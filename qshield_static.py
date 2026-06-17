#!/usr/bin/env python3
"""
QShield-Static: an abstract-interpretation-based state analyzer for Qiskit programs.

Design (matches the QShield-Static research plan):
  * Finite abstract domain of per-qubit labels:
        BOT  (unreachable / no info yet, lattice bottom)
        ZERO  |0>
        ONE   |1>
        PLUS  |+>
        MINUS |->
        MIXED (top: unknown / general superposition / post-measurement classical)
    plus an ENTANGLED status tracked structurally via union-find.
  * Monotone transfer functions per gate (h, x, y, z, s, t, cx, cz, ccx,
    swap, reset, measure, ...).
  * Union-find entanglement structure with a `split` operation applied on
    measurement (measured qubit leaves its class; survivors degrade to MIXED).
  * Lattice join for control-flow merges; fixpoint iteration for loops.
    The lattice is finite and transfer functions are monotone, so iteration
    terminates (we additionally cap iterations defensively).

Checks (silent, output-wrong oriented — the class LintQ cannot see):
  QS001 NoOpControlledGate   controlled gate whose control is definitely |0>
                             (the reversed-CNOT Bell-state bug)
  QS002 OpAfterMeasurement   quantum gate applied to an already-measured qubit
                             (state-aware analog of LintQ's OpAfterMeas)
  QS003 ConstantMeasurement  measuring a qubit in a definite basis state
                             (outcome is a compile-time constant)
  QS004 DoubleMeasurement    measuring the same qubit twice with no
                             intervening gate
  QS005 IdentityGatePair     adjacent self-inverse pair (H-H, X-X, ...) on the
                             same qubit — circuit no-op
  QS006 NeverMeasuredQubit   qubit is acted on but never measured in a circuit
                             that measures others (info-level; common symptom
                             of wrong-index bugs)

Usage:
    python qshield_static.py <file-or-directory> [more paths...] [--json out.json] [-q]

Scans every *.py file under each directory, analyzes each QuantumCircuit it
can resolve, and prints findings with file:line locations.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Abstract domain
# ---------------------------------------------------------------------------

class AbsState(Enum):
    BOT = "⊥"
    ZERO = "|0⟩"
    ONE = "|1⟩"
    PLUS = "|+⟩"
    MINUS = "|−⟩"
    MIXED = "⊤"      # unknown superposition / classical post-measurement

    def __str__(self) -> str:
        return self.value


def join(a: AbsState, b: AbsState) -> AbsState:
    """Least upper bound on the flat lattice BOT < {ZERO,ONE,PLUS,MINUS} < MIXED."""
    if a == b:
        return a
    if a == AbsState.BOT:
        return b
    if b == AbsState.BOT:
        return a
    return AbsState.MIXED


# ---------------------------------------------------------------------------
# Union-find with split (entanglement structure)
# ---------------------------------------------------------------------------

class EntanglementUF:
    """Union-find over qubit indices.

    Because standard union-find does not support efficient splitting, and our
    qubit counts are tiny, we represent classes explicitly as frozensets.
    `split(q)` removes q from its class; if the survivor class becomes a
    singleton it is dissolved.
    """

    def __init__(self, n: int):
        self.classes: list[set[int]] = []  # only classes of size >= 2 are stored
        self.n = n

    def class_of(self, q: int) -> Optional[set[int]]:
        for c in self.classes:
            if q in c:
                return c
        return None

    def entangled(self, q: int) -> bool:
        return self.class_of(q) is not None

    def union(self, a: int, b: int) -> None:
        ca, cb = self.class_of(a), self.class_of(b)
        if ca is not None and cb is not None:
            if ca is cb:
                return
            ca |= cb
            self.classes.remove(cb)
        elif ca is not None:
            ca.add(b)
        elif cb is not None:
            cb.add(a)
        else:
            self.classes.append({a, b})

    def split(self, q: int) -> set[int]:
        """Remove q from its class. Returns the set of former partners."""
        c = self.class_of(q)
        if c is None:
            return set()
        c.discard(q)
        partners = set(c)
        if len(c) < 2:
            self.classes.remove(c)
        return partners

    def copy(self) -> "EntanglementUF":
        uf = EntanglementUF(self.n)
        uf.classes = [set(c) for c in self.classes]
        return uf

    def join_with(self, other: "EntanglementUF") -> "EntanglementUF":
        """Sound join of two partitions: a pair is entangled in the join if it
        is entangled in either branch (over-approximation)."""
        out = self.copy()
        for c in other.classes:
            members = sorted(c)
            for x in members[1:]:
                out.union(members[0], x)
        return out

    def __eq__(self, other) -> bool:
        return sorted(map(sorted, self.classes)) == sorted(map(sorted, other.classes))


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    code: str
    severity: str          # "warning" | "info"
    message: str
    file: str
    line: int
    circuit: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Abstract circuit state
# ---------------------------------------------------------------------------

@dataclass
class CircuitState:
    name: str
    n_qubits: int
    qubit_state: list[AbsState] = field(default_factory=list)
    measured: list[bool] = field(default_factory=list)
    ops_since_measure: list[bool] = field(default_factory=list)  # gate after last measure?
    last_gate: list[Optional[str]] = field(default_factory=list)  # for identity-pair check
    last_gate_line: list[int] = field(default_factory=list)
    ever_gated: list[bool] = field(default_factory=list)
    ever_measured: list[bool] = field(default_factory=list)
    uf: EntanglementUF = None  # type: ignore

    def __post_init__(self):
        n = self.n_qubits
        if not self.qubit_state:
            self.qubit_state = [AbsState.ZERO] * n
            self.measured = [False] * n
            self.ops_since_measure = [False] * n
            self.last_gate = [None] * n
            self.last_gate_line = [0] * n
            self.ever_gated = [False] * n
            self.ever_measured = [False] * n
        if self.uf is None:
            self.uf = EntanglementUF(n)

    # -- lattice operations -------------------------------------------------

    def copy(self) -> "CircuitState":
        c = CircuitState(self.name, self.n_qubits)
        c.qubit_state = list(self.qubit_state)
        c.measured = list(self.measured)
        c.ops_since_measure = list(self.ops_since_measure)
        c.last_gate = list(self.last_gate)
        c.last_gate_line = list(self.last_gate_line)
        c.ever_gated = list(self.ever_gated)
        c.ever_measured = list(self.ever_measured)
        c.uf = self.uf.copy()
        return c

    def join_with(self, other: "CircuitState") -> "CircuitState":
        c = self.copy()
        for i in range(self.n_qubits):
            c.qubit_state[i] = join(self.qubit_state[i], other.qubit_state[i])
            c.measured[i] = self.measured[i] or other.measured[i]
            c.ops_since_measure[i] = self.ops_since_measure[i] or other.ops_since_measure[i]
            if self.last_gate[i] != other.last_gate[i]:
                c.last_gate[i] = None
            c.ever_gated[i] = self.ever_gated[i] or other.ever_gated[i]
            c.ever_measured[i] = self.ever_measured[i] or other.ever_measured[i]
        c.uf = self.uf.join_with(other.uf)
        return c

    def same_as(self, other: "CircuitState") -> bool:
        return (self.qubit_state == other.qubit_state
                and self.measured == other.measured
                and self.uf == other.uf)

    # -- helpers ------------------------------------------------------------

    def degrade_all(self) -> None:
        """Conservative top: used when we cannot resolve which qubit a gate hits."""
        for i in range(self.n_qubits):
            self.qubit_state[i] = AbsState.MIXED
            self.last_gate[i] = None
        # unknown multi-qubit gates may entangle anything: entangle all pairwise
        if self.n_qubits >= 2:
            for i in range(1, self.n_qubits):
                self.uf.union(0, i)


# ---------------------------------------------------------------------------
# Single-qubit transfer functions
# ---------------------------------------------------------------------------

_H = {AbsState.ZERO: AbsState.PLUS, AbsState.ONE: AbsState.MINUS,
      AbsState.PLUS: AbsState.ZERO, AbsState.MINUS: AbsState.ONE}
_X = {AbsState.ZERO: AbsState.ONE, AbsState.ONE: AbsState.ZERO,
      AbsState.PLUS: AbsState.PLUS, AbsState.MINUS: AbsState.MINUS}   # X|−⟩ = −|−⟩, phase irrelevant
_Y = {AbsState.ZERO: AbsState.ONE, AbsState.ONE: AbsState.ZERO,
      AbsState.PLUS: AbsState.MINUS, AbsState.MINUS: AbsState.PLUS}
_Z = {AbsState.ZERO: AbsState.ZERO, AbsState.ONE: AbsState.ONE,
      AbsState.PLUS: AbsState.MINUS, AbsState.MINUS: AbsState.PLUS}
# S/T (and daggers): preserve basis states up to phase; map |±⟩ out of the domain.
_PHASE = {AbsState.ZERO: AbsState.ZERO, AbsState.ONE: AbsState.ONE,
          AbsState.PLUS: AbsState.MIXED, AbsState.MINUS: AbsState.MIXED}

SINGLE_QUBIT_TABLES = {
    "h": _H, "x": _X, "y": _Y, "z": _Z,
    "s": _PHASE, "sdg": _PHASE, "t": _PHASE, "tdg": _PHASE,
    "p": _PHASE, "u1": _PHASE, "rz": _PHASE,  # diagonal: basis states preserved
}
SELF_INVERSE = {"h", "x", "y", "z", "cx", "cz", "swap", "ccx"}
PARAMETRIC_TO_MIXED = {"rx", "ry", "u", "u2", "u3", "r"}
NO_OPS = {"barrier", "id", "i", "delay"}
CONTROLLED = {"cx": 1, "cnot": 1, "cz": 1, "cy": 1, "ch": 1, "crz": 1,
              "crx": 1, "cry": 1, "cp": 1, "cu1": 1, "ccx": 2, "toffoli": 2,
              "mcx": -1}  # value = number of control qubits (-1 = variadic)


# ---------------------------------------------------------------------------
# The analyzer (AST interpreter over abstract states)
# ---------------------------------------------------------------------------

class QShieldAnalyzer:
    MAX_FIXPOINT_ITERS = 12  # defensive cap; lattice height is tiny

    def __init__(self, filename: str, source: str, in_function: bool = False):
        self.filename = filename
        self.source = source
        self.in_function = in_function  # circuits here are likely composable subcircuits
        self.findings: list[Finding] = []
        # symbol tables
        self.registers: dict[str, int] = {}            # qreg name -> size
        self.classical_regs: set[str] = set()          # creg names (skip in ctor)
        self.circuits: dict[str, CircuitState] = {}    # var name -> abstract state
        self.reg_offset: dict[tuple[str, str], int] = {}  # (circuit, reg) -> base index
        self.int_vars: dict[str, int] = {}             # simple constant propagation

    # -- public entry ---------------------------------------------------

    def run(self) -> list[Finding]:
        try:
            tree = ast.parse(self.source)
        except SyntaxError:
            return []  # not analyzable (some Bugs4Q entries are intentionally broken)
        self.exec_body(tree.body)
        self.final_checks()
        return self.findings

    # -- reporting ------------------------------------------------------

    def report(self, code: str, severity: str, msg: str, line: int, circ: str) -> None:
        f = Finding(code, severity, msg, self.filename, line, circ)
        # de-duplicate (loops revisit the same node)
        for existing in self.findings:
            if (existing.code, existing.line, existing.circuit) == (code, line, circ):
                return
        self.findings.append(f)

    # -- statement interpreter -------------------------------------------

    def exec_body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self.exec_stmt(stmt)

    def exec_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            self.handle_assign(stmt)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self.handle_call(stmt.value)
        elif isinstance(stmt, (ast.For, ast.While)):
            self.handle_loop(stmt)
        elif isinstance(stmt, ast.If):
            self.handle_if(stmt)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Analyze function bodies in isolation (fresh scope, shared findings).
            sub = QShieldAnalyzer(self.filename, "", in_function=True)
            sub.findings = self.findings
            sub.exec_body(stmt.body)
            sub.final_checks()
        elif isinstance(stmt, (ast.Try,)):
            self.exec_body(stmt.body)
            for h in stmt.handlers:
                self.exec_body(h.body)
            self.exec_body(stmt.finalbody)
        elif isinstance(stmt, ast.With):
            self.exec_body(stmt.body)
        # other statements: ignore

    # -- assignments -------------------------------------------------------

    def handle_assign(self, stmt: ast.Assign) -> None:
        if len(stmt.targets) != 1:
            return
        target = stmt.targets[0]
        value = stmt.value
        if not isinstance(target, ast.Name):
            return
        name = target.id

        if isinstance(value, ast.Call):
            callee = self.callee_name(value)
            if callee == "QuantumRegister":
                size = self.const_int(value.args[0]) if value.args else None
                if size is not None:
                    self.registers[name] = size
                return
            if callee == "ClassicalRegister":
                self.classical_regs.add(name)
                return
            if callee == "QuantumCircuit":
                self.make_circuit(name, value)
                return
            # circuit-producing method calls (e.g. qc2 = qc.copy()) — track conservatively
            if callee in {"copy", "compose", "inverse", "reverse_ops", "decompose",
                          "bind_parameters", "assign_parameters"}:
                base = self.call_receiver(value)
                if base in self.circuits:
                    self.circuits[name] = self.circuits[base].copy()
                    self.circuits[name].name = name
                return
            # any other call that we should still walk (qc.measure(...) used as value)
            self.handle_call(value)
            return

        cint = self.const_int(value)
        if cint is not None:
            self.int_vars[name] = cint

    def make_circuit(self, name: str, call: ast.Call) -> None:
        n_qubits = 0
        offsets: list[tuple[str, int]] = []
        ok = True
        for arg in call.args:
            if isinstance(arg, ast.Name) and arg.id in self.registers:
                offsets.append((arg.id, n_qubits))
                n_qubits += self.registers[arg.id]
            elif isinstance(arg, ast.Name) and arg.id in self.classical_regs:
                continue  # classical register: no qubits
            else:
                v = self.const_int(arg)
                if v is not None:
                    if n_qubits == 0:
                        n_qubits = v  # QuantumCircuit(nq) or (nq, nc): first int = qubits
                    # second int is classical bits — ignore
                elif isinstance(arg, ast.Name):
                    ok = False  # unknown register
        if n_qubits <= 0 or n_qubits > 64:
            ok = False
        if not ok:
            return
        cs = CircuitState(name, n_qubits)
        if self.in_function:
            # Circuits built inside functions are usually subcircuits composed
            # into a larger context: do NOT assume the |0...0> initial state.
            cs.qubit_state = [AbsState.MIXED] * n_qubits
        self.circuits[name] = cs
        for reg, off in offsets:
            self.reg_offset[(name, reg)] = off

    # -- control flow --------------------------------------------------------

    def handle_if(self, stmt: ast.If) -> None:
        snapshot = {k: v.copy() for k, v in self.circuits.items()}
        self.exec_body(stmt.body)
        then_state = self.circuits
        self.circuits = snapshot
        self.exec_body(stmt.orelse)
        # join
        merged: dict[str, CircuitState] = {}
        for k in set(then_state) | set(self.circuits):
            a, b = then_state.get(k), self.circuits.get(k)
            if a is not None and b is not None:
                merged[k] = a.join_with(b)
            else:
                merged[k] = (a or b).copy()
        self.circuits = merged

    def handle_loop(self, stmt) -> None:
        """Fixpoint: state* = state ⊔ F(state*) — terminates on the finite lattice."""
        # bind a constant range variable if trivially resolvable (helps precision
        # for `for i in range(n): qc.h(i)` — we still join over all iterations)
        for _ in range(self.MAX_FIXPOINT_ITERS):
            before = {k: v.copy() for k, v in self.circuits.items()}
            self.exec_body(stmt.body)
            # join post-body with pre-body (loop may execute 0..n times)
            changed = False
            for k in before:
                if k in self.circuits:
                    joined = before[k].join_with(self.circuits[k])
                    if not joined.same_as(self.circuits[k]) or not joined.same_as(before[k]):
                        changed = changed or not joined.same_as(self.circuits[k])
                    self.circuits[k] = joined
            if not changed:
                break
        if hasattr(stmt, "orelse"):
            self.exec_body(stmt.orelse)

    # -- call handling ---------------------------------------------------------

    def callee_name(self, call: ast.Call) -> Optional[str]:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    def call_receiver(self, call: ast.Call) -> Optional[str]:
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            return f.value.id
        return None

    def handle_call(self, call: ast.Call) -> None:
        recv = self.call_receiver(call)
        if recv is None or recv not in self.circuits:
            # still walk nested calls (e.g. execute(qc, backend))
            for a in call.args:
                if isinstance(a, ast.Call):
                    self.handle_call(a)
            return
        cs = self.circuits[recv]
        gate = (self.callee_name(call) or "").lower()
        line = call.lineno

        if gate in NO_OPS:
            return
        if gate in ("measure", "measure_all", "measure_active"):
            self.apply_measure(cs, call, gate, line)
            return
        if gate == "reset":
            for q in self.resolve_qubits(cs, call.args[:1], recv):
                self.do_reset(cs, q)
            return
        if gate in CONTROLLED:
            self.apply_controlled(cs, call, gate, line, recv)
            return
        if gate == "swap":
            qs = self.resolve_qubits(cs, call.args[:2], recv)
            if len(qs) == 2:
                a, b = qs
                self.check_post_measure(cs, [a, b], gate, line)
                cs.qubit_state[a], cs.qubit_state[b] = cs.qubit_state[b], cs.qubit_state[a]
                self.mark_gate(cs, a, gate, line)
                self.mark_gate(cs, b, gate, line)
            else:
                cs.degrade_all()
            return
        if gate in SINGLE_QUBIT_TABLES or gate in PARAMETRIC_TO_MIXED:
            self.apply_single(cs, call, gate, line, recv)
            return
        if gate in {"initialize", "append", "compose", "unitary"}:
            cs.degrade_all()
            return
        # unrecognized circuit method: ignore conservatively if clearly non-gate
        if gate in {"draw", "depth", "size", "count_ops", "qasm", "to_gate",
                    "add_register", "name"}:
            return
        # unknown method on a circuit -> conservative
        cs.degrade_all()

    # -- gate application -----------------------------------------------------

    def apply_single(self, cs: CircuitState, call: ast.Call, gate: str,
                     line: int, recv: str) -> None:
        targets = self.resolve_qubits(cs, call.args, recv, allow_params=True)
        if targets is None:
            cs.degrade_all()
            return
        for q in targets:
            self.check_post_measure(cs, [q], gate, line)
            # QS005: adjacent self-inverse pair
            if gate in SELF_INVERSE and cs.last_gate[q] == gate:
                self.report("QS005", "info",
                            f"adjacent '{gate}-{gate}' pair on qubit {q} is an identity "
                            f"(previous at line {cs.last_gate_line[q]}) — likely dead code "
                            f"or a copy-paste slip", line, cs.name)
            if cs.uf.entangled(q):
                # gate on an entangled qubit: keep entanglement, state stays MIXED
                cs.qubit_state[q] = AbsState.MIXED
            else:
                table = SINGLE_QUBIT_TABLES.get(gate)
                if table is None:  # parametric rotation
                    cs.qubit_state[q] = AbsState.MIXED
                else:
                    cs.qubit_state[q] = table.get(cs.qubit_state[q], AbsState.MIXED)
            self.mark_gate(cs, q, gate, line)

    def apply_controlled(self, cs: CircuitState, call: ast.Call, gate: str,
                         line: int, recv: str) -> None:
        # positional args: controls..., target (params first for crz etc. — strip non-qubit consts)
        args = [a for a in call.args if not isinstance(a, (ast.Constant,)) or
                isinstance(self.const_int(a), int)]
        qs = self.resolve_qubits(cs, call.args, recv)
        n_ctrl = CONTROLLED[gate]
        if qs is None or len(qs) < 2:
            cs.degrade_all()
            return
        if n_ctrl == -1:
            n_ctrl = len(qs) - 1
        controls, targets = qs[:n_ctrl], qs[n_ctrl:]
        if not targets:
            cs.degrade_all()
            return
        target = targets[0]
        self.check_post_measure(cs, qs, gate, line)

        # Qiskit's ctrl_state keyword inverts control polarity; if present and
        # not the default all-ones, our |0>-control reasoning doesn't apply.
        nondefault_ctrl = any(kw.arg == "ctrl_state" for kw in call.keywords)

        ctrl_states = [cs.qubit_state[c] for c in controls]

        # QS001: control definitely |0> -> controlled gate is a global no-op
        if not nondefault_ctrl and any(s == AbsState.ZERO and not cs.uf.entangled(c)
               for s, c in zip(ctrl_states, controls)):
            dead = [c for s, c in zip(ctrl_states, controls)
                    if s == AbsState.ZERO and not cs.uf.entangled(c)]
            hint = ""
            if gate in ("cx", "cnot") and len(qs) == 2:
                t_state = cs.qubit_state[target]
                if t_state in (AbsState.PLUS, AbsState.MINUS) or cs.uf.entangled(target):
                    hint = (f" — target qubit {target} is in superposition; "
                            f"did you mean {gate}({target}, {controls[0]})? "
                            f"(reversed control/target)")
            self.report("QS001", "warning",
                        f"'{gate}' at qubit(s) {qs}: control qubit {dead[0]} is in "
                        f"abstract state |0⟩, so this gate never fires (silent no-op)"
                        + hint, line, cs.name)
            self.mark_gate_many(cs, qs, gate, line)
            return  # no state change: the gate provably does nothing

        # control definitely |1>: gate reduces to its base gate on the target
        if not nondefault_ctrl and all(s == AbsState.ONE and not cs.uf.entangled(c)
               for s, c in zip(ctrl_states, controls)):
            base = {"cx": _X, "cnot": _X, "cy": _Y, "cz": _Z, "ccx": _X,
                    "toffoli": _X, "mcx": _X, "ch": _H}.get(gate, _PHASE)
            if cs.uf.entangled(target):
                cs.qubit_state[target] = AbsState.MIXED
            else:
                cs.qubit_state[target] = base.get(cs.qubit_state[target], AbsState.MIXED)
            self.mark_gate_many(cs, qs, gate, line)
            return

        # control in superposition (or unknown): entangle control(s) and target
        if nondefault_ctrl:
            # inverted-polarity control: be sound, not precise
            cs.qubit_state[target] = AbsState.MIXED
            for c in controls:
                if cs.qubit_state[c] not in (AbsState.ZERO, AbsState.ONE) or cs.uf.entangled(c):
                    cs.uf.union(c, target)
            self.mark_gate_many(cs, qs, gate, line)
            return
        for c in controls:
            if cs.qubit_state[c] in (AbsState.PLUS, AbsState.MINUS, AbsState.MIXED) \
                    or cs.uf.entangled(c):
                # cz with target in a basis Z-eigenstate doesn't entangle; keep it
                # simple and sound: entangle (over-approximation is safe here)
                cs.uf.union(c, target)
        if cs.uf.entangled(target):
            for q in qs:
                if cs.uf.entangled(q):
                    cs.qubit_state[q] = AbsState.MIXED
        self.mark_gate_many(cs, qs, gate, line)

    def do_reset(self, cs: CircuitState, q: int) -> None:
        partners = cs.uf.split(q)
        for p in partners:
            cs.qubit_state[p] = AbsState.MIXED
        cs.qubit_state[q] = AbsState.ZERO
        cs.measured[q] = False
        cs.last_gate[q] = None

    # -- measurement -----------------------------------------------------------

    def apply_measure(self, cs: CircuitState, call: ast.Call, gate: str,
                      line: int) -> None:
        if gate in ("measure_all", "measure_active"):
            qubits = list(range(cs.n_qubits))
        else:
            qubits = self.resolve_qubits(cs, call.args[:1], cs.name)
            if qubits is None:
                qubits = list(range(cs.n_qubits))  # conservative: may measure any

        for q in qubits:
            # QS003: constant measurement
            if cs.qubit_state[q] in (AbsState.ZERO, AbsState.ONE) and not cs.uf.entangled(q):
                self.report("QS003", "warning",
                            f"measurement of qubit {q} at a point where its abstract "
                            f"state is {cs.qubit_state[q]} — the outcome is a constant; "
                            f"if this is intentional the qubit is doing no quantum work",
                            line, cs.name)
            # QS004: double measurement with no gate in between
            if cs.measured[q] and not cs.ops_since_measure[q]:
                self.report("QS004", "warning",
                            f"qubit {q} is measured again with no intervening gate — "
                            f"redundant measurement (same classical outcome)",
                            line, cs.name)
            # split entanglement
            partners = cs.uf.split(q)
            for p in partners:
                cs.qubit_state[p] = AbsState.MIXED  # collapsed to unknown basis state
            cs.qubit_state[q] = AbsState.MIXED       # classical 0/1, unknown which
            cs.measured[q] = True
            cs.ever_measured[q] = True
            cs.ops_since_measure[q] = False
            cs.last_gate[q] = None

    def check_post_measure(self, cs: CircuitState, qubits: list[int],
                           gate: str, line: int) -> None:
        for q in qubits:
            if cs.measured[q]:
                self.report("QS002", "warning",
                            f"gate '{gate}' applied to qubit {q} after it was "
                            f"measured — operations after measurement are usually "
                            f"unintended (state already collapsed)", line, cs.name)
            cs.ops_since_measure[q] = True

    def mark_gate(self, cs: CircuitState, q: int, gate: str, line: int) -> None:
        cs.last_gate[q] = gate
        cs.last_gate_line[q] = line
        cs.ever_gated[q] = True

    def mark_gate_many(self, cs: CircuitState, qs: list[int], gate: str, line: int) -> None:
        for q in qs:
            self.mark_gate(cs, q, gate, line)

    # -- qubit-argument resolution ------------------------------------------------

    def resolve_qubits(self, cs: CircuitState, args, recv: str,
                       allow_params: bool = False):
        """Map AST call arguments to concrete qubit indices.

        Returns a list of indices, or None if any argument is unresolvable
        (caller should degrade to top).
        """
        out: list[int] = []
        for a in args:
            if isinstance(a, ast.keyword):
                continue
            v = self.const_int(a)
            if v is not None:
                if 0 <= v < cs.n_qubits:
                    out.append(v)
                elif allow_params:
                    continue  # probably an angle like qc.rz(2, 0) — skip big consts? keep simple
                else:
                    continue
                continue
            if isinstance(a, ast.Constant) and isinstance(a.value, float) and allow_params:
                continue  # rotation angle
            if isinstance(a, ast.Subscript):  # qreg[i]
                base = a.value.id if isinstance(a.value, ast.Name) else None
                idx = self.const_int(a.slice if not isinstance(a.slice, ast.Index)
                                     else a.slice.value)  # py<3.9 compat
                if base is not None and idx is not None and (cs.name, base) in self.reg_offset:
                    q = self.reg_offset[(cs.name, base)] + idx
                    if 0 <= q < cs.n_qubits:
                        out.append(q)
                        continue
                return None
            if isinstance(a, ast.Name):
                if a.id in self.int_vars:
                    v = self.int_vars[a.id]
                    if 0 <= v < cs.n_qubits:
                        out.append(v)
                        continue
                if a.id in self.registers:  # whole-register application
                    off = self.reg_offset.get((cs.name, a.id), 0)
                    out.extend(range(off, off + self.registers[a.id]))
                    continue
                return None
            if isinstance(a, (ast.List, ast.Tuple)):
                sub = self.resolve_qubits(cs, a.elts, recv, allow_params)
                if sub is None:
                    return None
                out.extend(sub)
                continue
            if isinstance(a, ast.Call) and self.callee_name(a) == "range":
                vals = [self.const_int(x) for x in a.args]
                if all(v is not None for v in vals):
                    out.extend(range(*vals))
                    continue
                return None
            if allow_params and isinstance(a, (ast.BinOp, ast.UnaryOp, ast.Attribute)):
                continue  # angle expression like pi/2
            return None
        return out

    def const_int(self, node) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self.const_int(node.operand)
            return -v if v is not None else None
        if isinstance(node, ast.Name) and node.id in self.int_vars:
            return self.int_vars[node.id]
        return None

    # -- end-of-program checks ------------------------------------------------------

    def final_checks(self) -> None:
        for cs in self.circuits.values():
            if any(cs.ever_measured) and not all(cs.ever_measured):
                for q in range(cs.n_qubits):
                    if cs.ever_gated[q] and not cs.ever_measured[q]:
                        self.report("QS006", "info",
                                    f"qubit {q} has gates applied but is never measured, "
                                    f"while other qubits are — possible wrong-index bug "
                                    f"or wasted qubit", 0, cs.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def analyze_path(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    for f in files:
        if f.name == Path(__file__).name:
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "QuantumCircuit" not in src and "qiskit" not in src:
            continue
        findings.extend(QShieldAnalyzer(str(f), src).run())
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="QShield-Static abstract state analyzer")
    ap.add_argument("paths", nargs="+", help="files or directories to scan")
    ap.add_argument("--json", metavar="OUT", help="also write findings as JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    all_findings: list[Finding] = []
    for p in args.paths:
        all_findings.extend(analyze_path(Path(p)))

    sev_order = {"warning": 0, "info": 1}
    all_findings.sort(key=lambda f: (f.file, f.line, sev_order.get(f.severity, 2)))

    if not args.quiet:
        for f in all_findings:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            print(f"{loc}: [{f.code}/{f.severity}] ({f.circuit}) {f.message}")

    n_warn = sum(1 for f in all_findings if f.severity == "warning")
    n_info = len(all_findings) - n_warn
    by_code: dict[str, int] = {}
    for f in all_findings:
        by_code[f.code] = by_code.get(f.code, 0) + 1
    print(f"\nQShield-Static: {n_warn} warning(s), {n_info} info finding(s)"
          + (f"  [{', '.join(f'{k}:{v}' for k, v in sorted(by_code.items()))}]"
             if by_code else ""))

    if args.json:
        Path(args.json).write_text(json.dumps([f.to_dict() for f in all_findings], indent=2))
        print(f"JSON written to {args.json}")
    return 1 if n_warn else 0


if __name__ == "__main__":
    sys.exit(main())
