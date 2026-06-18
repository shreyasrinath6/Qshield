#!/usr/bin/env python3
"""
QShield-Static: an abstract-interpretation-based state analyzer for Qiskit programs.

Design (matches the QShield-Static research plan):
  * Finite abstract domain of per-qubit labels (see AbsState below).
  * Monotone transfer functions per gate (h, x, y, z, s, t, cx, cz, ccx,
    swap, reset, measure, ...).
  * Union-find entanglement structure with a `split` operation applied on
    measurement (measured qubit leaves its class; survivors degrade).
  * Lattice join for control-flow merges; fixpoint iteration for loops.
    The lattice is finite and transfer functions are monotone, so iteration
    terminates (we additionally cap iterations defensively).

============================================================================
CHANGELOG (revision for Bugs4Q recall improvement)
============================================================================
This revision targets the three problem areas from the recall task. Every
change below is tagged [P1]/[P2]/[P3] for the area it addresses.

[P1] Abstract domain refinement (lattice still finite, height = 3):
  - Split the old single ⊤ ("MIXED") into TWO incomparable refinements:
        SUPER     — coherent superposition, unknown phase (was the common
                    meaning of MIXED: post-rotation / entangled component)
        CLASSICAL — post-measurement, a definite but statically-unknown 0/1
    Their join is the new top, TOP. `MIXED` is kept as an alias of SUPER so
    no external caller / message breaks. This lets QS002/QS004 distinguish
    "gate after measurement" (CLASSICAL) from "gate on a live superposition"
    (SUPER), and lets QS003 reason about post-measurement reuse precisely.
  - Added a lightweight, separate per-qubit PHASE tag (PLUS / MINUS phase /
    UNKNOWN) carried alongside the basis label. Z/S/Z-like diagonal gates now
    flip/advance the phase tag instead of collapsing |+>/|-> to top. This makes
    a missing Z correction (teleportation) visible to the new QS008 rule and
    keeps |+> vs |-> distinguishable through a Z.
  - Lattice height stays bounded: basis lattice is BOT < {ZERO,ONE,PLUS,MINUS}
    < {SUPER, CLASSICAL} < TOP (height 3); phase lattice is a 3-point flat
    lattice (UNKNOWN is its top). Product of two finite lattices is finite and
    all transfer functions remain monotone, so fixpoint still terminates.

[P2] Resolution / constant propagation:
  - Loop-variable binding: `for i in range(...)` now binds `i` to each concrete
    value (when the range is small and statically known) and the body is
    interpreted per iteration, then joined — instead of degrading on the first
    indexed gate. Falls back to the old fixpoint-join when the range is unknown
    or large.
  - int_vars now also handles augmented assignments (i += 1) and simple
    tuple/list unpacking (a, b = 0, 1).
  - Register-scoped degradation: degrade_register() replaces whole-circuit
    degrade_all() when we know which register's subscript failed to resolve.
    Only the qubits of that register go to top; the rest keep their info.

[P3] New rules (QS007+), all complementary to LintQ (no syntactic dup):
  QS007 WrongInitialStateAssumption — a subcircuit / composed circuit applies a
        gate whose meaning depends on a |0> start, but the qubit is SUPER/TOP
        (function-body circuits start at TOP). warning.
  QS008 MissingCorrectionGate — a measured qubit's outcome is used to condition
        later structure (teleportation-style) but the expected basis/phase
        correction is absent or of the wrong type (X where Z is needed, etc.).
        Detected via the phase tag + entanglement remnants. warning.
  QS009 QubitIndexTransposition — a 2-qubit gate is applied with arguments that
        are statically the reverse of a previously-applied gate on the same
        pair, with no symmetry justification (e.g. cx(1,0) after cx(0,1)).
        warning.
  QS010 RedundantReset — reset() on a qubit already provably in |0> and not
        entangled — a no-op signalling a misunderstanding of the initial
        state. info.
  QS011 EntanglementNotEstablished — a controlled gate intended to spread
        superposition is applied when BOTH control and target are definite
        basis states and unentangled, so no entanglement is created where the
        surrounding structure (later measurement of both) implies it should be.
        warning.

Checks (silent, output-wrong oriented — the class LintQ cannot see):
  QS001 NoOpControlledGate   controlled gate whose control is definitely |0>
  QS002 OpAfterMeasurement   quantum gate applied to an already-measured qubit
  QS003 ConstantMeasurement  measuring a qubit in a definite basis state
  QS004 DoubleMeasurement    measuring the same qubit twice with no gate between
  QS005 IdentityGatePair     adjacent self-inverse pair (H-H, X-X, ...)
  QS006 NeverMeasuredQubit   qubit acted on but never measured (info)
  QS007 WrongInitialStateAssumption   (new, see above)
  QS008 MissingCorrectionGate         (new, see above)
  QS009 QubitIndexTransposition       (new, see above)
  QS010 RedundantReset                (new, see above)
  QS011 EntanglementNotEstablished    (new, see above)

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
# [P1] Two refinements of the old ⊤: SUPER (live coherent superposition) and
# CLASSICAL (post-measurement, definite-but-unknown). TOP is their join.

class AbsState(Enum):
    BOT = "⊥"
    ZERO = "|0⟩"
    ONE = "|1⟩"
    PLUS = "|+⟩"
    MINUS = "|−⟩"
    SUPER = "⊤super"      # coherent superposition, unknown phase
    CLASSICAL = "⊤cl"     # post-measurement: a definite but unknown 0/1
    TOP = "⊤"             # super ⊔ classical: no information at all

    def __str__(self) -> str:
        return self.value


# Back-compat alias: historically the code/messages used MIXED for "live
# superposition / unknown". Keep the name working so callers & message text
# don't break; it now means SUPER specifically.
AbsState.MIXED = AbsState.SUPER  # type: ignore[attr-defined]

_DEFINITE = (AbsState.ZERO, AbsState.ONE, AbsState.PLUS, AbsState.MINUS)
_BASIS_Z = (AbsState.ZERO, AbsState.ONE)        # computational-basis eigenstates
_BASIS_X = (AbsState.PLUS, AbsState.MINUS)      # Hadamard-basis eigenstates
# states that still carry "this qubit is doing quantum work" meaning
_LIVE_SUPER = (AbsState.PLUS, AbsState.MINUS, AbsState.SUPER)


def join(a: AbsState, b: AbsState) -> AbsState:
    """Least upper bound on the lattice

            BOT
             |
        {ZERO,ONE,PLUS,MINUS}
           /          \\
        SUPER       CLASSICAL
           \\          /
             TOP

    (height 3, finite). A definite state joins UP to SUPER, since with no
    further info two distinct definite states are a coherent superposition;
    SUPER ⊔ CLASSICAL = TOP.
    """
    if a == b:
        return a
    if a == AbsState.BOT:
        return b
    if b == AbsState.BOT:
        return a
    if AbsState.TOP in (a, b):
        return AbsState.TOP
    # SUPER vs CLASSICAL (or either with a definite state below the other)
    def coarse(s: AbsState) -> AbsState:
        if s == AbsState.CLASSICAL:
            return AbsState.CLASSICAL
        if s == AbsState.SUPER:
            return AbsState.SUPER
        return AbsState.SUPER  # definite states sit below SUPER by default
    ca, cb = coarse(a), coarse(b)
    if ca == cb:
        return ca
    return AbsState.TOP


# ---------------------------------------------------------------------------
# [P1] Phase tag — lightweight relative-phase tracking for basis states.
# Flat 3-point lattice: PLUS_PH, MINUS_PH (definite) join to PH_UNKNOWN (top).
# ---------------------------------------------------------------------------

class Phase(Enum):
    PLUS_PH = "+"        # no nontrivial relative phase (e.g. |+>, |0>)
    MINUS_PH = "−"       # a Z-type phase has been applied (e.g. |->, Z|0>)
    UNKNOWN = "?"        # top: unknown relative phase

    def __str__(self) -> str:
        return self.value


def join_phase(a: Phase, b: Phase) -> Phase:
    if a == b:
        return a
    return Phase.UNKNOWN


def flip_phase(p: Phase) -> Phase:
    if p == Phase.PLUS_PH:
        return Phase.MINUS_PH
    if p == Phase.MINUS_PH:
        return Phase.PLUS_PH
    return Phase.UNKNOWN


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
    phase: list[Phase] = field(default_factory=list)          # [P1] phase tag
    measured: list[bool] = field(default_factory=list)
    ops_since_measure: list[bool] = field(default_factory=list)  # gate after last measure?
    last_gate: list[Optional[str]] = field(default_factory=list)  # for identity-pair check
    last_gate_line: list[int] = field(default_factory=list)
    ever_gated: list[bool] = field(default_factory=list)
    ever_measured: list[bool] = field(default_factory=list)
    # [P3] history of 2-qubit gate argument tuples, for QS009 transposition:
    pair_history: list[tuple[str, tuple[int, ...], int]] = field(default_factory=list)
    # [P3] qubits whose superposition collapsed at measurement and that still
    # have an entangled partner needing a correction (teleportation pattern):
    awaiting_correction: dict = field(default_factory=dict)  # q -> (line, phase)
    reg_of: dict = field(default_factory=dict)  # [P2] qubit index -> register name
    started_top: bool = False  # [P3] circuit built inside a function (subcircuit)
    uf: EntanglementUF = None  # type: ignore

    def __post_init__(self):
        n = self.n_qubits
        if not self.qubit_state:
            self.qubit_state = [AbsState.ZERO] * n
            self.phase = [Phase.PLUS_PH] * n
            self.measured = [False] * n
            self.ops_since_measure = [False] * n
            self.last_gate = [None] * n
            self.last_gate_line = [0] * n
            self.ever_gated = [False] * n
            self.ever_measured = [False] * n
        if not self.phase:
            self.phase = [Phase.PLUS_PH] * n
        if self.uf is None:
            self.uf = EntanglementUF(n)

    # -- lattice operations -------------------------------------------------

    def copy(self) -> "CircuitState":
        c = CircuitState(self.name, self.n_qubits)
        c.qubit_state = list(self.qubit_state)
        c.phase = list(self.phase)
        c.measured = list(self.measured)
        c.ops_since_measure = list(self.ops_since_measure)
        c.last_gate = list(self.last_gate)
        c.last_gate_line = list(self.last_gate_line)
        c.ever_gated = list(self.ever_gated)
        c.ever_measured = list(self.ever_measured)
        c.pair_history = list(self.pair_history)
        c.awaiting_correction = dict(self.awaiting_correction)
        c.reg_of = dict(self.reg_of)
        c.started_top = self.started_top
        c.uf = self.uf.copy()
        return c

    def join_with(self, other: "CircuitState") -> "CircuitState":
        c = self.copy()
        for i in range(self.n_qubits):
            c.qubit_state[i] = join(self.qubit_state[i], other.qubit_state[i])
            c.phase[i] = join_phase(self.phase[i], other.phase[i])
            c.measured[i] = self.measured[i] or other.measured[i]
            c.ops_since_measure[i] = self.ops_since_measure[i] or other.ops_since_measure[i]
            if self.last_gate[i] != other.last_gate[i]:
                c.last_gate[i] = None
            c.ever_gated[i] = self.ever_gated[i] or other.ever_gated[i]
            c.ever_measured[i] = self.ever_measured[i] or other.ever_measured[i]
        # awaiting_correction: keep entries present in either branch (sound)
        merged_await = dict(self.awaiting_correction)
        merged_await.update(other.awaiting_correction)
        c.awaiting_correction = merged_await
        c.uf = self.uf.join_with(other.uf)
        return c

    def same_as(self, other: "CircuitState") -> bool:
        return (self.qubit_state == other.qubit_state
                and self.phase == other.phase
                and self.measured == other.measured
                and self.uf == other.uf)

    # -- helpers ------------------------------------------------------------

    def degrade_all(self) -> None:
        """Conservative top: used when we cannot resolve which qubit a gate hits
        AND we cannot scope it to a single register."""
        for i in range(self.n_qubits):
            self.qubit_state[i] = AbsState.TOP
            self.phase[i] = Phase.UNKNOWN
            self.last_gate[i] = None
        # unknown multi-qubit gates may entangle anything: entangle all pairwise
        if self.n_qubits >= 2:
            for i in range(1, self.n_qubits):
                self.uf.union(0, i)

    def degrade_register(self, qubits: list[int]) -> None:
        """[P2] Register-scoped degradation: only the given qubits go to top;
        the rest of the circuit keeps its analyzed info. Used when a subscript
        base is known but the index is not."""
        if not qubits:
            self.degrade_all()
            return
        for i in qubits:
            if 0 <= i < self.n_qubits:
                self.qubit_state[i] = AbsState.TOP
                self.phase[i] = Phase.UNKNOWN
                self.last_gate[i] = None
        # those qubits may be entangled among themselves
        for j in qubits[1:]:
            if 0 <= qubits[0] < self.n_qubits and 0 <= j < self.n_qubits:
                self.uf.union(qubits[0], j)


# ---------------------------------------------------------------------------
# Single-qubit transfer functions  (basis-label part)
# ---------------------------------------------------------------------------

_H = {AbsState.ZERO: AbsState.PLUS, AbsState.ONE: AbsState.MINUS,
      AbsState.PLUS: AbsState.ZERO, AbsState.MINUS: AbsState.ONE}
_X = {AbsState.ZERO: AbsState.ONE, AbsState.ONE: AbsState.ZERO,
      AbsState.PLUS: AbsState.PLUS, AbsState.MINUS: AbsState.MINUS}
_Y = {AbsState.ZERO: AbsState.ONE, AbsState.ONE: AbsState.ZERO,
      AbsState.PLUS: AbsState.MINUS, AbsState.MINUS: AbsState.PLUS}
_Z = {AbsState.ZERO: AbsState.ZERO, AbsState.ONE: AbsState.ONE,
      AbsState.PLUS: AbsState.MINUS, AbsState.MINUS: AbsState.PLUS}
# S/T (and daggers): preserve basis Z-eigenstates; rotate the phase of |±⟩.
_PHASE = {AbsState.ZERO: AbsState.ZERO, AbsState.ONE: AbsState.ONE,
          AbsState.PLUS: AbsState.SUPER, AbsState.MINUS: AbsState.SUPER}

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
# [P1] gates whose Z-phase action we model on the phase tag:
_PHASE_FLIP_GATES = {"z", "s", "sdg", "t", "tdg", "p", "u1", "rz"}


# ---------------------------------------------------------------------------
# The analyzer (AST interpreter over abstract states)
# ---------------------------------------------------------------------------

class QShieldAnalyzer:
    MAX_FIXPOINT_ITERS = 12  # defensive cap; lattice height is tiny
    MAX_LOOP_UNROLL = 16     # [P2] concrete loop-variable binding limit

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
        elif isinstance(stmt, ast.AugAssign):
            self.handle_aug_assign(stmt)  # [P2]
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

        # [P2] simple tuple / list unpacking:  a, b = 0, 1
        if isinstance(target, (ast.Tuple, ast.List)) and \
                isinstance(value, (ast.Tuple, ast.List)) and \
                len(target.elts) == len(value.elts):
            for tnode, vnode in zip(target.elts, value.elts):
                if isinstance(tnode, ast.Name):
                    cv = self.const_int(vnode)
                    if cv is not None:
                        self.int_vars[tnode.id] = cv
            return

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
        elif name in self.int_vars:
            # reassigned to something non-constant: forget the binding
            del self.int_vars[name]

    def handle_aug_assign(self, stmt: ast.AugAssign) -> None:
        """[P2] handle i += k / i -= k / i *= k for simple constant tracking."""
        if not isinstance(stmt.target, ast.Name):
            return
        name = stmt.target.id
        if name not in self.int_vars:
            return
        rhs = self.const_int(stmt.value)
        if rhs is None:
            del self.int_vars[name]
            return
        cur = self.int_vars[name]
        if isinstance(stmt.op, ast.Add):
            self.int_vars[name] = cur + rhs
        elif isinstance(stmt.op, ast.Sub):
            self.int_vars[name] = cur - rhs
        elif isinstance(stmt.op, ast.Mult):
            self.int_vars[name] = cur * rhs
        else:
            del self.int_vars[name]

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
            # [P3] Circuits built inside functions are usually subcircuits
            # composed into a larger context: do NOT assume the |0...0> initial
            # state — start at TOP and remember that fact for QS007.
            cs.qubit_state = [AbsState.TOP] * n_qubits
            cs.phase = [Phase.UNKNOWN] * n_qubits
            cs.started_top = True
        self.circuits[name] = cs
        for reg, off in offsets:
            self.reg_offset[(name, reg)] = off
            for i in range(off, off + self.registers[reg]):
                if i < n_qubits:
                    cs.reg_of[i] = reg  # [P2] qubit -> register name

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
        """Loop handling. [P2] If this is a `for v in range(...)` with a small,
        statically-known range, bind v to each concrete value and interpret the
        body per iteration (joining circuit states across iterations) so indexed
        gates resolve. Otherwise fall back to the finite-lattice fixpoint."""
        if isinstance(stmt, ast.For):
            concrete = self._concrete_for_values(stmt)
            if concrete is not None:
                target = stmt.target
                if isinstance(target, ast.Name) and len(concrete) <= self.MAX_LOOP_UNROLL:
                    for val in concrete:
                        self.int_vars[target.id] = val
                        self.exec_body(stmt.body)
                    # drop the loop variable binding after the loop
                    self.int_vars.pop(target.id, None)
                    if stmt.orelse:
                        self.exec_body(stmt.orelse)
                    return
                # known but large/iterable range: bind to a representative join
                # by still running the fixpoint below.

        # Fallback fixpoint: state* = state ⊔ F(state*) — terminates on the
        # finite lattice.
        for _ in range(self.MAX_FIXPOINT_ITERS):
            before = {k: v.copy() for k, v in self.circuits.items()}
            self.exec_body(stmt.body)
            changed = False
            for k in before:
                if k in self.circuits:
                    joined = before[k].join_with(self.circuits[k])
                    if not joined.same_as(self.circuits[k]):
                        changed = True
                    self.circuits[k] = joined
            if not changed:
                break
        if hasattr(stmt, "orelse"):
            self.exec_body(stmt.orelse)

    def _concrete_for_values(self, stmt: ast.For):
        """[P2] Return the concrete list of values for `for v in range(...)`
        / `for v in [literals]` if statically resolvable, else None."""
        it = stmt.iter
        if isinstance(it, ast.Call) and self.callee_name(it) == "range":
            vals = [self.const_int(x) for x in it.args]
            if all(v is not None for v in vals) and vals:
                try:
                    return list(range(*vals))
                except (TypeError, ValueError):
                    return None
            return None
        if isinstance(it, (ast.List, ast.Tuple)):
            out = []
            for e in it.elts:
                cv = self.const_int(e)
                if cv is None:
                    return None
                out.append(cv)
            return out
        return None

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
            qs = self.resolve_qubits(cs, call.args[:1], recv)
            if qs is None:
                self.scoped_degrade(cs, call.args[:1])
                return
            for q in qs:
                self.do_reset(cs, q, line)
            return
        if gate in CONTROLLED:
            self.apply_controlled(cs, call, gate, line, recv)
            return
        if gate == "swap":
            qs = self.resolve_qubits(cs, call.args[:2], recv)
            if qs is not None and len(qs) == 2:
                a, b = qs
                self.check_post_measure(cs, [a, b], gate, line)
                self.record_pair(cs, gate, (a, b), line)  # [P3] QS009
                cs.qubit_state[a], cs.qubit_state[b] = cs.qubit_state[b], cs.qubit_state[a]
                cs.phase[a], cs.phase[b] = cs.phase[b], cs.phase[a]
                self.mark_gate(cs, a, gate, line)
                self.mark_gate(cs, b, gate, line)
            else:
                self.scoped_degrade(cs, call.args[:2])
            return
        if gate in SINGLE_QUBIT_TABLES or gate in PARAMETRIC_TO_MIXED:
            self.apply_single(cs, call, gate, line, recv)
            return
        if gate in {"initialize", "append", "compose", "unitary"}:
            # [P3] composition / append onto a circuit assumed at |0>: if the
            # circuit is NOT a subcircuit but the appended block likely expects
            # a fresh state, this can hide a wrong-initial-state bug. We degrade
            # conservatively but flag via QS007 when the receiver started_top.
            if cs.started_top:
                self.report("QS007", "warning",
                            f"circuit '{cs.name}' is a subcircuit (qubits not "
                            f"initialized to |0⟩) yet '{gate}' is applied as if "
                            f"on a fresh register — verify the composed block "
                            f"does not assume a |0…0⟩ start", line, cs.name)
            cs.degrade_all()
            return
        # unrecognized circuit method: ignore conservatively if clearly non-gate
        if gate in {"draw", "depth", "size", "count_ops", "qasm", "to_gate",
                    "add_register", "name"}:
            return
        # unknown method on a circuit -> conservative
        cs.degrade_all()

    def scoped_degrade(self, cs: CircuitState, args) -> None:
        """[P2] Degrade only the register implicated by an unresolvable
        subscript, if we can identify it; otherwise whole-circuit."""
        regs: set[str] = set()
        for a in args:
            if isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name):
                regs.add(a.value.id)
        if regs:
            affected: list[int] = []
            for reg in regs:
                off = self.reg_offset.get((cs.name, reg))
                size = self.registers.get(reg)
                if off is not None and size is not None:
                    affected.extend(range(off, off + size))
            if affected:
                cs.degrade_register(affected)
                return
        cs.degrade_all()

    # -- gate application -----------------------------------------------------

    def apply_single(self, cs: CircuitState, call: ast.Call, gate: str,
                     line: int, recv: str) -> None:
        targets = self.resolve_qubits(cs, call.args, recv, allow_params=True)
        if targets is None:
            self.scoped_degrade(cs, call.args)
            return
        for q in targets:
            self.check_post_measure(cs, [q], gate, line)

            # [P3] QS010: reset already handled separately; redundant identity-
            # creating gate on a known-classical post-measurement qubit is just
            # QS002 territory. Here we focus on QS005 + QS007.

            # QS005: adjacent self-inverse pair
            if gate in SELF_INVERSE and cs.last_gate[q] == gate:
                self.report("QS005", "info",
                            f"adjacent '{gate}-{gate}' pair on qubit {q} is an identity "
                            f"(previous at line {cs.last_gate_line[q]}) — likely dead code "
                            f"or a copy-paste slip", line, cs.name)

            # [P3] QS007: a gate whose effect depends on a |0> start applied on a
            # subcircuit qubit that started at TOP and has had no preparing gate.
            if cs.started_top and gate == "h" and cs.qubit_state[q] == AbsState.TOP \
                    and not cs.ever_gated[q]:
                self.report("QS007", "warning",
                            f"qubit {q} of subcircuit '{cs.name}' is used by '{gate}' "
                            f"assuming a |0⟩ start, but as a composed subcircuit its "
                            f"initial state is unknown — superposition prep may be "
                            f"applied to an already-excited qubit", line, cs.name)

            # [P1] phase-tag update
            prev_phase = cs.phase[q]
            if gate in _PHASE_FLIP_GATES:
                cs.phase[q] = flip_phase(prev_phase)
            elif gate == "h":
                # H swaps the X/Z bases; carry phase across deterministically
                cs.phase[q] = prev_phase
            elif gate in ("x", "y"):
                cs.phase[q] = prev_phase
            else:
                cs.phase[q] = Phase.UNKNOWN

            if cs.uf.entangled(q):
                # gate on an entangled qubit: keep entanglement, state stays SUPER
                cs.qubit_state[q] = AbsState.SUPER
            else:
                table = SINGLE_QUBIT_TABLES.get(gate)
                if table is None:  # parametric rotation
                    cs.qubit_state[q] = AbsState.SUPER
                else:
                    cs.qubit_state[q] = table.get(cs.qubit_state[q], AbsState.SUPER)
            self.mark_gate(cs, q, gate, line)

    def apply_controlled(self, cs: CircuitState, call: ast.Call, gate: str,
                         line: int, recv: str) -> None:
        qs = self.resolve_qubits(cs, call.args, recv)
        n_ctrl = CONTROLLED[gate]
        if qs is None or len(qs) < 2:
            self.scoped_degrade(cs, call.args)
            return
        if n_ctrl == -1:
            n_ctrl = len(qs) - 1
        controls, targets = qs[:n_ctrl], qs[n_ctrl:]
        if not targets:
            cs.degrade_all()
            return
        target = targets[0]
        self.check_post_measure(cs, qs, gate, line)
        self.record_pair(cs, gate, tuple(qs), line)  # [P3] QS009

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

        # [P3] QS011: entanglement-not-established. Both control(s) and target
        # are definite, unentangled basis states, and the controlled gate cannot
        # create the entanglement the surrounding code likely expects (e.g.
        # cx where the control is |1> would just flip; cx where control is a
        # Z-basis state and target Z-basis stays separable — no Bell pair).
        if not nondefault_ctrl and gate in ("cx", "cnot", "cz") \
                and all(s in _BASIS_Z and not cs.uf.entangled(c)
                        for s, c in zip(ctrl_states, controls)) \
                and cs.qubit_state[target] in _BASIS_Z \
                and not cs.uf.entangled(target):
            # this is the "tried to make a Bell pair without an H first" pattern:
            # report only when at least one control is |1> (so the gate fires but
            # produces a product state, not entanglement) OR all are |0> (handled
            # by QS001). The |1> case:
            if any(s == AbsState.ONE for s in ctrl_states):
                self.report("QS011", "warning",
                            f"'{gate}' on qubits {qs}: control is a definite basis "
                            f"state and target is a basis state, so no entanglement "
                            f"is created — a separable product results where a "
                            f"Bell/GHZ-type entangled state may have been intended "
                            f"(missing Hadamard on the control?)", line, cs.name)

        # control definitely |1>: gate reduces to its base gate on the target
        if not nondefault_ctrl and all(s == AbsState.ONE and not cs.uf.entangled(c)
               for s, c in zip(ctrl_states, controls)):
            base = {"cx": _X, "cnot": _X, "cy": _Y, "cz": _Z, "ccx": _X,
                    "toffoli": _X, "mcx": _X, "ch": _H}.get(gate, _PHASE)
            if cs.uf.entangled(target):
                cs.qubit_state[target] = AbsState.SUPER
            else:
                if gate == "cz":
                    cs.phase[target] = flip_phase(cs.phase[target])  # [P1]
                cs.qubit_state[target] = base.get(cs.qubit_state[target], AbsState.SUPER)
            self.mark_gate_many(cs, qs, gate, line)
            return

        # control in superposition (or unknown): entangle control(s) and target
        if nondefault_ctrl:
            # inverted-polarity control: be sound, not precise
            cs.qubit_state[target] = AbsState.SUPER
            cs.phase[target] = Phase.UNKNOWN
            for c in controls:
                if cs.qubit_state[c] not in _BASIS_Z or cs.uf.entangled(c):
                    cs.uf.union(c, target)
            self.mark_gate_many(cs, qs, gate, line)
            return
        for c in controls:
            if cs.qubit_state[c] in _LIVE_SUPER or cs.uf.entangled(c):
                cs.uf.union(c, target)
        if cs.uf.entangled(target):
            for q in qs:
                if cs.uf.entangled(q):
                    cs.qubit_state[q] = AbsState.SUPER
        self.mark_gate_many(cs, qs, gate, line)

    def record_pair(self, cs: CircuitState, gate: str, qs: tuple[int, ...],
                    line: int) -> None:
        """[P3] QS009 — qubit-index transposition. If we have already applied the
        same 2-qubit gate to the reversed argument tuple on this circuit, with no
        intervening symmetry justification, the order was likely transposed."""
        if len(qs) == 2:
            rev = (qs[1], qs[0])
            for (g, prev, pline) in cs.pair_history:
                if g == gate and prev == rev and gate not in ("cz", "swap"):
                    # cz and swap are symmetric in their arguments — exclude them
                    self.report("QS009", "warning",
                                f"'{gate}' applied to qubits {list(qs)} after an "
                                f"earlier '{gate}' on the reversed pair {list(rev)} "
                                f"(line {pline}) — control/target may be transposed",
                                line, cs.name)
                    break
        cs.pair_history.append((gate, qs, line))

    def do_reset(self, cs: CircuitState, q: int, line: int = 0) -> None:
        # [P3] QS010: reset on a qubit already provably in |0> and unentangled
        if cs.qubit_state[q] == AbsState.ZERO and not cs.uf.entangled(q) \
                and not cs.measured[q]:
            self.report("QS010", "info",
                        f"reset on qubit {q} which is already in |0⟩ and "
                        f"unentangled — redundant no-op; may signal a "
                        f"misunderstanding of the initial state", line, cs.name)
        partners = cs.uf.split(q)
        for p in partners:
            cs.qubit_state[p] = AbsState.SUPER
        cs.qubit_state[q] = AbsState.ZERO
        cs.phase[q] = Phase.PLUS_PH
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
            if cs.qubit_state[q] in _BASIS_Z and not cs.uf.entangled(q):
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

            # [P3] QS008: missing-correction detection. If q is entangled with a
            # partner that is in a phased superposition (|−⟩ / MINUS_PH) at the
            # moment q is measured, the protocol typically requires a basis or
            # phase correction on the partner after this measurement. Record the
            # obligation; final_checks verifies whether a correction gate fired.
            partners_pre = cs.uf.class_of(q)
            if partners_pre is not None:
                for p in partners_pre:
                    if p == q:
                        continue
                    # Only flag a *correction obligation* when the surviving
                    # partner carries a definite relative Z-phase (|−⟩ or a
                    # MINUS_PH tag). A generic entangled SUPER partner after a
                    # plain Bell measurement needs no correction, so we do NOT
                    # flag it — this keeps QS008 precise (teleportation/
                    # phase-kickback protocols specifically).
                    if cs.qubit_state[p] == AbsState.MINUS \
                            or cs.phase[p] == Phase.MINUS_PH:
                        cs.awaiting_correction[p] = (line, cs.phase[p])

            # split entanglement
            partners = cs.uf.split(q)
            for p in partners:
                cs.qubit_state[p] = AbsState.SUPER   # collapsed component, unknown
            cs.qubit_state[q] = AbsState.CLASSICAL   # [P1] definite 0/1, unknown which
            cs.phase[q] = Phase.PLUS_PH
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
            # [P3] QS008: a correction obligation on q is now being discharged by
            # a gate. Check it is a plausible correction (x/z/y); clear it.
            if q in cs.awaiting_correction:
                if gate in ("x", "z", "y", "cx", "cz", "cy"):
                    cs.awaiting_correction.pop(q, None)  # corrected — OK
                # any other gate leaves the obligation pending
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
        (caller should degrade — register-scoped where possible, see
        scoped_degrade).
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
                    continue  # probably an angle
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
                if all(v is not None for v in vals) and vals:
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
        # [P2] simple binary arithmetic over resolved ints: i+1, n-1, 2*k
        if isinstance(node, ast.BinOp):
            l = self.const_int(node.left)
            r = self.const_int(node.right)
            if l is not None and r is not None:
                if isinstance(node.op, ast.Add):
                    return l + r
                if isinstance(node.op, ast.Sub):
                    return l - r
                if isinstance(node.op, ast.Mult):
                    return l * r
                if isinstance(node.op, ast.FloorDiv) and r != 0:
                    return l // r
            return None
        if isinstance(node, ast.Name) and node.id in self.int_vars:
            return self.int_vars[node.id]
        return None

    # -- end-of-program checks ------------------------------------------------------

    def final_checks(self) -> None:
        for cs in self.circuits.values():
            # QS006: never-measured qubit
            if any(cs.ever_measured) and not all(cs.ever_measured):
                for q in range(cs.n_qubits):
                    if cs.ever_gated[q] and not cs.ever_measured[q]:
                        self.report("QS006", "info",
                                    f"qubit {q} has gates applied but is never measured, "
                                    f"while other qubits are — possible wrong-index bug "
                                    f"or wasted qubit", 0, cs.name)
            # [P3] QS008: any unfulfilled correction obligation at end of circuit
            for q, (mline, ph) in cs.awaiting_correction.items():
                want = "Z-phase" if ph == Phase.MINUS_PH else "basis"
                self.report("QS008", "warning",
                            f"qubit {q} was entangled with a qubit measured at line "
                            f"{mline} and carries a {want} that is never corrected — "
                            f"a conditional correction gate (e.g. X/Z) appears to be "
                            f"missing or wrong (teleportation-style protocol)",
                            mline, cs.name)


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