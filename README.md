# QShield-Static Prototype — Abstract State Analyzer for Qiskit

A working prototype of the QShield-Static design: abstract interpretation of
per-qubit quantum state over Qiskit programs, targeting silent Output Wrong
bugs that pattern-based linters (LintQ) cannot see.

## Design → implementation mapping

| Research design element | Where it lives |
|---|---|
| Finite abstract domain `|0⟩, |1⟩, |+⟩, |−⟩, MIXED(⊤), BOT(⊥)` | `AbsState` enum |
| Per-gate monotone transfer functions (H, X, Y, Z, S/T, CX, CZ, CCX, ...) | `SINGLE_QUBIT_TABLES`, `apply_controlled` |
| Union-find entanglement with `split` on measurement | `EntanglementUF` |
| Lattice join at control-flow merges | `CircuitState.join_with`, `handle_if` |
| Fixpoint iteration over loops (finite lattice ⇒ termination) | `handle_loop` (capped at 12 iters defensively) |

This prototype works over the Python AST directly rather than CodeQL — same
abstraction layer, lighter substrate — so you can run it today with zero
setup. The transfer functions and domain port directly to the CodeQL/QShield
architecture later.

## Checks

- **QS001 NoOpControlledGate** — controlled gate whose control is provably
  `|0⟩` (the reversed-CNOT Bell-state bug; includes a "did you mean
  cx(t, c)?" hint when the target is in superposition)
- **QS002 OpAfterMeasurement** — gate on an already-measured qubit
- **QS003 ConstantMeasurement** — measurement with a compile-time-constant outcome
- **QS004 DoubleMeasurement** — re-measurement with no intervening gate
- **QS005 IdentityGatePair** (info) — adjacent self-inverse pair (H-H, X-X)
- **QS006 NeverMeasuredQubit** (info) — gated but never measured while others are

## Usage

```bash
python3 qshield_static.py path/to/programs/            # scan a directory
python3 qshield_static.py file.py --json findings.json # single file + JSON
python3 bugs4q_differential.py path/to/Bugs4Q-Framework/qiskit
```

The differential script runs every buggy/fixed pair and reports findings
present in buggy but absent in fixed (robust to line shifts).

## Evaluation on Bugs4Q (42 programs)

**Differential detections (warning in buggy, gone in fixed): 2/42**

- **Bug 39** — missing initial Hadamard layer. The analyzer proves the whole
  circuit dead: 6× QS001 (every CNOT control stuck at `|0⟩`) + 4× QS003
  (all measurement outcomes constant). The fixed version's `for i in
  range(4): qc.h(i)` loop is handled by fixpoint iteration and is clean.
  Pure silent Output Wrong bug — invisible to LintQ.
- **Bug 21** — teleportation with corrections written as plain `cx`/`cz`
  after measurement instead of classically-conditioned ops. 2× QS002.

**Both-versions findings (bugs 6, 14, 17, 20, 25, 29, 31, 33):** warnings
that persist across buggy and fixed are properties of the program family,
not the defect — e.g., a deterministic ancilla measured in both versions
(QS003), or a genuine subcircuit pattern. Classify these manually: some are
benign true statements about the code, some are precision losses. Two
precision frontiers already handled:

1. `ctrl_state=` keyword inverts control polarity → QS001/|1⟩-specialization
   suppressed when present (this killed a spurious hit on bug 1, whose actual
   defect *is* ctrl_state argument handling).
2. Circuits constructed inside function bodies are treated as composable
   subcircuits and start at `⊤`, not `|0...0⟩` (killed spurious hits on
   bugs 14*/41). *Bug 14's circuit is top-level, so it still fires — worth a
   manual look.



## Known limitations (a.k.a. future work)

- No interprocedural composition (`qc.compose(sub)` degrades to ⊤)
- Unresolvable qubit indices degrade the whole circuit to ⊤ (sound, imprecise)
- Phase gates map `|±⟩` to ⊤ (domain has no Y-basis / phase labels)
- Parametric rotations always go to ⊤ (no angle reasoning, so e.g.
  `rx(π)` isn't recognized as X)
- `c_if` / dynamic circuits not modeled
