# QShield-Static

A quantum-aware static analyzer for Qiskit programs. QShield-Static performs
**abstract interpretation** over a finite per-qubit state domain to catch
*silent, state-semantic* bugs — the kind that produce a wrong output without
ever raising an error or tripping a syntactic linter.

It is designed to run alongside [LintQ](https://github.com/lintq) as a
complementary semantic layer: where LintQ flags structural/API misuse,
QShield targets bugs that require reasoning about what state each qubit is
actually in. The two tools detect disjoint bug classes.

## How it works

QShield never simulates the circuit. Instead it interprets the program over a
small, finite lattice, which guarantees termination:

- **Abstract domain.** Each qubit carries a basis label drawn from
  `BOT < {ZERO, ONE, PLUS, MINUS} < {SUPER, CLASSICAL} < TOP` (lattice height 3).
  `SUPER` is a live coherent superposition; `CLASSICAL` is a post-measurement
  definite-but-unknown 0/1; `TOP` is their join (no information). `MIXED` is
  kept as an alias of `SUPER` for backward compatibility.
- **Phase tag.** A separate flat 3-point phase lattice (`PLUS_PH`, `MINUS_PH`,
  `UNKNOWN`) is carried alongside each basis label so diagonal gates (Z, S, T)
  advance/flip phase instead of collapsing `|+⟩`/`|−⟩` to top. This keeps
  missing phase corrections visible.
- **Transfer functions.** One monotone transfer function per gate
  (`h, x, y, z, s, t, cx, cz, ccx, swap, reset, measure`, …).
- **Entanglement tracking.** A union-find structure with a `split` operation:
  on measurement the measured qubit leaves its class and survivors degrade.
- **Control flow.** Lattice join at merges; fixpoint iteration over loops
  (capped defensively). Small statically-known `range(...)` loops are unrolled
  and joined; unknown ranges fall back to fixpoint.
- **Resolution.** Constant propagation over int variables (including `i += 1`,
  simple unpacking, and arithmetic like `i+1`, `n-1`, `2*k`), register-offset
  mapping for `qreg[i]` subscripts, and **register-scoped degradation** — when
  an index is unresolvable, only the affected register goes to `TOP` rather
  than the whole circuit.

## Detection rules

Silent / output-wrong oriented (the class LintQ cannot see):

| Code  | Severity | What it catches |
|-------|----------|-----------------|
| QS001 | warning  | Controlled gate whose control is definitely `\|0⟩` (no-op). Suppressed when `ctrl_state=` inverts polarity. |
| QS002 | warning  | Quantum gate applied to an already-measured qubit. |
| QS003 | warning  | Measuring a qubit in a definite basis state (constant outcome). |
| QS004 | warning  | Measuring the same qubit twice with no gate in between. |
| QS005 | warning  | Adjacent self-inverse gate pair (H-H, X-X, …). |
| QS006 | info     | Qubit acted on but never measured while others are — possible wrong-index bug. |
| QS007 | warning  | Gate whose meaning depends on a `\|0⟩` start applied to a `SUPER`/`TOP` qubit (function-body circuits start at `TOP`). |
| QS008 | warning  | Teleportation-style missing/wrong correction gate, detected via phase tag + entanglement remnants. |
| QS009 | warning  | 2-qubit gate applied with statically-reversed arguments vs. a prior gate on the same pair (e.g. `cx(1,0)` after `cx(0,1)`). |
| QS010 | info     | `reset()` on a qubit provably already in `\|0⟩` and unentangled. |
| QS011 | warning  | Controlled gate meant to spread superposition applied when both qubits are definite basis states and unentangled. |

## Usage

```bash
python qshield_static.py <file-or-directory> [more paths...] [--json out.json] [-q]
```

It scans every `*.py` file under each given path, analyzes each
`QuantumCircuit` it can resolve, and prints findings as
`file:line: [CODE/severity] (circuit) message`.

Options:

- `--json OUT` — also write findings to `OUT` as JSON (SARIF-compatible
  structure), suitable for unified output alongside LintQ.
- `-q`, `--quiet` — print the summary line only.

Exit code is `1` if any warnings were found, else `0`.

### Example

```bash
# analyze a single file
python qshield_static.py bell_state.py

# scan a whole benchmark directory and emit JSON
python qshield_static.py Bugs4Q/ --json qshield_findings.json
```

## Requirements

- Python 3.8+ (uses only the standard library: `ast`, `argparse`, `json`,
  `dataclasses`, `enum`, `pathlib`).
- No Qiskit installation or circuit execution required — analysis is purely
  static.

## Output format

Each finding is a record with `code`, `severity` (`warning`/`info`),
`message`, `file`, `line`, and `circuit`. The text printer sorts findings by
file, then line, then severity, and ends with a summary tallying warnings,
info findings, and a per-code count.
