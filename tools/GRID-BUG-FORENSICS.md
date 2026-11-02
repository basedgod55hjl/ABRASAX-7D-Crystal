# GRID BUG forensics — quantum_evolution_2d

The claim: *"diagnose_quantum_grid documents a discovered GRID BUG: 8x64 mismatch
8.5e-01 FAIL because the loader launches with no grid argument → only one block
dispatched."*

**The bug was real and is fixed. The mechanism in that claim is wrong, and the
tool was still reporting the old failure long after the fix landed.**

Everything below is measured on this box today, by re-running both versions of the
loader against the same numpy reference.

## The docstring's thesis — proven, not argued

The original note says: *"The kernel is correct; the launch dimensions are not."*
That is decidable, so it was decided (`tools/_grid_stress.py`). Same kernel, same
`.ptx`, same 19 kernels bound, same field — only the grid handed to
`cuLaunchKernel` changes:

| launch path | 8x64 max_err | grid dispatched |
|---|---|---|
| pre-fix loader, as shipped | 8.29e-01 | (2, 2, 1) |
| pre-fix loader, **grid forced to ceil-div** | 7.91e-08 | (4, 1, 1) |
| current loader | 7.91e-08 | (4, 1, 1) |

And across all 15 shapes, the forced-grid pre-fix output is **byte-identical**
(SHA-256 of the output buffer) to the current loader's output — 0 of 15 differ.
Same kernel arithmetic, different dispatch geometry. The thesis holds exactly.

## The defect was broader than the one documented case

The square heuristic `side = int(n ** 0.5)` is wrong for every non-square field.
Of 15 shapes tested, the shipped pre-fix loader was wrong on **8**:

```
  1x64     (1,1,1)   want (4,1,1)   8.28e-01
  64x1     (1,1,1)   want (1,4,1)   8.28e-01
  1x100    (1,1,1)   want (7,1,1)   8.28e-01
  8x64     (2,2,1)   want (4,1,1)   8.29e-01
  64x8     (2,2,1)   want (1,4,1)   8.29e-01
  31x33    (2,2,1)   want (3,2,1)   8.28e-01
  33x31    (2,2,1)   want (2,3,1)   8.28e-01
  3x1000   (4,4,1)   want (63,1,1)  8.42e-01
  1000x3   (4,4,1)   want (1,63,1)  8.43e-01
  127x129  (8,8,1)   want (9,8,1)   8.28e-01
```

Only near-square fields survived: 1x1, 15x15, 16x16, 17x17, 256x256, where
`int(sqrt(n))` happens to cover both axes. `8x64` was not the bug — it was the
first case someone looked at.

## Measured, pre-fix loader (`.bak-gridfix-20261101`)

```
  16x16     grid=(1, 1, 1)    want=(1, 1, 1)    max_err=8.115e-08  OK
  32x32     grid=(2, 2, 1)    want=(2, 2, 1)    max_err=8.065e-08  OK
  8x64      grid=(2, 2, 1)    want=(4, 1, 1)    max_err=8.304e-01  GRID MISMATCH
  64x8      grid=(2, 2, 1)    want=(1, 4, 1)    max_err=8.315e-01  GRID MISMATCH
  5x100     grid=(2, 2, 1)    want=(7, 1, 1)    max_err=8.282e-01  GRID MISMATCH

8x64 forensics: max_err=8.3484e-01
bad_columns=[32..63]   (exactly 32 columns, i.e. everything past the 2x16 blocks)
```

`n=512 → side=22 → grid=(2,2,1)`, so x covered 32 of 64 columns and columns
**32..63 were never written**. Two blocks of 16 in x — not one block — plus a
second row in y that an 8-row field never used.

## Measured, current loader

```
  16x16     grid=(1, 1, 1)    max_err=8.115e-08  OK
  32x32     grid=(2, 2, 1)    max_err=8.065e-08  OK
  8x64      grid=(4, 1, 1)    max_err=8.323e-08  OK
  64x8      grid=(1, 4, 1)    max_err=7.899e-08  OK
  5x100     grid=(7, 1, 1)    max_err=7.922e-08  OK
```

The fix: the caller passes `first_array_shape=(h, w)` and the grid is ceil
division per axis. Exact on every one of the 15 stress shapes, including 1x1,
1x100, 3x1000 and 127x129.

## Two draws, same defect

The magnitude depends on the random field, so the gates use fixed seeds but
different draw orders:

| measurement | 8x64 max_err |
|---|---|
| `diagnose_quantum_grid.py --legacy` (first draw) | 8.348e-01 |
| `_check_grid_bug_history.py` (sixth draw) | 8.735e-01 |
| `_grid_stress.py` (seed 7) | 8.294e-01 |
| documented 2026-11-01 | 8.5e-01 |

All reproduce ~8.5e-01, and all put the damage in exactly columns 32..63 — the
*shape* of the failure is seed-independent, which is what the gates assert.

## What was wrong with the tool

`diagnose_quantum_grid.py` asserted the *old* behaviour:

```python
ok = covered.max() < 2e-5 and uncovered.max() > 1e-3
```

so after the fix it printed `NOT CONFIRMED` and **exited 1**. A forensic tool that
turns red when the bug is fixed is a false alarm generator — wire it into CI and
the green build looks broken.

Rewritten as two modes, both exiting 0 only when their own claim holds:

```
python diagnose_quantum_grid.py            regression: 5 shapes exact AND the
                                           dispatched grid == ceil-div(shape)
python diagnose_quantum_grid.py --legacy   forensics: pre-fix loader still
                                           reproduces ~8.5e-01, bad columns 32..63
```

The regression mode records the grid the loader actually dispatches (signature-
agnostic wrapper around `_get_grid_block`), so a future heuristic regression fails
on the grid values immediately instead of after the numerics drift. Its hardcoded
`C:\abrasax` path is gone too.

## Gates

| tool | proves |
|---|---|
| `diagnose_quantum_grid.py` | current grids == ceil division, numerics exact |
| `diagnose_quantum_grid.py --legacy` | the historical failure, reproduced from the backup loader |
| `_grid_stress.py` | 16 checks: 15 shapes, kernel-vs-geometry isolation, bit-identity |
| `_check_grid_bug_history.py` | pre-fix vs current side by side, 8 checks incl. non-vacuity |

All four are stages 9–12 of `python tools/verify_all.py` → **18 stages, 18 pass,
0 fail**.
