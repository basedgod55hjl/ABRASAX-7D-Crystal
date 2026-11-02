# Reproduce this session

Everything below runs on this box, offline, with no keys and no network. One
command does the whole ladder; the table underneath maps each claim to the single
command that proves it.

## The whole thing

```
cd C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal
tools\repro.cmd            :: full run, ~47 s, exit 0 == everything passed
tools\repro.cmd fast       :: static half only, ~20 s
```

```
============================================================
 ABRASAX-7D-Crystal reproduction   Sun 11/01/2026 15:04:44
 root: C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal
 log:  C:\Users\lucar\AppData\Local\Temp\7d_repro_58328697.log
============================================================

--- manifest (SHA-256 of every claimed artifact) ---
  OK   exit=0
--- verify_all ---
  OK   exit=0
--- verify_offline (network blocked) ---
  OK   exit=0

============================================================
 RESULT: ALL STAGES PASSED
============================================================
```

The transcript goes to `%TEMP%\7d_repro_<random>.log`, so a failing run can be
handed over whole. Nothing is written inside the repo.

## Claim → command → what to look for

| # | claim | command | expected |
|---|---|---|---|
| 1 | the nine original tools compile | `python tools\_compile_gate.py` | `failures=0` |
| 2 | every .py in the tree compiles | `python tools\_tree_gate.py .` | `files=N failures=0` (every .py in the tree) |
| 3 | no phantom imports remain | `python tools\_phantom_imports.py` | `sites=0 phantom=0` |
| 4 | every import resolves or is guarded | `python tools\_import_matrix.py` | `core_fatal=0 … external_only=28` |
| 5 | `cuda_kernel_loader` exists twice; the stub binds nothing | `python tools\_check_loader_shadow.py` | stub `kernel_count=0`, real `19` |
| 6 | `abrasax_pipeline` imports, fails loud on use | `python tools\_check_phantom_fix.py` | `4 checks: 4 pass` |
| 7 | grid bug: live state | `python tools\diagnose_quantum_grid.py` | `0 shape(s) failed` |
| 8 | grid bug: historical, columns 32..63 | `python tools\diagnose_quantum_grid.py --legacy` | `CONFIRMED` |
| 9 | kernel correct, geometry was not | `python tools\_grid_stress.py` | `16 checks: 16 pass`, `bit-identical … 0/15 shapes differ` |
| 10 | pre-fix vs current, 8 checks | `python tools\_check_grid_bug_history.py` | `8 checks: 8 pass` |
| 11 | gitignore tool is stdlib, safe, can fail | `python tools\_audit_verify_gitignore.py` | `13 checks: 13 pass` |
| 12 | gitignore current state | `python tools\verify_gitignore.py` | `0 misconfigured` |
| 13 | GPU gates need no cupy/torch | `python tools\_loader_requirements.py` | `8 checks: 8 pass` |
| 14 | GPU kernel numerics | `python tools\verify_kernel_math.py` | `0 failed`, errs ~7e-08..4e-07 |
| 15 | mH-Q quantum + multiplier discrimination | `python tools\verify_kernel_math_quantum.py` | `0 failed`, 12 OK lines |
| 16 | loader phases 1-5 | `python kernels\cuda\test_kernel_loader.py --quick` | `ALL TESTS PASSED` |
| 17 | artifacts are the verified ones | `python tools\verify_manifest.py` | `entries=N ok=N changed=0 missing=0 extra=0` (N grows as files are added) |
| 18 | no gate reaches for the network | `python tools\verify_offline.py` | `ARMED`, `13 executed … 0 fail` |
| 19 | the gates themselves can fail | `python tools\_self_test.py` | `6 controls: 6 pass` |
| 20 | sandbox and manifest controls | `python tools\_offline_controls.py` | `6 controls: 6 pass` |

## Reading the numbers honestly

* **Exit codes are the verdict.** Every tool exits 0 only when its own claim holds;
  the text is evidence, not decoration.
* **Skips are named, never silent.** `verify_offline` prints which stages the box
  cannot support and says so in the summary: a green there does not cover a skip.
* **`10 fail` is not `0 fail`.** The aggregate runner matches `", 0 fail"` for
  exactly this reason — a substring bug that made a passing run look like a
  catastrophe.
* **What a green does NOT cover:** the 28 `torch`/`tvm` imports under
  `kernels/gpu_compute/**` (optional backends, absent here, reported as
  `scope=external`), and anything requiring a different machine or a fresh
  checkout. The manifest proves *which files*, not that another box can build them.
* **`0.85` is not a magic constant.** The pre-fix loader's 8x64 error measures
  8.29e-01 / 8.35e-01 / 8.74e-01 depending on the random field. What is
  seed-independent — and what the gates assert — is *where* the damage lands
  (columns 32..63) and *why* (grid (2,2,1) instead of (4,1,1)).

## If something goes red

1. `static` or `manifest` red right after an edit → you skipped the housekeeping:
   `python -m compileall -q -f .` then `python tools\verify_manifest.py --update`.
   That is the gates working, not a broken tree.
2. `loader` stage red → check `nvcuda.dll` and a device: `python tools\verify_offline.py --matrix`.
3. `grid_regress` red with `GRID MISMATCH` → the grid heuristic regressed; the
   failing shape names the axis.
4. `gitignore` exits 3 → the path you gave is not the 7D-Crystal subtree (the
   marker `kernels/cuda/crystal_cortex.h` was not found, and no
   `SYSTEMS/ABRASAX-7D-Crystal` underneath it either). It refuses instead of
   reporting a vacuous pass -- pointing it at `C:\abrasax` used to "succeed"
   while 16 of the 17 committable paths did not exist at that level.
5. Anything else -> the log file has the full transcript including which control
   failed and with what message.
