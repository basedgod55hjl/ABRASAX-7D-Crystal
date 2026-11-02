# Verification you can do offline

No network, no API keys, no accounts. Python 3.11 + numpy is enough for the static
half; the GPU gates need the same box's CUDA driver. Everything here runs on the
machine itself.

## The one-liner

```
cd C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal
python tools\verify_all.py
```

Expected tail:

```
21 stages: 21 pass, 0 fail, 0 skip
```

Exit code 0 means every gate passed. Nonzero names the failing stage. `--fast`
skips the CUDA stages; `--no-selftest` skips the self test.

## If you want to check without trusting any of my code paths

```
python tools\verify_manifest.py
```

Rebuilds SHA-256 of all 34 covered files (the tools, the loader, the two source
files this work changed) and compares. Works with **zero dependencies** — no
numpy, no GPU, no git:

```
entries=34 ok=34 changed=0 missing=0 extra=0
```

`CHANGED` / `MISSING` / `EXTRA` lines name exactly what moved. If someone hands you
this tree, that command is the cheapest proof it is the tree I verified.

## If you want proof nothing here touches the network

```
python tools\verify_offline.py
```

It builds a throwaway `sitecustomize.py` that replaces `socket.connect`,
`socket.connect_ex`, `socket.create_connection` and `urllib.request.urlopen` with a
raiser, then runs every stage inside that sandbox. The first thing it prints is
whether the sandbox is actually armed — and a probe stage proves that, by making
the same call outside the sandbox and confirming it behaves *differently*:

```
== capability matrix ==
  python   3.11.9
  numpy    2.4.6
  cuda     19 kernels bound
  git      ...\git.EXE
  nvcc     ...\nvcc.EXE

== sandbox ==
  ARMED  SANDBOX_ARMED

== stages (offline, network blocked) ==
  [PASS] manifest       entries=34 ok=34 changed=0 missing=0 extra=0
  ...
13 stages: 13 executed (13 pass, 0 fail), 0 skipped for capability
```

The capability matrix exists so a green result cannot be misread: if a machine has
no CUDA device, the GPU stages are **named as skipped**, not silently passed, and
the summary says the skip is outside the coverage.

`python tools\verify_offline.py --matrix` prints just the capability matrix.

## Individual claims, each one command

| claim | command | expected |
|---|---|---|
| the 9 original tools compile | `python tools\_compile_gate.py` | `failures=0` |
| the whole tree compiles | `python tools\_tree_gate.py .` | `failures=0` (66+ files) |
| no phantom imports anywhere | `python tools\_phantom_imports.py` | `sites=0 phantom=0` |
| every import resolves or is guarded | `python tools\_import_matrix.py` | `core_fatal=0` |
| `abrasax_pipeline` imports and fails loud | `python tools\_check_phantom_fix.py` | `4 checks: 4 pass` |
| stub vs real loader | `python tools\_check_loader_shadow.py` | stub 0 kernels, real 19 |
| the grid bug, live | `python tools\diagnose_quantum_grid.py` | `0 shape(s) failed` |
| the grid bug, historically | `python tools\diagnose_quantum_grid.py --legacy` | `CONFIRMED` |
| kernel correct, geometry was not | `python tools\_grid_stress.py` | `16 checks: 16 pass`, bit-identical |
| the gates themselves can fail | `python tools\_self_test.py` | `6 controls: 6 pass` |
| the sandbox and manifest can fail | `python tools\_offline_controls.py` | `6 controls: 6 pass` |

## What cannot be verified offline here

* `kernels/gpu_compute/**` — imports `torch` and `tvm`, absent on this box. 28
  sites, reported as `scope=external` by `_import_matrix.py`. Nothing in the core
  crystal path imports them.
* Anything requiring a second machine, a fresh checkout, or a clean CUDA install.
  The manifest proves *which files*, not that a different box can build them.

## Reading the numbers honestly

`0.85` is not a magic constant. The pre-fix loader's 8x64 error measures
8.29e-01 / 8.35e-01 / 8.74e-01 depending on the random field; what is
seed-independent — and what the gates assert — is *where* the damage lands
(columns 32..63) and *why* (grid (2,2,1) instead of (4,1,1)).
