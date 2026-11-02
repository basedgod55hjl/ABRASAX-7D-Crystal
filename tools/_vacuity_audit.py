"""Do the gates actually execute what they claim, or do they pass vacuously?

A gate can be green three ways: it ran the check and it passed, it skipped
everything, or it asked about files that are not there. verify_gitignore.py was
the third kind until it was pointed at a parent directory -- "0 misconfigured"
while 16 of 17 paths were absent. This audits for that class of failure in every
tool, by demanding evidence of coverage, not just the absence of failure.

  A  STATIC   flag absence-tolerating logic: `if not path.exists(): return True`
              style branches, where a missing input becomes a pass
  B  RUNTIME  run each gate and require BOTH the success string AND a minimum
              amount of evidence (kernels bound, shapes exercised, controls run,
              rules attributed), so "0 checks executed, 0 failed" cannot pass
  C  SWEEP    flag skip/absent lines that are not annotated as expected

Note on C: a bare substring match on "absent" is the same trap that made "0 fail"
match "10 fail". Skip lines are matched per line and must either carry an
allowance (an explicit "expected", "note:", or a documented-field name) or be
reported.

Exit 0 only if every gate produced its expected evidence in the expected amount.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

# script, args, must-contain strings, {regex: minimum number of matches}
GATES = [
    ("verify_kernel_math.py", [], ["0 failed"],
     {"kernels bound: 19": 1, r"OK\s+manifold_projection_7d": 4,
      r"OK\s+holographic": 3, "sensitivity": 1}),
    ("verify_kernel_math_quantum.py", [], ["0 failed"],
     {"kernels bound: 19": 1, r"OK\s+mhq7d_attention": 3,
      r"OK\s+quantum_evolution_2d": 5, "distinguishes multiplier": 3}),
    ("diagnose_quantum_grid.py", [], ["0 shape(s) failed"],
     {r"grid=\(": 5, "want=": 5}),
    ("_grid_stress.py", [], ["0 fail"],
     {"bit-identical": 1, "checks: 16": 1, r"8x64": 2}),
    ("_check_grid_bug_history.py", [], ["0 fail"], {"checks: 8": 1}),
    ("verify_gitignore.py", [], ["0 misconfigured"],
     {"must-be-committable": 17, "must-be-ignored": 10, "note: ": 0}),
    ("_audit_verify_gitignore.py", [], ["0 fail"], {"checks: 16": 1}),
    ("_audit_loader_location.py", [], ["0 fail"], {"checks: 11": 1, "kernels=19": 5}),
    ("_loader_requirements.py", [], ["0 fail"], {"checks: 8": 1}),
    ("_phantom_imports.py", [], ["phantom=0"], {}),
    ("_static_audit.py", [], ["fatal=0"], {r"pyc=fresh": 30}),
    ("_self_test.py", [], ["0 fail"], {"controls: 6": 1}),
    ("_offline_controls.py", [], ["0 fail"], {"controls: 6": 1}),
]

SKIP_WORDS = re.compile(r"\b(?:absent|skipped|SKIP|MISSING|not found|<none>|"
                        r"no output|NOT CONFIRMED)\b")
# a skip-ish line is acceptable when it says why it is fine
ALLOWED = [
    re.compile(r"pyc_absent=\d+"),               # a counted field, not a skip
    re.compile(r"absent \(PTX path used\)"),     # documented optional artifact
    re.compile(r"\(expected"),                   # deliberate failure in a control
    re.compile(r"^\s*note: "),                   # the tool's own accounting line
    re.compile(r"are absent"),                   # section D of the loader audit
    re.compile(r"absent: "),                     # its explicit list
    re.compile(r"absent on disk\]"),             # per-path annotation
    re.compile(r"^\s*(cupy|torch|pycuda|numba|pyopencl)\s+absent"),  # section D table
    re.compile(r"pyc=absent"),                   # no bytecode yet, not a skip
]


def unmatched_skip_lines(out):
    bad = []
    for line in out.splitlines():
        if not SKIP_WORDS.search(line):
            continue
        if any(a.search(line) for a in ALLOWED):
            continue
        bad.append(line.strip()[:70])
    return bad


def run(script, args, timeout=420):
    path = os.path.join(HERE, script)
    proc = subprocess.run([PY, path] + args, cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def static_scan():
    """Absence-tolerating logic: missing input treated as a pass."""
    findings = []
    for name in sorted(f for f in os.listdir(HERE) if f.endswith(".py")):
        if name.startswith("_patch"):
            continue
        path = os.path.join(HERE, name)
        try:
            tree = ast.parse(open(path, "rb").read(), name)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            src = ast.unparse(node.test)
            if "exists()" not in src and "is_file()" not in src:
                continue
            if "not " not in src[:6]:
                continue
            body = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
            if re.search(r"return (True|0)\b|continue\b|return None\b", body):
                findings.append("%s:%d  if %s -> %s"
                                % (name, node.lineno, src[:44],
                                   body.replace("\n", "; ")[:44]))
    return findings


def main():
    cases = []
    print("== A. static: absence-tolerating branches ==")
    findings = static_scan()
    for line in findings:
        print("  REVIEW  %s" % line)
    if not findings:
        print("  none")

    print("\n== B/C. runtime: evidence per gate ==")
    for script, args, musts, minimums in GATES:
        try:
            rc, out = run(script, args)
        except subprocess.TimeoutExpired:
            cases.append((script, False, "timeout"))
            print("  [FAIL] %-32s timeout" % script)
            continue
        missing = [m for m in musts if m not in out]
        short = []
        for pattern, want in minimums.items():
            got = len(re.findall(pattern, out))
            if got < want:
                short.append("%s %d<%d" % (pattern, got, want))
        unexplained = unmatched_skip_lines(out)
        ok = rc == 0 and not missing and not short and not unexplained
        if missing:
            detail = "missing string(s): %s" % missing
        elif short:
            detail = "thin evidence: %s" % "; ".join(short)
        elif unexplained:
            detail = "%d unexplained skip line(s): %s" % (
                len(unexplained), unexplained[0])
        else:
            detail = "%d evidence line(s)" % len(
                [ln for ln in out.splitlines() if ln.strip()])
        cases.append((script, ok, detail))
        print("  [%s] %-32s %s" % ("PASS" if ok else "FAIL", script, detail))

    bad = [c for c in cases if not c[1]]
    print("\n%d gates: %d passed with evidence, %d failed"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
