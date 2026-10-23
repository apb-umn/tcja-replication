#!/usr/bin/env python3
"""Compare the paper/Inputs CSVs you just built with the authors' outputs.

    python3 compare_to_reference.py [--ref reference/clean_rebuild/paper_inputs]

Every CSV that tcja_draft5.tex loads is compared cell by cell (numeric cells
with relative tolerance 1e-6). 'clean_rebuild' is what this package produced on
the authors' machine; 'nber_draft' is the set behind the PDF sent to NBER
(see README, section 6, for the known differences between the two).
"""
import argparse, re, os, pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="reference/clean_rebuild/paper_inputs"); a = ap.parse_args()
tex = open("paper/tcja_draft5.tex").read(); pre = tex[: tex.index("\\begin{document}")]
live = sorted(set(re.findall(r"\\DTLloaddb\{\w+\}\{\\Inputs/([^}]+)\}", pre)) | set(re.findall(r"\\pgfplotstableread(?:\[[^]]*\])?\{\\Inputs/([^}]+)\}", tex)))
tally = {}
for f in live:
    mine, ref = os.path.join("paper/Inputs", f), os.path.join(a.ref, f)
    if not os.path.exists(mine): print(f"MISSING   {f}"); tally["MISSING"] = tally.get("MISSING", 0) + 1; continue
    A = pd.read_csv(ref, dtype=str, keep_default_na=False); B = pd.read_csv(mine, dtype=str, keep_default_na=False)
    if A.shape != B.shape or list(A.columns) != list(B.columns):
        print(f"SHAPE     {f}: reference {A.shape} vs yours {B.shape}"); tally["SHAPE"] = tally.get("SHAPE", 0) + 1; continue
    diffs = []
    for c in A.columns:
        for i, (x, y) in enumerate(zip(A[c], B[c])):
            if x == y: continue
            try:
                if np.isclose(float(x), float(y), rtol=1e-6, atol=1e-9): continue
            except ValueError: pass
            diffs.append(f"row {i} {c}: {x} -> {y}")
    if diffs: print(f"DIFF      {f}: {len(diffs)} cells, e.g. " + "; ".join(diffs[:3])); tally["DIFF"] = tally.get("DIFF", 0) + 1
    else: print(f"IDENTICAL {f}"); tally["IDENTICAL"] = tally.get("IDENTICAL", 0) + 1
print("summary:", tally)
