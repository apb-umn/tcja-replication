#!/usr/bin/env python3
"""
build_category_diagnostics.py

Diagnostics for Section 2: the assignment of reconciliation rows to the seven
channels of the paper's taxonomy (stage 3), and the agreement between the
effective tax rate implied by the extracted table and Compustat.

The assignment itself lives in build_etr_figures.py: classify() maps each
filer-written row label to one of ~30 fine categories by an ordered regular-
expression cascade (first match wins, OTHER as residual), and group_map folds
those into the seven channels. This script reports what that mapping does, so
the appendix can state it rather than assert it.

Writes paper/Inputs/tables/category_diagnostics.csv (key,value).
"""

import os
import pickle
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
LONG_PKL = os.path.join(ROOT, "data", "analysis", "cache", "long.pkl")
ETR_SRC = os.path.join(SCRIPT_DIR, "build_etr_figures.py")
OUT = os.path.join(ROOT, "paper", "Inputs", "tables", "category_diagnostics.csv")

YEARS = (2013, 2024)
CLIP = 15.0
# labels that name a foreign concept but are deliberately left in the residual
FOREIGN_TOKEN = re.compile(
    r"foreign|non-?u\.?s|internat|overseas|gilti|fdii|beat|subpart|withhold|repatriat", re.I
)


def main():
    src = open(ETR_SRC, encoding="utf-8").read()
    ns = {"re": re, "pd": pd}
    exec(src[src.index("group_map = {"):src.index("def add_tcja_line")], ns)
    n_rules = len(re.findall(r"re\.search", src[src.index("def classify"):src.index("def add_tcja_line")]))
    n_categories = len(set(ns["group_map"].keys()))

    L = pickle.load(open(LONG_PKL, "rb"))
    d = L[L.recon_year.between(*YEARS)].copy()
    sale = d.groupby(["gvkey", "recon_year"])["comp_sale"].first().rename("sale")

    # Channel shares follow the paper's construction exactly: sum the rows
    # assigned to a category within a firm-year, THEN winsorize that sum at
    # +/-15pp (build_etr_figures.build_driver_panel). Clipping row by row
    # instead would overstate the residual.
    g = (d.groupby(["gvkey", "recon_year", "group"])["cat_val"].sum()
           .reset_index(name="contrib"))
    g["cw"] = g["contrib"].clip(-CLIP, CLIP)
    g = g.join(sale, on=["gvkey", "recon_year"])
    g = g[g["sale"] > 0]
    tot_e = g["cw"].abs().sum()
    tot_s = (g["cw"].abs() * g["sale"]).sum()
    share = {}
    for grp, sub in g.groupby("group"):
        share[grp] = (100 * sub["cw"].abs().sum() / tot_e,
                      100 * (sub["cw"].abs() * sub["sale"]).sum() / tot_s)

    # The within-residual diagnostic is a row-level question (which labels
    # carry a foreign concept), so it stays at row level.
    d["w"] = d["cat_val"].clip(-CLIP, CLIP)

    n_fy = d.groupby(["gvkey", "recon_year"]).ngroups
    n_intl = d[d.group == "International"].groupby(["gvkey", "recon_year"]).ngroups
    o = d[d.category == "OTHER"]
    hits = o[o.raw_label.astype(str).str.contains(FOREIGN_TOKEN, na=False)]

    # Validation: extracted ETR vs. Compustat TXT/PI on linked firm-years.
    comb = pickle.load(open(os.path.join(ROOT, "data", "analysis", "cache", "combined.pkl"), "rb"))
    v = comb[(comb["comp_pi"] > 1)].copy()
    v["etr_comp"] = 100 * v["comp_txt"] / v["comp_pi"]
    v = v[v["etr_w"].notna() & v["etr_comp"].notna() & v["etr_comp"].between(-100, 100)]
    diff = (v["etr_w"] - v["etr_comp"]).abs()

    # Share of panel firm-years recovered from a prior-year (comparative)
    # column of a later filing rather than the filing's own year.
    off = comb["year_offset_from_filing"]
    out_off = {
        "panelOwnYearN": int((off == 0).sum()),
        "panelPriorYearN": int((off < 0).sum()),
        "panelPriorYearPct": 100 * (off < 0).mean(),
    }

    out = {
        **out_off,
        "etrAgreeHalfPct": 100 * (diff <= 0.5).mean(),
        "etrAgreeMedianAbs": diff.median(),
        "etrAgreeN": len(v),
        "catDistinctLabels": L.raw_label.nunique(),
        "catNumRules": n_rules,
        "catNumCategories": n_categories,
        "catFirmYears": n_fy,
        "catNoIntlFyPct": 100 * (1 - n_intl / n_fy),
        "catOtherEqualPct": share["Other"][0],
        "catOtherSalesPct": share["Other"][1],
        "catIntlEqualPct": share["International"][0],
        "catIntlSalesPct": share["International"][1],
        "catOtherForeignRowPct": 100 * len(hits) / len(o),
        "catOtherForeignPpPct": 100 * hits["w"].abs().sum() / o["w"].abs().sum(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("key,value\n")
        for k, v in out.items():
            f.write(f"{k},{float(v):.6f}\n")
    print(f"wrote {OUT} ({len(out)} keys)")
    for k, v in out.items():
        print(f"  {k:<24s} {v:12,.2f}")


if __name__ == "__main__":
    main()
