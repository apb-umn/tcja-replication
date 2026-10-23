#!/usr/bin/env python3
"""
Stage the profit-shifting heterogeneity estimates the draft asserts in prose.

Section 3.2.2 claims that the profit-location response is concentrated in
R&D-intensive and cash-rich firms and is not related to tax-haven presence,
and that domestic profits rise as foreign profits fall.  Those claims rest on
estimates that were never reported anywhere.  This stages them so the appendix
table and the text can be macro-driven like the rest of the paper.

Sources (regenerate them before running this):
  analysis/analyze_profit_shifting_mechanisms.py
  analysis/build_post_window_variants.py

Outputs:
  tables/heterogeneity_gamma.csv          gamma-hat by Z, for the appendix table
  tables/heterogeneity_headlines.csv      scalars quoted in the text
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


INPUTS = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = INPUTS / "tables"
ANALYSIS_OUT = REPO_ROOT / "data" / "analysis" / "output"
GENERATED_AT = datetime.now().astimezone().replace(microsecond=0).isoformat()

# The characteristics the text actually talks about, in the order the appendix
# table reports them: the two that predict the response, then the haven
# measures that do not, then scale as a reference.
PREDICTORS = [
    ("pre_rd_at_z", "R\\&D / assets"),
    ("pre_cash_at_z", "Cash / assets"),
    ("pre_intan_ex_gw_at_z", "Intangibles ex-goodwill / assets"),
    ("pre_foreign_rev_share_z", "Foreign revenue share"),
    ("pre_has_haven_or_conduit", "Any haven or conduit subsidiary"),
    ("pre_has_classic_haven", "Any classic-haven subsidiary"),
    ("pre_log1p_haven_subs_z", "Log haven/conduit subsidiary count"),
    ("pre_log_at_z", "Log assets"),
]


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main() -> None:
    oneway = pd.read_csv(ANALYSIS_OUT / "profit_shifting_mechanism_oneway_foreign_share.csv")
    by_pred = {r["predictor"]: r for _, r in oneway.iterrows()}

    rows = []
    for order, (key, label) in enumerate(PREDICTORS, start=1):
        if key not in by_pred:
            # [replication package] predictor not estimated (optional Compustat Segments /
            # Exhibit 21 data absent): emit the row with blank estimates. See README 4.3.
            print(f"  NOTE: {key} not estimated; Table B.2 row left blank")
            rows.append({"row_order": order, "row_key": key, "row_label": label, "binary": "", "gamma_pp": "",
                         "gamma_se_pp": "", "gamma_p": "", "stars": "", "n_obs": "", "n_firms": ""})
            continue
        r = by_pred[key]
        rows.append(
            {
                "row_order": order,
                "row_key": key,
                "row_label": label,
                "binary": bool(r["binary"]),
                # gamma is on the foreign-share scale; x100 gives percentage points
                "gamma_pp": 100.0 * float(r["interaction_gamma"]),
                "gamma_se_pp": 100.0 * float(r["interaction_se"]),
                "gamma_p": float(r["interaction_p_norm"]),
                "stars": stars(float(r["interaction_p_norm"])),
                "n_obs": int(r["n"]),
                "n_firms": int(r["firms"]),
            }
        )
    gamma = pd.DataFrame(rows)
    gamma.to_csv(OUT / "heterogeneity_gamma.csv", index=False, float_format="%.6f")

    # ---- scalars quoted in the text ------------------------------------
    pwv = pd.read_csv(REPO_ROOT / "data" / "analysis" / "output" / "robustness" / "post_window_variants.csv")
    base = pwv[pwv["variant"] == "Baseline"].set_index("outcome")
    pooled = {
        r["row_key"]: r
        for r in csv.DictReader(open(INPUTS / "tables/pooled_did_results_table.csv", newline=""))
    }
    beta_share_pp = float(pooled["foreign_profit_share"]["coef_table"])

    std = json.loads((ANALYSIS_OUT / "profit_shifting_mechanism_standardization.json").read_text())

    # "Typical exposed firm": the mean pre-TCJA exposure among firms with a
    # negative international contribution, and the mean foreign profit share of
    # those same firms.  Both are computed in the foreign-share sample so the
    # ratio is internally consistent.
    panel = pd.read_csv(ANALYSIS_OUT / "profit_shifting_mechanism_panel.csv")
    pre = panel[panel["recon_year"].between(2013, 2016)]
    exposed = pre[pre["intl_pre"] < 0]
    mean_exposure = float(exposed["intl_pre"].mean())
    mean_share = float(exposed["foreign_share"].mean()) * 100.0
    implied_drop_pp = beta_share_pp * abs(mean_exposure)

    head = [
        ("typicalExposedExposurePp", "Mean pre-TCJA international exposure among exposed firms (pp)", mean_exposure),
        ("typicalExposedSharePct", "Mean pre-TCJA foreign profit share of those firms (percent)", mean_share),
        ("typicalExposedDropPp", "Implied fall in foreign profit share for that firm (pp)", implied_drop_pp),
        ("typicalExposedDropPct", "That fall as a percent of their pre-TCJA share", 100.0 * implied_drop_pp / mean_share),
        ("betaPiDom", "Pooled DiD: domestic pretax income over lagged assets", 100.0 * float(base.loc["pidom_int", "coef"])),
        ("betaPiDomSe", "SE", 100.0 * float(base.loc["pidom_int", "se"])),
        ("betaPiFo", "Pooled DiD: foreign pretax income over lagged assets", 100.0 * float(base.loc["pifo_int", "coef"])),
        ("betaPiFoSe", "SE", 100.0 * float(base.loc["pifo_int", "se"])),
        ("zWinsorLowPct", "Z winsorization: lower percentile", 1.0),
        ("zWinsorHighPct", "Z winsorization: upper percentile", 99.0),
        ("zRdFirms", "Firms underlying the R&D standardization", float(std["compustat"]["pre_rd_at"]["n"])),
    ]
    pd.DataFrame(
        [{"row_key": k, "label": lab, "value": v} for k, lab, v in head]
    ).to_csv(OUT / "heterogeneity_headlines.csv", index=False, float_format="%.6f")

    print(f"Wrote heterogeneity_gamma.csv ({len(gamma)} rows) and heterogeneity_headlines.csv ({len(head)} keys)")


if __name__ == "__main__":
    main()
