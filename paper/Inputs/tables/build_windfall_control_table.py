#!/usr/bin/env python3
"""
Numbers for the windfall check in Section 3.2.4.

Section 3.2.4 asks whether the deferred-tax remeasurement muddles the reading
of Sections 3.2.1-3.2.3: exposed companies also received a balance-sheet
windfall, and if that relaxed binding constraints, the differential null on
real activity could be two effects cancelling and the profit-share result
could be picking up balance-sheet effects rather than booking incentives.

This stages what the check needs: how far the two treatments overlap, and the
main coefficients with and without W_i x post as a control.

Sources:
  data/analysis/output/robustness/windfall_partii_control_robustness.csv
  the windfall treatment and the international treatment, for their overlap

Outputs:
  tables/windfall_control_table.csv
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd


INPUTS = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import build_investment_elasticity as WE  # noqa: E402
import build_real_activity_dynamics as RA  # noqa: E402

ROBUST = (
    REPO_ROOT / "data" / "analysis" / "output" / "robustness"
    / "windfall_partii_control_robustness.csv"
)
BASE_PANEL = REPO_ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv"
CACHE = REPO_ROOT / "data" / "analysis" / "cache"


def main() -> None:
    rows: list[dict[str, object]] = []

    # --- how far the two treatments overlap -----------------------------
    with contextlib.redirect_stdout(io.StringIO()):
        comp = WE.load_comp_extra(str(BASE_PANEL))
        long_df = pd.read_pickle(CACHE / "long.pkl")
        treat = WE.build_treatment(comp, long_df)
        intl = RA.build_intl_pre(long_df).rename("intl_pre")
    f = treat[["gvkey", "W"]].merge(intl.reset_index(), on="gvkey", how="inner")
    exposed = f[f["intl_pre"] < -2]
    unexposed = f[f["intl_pre"] >= 0]
    rows += [
        {"row_key": "corr_w_intl", "row_label": "Correlation of W with pre-TCJA exposure",
         "value": float(f["W"].corr(f["intl_pre"]))},
        {"row_key": "mean_w_exposed", "row_label": "Mean W, exposure below -2pp",
         "value": float(exposed["W"].mean())},
        {"row_key": "mean_w_unexposed", "row_label": "Mean W, exposure at or above zero",
         "value": float(unexposed["W"].mean())},
        {"row_key": "n_overlap", "row_label": "Firms with both measures", "value": float(len(f))},
    ]

    # --- main coefficients with and without the control -----------------
    rob = pd.read_csv(ROBUST).set_index("outcome")
    for key, label in [("foreign_share", "Foreign profit share"), ("etr_w", "Effective tax rate")]:
        r = rob.loc[key]
        scale = 100.0 if key == "foreign_share" else 1.0
        rows += [
            {"row_key": f"{key}_base", "row_label": f"{label}: without control",
             "value": scale * float(r["base_coef_intl"])},
            {"row_key": f"{key}_base_se", "row_label": f"{label}: without control, SE",
             "value": scale * float(r["base_se_intl"])},
            {"row_key": f"{key}_ctrl", "row_label": f"{label}: with W x post control",
             "value": scale * float(r["wctrl_coef_intl"])},
            {"row_key": f"{key}_ctrl_se", "row_label": f"{label}: with control, SE",
             "value": scale * float(r["wctrl_se_intl"])},
            {"row_key": f"{key}_w_coef", "row_label": f"{label}: coefficient on W x post",
             "value": scale * float(r["wctrl_coef_w"])},
            {"row_key": f"{key}_w_se", "row_label": f"{label}: SE on W x post",
             "value": scale * float(r["wctrl_se_w"])},
            {"row_key": f"{key}_firms", "row_label": f"{label}: firms", "value": float(r["firms"])},
        ]

    df = pd.DataFrame(rows)
    out = INPUTS / "tables" / "windfall_control_table.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out.name} ({len(df)} keys)")


if __name__ == "__main__":
    main()
