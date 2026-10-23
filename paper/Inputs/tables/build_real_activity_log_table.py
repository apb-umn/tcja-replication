#!/usr/bin/env python3
"""
Real-activity outcomes in logs, for Table 9.

The scale measures enter the draft in logs rather than as ratios to lagged
assets.  A ratio can move because of its denominator, and for these outcomes it
does: the asset base falls for more exposed firms after repatriation and the
transition tax, which is a balance-sheet response rather than a change in
spending.  Logs of the levels put the question directly.

Also emits the split of pretax income into its foreign and domestic parts,
which is the object Section 4 needs when it holds total profits fixed while the
booking margin moves.

Outputs:
  tables/real_activity_log_table.csv
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUTS = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import build_real_activity_dynamics as M  # noqa: E402


PANEL_BASE = REPO_ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv"

ROWS = [
    ("log_capx", "comp_capx", "Log capital expenditures"),
    ("log_xrd", "comp_xrd", r"Log R\&D expenditures"),
    ("log_sale", "comp_sale", "Log sales"),
    ("log_emp", "comp_emp", "Log employment"),
    ("log_at", "comp_at", "Log assets"),
    ("log_pi", "comp_pi", "Log pretax income"),
    ("log_pifo", "comp_pifo", "Log foreign pretax income"),
    ("log_pidom", "comp_pidom", "Log domestic pretax income"),
]


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        combined, long_df, extra = M.load_data()
        intl_pre = M.build_intl_pre(long_df)
        panel = M.build_panel(combined, extra, intl_pre)

    pi = pd.read_csv(
        PANEL_BASE, usecols=["gvkey", "recon_year", "comp_pidom", "comp_pifo"], low_memory=False
    ).drop_duplicates(["gvkey", "recon_year"])
    panel = panel.merge(pi, on=["gvkey", "recon_year"], how="left")

    for col, src, _ in ROWS:
        if src is not None:
            panel[col] = np.where(panel[src] > 0, np.log(panel[src]), np.nan)

    always = panel[panel.assign(ok=panel["comp_pi"] > 1).groupby("gvkey")["ok"].transform("all")]
    variants = {"all": panel, "profitable": always}

    # The contrast used in the text: mean exposure among firms with a negative
    # pre-TCJA international contribution.
    exposed = panel.loc[panel["intl_pre"] < 0, ["gvkey", "intl_pre"]].drop_duplicates("gvkey")
    mean_exposure = abs(float(exposed["intl_pre"].mean()))

    rows: list[dict[str, object]] = []
    for order, (col, _src, label) in enumerate(ROWS, start=1):
        row: dict[str, object] = {"row_order": order, "row_key": col, "row_label": label}
        for name, frame in variants.items():
            r = M.run_pooled_did(frame, col, spec="ind_yr")
            t = r["coef"] / r["se"] if r["se"] else float("nan")
            row[f"{name}_coef"] = 100.0 * r["coef"]
            row[f"{name}_se"] = 100.0 * r["se"]
            row[f"{name}_stars"] = M.stars(t)
            row[f"{name}_firms"] = r["G"]
            row[f"{name}_obs"] = r["n"]
            # Implied effect for the mean exposed firm: percent for the log
            # rows, percentage points of assets for the two intensities.
            row[f"{name}_implied"] = 100.0 * r["coef"] * mean_exposure
        rows.append(row)

    df = pd.DataFrame(rows)
    df["mean_exposure_pp"] = mean_exposure
    out = INPUTS / "tables" / "real_activity_log_table.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out.name} ({len(df)} rows; contrast {mean_exposure:.2f}pp)")


if __name__ == "__main__":
    main()
