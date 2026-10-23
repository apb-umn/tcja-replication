#!/usr/bin/env python3
"""
Balance-sheet margins behind the cash sentence in Section 3.2.3.

The draft says the cash contraction at more exposed firms is tax driven, and
lists what did and did not move alongside it.  Those were six claims with no
estimates behind them.  This stages them, on the same panel and specification
as Table 9 so the numbers are comparable, with each margin scaled by lagged
assets and winsorized at the 1st and 99th percentiles.

Sign convention follows the rest of the paper: the treatment is negative for
more exposed firms, so a positive coefficient means the outcome FALLS for them.

Outputs:
  tables/balance_sheet_table.csv
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

import build_real_activity_dynamics as M  # noqa: E402


PANEL_BASE = REPO_ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv"
SOURCE_COLS = [
    "gvkey", "recon_year",
    "comp_che", "comp_txpd", "comp_dvc", "comp_prstkc", "comp_dltt", "comp_dlc", "comp_aqc",
]

ROWS = [
    ("cash", "comp_che", "Cash and equivalents"),
    ("taxes_paid", "comp_txpd", "Cash taxes paid"),
    ("dividends", "comp_dvc", "Dividends"),
    ("repurchases", "comp_prstkc", "Share repurchases"),
    ("debt", "debt", "Total debt"),
    ("acquisitions", "comp_aqc", "Acquisitions"),
]


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        combined, long_df, extra = M.load_data()
        intl_pre = M.build_intl_pre(long_df)
        panel = M.build_panel(combined, extra, intl_pre)

    base = pd.read_csv(PANEL_BASE, usecols=SOURCE_COLS, low_memory=False).drop_duplicates(
        ["gvkey", "recon_year"]
    )
    panel = panel.merge(base, on=["gvkey", "recon_year"], how="left")
    panel["debt"] = panel["comp_dltt"].fillna(0) + panel["comp_dlc"].fillna(0)

    rows: list[dict[str, object]] = []
    for order, (key, col, label) in enumerate(ROWS, start=1):
        y = panel[col] / panel["lag_at"]
        lo, hi = y.quantile([0.01, 0.99])
        frame = panel.assign(_y=y.clip(lo, hi)).dropna(subset=["_y"])
        r = M.run_pooled_did(frame, "_y", spec="ind_yr")
        t = r["coef"] / r["se"] if r["se"] else float("nan")
        rows.append(
            {
                "row_order": order,
                "row_key": key,
                "row_label": label,
                "coef": 100.0 * r["coef"],
                "se": 100.0 * r["se"],
                "stars": M.stars(t),
                "t": t,
                "n_firms": r["G"],
                "n_obs": r["n"],
            }
        )

    df = pd.DataFrame(rows)
    out = INPUTS / "tables" / "balance_sheet_table.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
