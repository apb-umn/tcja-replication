#!/usr/bin/env python3
"""
Windfall first-stage headline numbers, generated rather than hand-copied.

The draft quotes the 2017 net-income spike from the windfall design and the
pooled 2018-2022 coefficient.  These were previously maintained by hand -- the
build instructions recorded that the confidence band was "hardcoded from the
slide figure/text because the committed repo does not currently export those
uncertainty terms as structured CSV fields" -- and had drifted from what the
code produces.  This exports them directly.

Both the outcome and the treatment are fractions of lagged assets, so theta is
dimensionless: if the deferred-tax remeasurement passed one for one into
reported net income in the shock year, theta_2017 would be 1.

Outputs:
  tables/windfall_ni_headline.csv
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

import build_investment_elasticity as W  # noqa: E402


BASE_PANEL = REPO_ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv"
CACHE = REPO_ROOT / "data" / "analysis" / "cache"


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        comp = W.load_comp_extra(str(BASE_PANEL))
        long_df = pd.read_pickle(CACHE / "long.pkl")
        combined = pd.read_pickle(CACHE / "combined.pkl")
        treat = W.build_treatment(comp, long_df)
        panel = W.build_outcomes(
            W.build_event_panel(combined, treat, comp, W.EVENT_YEARS, W.PRE_YEARS, W.POST_YEARS)
        )
        event, _info = W.run_event_study(panel, "ni_intensity", "W", W.EVENT_YEARS, ind_yr_fe=True)
        pooled = W.run_did(panel, "ni_intensity", "W", ind_yr_fe=True)

    shock = event.loc[event["year"] == W.SHOCK_YEAR].iloc[0]
    rows = [
        {
            "row_key": "ni_shock_2017",
            "coef": float(shock["beta"]),
            "se": float(shock["se"]),
            "lower95": float(shock["beta"]) - 1.96 * float(shock["se"]),
            "upper95": float(shock["beta"]) + 1.96 * float(shock["se"]),
            "n_firms": int(pooled.get("G", 0)),
        },
        {
            "row_key": "ni_post_2018",
            "coef": float(pooled["coef"]),
            "se": float(pooled["se"]),
            "lower95": float(pooled["coef"]) - 1.96 * float(pooled["se"]),
            "upper95": float(pooled["coef"]) + 1.96 * float(pooled["se"]),
            "n_firms": int(pooled.get("G", 0)),
        },
    ]
    df = pd.DataFrame(rows)[["row_key", "coef", "se", "lower95", "upper95", "n_firms"]]
    out = INPUTS / "tables" / "windfall_ni_headline.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out.name}: theta_2017={rows[0]['coef']:.3f}, pooled={rows[1]['coef']:.3f}")


if __name__ == "__main__":
    main()
