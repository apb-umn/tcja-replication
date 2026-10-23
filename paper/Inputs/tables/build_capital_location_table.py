#!/usr/bin/env python3
"""
Segment-based capital-location estimates quoted in Section 3.2.3.

The draft says the foreign capital stock shows no shift and that for capital
expenditures a pre-TCJA differential trend cannot be ruled out, with post-TCJA
coefficients individually insignificant.  Both claims were made without
estimates.  This stages them from the capital-location run.

The distinction matters and the hedge is well founded: the pooled
foreign-capex-share coefficient is significant, but its event study has
negative pre-period coefficients and positive post-period ones that are each
insignificant, so the pooled estimate is partly the pre-trend the text
concedes.  The stock series shows nothing in any year.

Run analysis/capex_location_test.py first.

Outputs:
  tables/capital_location_table.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUTS = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_OUT = REPO_ROOT / "data" / "analysis" / "output"

SERIES = [
    ("ppe_share", "foreign_ppe_share", "Foreign share of PP\\&E"),
    ("capx_share", "foreign_capxs_share", "Foreign share of capital expenditures"),
]


def main() -> None:
    summary = json.loads((ANALYSIS_OUT / "capital_location_summary.json").read_text())

    rows: list[dict[str, object]] = []
    for order, (key, block, label) in enumerate(SERIES, start=1):
        o = summary[block]["outcome"]["ind_yr"]
        coef, se = float(o["coef"]), float(o["se"])
        rows.append(
            {
                "row_order": order,
                "row_key": key,
                "row_label": label,
                # Shares are fractions; x100 puts them in percentage points.
                "coef": 100.0 * coef,
                "se": 100.0 * se,
                "t": coef / se if se else float("nan"),
                "n_obs": int(o["n"]),
                "n_firms": int(o["G"]),
            }
        )

    df = pd.DataFrame(rows)
    out = INPUTS / "tables" / "capital_location_table.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
