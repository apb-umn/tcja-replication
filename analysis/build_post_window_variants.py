#!/usr/bin/env python3
"""
Post-window timing robustness for the TCJA pass-through and profit-location
designs.

The baseline slides use pre = 2013-2016, omit 2016 as the event-study base
year, exclude 2017-2018, and define post = 2019-2022. This script keeps the
same treatment construction and regression design, then varies only the post
window:

  - 2019-2021
  - 2019-2022 baseline
  - 2019-2024
  - 2022-2024

Outputs:
  data/analysis/output/robustness/post_window_variants.csv
  data/analysis/output/robustness/post_window_variants.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_pre_tcja_placebo_windows import (
    OUTCOMES,
    OUT_DIR,
    build_channels,
    build_panel,
    build_treatment,
    implied_effect,
    load_inputs,
    run_did,
    stars,
)


PRE_YEARS = list(range(2013, 2017))

POST_WINDOWS = [
    ("Early post", list(range(2019, 2022)), "2019-2021"),
    ("Baseline", list(range(2019, 2023)), "2019-2022"),
    ("Long post", list(range(2019, 2025)), "2019-2024"),
    ("Late post", list(range(2022, 2025)), "2022-2024"),
]


def fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.{digits}f}"


def coef_se(row: pd.Series, digits: int = 3) -> str:
    if pd.isna(row["coef"]):
        return ""
    return f"{row['coef']:+.{digits}f}{row['stars']} ({row['se']:.3f})"


def collect_results() -> pd.DataFrame:
    combined, long_df, pi = load_inputs()
    wide = build_channels(long_df)
    intl_pre = build_treatment(wide, PRE_YEARS)

    rows = []
    for label, post_years, post_label in POST_WINDOWS:
        panel = build_panel(combined, pi, wide, intl_pre, PRE_YEARS, post_years)
        for outcome, outcome_label, units in OUTCOMES:
            res = run_did(panel, outcome, PRE_YEARS, post_years)
            t = res["coef"] / res["se"] if res["se"] and not np.isnan(res["se"]) else np.nan
            rows.append(
                {
                    "variant": label,
                    "pre_years": f"{min(PRE_YEARS)}-{max(PRE_YEARS)}",
                    "post_years": post_label,
                    "balance_rule": ">=2 pre and >=2 post observations in the listed windows",
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "coef": res["coef"],
                    "se": res["se"],
                    "stars": stars(t) if not np.isnan(t) else "",
                    "n": res["n"],
                    "firms": res["G"],
                    "implied_minus5": implied_effect(outcome, res["coef"]),
                    "implied_units": "pp" if outcome in {"International", "etr_w", "foreign_share"} else units,
                }
            )

    out = pd.DataFrame(rows)
    baseline = (
        out[out["variant"] == "Baseline"][["outcome", "coef", "se", "n", "firms"]]
        .rename(
            columns={
                "coef": "baseline_coef",
                "se": "baseline_se",
                "n": "baseline_n",
                "firms": "baseline_firms",
            }
        )
    )
    out = out.merge(baseline, on="outcome", how="left")
    out["coef_minus_baseline"] = out["coef"] - out["baseline_coef"]
    out["pct_change_vs_baseline"] = np.where(
        out["baseline_coef"].abs() > 1e-12,
        100.0 * out["coef_minus_baseline"] / out["baseline_coef"],
        np.nan,
    )
    return out


def write_markdown(out: pd.DataFrame, path: Path) -> None:
    key_outcomes = ["International", "etr_w", "foreign_share"]
    sub = out[out["outcome"].isin(key_outcomes)].copy()

    def cell(variant: str, outcome: str) -> str:
        row = sub[(sub["variant"] == variant) & (sub["outcome"] == outcome)]
        if row.empty:
            return ""
        return coef_se(row.iloc[0])

    lines = [
        "# Post-Window Variants",
        "",
        "This robustness keeps the baseline pre period (`2013-2016`), treatment",
        "construction, omitted base year (`2016`), firm fixed effects,",
        "industry-by-year fixed effects, and firm-clustered standard errors.",
        "Only the post window changes. Each row applies the same balance rule:",
        "`>=2` pre observations and `>=2` observations in the listed post window.",
        "",
        "## Headline Outcomes",
        "",
        "| Variant | Post window | Firms | International | Book ETR | Foreign PI share |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for label, _, post_label in POST_WINDOWS:
        row = sub[sub["variant"] == label].iloc[0]
        lines.append(
            f"| {label} | {post_label} | {int(row['firms']):,} | "
            f"{cell(label, 'International')} | {cell(label, 'etr_w')} | "
            f"{cell(label, 'foreign_share')} |"
        )

    lines += [
        "",
        "## Read",
        "",
        "- The early-post row asks whether the effect is already present by 2021.",
        "- The long-post row asks whether the result survives adding 2023-2024.",
        "- The late-post row isolates 2022-2024, which is useful for late IP",
        "  onshoring cases but should be read as a timing diagnostic rather than",
        "  the main TCJA design.",
        "- Compare firm counts across rows before interpreting changes in",
        "  coefficients as pure timing effects.",
        "",
        "## Full Outcome Table",
        "",
        "| Variant | Post | Outcome | Coef. | SE | Stars | N | Firms | Delta vs baseline | Implied effect for `intl_pre=-5` | Units |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in out.iterrows():
        lines.append(
            f"| {row['variant']} | {row['post_years']} | {row['outcome_label']} | "
            f"{fmt_num(row['coef'], 5)} | {row['se']:.5f} | {row['stars']} | "
            f"{int(row['n']):,} | {int(row['firms']):,} | "
            f"{fmt_num(row['coef_minus_baseline'], 5)} | "
            f"{fmt_num(row['implied_minus5'], 3)} | {row['implied_units']} |"
        )

    lines += [
        "",
        "Source: `analysis/build_post_window_variants.py`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out = collect_results()
    csv_path = OUT_DIR / "post_window_variants.csv"
    md_path = OUT_DIR / "post_window_variants.md"
    out.to_csv(csv_path, index=False)
    write_markdown(out, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
