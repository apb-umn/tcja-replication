#!/usr/bin/env python3
"""
Post-window timing robustness for the real-activity DiD outcomes.

This script keeps the real-activity design from
build_real_activity_dynamics.py fixed and varies only the post window:

  - treatment: pre-TCJA International contribution, 2013-2016 mean
  - regression pre years: 2013-2015, omitting 2016
  - transition years excluded: 2017-2018
  - fixed effects: firm + industry-by-year
  - standard errors: clustered by firm

The diagnostic is whether near-zero effects on capex, employment, and sales
are just sluggish adjustments. Under that story, long/late post windows should
become more negative because beta < 0 means heavier shifters increased the
real outcome after TCJA.

Outputs:
  data/analysis/output/robustness/real_activity_post_window_variants.csv
  data/analysis/output/robustness/real_activity_post_window_variants.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_real_activity_dynamics import (
    OMIT_YEAR,
    OUTCOMES,
    PRE_YEARS,
    build_intl_pre,
    demean_ind_yr,
    load_data,
    stars,
    _fit_cluster,
)
from sic_industry import industry


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "analysis" / "output" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_YEARS = list(range(2011, 2025))
# Pre years match the paper baseline (Table 8): 2013-2015, 2016 omitted.
REG_PRE_YEARS = [yr for yr in range(2013, 2017) if yr != OMIT_YEAR]

POST_WINDOWS = [
    ("Early post", list(range(2019, 2022)), "2019-2021"),
    ("Real-activity baseline", list(range(2019, 2023)), "2019-2022"),
    ("Long post 2023", list(range(2019, 2024)), "2019-2023"),
    ("Long post", list(range(2019, 2025)), "2019-2024"),
    ("Late post", list(range(2022, 2025)), "2022-2024"),
]

OUTCOME_LABELS = {
    "capex_int": "Capex / lag assets",
    "rd_int": "R&D / lag assets",
    "total_inv": "Total inv / lag assets",
    "emp_growth": "Employment growth",
    "sale_growth": "Sales growth",
}


def fmt_num(x: float, digits: int = 5) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.{digits}f}"


def fmt_cell(row: pd.Series, digits: int = 5) -> str:
    if pd.isna(row["coef"]):
        return ""
    return f"{row['coef']:+.{digits}f}{row['stars']} ({row['se']:.5f})"


def fmt_implied(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.2f}"


def build_panel(
    combined: pd.DataFrame,
    extra: pd.DataFrame,
    intl_pre: pd.Series,
    post_years: list[int],
) -> pd.DataFrame:
    panel = combined[combined["recon_year"].isin(EVENT_YEARS)].copy()
    panel["industry"] = panel["comp_sic"].apply(industry)
    panel = panel[panel["industry"] != "Finance"].copy()
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(extra, on=["gvkey", "recon_year"], how="left")

    fy = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = fy.apply(lambda s: len(s & set(PRE_YEARS)) >= 2)
    has_post = fy.apply(lambda s: len(s & set(post_years)) >= 2)
    ok = fy[has_pre & has_post].index
    panel = panel[panel["gvkey"].isin(ok)].copy()

    lag_src = panel[["gvkey", "recon_year", "comp_at", "comp_emp", "comp_sale"]].copy()
    lag_src["recon_year"] = lag_src["recon_year"] + 1
    lag_src = lag_src.rename(
        columns={
            "comp_at": "lag_at",
            "comp_emp": "lag_emp",
            "comp_sale": "lag_sale",
        }
    )
    panel = panel.merge(lag_src, on=["gvkey", "recon_year"], how="left")

    valid_at = panel["lag_at"] > 1
    panel["capex_int"] = np.where(
        valid_at & panel["comp_capx"].notna(),
        panel["comp_capx"] / panel["lag_at"],
        np.nan,
    )
    panel["rd_int"] = np.where(
        valid_at & panel["comp_xrd"].notna(),
        panel["comp_xrd"] / panel["lag_at"],
        np.nan,
    )
    both_ok = valid_at & panel["comp_capx"].notna() & panel["comp_xrd"].notna()
    panel["total_inv"] = np.where(
        both_ok,
        (panel["comp_capx"] + panel["comp_xrd"]) / panel["lag_at"],
        np.nan,
    )
    panel["emp_growth"] = np.where(
        (panel["comp_emp"] > 0) & (panel["lag_emp"] > 0),
        np.log(panel["comp_emp"] / panel["lag_emp"]),
        np.nan,
    )
    panel["sale_growth"] = np.where(
        (panel["comp_sale"] > 0) & (panel["lag_sale"] > 0),
        np.log(panel["comp_sale"] / panel["lag_sale"]),
        np.nan,
    )

    for col, info in OUTCOMES.items():
        panel[col] = panel[col].clip(info["lo"], info["hi"])
    return panel


def run_did(panel: pd.DataFrame, outcome: str, post_years: list[int]) -> dict[str, float]:
    reg_years = REG_PRE_YEARS + post_years
    mask = (
        panel["recon_year"].isin(reg_years)
        & panel[outcome].notna()
        & panel["intl_pre"].notna()
    )
    sub = panel.loc[mask].copy()
    if len(sub) < 50 or sub["gvkey"].nunique() < 20:
        return {"coef": np.nan, "se": np.nan, "n": 0, "G": 0}

    sub["_post"] = sub["recon_year"].isin(post_years).astype(float)
    sub["_Di"] = sub["intl_pre"] * sub["_post"]
    dm = demean_ind_yr(sub, [outcome, "_Di"])
    mod, se = _fit_cluster(dm[outcome].values, dm[["_Di"]].values, sub["gvkey"].values)
    return {
        "coef": float(mod.params[0]),
        "se": float(se[0]),
        "n": int(len(sub)),
        "G": int(sub["gvkey"].nunique()),
    }


def collect_results() -> pd.DataFrame:
    combined, long_df, extra = load_data()
    intl_pre = build_intl_pre(long_df)

    rows: list[dict[str, object]] = []
    for variant, post_years, post_label in POST_WINDOWS:
        panel = build_panel(combined, extra, intl_pre, post_years)
        panel_firms = int(panel["gvkey"].nunique())
        panel_firm_years = int(len(panel))
        for outcome, info in OUTCOMES.items():
            res = run_did(panel, outcome, post_years)
            t = res["coef"] / res["se"] if res["se"] and not np.isnan(res["se"]) else np.nan
            outcome_obs = int(panel[outcome].notna().sum())
            outcome_firms = int(panel.loc[panel[outcome].notna(), "gvkey"].nunique())
            rows.append(
                {
                    "variant": variant,
                    "treatment_pre_years": f"{min(PRE_YEARS)}-{max(PRE_YEARS)}",
                    "regression_pre_years": f"{min(REG_PRE_YEARS)}-{max(REG_PRE_YEARS)}",
                    "omitted_base_year": OMIT_YEAR,
                    "post_years": post_label,
                    "balance_rule": ">=2 treatment-pre obs and >=2 post-window obs",
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS.get(outcome, info["label"]),
                    "coef": res["coef"],
                    "se": res["se"],
                    "t": t,
                    "stars": stars(t) if not np.isnan(t) else "",
                    "n": res["n"],
                    "firms": res["G"],
                    "panel_firms": panel_firms,
                    "panel_firm_years": panel_firm_years,
                    "outcome_obs": outcome_obs,
                    "outcome_firms": outcome_firms,
                    "implied_5pp_pp": -5.0 * res["coef"] * 100.0
                    if not np.isnan(res["coef"])
                    else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    baseline = (
        out[out["variant"] == "Real-activity baseline"][["outcome", "coef", "se", "n", "firms"]]
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
    out["implied_minus_baseline_pp"] = out["implied_5pp_pp"] - (
        -5.0 * out["baseline_coef"] * 100.0
    )
    return out


def write_markdown(out: pd.DataFrame, path: Path) -> None:
    outcome_order = list(OUTCOMES.keys())
    variants = [(label, post_label) for label, _, post_label in POST_WINDOWS]

    lines = [
        "# Real-Activity Post-Window Variants",
        "",
        "This robustness keeps the real-activity DiD design fixed and varies",
        "only the post window. Treatment is the 2013-2016 pre-TCJA",
        "International contribution. Regressions use 2011-2015 as the",
        "pre-period comparison years, omit 2016, exclude 2017-2018, include",
        "firm and industry-by-year fixed effects, and cluster standard errors",
        "by firm.",
        "",
        "Sign convention: beta < 0 means heavier pre-TCJA shifters increased",
        "the real outcome after TCJA. The implied-effect columns report the",
        "effect for a firm moving from -10 pp to -5 pp International exposure,",
        "in percentage points for intensities and approximately percent for",
        "log growth outcomes.",
        "",
        "## Coefficients",
        "",
        "| Outcome | 2019-2021 | 2019-2022 baseline | 2019-2023 | 2019-2024 | 2022-2024 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for outcome in outcome_order:
        cells = []
        for variant, _ in variants:
            row = out[(out["outcome"] == outcome) & (out["variant"] == variant)]
            cells.append(fmt_cell(row.iloc[0]) if not row.empty else "")
        label = OUTCOME_LABELS[outcome]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Implied Effect of a 5 pp Closure",
        "",
        "| Outcome | 2019-2021 | 2019-2022 baseline | 2019-2023 | 2019-2024 | 2022-2024 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome in outcome_order:
        cells = []
        for variant, _ in variants:
            row = out[(out["outcome"] == outcome) & (out["variant"] == variant)]
            cells.append(fmt_implied(row.iloc[0]["implied_5pp_pp"]) if not row.empty else "")
        lines.append(f"| {OUTCOME_LABELS[outcome]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Sample Counts",
        "",
        "| Variant | Post window | Panel firms | Capex firms | R&D firms | Total-inv firms | Employment firms | Sales firms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, post_label in variants:
        sub = out[out["variant"] == variant]
        panel_firms = int(sub["panel_firms"].iloc[0])

        def firms(outcome: str) -> int:
            return int(sub[sub["outcome"] == outcome]["firms"].iloc[0])

        lines.append(
            f"| {variant} | {post_label} | {panel_firms:,} | "
            f"{firms('capex_int'):,} | {firms('rd_int'):,} | "
            f"{firms('total_inv'):,} | {firms('emp_growth'):,} | "
            f"{firms('sale_growth'):,} |"
        )

    lines += [
        "",
        "## Read",
        "",
        "- A sluggish positive real response would show up as coefficients becoming",
        "  more negative in the long or late post windows.",
        "- The baseline real-activity table in the paper (Table 8) is the 2019-2022 row.",
        "- Compare firm counts before reading timing differences as pure dynamics;",
        "  the balance rule is applied separately within each post window.",
        "",
        "## Full Results",
        "",
        "| Variant | Post | Outcome | Coef. | SE | Stars | N | Firms | Implied 5 pp closure | Delta vs baseline |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in out.iterrows():
        lines.append(
            f"| {row['variant']} | {row['post_years']} | {row['outcome_label']} | "
            f"{fmt_num(row['coef'])} | {row['se']:.5f} | {row['stars']} | "
            f"{int(row['n']):,} | {int(row['firms']):,} | "
            f"{fmt_implied(row['implied_5pp_pp'])} | "
            f"{fmt_num(row['coef_minus_baseline'])} |"
        )

    lines += [
        "",
        "Source: `analysis/build_real_activity_post_window_variants.py`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out = collect_results()
    csv_path = OUT_DIR / "real_activity_post_window_variants.csv"
    md_path = OUT_DIR / "real_activity_post_window_variants.md"
    out.to_csv(csv_path, index=False)
    write_markdown(out, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
