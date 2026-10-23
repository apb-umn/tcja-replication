#!/usr/bin/env python3
"""
Integrate the Part III deferred-tax windfall with the Part I/II
foreign-exposure designs.

The script asks two questions:

1. How does the Part III windfall W_i vary across firms with low,
   moderate, and heavy pre-TCJA foreign exposure?
2. Do the Part II profit-location estimates survive a two-treatment DiD
   that adds W_i x post as a financial/balance-sheet control?

Outputs:
  data/analysis/output/robustness/windfall_foreign_exposure_bins.csv
  data/analysis/output/robustness/windfall_partii_control_robustness.csv
  data/analysis/output/robustness/windfall_integration.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "analysis" / "output" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep imported plotting modules from trying to write under the home folder.
MPL_DIR = ROOT / "data" / "analysis" / "cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import build_substitution_real_outcomes as real  # noqa: E402
import build_investment_elasticity as windfall  # noqa: E402


OUTCOMES = [
    ("International", "International contribution"),
    ("etr_w", "Book ETR"),
    ("foreign_share", "Foreign PI share"),
    ("foreign_share_log", "Log foreign PI share"),
    ("pidom_int", "Domestic PI / lag assets"),
    ("pifo_int", "Foreign PI / lag assets"),
]


def stars(t: float) -> str:
    a = abs(t)
    if a > 2.576:
        return "***"
    if a > 1.960:
        return "**"
    if a > 1.645:
        return "*"
    return ""


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.{digits}f}"


def load_merged_panel() -> pd.DataFrame:
    combined, long_df, pi = real.load_data()
    intl_pre = real.build_channels_and_treatment(long_df)
    intl_yearly = real.build_intl_yearly(long_df)
    panel = real.build_panel(combined, pi, intl_pre, intl_yearly)

    comp = windfall.load_comp_extra(ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv")
    treat = windfall.build_treatment(comp, long_df)

    panel = panel.merge(
        treat[["gvkey", "Z", "Z_w", "W", "va_2017_pp"]],
        on="gvkey",
        how="inner",
    )
    panel["W_pct"] = 100.0 * panel["W"]
    panel["foreign_exposure_pp"] = -panel["intl_pre"]
    return panel


def firm_level(panel: pd.DataFrame) -> pd.DataFrame:
    firm = panel.drop_duplicates("gvkey")[
        [
            "gvkey",
            "industry",
            "intl_pre",
            "foreign_exposure_pp",
            "Z",
            "Z_w",
            "W",
            "W_pct",
            "va_2017_pp",
        ]
    ].copy()
    pre = (
        panel[panel["recon_year"].isin(real.PRE_YEARS)]
        .groupby("gvkey")
        .agg(
            pre_foreign_share=("foreign_share", "mean"),
            pre_pifo_int=("pifo_int", "mean"),
            pre_pidom_int=("pidom_int", "mean"),
            pre_assets=("comp_at", "mean"),
            pre_etr=("etr_w", "mean"),
        )
    )
    firm = firm.merge(pre, on="gvkey", how="left")
    firm["net_dtl"] = firm["W"] > 0
    firm["exposure_bin"] = np.select(
        [firm["intl_pre"] < -5.0, firm["intl_pre"] < -1.0],
        ["heavy (<-5pp)", "moderate (-5,-1pp)"],
        default="low/no (>=-1pp)",
    )
    return firm


def summarize_bins(firm: pd.DataFrame) -> pd.DataFrame:
    out = (
        firm.groupby("exposure_bin")
        .agg(
            firms=("gvkey", "nunique"),
            intl_pre_mean=("intl_pre", "mean"),
            foreign_exposure_mean=("foreign_exposure_pp", "mean"),
            W_pct_mean=("W_pct", "mean"),
            W_pct_median=("W_pct", "median"),
            W_pct_p25=("W_pct", lambda x: x.quantile(0.25)),
            W_pct_p75=("W_pct", lambda x: x.quantile(0.75)),
            net_dtl_share=("net_dtl", "mean"),
            pre_foreign_share=("pre_foreign_share", "mean"),
            pre_assets_median=("pre_assets", "median"),
        )
        .reset_index()
    )
    order = {"low/no (>=-1pp)": 0, "moderate (-5,-1pp)": 1, "heavy (<-5pp)": 2}
    return out.sort_values("exposure_bin", key=lambda s: s.map(order)).reset_index(drop=True)


def cross_section_regs(firm: pd.DataFrame) -> pd.DataFrame:
    cs = firm.dropna(subset=["W_pct", "foreign_exposure_pp"]).copy()
    rows = []
    for fe in [False, True]:
        x = cs[["foreign_exposure_pp"]].copy()
        if fe:
            dummies = pd.get_dummies(cs["industry"], prefix="ind", drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)
        x = pd.concat([pd.Series(1.0, index=x.index, name="const"), x], axis=1)
        mod = OLS(cs["W_pct"].values, x.values).fit(cov_type="HC1")
        idx = list(x.columns).index("foreign_exposure_pp")
        rows.append(
            {
                "model": "industry_fe" if fe else "no_fe",
                "coef_foreign_exposure": float(mod.params[idx]),
                "se": float(mod.bse[idx]),
                "t": float(mod.tvalues[idx]),
                "n": int(len(cs)),
                "r2": float(mod.rsquared),
            }
        )
    return pd.DataFrame(rows)


def run_partii_did(panel: pd.DataFrame, outcome: str, include_w: bool) -> dict[str, float]:
    excluded = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [yr for yr in real.EVENT_YEARS if yr not in excluded]
    mask = (
        panel["recon_year"].isin(years)
        & panel[outcome].notna()
        & panel["intl_pre"].notna()
        & panel["W_pct"].notna()
    )
    sub = panel.loc[mask].copy()
    if len(sub) < 50 or sub["gvkey"].nunique() < 20:
        return {"coef_intl": np.nan, "se_intl": np.nan, "n": 0, "firms": 0}

    sub["_post"] = sub["recon_year"].isin(real.POST_YEARS).astype(float)
    sub["_intl_post"] = sub["intl_pre"] * sub["_post"]
    sub["_w_post"] = sub["W_pct"] * sub["_post"]

    cols = [outcome, "_intl_post"] + (["_w_post"] if include_w else [])
    dm = real.demean_ind_yr(sub, cols)
    xcols = ["_intl_post"] + (["_w_post"] if include_w else [])
    mod = OLS(dm[outcome].values, dm[xcols].values).fit()
    cov = cov_cluster(mod, sub["gvkey"].values)
    se = np.sqrt(np.diag(cov))

    out = {
        "coef_intl": float(mod.params[0]),
        "se_intl": float(se[0]),
        "t_intl": float(mod.params[0] / se[0]) if se[0] > 0 else np.nan,
        "n": int(len(sub)),
        "firms": int(sub["gvkey"].nunique()),
    }
    if include_w:
        out.update(
            {
                "coef_w": float(mod.params[1]),
                "se_w": float(se[1]),
                "t_w": float(mod.params[1] / se[1]) if se[1] > 0 else np.nan,
            }
        )
    return out


def partii_control_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, label in OUTCOMES:
        base = run_partii_did(panel, outcome, include_w=False)
        ctrl = run_partii_did(panel, outcome, include_w=True)
        rows.append(
            {
                "outcome": outcome,
                "outcome_label": label,
                "base_coef_intl": base["coef_intl"],
                "base_se_intl": base["se_intl"],
                "base_t_intl": base["t_intl"],
                "base_stars_intl": stars(base["t_intl"]),
                "wctrl_coef_intl": ctrl["coef_intl"],
                "wctrl_se_intl": ctrl["se_intl"],
                "wctrl_t_intl": ctrl["t_intl"],
                "wctrl_stars_intl": stars(ctrl["t_intl"]),
                "wctrl_coef_w": ctrl.get("coef_w", np.nan),
                "wctrl_se_w": ctrl.get("se_w", np.nan),
                "wctrl_t_w": ctrl.get("t_w", np.nan),
                "wctrl_stars_w": stars(ctrl.get("t_w", np.nan)),
                "delta_intl": ctrl["coef_intl"] - base["coef_intl"],
                "n": ctrl["n"],
                "firms": ctrl["firms"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    bins: pd.DataFrame,
    cross: pd.DataFrame,
    robust: pd.DataFrame,
    firm_count: int,
    panel_rows: int,
) -> None:
    lines = [
        "# Windfall and Foreign Exposure Integration",
        "",
        "This diagnostic merges Part III's predetermined deferred-tax windfall",
        "`W_i` with the Part I/II pre-TCJA International exposure treatment.",
        "The merged sample is the Part II profit-location panel restricted to",
        "firms with non-missing `W_i` after the 2017 valuation-allowance exclusion.",
        "",
        f"Merged sample: `{firm_count:,}` firms and `{panel_rows:,}` firm-years.",
        "",
        "## Windfall by Foreign-Exposure Bin",
        "",
        "| Bin | Firms | Mean intl_pre | Mean W (% assets) | Median W | Net-DTL share | Pre foreign PI share | Median pre assets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in bins.iterrows():
        lines.append(
            f"| {r['exposure_bin']} | {int(r['firms']):,} | "
            f"{r['intl_pre_mean']:+.2f} | {r['W_pct_mean']:+.2f} | "
            f"{r['W_pct_median']:+.2f} | {100*r['net_dtl_share']:.1f}% | "
            f"{100*r['pre_foreign_share']:.1f}% | {r['pre_assets_median']:,.0f} |"
        )

    lines += [
        "",
        "## Cross-Sectional Slope",
        "",
        "| Model | Coef. on foreign exposure | SE | t | N | R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in cross.iterrows():
        lines.append(
            f"| {r['model']} | {r['coef_foreign_exposure']:+.3f} | "
            f"{r['se']:.3f} | {r['t']:+.2f} | {int(r['n']):,} | {r['r2']:.3f} |"
        )

    lines += [
        "",
        "## Part II DiD with Windfall Control",
        "",
        "The comparison holds the sample fixed to firms with non-missing `W_i`.",
        "Columns compare the baseline `intl_pre x post` coefficient to the",
        "same regression with `W_i x post` added.",
        "",
        "| Outcome | Baseline beta_Intl | With W control | Delta | beta_W | Firms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in robust.iterrows():
        lines.append(
            f"| {r['outcome_label']} | {fmt(r['base_coef_intl'], 5)}{r['base_stars_intl']} "
            f"({r['base_se_intl']:.5f}) | {fmt(r['wctrl_coef_intl'], 5)}{r['wctrl_stars_intl']} "
            f"({r['wctrl_se_intl']:.5f}) | {fmt(r['delta_intl'], 5)} | "
            f"{fmt(r['wctrl_coef_w'], 5)}{r['wctrl_stars_w']} ({r['wctrl_se_w']:.5f}) | "
            f"{int(r['firms']):,} |"
        )

    lines += [
        "",
        "## Read",
        "",
        "- Windfall exposure is not mechanically the same object as foreign",
        "  profit-shifting exposure. Heavy shifters have higher pre foreign PI",
        "  shares, but lower average `W_i` in the raw bins.",
        "- The raw negative W-foreign-exposure slope is mostly industry composition:",
        "  it is large without industry FE and small after industry FE.",
        "- Adding `W_i x post` leaves the Part II foreign-share coefficient",
        "  unchanged at displayed precision and barely moves domestic/foreign",
        "  PI-intensity coefficients.",
        "- This makes Part III useful as a financial-confound robustness: the",
        "  profit-location estimates are not an artifact of the deferred-tax",
        "  balance-sheet windfall.",
        "- `W_i` is not a good instrument for the international exposure treatment.",
        "  It is a separate predetermined financial/accounting shock. The cleaner",
        "  integration is a two-treatment/control design, plus the existing",
        "  Part III use-of-funds result showing no investment response.",
        "",
        "Source: `analysis/build_windfall_foreign_exposure_integration.py`.",
    ]

    (OUT_DIR / "windfall_integration.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    panel = load_merged_panel()
    firm = firm_level(panel)
    bins = summarize_bins(firm)
    cross = cross_section_regs(firm)
    robust = partii_control_robustness(panel)

    bins.to_csv(OUT_DIR / "windfall_foreign_exposure_bins.csv", index=False)
    cross.to_csv(OUT_DIR / "windfall_foreign_exposure_cross_section.csv", index=False)
    robust.to_csv(OUT_DIR / "windfall_partii_control_robustness.csv", index=False)
    write_markdown(bins, cross, robust, firm["gvkey"].nunique(), len(panel))

    print(f"Wrote {OUT_DIR / 'windfall_foreign_exposure_bins.csv'}")
    print(f"Wrote {OUT_DIR / 'windfall_foreign_exposure_cross_section.csv'}")
    print(f"Wrote {OUT_DIR / 'windfall_partii_control_robustness.csv'}")
    print(f"Wrote {OUT_DIR / 'windfall_integration.md'}")


if __name__ == "__main__":
    main()
