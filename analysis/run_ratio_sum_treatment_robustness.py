"""Robustness: define pre-TCJA International exposure as a ratio of sums.

Baseline treatment in the deck:
    mean_t clip(c_Intl_it, [-15, 15]), then firm-level p1/p99 winsor.

This robustness:
    100 * sum_t International_amount_it / sum_t PI_base_it,
    then clip to [-15, 15] and firm-level p1/p99 winsor.

For amount-derived reconciliations PI_base is the implied pretax-income base
used in the amount-to-pp conversion. For rate-stated reconciliations PI_base
is Compustat pretax income.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import build_substitution_analysis as subst  # noqa: E402
import build_substitution_real_outcomes as real  # noqa: E402


OUT_DIR = ROOT / "data/analysis/output/robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_PP = 15.0


def build_ratio_sum_treatment(
    combined: pd.DataFrame,
    long_df: pd.DataFrame,
    pre_years: list[int] | tuple[int, ...] = subst.PRE_YEARS,
    clip_pp: float = CLIP_PP,
) -> pd.Series:
    """Build ratio-of-sums treatment with zero International rows retained."""
    denom = combined[["gvkey", "recon_year", "source", "comp_pi", "implied_pi"]].copy()
    denom["pi_base"] = np.where(
        denom["source"].eq("amount_derived") & denom["implied_pi"].notna(),
        denom["implied_pi"],
        denom["comp_pi"],
    )
    denom = denom[
        denom["recon_year"].isin(pre_years)
        & denom["pi_base"].notna()
        & denom["pi_base"].gt(1.0)
    ].drop_duplicates(["gvkey", "recon_year"])

    intl_pp = (
        long_df[
            long_df["recon_year"].isin(pre_years)
            & long_df["group"].eq("International")
        ]
        .groupby(["gvkey", "recon_year"])["cat_val"]
        .sum()
        .rename("intl_pp")
    )

    work = denom.merge(intl_pp.reset_index(), on=["gvkey", "recon_year"], how="left")
    work["intl_pp"] = work["intl_pp"].fillna(0.0)
    work["intl_amount"] = work["intl_pp"] / 100.0 * work["pi_base"]

    counts = work.groupby("gvkey")["recon_year"].nunique()
    elig = counts[counts >= 2].index
    work = work[work["gvkey"].isin(elig)].copy()

    summed = work.groupby("gvkey").agg(
        intl_amount_sum=("intl_amount", "sum"),
        pi_base_sum=("pi_base", "sum"),
        pre_obs=("recon_year", "nunique"),
    )
    ratio = (100.0 * summed["intl_amount_sum"] / summed["pi_base_sum"]).replace(
        [np.inf, -np.inf], np.nan
    )
    ratio = ratio.dropna().clip(-clip_pp, clip_pp)
    p1, p99 = ratio.quantile([0.01, 0.99])
    ratio = ratio.clip(p1, p99).rename("intl_pre")

    print(
        "ratio-sum intl_pre: "
        f"{len(ratio):,} firms  [p1,p99]=[{p1:.2f},{p99:.2f}]  "
        f"median={ratio.median():.2f}"
    )
    return ratio


def row(block: str, outcome: str, res: dict, effect_scale: str = "pp") -> dict:
    beta = res.get("coef", np.nan)
    if effect_scale == "share_pp":
        effect = -5.0 * beta * 100.0
        effect_label = "pp outcome for -10 vs -5"
    elif effect_scale == "log":
        effect = -5.0 * beta
        effect_label = "log points for -10 vs -5"
    else:
        effect = -5.0 * beta
        effect_label = "pp outcome for -10 vs -5"
    return {
        "block": block,
        "outcome": outcome,
        "beta": beta,
        "se": res.get("se", np.nan),
        "n": res.get("n", np.nan),
        "firms": res.get("G", np.nan),
        "effect_minus10_vs_minus5": effect,
        "effect_units": effect_label,
    }


def main() -> int:
    cache_dir = ROOT / "data" / "analysis" / "cache"
    combined, long_df = subst.load_data(str(cache_dir))

    wide = subst.build_channels(long_df)
    baseline_treat = subst.build_treatment(wide)
    ratio_treat = build_ratio_sum_treatment(combined, long_df)

    common = baseline_treat.index.intersection(ratio_treat.index)
    corr = baseline_treat.loc[common].corr(ratio_treat.loc[common])
    diag = {
        "baseline_firms": int(len(baseline_treat)),
        "ratio_sum_firms": int(len(ratio_treat)),
        "common_firms": int(len(common)),
        "corr_common": float(corr),
        "baseline_median": float(baseline_treat.median()),
        "ratio_sum_median": float(ratio_treat.median()),
        "baseline_p25": float(baseline_treat.quantile(0.25)),
        "baseline_p75": float(baseline_treat.quantile(0.75)),
        "ratio_sum_p25": float(ratio_treat.quantile(0.25)),
        "ratio_sum_p75": float(ratio_treat.quantile(0.75)),
    }

    rows: list[dict] = []

    panel_tax = subst.build_panel(combined, wide, ratio_treat)
    channel_labels = {**subst.CH_LABELS, "ETR": "ETR"}
    for outcome in subst.ALL_CHANNELS:
        if outcome in panel_tax.columns:
            rows.append(row(
                "tax_pass_through",
                channel_labels.get(outcome, outcome),
                subst.run_pooled_did(panel_tax, outcome, ind_yr_fe=True),
            ))
    rows.append(row("tax_pass_through", "ETR", subst.run_pooled_did(panel_tax, "etr_w", ind_yr_fe=True)))

    _, _, pi = real.load_data()
    intl_yearly = real.build_intl_yearly(long_df)
    panel_real = real.build_panel(combined, pi, ratio_treat, intl_yearly)
    real_outcomes = [
        ("foreign_share", "Foreign PI share", "share_pp"),
        ("foreign_share_log", "Log foreign PI share", "log"),
        ("pidom_int", "Domestic PI / lag assets", "share_pp"),
        ("pifo_int", "Foreign PI / lag assets", "share_pp"),
    ]
    for col, label, scale in real_outcomes:
        rows.append(row("income_location", label, real.run_pooled_did(panel_real, col, ind_yr_fe=True), scale))

    level = real.compute_level_ratio(panel_real)
    rows.append(row("income_location_common_sample", "Foreign PI share", level["indyr_share"], "share_pp"))
    rows.append(row("income_location_common_sample", "International", level["indyr_intl"]))

    out = pd.DataFrame(rows)
    out.insert(0, "treatment_definition", "ratio_sum_then_clip_winsor")
    out_path = OUT_DIR / "ratio_sum_treatment_robustness.csv"
    out.to_csv(out_path, index=False)

    diag_path = OUT_DIR / "ratio_sum_treatment_diagnostics.csv"
    pd.DataFrame([diag]).to_csv(diag_path, index=False)

    beta_share = out.loc[
        (out["block"].eq("income_location")) & (out["outcome"].eq("Foreign PI share")),
        "beta",
    ].iloc[0]
    beta_intl = out.loc[
        (out["block"].eq("tax_pass_through")) & (out["outcome"].eq("International")),
        "beta",
    ].iloc[0]
    beta_etr = out.loc[
        (out["block"].eq("tax_pass_through")) & (out["outcome"].eq("ETR")),
        "beta",
    ].iloc[0]

    md = OUT_DIR / "ratio_sum_treatment_robustness.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Ratio-of-sums treatment robustness\n\n")
        f.write("Treatment: `100 * sum(International amount) / sum(PI base)` over 2013-2016, ")
        f.write("then clipped at +/-15 pp and winsorized at p1/p99 across firms. ")
        f.write("Amount-derived rows use `implied_pi`; rate-stated rows use `comp_pi`.\n\n")
        f.write("## Treatment diagnostics\n\n")
        f.write(pd.DataFrame([diag]).to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Baseline-spec coefficients with ratio-of-sums treatment\n\n")
        f.write(out.to_markdown(index=False, floatfmt=".5f"))
        f.write("\n\n## Closure ratios\n\n")
        f.write(f"- `100 * beta_share / abs(beta_Intl)` = `{100 * beta_share / abs(beta_intl):.3f}` pp share per 1 pp International closure.\n")
        f.write(f"- `100 * beta_share / abs(beta_ETR)` = `{100 * beta_share / abs(beta_etr):.3f}` pp share per 1 pp ETR increase.\n")
        f.write(f"- `-10` vs `-5` treatment comparison: foreign share changes by `{-5 * beta_share * 100:.2f}` pp.\n")

    print(f"Wrote {out_path}")
    print(f"Wrote {diag_path}")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
