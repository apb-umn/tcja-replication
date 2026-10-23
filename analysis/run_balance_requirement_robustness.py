#!/usr/bin/env python3
"""Robustness to relaxing the balanced-panel rule.

This mirrors the slide-generating modules but changes only the sample-balance
threshold from >=2 pre and >=2 post observations to >=1 pre and >=1 post.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_CODE = ROOT / "analysis"
ANALYSIS = ROOT / "data" / "analysis"  # data / cache / output live here
OUT_DIR = ANALYSIS / "output" / "robustness"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_tax_panel(subst, combined, wide, intl_pre, min_pre: int, min_post: int) -> pd.DataFrame:
    panel = combined[combined["recon_year"].isin(subst.EVENT_YEARS)].copy()
    panel["industry"] = panel["comp_sic"].apply(subst.industry)
    panel = panel[panel["industry"] != "Finance"]
    if "comp_pi" in panel.columns:
        panel = panel[panel["comp_pi"] > 1]
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(wide, on=["gvkey", "recon_year"], how="left")
    for ch in subst.ALL_CHANNELS:
        if ch in panel.columns:
            panel[ch] = panel[ch].fillna(0.0)

    years = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = years.apply(lambda s: len(s & set(subst.PRE_YEARS)) >= min_pre)
    has_post = years.apply(lambda s: len(s & set(subst.POST_YEARS)) >= min_post)
    keep = years[has_pre & has_post].index
    return panel[panel["gvkey"].isin(keep)].copy()


def build_income_panel(real, combined, pi, intl_pre, intl_yearly, min_pre: int, min_post: int) -> pd.DataFrame:
    panel = combined[combined["recon_year"].isin(real.EVENT_YEARS)].copy()
    panel["industry"] = panel["comp_sic"].apply(real.industry)
    panel = panel[panel["industry"] != "Finance"]
    if "comp_pi" in panel.columns:
        panel = panel[panel["comp_pi"] > 1]
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(pi, on=["gvkey", "recon_year"], how="left")
    panel = panel.merge(intl_yearly, on=["gvkey", "recon_year"], how="left")
    panel["International"] = panel["International"].fillna(0.0)

    years = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = years.apply(lambda s: len(s & set(real.PRE_YEARS)) >= min_pre)
    has_post = years.apply(lambda s: len(s & set(real.POST_YEARS)) >= min_post)
    keep = years[has_pre & has_post].index
    panel = panel[panel["gvkey"].isin(keep)].copy()

    pi_total = panel["comp_pidom"].fillna(0) + panel["comp_pifo"].fillna(0)
    valid_share = (
        panel["comp_pidom"].notna()
        & panel["comp_pifo"].notna()
        & (pi_total.abs() > 0.5)
    )
    panel["foreign_share"] = np.where(valid_share, panel["comp_pifo"] / pi_total, np.nan)
    panel["foreign_share"] = panel["foreign_share"].clip(0, 1)
    panel["foreign_share_log"] = np.log(panel["foreign_share"] + real.EPS)

    panel = panel.sort_values(["gvkey", "recon_year"])
    panel["lag_at"] = panel.groupby("gvkey")["comp_at"].shift(1)
    panel["pidom_int"] = np.where(panel["lag_at"] > 1, panel["comp_pidom"] / panel["lag_at"], np.nan)
    panel["pifo_int"] = np.where(panel["lag_at"] > 1, panel["comp_pifo"] / panel["lag_at"], np.nan)
    for col in ["pidom_int", "pifo_int"]:
        panel[col] = panel[col].clip(-0.5, 0.5)
    return panel


def add_row(rows, block: str, balance: str, outcome: str, result: dict, scale: str) -> None:
    rows.append(
        {
            "block": block,
            "balance": balance,
            "outcome": outcome,
            "scale": scale,
            "beta": result["coef"],
            "se": result["se"],
            "n": result["n"],
            "firms": result["G"],
        }
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subst = load_module("subst", ANALYSIS_CODE / "build_substitution_analysis.py")
    real = load_module("real", ANALYSIS_CODE / "build_substitution_real_outcomes.py")

    combined, long_df = subst.load_data(str(ANALYSIS / "cache"))
    wide = subst.build_channels(long_df)
    subst_treat = subst.build_treatment(wide)

    real_combined, real_long, pi = real.load_data()
    real_treat = real.build_channels_and_treatment(real_long)
    intl_yearly = real.build_intl_yearly(real_long)

    rows: list[dict] = []
    for label, min_pre, min_post in [("ge2_ge2", 2, 2), ("ge1_ge1", 1, 1)]:
        tax_panel = build_tax_panel(subst, combined, wide, subst_treat, min_pre, min_post)
        for col in subst.ALL_CHANNELS:
            add_row(
                rows,
                "tax_pass_through",
                label,
                subst.CH_LABELS.get(col, col),
                subst.run_pooled_did(tax_panel, col, ind_yr_fe=True),
                "pp",
            )
        add_row(
            rows,
            "tax_pass_through",
            label,
            "ETR",
            subst.run_pooled_did(tax_panel, "etr_w", ind_yr_fe=True),
            "pp",
        )

        income_panel = build_income_panel(real, real_combined, pi, real_treat, intl_yearly, min_pre, min_post)
        for col, label_out, scale in [
            ("foreign_share", "Foreign PI share", "fraction"),
            ("foreign_share_log", "Log foreign PI share", "log"),
            ("pidom_int", "Domestic PI / lag assets", "fraction"),
            ("pifo_int", "Foreign PI / lag assets", "fraction"),
        ]:
            add_row(
                rows,
                "income_location",
                label,
                label_out,
                real.run_pooled_did(income_panel, col, ind_yr_fe=True),
                scale,
            )

    raw = pd.DataFrame(rows)
    raw_path = OUT_DIR / "balance_ge1_ge1_raw.csv"
    raw.to_csv(raw_path, index=False)

    cmp_rows = []
    for (block, outcome, scale), group in raw.groupby(["block", "outcome", "scale"], sort=False):
        by_balance = group.set_index("balance")
        base = by_balance.loc["ge2_ge2"]
        relaxed = by_balance.loc["ge1_ge1"]
        cmp_rows.append(
            {
                "block": block,
                "outcome": outcome,
                "scale": scale,
                "beta_ge2": base["beta"],
                "se_ge2": base["se"],
                "firms_ge2": int(base["firms"]),
                "n_ge2": int(base["n"]),
                "beta_ge1": relaxed["beta"],
                "se_ge1": relaxed["se"],
                "firms_ge1": int(relaxed["firms"]),
                "n_ge1": int(relaxed["n"]),
                "delta_beta": relaxed["beta"] - base["beta"],
                "pct_change_abs": (
                    (abs(relaxed["beta"]) / abs(base["beta"]) - 1) * 100
                    if abs(base["beta"]) > 1e-12
                    else np.nan
                ),
            }
        )
    comparison = pd.DataFrame(cmp_rows)
    comparison_path = OUT_DIR / "balance_ge1_ge1_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    md_path = OUT_DIR / "balance_ge1_ge1_comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Balance-rule robustness\n\n")
        f.write("Only change: relax the balanced-panel rule from `>=2` pre and `>=2` post ")
        f.write("profitable observations to `>=1` pre and `>=1` post. Treatment construction, ")
        f.write("firm FE + industry-year FE, and exclusion of 2017-2018 are unchanged.\n\n")
        f.write(comparison.to_markdown(index=False, floatfmt=".5f"))
        f.write("\n")

    print(f"Wrote {raw_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {md_path}")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
