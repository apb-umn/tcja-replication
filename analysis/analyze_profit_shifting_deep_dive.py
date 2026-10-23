#!/usr/bin/env python3
"""
analyze_profit_shifting_deep_dive.py

Deeper checks for the TCJA profit-shifting mechanism:

1. Alternative PI-location outcomes to guard against foreign-share denominator
   mechanics.
2. Event-study heterogeneity by top mechanism predictors.
3. Direct pre-TCJA domestic/foreign tax-rate wedge predictors.
4. Post-TCJA uses-of-funds outcomes.
5. Country-specific subsidiary-structure predictors.

All heterogeneity regressions use pre-TCJA predictors only and include the
lower-order Z_i x post_t term alongside intl_pre_i x post_t x Z_i.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_profit_shifting_mechanisms as mech
import build_substitution_real_outcomes as real


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUT_DIR = ROOT / "data" / "analysis" / "output"
FIG_DIR = ROOT / "data" / "analysis" / "figures" / "substitution" / "mechanisms"
BASE_CSV = OUT_DIR / "tax_reconciliation_panel_base.csv"

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})

TOP_PREDICTORS = [
    "pre_foreign_pi_share_z",
    "pre_rd_at_z",
    "pre_cash_at_z",
    "pre_foreign_rev_share_z",
    "pre_has_haven_or_conduit",
]

ALT_OUTCOMES = [
    "foreign_share",
    "foreign_share_log",
    "pifo_at",
    "pidom_at",
    "asinh_pifo_at",
    "asinh_pidom_at",
]

USES_OUTCOMES = [
    "capx_at",
    "rd_at",
    "total_inv_at",
    "debt_at",
    "cash_at",
    "dividends_at",
    "buybacks_at",
    "net_payout_at",
    "oancf_at",
]

COUNTRY_FLAGS = [
    "pre_has_ireland",
    "pre_has_netherlands",
    "pre_has_luxembourg",
    "pre_has_cayman",
    "pre_has_bermuda",
    "pre_has_singapore",
    "pre_has_hong_kong",
]

COUNTRY_LABELS = {
    "pre_has_ireland": "Ireland",
    "pre_has_netherlands": "Netherlands",
    "pre_has_luxembourg": "Luxembourg",
    "pre_has_cayman": "Cayman",
    "pre_has_bermuda": "Bermuda",
    "pre_has_singapore": "Singapore",
    "pre_has_hong_kong": "Hong Kong",
}

TAX_WEDGE_PREDICTORS = [
    "pre_foreign_tax_rate_z",
    "pre_domestic_tax_rate_z",
    "pre_domestic_foreign_tax_wedge_z",
    "pre_foreign_tax_paid_at_z",
]

TAX_WEDGE_LABELS = {
    "pre_foreign_tax_rate_z": "Foreign tax rate",
    "pre_domestic_tax_rate_z": "Domestic federal tax rate",
    "pre_domestic_foreign_tax_wedge_z": "Domestic - foreign tax rate",
    "pre_foreign_tax_paid_at_z": "Foreign tax paid / assets",
}


def load_extra_compustat() -> pd.DataFrame:
    usecols = [
        "gvkey",
        "recon_year",
        "comp_at",
        "comp_pidom",
        "comp_pifo",
        "comp_txfed",
        "comp_txfo",
        "comp_xrd",
        "comp_capx",
        "comp_dltt",
        "comp_dlc",
        "comp_che",
        "comp_dvc",
        "comp_prstkc",
        "comp_sstk",
        "comp_oancf",
    ]
    extra = pd.read_csv(BASE_CSV, usecols=usecols, low_memory=False)
    for col in usecols:
        extra[col] = pd.to_numeric(extra[col], errors="coerce")
    return extra.groupby(["gvkey", "recon_year"], as_index=False).mean(numeric_only=True)


def add_year_aware_lag(panel: pd.DataFrame, at_col: str = "comp_at_extra") -> pd.DataFrame:
    out = panel.copy()
    lag = out[["gvkey", "recon_year", at_col]].copy()
    lag["recon_year"] += 1
    lag = lag.rename(columns={at_col: "lag_at_year"})
    return out.merge(lag, on=["gvkey", "recon_year"], how="left")


def qclip(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().nunique() <= 2:
        return x
    a, b = x.quantile([lo, hi])
    return x.clip(a, b)


def add_deep_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    extra = load_extra_compustat().rename(columns={"comp_at": "comp_at_extra"})
    out = panel.merge(extra, on=["gvkey", "recon_year"], how="left", suffixes=("", "_extra"))
    out = add_year_aware_lag(out)

    valid_at = out["lag_at_year"].gt(1)
    out["pifo_at"] = np.where(valid_at, out["comp_pifo_extra"] / out["lag_at_year"], np.nan)
    out["pidom_at"] = np.where(valid_at, out["comp_pidom_extra"] / out["lag_at_year"], np.nan)
    out["asinh_pifo_at"] = np.where(valid_at, np.arcsinh(out["comp_pifo_extra"] / out["lag_at_year"]), np.nan)
    out["asinh_pidom_at"] = np.where(valid_at, np.arcsinh(out["comp_pidom_extra"] / out["lag_at_year"]), np.nan)

    out["capx_at"] = np.where(valid_at, out["comp_capx"] / out["lag_at_year"], np.nan)
    out["rd_at"] = np.where(valid_at, out["comp_xrd"] / out["lag_at_year"], np.nan)
    both_inv = valid_at & out["comp_capx"].notna() & out["comp_xrd"].notna()
    out["total_inv_at"] = np.where(both_inv, (out["comp_capx"] + out["comp_xrd"]) / out["lag_at_year"], np.nan)
    out["debt_at"] = np.where(valid_at, (out["comp_dltt"].fillna(0) + out["comp_dlc"].fillna(0)) / out["lag_at_year"], np.nan)
    out["cash_at"] = np.where(valid_at, out["comp_che"] / out["lag_at_year"], np.nan)
    out["dividends_at"] = np.where(valid_at, out["comp_dvc"] / out["lag_at_year"], np.nan)
    out["buybacks_at"] = np.where(valid_at, out["comp_prstkc"] / out["lag_at_year"], np.nan)
    out["net_payout_at"] = np.where(
        valid_at,
        (out["comp_dvc"].fillna(0) + out["comp_prstkc"].fillna(0) - out["comp_sstk"].fillna(0)) / out["lag_at_year"],
        np.nan,
    )
    out["oancf_at"] = np.where(valid_at, out["comp_oancf"] / out["lag_at_year"], np.nan)

    for col in ["pifo_at", "pidom_at"]:
        out[col] = out[col].clip(-0.5, 0.5)
    for col in ALT_OUTCOMES + USES_OUTCOMES:
        if col in out.columns:
            out[col] = qclip(out[col])
    return out


def build_tax_wedge_predictors() -> pd.DataFrame:
    usecols = [
        "gvkey",
        "recon_year",
        "comp_at",
        "comp_pidom",
        "comp_pifo",
        "comp_txfed",
        "comp_txfo",
    ]
    tax = pd.read_csv(BASE_CSV, usecols=usecols, low_memory=False)
    for col in usecols:
        tax[col] = pd.to_numeric(tax[col], errors="coerce")
    tax["gvkey_str"] = mech.zfill_gvkey(tax["gvkey"])
    tax = tax[tax["recon_year"].isin(mech.COMP_PRE_YEARS)].copy()
    tax = tax.groupby(["gvkey_str", "recon_year"], as_index=False).mean(numeric_only=True)
    tax["foreign_tax_rate"] = np.where(tax["comp_pifo"].gt(0), tax["comp_txfo"] / tax["comp_pifo"], np.nan)
    tax["domestic_tax_rate"] = np.where(tax["comp_pidom"].gt(0), tax["comp_txfed"] / tax["comp_pidom"], np.nan)
    tax["foreign_tax_rate"] = tax["foreign_tax_rate"].clip(-0.1, 0.8)
    tax["domestic_tax_rate"] = tax["domestic_tax_rate"].clip(-0.1, 0.8)
    tax["domestic_foreign_tax_wedge"] = tax["domestic_tax_rate"] - tax["foreign_tax_rate"]
    tax["foreign_tax_paid_at"] = np.where(tax["comp_at"].gt(0), tax["comp_txfo"] / tax["comp_at"], np.nan)
    pre = tax.groupby("gvkey_str").agg(
        pre_foreign_tax_rate=("foreign_tax_rate", "mean"),
        pre_domestic_tax_rate=("domestic_tax_rate", "mean"),
        pre_domestic_foreign_tax_wedge=("domestic_foreign_tax_wedge", "mean"),
        pre_foreign_tax_paid_at=("foreign_tax_paid_at", "mean"),
    ).reset_index()
    pre, _ = mech.standardize_columns(
        pre,
        [
            "pre_foreign_tax_rate",
            "pre_domestic_tax_rate",
            "pre_domestic_foreign_tax_wedge",
            "pre_foreign_tax_paid_at",
        ],
    )
    return pre


def prepare_panel() -> pd.DataFrame:
    base = mech.load_base_panel()
    predictors, _ = mech.build_predictors(base)
    panel = mech.add_predictors_to_panel(base, predictors)
    panel = panel.merge(build_tax_wedge_predictors(), on="gvkey_str", how="left")
    panel = add_deep_outcomes(panel)
    return panel


def interaction_rows(panel: pd.DataFrame, outcomes: list[str], predictors: list[str], label_map: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in panel.columns:
            continue
        for pred in predictors:
            if pred not in panel.columns:
                continue
            model = mech.run_interaction_model(panel, outcome, [pred], ind_yr_fe=True)
            if not model.get("ok"):
                continue
            gamma_name = f"_Di_x_{pred}"
            gamma = model["params"][gamma_name]
            se = model["ses"][gamma_name]
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": pred,
                    "label": (label_map or {}).get(pred, mech.PREDICTOR_META.get(pred, mech.Predictor(pred, pred, "custom")).label),
                    "n": model["n"],
                    "firms": model["firms"],
                    "base_coef": float(model["params"]["_Di"]),
                    "base_se": float(model["ses"]["_Di"]),
                    "interaction_gamma": float(gamma),
                    "interaction_se": float(se),
                    "interaction_t": float(gamma / se) if se > 0 else np.nan,
                    "interaction_p_norm": mech.normal_pvalue(gamma / se) if se > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main_effect_rows(panel: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in panel.columns:
            continue
        res = real.run_pooled_did(panel, outcome, ind_yr_fe=True)
        rows.append(
            {
                "outcome": outcome,
                "coef_on_intl_pre_post": res["coef"],
                "se": res["se"],
                "n": int(res["n"]),
                "firms": int(res["G"]),
                "effect_per_1pp_closure": -res["coef"],
            }
        )
    return pd.DataFrame(rows)


def group_definitions(panel: pd.DataFrame, pred: str) -> pd.Series:
    firm = panel.drop_duplicates("gvkey")[["gvkey", pred]].dropna().copy()
    if firm[pred].dropna().isin([0, 1]).all() and firm[pred].nunique() <= 2:
        mapping = {0.0: "No", 1.0: "Yes"}
        firm["group"] = firm[pred].map(mapping)
    else:
        q1, q2 = firm[pred].quantile([1 / 3, 2 / 3])
        firm["group"] = pd.NA
        firm.loc[firm[pred] <= q1, "group"] = "Low"
        firm.loc[firm[pred] >= q2, "group"] = "High"
    return firm.dropna(subset=["group"]).set_index("gvkey")["group"]


def event_study_groups(panel: pd.DataFrame, predictors: list[str], outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    pooled_rows = []
    for pred in predictors:
        if pred not in panel.columns:
            continue
        groups = group_definitions(panel, pred)
        for group_name in ["Low", "High", "No", "Yes"]:
            gvkeys = groups[groups.eq(group_name)].index
            if len(gvkeys) == 0:
                continue
            sub = panel[panel["gvkey"].isin(gvkeys)].copy()
            for outcome in outcomes:
                ev = real.run_event_study(sub, outcome, ind_yr_fe=True)
                if not ev.empty:
                    ev["predictor"] = pred
                    ev["label"] = mech.PREDICTOR_META.get(pred, mech.Predictor(pred, pred, "custom")).label
                    ev["group"] = group_name
                    ev["outcome"] = outcome
                    ev["firms_group"] = int(sub["gvkey"].nunique())
                    event_rows.append(ev)
                res = real.run_pooled_did(sub, outcome, ind_yr_fe=True)
                pooled_rows.append(
                    {
                        "predictor": pred,
                        "label": mech.PREDICTOR_META.get(pred, mech.Predictor(pred, pred, "custom")).label,
                        "group": group_name,
                        "outcome": outcome,
                        "coef": res["coef"],
                        "se": res["se"],
                        "n": int(res["n"]),
                        "firms": int(res["G"]),
                    }
                )
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    return events, pd.DataFrame(pooled_rows)


def plot_event_heterogeneity(events: pd.DataFrame) -> Path | None:
    if events.empty:
        return None
    plot_preds = [
        "pre_foreign_pi_share_z",
        "pre_rd_at_z",
        "pre_cash_at_z",
        "pre_foreign_rev_share_z",
        "pre_has_haven_or_conduit",
    ]
    fig, axes = plt.subplots(len(plot_preds), 1, figsize=(7.6, 12.5), sharex=True)
    colors = {"Low": "#7f8c8d", "High": "#c0392b", "No": "#7f8c8d", "Yes": "#c0392b"}
    for ax, pred in zip(axes, plot_preds):
        sub = events[(events["predictor"].eq(pred)) & (events["outcome"].eq("foreign_share"))].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        for group, g in sub.groupby("group"):
            g = g.sort_values("year")
            ax.errorbar(
                g["year"],
                g["beta"],
                yerr=1.96 * g["se"],
                fmt="o-",
                lw=1.3,
                ms=3.5,
                capsize=2,
                color=colors.get(group, "#333333"),
                label=group,
            )
        title = sub["label"].iloc[0]
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(2016, color="#c0392b", lw=0.8, ls="--", alpha=0.7)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylabel(r"$\beta_y$")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Foreign PI share event studies by pre-TCJA mechanism group", y=0.995)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig_profit_shifting_mechanism_event_groups.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(
    alt_main: pd.DataFrame,
    alt_interactions: pd.DataFrame,
    tax_wedge: pd.DataFrame,
    uses_main: pd.DataFrame,
    uses_interactions: pd.DataFrame,
    country: pd.DataFrame,
    event_pooled: pd.DataFrame,
    fig_path: Path | None,
) -> Path:
    path = OUT_DIR / "profit_shifting_deep_dive_summary.md"
    lines = [
        "# Profit-shifting mechanism deep dive",
        "",
        "All heterogeneity models include firm FE, industry x year FE, `intl_pre x post`, `Z x post`, and `intl_pre x post x Z`. Continuous pre-TCJA predictors are standardized.",
        "",
        "## Alternative PI-location outcomes: main treatment effect",
        "",
        alt_main.round(5).to_markdown(index=False),
        "",
        "## Alternative outcome heterogeneity by top predictors",
        "",
        alt_interactions.sort_values(["outcome", "interaction_p_norm"]).round(5).to_markdown(index=False),
        "",
        "## Direct tax-rate wedge predictors",
        "",
        tax_wedge.sort_values(["outcome", "interaction_p_norm"]).round(5).to_markdown(index=False),
        "",
        "## Uses-of-funds outcomes: main treatment effect",
        "",
        uses_main.round(5).to_markdown(index=False),
        "",
        "## Uses-of-funds heterogeneity by top predictors",
        "",
        uses_interactions.sort_values(["outcome", "interaction_p_norm"]).round(5).to_markdown(index=False),
        "",
        "## Country-specific subsidiary predictors",
        "",
        country.sort_values("interaction_p_norm").round(5).to_markdown(index=False),
        "",
        "## Grouped event-study pooled effects",
        "",
        event_pooled.round(5).to_markdown(index=False),
        "",
    ]
    if fig_path is not None:
        lines.extend(["## Event-study figure", "", str(fig_path), ""])
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    os.chdir(SCRIPT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()

    alt_main = main_effect_rows(panel, ALT_OUTCOMES)
    alt_interactions = interaction_rows(panel, ALT_OUTCOMES, TOP_PREDICTORS)
    tax_wedge = interaction_rows(panel, ["foreign_share", "pifo_at", "pidom_at"], TAX_WEDGE_PREDICTORS, TAX_WEDGE_LABELS)
    uses_main = main_effect_rows(panel, USES_OUTCOMES)
    uses_interactions = interaction_rows(panel, USES_OUTCOMES, ["pre_foreign_pi_share_z", "pre_rd_at_z", "pre_cash_at_z"])
    country = interaction_rows(panel, ["foreign_share"], COUNTRY_FLAGS, COUNTRY_LABELS)
    event_rows, event_pooled = event_study_groups(
        panel,
        TOP_PREDICTORS,
        ["foreign_share", "pifo_at", "pidom_at"],
    )
    fig_path = plot_event_heterogeneity(event_rows)

    panel.to_csv(OUT_DIR / "profit_shifting_deep_dive_panel.csv", index=False)
    alt_main.to_csv(OUT_DIR / "profit_shifting_deep_dive_alt_outcome_main.csv", index=False)
    alt_interactions.to_csv(OUT_DIR / "profit_shifting_deep_dive_alt_outcome_interactions.csv", index=False)
    tax_wedge.to_csv(OUT_DIR / "profit_shifting_deep_dive_tax_wedge_interactions.csv", index=False)
    uses_main.to_csv(OUT_DIR / "profit_shifting_deep_dive_uses_main.csv", index=False)
    uses_interactions.to_csv(OUT_DIR / "profit_shifting_deep_dive_uses_interactions.csv", index=False)
    country.to_csv(OUT_DIR / "profit_shifting_deep_dive_country_interactions.csv", index=False)
    event_rows.to_csv(OUT_DIR / "profit_shifting_deep_dive_event_study_groups.csv", index=False)
    event_pooled.to_csv(OUT_DIR / "profit_shifting_deep_dive_event_study_pooled.csv", index=False)
    summary_path = write_summary(
        alt_main,
        alt_interactions,
        tax_wedge,
        uses_main,
        uses_interactions,
        country,
        event_pooled,
        fig_path,
    )

    print("\nAlternative PI-location outcomes")
    print(alt_main.round(5).to_string(index=False))
    print("\nTop alternative-outcome interactions")
    print(
        alt_interactions.sort_values("interaction_p_norm")
        .head(18)
        .round(5)
        .to_string(index=False)
    )
    print("\nTax-wedge interactions")
    print(tax_wedge.sort_values("interaction_p_norm").round(5).to_string(index=False))
    print("\nUses-of-funds main effects")
    print(uses_main.round(5).to_string(index=False))
    print(f"\nWrote {summary_path}")
    if fig_path:
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
