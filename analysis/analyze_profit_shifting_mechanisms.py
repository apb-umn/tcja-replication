#!/usr/bin/env python3
"""
analyze_profit_shifting_mechanisms.py

Mechanism heterogeneity for the TCJA profit-shifting response.

The outcome and treatment match build_substitution_real_outcomes.py:

    foreign_share_it = pifo_it / (pidom_it + pifo_it)
    treatment_i      = pre-TCJA International reconciliation contribution
    post_t           = 2019-2022

For each pre-TCJA predictor Z_i, estimate:

    y_it = firm FE + industry x year FE
           + beta * intl_pre_i * post_t
           + delta * Z_i * post_t
           + gamma * intl_pre_i * post_t * Z_i + e_it

All Z_i variables are measured before TCJA.  Continuous predictors are
winsorized at p1/p99 and standardized at the firm level.  Binary predictors
remain 0/1.  Subsidiary-footprint predictors use 2013-2015 because annual
subsidiary coverage is sparse in 2016 in the WRDS export currently downloaded.
Other Compustat and geographic-segment predictors use 2013-2016.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

import build_substitution_real_outcomes as real


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ROOT_DIR = SCRIPT_DIR.parent
OUT_DIR = ROOT / "data" / "analysis" / "output"
TABLE_DIR = ROOT / "data" / "analysis" / "tables" / "substitution"
BASE_CSV = OUT_DIR / "tax_reconciliation_panel_base.csv"
GEO_PANEL = OUT_DIR / "geo_revenue_share_tcja_panel.csv"
SUBSIDIARY_PANEL = OUT_DIR / "subsidiary_footprint_panel.csv"

COMP_PRE_YEARS = [2013, 2014, 2015, 2016]
SUB_PRE_YEARS = [2013, 2014, 2015]
REG_YEARS = [yr for yr in real.EVENT_YEARS if yr not in {2017, 2018}]


@dataclass(frozen=True)
class Predictor:
    col: str
    label: str
    family: str
    binary: bool = False


PREDICTORS = [
    Predictor("pre_intl_benefit_z", "Pre-TCJA intl tax benefit", "tax_planning_ip"),
    Predictor("pre_rd_at_z", "R&D / assets", "tax_planning_ip"),
    Predictor("pre_xsga_cogs_share_z", "SG&A / (SG&A + COGS)", "tax_planning_ip"),
    Predictor("pre_intan_ex_gw_at_z", "Intangibles ex-goodwill / assets", "tax_planning_ip"),
    Predictor("pre_intan_at_z", "Total intangibles / assets", "tax_planning_ip"),
    Predictor("pre_foreign_pi_share_z", "Foreign PI share", "tax_planning_ip"),
    Predictor("is_tech_or_pharma", "Tech or pharma", "tax_planning_ip", binary=True),
    Predictor("pre_has_haven_or_conduit", "Any haven/conduit subsidiary", "entity_network", binary=True),
    Predictor("pre_has_classic_haven", "Any classic-haven subsidiary", "entity_network", binary=True),
    Predictor("pre_has_conduit", "Any conduit-hub subsidiary", "entity_network", binary=True),
    Predictor("pre_log1p_haven_subs_z", "Log haven/conduit subsidiary count", "entity_network"),
    Predictor("pre_log1p_classic_haven_subs_z", "Log classic-haven subsidiary count", "entity_network"),
    Predictor("pre_log1p_conduit_subs_z", "Log conduit subsidiary count", "entity_network"),
    Predictor("pre_has_ireland_netherlands_lux", "Ireland/Netherlands/Luxembourg", "entity_network", binary=True),
    Predictor("pre_has_cayman_bermuda", "Cayman/Bermuda", "entity_network", binary=True),
    Predictor("pre_foreign_rev_share_z", "Foreign revenue share", "real_footprint"),
    Predictor("pre_foreign_sub_share_z", "Foreign subsidiary share", "real_footprint"),
    Predictor("pre_log1p_foreign_subs_z", "Log foreign subsidiary count", "real_footprint"),
    Predictor("pre_log1p_foreign_country_count_z", "Log foreign country count", "real_footprint"),
    Predictor("pre_log_at_z", "Log assets", "financial_capacity"),
    Predictor("pre_log_sale_z", "Log sales", "financial_capacity"),
    Predictor("pre_leverage_z", "Debt / assets", "financial_capacity"),
    Predictor("pre_cash_at_z", "Cash / assets", "financial_capacity"),
    Predictor("pre_capx_at_z", "Capex / assets", "financial_capacity"),
    Predictor("pre_ppent_at_z", "PPE / assets", "financial_capacity"),
    Predictor("pre_market_to_book_z", "Market-to-book", "financial_capacity"),
]

HORSE_RACE_SPECS = {
    "tax_planning_ip": [
        "pre_intl_benefit_z",
        "pre_rd_at_z",
        "pre_xsga_cogs_share_z",
        "pre_intan_ex_gw_at_z",
        "is_tech_or_pharma",
    ],
    "entity_network": [
        "pre_has_haven_or_conduit",
        "pre_log1p_haven_subs_z",
        "pre_log1p_foreign_country_count_z",
        "pre_has_ireland_netherlands_lux",
    ],
    "real_footprint": [
        "pre_foreign_rev_share_z",
        "pre_foreign_sub_share_z",
        "pre_log1p_foreign_subs_z",
    ],
    "financial_capacity": [
        "pre_log_at_z",
        "pre_leverage_z",
        "pre_cash_at_z",
        "pre_market_to_book_z",
    ],
    "combined_core": [
        "pre_intl_benefit_z",
        "pre_rd_at_z",
        "pre_xsga_cogs_share_z",
        "pre_intan_ex_gw_at_z",
        "pre_foreign_rev_share_z",
        "pre_has_haven_or_conduit",
        "pre_log1p_haven_subs_z",
        "pre_log_at_z",
        "pre_leverage_z",
        "pre_cash_at_z",
    ],
}

PREDICTOR_META = {p.col: p for p in PREDICTORS}


def normal_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))


def zfill_gvkey(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def safe_ratio(num: pd.Series, den: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    out = np.where(den.gt(0), num / den, np.nan)
    out = pd.Series(out, index=num.index, dtype=float)
    if lower is not None or upper is not None:
        out = out.clip(lower=lower, upper=upper)
    return out


def winsorize(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().nunique() <= 2:
        return x
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


def standardize_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    stats = {}
    for col in columns:
        x = winsorize(out[col])
        mu = x.mean()
        sd = x.std(ddof=0)
        stats[col] = {
            "mean_after_winsor": float(mu) if pd.notna(mu) else None,
            "sd_after_winsor": float(sd) if pd.notna(sd) else None,
            "n": int(x.notna().sum()),
        }
        out[f"{col}_z"] = (x - mu) / sd if pd.notna(sd) and sd > 0 else np.nan
    return out, stats


def load_base_panel() -> pd.DataFrame:
    combined, long_df, pi = real.load_data()
    intl_pre = real.build_channels_and_treatment(long_df)
    intl_yearly = real.build_intl_yearly(long_df)
    panel = real.build_panel(combined, pi, intl_pre, intl_yearly)
    panel = panel.copy()
    panel["gvkey_str"] = zfill_gvkey(panel["gvkey"])
    panel["is_tech_or_pharma"] = panel["industry"].isin(["Tech", "Pharma/Biotech"]).astype(float)
    return panel


def load_compustat_pre_predictors() -> tuple[pd.DataFrame, dict]:
    usecols = [
        "gvkey",
        "recon_year",
        "comp_at",
        "comp_sale",
        "comp_cogs",
        "comp_xsga",
        "comp_xrd",
        "comp_intan",
        "comp_gdwl",
        "comp_capx",
        "comp_ppent",
        "comp_dltt",
        "comp_dlc",
        "comp_che",
        "comp_pidom",
        "comp_pifo",
        "comp_mkvalt",
        "comp_prcc_f",
        "comp_csho",
        "comp_ceq",
    ]
    comp = pd.read_csv(BASE_CSV, usecols=usecols, low_memory=False)
    comp["gvkey_str"] = zfill_gvkey(comp["gvkey"])
    for col in usecols:
        if col not in {"gvkey", "recon_year"}:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
    comp = comp[comp["recon_year"].isin(COMP_PRE_YEARS)].copy()
    comp = comp.groupby(["gvkey_str", "recon_year"], as_index=False).mean(numeric_only=True)

    at = comp["comp_at"]
    sale = comp["comp_sale"]
    xrd_zero = comp["comp_xrd"].fillna(0)
    gdwl_zero = comp["comp_gdwl"].fillna(0)
    intan = comp["comp_intan"]
    comp["rd_at"] = safe_ratio(xrd_zero, at, lower=0)
    comp["intan_at"] = safe_ratio(intan, at, lower=0)
    comp["gdwl_at"] = safe_ratio(gdwl_zero, at, lower=0)
    comp["intan_ex_gw_at"] = safe_ratio((intan.fillna(0) - gdwl_zero).clip(lower=0), at, lower=0)
    comp["leverage"] = safe_ratio(comp["comp_dltt"].fillna(0) + comp["comp_dlc"].fillna(0), at, lower=0)
    comp["cash_at"] = safe_ratio(comp["comp_che"], at, lower=0)
    comp["capx_at"] = safe_ratio(comp["comp_capx"], at, lower=0)
    comp["ppent_at"] = safe_ratio(comp["comp_ppent"], at, lower=0)
    ops_den = comp["comp_xsga"] + comp["comp_cogs"]
    comp["xsga_cogs_share"] = np.where(
        comp["comp_xsga"].notna()
        & comp["comp_cogs"].notna()
        & comp["comp_xsga"].ge(0)
        & comp["comp_cogs"].ge(0)
        & ops_den.gt(0),
        comp["comp_xsga"] / ops_den,
        np.nan,
    )
    comp["xsga_cogs_share"] = pd.Series(comp["xsga_cogs_share"], index=comp.index).clip(0, 1)
    comp["log_at"] = np.where(at.gt(0), np.log(at), np.nan)
    comp["log_sale"] = np.where(sale.gt(0), np.log(sale), np.nan)
    market_value = comp["comp_mkvalt"]
    fallback_market = comp["comp_prcc_f"] * comp["comp_csho"]
    market_value = market_value.where(market_value.notna(), fallback_market)
    comp["market_to_book"] = safe_ratio(market_value, comp["comp_ceq"], lower=0)
    pi_total = comp["comp_pidom"].fillna(0) + comp["comp_pifo"].fillna(0)
    comp["foreign_pi_share_pre"] = np.where(
        comp["comp_pidom"].notna() & comp["comp_pifo"].notna() & pi_total.abs().gt(0.5),
        comp["comp_pifo"] / pi_total,
        np.nan,
    )
    comp["foreign_pi_share_pre"] = pd.Series(comp["foreign_pi_share_pre"]).clip(0, 1)

    pre = comp.groupby("gvkey_str").agg(
        pre_log_at=("log_at", "mean"),
        pre_log_sale=("log_sale", "mean"),
        pre_rd_at=("rd_at", "mean"),
        pre_xsga_cogs_share=("xsga_cogs_share", "mean"),
        pre_intan_at=("intan_at", "mean"),
        pre_gdwl_at=("gdwl_at", "mean"),
        pre_intan_ex_gw_at=("intan_ex_gw_at", "mean"),
        pre_leverage=("leverage", "mean"),
        pre_cash_at=("cash_at", "mean"),
        pre_capx_at=("capx_at", "mean"),
        pre_ppent_at=("ppent_at", "mean"),
        pre_market_to_book=("market_to_book", "mean"),
        pre_foreign_pi_share=("foreign_pi_share_pre", "mean"),
    ).reset_index()

    z_cols = [
        "pre_log_at",
        "pre_log_sale",
        "pre_rd_at",
        "pre_xsga_cogs_share",
        "pre_intan_at",
        "pre_gdwl_at",
        "pre_intan_ex_gw_at",
        "pre_leverage",
        "pre_cash_at",
        "pre_capx_at",
        "pre_ppent_at",
        "pre_market_to_book",
        "pre_foreign_pi_share",
    ]
    return standardize_columns(pre, z_cols)


def load_geo_pre_predictors() -> tuple[pd.DataFrame, dict]:
    # [replication package] optional input: when the file is absent the
    # predictors built from it are skipped (see README, section 4.3).
    if not GEO_PANEL.exists():
        print(f"  NOTE: {GEO_PANEL.name} not present (needs Compustat Segments); foreign-revenue-share predictor skipped")
        return pd.DataFrame({"gvkey_str": pd.Series(dtype=str)}), {"skipped": "geo_revenue_share_tcja_panel.csv not present"}
    geo = pd.read_csv(GEO_PANEL, usecols=["gvkey", "recon_year", "foreign_rev_share", "total_geo_sales"])
    geo["gvkey_str"] = zfill_gvkey(geo["gvkey"])
    geo = geo[geo["recon_year"].isin(COMP_PRE_YEARS)].copy()
    pre = geo.groupby("gvkey_str").agg(
        pre_foreign_rev_share=("foreign_rev_share", "mean"),
        pre_total_geo_sales=("total_geo_sales", "mean"),
        pre_geo_years=("foreign_rev_share", lambda x: int(x.notna().sum())),
    ).reset_index()
    return standardize_columns(pre, ["pre_foreign_rev_share"])


def load_subsidiary_pre_predictors() -> tuple[pd.DataFrame, dict]:
    # [replication package] optional input: when the file is absent the
    # predictors built from it are skipped (see README, section 4.3).
    if not SUBSIDIARY_PANEL.exists():
        print(f"  NOTE: {SUBSIDIARY_PANEL.name} not present (needs Exhibit 21 subsidiary data); subsidiary predictors skipped")
        return pd.DataFrame({"gvkey_str": pd.Series(dtype=str)}), {"skipped": "subsidiary_footprint_panel.csv not present"}
    sub = pd.read_csv(SUBSIDIARY_PANEL, dtype={"gvkey": str}, low_memory=False)
    sub["gvkey_str"] = zfill_gvkey(sub["gvkey"])
    sub = sub[sub["report_year"].isin(SUB_PRE_YEARS)].copy()
    pre = sub.groupby("gvkey_str").agg(
        pre_sub_years=("report_year", "nunique"),
        pre_total_subs=("total_subs", "mean"),
        pre_foreign_subs=("foreign_subs", "mean"),
        pre_foreign_sub_share=("foreign_sub_share", "mean"),
        pre_foreign_country_count=("foreign_country_count", "mean"),
        pre_haven_subs=("haven_subs", "mean"),
        pre_classic_haven_subs=("classic_haven_subs", "mean"),
        pre_conduit_subs=("conduit_subs", "mean"),
        pre_haven_country_count=("haven_country_count", "mean"),
        pre_has_foreign_sub=("has_foreign_sub", "max"),
        pre_has_haven_or_conduit=("has_haven_sub", "max"),
        pre_has_classic_haven=("has_classic_haven_sub", "max"),
        pre_has_conduit=("has_conduit_sub", "max"),
        pre_has_ireland=("has_country_ireland", "max"),
        pre_has_netherlands=("has_country_netherlands", "max"),
        pre_has_luxembourg=("has_country_luxembourg", "max"),
        pre_has_cayman=("has_country_cayman", "max"),
        pre_has_bermuda=("has_country_bermuda", "max"),
        pre_has_singapore=("has_country_singapore", "max"),
        pre_has_hong_kong=("has_country_hong_kong", "max"),
    ).reset_index()
    count_cols = [
        "pre_total_subs",
        "pre_foreign_subs",
        "pre_foreign_country_count",
        "pre_haven_subs",
        "pre_classic_haven_subs",
        "pre_conduit_subs",
        "pre_haven_country_count",
    ]
    for col in count_cols:
        pre[f"pre_log1p_{col.removeprefix('pre_')}"] = np.log1p(pre[col])
    pre["pre_has_ireland_netherlands_lux"] = (
        pre[["pre_has_ireland", "pre_has_netherlands", "pre_has_luxembourg"]].max(axis=1)
    )
    pre["pre_has_cayman_bermuda"] = pre[["pre_has_cayman", "pre_has_bermuda"]].max(axis=1)

    binary_cols = [
        "pre_has_foreign_sub",
        "pre_has_haven_or_conduit",
        "pre_has_classic_haven",
        "pre_has_conduit",
        "pre_has_ireland_netherlands_lux",
        "pre_has_cayman_bermuda",
        "pre_has_singapore",
        "pre_has_hong_kong",
    ]
    for col in binary_cols:
        pre[col] = pre[col].astype(float)

    z_cols = [
        "pre_foreign_sub_share",
        "pre_log1p_foreign_subs",
        "pre_log1p_foreign_country_count",
        "pre_log1p_haven_subs",
        "pre_log1p_classic_haven_subs",
        "pre_log1p_conduit_subs",
        "pre_log1p_haven_country_count",
    ]
    return standardize_columns(pre, z_cols)


def build_predictors(base_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    firm = (
        base_panel.groupby("gvkey_str", as_index=False)
        .agg(
            gvkey=("gvkey", "first"),
            industry=("industry", "first"),
            intl_pre=("intl_pre", "first"),
            is_tech_or_pharma=("is_tech_or_pharma", "first"),
        )
    )
    firm["pre_intl_benefit"] = -firm["intl_pre"]
    firm, base_stats = standardize_columns(firm, ["pre_intl_benefit"])

    comp, comp_stats = load_compustat_pre_predictors()
    geo, geo_stats = load_geo_pre_predictors()
    sub, sub_stats = load_subsidiary_pre_predictors()
    predictors = firm.merge(comp, on="gvkey_str", how="left")
    predictors = predictors.merge(geo, on="gvkey_str", how="left")
    predictors = predictors.merge(sub, on="gvkey_str", how="left")

    stats = {
        "base": base_stats,
        "compustat": comp_stats,
        "geo_segments": geo_stats,
        "subsidiaries": sub_stats,
    }
    return predictors, stats


def add_predictors_to_panel(base_panel: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in predictors.columns if c not in {"gvkey", "industry", "intl_pre"}]
    return base_panel.merge(predictors[pred_cols], on="gvkey_str", how="left")


def regression_sample(panel: pd.DataFrame, outcome: str, z_cols: list[str]) -> pd.DataFrame:
    mask = (
        panel["recon_year"].isin(REG_YEARS)
        & panel[outcome].notna()
        & panel["intl_pre"].notna()
    )
    for col in z_cols:
        mask &= panel[col].notna()
    return panel[mask].copy()


def run_interaction_model(panel: pd.DataFrame, outcome: str, z_cols: list[str], ind_yr_fe: bool = True) -> dict:
    sub = regression_sample(panel, outcome, z_cols)
    if len(sub) < 50 or sub["gvkey"].nunique() < 20:
        return {"ok": False, "n": int(len(sub)), "firms": int(sub["gvkey"].nunique())}

    sub["_post"] = sub["recon_year"].isin(real.POST_YEARS).astype(float)
    sub["_Di"] = sub["intl_pre"] * sub["_post"]
    x_cols = ["_Di"]
    for col in z_cols:
        zpost = f"_post_x_{col}"
        ix = f"_Di_x_{col}"
        sub[zpost] = sub["_post"] * sub[col].astype(float)
        sub[ix] = sub["_Di"] * sub[col].astype(float)
        x_cols.extend([zpost, ix])

    cols = [outcome] + x_cols
    dm = real.demean_ind_yr(sub, cols) if ind_yr_fe else real.demean_twfe(sub, cols)
    mod = OLS(dm[outcome].values, dm[x_cols].values).fit()
    cv = cov_cluster(mod, sub["gvkey"].values)
    params = pd.Series(np.asarray(mod.params), index=x_cols)
    ses = pd.Series(np.sqrt(np.diag(cv)), index=x_cols)
    return {
        "ok": True,
        "params": params,
        "ses": ses,
        "cov": pd.DataFrame(cv, index=x_cols, columns=x_cols),
        "n": int(len(sub)),
        "firms": int(sub["gvkey"].nunique()),
        "z_cols": z_cols,
    }


def coeff_at(model: dict, z_col: str, z_value: float) -> tuple[float, float]:
    beta = model["params"]["_Di"]
    gamma_name = f"_Di_x_{z_col}"
    gamma = model["params"][gamma_name]
    cov = model["cov"]
    coef = beta + z_value * gamma
    var = (
        cov.loc["_Di", "_Di"]
        + z_value**2 * cov.loc[gamma_name, gamma_name]
        + 2 * z_value * cov.loc["_Di", gamma_name]
    )
    return float(coef), float(np.sqrt(max(var, 0)))


def run_oneway(panel: pd.DataFrame, outcome: str = "foreign_share") -> pd.DataFrame:
    rows = []
    first_stage_cache = {}
    for pred in PREDICTORS:
        z_col = pred.col
        if z_col not in panel.columns:
            continue
        model = run_interaction_model(panel, outcome, [z_col], ind_yr_fe=True)
        first_stage = run_interaction_model(panel, "International", [z_col], ind_yr_fe=True)
        if not model.get("ok"):
            continue
        first_stage_cache[z_col] = first_stage
        gamma_name = f"_Di_x_{z_col}"
        beta = model["params"]["_Di"]
        beta_se = model["ses"]["_Di"]
        gamma = model["params"][gamma_name]
        gamma_se = model["ses"][gamma_name]
        t_gamma = gamma / gamma_se if gamma_se > 0 else np.nan

        z_low, z_high = (0.0, 1.0) if pred.binary else (-1.0, 1.0)
        low_coef, low_se = coeff_at(model, z_col, z_low)
        high_coef, high_se = coeff_at(model, z_col, z_high)
        if first_stage.get("ok"):
            low_fs, _ = coeff_at(first_stage, z_col, z_low)
            high_fs, _ = coeff_at(first_stage, z_col, z_high)
            low_ratio = 100 * low_coef / abs(low_fs) if abs(low_fs) > 0 else np.nan
            high_ratio = 100 * high_coef / abs(high_fs) if abs(high_fs) > 0 else np.nan
        else:
            low_fs = high_fs = low_ratio = high_ratio = np.nan

        rows.append(
            {
                "outcome": outcome,
                "predictor": z_col,
                "label": pred.label,
                "family": pred.family,
                "binary": pred.binary,
                "n": model["n"],
                "firms": model["firms"],
                "coef_at_mean_or_zero": float(beta),
                "se_at_mean_or_zero": float(beta_se),
                "interaction_gamma": float(gamma),
                "interaction_se": float(gamma_se),
                "interaction_t": float(t_gamma),
                "interaction_p_norm": normal_pvalue(t_gamma),
                "coef_low_z_or_zero": low_coef,
                "se_low_z_or_zero": low_se,
                "first_stage_low_z_or_zero": low_fs,
                "closure_ratio_low_z_or_zero_pp": low_ratio,
                "coef_high_z_or_one": high_coef,
                "se_high_z_or_one": high_se,
                "first_stage_high_z_or_one": high_fs,
                "closure_ratio_high_z_or_one_pp": high_ratio,
                "high_minus_low_coef": high_coef - low_coef,
                "high_minus_low_closure_ratio_pp": high_ratio - low_ratio,
            }
        )
    return pd.DataFrame(rows)


def run_horse_races(panel: pd.DataFrame, outcome: str = "foreign_share") -> pd.DataFrame:
    rows = []
    for spec_name, z_cols in HORSE_RACE_SPECS.items():
        z_cols = [z for z in z_cols if z in panel.columns]
        model = run_interaction_model(panel, outcome, z_cols, ind_yr_fe=True)
        if not model.get("ok"):
            continue
        for z_col in z_cols:
            pred = PREDICTOR_META[z_col]
            gamma_name = f"_Di_x_{z_col}"
            gamma = model["params"][gamma_name]
            gamma_se = model["ses"][gamma_name]
            t_gamma = gamma / gamma_se if gamma_se > 0 else np.nan
            rows.append(
                {
                    "outcome": outcome,
                    "spec": spec_name,
                    "predictor": z_col,
                    "label": pred.label,
                    "family": pred.family,
                    "binary": pred.binary,
                    "n": model["n"],
                    "firms": model["firms"],
                    "base_coef": float(model["params"]["_Di"]),
                    "base_se": float(model["ses"]["_Di"]),
                    "interaction_gamma": float(gamma),
                    "interaction_se": float(gamma_se),
                    "interaction_t": float(t_gamma),
                    "interaction_p_norm": normal_pvalue(t_gamma),
                }
            )
    return pd.DataFrame(rows)


def predictor_coverage(predictors: pd.DataFrame, base_panel: pd.DataFrame) -> pd.DataFrame:
    reg_firms = (
        base_panel[
            base_panel["recon_year"].isin(REG_YEARS)
            & base_panel["foreign_share"].notna()
            & base_panel["intl_pre"].notna()
        ]["gvkey_str"]
        .drop_duplicates()
        .to_frame()
    )
    firm = reg_firms.merge(predictors, on="gvkey_str", how="left")
    rows = []
    for pred in PREDICTORS:
        if pred.col not in firm.columns:
            continue
        x = firm[pred.col]
        row = {
            "predictor": pred.col,
            "label": pred.label,
            "family": pred.family,
            "binary": pred.binary,
            "firms_with_value": int(x.notna().sum()),
            "coverage_frac": float(x.notna().mean()),
        }
        if pred.binary:
            row["mean_or_share"] = float(x.mean())
            row["sd"] = float(x.std(ddof=0))
        else:
            raw = pred.col.removesuffix("_z")
            if raw in firm.columns:
                row["mean_or_share"] = float(firm[raw].mean())
                row["sd"] = float(firm[raw].std(ddof=0))
            else:
                row["mean_or_share"] = float(x.mean())
                row["sd"] = float(x.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def write_mechanism_markdown(oneway: pd.DataFrame, horse: pd.DataFrame, coverage: pd.DataFrame) -> Path:
    path = OUT_DIR / "profit_shifting_mechanism_summary.md"
    top = oneway.sort_values("interaction_p_norm").head(12).copy()
    cols = [
        "label",
        "family",
        "n",
        "firms",
        "interaction_gamma",
        "interaction_se",
        "interaction_p_norm",
        "closure_ratio_low_z_or_zero_pp",
        "closure_ratio_high_z_or_one_pp",
    ]
    lines = [
        "# Profit-shifting mechanism heterogeneity",
        "",
        "Outcome: foreign PI share.  Treatment: pre-TCJA International reconciliation contribution × post.  Specification: firm FE + industry × year FE; SE clustered by firm.",
        "",
        "Continuous predictors are winsorized at p1/p99 and standardized.  Binary predictors compare 0 vs 1.  Larger positive `interaction_gamma` means the profit-share coefficient is higher for firms with more of that pre-TCJA predictor.",
        "",
        "## Strongest one-way interactions by normal-approximation p-value",
        "",
        top[cols].round(5).to_markdown(index=False),
        "",
        "## Horse-race interactions",
        "",
        horse.sort_values(["spec", "interaction_p_norm"])[
            ["spec", "label", "family", "n", "firms", "interaction_gamma", "interaction_se", "interaction_p_norm"]
        ]
        .round(5)
        .to_markdown(index=False),
        "",
        "## Predictor coverage",
        "",
        coverage[["label", "family", "firms_with_value", "coverage_frac", "mean_or_share", "sd"]]
        .round(5)
        .to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def write_latex_table(oneway: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selected = [
        "Pre-TCJA intl tax benefit",
        "R&D / assets",
        "SG&A / (SG&A + COGS)",
        "Intangibles ex-goodwill / assets",
        "Foreign PI share",
        "Any haven/conduit subsidiary",
        "Log haven/conduit subsidiary count",
        "Foreign revenue share",
        "Foreign subsidiary share",
        "Log foreign country count",
        "Log assets",
        "Debt / assets",
        "Cash / assets",
    ]
    tbl = oneway[oneway["label"].isin(selected)].copy()
    tbl["gamma_fmt"] = tbl.apply(lambda r: f"{r['interaction_gamma']:+.5f}", axis=1)
    tbl["se_fmt"] = tbl.apply(lambda r: f"({r['interaction_se']:.5f})", axis=1)
    tbl["low_ratio_fmt"] = tbl["closure_ratio_low_z_or_zero_pp"].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
    tbl["high_ratio_fmt"] = tbl["closure_ratio_high_z_or_one_pp"].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Pre-TCJA predictor & $\\hat\\gamma$ & SE & low/0 ratio & high/1 ratio \\\\",
        "\\midrule",
    ]
    for r in tbl.itertuples(index=False):
        rows.append(
            f"{r.label} & {r.gamma_fmt} & {r.se_fmt} & {r.low_ratio_fmt} & {r.high_ratio_fmt} \\\\"
        )
    rows.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    path = TABLE_DIR / "tbl_profit_shifting_mechanisms.tex"
    path.write_text("\n".join(rows) + "\n")
    return path


def main() -> None:
    os.chdir(SCRIPT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_panel = load_base_panel()
    predictors, stats = build_predictors(base_panel)
    panel = add_predictors_to_panel(base_panel, predictors)

    oneway_share = run_oneway(panel, "foreign_share")
    oneway_log = run_oneway(panel, "foreign_share_log")
    horse_share = run_horse_races(panel, "foreign_share")
    horse_log = run_horse_races(panel, "foreign_share_log")
    coverage = predictor_coverage(predictors, base_panel)

    predictors.to_csv(OUT_DIR / "profit_shifting_mechanism_predictors.csv", index=False)
    panel.to_csv(OUT_DIR / "profit_shifting_mechanism_panel.csv", index=False)
    oneway_share.to_csv(OUT_DIR / "profit_shifting_mechanism_oneway_foreign_share.csv", index=False)
    oneway_log.to_csv(OUT_DIR / "profit_shifting_mechanism_oneway_log_share.csv", index=False)
    horse_share.to_csv(OUT_DIR / "profit_shifting_mechanism_horserace_foreign_share.csv", index=False)
    horse_log.to_csv(OUT_DIR / "profit_shifting_mechanism_horserace_log_share.csv", index=False)
    coverage.to_csv(OUT_DIR / "profit_shifting_mechanism_predictor_coverage.csv", index=False)
    (OUT_DIR / "profit_shifting_mechanism_standardization.json").write_text(json.dumps(stats, indent=2))
    md_path = write_mechanism_markdown(oneway_share, horse_share, coverage)
    tex_path = write_latex_table(oneway_share)

    ranked_cols = [
        "label",
        "family",
        "n",
        "firms",
        "interaction_gamma",
        "interaction_se",
        "interaction_p_norm",
        "closure_ratio_low_z_or_zero_pp",
        "closure_ratio_high_z_or_one_pp",
    ]
    print("\nTop one-way mechanism interactions: foreign PI share")
    print(oneway_share.sort_values("interaction_p_norm")[ranked_cols].head(15).round(5).to_string(index=False))

    print("\nHorse-race interactions: foreign PI share")
    print(
        horse_share.sort_values(["spec", "interaction_p_norm"])[
            ["spec", "label", "family", "n", "firms", "interaction_gamma", "interaction_se", "interaction_p_norm"]
        ]
        .round(5)
        .to_string(index=False)
    )
    print(f"\nWrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
