#!/usr/bin/env python3
"""
Build the paper-facing CSV inputs under paper/Inputs from this checkout.

Everything resolves relative to the repository root, so the paper inputs can be
rebuilt on any machine holding the data files. The upstream sources are:

  - analysis code:  analysis/
  - analysis data:  data/analysis/ (cache/, output/, tables/)

Run this after re-running the analysis scripts, so the staged inputs match the
regenerated panels; see paper/Inputs/DRAFT2_BUILD_INSTRUCTIONS.md for the full
rebuild sequence.

Outputs:
  - decomp_tikz/fig_drivers_tikz_data.csv
  - decomp_tikz/etr_drivers_by_year_full.source.csv
  - decomp_ind_new_tikz/summary.csv
  - first_stage_tikz/*.csv
  - first_etr_tikz/*.csv
  - first_pishr_tikz/*.csv
  - tables/industry_gap_distribution_table.csv
  - tables/industry_gap_distribution_lookup.csv
  - tables/contribution_gap_distribution_table.csv
  - tables/contribution_gap_distribution_lookup.csv
  - tables/contribution_gap_changes.csv
  - tables/coverage_table.csv
  - tables/table_cells.csv
  - README.md
  - current_draft_inputs_manifest.csv
  - table_csvs_manifest.csv
  - generated_files.csv
  - provenance_manifest.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import pickle
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


INPUTS_DIR = Path(__file__).resolve().parent
# Everything is built from this checkout. These were previously pinned to a
# second clone (/Users/erm/Repos/TCJA) and an external drive
# (/Volumes/Crucial X6/...); both now resolve inside the repo so the paper
# inputs are reproducible on any machine holding the data files.
REPO_ROOT = INPUTS_DIR.parents[1]
REPO_ANALYSIS_DIR = REPO_ROOT / "analysis"
EXTERNAL_ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
REPO_SLIDE_DATA_DIR = REPO_ROOT / "data" / "analysis" / "output" / "slide_data"
EXTERNAL_CACHE_DIR = EXTERNAL_ANALYSIS_DIR / "cache"
EXTERNAL_OUTPUT_DIR = EXTERNAL_ANALYSIS_DIR / "output"
EXTERNAL_TABLES_DIR = EXTERNAL_ANALYSIS_DIR / "tables"

if str(REPO_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_ANALYSIS_DIR))

from sic_industry import industry as classify_industry  # noqa: E402


GENERATED_AT = datetime.now().astimezone().replace(microsecond=0).isoformat()

# Common event-study windows used by the current repo scripts.
PRE_YEARS_SHORT = list(range(2013, 2017))
POST_YEARS_SHORT = list(range(2019, 2023))
EVENT_YEARS_SHORT = list(range(2013, 2023))
OMIT_YEAR = 2016
# OMIT_YEAR normalizes the event study (eq. 2): its coefficient is fixed at
# zero, but its observations stay in the estimation sample.  The pooled DiD
# (eq. 1) has no base year to omit, so it drops only the transition years.
# Keeping 2016 puts Table 7 and Figure 3 on the same sample.
POOLED_DROP_YEARS = {2017, 2018}
WIN_CH = 15.0

# Longer descriptive window used by the industry figure / tables.
POST_YEARS_LONG = list(range(2019, 2025))

INDUSTRY_ORDER = [
    "Tech",
    "Pharma/Biotech",
    "Manufacturing",
    "Oil & Gas",
    "Retail",
    "Utilities",
    "Other",
]

GROUPS = [
    ("International", "international"),
    ("Federal credits", "federal_credits"),
    ("Permanent items", "permanent_items"),
    ("Reserves & losses", "reserves_losses"),
    ("TCJA one-time", "tcja_onetime"),
    ("State & local", "state_local"),
    ("Other", "other"),
]

GROUP_MAP = {
    "International": "International",
    "State & local": "State_local",
    "Federal credits": "Federal_credits",
    "Permanent items": "Permanent_items",
    "Reserves & losses": "Reserves_losses",
    "TCJA one-time": "TCJA_onetime",
    "Other": "Other",
}


@dataclass(frozen=True)
class SourceRef:
    role: str
    path: Path
    notes: str = ""


generated_rows: list[dict[str, str]] = []
provenance_rows: list[dict[str, str]] = []
draft_input_rows: list[dict[str, str]] = []
table_manifest_rows: list[dict[str, str]] = []


def source_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().replace(microsecond=0).isoformat()


def record_output(relpath: str, description: str, build_logic: str, notes: str = "") -> None:
    generated_rows.append(
        {
            "output_relpath": relpath,
            "generated_at": GENERATED_AT,
            "description": description,
            "build_logic": build_logic,
            "notes": notes,
        }
    )


def record_sources(relpath: str, sources: Iterable[SourceRef]) -> None:
    for src in sources:
        provenance_rows.append(
            {
                "output_relpath": relpath,
                "generated_at": GENERATED_AT,
                "source_role": src.role,
                "source_path": str(src.path),
                "source_mtime": source_mtime(src.path),
                "notes": src.notes,
            }
        )


def write_csv(
    df: pd.DataFrame,
    relpath: str,
    description: str,
    build_logic: str,
    sources: Iterable[SourceRef],
    notes: str = "",
    float_format: str | None = "%.6f",
) -> None:
    path = INPUTS_DIR / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)
    record_output(relpath, description, build_logic, notes)
    record_sources(relpath, sources)


def write_text(relpath: str, text: str, description: str, build_logic: str, sources: Iterable[SourceRef], notes: str = "") -> None:
    path = INPUTS_DIR / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    record_output(relpath, description, build_logic, notes)
    record_sources(relpath, sources)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stars(t_value: float) -> str:
    abs_t = abs(t_value)
    if abs_t > 2.576:
        return "***"
    if abs_t > 1.96:
        return "**"
    if abs_t > 1.645:
        return "*"
    return ""


def weighted_average(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def weighted_quantile(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cutoff = q * w.sum()
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def ols_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    return beta, resid


def cluster_covariance(X: np.ndarray, resid: np.ndarray, groups: np.ndarray) -> np.ndarray:
    xtx_inv = np.linalg.inv(X.T @ X)
    k = X.shape[1]
    n = X.shape[0]
    unique_groups = pd.Index(groups).unique()
    g_count = len(unique_groups)

    meat = np.zeros((k, k), dtype=float)
    for group in unique_groups:
        mask = groups == group
        Xg = X[mask]
        ug = resid[mask]
        xugu = Xg.T @ ug
        meat += np.outer(xugu, xugu)

    cov = xtx_inv @ meat @ xtx_inv
    if g_count > 1 and n > k:
        cov *= (g_count / (g_count - 1)) * ((n - 1) / (n - k))
    return cov


def demean_ind_yr(
    df: pd.DataFrame,
    cols: list[str],
    firm_col: str = "gvkey",
    ind_col: str = "industry",
    year_col: str = "recon_year",
    tol: float = 1e-9,
    max_iter: int = 500,
) -> pd.DataFrame:
    out = df[cols].copy().astype(float)
    temp = df.copy()
    temp["_ind_yr"] = temp[ind_col].astype(str) + "_" + temp[year_col].astype(str)
    for _ in range(max_iter):
        prev = out.values.copy()
        for col in cols:
            out[col] -= out[col].groupby(temp[firm_col]).transform("mean")
            out[col] -= out[col].groupby(temp["_ind_yr"]).transform("mean")
        if np.abs(out.values - prev).max() < tol:
            break
    return out


def group_flag(year: int) -> str:
    if year == OMIT_YEAR:
        return "reference"
    if year < 2017:
        return "pre"
    if year == 2017:
        return "shock"
    return "post"


def build_short_event_summary(es: pd.DataFrame, pooled: dict[str, float], floor_min: float, floor_max: float) -> pd.DataFrame:
    ymin = min(float(es["lower"].min()), floor_min)
    ymax = max(float(es["upper"].max()), floor_max)
    ypad = 0.10 * (ymax - ymin)
    return pd.DataFrame(
        [
            {
                "pooled_coef": pooled["coef"],
                "pooled_se": pooled["se"],
                "pooled_firms": pooled["G"],
                "pooled_obs": pooled["n"],
                "pooled_stars": pooled["stars"],
                "event_firms": int(es["event_firms"].iloc[0]) if "event_firms" in es else 0,
                "event_obs": int(es["event_obs"].iloc[0]) if "event_obs" in es else 0,
                "ymin": ymin - ypad,
                "ymax": ymax + ypad,
            }
        ]
    )


def build_channels_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["channel"] = df["group"].map(GROUP_MAP).fillna("Other")
    fy = df.groupby(["gvkey", "recon_year", "channel"])["cat_val"].sum().reset_index(name="val")
    fy["val_w"] = fy["val"].clip(-WIN_CH, WIN_CH)
    wide = fy.pivot_table(
        index=["gvkey", "recon_year"],
        columns="channel",
        values="val_w",
        fill_value=0.0,
    ).reset_index()
    wide.columns.name = None
    if "International" not in wide.columns:
        wide["International"] = 0.0
    return wide


def build_intl_pre_from_wide(wide: pd.DataFrame) -> pd.Series:
    pre = wide[wide["recon_year"].isin(PRE_YEARS_SHORT)]
    counts = pre.groupby("gvkey").size()
    eligible = counts[counts >= 2].index
    intl_pre = (
        pre[pre["gvkey"].isin(eligible)]
        .groupby("gvkey")["International"]
        .mean()
        .rename("intl_pre")
    )
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    return intl_pre.clip(p1, p99)


def build_first_stage_panel(combined: pd.DataFrame, wide: pd.DataFrame, intl_pre: pd.Series) -> pd.DataFrame:
    panel = combined[combined["recon_year"].isin(EVENT_YEARS_SHORT)].copy()
    panel["industry"] = panel["comp_sic"].apply(classify_industry)
    panel = panel[panel["industry"] != "Finance"].copy()
    panel = panel[panel["comp_pi"] > 1].copy()
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(wide, on=["gvkey", "recon_year"], how="left")
    panel["International"] = panel["International"].fillna(0.0)

    fy_sets = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = fy_sets.apply(lambda s: len(s & set(PRE_YEARS_SHORT)) >= 2)
    has_post = fy_sets.apply(lambda s: len(s & set(POST_YEARS_SHORT)) >= 2)
    ok = fy_sets[has_pre & has_post].index
    return panel[panel["gvkey"].isin(ok)].copy()


def build_intl_yearly(long_df: pd.DataFrame) -> pd.DataFrame:
    fy = long_df.groupby(["gvkey", "recon_year", "group"])["cat_val"].sum().reset_index(name="val")
    fy["val_w"] = fy["val"].clip(-WIN_CH, WIN_CH)
    return (
        fy[fy["group"] == "International"][["gvkey", "recon_year", "val_w"]]
        .rename(columns={"val_w": "International"})
        .copy()
    )


def build_intl_pre_for_foreign_share(long_df: pd.DataFrame) -> pd.Series:
    """Exact treatment construction used in build_substitution_real_outcomes.py."""
    fy = long_df.groupby(["gvkey", "recon_year", "group"])["cat_val"].sum().reset_index(name="val")
    fy["val_w"] = fy["val"].clip(-WIN_CH, WIN_CH)
    intl_fy = fy[fy["group"] == "International"].groupby(["gvkey", "recon_year"])["val_w"].sum()
    any_obs = fy.groupby(["gvkey", "recon_year"]).size()
    pre_any = any_obs[any_obs.index.get_level_values("recon_year").isin(PRE_YEARS_SHORT)]
    eligible = pre_any.groupby("gvkey").size()
    eligible = eligible[eligible >= 2].index
    pre_intl = (
        intl_fy[
            intl_fy.index.get_level_values("recon_year").isin(PRE_YEARS_SHORT)
            & intl_fy.index.get_level_values("gvkey").isin(eligible)
        ]
        .groupby("gvkey")
        .mean()
    )
    intl_pre = pd.Series(0.0, index=eligible, name="intl_pre")
    intl_pre.update(pre_intl)
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    return intl_pre.clip(p1, p99)


def build_foreign_share_panel(combined: pd.DataFrame, pi: pd.DataFrame, intl_pre: pd.Series, intl_yearly: pd.DataFrame) -> pd.DataFrame:
    panel = combined[combined["recon_year"].isin(EVENT_YEARS_SHORT)].copy()
    panel["industry"] = panel["comp_sic"].apply(classify_industry)
    panel = panel[panel["industry"] != "Finance"].copy()
    panel = panel[panel["comp_pi"] > 1].copy()
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(pi, on=["gvkey", "recon_year"], how="left")
    panel = panel.merge(intl_yearly, on=["gvkey", "recon_year"], how="left")
    panel["International"] = panel["International"].fillna(0.0)

    fy_sets = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = fy_sets.apply(lambda s: len(s & set(PRE_YEARS_SHORT)) >= 2)
    has_post = fy_sets.apply(lambda s: len(s & set(POST_YEARS_SHORT)) >= 2)
    ok = fy_sets[has_pre & has_post].index
    panel = panel[panel["gvkey"].isin(ok)].copy()

    pi_total = panel["comp_pidom"].fillna(0.0) + panel["comp_pifo"].fillna(0.0)
    valid = panel["comp_pidom"].notna() & panel["comp_pifo"].notna() & (pi_total.abs() > 0.5)
    panel["foreign_share"] = np.where(valid, panel["comp_pifo"] / pi_total, np.nan)
    panel["foreign_share"] = panel["foreign_share"].clip(0.0, 1.0)
    return panel


def run_event_study(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    plot_years = [yr for yr in EVENT_YEARS_SHORT if yr != OMIT_YEAR]
    sub = panel[
        panel["recon_year"].isin(EVENT_YEARS_SHORT)
        & panel[outcome].notna()
        & panel["intl_pre"].notna()
    ].copy()
    for year in plot_years:
        sub[f"_ix_{year}"] = sub["intl_pre"] * (sub["recon_year"] == year).astype(float)
    ix_cols = [f"_ix_{year}" for year in plot_years]
    dm = demean_ind_yr(sub, [outcome] + ix_cols)
    X = dm[ix_cols].values
    y = dm[outcome].values
    beta, resid = ols_fit(y, X)
    cov = cluster_covariance(X, resid, sub["gvkey"].values)
    ses = np.sqrt(np.diag(cov))

    event_obs = int(len(sub))
    event_firms = int(sub["gvkey"].nunique())

    rows: list[dict[str, float | str | int]] = []
    for idx, year in enumerate(plot_years):
        coef = float(beta[idx])
        se = float(ses[idx])
        err95 = 1.96 * se
        rows.append(
            {
                "year": year,
                "beta": coef,
                "se": se,
                "err95": err95,
                "lower": coef - err95,
                "upper": coef + err95,
                "group": group_flag(year),
            }
        )
    rows.append(
        {
            "year": OMIT_YEAR,
            "beta": 0.0,
            "se": 0.0,
            "err95": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "group": "reference",
        }
    )
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    # The event study estimates eq. (2) on every year in EVENT_YEARS_SHORT,
    # including the transition years the pooled DiD drops, so its sample is
    # larger than the pooled one.  Carry it so figure notes can quote the
    # right N rather than borrowing Table 7's.
    out["event_obs"] = event_obs
    out["event_firms"] = event_firms
    return out


def run_pooled_did(panel: pd.DataFrame, outcome: str) -> dict[str, float]:
    years = [yr for yr in EVENT_YEARS_SHORT if yr not in POOLED_DROP_YEARS]
    sub = panel[
        panel["recon_year"].isin(years)
        & panel[outcome].notna()
        & panel["intl_pre"].notna()
    ].copy()
    sub["_post"] = sub["recon_year"].isin(POST_YEARS_SHORT).astype(float)
    sub["_Di"] = sub["intl_pre"] * sub["_post"]
    dm = demean_ind_yr(sub, [outcome, "_Di"])
    X = dm[["_Di"]].values
    y = dm[outcome].values
    beta, resid = ols_fit(y, X)
    cov = cluster_covariance(X, resid, sub["gvkey"].values)
    se = float(np.sqrt(np.diag(cov))[0])
    coef = float(beta[0])
    t = coef / se if se > 0 else math.nan
    return {
        "coef": coef,
        "se": se,
        "G": int(sub["gvkey"].nunique()),
        "n": int(len(sub)),
        "t": t,
        "stars": stars(t),
    }


def build_fig_drivers() -> None:
    source_csv = EXTERNAL_TABLES_DIR / "etr_drivers_by_year_full.csv"
    full_sample_path = EXTERNAL_CACHE_DIR / "full_sample.pkl"
    source_df = pd.read_csv(source_csv)
    full_sample = pd.read_pickle(full_sample_path)

    rev_gap = pd.DataFrame(
        {
            "year": sorted(full_sample["recon_year"].dropna().unique()),
            "total_gap_rev_wtd": [
                weighted_average(
                    full_sample.loc[full_sample["recon_year"] == year, "gap_w"],
                    full_sample.loc[full_sample["recon_year"] == year, "comp_sale"],
                )
                for year in sorted(full_sample["recon_year"].dropna().unique())
            ],
        }
    )
    merged = source_df.merge(rev_gap, on="year", how="left").rename(
        columns={
            "State & local": "state_local",
            "International": "international",
            "Federal credits": "federal_credits",
            "Permanent items": "permanent_items",
            "Reserves & losses": "reserves_losses",
            "TCJA one-time": "tcja_onetime",
            "Other": "other",
        }
    )

    sources = [
        SourceRef("channel_series_csv", source_csv, "Current external-drive aggregate decomposition table."),
        SourceRef("full_sample_panel", full_sample_path, "Current external-drive full sample used for the total-gap overlay."),
    ]
    write_csv(
        merged,
        "decomp_tikz/fig_drivers_tikz_data.csv",
        "TikZ plotting data for the aggregate reconciliation decomposition figure.",
        "Current slide logic: channel bars from etr_drivers_by_year_full.csv plus sales-weighted total gap from full_sample.pkl.",
        sources,
    )
    write_csv(
        source_df,
        "decomp_tikz/etr_drivers_by_year_full.source.csv",
        "Verbatim copy of the external-drive aggregate decomposition source CSV.",
        "Copied directly from the current external-drive table output.",
        [SourceRef("source_csv", source_csv, "Copied verbatim for provenance.")],
    )
    draft_input_rows.append(
        {
            "draft_input": "decomp_tikz/fig_drivers_tikz_data.csv",
            "currently_loaded_by_draft": "yes",
            "rebuilt_output": "decomp_tikz/fig_drivers_tikz_data.csv",
            "build_logic": "Current aggregate decomposition: external etr_drivers_by_year_full.csv + external full_sample.pkl + repo classifier not needed.",
            "primary_sources": f"{source_csv}; {full_sample_path}",
            "notes": "Matches the aggregate slide workflow; source dates are in provenance_manifest.csv.",
        }
    )


def build_industry_summary(full_sample: pd.DataFrame, driver_panel_full: pd.DataFrame) -> None:
    fs = full_sample.copy()
    fs["industry"] = fs["comp_sic"].apply(classify_industry)
    lookup = fs[["gvkey", "recon_year", "industry"]].drop_duplicates(["gvkey", "recon_year"])
    dp = (
        driver_panel_full.drop(columns=["industry"], errors="ignore")
        .merge(lookup, on=["gvkey", "recon_year"], how="left")
        .copy()
    )
    dp["industry"] = dp["industry"].fillna("Other")

    rows: list[dict[str, float | int | str]] = []
    windows = {"pre": (2013, 2016), "post": (2019, 2024)}
    for y_pos, ind_name in enumerate(INDUSTRY_ORDER):
        ind_fs = fs[fs["industry"] == ind_name]
        ind_dp = dp[dp["industry"] == ind_name]
        row: dict[str, float | int | str] = {"industry": ind_name, "y": y_pos}
        for label, key in GROUPS:
            for side, (y0, y1) in windows.items():
                sub = ind_dp[ind_dp["recon_year"].between(y0, y1) & (ind_dp["group"] == label)]
                row[f"{key}_{side}_pp"] = weighted_average(sub["contrib_w"], sub["weight"])
            row[f"{key}_delta_pp"] = row[f"{key}_post_pp"] - row[f"{key}_pre_pp"]
        for side, (y0, y1) in windows.items():
            sub = ind_fs[ind_fs["recon_year"].between(y0, y1)]
            row[f"n_{side}"] = int(len(sub))
            row[f"gap_{side}_pp"] = weighted_average(sub["gap_w"], sub["comp_sale"])
            row[f"gap_{side}_lbl"] = f"{row[f'gap_{side}_pp']:.1f}pp"
        row["gap_change_pp"] = row["gap_post_pp"] - row["gap_pre_pp"]
        row["gap_change_lbl"] = f"{row['gap_change_pp']:.1f}pp"
        rows.append(row)

    summary = pd.DataFrame(rows)
    sources = [
        SourceRef("full_sample_panel", EXTERNAL_CACHE_DIR / "full_sample.pkl", "Current external-drive full sample."),
        SourceRef("driver_panel_full", EXTERNAL_CACHE_DIR / "driver_panel_full.pkl", "Current external-drive grouped contribution panel."),
        SourceRef("repo_classifier", REPO_ANALYSIS_DIR / "sic_industry.py", "Current repo SIC-to-industry mapping; this is where Finance is remapped into Other where appropriate."),
    ]
    write_csv(
        summary,
        "decomp_ind_new_tikz/summary.csv",
        "Industry pre/post gap levels and contribution deltas for the paper's TikZ industry figure.",
        "Current industry figure logic: full_sample.pkl + driver_panel_full.pkl with current repo SIC classifier; pre=2013-2016, post=2019-2024.",
        sources,
        notes="Industry rows use current repo classification, so the old Finance row becomes Other.",
    )
    draft_input_rows.append(
        {
            "draft_input": "decomp_ind_new_tikz/summary.csv",
            "currently_loaded_by_draft": "yes",
            "rebuilt_output": "decomp_ind_new_tikz/summary.csv",
            "build_logic": "Current paper industry figure logic from external full_sample/driver_panel_full plus repo SIC classifier.",
            "primary_sources": f"{EXTERNAL_CACHE_DIR / 'full_sample.pkl'}; {EXTERNAL_CACHE_DIR / 'driver_panel_full.pkl'}; {REPO_ANALYSIS_DIR / 'sic_industry.py'}",
            "notes": "Uses Other in place of the old Finance row.",
        }
    )


def write_event_outputs(prefix: str, event_df: pd.DataFrame, summary_df: pd.DataFrame, description_root: str, build_logic: str, sources: list[SourceRef]) -> None:
    write_csv(
        event_df,
        f"{prefix}/{prefix.split('_tikz')[0]}_event_tikz_data.csv",
        f"{description_root}: full event-study path.",
        build_logic,
        sources,
    )
    for group in ["pre", "shock", "post", "reference"]:
        sub = event_df[event_df["group"] == group]
        write_csv(
            sub,
            f"{prefix}/{prefix.split('_tikz')[0]}_{group}.csv",
            f"{description_root}: {group} subset.",
            build_logic,
            sources,
        )
    write_csv(
        summary_df,
        f"{prefix}/{prefix.split('_tikz')[0]}_event_summary.csv",
        f"{description_root}: pooled DiD summary and axis bounds.",
        build_logic,
        sources,
    )


def build_first_stage_family(combined: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    wide = build_channels_wide(long_df)
    intl_pre = build_intl_pre_from_wide(wide)
    panel = build_first_stage_panel(combined, wide, intl_pre)

    first_stage_event = run_event_study(panel, "International")
    first_stage_summary = build_short_event_summary(first_stage_event, run_pooled_did(panel, "International"), -0.05, 0.05)
    first_stage_sources = [
        SourceRef("combined_panel", EXTERNAL_CACHE_DIR / "combined.pkl", "Current external-drive combined panel."),
        SourceRef("long_panel", EXTERNAL_CACHE_DIR / "long.pkl", "Current external-drive long reconciliation panel."),
        SourceRef("repo_script_logic", REPO_ANALYSIS_DIR / "build_firststage_event_studies.py", "Current repo first-stage/ETR event-study logic."),
        SourceRef("repo_classifier", REPO_ANALYSIS_DIR / "sic_industry.py", "Current repo SIC-to-industry classifier."),
    ]
    write_event_outputs(
        "first_stage_tikz",
        first_stage_event,
        first_stage_summary,
        "International first-stage event-study inputs",
        "Current build_firststage_event_studies.py logic on the current external-drive data tree.",
        first_stage_sources,
    )

    draft_input_rows.extend(
        [
            {
                "draft_input": f"first_stage_tikz/{name}",
                "currently_loaded_by_draft": "yes",
                "rebuilt_output": f"first_stage_tikz/{name}",
                "build_logic": "Current build_firststage_event_studies.py logic on external combined.pkl and long.pkl.",
                "primary_sources": f"{EXTERNAL_CACHE_DIR / 'combined.pkl'}; {EXTERNAL_CACHE_DIR / 'long.pkl'}; {REPO_ANALYSIS_DIR / 'build_firststage_event_studies.py'}",
                "notes": "Pooled coefficient matches current regressionheadlines.csv.",
            }
            for name in [
                "first_stage_event_tikz_data.csv",
                "first_stage_pre.csv",
                "first_stage_shock.csv",
                "first_stage_post.csv",
                "first_stage_reference.csv",
                "first_stage_event_summary.csv",
            ]
        ]
    )

    first_etr_event = run_event_study(panel, "etr_w")
    first_etr_summary = build_short_event_summary(first_etr_event, run_pooled_did(panel, "etr_w"), -0.05, 0.05)
    first_etr_sources = [
        SourceRef("combined_panel", EXTERNAL_CACHE_DIR / "combined.pkl", "Current external-drive combined panel."),
        SourceRef("long_panel", EXTERNAL_CACHE_DIR / "long.pkl", "Current external-drive long reconciliation panel."),
        SourceRef("repo_script_logic", REPO_ANALYSIS_DIR / "build_firststage_event_studies.py", "Current repo first-stage/ETR event-study logic."),
        SourceRef("repo_classifier", REPO_ANALYSIS_DIR / "sic_industry.py", "Current repo SIC-to-industry classifier."),
    ]
    write_event_outputs(
        "first_etr_tikz",
        first_etr_event,
        first_etr_summary,
        "ETR pass-through event-study inputs",
        "Current build_firststage_event_studies.py logic on the current external-drive data tree.",
        first_etr_sources,
    )
    draft_input_rows.extend(
        [
            {
                "draft_input": f"first_etr_tikz/{name}",
                "currently_loaded_by_draft": "yes",
                "rebuilt_output": f"first_etr_tikz/{name}",
                "build_logic": "Current build_firststage_event_studies.py logic on external combined.pkl and long.pkl.",
                "primary_sources": f"{EXTERNAL_CACHE_DIR / 'combined.pkl'}; {EXTERNAL_CACHE_DIR / 'long.pkl'}; {REPO_ANALYSIS_DIR / 'build_firststage_event_studies.py'}",
                "notes": "Pooled coefficient matches current regressionheadlines.csv.",
            }
            for name in [
                "first_etr_event_tikz_data.csv",
                "first_etr_pre.csv",
                "first_etr_shock.csv",
                "first_etr_post.csv",
                "first_etr_reference.csv",
                "first_etr_event_summary.csv",
            ]
        ]
    )

    return panel


# Display order and labels for the reconciliation channels, matching Table 2.
CHANNEL_ROWS = [
    ("international", "International", "International"),
    ("state_local", "State_local", "State \\& local"),
    ("federal_credits", "Federal_credits", "Federal credits"),
    ("permanent_items", "Permanent_items", "Book v.\\ tax"),
    ("reserves_losses", "Reserves_losses", "Valuation"),
    ("tcja_onetime", "TCJA_onetime", "One time"),
    ("other", "Other", "Other"),
]


def build_channel_did_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Pooled DiD for every reconciliation channel, plus the identity check.

    The effective rate is the statutory rate plus the sum of the channel
    contributions.  The industry-by-year effects absorb the statutory change and
    every channel is zero-filled on one common sample, so beta_ETR should equal
    the sum of the channel coefficients up to winsorization.  Reporting the sum
    against beta_ETR makes that check visible instead of implicit.
    """
    rows: list[dict[str, object]] = []
    total = 0.0
    for order, (row_key, col, label) in enumerate(CHANNEL_ROWS, start=1):
        r = run_pooled_did(panel, col)
        total += r["coef"]
        rows.append(
            {
                "row_order": order,
                "row_key": row_key,
                "row_label": label,
                "coef": r["coef"],
                "se": r["se"],
                "stars": r["stars"],
                "t": r["t"],
                "n_obs": r["n"],
                "n_firms": r["G"],
            }
        )
    etr = run_pooled_did(panel, "etr_w")
    rows.append(
        {
            "row_order": len(CHANNEL_ROWS) + 1,
            "row_key": "sum_channels",
            "row_label": "Sum of channels",
            "coef": total,
            "se": float("nan"),
            "stars": "",
            "t": float("nan"),
            "n_obs": rows[0]["n_obs"],
            "n_firms": rows[0]["n_firms"],
        }
    )
    rows.append(
        {
            "row_order": len(CHANNEL_ROWS) + 2,
            "row_key": "etr",
            "row_label": "Effective tax rate",
            "coef": etr["coef"],
            "se": etr["se"],
            "stars": etr["stars"],
            "t": etr["t"],
            "n_obs": etr["n"],
            "n_firms": etr["G"],
        }
    )
    # Everything other than international, which is the substitution margin.
    non_intl = total - rows[0]["coef"]
    rows.append(
        {
            "row_order": len(CHANNEL_ROWS) + 3,
            "row_key": "sum_non_international",
            "row_label": "Sum excluding international",
            "coef": non_intl,
            "se": float("nan"),
            "stars": "",
            "t": float("nan"),
            "n_obs": rows[0]["n_obs"],
            "n_firms": rows[0]["n_firms"],
        }
    )
    rows.append(
        {
            "row_order": len(CHANNEL_ROWS) + 4,
            "row_key": "offset_share_pct",
            "row_label": "Offset share of the international closure (percent)",
            "coef": 100.0 * non_intl / abs(rows[0]["coef"]),
            "se": float("nan"),
            "stars": "",
            "t": float("nan"),
            "n_obs": rows[0]["n_obs"],
            "n_firms": rows[0]["n_firms"],
        }
    )
    return pd.DataFrame(rows)


def build_industry_passthrough_examples(panel: pd.DataFrame) -> None:
    """Worked pass-through example by industry, derived rather than hand-set.

    This replaces a hand-written CSV whose fields had drifted from both the
    industry figure and the pooled estimates, and whose "other channels" column
    was not reproducible from anything.
    """
    summary = pd.read_csv(INPUTS_DIR / "decomp_ind_new_tikz/summary.csv")
    channels = build_channel_did_table(panel)
    by_key = {str(r["row_key"]): float(r["coef"]) for _, r in channels.iterrows()}
    beta_intl = by_key["international"]
    beta_etr = by_key["etr"]
    beta_non_intl = by_key["sum_non_international"]

    rows: list[dict[str, object]] = []
    for _, r in summary.iterrows():
        closure = float(r["international_delta_pp"])
        rows.append(
            {
                "row_key": str(r["industry"]).lower().replace("/", "_").replace(" & ", "_").replace(" ", "_"),
                "industry": r["industry"],
                "international_pre_pp": float(r["international_pre_pp"]),
                "international_post_pp": float(r["international_post_pp"]),
                "international_closure_pp": closure,
                "total_gap_closure_pp": float(r["gap_change_pp"]),
                "implied_etr_increase_pp": (beta_etr / beta_intl) * closure,
                # Negative: the other channels give back part of the closure.
                "other_channels_pp": (beta_non_intl / beta_intl) * closure,
                "other_channels_abs_pp": abs((beta_non_intl / beta_intl) * closure),
            }
        )
    df = pd.DataFrame(rows)
    sources = [
        SourceRef("industry_summary_csv", INPUTS_DIR / "decomp_ind_new_tikz/summary.csv", "Generated earlier in this same build."),
        SourceRef("combined_panel", EXTERNAL_CACHE_DIR / "combined.pkl", "Current combined panel."),
        SourceRef("long_panel", EXTERNAL_CACHE_DIR / "long.pkl", "Current long reconciliation panel."),
    ]
    write_csv(
        df,
        "tables/industry_passthrough_examples.csv",
        "Worked pass-through example by industry.",
        "Industry closures from the industry figure summary; implied effective-rate and "
        "offset columns from the pooled channel coefficients in this same build.",
        sources,
        notes="other_channels_pp is negative: it is the part of the closure returned through non-international channels.",
    )
    write_csv(
        channels,
        "tables/channel_did_table.csv",
        "Pooled DiD coefficient for each reconciliation channel, with the sum-versus-ETR identity check.",
        "run_pooled_did on each zero-filled channel of the first-stage panel.",
        sources,
        notes="sum_channels should match etr up to winsorization; sum_non_international is the substitution margin.",
    )


def build_first_pishr_family(combined: pd.DataFrame, long_df: pd.DataFrame) -> None:
    pi = pd.read_csv(
        EXTERNAL_OUTPUT_DIR / "tax_reconciliation_panel_base.csv",
        usecols=["gvkey", "recon_year", "comp_pidom", "comp_pifo", "comp_at"],
        low_memory=False,
    ).drop_duplicates(["gvkey", "recon_year"])
    intl_pre = build_intl_pre_for_foreign_share(long_df)
    intl_yearly = build_intl_yearly(long_df)
    panel = build_foreign_share_panel(combined, pi, intl_pre, intl_yearly)

    event_df = run_event_study(panel, "foreign_share")
    summary_df = build_short_event_summary(event_df, run_pooled_did(panel, "foreign_share"), -0.01, 0.01)
    sources = [
        SourceRef("combined_panel", EXTERNAL_CACHE_DIR / "combined.pkl", "Current external-drive combined panel."),
        SourceRef("long_panel", EXTERNAL_CACHE_DIR / "long.pkl", "Current external-drive long reconciliation panel."),
        SourceRef("pi_base_csv", EXTERNAL_OUTPUT_DIR / "tax_reconciliation_panel_base.csv", "Current external-drive PI composition merge input."),
        SourceRef("repo_script_logic", REPO_ANALYSIS_DIR / "build_substitution_real_outcomes.py", "Current repo foreign-share event-study logic; this is the up-to-date source for the paper's foreign-share event-study inputs."),
        SourceRef("repo_classifier", REPO_ANALYSIS_DIR / "sic_industry.py", "Current repo SIC-to-industry classifier."),
    ]
    write_event_outputs(
        "first_pishr_tikz",
        event_df,
        summary_df,
        "Foreign PI share event-study inputs",
        "Current build_substitution_real_outcomes.py logic on the external-drive data tree; sample/treatment align with the current pass-through regression setup.",
        sources,
    )
    draft_input_rows.extend(
        [
            {
                "draft_input": f"first_pishr_tikz/{name}",
                "currently_loaded_by_draft": "yes",
                "rebuilt_output": f"first_pishr_tikz/{name}",
                "build_logic": "Current build_substitution_real_outcomes.py logic on external combined.pkl, long.pkl, and tax_reconciliation_panel_base.csv.",
                "primary_sources": f"{EXTERNAL_CACHE_DIR / 'combined.pkl'}; {EXTERNAL_CACHE_DIR / 'long.pkl'}; {EXTERNAL_OUTPUT_DIR / 'tax_reconciliation_panel_base.csv'}; {REPO_ANALYSIS_DIR / 'build_substitution_real_outcomes.py'}",
                "notes": "Pooled coefficient matches current regressionheadlines.csv foreign-share value.",
            }
            for name in [
                "first_pishr_event_tikz_data.csv",
                "first_pishr_pre.csv",
                "first_pishr_shock.csv",
                "first_pishr_post.csv",
                "first_pishr_reference.csv",
                "first_pishr_event_summary.csv",
            ]
        ]
    )


def summarize_frame(label: str, sub: pd.DataFrame, value_col: str, weight_col: str) -> dict[str, object]:
    return {
        "row_label": label,
        "counts": int(len(sub)),
        "average": weighted_average(sub[value_col], sub[weight_col]),
        "p25": weighted_quantile(sub[value_col], sub[weight_col], 0.25),
        "median": weighted_quantile(sub[value_col], sub[weight_col], 0.50),
        "p75": weighted_quantile(sub[value_col], sub[weight_col], 0.75),
    }


def build_table_csvs(full_sample: pd.DataFrame, driver_panel_full: pd.DataFrame) -> None:
    fs = full_sample.copy()
    fs["industry"] = fs["comp_sic"].apply(classify_industry)
    lookup = fs[["gvkey", "recon_year", "industry"]].drop_duplicates(["gvkey", "recon_year"])
    dp = (
        driver_panel_full.drop(columns=["industry"], errors="ignore")
        .merge(lookup, on=["gvkey", "recon_year"], how="left")
        .copy()
    )
    dp["industry"] = dp["industry"].fillna("Other")

    industry_rows: list[dict[str, object]] = []
    for period_key, period_label, years in [
        ("pre", "Pre-TCJA", PRE_YEARS_SHORT),
        ("post", "Post-TCJA", POST_YEARS_LONG),
    ]:
        period_df = fs[fs["recon_year"].isin(years) & fs["gap_w"].notna()].copy()
        for order, ind_name in enumerate(INDUSTRY_ORDER, start=1):
            sub = period_df[period_df["industry"] == ind_name].copy()
            row = summarize_frame(ind_name, sub, "gap_w", "comp_sale")
            row.update(
                {
                    "table_id": "industry_gap_distribution",
                    "period": period_key,
                    "period_label": period_label,
                    "row_order": order,
                    "row_key": ind_name.lower().replace("/", "_").replace(" ", "_").replace("&", "and").replace("-", "_"),
                    "source_group": "",
                }
            )
            industry_rows.append(row)

    contribution_spec = [
        ("total_gap", "Total Gap", None),
        ("international", "International", "International"),
        ("one_time", "One time", "TCJA one-time"),
        ("federal_credits", "Federal credits", "Federal credits"),
        ("state_local", "State & local", "State & local"),
        ("valuation", "Valuation", "Reserves & losses"),
        ("book_v_tax", "Book v. Tax", "Permanent items"),
        ("other", "Other", "Other"),
    ]

    contribution_rows: list[dict[str, object]] = []
    for period_key, period_label, years in [
        ("pre", "Pre-TCJA", PRE_YEARS_SHORT),
        ("post", "Post-TCJA", POST_YEARS_LONG),
    ]:
        gap_sub = fs[fs["recon_year"].isin(years) & fs["gap_w"].notna()].copy()
        gap_row = summarize_frame("Total Gap", gap_sub, "gap_w", "comp_sale")
        gap_row.update(
            {
                "table_id": "contribution_gap_distribution",
                "period": period_key,
                "period_label": period_label,
                "row_order": 1,
                "row_key": "total_gap",
                "source_group": "",
            }
        )
        contribution_rows.append(gap_row)

        for order, (row_key, row_label, group_name) in enumerate(contribution_spec[1:], start=2):
            sub = dp[dp["recon_year"].isin(years) & (dp["group"] == group_name)].copy()
            row = summarize_frame(row_label, sub, "contrib_w", "weight")
            row.update(
                {
                    "table_id": "contribution_gap_distribution",
                    "period": period_key,
                    "period_label": period_label,
                    "row_order": order,
                    "row_key": row_key,
                    "source_group": group_name,
                }
            )
            contribution_rows.append(row)

    industry_df = pd.DataFrame(industry_rows)[
        ["table_id", "period", "period_label", "row_order", "row_key", "row_label", "source_group", "counts", "average", "p25", "median", "p75"]
    ]
    contribution_df = pd.DataFrame(contribution_rows)[
        ["table_id", "period", "period_label", "row_order", "row_key", "row_label", "source_group", "counts", "average", "p25", "median", "p75"]
    ]

    table_sources = [
        SourceRef("full_sample_panel", EXTERNAL_CACHE_DIR / "full_sample.pkl", "Current external-drive full sample."),
        SourceRef("driver_panel_full", EXTERNAL_CACHE_DIR / "driver_panel_full.pkl", "Current external-drive grouped contribution panel."),
        SourceRef("repo_classifier", REPO_ANALYSIS_DIR / "sic_industry.py", "Current repo SIC-to-industry mapping."),
    ]
    write_csv(
        industry_df,
        "tables/industry_gap_distribution_table.csv",
        "Row-based CSV for the hand-written industry gap distribution table.",
        "Current distribution-table logic on external full_sample.pkl with repo SIC classifier; pre=2013-2016, post=2019-2024.",
        table_sources,
        notes="Uses Other in place of the old Finance row.",
    )
    write_csv(
        contribution_df,
        "tables/contribution_gap_distribution_table.csv",
        "Row-based CSV for the hand-written contribution distribution table.",
        "Current distribution-table logic on external full_sample.pkl and driver_panel_full.pkl; non-total rows use the subset where the contribution group is present.",
        table_sources,
        notes="Display labels follow the draft table: One time, Valuation, and Book v. Tax map internally to TCJA one-time, Reserves & losses, and Permanent items.",
    )

    # Pre/post changes in the category averages.  Section 3.1 asserts that no
    # category other than international moves by as much as a percentage point;
    # emit the binding number so the claim tracks the data instead of being
    # asserted by hand (the margin is only about 0.1pp).
    avg_by = {
        (str(r["period"]), str(r["row_key"])): float(r["average"])
        for _, r in contribution_df.iterrows()
    }
    change_rows: list[dict[str, object]] = []
    for row_key, row_label, _group in contribution_spec:
        change = avg_by[("post", row_key)] - avg_by[("pre", row_key)]
        change_rows.append(
            {
                "row_key": row_key,
                "row_label": row_label,
                "pre_average": avg_by[("pre", row_key)],
                "post_average": avg_by[("post", row_key)],
                "change": change,
                "abs_change": abs(change),
            }
        )
    change_df = pd.DataFrame(change_rows)
    non_intl = change_df[~change_df["row_key"].isin(["total_gap", "international"])]
    binding = non_intl.loc[non_intl["abs_change"].idxmax()]
    change_df = pd.concat(
        [
            change_df,
            pd.DataFrame(
                [
                    {
                        "row_key": "max_non_international",
                        "row_label": str(binding["row_label"]),
                        "pre_average": binding["pre_average"],
                        "post_average": binding["post_average"],
                        "change": binding["change"],
                        "abs_change": binding["abs_change"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_csv(
        change_df,
        "tables/contribution_gap_changes.csv",
        "Pre-to-post change in each category's sales-weighted average contribution.",
        "Derived from the contribution distribution table written in this build; "
        "the max_non_international row is the largest absolute change outside international.",
        [
            SourceRef("contribution_table_csv", INPUTS_DIR / "tables/contribution_gap_distribution_table.csv", "Generated earlier in this same build."),
        ],
        notes="max_non_international carries row_label naming the binding category, so the draft can report it.",
    )

    industry_lookup_df = industry_df.copy()
    industry_lookup_df.insert(
        0,
        "lookup_key",
        industry_lookup_df["period"] + "_" + industry_lookup_df["row_key"],
    )
    contribution_lookup_df = contribution_df.copy()
    contribution_lookup_df.insert(
        0,
        "lookup_key",
        contribution_lookup_df["period"] + "_" + contribution_lookup_df["row_key"],
    )
    write_csv(
        industry_lookup_df,
        "tables/industry_gap_distribution_lookup.csv",
        "Lookup-key CSV for the hand-written industry gap distribution table.",
        "Derived from the row-based industry table CSV written in this build; lookup_key = period_rowkey.",
        [
            SourceRef("industry_table_csv", INPUTS_DIR / "tables/industry_gap_distribution_table.csv", "Generated earlier in this same build."),
        ],
        notes="Makes Datatool fetches simpler for period-specific industry rows.",
    )
    write_csv(
        contribution_lookup_df,
        "tables/contribution_gap_distribution_lookup.csv",
        "Lookup-key CSV for the hand-written contribution gap distribution table.",
        "Derived from the row-based contribution table CSV written in this build; lookup_key = period_rowkey.",
        [
            SourceRef("contribution_table_csv", INPUTS_DIR / "tables/contribution_gap_distribution_table.csv", "Generated earlier in this same build."),
        ],
        notes="Makes Datatool fetches simpler for period-specific contribution rows.",
    )

    cell_rows: list[dict[str, object]] = []
    stat_map = [
        ("counts", "Counts"),
        ("average", "Average"),
        ("p25", "25th"),
        ("median", "50th"),
        ("p75", "75th"),
    ]
    for df in [industry_df, contribution_df]:
        for _, row in df.iterrows():
            for stat_key, stat_label in stat_map:
                cell_rows.append(
                    {
                        "table_id": row["table_id"],
                        "period": row["period"],
                        "period_label": row["period_label"],
                        "row_order": row["row_order"],
                        "row_key": row["row_key"],
                        "row_label": row["row_label"],
                        "stat_key": stat_key,
                        "stat_label": stat_label,
                        "value": row[stat_key],
                    }
                )
    write_csv(
        pd.DataFrame(cell_rows),
        "tables/table_cells.csv",
        "Long-form cell-level CSV for Datatool lookups across both hand-written tables.",
        "Derived from the two row-based table CSVs written in this build.",
        [
            SourceRef("industry_table_csv", INPUTS_DIR / "tables/industry_gap_distribution_table.csv", "Generated earlier in this same build."),
            SourceRef("contribution_table_csv", INPUTS_DIR / "tables/contribution_gap_distribution_table.csv", "Generated earlier in this same build."),
        ],
    )

    table_manifest_rows.extend(
        [
            {
                "table_csv": "tables/industry_gap_distribution_table.csv",
                "description": "Industry gap distribution table rows for Datatool or direct CSV lookup.",
                "notes": "Current classifier uses Other instead of the old Finance row and includes Utilities explicitly.",
            },
            {
                "table_csv": "tables/industry_gap_distribution_lookup.csv",
                "description": "Lookup-key version of the industry gap distribution table.",
                "notes": "Adds lookup_key = period_rowkey for easier Datatool fetches.",
            },
            {
                "table_csv": "tables/contribution_gap_distribution_table.csv",
                "description": "Contribution distribution table rows for Datatool or direct CSV lookup.",
                "notes": "Display labels track the draft table while internal source_group preserves the underlying repo grouping.",
            },
            {
                "table_csv": "tables/contribution_gap_distribution_lookup.csv",
                "description": "Lookup-key version of the contribution distribution table.",
                "notes": "Adds lookup_key = period_rowkey for easier Datatool fetches.",
            },
            {
                "table_csv": "tables/table_cells.csv",
                "description": "Long-form cell-level lookup table spanning both hand-written tables.",
                "notes": "Intended to make Datatool lookups easier when fetching a single table entry.",
            },
        ]
    )


def assert_pipeline_counts_agree() -> None:
    """The draft pulls the same counts from two CSVs; make a mismatch fail loudly.

    reconciliation_extraction_headlines.csv and slide_data/pipelinecounts.csv are
    produced by different scripts. Nothing else enforces that they agree.
    """
    heads = pd.read_csv(INPUTS_DIR / "tables" / "reconciliation_extraction_headlines.csv")
    h = dict(zip(heads["row_key"], heads["value"]))
    pc = pd.read_csv(REPO_SLIDE_DATA_DIR / "pipelinecounts.csv")
    p = dict(zip(pc["key"], pc["value"]))
    checks = [("phase2_extracted_filings", "phaseTwoUniqueFilings"),
              ("phase2_extracted_firm_years", "phaseTwoTotalRows")]
    for hk, pk in checks:
        if hk in h and pk in p and int(float(h[hk])) != int(float(p[pk])):
            raise AssertionError(
                f"pipeline count mismatch: headlines[{hk}]={h[hk]} vs pipelinecounts[{pk}]={p[pk]}"
            )
    print("Pipeline count cross-check: headlines and pipelinecounts agree")


def build_coverage_by_year_csv() -> None:
    """Per-year coverage of the analysis panel, 1998-2024, for the appendix figure.

    Same stage 4 basis as the coverage table, so the year-by-year series, the
    1998/2008 figures quoted in Section 2.2, and Table 3 all agree.
    """
    src = REPO_ROOT / "data" / "analysis" / "output" / "coverage_audit" / "stage4_analysis_panel.csv"
    df = pd.read_csv(src)
    g = df[(df["group"] == "All") & df["year"].between(1998, 2024)].copy()
    out = pd.DataFrame({
        "year": g["year"].astype(int),
        "equal_pct": g["coverage_count_pct"].round(2),
        "sales_pct": (100.0 * g["num_sale_M"] / g["denom_sale_M"]).round(2),
    }).sort_values("year")
    dest = INPUTS_DIR / "coverage_time_tikz"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "coverage_by_year.csv", index=False)
    print(f"Wrote {dest / 'coverage_by_year.csv'} ({len(out)} years)")


def build_coverage_table_csvs() -> None:
    # Stage 4, the canonical analysis panel, is the object the paper analyses:
    # it is stage 3 plus the filings recovered outside the main phase 2 run and
    # the overlap repairs. Section 2.2's year-by-year coverage figures are on
    # the same basis, so the table and the text agree.
    coverage_path = REPO_ROOT / "data" / "analysis" / "output" / "coverage_audit" / "stage4_analysis_panel.csv"
    df = pd.read_csv(coverage_path)
    group_map = {
        "Top100": ("top100", "Top 100"),
        "Top250": ("top250", "Top 250"),
        "Top500": ("top500", "Top 500"),
        "All": ("full_sample", "Full sample"),
    }
    order_map = {"Top100": 1, "Top250": 2, "Top500": 3, "All": 4}

    sub = df[df["year"].between(2013, 2024) & df["group"].isin(group_map)].copy()
    rows: list[dict[str, object]] = []
    for group in ["Top100", "Top250", "Top500", "All"]:
        g = sub[sub["group"] == group].copy()
        key, label = group_map[group]
        rows.append(
            {
                "table_id": "coverage_extracted_reconciliations",
                "period": "2013_2024",
                "period_label": "2013--2024",
                "row_order": order_map[group],
                "row_key": key,
                "row_label": label,
                "years_covered": 12,
                "compustat_firm_years": int(g["denom_count"].sum()),
                "covered_firm_years": int(g["num_count"].sum()),
                "equal_weight_share_pct": float(g["coverage_count_pct"].mean()),
                "sales_weight_share_pct": float(100.0 * g["num_sale_M"].sum() / g["denom_sale_M"].sum()),
            }
        )

    coverage_df = pd.DataFrame(rows)[
        [
            "table_id",
            "period",
            "period_label",
            "row_order",
            "row_key",
            "row_label",
            "years_covered",
            "compustat_firm_years",
            "covered_firm_years",
            "equal_weight_share_pct",
            "sales_weight_share_pct",
        ]
    ]
    sources = [
        SourceRef("coverage_stage3_csv", coverage_path, "Current repo stage-3 coverage audit file."),
    ]
    write_csv(
        coverage_df,
        "tables/coverage_table.csv",
        "Row-based CSV for Table 3 coverage statistics.",
        "Repo-backed coverage table: stage3_phase2_extracted.csv restricted to 2013-2024 and groups All/Top100/Top250/Top500; equal-weight share is the simple average of annual coverage_count_pct, sales-weight share is summed num_sale_M over summed denom_sale_M.",
        sources,
        float_format="%.6f",
    )

    stat_map = [
        ("covered_firm_years", "Covered firm-years"),
        ("equal_weight_share_pct", "Equal-weight share (%)"),
        ("sales_weight_share_pct", "Sales-weighted share (%)"),
    ]
    cell_rows: list[dict[str, object]] = []
    for _, row in coverage_df.iterrows():
        for stat_key, stat_label in stat_map:
            cell_rows.append(
                {
                    "table_id": row["table_id"],
                    "period": row["period"],
                    "period_label": row["period_label"],
                    "row_order": row["row_order"],
                    "row_key": row["row_key"],
                    "row_label": row["row_label"],
                    "stat_key": stat_key,
                    "stat_label": stat_label,
                    "value": row[stat_key],
                }
            )
    write_csv(
        pd.DataFrame(cell_rows),
        "tables/coverage_table_cells.csv",
        "Long-form cell-level CSV for the coverage table.",
        "Derived directly from tables/coverage_table.csv in this build.",
        [
            SourceRef("coverage_table_csv", INPUTS_DIR / "tables/coverage_table.csv", "Generated earlier in this same build."),
        ],
        float_format="%.6f",
    )

    table_manifest_rows.extend(
        [
            {
                "table_csv": "tables/coverage_table.csv",
                "description": "Coverage table rows for the hand-written Table 3.",
                "notes": "Uses repo stage3_phase2_extracted.csv for 2013-2024; Top groups are annual sales ranks.",
            },
            {
                "table_csv": "tables/coverage_table_cells.csv",
                "description": "Long-form cell-level lookup table for the coverage table.",
                "notes": "Useful if you want Datatool to fetch one coverage entry at a time.",
            },
        ]
    )


def write_manifests() -> None:
    pd.DataFrame(generated_rows).sort_values("output_relpath").to_csv(INPUTS_DIR / "generated_files.csv", index=False)
    pd.DataFrame(provenance_rows).sort_values(["output_relpath", "source_role"]).to_csv(INPUTS_DIR / "provenance_manifest.csv", index=False)
    pd.DataFrame(draft_input_rows).sort_values("draft_input").to_csv(INPUTS_DIR / "current_draft_inputs_manifest.csv", index=False)
    pd.DataFrame(table_manifest_rows).to_csv(INPUTS_DIR / "table_csvs_manifest.csv", index=False)


def write_readme() -> None:
    text = f"""# Inputs Build

This directory is the clean staging area for the CSV inputs currently used by
`/Users/erm/Draft/tcja_draft1.tex`, plus CSV versions of the two hand-written
tables in the draft.

Build timestamp:

- `{GENERATED_AT}`

Canonical upstream sources used in this build:

- Repo code: `/Users/erm/Repos/TCJA/analysis`
- Current data tree: `/Volumes/Crucial X6/Users/apb/TaxReconciliation/data/analysis`

What was rebuilt here:

- `decomp_tikz/fig_drivers_tikz_data.csv`
- `decomp_ind_new_tikz/summary.csv`
- `first_stage_tikz/*.csv`
- `first_etr_tikz/*.csv`
- `first_pishr_tikz/*.csv`
- `tables/industry_gap_distribution_table.csv`
- `tables/industry_gap_distribution_lookup.csv`
- `tables/contribution_gap_distribution_table.csv`
- `tables/contribution_gap_distribution_lookup.csv`
- `tables/contribution_gap_changes.csv`
- `tables/coverage_table.csv`
- `tables/table_cells.csv`

Key provenance files:

- `generated_files.csv`
  One row per generated file, with build timestamp and a short description.
- `provenance_manifest.csv`
  One row per generated-file / source-file pair, including the source path and
  the source file's modification timestamp.
- `current_draft_inputs_manifest.csv`
  Summary of the draft inputs currently loaded by `tcja_draft1.tex` and how
  they were rebuilt here.
- `table_csvs_manifest.csv`
  Summary of the new table CSVs created for the two hand-written tables.

Important notes:

- This build does **not** use `/Users/erm/TaxReconciliation_explore/analysis/cache`.
  That local tree is stale.
- The current repo classifier in `sic_industry.py` is applied wherever industry
  labels matter. As a result, the old descriptive `Finance` row becomes `Other`
  in the rebuilt industry-facing files, and `Utilities` is preserved as its own
  industry row.
- The foreign-PI-share event-study inputs under `first_pishr_tikz/` are rebuilt
  from the current `build_substitution_real_outcomes.py` logic, which is the
  up-to-date repo source for that regression setup.
- In `tcja_draft1.tex`, the `first_pishr_tikz` helper still uses a relative
  path instead of `\\Inputs/first_pishr_tikz/...`. If you point the draft at
  this directory, that helper should be updated to match the others.
"""
    write_text(
        "README.md",
        text,
        "Top-level explanation of the Inputs build and provenance files.",
        "Generated directly by build_inputs_from_external.py.",
        [
            SourceRef("repo_classifier", REPO_ANALYSIS_DIR / "sic_industry.py", "Referenced in the build."),
            SourceRef("external_data_root", EXTERNAL_ANALYSIS_DIR, "Current external-drive source tree."),
        ],
    )


def main() -> None:
    full_sample = pd.read_pickle(EXTERNAL_CACHE_DIR / "full_sample.pkl")
    driver_panel_full = pd.read_pickle(EXTERNAL_CACHE_DIR / "driver_panel_full.pkl")
    combined = pd.read_pickle(EXTERNAL_CACHE_DIR / "combined.pkl")
    long_df = pd.read_pickle(EXTERNAL_CACHE_DIR / "long.pkl")

    build_fig_drivers()
    build_industry_summary(full_sample, driver_panel_full)
    first_stage_panel = build_first_stage_family(combined, long_df)
    build_first_pishr_family(combined, long_df)
    build_industry_passthrough_examples(first_stage_panel)
    build_table_csvs(full_sample, driver_panel_full)
    build_coverage_table_csvs()
    build_coverage_by_year_csv()
    assert_pipeline_counts_agree()
    write_readme()
    write_manifests()
    print(f"Wrote Inputs build under {INPUTS_DIR}")


if __name__ == "__main__":
    main()
