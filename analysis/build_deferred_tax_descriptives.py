"""
Build deferred-tax descriptive figures + data for appendix A.16a-g.

Produces:
  data/analysis/figures/deferred_tax/fig_dt_aggregate_trend.png
      Two-panel time series: aggregate gross DTA and DTL for the Part-III
      windfall cohort, panel A normalized by aggregate assets (%), panel B
      in USD billions.

Reusable helpers used by build_slide_data.py to emit slide-data macros:
  build_windfall_cohort()         -> gvkey-indexed treatment cohort
  build_aggregate_trend()         -> per-year aggregates over that cohort
  build_cross_section_2016()      -> per-industry summary stats at end-2016

Cohort construction mirrors analysis/build_investment_elasticity.py
build_treatment() (Z = -txndb_2016/at_2015, |VA 2017| <= 3pp, non-Finance).
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

WINDFALL_FRAC = 0.40   # (35 - 21) / 35
VA_THRESHOLD  = 3.0    # |2017 valuation-allowance pp| <= this
YEARS         = range(1998, 2025)

from sic_industry import industry  # canonical SIC -> industry classifier


# ── Cohort construction ─────────────────────────────────────────────────────

def _load_base_panel():
    """Load the columns we need from tax_reconciliation_panel_base.csv."""
    p = ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv"
    cols = ["gvkey", "recon_year",
            "comp_txndb", "comp_txndba", "comp_txndbl",
            "comp_at", "comp_sic"]
    df = pd.read_csv(p, usecols=cols, low_memory=False)
    df = df.drop_duplicates(["gvkey", "recon_year"]).rename(
        columns={c: c.replace("comp_", "") for c in cols if c.startswith("comp_")}
    )
    return df


def _load_long_df():
    p = ROOT / "data" / "analysis" / "cache" / "long.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def _load_combined():
    p = ROOT / "data" / "analysis" / "cache" / "combined.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


PRE_YEARS  = list(range(2013, 2017))     # 2013-2016
POST_YEARS = list(range(2018, 2023))     # 2018-2022


def build_windfall_cohort():
    """Return the windfall cohort with industry + Z, after the same
    treatment + balanced-panel filters as analysis/build_investment_elasticity.py.

    Columns: gvkey, sic, industry, txndb_2016, at_2015, Z, Z_w, W.
    """
    panel = _load_base_panel()
    long_df = _load_long_df()
    combined = _load_combined()

    comp2016 = (panel[panel["recon_year"] == 2016]
                [["gvkey", "txndb", "sic"]]
                .rename(columns={"txndb": "txndb_2016"})
                .dropna(subset=["txndb_2016"]))
    comp2015 = (panel[panel["recon_year"] == 2015]
                [["gvkey", "at"]]
                .rename(columns={"at": "at_2015"}))
    comp2015 = comp2015[comp2015["at_2015"] > 0]

    treat = comp2016.merge(comp2015, on="gvkey", how="inner")
    treat["Z"] = -treat["txndb_2016"] / treat["at_2015"]

    p1, p99 = treat["Z"].quantile([0.01, 0.99])
    treat["Z_w"] = treat["Z"].clip(p1, p99)
    treat["W"] = WINDFALL_FRAC * treat["Z_w"]

    # 2017 VA exclusion: |sum of 2017 'Valuation allowance' category pp| > 3
    va_2017 = (long_df[(long_df["recon_year"] == 2017) &
                       (long_df["category"] == "Valuation allowance")]
               .groupby("gvkey")["cat_val"].sum()
               .rename("va_2017_pp"))
    treat = treat.merge(va_2017, on="gvkey", how="left")
    treat["va_2017_pp"] = treat["va_2017_pp"].fillna(0)
    treat = treat[treat["va_2017_pp"].abs() <= VA_THRESHOLD].copy()

    # Non-Finance restriction matches build_event_panel().
    treat["industry"] = treat["sic"].apply(industry)
    treat = treat[treat["industry"] != "Finance"].copy()

    # Balanced-panel filter: must have >=2 pre + >=2 post reconciliation years
    # in the combined panel.  This produces the 1,354-firm slide-77 sample.
    cyrs = combined.groupby("gvkey")["recon_year"].apply(set)
    has_pre  = cyrs.apply(lambda s: len(s & set(PRE_YEARS))  >= 2)
    has_post = cyrs.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok = cyrs[has_pre & has_post].index
    treat = treat[treat["gvkey"].isin(ok)].copy()

    return treat.reset_index(drop=True)


# ── Aggregate trend over the cohort ─────────────────────────────────────────

def build_aggregate_trend(cohort):
    """For the cohort, return year-by-year aggregates of DTA, DTL, and assets.

    Columns: year, n_firms, agg_dta_bn, agg_dtl_bn, agg_at_bn,
             dta_share_pct, dtl_share_pct.
    All in USD billions; shares in percent of aggregate assets.
    """
    panel = _load_base_panel()
    gvkeys = set(cohort["gvkey"])

    sub = panel[panel["gvkey"].isin(gvkeys)
                & panel["recon_year"].between(min(YEARS), max(YEARS))
                & (panel["at"] > 0)].copy()

    rows = []
    for yr in YEARS:
        sl = sub[sub["recon_year"] == yr]
        with_both = sl.dropna(subset=["txndba", "txndbl"])
        agg_dta = with_both["txndba"].sum()    # $millions
        agg_dtl = with_both["txndbl"].sum()    # $millions
        agg_at  = with_both["at"].sum()        # $millions
        n_firms = with_both["gvkey"].nunique()
        rows.append({
            "year": yr,
            "n_firms": n_firms,
            "agg_dta_bn": agg_dta / 1e3,       # convert M -> B
            "agg_dtl_bn": agg_dtl / 1e3,
            "agg_at_bn":  agg_at  / 1e3,
            "dta_share_pct": 100 * agg_dta / agg_at if agg_at > 0 else np.nan,
            "dtl_share_pct": 100 * agg_dtl / agg_at if agg_at > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ── Cross-section at end-2016 ───────────────────────────────────────────────

def build_cross_section_2016(cohort):
    """Per-industry summary of txndb_2016 / at_2015 (= -Z).

    Reported with -txndb/at sign convention so that a *positive* value
    means net DTL = windfall winner. Columns:
    industry, n_firms, mean_pct, p10_pct, p25_pct, median_pct, p75_pct, p90_pct,
    pct_net_dtl, pct_net_dta.
    """
    df = cohort.copy()
    # -Z = txndb_2016/at_2015 (raw, no winsorization). Positive = net DTL.
    df["neg_z_pct"] = 100 * df["Z"]   # Z is already -txndb/at, in fraction units

    industries = ["Tech", "Pharma/Biotech", "Manufacturing", "Retail",
                  "Oil & Gas", "Utilities", "Other"]
    rows = []
    for ind in industries + ["All"]:
        sl = df if ind == "All" else df[df["industry"] == ind]
        if len(sl) == 0:
            continue
        x = sl["neg_z_pct"]
        rows.append({
            "industry":    ind,
            "n_firms":     len(sl),
            "mean_pct":    x.mean(),
            "p10_pct":     x.quantile(0.10),
            "p25_pct":     x.quantile(0.25),
            "median_pct":  x.median(),
            "p75_pct":     x.quantile(0.75),
            "p90_pct":     x.quantile(0.90),
            "pct_net_dtl": 100 * (sl["Z"] > 0).mean(),
            "pct_net_dta": 100 * (sl["Z"] < 0).mean(),
        })
    return pd.DataFrame(rows)


# ── Figure ──────────────────────────────────────────────────────────────────

def make_figure(trend, out_path):
    """Two-panel time series. Panel A: normalized %. Panel B: USD billions."""
    matplotlib.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10.5,
        "axes.labelsize": 10.5, "axes.titlesize": 11,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))

    DTA_COLOR = "#2c7fb8"
    DTL_COLOR = "#c0392b"
    SHOCK_X   = 2017.5

    # Panel A: shares of assets
    ax = axes[0]
    ax.plot(trend["year"], trend["dta_share_pct"], color=DTA_COLOR, lw=2.2,
            marker="o", ms=4.0, label="Gross DTA / aggregate assets")
    ax.plot(trend["year"], trend["dtl_share_pct"], color=DTL_COLOR, lw=2.2,
            marker="s", ms=4.0, label="Gross DTL / aggregate assets")
    ax.axvline(SHOCK_X, color="dimgray", lw=1.2, ls="--", alpha=0.7)
    ax.text(SHOCK_X + 0.15, ax.get_ylim()[1] * 0.95, "TCJA", color="dimgray",
            fontsize=9, va="top", ha="left")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("% of aggregate assets")
    ax.set_title("A. Normalized (windfall cohort, N = "
                 f"{int(trend['n_firms'].max()):,} firms peak coverage)")
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: levels
    ax = axes[1]
    ax.plot(trend["year"], trend["agg_dta_bn"], color=DTA_COLOR, lw=2.2,
            marker="o", ms=4.0, label="Aggregate gross DTA")
    ax.plot(trend["year"], trend["agg_dtl_bn"], color=DTL_COLOR, lw=2.2,
            marker="s", ms=4.0, label="Aggregate gross DTL")
    ax.axvline(SHOCK_X, color="dimgray", lw=1.2, ls="--", alpha=0.7)
    ax.text(SHOCK_X + 0.15, ax.get_ylim()[1] * 0.95, "TCJA", color="dimgray",
            fontsize=9, va="top", ha="left")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("USD billions")
    ax.set_title("B. Levels (sum across cohort firms with both items)")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--figures-dir",
        default=str(ROOT / "data" / "analysis" / "figures" / "deferred_tax"),
    )
    args = p.parse_args()

    print("Building windfall cohort ...")
    cohort = build_windfall_cohort()
    print(f"  Cohort: {len(cohort):,} firms")
    print(f"  Net-DTL share: {(cohort['Z'] > 0).mean() * 100:.1f}%")
    print(f"  Net-DTA share: {(cohort['Z'] < 0).mean() * 100:.1f}%")

    print("\nBuilding aggregate trend ...")
    trend = build_aggregate_trend(cohort)
    print(trend[["year", "n_firms", "dta_share_pct", "dtl_share_pct"]]
          .to_string(index=False))

    print("\nBuilding 2016 cross-section ...")
    xs = build_cross_section_2016(cohort)
    print(xs.to_string(index=False))

    fig_path = os.path.join(args.figures_dir, "fig_dt_aggregate_trend.pdf")
    make_figure(trend, fig_path)
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
