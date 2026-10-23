#!/usr/bin/env python3
"""
build_geo_revenue_share_outcomes.py

Parallel extension to build_substitution_real_outcomes.py:
does TCJA's closure of the International reconciliation channel also change
where firms generate revenue geographically?

Uses the same 1,277-firm substitution panel and treatment as the income-location
slides. The outcome is built from Compustat Historical Segments GEOSEG rows:

    foreign_rev_share = foreign positive GEOSEG sales / total positive GEOSEG sales

where foreign is geotp == 3. Domestic-only geography reporters receive a zero
foreign revenue share when total GEOSEG sales are positive.
"""

import json
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_substitution_real_outcomes as real

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SEGMENT_CSV = os.path.join(ROOT_DIR, "data", "original", "compustat_segments.csv")
FIG_DIR = os.path.join(ROOT_DIR, "data", "analysis", "figures", "substitution")
TBL_DIR = os.path.join(ROOT_DIR, "data", "analysis", "tables", "substitution")
OUT_DIR = os.path.join(ROOT_DIR, "data", "analysis", "output")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def build_base_panel():
    combined, long_df, pi = real.load_data()
    intl_pre = real.build_channels_and_treatment(long_df)
    intl_yearly = real.build_intl_yearly(long_df)
    return real.build_panel(combined, pi, intl_pre, intl_yearly)


def load_geoseg_rows(panel):
    base_years = panel[["gvkey", "recon_year"]].drop_duplicates().copy()
    base_years["gvkey"] = pd.to_numeric(base_years["gvkey"], errors="coerce").astype("Int64")
    base_years["year"] = pd.to_numeric(base_years["recon_year"], errors="coerce").astype("Int64")
    base_years = base_years.dropna(subset=["gvkey", "year"])
    base_years["gvkey"] = base_years["gvkey"].astype(int)
    base_years["year"] = base_years["year"].astype(int)

    keep_gvkeys = set(base_years["gvkey"].unique())
    usecols = ["stype", "datadate", "gvkey", "snms", "geotp", "sales", "sid"]
    chunks = []
    for chunk in pd.read_csv(SEGMENT_CSV, usecols=usecols, chunksize=300_000, low_memory=False):
        chunk = chunk[chunk["stype"].eq("GEOSEG")].copy()
        chunk["gvkey_num"] = pd.to_numeric(chunk["gvkey"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["gvkey_num"].isin(keep_gvkeys)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    seg = pd.concat(chunks, ignore_index=True)
    seg["gvkey"] = pd.to_numeric(seg["gvkey"], errors="coerce").astype("Int64")
    seg["year"] = pd.to_datetime(seg["datadate"], errors="coerce").dt.year.astype("Int64")
    seg["geotp_num"] = pd.to_numeric(seg["geotp"], errors="coerce")
    seg["sales_num"] = pd.to_numeric(seg["sales"], errors="coerce")
    seg = seg.dropna(subset=["gvkey", "year"])
    seg["gvkey"] = seg["gvkey"].astype(int)
    seg["year"] = seg["year"].astype(int)
    seg = seg.merge(base_years[["gvkey", "year"]], on=["gvkey", "year"], how="inner")

    sid_key = seg["sid"].astype("string")
    sid_key = sid_key.where(sid_key.notna(), "name:" + seg["snms"].astype("string").fillna(""))
    seg["sid_key"] = sid_key.astype(str)

    return seg.drop_duplicates(subset=["gvkey", "year", "sid_key", "geotp_num", "sales_num"])


def build_geo_revenue_share(panel):
    seg = load_geoseg_rows(panel)
    if seg.empty:
        raise RuntimeError(f"No GEOSEG rows found in {SEGMENT_CSV}")

    seg["positive_sales"] = seg["sales_num"].where(seg["sales_num"].gt(0), 0.0)
    seg["foreign_positive_sales"] = np.where(
        seg["geotp_num"].eq(3) & seg["sales_num"].gt(0), seg["sales_num"], 0.0
    )
    seg["domestic_positive_sales"] = np.where(
        seg["geotp_num"].eq(2) & seg["sales_num"].gt(0), seg["sales_num"], 0.0
    )
    seg["foreign_row_missing_sales"] = seg["geotp_num"].eq(3) & seg["sales_num"].isna()
    seg["foreign_row_nonpositive_sales"] = (
        seg["geotp_num"].eq(3) & seg["sales_num"].notna() & ~seg["sales_num"].gt(0)
    )

    geo = seg.groupby(["gvkey", "year"]).agg(
        foreign_geo_sales=("foreign_positive_sales", "sum"),
        domestic_geo_sales=("domestic_positive_sales", "sum"),
        total_geo_sales=("positive_sales", "sum"),
        n_geo_rows=("stype", "size"),
        n_geo_names=("snms", lambda x: x.dropna().nunique()),
        has_foreign_geo_row=("geotp_num", lambda x: bool(pd.Series(x).eq(3).any())),
        has_foreign_row_missing_sales=("foreign_row_missing_sales", "any"),
        has_foreign_row_nonpositive_sales=("foreign_row_nonpositive_sales", "any"),
    ).reset_index()

    geo["foreign_rev_share"] = np.where(
        geo["total_geo_sales"].gt(0),
        geo["foreign_geo_sales"] / geo["total_geo_sales"],
        np.nan,
    )
    geo["foreign_rev_share"] = geo["foreign_rev_share"].clip(0, 1)
    geo = geo.rename(columns={"year": "recon_year"})
    return panel.merge(geo, on=["gvkey", "recon_year"], how="left")


def run_results(panel):
    panel = panel.copy()
    panel["International_geo_rev_sample"] = np.where(
        panel["foreign_rev_share"].notna(), panel["International"], np.nan
    )

    results = {
        "foreign_rev_share": {
            "baseline": real.run_pooled_did(panel, "foreign_rev_share", ind_yr_fe=False),
            "ind_yr": real.run_pooled_did(panel, "foreign_rev_share", ind_yr_fe=True),
        },
        "international_same_sample": {
            "baseline": real.run_pooled_did(panel, "International_geo_rev_sample", ind_yr_fe=False),
            "ind_yr": real.run_pooled_did(panel, "International_geo_rev_sample", ind_yr_fe=True),
        },
    }

    results["closure_ratio"] = {
        spec: 100
        * results["foreign_rev_share"][spec]["coef"]
        / abs(results["international_same_sample"][spec]["coef"])
        for spec in ["baseline", "ind_yr"]
    }
    return results


def event_study(panel):
    base = real.run_event_study(panel, "foreign_rev_share", ind_yr_fe=False)
    ind = real.run_event_study(panel, "foreign_rev_share", ind_yr_fe=True)
    if not base.empty:
        base["spec"] = "Firm + year FE"
    if not ind.empty:
        ind["spec"] = "Firm + industry x year FE"
    return pd.concat([base, ind], ignore_index=True)


def plot_event_study(es):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {
        "Firm + year FE": "#7f8c8d",
        "Firm + industry x year FE": "#2980b9",
    }
    offsets = {
        "Firm + year FE": -0.06,
        "Firm + industry x year FE": 0.06,
    }
    for spec, grp in es.groupby("spec"):
        grp = grp.sort_values("year")
        x = grp["year"].astype(float) + offsets.get(spec, 0.0)
        ax.errorbar(
            x,
            grp["beta"],
            yerr=1.96 * grp["se"],
            fmt="o-",
            lw=1.6,
            ms=4,
            capsize=3,
            color=colors.get(spec, "#333333"),
            label=spec,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(2016, color="#c0392b", lw=0.9, ls="--", alpha=0.75)
    ax.axvspan(2017, 2018, color="#d9d9d9", alpha=0.35, label="transition years")
    ax.set_title("Foreign revenue-share response")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\hat\beta_y$: intl_pre $\times$ year")
    ax.set_xticks(sorted(es["year"].unique()))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_geo_revenue_share_event.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def stars(coef, se):
    if se <= 0 or np.isnan(se):
        return ""
    return real.stars(abs(coef / se))


def fmt_coef(res):
    sig = stars(res["coef"], res["se"])
    suffix = f"^{{{sig}}}" if sig else ""
    return f"${res['coef']:+.5f}{suffix}$"


def fmt_se(res):
    return f"$({res['se']:.5f})$"


def write_table(results):
    rev = results["foreign_rev_share"]
    intl = results["international_same_sample"]
    ratio = results["closure_ratio"]
    text = rf"""\begin{{table}}[htbp]
\centering
\caption{{Foreign revenue-share effects from Compustat geographic segments}}
\label{{tab:geo_revenue_share}}
\small
\begin{{tabular}}{{lccc}}
\toprule
 & (1) Baseline & (2) + Ind$\times$yr FE & $N$ firms \\
\midrule
Foreign revenue share & {fmt_coef(rev["baseline"])} & {fmt_coef(rev["ind_yr"])} & {rev["ind_yr"]["G"]:,} \\
 & {fmt_se(rev["baseline"])} & {fmt_se(rev["ind_yr"])} & \\
International channel, same sample & {fmt_coef(intl["baseline"])} & {fmt_coef(intl["ind_yr"])} & {intl["ind_yr"]["G"]:,} \\
 & {fmt_se(intl["baseline"])} & {fmt_se(intl["ind_yr"])} & \\
\midrule
Closure ratio & {ratio["baseline"]:.2f} pp & {ratio["ind_yr"]:.2f} pp & \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}\footnotesize
\item Foreign revenue share is positive foreign GEOSEG sales divided by total positive GEOSEG sales. Foreign rows have \texttt{{geotp}}=3; domestic rows have \texttt{{geotp}}=2. Duplicate segment aliases are removed at the gvkey-year-sid-geography-sales level.
\item Sample: same 1,277-firm substitution panel; regression mask requires non-missing GEOSEG revenue share. DiD excludes 2017 and 2018 transition years. SE clustered at firm.
\item Closure ratio is $100\times\hat\beta_{{\text{{foreign rev share}}}}/|\hat\beta_{{\text{{International}}}}|$, in pp foreign revenue share per 1 pp International-channel closure.
\end{{tablenotes}}
\end{{table}}
"""
    out = os.path.join(TBL_DIR, "tbl_geo_revenue_share.tex")
    with open(out, "w") as fh:
        fh.write(text)
    return out


def summarize(panel, results):
    reg_years = [yr for yr in real.EVENT_YEARS if yr not in {2017, 2018}]
    reg_sample = panel[
        panel["recon_year"].isin(reg_years)
        & panel["foreign_rev_share"].notna()
        & panel["intl_pre"].notna()
    ]
    all_geo = panel[panel["foreign_rev_share"].notna()]
    return {
        "definition": (
            "foreign_geo_sales / total_positive_geo_sales using Compustat GEOSEG rows; "
            "foreign is geotp==3 and domestic is geotp==2; duplicate segment aliases "
            "removed by gvkey-year-sid-geotp-sales."
        ),
        "panel": {"firms": int(panel["gvkey"].nunique()), "firm_years": int(len(panel))},
        "all_event_years_with_geo_revenue_share": {
            "firms": int(all_geo["gvkey"].nunique()),
            "firm_years": int(len(all_geo)),
        },
        "pooled_did_regression_sample": {
            "firms": int(reg_sample["gvkey"].nunique()),
            "firm_years": int(len(reg_sample)),
        },
        "pre_mean_foreign_rev_share": float(
            panel.loc[panel["recon_year"].isin(real.PRE_YEARS), "foreign_rev_share"].mean()
        ),
        "post_mean_foreign_rev_share": float(
            panel.loc[panel["recon_year"].isin(real.POST_YEARS), "foreign_rev_share"].mean()
        ),
        "share_zero_firm_years_in_reg_sample": float(reg_sample["foreign_rev_share"].eq(0).mean()),
        "share_one_firm_years_in_reg_sample": float(reg_sample["foreign_rev_share"].eq(1).mean()),
        "results": results["foreign_rev_share"],
        "international_same_sample": results["international_same_sample"],
        "closure_ratio_pp_revenue_share_per_1pp_international_closure": results["closure_ratio"],
    }


def main():
    panel = build_base_panel()
    panel = build_geo_revenue_share(panel)
    results = run_results(panel)
    es = event_study(panel)
    fig = plot_event_study(es)
    table = write_table(results)
    summary = summarize(panel, results)

    summary_path = os.path.join(OUT_DIR, "geo_revenue_share_tcja_regression_summary.json")
    panel_path = os.path.join(OUT_DIR, "geo_revenue_share_tcja_panel.csv")
    es_path = os.path.join(OUT_DIR, "geo_revenue_share_tcja_event_study.csv")

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    panel[
        [
            "gvkey",
            "recon_year",
            "industry",
            "intl_pre",
            "International",
            "foreign_rev_share",
            "foreign_geo_sales",
            "domestic_geo_sales",
            "total_geo_sales",
            "n_geo_rows",
            "n_geo_names",
            "has_foreign_geo_row",
            "has_foreign_row_missing_sales",
            "has_foreign_row_nonpositive_sales",
        ]
    ].to_csv(panel_path, index=False)
    es.to_csv(es_path, index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote:\n  {fig}\n  {table}\n  {summary_path}\n  {panel_path}\n  {es_path}")


if __name__ == "__main__":
    main()
