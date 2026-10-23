#!/usr/bin/env python3
"""
capex_location_test.py

Does TCJA's closure of the International channel move the LOCATION of capital,
not just the location of booked profits? The consolidated capx/at outcome in
build_real_activity_dynamics.py nets out cross-border relocation by
construction. This test uses Compustat Historical Segments GEOSEG rows to
build geographic capital outcomes:

    foreign_ppe_share  = foreign positive ppents / total positive ppents
    foreign_capx_share = foreign positive capxs  / total positive capxs

ppents (PP&E by geography) is the ASC 280-mandated long-lived-assets
disclosure, so coverage is materially better than capxs. Construction mirrors
build_geo_revenue_share_outcomes.py (geotp==3 foreign, geotp==2 domestic,
duplicate aliases dropped at gvkey-year-sid-geotp-value level) with one
tightening: because ppents/capxs are sparsely populated within filings, a
firm-year with a foreign GEOSEG row whose measure is MISSING is excluded from
the strict share (missing foreign capital is not zero foreign capital).
Domestic-only geography reporters keep share = 0. A loose (sales-style)
variant is also reported.

Same panel, treatment, spec and estimator as the income-location results
(via build_substitution_real_outcomes): firm FE + industry x year FE,
intl_pre x post, pre 2013-16, post 2019-22, 2017-18 dropped, SE clustered
by firm.

Run:  python3 capex_location_test.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import build_substitution_real_outcomes as real
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SEGMENT_CSV = os.path.join(ROOT_DIR, "data", "original", "compustat_segments.csv")
COMPUSTAT_CSV = os.path.join(ROOT_DIR, "data", "original", "compustat_data.csv")
OUT_DIR = os.path.join(ROOT_DIR, "data", "analysis", "output")

MEASURES = ["ppents", "capxs", "sales"]
PRE_YEARS = list(range(2013, 2017))


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
    usecols = ["stype", "datadate", "gvkey", "snms", "geotp", "sid"] + MEASURES
    chunks = []
    for chunk in pd.read_csv(SEGMENT_CSV, usecols=usecols, chunksize=300_000, low_memory=False):
        chunk = chunk[chunk["stype"].eq("GEOSEG")].copy()
        chunk["gvkey_num"] = pd.to_numeric(chunk["gvkey"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["gvkey_num"].isin(keep_gvkeys)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No GEOSEG rows found in {SEGMENT_CSV}")

    seg = pd.concat(chunks, ignore_index=True)
    seg["gvkey"] = pd.to_numeric(seg["gvkey"], errors="coerce").astype("Int64")
    seg["year"] = pd.to_datetime(seg["datadate"], errors="coerce").dt.year.astype("Int64")
    seg["geotp_num"] = pd.to_numeric(seg["geotp"], errors="coerce")
    for m in MEASURES:
        seg[f"{m}_num"] = pd.to_numeric(seg[m], errors="coerce")
    seg = seg.dropna(subset=["gvkey", "year"])
    seg["gvkey"] = seg["gvkey"].astype(int)
    seg["year"] = seg["year"].astype(int)
    seg = seg.merge(base_years[["gvkey", "year"]], on=["gvkey", "year"], how="inner")

    sid_key = seg["sid"].astype("string")
    sid_key = sid_key.where(sid_key.notna(), "name:" + seg["snms"].astype("string").fillna(""))
    seg["sid_key"] = sid_key.astype(str)

    # Alias dedup per measure-bundle (same convention as the revenue script,
    # extended to the capital measures)
    return seg.drop_duplicates(
        subset=["gvkey", "year", "sid_key", "geotp_num"] + [f"{m}_num" for m in MEASURES]
    )


def build_shares(panel, seg):
    out = panel.copy()
    for m in MEASURES:
        v = f"{m}_num"
        seg["_pos"] = seg[v].where(seg[v].gt(0), 0.0)
        seg["_fpos"] = np.where(seg["geotp_num"].eq(3) & seg[v].gt(0), seg[v], 0.0)
        seg["_f_missing"] = seg["geotp_num"].eq(3) & seg[v].isna()
        g = seg.groupby(["gvkey", "year"]).agg(
            total=("_pos", "sum"),
            foreign=("_fpos", "sum"),
            has_foreign_row=("geotp_num", lambda x: bool(pd.Series(x).eq(3).any())),
            has_foreign_missing=("_f_missing", "any"),
        ).reset_index()

        share = np.where(g["total"].gt(0), g["foreign"] / g["total"], np.nan)
        g[f"foreign_{m}_share_loose"] = pd.Series(share).clip(0, 1)
        # Strict: drop firm-years where a foreign row exists with missing measure
        g[f"foreign_{m}_share"] = g[f"foreign_{m}_share_loose"].where(~g["has_foreign_missing"])
        g = g.rename(columns={"year": "recon_year",
                              "total": f"total_geo_{m}",
                              "foreign": f"foreign_geo_{m}",
                              "has_foreign_row": f"has_foreign_row_{m}",
                              "has_foreign_missing": f"has_foreign_missing_{m}"})
        out = out.merge(g, on=["gvkey", "recon_year"], how="left")
    return out


def did_block(panel, outcome):
    """Pooled DiD on `outcome` + International channel on the same subsample."""
    p = panel.copy()
    p["_intl_same"] = np.where(p[outcome].notna(), p["International"], np.nan)
    res = {
        "outcome": {
            "baseline": real.run_pooled_did(p, outcome, ind_yr_fe=False),
            "ind_yr": real.run_pooled_did(p, outcome, ind_yr_fe=True),
        },
        "international_same_sample": {
            "baseline": real.run_pooled_did(p, "_intl_same", ind_yr_fe=False),
            "ind_yr": real.run_pooled_did(p, "_intl_same", ind_yr_fe=True),
        },
    }
    res["closure_ratio"] = {}
    for spec in ["baseline", "ind_yr"]:
        denom = res["international_same_sample"][spec]["coef"]
        num = res["outcome"][spec]["coef"]
        res["closure_ratio"][spec] = (100 * num / abs(denom)) if denom else np.nan
    return res


def fmt(res):
    c, s, n, g = res["coef"], res["se"], res["n"], res["G"]
    if np.isnan(c):
        return "    (insufficient sample)"
    t = c / s if s else np.nan
    return f"    beta={c:+.5f}  SE={s:.5f}  t={t:+.2f}{real.stars(abs(t))}  N={n:,}  firms={g:,}"


# ── Heterogeneity: which firm types carry each margin (F4 mechanism test) ──

def _firm_intensity(usecol, num, den, lo_year=2013, hi_year=2016):
    """Firm-level pre-period mean of num/den from Compustat (missing num -> 0)."""
    df = pd.read_csv(COMPUSTAT_CSV, usecols=["gvkey", "fyear", num, den], low_memory=False)
    w = df[df["fyear"].between(lo_year, hi_year)].copy()
    w[usecol] = np.where(w[den] > 0, w[num].fillna(0.0) / w[den], np.nan)
    return w.groupby("gvkey")[usecol].mean()


def _standardize(s):
    lo, hi = s.quantile([0.01, 0.99])
    w = s.clip(lo, hi)
    sd = w.std()
    return (w - w.mean()) / sd if sd else w * 0.0


def _triple(panel, outcome, zcol):
    """Pooled triple-interaction DiD: outcome on [Di, Di*z, z*post], firm+indxyr FE."""
    excl = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [y for y in real.EVENT_YEARS if y not in excl]
    mask = (panel["recon_year"].isin(years) & panel[outcome].notna()
            & panel["intl_pre"].notna() & panel[zcol].notna())
    sub = panel[mask].copy()
    if len(sub) < 50 or sub["gvkey"].nunique() < 20:
        return None
    sub["_post"] = sub["recon_year"].isin(real.POST_YEARS).astype(float)
    sub["_Di"] = sub["intl_pre"] * sub["_post"]
    sub["_DiZ"] = sub["_Di"] * sub[zcol]
    sub["_Zpost"] = sub[zcol] * sub["_post"]
    cols = [outcome, "_Di", "_DiZ", "_Zpost"]
    dm = real.demean_ind_yr(sub, cols)
    mod = OLS(dm[outcome].values, dm[["_Di", "_DiZ", "_Zpost"]].values).fit()
    se = np.sqrt(np.diag(cov_cluster(mod, sub["gvkey"].values)))
    di_z, di_z_se = float(mod.params[1]), float(se[1])
    return {"di": float(mod.params[0]), "diz": di_z, "diz_se": di_z_se,
            "diz_t": di_z / di_z_se if di_z_se else float("nan"),
            "n": len(sub), "G": sub["gvkey"].nunique()}


def _split(panel, outcome, zcol):
    """Median split on firm-level zcol (within the outcome sample); high vs low DiD."""
    firm_z = panel.dropna(subset=[outcome]).drop_duplicates("gvkey").set_index("gvkey")[zcol]
    med = float(firm_z.median())
    hi = set(firm_z[firm_z > med].index)
    rh = real.run_pooled_did(panel[panel["gvkey"].isin(hi)], outcome, ind_yr_fe=True)
    rl = real.run_pooled_did(panel[~panel["gvkey"].isin(hi)], outcome, ind_yr_fe=True)
    return {"median": med, "high": rh, "low": rl}


def heterogeneity_block(panel):
    """R&D- and capital-intensity heterogeneity of the profit and capex margins.

    Tests the F4 mechanism: the weak foreign-capex tilt is NOT concentrated in
    R&D-intensive firms (rejecting IP--K complementarity) but IS concentrated in
    capital-intensive firms; profit relocation is the mirror image (asset-light,
    R&D-intensive). Returns a dict consumed by build_slide_data.py.
    """
    p = panel.copy()
    p = p.merge(_firm_intensity("rd_int", "xrd", "at").rename("rd_int"),
                on="gvkey", how="left")
    p = p.merge(_firm_intensity("capint", "ppent", "at").rename("capint"),
                on="gvkey", how="left")
    p["rd_int"] = p["rd_int"].fillna(0.0)
    p["rd_z"] = _standardize(p["rd_int"])
    p["capint_z"] = _standardize(p["capint"])

    out = {}
    for key, col in [("profit", "foreign_share"), ("capex", "foreign_capxs_share")]:
        if col not in p.columns:
            continue
        out[key] = {
            "rd_split": _split(p, col, "rd_int"),
            "capint_split": _split(p, col, "capint"),
            "rd_triple": _triple(p, col, "rd_z"),
            "capint_triple": _triple(p, col, "capint_z"),
        }
    return out


def main():
    panel = build_base_panel()
    seg = load_geoseg_rows(panel)
    panel = build_shares(panel, seg)

    print("\n" + "=" * 72)
    print("Capital-location test: foreign shares from GEOSEG segments")
    print("Spec: firm FE + ind x yr FE (and firm FE + yr FE), intl_pre x post,")
    print("pre 2013-16, post 2019-22, 2017-18 dropped, SE clustered by firm")
    print("=" * 72)

    outcomes = [
        ("foreign_ppe_share", "Foreign PP&E share (strict)   [capital STOCK location]"),
        ("foreign_ppe_share_loose", "Foreign PP&E share (loose, sales-style construction)"),
        ("foreign_capxs_share", "Foreign capex share (strict)  [capital FLOW location]"),
        ("foreign_sales_share", "Foreign revenue share, same construction [benchmark]"),
    ]
    # column name fix: ppents share built as foreign_ppents_share
    panel = panel.rename(columns={
        "foreign_ppents_share": "foreign_ppe_share",
        "foreign_ppents_share_loose": "foreign_ppe_share_loose",
    })

    summary = {}
    for col, label in outcomes:
        if col not in panel.columns:
            print(f"\n{label}: column missing, skipped")
            continue
        nn = panel[col].notna().sum()
        nf = panel.loc[panel[col].notna(), "gvkey"].nunique()
        print(f"\n{label}")
        print(f"  coverage: {nn:,} firm-years / {nf:,} firms "
              f"(of {len(panel):,} / {panel['gvkey'].nunique():,} panel)")
        pre_m = panel.loc[panel["recon_year"].isin(real.PRE_YEARS), col].mean()
        post_m = panel.loc[panel["recon_year"].isin(real.POST_YEARS), col].mean()
        print(f"  mean: pre={pre_m:.3f}  post={post_m:.3f}")
        blk = did_block(panel, col)
        print("  Firm + yr FE:")
        print(fmt(blk["outcome"]["baseline"]))
        print("  Firm + ind x yr FE:")
        print(fmt(blk["outcome"]["ind_yr"]))
        print("  International channel, same sample (ind x yr FE):")
        print(fmt(blk["international_same_sample"]["ind_yr"]))
        print(f"  closure ratio (pp share per 1pp Intl closure): "
              f"baseline {blk['closure_ratio']['baseline']:+.2f}, "
              f"ind_yr {blk['closure_ratio']['ind_yr']:+.2f}")
        summary[col] = blk

    # Event study on the primary outcome
    print("\n" + "=" * 72)
    print("Event study: foreign PP&E share (strict), firm + ind x yr FE, omit 2016")
    print("=" * 72)
    es = real.run_event_study(panel, "foreign_ppe_share", ind_yr_fe=True)
    if not es.empty:
        for _, r in es.iterrows():
            print(f"  {int(r['year'])}:  beta={r['beta']:+.5f}  SE={r['se']:.5f}")
        es.to_csv(os.path.join(OUT_DIR, "capital_location_event_study.csv"), index=False)

    # Heterogeneity: which firm types carry each margin (F4 mechanism test)
    print("\n" + "=" * 72)
    print("Heterogeneity: R&D- and capital-intensity splits (profit vs capex margin)")
    print("=" * 72)
    het = heterogeneity_block(panel)
    summary["heterogeneity"] = het
    for marg in ("profit", "capex"):
        if marg not in het:
            continue
        print(f"\n  {marg.upper()} margin:")
        for sp in ("rd_triple", "capint_triple"):
            tr = het[marg][sp]
            if tr:
                print(f"    {sp}: Di x z beta={tr['diz']:+.5f} (t={tr['diz_t']:+.2f}"
                      f"{real.stars(abs(tr['diz_t']))})  Di={tr['di']:+.5f}  firms={tr['G']:,}")
        cs = het[marg]["capint_split"]
        print(f"    capint split (med={cs['median']:.3f}): "
              f"high {fmt(cs['high']).strip()} | low {fmt(cs['low']).strip()}")

    with open(os.path.join(OUT_DIR, "capital_location_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    keep = ["gvkey", "recon_year", "industry", "intl_pre", "International"] + [
        c for c in panel.columns
        if c.startswith(("foreign_ppe_share", "foreign_capxs_share", "foreign_sales_share",
                         "total_geo_", "foreign_geo_", "has_foreign_"))
    ]
    panel[[c for c in keep if c in panel.columns]].to_csv(
        os.path.join(OUT_DIR, "capital_location_panel.csv"), index=False)
    print(f"\nWrote: {OUT_DIR}/capital_location_{{summary.json,panel.csv,event_study.csv}}")


if __name__ == "__main__":
    main()
