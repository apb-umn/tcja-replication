#!/usr/bin/env python3
"""
build_section4_same_sample.py

Baseline inputs, robustness and uncertainty for the Section 4 inversion.

BASELINE CONVENTION (matches Ellen's draft 3): the inversion runs on the
foreign-profit-share sample, where all three coefficients are estimated on the
same 876 firms -- with heterogeneous psi_i, coefficients from different
samples are projections on different populations and cannot be inverted
jointly. beta_Intl = -0.737 there, against -0.791 on the full panel (Table 7).

Table 7's rows come from two constructions that share the same sample filters
but build the treatment differently:

  A (first stage; build_firststage_event_studies.py): the pre-period mean of
    the International item includes zeros for firm-years that have
    reconciliation rows but no International line. beta_Intl and beta_ETR.
  B (substitution; build_substitution_real_outcomes.py): the pre-period mean
    is over years where the International line exists, zeros only for firms
    that never report one. beta_share, on the subsample reporting pidom/pifo.

This script reproduces each row with its own construction -- and asserts the
reproduction against the staged pooled_did_results_table.csv to 1e-6 -- then
supplies the robustness footnote and the uncertainty:

  1. the x_pre <= 0 restriction, per construction;
  2. the same-sample re-estimates of beta_Intl / beta_ETR on the
     foreign-profit-share sample, plus a DiD on log pretax income there;
  3. a firm-level block bootstrap (B = 999) under the baseline convention:
     the share-sample firms are resampled and all three coefficients computed
     on the resample, so the intervals are centred on the same-sample points;
  4. the Table 9 auxiliaries (foreign PP&E share, log R&D).

Writes  paper/Inputs/tables/section4_same_sample.csv        (key,value)
        paper/Inputs/tables/section4_bootstrap_draws.csv    (one row per draw)
Both feed section4/code/theory_calibration.{jl,py}.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import build_substitution_real_outcomes as real
from sic_industry import industry

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TBL_DIR = os.path.join(ROOT_DIR, "paper", "Inputs", "tables")
CAPLOC_JSON = os.path.join(ROOT_DIR, "data", "analysis", "output", "capital_location_summary.json")
RAROB_CSV = os.path.join(ROOT_DIR, "data", "analysis", "output", "slide_data", "realactivityrobust.csv")

B_BOOT = 999
SEED = 20260830


def build_firststage_panel(combined, long_df):
    """Construction A: mirrors build_firststage_event_studies.py exactly."""
    fy = (long_df.groupby(["gvkey", "recon_year", "group"])["cat_val"]
          .sum().reset_index(name="val"))
    fy["val_w"] = fy["val"].clip(-real.WIN_CH, real.WIN_CH)
    wide = fy.pivot_table(index=["gvkey", "recon_year"], columns="group",
                          values="val_w", fill_value=0).reset_index()
    wide.columns.name = None
    pre = wide[wide["recon_year"].isin(real.PRE_YEARS)]
    counts = pre.groupby("gvkey").size()
    elig = counts[counts >= 2].index
    intl_pre = (pre[pre["gvkey"].isin(elig)]
                .groupby("gvkey")["International"].mean().rename("intl_pre"))
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)

    panel = combined[combined["recon_year"].isin(real.EVENT_YEARS)].copy()
    panel["industry"] = panel["comp_sic"].apply(industry)
    panel = panel[panel["industry"] != "Finance"]
    panel = panel[panel["comp_pi"] > 1]
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(wide[["gvkey", "recon_year", "International"]],
                        on=["gvkey", "recon_year"], how="left")
    panel["International"] = panel["International"].fillna(0.0)
    fs = panel.groupby("gvkey")["recon_year"].apply(set)
    ok = fs[(fs.apply(lambda s: len(s & set(real.PRE_YEARS)) >= 2))
            & (fs.apply(lambda s: len(s & set(real.POST_YEARS)) >= 2))].index
    return panel[panel["gvkey"].isin(ok)].copy()


def did_slopes(sub, outcomes):
    """Pooled-DiD slopes, firm + industry-year FE, joint demeaning
    (mirrors real.run_pooled_did(..., ind_yr_fe=True))."""
    excl = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [yr for yr in real.EVENT_YEARS if yr not in excl]
    d = sub[sub["recon_year"].isin(years)].copy()
    d["_post"] = d["recon_year"].isin(real.POST_YEARS).astype(float)
    d["_Di"] = d["intl_pre"] * d["_post"]
    dm = real.demean_ind_yr(d, list(outcomes) + ["_Di"])
    x = dm["_Di"].to_numpy()
    denom = (x * x).sum()
    return {c: float((dm[c].to_numpy() * x).sum() / denom) for c in outcomes}


def resample(panel, indices, pick):
    """Cluster resample: gather each picked firm's rows, re-ID clusters."""
    idx = [indices[g] for g in pick]
    frame = panel.iloc[np.concatenate(idx)].copy()
    frame["gvkey"] = np.concatenate([np.full(len(ix), j) for j, ix in enumerate(idx)])
    return frame


def main():
    print("Section 4 robustness inputs, aligned with Table 7 row by row")
    print("=" * 74)

    combined, long_df, pi = real.load_data()
    panelA = build_firststage_panel(combined, long_df)
    intl_pre_B = real.build_channels_and_treatment(long_df)
    intl_yearly = real.build_intl_yearly(long_df)
    panelB = real.build_panel(combined, pi, intl_pre_B, intl_yearly)
    shareB = panelB.dropna(subset=["foreign_share"]).copy()
    shareB["log_pi"] = np.log(shareB["comp_pi"].where(shareB["comp_pi"] > 1))
    print(f"  A (first stage): {panelA['gvkey'].nunique():,} firms; "
          f"B (share sample): {shareB['gvkey'].nunique():,} firms")

    out = {}

    def run(sub, col, key, sc):
        r = real.run_pooled_did(sub, col, ind_yr_fe=True)
        out[key] = r["coef"] * sc
        out[key + "Se"] = r["se"] * sc
        out[key + "N"] = r["n"]
        out[key + "G"] = r["G"]
        print(f"  {key:<16s} {r['coef']*sc:+.4f} ({r['se']*sc:.4f}) n={r['n']:,} G={r['G']:,}")
        return r

    # ---- (1) baseline reproduction, asserted against the staged Table 7 ----
    print("\n[1] Baseline betas, each row on its own construction")
    ra = run(panelA, "International", "betaIntlFull", 1.0)
    rb = run(panelA, "etr_w", "betaEtrFull", 1.0)
    rc = run(shareB, "foreign_share", "betaShareFull", 100.0)
    staged = pd.read_csv(os.path.join(TBL_DIR, "pooled_did_results_table.csv")).set_index("row_key")
    for key, r, sc in (("international", ra, 1.0), ("etr", rb, 1.0),
                       ("foreign_profit_share", rc, 100.0)):
        want = staged.loc[key, "coef_table"]
        got = r["coef"] * sc
        assert abs(got - want) < 5e-4, (key, got, want)  # staged file is rounded to 6dp on the raw scale
    print("  == matches staged pooled_did_results_table.csv to its stored precision ==")

    # ---- (2) x_pre <= 0 restriction, per construction ----
    print("\n[2] Restricting to x_pre <= 0 (exposed) plus zero-exposure controls")
    negA = panelA[panelA["intl_pre"] <= 0]
    negB = shareB[shareB["intl_pre"] <= 0]
    fdrop = 1 - negA["gvkey"].nunique() / panelA["gvkey"].nunique()
    odrop = 1 - len(negA) / len(panelA)
    v = panelA.groupby("gvkey")["intl_pre"].first()
    vdrop = (v[v > 0] ** 2).sum() / (v ** 2).sum()
    out["negDropFirmsPct"] = 100 * fdrop
    out["negDropObsPct"] = 100 * odrop
    out["negDropTreatVarPct"] = 100 * vdrop
    print(f"  drops {100*fdrop:.1f}% of firms, {100*odrop:.1f}% of obs, "
          f"{100*vdrop:.1f}% of treatment variance (construction A)")
    run(negA, "International", "betaIntlNeg", 1.0)
    run(negA, "etr_w", "betaEtrNeg", 1.0)
    run(negB, "foreign_share", "betaShareNeg", 100.0)
    # and under the baseline (share-sample) convention:
    run(negB, "International", "betaIntlNegSame", 1.0)
    out["negDropFirmsSamePct"] = 100 * (1 - negB["gvkey"].nunique() / shareB["gvkey"].nunique())
    vs = shareB.groupby("gvkey")["intl_pre"].first()
    out["negDropTreatVarSamePct"] = 100 * (vs[vs > 0] ** 2).sum() / (vs ** 2).sum()
    print(f"  share-sample restriction drops {out['negDropFirmsSamePct']:.1f}% of firms, "
          f"{out['negDropTreatVarSamePct']:.1f}% of treatment variance")

    # ---- (3) same-sample estimates: the BASELINE for the inversion ----
    print("\n[3] Same-sample estimates (BASELINE for the inversion)")
    run(shareB, "International", "betaIntlSame", 1.0)
    run(shareB, "etr_w", "betaEtrSame", 1.0)
    run(shareB, "log_pi", "betaLogPiSame", 100.0)

    # ---- (4) bootstrap under the baseline convention (share sample) ----
    print(f"\n[4] Firm bootstrap (B={B_BOOT}) on the share sample: all three betas together")
    cols = ["International", "etr_w", "foreign_share"]
    pB = did_slopes(shareB, cols)
    assert abs(pB["International"] - out["betaIntlSame"]) < 1e-6
    assert abs(pB["foreign_share"] * 100 - out["betaShareFull"]) < 1e-6

    firms = np.array(sorted(set(shareB["gvkey"])))
    print(f"  resampling {len(firms):,} share-sample firms per draw")
    idxB = {g: ix for g, ix in shareB.groupby("gvkey").indices.items()}

    rng = np.random.default_rng(SEED)
    draws = []
    for b in range(B_BOOT):
        pick = rng.choice(firms, size=len(firms), replace=True)
        fb = resample(shareB, idxB, pick)
        bs = did_slopes(fb, cols)
        draws.append({"beta_intl": bs["International"], "beta_etr": bs["etr_w"],
                      "beta_share": bs["foreign_share"] * 100.0})
        if (b + 1) % 100 == 0:
            print(f"  {b+1}/{B_BOOT}")
    bdf = pd.DataFrame(draws)
    for c, key in [("beta_intl", "betaIntlSame"), ("beta_etr", "betaEtrSame"),
                   ("beta_share", "betaShareFull")]:
        lo, hi = bdf[c].quantile([0.025, 0.975])
        print(f"  {c:<11s} boot SE {bdf[c].std():.4f}  95% [{lo:+.4f}, {hi:+.4f}]"
              f"  (point {out[key]:+.4f})")

    # ---- (5) Table 9 auxiliaries ----
    print("\n[5] Table 9 auxiliary estimates")
    with open(CAPLOC_JSON) as f:
        cap = json.load(f)["foreign_ppe_share"]["outcome"]["ind_yr"]
    out["betaPpeShare"] = cap["coef"] * 100.0
    out["betaPpeShareSe"] = cap["se"] * 100.0
    out["betaPpeShareN"] = cap["n"]
    out["betaPpeShareG"] = cap["G"]
    print(f"  foreign PP&E share  {out['betaPpeShare']:+.4f} ({out['betaPpeShareSe']:.4f})")
    rarob = pd.read_csv(RAROB_CSV, header=None, names=["key", "label", "value", "type"])
    rd = rarob.set_index("key")["value"]
    out["betaLogRd"] = float(rd["raRobLogRd"]) * 100.0
    print(f"  log R&D             {out['betaLogRd']:+.4f} (insignificant)")

    # ---- write ----
    os.makedirs(TBL_DIR, exist_ok=True)
    p1 = os.path.join(TBL_DIR, "section4_same_sample.csv")
    with open(p1, "w") as f:
        f.write("key,value\n")
        for k, v in out.items():
            f.write(f"{k},{float(v):.6f}\n")
    p2 = os.path.join(TBL_DIR, "section4_bootstrap_draws.csv")
    bdf.to_csv(p2, index=False, float_format="%.6f")
    print(f"\nwrote {p1} ({len(out)} keys)")
    print(f"wrote {p2} ({len(bdf)} draws)")


if __name__ == "__main__":
    main()
