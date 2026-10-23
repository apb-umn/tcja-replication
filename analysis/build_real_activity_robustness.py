#!/usr/bin/env python3
"""
build_real_activity_robustness.py

Production consolidation of the real-activity robustness digs (exploratory
scripts: real_activity_spec_robustness.py, real_activity_denominator_test.py,
real_activity_balance_sheet.py, real_activity_repatriation_ppe.py). Builds the
real-activity panel once and computes the numbers behind appendix Section 10
(A.38-A.40), writing data/analysis/output/real_activity_robustness_summary.json
for build_slide_data.py::emit_real_activity_robustness.

All estimates: pooled DiD, intl_pre x post, firm + industry x year FE, SE
clustered by firm, window 2011-2023, 2017-18 dropped. Sign: beta<0 => outcome
RISES for exposed firms (intl_pre<0); beta>0 => FALLS.

Run:  python3 build_real_activity_robustness.py
"""
import json
import os
import numpy as np
import pandas as pd

import build_real_activity_dynamics as ra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
COMPUSTAT = os.path.join(ROOT, "data", "original", "compustat_data.csv")
OUT = os.path.join(ROOT, "data", "analysis", "output", "real_activity_robustness_summary.json")
SPEC = "ind_yr"
PRE = list(range(2013, 2017))


def winsor(s, lo=0.01, hi=0.99):
    a, b = s.quantile([lo, hi]); return s.clip(a, b)


def standardize(s):
    w = s.clip(*s.quantile([0.01, 0.99])); sd = w.std()
    return (w - w.mean()) / sd if sd else w * 0.0


def did(panel, col):
    r = ra.run_pooled_did(panel, col, treat_col="intl_pre", spec=SPEC)
    t = r["coef"] / r["se"] if r["se"] else float("nan")
    return {"coef": r["coef"], "se": r["se"], "t": t, "n": r["n"], "G": r["G"],
            "sig": ra.stars(abs(t)) if r["se"] else ""}


def main():
    combined, long_df, extra = ra.load_data()
    intl_pre = ra.build_intl_pre(long_df)
    panel = ra.build_panel(combined, extra, intl_pre)

    cs = pd.read_csv(COMPUSTAT, usecols=["gvkey", "fyear", "che", "pidom", "pifo",
                                         "sppe", "aqc", "dp", "ppent", "at", "capx",
                                         "txpd", "prstkc", "dltt", "dlc", "dvc"],
                     low_memory=False).rename(columns={"fyear": "recon_year"})
    cs = cs.drop_duplicates(["gvkey", "recon_year"])

    pre = cs[cs["recon_year"].isin(PRE)].copy()
    pre["cash"] = np.where(pre["at"] > 0, pre["che"] / pre["at"], np.nan)
    zpre = pre.groupby("gvkey")["cash"].mean().rename("cash_pre")
    panel = panel.merge(zpre, on="gvkey", how="left")
    panel = panel.merge(cs[["gvkey", "recon_year", "che", "ppent", "sppe", "aqc",
                            "dp", "txpd", "prstkc", "dltt", "dlc", "dvc"]],
                        on=["gvkey", "recon_year"], how="left")

    # lags
    lag = panel[["gvkey", "recon_year", "comp_xrd", "ppent"]].copy()
    lag["recon_year"] += 1
    lag = lag.rename(columns={"comp_xrd": "lag_xrd", "ppent": "lag_ppent"})
    panel = panel.merge(lag, on=["gvkey", "recon_year"], how="left")

    L = lambda s: pd.Series(np.where(s > 0, np.log(s.where(s > 0)), np.nan), index=panel.index)
    DL = lambda s, sl: pd.Series(np.where((s > 0) & (sl > 0), np.log(s.where(s > 0) / sl.where(sl > 0)), np.nan), index=panel.index).clip(-1, 1)
    at = panel["lag_at"]; okat = at > 1
    rat = lambda x: winsor(pd.Series(np.where(okat & x.notna(), x / at, np.nan), index=panel.index))

    summary = {}

    # ── (A.38) outcome-spec robustness: capex / R&D / total inv ───────────────
    panel["totinv_lvl"] = np.where(panel["comp_capx"].notna() & panel["comp_xrd"].notna(),
                                   panel["comp_capx"] + panel["comp_xrd"], np.nan)
    spec = {}
    for nm, lvl, base in [("capex", "comp_capx", "capex_int"),
                          ("rd", "comp_xrd", "rd_int"),
                          ("totalinv", "totinv_lvl", "total_inv")]:
        x = panel[lvl]
        panel[f"_{nm}_log"] = L(x)
        # dlog: need lag of the level; build for capex/rd/totinv via merge on the level series
        lg = panel[["gvkey", "recon_year"]].copy(); lg["recon_year"] += 1; lg[f"_{nm}_lvl"] = x.values
        lg = lg.rename(columns={f"_{nm}_lvl": f"_{nm}_laglvl"})
        panel = panel.merge(lg, on=["gvkey", "recon_year"], how="left")
        panel[f"_{nm}_dlog"] = DL(x, panel[f"_{nm}_laglvl"])
        panel[f"_{nm}_ppe"] = winsor(pd.Series(np.where((panel["lag_ppent"] > 1) & x.notna(),
                                                        x / panel["lag_ppent"], np.nan), index=panel.index))
        spec[nm] = {
            "ratio_at": did(panel, base),
            "log":      did(panel, f"_{nm}_log"),
            "dlog":     did(panel, f"_{nm}_dlog"),
            "ratio_ppe": did(panel, f"_{nm}_ppe"),
        }
    summary["spec"] = spec

    # ── (A.39) denominator test + exact log-ratio decomposition ───────────────
    panel["log_at"] = L(panel["comp_at"]); panel["log_lagat"] = L(panel["lag_at"])
    panel["log_ppe"] = L(panel["ppent"]); panel["log_capx"] = L(panel["comp_capx"]); panel["log_rd"] = L(panel["comp_xrd"])
    common = panel["comp_xrd"].gt(0) & panel["lag_at"].gt(1)
    panel["logratio_rd"] = np.where(common, panel["log_rd"] - panel["log_lagat"], np.nan)
    panel["log_rd_c"]    = np.where(common, panel["log_rd"], np.nan)
    panel["log_lagat_c"] = np.where(common, panel["log_lagat"], np.nan)
    den = {
        "log_at": did(panel, "log_at"), "log_ppe": did(panel, "log_ppe"),
        "log_capx": did(panel, "log_capx"), "log_rd": did(panel, "log_rd"),
        "ratio": did(panel, "logratio_rd"), "num": did(panel, "log_rd_c"),
        "den": did(panel, "log_lagat_c"),
    }
    dnum, dden = den["num"]["coef"], den["den"]["coef"]
    den["denominator_share_pct"] = round(-dden / (dnum - dden) * 100) if (dnum - dden) else None
    summary["denominator"] = den

    # ── (A.40) balance sheet + repatriation split + PP&E flows ────────────────
    panel["cash_int"] = rat(panel["che"]); panel["log_che"] = L(panel["che"])
    panel["txpd_int"] = rat(panel["txpd"]); panel["buyback_int"] = rat(panel["prstkc"])
    panel["debt_int"] = rat(panel["dltt"].fillna(0) + panel["dlc"].fillna(0))
    panel["div_int"]  = rat(panel["dvc"])
    panel["sppe_int"] = rat(panel["sppe"]); panel["aqc_int"] = rat(panel["aqc"]); panel["dp_int"] = rat(panel["dp"])
    bs = {k: did(panel, c) for k, c in [
        ("cash_int", "cash_int"), ("log_che", "log_che"), ("txpd_int", "txpd_int"),
        ("buyback_int", "buyback_int"), ("debt_int", "debt_int"), ("div_int", "div_int"),
        ("sppe_int", "sppe_int"), ("aqc_int", "aqc_int"), ("dp_int", "dp_int")]}
    # repatriation split: asset contraction high vs low pre-cash
    fz = panel.dropna(subset=["log_at"]).drop_duplicates("gvkey").set_index("gvkey")["cash_pre"]
    med = float(fz.median()); hi = set(fz[fz > med].index)
    bs["assetcontraction_hicash"] = did(panel[panel["gvkey"].isin(hi)], "log_at")
    bs["assetcontraction_locash"] = did(panel[~panel["gvkey"].isin(hi)], "log_at")
    bs["cashdrawdown_hicash"] = did(panel[panel["gvkey"].isin(hi)], "log_che")
    bs["cashdrawdown_locash"] = did(panel[~panel["gvkey"].isin(hi)], "log_che")
    summary["balance_sheet"] = bs

    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # console summary
    def fmt(d): return f"beta={d['coef']:+.5f} t={d['t']:+.2f}{d['sig']}"
    print("\n=== A.38 outcome-spec robustness ===")
    for nm in ("capex", "rd", "totalinv"):
        print(f"  {nm:9s} " + " | ".join(f"{k}: {fmt(spec[nm][k])}" for k in ("ratio_at", "log", "dlog", "ratio_ppe")))
    print("\n=== A.39 denominator ===")
    for k in ("log_capx", "log_rd", "log_at", "log_ppe", "ratio", "num", "den"):
        print(f"  {k:10s} {fmt(den[k])}")
    print(f"  denominator share of R&D log-ratio response: {den['denominator_share_pct']}%")
    print("\n=== A.40 balance sheet ===")
    for k in ("log_che", "cash_int", "txpd_int", "buyback_int", "debt_int", "div_int",
              "sppe_int", "aqc_int", "dp_int", "assetcontraction_hicash", "assetcontraction_locash"):
        print(f"  {k:24s} {fmt(bs[k])}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
