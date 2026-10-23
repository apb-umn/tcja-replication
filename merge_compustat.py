#!/usr/bin/env python3
"""
merge_compustat.py -- refill the blank Compustat columns of the shareable panel.

Usage (from the package root):
    python3 merge_compustat.py [--compustat data/original/compustat_data.csv]
                               [--panel data/tax_reconciliation_panel_shareable.csv]
                               [--out data/analysis/output/tax_reconciliation_panel_base.csv]

Input  : a Compustat Fundamentals Annual extract (WRDS, comp.funda) as CSV with
         at least the variables listed in REQUIRED_VARS below (see
         data/original/README.md for the WRDS query).
Output : the analysis panel every script in analysis/ and paper/ reads.

The merge reproduces analysis/build_analysis_base.py::build_compustat_panel:
  1. one Compustat row per (gvkey, fyear), preferring INDL > FS, STD, consolidated,
     rows with non-missing sale/ni/pi, then the latest datadate;
  2. left-join onto the panel on (gvkey, recon_year = fyear);
  3. has_compustat_year_match = a Compustat row was found.
"""
import argparse, sys
from pathlib import Path
import pandas as pd

COMP_VARS = ["pi", "pidom", "pifo", "ni", "txt", "txpd", "sale", "cogs", "xsga", "xrd", "emp",
             "at", "capx", "ppent", "intan", "gdwl", "aqc", "che", "dltt", "dlc", "ceq", "dvc",
             "prstkc", "mkvalt", "prcc_f", "csho", "txndb", "txndba", "txndbl", "txdb", "txditc",
             "tlcf", "txfed", "txfo", "oancf", "sstk", "sic", "fic"]
KEY_VARS = ["gvkey", "fyear", "datadate", "indfmt", "datafmt", "consol"]
REQUIRED_VARS = KEY_VARS + COMP_VARS
STRING_VARS = {"comp_datadate", "comp_fic"}


def normalize_year(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).replace({"": pd.NA, "nan": pd.NA})


def gvkey_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True), errors="coerce").astype("Int64")


def build_compustat_panel(path: Path) -> pd.DataFrame:
    comp = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = [v for v in REQUIRED_VARS if v not in comp.columns]
    if missing:
        sys.exit(f"ERROR: {path} lacks required Compustat variables: {missing}")
    comp["recon_year"] = normalize_year(comp["fyear"])
    comp["gvkey"] = gvkey_int(comp["gvkey"])
    comp = comp.dropna(subset=["gvkey", "recon_year"])
    rank = {
        "_r_indfmt": comp["indfmt"].eq("INDL").astype(int),
        "_r_datafmt": comp["datafmt"].eq("STD").astype(int),
        "_r_consol": comp["consol"].eq("C").astype(int),
        "_r_sale": comp["sale"].ne("").astype(int),
        "_r_ni": comp["ni"].ne("").astype(int),
        "_r_pi": comp["pi"].ne("").astype(int),
    }
    for k, v in rank.items():
        comp[k] = v
    comp = comp.sort_values(
        ["gvkey", "recon_year"] + list(rank) + ["datadate"],
        ascending=[True, True] + [False] * len(rank) + [False], kind="stable")
    comp = comp.drop_duplicates(["gvkey", "recon_year"], keep="first")
    keep = ["gvkey", "recon_year", "datadate", "fyear"] + COMP_VARS
    comp = comp[keep].rename(columns={c: f"comp_{c}" for c in keep if c not in ("gvkey", "recon_year")})
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compustat", default="data/original/compustat_data.csv")
    ap.add_argument("--panel", default="data/tax_reconciliation_panel_shareable.csv")
    ap.add_argument("--out", default="data/analysis/output/tax_reconciliation_panel_base.csv")
    ap.add_argument("--ignore-shipped-flag", action="store_true", help="fill every row your extract matches, even rows the authors left unmatched")
    a = ap.parse_args()
    if not Path(a.compustat).exists():
        sys.exit(f"ERROR: {a.compustat} not found. Download it from WRDS first (see data/original/README.md).")

    if not Path(a.panel).exists() and Path(a.panel + ".gz").exists():
        a.panel = a.panel + ".gz"   # the git repository ships the panel gzipped
    print(f"Reading panel {a.panel} ...")
    panel = pd.read_csv(a.panel, dtype=str, keep_default_na=False, low_memory=False)
    comp_cols = [c for c in panel.columns if c.startswith("comp_")]
    shipped_flag = panel["has_compustat_year_match"].str.lower().eq("true")
    panel = panel.drop(columns=comp_cols)
    panel["gvkey"] = gvkey_int(panel["gvkey"])
    panel["recon_year"] = normalize_year(panel["recon_year"])

    print(f"Reading Compustat {a.compustat} ...")
    comp = build_compustat_panel(Path(a.compustat))
    print(f"  {len(comp):,} unique (gvkey, fyear) Compustat rows")

    merged = panel.merge(comp, on=["gvkey", "recon_year"], how="left", validate="many_to_one")
    found = merged["comp_datadate"].notna() & merged["comp_datadate"].ne("")
    # The authors' sample is defined by the shipped has_compustat_year_match flag.
    # Rows the authors could not match are kept unmatched even if your extract has a
    # record (with the authors' extract this is a single empty Compustat shell record);
    # rows the authors matched but your extract lacks are reported below.
    if a.ignore_shipped_flag:
        merged["has_compustat_year_match"] = found
    else:
        extra = found & ~shipped_flag.values
        if extra.any():
            print(f"  {int(extra.sum()):,} row(s) have a Compustat record in your extract but were unmatched in the authors' panel; "
                  "kept unmatched to preserve the authors' sample (use --ignore-shipped-flag to fill them)")
            comp_fill = [c for c in merged.columns if c.startswith("comp_")]
            merged.loc[extra, comp_fill] = ""
        merged["has_compustat_year_match"] = found & shipped_flag.values
    for c in [c for c in merged.columns if c.startswith("comp_") and c not in STRING_VARS]:
        merged[c] = pd.to_numeric(merged[c].replace({"": pd.NA}), errors="coerce")
    merged["gvkey"] = merged["gvkey"].astype("int64")
    # restore the original column order (Compustat block after has_compustat_year_match)
    order = [c for c in panel.columns if c != "has_compustat_year_match"]
    i = order.index("israte") if "israte" in order else len(order)
    order = order[:order.index("statutory_tax")] + ["has_compustat_year_match"] + [c for c in merged.columns if c.startswith("comp_")] + ["filing_compustat_fyear"] + order[order.index("statutory_tax"):]
    order = [c for c in dict.fromkeys(order) if c in merged.columns]
    merged = merged[order]

    n_match = int(merged["has_compustat_year_match"].sum())
    n_ship = int(shipped_flag.sum())
    print(f"  rows: {len(merged):,}; rows with a Compustat match: {n_match:,} (authors' panel: {n_ship:,})")
    if n_match < n_ship:
        print(f"  NOTE: {n_ship - n_match:,} rows matched in the authors' panel but not in your extract (their Compustat columns stay blank). "
              "Check that your WRDS query covers fiscal years 1995-2024, all formats (INDL and FS), and is not restricted to USA.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
