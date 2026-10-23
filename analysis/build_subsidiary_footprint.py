#!/usr/bin/env python3
"""
build_subsidiary_footprint.py

Construct annual subsidiary-footprint variables from the WRDS/Compustat
subsidiary CSV export.

The raw WRDS file includes transaction filings such as S-4. For a comparable
firm-year panel, this script keeps annual filings only:

    10-K, 10-K/A, 10-KT, 10-KT/A

Rows are subsidiary-level Exhibit 21 disclosures. We deduplicate subsidiary
rows within a filing, select one annual filing per gvkey x fiscal year, and
then construct total, foreign, tax-haven/conduit, and country-specific
subsidiary counts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUT_DIR = ROOT_DIR / "data" / "analysis" / "output"  # [replication package] was SCRIPT_DIR / "output"

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}

# Main headline definition: classic offshore havens plus conduit / treaty hubs.
CLASSIC_HAVENS = {
    "AD", "AI", "AG", "AW", "BS", "BB", "BZ", "BM", "VG", "KY",
    "CK", "CW", "DM", "GI", "GD", "GG", "IM", "JE", "LI", "MH",
    "MC", "MS", "NR", "NU", "PA", "KN", "LC", "VC", "WS", "SC",
    "TC", "VU", "VI",
}
CONDUIT_HUBS = {"IE", "NL", "LU", "CH", "SG", "HK", "MT", "CY", "MU"}
HAVEN_OR_CONDUIT = CLASSIC_HAVENS | CONDUIT_HUBS

FOCUS_COUNTRIES = {
    "IE": "ireland",
    "NL": "netherlands",
    "LU": "luxembourg",
    "KY": "cayman",
    "BM": "bermuda",
    "SG": "singapore",
    "HK": "hong_kong",
    "CH": "switzerland",
    "VG": "british_virgin_islands",
    "MU": "mauritius",
    "PA": "panama",
    "GB": "united_kingdom",
    "CA": "canada",
    "CN": "china",
    "DE": "germany",
    "MX": "mexico",
}

USECOLS = [
    "cik",
    "gvkey",
    "accession",
    "coname",
    "fdate",
    "rdate",
    "secpdate",
    "subsidiary_name",
    "clean_company",
    "location",
    "country_code",
    "subdiv_code",
    "form",
    "extype",
    "ex21url",
    "formurl",
]


def default_input_files() -> list[Path]:
    patterns = [
        "subsidiaries_pre_tcja.csv",
        "subsidiaries_post_tcja.csv",
        "subsidiaries*_tcja*.csv",
        "subsidiaries_202*.csv",
    ]
    files: list[Path] = []
    seen = set()
    for pattern in patterns:
        for match in glob.glob(str(ROOT_DIR / pattern)):
            path = Path(match).resolve()
            if path.exists() and path not in seen:
                files.append(path)
                seen.add(path)
    return sorted(files)


def zfill_gvkey(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def source_period(path: Path) -> str:
    name = path.name.lower()
    if "pre" in name and "tcja" in name:
        return "pre_tcja_file"
    if "post" in name and "tcja" in name:
        return "post_tcja_file"
    return "unspecified"


def load_annual_rows(paths: list[Path], chunksize: int, keep_gvkey_zero: bool) -> tuple[pd.DataFrame, dict]:
    frames = []
    input_stats = {}

    for path in paths:
        raw_rows = 0
        annual_rows = 0
        kept_rows = 0
        form_counts = Counter()

        print(f"Reading {path.name}...")
        for chunk in pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=chunksize, low_memory=False):
            raw_rows += len(chunk)
            chunk["form"] = chunk["form"].fillna("").str.strip().str.upper()
            form_counts.update(chunk["form"].tolist())
            chunk = chunk[chunk["form"].isin(ANNUAL_FORMS)].copy()
            annual_rows += len(chunk)
            if chunk.empty:
                continue

            chunk["gvkey"] = zfill_gvkey(chunk["gvkey"])
            if not keep_gvkey_zero:
                chunk = chunk[chunk["gvkey"] != "000000"].copy()
            chunk = chunk[chunk["gvkey"].ne("") & chunk["accession"].notna()].copy()
            if chunk.empty:
                continue

            chunk["fdate_dt"] = pd.to_datetime(chunk["fdate"], errors="coerce")
            chunk["rdate_dt"] = pd.to_datetime(chunk["rdate"], errors="coerce")
            chunk["secpdate_dt"] = pd.to_datetime(chunk["secpdate"], errors="coerce")
            chunk["report_year"] = chunk["rdate_dt"].dt.year.astype("Int64")
            chunk = chunk[chunk["report_year"].notna()].copy()
            if chunk.empty:
                continue

            chunk["cik"] = chunk["cik"].fillna("").astype(str).str.strip().str.zfill(10)
            chunk["country_code"] = chunk["country_code"].fillna("").str.strip().str.upper()
            chunk["subdiv_code"] = chunk["subdiv_code"].fillna("").str.strip().str.upper()
            chunk["location"] = normalize_text(chunk["location"])
            chunk["subsidiary_name"] = normalize_text(chunk["subsidiary_name"])
            chunk["clean_company"] = normalize_text(chunk["clean_company"])
            chunk["subsidiary_key"] = chunk["clean_company"].where(
                chunk["clean_company"].ne(""), chunk["subsidiary_name"]
            )
            chunk["source_file"] = path.name
            chunk["source_period"] = source_period(path)
            frames.append(chunk)
            kept_rows += len(chunk)

        input_stats[path.name] = {
            "raw_rows": raw_rows,
            "annual_rows_before_gvkey_filter": annual_rows,
            "kept_annual_rows": kept_rows,
            "top_forms_raw": form_counts.most_common(20),
            "source_period": source_period(path),
        }

    if not frames:
        raise FileNotFoundError("No annual subsidiary rows found in input files.")

    annual = pd.concat(frames, ignore_index=True)
    annual["report_year"] = annual["report_year"].astype(int)
    return annual, input_stats


def dedupe_rows(annual: pd.DataFrame) -> pd.DataFrame:
    keys = ["accession", "subsidiary_key", "location", "country_code", "subdiv_code"]
    before = len(annual)
    deduped = annual.drop_duplicates(subset=keys).copy()
    print(f"Deduped annual subsidiary rows: {before:,} -> {len(deduped):,}")
    return deduped


def select_latest_annual_filing(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filing_cols = [
        "gvkey",
        "report_year",
        "cik",
        "coname",
        "accession",
        "form",
        "fdate",
        "rdate",
        "secpdate",
        "fdate_dt",
        "rdate_dt",
        "secpdate_dt",
        "extype",
        "ex21url",
        "formurl",
        "source_file",
        "source_period",
    ]
    filings = rows[filing_cols].drop_duplicates().copy()
    filings["form_rank"] = filings["form"].map({"10-K": 1, "10-KT": 2, "10-K/A": 3, "10-KT/A": 4}).fillna(0)
    filings["fdate_sort"] = filings["fdate_dt"].fillna(pd.Timestamp("1900-01-01"))
    filings["secpdate_sort"] = filings["secpdate_dt"].fillna(pd.Timestamp("1900-01-01"))
    filings = filings.sort_values(
        ["gvkey", "report_year", "fdate_sort", "secpdate_sort", "form_rank", "accession"]
    )
    filings["n_accessions_in_parent_year"] = filings.groupby(["gvkey", "report_year"])["accession"].transform("nunique")
    selected = filings.drop_duplicates(subset=["gvkey", "report_year"], keep="last").copy()
    selected["selected_annual_filing"] = 1

    selected_keys = selected[["gvkey", "report_year", "accession"]]
    selected_rows = rows.merge(selected_keys, on=["gvkey", "report_year", "accession"], how="inner")
    return selected_rows, selected


def join_unique(values: pd.Series) -> str:
    vals = sorted({str(v) for v in values.dropna() if str(v).strip()})
    return "|".join(vals)


def build_country_long(rows: pd.DataFrame) -> pd.DataFrame:
    tmp = rows.copy()
    tmp["country_code_clean"] = tmp["country_code"].where(tmp["country_code"].ne(""), "<missing>")
    tmp["is_us"] = tmp["country_code_clean"].eq("US")
    tmp["is_foreign"] = ~tmp["country_code_clean"].isin(["US", "<missing>"])
    tmp["is_classic_haven"] = tmp["country_code_clean"].isin(CLASSIC_HAVENS)
    tmp["is_conduit_hub"] = tmp["country_code_clean"].isin(CONDUIT_HUBS)
    tmp["is_haven_or_conduit"] = tmp["country_code_clean"].isin(HAVEN_OR_CONDUIT)

    country = tmp.groupby(["gvkey", "report_year", "country_code_clean"]).agg(
        sub_count=("subsidiary_key", "size"),
        location_names=("location", join_unique),
        is_us=("is_us", "max"),
        is_foreign=("is_foreign", "max"),
        is_classic_haven=("is_classic_haven", "max"),
        is_conduit_hub=("is_conduit_hub", "max"),
        is_haven_or_conduit=("is_haven_or_conduit", "max"),
    ).reset_index()
    return country.rename(columns={"country_code_clean": "country_code"})


def build_parent_year_panel(rows: pd.DataFrame, selected_filings: pd.DataFrame) -> pd.DataFrame:
    tmp = rows.copy()
    tmp["country_code_clean"] = tmp["country_code"].where(tmp["country_code"].ne(""), "<missing>")
    tmp["is_us_sub"] = tmp["country_code_clean"].eq("US")
    tmp["is_missing_country_sub"] = tmp["country_code_clean"].eq("<missing>")
    tmp["is_foreign_sub"] = ~tmp["country_code_clean"].isin(["US", "<missing>"])
    tmp["is_classic_haven_sub"] = tmp["country_code_clean"].isin(CLASSIC_HAVENS)
    tmp["is_conduit_sub"] = tmp["country_code_clean"].isin(CONDUIT_HUBS)
    tmp["is_haven_sub"] = tmp["country_code_clean"].isin(HAVEN_OR_CONDUIT)

    panel = tmp.groupby(["gvkey", "report_year"]).agg(
        total_subs=("subsidiary_key", "size"),
        us_subs=("is_us_sub", "sum"),
        foreign_subs=("is_foreign_sub", "sum"),
        missing_country_subs=("is_missing_country_sub", "sum"),
        classic_haven_subs=("is_classic_haven_sub", "sum"),
        conduit_subs=("is_conduit_sub", "sum"),
        haven_subs=("is_haven_sub", "sum"),
        country_list=("country_code_clean", join_unique),
        foreign_country_list=("country_code_clean", lambda x: join_unique(x[~x.isin(["US", "<missing>"])])),
        haven_country_list=("country_code_clean", lambda x: join_unique(x[x.isin(HAVEN_OR_CONDUIT)])),
        n_subsidiary_names=("subsidiary_key", "nunique"),
    ).reset_index()

    country_sets = tmp.groupby(["gvkey", "report_year"]).agg(
        country_count=("country_code_clean", lambda x: len({v for v in x if v != "<missing>"})),
        foreign_country_count=("country_code_clean", lambda x: len({v for v in x if v not in {"US", "<missing>"}})),
        classic_haven_country_count=("country_code_clean", lambda x: len(set(x) & CLASSIC_HAVENS)),
        conduit_country_count=("country_code_clean", lambda x: len(set(x) & CONDUIT_HUBS)),
        haven_country_count=("country_code_clean", lambda x: len(set(x) & HAVEN_OR_CONDUIT)),
    ).reset_index()
    panel = panel.merge(country_sets, on=["gvkey", "report_year"], how="left")

    for code, label in FOCUS_COUNTRIES.items():
        mask_col = f"country_{label}_subs"
        tmp[mask_col] = tmp["country_code_clean"].eq(code)
        counts = tmp.groupby(["gvkey", "report_year"])[mask_col].sum().reset_index()
        panel = panel.merge(counts, on=["gvkey", "report_year"], how="left")
        panel[mask_col] = panel[mask_col].fillna(0).astype(int)
        panel[f"has_country_{label}"] = panel[mask_col].gt(0).astype(int)

    panel["foreign_sub_share"] = np.where(panel["total_subs"].gt(0), panel["foreign_subs"] / panel["total_subs"], np.nan)
    panel["haven_sub_share"] = np.where(panel["total_subs"].gt(0), panel["haven_subs"] / panel["total_subs"], np.nan)
    panel["haven_foreign_sub_share"] = np.where(
        panel["foreign_subs"].gt(0), panel["haven_subs"] / panel["foreign_subs"], np.nan
    )
    panel["classic_haven_sub_share"] = np.where(
        panel["total_subs"].gt(0), panel["classic_haven_subs"] / panel["total_subs"], np.nan
    )
    panel["conduit_sub_share"] = np.where(panel["total_subs"].gt(0), panel["conduit_subs"] / panel["total_subs"], np.nan)

    for col in [
        "total_subs",
        "foreign_subs",
        "haven_subs",
        "classic_haven_subs",
        "conduit_subs",
        "foreign_country_count",
        "haven_country_count",
    ]:
        panel[f"log1p_{col}"] = np.log1p(panel[col])

    panel["has_foreign_sub"] = panel["foreign_subs"].gt(0).astype(int)
    panel["has_haven_sub"] = panel["haven_subs"].gt(0).astype(int)
    panel["has_classic_haven_sub"] = panel["classic_haven_subs"].gt(0).astype(int)
    panel["has_conduit_sub"] = panel["conduit_subs"].gt(0).astype(int)

    meta_cols = [
        "gvkey",
        "report_year",
        "cik",
        "coname",
        "accession",
        "form",
        "fdate",
        "rdate",
        "secpdate",
        "extype",
        "ex21url",
        "formurl",
        "source_file",
        "source_period",
        "n_accessions_in_parent_year",
    ]
    selected_meta = selected_filings[meta_cols].drop_duplicates(subset=["gvkey", "report_year"])
    panel = panel.merge(selected_meta, on=["gvkey", "report_year"], how="left")

    front = [
        "gvkey",
        "report_year",
        "cik",
        "coname",
        "accession",
        "form",
        "fdate",
        "rdate",
        "source_file",
        "source_period",
        "n_accessions_in_parent_year",
    ]
    rest = [c for c in panel.columns if c not in front]
    return panel[front + rest].sort_values(["gvkey", "report_year"]).reset_index(drop=True)


def build_summary(
    paths: list[Path],
    input_stats: dict,
    annual_rows: pd.DataFrame,
    deduped_rows: pd.DataFrame,
    selected_rows: pd.DataFrame,
    selected_filings: pd.DataFrame,
    panel: pd.DataFrame,
    country_long: pd.DataFrame,
) -> dict:
    top_countries = (
        country_long[country_long["country_code"].ne("<missing>")]
        .groupby("country_code")["sub_count"]
        .sum()
        .sort_values(ascending=False)
        .head(50)
    )
    top_foreign = (
        country_long[~country_long["country_code"].isin(["US", "<missing>"])]
        .groupby("country_code")["sub_count"]
        .sum()
        .sort_values(ascending=False)
        .head(50)
    )
    top_havens = (
        country_long[country_long["country_code"].isin(HAVEN_OR_CONDUIT)]
        .groupby("country_code")["sub_count"]
        .sum()
        .sort_values(ascending=False)
        .head(50)
    )

    def describe(series: pd.Series) -> dict:
        s = series.dropna().astype(float)
        if s.empty:
            return {}
        return {
            "n": int(s.size),
            "mean": float(s.mean()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }

    return {
        "input_files": [str(p) for p in paths],
        "annual_forms_kept": sorted(ANNUAL_FORMS),
        "dedupe_key": ["accession", "subsidiary_key", "location", "country_code", "subdiv_code"],
        "filing_selection_rule": "latest annual filing by fdate, secpdate, form rank, accession within gvkey x report_year",
        "haven_or_conduit_codes": sorted(HAVEN_OR_CONDUIT),
        "classic_haven_codes": sorted(CLASSIC_HAVENS),
        "conduit_hub_codes": sorted(CONDUIT_HUBS),
        "focus_country_codes": FOCUS_COUNTRIES,
        "input_stats": input_stats,
        "rows": {
            "annual_rows_after_gvkey_filter": int(len(annual_rows)),
            "deduped_annual_rows": int(len(deduped_rows)),
            "selected_subsidiary_rows": int(len(selected_rows)),
            "selected_filings": int(len(selected_filings)),
            "parent_year_rows": int(len(panel)),
        },
        "coverage": {
            "unique_gvkeys": int(panel["gvkey"].nunique()),
            "report_year_min": int(panel["report_year"].min()) if not panel.empty else None,
            "report_year_max": int(panel["report_year"].max()) if not panel.empty else None,
            "parent_years_with_foreign_subs": int(panel["has_foreign_sub"].sum()),
            "parent_years_with_foreign_subs_frac": float(panel["has_foreign_sub"].mean()) if not panel.empty else None,
            "parent_years_with_haven_subs": int(panel["has_haven_sub"].sum()),
            "parent_years_with_haven_subs_frac": float(panel["has_haven_sub"].mean()) if not panel.empty else None,
            "parent_years_multiple_annual_accessions": int(panel["n_accessions_in_parent_year"].gt(1).sum()),
        },
        "report_year_counts": panel["report_year"].value_counts().sort_index().astype(int).to_dict(),
        "distributions": {
            "total_subs": describe(panel["total_subs"]),
            "foreign_subs": describe(panel["foreign_subs"]),
            "foreign_sub_share": describe(panel["foreign_sub_share"]),
            "foreign_country_count": describe(panel["foreign_country_count"]),
            "haven_subs": describe(panel["haven_subs"]),
            "haven_country_count": describe(panel["haven_country_count"]),
        },
        "top_country_counts": top_countries.astype(int).to_dict(),
        "top_foreign_country_counts": top_foreign.astype(int).to_dict(),
        "top_haven_or_conduit_country_counts": top_havens.astype(int).to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Subsidiary CSV file(s). Defaults to subsidiaries*_tcja*.csv.")
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--keep-gvkey-zero", action="store_true", help="Keep gvkey=000000 rows. Default drops them.")
    parser.add_argument("--output-prefix", default="subsidiary_footprint", help="Prefix for output files in analysis/output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [p.resolve() for p in args.inputs] if args.inputs else default_input_files()
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No subsidiary CSV files found. Expected subsidiaries_pre_tcja.csv and/or subsidiaries_post_tcja.csv.")

    OUT_DIR.mkdir(exist_ok=True)

    annual_rows, input_stats = load_annual_rows(paths, args.chunksize, args.keep_gvkey_zero)
    deduped_rows = dedupe_rows(annual_rows)
    selected_rows, selected_filings = select_latest_annual_filing(deduped_rows)
    country_long = build_country_long(selected_rows)
    panel = build_parent_year_panel(selected_rows, selected_filings)

    summary = build_summary(paths, input_stats, annual_rows, deduped_rows, selected_rows, selected_filings, panel, country_long)

    prefix = args.output_prefix
    panel_path = OUT_DIR / f"{prefix}_panel.csv"
    country_path = OUT_DIR / f"{prefix}_country_counts_long.csv"
    filings_path = OUT_DIR / f"{prefix}_selected_filings.csv"
    summary_path = OUT_DIR / f"{prefix}_summary.json"

    panel.to_csv(panel_path, index=False)
    country_long.sort_values(["gvkey", "report_year", "country_code"]).to_csv(country_path, index=False)
    selected_filings.sort_values(["gvkey", "report_year", "fdate_sort", "accession"]).to_csv(filings_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "inputs": [p.name for p in paths],
        "parent_year_rows": summary["rows"]["parent_year_rows"],
        "unique_gvkeys": summary["coverage"]["unique_gvkeys"],
        "report_year_counts": summary["report_year_counts"],
        "parent_years_with_foreign_subs_frac": summary["coverage"]["parent_years_with_foreign_subs_frac"],
        "parent_years_with_haven_subs_frac": summary["coverage"]["parent_years_with_haven_subs_frac"],
        "outputs": {
            "panel": str(panel_path),
            "country_counts_long": str(country_path),
            "selected_filings": str(filings_path),
            "summary": str(summary_path),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
