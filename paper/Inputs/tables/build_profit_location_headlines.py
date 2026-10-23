#!/usr/bin/env python3
"""
Build a paper-facing CSV of profit-location headline scalars from the current
repo slide-data exports.

Outputs:
- profit_location_headlines.csv
- profit_location_headlines_preview.md
- profit_location_headlines_provenance.csv
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


# Paths resolve inside this checkout; they were previously pinned to a
# second clone (/Users/erm/...) and an external drive (/Volumes/Crucial X6/...).
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "tables"
REPO_SLIDE_DATA = REPO_ROOT / "data" / "analysis" / "output" / "slide_data"
GENERATED_AT = datetime.now().astimezone().replace(microsecond=0).isoformat()

SOURCES = [
    {
        "row_order": 1,
        "row_key": "incomeLocBetaForeignShare",
        "source_key": "incomeLocBetaForeignShare",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: foreign profit share coefficient",
        "notes": "Raw regression coefficient in share units from regressionheadlines.csv.",
    },
    {
        "row_order": 2,
        "row_key": "incomeLocBetaForeignSharePp",
        "source_key": "incomeLocBetaForeignShare",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: foreign profit share coefficient (pp)",
        "notes": "Same coefficient rescaled by 100 so the paper can report percentage points directly.",
        "scale_factor": 100.0,
    },
    {
        "row_order": 3,
        "row_key": "incomeLocBetaForeignShareSe",
        "source_key": "incomeLocBetaForeignShareSe",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: foreign profit share standard error",
        "notes": "Raw regression standard error in share units from regressionheadlines.csv.",
    },
    {
        "row_order": 4,
        "row_key": "incomeLocBetaForeignShareSePp",
        "source_key": "incomeLocBetaForeignShareSe",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: foreign profit share standard error (pp)",
        "notes": "Same standard error rescaled by 100 so the paper can report percentage points directly.",
        "scale_factor": 100.0,
    },
    {
        "row_order": 5,
        "row_key": "incomeLocBetaIntlSubsample",
        "source_key": "incomeLocBetaIntlSubsample",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: matched-subsample International coefficient",
        "notes": "International coefficient on the income-location subsample; denominator for profitPerPpClosure.",
    },
    {
        "row_order": 5,
        "row_key": "incomeLocBetaEtrSubsample",
        "source_key": "incomeLocBetaEtrSubsample",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: matched-subsample ETR coefficient",
        "notes": "ETR coefficient on the income-location subsample; denominator for the closure-ratio calculation.",
    },
    {
        "row_order": 6,
        "row_key": "incomeLocClosureRatio",
        "source_key": "incomeLocClosureRatio",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Income location: implied foreign-share response per 1 pp ETR increase",
        "notes": "Stored repo scalar. Equals 100 x |beta_foreign_share| / |beta_etr_subsample|.",
    },
    {
        "row_order": 7,
        "row_key": "geoRevenueBetaForeignShare",
        "source_key": "geoRevenueBetaForeignShare",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Foreign revenue share coefficient",
        "notes": "Raw foreign-revenue-share coefficient in share units from regressionheadlines.csv.",
    },
    {
        "row_order": 8,
        "row_key": "geoRevenueBetaForeignSharePp",
        "source_key": "geoRevenueBetaForeignShare",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Foreign revenue share coefficient (pp)",
        "notes": "Same foreign-revenue-share coefficient rescaled by 100 so the paper can report percentage points directly.",
        "scale_factor": 100.0,
    },
    {
        "row_order": 9,
        "row_key": "geoRevenueBetaForeignShareSe",
        "source_key": "geoRevenueBetaForeignShareSe",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Foreign revenue share standard error",
        "notes": "Raw foreign-revenue-share standard error in share units from regressionheadlines.csv.",
    },
    {
        "row_order": 10,
        "row_key": "geoRevenueBetaForeignShareSePp",
        "source_key": "geoRevenueBetaForeignShareSe",
        "source_csv": REPO_SLIDE_DATA / "regressionheadlines.csv",
        "label": "Foreign revenue share standard error (pp)",
        "notes": "Same foreign-revenue-share standard error rescaled by 100 so the paper can report percentage points directly.",
        "scale_factor": 100.0,
    },
    {
        "row_order": 11,
        "row_key": "geoRevenueShareFirms",
        "source_key": "geoRevenueShareFirms",
        "source_csv": REPO_SLIDE_DATA / "samplesanddescriptive.csv",
        "label": "Foreign revenue share: distinct firms with usable segment data",
        "notes": "Repo slide scalar used in the profit-location / foreign-revenue-share section.",
    },
    {
        "row_order": 12,
        "row_key": "geoRevenueShareFirmYears",
        "source_key": "geoRevenueShareFirmYears",
        "source_csv": REPO_SLIDE_DATA / "samplesanddescriptive.csv",
        "label": "Foreign revenue share: pooled DiD firm-years",
        "notes": "Repo slide scalar used in the profit-location / foreign-revenue-share section.",
    },
    {
        "row_order": 13,
        "row_key": "profitVsRevenueResponseRatio",
        "source_key": "profitVsRevenueResponseRatio",
        "source_csv": REPO_SLIDE_DATA / "samplesanddescriptive.csv",
        "label": "Profit vs revenue response ratio",
        "notes": "Repo slide scalar comparing the implied profit-share response to the implied revenue-share response.",
    },
    {
        "row_order": 14,
        "row_key": "foreignShareDeclinePct",
        "source_key": "foreignShareDeclinePct",
        "source_csv": REPO_SLIDE_DATA / "samplesanddescriptive.csv",
        "label": "Typical exposed firm: percent decline in foreign profit share",
        "notes": "Repo slide scalar for the typical-exposed-firm interpretation.",
    },
    {
        "row_order": 15,
        "row_key": "foreignShareDeclinePp",
        "source_key": "foreignShareDeclinePp",
        "source_csv": REPO_SLIDE_DATA / "samplesanddescriptive.csv",
        "label": "Typical exposed firm: decline in foreign profit share (pp)",
        "notes": "Repo slide scalar for the typical-exposed-firm interpretation, in percentage points.",
    },
]


def source_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().replace(microsecond=0).isoformat()


def load_slide_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "key" not in df.columns:
        raise ValueError(f"{path} does not have the expected slide-data schema.")
    return df.set_index("key", drop=False)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, pd.DataFrame] = {}
    rows = []
    provenance_rows = []

    for spec in SOURCES:
        if spec["source_csv"] not in cache:
            cache[spec["source_csv"]] = load_slide_data(spec["source_csv"])
        src = cache[spec["source_csv"]]
        row = src.loc[spec["source_key"]]
        value = row["value"]
        scale_factor = spec.get("scale_factor", 1.0)
        value_num = float(value) * scale_factor
        rows.append(
            {
                "row_order": spec["row_order"],
                "row_key": spec["row_key"],
                "row_label": spec["label"],
                "value": value_num,
                "display_0dp": f"{value_num:.0f}",
                "display_1dp": f"{value_num:.1f}",
                "display_3dp": f"{value_num:.3f}",
                "format": row["format"],
                "source_file": spec["source_csv"].name,
                "notes": spec["notes"],
            }
        )
        provenance_rows.append(
            {
                "generated_at": GENERATED_AT,
                "row_key": spec["row_key"],
                "source_path": str(spec["source_csv"]),
                "source_mtime": source_mtime(spec["source_csv"]),
                "notes": spec["notes"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTDIR / "profit_location_headlines.csv", index=False, float_format="%.6f")

    preview = [
        "# Profit Location Headlines Preview",
        "",
        f"Generated: `{GENERATED_AT}`",
        "",
        "| Key | Value | Notes |",
        "| --- | ---: | --- |",
    ]
    for row in out_df.itertuples(index=False):
        preview.append(f"| `{row.row_key}` | {row.display_3dp} | {row.notes} |")
    (OUTDIR / "profit_location_headlines_preview.md").write_text("\n".join(preview) + "\n")

    pd.DataFrame(provenance_rows).to_csv(
        OUTDIR / "profit_location_headlines_provenance.csv", index=False
    )


if __name__ == "__main__":
    main()
