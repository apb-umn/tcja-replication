#!/usr/bin/env python3
"""
Build identification-table inputs for the paper.

This table is designed to support the causal section by showing the
distribution of the treatment variable

    \bar c^{pre}_{Intl,i}

within the exact first-stage regression sample used in the current repo's
event-study pipeline.

Upstream sources:
  - repo logic:
      /Users/erm/Repos/TCJA/analysis/build_firststage_event_studies.py
      /Users/erm/Repos/TCJA/analysis/sic_industry.py
  - current data:
      /Volumes/Crucial X6/Users/apb/TaxReconciliation/data/analysis/cache/combined.pkl
      /Volumes/Crucial X6/Users/apb/TaxReconciliation/data/analysis/cache/long.pkl
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
OUT_DIR = SCRIPT_PATH.parent
# Paths resolve inside this checkout; they were previously pinned to a
# second clone (/Users/erm/...) and an external drive (/Volumes/Crucial X6/...).
REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ANALYSIS_DIR = REPO_ROOT / "analysis"
EXTERNAL_ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
EXTERNAL_CACHE_DIR = EXTERNAL_ANALYSIS_DIR / "cache"

if str(REPO_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_ANALYSIS_DIR))

from sic_industry import industry as classify_industry  # noqa: E402


GENERATED_AT = datetime.now().astimezone().replace(microsecond=0).isoformat()
PRE_YEARS = list(range(2013, 2017))
POST_YEARS = list(range(2019, 2023))
EVENT_YEARS = list(range(2013, 2023))
WIN_CH = 15.0

GROUP_MAP = {
    "International": "International",
    "State & local": "State_local",
    "Federal credits": "Federal_credits",
    "Permanent items": "Permanent_items",
    "Reserves & losses": "Reserves_losses",
    "TCJA one-time": "TCJA_onetime",
    "Other": "Other",
}

DISPLAY_ORDER = [
    ("all_in_sample", "All in-sample firms"),
    ("tech", "Tech"),
    ("pharma_biotech", "Pharma/Biotech"),
    ("manufacturing", "Manufacturing"),
    ("oil_and_gas", "Oil & Gas"),
    ("retail", "Retail"),
    ("utilities", "Utilities"),
    ("other", "Other"),
]

LABEL_TO_KEY = {
    "Tech": "tech",
    "Pharma/Biotech": "pharma_biotech",
    "Manufacturing": "manufacturing",
    "Oil & Gas": "oil_and_gas",
    "Retail": "retail",
    "Utilities": "utilities",
    "Other": "other",
}


def source_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().replace(microsecond=0).isoformat()


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize(series: pd.Series) -> dict[str, float]:
    s = pd.Series(series, dtype=float).dropna()
    share_zero = float((s == 0).mean() * 100.0)
    return {
        "n_firms": int(s.shape[0]),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "min": float(s.min()),
        "max": float(s.max()),
        "share_zero_pct": share_zero,
        "share_nonzero_pct": float(100.0 - share_zero),
    }


def build_first_stage_sample() -> pd.DataFrame:
    combined = load_pickle(EXTERNAL_CACHE_DIR / "combined.pkl")
    long_df = load_pickle(EXTERNAL_CACHE_DIR / "long.pkl")

    df = long_df.copy()
    df["channel"] = df["group"].map(GROUP_MAP).fillna("Other")
    fy = (
        df.groupby(["gvkey", "recon_year", "channel"])["cat_val"]
        .sum()
        .reset_index(name="val")
    )
    fy["val_w"] = fy["val"].clip(-WIN_CH, WIN_CH)
    wide = (
        fy.pivot_table(
            index=["gvkey", "recon_year"],
            columns="channel",
            values="val_w",
            fill_value=0,
        )
        .reset_index()
    )
    wide.columns.name = None

    pre = wide[wide["recon_year"].isin(PRE_YEARS)]
    counts = pre.groupby("gvkey").size()
    elig = counts[counts >= 2].index
    intl_pre = (
        pre[pre["gvkey"].isin(elig)]
        .groupby("gvkey")["International"]
        .mean()
        .rename("intl_pre")
    )
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)

    panel = combined[combined["recon_year"].isin(EVENT_YEARS)].copy()
    panel["industry"] = panel["comp_sic"].apply(classify_industry)
    panel = panel[panel["industry"] != "Finance"]
    if "comp_pi" in panel.columns:
        panel = panel[panel["comp_pi"] > 1]
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(
        wide[["gvkey", "recon_year", "International"]],
        on=["gvkey", "recon_year"],
        how="left",
    )
    panel["International"] = panel["International"].fillna(0.0)

    fy_sets = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = fy_sets.apply(lambda s: len(s & set(PRE_YEARS)) >= 2)
    has_post = fy_sets.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok = fy_sets[has_pre & has_post].index
    panel = panel[panel["gvkey"].isin(ok)].copy()

    firm = panel[["gvkey", "industry", "intl_pre"]].drop_duplicates("gvkey")
    return firm


def build_rows(firm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    overall = summarize(firm["intl_pre"])
    rows.append(
        {
            "row_order": 0,
            "row_key": "all_in_sample",
            "row_label": "All in-sample firms",
            **overall,
        }
    )

    for idx, (row_key, row_label) in enumerate(DISPLAY_ORDER[1:], start=1):
        sub = firm.loc[firm["industry"] == row_label, "intl_pre"]
        stats = summarize(sub)
        rows.append(
            {
                "row_order": idx,
                "row_key": row_key,
                "row_label": row_label,
                **stats,
            }
        )

    return pd.DataFrame(rows)


def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    stat_map = [
        ("n_firms", "N firms"),
        ("mean", "Mean"),
        ("sd", "SD"),
        ("p25", "25th"),
        ("median", "50th"),
        ("p75", "75th"),
        ("iqr", "IQR"),
        ("min", "Min"),
        ("max", "Max"),
        ("share_zero_pct", "Share zero"),
        ("share_nonzero_pct", "Share nonzero"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for record in df.to_dict(orient="records"):
        for stat_key, stat_label in stat_map:
            rows.append(
                {
                    "row_order": record["row_order"],
                    "row_key": record["row_key"],
                    "row_label": record["row_label"],
                    "stat_key": stat_key,
                    "stat_label": stat_label,
                    "value": record[stat_key],
                }
            )
    return pd.DataFrame(rows)


def write_preview(df: pd.DataFrame) -> str:
    header = [
        "# Identification Table Preview",
        "",
        f"Generated: `{GENERATED_AT}`",
        "",
        "Treatment variable: firm-level pre-TCJA average International contribution,",
        "computed over `2013-2016` and clipped at the 1st/99th percentile across firms,",
        "within the exact first-stage regression sample.",
        "",
        "| Row | N firms | Mean | SD | P25 | Median | P75 | % nonzero |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines = header[:]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.row_label} | {row.n_firms:,} | {row.mean:.2f} | {row.sd:.2f} | "
            f"{row.p25:.2f} | {row.median:.2f} | {row.p75:.2f} | {row.share_nonzero_pct:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_readme(firm: pd.DataFrame) -> str:
    return f"""# identification_tikz

Generated: `{GENERATED_AT}`

This directory contains paper-facing inputs for a small identification table
describing the distribution of the treatment variable
`\\bar c^{{pre}}_{{Intl,i}}` by broad industry.

## Sample

The table uses the exact first-stage regression sample from the current repo's
`build_firststage_event_studies.py` logic:

- event-study window: `2013-2022`
- pre-treatment years: `2013-2016`
- post-treatment years: `2019-2022`
- broad SIC-based industry groups from `sic_industry.py`
- Finance excluded using the current repo definition
- profitable observations only: `comp_pi > 1`
- firms must have at least `2` pre and `2` post observations

The treatment variable is the firm-level average pre-TCJA International
contribution over `2013-2016`, after clipping the annual channel contributions
at `±15` percentage points and then clipping the firm-level distribution at the
1st and 99th percentiles.

The resulting first-stage sample contains `{firm['gvkey'].nunique():,}` firms.

## Files

- `intl_pre_by_industry.csv`
  Wide table for direct lookup in LaTeX.
- `intl_pre_by_industry_cells.csv`
  Long-form cell table if a row/stat layout is more convenient.
- `preview.md`
  Quick human-readable preview.
- `provenance.csv`
  Source paths and mtimes.

## Sources

- repo script logic:
  - `/Users/erm/Repos/TCJA/analysis/build_firststage_event_studies.py`
  - `/Users/erm/Repos/TCJA/analysis/sic_industry.py`
- current data:
  - `/Volumes/Crucial X6/Users/apb/TaxReconciliation/data/analysis/cache/combined.pkl`
  - `/Volumes/Crucial X6/Users/apb/TaxReconciliation/data/analysis/cache/long.pkl`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    firm = build_first_stage_sample()
    table = build_rows(firm)
    cells = build_cells(table)

    table.to_csv(OUT_DIR / "intl_pre_by_industry.csv", index=False, float_format="%.6f")
    cells.to_csv(OUT_DIR / "intl_pre_by_industry_cells.csv", index=False, float_format="%.6f")
    (OUT_DIR / "preview.md").write_text(write_preview(table))
    (OUT_DIR / "README.md").write_text(write_readme(firm))

    provenance = pd.DataFrame(
        [
            {
                "generated_at": GENERATED_AT,
                "source_role": "repo_script_logic",
                "source_path": str(REPO_ANALYSIS_DIR / "build_firststage_event_studies.py"),
                "source_mtime": source_mtime(REPO_ANALYSIS_DIR / "build_firststage_event_studies.py"),
                "notes": "Current repo first-stage event-study sample logic.",
            },
            {
                "generated_at": GENERATED_AT,
                "source_role": "repo_classifier",
                "source_path": str(REPO_ANALYSIS_DIR / "sic_industry.py"),
                "source_mtime": source_mtime(REPO_ANALYSIS_DIR / "sic_industry.py"),
                "notes": "Current repo SIC-to-industry mapping.",
            },
            {
                "generated_at": GENERATED_AT,
                "source_role": "combined_panel",
                "source_path": str(EXTERNAL_CACHE_DIR / "combined.pkl"),
                "source_mtime": source_mtime(EXTERNAL_CACHE_DIR / "combined.pkl"),
                "notes": "Current external-drive combined panel.",
            },
            {
                "generated_at": GENERATED_AT,
                "source_role": "long_panel",
                "source_path": str(EXTERNAL_CACHE_DIR / "long.pkl"),
                "source_mtime": source_mtime(EXTERNAL_CACHE_DIR / "long.pkl"),
                "notes": "Current external-drive long reconciliation panel.",
            },
        ]
    )
    provenance.to_csv(OUT_DIR / "provenance.csv", index=False)


if __name__ == "__main__":
    main()
