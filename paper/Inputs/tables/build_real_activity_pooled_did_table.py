#!/usr/bin/env python3
"""
Build a paper-facing CSV for the pooled real-activity DiD table.

Outputs:
- real_activity_pooled_did_table.csv
- real_activity_pooled_did_preview.md
- real_activity_pooled_did_provenance.csv

Design note:
- Coefficients, standard errors, stars, and row-level firm counts are parsed
  from the canonical real-activity TeX table on the external drive.
- Row-level firm-year counts are not exported in that TeX table or in the
  repo slide-data CSVs, so they are reconstructed from the current repo's
  sample-building logic and stored separately as `n_firmyears_current`.
- Main `coef` and `se` fields are rescaled by 100 into paper-facing units:
  percentage points of lagged assets for the investment rows and 100 x log
  points for the growth rows. Raw repo-exported values are preserved in
  `coef_raw_repo` and `se_raw_repo`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd


# Paths resolve inside this checkout; they were previously pinned to a
# second clone (/Users/erm/...) and an external drive (/Volumes/Crucial X6/...).
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "tables"
REPO_ANALYSIS = REPO_ROOT / "analysis"
REPO_SLIDE_DATA = REPO_ROOT / "data" / "analysis" / "output" / "slide_data"
EXTERNAL_ROOT = REPO_ROOT
REAL_ACTIVITY_TEX = (
    EXTERNAL_ROOT
    / "data"
    / "analysis"
    / "tables"
    / "real_activity_dynamics"
    / "tbl_real_activity_dynamics.tex"
)
COMBINED_PKL = EXTERNAL_ROOT / "data" / "analysis" / "cache" / "combined.pkl"
LONG_PKL = EXTERNAL_ROOT / "data" / "analysis" / "cache" / "long.pkl"
PANEL_BASE_CSV = (
    EXTERNAL_ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv"
)
GENERATED_AT = datetime.now().astimezone().replace(microsecond=0).isoformat()

PRE_YEARS = list(range(2013, 2017))
POST_YEARS = list(range(2019, 2023))
EVENT_YEARS = list(range(2011, 2024))
OMIT_YEARS = {2017, 2018}
# Must match POOLED_YEARS in analysis/build_real_activity_dynamics.py: the
# pooled DiD runs on the tax-side window (pre 2013-2016, post 2019-2022), while
# EVENT_YEARS stays wider so lag(at) exists for the first regression year.
POOLED_YEARS = [yr for yr in range(2013, 2023) if yr not in OMIT_YEARS]
WIN_CH = 15.0

sys.path.insert(0, str(REPO_ANALYSIS))
from sic_industry import industry  # noqa: E402


OUTCOME_SPECS = [
    {
        "row_order": 1,
        "row_key": "capex_intensity",
        "tex_label": "Capex intensity",
        "row_label": "Capex / lag assets",
        "outcome_col": "capex_int",
        "display_unit": "pp of lagged assets",
    },
    {
        "row_order": 2,
        "row_key": "rd_intensity",
        "tex_label": r"R\&D intensity",
        "row_label": "R&D / lag assets",
        "outcome_col": "rd_int",
        "display_unit": "pp of lagged assets",
    },
    {
        "row_order": 3,
        "row_key": "total_investment",
        "tex_label": "Total investment",
        "row_label": "Total investment / lag assets",
        "outcome_col": "total_inv",
        "display_unit": "pp of lagged assets",
    },
    {
        "row_order": 4,
        "row_key": "employment_growth",
        "tex_label": "Employment growth",
        "row_label": "Employment growth",
        "outcome_col": "emp_growth",
        "display_unit": "100 x log points",
    },
    {
        "row_order": 5,
        "row_key": "sales_growth",
        "tex_label": "Sales growth",
        "row_label": "Sales growth",
        "outcome_col": "sale_growth",
        "display_unit": "100 x log points",
    },
]


def source_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().replace(microsecond=0).isoformat()


def parse_real_activity_tex() -> dict[str, dict[str, object]]:
    text = REAL_ACTIVITY_TEX.read_text()
    lines = text.splitlines()
    parsed: dict[str, dict[str, object]] = {}

    for spec in OUTCOME_SPECS:
        label = spec["tex_label"]
        row_pat = re.compile(
            rf"^{re.escape(label)}\s*&\s*(?P<definition>[^&]+)\s*&\s*\$(?P<coef>[+\-]?\d+\.\d+)(?P<sig>\^\{{?\*+\}})?\$.*?&\s*(?P<n_firms>[0-9,]+)\s*\\\\"
        )
        se_pat = re.compile(r"^\s*&\s*&\s*\$\((?P<se>\d+\.\d+)\)\$")

        for idx, line in enumerate(lines):
            m = row_pat.search(line)
            if not m:
                continue
            if idx + 1 >= len(lines):
                raise ValueError(f"Missing SE line after {label!r} in {REAL_ACTIVITY_TEX}.")
            se_match = se_pat.search(lines[idx + 1])
            if not se_match:
                raise ValueError(f"Could not parse SE line after {label!r} in {REAL_ACTIVITY_TEX}.")
            sig_raw = m.group("sig") or ""
            parsed[spec["row_key"]] = {
                "definition": m.group("definition").strip(),
                "coef": float(m.group("coef")),
                "se": float(se_match.group("se")),
                "stars": re.sub(r"[\^{}]", "", sig_raw),
                "n_firms_repo": int(m.group("n_firms").replace(",", "")),
            }
            break
        else:
            raise ValueError(f"Could not find row for {label!r} in {REAL_ACTIVITY_TEX}.")

    return parsed


def build_current_counts() -> dict[str, dict[str, int]]:
    with open(COMBINED_PKL, "rb") as f:
        combined = pickle.load(f)
    with open(LONG_PKL, "rb") as f:
        long_df = pickle.load(f)

    extra = pd.read_csv(
        PANEL_BASE_CSV,
        usecols=["gvkey", "recon_year", "comp_at", "comp_capx", "comp_xrd", "comp_emp"],
        low_memory=False,
    ).drop_duplicates(["gvkey", "recon_year"])

    fy = (
        long_df.groupby(["gvkey", "recon_year", "group"])["cat_val"]
        .sum()
        .reset_index(name="val")
    )
    fy["val_w"] = fy["val"].clip(-WIN_CH, WIN_CH)
    intl_fy = (
        fy[fy["group"] == "International"]
        .groupby(["gvkey", "recon_year"])["val_w"]
        .sum()
    )
    any_obs = fy.groupby(["gvkey", "recon_year"]).size()
    pre_any = any_obs[any_obs.index.get_level_values("recon_year").isin(PRE_YEARS)]
    elig = pre_any.groupby("gvkey").size()
    elig = elig[elig >= 2].index

    pre_intl = (
        intl_fy[
            intl_fy.index.get_level_values("recon_year").isin(PRE_YEARS)
            & intl_fy.index.get_level_values("gvkey").isin(elig)
        ]
        .groupby("gvkey")
        .mean()
    )
    intl_pre = pd.Series(0.0, index=elig, name="intl_pre")
    intl_pre.update(pre_intl)
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)

    panel = combined[combined["recon_year"].isin(EVENT_YEARS)].copy()
    panel["industry"] = panel["comp_sic"].apply(industry)
    panel = panel[panel["industry"] != "Finance"]
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")
    panel = panel.merge(extra, on=["gvkey", "recon_year"], how="left")

    fyset = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = fyset.apply(lambda s: len(s & set(PRE_YEARS)) >= 2)
    has_post = fyset.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok = fyset[has_pre & has_post].index
    panel = panel[panel["gvkey"].isin(ok)].copy()

    lag_src = panel[["gvkey", "recon_year", "comp_at", "comp_emp", "comp_sale"]].copy()
    lag_src["recon_year"] = lag_src["recon_year"] + 1
    lag_src = lag_src.rename(
        columns={"comp_at": "lag_at", "comp_emp": "lag_emp", "comp_sale": "lag_sale"}
    )
    panel = panel.merge(lag_src, on=["gvkey", "recon_year"], how="left")

    valid_at = panel["lag_at"] > 1
    panel["capex_int"] = np.where(
        valid_at & panel["comp_capx"].notna(), panel["comp_capx"] / panel["lag_at"], np.nan
    )
    panel["rd_int"] = np.where(
        valid_at & panel["comp_xrd"].notna(), panel["comp_xrd"] / panel["lag_at"], np.nan
    )
    both_ok = valid_at & panel["comp_capx"].notna() & panel["comp_xrd"].notna()
    panel["total_inv"] = np.where(
        both_ok, (panel["comp_capx"] + panel["comp_xrd"]) / panel["lag_at"], np.nan
    )
    panel["emp_growth"] = np.where(
        (panel["comp_emp"] > 0) & (panel["lag_emp"] > 0),
        np.log(panel["comp_emp"] / panel["lag_emp"]),
        np.nan,
    )
    panel["sale_growth"] = np.where(
        (panel["comp_sale"] > 0) & (panel["lag_sale"] > 0),
        np.log(panel["comp_sale"] / panel["lag_sale"]),
        np.nan,
    )

    bounds = {
        "capex_int": (0.0, 0.5),
        "rd_int": (0.0, 0.5),
        "total_inv": (0.0, 1.0),
        "emp_growth": (-1.0, 1.0),
        "sale_growth": (-1.0, 1.0),
    }
    for col, (lo, hi) in bounds.items():
        panel[col] = panel[col].clip(lo, hi)

    event_keep = POOLED_YEARS
    counts: dict[str, dict[str, int]] = {}
    for spec in OUTCOME_SPECS:
        col = spec["outcome_col"]
        sub = panel[
            panel["recon_year"].isin(event_keep) & panel[col].notna() & panel["intl_pre"].notna()
        ].copy()
        counts[spec["row_key"]] = {
            "n_firms_current": int(sub["gvkey"].nunique()),
            "n_firmyears_current": int(len(sub)),
        }
    counts["_meta"] = {"sample_firms_current": int(panel["gvkey"].nunique())}
    return counts


def load_repo_real_activity_sample_firms() -> int:
    df = pd.read_csv(REPO_SLIDE_DATA / "samplesanddescriptive.csv")
    row = df.loc[df["key"] == "realActivitySampleFirms"]
    if row.empty:
        raise ValueError("Could not find realActivitySampleFirms in samplesanddescriptive.csv.")
    return int(float(row.iloc[0]["value"]))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    tex_rows = parse_real_activity_tex()
    current_counts = build_current_counts()
    repo_sample_firms = load_repo_real_activity_sample_firms()

    rows = []
    provenance_rows = []

    for spec in OUTCOME_SPECS:
        tex_row = tex_rows[spec["row_key"]]
        cur = current_counts[spec["row_key"]]
        coef_scaled = tex_row["coef"] * 100.0
        se_scaled = tex_row["se"] * 100.0
        notes = (
            "Coefficient, SE, stars, and n_firms_repo parsed from the canonical real-activity "
            "TeX table. n_firmyears_current and n_firms_current reconstructed from the current "
            "repo sample-building logic because row-level firm-years are not exported in the table. "
            "Main coef/se fields are scaled by 100 into paper-facing units."
        )
        rows.append(
            {
                "row_order": spec["row_order"],
                "row_key": spec["row_key"],
                "row_label": spec["row_label"],
                "definition_repo": tex_row["definition"],
                "display_unit": spec["display_unit"],
                "coef": coef_scaled,
                "se": se_scaled,
                "stars": tex_row["stars"],
                "coef_raw_repo": tex_row["coef"],
                "se_raw_repo": tex_row["se"],
                "coef_display_3dp": f"{coef_scaled:+.3f}",
                "se_display_3dp": f"{se_scaled:.3f}",
                "coef_raw_display_5dp": f"{tex_row['coef']:+.5f}",
                "se_raw_display_5dp": f"{tex_row['se']:.5f}",
                "n_firms_repo": tex_row["n_firms_repo"],
                "n_firms_current": cur["n_firms_current"],
                "n_firmyears_current": cur["n_firmyears_current"],
                "sample_firms_repo_slide": repo_sample_firms,
                "notes": notes,
            }
        )

        for source_path, source_note in [
            (
                REAL_ACTIVITY_TEX,
                "Canonical external TeX table for coefficients, SEs, stars, and row firm counts.",
            ),
            (
                REPO_ANALYSIS / "build_real_activity_dynamics.py",
                "Current repo logic used to reconstruct row-level firm-year counts.",
            ),
            (REPO_ANALYSIS / "sic_industry.py", "Current repo industry classifier used in the reconstruction."),
            (COMBINED_PKL, "Current external combined panel used in the reconstruction."),
            (LONG_PKL, "Current external long-form reconciliation panel used for intl_pre construction."),
            (PANEL_BASE_CSV, "Current external Compustat extra-vars CSV used for real-activity outcomes."),
        ]:
            provenance_rows.append(
                {
                    "generated_at": GENERATED_AT,
                    "row_key": spec["row_key"],
                    "source_path": str(source_path),
                    "source_mtime": source_mtime(source_path),
                    "notes": source_note,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        OUTDIR / "real_activity_pooled_did_table.csv", index=False, float_format="%.6f"
    )

    preview = [
        "# Real Activity Pooled DiD Preview",
        "",
        f"Generated: `{GENERATED_AT}`",
        "",
        f"Repo slide-data overall real-activity sample firms: `{repo_sample_firms}`",
        "",
        "| Outcome | Coef | SE | Stars | Unit | N firms (repo table) | N firms (current script) | Firm-years (current script) |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for row in out_df.itertuples(index=False):
        preview.append(
            f"| {row.row_label} | {row.coef_display_3dp} | {row.se_display_3dp} | {row.stars or ''} | "
            f"{row.display_unit} | {row.n_firms_repo:,} | {row.n_firms_current:,} | {row.n_firmyears_current:,} |"
        )
    preview.append("")
    preview.append(
        "Note: main `coef` and `se` fields are rescaled by 100 into paper-facing units. "
        "`coef_raw_repo` and `se_raw_repo` preserve the canonical raw values. "
        "`n_firms_repo` is the strict repo-backed row count from the canonical TeX table. "
        "`n_firms_current` and `n_firmyears_current` are reconstructed from the current repo script "
        "because row-level firm-years are not exported in the table or the slide-data CSVs. "
        "The published slide-data scalar for the overall real-activity sample is 2,016 firms."
    )
    (OUTDIR / "real_activity_pooled_did_preview.md").write_text("\n".join(preview) + "\n")

    pd.DataFrame(provenance_rows).to_csv(
        OUTDIR / "real_activity_pooled_did_provenance.csv", index=False
    )


if __name__ == "__main__":
    main()
