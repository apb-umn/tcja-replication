#!/usr/bin/env python3
"""
Build a paper-facing pooled DiD results table from the current event-study
summary CSVs already staged under Inputs/.

Outputs:
- pooled_did_results_table.csv
- pooled_did_results_preview.md
- pooled_did_results_provenance.csv
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


# Paths resolve inside this checkout; they were previously pinned to a
# second clone (/Users/erm/...) and an external drive (/Volumes/Crucial X6/...).
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "tables"
GENERATED_AT = datetime.now().astimezone().replace(microsecond=0).isoformat()

SOURCES = [
    {
        "row_order": 1,
        "row_key": "international",
        "row_label": "International contribution",
        "source_csv": ROOT / "first_stage_tikz" / "first_stage_event_summary.csv",
        "scale_factor": 1.0,
        "display_unit": "pp",
        "notes": "Direct pooled DiD from first-stage event-study summary.",
    },
    {
        "row_order": 2,
        "row_key": "etr",
        "row_label": "Effective tax rate",
        "source_csv": ROOT / "first_etr_tikz" / "first_etr_event_summary.csv",
        "scale_factor": 1.0,
        "display_unit": "pp",
        "notes": "Direct pooled DiD from ETR event-study summary.",
    },
    {
        "row_order": 3,
        "row_key": "foreign_profit_share",
        "row_label": "Foreign profit share",
        "source_csv": ROOT / "first_pishr_tikz" / "first_pishr_event_summary.csv",
        "scale_factor": 100.0,
        "display_unit": "pp",
        "notes": "Raw regression is in share units; table rescales by 100 so entries are percentage points.",
    },
]


def source_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().replace(microsecond=0).isoformat()


def long_window_rows() -> list[dict]:
    """The 2019-2024 post window, staged so Section 2 can cite it.

    Section 2 says the regressions stop at 2022 where the descriptive tables
    run to 2024, and that extending them strengthens the estimates.  That
    comparison has to come from the robustness run rather than from a literal
    in the text.
    """
    path = (
        ROOT.parents[1] / "data" / "analysis" / "output" / "robustness"
        / "post_window_variants.csv"
    )
    df = pd.read_csv(path)
    row = df[(df["variant"] == "Long post") & (df["outcome"] == "International")]
    if row.empty:
        raise ValueError(f"no Long post / International row in {path}")
    r = row.iloc[0]
    return [
        {
            "row_order": 90,
            "row_key": "international_long_window",
            "row_label": "International contribution, post 2019-2024",
            "coef_table": float(r["coef"]),
            "coef_abs": abs(float(r["coef"])),
            "se_table": float(r["se"]),
            "coef_raw": float(r["coef"]),
            "se_raw": float(r["se"]),
            "scale_factor": 1.0,
            "display_unit": "pp",
            "n_firms": int(r["firms"]),
            "n_obs": int(r["n"]),
            "stars": str(r["stars"]) if pd.notna(r["stars"]) else "",
            "notes": "Post window extended to 2024; from build_post_window_variants.py.",
        }
    ]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []
    provenance_rows = []
    for spec in SOURCES:
        df = pd.read_csv(spec["source_csv"])
        row = df.iloc[0]
        rows.append(
            {
                "row_order": spec["row_order"],
                "row_key": spec["row_key"],
                "row_label": spec["row_label"],
                "coef_table": float(row["pooled_coef"]) * spec["scale_factor"],
                # Magnitude, so the text can say "closes by X" without the sign
                # words fighting the negative coefficient.
                "coef_abs": abs(float(row["pooled_coef"]) * spec["scale_factor"]),
                "se_table": float(row["pooled_se"]) * spec["scale_factor"],
                "coef_raw": float(row["pooled_coef"]),
                "se_raw": float(row["pooled_se"]),
                "scale_factor": spec["scale_factor"],
                "display_unit": spec["display_unit"],
                "n_firms": int(row["pooled_firms"]),
                "n_obs": int(row["pooled_obs"]),
                "stars": row["pooled_stars"],
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

    out_df = pd.DataFrame(rows + long_window_rows())
    out_df.to_csv(OUTDIR / "pooled_did_results_table.csv", index=False, float_format="%.6f")

    preview = [
        "# Pooled DiD Results Preview",
        "",
        f"Generated: `{GENERATED_AT}`",
        "",
        "| Outcome | Coef. | SE | N firms | N obs. |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in out_df.itertuples(index=False):
        preview.append(
            f"| {row.row_label} | {row.coef_table:.3f}{row.stars} | {row.se_table:.3f} | "
            f"{row.n_firms:,} | {row.n_obs:,} |"
        )
    (OUTDIR / "pooled_did_results_preview.md").write_text("\n".join(preview) + "\n")

    pd.DataFrame(provenance_rows).to_csv(
        OUTDIR / "pooled_did_results_provenance.csv", index=False
    )


if __name__ == "__main__":
    main()
