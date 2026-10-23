#!/usr/bin/env python3
"""
Build a paper-facing CSV of the page A.14 clipping-robustness headline scalars
from the current repo slide-data exports.

Outputs:
- clip_robustness_headlines.csv
- clip_robustness_headlines_preview.md
- clip_robustness_headlines_provenance.csv
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

SOURCE_CSV = REPO_SLIDE_DATA / "remainingpipeline.csv"

SOURCES = [
    {
        "row_order": 1,
        "row_key": "clipThirtyPpBetaShare",
        "source_key": "clipThirtyPpBetaShare",
        "label": "A.14 clip 30pp: foreign profit share coefficient",
        "notes": "Raw foreign-profit-share coefficient in share units from remainingpipeline.csv.",
    },
    {
        "row_order": 2,
        "row_key": "clipThirtyPpBetaSharePp",
        "source_key": "clipThirtyPpBetaShare",
        "label": "A.14 clip 30pp: foreign profit share coefficient (pp)",
        "notes": "Same foreign-profit-share coefficient rescaled by 100 so the paper can report percentage points directly.",
        "scale_factor": 100.0,
    },
    {
        "row_order": 3,
        "row_key": "clipThirtyPpBetaIntl",
        "source_key": "clipThirtyPpBetaIntl",
        "label": "A.14 clip 30pp: International coefficient",
        "notes": "Signed International coefficient on the foreign-share subsample from remainingpipeline.csv.",
    },
    {
        "row_order": 4,
        "row_key": "clipThirtyPpBetaIntlAbs",
        "source_key": "clipThirtyPpBetaIntl",
        "label": "A.14 clip 30pp: International coefficient (absolute pp)",
        "notes": "Absolute-value version of the signed International coefficient, convenient for prose that discusses magnitudes.",
        "transform": "abs",
    },
    {
        "row_order": 5,
        "row_key": "clipThirtyPpBetaEtr",
        "source_key": "clipThirtyPpBetaEtr",
        "label": "A.14 clip 30pp: ETR coefficient",
        "notes": "Signed matched-subsample ETR coefficient from remainingpipeline.csv.",
    },
    {
        "row_order": 6,
        "row_key": "clipThirtyPpBetaEtrAbs",
        "source_key": "clipThirtyPpBetaEtr",
        "label": "A.14 clip 30pp: ETR coefficient (absolute pp)",
        "notes": "Absolute-value version of the signed ETR coefficient, convenient for prose that discusses magnitudes.",
        "transform": "abs",
    },
    {
        "row_order": 7,
        "row_key": "clipThirtyPpRatioShareIntl",
        "source_key": "clipThirtyPpRatioShareIntl",
        "label": "A.14 clip 30pp: Share/Intl ratio",
        "notes": "Stored repo scalar for 100 x beta_share / |beta_Intl| under the 30pp clipping rule.",
    },
    {
        "row_order": 8,
        "row_key": "clipThirtyPpRatioShareEtr",
        "source_key": "clipThirtyPpRatioShareEtr",
        "label": "A.14 clip 30pp: Share/ETR ratio",
        "notes": "Stored repo scalar for 100 x beta_share / |beta_ETR| under the 30pp clipping rule.",
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

    src = load_slide_data(SOURCE_CSV)
    rows = []
    provenance_rows = []

    for spec in SOURCES:
        row = src.loc[spec["source_key"]]
        value_num = float(row["value"])
        if spec.get("transform") == "abs":
            value_num = abs(value_num)
        value_num *= spec.get("scale_factor", 1.0)

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
                "source_file": SOURCE_CSV.name,
                "notes": spec["notes"],
            }
        )
        provenance_rows.append(
            {
                "generated_at": GENERATED_AT,
                "row_key": spec["row_key"],
                "source_path": str(SOURCE_CSV),
                "source_mtime": source_mtime(SOURCE_CSV),
                "notes": spec["notes"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTDIR / "clip_robustness_headlines.csv", index=False, float_format="%.6f")

    preview = [
        "# Clip Robustness Headlines Preview",
        "",
        f"Generated: `{GENERATED_AT}`",
        "",
        "| Key | Value | Notes |",
        "| --- | ---: | --- |",
    ]
    for row in out_df.itertuples(index=False):
        preview.append(f"| `{row.row_key}` | {row.display_3dp} | {row.notes} |")
    (OUTDIR / "clip_robustness_headlines_preview.md").write_text("\n".join(preview) + "\n")

    pd.DataFrame(provenance_rows).to_csv(
        OUTDIR / "clip_robustness_headlines_provenance.csv", index=False
    )


if __name__ == "__main__":
    main()
