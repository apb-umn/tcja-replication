#!/usr/bin/env python3
"""
Two reconciliations the appendix tables need in order to agree with each other.

A-1.  Stage 1 records more filings with status OK than stage 2 receives packets
for.  The gap is filings that finished stage 1 without a selected candidate
table, plus the rescue pass, whose packets are processed separately.

A-2.  The arithmetic check is applied twice, and the appendix reports both
counts as though they were one.  It runs on the raw extraction in stage 2 and
again when the base panel is assembled, on filing-years that have been
deduplicated and put on a common schema.  The second application is stricter,
so the second count is larger.  This measures by how much.

Outputs:
  tables/appendix_reconciliation.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUTS = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE1 = REPO_ROOT / "data" / "phase1"
PHASE2 = REPO_ROOT / "data" / "phase2"
ANALYSIS_OUT = REPO_ROOT / "data" / "analysis" / "output"


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def main() -> None:
    rows: list[dict[str, object]] = []

    # --- A-1: filings with status OK that yielded no packet ----------------
    bs = pd.read_csv(PHASE1 / "batch_summary_v4_combined.csv", dtype=str, low_memory=False)
    rescue = pd.read_csv(
        PHASE1 / "output_rescue_year_mismatch_v1" / "main_out" / "batch_summary.csv",
        dtype=str, low_memory=False,
    )
    sel = pd.read_csv(
        PHASE1 / "output_full_run_v4" / "selected_matches.csv",
        usecols=["filing_url", "filing_id"], low_memory=False,
    )
    ok = bs[bs["status"] == "OK"]
    # build_slide_data.py dedupes the rescue summary by filing_url before
    # counting, so match that or the totals will not agree with Table 11.
    rescue = rescue.drop_duplicates("filing_url", keep="first")
    ok_rescue = int((rescue["status"] == "OK").sum())
    no_packet = sorted(set(ok["filing_url"]) - set(sel["filing_url"]))
    rows += [
        {"row_key": "stage1_ok", "row_label": "Stage 1 filings with status OK",
         "value": float(len(ok) + ok_rescue)},
        {"row_key": "stage2_packets", "row_label": "Packets received by stage 2",
         "value": float(sel["filing_id"].nunique())},
        {"row_key": "no_packet_canonical", "row_label": "OK in stage 1 but no candidate table selected",
         "value": float(len(no_packet))},
        {"row_key": "rescue_ok", "row_label": "Rescue-pass filings, processed separately",
         "value": float(ok_rescue)},
    ]

    # --- A-2: the same check applied at two points -------------------------
    p2 = pd.read_csv(PHASE2 / "full_run_v4_llm.csv", usecols=["id", "year", "check"], low_memory=False)
    base = pd.read_csv(
        ANALYSIS_OUT / "tax_reconciliation_panel_base.csv",
        usecols=["filing_id", "recon_year", "recon_check"], low_memory=False,
    )
    merged = base.merge(
        p2.rename(columns={"id": "filing_id", "year": "recon_year", "check": "extract_check"}),
        on=["filing_id", "recon_year"], how="inner",
    )
    at_extract = truthy(merged["extract_check"])
    at_panel = truthy(merged["recon_check"])
    rows += [
        {"row_key": "fail_at_extraction", "row_label": "Fail the check at extraction",
         "value": float((~truthy(p2["check"])).sum())},
        {"row_key": "fail_at_panel", "row_label": "Fail it when the panel is assembled",
         "value": float((~truthy(base["recon_check"])).sum())},
        {"row_key": "pass_then_fail", "row_label": "Pass at extraction, fail on the panel",
         "value": float((at_extract & ~at_panel).sum())},
        {"row_key": "fail_then_pass", "row_label": "Fail at extraction, pass on the panel",
         "value": float((~at_extract & at_panel).sum())},
        {"row_key": "matched_filing_years", "row_label": "Filing-years present at both points",
         "value": float(len(merged))},
    ]

    df = pd.DataFrame(rows)
    out = INPUTS / "tables" / "appendix_reconciliation.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out.name} ({len(df)} keys)")


if __name__ == "__main__":
    main()
