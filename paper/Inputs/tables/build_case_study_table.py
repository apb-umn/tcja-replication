#!/usr/bin/env python3
"""
Case-study numbers and the litigation robustness check for Section 3.3.

Section 3.3 reads company filings against the panel to ask whether the closure
the screen picks up is the reform.  This stages two things: the panel numbers
for the companies discussed, and the check that drops every company we can
identify as having litigated a transfer-pricing case with the IRS.

The litigation list is the set of companies appearing in the collected court
documents under TPcases/ that can be matched to Compustat.  It is a lower
bound on litigants -- it covers cases that reached a published opinion -- which
is the right direction for a robustness check: if dropping the ones we can
name moves nothing, unnamed ones are unlikely to matter either.

Outputs:
  tables/case_study_table.csv     per-company panel numbers
  tables/case_study_litigation.csv   the drop-litigants check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUTS = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(INPUTS))

import build_inputs_from_external as B  # noqa: E402

# Companies discussed in the text, with the filing evidence for each.
CASES = [
    (160329, "Alphabet", "Irish holding structure wound up"),
    (12141, "Microsoft", "intangible property transferred to the United States"),
    (134932, "ON Semiconductor", "Bermuda subsidiaries domesticated into Delaware"),
    (24800, "Qualcomm", "FDII claimed on export income"),
    (3144, "Coca-Cola", "Section 482 reallocation by the IRS"),
]

# Companies in TPcases/ that match a Compustat gvkey.  Names are recorded so the
# list can be audited against the folder.
LITIGANTS = {
    7435: "3M", 14324: "Altera", 64768: "Amazon", 1632: "Analog Devices",
    2285: "Boeing", 25279: "Boston Scientific", 3144: "Coca-Cola",
    8539: "Exelon", 31166: "Guidant", 5606: "HP", 170617: "Meta",
    12141: "Microsoft", 8479: "PepsiCo", 9667: "Sherwin-Williams",
    121142: "Skechers", 29264: "Talbots", 13554: "Timberland",
    29356: "Veritas", 11465: "Whirlpool", 22325: "Xilinx",
}

PRE = range(2013, 2017)
POST = range(2019, 2023)


def main() -> None:
    long_df = pd.read_pickle(B.EXTERNAL_CACHE_DIR / "long.pkl")
    combined = pd.read_pickle(B.EXTERNAL_CACHE_DIR / "combined.pkl")
    wide = B.build_channels_wide(long_df)
    intl_pre = B.build_intl_pre_from_wide(wide)
    panel = B.build_first_stage_panel(combined, wide, intl_pre)

    pi = pd.read_csv(
        B.EXTERNAL_OUTPUT_DIR / "tax_reconciliation_panel_base.csv",
        usecols=["gvkey", "recon_year", "comp_pidom", "comp_pifo", "comp_at"],
        low_memory=False,
    ).drop_duplicates(["gvkey", "recon_year"])
    share_panel = B.build_foreign_share_panel(
        combined, pi, B.build_intl_pre_for_foreign_share(long_df), B.build_intl_yearly(long_df)
    )

    # --- per-company numbers, on the paper's own definitions ---------------
    rows: list[dict[str, object]] = []
    for order, (gvkey, name, evidence) in enumerate(CASES, start=1):
        w = wide[wide["gvkey"] == gvkey]
        pre = w[w["recon_year"].isin(PRE)]["International"]
        post = w[w["recon_year"].isin(POST)]["International"]
        s = share_panel[share_panel["gvkey"] == gvkey]
        s_pre = s[s["recon_year"].isin(PRE)]["foreign_share"]
        s_post = s[s["recon_year"].isin(POST)]["foreign_share"]
        rows.append(
            {
                "row_order": order,
                "row_key": name.lower().replace(" ", "_").replace("-", "_"),
                "row_label": name,
                "evidence": evidence,
                "intl_pre": float(pre.mean()) if len(pre) else np.nan,
                "intl_post": float(post.mean()) if len(post) else np.nan,
                "intl_closure": float(post.mean() - pre.mean()) if len(pre) and len(post) else np.nan,
                "share_pre_pct": 100.0 * float(s_pre.mean()) if len(s_pre) else np.nan,
                "share_post_pct": 100.0 * float(s_post.mean()) if len(s_post) else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(
        INPUTS / "tables" / "case_study_table.csv", index=False, float_format="%.6f"
    )

    # --- drop the identified litigants and re-estimate ---------------------
    lit = set(LITIGANTS)
    in_sample = sorted(set(panel["gvkey"].unique()) & lit)
    checks: list[dict[str, object]] = [
        {"row_key": "n_litigants_identified", "row_label": "Litigants matched to Compustat",
         "value": float(len(lit))},
        {"row_key": "n_litigants_in_sample", "row_label": "Of those, in the estimation sample",
         "value": float(len(in_sample))},
        {"row_key": "n_sample_firms", "row_label": "Estimation-sample firms",
         "value": float(panel["gvkey"].nunique())},
    ]
    for label, frame, outcome in [
        ("intl", panel, "International"),
        ("etr", panel, "etr_w"),
        ("share", share_panel, "foreign_share"),
    ]:
        scale = 100.0 if outcome == "foreign_share" else 1.0
        full = B.run_pooled_did(frame, outcome)
        drop = B.run_pooled_did(frame[~frame["gvkey"].isin(lit)], outcome)
        checks += [
            {"row_key": f"{label}_full", "row_label": f"{label}: all firms",
             "value": scale * full["coef"]},
            {"row_key": f"{label}_full_se", "row_label": f"{label}: all firms, SE",
             "value": scale * full["se"]},
            {"row_key": f"{label}_drop", "row_label": f"{label}: excluding litigants",
             "value": scale * drop["coef"]},
            {"row_key": f"{label}_drop_se", "row_label": f"{label}: excluding litigants, SE",
             "value": scale * drop["se"]},
        ]
    pd.DataFrame(checks).to_csv(
        INPUTS / "tables" / "case_study_litigation.csv", index=False, float_format="%.6f"
    )
    print(
        f"Wrote case_study_table.csv ({len(rows)} companies) and "
        f"case_study_litigation.csv ({len(in_sample)} litigants in sample)"
    )


if __name__ == "__main__":
    main()
