#!/usr/bin/env python3
"""
Cross-check the staged paper inputs against each other.

The build writes the same estimate into several CSVs -- Table 7's coefficients
land in pooled_did_results_table.csv, the derived ratios in
profit_location_headlines.csv, and the slide scalars in slide_data/ -- and
those files are produced by different scripts run at different times.  When one
is regenerated and another is not, the draft prints two numbers that no longer
belong to the same regression, and nothing complains: the paper still compiles,
because every value is a valid entry in a valid CSV.

This has happened repeatedly.  The pass-through ratios were computed from a
superseded beta_Share; Table 6's committed CSV was stale against its own
builder; the worked technology example was hand-written and had drifted from
both the industry figure and the pooled estimates.

Run this last in the rebuild sequence (see DRAFT2_BUILD_INSTRUCTIONS.md).  It
asserts the identities that must hold if every input came from one run, and
exits non-zero listing whatever no longer agrees.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


INPUTS = Path(__file__).resolve().parent

# Displayed values are rounded to two or three decimals before they are stored,
# so identities hold to display precision rather than exactly.
TOL_DISPLAY = 5e-3
# The channel decomposition is exact only up to the per-channel winsorization.
TOL_IDENTITY = 0.10

failures: list[str] = []
checks_run = 0


def load(relpath: str, key_col: str) -> dict[str, dict[str, str]]:
    with open(INPUTS / relpath, newline="") as f:
        return {row[key_col]: row for row in csv.DictReader(f)}


def check(label: str, left: float, right: float, tol: float) -> None:
    global checks_run
    checks_run += 1
    if abs(left - right) > tol:
        failures.append(f"{label}: {left:.6f} vs {right:.6f} (tolerance {tol})")


def main() -> None:
    pooled = load("tables/pooled_did_results_table.csv", "row_key")
    profloc = load("tables/profit_location_headlines.csv", "row_key")
    channels = load("tables/channel_did_table.csv", "row_key")
    passthru = load("tables/industry_passthrough_examples.csv", "row_key")
    industry = load("decomp_ind_new_tikz/summary.csv", "industry")
    slides = {
        r["key"]: r["value"]
        for r in csv.DictReader(open(INPUTS / "slide_data/samplesanddescriptive.csv", newline=""))
    }

    beta_share = float(pooled["foreign_profit_share"]["coef_table"])
    beta_intl = float(pooled["international"]["coef_table"])
    beta_etr = float(pooled["etr"]["coef_table"])

    # 1. The same foreign-profit-share regression is reported in two files.
    check(
        "beta_Share: pooled_did_results_table vs profit_location_headlines",
        beta_share,
        float(profloc["incomeLocBetaForeignSharePp"]["value"]),
        TOL_DISPLAY,
    )

    # 2-3. The published ratios must come from the betas the paper prints.
    #      Both denominators are estimated on the foreign-share subsample, not
    #      the full panel, so they are read from the same headline file.
    # A missing key is a failure, not a skip: a check that quietly does not run
    # is worse than no check, because it reads as a pass.
    if "incomeLocBetaIntlSubsample" not in profloc:
        failures.append("profit_location_headlines is missing incomeLocBetaIntlSubsample")
    else:
        check(
            "profitPerPpClosure == beta_Share / |beta_Intl on the share subsample|",
            float(slides["profitPerPpClosure"]),
            beta_share / abs(float(profloc["incomeLocBetaIntlSubsample"]["value"])),
            TOL_DISPLAY,
        )
    check(
        "incomeLocClosureRatio == beta_Share / |beta_ETR on the share subsample|",
        float(profloc["incomeLocClosureRatio"]["value"]),
        beta_share / abs(float(profloc["incomeLocBetaEtrSubsample"]["value"])),
        TOL_DISPLAY,
    )

    # 4. The effective rate is the statutory rate plus the sum of the channels,
    #    and the industry-by-year effects absorb the statutory change, so the
    #    channel coefficients must sum to beta_ETR up to winsorization.
    check(
        "sum of channel betas == beta_ETR",
        float(channels["sum_channels"]["coef"]),
        beta_etr,
        TOL_IDENTITY,
    )
    check(
        "sum excluding international == sum of channels - beta_Intl",
        float(channels["sum_non_international"]["coef"]),
        float(channels["sum_channels"]["coef"]) - float(channels["international"]["coef"]),
        1e-6,
    )
    check(
        "channel table beta_Intl == Table 7 beta_Intl",
        float(channels["international"]["coef"]),
        beta_intl,
        1e-6,
    )

    # 5. The worked example must track the industry figure it points the reader at.
    check(
        "technology closure == industry figure international delta",
        float(passthru["tech"]["international_closure_pp"]),
        float(industry["Tech"]["international_delta_pp"]),
        1e-6,
    )
    check(
        "technology implied ETR increase == (beta_ETR / beta_Intl) x closure",
        float(passthru["tech"]["implied_etr_increase_pp"]),
        (beta_etr / beta_intl) * float(passthru["tech"]["international_closure_pp"]),
        # CSVs are written with %.6f, so derived columns round-trip to ~1e-6.
        1e-4,
    )

    # 6. Table 7's N must be the pooled N, not the event study's.
    with open(INPUTS / "first_stage_tikz/first_stage_event_summary.csv", newline="") as f:
        es = next(csv.DictReader(f))
    check(
        "Table 7 firm-years == pooled N from the event summary",
        float(pooled["international"]["n_obs"]),
        float(es["pooled_obs"]),
        1e-6,
    )
    if float(es["event_obs"]) <= float(es["pooled_obs"]):
        failures.append(
            "event-study N should exceed pooled N (it keeps the 2017-2018 transition years): "
            f"{es['event_obs']} vs {es['pooled_obs']}"
        )

    if failures:
        print(f"Input consistency: {len(failures)} of {checks_run} checks FAILED\n")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nThese files are built by different scripts. A failure usually means one was "
            "regenerated and another was not; rerun the full sequence in "
            "DRAFT2_BUILD_INSTRUCTIONS.md."
        )
        sys.exit(1)

    print(f"Input consistency: all {checks_run} cross-CSV checks agree")


if __name__ == "__main__":
    main()
