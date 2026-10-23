"""build_robustness_ladder.py

Regenerate the channel-clip robustness ladder that feeds the deck's main
headline coefficients (slides 36 / 40 / 42 pass-through, slides 49 / 50 /
51 income location) and the A.14 robustness table.

For each (family, outcome, clip_variant), rerun the pooled DiD with that
per-firm-year-channel clip applied and record the result.

Clip variants:
  - clip_10pp           (+/- 10 pp per-obs winsor)
  - clip_15pp_baseline  (+/- 15 pp; matches the deck's main spec)
  - clip_20pp           (+/- 20 pp)
  - clip_30pp           (+/- 30 pp)
  - no_per_obs_clip     (no per-obs winsor; firm-level p1/p99 still applied)

Families and outcomes (must stay aligned with channel_clip_robustness_headlines.csv
schema and the cells consumed by build_slide_data.py):

  substitution_channels  : International, ETR
  income_location        : foreign_share, foreign_share_log
  profit_shifting_dynamics: foreign_pi_share, foreign_share_log
  real_activity          : capex_int, rd_int, total_inv

Spec: pooled DiD with firm FE + industry x year FE, SE clustered by firm,
post = 2019-2022, pre = 2013-2016, transition years 2017-2018 excluded.

Writes:
  data/analysis/output/robustness/channel_clip_robustness_headlines.csv
  data/analysis/output/robustness/channel_clip_robustness_summary.md (refreshed)

Invoked by `make data` before build_slide_data.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))

# Import the underlying machinery from the per-family scripts so the
# regression definitions stay consistent.  Each helper accepts a `win_ch`
# parameter we just plumbed through.
import build_substitution_analysis as bsa
import build_substitution_real_outcomes as bsro
import build_real_activity_dynamics as brad
import build_profit_shifting_dynamics as bpsd


CLIP_VARIANTS = [
    ("clip_10pp",          10.0),
    ("clip_15pp_baseline", 15.0),
    ("clip_20pp",          20.0),
    ("clip_30pp",          30.0),
    ("no_per_obs_clip",    None),
]

OUT_DIR = ROOT / "data" / "analysis" / "output" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _treat_quantiles(intl_pre: pd.Series) -> dict[str, float]:
    return {
        "treat_p1":     float(intl_pre.quantile(0.01)),
        "treat_p25":    float(intl_pre.quantile(0.25)),
        "treat_median": float(intl_pre.quantile(0.50)),
        "treat_p75":    float(intl_pre.quantile(0.75)),
        "treat_p99":    float(intl_pre.quantile(0.99)),
    }


def _make_row(family, variant, clip_pp, outcome, result, treat_n_firms, treat_quants):
    coef, se = result["coef"], result["se"]
    n = result.get("n", 0)
    firms = result.get("G", 0)
    t = coef / se if se and se > 0 else np.nan
    return {
        "family":       family,
        "clip_variant": variant,
        "clip_pp":      clip_pp,
        "outcome":      outcome,
        "spec":         "firm_fe_industry_year_fe",
        "coef":         coef,
        "se":           se,
        "t":            t,
        "n":            n,
        "firms":        firms,
        "treat_firms":  treat_n_firms,
        **treat_quants,
    }


# ─── Substitution channels & income location share the same panel ────────────

def run_substitution_and_income_location(combined, long_df, pi):
    """Returns rows for substitution_channels and income_location families
    across all clip variants. Both families use the same balanced panel
    (profitable, ex-Finance, >=2 pre + >=2 post) but with different outcomes."""
    rows = []
    for variant, clip_pp in CLIP_VARIANTS:
        print(f"\n[substitution + income_location] {variant}")
        # Channel-wide panel for substitution
        wide = bsa.build_channels(long_df, win_ch=clip_pp)
        intl_pre = bsa.build_treatment(wide)
        treat_quants = _treat_quantiles(intl_pre)
        treat_n_firms = int(len(intl_pre))
        panel = bsa.build_panel(combined, wide, intl_pre)

        # substitution_channels: International, ETR
        for outcome in ("International", "etr_w"):
            res = bsa.run_pooled_did(panel, outcome,
                                     treat_col="intl_pre", ind_yr_fe=True)
            outcome_label = "International" if outcome == "International" else "ETR"
            rows.append(_make_row("substitution_channels", variant, clip_pp,
                                  outcome_label, res, treat_n_firms, treat_quants))

        # income_location uses a slightly different panel (requires pidom/pifo).
        # Build using bsro's machinery with the same clip so the treatment is
        # constructed identically.
        intl_pre_il  = bsro.build_channels_and_treatment(long_df, win_ch=clip_pp)
        intl_yearly  = bsro.build_intl_yearly(long_df, win_ch=clip_pp)
        panel_il     = bsro.build_panel(combined, pi, intl_pre_il, intl_yearly)
        for outcome in ("foreign_share", "foreign_share_log"):
            res = bsro.run_pooled_did(panel_il, outcome,
                                      treat_col="intl_pre", ind_yr_fe=True)
            rows.append(_make_row("income_location", variant, clip_pp,
                                  outcome, res, int(len(intl_pre_il)),
                                  _treat_quantiles(intl_pre_il)))
    return rows


# ─── Profit-shifting dynamics ────────────────────────────────────────────────

def run_profit_shifting_dynamics(combined, long_df, pi):
    rows = []
    for variant, clip_pp in CLIP_VARIANTS:
        print(f"\n[profit_shifting_dynamics] {variant}")
        intl_pre = bpsd.build_intl_pre(long_df, win_ch=clip_pp)
        panel = bpsd.build_panel(combined, pi, intl_pre)
        for outcome in ("foreign_pi_share", "foreign_share_log"):
            res = bpsd.run_pooled_did(panel, outcome,
                                      treat_col="intl_pre", spec="ind_yr")
            rows.append(_make_row("profit_shifting_dynamics", variant, clip_pp,
                                  outcome, res, int(len(intl_pre)),
                                  _treat_quantiles(intl_pre)))
    return rows


# ─── Real activity ───────────────────────────────────────────────────────────

def run_real_activity(combined, long_df, extra):
    rows = []
    for variant, clip_pp in CLIP_VARIANTS:
        print(f"\n[real_activity] {variant}")
        intl_pre = brad.build_intl_pre(long_df, win_ch=clip_pp)
        panel = brad.build_panel(combined, extra, intl_pre)
        for outcome, label in (
            ("capex_int", "capex_int"),
            ("rd_int",    "rd_int"),
            ("total_inv", "total_inv"),
        ):
            res = brad.run_pooled_did(panel, outcome,
                                      treat_col="intl_pre", spec="ind_yr")
            rows.append(_make_row("real_activity", variant, clip_pp,
                                  label, res, int(len(intl_pre)),
                                  _treat_quantiles(intl_pre)))
    return rows


# ─── Diff vs baseline + write ────────────────────────────────────────────────

def add_baseline_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Add baseline_coef / baseline_se / coef_minus_baseline / pct_change cols."""
    baselines = df[df["clip_variant"] == "clip_15pp_baseline"][
        ["family", "outcome", "coef", "se"]
    ].rename(columns={"coef": "baseline_coef", "se": "baseline_se"})
    df = df.merge(baselines, on=["family", "outcome"], how="left")
    df["coef_minus_baseline"]   = df["coef"] - df["baseline_coef"]
    df["pct_change_vs_baseline"] = np.where(
        df["baseline_coef"].abs() > 1e-12,
        100 * df["coef_minus_baseline"] / df["baseline_coef"],
        np.nan,
    )
    return df


def write_summary_md(df: pd.DataFrame, path: Path) -> None:
    """Refresh the human-readable markdown alongside the headlines CSV."""
    lines = []
    lines.append("# Channel-clip robustness ladder\n")
    lines.append("Regenerated by `analysis/build_robustness_ladder.py` "
                 "as part of `make data`.\n")
    for family in df["family"].unique():
        fam = df[df["family"] == family]
        lines.append(f"\n## {family}\n")
        for outcome in fam["outcome"].unique():
            sub = fam[fam["outcome"] == outcome].sort_values("clip_pp",
                                                              na_position="last")
            lines.append(f"\n### outcome = `{outcome}`\n")
            lines.append("| Clip variant | coef | SE | t | n | firms |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['clip_variant']} | {r['coef']:+.5f} | {r['se']:.5f} | "
                    f"{r['t']:+.3f} | {int(r['n']):,} | {int(r['firms']):,} |"
                )
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",
                   default=str(ROOT / "data" / "analysis" / "cache"))
    args = p.parse_args()

    print("Loading caches once for all families ...")
    combined, long_df = bsa.load_data(args.cache_dir)
    pi = bsro.load_data()[2]      # bsro.load_data returns (combined, long_df, pi)
    extra = brad.load_data()[2]   # brad.load_data returns (combined, long_df, extra)

    all_rows = []
    all_rows += run_substitution_and_income_location(combined, long_df, pi)
    all_rows += run_profit_shifting_dynamics(combined, long_df, pi)
    all_rows += run_real_activity(combined, long_df, extra)

    df = pd.DataFrame(all_rows)
    df = add_baseline_diff(df)

    # Reorder columns to match the prior CSV schema build_slide_data.py expects.
    column_order = [
        "family", "clip_variant", "clip_pp", "outcome", "spec",
        "coef", "se", "t",
        "n", "firms", "treat_firms",
        "treat_p1", "treat_p25", "treat_median", "treat_p75", "treat_p99",
        "baseline_coef", "baseline_se",
        "coef_minus_baseline", "pct_change_vs_baseline",
    ]
    df = df[column_order]

    out_csv = OUT_DIR / "channel_clip_robustness_headlines.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows -> {out_csv.relative_to(ROOT)}")

    write_summary_md(df, OUT_DIR / "channel_clip_robustness_summary.md")
    print(f"Wrote markdown summary -> "
          f"{(OUT_DIR / 'channel_clip_robustness_summary.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
