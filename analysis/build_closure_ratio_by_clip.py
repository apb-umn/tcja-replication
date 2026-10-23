#!/usr/bin/env python3
"""Regenerate `data/analysis/output/robustness/foreign_share_closure_ratio_by_clip.csv`
on the current sample.

For each clip variant, restricts the panel to firms with non-missing
foreign_share, then runs three pooled-DiD regressions on that *same*
subsample:

  beta_share              — outcome = foreign PI share
  beta_intl_same_sample   — outcome = International reconciliation channel
  beta_etr_same_sample    — outcome = effective tax rate (ETR)

The slide-data builder (analysis/build_slide_data.py) reads this CSV
to populate slide A.14's two ratio columns (`Share/Intl`, `Share/ETR`)
and the per-clip beta columns.

Run via:
    python3 analysis/build_closure_ratio_by_clip.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from build_substitution_real_outcomes import (
    load_data, build_channels_and_treatment, build_intl_yearly,
    build_panel, run_pooled_did,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "analysis" / "output" / "robustness"

CLIP_VARIANTS = [
    ("clip_10pp",           10.0),
    ("clip_15pp_baseline",  15.0),  # baseline (matches existing CSV convention)
    ("clip_20pp",           20.0),
    ("clip_30pp",           30.0),
    ("clip_50pp",           50.0),
    ("no_per_obs_clip",     None),
]

SPECS = [
    ("firm_year_fe",            False),
    ("firm_fe_industry_year_fe", True),
]


def run_for_clip(combined, long_df, pi, clip_pp):
    """Build the panel under one clip variant, return the subsample regressions."""
    intl_pre  = build_channels_and_treatment(long_df, win_ch=clip_pp)
    intl_year = build_intl_yearly(long_df, win_ch=clip_pp)
    panel     = build_panel(combined, pi, intl_pre, intl_year)

    # Add ETR column for the third regression. The reduced-form ETR
    # used in build_substitution_analysis is `etr_w` -- a 100x scaled,
    # winsorized version. Mirror that here.
    if 'comp_txt' in panel.columns and 'comp_pi' in panel.columns:
        etr = panel['comp_txt'] / panel['comp_pi'].replace(0, np.nan) * 100
        etr = etr.clip(-100, 100)
        panel['etr'] = etr
    else:
        panel['etr'] = np.nan

    # Restrict to firms with observed foreign_share (the income-location subsample)
    sub = panel.dropna(subset=['foreign_share', 'International', 'intl_pre']).copy()
    rows = []
    for spec_name, ind_yr_fe in SPECS:
        b_share = run_pooled_did(sub, 'foreign_share',  ind_yr_fe=ind_yr_fe)
        b_intl  = run_pooled_did(sub, 'International',   ind_yr_fe=ind_yr_fe)
        sub_etr = sub.dropna(subset=['etr'])
        b_etr   = run_pooled_did(sub_etr, 'etr',          ind_yr_fe=ind_yr_fe) if len(sub_etr) else {}

        bs, bse_share = b_share.get('coef', np.nan), b_share.get('se', np.nan)
        bi, bse_intl  = b_intl.get('coef', np.nan),  b_intl.get('se', np.nan)
        be, bse_etr   = b_etr.get('coef',  np.nan),  b_etr.get('se',  np.nan)

        def safe_ratio(num, denom):
            if np.isnan(num) or np.isnan(denom) or abs(denom) < 1e-9:
                return np.nan
            return 100.0 * num / abs(denom)

        rows.append({
            'spec': spec_name,
            'beta_share':                bs,
            'se_share':                  bse_share,
            'beta_intl_same_sample':     bi,
            'se_intl_same_sample':       bse_intl,
            'ratio_100x_beta_share_over_abs_beta_intl': safe_ratio(bs, bi),
            'beta_etr_same_sample':      be,
            'se_etr_same_sample':        bse_etr,
            'ratio_100x_beta_share_over_abs_beta_etr':  safe_ratio(bs, be),
            'regression_obs':            int(b_share.get('n', 0)),
            'firms':                     int(b_share.get('G', 0)),
            'event_window_obs':          int(len(sub)),
            'event_window_firms':        int(sub['gvkey'].nunique()),
            'treat_p1':                  float(sub['intl_pre'].quantile(0.01)),
            'treat_p99':                 float(sub['intl_pre'].quantile(0.99)),
        })
    return rows


def main():
    combined, long_df, pi = load_data()

    all_rows = []
    for clip_name, clip_pp in CLIP_VARIANTS:
        print(f'\n== clip variant: {clip_name} (clip_pp={clip_pp}) ==')
        rows = run_for_clip(combined, long_df, pi, clip_pp)
        for r in rows:
            r['clip_variant'] = clip_name
            r['clip_pp']      = clip_pp if clip_pp is not None else np.nan
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # Match the schema build_slide_data.py expects
    col_order = [
        'clip_variant', 'clip_pp', 'spec',
        'beta_share', 'se_share',
        'beta_intl_same_sample', 'se_intl_same_sample',
        'ratio_100x_beta_share_over_abs_beta_intl',
        'beta_etr_same_sample', 'se_etr_same_sample',
        'ratio_100x_beta_share_over_abs_beta_etr',
        'regression_obs', 'firms', 'event_window_obs', 'event_window_firms',
        'treat_p1', 'treat_p99',
    ]
    df = df[col_order]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "foreign_share_closure_ratio_by_clip.csv"
    df.to_csv(out_path, index=False)
    print(f'\nWrote {len(df)} rows -> {out_path.relative_to(ROOT)}')


if __name__ == "__main__":
    main()
