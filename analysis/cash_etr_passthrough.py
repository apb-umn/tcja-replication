#!/usr/bin/env python3
"""
cash_etr_passthrough.py
Pass-through DiD (Part I baseline spec) with CASH effective tax rate as the
outcome:  y_it = 100 * comp_txpd_it / comp_pi_it   (income taxes PAID / pretax income).

Reuses the exact treatment, sample, FE structure and estimator from
build_substitution_analysis.py (firm FE + industry x year FE, SE clustered by
firm, pre 2013-16, post 2019-22, transition 2017-18 dropped). comp_txpd is not
in the cached combined.pkl, so it is merged in from the canonical panel.

Run:  python3 cash_etr_passthrough.py
"""
import json
import os
import numpy as np
import pandas as pd

import build_substitution_analysis as B

CACHE = os.path.join(B.script_dir, '..', 'data', 'analysis', 'cache')
PANEL = os.path.join(B.script_dir, '..', 'data', 'analysis', 'output',
                     'tax_reconciliation_panel_v4.csv')
OUT_JSON = os.path.join(B.script_dir, '..', 'data', 'analysis', 'output',
                        'cash_etr_passthrough_summary.json')


def main():
    # 1. Exact deck pipeline: caches -> channels -> treatment -> balanced panel
    combined, long_df = B.load_data(CACHE)
    wide     = B.build_channels(long_df)
    intl_pre = B.build_treatment(wide)
    panel    = B.build_panel(combined, wide, intl_pre)

    # 2. Merge comp_txpd (cash taxes paid) from the canonical panel
    # [replication package] the canonical panel (v4) is not shipped; fall back to the
    # base panel, one row per (gvkey, recon_year). Affects slide A.31 only.
    _src = PANEL if os.path.exists(PANEL) else os.path.join(B.script_dir, '..', 'data', 'analysis', 'output', 'tax_reconciliation_panel_base.csv')
    cp = pd.read_csv(_src, usecols=['gvkey', 'recon_year', 'comp_txpd'],
                     low_memory=False).drop_duplicates(['gvkey', 'recon_year'])
    panel = panel.merge(cp, on=['gvkey', 'recon_year'], how='left')

    # 3. Build cash ETR (pp), winsorized like etr_w (clip +/-100)
    #    Panel already restricts to comp_pi > 1 ($1M), so the denominator is
    #    bounded away from zero.
    panel['cash_etr_w'] = (100.0 * panel['comp_txpd'] / panel['comp_pi']).clip(-100, 100)

    n_obs = panel['cash_etr_w'].notna().sum()
    print(f"\ncomp_txpd coverage in regression panel: {n_obs:,}/{len(panel):,} "
          f"({100*n_obs/len(panel):.1f}%)")
    print("cash_etr_w summary (pp):")
    print(panel['cash_etr_w'].describe().round(2).to_string())

    # 4. Pooled DiD, baseline spec (firm FE + industry x year FE)
    print("\n" + "=" * 64)
    print("Pooled DiD: beta on (intl_pre x post), firm + industry x year FE")
    print("=" * 64)
    rows = [
        ('ETR (recon, etr_w)  [deck headline]', 'etr_w', 'book_etr'),
        ('Cash ETR (txpd/PI)               ', 'cash_etr_w', 'cash_etr'),
    ]
    summary = {'coverage': {'nonmissing': int(n_obs), 'panel_rows': int(len(panel)),
                            'pct': round(100 * n_obs / len(panel), 1)}}
    for label, outc, key in rows:
        r = B.run_pooled_did(panel, outc, treat_col='intl_pre', ind_yr_fe=True)
        t = r['coef'] / r['se'] if r['se'] else np.nan
        print(f"{label}: beta={r['coef']:+.4f}  SE={r['se']:.4f}  "
              f"t={t:+.2f}{B.stars(abs(t))}  N={r['n']:,}  firms={r['G']:,}")
        summary[key] = {**r, 't': float(t), 'stars': B.stars(abs(t))}

    # 5. Event study for the cash-ETR outcome
    print("\n" + "=" * 64)
    print("Event study (cash ETR), firm + industry x year FE, omit 2016")
    print("=" * 64)
    ev = B.run_panel_event(panel, 'cash_etr_w', treat_col='intl_pre', ind_yr_fe=True)
    if not ev.empty:
        for _, row in ev.iterrows():
            print(f"  {int(row['year'])}:  beta={row['beta']:+.4f}  "
                  f"SE={row['se']:.4f}")
        summary['cash_etr_event'] = ev[['year', 'beta', 'se']].to_dict('records')

    with open(OUT_JSON, 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")


if __name__ == '__main__':
    main()
