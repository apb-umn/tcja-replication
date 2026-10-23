#!/usr/bin/env python3
"""
build_substitution_real_outcomes.py

Extension of the substitution analysis: did TCJA's closure of the international
tax-planning channel translate into a real shift in where firms book income
(domestic vs foreign PI)?

Uses the identical sample and treatment as build_substitution_analysis.py:
  - 1,277-firm balanced panel, 2013–2022, profitable, Finance excluded
  - intl_pre: ±15 pp per-obs winsor → pre-period mean → [p1,p99] clip
  - Event years 2013–2022, omit 2016, post = 2019–2022 (excl 2017/18)

Outcomes:
  1. Foreign PI share = pifo / (pidom + pifo), clipped [0,1]
  2. Domestic PI intensity = pidom / lag(at)  (where booked, % of assets)
  3. Foreign PI intensity  = pifo / lag(at)
  4. Log foreign PI share = log(foreign share + 0.01)
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

from sic_industry import industry  # canonical SIC -> industry classifier

warnings.filterwarnings('ignore')
mpl.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 100})

script_dir = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'figures', 'substitution')
TBL_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'tables',  'substitution')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

PRE_YEARS   = list(range(2013, 2017))   # 2013–2016
POST_YEARS  = list(range(2019, 2023))   # 2019–2022
EVENT_YEARS = list(range(2013, 2023))
OMIT_YEAR   = 2016
WIN_CH      = 15.0
EPS         = 0.01
WEDGE_SHARE_MIN = 0.01

# ──────────────────────────────────────────────────────────────────────────────

def stars(t):
    a = abs(t)
    return '***' if a>2.576 else ('**' if a>1.96 else ('*' if a>1.645 else ''))


def demean_twfe(df, cols, firm_col='gvkey', year_col='recon_year',
                tol=1e-9, max_iter=500):
    out = df[cols].copy().astype(float)
    for _ in range(max_iter):
        prev = out.values.copy()
        for col in cols:
            out[col] -= out[col].groupby(df[firm_col]).transform('mean')
            out[col] -= out[col].groupby(df[year_col]).transform('mean')
        if np.abs(out.values - prev).max() < tol:
            break
    return out


def demean_ind_yr(df, cols, firm_col='gvkey', ind_col='industry',
                  year_col='recon_year', tol=1e-9, max_iter=500):
    df = df.copy()
    df['_ind_yr'] = df[ind_col].astype(str) + '_' + df[year_col].astype(str)
    out = df[cols].copy().astype(float)
    for _ in range(max_iter):
        prev = out.values.copy()
        for col in cols:
            out[col] -= out[col].groupby(df[firm_col]).transform('mean')
            out[col] -= out[col].groupby(df['_ind_yr']).transform('mean')
        if np.abs(out.values - prev).max() < tol:
            break
    return out


def run_pooled_did(panel, outcome, treat_col='intl_pre', ind_yr_fe=False):
    excl  = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [yr for yr in EVENT_YEARS if yr not in excl]
    mask  = (panel['recon_year'].isin(years) &
             panel[outcome].notna() & panel[treat_col].notna())
    sub   = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return {'coef': np.nan, 'se': np.nan, 'n': 0, 'G': 0}

    sub['_post'] = sub['recon_year'].isin(POST_YEARS).astype(float)
    sub['_Di']   = sub[treat_col] * sub['_post']
    cols  = [outcome, '_Di']

    dm = demean_ind_yr(sub, cols) if ind_yr_fe else demean_twfe(sub, cols)
    mod = OLS(dm[outcome].values, dm[['_Di']].values).fit()
    cv  = cov_cluster(mod, sub['gvkey'].values)
    return {'coef': float(mod.params[0]), 'se': float(np.sqrt(np.diag(cv))[0]),
            'n': len(sub), 'G': sub['gvkey'].nunique()}


def run_event_study(panel, outcome, treat_col='intl_pre', ind_yr_fe=False):
    plot_yrs = [yr for yr in sorted(EVENT_YEARS) if yr != OMIT_YEAR]
    mask = (panel['recon_year'].isin(EVENT_YEARS) &
            panel[outcome].notna() & panel[treat_col].notna())
    sub  = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return pd.DataFrame()
    for yr in plot_yrs:
        sub[f'_ix_{yr}'] = sub[treat_col] * (sub['recon_year'] == yr).astype(float)
    ix_cols = [f'_ix_{yr}' for yr in plot_yrs]
    dm      = demean_ind_yr(sub, [outcome] + ix_cols) if ind_yr_fe else demean_twfe(sub, [outcome] + ix_cols)
    mod     = OLS(dm[outcome].values, dm[ix_cols].values).fit()
    cv      = cov_cluster(mod, sub['gvkey'].values)
    cl_se   = np.sqrt(np.diag(cv))

    rows = [{'year': yr, 'beta': float(mod.params[i]), 'se': float(cl_se[i]),
             'lower': float(mod.params[i] - 1.96*cl_se[i]),
             'upper': float(mod.params[i] + 1.96*cl_se[i])}
            for i, yr in enumerate(plot_yrs)]
    rows.append({'year': OMIT_YEAR, 'beta': 0., 'se': 0., 'lower': 0., 'upper': 0.})
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'combined.pkl'), 'rb') as f:
        combined = pickle.load(f)
    with open(os.path.join(cache := os.path.join(script_dir, '..', 'data', 'analysis', 'cache'), 'long.pkl'), 'rb') as f:
        long_df = pickle.load(f)
    pi_csv = os.path.join(script_dir, '..', 'data', 'analysis', 'output', 'tax_reconciliation_panel_base.csv')
    pi = pd.read_csv(pi_csv,
                     usecols=['gvkey','recon_year','comp_pidom','comp_pifo','comp_at'],
                     low_memory=False).drop_duplicates(['gvkey','recon_year'])
    print(f'  combined: {len(combined):,} rows, {combined["gvkey"].nunique():,} firms')
    print(f'  long_df:  {len(long_df):,} rows')
    print(f'  pi merge: {len(pi):,} rows (pidom/pifo)')
    return combined, long_df, pi


def build_channels_and_treatment(long_df, win_ch=WIN_CH):
    """Mirrors build_substitution_analysis: channel wide + intl_pre treatment.

    `win_ch` parametrizes the per-firm-year-channel winsorization. None
    disables it (no-per-obs-clip variant in robustness ladder).
    """
    fy = (long_df.groupby(['gvkey','recon_year','group'])['cat_val']
          .sum().reset_index(name='val'))
    if win_ch is None:
        fy['val_w'] = fy['val']
    else:
        fy['val_w'] = fy['val'].clip(-win_ch, win_ch)
    intl_fy = (fy[fy['group'] == 'International']
               .groupby(['gvkey','recon_year'])['val_w'].sum())
    # Eligibility: ≥2 pre-period obs in any group
    any_obs = fy.groupby(['gvkey','recon_year']).size()
    pre_any = any_obs[any_obs.index.get_level_values('recon_year').isin(PRE_YEARS)]
    elig = pre_any.groupby('gvkey').size()
    elig = elig[elig >= 2].index
    # Pre-period mean International (zeros for firms with no International entries)
    pre_intl = (intl_fy[intl_fy.index.get_level_values('recon_year').isin(PRE_YEARS)
                        & intl_fy.index.get_level_values('gvkey').isin(elig)]
                .groupby('gvkey').mean())
    intl_pre = pd.Series(0.0, index=elig, name='intl_pre')
    intl_pre.update(pre_intl)
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)
    print(f'  Treatment built: {len(intl_pre):,} firms  '
          f'[p1,p99]=[{p1:.2f},{p99:.2f}]  median={intl_pre.median():.2f}')
    return intl_pre


def build_intl_yearly(long_df, win_ch=WIN_CH):
    """Firm-year International contribution, winsorized the same way as substitution analysis."""
    fy = (long_df.groupby(['gvkey','recon_year','group'])['cat_val']
          .sum().reset_index(name='val'))
    if win_ch is None:
        fy['val_w'] = fy['val']
    else:
        fy['val_w'] = fy['val'].clip(-win_ch, win_ch)
    intl = (fy[fy['group'] == 'International']
            [['gvkey','recon_year','val_w']]
            .rename(columns={'val_w': 'International'}))
    return intl


def build_panel(combined, pi, intl_pre, intl_yearly):
    """Same balanced panel as substitution analysis, plus pi composition."""
    panel = combined[combined['recon_year'].isin(EVENT_YEARS)].copy()
    panel['industry'] = panel['comp_sic'].apply(industry)
    panel = panel[panel['industry'] != 'Finance']
    if 'comp_pi' in panel.columns:
        panel = panel[panel['comp_pi'] > 1]
    panel = panel.merge(intl_pre.reset_index(), on='gvkey', how='inner')
    panel = panel.merge(pi, on=['gvkey','recon_year'], how='left')
    panel = panel.merge(intl_yearly, on=['gvkey','recon_year'], how='left')
    panel['International'] = panel['International'].fillna(0.0)
    # Balanced filter
    fy = panel.groupby('gvkey')['recon_year'].apply(set)
    has_pre  = fy.apply(lambda s: len(s & set(PRE_YEARS))  >= 2)
    has_post = fy.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok = fy[has_pre & has_post].index
    panel = panel[panel['gvkey'].isin(ok)].copy()

    # ── Real outcomes ─────────────────────────────────────────────────────────
    # Foreign PI share: guard against zero/negative total PI
    pi_tot = panel['comp_pidom'].fillna(0) + panel['comp_pifo'].fillna(0)
    valid  = (panel['comp_pidom'].notna() & panel['comp_pifo'].notna() &
              (pi_tot.abs() > 0.5))
    panel['foreign_share'] = np.where(valid, panel['comp_pifo'] / pi_tot, np.nan)
    panel['foreign_share'] = panel['foreign_share'].clip(0, 1)

    panel['foreign_share_log'] = np.log(panel['foreign_share'] + EPS)

    # Implied effective US-foreign tax wedge:
    # International contribution ≈ - foreign_share × wedge
    panel['tax_wedge'] = np.where(
        panel['foreign_share'] > WEDGE_SHARE_MIN,
        -panel['International'] / panel['foreign_share'],
        np.nan,
    )
    panel['tax_wedge'] = panel['tax_wedge'].replace([np.inf, -np.inf], np.nan)
    if panel['tax_wedge'].notna().any():
        p1, p99 = panel['tax_wedge'].quantile([0.01, 0.99])
        panel['tax_wedge_w'] = panel['tax_wedge'].clip(p1, p99)
    else:
        panel['tax_wedge_w'] = np.nan

    # PI intensities (levels / lag assets, winsorized)
    panel = panel.sort_values(['gvkey','recon_year'])
    panel['lag_at']    = panel.groupby('gvkey')['comp_at'].shift(1)
    panel['pidom_int'] = np.where(panel['lag_at'] > 1,
                                   panel['comp_pidom'] / panel['lag_at'], np.nan)
    panel['pifo_int']  = np.where(panel['lag_at'] > 1,
                                   panel['comp_pifo']  / panel['lag_at'], np.nan)
    for col in ['pidom_int', 'pifo_int']:
        panel[col] = panel[col].clip(-0.5, 0.5)

    print(f'  Panel built: {panel["gvkey"].nunique():,} firms, {len(panel):,} firm-years')
    n_fs = panel.dropna(subset=['foreign_share'])['gvkey'].nunique()
    print(f'  Firms with foreign_share observed: {n_fs:,}')
    n_tw = panel.dropna(subset=['tax_wedge_w'])['gvkey'].nunique()
    print(f'  Firms with implied tax wedge observed: {n_tw:,}')
    return panel


def compute_wedge_elasticity(panel):
    """Common-sample log-share semi-elasticity with respect to the implied tax wedge."""
    sub = panel.dropna(subset=['foreign_share_log', 'tax_wedge_w', 'intl_pre']).copy()
    base_log = run_pooled_did(sub, 'foreign_share_log', ind_yr_fe=False)
    base_wedge = run_pooled_did(sub, 'tax_wedge_w', ind_yr_fe=False)
    ind_log = run_pooled_did(sub, 'foreign_share_log', ind_yr_fe=True)
    ind_wedge = run_pooled_did(sub, 'tax_wedge_w', ind_yr_fe=True)

    def ratio(num, den):
        if np.isnan(num.get('coef', np.nan)) or np.isnan(den.get('coef', np.nan)) or abs(den['coef']) < 1e-9:
            return np.nan
        return num['coef'] / abs(den['coef'])

    return {
        'sample_firms': base_wedge['G'],
        'sample_obs': base_wedge['n'],
        'event_window_firms': sub['gvkey'].nunique(),
        'event_window_obs': len(sub),
        'baseline_log': base_log,
        'baseline_wedge': base_wedge,
        'baseline_ratio': ratio(base_log, base_wedge),
        'indyr_log': ind_log,
        'indyr_wedge': ind_wedge,
        'indyr_ratio': ratio(ind_log, ind_wedge),
    }


def compute_level_ratio(panel):
    """Common-sample ratio of level betas: foreign-share change over International-component change."""
    sub = panel.dropna(subset=['foreign_share', 'International', 'intl_pre']).copy()
    base_share = run_pooled_did(sub, 'foreign_share', ind_yr_fe=False)
    base_intl = run_pooled_did(sub, 'International', ind_yr_fe=False)
    ind_share = run_pooled_did(sub, 'foreign_share', ind_yr_fe=True)
    ind_intl = run_pooled_did(sub, 'International', ind_yr_fe=True)

    def ratio(num, den):
        if np.isnan(num.get('coef', np.nan)) or np.isnan(den.get('coef', np.nan)) or abs(den['coef']) < 1e-9:
            return np.nan
        return num['coef'] / abs(den['coef'])

    return {
        'sample_firms': base_share['G'],
        'sample_obs': base_share['n'],
        'event_window_firms': sub['gvkey'].nunique(),
        'event_window_obs': len(sub),
        'baseline_share': base_share,
        'baseline_intl': base_intl,
        'baseline_ratio': ratio(base_share, base_intl),
        'indyr_share': ind_share,
        'indyr_intl': ind_intl,
        'indyr_ratio': ratio(ind_share, ind_intl),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Event-study figure: 3-panel (foreign share, pidom intensity, pifo intensity)
# ──────────────────────────────────────────────────────────────────────────────

def fig_real_event(panel):
    print('  Event study: foreign share (level and log)...')
    outcomes = [
        ('foreign_share', 'Foreign PI share $= \\pi^{fo}/(\\pi^{dom}+\\pi^{fo})$'),
        ('foreign_share_log', r'$\log(\mathrm{foreign\ share}+0.01)$'),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
    pooled = {}

    for ax, (y, title) in zip(axes, outcomes):
        ev = run_event_study(panel, y, ind_yr_fe=True)
        pooled[y] = {
            'baseline': run_pooled_did(panel, y, ind_yr_fe=False),
            'ind_yr':   run_pooled_did(panel, y, ind_yr_fe=True),
        }
        if ev.empty:
            ax.set_title(f'{title} (skipped)')
            continue

        pre  = ev[ev['year'] <  2017]
        t17  = ev[ev['year'] == 2017]
        post = ev[ev['year'] >  2017]
        kw = dict(fmt='o', ms=5, lw=1.3, capsize=3)
        ax.errorbar(pre['year'],  pre['beta'],  yerr=1.96*pre['se'],
                    color='#888888', **kw, label='Pre-TCJA')
        if not t17.empty:
            ax.errorbar(t17['year'], t17['beta'], yerr=1.96*t17['se'],
                        fmt='s', color='darkorange', ms=6, lw=1.3, capsize=3,
                        label='2017 (transition)')
        ax.errorbar(post['year'], post['beta'], yerr=1.96*post['se'],
                    color='#2980b9', **kw, label='Post-TCJA')

        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.axvline(2017.5, color='black', ls='--', lw=1.0, alpha=0.55)
        ax.set_ylabel(r'$\hat{\beta}_t$', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)

        pri = pooled[y]['ind_yr']
        ti = abs(pri['coef']/pri['se']) if pri.get('se', 0) > 0 else 0
        ax.text(0.99, 0.94,
                f"Baseline spec: {pri['coef']:+.5f} (SE {pri['se']:.5f}){stars(ti)}\n"
                f"firm + industry$\\times$year FE",
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    axes[0].legend(fontsize=7, loc='lower left', ncol=3)
    axes[-1].set_xlabel('Year')
    fig.suptitle('Income-location event studies (substitution panel)\n'
                 r'Firm + industry$\times$year FE; $\hat{\beta}_t>0$ means heavier shifters reduced foreign share post-TCJA',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_real_outcomes_event.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    for y in ['pidom_int', 'pifo_int']:
        pooled[y] = {
            'baseline': run_pooled_did(panel, y, ind_yr_fe=False),
            'ind_yr':   run_pooled_did(panel, y, ind_yr_fe=True),
        }
    return pooled


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table: Pooled DiD by outcome × spec
# ──────────────────────────────────────────────────────────────────────────────

def write_table(pooled):
    def fc(r):
        if not r or np.isnan(r.get('coef', np.nan)):
            return '---'
        v, s = r['coef'], r['se']
        t = abs(v/s) if s > 0 else 0
        return f'${v:+.5f}^{{{stars(t)}}}$' if stars(t) else f'${v:+.5f}$'
    def fse(r):
        return f'$({r["se"]:.5f})$' if r and not np.isnan(r.get('se', np.nan)) else ''

    outcomes = [
        ('foreign_share', 'Foreign PI share'),
        ('foreign_share_log', r'$\log(\mathrm{foreign\ share}+0.01)$'),
        ('pidom_int',     r'$\pi^{\text{dom}}/\text{assets}$'),
        ('pifo_int',      r'$\pi^{\text{fo}}/\text{assets}$'),
    ]
    lines = [
        '\\begin{table}[htbp]', '\\centering',
        '\\caption{Real-outcome effects (substitution panel, 1{,}277 firms)}',
        '\\label{tab:real_outcomes}', '\\small',
        '\\begin{tabular}{lccc}', '\\toprule',
        ' & (1) Baseline & (2) + Ind$\\times$yr FE & $N$ firms \\\\',
        '\\midrule',
    ]
    for y_col, lbl in outcomes:
        p = pooled[y_col]['baseline']; q = pooled[y_col]['ind_yr']
        lines.append(f'{lbl} & {fc(p)} & {fc(q)} & {p.get("G",""):,} \\\\')
        lines.append(f' & {fse(p)} & {fse(q)} & \\\\')
    lines += [
        '\\bottomrule', '\\end{tabular}',
        '\\begin{tablenotes}\\footnotesize',
        '\\item Sample: identical to substitution analysis (profitable, non-Finance, balanced 2013--2022).',
        '\\item Treatment $\\bar{c}^{\\text{pre}}_{\\text{Intl}}$: pre-TCJA international contribution (2013--16 mean, pp); '
        '$\\pm 15$\\,pp per-obs winsor, $[p_1,p_{99}]$ firm-level clip.',
        '\\item Log-share outcome uses $\\log(\\mathrm{foreign\\ share}+0.01)$ so zero-share observations remain in sample.',
        '\\item DiD excludes 2017 and 2018 transition years.  Clustered SE at firm.',
        '\\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
        '\\end{tablenotes}', '\\end{table}',
    ]
    path = os.path.join(TBL_DIR, 'tbl_real_outcomes.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data...')
    combined, long_df, pi = load_data()

    print('\nBuilding treatment (mirrors substitution analysis)...')
    intl_pre = build_channels_and_treatment(long_df)
    intl_yearly = build_intl_yearly(long_df)

    print('\nBuilding panel...')
    panel = build_panel(combined, pi, intl_pre, intl_yearly)

    # Descriptive stats: where firms booked income pre vs post
    print('\nDescriptives (PI-weighted means, pre vs post):')
    for lbl, yrs in [('Pre (2013-16)', PRE_YEARS), ('Post (2019-22)', POST_YEARS)]:
        s = panel[panel['recon_year'].isin(yrs)]
        fs = s['foreign_share'].mean()
        pd_int = s['pidom_int'].mean()
        pf_int = s['pifo_int'].mean()
        print(f'  {lbl:<16}  foreign_share={fs:.3f}  pidom/at={pd_int:.4f}  pifo/at={pf_int:.4f}')

    print('\nRunning event study + pooled DiD...')
    pooled = fig_real_event(panel)

    print('\nPooled DiD summary:')
    print(f'{"Outcome":<18} {"Baseline β":>14} {"Ind×Yr β":>14} {"N firms":>10}')
    for y in ['foreign_share', 'foreign_share_log', 'pidom_int', 'pifo_int']:
        pb = pooled[y]['baseline']; pi_ = pooled[y]['ind_yr']
        print(f'  {y:<16} {pb["coef"]:>+13.5f}{stars(abs(pb["coef"]/pb["se"]) if pb["se"]>0 else 0):<4}'
              f'{pi_["coef"]:>+13.5f}{stars(abs(pi_["coef"]/pi_["se"]) if pi_["se"]>0 else 0):<4}'
              f' {pb["G"]:>10,}')

    level = compute_level_ratio(panel)
    print('\nInternational-closure level ratio (common sample)')
    print(f'  Sample: {level["sample_firms"]:,} firms, {level["sample_obs"]:,} firm-years')
    print(f'  Baseline β_share:          {level["baseline_share"]["coef"]:+.5f}  SE={level["baseline_share"]["se"]:.5f}')
    print(f'  Baseline β_International:  {level["baseline_intl"]["coef"]:+.5f}  SE={level["baseline_intl"]["se"]:.5f}')
    print(f'  Baseline ratio:            {100*level["baseline_ratio"]:+.3f} pp share per 1 pp International')
    print(f'  + Ind×year ratio:          {100*level["indyr_ratio"]:+.3f} pp share per 1 pp International')

    wedge = compute_wedge_elasticity(panel)
    print('\nImplied tax-wedge semi-elasticity (common sample)')
    print(f'  Sample: {wedge["sample_firms"]:,} firms, {wedge["sample_obs"]:,} firm-years')
    print(f'  Baseline β_log(share+0.01):  {wedge["baseline_log"]["coef"]:+.5f}  SE={wedge["baseline_log"]["se"]:.5f}')
    print(f'  Baseline β_wedge:           {wedge["baseline_wedge"]["coef"]:+.5f}  SE={wedge["baseline_wedge"]["se"]:.5f}')
    print(f'  Baseline semi-elasticity:   {wedge["baseline_ratio"]:+.5f} log points per 1 pp wedge')
    print(f'  + Ind×year semi-elasticity: {wedge["indyr_ratio"]:+.5f} log points per 1 pp wedge')

    print('\nWriting table...')
    write_table(pooled)
    print('\nDone.')
