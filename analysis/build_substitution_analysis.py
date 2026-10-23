#!/usr/bin/env python3
"""
build_substitution_analysis.py
Cross-channel substitution in corporate tax planning around TCJA.

Economic question: when TCJA closed the international profit-shifting channel,
did firms substitute into other ETR-reducing strategies, or accept higher ETRs?

Run:
    python build_substitution_analysis.py [--cache-dir ./cache]
"""

import argparse, os, pickle, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_cluster

from sic_industry import industry  # canonical SIC -> industry classifier

mpl.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 100})

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'figures', 'substitution')
TBL_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'tables',  'substitution')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

PRE_YEARS   = list(range(2013, 2017))   # 2013–2016
POST_YEARS  = list(range(2019, 2023))   # 2019–2022  (skip 2017, 2018 transition)
EVENT_YEARS = list(range(2013, 2023))   # full event window
OMIT_YEAR   = 2016
WIN_CH      = 15.0    # ±pp winsorization per firm-year-channel

# Map the 7-group `group` column values to canonical internal names
GROUP_MAP = {
    'International':  'International',
    'State & local':  'State_local',
    'Federal credits':'Federal_credits',
    'Permanent items':'Permanent_items',
    'Reserves & losses':'Reserves_losses',
    'TCJA one-time':  'TCJA_onetime',
    'Other':          'Other',
}

# Channels kept for substitution analysis
SUBST_CHANNELS = ['State_local', 'Federal_credits', 'Permanent_items', 'Other']
ALL_CHANNELS   = ['International'] + SUBST_CHANNELS

CH_COLORS = {
    'International':   '#c0392b',
    'State_local':     '#2980b9',
    'Federal_credits': '#16a085',
    'Permanent_items': '#e67e22',
    'Other':           '#7f8c8d',
}
CH_LABELS = {
    'International':   'International',
    'State_local':     'State & local',
    'Federal_credits': 'Federal credits',
    'Permanent_items': 'Permanent items',
    'Other':           'Other',
}
# LaTeX-escaped versions for tables
CH_LABELS_TEX = {
    'International':   'International',
    'State_local':     'State \\& local',
    'Federal_credits': 'Federal credits',
    'Permanent_items': 'Permanent items',
    'Other':           'Other',
    'ETR':             '$\\Delta$ETR',
}

IND_ORDER   = ['Tech', 'Pharma/Biotech', 'Manufacturing', 'Retail', 'Oil & Gas', 'Other']
IND_COLORS  = dict(zip(IND_ORDER, plt.cm.tab10(np.linspace(0, 0.9, len(IND_ORDER)))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def stars(t):
    if t > 2.576: return '***'
    if t > 1.960: return '**'
    if t > 1.645: return '*'
    return ''


def demean_twfe(df, cols, firm_col='gvkey', year_col='recon_year',
                tol=1e-9, max_iter=500):
    """Alternating-projections within-estimator (firm + year FE)."""
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
    """Alternating-projections within-estimator (firm + industry×year FE)."""
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


def run_cs_reg(y_ser, x_ser, controls=None):
    """Cross-sectional OLS with HC1 robust SE. Returns dict."""
    idx = y_ser.dropna().index.intersection(x_ser.dropna().index)
    if controls:
        for c in controls:
            idx = idx.intersection(c.dropna().index)
    if len(idx) < 25:
        return {'coef': np.nan, 'se': np.nan, 'n': 0, 'r2': np.nan}
    y = y_ser.loc[idx].values
    xparts = [x_ser.loc[idx].values]
    if controls:
        for c in controls:
            xparts.append(c.loc[idx].values)
    X = add_constant(np.column_stack(xparts))
    m = OLS(y, X).fit(cov_type='HC1')
    return {'coef': float(m.params[1]), 'se': float(m.bse[1]),
            'n': int(m.nobs), 'r2': float(m.rsquared)}


def run_panel_event(panel, outcome, treat_col='intl_pre', ind_yr_fe=False):
    """TWFE event study via alternating projections. Returns DataFrame."""
    plot_yrs = [yr for yr in sorted(EVENT_YEARS) if yr != OMIT_YEAR]
    mask = (panel['recon_year'].isin(EVENT_YEARS) &
            panel[outcome].notna() & panel[treat_col].notna())
    sub = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return pd.DataFrame()

    for yr in plot_yrs:
        sub[f'_ix_{yr}'] = sub[treat_col] * (sub['recon_year'] == yr).astype(float)
    ix_cols  = [f'_ix_{yr}' for yr in plot_yrs]
    all_cols = [outcome] + ix_cols

    if ind_yr_fe:
        sub_dm = demean_ind_yr(sub, all_cols)
    else:
        sub_dm = demean_twfe(sub, all_cols)

    X      = sub_dm[ix_cols].values
    y      = sub_dm[outcome].values
    groups = sub['gvkey'].values
    mod    = OLS(y, X).fit()
    cl_cov = cov_cluster(mod, groups)
    cl_se  = np.sqrt(np.diag(cl_cov))

    records = [{'year': yr, 'beta': float(mod.params[i]), 'se': float(cl_se[i]),
                'lower': float(mod.params[i] - 1.96*cl_se[i]),
                'upper': float(mod.params[i] + 1.96*cl_se[i])}
               for i, yr in enumerate(plot_yrs)]
    records.append({'year': OMIT_YEAR, 'beta': 0., 'se': 0., 'lower': 0., 'upper': 0.})
    return pd.DataFrame(records).sort_values('year').reset_index(drop=True)


def run_pooled_did(panel, outcome, treat_col='intl_pre', ind_yr_fe=False):
    """Pooled DiD (excludes 2017, 2018, OMIT_YEAR). Returns dict."""
    excl  = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [yr for yr in EVENT_YEARS if yr not in excl]
    mask  = (panel['recon_year'].isin(years) &
             panel[outcome].notna() & panel[treat_col].notna())
    sub   = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return {'coef': np.nan, 'se': np.nan, 'n': 0, 'G': 0}

    sub['_post'] = sub['recon_year'].isin(POST_YEARS).astype(float)
    sub['_Di']   = sub[treat_col] * sub['_post']
    all_cols     = [outcome, '_Di']

    if ind_yr_fe:
        sub_dm = demean_ind_yr(sub, all_cols)
    else:
        sub_dm = demean_twfe(sub, all_cols)

    X      = sub_dm[['_Di']].values
    y      = sub_dm[outcome].values
    groups = sub['gvkey'].values
    mod    = OLS(y, X).fit()
    cl_cov = cov_cluster(mod, groups)
    cl_se  = np.sqrt(np.diag(cl_cov))
    return {'coef': float(mod.params[0]), 'se': float(cl_se[0]),
            'n': len(sub), 'G': sub['gvkey'].nunique()}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cache_dir):
    print('Loading caches...')
    with open(os.path.join(cache_dir, 'combined.pkl'), 'rb') as f:
        combined = pickle.load(f)
    with open(os.path.join(cache_dir, 'long.pkl'), 'rb') as f:
        long_df = pickle.load(f)
    print(f'  combined: {len(combined):,} rows, {combined["gvkey"].nunique():,} firms')
    print(f'  long_df:  {len(long_df):,} rows')
    assert 'group' in long_df.columns, "long_df missing 'group' column"
    assert 'cat_val' in long_df.columns, "long_df missing 'cat_val' column"
    return combined, long_df


def build_channels(long_df, win_ch=WIN_CH):
    """Map groups to canonical names, winsorize, pivot to firm-year x channel.

    `win_ch` is the per-firm-year-channel winsorization. None disables it
    (lets values pass through unclipped) and is used by the no-per-obs-clip
    robustness variant in build_robustness_ladder.py.
    """
    print('Building channel contributions...')
    df = long_df.copy()
    df['channel'] = df['group'].map(GROUP_MAP).fillna('Other')

    # Sum to firm-year-channel (handles multiple items per group)
    fy = (df.groupby(['gvkey', 'recon_year', 'channel'])['cat_val']
          .sum().reset_index(name='val'))
    if win_ch is None:
        fy['val_w'] = fy['val']
    else:
        fy['val_w'] = fy['val'].clip(-win_ch, win_ch)

    wide = fy.pivot_table(index=['gvkey', 'recon_year'], columns='channel',
                          values='val_w', fill_value=0).reset_index()
    wide.columns.name = None

    # Ensure all channels exist
    for ch in ALL_CHANNELS + ['TCJA_onetime', 'Reserves_losses']:
        if ch not in wide.columns:
            wide[ch] = 0.0

    print(f'  Channel panel: {len(wide):,} firm-years, {wide["gvkey"].nunique():,} firms')
    return wide


def build_treatment(wide):
    """Pre-period (2013–2016) average International contribution, winsorized."""
    print('Building treatment variable...')
    pre    = wide[wide['recon_year'].isin(PRE_YEARS)]
    counts = pre.groupby('gvkey').size()
    elig   = counts[counts >= 2].index
    intl_pre = (pre[pre['gvkey'].isin(elig)]
                .groupby('gvkey')['International']
                .mean()
                .rename('intl_pre'))
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)
    print(f'  Firms with ≥2 pre-period obs: {len(intl_pre):,}')
    print(f'  intl_pre: median={intl_pre.median():.2f}  '
          f'P25={intl_pre.quantile(0.25):.2f}  P75={intl_pre.quantile(0.75):.2f}')
    return intl_pre


def build_panel(combined, wide, intl_pre):
    """Balanced TWFE panel for event studies."""
    print('Building event panel...')
    panel = combined[combined['recon_year'].isin(EVENT_YEARS)].copy()
    panel['industry'] = panel['comp_sic'].apply(industry)
    panel = panel[panel['industry'] != 'Finance']

    # Profitable subsample
    if 'comp_pi' in panel.columns:
        panel = panel[panel['comp_pi'] > 1]

    # Merge treatment
    panel = panel.merge(intl_pre.reset_index(), on='gvkey', how='inner')

    # Merge channel contributions
    panel = panel.merge(wide, on=['gvkey', 'recon_year'], how='left')
    for ch in ALL_CHANNELS:
        if ch in panel.columns:
            panel[ch] = panel[ch].fillna(0.0)

    # Balanced filter: ≥2 pre AND ≥2 post observations per firm
    fy      = panel.groupby('gvkey')['recon_year'].apply(set)
    has_pre  = fy.apply(lambda s: len(s & set(PRE_YEARS))  >= 2)
    has_post = fy.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok      = fy[has_pre & has_post].index
    panel   = panel[panel['gvkey'].isin(ok)].copy()

    print(f'  Balanced panel: {panel["gvkey"].nunique():,} firms, {len(panel):,} firm-years')
    return panel


def build_firm_level(panel, intl_pre):
    """Firm-level pre/post averages for each channel (for Spec 1)."""
    fl = intl_pre.to_frame()

    for ch in ALL_CHANNELS:
        if ch not in panel.columns:
            continue
        pre  = (panel[panel['recon_year'].isin(PRE_YEARS)]
                .groupby('gvkey')[ch].mean().rename(f'{ch}_pre'))
        post = (panel[panel['recon_year'].isin(POST_YEARS)]
                .groupby('gvkey')[ch].mean().rename(f'{ch}_post'))
        fl = fl.join(pre, how='left').join(post, how='left')
        fl[f'{ch}_delta'] = fl[f'{ch}_post'] - fl[f'{ch}_pre']

    # ETR
    if 'etr_w' in panel.columns:
        pre_etr  = (panel[panel['recon_year'].isin(PRE_YEARS)]
                    .groupby('gvkey')['etr_w'].mean().rename('etr_pre'))
        post_etr = (panel[panel['recon_year'].isin(POST_YEARS)]
                    .groupby('gvkey')['etr_w'].mean().rename('etr_post'))
        fl = fl.join(pre_etr, how='left').join(post_etr, how='left')
        fl['etr_delta'] = fl['etr_post'] - fl['etr_pre']

    # Industry
    ind = panel.drop_duplicates('gvkey').set_index('gvkey')['industry']
    fl  = fl.join(ind, how='left')

    # Size proxy: log(avg sales in pre-period)
    if 'comp_sale' in panel.columns:
        sale = (panel[panel['recon_year'].isin(PRE_YEARS)]
                .groupby('gvkey')['comp_sale'].mean())
        fl['log_sale'] = np.log(sale.clip(lower=0.1))
        fl = fl.join(fl['log_sale'], how='left', rsuffix='_dup')
        fl = fl[[c for c in fl.columns if not c.endswith('_dup')]]

    fl = fl.dropna(subset=['International_delta'])
    print(f'  Firm-level cross-section: {len(fl):,} firms')
    return fl


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Channel time series
# ─────────────────────────────────────────────────────────────────────────────

def fig_channel_timeseries(combined, wide):
    print('  Fig 1: channel time series...')
    df = combined.merge(wide, on=['gvkey', 'recon_year'], how='inner')
    df['industry'] = df['comp_sic'].apply(industry)
    df = df[(df['industry'] != 'Finance') &
            df['recon_year'].between(2010, 2023)]
    if 'comp_pi' in df.columns:
        df = df[df['comp_pi'] > 1]

    records = []
    for yr, grp in df.groupby('recon_year'):
        w = grp['comp_pi'].clip(lower=0)
        tot = w.sum()
        if tot == 0:
            continue
        row = {'year': yr}
        for ch in ALL_CHANNELS:
            if ch in grp.columns:
                row[ch] = float((grp[ch] * w).sum() / tot)
        records.append(row)
    ts = pd.DataFrame(records).set_index('year')

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for ch in ALL_CHANNELS:
        if ch in ts.columns:
            ax.plot(ts.index, ts[ch], color=CH_COLORS[ch], lw=2.0,
                    label=CH_LABELS[ch], marker='o', ms=3)
    ax.axvline(2017.5, color='black', ls='--', lw=1.2, alpha=0.7, label='TCJA')
    ax.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax.axvspan(2013, 2016.5, alpha=0.06, color='steelblue')
    ax.axvspan(2018.5, 2022, alpha=0.06, color='coral')
    ax.text(2014.5, ax.get_ylim()[1]*0.95 if len(ts) else 1,
            'Pre', ha='center', fontsize=8, color='steelblue', va='top')
    ax.text(2020.5, ax.get_ylim()[1]*0.95 if len(ts) else 1,
            'Post', ha='center', fontsize=8, color='coral', va='top')
    ax.set_xlabel('Year')
    ax.set_ylabel('PI-weighted mean contribution (pp)')
    ax.set_title('Tax-planning channel contributions, 2010–2023\n'
                 '(Finance excluded; profitable subsample; winsorized ±15 pp)', fontweight='bold')
    ax.legend(ncol=3, fontsize=8, loc='lower left')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_channel_timeseries.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    return ts


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: First-stage scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig_firststage_intl(fl):
    print('  Fig 2: first-stage scatter...')
    sub = fl.dropna(subset=['intl_pre', 'International_delta'])

    r = run_cs_reg(sub['International_delta'].clip(-10, 10), sub['intl_pre'])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ind_name in IND_ORDER:
        g = sub[sub['industry'] == ind_name]
        if len(g) == 0:
            continue
        ax.scatter(g['intl_pre'], g['International_delta'].clip(-10, 10),
                   s=14, alpha=0.35, color=IND_COLORS.get(ind_name, 'gray'),
                   label=f'{ind_name} ({len(g)})')

    x_line = np.linspace(sub['intl_pre'].min(), sub['intl_pre'].max(), 200)
    X_fit  = add_constant(sub['intl_pre'].values)
    m_fit  = OLS(sub['International_delta'].clip(-10, 10).values, X_fit).fit()
    ax.plot(x_line, m_fit.params[0] + m_fit.params[1]*x_line,
            color='black', lw=2, zorder=5, label='OLS fit')
    ax.plot(x_line, -x_line, color='gray', ls='--', lw=1.2,
            alpha=0.7, label='Slope $=-1$ (full closure)')

    t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
    ax.text(0.04, 0.97,
            f"Slope = {r['coef']:+.3f} (SE {r['se']:.3f}){stars(t)}\n"
            f"$R^2 = {r['r2']:.3f}$,  $N = {r['n']:,}$",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.axhline(0, color='black', lw=0.7, alpha=0.4)
    ax.axvline(0, color='black', lw=0.7, alpha=0.4)
    ax.set_xlabel('Pre-TCJA international contribution (avg 2013–2016, pp)', fontsize=10)
    ax.set_ylabel('$\\Delta$ International (post $-$ pre avg, pp)', fontsize=10)
    ax.set_title('First stage: closing of the international channel\n'
                 'Slope $\\approx -1$ = channel closed fully for heavy shifters', fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_firststage_intl.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Substitution event study (4 panels)
# ─────────────────────────────────────────────────────────────────────────────

def fig_substitution_event(event_results, pooled_results):
    print('  Fig 3: substitution event study...')
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    for ax, ch in zip(axes, SUBST_CHANNELS):
        ev = event_results.get(ch, pd.DataFrame())
        if ev.empty:
            ax.set_title(CH_LABELS[ch])
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
                    color=CH_COLORS[ch], **kw, label='Post-TCJA')

        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.axvline(2017.5, color='black', ls='--', lw=1.0, alpha=0.55)
        ax.set_ylabel('$\\hat{\\beta}_t$ (pp / pp)', fontsize=8)
        ax.set_title(CH_LABELS[ch], fontsize=10, fontweight='bold',
                     color=CH_COLORS[ch])
        ax.grid(axis='y', alpha=0.25)

        pr = pooled_results.get(ch, {})
        if pr and not np.isnan(pr.get('coef', np.nan)):
            t = abs(pr['coef']/pr['se']) if pr.get('se', 0) > 0 else 0
            ax.text(0.97, 0.94,
                    f"Pooled DiD: {pr['coef']:+.3f} ({stars(t) or 'n.s.'})",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    axes[0].legend(fontsize=7, loc='lower left', ncol=3)
    axes[-1].set_xlabel('Year')
    fig.suptitle('Substitution event study\n'
                 'Firm + industry$\\times$year FE; positive $\\hat{\\beta}_t$ means firms with more negative '
                 'pre-TCJA international contribution expanded this channel post-TCJA',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_substitution_event.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Substitution scatter matrix
# ─────────────────────────────────────────────────────────────────────────────

def fig_substitution_scatter(fl):
    print('  Fig 4: substitution scatter matrix...')
    sub  = fl.dropna(subset=['International_delta'])
    x_delt = sub['International_delta'].clip(-10, 10)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    x_ref = np.linspace(x_delt.min() - 0.3, x_delt.max() + 0.3, 200)

    for ax, ch in zip(axes, SUBST_CHANNELS):
        delta_col = f'{ch}_delta'
        if delta_col not in sub.columns:
            ax.set_visible(False)
            continue
        sub2 = sub.dropna(subset=[delta_col])
        x    = sub2['International_delta'].clip(-10, 10)
        y    = sub2[delta_col].clip(-10, 10)

        for ind_name in IND_ORDER:
            g = sub2[sub2['industry'] == ind_name]
            if len(g) == 0:
                continue
            ax.scatter(g['International_delta'].clip(-10, 10),
                       g[delta_col].clip(-10, 10),
                       s=12, alpha=0.3, color=IND_COLORS.get(ind_name, 'gray'),
                       label=ind_name)

        # OLS fit line
        r = run_cs_reg(pd.Series(y.values), pd.Series(x.values))
        if not np.isnan(r['coef']):
            m_fit = OLS(y.values, add_constant(x.values)).fit()
            ax.plot(x_ref, m_fit.params[0] + m_fit.params[1]*x_ref,
                    color=CH_COLORS[ch], lw=2.2, zorder=5)

        # Reference: perfect 1:1 substitution (slope = -1)
        ax.plot(x_ref, -x_ref, color='gray', ls='--', lw=1.2, alpha=0.65,
                label='$-$1:1 ref.')
        ax.axhline(0, color='black', lw=0.6, alpha=0.35)
        ax.axvline(0, color='black', lw=0.6, alpha=0.35)

        t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
        ax.text(0.04, 0.97,
                f"$\\beta = {r['coef']:+.3f}$ (SE {r['se']:.3f}){stars(t)}\n"
                f"$R^2 = {r['r2']:.3f}$",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_xlabel('$\\Delta$ International (pp)', fontsize=9)
        ax.set_ylabel(f'$\\Delta$ {CH_LABELS[ch]} (pp)', fontsize=9)
        ax.set_title(CH_LABELS[ch], fontsize=10, fontweight='bold',
                     color=CH_COLORS[ch])

    axes[0].legend(fontsize=7, loc='upper left', ncol=2)
    fig.suptitle('Cross-channel substitution: $\\Delta$(channel $k$) vs $\\Delta$(International)\n'
                 'Dashed line = $-1:1$ reference (perfect substitution)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_substitution_scatter.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Substitution matrix heatmap (industry × channel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_substitution_matrix(fl):
    print('  Fig 5: substitution matrix heatmap...')
    coef_mat  = pd.DataFrame(index=IND_ORDER, columns=SUBST_CHANNELS, dtype=float)
    stars_mat = pd.DataFrame(index=IND_ORDER, columns=SUBST_CHANNELS, dtype=object)

    for ind_name in IND_ORDER:
        sub_i = fl[fl['industry'] == ind_name]
        for ch in SUBST_CHANNELS:
            delta_col = f'{ch}_delta'
            if delta_col not in sub_i.columns or len(sub_i.dropna(subset=[delta_col])) < 25:
                coef_mat.loc[ind_name, ch]  = np.nan
                stars_mat.loc[ind_name, ch] = 'n.a.'
                continue
            r = run_cs_reg(sub_i[delta_col], sub_i['intl_pre'])
            coef_mat.loc[ind_name, ch]  = r['coef']
            t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
            stars_mat.loc[ind_name, ch] = stars(t)

    valid_rows = [i for i in IND_ORDER if not coef_mat.loc[i].isna().all()]
    coef_val   = coef_mat.loc[valid_rows].astype(float)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(valid_rows)*0.75 + 1.5)))
    vmax = max(0.4, float(coef_val.abs().max().max()))
    im   = ax.imshow(coef_val.values, aspect='auto',
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(SUBST_CHANNELS)))
    ax.set_xticklabels([CH_LABELS[c] for c in SUBST_CHANNELS], fontsize=10)
    ax.set_yticks(range(len(valid_rows)))
    ax.set_yticklabels(valid_rows, fontsize=10)

    for i, ind_name in enumerate(valid_rows):
        for j, ch in enumerate(SUBST_CHANNELS):
            val = coef_mat.loc[ind_name, ch]
            st  = stars_mat.loc[ind_name, ch]
            if pd.isna(val):
                ax.text(j, i, 'n.a.', ha='center', va='center', fontsize=9)
            else:
                txt_col = 'white' if abs(val) > vmax * 0.55 else 'black'
                ax.text(j, i, f'{val:+.3f}{st}',
                        ha='center', va='center', fontsize=9, color=txt_col,
                        fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Substitution $\\beta_k$ (pp / pp)')
    ax.set_title('Cross-channel substitution by industry\n'
                 '$\\beta_k > 0$ = substitute; cells with $N < 25$ shown as n.a.',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_substitution_matrix.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    return coef_mat, stars_mat


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Heterogeneity by firm size
# ─────────────────────────────────────────────────────────────────────────────

def fig_heterogeneity(fl):
    print('  Fig 6: heterogeneity by firm size...')
    size_col = 'log_sale' if 'log_sale' in fl.columns else None
    if size_col is None:
        print('    (skipped — no firm size variable)')
        return

    fl2 = fl.dropna(subset=[size_col]).copy()
    fl2['size_tercile'] = pd.qcut(fl2[size_col], q=3,
                                   labels=['Small', 'Mid', 'Large'],
                                   duplicates='drop')
    terciles = ['Small', 'Mid', 'Large']
    res = {t: {} for t in terciles}
    for t in terciles:
        sub_t = fl2[fl2['size_tercile'] == t]
        for ch in SUBST_CHANNELS:
            delta_col = f'{ch}_delta'
            if delta_col not in sub_t.columns:
                continue
            res[t][ch] = run_cs_reg(sub_t[delta_col], sub_t['intl_pre'])

    x        = np.arange(len(SUBST_CHANNELS))
    width    = 0.25
    offsets  = [-width, 0, width]
    t_colors = ['#3498db', '#e67e22', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (terc, col) in enumerate(zip(terciles, t_colors)):
        n_t    = int((fl2['size_tercile'] == terc).sum())
        coefs  = [res[terc].get(ch, {}).get('coef', np.nan) for ch in SUBST_CHANNELS]
        ses    = [res[terc].get(ch, {}).get('se',   np.nan) for ch in SUBST_CHANNELS]
        ax.bar(x + offsets[i], coefs, width=width*0.92,
               color=col, alpha=0.78, label=f'{terc} ($N={n_t}$)',
               yerr=ses, capsize=4, error_kw={'linewidth': 1.2})

    ax.axhline(0, color='black', lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([CH_LABELS[c] for c in SUBST_CHANNELS], fontsize=10)
    ax.set_ylabel('Substitution coefficient $\\beta_k$ (pp / pp)', fontsize=10)
    ax.set_title('Substitution elasticities by firm size (revenue tercile)\n'
                 '$\\beta_k > 0$ = substitute channel; 95\\% CI error bars',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_heterogeneity.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def write_main_table(spec1, spec1_ctrl, spec3_pooled):
    """Table 1: Main substitution coefficients — Specs 1 / 1+controls / 3."""
    outcome_cols = SUBST_CHANNELS + ['ETR']

    def fc(r):
        if not r or np.isnan(r.get('coef', np.nan)):
            return '---'
        v, s = r['coef'], r.get('se', 0)
        t = abs(v/s) if s > 0 else 0
        return f'${v:+.3f}^{{{stars(t)}}}$' if stars(t) else f'${v:+.3f}$'

    def fse(r):
        if not r or np.isnan(r.get('se', np.nan)):
            return ''
        return f'$({r["se"]:.3f})$'

    def fn(r):
        return str(r.get('n', '')) if r else ''

    col_hdrs = ' & '.join(
        f'\\multicolumn{{3}}{{c}}{{{CH_LABELS_TEX[c]}}}' for c in outcome_cols)
    cmidrules = ''.join(
        f'\\cmidrule(lr){{{2+3*i}-{4+3*i}}}' for i in range(len(outcome_cols)))
    spec_hdrs = ' & '.join(['(1) & (2) & (3)'] * len(outcome_cols))

    def make_row(label, d):
        return label + ' & ' + ' & '.join(
            fc(d.get(c, {})) + ' & ' + fc(d.get(c, {})) + ' & ' + fc(d.get(c, {}))
            if False else  # placeholder — expand below
            '&'.join([fc(d[0].get(c,{})), fc(d[1].get(c,{})), fc(d[2].get(c,{}))])
            for c in outcome_cols) + ' \\\\'

    rows = []
    # Coef rows
    coef_parts = []
    se_parts   = []
    for c in outcome_cols:
        r1 = spec1.get(c, {})
        r2 = spec1_ctrl.get(c, {})
        r3 = spec3_pooled.get(c, {})
        coef_parts.append(f'{fc(r1)} & {fc(r2)} & {fc(r3)}')
        se_parts.append(f'{fse(r1)} & {fse(r2)} & {fse(r3)}')

    n_parts = []
    for c in outcome_cols:
        r1 = spec1.get(c, {})
        n_parts.append(f'{fn(r1)} & & ')

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Cross-channel tax-planning substitution around TCJA}',
        '\\label{tab:substitution_main}',
        '\\small',
        '\\begin{tabular}{l' + 'ccc' * len(outcome_cols) + '}',
        '\\toprule',
        ' & ' + col_hdrs + ' \\\\',
        cmidrules,
        ' & ' + spec_hdrs + ' \\\\',
        '\\midrule',
        '$\\bar{c}^{\\text{pre}}_{\\text{Intl}}$ & ' +
        ' & '.join(coef_parts) + ' \\\\',
        ' & ' + ' & '.join(se_parts) + ' \\\\',
        '\\midrule',
        'Controls & No & Yes & --- & ' * (len(outcome_cols)-1) + 'No & Yes & --- \\\\',
        'Industry $\\times$ yr FE & No & No & Yes & ' * (len(outcome_cols)-1) +
        'No & No & Yes \\\\',
        '$N$ firms & ' + ' & '.join(n_parts) + ' \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\begin{tablenotes}\\footnotesize',
        '\\item Spec (1): baseline cross-section, HC1 SE. '
        'Spec (2): adds industry FE. Spec (3): panel pooled DiD with industry$\\times$year FE, clustered SE.',
        '\\item $\\bar{c}^{\\text{pre}}_{\\text{Intl}}$: firm pre-TCJA international contribution (2013--2016 mean, pp).',
        '\\item Outcome: firm-level $\\Delta c_k$ (post minus pre average). '
        '$\\hat{\\beta}_k > 0$ = substitute channel.',
        '\\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
        '\\end{tablenotes}',
        '\\end{table}',
    ]
    path = os.path.join(TBL_DIR, 'tbl_substitution_main.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


def write_channel_summary(fl):
    """Table 2: Descriptive statistics by channel."""
    rows_data = []
    for ch in ALL_CHANNELS:
        pre_c, post_c, delt_c = f'{ch}_pre', f'{ch}_post', f'{ch}_delta'
        if pre_c not in fl.columns:
            continue
        pre  = fl[pre_c].dropna()
        post = fl[post_c].dropna() if post_c in fl.columns else pd.Series(dtype=float)
        delt = fl[delt_c].dropna() if delt_c in fl.columns else pd.Series(dtype=float)
        rows_data.append({
            'label': CH_LABELS_TEX.get(ch, ch).replace('\\&', '\\&'),
            'pre_m': pre.mean(),  'pre_s': pre.std(),
            'post_m': post.mean() if len(post) else np.nan,
            'post_s': post.std()  if len(post) else np.nan,
            'delt_m': delt.mean() if len(delt) else np.nan,
            'delt_s': delt.std()  if len(delt) else np.nan,
        })

    lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Channel contribution descriptive statistics (pp, firm-level averages)}',
        '\\label{tab:channel_summary}',
        '\\small',
        '\\begin{tabular}{lrrrrrr}',
        '\\toprule',
        ' & \\multicolumn{2}{c}{Pre-TCJA (2013--16)} '
        '& \\multicolumn{2}{c}{Post-TCJA (2019--22)} '
        '& \\multicolumn{2}{c}{$\\Delta$ (post$-$pre)} \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
        'Channel & Mean & SD & Mean & SD & Mean & SD \\\\',
        '\\midrule',
    ]
    for r in rows_data:
        def fmt(v): return f'{v:+.2f}' if not np.isnan(v) else '---'
        def fmtsd(v): return f'{v:.2f}' if not np.isnan(v) else '---'
        lines.append(
            f"{r['label']} & {fmt(r['pre_m'])} & {fmtsd(r['pre_s'])} & "
            f"{fmt(r['post_m'])} & {fmtsd(r['post_s'])} & "
            f"{fmt(r['delt_m'])} & {fmtsd(r['delt_s'])} \\\\")
    lines.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\begin{tablenotes}\\footnotesize',
        '\\item Balanced panel sample (Finance excluded; profitable firms). '
        'Winsorized at $\\pm 15$\\,pp per firm-year-channel.',
        '\\end{tablenotes}',
        '\\end{table}',
    ])
    path = os.path.join(TBL_DIR, 'tbl_channel_summary.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default=os.path.join(script_dir, '..', 'data', 'analysis', 'cache'))
    args = parser.parse_args()

    # ── I. Load ──────────────────────────────────────────────────────────────
    combined, long_df = load_data(args.cache_dir)

    # ── II. Build channel panel ───────────────────────────────────────────────
    wide = build_channels(long_df)

    # ── III. Treatment variable ───────────────────────────────────────────────
    intl_pre = build_treatment(wide)

    # ── IV. Event panel ───────────────────────────────────────────────────────
    panel = build_panel(combined, wide, intl_pre)

    # ── V. Firm-level cross-section ───────────────────────────────────────────
    fl = build_firm_level(panel, intl_pre)

    # ── VI. Spec 1: cross-sectional substitution ──────────────────────────────
    print('\nSection I: Spec 1 — cross-sectional substitution...')
    spec1 = {}
    for ch in SUBST_CHANNELS:
        delta = f'{ch}_delta'
        if delta not in fl.columns:
            continue
        r = run_cs_reg(fl[delta], fl['intl_pre'])
        spec1[ch] = r
        t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
        print(f'  Δ{ch:<20} β={r["coef"]:+.4f} SE={r["se"]:.4f} {stars(t):3s}  N={r["n"]}')

    # ETR
    if 'etr_delta' in fl.columns:
        r_etr = run_cs_reg(fl['etr_delta'], fl['intl_pre'])
        spec1['ETR'] = r_etr
        t = abs(r_etr['coef']/r_etr['se']) if r_etr.get('se', 0) > 0 else 0
        print(f'  ΔETR                     β={r_etr["coef"]:+.4f} SE={r_etr["se"]:.4f} {stars(t):3s}  N={r_etr["n"]}')

    # International self-regression (first stage)
    r_intl = {'coef': np.nan, 'se': np.nan, 'r2': np.nan, 'n': 0}
    if 'International_delta' in fl.columns:
        r_intl = run_cs_reg(fl['International_delta'], fl['intl_pre'])
        t = abs(r_intl['coef']/r_intl['se']) if r_intl.get('se', 0) > 0 else 0
        print(f'  ΔInternational           β={r_intl["coef"]:+.4f} SE={r_intl["se"]:.4f} {stars(t):3s}  (first stage)')

    # Mechanical identity: Σ_k β(channel_k) ≈ β(ETR)
    # (since channels sum to ETR-STR gap; STR change is common to all firms so its
    #  regression coefficient on intl_pre ≈ 0)
    ch_sum     = sum(spec1.get(ch, {}).get('coef', 0) for ch in SUBST_CHANNELS)
    etr_c      = spec1.get('ETR', {}).get('coef', np.nan)
    intl_c     = r_intl.get('coef', np.nan)
    chan_total  = ch_sum + (intl_c if not np.isnan(intl_c) else 0)
    identity_resid = chan_total - (etr_c if not np.isnan(etr_c) else 0)
    print(f'\n  Identity check: β(Intl) + Σβ(subst channels) = {chan_total:+.4f}')
    print(f'  β(ETR) = {etr_c:+.4f}')
    print(f'  Residual (should ≈ 0; remainder = omitted Reserves & TCJA channels): {identity_resid:+.4f}')
    total = chan_total  # store for validation report

    # Spec 1 with industry FE
    print('\n  Spec 1 + industry FE...')
    ind_dum = pd.get_dummies(fl['industry'], prefix='ind', drop_first=True, dtype=float)
    spec1_ctrl = {}
    for ch in SUBST_CHANNELS + ['ETR']:
        delta = 'etr_delta' if ch == 'ETR' else f'{ch}_delta'
        if delta not in fl.columns:
            continue
        sub = fl.dropna(subset=[delta, 'intl_pre'])
        d_sub = ind_dum.loc[sub.index]
        ctrls = [d_sub[c] for c in d_sub.columns]
        r = run_cs_reg(sub[delta], sub['intl_pre'], controls=ctrls)
        spec1_ctrl[ch] = r
        t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
        print(f'  Δ{ch:<20} β={r["coef"]:+.4f} SE={r["se"]:.4f} {stars(t):3s}  N={r["n"]}')

    # ── VII. Spec 2: panel event studies ──────────────────────────────────────
    print('\nSection II: Spec 2 — panel event studies...')
    event_results  = {}
    pooled_results = {}
    for ch in ALL_CHANNELS:
        if ch not in panel.columns:
            continue
        print(f'  {ch}...', end=' ', flush=True)
        event_results[ch]  = run_panel_event(panel, ch, ind_yr_fe=True)
        pooled_results[ch] = run_pooled_did(panel, ch, ind_yr_fe=True)
        pr = pooled_results[ch]
        t  = abs(pr['coef']/pr['se']) if pr.get('se', 0) > 0 else 0
        print(f'pooled DiD: {pr["coef"]:+.4f} (SE {pr["se"]:.4f}){stars(t)}')

    if 'etr_w' in panel.columns:
        print('  ETR...', end=' ', flush=True)
        event_results['ETR']  = run_panel_event(panel, 'etr_w', ind_yr_fe=True)
        pooled_results['ETR'] = run_pooled_did(panel, 'etr_w', ind_yr_fe=True)
        pr = pooled_results['ETR']
        t  = abs(pr['coef']/pr['se']) if pr.get('se', 0) > 0 else 0
        print(f'pooled DiD: {pr["coef"]:+.4f} (SE {pr["se"]:.4f}){stars(t)}')

    # ── VIII. Spec 3: industry × year FE robustness ───────────────────────────
    print('\nSection III: Spec 3 — industry × year FE...')
    spec3_pooled = {}
    for ch in SUBST_CHANNELS + ['ETR']:
        col = 'etr_w' if ch == 'ETR' else ch
        if col not in panel.columns:
            continue
        r = run_pooled_did(panel, col, ind_yr_fe=True)
        spec3_pooled[ch] = r
        t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
        print(f'  {ch:<20} β={r["coef"]:+.4f} SE={r["se"]:.4f} {stars(t):3s}')

    # ── IX. Figures ───────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    ts = fig_channel_timeseries(combined, wide)
    fs_r = fig_firststage_intl(fl)
    fig_substitution_event(event_results, pooled_results)
    fig_substitution_scatter(fl)
    heat_coef, heat_stars = fig_substitution_matrix(fl)
    fig_heterogeneity(fl)

    # ── X. LaTeX tables ───────────────────────────────────────────────────────
    print('\nWriting LaTeX tables...')
    write_main_table(spec1, spec1_ctrl, spec3_pooled)
    write_channel_summary(fl)

    # ── XI. Validation report ─────────────────────────────────────────────────
    print()
    print('=' * 68)
    print('  SUBSTITUTION ANALYSIS — VALIDATION REPORT')
    print('=' * 68)
    print()
    print('SAMPLE')
    print(f'  Treatment firms (≥2 pre obs):     {len(intl_pre):,}')
    print(f'  Balanced panel firms:              {panel["gvkey"].nunique():,}')
    print(f'  Firm-year obs (panel):             {len(panel):,}')
    print(f'  Firm-level cross-section (Spec 1): {len(fl):,}')
    print()
    print('TREATMENT (intl_pre, pp)')
    print(f'  Median: {intl_pre.median():.2f}  '
          f'P25: {intl_pre.quantile(0.25):.2f}  '
          f'P75: {intl_pre.quantile(0.75):.2f}')
    print()
    print('FIRST STAGE: Δ International on intl_pre')
    print(f'  Slope: {r_intl["coef"]:+.4f}  SE: {r_intl["se"]:.4f}  R²: {r_intl["r2"]:.3f}')
    print(f'  (Slope ≈ -1 means channel closed fully; ≈ 0 means no closure)')
    print()
    print('SPEC 1: CROSS-SECTIONAL SUBSTITUTION (baseline)')
    for ch in SUBST_CHANNELS + ['ETR']:
        r = spec1.get(ch, {})
        if r:
            t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
            print(f'  Δ{ch:<20} β={r["coef"]:+.4f}  SE={r["se"]:.4f}  {stars(t)}')
    print(f'  β(Intl)+Σβ(sub channels)={chan_total:+.4f}  β(ETR)={etr_c:+.4f}  '
          f'residual={identity_resid:+.4f}  (pass if ≈ 0)')
    print()
    print('SPEC 2: POOLED DiD (PANEL)')
    for ch in SUBST_CHANNELS + ['ETR']:
        pr = pooled_results.get(ch, {})
        if pr:
            t = abs(pr['coef']/pr['se']) if pr.get('se', 0) > 0 else 0
            print(f'  {ch:<22} θ={pr["coef"]:+.4f}  SE={pr["se"]:.4f}  {stars(t)}')
    print()
    print('SPEC 3: INDUSTRY × YEAR FE')
    for ch in SUBST_CHANNELS + ['ETR']:
        pr = spec3_pooled.get(ch, {})
        if pr:
            t = abs(pr['coef']/pr['se']) if pr.get('se', 0) > 0 else 0
            print(f'  {ch:<22} θ={pr["coef"]:+.4f}  SE={pr["se"]:.4f}  {stars(t)}')
    print()
    print('Done.')
