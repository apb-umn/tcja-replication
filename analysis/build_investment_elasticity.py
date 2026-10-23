#!/usr/bin/env python3
"""
Investment Elasticity from TCJA Deferred Tax Shock
====================================================
Strategy A: Z_i = txdb_2016 / at_2015  (raw variation)
Strategy B: within-industry deviation of Z_i  (more conservative)

Usage:
    python3 analysis/build_investment_elasticity.py
"""

import os, pickle, warnings, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

from sic_industry import industry  # canonical SIC -> industry classifier

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))
DPI = 300

# ── Constants ────────────────────────────────────────────────────────────────
PRE_YEARS     = list(range(2013, 2017))
POST_YEARS    = list(range(2018, 2023))   # exclude 2017 shock year
EVENT_YEARS   = list(range(2013, 2023))
OMIT_YEAR     = 2016
SHOCK_YEAR    = 2017
WINDFALL_FRAC = 0.40   # (35-21)/35
VA_THRESHOLD  = 3.0    # |VA 2017 pp| > this → exclude

COMP_COLS = [
    'comp_txndb', 'comp_txdb', 'comp_txditc', 'comp_at', 'comp_capx', 'comp_xrd',
    'comp_sale', 'comp_emp', 'comp_ni', 'comp_pi', 'comp_che',
    'comp_dltt', 'comp_dlc', 'comp_ppent', 'comp_dvc', 'comp_prstkc',
    'comp_ceq', 'comp_intan', 'comp_gdwl', 'comp_tlcf',
]

IND_COLORS = {
    'Tech':           '#3498db',
    'Pharma/Biotech': '#9b59b6',
    'Manufacturing':  '#e67e22',
    'Oil & Gas':      '#f1c40f',
    'Retail':         '#27ae60',
    'Utilities':      '#1abc9c',
    'Finance':        '#7f8c8d',
    'Other':          '#bdc3c7',
}

matplotlib.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 10.5,
    'axes.labelsize': 10.5, 'axes.titlesize': 11,
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════



def stars(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))


def _winsorize_within_year(df, col, year_col='recon_year', lo=0.01, hi=0.99):
    result = df[col].copy().astype(float)
    for yr in df[year_col].unique():
        mask = df[year_col] == yr
        s = df.loc[mask, col].astype(float)
        if s.notna().sum() < 10:
            continue
        result.loc[mask] = s.clip(s.quantile(lo), s.quantile(hi))
    return result


def demean_twfe(df, cols, firm_col='gvkey', year_col='recon_year',
                max_iter=80, tol=1e-10):
    df = df.copy()
    for _ in range(max_iter):
        prev = df[cols[0]].copy()
        fm = df.groupby(firm_col)[cols].transform('mean')
        df[cols] = df[cols].values - fm.values
        ym = df.groupby(year_col)[cols].transform('mean')
        df[cols] = df[cols].values - ym.values
        if (df[cols[0]] - prev).abs().max() < tol:
            break
    return df


def demean_ind_yr(df, cols, firm_col='gvkey', year_col='recon_year',
                  ind_col='industry', max_iter=80, tol=1e-10):
    df = df.copy()
    df['_iyr'] = df[ind_col].astype(str) + '_' + df[year_col].astype(str)
    for _ in range(max_iter):
        prev = df[cols[0]].copy()
        fm  = df.groupby(firm_col)[cols].transform('mean')
        df[cols] = df[cols].values - fm.values
        iym = df.groupby('_iyr')[cols].transform('mean')
        df[cols] = df[cols].values - iym.values
        if (df[cols[0]] - prev).abs().max() < tol:
            break
    return df


def _event_axes(ax, all_years, y_pts, lo_pts, hi_pts,
                omit=OMIT_YEAR, shock=SHOCK_YEAR):
    ax.axvspan(min(all_years) - 0.5, shock - 0.2, color='#7f8c8d', alpha=0.04, zorder=0)
    ax.axvspan(shock + 0.2, max(all_years) + 0.5, color='#c0392b', alpha=0.05, zorder=0)
    ax.axhline(0, color='black', lw=0.9, zorder=1)
    ax.axvline(shock, color='dimgray', lw=1.3, ls='--', zorder=2, alpha=0.75)
    ax.errorbar(all_years, y_pts,
                yerr=np.vstack([y_pts - lo_pts, hi_pts - y_pts]),
                fmt='none', ecolor='#34495e', capsize=3, lw=1.2, alpha=0.75, zorder=3)
    for i, yr in enumerate(all_years):
        if yr == omit:
            ax.scatter(yr, 0, s=80, marker='D', facecolors='white',
                       edgecolors='black', lw=1.2, zorder=5)
        elif yr >= shock:
            ax.scatter(yr, y_pts[i], s=65, color='#c0392b', zorder=5)
        else:
            ax.scatter(yr, y_pts[i], s=65, color='#7f8c8d', zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading & construction
# ═══════════════════════════════════════════════════════════════════════════

def load_comp_extra(base_panel_csv):
    all_cols  = pd.read_csv(base_panel_csv, nrows=0).columns.tolist()
    available = [c for c in COMP_COLS if c in all_cols]
    missing   = [c for c in COMP_COLS if c not in all_cols]
    if missing:
        print(f'  WARNING: missing from base panel: {missing}')

    load_cols = ['gvkey', 'recon_year'] + available
    comp = pd.read_csv(base_panel_csv, usecols=load_cols, low_memory=False)
    comp = comp.drop_duplicates(['gvkey', 'recon_year'])
    rename = {c: c.replace('comp_', '') for c in available}
    comp = comp.rename(columns=rename)

    print(f'  Loaded {len(comp):,} deduplicated firm-year rows')
    for v in ['txdb', 'at', 'capx', 'xrd', 'dvc', 'prstkc', 'che']:
        if v in comp.columns:
            pct = 100 * comp[v].notna().mean()
            print(f'    {v:<12}: {pct:.1f}% non-missing')
    return comp


def build_treatment(comp, long_df):
    """Z_i = txdb_2016 / at_2015, W_i = 0.40 × Z_i. Excludes |VA 2017| > 3 pp."""
    # Use txndb, the net deferred tax position, rather than txdb, which is
    # DTL-only in Compustat and never negative.  txndb is positive for a net
    # deferred tax ASSET: firms with txndb < 0 are the net-DTL firms, and they
    # are the ones that booked a benefit on the 2017 remeasurement (median
    # TCJA reconciliation contribution -12.9pp against +8.5pp for txndb > 0).
    # Hence the negation below.
    comp2016 = (comp[comp['recon_year'] == 2016][['gvkey', 'txndb']]
                .rename(columns={'txndb': 'txdb_2016'}).dropna())
    comp2015 = (comp[comp['recon_year'] == 2015][['gvkey', 'at']]
                .rename(columns={'at': 'at_2015'}))
    comp2015 = comp2015[comp2015['at_2015'] > 0]

    treat = comp2016.merge(comp2015, on='gvkey', how='inner')
    # txndb = DTA - DTL (asset-side convention in Compustat): negate so that
    # positive Z means net-DTL = TCJA windfall (future tax payments revalued down).
    treat['Z'] = -treat['txdb_2016'] / treat['at_2015']
    print(f'  Initial: {len(treat)} firms')
    print(f'  Sign dist: {(treat["Z"]>0).mean()*100:.1f}% net-DTL (windfall), '
          f'{(treat["Z"]<0).mean()*100:.1f}% net-DTA (hit)')

    p1, p99 = treat['Z'].quantile([0.01, 0.99])
    treat['Z_w'] = treat['Z'].clip(p1, p99)
    treat['W']   = WINDFALL_FRAC * treat['Z_w']
    print(f'  Z_w: median={treat["Z_w"].median():.4f}, '
          f'IQR=[{treat["Z_w"].quantile(.25):.4f}, {treat["Z_w"].quantile(.75):.4f}]')

    # Valuation allowance exclusion (uses long_df, not Compustat)
    va_2017 = (long_df[(long_df['recon_year'] == 2017) &
                       (long_df['category'] == 'Valuation allowance')]
               .groupby('gvkey')['cat_val'].sum()
               .rename('va_2017_pp'))
    treat = treat.merge(va_2017, on='gvkey', how='left')
    treat['va_2017_pp'] = treat['va_2017_pp'].fillna(0)
    n_before = len(treat)
    treat = treat[treat['va_2017_pp'].abs() <= VA_THRESHOLD].copy()
    print(f'  After VA exclusion: {len(treat)} firms '
          f'(removed {n_before - len(treat)}, '
          f'{100*(n_before-len(treat))/n_before:.1f}%)')
    return treat


def build_event_panel(combined, treat, comp, event_years, pre_years, post_years):
    df = combined.copy()
    df['industry'] = df['comp_sic'].apply(industry)
    df = df[df['industry'] != 'Finance']
    df = df.merge(treat[['gvkey', 'Z', 'Z_w', 'W']], on='gvkey', how='inner')

    # Merge comp extras
    extra_cols = [c for c in comp.columns if c not in ('gvkey', 'recon_year')]
    df = df.merge(comp[['gvkey', 'recon_year'] + extra_cols],
                  on=['gvkey', 'recon_year'], how='left')

    df = df[df['recon_year'].isin(event_years)].copy()

    firm_yrs = df.groupby('gvkey')['recon_year'].apply(set)
    has_pre  = firm_yrs.apply(lambda s: len(s & set(pre_years)) >= 2)
    has_post = firm_yrs.apply(lambda s: len(s & set(post_years)) >= 2)
    ok = firm_yrs[has_pre & has_post].index
    df = df[df['gvkey'].isin(ok)].copy()
    print(f'  Balanced panel: {df["gvkey"].nunique()} firms, {len(df):,} firm-years')
    return df


def build_outcomes(panel):
    df = panel.sort_values(['gvkey', 'recon_year']).copy()

    for v in ['at', 'emp', 'dltt', 'dlc']:
        df[f'{v}_lag1'] = df.groupby('gvkey')[v].shift(1)

    valid_at = df['at_lag1'] > 0

    df['capx_intensity']      = np.where(valid_at & df['capx'].notna(),
                                          df['capx'] / df['at_lag1'], np.nan)
    df['rd_intensity']        = np.where(valid_at & df['xrd'].notna(),
                                          df['xrd'] / df['at_lag1'], np.nan)
    df['total_inv_intensity'] = np.where(valid_at,
                                          (df['capx'].fillna(0) + df['xrd'].fillna(0)) / df['at_lag1'],
                                          np.nan)
    df['payout_intensity']    = np.where(valid_at,
                                          (df['dvc'].fillna(0) + df['prstkc'].fillna(0)) / df['at_lag1'],
                                          np.nan)
    df['ni_intensity']        = np.where(valid_at & df['ni'].notna(),
                                          df['ni'] / df['at_lag1'], np.nan)
    df['total_debt']          = df['dltt'].fillna(0) + df['dlc'].fillna(0)
    df['total_debt_lag1']     = df['dltt_lag1'].fillna(0) + df['dlc_lag1'].fillna(0)
    df['debt_change_intensity'] = np.where(valid_at,
                                            (df['total_debt'] - df['total_debt_lag1']) / df['at_lag1'],
                                            np.nan)

    emp_valid = (df['emp_lag1'] > 0) & df['emp_lag1'].notna() & (df['emp'] > 0)
    df['emp_growth'] = np.where(emp_valid, np.log(df['emp'] / df['emp_lag1']), np.nan)

    # Cash accumulation: level and change, both scaled by lagged assets
    df['che_lag1'] = df.groupby('gvkey')['che'].shift(1)
    df['che_intensity']  = np.where(valid_at & df['che'].notna(),
                                     df['che'] / df['at_lag1'], np.nan)
    df['che_change']     = np.where(valid_at & df['che'].notna() & df['che_lag1'].notna(),
                                     (df['che'] - df['che_lag1']) / df['at_lag1'], np.nan)

    outcome_cols = ['capx_intensity', 'rd_intensity', 'total_inv_intensity',
                    'payout_intensity', 'ni_intensity', 'debt_change_intensity',
                    'emp_growth', 'che_intensity', 'che_change']
    for col in outcome_cols:
        if df[col].notna().sum() > 50:
            df[col] = _winsorize_within_year(df, col)

    for col in outcome_cols:
        n = df[col].notna().sum()
        g = df.loc[df[col].notna(), 'gvkey'].nunique()
        print(f'  {col:<28}: {n:,} obs, {g} firms')
    return df


def build_controls(panel):
    df = panel.copy()
    df['log_at']    = np.log(df['at'].clip(lower=1e-6))
    df['leverage']  = df['total_debt'] / df['at'].clip(lower=1e-6)
    df['cash_ratio'] = df['che'] / df['at'].clip(lower=1e-6)

    snap2016 = (df[df['recon_year'] == 2016]
                .groupby('gvkey')[['log_at', 'leverage', 'cash_ratio']].first()
                .rename(columns=lambda c: c + '_2016'))
    df = df.merge(snap2016, on='gvkey', how='left')

    pre_mask = df['recon_year'].between(2014, 2016)
    for v in ['capx_intensity', 'rd_intensity', 'cash_ratio']:
        pre_avg = (df[pre_mask].groupby('gvkey')[v].mean().rename(v + '_pre3y'))
        df = df.merge(pre_avg, on='gvkey', how='left')

    if 'intan' in df.columns:
        df['intan_ratio'] = df['intan'] / df['at'].clip(lower=1e-6)
        pre_intan = (df[pre_mask].groupby('gvkey')['intan_ratio'].mean()
                     .rename('intan_pre3y'))
        df = df.merge(pre_intan, on='gvkey', how='left')
    return df


def build_strategy_b(panel, treat):
    fl = treat[['gvkey', 'Z_w', 'W']].copy()
    ind_map = panel.drop_duplicates('gvkey').set_index('gvkey')['industry']
    fl = fl.merge(ind_map, on='gvkey', how='left')
    ind_mean = fl.groupby('industry')['Z_w'].mean().rename('Z_ind_mean')
    fl = fl.merge(ind_mean, on='industry', how='left')
    fl['Z_within'] = fl['Z_w'] - fl['Z_ind_mean']
    fl['W_within'] = WINDFALL_FRAC * fl['Z_within']

    total_var   = fl['Z_w'].var()
    within_var  = fl['Z_within'].var()
    between_var = total_var - within_var
    print(f'  Z_w variance decomp: total={total_var:.5f}, '
          f'between={between_var:.5f} ({100*between_var/total_var:.1f}%), '
          f'within={within_var:.5f} ({100*within_var/total_var:.1f}%)')
    return fl[['gvkey', 'Z_within', 'W_within', 'Z_ind_mean']]


# ═══════════════════════════════════════════════════════════════════════════
# Regression runners
# ═══════════════════════════════════════════════════════════════════════════

def run_event_study(panel, outcome, treat_col, event_years,
                    omit=OMIT_YEAR, controls=None, ind_yr_fe=False):
    plot_yrs = [yr for yr in sorted(event_years) if yr != omit]
    ctrl_cols = [c for c in (controls or []) if c in panel.columns]

    mask = (panel['recon_year'].isin(event_years) &
            panel[outcome].notna() & panel[treat_col].notna())
    for c in ctrl_cols:
        mask = mask & panel[c].notna()
    sub = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return pd.DataFrame(), {}

    ix_cols = []
    for yr in plot_yrs:
        col = f'_ix_{yr}'
        sub[col] = sub[treat_col] * (sub['recon_year'] == yr).astype(float)
        ix_cols.append(col)

    all_cols = [outcome] + ix_cols + ctrl_cols
    sub_dm = demean_ind_yr(sub, all_cols) if ind_yr_fe else demean_twfe(sub, all_cols)

    X      = sub_dm[ix_cols + ctrl_cols].values
    y      = sub_dm[outcome].values
    groups = sub['gvkey'].values
    mod    = OLS(y, X).fit()
    cl_cov = cov_cluster(mod, groups)
    cl_se  = np.sqrt(np.diag(cl_cov))

    records = [{'year': yr, 'beta': mod.params[i], 'se': cl_se[i],
                'lower': mod.params[i] - 1.96*cl_se[i],
                'upper': mod.params[i] + 1.96*cl_se[i]}
               for i, yr in enumerate(plot_yrs)]
    records.append({'year': omit, 'beta': 0., 'se': 0., 'lower': 0., 'upper': 0.})
    ev = pd.DataFrame(records).sort_values('year').reset_index(drop=True)
    info = {'n': len(sub), 'G': sub['gvkey'].nunique(), 'rsq': float(mod.rsquared)}
    return ev, info


def run_did(panel, outcome, treat_col, post_years=None, drop_years=None,
            controls=None, ind_yr_fe=False):
    p_yrs = post_years or POST_YEARS
    d_yrs = list(set((drop_years or []) + [SHOCK_YEAR]))
    years = [yr for yr in EVENT_YEARS if yr not in d_yrs]
    ctrl_cols = [c for c in (controls or []) if c in panel.columns]

    mask = (panel['recon_year'].isin(years) &
            panel[outcome].notna() & panel[treat_col].notna())
    for c in ctrl_cols:
        mask = mask & panel[c].notna()
    sub = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return {'coef': np.nan, 'se': np.nan, 'n': 0, 'G': 0, 'rsq': np.nan, 't': np.nan}

    sub['_post'] = sub['recon_year'].isin(p_yrs).astype(float)
    sub['_Di']   = sub[treat_col] * sub['_post']

    all_cols = [outcome, '_Di'] + ctrl_cols
    sub_dm = demean_ind_yr(sub, all_cols) if ind_yr_fe else demean_twfe(sub, all_cols)

    X      = sub_dm[['_Di'] + ctrl_cols].values
    y      = sub_dm[outcome].values
    groups = sub['gvkey'].values
    mod    = OLS(y, X).fit()
    cl_cov = cov_cluster(mod, groups)
    cl_se  = np.sqrt(np.diag(cl_cov))

    b, s = float(mod.params[0]), float(cl_se[0])
    return {'coef': b, 'se': s, 'n': len(sub),
            'G': sub['gvkey'].nunique(), 'rsq': float(mod.rsquared),
            't': b/s if s > 0 else 0.}


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_firststage_dtl(firm_level, fig_dir):
    sub = firm_level.dropna(subset=['Z_w', 'tcja_2017_pp'])
    X   = np.column_stack([np.ones(len(sub)), sub['Z_w'].values])
    mod = OLS(sub['tcja_2017_pp'].values, X).fit()
    b0, b1, r2 = mod.params[0], mod.params[1], mod.rsquared

    fig, ax = plt.subplots(figsize=(9, 6))
    for ind, grp in sub.groupby('industry'):
        ax.scatter(grp['Z_w'], grp['tcja_2017_pp'],
                   s=15, alpha=0.35, color=IND_COLORS.get(ind, '#aaa'), label=ind, zorder=2)

    xl = np.linspace(sub['Z_w'].min(), sub['Z_w'].max(), 200)
    ax.plot(xl, b0 + b1 * xl, color='black', lw=1.8, zorder=4)
    ax.text(0.04, 0.96,
            f'Slope: {b1:.2f}  (intercept {b0:.2f})\n$R^2 = {r2:.2f}$',
            transform=ax.transAxes, va='top', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', lw=0.8))
    ax.axhline(0, color='black', lw=0.7)
    ax.axvline(0, color='black', lw=0.7)
    ax.set_xlabel(r'$Z_i = \mathtt{txdb}_{2016} / \mathtt{at}_{2015}$ (net DTL / assets)')
    ax.set_ylabel('2017 TCJA rate-remeasurement contribution (pp)')
    ax.set_title('Mechanical validation: pre-period DTL balance predicts\n'
                 '2017 rate-remeasurement contribution')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'fig_firststage_dtl.pdf')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def _render_event_panel(ax, ev, title, ylabel):
    if ev.empty:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        ax.set_title(title)
        return
    all_years = sorted(ev['year'].tolist())
    idx = ev.set_index('year')
    y_pts  = np.array([idx.loc[yr, 'beta']  if yr in idx.index else 0. for yr in all_years])
    lo_pts = np.array([idx.loc[yr, 'lower'] if yr in idx.index else 0. for yr in all_years])
    hi_pts = np.array([idx.loc[yr, 'upper'] if yr in idx.index else 0. for yr in all_years])
    _event_axes(ax, all_years, y_pts, lo_pts, hi_pts)
    ax.set_title(title, fontsize=10.5, pad=4)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(all_years)
    ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha='right', fontsize=8.5)


def fig_firststage_ni(ev_ni, did_ni_post, n_firms, n_obs, fig_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _render_event_panel(ax, ev_ni,
                        title=(f'First stage: net income / lagged assets on predicted TCJA windfall\n'
                               f'N = {n_firms} firms, {n_obs:,} firm-years; firm + industry$\\times$year FE; clustered SE'),
                        ylabel=r'$\hat\beta_t$ (NI/A per unit of $W_i$)')

    b_post = did_ni_post.get('coef', np.nan)
    se_post = did_ni_post.get('se', np.nan)
    if not np.isnan(b_post):
        ax.text(0.72, 0.97,
                f'Pooled 2018–22:\n$\\hat{{\\beta}} = {b_post:.3f}$ (SE {se_post:.3f})',
                transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', lw=0.8))
    ax.set_xlabel('Fiscal year', fontsize=10.5)
    ax.text(0.03, 0.03,
            r'$W_i = 0.40\times(\mathtt{txdb}_{2016}/\mathtt{at}_{2015})$; one-time effect concentrated in 2017.',
            transform=ax.transAxes, fontsize=7.5, color='#555', style='italic')
    fig.tight_layout()
    out = os.path.join(fig_dir, 'fig_firststage_ni.pdf')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_investment_event(ev_capx, ev_rd, ev_total, n_firms, n_obs, fig_dir):
    panels = [
        (ev_capx,  'A. Capital expenditure / lagged assets', r'$\hat\beta_t$ (Capex/A per $W_i$)'),
        (ev_rd,    'B. R&D expense / lagged assets',         r'$\hat\beta_t$ (R\&D/A per $W_i$)'),
        (ev_total, 'C. Total investment / lagged assets',    r'$\hat\beta_t$ (Inv/A per $W_i$)'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    for ax, (ev, title, ylabel) in zip(axes, panels):
        _render_event_panel(ax, ev, title=title, ylabel=ylabel)
    axes[-1].set_xlabel('Fiscal year', fontsize=10.5)
    fig.suptitle(
        f'Event study: investment response to predicted TCJA windfall\n'
        f'N = {n_firms} balanced firms; firm + industry$\\times$year FE; cluster-robust SEs',
        fontsize=11, y=1.005)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'fig_investment_event.pdf')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_investment_hetero(hetero_res, fig_dir):
    """
    hetero_res: {dim: {cell: {outcome: result}}}
    Two panels: capx_intensity (left) and total_inv_intensity (right).
    """
    outcomes   = ['capx_intensity', 'total_inv_intensity']
    oc_labels  = {'capx_intensity': 'Capex / lagged assets',
                  'total_inv_intensity': 'Total investment / lagged assets'}
    dims       = sorted(hetero_res.keys())
    bar_colors = ['#2c3e50', '#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, len(outcomes), figsize=(13, 5.5), sharey=False)

    for ax, oc in zip(axes, outcomes):
        labels, betas, errs, cols = [], [], [], []
        pos_map = {}   # label → x position (for separator logic)
        ci = 0
        for dim in dims:
            cells = sorted(hetero_res[dim].keys())
            dim_cells_added = []
            for cell in cells:
                r = hetero_res[dim][cell].get(oc, {})
                b, s = r.get('coef', np.nan), r.get('se', np.nan)
                if np.isnan(b):
                    continue
                labels.append(cell)
                betas.append(b)
                errs.append(1.96 * s)
                cols.append(bar_colors[ci % len(bar_colors)])
                dim_cells_added.append(cell)
            if dim_cells_added:
                pos_map[dim] = (len(labels) - len(dim_cells_added), len(labels))
            ci += 1

        if not labels:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(oc_labels[oc], fontsize=11)
            continue

        x = np.arange(len(labels))
        ax.bar(x, betas, color=cols, alpha=0.82, width=0.6, zorder=2)
        ax.errorbar(x, betas, yerr=errs, fmt='none', ecolor='#333',
                    capsize=4, lw=1.2, zorder=3)
        ax.axhline(0, color='black', lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(oc_labels[oc], fontsize=11)
        ax.set_ylabel(r'DiD $\hat\theta$ (per unit $W_i$)', fontsize=9.5)

        # Group separators and labels
        ymax = max(abs(b) for b in betas) * 1.3
        for dim, (start, end) in pos_map.items():
            if start > 0:
                ax.axvline(start - 0.5, color='gray', lw=0.8, ls=':', zorder=1)
            ax.text((start + end - 1) / 2, ymax,
                    dim, ha='center', va='bottom', fontsize=8.5, color='#555',
                    fontweight='bold')

    fig.suptitle('Investment response heterogeneity by size, cash, leverage, and intangibles\n'
                 '(Pooled DiD; firm + year FE; ≥30 firms per cell; tercile splits)',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'fig_investment_hetero.pdf')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_windfall_decomp(windfall_res, fig_dir):
    display = [
        ('capx_intensity',        'Capital expenditure',              False),
        ('rd_intensity',          'R&D expense',                      False),
        ('total_inv_intensity',   'Total investment (capex + R&D)',   False),
        ('payout_intensity',      'Payouts (div. + buybacks)',        False),
        ('debt_change_intensity', 'Net debt reduction (sign-flipped)',True),
        ('che_change',            'Cash accumulation (Δche/A)',       False),
        ('emp_growth',            'Employment growth*',               False),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    y_pos = np.arange(len(display))

    implied_sum = 0.
    for i, (key, label, flip) in enumerate(display):
        r = windfall_res.get(key, {})
        b, s = r.get('coef', np.nan), r.get('se', np.nan)
        if np.isnan(b):
            continue
        plot_b = -b if flip else b
        t_val  = abs(b / s) if s > 0 else 0
        st     = stars(t_val)
        color  = '#2ecc71' if plot_b > 0 else '#e74c3c'
        ax.barh(i, plot_b, color=color, alpha=0.82, height=0.55, zorder=2)
        ax.errorbar(plot_b, i, xerr=[[1.96*s], [1.96*s]],
                    fmt='none', ecolor='#333', capsize=4, lw=1.2, zorder=3)
        sign_str = '+' if plot_b >= 0 else ''
        ax.text(plot_b + (0.003 if plot_b >= 0 else -0.003), i,
                f'{sign_str}{plot_b:.3f}{st}',
                va='center', ha='left' if plot_b >= 0 else 'right',
                fontsize=8.5, color='#222')
        if key not in ('emp_growth', 'total_inv_intensity'):
            implied_sum += plot_b

    # Implied total annotation
    ax.text(0.98, 0.04,
            f'Sum (excl. emp.): {implied_sum:+.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#333',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='#ccc', lw=0.8))

    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl, _ in display], fontsize=9.5)
    ax.set_xlabel(r'DiD $\hat\theta$ (fraction of lagged assets per unit $W_i$)',
                  fontsize=10)
    ax.set_title('How did the TCJA windfall get spent?\n'
                 'Pooled DiD decomposition (2018–22 vs 2013–16, shock year excluded)',
                 fontsize=11)
    ax.text(0.99, 0.02,
            '*Employment growth is a log-change;\nnot directly comparable to level intensities.',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#777', style='italic')
    fig.tight_layout()
    out = os.path.join(fig_dir, 'fig_windfall_decomp.pdf')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_strategy_comparison(strat_comp, fig_dir):
    outcomes  = ['capx_intensity', 'rd_intensity', 'total_inv_intensity']
    oc_labels = ['Capex / A', 'R&D / A', 'Total inv. / A']
    strategies = [('A', '#2c3e50', 'Strategy A (raw $Z_i$)'),
                  ('B', '#e74c3c', 'Strategy B (within-industry $Z_i$)')]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(outcomes))
    bar_w = 0.32

    for j, (strat, color, lbl) in enumerate(strategies):
        betas = np.array([strat_comp.get(strat, {}).get(oc, {}).get('coef', np.nan)
                          for oc in outcomes])
        ses   = np.array([strat_comp.get(strat, {}).get(oc, {}).get('se', 0.)
                          for oc in outcomes])
        offset = x + (j - 0.5) * bar_w
        valid  = ~np.isnan(betas)
        ax.bar(offset[valid], betas[valid], width=bar_w * 0.88,
               color=color, alpha=0.85, label=lbl, zorder=2)
        ax.errorbar(offset[valid], betas[valid], yerr=1.96 * ses[valid],
                    fmt='none', ecolor='#333', capsize=4, lw=1.2, zorder=3)

    ax.axhline(0, color='black', lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(oc_labels, fontsize=11)
    ax.set_ylabel(r'DiD $\hat\theta$ ($W_i \times$ post; per unit $W_i$)', fontsize=10)
    ax.set_title('Strategy A vs. Strategy B: investment elasticity estimates\n'
                 'Baseline spec; firm + year FE; SE clustered at firm',
                 fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'fig_strategy_comparison.pdf')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX tables
# ═══════════════════════════════════════════════════════════════════════════

def write_elasticity_table(main_results, tables_dir):
    outcomes = ['capx_intensity', 'rd_intensity', 'total_inv_intensity']
    oc_hdrs  = ['Capex/A', 'R\\&D/A', 'Total Inv/A']
    specs    = ['Baseline', 'w/ controls', '+ Ind\\texttimes yr FE', 'Strategy B']

    def _fmt(r):
        b, s = r.get('coef', np.nan), r.get('se', np.nan)
        if np.isnan(b): return '---', ''
        t  = abs(b/s) if s > 0 else 0
        st = '***' if t>2.576 else ('**' if t>1.96 else ('*' if t>1.645 else ''))
        return f'{b:.3f}{st}', f'({s:.3f})'

    n_spec = len(specs); n_oc = len(outcomes); total = n_spec * n_oc
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Investment elasticity to TCJA deferred tax windfall}',
        r'\label{tab:inv_elasticity}', r'\small',
        r'\begin{tabular}{l' + 'c'*total + r'}', r'\toprule',
    ]
    # Outcome headers
    oc_row = ''; midr = ''
    col = 2
    for h in oc_hdrs:
        oc_row += f' & \\multicolumn{{{n_spec}}}{{c}}{{{h}}}'
        midr   += f'\\cmidrule(lr){{{col}-{col+n_spec-1}}}'
        col    += n_spec
    lines += [oc_row + r' \\', midr]
    lines.append('& ' + ' & '.join(f'({i+1})' for i in range(total)) + r' \\')
    spec_names_clean = ['Baseline', 'Controls', '+Ind$\\times$yr', 'Strat. B']
    lines.append('& ' + ' & '.join(spec_names_clean * n_oc) + r' \\')
    lines.append(r'\midrule')

    coef_row, se_row, n_row, g_row, r2_row = [], [], [], [], []
    for oc in outcomes:
        for sp in specs:
            r = main_results.get(oc, {}).get(sp, {})
            cv, sv = _fmt(r)
            coef_row.append(cv); se_row.append(sv)
            n_row.append(str(r.get('n', '---')))
            g_row.append(str(r.get('G', '---')))
            rsq = r.get('rsq', float('nan'))
            r2_row.append(f'{rsq:.3f}' if not np.isnan(rsq) else '---')

    lines += [
        r'$W_i \times \text{post}$ & ' + ' & '.join(coef_row) + r' \\',
        '& ' + ' & '.join(se_row) + r' \\',
        r'\midrule',
        r'Observations & ' + ' & '.join(n_row) + r' \\',
        r'Firms & '        + ' & '.join(g_row) + r' \\',
        r'$R^2$ & '        + ' & '.join(r2_row) + r' \\',
        r'\midrule',
    ]
    fe_firm  = ['Y'] * total
    fe_yr    = (['Y', 'Y', '--', 'Y'] * n_oc)[:total]
    fe_indyr = (['--', '--', 'Y', '--'] * n_oc)[:total]
    fe_ctrl  = (['--', 'Y', 'Y', '--'] * n_oc)[:total]
    lines += [
        r'Firm FE & '                + ' & '.join(fe_firm)  + r' \\',
        r'Year FE & '                + ' & '.join(fe_yr)    + r' \\',
        r'Ind.$\times$year FE & '    + ' & '.join(fe_indyr) + r' \\',
        r'Pre-period controls & '    + ' & '.join(fe_ctrl)  + r' \\',
        r'\bottomrule', r'\end{tabular}',
        r'\begin{tablenotes}\footnotesize',
        r'\item $W_i = 0.40\times(\mathtt{txdb}_{2016}/\mathtt{at}_{2015})$. '
        r'Shock year 2017 excluded from post-period. '
        r'Strategy B uses within-industry deviation of $Z_i$. '
        r'All outcomes scaled by lagged assets.',
        r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. SE clustered at firm.',
        r'\end{tablenotes}', r'\end{table}',
    ]
    path = os.path.join(tables_dir, 'tbl_main_elasticity.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


def write_windfall_table(windfall_res, tables_dir):
    display = [
        ('capx_intensity',        'Capital expenditure'),
        ('rd_intensity',          'R\\&D expense'),
        ('payout_intensity',      'Payouts (div.\\ + buybacks)'),
        ('emp_growth',            'Employment growth$^{*}$'),
        ('debt_change_intensity', 'Debt change (+ = increase)'),
    ]
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{TCJA windfall decomposition}',
        r'\label{tab:windfall}', r'\small',
        r'\begin{tabular}{lccc}', r'\toprule',
        r'Outcome & Coef. & SE & 95\% CI \\', r'\midrule',
    ]
    implied = 0.
    for key, label in display:
        r = windfall_res.get(key, {})
        b, s = r.get('coef', np.nan), r.get('se', np.nan)
        if np.isnan(b):
            lines.append(f'{label} & --- & --- & --- \\\\')
            continue
        t  = abs(b/s) if s > 0 else 0
        st = '***' if t>2.576 else ('**' if t>1.96 else ('*' if t>1.645 else ''))
        lo, hi = b - 1.96*s, b + 1.96*s
        lines.append(f'{label} & ${b:+.3f}{st}$ & $({s:.3f})$ & $[{lo:.3f},\\ {hi:.3f}]$ \\\\')
        if key != 'emp_growth':
            implied += b
    lines += [
        r'\midrule',
        f'\\emph{{Implied total (excl.\ emp.)}} & ${implied:+.3f}$ & & \\\\',
        r'\bottomrule', r'\end{tabular}',
        r'\begin{tablenotes}\footnotesize',
        r'\item All outcomes (except employment growth) scaled by lagged total assets; '
        r'coefficients directly comparable. Sign of debt change not flipped.',
        r'\item $^{*}$ Log-change; cannot be summed with level intensities.',
        r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. SE clustered at firm.',
        r'\end{tablenotes}', r'\end{table}',
    ]
    path = os.path.join(tables_dir, 'tbl_windfall_decomposition.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


# ═══════════════════════════════════════════════════════════════════════════
# Validation report
# ═══════════════════════════════════════════════════════════════════════════

def validation_report(treat, panel, mech_coef, mech_r2,
                       b17_A, b17_B, did_ni_post,
                       main_results, windfall_res):
    print('\n' + '='*65)
    print('  TCJA INVESTMENT ELASTICITY — VALIDATION REPORT')
    print('='*65)

    print(f'\nSAMPLE CONSTRUCTION')
    print(f'  Final balanced firms (non-Finance):  {panel["gvkey"].nunique():,}')
    print(f'  Final panel firm-years:              {len(panel):,}')

    print(f'\nTREATMENT DISTRIBUTION')
    print(f'  Z_i median:    {treat["Z_w"].median():.4f}  (expected ~0.02)')
    print(f'  Z_i p5 / p95:  {treat["Z_w"].quantile(.05):.4f} / {treat["Z_w"].quantile(.95):.4f}')
    print(f'  Share net-DTL: {(treat["Z"]>0).mean()*100:.1f}%  (expected 60–70%)')

    print(f'\nMECHANICAL PASS-THROUGH')
    print(f'  Slope of 2017 rate-change on Z_i:  {mech_coef:.2f}')
    print(f'  R²:                                 {mech_r2:.3f}  (expected >0.10)')

    print(f'\nFIRST STAGE ON NET INCOME')
    print(f'  β_2017 (Strategy A):              {b17_A:.3f}  (expected: large +)')
    print(f'  β_2017 (Strategy B, within-ind):  {b17_B:.3f}  (should be similar to A)')
    print(f'  β_pooled_2018-22 (A):             {did_ni_post.get("coef", float("nan")):.3f}  (expected: ~0)')

    print(f'\nMAIN INVESTMENT RESULTS')
    for oc in ['capx_intensity', 'rd_intensity', 'total_inv_intensity']:
        for sp in ['Baseline', 'w/ controls', 'Strategy B']:
            r = main_results.get(oc, {}).get(sp, {})
            b, s = r.get('coef', float('nan')), r.get('se', float('nan'))
            t = abs(b/s) if s > 0 else 0
            print(f'  {oc[:18]:<18} [{sp:<15}]  θ={b:+.4f}  SE={s:.4f}  {stars(t)}')

    print(f'\nWINDFALL DECOMPOSITION')
    implied = 0.
    for key in ['capx_intensity', 'rd_intensity', 'payout_intensity',
                'che_change', 'debt_change_intensity']:
        r = windfall_res.get(key, {})
        b = r.get('coef', float('nan'))
        s = r.get('se', float('nan'))
        if not np.isnan(b):
            t = abs(b/s) if s > 0 else 0
            print(f'  {key:<28}: {b:+.4f}  {stars(t)}')
            implied += b
    print(f'  Implied total (capx+rd+payout+che+debt): {implied:+.4f}')
    print(f'  (Fraction accounted for: ~{100*implied:.0f}% of W unit)')
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',
                        default=os.path.join(script_dir, '..', 'data', 'analysis', 'cache'))
    parser.add_argument('--base-panel',
                        default=os.path.join(script_dir, '..', 'data', 'analysis', 'output',
                                             'tax_reconciliation_panel_base.csv'))
    parser.add_argument('--figures-dir',
                        default=os.path.join(script_dir, '..', 'data', 'analysis', 'figures', 'investment'))
    parser.add_argument('--tables-dir',
                        default=os.path.join(script_dir, '..', 'data', 'analysis', 'tables', 'investment'))
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.tables_dir,  exist_ok=True)

    # ── Load caches ──────────────────────────────────────────────────────────
    print('Loading caches...')
    combined = pickle.load(open(os.path.join(args.cache_dir, 'combined.pkl'), 'rb'))
    long_df  = pickle.load(open(os.path.join(args.cache_dir, 'long.pkl'),    'rb'))

    print('\nLoading extended Compustat from base panel...')
    comp = load_comp_extra(args.base_panel)

    # ── Treatment construction ───────────────────────────────────────────────
    print('\nBuilding treatment variable...')
    treat = build_treatment(comp, long_df)

    print('\nBuilding balanced event panel...')
    panel_raw = build_event_panel(combined, treat, comp, EVENT_YEARS, PRE_YEARS, POST_YEARS)

    print('\nBuilding outcome variables...')
    panel = build_outcomes(panel_raw)

    print('\nBuilding controls...')
    panel = build_controls(panel)

    print('\nBuilding Strategy B (within-industry)...')
    strat_b = build_strategy_b(panel, treat)
    panel = panel.merge(strat_b, on='gvkey', how='left')

    # ── First-stage validation ───────────────────────────────────────────────
    print('\nSection II: First-stage validation...')

    # Mechanical pass-through (Z_i → 2017 TCJA/rate change contribution)
    tcja_2017 = (long_df[(long_df['recon_year'] == 2017) &
                         (long_df['category'] == 'TCJA/rate change')]
                 .groupby('gvkey')['cat_val'].sum()
                 .rename('tcja_2017_pp'))

    ind_map    = panel.drop_duplicates('gvkey').set_index('gvkey')['industry']
    firm_level = treat.merge(tcja_2017, on='gvkey', how='left')
    firm_level['tcja_2017_pp'] = firm_level['tcja_2017_pp'].fillna(0)
    firm_level = firm_level.merge(ind_map, on='gvkey', how='left')

    sub_fs = firm_level.dropna(subset=['Z_w', 'tcja_2017_pp'])
    X_fs   = np.column_stack([np.ones(len(sub_fs)), sub_fs['Z_w'].values])
    m_fs   = OLS(sub_fs['tcja_2017_pp'].values, X_fs).fit()
    mech_coef, mech_r2 = float(m_fs.params[1]), float(m_fs.rsquared)
    print(f'  Pass-through slope (raw): {mech_coef:.3f}  R²={mech_r2:.3f}')

    # Winsorize reconciliation contribution at [-15, +15] pp (realistic range)
    firm_level['tcja_2017_pp_w'] = firm_level['tcja_2017_pp'].clip(-15, 15)
    sub_w  = firm_level.dropna(subset=['Z_w', 'tcja_2017_pp_w'])
    X_w    = np.column_stack([np.ones(len(sub_w)), sub_w['Z_w'].values])
    m_w    = OLS(sub_w['tcja_2017_pp_w'].values, X_w).fit()
    mech_coef_w, mech_r2_w = float(m_w.params[1]), float(m_w.rsquared)
    print(f'  Pass-through slope (winsorized ±15pp): {mech_coef_w:.3f}  R²={mech_r2_w:.3f}')
    firm_level['tcja_2017_pp_orig'] = firm_level['tcja_2017_pp']
    firm_level['tcja_2017_pp'] = firm_level['tcja_2017_pp_w']  # use winsorized for scatter

    # Event study: NI
    print('  Event study: net income...')
    ev_ni_A, info_ni = run_event_study(panel, 'ni_intensity', 'W', EVENT_YEARS,
                                       ind_yr_fe=True)
    ev_ni_B, _       = run_event_study(panel, 'ni_intensity', 'W_within', EVENT_YEARS,
                                       ind_yr_fe=True)
    did_ni_post      = run_did(panel, 'ni_intensity', 'W', ind_yr_fe=True)

    b17_A = (float(ev_ni_A.loc[ev_ni_A['year']==2017, 'beta'].iloc[0])
             if not ev_ni_A.empty and 2017 in ev_ni_A['year'].values else float('nan'))
    b17_B = (float(ev_ni_B.loc[ev_ni_B['year']==2017, 'beta'].iloc[0])
             if not ev_ni_B.empty and 2017 in ev_ni_B['year'].values else float('nan'))
    print(f'  β_2017 (A)={b17_A:.3f}  β_2017 (B)={b17_B:.3f}  '
          f'β_post={did_ni_post.get("coef", float("nan")):.3f}')

    n_firms = panel['gvkey'].nunique()
    n_obs   = len(panel)

    # ── First-stage figures ──────────────────────────────────────────────────
    print('\nGenerating first-stage figures...')
    fig_firststage_dtl(firm_level, args.figures_dir)
    fig_firststage_ni(ev_ni_A, did_ni_post, n_firms, n_obs, args.figures_dir)

    # ── Section III: Investment elasticity ───────────────────────────────────
    print('\nSection III: Investment elasticity regressions...')
    ctrl_base = [c for c in ['log_at_2016', 'leverage_2016',
                              'cash_ratio_pre3y', 'capx_intensity_pre3y']
                 if c in panel.columns]

    main_results = {oc: {} for oc in ['capx_intensity', 'rd_intensity', 'total_inv_intensity']}
    for oc in main_results:
        print(f'  {oc}:')
        main_results[oc]['Baseline']             = run_did(panel, oc, 'W',
                                                           controls=ctrl_base, ind_yr_fe=True)
        main_results[oc]['w/ controls']          = run_did(panel, oc, 'W', controls=ctrl_base,
                                                           ind_yr_fe=True)
        main_results[oc]['+ Ind\\texttimes yr FE'] = run_did(panel, oc, 'W',
                                                              controls=ctrl_base, ind_yr_fe=True)
        main_results[oc]['Strategy B']           = run_did(panel, oc, 'W_within',
                                                           ind_yr_fe=True)
        for sp, r in main_results[oc].items():
            t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
            print(f'    [{sp:<25}] θ={r["coef"]:+.4f} SE={r["se"]:.4f} {stars(t)} n={r["n"]:,}')

    # Event studies on investment
    print('\nEvent studies: investment outcomes...')
    ctrl_ev = [c for c in ['log_at_2016', 'cash_ratio_pre3y'] if c in panel.columns]
    ev_capx,  _ = run_event_study(panel, 'capx_intensity',     'W', EVENT_YEARS,
                                   controls=ctrl_ev if ctrl_ev else None,
                                   ind_yr_fe=True)
    ev_rd,    _ = run_event_study(panel, 'rd_intensity',        'W', EVENT_YEARS,
                                   controls=ctrl_ev if ctrl_ev else None,
                                   ind_yr_fe=True)
    ev_total, _ = run_event_study(panel, 'total_inv_intensity', 'W', EVENT_YEARS,
                                   controls=ctrl_ev if ctrl_ev else None,
                                   ind_yr_fe=True)

    # §965 repatriation controls — merge firm-level repatriation exposure
    print('\nBuilding §965 repatriation controls...')
    repat_2017 = (long_df[(long_df['recon_year'] == 2017) &
                          (long_df['category'] == 'TCJA/repatriation')]
                  .groupby('gvkey')['cat_val'].sum()
                  .rename('repat_2017_pp'))
    panel = panel.merge(repat_2017, on='gvkey', how='left')
    panel['repat_2017_pp'] = panel['repat_2017_pp'].fillna(0)
    # Share of firms with non-trivial repatriation exposure
    repat_exposed = (panel.drop_duplicates('gvkey')['repat_2017_pp'].abs() > 1).mean()
    print(f'  Firms with |repat_2017| > 1pp: {repat_exposed*100:.1f}%')

    # Heterogeneity: size, cash, leverage, intangibles terciles
    # hetero_res structure: {dim: {cell: {outcome: result}}}
    print('\nHeterogeneity analysis...')
    at_2016 = panel[panel['recon_year'] == 2016].groupby('gvkey')['at'].first()
    panel['at_2016'] = panel['gvkey'].map(at_2016)

    hetero_dims = [('Size',       'at_2016'),
                   ('Cash',       'cash_ratio_pre3y'),
                   ('Leverage',   'leverage_2016'),
                   ('Intangibles','intan_pre3y')]
    hetero_ocs = ['capx_intensity', 'total_inv_intensity']

    hetero_res = {}
    for dim_name, col in hetero_dims:
        if col not in panel.columns:
            continue
        firm_vals = panel.drop_duplicates('gvkey').set_index('gvkey')[col].dropna()
        if firm_vals.nunique() < 6:
            continue
        q = pd.qcut(firm_vals, q=3,
                    labels=[f'{dim_name}-Low', f'{dim_name}-Mid', f'{dim_name}-High'],
                    duplicates='drop')
        panel[f'_terc_{dim_name}'] = panel['gvkey'].map(q)
        hetero_res[dim_name] = {}
        for cell in sorted(panel[f'_terc_{dim_name}'].dropna().unique()):
            sub_h = panel[panel[f'_terc_{dim_name}'] == cell]
            if sub_h['gvkey'].nunique() < 30:
                continue
            hetero_res[dim_name][str(cell)] = {}
            for oc in hetero_ocs:
                r = run_did(sub_h, oc, 'W', ind_yr_fe=True)
                if not np.isnan(r['coef']):
                    hetero_res[dim_name][str(cell)][oc] = r
                    t = abs(r['coef']/r['se']) if r.get('se',0) > 0 else 0
                    print(f'  {dim_name}/{cell}/{oc[:8]}: θ={r["coef"]:+.4f} SE={r["se"]:.4f} {stars(t)}')

    # Windfall decomposition
    print('\nWindfall decomposition...')
    decomp_ocs = ['capx_intensity', 'rd_intensity', 'payout_intensity',
                  'emp_growth', 'debt_change_intensity', 'che_change']
    windfall_res = {}
    for oc in decomp_ocs:
        r = run_did(panel, oc, 'W', ind_yr_fe=True)
        windfall_res[oc] = r
        t = abs(r['coef']/r['se']) if r.get('se', 0) > 0 else 0
        print(f'  {oc:<28} θ={r["coef"]:+.4f} SE={r["se"]:.4f} {stars(t)}')

    # Debt regression conditioned on §965 repatriation exposure
    # repat_2017_pp is firm-level (time-invariant) → absorbed by firm FE.
    # Must interact with post to get within-firm variation that survives demeaning.
    print('\nDebt regression conditioned on §965 repatriation...')
    panel['repat_x_post'] = (panel['repat_2017_pp'] *
                              panel['recon_year'].isin(POST_YEARS).astype(float))
    r_debt_ctrl = run_did(panel, 'debt_change_intensity', 'W',
                          controls=['repat_x_post'], ind_yr_fe=True)
    t_dc = abs(r_debt_ctrl['coef']/r_debt_ctrl['se']) if r_debt_ctrl.get('se',0) > 0 else 0
    print(f'  Debt DiD (+ repat×post control): θ={r_debt_ctrl["coef"]:+.4f} '
          f'SE={r_debt_ctrl["se"]:.4f} {stars(t_dc)}  (baseline was {windfall_res["debt_change_intensity"]["coef"]:+.4f})')

    # Strategy A vs B comparison
    print('\nStrategy A vs B comparison...')
    strat_comp = {'A': {}, 'B': {}}
    for oc in ['capx_intensity', 'rd_intensity', 'total_inv_intensity']:
        strat_comp['A'][oc] = run_did(panel, oc, 'W', ind_yr_fe=True)
        strat_comp['B'][oc] = run_did(panel, oc, 'W_within', ind_yr_fe=True)

    # ── Figures ──────────────────────────────────────────────────────────────
    print('\nGenerating investment figures...')
    fig_investment_event(ev_capx, ev_rd, ev_total, n_firms, n_obs, args.figures_dir)
    if hetero_res:
        fig_investment_hetero(hetero_res, args.figures_dir)
    fig_windfall_decomp(windfall_res, args.figures_dir)
    fig_strategy_comparison(strat_comp, args.figures_dir)

    # ── Tables ───────────────────────────────────────────────────────────────
    print('\nWriting LaTeX tables...')
    write_elasticity_table(main_results, args.tables_dir)
    write_windfall_table(windfall_res, args.tables_dir)

    # ── Summary JSON consumed by analysis/build_slide_data.py ────────────────
    debt_baseline = windfall_res["debt_change_intensity"]
    windfall_summary = {
        "ni_shock_2017_strat_a": {"coef": float(b17_A)},
        "ni_shock_2017_strat_b": {"coef": float(b17_B)},
        "ni_shock_post": {
            "coef": float(did_ni_post.get("coef", float("nan"))),
            "se":   float(did_ni_post.get("se",   float("nan"))),
        },
        "debt_baseline": {
            "coef": float(debt_baseline["coef"]),
            "se":   float(debt_baseline["se"]),
        },
        "debt_sec965_controlled": {
            "coef": float(r_debt_ctrl["coef"]),
            "se":   float(r_debt_ctrl["se"]),
            "stars": stars(t_dc),
        },
    }
    summary_path = os.path.join(args.tables_dir, 'windfall_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(windfall_summary, f, indent=2)
    print(f'  wrote {summary_path}')

    # ── Validation ───────────────────────────────────────────────────────────
    validation_report(treat, panel, mech_coef, mech_r2,
                       b17_A, b17_B, did_ni_post,
                       main_results, windfall_res)

    print('Done.')
