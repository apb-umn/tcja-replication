#!/usr/bin/env python3
"""
build_profit_shifting_dynamics.py

Dynamic effects of TCJA on firm-level domestic-vs-foreign PI composition.

Extends build_substitution_real_outcomes.py with:
  - Longer event window (2011–2023) to study adjustment dynamics.
  - Horizon-specific ATT estimates at h ∈ {1, 2, 3, 5} post-TCJA.
  - Log-share transformation of the foreign share as a semi-elasticity robustness check.
  - Industry-stratified pooled DiD.
  - Firm-specific linear-trend spec.

Sign convention (see spec §"Sign convention"):
  intl_pre is in pp, typically NEGATIVE for profit shifters.
  foreign_pi_share is in [0, 1].
  A POSITIVE β on (intl_pre × post) means:
    heavy shifter (intl_pre = -5) → predicted Δshare = -5 × β < 0,
  i.e. foreign share FALLS post-TCJA. This is the re-shoring prediction.
  Coefficients are reported in RAW sign (not flipped).
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import f as f_dist
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

from sic_industry import industry  # canonical SIC -> industry classifier

warnings.filterwarnings('ignore')
mpl.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'axes.grid': True, 'grid.alpha': 0.22,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'figures', 'shifting_dynamics')
TBL_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'tables',  'shifting_dynamics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

PRE_YEARS   = list(range(2013, 2017))       # 2013–2016
POST_YEARS  = list(range(2019, 2024))       # 2019–2023 (includes 2023)
EVENT_YEARS = list(range(2011, 2024))       # 2011–2023
OMIT_YEAR   = 2016
WIN_CH      = 15.0                          # ±pp per-obs winsor on cat_val
EPS         = 0.01                          # log-share smoothing
HORIZONS    = [1, 2, 3, 5]                  # years after 2017 → 2018,19,20,22

IND_ORDER   = ['Tech', 'Pharma/Biotech', 'Manufacturing', 'Retail', 'Oil & Gas', 'Other']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────



def stars(t):
    a = abs(t)
    return '***' if a>2.576 else ('**' if a>1.96 else ('*' if a>1.645 else ''))


def piw_mean(vals, weights):
    v = np.asarray(vals, dtype=float); w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Within-transformations (alternating projections)
# ─────────────────────────────────────────────────────────────────────────────

def demean_twfe(df, cols, firm_col='gvkey', year_col='recon_year',
                tol=1e-9, max_iter=500):
    """Firm + year FE demeaning."""
    out = df[cols].copy().astype(float)
    for _ in range(max_iter):
        prev = out.values.copy()
        for c in cols:
            out[c] -= out[c].groupby(df[firm_col]).transform('mean')
            out[c] -= out[c].groupby(df[year_col]).transform('mean')
        if np.abs(out.values - prev).max() < tol:
            break
    return out


def demean_ind_yr(df, cols, firm_col='gvkey', ind_col='industry',
                  year_col='recon_year', tol=1e-9, max_iter=500):
    """Firm + industry×year FE demeaning."""
    df = df.copy()
    df['_ind_yr'] = df[ind_col].astype(str) + '_' + df[year_col].astype(str)
    out = df[cols].copy().astype(float)
    for _ in range(max_iter):
        prev = out.values.copy()
        for c in cols:
            out[c] -= out[c].groupby(df[firm_col]).transform('mean')
            out[c] -= out[c].groupby(df['_ind_yr']).transform('mean')
        if np.abs(out.values - prev).max() < tol:
            break
    return out


def residualize_firm_trend(df, col, firm_col='gvkey', year_col='recon_year'):
    """Within-firm residual after (1, year) regression for one column."""
    x = df[[firm_col, year_col, col]].copy()
    x['t'] = x[year_col].astype(float)
    t_mean = x.groupby(firm_col)['t'].transform('mean')
    y_mean = x.groupby(firm_col)[col].transform('mean')
    t_dm   = x['t'] - t_mean
    y_dm   = x[col] - y_mean
    num    = (t_dm * y_dm).groupby(x[firm_col]).transform('sum')
    den    = (t_dm ** 2).groupby(x[firm_col]).transform('sum')
    slope  = np.where(den > 0, num / den, 0.0)
    return (y_dm - slope * t_dm).values


def demean_firm_year_trend(df, cols, firm_col='gvkey', year_col='recon_year',
                            tol=1e-9, max_iter=500):
    """Firm FE + year FE + firm-specific linear year trend."""
    out = df[cols].copy().astype(float)
    # Initial residualize: firm FE + firm trend
    for c in cols:
        out[c] = residualize_firm_trend(pd.concat([df[[firm_col, year_col]], out[[c]]],
                                                   axis=1), c,
                                        firm_col=firm_col, year_col=year_col)
    # Alternate: year FE → re-residualize firm+trend
    for _ in range(max_iter):
        prev = out.values.copy()
        for c in cols:
            out[c] -= out[c].groupby(df[year_col]).transform('mean')
        for c in cols:
            out[c] = residualize_firm_trend(
                pd.concat([df[[firm_col, year_col]], out[[c]]], axis=1),
                c, firm_col=firm_col, year_col=year_col)
        if np.abs(out.values - prev).max() < tol:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────────────

def _fit_cluster(y, X, groups):
    mod    = OLS(y, X).fit()
    cl_cov = cov_cluster(mod, groups)
    cl_se  = np.sqrt(np.diag(cl_cov))
    return mod, cl_se, cl_cov


def run_event_study(panel, outcome, treat_col='intl_pre', spec='baseline'):
    """Event study with chosen FE structure.

    spec ∈ {'baseline', 'ind_yr', 'firm_trend'}.
    Returns DataFrame with year, beta, se, lower, upper.
    """
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

    if   spec == 'ind_yr':     dm = demean_ind_yr(sub, all_cols)
    elif spec == 'firm_trend': dm = demean_firm_year_trend(sub, all_cols)
    else:                       dm = demean_twfe(sub, all_cols)

    mod, se, _ = _fit_cluster(dm[outcome].values, dm[ix_cols].values,
                               sub['gvkey'].values)
    rows = [{'year': yr, 'beta': float(mod.params[i]), 'se': float(se[i]),
             'lower': float(mod.params[i] - 1.96*se[i]),
             'upper': float(mod.params[i] + 1.96*se[i])}
            for i, yr in enumerate(plot_yrs)]
    rows.append({'year': OMIT_YEAR, 'beta': 0., 'se': 0., 'lower': 0., 'upper': 0.})
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


def run_pooled_did(panel, outcome, treat_col='intl_pre', spec='baseline'):
    excl  = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [yr for yr in EVENT_YEARS if yr not in excl]
    mask = (panel['recon_year'].isin(years) &
            panel[outcome].notna() & panel[treat_col].notna())
    sub = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return {'coef': np.nan, 'se': np.nan, 'n': 0, 'G': 0, 'r2': np.nan}
    sub['_post'] = sub['recon_year'].isin(POST_YEARS).astype(float)
    sub['_Di']   = sub[treat_col] * sub['_post']
    cols = [outcome, '_Di']

    if   spec == 'ind_yr':     dm = demean_ind_yr(sub, cols)
    elif spec == 'firm_trend': dm = demean_firm_year_trend(sub, cols)
    else:                       dm = demean_twfe(sub, cols)

    mod, se, _ = _fit_cluster(dm[outcome].values, dm[['_Di']].values,
                               sub['gvkey'].values)
    y = dm[outcome].values
    y_hat = dm[['_Di']].values @ mod.params
    ss_res = ((y - y_hat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'coef': float(mod.params[0]), 'se': float(se[0]),
            'n': len(sub), 'G': sub['gvkey'].nunique(), 'r2': float(r2)}


def pre_trend_ftest(es, pre_years=(2011,2012,2013,2014,2015)):
    """Joint Wald/F-test that pre-period event-study coefs are all zero.
    Uses the reported se as diagonal; returns a χ²/df approx F-statistic."""
    sub = es[es['year'].isin(pre_years) & (es['year'] != OMIT_YEAR) & (es['se'] > 0)]
    if len(sub) == 0: return {'F': np.nan, 'df': 0, 'p': np.nan}
    z = (sub['beta'] / sub['se']).values
    chi2 = (z**2).sum()
    df = len(sub)
    F = chi2 / df
    p = 1 - f_dist.cdf(F, df, 1000)   # approx, large-n df for denom
    return {'F': float(F), 'df': int(df), 'p': float(p), 'chi2': float(chi2)}


# ─────────────────────────────────────────────────────────────────────────────
# Section I: Load data + build panel
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'combined.pkl'), 'rb') as f:
        combined = pickle.load(f)
    with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'long.pkl'), 'rb') as f:
        long_df = pickle.load(f)
    pi_csv = os.path.join(script_dir, '..', 'data', 'analysis', 'output', 'tax_reconciliation_panel_base.csv')
    pi = pd.read_csv(pi_csv,
                     usecols=['gvkey','recon_year','comp_pidom','comp_pifo','comp_at'],
                     low_memory=False).drop_duplicates(['gvkey','recon_year'])
    print(f'  combined: {len(combined):,} rows, {combined["gvkey"].nunique():,} firms')
    print(f'  long_df:  {len(long_df):,} rows')
    print(f'  pi merge: {len(pi):,} rows (pidom/pifo/at)')
    return combined, long_df, pi


def build_intl_pre(long_df, win_ch=WIN_CH):
    """Same construction as build_substitution_analysis: +/- win_ch per-obs clip
    on all groups, then mean over pre-period, then [p1,p99] clip across firms.
    Includes domestic zeros for firms with >=2 pre-period obs but no Intl entries.
    `win_ch=None` disables the per-firm-year-channel clip.
    """
    fy = (long_df.groupby(['gvkey','recon_year','group'])['cat_val']
          .sum().reset_index(name='val'))
    if win_ch is None:
        fy['val_w'] = fy['val']
    else:
        fy['val_w'] = fy['val'].clip(-win_ch, win_ch)
    intl_fy = (fy[fy['group'] == 'International']
               .groupby(['gvkey','recon_year'])['val_w'].sum())
    any_obs = fy.groupby(['gvkey','recon_year']).size()
    pre_any = any_obs[any_obs.index.get_level_values('recon_year').isin(PRE_YEARS)]
    elig = pre_any.groupby('gvkey').size()
    elig = elig[elig >= 2].index
    pre_intl = (intl_fy[intl_fy.index.get_level_values('recon_year').isin(PRE_YEARS)
                        & intl_fy.index.get_level_values('gvkey').isin(elig)]
                .groupby('gvkey').mean())
    intl_pre = pd.Series(0.0, index=elig, name='intl_pre')
    intl_pre.update(pre_intl)
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)
    print(f'  intl_pre: {len(intl_pre):,} firms  '
          f'[p1,p99]=[{p1:.2f},{p99:.2f}]  median={intl_pre.median():.2f}')
    return intl_pre


def build_panel(combined, pi, intl_pre):
    panel = combined[combined['recon_year'].isin(EVENT_YEARS)].copy()
    panel['industry'] = panel['comp_sic'].apply(industry)
    panel = panel[panel['industry'] != 'Finance']

    # Merge treatment + PI composition
    panel = panel.merge(intl_pre.reset_index(), on='gvkey', how='inner')
    panel = panel.merge(pi, on=['gvkey','recon_year'], how='left')

    # ── Reconciliation check: |pidom+pifo - pi| / |pi| ≤ 0.20 ────────────────
    panel['total_pi_footnote'] = panel['comp_pidom'].fillna(0) + panel['comp_pifo'].fillna(0)
    panel['pi_gap_rel'] = np.where(
        panel['comp_pi'].abs() > 1,
        (panel['total_pi_footnote'] - panel['comp_pi']).abs() / panel['comp_pi'].abs(),
        np.nan)
    panel['pi_recon_ok'] = panel['pi_gap_rel'] < 0.20
    firm_ok_frac = panel.groupby('gvkey')['pi_recon_ok'].mean()
    ok_firms = firm_ok_frac[firm_ok_frac >= 0.5].index
    panel = panel[panel['gvkey'].isin(ok_firms)].copy()

    # ── Primary outcome: foreign_pi_share ─────────────────────────────────────
    valid = (panel['comp_pidom'].notna() & panel['comp_pifo'].notna()
             & (panel['total_pi_footnote'] > 0))
    panel['foreign_pi_share'] = np.where(valid,
        panel['comp_pifo'] / panel['total_pi_footnote'], np.nan)
    panel['foreign_pi_share'] = panel['foreign_pi_share'].clip(0, 1)
    panel['domestic_pi_share'] = 1 - panel['foreign_pi_share']

    # ── Log-share transformation ──────────────────────────────────────────────
    panel['foreign_share_log'] = np.log(panel['foreign_pi_share'] + EPS)

    # ── Require ≥2 pre AND ≥2 post observations of the OUTCOME ────────────────
    share_obs = panel.dropna(subset=['foreign_pi_share'])
    fy = share_obs.groupby('gvkey')['recon_year'].apply(set)
    has_pre  = fy.apply(lambda s: len(s & set(PRE_YEARS)) >= 2)
    has_post = fy.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok = fy[has_pre & has_post].index
    panel = panel[panel['gvkey'].isin(ok)].copy()

    print(f'  Panel: {panel["gvkey"].nunique():,} firms, {len(panel):,} firm-years')
    print(f'  Firms with foreign_share observed: '
          f'{panel.dropna(subset=["foreign_pi_share"])["gvkey"].nunique():,}')

    # ── Outcome diagnostics ───────────────────────────────────────────────────
    fs = panel['foreign_pi_share'].dropna()
    print(f'  foreign_pi_share: median={fs.median():.3f}  '
          f'P25={fs.quantile(.25):.3f}  P75={fs.quantile(.75):.3f}  '
          f'P90={fs.quantile(.90):.3f}')
    print(f'    share == 0: {(fs==0).mean()*100:.1f}%   '
          f'share == 1: {(fs==1).mean()*100:.1f}%')
    wfsd = panel.groupby('gvkey')['foreign_pi_share'].std().mean()
    print(f'    within-firm SD (mean across firms): {wfsd:.4f}')
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Section II: Time-series by pre-TCJA exposure tercile
# ─────────────────────────────────────────────────────────────────────────────

def fig_shares_timeseries(panel):
    print('  Fig 1: foreign share time-series by intl_pre tercile...')
    firm_t = panel.drop_duplicates('gvkey').set_index('gvkey')['intl_pre']
    firm_t = firm_t.dropna()
    try:
        terc = pd.qcut(firm_t, q=3, labels=['Heavy (most intl)','Medium','Light (domestic)'],
                       duplicates='drop')
    except ValueError:
        terc = pd.cut(firm_t, bins=3, labels=['Heavy (most intl)','Medium','Light (domestic)'])
    panel = panel.copy()
    panel['terc'] = panel['gvkey'].map(terc)

    colors = {'Heavy (most intl)':   '#c0392b',
              'Medium':               '#e67e22',
              'Light (domestic)':     '#2980b9'}

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    for label, col in colors.items():
        sub = panel[panel['terc'] == label]
        ser = sub.dropna(subset=['foreign_pi_share'])
        yearly = (ser.groupby('recon_year')
                     .apply(lambda g: piw_mean(g['foreign_pi_share'].values,
                                               g['comp_pi'].values))
                     .rename('fs').reset_index())
        n_firms = sub['gvkey'].nunique()
        mean_t  = firm_t[firm_t.index.isin(sub['gvkey'].unique())].mean()
        ax.plot(yearly['recon_year'], yearly['fs'],
                color=col, lw=2.1, marker='o', ms=5,
                label=f'{label} (N={n_firms}, $\\bar{{c}}^{{pre}}_{{Intl}}\\approx{mean_t:.1f}$ pp)')

    ax.axvline(2017.5, color='black', ls='--', lw=1.2, alpha=0.65)
    ax.text(2017.6, ax.get_ylim()[1]*0.97 if ax.get_ylim()[1]>0 else 0.7,
            'TCJA', fontsize=9, va='top', color='black')
    ax.set_xlabel('Fiscal year')
    ax.set_ylabel('Foreign PI share (PI-weighted mean)')
    ax.set_xticks(EVENT_YEARS)
    ax.set_xticklabels([str(y) for y in EVENT_YEARS], rotation=45, ha='right', fontsize=8.5)
    ax.set_title('Foreign PI share over time, by pre-TCJA international-exposure tercile\n'
                 'Heavy-shifter tercile has lower foreign share post-TCJA — income re-booked to the US',
                 fontsize=10.5)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_shares_timeseries.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Section III-IV: Main event-study figure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_event(ax, es, title, pooled=None, ylabel=None,
                 color_pre='#7f8c8d', color_post='#c0392b'):
    pre = es[es['year'] < 2017]
    t17 = es[es['year'] == 2017]
    post = es[es['year'] > 2017]
    ref  = es[es['year'] == OMIT_YEAR]
    ax.axhline(0, color='black', lw=0.9)
    ax.axvline(2017.5, color='black', ls='--', lw=1.2, alpha=0.65)
    ax.axvspan(2010.5, 2017, color=color_pre, alpha=0.04)
    ax.axvspan(2017.5, 2023.5, color=color_post, alpha=0.04)
    kw = dict(fmt='o', ms=5, lw=1.4, capsize=3, color='#34495e', ecolor='#7f8c8d')
    if not pre.empty:
        ax.errorbar(pre['year'], pre['beta'], yerr=1.96*pre['se'],
                     mfc=color_pre, mec='black', **kw)
    if not t17.empty:
        ax.errorbar(t17['year'], t17['beta'], yerr=1.96*t17['se'],
                     fmt='s', ms=6, lw=1.4, capsize=3,
                     color='darkorange', ecolor='#7f8c8d')
    if not post.empty:
        ax.errorbar(post['year'], post['beta'], yerr=1.96*post['se'],
                     mfc=color_post, mec='black', **kw)
    if not ref.empty:
        ax.scatter(ref['year'], ref['beta'], marker='D', s=60,
                    facecolor='white', edgecolor='black', zorder=6,
                    label='Reference (2016)')
    ax.text(2017.6, ax.get_ylim()[1]*0.97, 'TCJA',
             style='italic', fontsize=9, color='black', va='top')
    if pooled is not None and not np.isnan(pooled['coef']):
        t = abs(pooled['coef']/pooled['se']) if pooled['se']>0 else 0
        ax.text(0.02, 0.04,
                f"Pooled DiD 2019-23: $\\hat\\beta={pooled['coef']:+.5f}$ "
                f"(SE {pooled['se']:.5f}){stars(t)}  "
                f"$N_{{firm}}={pooled['G']}$",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_xticks(EVENT_YEARS)
    ax.set_xticklabels([str(y) for y in EVENT_YEARS], rotation=45, ha='right', fontsize=8.5)
    ax.set_ylabel(ylabel or r'$\hat\beta_t$ on $\bar c^{pre}_{Intl}\cdot\mathbf{1}[t=s]$')
    ax.set_title(title, fontsize=10.5)


def fig_foreignshare_event(panel, outcome='foreign_pi_share',
                            spec='baseline', fname='fig_foreignshare_event.pdf',
                            outcome_label='foreign PI share',
                            ylabel=None):
    print(f'  Fig 2: foreign-share event study (spec={spec})...')
    es  = run_event_study(panel, outcome, spec=spec)
    pd_ = run_pooled_did(panel, outcome, spec=spec)
    sub = panel.dropna(subset=[outcome])
    n_firm = sub['gvkey'].nunique(); n_obs = len(sub)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    spec_lbl = {'baseline':'firm + year FE',
                'ind_yr':  'firm + industry$\\times$year FE',
                'firm_trend':'firm + year + firm-linear-trend FE'}.get(spec, spec)
    _plot_event(ax, es,
                title=(f'Event study: {outcome_label} on pre-TCJA international exposure\n'
                       f'N = {n_firm} balanced firms, {n_obs:,} firm-years; {spec_lbl}; SE clustered at firm'),
                pooled=pd_,
                ylabel=(ylabel or
                        r'$\hat\beta_t$ (share units; $>0$ $\Rightarrow$ heavy shifter reduced foreign share)'))
    # Pre-trend joint F-test annotation
    ft = pre_trend_ftest(es)
    ax.text(0.02, 0.94,
            f'Pre-trend joint F (df={ft["df"]}): $F={ft["F"]:.2f}$,  $p={ft["p"]:.2f}$',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    return {'es': es, 'pooled': pd_}


# ─────────────────────────────────────────────────────────────────────────────
# Section V: Horizon-specific effects
# ─────────────────────────────────────────────────────────────────────────────

def fig_horizon_effects(panel, outcome='foreign_pi_share'):
    print('  Fig 3: horizon-specific effects (h=1,2,3,5)...')
    es = run_event_study(panel, outcome, spec='baseline')
    target_years = [2017 + h for h in HORIZONS]  # 2018, 19, 20, 22
    sel = es[es['year'].isin(target_years)].copy()
    sel['horizon'] = sel['year'] - 2017
    if sel.empty:
        print('    (no horizon coefs found)')
        return

    # Color gradient: light red → dark red
    cmap = plt.cm.Reds
    n = len(HORIZONS)
    colors = [cmap(0.35 + 0.55*i/(n-1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(sel))
    bars = ax.bar(x, sel['beta'].values, width=0.62,
                  color=colors, edgecolor='black', linewidth=0.6,
                  yerr=1.96*sel['se'].values, capsize=5,
                  error_kw={'linewidth':1.1, 'ecolor':'#34495e'})
    ax.axhline(0, color='black', lw=0.9)
    n_firm = panel.dropna(subset=[outcome])['gvkey'].nunique()
    for i, row in enumerate(sel.itertuples()):
        t = abs(row.beta/row.se) if row.se>0 else 0
        ax.text(i, row.beta + 1.96*row.se + 0.0005*np.sign(row.beta+1e-8),
                f'{row.beta:+.4f}{stars(t)}\n(SE {row.se:.4f})',
                ha='center', va='bottom' if row.beta>=0 else 'top', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}\n({yr})' for h, yr in zip(sel['horizon'], sel['year'])],
                       fontsize=9.5)
    ax.set_xlabel('Horizon (years after 2017)')
    ax.set_ylabel(r'$\hat\beta_h$: coefficient on $\bar c^{pre}_{Intl}\cdot\mathbf{1}[t=2017+h]$')
    ax.set_title(f'Dynamic adjustment of foreign PI share\n'
                 f'Response coefficient by years post-TCJA (N firms = {n_firm})',
                 fontsize=10.5)
    # Compare β_{h=5} / β_{h=1}
    b1 = sel[sel['horizon']==1]['beta'].values
    b5 = sel[sel['horizon']==5]['beta'].values
    if len(b1)>0 and len(b5)>0 and abs(b1[0])>1e-6:
        ax.text(0.98, 0.95,
                f'$\\hat\\beta_{{h=5}}/\\hat\\beta_{{h=1}} = {b5[0]/b1[0]:+.2f}$'
                f'\n($>1$: gradual; $\\approx 1$: onset-driven)',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_horizon_effects.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Section VI: Log-share robustness
# ─────────────────────────────────────────────────────────────────────────────

def fig_logshare_robustness(panel):
    print('  Fig 4: log-share robustness...')
    return fig_foreignshare_event(panel,
                                   outcome='foreign_share_log',
                                   spec='baseline',
                                   fname='fig_logshare_robustness.pdf',
                                   outcome_label='log(foreign PI share + 0.01)',
                                   ylabel=r'$\hat\beta_t$ (log-share units; $>0$ $\Rightarrow$ heavy shifter reduced foreign share)')


# ─────────────────────────────────────────────────────────────────────────────
# Section VII: Industry heterogeneity
# ─────────────────────────────────────────────────────────────────────────────

def fig_heterogeneity_byindustry(panel, outcome='foreign_pi_share'):
    print('  Fig 5: industry heterogeneity (pooled DiD)...')
    results = {}
    for ind_ in IND_ORDER:
        sub = panel[panel['industry'] == ind_]
        if sub.dropna(subset=[outcome])['gvkey'].nunique() < 30:
            continue
        results[ind_] = run_pooled_did(sub, outcome, spec='baseline')
    if not results:
        print('    (no industries with ≥30 firms — skipping)')
        return results

    inds   = list(results.keys())
    betas  = [results[i]['coef'] for i in inds]
    ses    = [results[i]['se']   for i in inds]
    Gs     = [results[i]['G']    for i in inds]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(inds))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(inds)))
    ax.bar(x, betas, width=0.62, color=colors, alpha=0.85,
           yerr=[1.96*s for s in ses], capsize=5,
           edgecolor='black', linewidth=0.5,
           error_kw={'linewidth':1.2, 'ecolor':'#34495e'})
    ax.axhline(0, color='black', lw=0.9)
    for i, (b, s) in enumerate(zip(betas, ses)):
        t = abs(b/s) if s>0 else 0
        ax.text(i, b + 1.96*s + 0.0005*np.sign(b+1e-8),
                f'{b:+.4f}{stars(t)}\nN={Gs[i]}',
                ha='center', va='bottom' if b>=0 else 'top', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(inds, fontsize=10)
    ax.set_xlabel('Industry')
    ax.set_ylabel(r'$\hat\beta$ pooled DiD (foreign share)')
    ax.set_title('Foreign-share response to TCJA by industry\n'
                 'Pooled DiD coefficient (2019-2023 vs 2013-2016)',
                 fontsize=10.5)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_heterogeneity_byindustry.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section VIII: LaTeX table — 5 specs
# ─────────────────────────────────────────────────────────────────────────────

def write_main_table(es1, es2, es3, pd1, pd5):
    """Main table — columns = (1)-(5) specs; rows = intl_pre × year coefs + pooled DiD."""
    def fc(b, s):
        if np.isnan(b) or np.isnan(s): return '---'
        t = abs(b/s) if s>0 else 0
        return (f'${b:+.5f}^{{{stars(t)}}}$'
                if stars(t) else f'${b:+.5f}$')
    def fse(s):
        return f'$({s:.5f})$' if not np.isnan(s) else ''

    yrs_in_tbl = [2018, 2019, 2020, 2021, 2022, 2023]

    def row_for(es, yr):
        if es is None or es.empty: return '---', ''
        r = es[es['year'] == yr]
        if r.empty: return '---', ''
        return fc(r.iloc[0]['beta'], r.iloc[0]['se']), fse(r.iloc[0]['se'])

    lines = [
        '\\begin{table}[htbp]', '\\centering',
        '\\caption{Dynamic profit-shifting response: foreign PI share on pre-TCJA international exposure}',
        '\\label{tab:dynamics_main}', '\\small',
        '\\begin{tabular}{lccccc}', '\\toprule',
        ' & (1) & (2) & (3) & (4) & (5) \\\\',
        ' & Firm+yr FE & Firm+Ind$\\times$yr FE & Firm+yr+firm trend & Pooled DiD & Log-share pooled \\\\',
        '\\midrule',
    ]
    for yr in yrs_in_tbl:
        c1, s1 = row_for(es1, yr)
        c2, s2 = row_for(es2, yr)
        c3, s3 = row_for(es3, yr)
        lines.append(f'$\\bar c^{{pre}}_{{Intl}} \\cdot \\mathbf{{1}}[t={yr}]$ '
                     f'& {c1} & {c2} & {c3} & --- & --- \\\\')
        lines.append(f' & {s1} & {s2} & {s3} & & \\\\')
    # Pooled rows
    lines.append(f'$\\bar c^{{pre}}_{{Intl}} \\cdot \\mathrm{{post}}$ & --- & --- & --- & '
                 f'{fc(pd1["coef"], pd1["se"])} & {fc(pd5["coef"], pd5["se"])} \\\\')
    lines.append(f' & & & & {fse(pd1["se"])} & {fse(pd5["se"])} \\\\')
    lines.append('\\midrule')
    lines.append(f'$N$ firm-years & {len(es1)-1 if es1 is not None else "---"} &'
                 f' & & {pd1["n"]:,} & {pd5["n"]:,} \\\\')
    lines.append(f'$N$ firms & & & & {pd1["G"]:,} & {pd5["G"]:,} \\\\')
    lines.append(f'Firm FE & Y & Y & Y & Y & Y \\\\')
    lines.append(f'Year FE & Y & --- & Y & Y & Y \\\\')
    lines.append(f'Industry $\\times$ year FE & --- & Y & --- & --- & --- \\\\')
    lines.append(f'Firm linear trend & --- & --- & Y & --- & --- \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines += [
        '\\begin{tablenotes}\\footnotesize',
        '\\item Outcome: columns (1)-(4) use foreign PI share = $\\pi^{fo}/(\\pi^{dom}+\\pi^{fo})\\in[0,1]$; '
        'column (5) uses $\\log(\\mathrm{share}+\\epsilon)$ with $\\epsilon=0.01$.',
        '\\item Sign: $\\hat\\beta > 0$ means firms with more negative pre-period International '
        'contribution reduced their foreign PI share post-TCJA (re-shoring).',
        '\\item Event window 2011--2023 (omit 2016); pooled DiD excludes 2017, 2018.  '
        'SE clustered at firm.  $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
        '\\end{tablenotes}', '\\end{table}',
    ]
    path = os.path.join(TBL_DIR, 'tbl_dynamics_main.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Validation report
# ─────────────────────────────────────────────────────────────────────────────

def print_validation(panel, es1, es2, es3, pd1, pd5, horizon_es, ind_results):
    print()
    print('=' * 70)
    print('  PROFIT SHIFTING DYNAMICS — VALIDATION REPORT')
    print('=' * 70)
    print()
    print('SAMPLE')
    n_fy = len(panel.dropna(subset=['foreign_pi_share']))
    n_fm = panel.dropna(subset=['foreign_pi_share'])['gvkey'].nunique()
    print(f'  Balanced panel, firm-years: {n_fy:,}')
    print(f'  Firms:                      {n_fm:,}')
    print(f'  Years:                      {min(EVENT_YEARS)}–{max(EVENT_YEARS)}')
    print()
    t = panel.drop_duplicates('gvkey')['intl_pre'].dropna()
    print('TREATMENT VARIABLE (intl_pre)')
    print(f'  Median:                     {t.median():+.2f}')
    print(f'  IQR:                        [{t.quantile(.25):+.2f}, {t.quantile(.75):+.2f}]')
    print(f'  Share with intl_pre < -5:   {(t < -5).mean()*100:.1f}%')
    print()
    fs = panel['foreign_pi_share'].dropna()
    print('OUTCOME QUALITY')
    print(f'  share == 0:                 {(fs==0).mean()*100:.1f}%')
    print(f'  share == 1:                 {(fs==1).mean()*100:.1f}%')
    wfsd = panel.groupby('gvkey')['foreign_pi_share'].std().dropna().mean()
    print(f'  Within-firm SD (avg):       {wfsd:.4f}')
    print()
    ft = pre_trend_ftest(es1, pre_years=(2011,2012,2013,2014,2015))
    print('PRE-TRENDS TEST (Spec 1)')
    print(f'  Joint F (df={ft["df"]}):             F={ft["F"]:.2f}  p={ft["p"]:.3f}')
    print()
    print('MAIN EVENT-STUDY COEFS (Spec 1, firm+year FE)')
    for yr in [2013,2014,2015,2018,2019,2020,2021,2022,2023]:
        r = es1[es1['year']==yr]
        if r.empty: continue
        b, s = r.iloc[0]['beta'], r.iloc[0]['se']
        t = abs(b/s) if s>0 else 0
        print(f'  β_{yr}: {b:+.5f}{stars(t):3s}  SE={s:.5f}')
    print()
    print('POOLED DID (Spec 4, baseline)')
    t = abs(pd1['coef']/pd1['se']) if pd1['se']>0 else 0
    print(f'  β (post × intl_pre):        {pd1["coef"]:+.5f}{stars(t):3s}  SE={pd1["se"]:.5f}')
    print(f'  Implied for intl_pre=-5:    {-5*pd1["coef"]*100:+.2f} pp foreign share')
    print()
    print('HORIZON-SPECIFIC EFFECTS')
    for h in HORIZONS:
        yr = 2017 + h
        r = horizon_es[horizon_es['year']==yr]
        if r.empty: continue
        b, s = r.iloc[0]['beta'], r.iloc[0]['se']
        print(f'  β_{{h={h}}} ({yr}):             {b:+.5f}  SE={s:.5f}')
    b1 = horizon_es[horizon_es['year']==2018]['beta'].values
    b5 = horizon_es[horizon_es['year']==2022]['beta'].values
    if len(b1)>0 and len(b5)>0 and abs(b1[0])>1e-6:
        print(f'  Ratio β_h=5 / β_h=1:        {b5[0]/b1[0]:+.2f}')
    print()
    print('ROBUSTNESS')
    print(f'  Log-share pooled:           {pd5["coef"]:+.5f}  SE={pd5["se"]:.5f}')
    if es2 is not None and not es2.empty:
        r = es2[es2['year']==2022]
        if not r.empty:
            print(f'  Ind×year FE β_2022:         {r.iloc[0]["beta"]:+.5f}')
    if es3 is not None and not es3.empty:
        r = es3[es3['year']==2022]
        if not r.empty:
            print(f'  Firm-trend β_2022:          {r.iloc[0]["beta"]:+.5f}')
    print()
    print('INDUSTRY HETEROGENEITY (pooled DiD)')
    for ind_, r in ind_results.items():
        t = abs(r['coef']/r['se']) if r['se']>0 else 0
        print(f'  {ind_:<16}  β={r["coef"]:+.5f}{stars(t):3s}  SE={r["se"]:.5f}  N={r["G"]}')
    print()
    print('INTERPRETATION')
    med_t = t if not isinstance(t, float) else t  # dummy — recompute
    med_t = panel.drop_duplicates('gvkey')['intl_pre'].dropna().median()
    implied = med_t * pd1['coef'] * 100
    print(f'  Median intl_pre = {med_t:+.2f} pp')
    print(f'  Implied foreign-share change for median firm: {implied:+.2f} pp')
    # Compare ETR first-stage check
    print()
    print('  (Cross-check): build_substitution_real_outcomes.py gave β_share = +0.00551 baseline / +0.00663 ind×year')
    print(f'                  this script baseline β = {pd1["coef"]:+.5f}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data...')
    combined, long_df, pi = load_data()

    print('\nBuilding treatment variable...')
    intl_pre = build_intl_pre(long_df)

    print('\nBuilding panel...')
    panel = build_panel(combined, pi, intl_pre)

    print('\nRunning Section II: foreign-share time-series by exposure tercile...')
    fig_shares_timeseries(panel)

    print('\nRunning Section III: main event studies...')
    print('  Spec 1: firm + year FE')
    r1 = fig_foreignshare_event(panel, outcome='foreign_pi_share', spec='baseline',
                                 fname='fig_foreignshare_event.pdf')
    print('  Spec 2: + industry × year FE')
    r2_es = run_event_study(panel, 'foreign_pi_share', spec='ind_yr')
    r2_pd = run_pooled_did(panel, 'foreign_pi_share', spec='ind_yr')
    print('  Spec 3: + firm linear trend')
    r3_es = run_event_study(panel, 'foreign_pi_share', spec='firm_trend')
    r3_pd = run_pooled_did(panel, 'foreign_pi_share', spec='firm_trend')

    print('\nRunning Section V: horizon-specific effects...')
    fig_horizon_effects(panel)

    print('\nRunning Section VI: log-share robustness...')
    r5 = fig_logshare_robustness(panel)

    print('\nRunning Section VII: industry heterogeneity...')
    ind_results = fig_heterogeneity_byindustry(panel)

    print('\nWriting main table...')
    write_main_table(r1['es'], r2_es, r3_es, r1['pooled'], r5['pooled'])

    # Validation
    print_validation(panel,
                      r1['es'], r2_es, r3_es,
                      r1['pooled'], r5['pooled'],
                      r1['es'], ind_results)
    print('\nDone.')
