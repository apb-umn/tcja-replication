#!/usr/bin/env python3
"""
build_real_activity_dynamics.py

Parallel of build_profit_shifting_dynamics.py — same treatment
(pre-TCJA International contribution) and event window (2011–2023),
but with a maximum-information real-activity panel:

  1. capex intensity   = capx / lag(at)
  2. R&D intensity     = xrd  / lag(at)
  3. Total investment  = (capx + xrd) / lag(at)
  4. Employment growth = log(emp_t / emp_{t-1})
  5. Sales growth      = log(sale_t / sale_{t-1})

Question: when TCJA closed the international profit-shifting channel,
did firms re-allocate REAL activity (investment, hiring) toward the US,
or only re-allocate where income was BOOKED (the income-side response
already documented in build_profit_shifting_dynamics.py)?

Sign convention (no flip):
  intl_pre is in pp (NEGATIVE for shifters); outcomes are in level
  intensities or log-growth rates.
  β < 0 means heavy shifters (intl_pre < 0) had POSITIVE outcome change
    (e.g. invested more) post-TCJA — the "re-shoring" prediction.
  β > 0 means heavy shifters reduced the outcome.
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
FIG_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'figures', 'real_activity_dynamics')
TBL_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'tables',  'real_activity_dynamics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

PRE_YEARS   = list(range(2013, 2017))
POST_YEARS  = list(range(2019, 2023))   # 2019–2022
EVENT_YEARS = list(range(2011, 2024))   # 2011–2023
OMIT_YEAR   = 2016

# The pooled DiD runs on the same window as the tax-side DiD (Table 7): pre
# 2013-2016, post 2019-2022, dropping only the 2017-2018 transition years.
# OMIT_YEAR normalizes the event study (eq. 2) but its observations stay in the
# pooled sample -- eq. (1) has no base year to omit.  EVENT_YEARS stays wider so
# the event study keeps its longer pre-trend and lag(at) exists for the first
# regression year.
POOLED_YEARS = [yr for yr in range(2013, 2023) if yr not in {2017, 2018}]
WIN_CH      = 15.0
HORIZONS    = [1, 2, 3, 5]

IND_ORDER   = ['Tech', 'Pharma/Biotech', 'Manufacturing', 'Retail', 'Oil & Gas', 'Other']

# Per-outcome winsor bounds (level intensities and log growth rates)
OUTCOMES = {
    'capex_int':    {'label': 'Capex intensity', 'unit': 'capx/lag(at)',
                     'lo': 0.0,  'hi': 0.5},
    'rd_int':       {'label': 'R\\&D intensity', 'unit': 'xrd/lag(at)',
                     'lo': 0.0,  'hi': 0.5},
    'total_inv':    {'label': 'Total investment', 'unit': '(capx+xrd)/lag(at)',
                     'lo': 0.0,  'hi': 1.0},
    'emp_growth':   {'label': 'Employment growth', 'unit': '$\\Delta\\log$ emp',
                     'lo': -1.0, 'hi': 1.0},
    'sale_growth':  {'label': 'Sales growth', 'unit': '$\\Delta\\log$ sale',
                     'lo': -1.0, 'hi': 1.0},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (copied from build_profit_shifting_dynamics.py)
# ─────────────────────────────────────────────────────────────────────────────



def stars(t):
    a = abs(t)
    return '***' if a>2.576 else ('**' if a>1.96 else ('*' if a>1.645 else ''))


def piw_mean(vals, weights):
    v = np.asarray(vals, dtype=float); w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


def demean_twfe(df, cols, firm_col='gvkey', year_col='recon_year',
                tol=1e-9, max_iter=500):
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
    """Within-firm residual after (1, year) regression."""
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
    """Firm FE + year FE + firm-specific linear time trend."""
    out = df[cols].copy().astype(float)
    for c in cols:
        out[c] = residualize_firm_trend(
            pd.concat([df[[firm_col, year_col]], out[[c]]], axis=1),
            c, firm_col=firm_col, year_col=year_col)
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


def _fit_cluster(y, X, groups):
    mod    = OLS(y, X).fit()
    cl_cov = cov_cluster(mod, groups)
    cl_se  = np.sqrt(np.diag(cl_cov))
    return mod, cl_se


def run_event_study(panel, outcome, treat_col='intl_pre', spec='baseline'):
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
    mod, se = _fit_cluster(dm[outcome].values, dm[ix_cols].values,
                            sub['gvkey'].values)
    rows = [{'year': yr, 'beta': float(mod.params[i]), 'se': float(se[i]),
             'lower': float(mod.params[i] - 1.96*se[i]),
             'upper': float(mod.params[i] + 1.96*se[i])}
            for i, yr in enumerate(plot_yrs)]
    rows.append({'year': OMIT_YEAR, 'beta': 0., 'se': 0., 'lower': 0., 'upper': 0.})
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


def run_pooled_did(panel, outcome, treat_col='intl_pre', spec='baseline'):
    years = POOLED_YEARS
    mask = (panel['recon_year'].isin(years) &
            panel[outcome].notna() & panel[treat_col].notna())
    sub = panel[mask].copy()
    if len(sub) < 50 or sub['gvkey'].nunique() < 20:
        return {'coef': np.nan, 'se': np.nan, 'n': 0, 'G': 0}
    sub['_post'] = sub['recon_year'].isin(POST_YEARS).astype(float)
    sub['_Di']   = sub[treat_col] * sub['_post']
    cols = [outcome, '_Di']
    if   spec == 'ind_yr':     dm = demean_ind_yr(sub, cols)
    elif spec == 'firm_trend': dm = demean_firm_year_trend(sub, cols)
    else:                       dm = demean_twfe(sub, cols)
    mod, se = _fit_cluster(dm[outcome].values, dm[['_Di']].values,
                            sub['gvkey'].values)
    return {'coef': float(mod.params[0]), 'se': float(se[0]),
            'n': len(sub), 'G': sub['gvkey'].nunique()}


def pre_trend_ftest(es, pre_years=(2011,2012,2013,2014,2015)):
    sub = es[es['year'].isin(pre_years) & (es['year'] != OMIT_YEAR) & (es['se'] > 0)]
    if len(sub) == 0: return {'F': np.nan, 'df': 0, 'p': np.nan}
    z = (sub['beta'] / sub['se']).values
    chi2 = (z**2).sum()
    df = len(sub)
    F = chi2 / df
    p = 1 - f_dist.cdf(F, df, 1000)
    return {'F': float(F), 'df': int(df), 'p': float(p)}


# ─────────────────────────────────────────────────────────────────────────────
# Section I: Load + build panel
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'combined.pkl'), 'rb') as f:
        combined = pickle.load(f)
    with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'long.pkl'), 'rb') as f:
        long_df = pickle.load(f)
    csv = os.path.join(script_dir, '..', 'data', 'analysis', 'output', 'tax_reconciliation_panel_base.csv')
    extra = pd.read_csv(csv,
                        usecols=['gvkey','recon_year','comp_at','comp_capx',
                                 'comp_xrd','comp_emp'],
                        low_memory=False).drop_duplicates(['gvkey','recon_year'])
    print(f'  combined: {len(combined):,} rows, {combined["gvkey"].nunique():,} firms')
    print(f'  long_df:  {len(long_df):,} rows')
    print(f'  extra:    {len(extra):,} rows (capx/xrd/emp/at)')
    return combined, long_df, extra


def build_intl_pre(long_df, win_ch=WIN_CH):
    """Pre-period International contribution per firm, winsorized at +/- win_ch.

    `win_ch=None` disables per-firm-year clipping (no-per-obs-clip variant).
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


def build_panel(combined, extra, intl_pre):
    """Build §4c real-activity panel.

    Goal: maximum data for each real-activity outcome, given that we only
    need (treatment T_i, outcome y_it) — no PI composition required.

    Why the sample differs from §4b (build_profit_shifting_dynamics.py):
      §4b's outcome IS foreign_pi_share, so it requires firms to report
      pidom/pifo and pass the PI-reconciliation check.  §4c's outcomes
      (capex, R&D, employment, sales growth) don't need PI composition;
      restricting to the §4b sample throws out many firms with valid
      treatment + outcome data.  We use the maximum-information sample
      here and document the difference.

    Sample restrictions (this script):
      1. Finance excluded
      2. Merge intl_pre treatment (≥2 pre-period reconciliation obs)
      3. ≥2 pre AND ≥2 post observations in the event window
         (i.e., firm appears in the panel; per-outcome missingness is
         handled by the regression masks)

    Methodology (kept from prior fix):
      - Year-aware lags via merge (NOT groupby.shift)
      - total_inv requires BOTH capex AND xrd observed (NO zero-fill)
    """
    panel = combined[combined['recon_year'].isin(EVENT_YEARS)].copy()
    panel['industry'] = panel['comp_sic'].apply(industry)
    panel = panel[panel['industry'] != 'Finance']
    panel = panel.merge(intl_pre.reset_index(), on='gvkey', how='inner')
    panel = panel.merge(extra, on=['gvkey','recon_year'], how='left')

    # ── Sample-defining balance: ≥2 pre AND ≥2 post observations ────────────
    fy = panel.groupby('gvkey')['recon_year'].apply(set)
    has_pre  = fy.apply(lambda s: len(s & set(PRE_YEARS))  >= 2)
    has_post = fy.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
    ok = fy[has_pre & has_post].index
    panel = panel[panel['gvkey'].isin(ok)].copy()

    # ── Year-aware lags via merge (handles gaps correctly) ──────────────────
    lag_src = (panel[['gvkey','recon_year','comp_at','comp_emp','comp_sale']]
               .copy())
    lag_src['recon_year'] = lag_src['recon_year'] + 1   # year t-1 obs maps to year t
    lag_src = lag_src.rename(columns={'comp_at':'lag_at',
                                       'comp_emp':'lag_emp',
                                       'comp_sale':'lag_sale'})
    panel = panel.merge(lag_src, on=['gvkey','recon_year'], how='left')

    # ── Outcomes (no zero-fill on missing capex/xrd) ────────────────────────
    valid_at = panel['lag_at'] > 1
    panel['capex_int'] = np.where(valid_at & panel['comp_capx'].notna(),
                                   panel['comp_capx'] / panel['lag_at'], np.nan)
    panel['rd_int']    = np.where(valid_at & panel['comp_xrd'].notna(),
                                   panel['comp_xrd']  / panel['lag_at'], np.nan)
    # total_inv: require BOTH capex AND xrd observed (no fillna)
    both_ok = valid_at & panel['comp_capx'].notna() & panel['comp_xrd'].notna()
    panel['total_inv'] = np.where(both_ok,
        (panel['comp_capx'] + panel['comp_xrd']) / panel['lag_at'], np.nan)
    panel['emp_growth']  = np.where((panel['comp_emp'] > 0) & (panel['lag_emp'] > 0),
        np.log(panel['comp_emp'] / panel['lag_emp']), np.nan)
    panel['sale_growth'] = np.where((panel['comp_sale'] > 0) & (panel['lag_sale'] > 0),
        np.log(panel['comp_sale'] / panel['lag_sale']), np.nan)

    # Per-outcome winsor
    for col, info in OUTCOMES.items():
        panel[col] = panel[col].clip(info['lo'], info['hi'])

    print(f'  Panel: {panel["gvkey"].nunique():,} firms, {len(panel):,} firm-years')
    print(f'  (§4c uses BROADER sample than §4b — no PI-share restriction.)')
    print(f'  Outcome coverage (per-outcome missingness handled by regression mask):')
    for col, info in OUTCOMES.items():
        sub = panel.dropna(subset=[col])
        print(f'    {col:<14} {sub["gvkey"].nunique():>5} firms  '
              f'{len(sub):>6} obs  '
              f'mean={sub[col].mean():+.4f}  '
              f'median={sub[col].median():+.4f}')
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Section II: Time-series by tercile
# ─────────────────────────────────────────────────────────────────────────────

def fig_timeseries(panel):
    print('  Fig 1: real-activity time series by intl_pre tercile...')
    firm_t = panel.drop_duplicates('gvkey').set_index('gvkey')['intl_pre'].dropna()
    try:
        terc = pd.qcut(firm_t, q=3, labels=['Heavy','Medium','Light'],
                       duplicates='drop')
    except ValueError:
        terc = pd.cut(firm_t, bins=3, labels=['Heavy','Medium','Light'])
    panel = panel.copy()
    panel['terc'] = panel['gvkey'].map(terc)
    colors = {'Heavy':'#c0392b','Medium':'#e67e22','Light':'#2980b9'}

    n_out = len(OUTCOMES)
    fig, axes = plt.subplots(n_out, 1, figsize=(11.5, 2.8*n_out), sharex=True)

    for ax, (col, info) in zip(axes, OUTCOMES.items()):
        for label, c in colors.items():
            sub = panel[panel['terc'] == label]
            ser = sub.dropna(subset=[col])
            yr  = (ser.groupby('recon_year')
                       .apply(lambda g: piw_mean(g[col].values, g['comp_pi'].values))
                       .rename('y').reset_index())
            n = sub['gvkey'].nunique()
            ax.plot(yr['recon_year'], yr['y'], color=c, lw=1.8, marker='o', ms=4,
                    label=f'{label} (N={n})' if col == list(OUTCOMES.keys())[0] else None)
        ax.axvline(2017.5, color='black', ls='--', lw=1.0, alpha=0.6)
        ax.set_ylabel(f"{info['label']}\n({info['unit']})", fontsize=8.5)
        ax.set_title(info['label'], fontsize=9, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.25)
    axes[0].legend(fontsize=8, loc='upper left', ncol=3)
    axes[-1].set_xlabel('Fiscal year')
    axes[-1].set_xticks(EVENT_YEARS)
    axes[-1].set_xticklabels([str(y) for y in EVENT_YEARS], rotation=45, ha='right',
                             fontsize=8)
    fig.suptitle('Real activity by pre-TCJA intl-exposure tercile (PI-weighted means)',
                 fontsize=10.5)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_real_activity_timeseries.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Section III: Multi-outcome event-study figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_event_studies(panel):
    print('  Fig 2: multi-outcome event studies...')
    n_out = len(OUTCOMES)
    fig, axes = plt.subplots(n_out, 1, figsize=(11.5, 2.5*n_out), sharex=True)
    es_results = {}
    pd_results = {}
    for ax, (col, info) in zip(axes, OUTCOMES.items()):
        es = run_event_study(panel, col, spec='ind_yr')
        pdres = run_pooled_did(panel, col, spec='ind_yr')
        es_results[col] = es; pd_results[col] = pdres
        if es.empty:
            ax.set_title(f'{info["label"]} — too few obs')
            continue
        pre  = es[es['year'] <  2017]
        t17  = es[es['year'] == 2017]
        post = es[es['year'] >  2017]
        ref  = es[es['year'] == OMIT_YEAR]
        ax.axhline(0, color='black', lw=0.9)
        ax.axvline(2017.5, color='black', ls='--', lw=1.1, alpha=0.65)
        kw = dict(fmt='o', ms=5, lw=1.4, capsize=3, color='#34495e', ecolor='#7f8c8d')
        ax.errorbar(pre['year'],  pre['beta'],  yerr=1.96*pre['se'],  mfc='#7f8c8d',
                    mec='black', **kw)
        if not t17.empty:
            ax.errorbar(t17['year'], t17['beta'], yerr=1.96*t17['se'],
                        fmt='s', ms=6, lw=1.4, capsize=3,
                        color='darkorange', ecolor='#7f8c8d')
        ax.errorbar(post['year'], post['beta'], yerr=1.96*post['se'], mfc='#c0392b',
                    mec='black', **kw)
        if not ref.empty:
            ax.scatter(ref['year'], ref['beta'], marker='D', s=55,
                       facecolor='white', edgecolor='black', zorder=6)
        t = abs(pdres['coef']/pdres['se']) if pdres['se']>0 else 0
        ax.text(0.99, 0.04,
                f"Pooled DiD: $\\hat\\beta={pdres['coef']:+.5f}$ "
                f"(SE {pdres['se']:.5f}){stars(t)}  N={pdres['G']}",
                transform=ax.transAxes, ha='right', fontsize=8.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_ylabel(rf'$\hat\beta_t$', fontsize=9)
        ax.set_title(info['label'], fontsize=9.5, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.25)
    axes[-1].set_xlabel('Fiscal year')
    axes[-1].set_xticks(EVENT_YEARS)
    axes[-1].set_xticklabels([str(y) for y in EVENT_YEARS], rotation=45, ha='right',
                             fontsize=8)
    fig.suptitle('Real-activity event studies on pre-TCJA international exposure\n'
                 r'Firm + industry$\times$year FE; $\hat\beta < 0$ means heavier shifters increased the outcome post-TCJA',
                 fontsize=10.5)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_real_activity_event.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')
    return es_results, pd_results


# ─────────────────────────────────────────────────────────────────────────────
# Section V: Horizon-specific effects (one panel per outcome)
# ─────────────────────────────────────────────────────────────────────────────

def fig_horizon_effects(es_results):
    print('  Fig 3: horizon-specific effects (h=1,2,3,5)...')
    n_out = len(OUTCOMES)
    target_yrs = [2017 + h for h in HORIZONS]
    cmap = plt.cm.Reds
    cols = [cmap(0.35 + 0.55*i/(len(HORIZONS)-1)) for i in range(len(HORIZONS))]

    fig, axes = plt.subplots(1, n_out, figsize=(3.4*n_out, 4.5), sharey=False)
    if n_out == 1: axes = [axes]
    for ax, (col, info) in zip(axes, OUTCOMES.items()):
        es = es_results.get(col, pd.DataFrame())
        if es.empty:
            ax.set_title(info['label'] + ' (n.a.)'); continue
        sel = es[es['year'].isin(target_yrs)].copy()
        sel['horizon'] = sel['year'] - 2017
        x = np.arange(len(sel))
        ax.bar(x, sel['beta'].values, width=0.62, color=cols,
               edgecolor='black', linewidth=0.5,
               yerr=1.96*sel['se'].values, capsize=4,
               error_kw={'linewidth':1.0, 'ecolor':'#34495e'})
        ax.axhline(0, color='black', lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'h={h}\n({yr})' for h,yr in zip(sel['horizon'], sel['year'])],
                           fontsize=8)
        ax.set_title(info['label'], fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        for i, row in enumerate(sel.itertuples()):
            t = abs(row.beta/row.se) if row.se>0 else 0
            ax.text(i, row.beta + 1.96*row.se + 0.0001*np.sign(row.beta+1e-12),
                    f'{stars(t)}',
                    ha='center', va='bottom' if row.beta>=0 else 'top', fontsize=8)
    fig.suptitle(r'Horizon-specific effects on real activity ($\hat\beta_h$ at $h\in\{1,2,3,5\}$)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_real_activity_horizon.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Section VII: Industry heterogeneity (one outcome per panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_industry_heterogeneity(panel):
    print('  Fig 4: industry heterogeneity by outcome...')
    n_out = len(OUTCOMES)
    fig, axes = plt.subplots(1, n_out, figsize=(3.4*n_out, 4.7), sharey=False)
    if n_out == 1: axes = [axes]

    for ax, (col, info) in zip(axes, OUTCOMES.items()):
        results = {}
        for ind_ in IND_ORDER:
            sub = panel[panel['industry'] == ind_]
            if sub.dropna(subset=[col])['gvkey'].nunique() < 30:
                continue
            results[ind_] = run_pooled_did(sub, col, spec='baseline')
        if not results:
            ax.set_title(info['label'] + ' (n.a.)'); continue
        inds = list(results.keys())
        bs   = [results[i]['coef'] for i in inds]
        ss   = [results[i]['se']   for i in inds]
        Gs   = [results[i]['G']    for i in inds]
        x = np.arange(len(inds))
        cols = plt.cm.tab10(np.linspace(0, 0.9, len(inds)))
        ax.bar(x, bs, width=0.62, color=cols, alpha=0.85,
               edgecolor='black', linewidth=0.5,
               yerr=[1.96*s for s in ss], capsize=4,
               error_kw={'linewidth':1.0, 'ecolor':'#34495e'})
        ax.axhline(0, color='black', lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(inds, fontsize=8.5, rotation=30, ha='right')
        ax.set_title(info['label'], fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        for i, (b, s, g) in enumerate(zip(bs, ss, Gs)):
            t = abs(b/s) if s>0 else 0
            ax.text(i, b + 1.96*s + 0.0001*np.sign(b+1e-12),
                    f'{stars(t)}\nN={g}',
                    ha='center', va='bottom' if b>=0 else 'top', fontsize=7.5)
    fig.suptitle('Industry heterogeneity: pooled DiD by outcome and industry',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_real_activity_byindustry.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Section VIII: LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def write_table(panel):
    print('  Writing LaTeX table...')
    rows = []
    for col, info in OUTCOMES.items():
        b = run_pooled_did(panel, col, spec='baseline')
        i = run_pooled_did(panel, col, spec='ind_yr')
        f = run_pooled_did(panel, col, spec='firm_trend')
        rows.append((info['label'], info['unit'], b, i, f))

    def fc(r):
        if not r or np.isnan(r.get('coef', np.nan)): return '---'
        v, s = r['coef'], r['se']
        t = abs(v/s) if s>0 else 0
        return f'${v:+.5f}^{{{stars(t)}}}$' if stars(t) else f'${v:+.5f}$'
    def fse(r):
        return f'$({r["se"]:.5f})$' if r and not np.isnan(r.get('se', np.nan)) else ''

    lines = [
        '\\begin{table}[htbp]', '\\centering',
        '\\caption{Real-activity dynamic response to pre-TCJA international exposure}',
        '\\label{tab:real_activity_dynamics}', '\\small',
        '\\begin{tabular}{lccccc}', '\\toprule',
        'Outcome & Definition & (1) Firm+Ind$\\times$yr FE & (2) Firm+yr FE & (3) Firm trend & $N$ firms \\\\',
        '\\midrule',
    ]
    for lbl, unit, b, i, f_ in rows:
        lines.append(f'{lbl} & {unit} & {fc(i)} & {fc(b)} & {fc(f_)} & {i["G"]:,} \\\\')
        lines.append(f' & & {fse(i)} & {fse(b)} & {fse(f_)} & \\\\')
    lines += [
        '\\bottomrule', '\\end{tabular}',
        '\\begin{tablenotes}\\footnotesize',
        '\\item Sample: maximum-information real-activity panel; profitable, Finance excluded; ',
        '$\\geq 2$ pre and $\\geq 2$ post observations in the event window. ',
        'PI-composition variables are not required because the outcomes are Compustat real-activity measures.',
        '\\item Treatment $\\bar c^{\\text{pre}}_{\\text{Intl}}$: pre-TCJA international '
        'reconciliation contribution (2013--16 mean, pp); $\\pm 15$\\,pp per-obs winsor, '
        '$[p_1,p_{99}]$ across firms.',
        '\\item \\texttt{total\\_inv} requires both \\texttt{capx} and \\texttt{xrd} observed; '
        'lags computed via year-aware merge (handles gaps).',
        '\\item Pooled DiD excludes 2017 and 2018 transition years; SE clustered at firm.',
        '\\item Sign: $\\hat\\beta < 0$ means heavy shifters ($\\bar c^{\\text{pre}}_{\\text{Intl}}<0$) '
        'showed POSITIVE outcome change post-TCJA.',
        '\\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
        '\\end{tablenotes}', '\\end{table}',
    ]
    path = os.path.join(TBL_DIR, 'tbl_real_activity_dynamics.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'    Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def print_validation(panel, es_results, pd_results):
    print()
    print('=' * 70)
    print('  REAL-ACTIVITY DYNAMICS — VALIDATION REPORT')
    print('=' * 70)
    print()
    n_fm = panel['gvkey'].nunique(); n_fy = len(panel)
    print(f'SAMPLE: {n_fm:,} firms, {n_fy:,} firm-years, 2011–2023')
    t = panel.drop_duplicates('gvkey')['intl_pre'].dropna()
    print(f'TREATMENT (intl_pre): median={t.median():+.2f}  '
          f'IQR=[{t.quantile(.25):+.2f},{t.quantile(.75):+.2f}]  '
          f'<-5: {(t<-5).mean()*100:.1f}%')
    print()
    print('POOLED DID BY OUTCOME (baseline firm+industry×year FE)')
    for col, info in OUTCOMES.items():
        r = pd_results.get(col, {})
        if not r or np.isnan(r.get('coef', np.nan)):
            print(f'  {info["label"]:<22} (n.a.)'); continue
        t_ = abs(r['coef']/r['se']) if r['se']>0 else 0
        print(f'  {info["label"]:<22} β={r["coef"]:+.5f}{stars(t_):<3}  '
              f'SE={r["se"]:.5f}  N={r["G"]:,}')
    print()
    print('PRE-TREND F-TESTS (Spec 1)')
    for col, info in OUTCOMES.items():
        es = es_results.get(col, pd.DataFrame())
        if es.empty: continue
        ft = pre_trend_ftest(es)
        print(f'  {info["label"]:<22} F(df={ft["df"]})={ft["F"]:.2f}  p={ft["p"]:.3f}')
    print()
    print('HORIZON-SPECIFIC EFFECTS (capex_int)')
    es_capex = es_results.get('capex_int', pd.DataFrame())
    if not es_capex.empty:
        for h in HORIZONS:
            yr = 2017 + h
            r = es_capex[es_capex['year']==yr]
            if r.empty: continue
            b, s = r.iloc[0]['beta'], r.iloc[0]['se']
            print(f'  capex β_{{h={h}}} ({yr}): {b:+.5f}  SE={s:.5f}')
    print()
    print('INTERPRETATION (per heavy shifter, intl_pre = -5 pp):')
    for col, info in OUTCOMES.items():
        r = pd_results.get(col, {})
        if not r or np.isnan(r.get('coef', np.nan)): continue
        impl = -5 * r['coef']
        print(f'  Δ {info["label"]:<22}: {impl:+.5f}  ({info["unit"]})')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data...')
    combined, long_df, extra = load_data()
    print('\nBuilding treatment...')
    intl_pre = build_intl_pre(long_df)
    print('\nBuilding panel...')
    panel = build_panel(combined, extra, intl_pre)
    print('\nFig 1: time series by tercile...')
    fig_timeseries(panel)
    print('\nFig 2: event studies...')
    es_res, pd_res = fig_event_studies(panel)
    print('\nFig 3: horizon effects...')
    fig_horizon_effects(es_res)
    print('\nFig 4: industry heterogeneity...')
    fig_industry_heterogeneity(panel)
    print('\nWriting table...')
    write_table(panel)
    print_validation(panel, es_res, pd_res)
    print('\nDone.')
