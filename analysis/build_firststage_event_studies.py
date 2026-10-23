#!/usr/bin/env python3
"""
build_firststage_event_studies.py

Event-study figures for the TCJA first-stage and headline slides:
  1. International contribution event study (first stage dynamics)
  2. ETR event study (headline reduced-form dynamics)

Uses the exact same panel as build_substitution_analysis.py so the
pooled DiDs reproduce the scatter-based slopes.

Each figure overlays the event-study path with the pooled DiD estimate.
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
mpl.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'axes.grid': True, 'grid.alpha': 0.22,
                     'axes.spines.top': False, 'axes.spines.right': False})

script_dir = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(script_dir, '..', 'data', 'analysis', 'figures', 'substitution')
os.makedirs(FIG_DIR, exist_ok=True)

PRE_YEARS    = list(range(2013, 2017))
POST_YEARS   = list(range(2019, 2023))   # matches substitution analysis
EVENT_YEARS  = list(range(2013, 2023))
OMIT_YEAR    = 2016
WIN_CH       = 15.0


# ── Helpers ──────────────────────────────────────────────────────────────────



def stars(t):
    a = abs(t)
    return '***' if a>2.576 else ('**' if a>1.96 else ('*' if a>1.645 else ''))


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
    out = df[cols].copy().astype(float)
    df = df.copy()
    df['_ind_yr'] = df[ind_col].astype(str) + '_' + df[year_col].astype(str)
    for _ in range(max_iter):
        prev = out.values.copy()
        for c in cols:
            out[c] -= out[c].groupby(df[firm_col]).transform('mean')
            out[c] -= out[c].groupby(df['_ind_yr']).transform('mean')
        if np.abs(out.values - prev).max() < tol:
            break
    return out


def run_event_study(panel, outcome, treat_col='intl_pre', ind_yr_fe=False):
    """Event study of y on intl_pre × 1[t=s]."""
    plot_yrs = [yr for yr in sorted(EVENT_YEARS) if yr != OMIT_YEAR]
    mask = (panel['recon_year'].isin(EVENT_YEARS) &
            panel[outcome].notna() & panel[treat_col].notna())
    sub = panel[mask].copy()
    if len(sub) < 50:
        return pd.DataFrame()
    for yr in plot_yrs:
        sub[f'_ix_{yr}'] = sub[treat_col] * (sub['recon_year'] == yr).astype(float)
    ix_cols  = [f'_ix_{yr}' for yr in plot_yrs]
    dm = demean_ind_yr(sub, [outcome] + ix_cols) if ind_yr_fe else demean_twfe(sub, [outcome] + ix_cols)
    mod = OLS(dm[outcome].values, dm[ix_cols].values).fit()
    cv  = cov_cluster(mod, sub['gvkey'].values)
    se  = np.sqrt(np.diag(cv))
    rows = [{'year': yr, 'beta': float(mod.params[i]), 'se': float(se[i])}
            for i, yr in enumerate(plot_yrs)]
    rows.append({'year': OMIT_YEAR, 'beta': 0., 'se': 0.})
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


def run_pooled_did(panel, outcome, treat_col='intl_pre', ind_yr_fe=False):
    excl = {2017, 2018}   # 2016 is a pre-year; OMIT_YEAR only normalizes eq. (2)
    years = [yr for yr in EVENT_YEARS if yr not in excl]
    mask = (panel['recon_year'].isin(years) &
            panel[outcome].notna() & panel[treat_col].notna())
    sub = panel[mask].copy()
    sub['_post'] = sub['recon_year'].isin(POST_YEARS).astype(float)
    sub['_Di']   = sub[treat_col] * sub['_post']
    dm = demean_ind_yr(sub, [outcome, '_Di']) if ind_yr_fe else demean_twfe(sub, [outcome, '_Di'])
    mod = OLS(dm[outcome].values, dm[['_Di']].values).fit()
    cv  = cov_cluster(mod, sub['gvkey'].values)
    se  = float(np.sqrt(np.diag(cv))[0])
    return {'coef': float(mod.params[0]), 'se': se,
            'n': len(sub), 'G': sub['gvkey'].nunique()}


# ── Build panel (matches build_substitution_analysis.py exactly) ─────────────

print('Loading data...')
with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'combined.pkl'), 'rb') as f:
    combined = pickle.load(f)
with open(os.path.join(script_dir, '..', 'data', 'analysis', 'cache', 'long.pkl'), 'rb') as f:
    long_df = pickle.load(f)

print('Building channels + intl_pre...')
GROUP_MAP = {
    'International':  'International',
    'State & local':  'State_local',
    'Federal credits':'Federal_credits',
    'Permanent items':'Permanent_items',
    'Reserves & losses':'Reserves_losses',
    'TCJA one-time':  'TCJA_onetime',
    'Other':          'Other',
}
df = long_df.copy()
df['channel'] = df['group'].map(GROUP_MAP).fillna('Other')
fy = (df.groupby(['gvkey', 'recon_year', 'channel'])['cat_val']
      .sum().reset_index(name='val'))
fy['val_w'] = fy['val'].clip(-WIN_CH, WIN_CH)
wide = fy.pivot_table(index=['gvkey', 'recon_year'], columns='channel',
                      values='val_w', fill_value=0).reset_index()
wide.columns.name = None

# intl_pre: pre-period mean of International, [p1,p99] clip
pre = wide[wide['recon_year'].isin(PRE_YEARS)]
counts = pre.groupby('gvkey').size()
elig = counts[counts >= 2].index
intl_pre = (pre[pre['gvkey'].isin(elig)]
            .groupby('gvkey')['International'].mean()
            .rename('intl_pre'))
p1, p99 = intl_pre.quantile([0.01, 0.99])
intl_pre = intl_pre.clip(p1, p99)

# Panel
panel = combined[combined['recon_year'].isin(EVENT_YEARS)].copy()
panel['industry'] = panel['comp_sic'].apply(industry)
panel = panel[panel['industry'] != 'Finance']
if 'comp_pi' in panel.columns:
    panel = panel[panel['comp_pi'] > 1]
panel = panel.merge(intl_pre.reset_index(), on='gvkey', how='inner')
panel = panel.merge(wide, on=['gvkey', 'recon_year'], how='left')
for ch in ['International']:
    panel[ch] = panel[ch].fillna(0.0)
# Balanced: ≥2 pre AND ≥2 post
fy_sets = panel.groupby('gvkey')['recon_year'].apply(set)
has_pre  = fy_sets.apply(lambda s: len(s & set(PRE_YEARS))  >= 2)
has_post = fy_sets.apply(lambda s: len(s & set(POST_YEARS)) >= 2)
ok = fy_sets[has_pre & has_post].index
panel = panel[panel['gvkey'].isin(ok)].copy()

print(f'  Panel: {panel["gvkey"].nunique():,} firms, {len(panel):,} firm-years')


# ── Event studies ────────────────────────────────────────────────────────────

print('\nRunning event studies...')
es_intl = run_event_study(panel, 'International', treat_col='intl_pre', ind_yr_fe=True)
pd_intl = run_pooled_did(panel, 'International', treat_col='intl_pre', ind_yr_fe=True)
es_etr  = run_event_study(panel, 'etr_w',         treat_col='intl_pre', ind_yr_fe=True)
pd_etr  = run_pooled_did(panel, 'etr_w',          treat_col='intl_pre', ind_yr_fe=True)

print(f'  International pooled DiD: β = {pd_intl["coef"]:+.4f} '
      f'(SE {pd_intl["se"]:.4f}), N_firm = {pd_intl["G"]}')
print(f'  ETR pooled DiD:           β = {pd_etr["coef"]:+.4f} '
      f'(SE {pd_etr["se"]:.4f}), N_firm = {pd_etr["G"]}')


# ── Figure — single panel per outcome ────────────────────────────────────────

def _plot_event(ax, es, pooled, ylabel, title, color_post='#c0392b'):
    pre  = es[es['year'] <  2017]
    t17  = es[es['year'] == 2017]
    post = es[es['year'] >  2017]
    ref  = es[es['year'] == OMIT_YEAR]

    ax.axhline(0, color='black', lw=0.9)
    ax.axvline(2017.5, color='black', ls='--', lw=1.2, alpha=0.65)
    ax.axvspan(2012.5, 2017, color='#7f8c8d', alpha=0.06)
    ax.axvspan(2017.5, 2022.5, color=color_post, alpha=0.06)

    kw = dict(fmt='o', ms=6, lw=1.4, capsize=3, color='#34495e', ecolor='#7f8c8d')
    if not pre.empty:
        ax.errorbar(pre['year'], pre['beta'], yerr=1.96*pre['se'],
                     mfc='#7f8c8d', mec='black', **kw)
    if not t17.empty:
        ax.errorbar(t17['year'], t17['beta'], yerr=1.96*t17['se'],
                     fmt='s', ms=7, lw=1.4, capsize=3,
                     color='darkorange', ecolor='#7f8c8d')
    if not post.empty:
        ax.errorbar(post['year'], post['beta'], yerr=1.96*post['se'],
                     mfc=color_post, mec='black', **kw)
    if not ref.empty:
        ax.scatter(ref['year'], ref['beta'], marker='D', s=65,
                    facecolor='white', edgecolor='black', zorder=6,
                    label='Reference (2016)')

    ax.text(2017.6, ax.get_ylim()[1]*0.92, 'TCJA', style='italic',
             fontsize=10, color='black', va='top')
    t = abs(pooled['coef']/pooled['se']) if pooled['se']>0 else 0
    ax.text(0.02, 0.04,
            f"Pooled DiD (post 2019-22, excl.\\ 2017-18):\n"
            f"$\\hat\\beta = {pooled['coef']:+.3f}$ (SE {pooled['se']:.3f}){stars(t)}  "
            f"$N_{{firm}} = {pooled['G']:,}$",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_xticks(list(range(2013, 2023)))
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)


# Figure 1: International contribution event study (FIRST STAGE)
fig, ax = plt.subplots(figsize=(10.5, 5.5))
_plot_event(ax, es_intl, pd_intl,
            ylabel=r'$\hat{\beta}_t$ on $\bar{c}^{\text{pre}}_{\text{Intl}} \cdot \mathbf{1}[t=s]$',
            title=f'First stage: International contribution on pre-TCJA exposure '
                  f'(N = {pd_intl["G"]:,} firms; firm + industry$\\times$year FE)')
fig.tight_layout()
p1 = os.path.join(FIG_DIR, 'fig_firststage_intl_event.pdf')
fig.savefig(p1, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {p1}')

# Figure 2: ETR event study (REDUCED FORM / HEADLINE)
fig, ax = plt.subplots(figsize=(10.5, 5.5))
_plot_event(ax, es_etr, pd_etr,
            ylabel=r'$\hat{\beta}_t$ on $\bar{c}^{\text{pre}}_{\text{Intl}} \cdot \mathbf{1}[t=s]$  (ETR)',
            title=f'Headline: ETR response to pre-TCJA international exposure '
                  f'(N = {pd_etr["G"]:,} firms; firm + industry$\\times$year FE)',
            color_post='#2980b9')
fig.tight_layout()
p2 = os.path.join(FIG_DIR, 'fig_etr_event.pdf')
fig.savefig(p2, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {p2}')

# Save summary table for reference
print()
print('Event-study coefficients:')
print(f'{"year":>6} {"β(Intl)":>12} {"SE(Intl)":>10} {"β(ETR)":>12} {"SE(ETR)":>10}')
for yr in sorted(set(es_intl['year']).union(es_etr['year'])):
    ri = es_intl[es_intl['year']==yr]
    re = es_etr[es_etr['year']==yr]
    bi = ri.iloc[0]['beta'] if not ri.empty else np.nan
    si = ri.iloc[0]['se']   if not ri.empty else np.nan
    be = re.iloc[0]['beta'] if not re.empty else np.nan
    se_ = re.iloc[0]['se']  if not re.empty else np.nan
    print(f'{yr:>6} {bi:>+12.4f} {si:>10.4f} {be:>+12.4f} {se_:>10.4f}')

print('\nDone.')
