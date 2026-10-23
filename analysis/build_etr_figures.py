"""
build_etr_figures.py
Reproduce US corporate ETR descriptive analysis from 10-K tax reconciliation panel.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import re
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sic_industry import industry  # canonical SIC -> industry classifier


np.random.seed(42)

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

DPI = 300
CACHE_VERSION = 1

PALETTE = {
    'State & local':      '#2c3e50',
    'International':      '#e74c3c',
    'Federal credits':    '#1abc9c',
    'Permanent items':    '#f39c12',
    'Reserves & losses':  '#95a5a6',
    'TCJA one-time':      '#c2185b',
    'Other':              '#d5dbdb',
}

GROUP_ORDER = ['State & local', 'International', 'Federal credits',
               'Permanent items', 'Reserves & losses', 'TCJA one-time', 'Other']

group_map = {
    'State & local':                   'State & local',
    'Foreign rate differential':        'International',
    'GILTI':                            'International',
    'FDII':                             'International',
    'Subpart F':                        'International',
    'BEAT':                             'International',
    'Tax holiday':                      'International',
    'Foreign tax credit':               'International',
    'R&D credit':                       'Federal credits',
    'Domestic prod./§199':              'Federal credits',
    'Other credits':                    'Federal credits',
    'Permanent diffs / non-ded.':       'Permanent items',
    'Stock-based comp':                 'Permanent items',
    'Goodwill impairment':              'Permanent items',
    'Compensation (other)':             'Permanent items',
    'Tax-exempt income / DRD':          'Permanent items',
    'M&A / dispositions':               'Permanent items',
    'Depletion':                        'Permanent items',
    'Noncontrolling interest':          'Permanent items',
    'Valuation allowance':              'Reserves & losses',
    'Uncertain tax positions':          'Reserves & losses',
    'NOL / loss carryforward':          'Reserves & losses',
    'Return-to-provision / prior-yr':   'Reserves & losses',
    'TCJA/rate change':                 'TCJA one-time',
    'TCJA/repatriation':                'TCJA one-time',
    'OTHER':                            'Other',
}

IND_COLORS = {'Pharma/Biotech': '#9b59b6', 'Tech': '#3498db',
              'Manufacturing': '#e67e22', 'Retail': '#27ae60'}
IND_MARKERS = {'Pharma/Biotech': 'o', 'Tech': 's', 'Manufacturing': 'D', 'Retail': '^'}

COLS = [
    'filing_id', 'gvkey', 'recon_year', 'year_offset_from_filing',
    'within_expected_filing_window', 'has_compustat_year_match',
    'statutory_tax', 'israte', 'effective_tax', 'recon_check',
    'extracted_category_count', 'extraction_confidence', 'extraction_mode',
    'extraction_warnings', 'filing_entity_count',
    'filing_compustat_fyear', 'comp_pi', 'comp_ni', 'comp_txt', 'comp_sale',
    'comp_fic', 'comp_sic', 'filed_at',
    'overlap_repair_role', 'overlap_repair_tag', 'overlap_repair_applied',
] + [f'category_{i}_label' for i in range(1, 16)] + [f'category_{i}_val' for i in range(1, 16)]


# ── Helper functions ──────────────────────────────────────────────────────────

def piw_mean(vals, weights):
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vals, dtype=float)
    mask = (w > 0) & np.isfinite(v) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan
    return np.average(v[mask], weights=w[mask])


def repair_rank(value):
    s = str(value or '').strip().lower().replace('|', ';')
    parts = {part.strip() for part in s.split(';') if part.strip()}
    if 'replacement' in parts:
        return 0
    if 'replaced_source' in parts or 'blocked_misaligned_transition' in parts:
        return 2
    return 1


def classify(lbl):
    if pd.isna(lbl):
        return 'OTHER'
    s = str(lbl).lower()
    if re.search(r'tcja|tax cuts and jobs|rate chang|change in (enacted |statutory |federal |u\.?s\.? )?(tax )?rate|remeasurement|deferred tax re[- ]?measur', s):
        return 'TCJA/rate change'
    if re.search(r'transition tax|toll charge|deemed repat|repatriat', s):
        return 'TCJA/repatriation'
    if re.search(r'gilti|global intangible', s):
        return 'GILTI'
    # BEAT: the third international provision of TCJA, alongside GILTI and FDII.
    if re.search(r'\bbeat\b|base erosion', s):
        return 'BEAT'
    if re.search(r'fdii|foreign[- ]derived intangible', s):
        return 'FDII'
    # Subpart F: the pre-TCJA anti-deferral regime and direct ancestor of GILTI.
    # A current U.S. tax on foreign income, so it belongs with the international
    # items rather than the OTHER residual it fell through to before.
    if re.search(r'subpart\s*f\b', s):
        return 'Subpart F'
    # Jurisdiction-specific tax holidays are why a firm's foreign rate is low,
    # i.e. a foreign rate differential. Without this rule the labels split
    # arbitrarily: "Singapore tax holiday" fell to OTHER while any variant
    # naming "foreign" was caught by the broad rate rule below.
    if re.search(r'tax holiday|holiday', s):
        return 'Tax holiday'
    if re.search(r'losses in foreign jurisdictions for which no benefit|foreign jurisdictions for which no benefit', s):
        return 'NOL / loss carryforward'
    if re.search(r'taxes on non[- ]?u\.?s\.? income\s*-\s*european reorganization', s):
        return 'OTHER'
    if re.search(
        r'non[- ]?u\.?s\.?.*(operations|subsidiar|tax|taxes|earnings|income|rate|jurisd)|'
        r'u\.?s\.?\s+tax\s+on\s+non[- ]?u\.?s\.?|'
        r'jurisdictions taxed at rates? different|'
        r'tax on global activities including exports|'
        r'net impact of foreign results|'
        r'lower rates in other jurisdictions|'
        r'lower taxes on foreign results|'
        r'non[- ]united states tax rate difference|'
        r'(other )?non[- ]united states income taxes at other than|'
        r'foreign impact, net|'
        r'earnings of subsidiaries taxed at other than u\.?s\.? statutory rate|'
        r'jurisdictional rate differences|'
        r'income tax at rate other than the u\.?s\.? statutory tax rate|'
        r'statutory rate other than u\.?s\.? statutory rate|'
        r'^foreign$',
        s):
        return 'Foreign rate differential'
    if re.search(
        r'dividend income from foreign subsidiaries|'
        r'distributions from foreign subsidiaries|'
        r'tax on foreign subsidiary earnings|'
        r'distribution related foreign taxes|'
        r'foreign source income, net of tax credits|'
        r'reversal of prior years.? indefinite reinvestment assertion',
        s):
        return 'OTHER'
    if re.search(r'^(hong kong|bermuda|dominican republic)$', s.strip()):
        return 'Foreign rate differential'
    if re.search(r'\bforeign\b.*(rate|differential|jurisd)|rate differential|foreign[- ]us rate|foreign operations', s):
        return 'Foreign rate differential'
    if re.search(r'foreign tax credit', s):
        return 'Foreign tax credit'
    # Exclude currency/FX from the broad International catch — it's FX, not tax
    if re.search(r'foreign (exchange|currency)|currency translation|\bfx\b.*(gain|loss|translation|remeasure)', s):
        return 'OTHER'
    # Broad International catch — matches reinvestment language, "X of foreign" word order,
    # FSC, "US tax on foreign", international/overseas, and most foreign-X tax-noun combos.
    if re.search(
        r'foreign (income )?tax|'
        r'foreign (earnings|subsidiar|operations|dividend|withhold|provision|source|losses?|income|rate)|'
        r'(earnings|income|dividends|losses|subsidiar)\b[^.]{0,40}\bforeign\b|'
        r'(permanent|indefinite)(ly)?\s+(re)?invest|'
        r'u\.?\s?s\.?\s+tax\s+on\s+foreign|tax\s+on\s+foreign|'
        r'foreign\s+sales\s+(corporation|benefit|exempt|exclusion)|\bfsc\b|'
        r'outside\s+the\s+u\.?\s?s|outside\s+the\s+united\s+states|'
        r'\binternational\b|\boverseas\b|'
        r'effect of foreign',
        s):
        return 'Foreign rate differential'
    if re.search(r'\bstate\b|local income|state and local', s):
        return 'State & local'
    if re.search(r'research.*(credit|r&d)|r\s*&\s*d credit|r&d tax', s):
        return 'R&D credit'
    if re.search(r'credit', s):
        return 'Other credits'
    if re.search(r'valuation allow|valuation allowance|increase in valuation|decrease in valuation|change in (dta )?valuation', s):
        return 'Valuation allowance'
    if re.search(r'stock[- ]?based compensation|share[- ]?based|stock compensation|equity compensation|incentive stock|iso', s):
        return 'Stock-based comp'
    if re.search(r'goodwill', s):
        return 'Goodwill impairment'
    if re.search(r'uncertain tax|unrecognized tax|utb|utp|fin ?48|reserv.*tax|tax reserv', s):
        return 'Uncertain tax positions'
    if re.search(r'domestic (production|manufactur)|section 199\b|manufacturing deduction', s):
        return 'Domestic prod./§199'
    if re.search(r'meals|entertainment|\bm\s*&\s*e\b|non[- ]deductible|nondeductible|permanent (difference|item)|permanent tax|permanent diff', s):
        return 'Permanent diffs / non-ded.'
    if re.search(r'nol|net operating loss|loss carry', s):
        return 'NOL / loss carryforward'
    if re.search(r'tax[- ]exempt|municipal|dividends received|dividend[- ]received', s):
        return 'Tax-exempt income / DRD'
    if re.search(r'return to provision|true[- ]?up|prior year|prior period', s):
        return 'Return-to-provision / prior-yr'
    if re.search(r'acquisition|merger|divestit|disposal|dispositio', s):
        return 'M&A / dispositions'
    if re.search(r'depletion|percentage depletion', s):
        return 'Depletion'
    if re.search(r'noncontrolling|non[- ]controll|minority interest', s):
        return 'Noncontrolling interest'
    if re.search(r'compensation', s):
        return 'Compensation (other)'
    return 'OTHER'


def add_tcja_line(ax, y_text=None, text='TCJA'):
    ax.axvline(2017.5, color='#555', lw=1.2, ls='--', zorder=3)
    if y_text is not None:
        ax.text(2017.7, y_text, text, fontsize=9, style='italic', color='#555')


def statutory_rate(year):
    return 35.0 if year < 2018 else 21.0


def moving_avg(series, window=3):
    return series.rolling(window, center=True, min_periods=2).mean()


def cache_meta_path(cache_file):
    return f'{cache_file}.meta.json'


def path_signature(path):
    abspath = os.path.abspath(os.fspath(path))
    stat = os.stat(abspath)
    return {'path': abspath, 'mtime_ns': stat.st_mtime_ns, 'size': stat.st_size}


def normalize_cache_dep(value):
    if isinstance(value, os.PathLike):
        return path_signature(value)
    if isinstance(value, str) and os.path.exists(value):
        return path_signature(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalize_cache_dep(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_cache_dep(v) for v in value]
    if isinstance(value, set):
        return [normalize_cache_dep(v) for v in sorted(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def cache_signature(cache_file):
    meta_file = cache_meta_path(cache_file)
    return {
        'payload': path_signature(cache_file),
        'meta': path_signature(meta_file),
    }


def load_pickle_cache(cache_file, label, deps):
    meta_file = cache_meta_path(cache_file)
    if not os.path.exists(cache_file):
        return None
    if not os.path.exists(meta_file):
        print(f'{label}: cache metadata missing, rebuilding...')
        return None

    expected = {'cache_version': CACHE_VERSION, 'deps': normalize_cache_dep(deps)}
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            actual = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f'{label}: cache metadata unreadable, rebuilding...')
        return None

    if actual != expected:
        print(f'{label}: cache stale, rebuilding...')
        return None

    print(f'{label}: loading from cache...')
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def write_pickle_cache(cache_file, payload, deps):
    meta_file = cache_meta_path(cache_file)
    meta = {'cache_version': CACHE_VERSION, 'deps': normalize_cache_dep(deps)}
    with open(cache_file, 'wb') as f:
        pickle.dump(payload, f)
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


# ── Co-registrant collapse ───────────────────────────────────────────────────

def coregistrant_score(df):
    """Relative mismatch between the extracted reconciliation and each firm's
    own Compustat data.

    Amount-form rows are compared on tax expense.  The filing's reporting unit
    (units / thousands / millions) is not recorded, so the best of the plausible
    scales is taken.  Rate-form rows are compared on the effective rate itself,
    expressed on the same relative scale so that a single tolerance covers both.
    """
    score = pd.Series(np.inf, index=df.index)
    amt = df['israte'] == 0
    for scale in (1.0, 1e3, 1e6):
        rel = ((df['comp_txt'] - df['effective_tax'] / scale).abs()
               / df['comp_txt'].abs().clip(lower=0.01))
        score = score.where(~(amt & rel.notna()), np.minimum(score, rel.fillna(np.inf)))
    rate = (df['israte'] == 1) & df['comp_pi'].abs().gt(1)
    etr_diff = (df['comp_txt'] / df['comp_pi'] * 100 - df['effective_tax']).abs() / 100
    return score.mask(rate, etr_diff)


def collapse_coregistrants(df, tol=0.01):
    """Keep one firm per (filing, reconciliation year) for combined 10-Ks.

    Utility holding companies and their subsidiaries file a single combined
    10-K as co-registrants.  Stage 2 extracts one reconciliation from that
    document, and the base panel attaches it to every registrant, so the same
    tax table is repeated across firms that did not report it.

    The reconciliation belongs to exactly one registrant, but not always the
    parent -- it is the operating subsidiary in a sizeable minority of cases.
    Keep the firm whose own Compustat data matches the extracted figures, and
    fall back to the largest registrant by sales when no unique match clears
    ``tol``.
    """
    multi = df['filing_entity_count'] > 1
    if not multi.any():
        return df

    m = df[multi].copy()
    m['_score'] = coregistrant_score(m)
    g = m.groupby(['filing_id', 'recon_year'], sort=False)
    m['_win'] = (m['_score'] == g['_score'].transform('min')) & (m['_score'] <= tol)
    n_win = g['_win'].transform('sum')

    unique = m['_win'] & (n_win == 1)
    fallback = (n_win != 1) & (m['comp_sale'] == g['comp_sale'].transform('max'))
    # a sales tie would keep two rows; take the first deterministically
    fallback &= ~fallback.groupby([m['filing_id'], m['recon_year']]).cumsum().gt(1)
    keep = unique | fallback

    n_groups = g.ngroups
    n_matched = int(unique.groupby([m['filing_id'], m['recon_year']]).any().sum())
    print(f"  Co-registrant collapse: {len(m):,} rows across {n_groups:,} "
          f"filing-years -> {int(keep.sum()):,} "
          f"({n_matched:,} matched, {n_groups - n_matched:,} by largest-sales fallback)")
    if int(keep.sum()) != n_groups:
        raise RuntimeError(
            f"Co-registrant collapse kept {int(keep.sum()):,} rows for "
            f"{n_groups:,} filing-years; expected exactly one row each."
        )
    return pd.concat([df[~multi], m[keep].drop(columns=['_score', '_win'])])


# ── Section 1: Load and filter ────────────────────────────────────────────────

def load_and_filter(csv_path, cache_dir):
    cache_file = os.path.join(cache_dir, 'filtered.pkl')
    deps = {
        'script': __file__,
        'input_csv': csv_path,
        'cols': COLS,
        'year_range': [1998, 2024],
        'coregistrant_collapse_tol': 0.01,
        'coregistrant_collapse_order': 'before_filters',
    }
    cached = load_pickle_cache(cache_file, 'Section 1', deps)
    if cached is not None:
        return cached

    print(f"Section 1: reading CSV from {csv_path} ...")
    # Only load columns we need; ignore missing ones silently
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    load_cols = [c for c in COLS if c in all_cols]
    df = pd.read_csv(csv_path, usecols=load_cols, low_memory=False)
    print(f"  Loaded: {len(df):,} rows")

    # Resolve co-registrants first, so the reconciliation is attributed using
    # every candidate registrant rather than whichever ones happen to survive
    # the filters below. This leaves exactly one row per (filing, recon_year),
    # so the filters that follow operate on filing-years rather than on the
    # firm-expanded panel.
    df = collapse_coregistrants(df)
    print(f"  After co-registrant resolution: {len(df):,}")

    # Filter 1: year range
    df = df[df['recon_year'].between(1998, 2024)]
    print(f"  After year filter [1998,2024]: {len(df):,}")

    # Filter 2: within_expected_filing_window
    df = df[df['within_expected_filing_window'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    print(f"  After filing window filter: {len(df):,}")

    # Filter 3: has_compustat_year_match
    df = df[df['has_compustat_year_match'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    print(f"  After compustat match filter: {len(df):,}")

    # Filter 4: recon_check
    df = df[df['recon_check'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    print(f"  After recon_check filter: {len(df):,}")

    write_pickle_cache(cache_file, df, deps)
    return df


# ── Section 2: Deduplicate to firm-year ──────────────────────────────────────

def deduplicate(df, cache_dir):
    cache_file = os.path.join(cache_dir, 'deduped.pkl')
    deps = {
        'script': __file__,
        'filtered_cache': cache_signature(os.path.join(cache_dir, 'filtered.pkl')),
    }
    cached = load_pickle_cache(cache_file, 'Section 2', deps)
    if cached is not None:
        return cached

    print("Section 2: deduplicating to firm-year...")
    conf_rank_map = {'source_reviewed': 3, 'high': 2, 'medium': 1, 'low': 0}
    df = df.copy()
    df['conf_rank'] = df['extraction_confidence'].map(conf_rank_map).fillna(0).astype(int)
    df['abs_offset'] = df['year_offset_from_filing'].abs()
    df['repair_rank'] = df.get('overlap_repair_role', pd.Series('', index=df.index)).apply(repair_rank)
    df['filed_at_dt'] = pd.to_datetime(df['filed_at'], utc=True, errors='coerce')

    df = df.sort_values(
        ['gvkey', 'recon_year', 'israte', 'repair_rank', 'abs_offset', 'conf_rank', 'filed_at_dt'],
        ascending=[True, True, True, True, True, False, False]
    )

    rate_df = df[df['israte'] == 1].drop_duplicates(['gvkey', 'recon_year'], keep='first').copy()
    amt_df  = df[df['israte'] == 0].drop_duplicates(['gvkey', 'recon_year'], keep='first').copy()

    print(f"  Rate-form rows after dedup: {len(rate_df):,}")
    print(f"  Amount-form rows after dedup: {len(amt_df):,}")

    result = {'rate': rate_df, 'amount': amt_df}
    write_pickle_cache(cache_file, result, deps)
    return result


# ── Section 3: Standardize rate reconciliations ───────────────────────────────

def standardize_rate(rate_df, cache_dir):
    cache_file = os.path.join(cache_dir, 'rate_clean.pkl')
    deps = {
        'script': __file__,
        'deduped_cache': cache_signature(os.path.join(cache_dir, 'deduped.pkl')),
        'statutory_tolerance_pp': 2.0,
        'etr_clip': [-100, 100],
        'gap_clip': [-120, 120],
    }
    cached = load_pickle_cache(cache_file, 'Section 3', deps)
    if cached is not None:
        return cached

    print("Section 3: standardizing rate-form rows...")
    df = rate_df.copy()

    df['stat_abs'] = df['statutory_tax'].abs()
    df['expected_stat'] = df['recon_year'].apply(lambda y: 35.0 if y < 2018 else 21.0)
    df = df[((df['stat_abs'] - df['expected_stat']).abs() <= 2.0)].copy()
    print(f"  After statutory proximity filter: {len(df):,}")

    sign = np.sign(df['statutory_tax'])
    df['etr_signed'] = df['effective_tax'] * sign
    df['stat_signed'] = df['statutory_tax'] * sign

    cat_val_cols = [f'category_{i}_val' for i in range(1, 16)]
    for col in cat_val_cols:
        if col in df.columns:
            df[col] = df[col] * sign

    df['etr_w'] = df['etr_signed'].clip(-100, 100)
    df['gap'] = df['etr_signed'] - df['stat_signed']
    df['gap_w'] = df['gap'].clip(-120, 120)
    df['source'] = 'rate_stated'

    write_pickle_cache(cache_file, df, deps)
    return df


# ── Section 4: Convert amount reconciliations to rate form ───────────────────

def convert_amount(amt_df, cache_dir):
    cache_file = os.path.join(cache_dir, 'amount_clean.pkl')
    deps = {
        'script': __file__,
        'deduped_cache': cache_signature(os.path.join(cache_dir, 'deduped.pkl')),
        'min_abs_statutory_tax': 10,
        'etr_clip': [-100, 100],
        'gap_clip': [-120, 120],
    }
    cached = load_pickle_cache(cache_file, 'Section 4', deps)
    if cached is not None:
        return cached

    print("Section 4: converting amount-form rows to rate form...")
    df = amt_df.copy()

    df = df[df['statutory_tax'].abs() >= 10].copy()

    df['stat_rate'] = df['recon_year'].apply(lambda y: 0.21 if y >= 2018 else 0.35)
    df['implied_pi'] = df['statutory_tax'] / df['stat_rate']

    df['etr_from_amount'] = 100 * df['effective_tax'] / df['implied_pi']

    for i in range(1, 16):
        val_col = f'category_{i}_val'
        new_col = f'c{i}_pp'
        if val_col in df.columns:
            df[new_col] = 100 * df[val_col] / df['implied_pi']
        else:
            df[new_col] = np.nan

    # Rename pp columns back to category_{i}_val for uniform schema
    for i in range(1, 16):
        df[f'category_{i}_val'] = df[f'c{i}_pp']

    df['stat_signed'] = df['stat_rate'] * 100
    df['etr_signed'] = df['etr_from_amount']

    # Keep only: implied_pi > 0 AND etr in [-100, 100]
    df = df[(df['implied_pi'] > 0) & (df['etr_from_amount'].between(-100, 100))].copy()
    print(f"  Amount-form after conversion filter: {len(df):,}")

    df['etr_w'] = df['etr_signed'].clip(-100, 100)
    df['gap'] = df['etr_signed'] - df['stat_signed']
    df['gap_w'] = df['gap'].clip(-120, 120)
    df['source'] = 'amount_derived'

    write_pickle_cache(cache_file, df, deps)
    return df


# ── Section 5: Combine forms ──────────────────────────────────────────────────

def combine_forms(rate_clean, amount_clean, cache_dir):
    cache_file = os.path.join(cache_dir, 'combined.pkl')
    deps = {
        'script': __file__,
        'rate_cache': cache_signature(os.path.join(cache_dir, 'rate_clean.pkl')),
        'amount_cache': cache_signature(os.path.join(cache_dir, 'amount_clean.pkl')),
    }
    cached = load_pickle_cache(cache_file, 'Section 5', deps)
    if cached is not None:
        return cached

    print("Section 5: combining rate and amount forms...")
    rate_keys = set(zip(rate_clean['gvkey'], rate_clean['recon_year']))
    amount_only = amount_clean[
        ~amount_clean.apply(lambda r: (r['gvkey'], r['recon_year']) in rate_keys, axis=1)
    ].copy()

    combined = pd.concat([rate_clean, amount_only], ignore_index=True, sort=False)
    print(f"  Combined firm-year count: {len(combined):,}")

    write_pickle_cache(cache_file, combined, deps)
    return combined


# ── Section 6: Build profitable subsample ────────────────────────────────────

def build_profitable(combined, cache_dir):
    cache_file = os.path.join(cache_dir, 'profitable.pkl')
    deps = {
        'script': __file__,
        'combined_cache': cache_signature(os.path.join(cache_dir, 'combined.pkl')),
        'min_comp_pi': 1.0,
        'min_comp_sale': 0.0,
        'finance_filter': 'sic_industry_v1_2026_05_27',
    }
    cached = load_pickle_cache(cache_file, 'Section 6', deps)
    if cached is not None:
        return cached

    print("Section 6: building profitable subsample (ex Finance)...")
    df = combined.copy()
    df = df[df['comp_pi'].notna() & (df['comp_pi'] > 1.0)].copy()
    df = df[df['comp_sale'].notna() & (df['comp_sale'] > 0)].copy()
    print(f"  Profitable firm-years (pre Finance filter): {len(df):,}")

    df['industry'] = df['comp_sic'].apply(industry)
    n_before = len(df)
    df = df[df['industry'] != 'Finance'].copy()
    print(f"  Dropped {n_before - len(df):,} Finance firm-years")
    print(f"  Profitable firm-years (post Finance filter): {len(df):,}")

    # size_bin: qcut within each year
    def qcut_year(grp):
        try:
            grp['size_bin'] = pd.qcut(grp['comp_sale'], q=4,
                                       labels=['Q1 Small', 'Q2 Mid-small', 'Q3 Mid-large', 'Q4 Large'],
                                       duplicates='drop')
        except Exception:
            grp['size_bin'] = np.nan
        return grp
    _yr = df['recon_year']  # pandas 3.0 drops group key from apply result
    df = df.groupby('recon_year', group_keys=False).apply(qcut_year)
    df['recon_year'] = _yr

    write_pickle_cache(cache_file, df, deps)
    return df


# ── Section 7: Reshape adjustments to long format ────────────────────────────

def build_long(combined, cache_dir):
    cache_file = os.path.join(cache_dir, 'long.pkl')
    deps = {
        'script': __file__,
        'combined_cache': cache_signature(os.path.join(cache_dir, 'combined.pkl')),
    }
    cached = load_pickle_cache(cache_file, 'Section 7', deps)
    if cached is not None:
        return cached

    print("Section 7: reshaping to long format...")
    slices = []
    for i in range(1, 16):
        lbl_col = f'category_{i}_label'
        val_col = f'category_{i}_val'
        if lbl_col not in combined.columns:
            continue
        sub = combined[['gvkey', 'recon_year', 'source', 'comp_pi', 'comp_sale', lbl_col, val_col]].copy()
        sub = sub.rename(columns={lbl_col: 'raw_label', val_col: 'cat_val'})
        sub = sub[sub['raw_label'].notna() & (sub['raw_label'] != '')]
        slices.append(sub)

    long_df = pd.concat(slices, ignore_index=True)
    print(f"  Long-format rows: {len(long_df):,}")

    long_df['category'] = long_df['raw_label'].apply(classify)
    long_df['group'] = long_df['category'].map(group_map).fillna('Other')

    write_pickle_cache(cache_file, long_df, deps)
    return long_df


# ── Section 8 & 9: Category and group classification already defined above ───

# ── Section 10: Industry classification already defined above ────────────────

# ── Section 6b: Full sample (includes loss firms) ────────────────────────────

def build_full_sample(combined, cache_dir):
    """All firm-years with comp_sale > 0, regardless of profitability.

    Finance firms (per the shared `sic_industry.is_finance_sic` rule)
    are excluded so the descriptive figures share the same industry
    universe as every regression sample.
    """
    cache_file = os.path.join(cache_dir, 'full_sample.pkl')
    deps = {
        'script': __file__,
        'combined_cache': cache_signature(os.path.join(cache_dir, 'combined.pkl')),
        'min_comp_sale': 0.0,
        'finance_filter': 'sic_industry_v1_2026_05_27',
    }
    cached = load_pickle_cache(cache_file, 'Section 6b', deps)
    if cached is not None:
        return cached

    print("Section 6b: building full sample (incl. loss firms, ex Finance)...")
    df = combined[combined['comp_sale'].notna() & (combined['comp_sale'] > 0)].copy()
    print(f"  Full sample firm-years (pre Finance filter): {len(df):,}")

    df['industry'] = df['comp_sic'].apply(industry)
    n_before = len(df)
    df = df[df['industry'] != 'Finance'].copy()
    print(f"  Dropped {n_before - len(df):,} Finance firm-years "
          f"({100 * (n_before - len(df)) / max(n_before, 1):.1f}%)")
    print(f"  Full sample firm-years (post Finance filter): {len(df):,}")

    def qcut_year(grp):
        try:
            grp['size_bin'] = pd.qcut(grp['comp_sale'], q=4,
                                       labels=['Q1 Small', 'Q2 Mid-small', 'Q3 Mid-large', 'Q4 Large'],
                                       duplicates='drop')
        except Exception:
            grp['size_bin'] = np.nan
        return grp
    _yr2 = df['recon_year']
    df = df.groupby('recon_year', group_keys=False).apply(qcut_year)
    df['recon_year'] = _yr2

    write_pickle_cache(cache_file, df, deps)
    return df


# ── Build driver summary (firm-year × group) ─────────────────────────────────

def build_driver_panel(long_df, sample_df, cache_dir, weight_col='comp_pi',
                       cache_name='driver_panel', sample_cache_name='profitable'):
    """Aggregate long adjustments to firm-year × group contributions (winsorized ±15).

    weight_col: column from sample_df used for weighted averaging in figure functions.
                Use 'comp_pi' for the profitable sample, 'comp_sale' for the full sample.
    """
    cache_file = os.path.join(cache_dir, f'{cache_name}.pkl')
    deps = {
        'script': __file__,
        'long_cache': cache_signature(os.path.join(cache_dir, 'long.pkl')),
        'sample_cache': cache_signature(os.path.join(cache_dir, f'{sample_cache_name}.pkl')),
        'weight_col': weight_col,
    }
    cached = load_pickle_cache(cache_file, f'Driver panel ({cache_name})', deps)
    if cached is not None:
        return cached

    print(f"Building firm-year × group driver panel ({cache_name})...")
    grp_sum = (long_df.groupby(['gvkey', 'recon_year', 'group'])['cat_val']
               .sum()
               .reset_index()
               .rename(columns={'cat_val': 'contrib'}))
    grp_sum['contrib_w'] = grp_sum['contrib'].clip(-15, 15)

    # deduplicate column list in case weight_col == 'comp_sale'
    slim_cols = list(dict.fromkeys(
        ['gvkey', 'recon_year', weight_col, 'industry', 'comp_sale', 'etr_w', 'stat_signed', 'gap_w']
    ))
    slim = (sample_df[slim_cols]
            .drop_duplicates(['gvkey', 'recon_year'])
            .rename(columns={weight_col: 'weight'}))
    grp_sum = grp_sum.merge(slim, on=['gvkey', 'recon_year'], how='inner')

    write_pickle_cache(cache_file, grp_sum, deps)
    return grp_sum


# ── Section 11: Figure 1 — ETR trend ─────────────────────────────────────────

def fig_etr_trend(profitable, out_dir, tables_dir):
    print("Figure 1: ETR trend...")

    years = sorted(profitable['recon_year'].unique())

    yearly = []
    for yr in years:
        sub = profitable[profitable['recon_year'] == yr]
        sub_pos = sub[sub['comp_pi'] > 0]
        stat = statutory_rate(yr)
        row = {
            'year': yr,
            'n_firms': len(sub),
            'statutory_rate': stat,
            'etr_mean': sub['etr_w'].mean(),
            'etr_median': sub['etr_w'].median(),
            'etr_p25': sub['etr_w'].quantile(0.25),
            'etr_p75': sub['etr_w'].quantile(0.75),
            'etr_mean_piweighted': piw_mean(sub['etr_w'].values, sub['comp_pi'].values),
            'gap_median': sub['gap_w'].median(),
            'gap_mean_piweighted': piw_mean(sub['gap_w'].values, sub['comp_pi'].values),
            'aggregate_etr_compustat': (
                sub_pos['comp_txt'].sum() / sub_pos['comp_pi'].sum()
                if sub_pos['comp_pi'].sum() > 0 else np.nan
            ) * 100,
        }
        yearly.append(row)

    tbl = pd.DataFrame(yearly).set_index('year')
    tbl.to_csv(os.path.join(tables_dir, 'etr_summary_by_year.csv'), float_format='%.2f')

    N = len(profitable)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Panel A
    ax = axes[0]
    ax.plot(tbl.index, tbl['statutory_rate'], color='black', lw=2, label='Statutory federal rate')
    ax.plot(tbl.index, tbl['etr_median'], color='#c0392b', marker='o', ms=4, lw=1.6,
            label='Median ETR (profitable)')
    ax.plot(tbl.index, tbl['etr_mean_piweighted'], color='#2980b9', marker='s', ms=4, lw=1.6,
            label='PI-weighted mean ETR')
    ax.plot(tbl.index, tbl['aggregate_etr_compustat'], color='#27ae60', lw=1.6, ls='--',
            label='Aggregate Compustat ETR')
    add_tcja_line(ax, y_text=42)
    ax.set_ylim(0, 45)
    ax.set_xlim(1997.5, 2024.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Tax rate (%)')
    ax.set_title(f'A. Statutory vs. effective tax rate (N = {N:,} profitable firm-years)')
    ax.legend(loc='lower left', fontsize=8.5)

    # Panel B
    ax = axes[1]
    ax.axhline(0, color='black', lw=0.8)
    ax.plot(tbl.index, tbl['gap_median'], color='#c0392b', marker='o', ms=4, lw=1.6,
            label='Median gap')
    ax.plot(tbl.index, tbl['gap_mean_piweighted'], color='#2980b9', marker='s', ms=4, lw=1.6,
            label='PI-weighted mean gap')
    add_tcja_line(ax, y_text=5.0)
    ax.set_ylim(-14, 6)
    ax.set_xlim(1997.5, 2024.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('ETR − Statutory (pp)')
    ax.set_title('B. Effective-to-statutory gap')
    ax.legend(loc='lower left', fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_etr_trend.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_etr_trend.png")
    return tbl


# ── Section 12: Figure 2 — Seven-group decomposition ─────────────────────────

def fig_drivers(driver_panel_full, full_sample, out_dir, tables_dir):
    """Fig 2: seven-group decomposition — full sample (incl. loss firms), revenue-weighted."""
    print("Figure 2: driver decomposition (full sample, revenue-weighted)...")

    years = sorted(full_sample['recon_year'].unique())

    # Revenue-weighted mean per year per group
    records = []
    for yr in years:
        sub = driver_panel_full[driver_panel_full['recon_year'] == yr]
        row = {'year': yr}
        for grp in GROUP_ORDER:
            gs = sub[sub['group'] == grp]
            row[grp] = piw_mean(gs['contrib_w'].values, gs['weight'].values) if len(gs) > 0 else 0.0
        records.append(row)

    wtd_grp = pd.DataFrame(records).set_index('year').fillna(0)
    wtd_grp.to_csv(os.path.join(tables_dir, 'etr_drivers_by_year_full.csv'), float_format='%.2f')

    draw_order = ['State & local', 'Permanent items', 'International', 'Federal credits',
                  'Reserves & losses', 'TCJA one-time', 'Other']

    N = len(full_sample)
    years_arr = np.array(years)
    n_years = len(years_arr)
    pos = wtd_grp[draw_order].clip(lower=0)
    neg = wtd_grp[draw_order].clip(upper=0)

    fig, ax = plt.subplots(figsize=(13, 6.2))

    ax.axvspan(1997.5, 2017.5, color='#e74c3c', alpha=0.02, zorder=0)
    ax.axvspan(2017.5, 2024.5, color='#3498db', alpha=0.04, zorder=0)
    add_tcja_line(ax)

    bot_pos = np.zeros(n_years)
    bot_neg = np.zeros(n_years)
    for c in draw_order:
        ax.bar(years_arr, pos[c].values, bottom=bot_pos, color=PALETTE[c],
               edgecolor='white', lw=0.4, width=0.82, label=c)
        bot_pos += pos[c].values
        ax.bar(years_arr, neg[c].values, bottom=bot_neg, color=PALETTE[c],
               edgecolor='white', lw=0.4, width=0.82)
        bot_neg += neg[c].values

    # Total gap overlay — revenue-weighted
    rev_gap = [piw_mean(
        full_sample[full_sample['recon_year'] == yr]['gap_w'].values,
        full_sample[full_sample['recon_year'] == yr]['comp_sale'].values
    ) for yr in years_arr]
    ax.plot(years_arr, rev_gap, color='black', lw=2.2, marker='o', ms=5,
            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
            label='Total gap (rev-wtd)', zorder=5)

    ax.text(2007.5, 4.8, '35% statutory regime', style='italic', fontsize=8.5,
            ha='center', color='#555')
    ax.text(2021, 4.8, '21% statutory\nregime', style='italic', fontsize=8.5,
            ha='center', color='#555')
    ax.text(2017.7, 3, 'TCJA', style='italic', fontsize=8.5, color='#555')

    ax.set_ylim(-10, 5.5)
    ax.set_xlim(1997.5, 2024.5)
    ax.set_xticks(range(1998, 2025, 2))
    ax.set_xlabel('Year')
    ax.set_ylabel('Contribution to ETR−Statutory gap (pp)')
    ax.set_title(f'ETR–Statutory gap decomposition by driver group\n'
                 f'(Revenue-weighted, all firms incl. losses, N = {N:,})')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_drivers.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_drivers.png")


# ── Section 13: Figure 3 — Industry ETR trajectories ─────────────────────────

def fig_ind_etr(profitable, out_dir):
    print("Figure 3: industry ETR trajectories...")

    focal = ['Pharma/Biotech', 'Tech', 'Manufacturing', 'Retail']
    sub = profitable[(profitable['recon_year'] >= 2002) & profitable['industry'].isin(focal)].copy()
    N = len(sub)

    fig, ax = plt.subplots(figsize=(12, 6))

    years = range(2002, 2025)
    stat_line = [statutory_rate(y) for y in years]
    ax.plot(list(years), stat_line, color='black', lw=2.3, label='Statutory rate', zorder=2)

    for ind in focal:
        idf = sub[sub['industry'] == ind]
        yearly = []
        for yr in range(2002, 2025):
            ys = idf[idf['recon_year'] == yr]
            if len(ys) < 25:
                yearly.append(np.nan)
            else:
                yearly.append(piw_mean(ys['etr_w'].values, ys['comp_pi'].values))
        s = pd.Series(yearly, index=range(2002, 2025))
        s_smooth = moving_avg(s, 3)
        ax.plot(range(2002, 2025), s_smooth, color=IND_COLORS[ind],
                marker=IND_MARKERS[ind], ms=4, lw=1.8, label=ind)

    add_tcja_line(ax, y_text=39)
    ax.set_ylim(10, 42)
    ax.set_xlim(2001, 2025)
    ax.set_xlabel('Year')
    ax.set_ylabel('PI-weighted ETR (%)')
    ax.set_title(
        f'Effective tax rate trajectories by industry, 2002–2024\n'
        f'(Profitable firms, N = {N:,}; 3-year moving avg, min 25 firm-years/year)'
    )
    ax.legend(loc='lower left', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_ind_etr.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_ind_etr.png")


# ── Section 14: Figure 4 — Industry × period decomposition ───────────────────

def fig_ind_drivers(driver_panel_full, full_sample, out_dir):
    print("Figure 4: industry × period decomposition...")

    # 'Other' was previously 'Finance' here, but the Finance bucket is filtered
    # out of full_sample (see build_full_sample); the 6th panel rendered NaN.
    # 'Other' is the catch-all that now contains the carve-in firms (Goldman
    # Sachs, State Street, Marsh, CBRE, WillScot, etc. — see slide A.7a).
    focal6 = ['Tech', 'Pharma/Biotech', 'Manufacturing', 'Retail', 'Oil & Gas', 'Other']
    period_defs = {
        'Late 35% (2013-17)': (2013, 2017),
        'Post-TCJA (2019-24)': (2019, 2024),
    }
    periods = list(period_defs.keys())
    N = len(full_sample[full_sample['industry'].isin(focal6) &
                         (full_sample['recon_year'].between(2013, 2017) |
                          full_sample['recon_year'].between(2019, 2024))])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    draw_order = ['State & local', 'Permanent items', 'International', 'Federal credits',
                  'Reserves & losses', 'TCJA one-time', 'Other']

    handles_legend = None

    for idx, ind in enumerate(focal6):
        ax = axes_flat[idx]
        ind_fs = full_sample[full_sample['industry'] == ind]
        ind_dp = driver_panel_full[driver_panel_full['industry'] == ind]

        etr_vals = {}
        bar_data = {}
        total_gap = {}

        for pid, (y0, y1) in period_defs.items():
            sub_fs = ind_fs[ind_fs['recon_year'].between(y0, y1)]
            sub_dp = ind_dp[ind_dp['recon_year'].between(y0, y1)]
            # ETR box: profitable firms only (ratio is meaningful)
            sub_prof = sub_fs[sub_fs['comp_pi'] > 1.0]
            etr_vals[pid] = piw_mean(sub_prof['etr_w'].values, sub_prof['comp_pi'].values)
            # Gap and driver bars: full sample, revenue-weighted
            total_gap[pid] = piw_mean(sub_fs['gap_w'].values, sub_fs['comp_sale'].values)
            grp_contribs = {}
            for grp in GROUP_ORDER:
                gs = sub_dp[sub_dp['group'] == grp]
                grp_contribs[grp] = piw_mean(gs['contrib_w'].values, gs['weight'].values) if len(gs) > 0 else 0.0
            bar_data[pid] = grp_contribs

        x = np.array([0, 1])
        bot_pos = np.zeros(2)
        bot_neg = np.zeros(2)
        patch_handles = []
        for grp in draw_order:
            vals = np.array([bar_data[p].get(grp, 0.0) for p in periods])
            vals = np.nan_to_num(vals)
            pos_v = np.clip(vals, 0, None)
            neg_v = np.clip(vals, None, 0)
            h = ax.bar(x, pos_v, bottom=bot_pos, color=PALETTE[grp],
                       edgecolor='white', lw=0.4, width=0.6, label=grp)
            patch_handles.append(h)
            bot_pos += pos_v
            ax.bar(x, neg_v, bottom=bot_neg, color=PALETTE[grp],
                   edgecolor='white', lw=0.4, width=0.6)
            bot_neg += neg_v

        if handles_legend is None:
            handles_legend = patch_handles

        # Total gap annotations — place outside the outermost stack edge
        for xi, pid in enumerate(periods):
            tg = total_gap.get(pid, np.nan)
            if not np.isnan(tg):
                if bot_pos[xi] >= abs(bot_neg[xi]):
                    ann_y = bot_pos[xi] + 0.55
                    va = 'bottom'
                else:
                    ann_y = bot_neg[xi] - 0.55
                    va = 'top'
                ax.text(xi, ann_y, f'{tg:.1f}pp', ha='center', va=va,
                        fontsize=8, fontweight='bold', color='#2c3e50')

        # ETR textbox
        etr_pre = etr_vals.get(periods[0], np.nan)
        etr_post = etr_vals.get(periods[1], np.nan)
        txt = f'ETR (pre):  {etr_pre:.1f}%\nETR (post): {etr_post:.1f}%'
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(periods, fontsize=8, rotation=10, ha='right')
        ax.set_title(ind, fontweight='bold', fontsize=12)
        ax.set_ylim(-14, 4)
        ax.axhline(0, color='black', lw=0.8)

    fig.suptitle(
        f'ETR–Statutory gap decomposition by industry, pre- vs. post-TCJA\n'
        f'(Revenue-weighted, all firms incl. losses; N = {N:,})',
        fontsize=12, y=1.01
    )

    if handles_legend:
        labels = [draw_order[i] for i in range(len(handles_legend))]
        fig.legend(handles_legend, labels,
                   loc='lower center', bbox_to_anchor=(0.5, -0.06),
                   ncol=7, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_ind_drivers.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_ind_drivers.png")


# ── Section 15: Figure 5 — U-shape binscatter ────────────────────────────────

def fig_u_gap(profitable, out_dir):
    print("Figure 5: U-shape binscatter...")

    period_specs = {
        '2013–17 (pre-TCJA)': (2013, 2017, '#c0392b', 'o'),
        '2019–24 (post-TCJA)': (2019, 2024, '#2980b9', 's'),
    }

    sub = profitable[
        (profitable['recon_year'].between(2013, 2017) | profitable['recon_year'].between(2019, 2024)) &
        (profitable['comp_sale'] > 0) & (profitable['comp_pi'] > 0)
    ].copy()
    sub['log_rev'] = np.log10(sub['comp_sale'])
    # Winsorize at a generous ±20pp so a single extreme-ETR firm can't
    # dominate the PI-weighted bin average, but preserves real variation
    sub['gap_for_fig'] = (sub['etr_w'] - sub['stat_signed']).clip(-20, 10)
    N = len(sub)

    fig, ax = plt.subplots(figsize=(11, 6))

    for label, (y0, y1, color, marker) in period_specs.items():
        ps = sub[sub['recon_year'].between(y0, y1)].copy()
        ps['bin'] = pd.qcut(ps['log_rev'], q=20, labels=False, duplicates='drop')
        bins = (ps.groupby('bin')[['log_rev', 'gap_for_fig', 'comp_pi']]
                .apply(lambda g: pd.Series({
                    'x_mean': g['log_rev'].mean(),
                    'y_mean': piw_mean(g['gap_for_fig'].values, g['comp_pi'].values),
                }), include_groups=False)
                .dropna()
                .sort_values('x_mean'))
        y_smooth = moving_avg(pd.Series(bins['y_mean'].values), window=7)

        ax.scatter(bins['x_mean'], bins['y_mean'], color=color, marker=marker,
                   s=70, alpha=0.55, edgecolors='white', lw=0.8, label=label, zorder=4)
        ax.plot(bins['x_mean'].values, y_smooth.values, color=color, lw=2.5, alpha=0.85, zorder=3)

    ax.axhline(0, color='black', lw=1.2, zorder=2)
    ax.text(5.85, 0.5, 'ETR = Statutory', fontsize=8.5, style='italic', color='#333',
            ha='right')

    # Annotations: arms of the ∩ (gap most negative at both ends)
    ax.annotate('', xy=(1.2, -6.8), xytext=(2.0, -4.0),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3))
    ax.annotate('', xy=(4.9, -8.5), xytext=(4.2, -4.0),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3))
    ax.text(3.1, -3.2, 'Gap largest at both size extremes',
            ha='center', fontsize=8.5, color='#c0392b')
    ax.text(1.35, -3.8, 'reserves\n& credits', ha='center', fontsize=7.5,
            color='#c0392b', style='italic')
    ax.text(5.05, -3.8, 'intl.\nshifting', ha='center', fontsize=7.5,
            color='#c0392b', style='italic')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['$10M', '$100M', '$1B', '$10B', '$100B'])
    ax.set_xlim(0.8, 5.9)
    ax.set_ylim(-13, 2)
    ax.set_xlabel('Revenue (log scale)')
    ax.set_ylabel('ETR − Statutory gap (pp)')
    ax.set_title(
        f'ETR–Statutory gap by firm size: largest at both ends (∩-shaped pre-TCJA)\n'
        f'(Binned scatter, 20 revenue bins/period; profitable sample N = {N:,} in 2013–24)'
    )
    ax.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_U_gap.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_U_gap.png")


# ── Section 16: Figure 6 — Drivers by size ───────────────────────────────────

def fig_drivers_by_size(driver_panel_full, full_sample, out_dir):
    """Fig 6: driver contributions by size — full sample (incl. losses), revenue-weighted."""
    print("Figure 6: drivers by size (full sample, revenue-weighted)...")

    focal_groups = ['State & local', 'International', 'Federal credits',
                    'Permanent items', 'Reserves & losses', 'Other']

    period_specs = {
        '2013–17 (pre)': (2013, 2017),
        '2019–24 (post)': (2019, 2024),
    }
    period_styles = {
        '2013–17 (pre)':  ('-',  'o', '#c0392b'),
        '2019–24 (post)': ('--', 's', '#2980b9'),
    }

    sub_fs = full_sample[
        full_sample['recon_year'].between(2013, 2017) |
        full_sample['recon_year'].between(2019, 2024)
    ].copy()
    sub_fs['log_rev'] = np.log10(sub_fs['comp_sale'])

    sub_dp = driver_panel_full[
        driver_panel_full['recon_year'].between(2013, 2017) |
        driver_panel_full['recon_year'].between(2019, 2024)
    ].copy()

    # Merge log_rev into driver panel
    fs_slim = sub_fs[['gvkey', 'recon_year', 'log_rev']].drop_duplicates()
    sub_dp = sub_dp.merge(fs_slim, on=['gvkey', 'recon_year'], how='inner',
                          suffixes=('', '_fs'))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    legend_drawn = False

    for idx, grp in enumerate(focal_groups):
        ax = axes_flat[idx]
        grp_df = sub_dp[sub_dp['group'] == grp]

        for label, (y0, y1) in period_specs.items():
            ls, mk, color = period_styles[label]
            ps = grp_df[grp_df['recon_year'].between(y0, y1)].copy()
            if len(ps) == 0:
                continue

            # Bin on full-sample log_rev for that period
            fs_per = sub_fs[sub_fs['recon_year'].between(y0, y1)].copy()
            fs_per['bin'] = pd.qcut(fs_per['log_rev'], q=20, labels=False, duplicates='drop')
            bin_map = fs_per.set_index(['gvkey', 'recon_year'])['bin'].to_dict()

            ps['bin'] = ps.apply(lambda r: bin_map.get((r['gvkey'], r['recon_year']), np.nan), axis=1)
            ps = ps.dropna(subset=['bin'])

            bins = (ps.groupby('bin')[['log_rev', 'contrib_w', 'weight']]
                    .apply(lambda g: pd.Series({
                        'x_mean': g['log_rev'].mean(),
                        'y_mean': piw_mean(g['contrib_w'].values, g['weight'].values),
                    }), include_groups=False)
                    .dropna().sort_values('x_mean'))

            y_smooth = moving_avg(pd.Series(bins['y_mean'].values), window=3)

            ax.plot(bins['x_mean'].values, y_smooth.values,
                    color=color, ls=ls, lw=1.8, alpha=0.85, label=label)
            ax.scatter(bins['x_mean'], bins['y_mean'], color=color, marker=mk,
                       s=35, alpha=0.55, edgecolors='white', lw=0.5)

        ax.axhline(0, color='black', lw=0.8)
        title_color_override = {'Reserves & losses': '#6b7a7b', 'Other': '#8a9090'}
        title_color = title_color_override.get(grp, PALETTE.get(grp, 'black'))
        ax.set_title(grp, fontweight='bold', color=title_color, fontsize=10)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['$10M', '$100M', '$1B', '$10B', '$100B'])
        ax.set_xlim(1.2, 5.2)
        ax.set_ylim(-7.5, 3.5)

        if idx in (3, 4, 5):
            ax.set_xlabel('Revenue (log scale)')
        if idx in (0, 3):
            ax.set_ylabel('Contribution to gap (pp)')

        if idx == 0 and not legend_drawn:
            ax.legend(loc='lower left', fontsize=8.5)
            legend_drawn = True

    fig.suptitle(
        'Contribution to ETR–Statutory gap by driver group and firm size\n'
        '(All firms incl. losses; revenue-weighted within 20 equal-count bins; 3-bin moving avg)',
        fontsize=10
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_drivers_by_size.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_drivers_by_size.png")


# ── Validation ────────────────────────────────────────────────────────────────

def validation(profitable):
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)

    # Yearly counts
    yr_counts = profitable.groupby('recon_year').size()
    print("\nYear-by-year profitable firm-year counts:")
    expected = {
        1998: 210, 1999: 628, 2000: 1125, 2001: 1345, 2002: 1733,
        2003: 2009, 2004: 2367, 2005: 2443, 2006: 2417, 2007: 2336,
        2008: 1927, 2009: 1897, 2010: 2187, 2011: 2151, 2012: 2077,
        2013: 2058, 2014: 2018, 2015: 1850, 2016: 1789, 2017: 1665,
        2018: 1519, 2019: 1638, 2020: 1485, 2021: 1813, 2022: 1706,
        2023: 1626, 2024: 1509,
    }
    for yr in sorted(yr_counts.index):
        actual = yr_counts[yr]
        exp = expected.get(yr, '?')
        flag = ''
        if isinstance(exp, int):
            pct_diff = abs(actual - exp) / exp * 100
            flag = '  *** >5% off' if pct_diff > 5 else ''
        print(f"  {yr}: actual={actual:5d}  expected={exp}{flag}")

    # Confidence breakdown
    print("\nConfidence breakdown (profitable sample):")
    print(profitable['extraction_confidence'].value_counts().to_string())

    # Reconciliation identity check
    cat_val_cols = [f'category_{i}_val' for i in range(1, 16)]
    avail = [c for c in cat_val_cols if c in profitable.columns]
    sum_contribs = profitable[avail].sum(axis=1)
    residual = (profitable['gap'] - sum_contribs).abs() if 'gap' in profitable.columns else None
    if residual is not None:
        print(f"\nReconciliation identity: median residual = {residual.median():.2e}")
        if residual.median() < 1e-10:
            print("  PASS (median residual < 1e-10)")
        else:
            print("  NOTE: median residual > 1e-10 (winsorization may have been applied)")

    # ETR headlines
    print("\nPI-weighted ETR headlines (profitable):")
    for yr in [2000, 2010, 2015, 2017, 2018, 2024]:
        sub = profitable[profitable['recon_year'] == yr]
        etr = piw_mean(sub['etr_w'].values, sub['comp_pi'].values)
        print(f"  {yr}: {etr:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '--csv', dest='input',
                        default=os.path.join(script_dir, '..', 'data', 'analysis', 'output', 'tax_reconciliation_panel_base.csv'))
    parser.add_argument('--figures-dir', default=os.path.join(script_dir, '..', 'data', 'analysis', 'figures'))
    parser.add_argument('--tables-dir', default=os.path.join(script_dir, '..', 'data', 'analysis', 'tables'))
    parser.add_argument('--cache-dir', default=os.path.join(script_dir, '..', 'data', 'analysis', 'cache'))
    args = parser.parse_args()

    fig_dir   = args.figures_dir
    tbl_dir   = args.tables_dir
    cache_dir = args.cache_dir

    os.makedirs(fig_dir,   exist_ok=True)
    os.makedirs(tbl_dir,   exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ── Section 1-2: Load, filter, deduplicate ────────────────────────────────
    raw = load_and_filter(args.input, cache_dir)
    deduped = deduplicate(raw, cache_dir)

    # ── Section 3-4: Standardize forms ───────────────────────────────────────
    rate_clean = standardize_rate(deduped['rate'], cache_dir)
    amount_clean = convert_amount(deduped['amount'], cache_dir)

    # ── Section 5: Combine ────────────────────────────────────────────────────
    combined = combine_forms(rate_clean, amount_clean, cache_dir)

    # ── Section 6: Profitable subsample (ETR-level figures) ──────────────────
    profitable = build_profitable(combined, cache_dir)

    # ── Section 6b: Full sample incl. losses (decomposition figures) ─────────
    full_sample = build_full_sample(combined, cache_dir)

    # ── Section 7: Long format ────────────────────────────────────────────────
    long_df = build_long(combined, cache_dir)

    # ── Driver panels ─────────────────────────────────────────────────────────
    # Profitable × PI-weighted  → used only for validation / tables
    driver_panel = build_driver_panel(long_df, profitable, cache_dir,
                                      weight_col='comp_pi', cache_name='driver_panel',
                                      sample_cache_name='profitable')
    # Full sample × revenue-weighted  → used for Figs 2, 4, 6
    driver_panel_full = build_driver_panel(long_df, full_sample, cache_dir,
                                           weight_col='comp_sale', cache_name='driver_panel_full',
                                           sample_cache_name='full_sample')

    # ── Figures ───────────────────────────────────────────────────────────────
    # ETR-level figures: profitable sample only (ratio is meaningful)
    fig_etr_trend(profitable, fig_dir, tbl_dir)
    fig_ind_etr(profitable, fig_dir)
    fig_u_gap(profitable, fig_dir)
    # Decomposition figures: full sample, revenue-weighted
    fig_drivers(driver_panel_full, full_sample, fig_dir, tbl_dir)
    fig_ind_drivers(driver_panel_full, full_sample, fig_dir)
    fig_drivers_by_size(driver_panel_full, full_sample, fig_dir)

    # ── Validation ────────────────────────────────────────────────────────────
    validation(profitable)

    print(f"\nDone. All figures saved to {fig_dir}, tables to {tbl_dir}")
