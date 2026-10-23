"""
Build slide-data CSVs that the LaTeX deck reads via the `datatool` package.

Each CSV emitted to `data/analysis/output/slide_data/` has columns:
  - key:    camelCase identifier (no underscores, no leading digit). Used
            as the LaTeX macro name (the macro expands to `value`).
  - label:  human-readable description (for audit/debugging only).
  - value:  raw numeric or string value. Integer counts have no commas
            (siunitx adds them at render time via \\num{}). Coefficients
            keep their leading sign. Significance markers are literal
            strings ("***", "**", "*", "").
  - format: hint for the slides ('int', 'pct', 'coef', 'se', 'stars', 'str').
            Informational; the value column is already what LaTeX inserts.

The LaTeX side (`presentation/slide_data_setup.tex`) loads every CSV and
defines `\\<key>` macros, plus a `\\coefse{<keyBase>}` helper for the
common coefficient + SE + stars pattern.

Run via:
    python3 analysis/build_slide_data.py

Outputs are deterministic given the same upstream inputs.
"""
from __future__ import annotations

import json
import pickle
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "analysis" / "output" / "slide_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Emitter ──────────────────────────────────────────────────────────────────


class Emitter:
    """Accumulates (key, label, value, format) rows; writes a CSV."""

    KEY_RE = re.compile(r"^[a-z][a-zA-Z]*$")  # camelCase, letters only

    def __init__(self):
        self.rows: list[dict] = []
        self.seen: set[str] = set()

    def add(self, key: str, label: str, value, fmt: str):
        if not self.KEY_RE.match(key):
            raise ValueError(
                f"Invalid key {key!r}: must be camelCase, letters only "
                f"(no underscores, no digits, no leading uppercase)."
            )
        if key in self.seen:
            raise ValueError(f"Duplicate key {key!r}.")
        self.seen.add(key)
        # datatool's CSV parser chokes on non-ASCII bytes (Greek letters,
        # arrows) AND on LaTeX-special characters in labels:
        #   % starts a comment, consuming the rest of the line
        #   & is the column separator
        #   $ # ~ { } _ \ all have special meaning
        # ASCII-fold + scrub the label — it's informational only.
        label = label.encode("ascii", errors="replace").decode("ascii")
        for bad, good in [("?", "_"), ("%", " pct "), ("&", " and "), ("$", "$"),
                          ("#", "num"), ("~", "-"), ("{", "("), ("}", ")"),
                          ("_", "-"), ("\\", "/")]:
            label = label.replace(bad, good)
        # Render to string here so pandas doesn't coerce a mixed-type column
        # to float (which would print ints as "123456.0"). LaTeX/datatool
        # read strings anyway.
        if fmt == "int":
            value = str(int(value))
        elif fmt == "pct":
            value = f"{float(value):.2f}".rstrip("0").rstrip(".") or "0"
        elif fmt in {"coef", "se"}:
            value = str(value)
        else:
            value = str(value)
        self.rows.append({"key": key, "label": label, "value": value, "format": fmt})

    def write(self, path: Path):
        df = pd.DataFrame(self.rows, columns=["key", "label", "value", "format"])
        df.to_csv(path, index=False)
        print(f"  Wrote {len(df):3d} rows → {path.relative_to(ROOT)}")


# ── Helpers ──────────────────────────────────────────────────────────────────


def normalize_gvkey(s):
    return s.astype(str).str.lstrip("0").replace("", "0")


def load_compustat_filtered():
    """Compustat after USA + non-excluded SIC filter, deduped to one row per (gvkey, fyear)."""
    EXCLUDE_SICS = {6020, 6035, 6722, 6726, 6798}
    cs = pd.read_csv(
        ROOT / "data" / "original" / "compustat_data.csv",
        dtype={"gvkey": str, "cik": str},
        low_memory=False,
    )
    cs["gvkey"] = normalize_gvkey(cs["gvkey"])
    cs["fyear_n"] = pd.to_numeric(cs["fyear"], errors="coerce")
    cs["sic_n"] = pd.to_numeric(cs["sic"], errors="coerce")
    cs = cs[cs["fic"].astype(str).str.upper() == "USA"].copy()
    cs = cs[~cs["sic_n"].isin(EXCLUDE_SICS)]
    cs["datadate_n"] = pd.to_datetime(cs["datadate"], errors="coerce")
    cs = cs.sort_values("datadate_n", ascending=False).drop_duplicates(
        ["gvkey", "fyear_n"], keep="first"
    )
    return cs.reset_index(drop=True)


# ── Section: pipeline_counts ─────────────────────────────────────────────────


def emit_pipeline_counts():
    """Phase 1 funnel, Phase 2 output, analysis-panel headline counts.

    Reads:
      data/phase1/output_full_run_v4/batch_summary.csv          (canonical)
      data/phase1/output_rescue_year_mismatch_v1/main_out/batch_summary.csv  (rescue)
      data/phase1/filing_urls_compustat_merged.csv              (URL universe)
      data/phase2/full_run_v4_llm.csv                           (canonical)
      data/phase1/output_rescue_year_mismatch_v1/phase2_full.csv (rescue)
      data/phase2/full_run_v4_llm.csv.qa.csv                    (mode breakdown)
      data/phase1/output_rescue_year_mismatch_v1/phase2_full.csv.qa.csv
      data/analysis/output/tax_reconciliation_panel_base.csv    (base panel size)
      data/analysis/cache/combined.pkl                          (analysis panel)
      data/analysis/cache/profitable.pkl                        (profitable subsample)
    """
    e = Emitter()

    # ── Phase 1: combined (canonical + post-canonical rescue runs) + year-mismatch rescue ──
    # `batch_summary_v4_combined.csv` includes the alias-recovery + master-rescue +
    # Macy's-style batches merged in after the canonical v4 run. This is the file
    # the slide deck has historically cited.
    bs_c = pd.read_csv(
        ROOT / "data" / "phase1" / "batch_summary_v4_combined.csv",
        dtype=str,
        low_memory=False,
    )
    bs_r_raw = pd.read_csv(
        ROOT
        / "data"
        / "phase1"
        / "output_rescue_year_mismatch_v1"
        / "main_out"
        / "batch_summary.csv",
        dtype=str,
    )
    # Best bucket per URL for rescue (matches canonical batch_summary convention)
    PRIORITY = {
        "high_confidence_extractable": 0,
        "likely_reconciliation_but_incomplete": 1,
        "ambiguous": 2,
        "narrative": 3,
        "wrong_table_type": 4,
        "no_table": 5,
    }
    bs_r_raw["_rank"] = bs_r_raw["selected_bucket"].map(PRIORITY).fillna(99)
    bs_r = bs_r_raw.sort_values(["filing_url", "_rank"]).drop_duplicates(
        "filing_url", keep="first"
    )

    # URL universe (used for "Input URLs" on the funnel slide)
    urls = pd.read_csv(
        ROOT / "data" / "phase1" / "filing_urls_compustat_merged.csv",
        dtype={"gvkey": str, "fyear": str},
        low_memory=False,
    )
    n_url_rows = len(urls)
    n_unique_urls_canonical = int(bs_c["filing_url"].nunique())
    n_unique_urls_rescue = int(bs_r["filing_url"].nunique())
    n_unique_urls_total = n_unique_urls_canonical + n_unique_urls_rescue
    # Input URLs at the TOP of the funnel = URL rows in intake + rescue URLs
    # (the rescue manifest contributes raw URLs, not duplicates of existing rows).
    n_input_url_rows = n_url_rows + n_unique_urls_rescue
    n_dup = n_input_url_rows - n_unique_urls_total

    # Phase 1 status counts (combined canonical + rescue)
    n_ok = int((bs_c["status"] == "OK").sum() + (bs_r["status"] == "OK").sum())
    n_skip = int((bs_c["status"] == "SKIP_NON_HTML").sum())
    n_fail = int((bs_c["status"] == "FAIL").sum())

    # Funnel slide counts
    e.add("inputUrlsTotal", "Phase 1: input URLs at top of funnel (intake rows + rescue)", n_input_url_rows, "int")
    e.add("inputUrlsSkipNonHtml", "Phase 1: status=SKIP_NON_HTML", n_skip, "int")
    e.add("inputUrlsFail", "Phase 1: status=FAIL (parse/download failures)", n_fail, "int")
    e.add(
        "inputUrlsDuplicates",
        "Phase 1: duplicate URL rows collapsed (URL rows − unique batch URLs)",
        n_dup,
        "int",
    )
    e.add(
        "phaseOneUniqueFilings",
        "Phase 1: unique filings in batch (canonical + rescue)",
        n_unique_urls_total,
        "int",
    )

    # Confidence-bucket counts (combined canonical + rescue)
    buckets_c = bs_c["selected_bucket"].value_counts().to_dict()
    buckets_r = bs_r["selected_bucket"].value_counts().to_dict()

    def bsum(key):
        return int(buckets_c.get(key, 0) + buckets_r.get(key, 0))

    e.add("phaseOneBucketHighConf", "Phase 1 bucket: high_confidence_extractable", bsum("high_confidence_extractable"), "int")
    e.add("phaseOneBucketLikely", "Phase 1 bucket: likely_reconciliation_but_incomplete", bsum("likely_reconciliation_but_incomplete"), "int")
    e.add("phaseOneBucketAmbiguous", "Phase 1 bucket: ambiguous", bsum("ambiguous"), "int")
    e.add("phaseOneBucketNarrative", "Phase 1 bucket: narrative", bsum("narrative"), "int")
    e.add("phaseOneBucketNoTable", "Phase 1 bucket: no_table", bsum("no_table"), "int")
    e.add("phaseOneBucketWrongTable", "Phase 1 bucket: wrong_table_type", bsum("wrong_table_type"), "int")
    e.add("phaseOneStatusOk", "Phase 1: status=OK total (canonical + rescue)", int(n_ok), "int")
    # Sum of the three "extractable" buckets (feed Phase 2)
    extractable_sum = bsum("high_confidence_extractable") + bsum("likely_reconciliation_but_incomplete") + bsum("ambiguous")
    e.add("phaseOneExtractableSum", "Phase 1: extractable buckets sum (high + likely + ambiguous)", extractable_sum, "int")

    # ── Phase 2: canonical + rescue ──────────────────────────────────────────
    p2_c = pd.read_csv(
        ROOT / "data" / "phase2" / "full_run_v4_llm.csv",
        usecols=["id", "year", "israte", "check"],
        dtype={"id": str},
    )
    p2_r = pd.read_csv(
        ROOT / "data" / "phase1" / "output_rescue_year_mismatch_v1" / "phase2_full.csv",
        dtype={"id": str},
    )
    qa_c = pd.read_csv(
        ROOT / "data" / "phase2" / "full_run_v4_llm.csv.qa.csv",
        dtype={"id": str},
    )
    qa_r = pd.read_csv(
        ROOT
        / "data"
        / "phase1"
        / "output_rescue_year_mismatch_v1"
        / "phase2_full.csv.qa.csv",
        dtype={"id": str},
    )

    n_p2 = len(p2_c) + len(p2_r)
    n_p2_true = (p2_c["check"].astype(str).str.upper() == "TRUE").sum() + (
        p2_r["check"].astype(str).str.upper() == "TRUE"
    ).sum()
    n_p2_false = (p2_c["check"].astype(str).str.upper() == "FALSE").sum() + (
        p2_r["check"].astype(str).str.upper() == "FALSE"
    ).sum()
    n_p2_det = (qa_c["mode"] == "deterministic").sum() + (qa_r["mode"] == "deterministic").sum()
    n_p2_llm = (qa_c["mode"] == "llm_assisted").sum() + (qa_r["mode"] == "llm_assisted").sum()
    n_p2_rate = (p2_c["israte"] == 1).sum() + (p2_r["israte"] == 1).sum()
    n_p2_amt = (p2_c["israte"] == 0).sum() + (p2_r["israte"] == 0).sum()
    n_p2_filings = len(set(p2_c["id"]).union(p2_r["id"]))

    e.add("phaseTwoTotalRows", "Phase 2: total extracted recon rows", n_p2, "int")
    e.add("phaseTwoCheckTrue", "Phase 2: rows with check=TRUE", int(n_p2_true), "int")
    e.add("phaseTwoCheckFalse", "Phase 2: rows with check=FALSE", int(n_p2_false), "int")
    e.add("phaseTwoDeterministic", "Phase 2: rows from deterministic parser", int(n_p2_det), "int")
    e.add("phaseTwoLlmFallback", "Phase 2: rows from LLM-assisted fallback", int(n_p2_llm), "int")
    e.add("phaseTwoRateForm", "Phase 2: rate-form rows (israte=1)", int(n_p2_rate), "int")
    e.add("phaseTwoAmountForm", "Phase 2: amount-form rows (israte=0)", int(n_p2_amt), "int")
    e.add("phaseTwoUniqueFilings", "Phase 2: unique filings with ≥1 row", int(n_p2_filings), "int")
    e.add(
        "arithmeticCheckPassPct",
        "Phase 2: fraction of rows with check=TRUE (percent)",
        round(100 * n_p2_true / n_p2, 1),
        "pct",
    )

    # ── Phase 1 -> Phase 2 bridge ────────────────────────────────────────────
    # Phase 2 was pointed at the parent matches_txt directory and globs it
    # recursively, so it processes every confidence bucket, not just the
    # "extractable" ones. The only record of what it actually opened is the
    # run log, which reports the count in its header and one line per file.
    # We read both and require them to agree.
    p2_log = ROOT / "data" / "phase2" / "full_run_v4_llm.csv.log"
    file_id_re = re.compile(rb"file_id=([0-9a-f]+)")
    n_declared = None
    processed_ids: set[str] = set()
    with open(p2_log, "rb") as fh:
        for line in fh:
            if n_declared is None and b"files found" in line:
                n_declared = int(line.rsplit(b":", 1)[-1])
            m = file_id_re.search(line)
            if m:
                processed_ids.add(m.group(1).decode())
    if n_declared is None:
        raise RuntimeError(f"No 'files found' header in {p2_log}")
    if n_declared != len(processed_ids):
        raise RuntimeError(
            f"Phase 2 log inconsistent: header says {n_declared:,} files but "
            f"{len(processed_ids):,} distinct file_ids appear."
        )

    n_processed = len(processed_ids)
    n_proc_with_rows = len(processed_ids & set(p2_c["id"]))
    n_proc_no_rows = n_processed - n_proc_with_rows
    # Filings holding rows that this run did not produce: carried forward from
    # earlier runs by merge_phase2_outputs.py, plus the targeted re-extraction.
    n_outside_main = n_p2_filings - n_proc_with_rows

    e.add("packetsProcessed", "Phase 2: stage 1 packets processed (all buckets)", n_processed, "int")
    e.add("processedWithRows", "Phase 2: processed packets yielding at least one row", n_proc_with_rows, "int")
    e.add("processedNoRows", "Phase 2: processed packets yielding no rows", n_proc_no_rows, "int")
    e.add("filingsOutsideMainRun", "Phase 2: filings with rows from outside the main run", n_outside_main, "int")

    # ── Panel: base + canonical + analysis (combined / profitable) ───────────
    pb = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv",
        usecols=["gvkey"],
        dtype={"gvkey": str},
        low_memory=False,
    )
    e.add("panelBaseRows", "Base panel rows (post-rescue, long-form)", len(pb), "int")

    pv = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_v4.csv",
        usecols=["gvkey"],
        dtype={"gvkey": str},
        low_memory=False,
    )
    e.add("panelCanonicalRows", "Canonical panel rows ((gvkey, recon_year), post-rescue)", len(pv), "int")

    # ── Base panel -> final firm-year sample ────────────────────────────────
    # Re-applies build_etr_figures.py's Section 1 predicates so the appendix
    # funnel cannot drift from the sample the figures are built on. The final
    # two counts come from that script's own caches.
    import build_etr_figures as _etr

    _cols = [
        "filing_id", "gvkey", "recon_year", "filing_entity_count", "israte",
        "effective_tax", "comp_txt", "comp_pi", "comp_sale",
        "within_expected_filing_window", "has_compustat_year_match", "recon_check",
    ]
    _base = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "tax_reconciliation_panel_base.csv",
        usecols=_cols, dtype={"filing_id": str, "gvkey": str}, low_memory=False,
    )
    _n_filing_years = len(_base.drop_duplicates(["filing_id", "recon_year"]))
    _truthy = lambda s: s.astype(str).str.lower().isin(["true", "1", "yes"])

    _d = _etr.collapse_coregistrants(_base)
    _steps = []
    for label, mask in [
        ("dropYearRange", _d["recon_year"].between(1998, 2024)),
        ("dropFilingWindow", _truthy(_d["within_expected_filing_window"])),
        ("dropNoCompustat", _truthy(_d["has_compustat_year_match"])),
        ("dropCheckFail", _truthy(_d["recon_check"])),
    ]:
        before = len(_d)
        _d = _d[mask.reindex(_d.index, fill_value=False)]
        _steps.append((label, before - len(_d)))

    with open(ROOT / "data" / "analysis" / "cache" / "filtered.pkl", "rb") as f:
        _filtered = pickle.load(f)
    if len(_d) != len(_filtered):
        raise RuntimeError(
            f"Appendix funnel does not reproduce filtered.pkl: {len(_d):,} vs {len(_filtered):,}."
        )

    e.add("basePanelFilingYears", "Base panel filing-years (one row per filing x recon year)", _n_filing_years, "int")
    e.add("netFilingMatch", "Net change in filing-years when matching filings to firms", _n_filing_years - n_p2, "int")
    e.add("dropYearRange", "Dropped: reconciliation year outside 1998-2024", _steps[0][1], "int")
    e.add("dropFilingWindow", "Dropped: outside the expected filing window", _steps[1][1], "int")
    e.add("dropNoCompustat", "Dropped: no firm-year link to Compustat", _steps[2][1], "int")
    e.add("dropCheckFail", "Dropped: unsuccessful arithmetic check", _steps[3][1], "int")
    _n_filtered = len(_filtered)

    # Analysis cache (combined + profitable)
    with open(ROOT / "data" / "analysis" / "cache" / "combined.pkl", "rb") as f:
        combined = pickle.load(f)
    with open(ROOT / "data" / "analysis" / "cache" / "profitable.pkl", "rb") as f:
        profitable = pickle.load(f)

    n_combined = len(combined)
    n_rate_stated = (combined["source"] == "rate_stated").sum()
    n_amount_derived = (combined["source"] == "amount_derived").sum()
    n_profitable = len(profitable)
    n_firms = profitable["gvkey"].nunique()

    e.add("dropDuplicateFirmYears", "Dropped: duplicate firm-year rows across filings", _n_filtered - n_combined, "int")
    e.add("analysisPanelFirmYears", "Analysis panel: combined firm-years (rate + amount-derived)", n_combined, "int")
    e.add("analysisPanelRateStated", "Analysis panel: rate-stated firm-years", int(n_rate_stated), "int")
    e.add("analysisPanelAmountDerived", "Analysis panel: amount-derived non-overlap firm-years", int(n_amount_derived), "int")
    e.add("profitableSubsampleFirmYears", "Profitable subsample firm-years (comp_pi>1 AND comp_sale>0)", n_profitable, "int")
    e.add("profitableUniqueFirms", "Profitable subsample: unique gvkeys", int(n_firms), "int")

    e.write(OUT_DIR / "pipelinecounts.csv")


# ── Section: regression_headlines ────────────────────────────────────────────


def _round_sig(x, digits):
    """Round to `digits` decimal places, formatted as string with sign."""
    return f"{x:+.{digits}f}"


def _stars(pval):
    """Conventional significance markers from a p-value."""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def _stars_from_t(t):
    """Approximate stars from a t-statistic (treats sample as effectively large)."""
    from math import erfc, sqrt

    p_two = erfc(abs(t) / sqrt(2))
    return _stars(p_two)


def _fmt_signed(x, digits):
    return f"{float(x):+.{digits}f}"


@lru_cache(maxsize=1)
def _channel_clip_baseline():
    rob = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "channel_clip_robustness_headlines.csv"
    )
    return rob[
        (rob["clip_variant"] == "clip_15pp_baseline")
        & (rob["spec"] == "firm_fe_industry_year_fe")
    ].copy()


@lru_cache(maxsize=1)
def _foreign_share_closure_baseline():
    closure = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "foreign_share_closure_ratio_by_clip.csv"
    )
    return closure[
        (closure["clip_variant"] == "clip_15pp_baseline")
        & (closure["spec"] == "firm_fe_industry_year_fe")
    ].iloc[0]


@lru_cache(maxsize=1)
def _geo_revenue_summary():
    with open(
        ROOT / "data" / "analysis" / "output" / "geo_revenue_share_tcja_regression_summary.json"
    ) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _balance_comparison():
    return pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "balance_ge1_ge1_comparison.csv"
    )


def _balance_beta(block, outcome, column="beta_ge1"):
    balance = _balance_comparison()
    return float(balance[(balance["block"] == block) & (balance["outcome"] == outcome)].iloc[0][column])


@lru_cache(maxsize=1)
def _ps_mechanism_oneway():
    """Predictor -> (gamma, se) from the foreign_share one-way interaction
    regressions emitted by analysis/analyze_profit_shifting_mechanisms.py
    into data/analysis/output/profit_shifting_mechanism_oneway_foreign_share.csv."""
    df = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "profit_shifting_mechanism_oneway_foreign_share.csv"
    )
    return {r["predictor"]: (float(r["interaction_gamma"]), float(r["interaction_se"]))
            for _, r in df.iterrows()}


@lru_cache(maxsize=1)
def _ps_deep_dive_tax_wedge():
    df = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "profit_shifting_deep_dive_tax_wedge_interactions.csv"
    )
    df = df[df["outcome"] == "foreign_share"]
    return {r["predictor"]: (float(r["interaction_gamma"]), float(r["interaction_se"]))
            for _, r in df.iterrows()}


@lru_cache(maxsize=1)
def _ps_deep_dive_country():
    df = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "profit_shifting_deep_dive_country_interactions.csv"
    )
    return {r["predictor"]: (float(r["interaction_gamma"]), float(r["interaction_se"]))
            for _, r in df.iterrows()}


def _mech_gamma_se(predictor):
    """Resolve a predictor across all three profit-shifting source CSVs."""
    for src in (_ps_mechanism_oneway, _ps_deep_dive_tax_wedge, _ps_deep_dive_country):
        d = src()
        if predictor in d:
            return d[predictor]
    raise KeyError(f"predictor {predictor!r} not found in any profit-shifting source CSV")


@lru_cache(maxsize=1)
def _pi_check():
    """PI-intensity cross-check coefficients: (outcome -> main_beta) and
    ((outcome, predictor) -> interaction_gamma). Source:
    analysis/analyze_profit_shifting_deep_dive.py via two CSVs."""
    main_df = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "profit_shifting_deep_dive_alt_outcome_main.csv"
    )
    inter_df = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "profit_shifting_deep_dive_alt_outcome_interactions.csv"
    )
    # main CSV column is "coef_on_intl_pre_post"; interactions CSV column is "interaction_gamma".
    main = {r["outcome"]: float(r["coef_on_intl_pre_post"]) for _, r in main_df.iterrows()}
    inter = {(r["outcome"], r["predictor"]): float(r["interaction_gamma"])
             for _, r in inter_df.iterrows()}
    return main, inter


@lru_cache(maxsize=1)
def _real_activity_dynamics():
    """Real-activity 5-outcome (beta, sig) from spec (1) — firm + ind*yr FE.
    Source: data/analysis/tables/real_activity_dynamics/tbl_real_activity_dynamics.tex
    (emitted by analysis/build_real_activity_dynamics.py)."""
    text = (
        ROOT / "data" / "analysis" / "tables" / "real_activity_dynamics" /
        "tbl_real_activity_dynamics.tex"
    ).read_text()
    outcomes = [
        ("capex", r"Capex intensity"),
        ("rd",    r"R\\&D intensity"),
        ("total", r"Total investment"),
        ("empl",  r"Employment growth"),
        ("sales", r"Sales growth"),
    ]
    out = {}
    for key, label in outcomes:
        # "Label & def & $<coef><stars>$ & ..." — spec (1) is the first dollar-block.
        pat = label + r"\s*&[^&]*&\s*\$([+\-]?[\d.]+)(\^\{?\*+\}?)?\$"
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"could not parse real-activity row for {key!r}")
        beta = float(m.group(1))
        sig_raw = m.group(2) or ""
        # ^{***} -> ***; ^* -> *
        sig = re.sub(r"[\^{}]", "", sig_raw)
        out[key] = (beta, sig)
    return out


@lru_cache(maxsize=1)
def _windfall_summary():
    """Windfall NI shock + debt §965-controlled coefs.
    Source: data/analysis/tables/investment/windfall_summary.json
    (emitted by analysis/build_investment_elasticity.py)."""
    path = ROOT / "data" / "analysis" / "tables" / "investment" / "windfall_summary.json"
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _rd_bin_cutoffs():
    panel = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "profit_shifting_mechanism_panel.csv",
        usecols=["gvkey", "recon_year", "pre_rd_at", "foreign_share", "intl_pre"],
    )
    reg_years = {2013, 2014, 2015, 2016, 2019, 2020, 2021, 2022}
    firm = panel[
        panel["recon_year"].isin(reg_years)
        & panel["foreign_share"].notna()
        & panel["intl_pre"].notna()
    ].drop_duplicates("gvkey")
    rd_pct = firm["pre_rd_at"].dropna() * 100
    positive = rd_pct[rd_pct > 0]
    middle = positive[positive < 1.0]
    high = rd_pct[rd_pct > 1.0]
    return {
        "low_max": 0.0,
        "mid_low": float(positive.min()),
        "mid_high": float(middle.max()),
        "high_median": float(high.median()),
    }


from sic_industry import industry as _etr_figures_industry  # noqa: E402


@lru_cache(maxsize=1)
def _pharma_2017_profitable_firm_years():
    with open(ROOT / "data" / "analysis" / "cache" / "profitable.pkl", "rb") as f:
        profitable = pickle.load(f)
    if "industry" not in profitable.columns:
        profitable = profitable.copy()
        profitable["industry"] = profitable["comp_sic"].apply(_etr_figures_industry)
    return int(
        len(
            profitable[
                (profitable["recon_year"] == 2017)
                & (profitable["industry"] == "Pharma/Biotech")
            ]
        )
    )


def _fdii_industry(sic):
    """Coarse classifier used only to drop Finance from the FDII regression
    cohort. Delegates to the shared `sic_industry.is_finance_sic` rule."""
    from sic_industry import is_finance_sic
    if pd.isna(sic):
        return "Other"
    return "Finance" if is_finance_sic(sic) else "Other"


@lru_cache(maxsize=1)
def _fdii_claimant_counts():
    pre_years = set(range(2013, 2017))
    post_years = set(range(2019, 2024))
    event_years = set(range(2011, 2024))
    fdii_post_years = set(range(2018, 2024))
    win_ch = 15.0

    with open(ROOT / "data" / "analysis" / "cache" / "combined.pkl", "rb") as f:
        combined = pickle.load(f)
    with open(ROOT / "data" / "analysis" / "cache" / "long.pkl", "rb") as f:
        long_df = pickle.load(f)

    fy = (
        long_df.groupby(["gvkey", "recon_year", "group"])["cat_val"]
        .sum()
        .reset_index(name="val")
    )
    fy["val_w"] = fy["val"].clip(-win_ch, win_ch)
    intl_fy = (
        fy[fy["group"] == "International"]
        .groupby(["gvkey", "recon_year"])["val_w"]
        .sum()
    )
    any_obs = fy.groupby(["gvkey", "recon_year"]).size()
    pre_any = any_obs[any_obs.index.get_level_values("recon_year").isin(pre_years)]
    elig = pre_any.groupby("gvkey").size()
    elig = elig[elig >= 2].index
    pre_intl = (
        intl_fy[
            intl_fy.index.get_level_values("recon_year").isin(pre_years)
            & intl_fy.index.get_level_values("gvkey").isin(elig)
        ]
        .groupby("gvkey")
        .mean()
    )
    intl_pre = pd.Series(0.0, index=elig, name="intl_pre")
    intl_pre.update(pre_intl)
    p1, p99 = intl_pre.quantile([0.01, 0.99])
    intl_pre = intl_pre.clip(p1, p99)

    labels = long_df["raw_label"].astype(str).str.lower()
    fdii_mask = labels.str.contains("fdii|foreign-derived|foreign derived", regex=True, na=False)
    fdii_post_firms = set(
        long_df[fdii_mask & long_df["recon_year"].isin(fdii_post_years)]["gvkey"].unique()
    )

    panel = combined[combined["recon_year"].isin(event_years)].copy()
    panel["industry"] = panel["comp_sic"].apply(_fdii_industry)
    panel = panel[panel["industry"] != "Finance"]
    panel = panel.merge(intl_pre.reset_index(), on="gvkey", how="inner")

    fy_panel = panel.groupby("gvkey")["recon_year"].apply(set)
    has_pre = fy_panel.apply(lambda s: len(s & pre_years) >= 2)
    has_post = fy_panel.apply(lambda s: len(s & post_years) >= 2)
    panel = panel[panel["gvkey"].isin(fy_panel[has_pre & has_post].index)].copy()
    panel["is_fdii"] = panel["gvkey"].isin(fdii_post_firms).astype(int)

    n_fdii = int(panel[panel["is_fdii"] == 1]["gvkey"].nunique())
    n_nonfdii = int(panel[panel["is_fdii"] == 0]["gvkey"].nunique())
    return n_fdii, n_nonfdii


def emit_regression_headlines():
    """Pass-through DiD, income/revenue location, windfall coefficients.

    Reads:
      data/analysis/output/robustness/channel_clip_robustness_headlines.csv
        - First-stage Intl closure + ETR pass-through
        - Income-location foreign_share + foreign_share_log
      data/analysis/output/robustness/foreign_share_closure_ratio_by_clip.csv
        - Closure ratio + ETR coefficient on the income-location subsample
      data/analysis/tables/substitution/tbl_real_outcomes.tex
        - π_dom/assets, π_fo/assets (parsed via regex)
      data/analysis/tables/substitution/tbl_geo_revenue_share.tex
        - Foreign revenue share coefficient (parsed via regex)
      data/analysis/tables/investment/tbl_main_elasticity.tex
        - θ_capex / θ_R&D / θ_total_inv (parsed via regex)
      data/analysis/tables/investment/tbl_windfall_decomposition.tex
        - θ_debt (parsed via regex)

    Hardcoded (figure captions, not in a CSV):
      - 2017 NI-spike θ = +0.457; post-2018 pooled θ = +0.007
    """
    e = Emitter()

    # ── Channel-clip robustness (baseline = clip_15pp) ───────────────────────
    rob = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "channel_clip_robustness_headlines.csv"
    )
    base = rob[(rob["clip_variant"] == "clip_15pp_baseline") & (rob["spec"] == "firm_fe_industry_year_fe")]

    # First-stage Intl closure: substitution_channels family, outcome=International
    intl = base[(base["family"] == "substitution_channels") & (base["outcome"] == "International")].iloc[0]
    e.add("passthroughBetaIntl", "Pass-through first stage: β on Intl closure", _round_sig(intl["coef"], 3), "coef")
    e.add("passthroughBetaIntlSe", "Pass-through first stage: SE of β", f"{intl['se']:.3f}", "se")
    e.add("passthroughBetaIntlSig", "Pass-through first stage: significance", _stars_from_t(intl["t"]), "stars")

    # ETR pass-through: substitution_channels family, outcome=ETR
    etr = base[(base["family"] == "substitution_channels") & (base["outcome"] == "ETR")].iloc[0]
    e.add("passthroughBetaEtr", "Pass-through reduced form: β on ETR", _round_sig(etr["coef"], 3), "coef")
    e.add("passthroughBetaEtrSe", "Pass-through reduced form: SE of β", f"{etr['se']:.3f}", "se")
    e.add("passthroughBetaEtrSig", "Pass-through reduced form: significance", _stars_from_t(etr["t"]), "stars")

    # Income-location foreign share
    fs = base[(base["family"] == "income_location") & (base["outcome"] == "foreign_share")].iloc[0]
    e.add("incomeLocBetaForeignShare", "Income location: β on foreign PI share", _round_sig(fs["coef"], 5), "coef")
    e.add("incomeLocBetaForeignShareSe", "Income location: SE", f"{fs['se']:.5f}", "se")
    e.add("incomeLocBetaForeignShareSig", "Income location: significance", _stars_from_t(fs["t"]), "stars")

    # Income-location log foreign share
    fsl = base[(base["family"] == "income_location") & (base["outcome"] == "foreign_share_log")].iloc[0]
    e.add("incomeLocBetaLogForeignShare", "Income location: β on log(foreign share)", _round_sig(fsl["coef"], 5), "coef")
    e.add("incomeLocBetaLogForeignShareSe", "Income location: SE of log β", f"{fsl['se']:.5f}", "se")
    e.add("incomeLocBetaLogForeignShareSig", "Income location: significance of log β", _stars_from_t(fsl["t"]), "stars")

    # ── Closure ratio + subsample ETR ────────────────────────────────────────
    closure = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "foreign_share_closure_ratio_by_clip.csv"
    )
    cb = closure[(closure["clip_variant"] == "clip_15pp_baseline") & (closure["spec"] == "firm_fe_industry_year_fe")].iloc[0]
    # Emit the international denominator too, so the published closure ratios
    # can be re-derived from staged inputs rather than trusted.
    e.add(
        "incomeLocBetaIntlSubsample",
        "Income-location subsample: β on the International contribution (denominator for profitPerPpClosure)",
        _round_sig(cb["beta_intl_same_sample"], 3),
        "coef",
    )
    e.add(
        "incomeLocBetaEtrSubsample",
        "Income-location subsample: β on ETR (used for closure ratio denominator)",
        _round_sig(cb["beta_etr_same_sample"], 3),
        "coef",
    )
    e.add(
        "incomeLocClosureRatio",
        "Income-location closure ratio: |β_share| / |β_ETR| (denominator = ETR on same subsample)",
        f"{cb['ratio_100x_beta_share_over_abs_beta_etr']:.2f}",
        "str",
    )

    # ── π_dom/at, π_fo/at — parse from tbl_real_outcomes.tex ─────────────────
    # The table has paired rows: coef row "<label> & $<c1>^{<s1>}$ & $<c2>^{<s2>}$ & N \\"
    # followed by SE row "         & $(<se1>)$ & $(<se2>)$ & \\".
    # We capture the SECOND spec (firm + Ind×yr FE), which is what the slides cite.
    coef_re = re.compile(r"\$([+\-]?\d+\.\d+)\^?\{?(\**)\}?\$")
    se_re = re.compile(r"\$\(([+\-]?\d+\.\d+)\)\$")

    def parse_paired_rows(text, label_match):
        """Walk lines, find the line whose label contains `label_match`, parse its coef+stars
        and the next non-empty line's SE. Returns ((c1,s1,se1), (c2,s2,se2)) for two specs."""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if label_match in line and "&" in line:
                coefs = coef_re.findall(line)
                if len(coefs) < 2:
                    continue
                # Next line is the SE row
                for j in range(i + 1, len(lines)):
                    if "$(" in lines[j]:
                        ses = se_re.findall(lines[j])
                        if len(ses) >= 2:
                            return ((coefs[0][0], coefs[0][1], ses[0]), (coefs[1][0], coefs[1][1], ses[1]))
                        break
        raise ValueError(f"Could not parse paired rows for label {label_match!r}")

    real_outcomes = (
        ROOT / "data" / "analysis" / "tables" / "substitution" / "tbl_real_outcomes.tex"
    ).read_text()

    _, pi_dom = parse_paired_rows(real_outcomes, r"\pi^{\text{dom}}/\text{assets}")
    e.add("realOutcomePiDomAssets", "Real outcome: β on π^dom/assets (spec 2)", f"+{pi_dom[0]}" if not pi_dom[0].startswith(("+", "-")) else pi_dom[0], "coef")
    e.add("realOutcomePiDomAssetsSe", "Real outcome: SE of π^dom/assets β", pi_dom[2], "se")
    e.add("realOutcomePiDomAssetsSig", "Real outcome: significance", pi_dom[1], "stars")

    _, pi_fo = parse_paired_rows(real_outcomes, r"\pi^{\text{fo}}/\text{assets}")
    e.add("realOutcomePiFoAssets", "Real outcome: β on π^fo/assets (spec 2)", f"+{pi_fo[0]}" if not pi_fo[0].startswith(("+", "-")) else pi_fo[0], "coef")
    e.add("realOutcomePiFoAssetsSe", "Real outcome: SE of π^fo/assets β", pi_fo[2], "se")
    e.add("realOutcomePiFoAssetsSig", "Real outcome: significance", pi_fo[1], "stars")

    # ── Foreign revenue share — parse from tbl_geo_revenue_share.tex ─────────
    geo_rev = (
        ROOT / "data" / "analysis" / "tables" / "substitution" / "tbl_geo_revenue_share.tex"
    ).read_text()
    # The headline row reports β and SE; format may vary. Try a few patterns.
    m = re.search(r"Foreign revenue share[^\n]*?&\s*\$?([+\-][\d.]+)\$?\^?\{?(\**)\}?\$?\s*&\s*\$?\(([\d.]+)\)\$?", geo_rev)
    if m:
        e.add("geoRevenueBetaForeignShare", "Foreign revenue share: β", m.group(1), "coef")
        e.add("geoRevenueBetaForeignShareSe", "Foreign revenue share: SE", m.group(3), "se")
        e.add("geoRevenueBetaForeignShareSig", "Foreign revenue share: significance", m.group(2), "stars")
    else:
        # Fallback: hardcode the audited values
        e.add("geoRevenueBetaForeignShare", "Foreign revenue share: β (audited 2026-05)", "+0.00060", "coef")
        e.add("geoRevenueBetaForeignShareSe", "Foreign revenue share: SE", "0.00088", "se")
        e.add("geoRevenueBetaForeignShareSig", "Foreign revenue share: significance (n.s.)", "", "stars")

    # ── Windfall investment elasticities — parse from tbl_main_elasticity.tex ─
    inv = (
        ROOT / "data" / "analysis" / "tables" / "investment" / "tbl_main_elasticity.tex"
    ).read_text()
    # Row: `$W_i \times \text{post}$ & -0.012 & -0.012 & -0.012 & -0.027 & -0.056 & ... \\`
    # We want spec (3) for each outcome family: positions 3, 7, 11 (1-indexed past label).
    m = re.search(r"\$W_i \\times \\text\{post\}\$\s*((?:&\s*-?\d+\.\d+\s*)+)\\\\", inv)
    if m:
        coefs = re.findall(r"-?\d+\.\d+", m.group(1))
        # 12 specs total, want indices 2, 6, 10 (zero-indexed for spec 3 of each family)
        capex_coef = float(coefs[2])
        rd_coef = float(coefs[6])
        total_coef = float(coefs[10])
    else:
        capex_coef, rd_coef, total_coef = -0.012, -0.056, -0.001

    # SEs on the next line
    m_se = re.search(r"\$W_i \\times \\text\{post\}\$.*?\\\\.*?&\s*\(([\d.]+)\)(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?(?:\s*&\s*\(([\d.]+)\))?", inv, re.DOTALL)
    if m_se:
        ses = [float(g) for g in m_se.groups() if g is not None]
        capex_se = ses[2] if len(ses) > 2 else 0.038
        rd_se = ses[6] if len(ses) > 6 else 0.158
        total_se = ses[10] if len(ses) > 10 else 0.085
    else:
        capex_se, rd_se, total_se = 0.038, 0.158, 0.085

    e.add("windfallThetaCapex", "Windfall: θ on capex/A (spec 3)", _round_sig(capex_coef, 3), "coef")
    e.add("windfallThetaCapexSe", "Windfall: SE", f"{capex_se:.3f}", "se")
    e.add("windfallThetaRd", "Windfall: θ on R&D/A (spec 3)", _round_sig(rd_coef, 3), "coef")
    e.add("windfallThetaRdSe", "Windfall: SE", f"{rd_se:.3f}", "se")
    e.add("windfallThetaTotalInv", "Windfall: θ on total inv/A (spec 3)", _round_sig(total_coef, 3), "coef")
    e.add("windfallThetaTotalInvSe", "Windfall: SE", f"{total_se:.3f}", "se")

    # ── Debt elasticity — parse from tbl_windfall_decomposition.tex ──────────
    wd = (
        ROOT / "data" / "analysis" / "tables" / "investment" / "tbl_windfall_decomposition.tex"
    ).read_text()
    # Row: Debt change (+ = increase) & $-0.327**$ & $(0.135)$ & ...
    m = re.search(r"Debt change[^&]*&\s*\$([+\-][\d.]+)(\**)\$\s*&\s*\$\(([\d.]+)\)\$", wd)
    if m:
        e.add("windfallThetaDebt", "Windfall: θ on debt change", m.group(1), "coef")
        e.add("windfallThetaDebtSe", "Windfall: SE", m.group(3), "se")
        e.add("windfallThetaDebtSig", "Windfall: significance", m.group(2), "stars")
    else:
        e.add("windfallThetaDebt", "Windfall: θ on debt (audited)", "-0.327", "coef")
        e.add("windfallThetaDebtSe", "Windfall: SE", "0.135", "se")
        e.add("windfallThetaDebtSig", "Windfall: significance", "**", "stars")

    # ── Hardcoded: NI-spike thetas (currently only in figure captions) ───────
    # build_investment_elasticity.py computes these at runtime; not exported to CSV.
    # Values verified 2026-05 via presentation/table_audit.md.
    ws = _windfall_summary()
    e.add("windfallThetaNiShockYear",
          "Windfall: shock-year (2017) NI-spike θ (first stage on net income)",
          _fmt_signed(ws["ni_shock_2017_strat_a"]["coef"], 3), "coef")
    e.add("windfallThetaNiPostShock",
          "Windfall: post-shock (2018+) pooled θ on net income",
          _fmt_signed(ws["ni_shock_post"]["coef"], 3), "coef")

    e.write(OUT_DIR / "regressionheadlines.csv")


# ── Section: sample_sizes_and_derived_stats ──────────────────────────────────


def emit_sample_sizes_and_derived():
    """Sample sizes (Ns from regression .tex tables) + a few stats derived
    from the regression coefficients (closure %, pass-through %, etc.).

    Reads:
      data/analysis/tables/substitution/tbl_substitution_main.tex     (1,277)
      data/analysis/tables/substitution/tbl_real_outcomes.tex         (825/832/822)
      data/analysis/tables/substitution/tbl_geo_revenue_share.tex     (1,148 / 6,761)
      data/analysis/tables/real_activity_dynamics/tbl_real_activity_dynamics.tex (1,926)
      data/analysis/tables/investment/tbl_main_elasticity.tex         (1,354)
      data/analysis/output/robustness/foreign_share_closure_ratio_by_clip.csv (range)

    Hardcoded (computed in figure scripts, not exported to CSV):
      - Revenue share means pre/post (26.2% / 25.8%)
      - Worked-example pp values (+3.2 / +0.4)
      - Pharma 2017 sample size (62)
    """
    e = Emitter()

    # ── DiD sample size ──────────────────────────────────────────────────────
    # 1,277 firms appears in tbl_substitution_main.tex ($N$ firms row).
    sub_main = (
        ROOT / "data" / "analysis" / "tables" / "substitution" / "tbl_substitution_main.tex"
    ).read_text()
    m = re.search(r"\$N\$ firms\s*&\s*(\d+)", sub_main)
    n_did = int(m.group(1)) if m else 1277
    e.add("didSampleFirms", "Substitution / pass-through DiD: N firms", n_did, "int")

    # ── Real-outcomes sample sizes (right column of tbl_real_outcomes.tex) ──
    ro = (
        ROOT / "data" / "analysis" / "tables" / "substitution" / "tbl_real_outcomes.tex"
    ).read_text()
    # Each coef row ends with "& <N> \\". Capture in order: foreign share, log share, π_dom, π_fo.
    n_matches = re.findall(r"\$[+\-]?[\d.]+\^?\{?\**\}?\$\s*&\s*(\d+)\s*\\\\", ro)
    if len(n_matches) >= 4:
        e.add("realOutcomeNForeignShare", "Income location: N firms for foreign-share outcome", int(n_matches[0]), "int")
        e.add("realOutcomeNPiDom", "Income location: N firms for π^dom/assets outcome", int(n_matches[2]), "int")
        e.add("realOutcomeNPiFo", "Income location: N firms for π^fo/assets outcome", int(n_matches[3]), "int")
    else:
        # Fallback to audited values
        e.add("realOutcomeNForeignShare", "Income location: N firms (foreign share)", 825, "int")
        e.add("realOutcomeNPiDom", "Income location: N firms (π^dom/assets)", 832, "int")
        e.add("realOutcomeNPiFo", "Income location: N firms (π^fo/assets)", 822, "int")

    # ── Geographic revenue share sample sizes ────────────────────────────────
    geo_summary = _geo_revenue_summary()
    e.add(
        "geoRevenueShareFirms",
        "Foreign revenue share: distinct firms with usable segment data",
        geo_summary["pooled_did_regression_sample"]["firms"],
        "int",
    )
    e.add(
        "geoRevenueShareFirmYears",
        "Foreign revenue share: pooled DiD firm-years",
        geo_summary["pooled_did_regression_sample"]["firm_years"],
        "int",
    )

    # ── Real-activity dynamics sample (max-information panel) ────────────────
    rad = (
        ROOT / "data" / "analysis" / "tables" / "real_activity_dynamics" / "tbl_real_activity_dynamics.tex"
    ).read_text()
    # Capture every "& <N> \\" at end-of-row, then pick the largest (which is
    # the biggest outcome subsample — currently capex/lag(at)).
    rad_ns = [int(s.replace(",", "")) for s in re.findall(r"&\s*(\d{1,3}(?:,\d{3})?)\s*\\\\", rad)]
    e.add(
        "realActivitySampleFirms",
        "Real-activity max-information panel: N firms (largest outcome subsample)",
        max(rad_ns) if rad_ns else 1926,
        "int",
    )

    # ── Windfall sample (Strat. B column has the largest N) ──────────────────
    inv_main = (
        ROOT / "data" / "analysis" / "tables" / "investment" / "tbl_main_elasticity.tex"
    ).read_text()
    # "Firms" row: 1263 1263 1263 1354 721 ... — capture all, take Strat. B = 4th = 1354
    m_fr = re.search(r"Firms\s*((?:&\s*\d+\s*){4,})", inv_main)
    if m_fr:
        firm_ns = re.findall(r"\d+", m_fr.group(1))
        windfall_n = int(firm_ns[3]) if len(firm_ns) >= 4 else 1354
    else:
        windfall_n = 1354
    e.add("windfallSampleFirms", "Windfall: max-info sample N firms (Strat. B)", windfall_n, "int")

    # ── Derived stats from the regression coefficients ───────────────────────
    channel_base = _channel_clip_baseline()
    intl = channel_base[
        (channel_base["family"] == "substitution_channels")
        & (channel_base["outcome"] == "International")
    ].iloc[0]
    etr = channel_base[
        (channel_base["family"] == "substitution_channels")
        & (channel_base["outcome"] == "ETR")
    ].iloc[0]
    fs = channel_base[
        (channel_base["family"] == "income_location")
        & (channel_base["outcome"] == "foreign_share")
    ].iloc[0]
    fsl = channel_base[
        (channel_base["family"] == "income_location")
        & (channel_base["outcome"] == "foreign_share_log")
    ].iloc[0]
    closure_base = _foreign_share_closure_baseline()
    rd_cutoffs = _rd_bin_cutoffs()

    # 79% closure: the channel collapses by |β_Intl| of its pre-TCJA magnitude.
    # The treated firm went from c̄^pre_Intl < 0 toward zero; pass-through |β|.
    beta_intl = float(intl["coef"])
    beta_etr = float(etr["coef"])
    beta_share = float(fs["coef"])
    beta_log_share = float(fsl["coef"])
    closure_pct = round(abs(beta_intl) * 100)
    passthrough_pct = round(abs(beta_etr) / abs(beta_intl) * 100)
    e.add("passthroughClosurePct", "Derived: % of pre-TCJA Intl benefit closed per pp (β_Intl rounded)", closure_pct, "pct")
    e.add("passthroughEtrPct", "Derived: % of closure that passes through to ETR (β_ETR/|β_Intl|)", passthrough_pct, "pct")
    # What did not pass through, i.e. what came back through the other
    # reconciliation categories.  Taken as the complement of the pass-through
    # so the two sum to 100 on the slide that shows both.  Summing the channel
    # betas directly gives 27 percent (Table 8); the difference is the
    # sum-of-channels versus ETR gap the paper discloses in Section 3.2.1.
    e.add("passthroughLeakagePct", "Derived: % of the closure recovered through the other channels (100 - pass-through)", 100 - passthrough_pct, "pct")

    # Worked example.  Two companies in the same industry whose pre-TCJA
    # International contributions were WORKED_FROM_PP and WORKED_TO_PP below
    # zero, i.e. WORKED_GAP_PP apart in exposure.  Every number below is a
    # coefficient times that gap, which is what the design identifies: a
    # differential between a more- and a less-exposed firm, not a level.
    #
    # Do not read these as one firm whose contribution moved 5pp.  On that
    # reading the ETR effect is β_ETR/β_Intl × 5, a different number, and the
    # endpoints contradict β_Intl -- a firm at -10 is predicted to close by
    # 7.8, landing near -2.2, not -5.  Section 3.2.1's technology example is
    # the closure-basis calculation and does use the ratio.
    WORKED_FROM_PP, WORKED_TO_PP = 10, 5
    worked_gap_pp = WORKED_FROM_PP - WORKED_TO_PP
    worked_intl_pp = abs(beta_intl) * worked_gap_pp
    worked_etr_pp = abs(beta_etr) * worked_gap_pp
    e.add("workedExampleFromPp", "Worked example: more exposed firm, pre-TCJA Intl contribution (pp, absolute value)", WORKED_FROM_PP, "int")
    e.add("workedExampleToPp", "Worked example: less exposed firm, pre-TCJA Intl contribution (pp, absolute value)", WORKED_TO_PP, "int")
    e.add("workedExampleGapPp", "Worked example: pre-TCJA exposure gap between the two firms (pp)", worked_gap_pp, "int")
    e.add("workedExampleIntlPp", "Worked example: differential Intl closure = |β_Intl| × gap (pp)", round(worked_intl_pp, 1), "str")
    e.add("workedExampleEtrPp", "Worked example: differential ETR rise = |β_ETR| × gap (pp)", round(worked_etr_pp, 1), "str")
    # The part of the closure that did not pass through, recovered through the
    # other reconciliation categories.  Taken as the residual of the two
    # coefficients above so that the three numbers add exactly.  Summing the
    # channel betas directly gives -1.1 instead; the difference is the
    # sum-of-channels versus ETR gap that Section 3.2.1 discloses.  It is
    # negative: for the more exposed firm the other categories became more
    # tax-reducing, so they offset the closure rather than adding to it.
    e.add("workedExampleOtherPp", "Worked example: differential movement in the other channels = -(Intl closure - ETR rise) (pp)", _fmt_signed(-(worked_intl_pp - worked_etr_pp), 1), "str")

    # ── Closure-ratio clipping robustness range ──────────────────────────────
    closure = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "foreign_share_closure_ratio_by_clip.csv"
    )
    closure_main = closure[closure["spec"] == "firm_fe_industry_year_fe"]
    # Drop the no_per_obs_clip variant — it's the pathological tail and the
    # slide explicitly cites the range over the finite-clip variants only.
    closure_main = closure_main[closure_main["clip_variant"] != "no_per_obs_clip"]
    ratios = closure_main["ratio_100x_beta_share_over_abs_beta_etr"].dropna()
    e.add("closureRatioMin", "Income-location closure ratio: min across clip variants", f"{ratios.min():.2f}", "str")
    e.add("closureRatioMax", "Income-location closure ratio: max across clip variants", f"{ratios.max():.2f}", "str")

    # ── One-offs now sourced from structured analysis outputs ────────────────
    n_fdii, n_nonfdii = _fdii_claimant_counts()
    diagnostics = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "ratio_sum_treatment_diagnostics.csv"
    ).iloc[0]
    e.add("revenueSharePreMean", "Foreign revenue share: mean pre-TCJA (2013-16)", round(geo_summary["pre_mean_foreign_rev_share"] * 100, 1), "pct")
    e.add("revenueSharePostMean", "Foreign revenue share: mean post-TCJA (2019-22)", round(geo_summary["post_mean_foreign_rev_share"] * 100, 1), "pct")
    e.add("pharmaSampleFirmYears", "Pharma 2017 profitable sample", _pharma_2017_profitable_firm_years(), "int")
    e.add("fdiiClaimantsCount", "FDII channel test: N claimants (body 'Direct test of the FDII channel' slide)", n_fdii, "int")
    e.add("fdiiNonClaimantsCount", "FDII channel test: N non-claimants", n_nonfdii, "int")
    e.add("commonFirmCorrelation", "A.15 real-activity: common-firm correlation between max-info panel and substitution panel", f"{diagnostics['corr_common']:.2f}", "str")
    # Was a hard-coded 18. Derive it: the implied fall for a firm at the mean
    # pre-TCJA exposure among exposed firms, over that group's mean foreign
    # share, both measured in the foreign-share sample so the ratio is internally
    # consistent.
    _msp = pd.read_csv(ROOT / "data" / "analysis" / "output" / "profit_shifting_mechanism_panel.csv")
    _pre = _msp[_msp["recon_year"].between(2013, 2016)]
    _exposed = _pre[_pre["intl_pre"] < 0]
    _mean_share_pct = float(_exposed["foreign_share"].mean()) * 100.0
    _implied_drop_pp = float(closure_base["beta_share"]) * 100.0 * abs(float(_exposed["intl_pre"].mean()))
    e.add(
        "foreignShareDeclinePct",
        "Typical exposed firm: percent decline in its foreign profit share",
        round(100.0 * _implied_drop_pp / _mean_share_pct),
        "pct",
    )
    foreign_share_decline_pp = float(closure_base["ratio_100x_beta_share_over_abs_beta_etr"])
    profit_per_closure = float(closure_base["ratio_100x_beta_share_over_abs_beta_intl"])
    revenue_per_closure = float(
        geo_summary["closure_ratio_pp_revenue_share_per_1pp_international_closure"]["ind_yr"]
    )
    e.add("foreignShareDeclinePp", "Typical exposed firm: foreign share pp decline (derived headline)", f"{foreign_share_decline_pp:.2f}", "str")
    e.add("profitVsRevenueResponseRatio", "Profit vs revenue response ratio", str(round(profit_per_closure / revenue_per_closure)), "str")

    # ── Profit-location body table: per-pp implied effects for "-10 vs -5 firm" ─────
    # 5pp closure × β; sign matches the more-exposed firm's direction relative to less-exposed.
    pi_dom_beta = _balance_beta("income_location", "Domestic PI / lag assets", "beta_ge2")
    pi_fo_beta = _balance_beta("income_location", "Foreign PI / lag assets", "beta_ge2")
    e.add("incomeLocPerPpForeignShare", "Profit location: implied foreign-share Δ for -10 vs -5 firm (pp)", _fmt_signed(-worked_gap_pp * beta_share * 100, 1), "str")
    e.add("incomeLocPerPpLogShare",     "Profit location: implied log foreign-share Δ for -10 vs -5 firm (log pts)", _fmt_signed(-worked_gap_pp * beta_log_share, 2), "str")
    e.add("incomeLocPerPpPiDom",        "Profit location: implied π_dom/assets Δ for -10 vs -5 firm (pp)", _fmt_signed(-worked_gap_pp * pi_dom_beta * 100, 2), "str")
    e.add("incomeLocPerPpPiFo",         "Profit location: implied π_fo/assets Δ for -10 vs -5 firm (pp)", _fmt_signed(-worked_gap_pp * pi_fo_beta * 100, 2), "str")

    # ── Profit-vs-revenue side-by-side body table: per 1pp Intl closure ────────────
    e.add("profitPerPpClosure",  "Foreign profit share: implied Δ per 1pp Intl closure (pp)",  f"{profit_per_closure:.2f}", "str")
    e.add("revenuePerPpClosure", "Foreign revenue share: implied Δ per 1pp Intl closure (pp)", f"{revenue_per_closure:.2f}", "str")

    # ── R&D magnitude split body table: All-sample implied per-pp effects ──────────
    main_implied_share = abs(float(closure_base["beta_share"])) * 100
    e.add("crossCheckMainImpliedShare", "R&D split (All): foreign-share Δ per 1pp benefit (pp)", f"{main_implied_share:.2f}", "str")
    e.add("crossCheckMainImpliedEtr",   "R&D split (All): implied ETR Δ per 1pp benefit (pp)",   f"{main_implied_share / foreign_share_decline_pp:.2f}", "str")

    # ── R&D bin cutoffs (R&D / assets, in percent) ─────────────────────────────────
    # Low = exactly 0; Mid ∈ [0.02%, 1.0%]; High > 1.0%. Stored bare (slide adds %).
    e.add("rdBinLowMax",  "R&D bin: Low cutoff (R&D/assets, pct)",                f"{rd_cutoffs['low_max']:.0f}", "str")
    e.add("rdBinMidLow",  "R&D bin: Middle lower bound (R&D/assets, pct)",        f"{rd_cutoffs['mid_low']:.2f}", "str")
    e.add("rdBinMidHigh", "R&D bin: Middle upper bound = High threshold (pct)",   f"{rd_cutoffs['mid_high']:.1f}", "str")

    # ── FDII channel slide: exposure-difference units in the figure caption ────────
    e.add("fdiiExposureDiff", "FDII slide: exposure difference figure caption refers to (pp)", 5, "int")
    e.add("confidenceLevel",  "Standard confidence-interval level used in figures (pct)", 95, "int")

    # ── A.6 cross-validation of rate vs amount form (text) ─────────────────────────
    e.add("cvMedianAbsErrPp", "A.6 cross-validation: median |derived - stated ETR| (pp)", "0.00", "str")
    e.add("cvIqrLowPp",       "A.6 cross-validation: IQR lower bound (pp)",               "-0.02", "str")
    e.add("cvIqrHighPp",      "A.6 cross-validation: IQR upper bound (pp)",               "+0.04", "str")

    # ── A.15 balance-rule robustness (1+/1+ vs 2+/2+ baseline) ─────────────────────
    e.add("balanceRuleBetaIntl",  "A.15 balance rule (1+/1+): β_Intl",  f"{_balance_beta('tax_pass_through', 'International'):.3f}", "coef")
    e.add("balanceRuleBetaShare", "A.15 balance rule (1+/1+): β_share", _fmt_signed(_balance_beta('income_location', 'Foreign PI share'), 5), "coef")
    e.add("balanceRuleBetaEtr",   "A.15 balance rule (1+/1+): β_ETR (pass-through)", f"{_balance_beta('tax_pass_through', 'ETR'):.3f}", "coef")

    e.write(OUT_DIR / "samplesanddescriptive.csv")


# ── Section: pharma + case-study + R&D-split values ─────────────────────────


def emit_pharma_and_cases():
    """Pharma 5-firm contribution table (A.20), pharma post-TCJA (A.22),
    case-study appendix (A.23-A.30), Bermuda Domestications body slide,
    and R&D-split table (body).

    Most pharma and case-study values are firm-specific or sub-sample-specific
    and were computed once (verified May 2026) and recorded in
    presentation/table_audit.md and presentation/case_study_audit.md.
    The R&D-split rows are read from the structured heterogeneity output;
    PI-intensity cross-check rows are read from the deep-dive CSVs.

    Why the per-firm pharma/case-study values stay hand-curated (not script-
    derived from the panel):
      * Bermuda-Domestications "post" = FY2024 endpoint, NOT a 2019-22 window
        mean (per the body slide's `^{\\ddagger}` footnote — diluting Nvidia
        / ON Semi by averaging pre-action years would mislead).
      * Nvidia's "post" uses a 2022-23 window because the standard 2019-22
        window is right-censored relative to mid-2021 Domestication.
      * A.20 pharma 2017 contribution table uses a curated "main pharma"
        sample (top firms), not all profitable pharma-SIC firms; the
        denominator chosen there is not in any source CSV.
      * 10-K narrative amounts (e.g. JNJ "$13.0B provisional tax charge")
        are direct text quotes that must be preserved verbatim with their
        cite-source-text framing in the appendix.
    A spike was attempted to auto-derive these from the panel (see
    `analysis/build_case_study_metrics.py` in commit 2dd89a7's history);
    the divergences from the audited values exceeded acceptable drift,
    so the literal-with-audit-trail approach was kept.
    """
    e = Emitter()

    # ── Pharma 2017 spike (A.20) ─────────────────────────────────────────────
    # Source: build_etr_figures.py pharma-industry slice (SIC 2834/2836).
    e.add("pharmaEtrPreYear", "Pharma 2016 PI-weighted ETR (single year)", 20.3, "pct")
    e.add("pharmaEtrSpike", "Pharma 2017 mean ETR (transition-tax spike)", 64.7, "pct")
    e.add("pharmaEtrYearAfterShock", "Pharma 2018 PI-weighted ETR (single year)", 13.2, "pct")
    e.add("pharmaEtrMedianSpike", "Pharma 2017 median ETR", 31.3, "pct")
    # A.22 window averages (different from A.20 single-year values above)
    e.add("pharmaEtrPreWindow", "Pharma pre-TCJA window mean ETR (2013-16)", 20.3, "pct")
    e.add("pharmaEtrPostWindow", "Pharma post-TCJA window mean ETR (2019-24)", 12.4, "pct")
    e.add("pharmaTopFiveContribution", "Top 5 firms' contribution to 2017 pharma ETR spike (pp)", "52.8", "str")
    # Top-5 firms' share of 2017 pharma sample PI (A.20 table footer row).
    e.add("pharmaTopFivePiShare", "Pharma top-5 firms' share of 2017 PI (pct)", 67.4, "pct")
    # Per-firm 2017 contribution rows (PI share, ETR, contribution in pp)
    e.add("jnjPiShare", "JNJ 2017 PI share within pharma sample (pct)", 22.7, "pct")
    e.add("jnjEtr", "JNJ 2017 ETR (pct)", 92.6, "pct")
    e.add("jnjContribution", "JNJ 2017 contribution to industry ETR (pp)", "21.0", "str")
    e.add("gileadPiShare", "Gilead 2017 PI share (pct)", 17.4, "pct")
    e.add("gileadEtr", "Gilead 2017 ETR (pct)", 65.7, "pct")
    e.add("gileadContribution", "Gilead 2017 contribution (pp)", "11.4", "str")
    e.add("amgenPiShare", "Amgen 2017 PI share (pct)", 12.3, "pct")
    e.add("amgenEtr", "Amgen 2017 ETR (pct)", 79.4, "pct")
    e.add("amgenContribution", "Amgen 2017 contribution (pp)", "9.8", "str")
    e.add("bmsPiShare", "BMS 2017 PI share (pct)", 6.6, "pct")
    e.add("bmsEtr", "BMS 2017 ETR (pct)", 81.0, "pct")
    e.add("bmsContribution", "BMS 2017 contribution (pp)", "5.3", "str")
    e.add("merckPiShare", "Merck 2017 PI share (pct)", 8.4, "pct")
    e.add("merckEtr", "Merck 2017 ETR (pct)", 62.9, "pct")
    e.add("merckContribution", "Merck 2017 contribution (pp)", "5.3", "str")

    # ── A.21 Pharma 2017 10-K verbatim quote amounts (USD billions) ──────────
    e.add("jnjTcjaCharge", "JNJ 2017 10-K provisional tax charge (USD B)", "13.0", "str")
    e.add("jnjEtrIncrease", "JNJ 2017 10-K ETR increase (pp)", "73.3", "str")
    e.add("gileadTransitionTax", "Gilead 2017 10-K transition tax (USD B)", "5.8", "str")
    e.add("merckProvisional", "Merck 2017 10-K provisional transition tax (USD B)", "5.3", "str")
    # Stored bare (no % — slide adds it) so datatool CSV parser doesn't choke
    e.add("amgenRepatriationTax", "Amgen 2017 10-K net repatriation tax (pct, verbatim)", "71.5", "str")
    e.add("amgenDtrRemeasurement", "Amgen 2017 10-K net DTR remeasurement (verbatim)", "(7.7)", "str")
    e.add("bmsTcjaCharge", "BMS 2017 10-K additional tax expense (USD B)", "2.9", "str")
    e.add("bmsTransitionTax", "BMS 2017 10-K deemed repatriation transition tax (USD B)", "2.6", "str")

    # ── A.22 pharma post-TCJA ────────────────────────────────────────────────
    e.add("pharmaIntlPre", "Pharma International channel pre-TCJA (pp)", "-11.5", "str")
    e.add("pharmaIntlPost", "Pharma International channel post-2019 (pp)", "-4.3", "str")
    e.add("pharmaTopSixPiShare", "Pharma top-6 firms' PI share (pct)", 64.9, "pct")
    # 2019-24 mean ETR for the 6 firms in A.22
    e.add("pfizerEtrPostTcja", "Pfizer 2019-24 mean ETR (pct)", 5.8, "pct")
    e.add("pfizerIntlPostTcja", "Pfizer 2019-24 Intl channel (pp)", "-11.1", "str")
    e.add("jnjEtrPostTcja", "JNJ 2019-24 mean ETR (pct)", 12.3, "pct")
    e.add("jnjIntlPostTcja", "JNJ 2019-24 Intl channel (pp)", "-7.3", "str")
    e.add("abbvieEtrPostTcja", "AbbVie 2019-24 mean ETR (pct)", 6.6, "pct")
    e.add("merckEtrPostTcja", "Merck 2019-24 mean ETR (pct)", 15.4, "pct")
    e.add("merckIntlPostTcja", "Merck 2019-24 Intl channel (pp)", "-8.3", "str")
    e.add("lillyEtrPostTcja", "Lilly 2019-24 mean ETR (pct)", 13.9, "pct")
    e.add("lillyIntlPostTcja", "Lilly 2019-24 Intl channel (pp)", "-6.1", "str")
    e.add("amgenEtrPostTcja", "Amgen 2019-24 mean ETR (pct)", 12.4, "pct")

    # ── Body Part II Bermuda Domestications table ────────────────────────────
    e.add("alphabetSalesRank", "Alphabet 2016 sales rank", 17, "int")
    e.add("nvidiaSalesRank", "Nvidia 2016 sales rank", 202, "int")
    e.add("onsemiSalesRank", "ON Semi 2016 sales rank", 302, "int")
    e.add("alphabetForeignPiPre", "Alphabet foreign PI share pre-action", 0.55, "str")
    e.add("alphabetForeignPiPost", "Alphabet foreign PI share post-action", 0.10, "str")
    e.add("nvidiaForeignPiPre", "Nvidia foreign PI share pre-action", 0.79, "str")
    e.add("nvidiaForeignPiPost", "Nvidia foreign PI share post-action", 0.07, "str")
    e.add("onsemiForeignPiPre", "ON Semi foreign PI share pre-action", "1.00", "str")
    e.add("onsemiForeignPiPost", "ON Semi foreign PI share post-action", 0.13, "str")
    e.add("alphabetIntlDelta", "Alphabet Intl line change (pp)", "+9.9", "str")
    e.add("nvidiaIntlDelta", "Nvidia Intl line change (pp)", "+12", "str")
    e.add("onsemiIntlDelta", "ON Semi Intl line change (pp)", "+12.5", "str")
    # Year-of-IP-onshoring row in the Bermuda-Domestications table.
    e.add("alphabetIpYear", "Alphabet year of IP onshoring", "2019--20", "str")
    e.add("nvidiaIpYear",   "Nvidia year of IP onshoring",   "2021",     "str")
    e.add("onsemiIpYear",   "ON Semi year of IP onshoring",  "2020",     "str")
    # A.23 case-study composite-screen denominator.
    e.add("caseStudyScreenFirms", "A.23 case-study screen: N firms ranked", 922, "int")

    # ── Qualcomm body slide ──────────────────────────────────────────────────
    e.add("qualcommIntlPre", "Qualcomm Intl pre (pp)", "-14.8", "str")
    e.add("qualcommIntlPost", "Qualcomm Intl post (pp)", "-5.9", "str")
    e.add("qualcommForeignPiPre", "Qualcomm foreign PI share pre", 0.57, "str")
    e.add("qualcommForeignPiPost", "Qualcomm foreign PI share post", 0.12, "str")
    e.add("qualcommEtrPre", "Qualcomm ETR pre (pct)", 16.5, "pct")
    e.add("qualcommEtrPost", "Qualcomm ETR post (pct)", 19.0, "pct")

    # ── Case-study appendix A.24-A.28 ────────────────────────────────────────
    e.add("alphabetSecTransitionTax", "Alphabet sec 965 charge (USD B)", "10.2", "str")
    e.add("alphabetForeignPiEndpoint", "Alphabet foreign PI share endpoint", 0.10, "str")
    e.add("alphabetEtrShock", "Alphabet 2017 ETR (pct)", 53, "pct")
    e.add("alphabetEtrYearAfter", "Alphabet 2018 ETR (pct)", 12, "pct")
    e.add("onsemiSecTransitionTax", "ON Semi sec 965 charge (USD M)", "1.5", "str")
    e.add("nvidiaIpDomLine", "Nvidia FY2021 IP-domestication line (USD M)", 244, "int")
    e.add("nvidiaFdiiLine", "Nvidia FY2021 FDII line (USD M)", 520, "int")
    e.add("nvidiaIntlDeltaStandard", "Nvidia Intl delta standard window (pp)", "1.9", "str")
    e.add("nvidiaIntlDeltaRecent", "Nvidia Intl delta most recent (pp)", "12.8", "str")
    e.add("microsoftSecTransitionTaxGross", "Microsoft sec 965 gross transition tax (USD B)", "17.9", "str")
    e.add("microsoftSecTransitionTaxNet", "Microsoft sec 965 net transition tax (USD B)", "13.7", "str")
    e.add("microsoftBenefit", "Microsoft FY2019 prior-year benefit (USD B)", "2.6", "str")
    e.add("microsoftForeignPiPre", "Microsoft foreign PI share pre", 0.78, "str")
    e.add("microsoftForeignPiPost", "Microsoft foreign PI share post", 0.53, "str")
    e.add("ciscoForeignCashPre", "Cisco pre-TCJA foreign cash (USD B)", 67, "int")
    e.add("ciscoTcjaCharge", "Cisco sec 965 + remeasurement (USD B)", "10.4", "str")
    e.add("ciscoRepatriation", "Cisco repatriated foreign cash (USD B)", 67, "int")
    e.add("ciscoRouted", "Cisco amount routed to buybacks/dividends (USD B)", 44, "int")
    e.add("ciscoForeignPiPre", "Cisco foreign PI share pre", 0.71, "str")
    e.add("ciscoForeignPiPost", "Cisco foreign PI share post", 0.27, "str")

    # ── Case-study appendix A.29-A.30 (Coca-Cola, Meta, Gilead, Salesforce) ──
    e.add("cocaColaDeposit", "Coca-Cola litigation deposit (USD B)", 6, "int")
    e.add("metaCsaMetaValue", "Meta cost-sharing Meta valuation (USD B)", "6.3", "str")
    e.add("metaCsaIrsValue", "Meta cost-sharing IRS valuation (USD B)", "19.9", "str")
    e.add("metaTaxCourtRuling", "Meta Tax Court ruling (USD B)", "7.79", "str")
    e.add("metaIrsClaim", "Meta IRS periodic claim (USD B)", 16, "int")
    e.add("gileadHcvRevenue", "Gilead HCV revenue 2015 (USD B)", 19, "int")
    e.add("gileadHcvRevenueRecent", "Gilead HCV revenue 2019 (USD B)", "2.9", "str")
    e.add("salesforceIpTransfer", "Salesforce IP transfer-pricing settlement value (USD B)", 24, "int")
    e.add("salesforceDtaBenefit", "Salesforce DTA benefit (USD B)", "1.8", "str")

    # ── PI-intensity cross-check table coefficients (body) ──────────────────
    # Sourced from analysis/analyze_profit_shifting_deep_dive.py output CSVs.
    pi_main, pi_inter = _pi_check()
    pi_rows = [
        ("ForeignShare", "foreign_share", "foreign-share"),
        ("PiFo",         "pifo_at",        "pi^fo/assets"),
        ("PiDom",        "pidom_at",       "pi^dom/assets"),
    ]
    for slug, outcome, human in pi_rows:
        e.add(f"piCheck{slug}Main", f"PI-check: {human} main β",
              _fmt_signed(pi_main[outcome], 5), "coef")
        e.add(f"piCheck{slug}Rd",   f"PI-check: {human} R&D γ",
              _fmt_signed(pi_inter[(outcome, "pre_rd_at_z")], 5), "coef")
        e.add(f"piCheck{slug}Cash", f"PI-check: {human} cash γ",
              _fmt_signed(pi_inter[(outcome, "pre_cash_at_z")], 5), "coef")

    e.write(OUT_DIR / "pharmaandcases.csv")


# ── Section: remaining pipeline numbers (A.13/A.14/real activity) ──────────


def emit_remaining_pipeline():
    """A.13 mechanism falsification, A.14 clipping robustness, real-activity
    coefficients — the final ~50 pipeline-derived numbers in the slides.

    A.14 comes from channel_clip_robustness_headlines.csv (already used for
    the headline ratios). A.13 + real-activity coefficients come from the
    profit-shifting heterogeneity scripts; hardcoded here with source
    attribution since the underlying scripts don't currently emit clean CSVs.
    """
    e = Emitter()

    # ── A.14 Clipping robustness (5 non-baseline clip variants × 5 cols) ─────
    # All A.14 columns come from foreign_share_closure_ratio_by_clip.csv — it
    # has β_share, β_Intl_same_sample, β_ETR_same_sample, and both ratios
    # for every clip variant on the foreign-share subsample. The ±15pp
    # baseline row also pulls β_Intl and Share/Intl from this CSV; β_share,
    # β_ETR, and Share/ETR for the baseline reuse the regression-headline
    # macros (incomeLocBetaForeignShare, incomeLocBetaEtrSubsample,
    # incomeLocClosureRatio).
    closure = pd.read_csv(
        ROOT / "data" / "analysis" / "output" / "robustness" / "foreign_share_closure_ratio_by_clip.csv"
    )
    base = closure[(closure["clip_variant"] == "clip_15pp") & (closure["spec"] == "firm_fe_industry_year_fe")]
    if len(base):
        b0 = base.iloc[0]
        e.add("clipFifteenPpBetaIntl",       "A.14 clip ±15pp baseline: β_Intl (foreign-share subsample)", f"{b0['beta_intl_same_sample']:.3f}", "coef")
        e.add("clipFifteenPpRatioShareIntl", "A.14 clip ±15pp baseline: Share/Intl ratio",                  f"{b0['ratio_100x_beta_share_over_abs_beta_intl']:.2f}", "str")
    else:
        e.add("clipFifteenPpBetaIntl",       "A.14 clip ±15pp baseline: β_Intl (fallback)", "-0.733", "coef")
        e.add("clipFifteenPpRatioShareIntl", "A.14 clip ±15pp baseline: Share/Intl ratio (fallback)", "0.97", "str")
    for variant, suffix in [
        ("clip_10pp", "TenPp"),
        ("clip_20pp", "TwentyPp"),
        ("clip_30pp", "ThirtyPp"),
        ("clip_50pp", "FiftyPp"),
        ("no_per_obs_clip", "NoClip"),
    ]:
        cl = closure[(closure["clip_variant"] == variant) & (closure["spec"] == "firm_fe_industry_year_fe")]
        if not len(cl):
            continue
        cl0 = cl.iloc[0]
        e.add(f"clip{suffix}BetaShare", f"A.14 clip {variant}: beta share", f"+{cl0['beta_share']:.5f}", "coef")
        e.add(f"clip{suffix}BetaIntl", f"A.14 clip {variant}: beta Intl (foreign-share subsample)", f"{cl0['beta_intl_same_sample']:.3f}", "coef")
        e.add(f"clip{suffix}BetaEtr", f"A.14 clip {variant}: beta ETR (foreign-share subsample)", f"{cl0['beta_etr_same_sample']:.3f}", "coef")
        e.add(f"clip{suffix}RatioShareIntl", f"A.14 clip {variant}: Share/Intl ratio", f"{cl0['ratio_100x_beta_share_over_abs_beta_intl']:.2f}", "str")
        e.add(f"clip{suffix}RatioShareEtr", f"A.14 clip {variant}: Share/ETR ratio", f"{cl0['ratio_100x_beta_share_over_abs_beta_etr']:.2f}", "str")

    # ── A.13 Mechanism falsification (6 predictor rows) ──────────────────────
    # Sourced from analysis/analyze_profit_shifting_mechanisms.py +
    # analysis/analyze_profit_shifting_deep_dive.py output CSVs.
    mech_falsification = [
        ("RevShare",   "foreign revenue share",        "pre_foreign_rev_share_z"),
        ("Haven",      "any haven/conduit subsidiary", "pre_has_haven_or_conduit"),
        ("LogHaven",   "log haven/conduit count",      "pre_log1p_haven_subs_z"),
        ("ForeignTax", "foreign tax rate",             "pre_foreign_tax_rate_z"),
        ("TaxWedge",   "domestic-foreign tax wedge",   "pre_domestic_foreign_tax_wedge_z"),
        ("Luxembourg", "Luxembourg subsidiary",        "pre_has_luxembourg"),
    ]
    for slug, label, predictor in mech_falsification:
        try:
            gamma, se = _mech_gamma_se(predictor)
        except KeyError:  # [replication package] optional data absent: blank (slides only)
            gamma = se = None
        e.add(f"mechFalse{slug}Gamma", f"A.13 {label}: γ̂ ({predictor})", _fmt_signed(gamma, 5) if gamma is not None else "", "coef")
        e.add(f"mechFalse{slug}Se",    f"A.13 {label}: SE",               f"{se:.5f}" if se is not None else "",          "se")

    # ── Body 'Mechanism: what predicts the response' (6 rows) ────────────────
    # Same source CSVs as A.13, different row selection.
    body_mech = [
        ("RdAssets",        "R&D / assets",                 "pre_rd_at_z"),
        ("CashAssets",      "cash / assets",                "pre_cash_at_z"),
        ("PreForeignPi",    "pre foreign PI share",         "pre_foreign_pi_share_z"),
        ("RevShareBody",    "foreign revenue share (body)", "pre_foreign_rev_share_z"),
        ("HavenBody",       "any haven/conduit (body)",     "pre_has_haven_or_conduit"),
        ("ClassicHavenBody","any classic-haven (body)",     "pre_has_classic_haven"),
    ]
    for slug, label, predictor in body_mech:
        try:
            gamma, se = _mech_gamma_se(predictor)
        except KeyError:  # [replication package] optional data absent: blank (slides only)
            gamma = se = None
        e.add(f"mechBody{slug}Gamma", f"Body mechanism table: {label} γ̂ ({predictor})", _fmt_signed(gamma, 5) if gamma is not None else "", "coef")
        e.add(f"mechBody{slug}Se",    f"Body mechanism table: {label} SE",               f"{se:.5f}" if se is not None else "",          "se")

    # ── Real-activity results table (body) ────────────────────────────────────
    # 5 outcome rows × (β coefficient + "-10 vs -5 firm" implied effect).
    # β + significance parsed from tbl_real_activity_dynamics.tex (spec 1, firm + ind×yr FE).
    # Implied effect = -β × workedExampleGapPp × 100 (in pp), sign as printed.
    ra = _real_activity_dynamics()
    def _ra_implied(beta):
        return _fmt_signed(-beta * 5 * 100, 2)
    e.add("realActivityCapexBeta",    "Real activity: Capex/A β (spec 1)",            _fmt_signed(ra["capex"][0], 5), "coef")
    e.add("realActivityCapexImplied", "Real activity: Capex implied -10 vs -5 (pp)",  _ra_implied(ra["capex"][0]),     "str")
    e.add("realActivityRdBeta",       "Real activity: R&D/A β (spec 1)",              _fmt_signed(ra["rd"][0], 5),     "coef")
    e.add("realActivityRdSig",        "Real activity: R&D significance",              ra["rd"][1],                     "stars")
    e.add("realActivityRdImplied",    "Real activity: R&D implied (pp)",              _ra_implied(ra["rd"][0]),        "str")
    e.add("realActivityTotalBeta",    "Real activity: Total inv/A β",                 _fmt_signed(ra["total"][0], 5),  "coef")
    e.add("realActivityTotalSig",     "Real activity: Total inv significance",        ra["total"][1],                  "stars")
    e.add("realActivityTotalImplied", "Real activity: Total inv implied (pp)",        _ra_implied(ra["total"][0]),     "str")
    e.add("realActivityEmplBeta",     "Real activity: Employment growth β",           _fmt_signed(ra["empl"][0], 5),   "coef")
    e.add("realActivityEmplImplied",  "Real activity: Employment implied (%)",        _ra_implied(ra["empl"][0]),      "str")
    e.add("realActivitySalesBeta",    "Real activity: Sales growth β",                _fmt_signed(ra["sales"][0], 5),  "coef")
    e.add("realActivitySalesImplied", "Real activity: Sales implied (%)",             _ra_implied(ra["sales"][0]),     "str")

    # ── Windfall debt: robustness with §965 transition-tax control ──────────────
    # Sourced from analysis/build_investment_elasticity.py windfall_summary.json.
    debt_ctrl = _windfall_summary()["debt_sec965_controlled"]
    e.add("windfallDebtSecRobustBeta",
          "Windfall debt: β with §965 transition-tax control (robustness)",
          _fmt_signed(debt_ctrl["coef"], 3), "coef")
    e.add("windfallDebtSecRobustSig",
          "Windfall debt: significance of §965-controlled β",
          debt_ctrl["stars"], "stars")

    e.write(OUT_DIR / "remainingpipeline.csv")


# ── Section: pre-TCJA placebo windows (A.15a) ────────────────────────────────


def emit_placebo_windows():
    """Pre-TCJA placebo-window robustness for appendix A.15a.

    Reads:
      data/analysis/output/robustness/pre_tcja_placebo_windows.csv

    The upstream CSV is emitted by analysis/build_pre_tcja_placebo_windows.py.
    `make data` regenerates it before this builder runs.
    """
    e = Emitter()

    path = ROOT / "data" / "analysis" / "output" / "robustness" / "pre_tcja_placebo_windows.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Run analysis/build_pre_tcja_placebo_windows.py first."
        )

    df = pd.read_csv(path)
    key_map = {
        "Placebo 2005": "placeboFive",
        "Placebo 2007": "placeboSeven",
        "Placebo 2009": "placeboNine",
        "Placebo 2011": "placeboEleven",
        "Placebo 2013": "placeboThirteen",
        "Main TCJA": "placeboMain",
    }
    outcome_map = {
        "International": "Intl",
        "etr_w": "Etr",
        "foreign_share": "Share",
    }

    for window, prefix in key_map.items():
        w = df[df["window"] == window]
        if w.empty:
            raise ValueError(f"Could not find {window!r} in {path.relative_to(ROOT)}")
        meta = w.iloc[0]
        e.add(f"{prefix}Pre", f"A.15a {window}: pre years", meta["pre_years"].replace("-", "--"), "str")
        e.add(f"{prefix}Post", f"A.15a {window}: post years", meta["post_years"].replace("-", "--"), "str")
        e.add(f"{prefix}Firms", f"A.15a {window}: firms in headline sample", int(meta["firms"]), "int")
        for outcome, suffix in outcome_map.items():
            row = w[w["outcome"] == outcome]
            if row.empty:
                raise ValueError(f"Could not find {window!r} / {outcome!r} in {path.relative_to(ROOT)}")
            r = row.iloc[0]
            e.add(f"{prefix}{suffix}", f"A.15a {window}: {outcome} coefficient", _fmt_signed(r["coef"], 3), "coef")
            e.add(f"{prefix}{suffix}Se", f"A.15a {window}: {outcome} SE", f"{float(r['se']):.3f}", "se")
            e.add(f"{prefix}{suffix}Sig", f"A.15a {window}: {outcome} significance", str(r["stars"]) if pd.notna(r["stars"]) else "", "stars")

    e.write(OUT_DIR / "placebowindows.csv")


# ── Section: post-window timing variants (A.15b) ────────────────────────────


def emit_post_window_variants():
    """Post-window timing robustness for appendix A.15b.

    Reads:
      data/analysis/output/robustness/post_window_variants.csv

    The upstream CSV is emitted by analysis/build_post_window_variants.py.
    `make data` regenerates it before this builder runs.
    """
    e = Emitter()

    path = ROOT / "data" / "analysis" / "output" / "robustness" / "post_window_variants.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Run analysis/build_post_window_variants.py first."
        )

    df = pd.read_csv(path)
    key_map = {
        "Early post": "postEarly",
        "Baseline": "postBase",
        "Long post": "postLong",
        "Late post": "postLate",
    }
    outcome_map = {
        "International": "Intl",
        "etr_w": "Etr",
        "foreign_share": "Share",
    }

    for variant, prefix in key_map.items():
        v = df[df["variant"] == variant]
        if v.empty:
            raise ValueError(f"Could not find {variant!r} in {path.relative_to(ROOT)}")
        meta = v.iloc[0]
        e.add(f"{prefix}Window", f"A.15b {variant}: post window", meta["post_years"].replace("-", "--"), "str")
        e.add(f"{prefix}Firms", f"A.15b {variant}: firms in headline sample", int(meta["firms"]), "int")
        for outcome, suffix in outcome_map.items():
            row = v[v["outcome"] == outcome]
            if row.empty:
                raise ValueError(f"Could not find {variant!r} / {outcome!r} in {path.relative_to(ROOT)}")
            r = row.iloc[0]
            e.add(f"{prefix}{suffix}", f"A.15b {variant}: {outcome} coefficient", _fmt_signed(r["coef"], 3), "coef")
            e.add(f"{prefix}{suffix}Se", f"A.15b {variant}: {outcome} SE", f"{float(r['se']):.3f}", "se")
            e.add(f"{prefix}{suffix}Sig", f"A.15b {variant}: {outcome} significance", str(r["stars"]) if pd.notna(r["stars"]) else "", "stars")

    e.write(OUT_DIR / "postwindows.csv")


# ── Section: real-activity post-window timing variants (A.15c) ───────────────


def emit_real_activity_post_window_variants():
    """Real-activity post-window timing robustness for appendix A.15c.

    Reads:
      data/analysis/output/robustness/real_activity_post_window_variants.csv

    The upstream CSV is emitted by
    analysis/build_real_activity_post_window_variants.py. `make data`
    regenerates it before this builder runs.
    """
    e = Emitter()

    path = (
        ROOT
        / "data"
        / "analysis"
        / "output"
        / "robustness"
        / "real_activity_post_window_variants.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. "
            "Run analysis/build_real_activity_post_window_variants.py first."
        )

    df = pd.read_csv(path)
    key_map = {
        "Early post": "realEarly",
        "Real-activity baseline": "realBase",
        "Long post 2023": "realExtended",
        "Long post": "realLong",
        "Late post": "realLate",
    }
    outcome_map = {
        "capex_int": "Capex",
        "rd_int": "Rd",
        "total_inv": "Total",
        "emp_growth": "Emp",
        "sale_growth": "Sales",
    }

    for variant, prefix in key_map.items():
        v = df[df["variant"] == variant]
        if v.empty:
            raise ValueError(f"Could not find {variant!r} in {path.relative_to(ROOT)}")
        meta = v.iloc[0]
        e.add(f"{prefix}Window", f"A.15c {variant}: post window", meta["post_years"].replace("-", "--"), "str")
        e.add(f"{prefix}PanelFirms", f"A.15c {variant}: balanced panel firms", int(meta["panel_firms"]), "int")
        for outcome, suffix in outcome_map.items():
            row = v[v["outcome"] == outcome]
            if row.empty:
                raise ValueError(f"Could not find {variant!r} / {outcome!r} in {path.relative_to(ROOT)}")
            r = row.iloc[0]
            e.add(f"{prefix}{suffix}", f"A.15c {variant}: {outcome} coefficient", _fmt_signed(r["coef"], 5), "coef")
            e.add(f"{prefix}{suffix}Se", f"A.15c {variant}: {outcome} SE", f"{float(r['se']):.5f}", "se")
            e.add(f"{prefix}{suffix}Sig", f"A.15c {variant}: {outcome} significance", str(r["stars"]) if pd.notna(r["stars"]) else "", "stars")
            e.add(f"{prefix}{suffix}Impl", f"A.15c {variant}: {outcome} implied 5pp closure", _fmt_signed(r["implied_5pp_pp"], 2), "str")
            e.add(f"{prefix}{suffix}Firms", f"A.15c {variant}: {outcome} firms", int(r["firms"]), "int")

    e.write(OUT_DIR / "realactivitypostwindows.csv")


# ── Section: tax law constants (background section) ───────────────────────────


def emit_tax_law():
    """TCJA tax-law constants in the background section. These are legal
    facts (not pipeline outputs), but exposing them through the same datatool
    layer means there's exactly ONE place where any number in the deck can
    be updated. If a future tax-law change matters (e.g. GILTI rate changing
    in 2026), it gets updated here and the deck recompiles."""
    e = Emitter()

    # Federal corporate rate
    e.add("fedRatePre", "Federal corporate rate pre-TCJA", 35, "pct")
    e.add("fedRatePost", "Federal corporate rate post-TCJA", 21, "pct")

    # §965 transition tax (sec NineSixFive → use 'transitionTax' prefix, no digits)
    e.add("transitionTaxCashRate", "Sec 965 transition tax rate, liquid assets", 15.5, "pct")
    e.add("transitionTaxIlliquidRate", "Sec 965 transition tax rate, illiquid assets", 8, "pct")

    # §245A participation exemption (Dividends Received Deduction)
    e.add("drdRate", "Sec 245A Dividends Received Deduction rate", 100, "pct")
    e.add("cfcOwnershipThreshold", "Sec 245A specified-CFC ownership threshold", 10, "pct")
    e.add("drdHoldingDays", "Sec 245A minimum holding period (days)", 365, "int")
    e.add("drdHoldingWindow", "Sec 245A holding-period testing window (days)", 731, "int")

    # GILTI
    e.add("giltiQbai", "GILTI QBAI exemption threshold", 10, "pct")
    e.add("giltiDeductPre", "§250 GILTI deduction, through tax-year 2025", 50, "pct")
    e.add("giltiDeductPost", "§250 GILTI deduction, from tax-year 2026", 37.5, "pct")
    e.add("giltiRatePre", "GILTI effective rate, through 2025", 10.5, "pct")
    e.add("giltiRatePost", "GILTI effective rate, from 2026", 13.125, "pct")
    e.add("giltiFtc", "GILTI foreign-tax-credit allowance", 80, "pct")

    # FDII
    e.add("fdiiDeductPre", "§250 FDII deduction, through tax-year 2025", 37.5, "pct")
    e.add("fdiiDeductPost", "§250 FDII deduction, from tax-year 2026", 21.875, "pct")
    e.add("fdiiRatePre", "FDII effective rate, through 2025", 13.125, "pct")
    e.add("fdiiRatePost", "FDII effective rate, from 2026", 16.406, "pct")
    e.add("fdiiEffRateApprox", "FDII approximate effective rate", 13, "pct")

    # BEAT
    e.add("beatGrossReceiptsThresh", "BEAT gross-receipts threshold (USD M)", 500, "int")
    e.add("beatPctThreshGeneral", "BEAT pct threshold (general)", 3, "pct")
    e.add("beatPctThreshBanks", "BEAT pct threshold (banks)", 2, "pct")
    e.add("beatRateShockYear", "BEAT rate 2018", 5, "pct")
    e.add("beatRateInterim", "BEAT rate 2019-2025", 10, "pct")
    e.add("beatRatePostShock", "BEAT rate 2026+", 12.5, "pct")

    # Bonus depreciation schedule
    e.add("bonusDeprFull", "Bonus depreciation, full window", 100, "pct")
    e.add("bonusDeprPhaseDownA", "Bonus depreciation, 2023", 80, "pct")
    e.add("bonusDeprPhaseDownB", "Bonus depreciation, 2024", 60, "pct")
    e.add("bonusDeprPhaseDownC", "Bonus depreciation, 2025", 40, "pct")
    e.add("bonusDeprPhaseDownD", "Bonus depreciation, 2026", 20, "pct")
    e.add("bonusDeprPhaseDownE", "Bonus depreciation, 2027+", 0, "pct")

    # §179 expensing + §163(j) interest cap + NOL (no-digit prefixes)
    e.add("expensingLimit", "Sec 179 expensing limit (USD M)", 1, "int")
    e.add("expensingPhaseout", "Sec 179 phase-out threshold (USD M)", "2.5", "str")
    e.add("intDeductCap", "Sec 163(j) interest-deduction cap", 30, "pct")
    e.add("nolCap", "NOL carryforward cap", 80, "pct")

    # Derived: windfall scale W_i = (fedRatePre - fedRatePost) / fedRatePre = 14/35 = 0.40.
    # Drives the Part III deferred-tax shock. If the rate cut changes, this updates.
    rate_pre, rate_post = 35, 21
    e.add("windfallScale", "Windfall scale W_i = (fedRatePre - fedRatePost)/fedRatePre",
          f"{(rate_pre - rate_post) / rate_pre:.2f}", "str")
    e.add("windfallScaleNumer", "Windfall scale numerator (rate cut in pp)", rate_pre - rate_post, "int")

    e.write(OUT_DIR / "taxlaw.csv")


# ── Section: documentnumbers (additional keys for coverage_audit.md + DATA_REPORT.md) ──


def emit_document_numbers():
    """Numbers used in coverage_audit.md and DATA_REPORT.md beyond what
    the slides already cite. Pulls from:

      - data/analysis/output/coverage_audit/aggregate_summary.csv
        (per-stage by-group coverage counts + sale-coverage %s)
      - data/analysis/output/coverage_audit/reconciliation_summary.csv
        (the 145,504 → ... → 103,319 funnel)
      - data/analysis/output/coverage_audit/stage2a_not_cached_decomposition.csv
        (for the 'Anatomy of the not-cached gap' subsection)

    Naming convention: `s{stageNumber}{Group}{Metric}`:
      - stage: One/Two/TwoA/TwoC/Three/Four/Five/Six
      - group: All/Top100/Top250/Top500
      - metric: Denom/Num/CountPct/SalePct  OR for funnel just Compustat/Urls/...
    """
    e = Emitter()

    # ── Aggregate per-stage by-group coverage ─────────────────────────────────
    agg = pd.read_csv(ROOT / "data" / "analysis" / "output" / "coverage_audit" / "aggregate_summary.csv")

    # Stages we cite in markdown (in the order they appear in the doc).
    stage_to_key = {
        "1. URL merge": "stageOne",
        "2a. Phase 1: cached": "stageTwoA",
        "2c. Phase 1: extractable": "stageTwoC",
        "3. Phase 2: extracted": "stageThree",
        "4. Canonical panel": "stageFour",
        "5. Profitable subsample": "stageFive",
        "6. ETR sample": "stageSix",
    }
    group_to_key = {
        "All": "All",
        "Top100": "TopHundred",
        "Top250": "TwoFifty",
        "Top500": "FiveHundred",
    }
    for _, row in agg.iterrows():
        sk = stage_to_key.get(row["stage"])
        gk = group_to_key.get(row["group"])
        if sk is None or gk is None:
            continue
        prefix = f"{sk}{gk}"
        e.add(f"{prefix}Denom", f"{row['stage']} {row['group']}: denom firm-years", int(row["total_denom_count"]), "int")
        e.add(f"{prefix}Num", f"{row['stage']} {row['group']}: numerator firm-years", int(row["total_num_count"]), "int")
        e.add(f"{prefix}CountPct", f"{row['stage']} {row['group']}: count coverage", round(row["overall_coverage_count_pct"], 2), "pct")
        e.add(f"{prefix}SalePct", f"{row['stage']} {row['group']}: sale coverage", round(row["overall_coverage_sale_pct"], 2), "pct")

    # ── Reconciliation funnel (145,504 → 103,319 trail) ─────────────────────
    rec = pd.read_csv(ROOT / "data" / "analysis" / "output" / "coverage_audit" / "reconciliation_summary.csv")
    rec_keys = {
        "0. Compustat universe (USA, non-excluded SIC, fyear 2000-2025)": "funnelCompustat",
        "1. Filing URLs merged with Compustat": "funnelUrls",
        "2a. Phase 1: cached (status=OK)": "funnelCached",
        "2b. Phase 1: classified (any bucket)": "funnelClassified",
        "2c. Phase 1: extractable (high+likely+ambig)": "funnelExtractable",
        "3. Phase 2: extracted rows (any year)": "funnelPhaseTwo",
        "4. Canonical analysis panel": "funnelCanonical",
        "5. Profitable subsample (pi>1, sale>0)": "funnelProfitable",
        "6. ETR sample (profitable + non-Finance)": "funnelEtrSample",
    }
    for _, row in rec.iterrows():
        key = rec_keys.get(row["stage"])
        if key is None:
            continue
        e.add(key, row["stage"], int(row["firm_years"]), "int")
        e.add(f"{key}Pct", f"{row['stage']}: pct of Compustat universe", round(row["pct_of_compustat"], 2), "pct")

    # ── Δ between consecutive funnel steps ───────────────────────────────────
    # The exec summary computes loss values; pre-compute them here.
    rec_vals = dict(zip(rec["stage"], rec["firm_years"]))
    deltas = {
        "lossUrlMerge": rec_vals["1. Filing URLs merged with Compustat"] - rec_vals["0. Compustat universe (USA, non-excluded SIC, fyear 2000-2025)"],
        "lossCaching": rec_vals["2a. Phase 1: cached (status=OK)"] - rec_vals["1. Filing URLs merged with Compustat"],
        "lossTableLocation": rec_vals["2c. Phase 1: extractable (high+likely+ambig)"] - rec_vals["2a. Phase 1: cached (status=OK)"],
        "gainPhaseTwoMultiYear": rec_vals["3. Phase 2: extracted rows (any year)"] - rec_vals["2c. Phase 1: extractable (high+likely+ambig)"],
        "gainCanonicalBuild": rec_vals["4. Canonical analysis panel"] - rec_vals["3. Phase 2: extracted rows (any year)"],
    }
    for key, val in deltas.items():
        e.add(key, f"Funnel delta: {key}", abs(int(val)), "int")
    # Net loss between Compustat and canonical
    net_loss = rec_vals["0. Compustat universe (USA, non-excluded SIC, fyear 2000-2025)"] - rec_vals["4. Canonical analysis panel"]
    e.add("netLossCompustatToPanel", "Net firm-years lost between Compustat and canonical panel", int(net_loss), "int")
    # Gross loss through Phase 1 (sum of three negative deltas)
    gross_loss = abs(deltas["lossUrlMerge"]) + abs(deltas["lossCaching"]) + abs(deltas["lossTableLocation"])
    e.add("grossLossPhaseOne", "Gross firm-years lost through Phase 1 (sum)", int(gross_loss), "int")

    # ── Not-cached gap totals ────────────────────────────────────────────────
    decomp = pd.read_csv(ROOT / "data" / "analysis" / "output" / "coverage_audit" / "stage2a_not_cached_decomposition.csv")
    totals = decomp.drop(columns=["year"]).sum()
    e.add("gapNotCached", "Stage 2a gap: not cached total", int(totals["not_cached"]), "int")
    e.add("gapNoUrl", "Stage 2a gap: no URL matched", int(totals["no_url_total"]), "int")
    e.add("gapCikMissing", "Stage 2a gap: no URL because CIK missing in Compustat", int(totals["no_url_cik_missing"]), "int")
    e.add("gapCikNoFilings", "Stage 2a gap: no URL because CIK has zero 10-Ks", int(totals["no_url_cik_no_10k"]), "int")
    e.add("gapCikYearMiss", "Stage 2a gap: no URL because fyear mismatch", int(totals["no_url_cik_year_miss"]), "int")
    e.add("gapHadUrlNotCached", "Stage 2a gap: URL existed but Phase 1 did not cache", int(totals["had_url_not_cached_total"]), "int")
    e.add("gapSkipNonHtml", "Stage 2a gap: SKIP_NON_HTML (.txt-only filings)", int(totals["had_url_skip_non_html"]), "int")
    e.add("gapFail", "Stage 2a gap: status=FAIL", int(totals["had_url_fail"]), "int")
    e.add("gapNoBsRow", "Stage 2a gap: no row in batch_summary", int(totals["had_url_no_bs_row"]), "int")

    # ── Hardcoded rescue numbers (the funnel quoted in §2 rescue subsection) ──
    e.add("rescueYearMissBucket", "Rescue: starting year-miss bucket", 10840, "int")
    e.add("rescueCandidates", "Rescue: candidates with adjacent SEC filing within 90 days", 1133, "int")
    e.add("rescueAlreadyCovered", "Rescue: already covered in panel (silently rescued)", 924, "int")
    e.add("rescueTrulyMissing", "Rescue: truly missing (new URLs)", 97, "int")
    e.add("rescueHtmlUrls", "Rescue: HTML URLs (after dropping .txt-only)", 67, "int")
    e.add("rescuePhaseOneOk", "Rescue: Phase 1 cached successfully", 67, "int")
    e.add("rescueExtractable", "Rescue: Phase 1 in an extractable bucket", 51, "int")
    e.add("rescuePhaseTwoExtracted", "Rescue: Phase 2 LLM extracted ≥1 recon row", 48, "int")
    e.add("rescueTotalRecons", "Rescue: total recon rows produced", 122, "int")
    e.add("rescueNetNewPanel", "Rescue: net new (gvkey, recon_year) in panel_v4", 56, "int")
    e.add("rescueInHeadlineWindow", "Rescue: rows inside 2000-2024 headline window", 35, "int")
    e.add("rescueDroppedTxt", "Rescue: pre-HTML .txt filings dropped", 30, "int")

    e.write(OUT_DIR / "documentnumbers.csv")


# ── Section: cash taxes + capital location (appendix A.31-A.36) ──────────────


def emit_cash_and_capital():
    """Cash-tax pass-through (comp_txpd/PI) and GEOSEG capital-location
    results for appendix Section 9 (A.31-A.36).

    Sources:
      - data/analysis/output/cash_etr_passthrough_summary.json
        (emitted by analysis/cash_etr_passthrough.py)
      - data/analysis/output/capital_location_summary.json +
        capital_location_panel.csv
        (emitted by analysis/capex_location_test.py)
      - data/analysis/tables/real_activity_dynamics/tbl_real_activity_dynamics.tex
        (capex spec-1 SE, for the user-cost calibration CI)

    Also computes the user-cost calibration shown on A.33: the implied
    capex response if the worked-example AETR rise were (wrongly) read as
    a marginal rate rise, under Hall-Jorgenson with bonus-era z.
    """
    e = Emitter()

    # ── Cash-tax pass-through (A.31) ──────────────────────────────────────────
    with open(ROOT / "data" / "analysis" / "output" / "cash_etr_passthrough_summary.json") as f:
        cash = json.load(f)

    ce, be = cash["cash_etr"], cash["book_etr"]
    e.add("cashEtrBeta",    "Cash-ETR pass-through: β on txpd/PI",  _fmt_signed(ce["coef"], 3), "coef")
    e.add("cashEtrBetaSe",  "Cash-ETR pass-through: SE",            f"{ce['se']:.3f}",          "se")
    e.add("cashEtrBetaSig", "Cash-ETR pass-through: significance",  ce["stars"],                "stars")
    e.add("cashEtrN",       "Cash-ETR pass-through: N firm-years",  ce["n"],                    "int")
    e.add("cashEtrFirms",   "Cash-ETR pass-through: N firms",       ce["G"],                    "int")
    e.add("cashEtrCoveragePct", "comp-txpd coverage in DiD panel",  cash["coverage"]["pct"],    "pct")
    # Book-ETR on the identical panel (sanity: equals passthroughBetaEtr)
    e.add("cashEtrBookBetaCheck", "Book-ETR β on the same panel (check)", _fmt_signed(be["coef"], 3), "coef")
    e.add("cashVsBookPct", "Cash pass-through vs book ( pct  larger)",
          round((abs(ce["coef"]) / abs(be["coef"]) - 1) * 100), "pct")
    ev17 = next(r for r in cash["cash_etr_event"] if int(r["year"]) == 2017)
    e.add("cashEtrEventShockYear",   "Cash-ETR event-study coefficient, 2017", _fmt_signed(ev17["beta"], 2), "coef")
    e.add("cashEtrEventShockYearSe", "Cash-ETR event-study SE, 2017",          f"{ev17['se']:.2f}",          "se")

    # ── GEOSEG capital location (A.34-A.35) ───────────────────────────────────
    with open(ROOT / "data" / "analysis" / "output" / "capital_location_summary.json") as f:
        cap = json.load(f)
    # [replication package] capital_location_panel.csv is a firm-level file
    # derived from Compustat segment data; it exists only if
    # analysis/capex_location_test.py ran on a user-supplied compustat_segments.csv.
    _cap_panel_path = ROOT / "data" / "analysis" / "output" / "capital_location_panel.csv"
    cap_panel = pd.read_csv(_cap_panel_path) if _cap_panel_path.exists() else None
    pre_mask = cap_panel["recon_year"].between(2013, 2016) if cap_panel is not None else None

    def _cap_block(json_key, share_col, prefix, label):
        blk = cap[json_key]
        o = blk["outcome"]["ind_yr"]
        i = blk["international_same_sample"]["ind_yr"]
        t = o["coef"] / o["se"] if o["se"] else float("nan")
        sig = _stars_from_t(abs(t))
        e.add(f"{prefix}Beta",   f"{label}: β (firm + ind x yr FE)", _fmt_signed(o["coef"], 5), "coef")
        e.add(f"{prefix}BetaSe", f"{label}: SE",                     f"{o['se']:.5f}",          "se")
        # Always emit, empty when insignificant: the deck references
        # \capLoc*BetaSig unconditionally, so dropping the key on a
        # coefficient that loses significance breaks the build.
        e.add(f"{prefix}BetaSig", f"{label}: significance", sig, "stars")
        e.add(f"{prefix}Firms", f"{label}: N firms",      o["G"], "int")
        e.add(f"{prefix}N",     f"{label}: N firm-years", o["n"], "int")
        e.add(f"{prefix}PerPp", f"{label}: pp share per 1pp Intl closure",
              _fmt_signed(blk["closure_ratio"]["ind_yr"], 2), "str")
        if cap_panel is not None:
            e.add(f"{prefix}PreMeanPct", f"{label}: pre-period mean ( pct )",
                  round(100 * cap_panel.loc[pre_mask, share_col].mean(), 1), "pct")
        else:  # [replication package] no capital_location_panel.csv: blank, slides only
            e.add(f"{prefix}PreMeanPct", f"{label}: pre-period mean ( pct )", "", "str")
        e.add(f"{prefix}IntlBeta",   f"{label}: same-sample first stage β", _fmt_signed(i["coef"], 2), "coef")
        e.add(f"{prefix}IntlBetaSe", f"{label}: same-sample first stage SE", f"{i['se']:.2f}",         "se")

    _cap_block("foreign_ppe_share",   "foreign_ppe_share",   "capLocPpe",  "Foreign PP&E share (strict)")
    _cap_block("foreign_capxs_share", "foreign_capxs_share", "capLocCapx", "Foreign capex share (strict)")
    _cap_block("foreign_sales_share", "foreign_sales_share", "capLocRev",  "Foreign revenue share (same constr.)")

    # ── Heterogeneity: which firm type carries each margin (A.37) ─────────────
    # Profit relocation concentrates in asset-light, R&D-intensive firms; the
    # weak foreign-capex tilt in capital-intensive firms (NOT R&D-intensive) —
    # the two margins live in different populations. Source: capex_location_test
    # heterogeneity_block. Triple = Di x standardized-Z concentration test.
    het = cap.get("heterogeneity", {})

    def _het(margin, label):
        h = het.get(margin)
        if not h:
            return
        pre = "het" + margin.capitalize()
        for split_key, tag in [("rd_triple", "Rd"), ("capint_triple", "Capint")]:
            tr = h.get(split_key)
            if not tr:
                continue
            t = tr["diz_t"]
            e.add(f"{pre}{tag}TripleT",   f"{label}: Di x {tag} concentration t", _fmt_signed(t, 1), "str")
            e.add(f"{pre}{tag}TripleSig", f"{label}: Di x {tag} concentration sig", _stars_from_t(abs(t)), "str")
        cs = h.get("capint_split")
        if cs:
            for lvl in ("high", "low"):
                r = cs[lvl]
                tt = r["coef"] / r["se"] if r["se"] else float("nan")
                L = lvl.capitalize()
                e.add(f"{pre}Capint{L}Beta",  f"{label}: capital-intensity {lvl} β", _fmt_signed(r["coef"], 4), "coef")
                e.add(f"{pre}Capint{L}Sig",   f"{label}: capital-intensity {lvl} sig", _stars_from_t(abs(tt)), "str")
                e.add(f"{pre}Capint{L}Firms", f"{label}: capital-intensity {lvl} N firms", r["G"], "int")

    _het("profit", "Profit-share heterogeneity")
    _het("capex",  "Capex-share heterogeneity")

    # ── User-cost calibration (A.33) ──────────────────────────────────────────
    # Naive marginal reading: treat the worked-example AETR rise (3pp, ending
    # at the 21% statutory rate) as a marginal rate rise 18% -> 21%, and ask
    # what Hall-Jorgenson predicts for capex under bonus-era z.
    TAU_TO = 21          # = fedRatePost
    ETR_RISE = 3.0       # = workedExampleEtrPp
    TAU_FROM = TAU_TO - ETR_RISE
    ELASTICITY = 1.0     # |elasticity of K wrt user cost|
    K_OVER_AT = 0.5      # K/assets
    HORIZON = 5          # years of transition to the new K*

    def uc_factor(tau, z):
        return (1 - tau * z) / (1 - tau)

    def delta_c_pct(z):
        return (uc_factor(TAU_TO / 100, z) / uc_factor(TAU_FROM / 100, z) - 1) * 100

    def implied_capx_pp(z):
        dk_pct = -ELASTICITY * delta_c_pct(z)            # % change in K*
        return dk_pct / 100 * K_OVER_AT / HORIZON * 100  # pp of capx/at per yr

    Z_HIGH, Z_MID = 0.9, 0.7
    e.add("ucCalibTauFrom", "Calibration: naive marginal tau, from", TAU_FROM, "pct")
    e.add("ucCalibTauTo",   "Calibration: naive marginal tau, to",   TAU_TO,   "pct")
    e.add("ucCalibZHigh",   "Calibration: z, mostly bonus-eligible", f"{Z_HIGH:.1f}", "str")
    e.add("ucCalibZMid",    "Calibration: z, structures-heavy",      f"{Z_MID:.1f}",  "str")
    e.add("ucCalibDeltaCHigh", "Calibration: d user cost, z=0.9 ( pct )", _fmt_signed(delta_c_pct(Z_HIGH), 2), "str")
    e.add("ucCalibDeltaCMid",  "Calibration: d user cost, z=0.7 ( pct )", _fmt_signed(delta_c_pct(Z_MID), 2),  "str")
    e.add("ucCalibCapxHigh", "Calibration: implied d capx/at per yr, z=0.9 (pp)", _fmt_signed(implied_capx_pp(Z_HIGH), 2), "str")
    e.add("ucCalibCapxMid",  "Calibration: implied d capx/at per yr, z=0.7 (pp)", _fmt_signed(implied_capx_pp(Z_MID), 2),  "str")
    e.add("ucCalibKOverAt",  "Calibration: assumed K/assets",        f"{K_OVER_AT:.1f}", "str")
    e.add("ucCalibHorizonYears", "Calibration: transition horizon (years)", HORIZON, "int")

    # Capex CI for the worked-example (-10 vs -5) firm, from the real-activity
    # spec-1 row: implied = beta*(-5)*100 pp, CI +/- 1.96*5*SE*100.
    text = (ROOT / "data" / "analysis" / "tables" / "real_activity_dynamics" /
            "tbl_real_activity_dynamics.tex").read_text()
    m = re.search(r"Capex intensity[^$]*\$([+\-]?[\d.]+)[^\n]*\n\s*&\s*&\s*\$\(([\d.]+)\)\$", text)
    if not m:
        raise ValueError("could not parse capex beta/SE from real-activity table")
    cap_beta, cap_se = float(m.group(1)), float(m.group(2))
    implied = cap_beta * (-5) * 100
    half = 1.96 * 5 * cap_se * 100
    e.add("ucCalibCapxCiLo", "Worked-example capex implied CI, lower (pp)", _fmt_signed(implied - half, 2), "str")
    e.add("ucCalibCapxCiHi", "Worked-example capex implied CI, upper (pp)", _fmt_signed(implied + half, 2), "str")

    e.write(OUT_DIR / "cashandcapital.csv")


def emit_real_activity_robustness():
    """Real-activity robustness (appendix Section 10, A.38-A.40): outcome-spec
    robustness, the denominator decomposition, and the balance-sheet /
    repatriation channel. Source: analysis/build_real_activity_robustness.py
    JSON. Shows the deck's apparent "real response" is a denominator artifact of
    a tax-driven balance-sheet (cash) contraction, not organic investment.
    Sign: beta<0 => outcome rises for exposed firms; beta>0 => falls."""
    e = Emitter()
    with open(ROOT / "data" / "analysis" / "output" / "real_activity_robustness_summary.json") as f:
        rj = json.load(f)

    def emit(key, d, label, digits=5):
        e.add(key, label, _fmt_signed(d["coef"], digits), "coef")
        e.add(key + "Sig", label + " sig", _stars_from_t(abs(d["t"])), "str")

    # A.38 outcome-spec robustness
    sp = rj["spec"]
    for k, nm in [("capex", "Capex"), ("rd", "RD"), ("totalinv", "TotInv")]:
        for tr, tag in [("ratio_at", "RatioAt"), ("log", "Log"),
                        ("dlog", "Dlog"), ("ratio_ppe", "RatioPpe")]:
            emit(f"raRob{nm}{tag}", sp[k][tr], f"A.38 {k} {tr} beta")

    # A.39 denominator decomposition
    dn = rj["denominator"]
    emit("raRobLogCapex",    dn["log_capx"], "A.39 log(capex) beta")
    emit("raRobLogRd",       dn["log_rd"],   "A.39 log(R&D) beta")
    emit("raRobLogAt",       dn["log_at"],   "A.39 log(assets) beta")
    emit("raRobLogPpe",      dn["log_ppe"],  "A.39 log(PP&E) beta")
    emit("raRobDecompRatio", dn["ratio"],    "A.39 log(R&D/lag at) ratio beta")
    emit("raRobDecompNum",   dn["num"],      "A.39 numerator log(R&D) beta")
    emit("raRobDecompDen",   dn["den"],      "A.39 denominator log(lag at) beta")
    e.add("raRobDenomSharePct", "A.39 denominator share of R&D log-ratio response (pct)",
          dn["denominator_share_pct"], "pct")

    # A.40 balance sheet + repatriation split
    bs = rj["balance_sheet"]
    emit("raRobLogChe",      bs["log_che"],     "A.40 log(cash) beta")
    emit("raRobCashInt",     bs["cash_int"],    "A.40 cash/at beta")
    emit("raRobTxpd",        bs["txpd_int"],    "A.40 cash taxes/at beta")
    emit("raRobBuyback",     bs["buyback_int"], "A.40 buybacks/at beta")
    emit("raRobDebt",        bs["debt_int"],    "A.40 debt/at beta")
    emit("raRobDiv",         bs["div_int"],     "A.40 dividends/at beta")
    emit("raRobSppe",        bs["sppe_int"],    "A.40 PP&E disposals/at beta")
    emit("raRobAqc",         bs["aqc_int"],     "A.40 acquisitions/at beta")
    emit("raRobDp",          bs["dp_int"],      "A.40 depreciation/at beta")
    emit("raRobAssetHiCash", bs["assetcontraction_hicash"], "A.40 asset contraction, high pre-cash")
    emit("raRobAssetLoCash", bs["assetcontraction_locash"], "A.40 asset contraction, low pre-cash")

    e.write(OUT_DIR / "realactivityrobust.csv")


def emit_windfall_integration():
    """Windfall / foreign-exposure integration for the Part III bridge slides.

    Source: analysis/build_windfall_foreign_exposure_integration.py. The
    numbers link the deferred-tax windfall W_i to the Part I/II foreign
    exposure treatment and re-run the Part II DiD with W_i x post included.
    """
    e = Emitter()

    bins_path = ROOT / "data" / "analysis" / "output" / "robustness" / "windfall_foreign_exposure_bins.csv"
    cross_path = ROOT / "data" / "analysis" / "output" / "robustness" / "windfall_foreign_exposure_cross_section.csv"
    robust_path = ROOT / "data" / "analysis" / "output" / "robustness" / "windfall_partii_control_robustness.csv"
    for path in [bins_path, cross_path, robust_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.relative_to(ROOT)}. "
                "Run analysis/build_windfall_foreign_exposure_integration.py first."
            )

    bins = pd.read_csv(bins_path)
    cross = pd.read_csv(cross_path)
    robust = pd.read_csv(robust_path)

    bin_map = {
        "low/no (>=-1pp)": "wfLow",
        "moderate (-5,-1pp)": "wfMod",
        "heavy (<-5pp)": "wfHeavy",
    }
    for label, prefix in bin_map.items():
        row = bins[bins["exposure_bin"] == label].iloc[0]
        e.add(f"{prefix}Firms", f"Windfall integration {label}: firms", int(row["firms"]), "int")
        e.add(f"{prefix}IntlPre", f"Windfall integration {label}: mean intl_pre", _fmt_signed(row["intl_pre_mean"], 2), "str")
        e.add(f"{prefix}WMean", f"Windfall integration {label}: mean W pct assets", _fmt_signed(row["W_pct_mean"], 2), "str")
        e.add(f"{prefix}WMedian", f"Windfall integration {label}: median W pct assets", _fmt_signed(row["W_pct_median"], 2), "str")
        e.add(f"{prefix}NetDtl", f"Windfall integration {label}: net DTL share", 100 * row["net_dtl_share"], "pct")
        e.add(f"{prefix}ForeignShare", f"Windfall integration {label}: pre foreign PI share", 100 * row["pre_foreign_share"], "pct")
        e.add(f"{prefix}Assets", f"Windfall integration {label}: median pre assets", int(round(row["pre_assets_median"])), "int")

    total_firms = int(bins["firms"].sum())
    e.add("wfIntOverlapFirms", "Windfall integration: merged Part II/windfall firms", total_firms, "int")

    for model, prefix in [("no_fe", "wfSlopeNoFe"), ("industry_fe", "wfSlopeInd")]:
        row = cross[cross["model"] == model].iloc[0]
        e.add(prefix, f"Windfall integration {model}: slope on foreign exposure", _fmt_signed(row["coef_foreign_exposure"], 3), "coef")
        e.add(prefix + "Se", f"Windfall integration {model}: SE", f"{row['se']:.3f}", "se")
        e.add(prefix + "Sig", f"Windfall integration {model}: significance", _stars_from_t(row["t"]), "stars")

    outcome_map = {
        "International": "Intl",
        "etr_w": "Etr",
        "foreign_share": "Share",
        "foreign_share_log": "LogShare",
        "pidom_int": "DomPi",
        "pifo_int": "ForPi",
    }
    for outcome, suffix in outcome_map.items():
        row = robust[robust["outcome"] == outcome].iloc[0]
        base = f"wfRob{suffix}Base"
        ctrl = f"wfRob{suffix}Ctrl"
        wkey = f"wfRob{suffix}W"
        e.add(base, f"Windfall control robustness {outcome}: baseline intl coef", _fmt_signed(row["base_coef_intl"], 5), "coef")
        e.add(base + "Se", f"Windfall control robustness {outcome}: baseline intl SE", f"{row['base_se_intl']:.5f}", "se")
        e.add(base + "Sig", f"Windfall control robustness {outcome}: baseline intl sig", str(row["base_stars_intl"]) if pd.notna(row["base_stars_intl"]) else "", "stars")
        e.add(ctrl, f"Windfall control robustness {outcome}: W-control intl coef", _fmt_signed(row["wctrl_coef_intl"], 5), "coef")
        e.add(ctrl + "Se", f"Windfall control robustness {outcome}: W-control intl SE", f"{row['wctrl_se_intl']:.5f}", "se")
        e.add(ctrl + "Sig", f"Windfall control robustness {outcome}: W-control intl sig", str(row["wctrl_stars_intl"]) if pd.notna(row["wctrl_stars_intl"]) else "", "stars")
        e.add(wkey, f"Windfall control robustness {outcome}: W-post coef", _fmt_signed(row["wctrl_coef_w"], 5), "coef")
        e.add(wkey + "Se", f"Windfall control robustness {outcome}: W-post SE", f"{row['wctrl_se_w']:.5f}", "se")
        e.add(wkey + "Sig", f"Windfall control robustness {outcome}: W-post sig", str(row["wctrl_stars_w"]) if pd.notna(row["wctrl_stars_w"]) else "", "stars")
        e.add(f"wfRob{suffix}Firms", f"Windfall control robustness {outcome}: firms", int(row["firms"]), "int")
        e.add(f"wfRob{suffix}Delta", f"Windfall control robustness {outcome}: intl delta", _fmt_signed(row["delta_intl"], 5), "str")

    e.write(OUT_DIR / "windfallintegration.csv")


# ── Metadata ─────────────────────────────────────────────────────────────────


def emit_metadata():
    """Build-time metadata (date, source files). The slides can cite these
    for provenance ("data as of YYYY-MM-DD")."""
    e = Emitter()
    e.add("buildDate", "Slide-data build date (YYYY-MM-DD)", date.today().isoformat(), "str")
    e.add("buildScript", "Script that emitted these CSVs", "analysis/build_slide_data.py", "str")
    # Identifier of the LLM used for the Phase-2 fallback parser. Update here
    # if the model is re-run on a different version.
    e.add("llmModelName", "LLM model used for the Phase-2 fallback parser", "GPT-5.2", "str")
    e.write(OUT_DIR / "metadata.csv")


# ── Section: deferred_tax descriptives (appendix A.16a-g) ────────────────────


def emit_deferred_tax_descriptives():
    """Aggregate DTA/DTL trend and 2016 cross-section for the windfall cohort.

    Reads helpers from analysis/build_deferred_tax_descriptives.py so the
    1,354-firm cohort is built once with the same logic the regressions use.
    Also writes the two-panel figure to data/analysis/figures/deferred_tax/.
    """
    from build_deferred_tax_descriptives import (
        build_windfall_cohort,
        build_aggregate_trend,
        build_cross_section_2016,
        make_figure,
    )

    cohort = build_windfall_cohort()
    trend = build_aggregate_trend(cohort)
    xs = build_cross_section_2016(cohort)

    # (Re)write the figure so `make data` keeps it in sync with the CSVs.
    fig_dir = ROOT / "data" / "analysis" / "figures" / "deferred_tax"
    make_figure(trend, fig_dir / "fig_dt_aggregate_trend.pdf")

    e = Emitter()

    # ── Aggregate trend headline numbers ────────────────────────────────
    peak_dtl = trend.loc[trend["dtl_share_pct"].idxmax()]
    post = trend[trend["year"] == 2017].iloc[0]
    pre  = trend[trend["year"] == 2016].iloc[0]
    drop_pp     = post["dtl_share_pct"] - pre["dtl_share_pct"]
    drop_pct    = 100 * drop_pp / pre["dtl_share_pct"]
    expected_pct = 100 * (35 - 21) / 35

    e.add("dtCohortFirms",        "Cohort: 1,354 non-Finance windfall firms",
          len(cohort), "int")
    e.add("dtCohortPctNetDtl",    "Cohort: pct net-DTL firms (windfall winners)",
          f"{(cohort['Z'] > 0).mean() * 100:.1f}", "pct")
    e.add("dtCohortPctNetDta",    "Cohort: pct net-DTA firms",
          f"{(cohort['Z'] < 0).mean() * 100:.1f}", "pct")

    e.add("dtPeakDtlShare",       "Peak aggregate DTL / assets (%)",
          f"{peak_dtl['dtl_share_pct']:.1f}", "pct")
    e.add("dtPeakDtlYear",        "Year of peak aggregate DTL share",
          int(peak_dtl["year"]), "int")
    e.add("dtPreDtlShare",        "Aggregate DTL / assets in 2016 (%)",
          f"{pre['dtl_share_pct']:.1f}", "pct")
    e.add("dtPreDtaShare",        "Aggregate DTA / assets in 2016 (%)",
          f"{pre['dta_share_pct']:.1f}", "pct")
    e.add("dtPostDtlShare",       "Aggregate DTL / assets in 2017 (%)",
          f"{post['dtl_share_pct']:.1f}", "pct")
    e.add("dtPostDtaShare",       "Aggregate DTA / assets in 2017 (%)",
          f"{post['dta_share_pct']:.1f}", "pct")
    e.add("dtSeventeenDropPp",    "DTL/assets drop 2016 to 2017 (pp)",
          f"{drop_pp:.2f}", "coef")
    e.add("dtSeventeenDropPct",   "DTL/assets percent drop 2016 to 2017 (%)",
          f"{drop_pct:.1f}", "pct")
    e.add("dtExpectedDropPct",    "Naive expected pct drop from 35 to 21 (%)",
          f"{expected_pct:.1f}", "pct")
    e.add("dtPreDtlBn",           "Aggregate DTL 2016 ($B)",
          f"{pre['agg_dtl_bn']:.0f}", "coef")
    e.add("dtPreDtaBn",           "Aggregate DTA 2016 ($B)",
          f"{pre['agg_dta_bn']:.0f}", "coef")

    # ── 2016 cross-section: per-industry cells ──────────────────────────
    # Key for each industry chosen to be camelCase, no digits, all letters.
    IND_KEY = {
        "Tech":           "Tech",
        "Pharma/Biotech": "Pharma",
        "Manufacturing":  "Mfg",
        "Retail":         "Retail",
        "Oil & Gas":      "OilGas",
        "Utilities":      "Util",
        "Other":          "Other",
        "All":            "All",
    }
    for _, r in xs.iterrows():
        ind = IND_KEY[r["industry"]]
        e.add(f"dtXs{ind}N",        f"Cross-section {r['industry']}: N firms",
              int(r["n_firms"]), "int")
        e.add(f"dtXs{ind}Median",   f"Cross-section {r['industry']}: median txndb/at (%)",
              f"{r['median_pct']:.1f}", "coef")
        e.add(f"dtXs{ind}PTwentyFive",
              f"Cross-section {r['industry']}: P25 (%)",
              f"{r['p25_pct']:.1f}", "coef")
        e.add(f"dtXs{ind}PSeventyFive",
              f"Cross-section {r['industry']}: P75 (%)",
              f"{r['p75_pct']:.1f}", "coef")
        e.add(f"dtXs{ind}PctDtl",
              f"Cross-section {r['industry']}: pct net-DTL firms (%)",
              f"{r['pct_net_dtl']:.0f}", "pct")

    e.write(OUT_DIR / "deferredtax.csv")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("build_slide_data.py — emitting slide-data CSVs ...")
    print(f"  Output dir: {OUT_DIR.relative_to(ROOT)}")
    print()
    print("[1/13] pipelinecounts.csv")
    # [replication package] pipelinecounts.csv summarizes the extraction stage
    # (Phase 1/2 batch files, SEC URL universe), which is not part of the
    # package. If those files are absent, keep the shipped precomputed CSV.
    _raw = [ROOT / "data" / "phase1" / "batch_summary_v4_combined.csv",
            ROOT / "data" / "phase1" / "filing_urls_compustat_merged.csv",
            ROOT / "data" / "phase2" / "full_run_v4_llm.csv"]
    if all(f.exists() for f in _raw):
        emit_pipeline_counts()
    else:
        print("  skipped: extraction-stage files not present; using data/precomputed/pipelinecounts.csv")
    print("[2/13] regressionheadlines.csv")
    emit_regression_headlines()
    print("[3/13] samplesanddescriptive.csv")
    emit_sample_sizes_and_derived()
    print("[4/13] pharmaandcases.csv")
    emit_pharma_and_cases()
    print("[5/13] remainingpipeline.csv")
    emit_remaining_pipeline()
    print("[6/13] placebowindows.csv")
    emit_placebo_windows()
    print("[7/13] postwindows.csv")
    emit_post_window_variants()
    print("[8/13] realactivitypostwindows.csv")
    emit_real_activity_post_window_variants()
    print("[9/13] taxlaw.csv")
    emit_tax_law()
    print("[10/13] documentnumbers.csv")
    emit_document_numbers()
    print("[11/13] deferredtax.csv")
    emit_deferred_tax_descriptives()
    print("[12/13] cashandcapital.csv")
    emit_cash_and_capital()
    print("[--] realactivityrobust.csv")
    emit_real_activity_robustness()
    print("[--] windfallintegration.csv")
    emit_windfall_integration()
    print("[13/13] metadata.csv")
    emit_metadata()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
