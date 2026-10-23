#!/usr/bin/env python3
"""
make_shareable_panel.py  (authors' tool; included for transparency)

Builds data/tax_reconciliation_panel_shareable.csv from the authors' full
analysis panel (data/analysis/output/tax_reconciliation_panel_base.csv):

  * keeps every reconciliation / extraction / SEC-metadata column populated;
  * keeps only the Compustat columns that the replication code reads, and
    BLANKS them (Compustat is licensed and cannot be redistributed);
  * drops two unused provenance helper columns (gvkey_str, fyear_n) and the
    ~950 Compustat columns the code never touches.

Users do not need to run this. See merge_compustat.py to refill the blanks.
"""
import sys, pandas as pd

SRC = sys.argv[1]; DST = sys.argv[2]; CODEBOOK = sys.argv[3]

# Compustat columns the replication code reads (union over all live scripts).
COMP_USED = [
    "comp_datadate", "comp_fyear",                      # match identifiers
    "comp_pi", "comp_pidom", "comp_pifo", "comp_ni", "comp_txt", "comp_txpd",
    "comp_sale", "comp_cogs", "comp_xsga", "comp_xrd", "comp_emp",
    "comp_at", "comp_capx", "comp_ppent", "comp_intan", "comp_gdwl", "comp_aqc",
    "comp_che", "comp_dltt", "comp_dlc", "comp_ceq", "comp_dvc", "comp_prstkc",
    "comp_mkvalt", "comp_prcc_f", "comp_csho",
    "comp_txndb", "comp_txndba", "comp_txndbl", "comp_txdb", "comp_txditc", "comp_tlcf",
    "comp_txfed", "comp_txfo", "comp_oancf", "comp_sstk",
    "comp_sic", "comp_fic",
    "filing_compustat_fyear",                            # loaded by build_etr_figures, not used in any computation
]
DROP = ["gvkey_str", "fyear_n"]

cols = pd.read_csv(SRC, nrows=0).columns.tolist()
keep_open = [c for c in cols if not c.startswith(("comp_", "filing_compustat_")) and c not in DROP]
df = pd.read_csv(SRC, usecols=keep_open, dtype=str, keep_default_na=False, low_memory=False)
for c in COMP_USED:
    df[c] = ""
order = keep_open[:keep_open.index("statutory_tax")] + COMP_USED + keep_open[keep_open.index("statutory_tax"):]
df = df[order]
df.to_csv(DST, index=False)
print(f"wrote {DST}: {len(df):,} rows x {df.shape[1]} columns ({len(COMP_USED)} Compustat columns blank)")

# ---- codebook -------------------------------------------------------------
D = {
 "filing_id": ("SEC filing", "Stable 12-hex id of the 10-K filing (hash of the canonical EDGAR URL)."),
 "filing_url": ("SEC filing", "EDGAR URL of the 10-K document the reconciliation was extracted from."),
 "gvkey": ("Identifier", "Compustat GVKEY of the filer (link key for the Compustat merge; integer, no leading zeros)."),
 "recon_year": ("Reconciliation", "Fiscal year the reconciliation column refers to (a 10-K reports up to 3 years; one row per filing x year)."),
 "year_offset_from_filing": ("SEC filing", "recon_year minus the filing's own fiscal year (0 = current year, -1 = prior year, ...)."),
 "within_expected_filing_window": ("SEC filing", "True if recon_year lies in the window a 10-K is expected to cover."),
 "filing_entity_count": ("SEC filing", "Number of Compustat entities mapped to the same filing URL (combined filings)."),
 "has_compustat_year_match": ("Compustat flag", "True if a Compustat Fundamentals Annual row exists for (gvkey, recon_year). Recomputed by merge_compustat.py."),
 "statutory_tax": ("Reconciliation", "Starting line of the reconciliation: statutory rate in percentage points if israte=1; expected tax at the statutory rate in filing-native currency units if israte=0."),
 "israte": ("Reconciliation", "1 = rate reconciliation (percentage points); 0 = dollar/provision reconciliation (filing-native units, usually thousands)."),
 "effective_tax": ("Reconciliation", "Final line: effective tax rate (israte=1) or total income tax expense/benefit (israte=0)."),
 "recon_check": ("Reconciliation", "True if statutory_tax + sum(category_i_val) equals effective_tax within tolerance."),
 "extraction_mode": ("Extraction", "How the row was extracted: deterministic parser, llm_assisted fallback, or a manual/scripted repair mode."),
 "extraction_candidate_id": ("Extraction", "Which candidate table in the filing packet was used."),
 "extraction_confidence": ("Extraction", "Extractor confidence (high / medium / low)."),
 "extraction_warnings": ("Extraction", "Pipe-separated extractor warnings (e.g. statutory_value_inferred)."),
 "phase1_source_score": ("Extraction", "Phase 1 (table-finding) score of the selected table."),
 "extracted_category_count": ("Reconciliation", "Number of non-blank reconciling items (category_*_label) in the row."),
 "phase1_status": ("Extraction", "Phase 1 status for the filing (OK / SKIP_NON_HTML / FAIL)."),
 "phase1_selected_type": ("Extraction", "Phase 1 selected object type (table / narrative / skip)."),
 "phase1_selected_bucket": ("Extraction", "Phase 1 confidence bucket of the selected table."),
 "phase1_selected_reason": ("Extraction", "Semicolon-separated reasons for the Phase 1 selection."),
 "phase1_retrieval_path": ("Extraction", "Phase 1 retrieval path (best_region / global_search / xbrl_fallback ...)."),
 "phase1_cache_source": ("Extraction", "Phase 1 cache source."),
 "phase1_best_region_score": ("Extraction", "Phase 1 best-region score."),
 "phase1_num_tables_total": ("Extraction", "Number of HTML tables in the filing."),
 "phase1_selected_count": ("Extraction", "Number of tables Phase 1 selected."),
 "sec_cik": ("SEC filing", "SEC Central Index Key of the filer."),
 "sec_cik10": ("SEC filing", "CIK, 10-digit zero-padded form."),
 "sec_ticker": ("SEC filing", "Ticker as recorded by SEC-API at filing time."),
 "sec_company_name": ("SEC filing", "Company name as recorded in the EDGAR filing index."),
 "sec_form_type": ("SEC filing", "Form type (10-K, 10-K405, 10KSB, ...)."),
 "filed_at": ("SEC filing", "Filing timestamp."),
 "period_of_report": ("SEC filing", "Fiscal period end date of the filing."),
 "sec_fyear": ("SEC filing", "Fiscal year of the filing (from period_of_report)."),
 "phase1_out_txt_path": ("Extraction", "Path of the Phase 1 text packet inside the authors' pipeline (not shipped)."),
 "phase1_error": ("Extraction", "Phase 1 error message, if any."),
 "source_dir": ("Provenance", "Which pipeline run (canonical or a rescue run) the row came from."),
 "_status_rank": ("Provenance", "Internal ranking used when the same filing appeared in several runs."),
 "overlap_repair_role": ("Provenance", "Role of the row in the overlapping-filing repair step (replacement / replaced_source / value_repaired)."),
 "overlap_repair_tag": ("Provenance", "Which repair pass touched the row."),
 "overlap_repair_counterparty_filing_id": ("Provenance", "filing_id of the counterpart row in an overlap repair."),
 "overlap_repair_applied": ("Provenance", "True if an overlap repair was applied to the row."),
 "comp_datadate": ("Compustat (blank)", "DATADATE of the matched Compustat row."),
 "comp_fyear": ("Compustat (blank)", "FYEAR of the matched Compustat row (= recon_year when matched)."),
 "filing_compustat_fyear": ("Compustat (blank)", "FYEAR of the Compustat row matched to the filing itself (loaded by build_etr_figures.py but not used in any computation; may be left blank)."),
}
COMP_DESC = {"pi":"Pretax income","pidom":"Pretax income, domestic","pifo":"Pretax income, foreign","ni":"Net income","txt":"Income taxes, total","txpd":"Income taxes paid",
 "sale":"Sales/turnover (net)","cogs":"Cost of goods sold","xsga":"SG&A expense","xrd":"R&D expense","emp":"Employees (thousands)","at":"Total assets","capx":"Capital expenditures",
 "ppent":"PP&E, net","intan":"Intangible assets","gdwl":"Goodwill","aqc":"Acquisitions","che":"Cash and short-term investments","dltt":"Long-term debt","dlc":"Debt in current liabilities",
 "ceq":"Common equity","dvc":"Dividends, common","prstkc":"Purchase of common and preferred stock","mkvalt":"Market value, total","prcc_f":"Price close, fiscal year","csho":"Common shares outstanding",
 "txndb":"Net deferred tax liability (balance sheet)","txndba":"Net deferred tax assets","txndbl":"Net deferred tax liabilities","txdb":"Deferred taxes (balance sheet)","txditc":"Deferred taxes and investment tax credit","tlcf":"Tax loss carryforward","txfed":"Income taxes, federal","txfo":"Income taxes, foreign","oancf":"Operating activities, net cash flow","sstk":"Sale of common and preferred stock",
 "sic":"SIC code (header)","fic":"Country of incorporation (ISO code)"}
rows=[]
for c in order:
    if c in D: src,desc=D[c]
    elif c.startswith("category_"):
        i=c.split("_")[1]; src="Reconciliation"; desc=(f"Label of reconciling item {i}, as written in the filing." if c.endswith("label") else f"Value of reconciling item {i} (pp if israte=1, native currency units if israte=0).")
    elif c.startswith("comp_"): src="Compustat (blank)"; desc=f"Compustat Fundamentals Annual {c[5:].upper()}: {COMP_DESC[c[5:]]}."
    else: raise KeyError(c)
    rows.append({"column":c,"source":src,"populated_in_package":"no" if src.startswith("Compustat (") else "yes","description":desc})
pd.DataFrame(rows).to_csv(CODEBOOK,index=False)
print(f"wrote {CODEBOOK}: {len(rows)} columns")
