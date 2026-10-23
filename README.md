# Replication package

**Foreign Tax Advantages after the U.S. Tax Reform of 2017**
Anmol Bhandari (University of Minnesota) and Ellen R. McGrattan (University of Minnesota)
Draft of September 2026 (`paper/tcja_draft5.tex`).

This package rebuilds every table, figure and in-text number of the paper from
a firm-year panel of effective-tax-rate (ETR) reconciliations extracted from
10-K filings. The reconciliation data are included. The Compustat variables the
paper merges in are **not** included (Compustat is licensed); the package ships
the column names, a codebook and a script that refills them from your own WRDS
extract. With that extract in place, one command (`bash run_all.sh`) reproduces
the paper and compiles the PDF.

---

## 1. What is and is not reproduced

| Layer | In this package? | Notes |
|---|---|---|
| Extraction of reconciliation tables from 10-K HTML (Phase 1 table finding, Phase 2 parsing with an LLM fallback, rescue and repair passes) | **No** (its output is) | Runs for ~30 hours against SEC EDGAR and the OpenAI API. Its output, the reconciliation panel, is shipped. |
| Reconciliation panel, 227,835 filing-year rows (1998–2024) | **Yes** (`data/tax_reconciliation_panel_shareable.csv`) | All reconciliation, extraction and SEC-metadata columns populated. |
| Compustat Fundamentals Annual variables (36 data items) | **Names only** | Blank columns; refilled by `merge_compustat.py` from your WRDS extract. |
| Compustat Segments (foreign revenue share; capital location) | **No** | Optional input. Without it, the authors' aggregate results are substituted for those two blocks (two in-text counts, Table B.2 row 4, the capital-location sentences). |
| Exhibit 21 subsidiary lists (WRDS SEC Analytics) | **No** | Optional input. Without it, Table B.2 rows 5–7 (subsidiary-network predictors) are left blank. |
| Analysis: cleaning, category mapping, event studies, difference-in-differences, windfall, heterogeneity, Section 4 inversion | **Yes** | 27 Python scripts in `analysis/`, 1 Julia script in `paper/section4/code/`. |
| Paper inputs (every number in the draft is read from a CSV through `datatool`) | **Yes** | Built by the scripts in `paper/Inputs/`. |
| Appendix A pipeline counts (Tables A.1–A.5, coverage by year, Table 3) | **Precomputed** | These summarize the extraction funnel over the full SEC/Compustat universe. Shipped in `data/precomputed/`. |
| PDF | **Yes** | `pdflatex` + `bibtex`. |

Everything the draft prints comes from `paper/Inputs/*.csv`; nothing is typed by
hand except Table 1 (Alphabet's own 10-K numbers), Table 2 (the taxonomy),
Appendix Table A.6 (SIC codes), statutory constants and literature figures.

---

## 2. Package contents

```
tcja_replication/
├── README.md                       this file
├── run_all.sh                      one-command driver (merge → analysis → paper inputs → PDF → check)
├── merge_compustat.py              refills the blank Compustat columns from your WRDS extract
├── compare_to_reference.py         diffs your paper/Inputs against the authors' outputs
├── data/
│   ├── tax_reconciliation_panel_shareable.csv   THE DATA (227,835 rows × 114 columns, 199 MB; the GitHub copy ships it as .csv.gz, 22 MB)
│   ├── codebook_tax_reconciliation_panel.csv    one line per column: source, populated?, description
│   ├── original/README.md                       what to download from WRDS and where to put it
│   └── precomputed/                             extraction-stage summaries and fallbacks for optional inputs (see §4.4)
├── analysis/                       27 Python scripts (the analysis layer) incl. sic_industry.py
├── paper/
│   ├── tcja_draft5.tex, tcja_refs.bib, jpe.bst
│   ├── Inputs/                     builders of the datatool CSVs (+ 2 frozen CSVs, see §4.4)
│   └── section4/code/theory_calibration.jl      Table 9 (Julia)
├── reference/
│   ├── clean_rebuild/              paper/Inputs + PDF as produced by THIS package on the authors' machine
│   └── nber_draft/                 paper/Inputs + PDF as sent to NBER (differs in two sentences, see §6)
└── tools/make_shareable_panel.py   how the shareable CSV was made from the authors' panel (for transparency)
```

The directory layout (`analysis/`, `data/analysis/...`, `paper/Inputs/`) mirrors
the authors' repository because the scripts locate their inputs relative to
their own position. Do not move files between folders.

---

## 3. Requirements

* **Python 3.12** with `pandas 2.2`, `numpy 2.2`, `scipy 1.15`, `statsmodels 0.14`, `matplotlib 3.10`
  (these are the versions the reference outputs were produced with; pandas 2.x is required, pandas 3 is untested).
  ```
  pip install "pandas>=2.2,<3" "numpy>=2,<3" scipy statsmodels matplotlib
  ```
* **Julia 1.12** (standard library only; no packages). Used for Table 9.
* **LaTeX** (TeX Live 2024): `datatool`, `pgfplots`, `pgfplotstable`, `siunitx`, `booktabs`, `threeparttable`,
  `natbib`, `hyperref`, `tikz`, `placeins`, `chngcntr`, `sectsty`, `setspace`, `comment`, `etoolbox`.
  Optional: without `pdflatex` the run stops after building `paper/Inputs`.
* **Disk**: ~1.5 GB free (your Compustat extract, the merged panel, caches).
* **Memory**: 8 GB is enough; the largest step loads the 880 MB merged panel once.
* **Time**: about 10 minutes end to end on a 2024 laptop with the required input only
  (merge 1.5 min, analysis 5 min, paper inputs 2 min, PDF 1 min); add ~5 minutes for the
  optional Segments and Exhibit 21 steps.

Set `PYTHON=` and `JULIA=` if the binaries are not on your `PATH`:
`PYTHON=/path/to/python3 JULIA=/path/to/julia bash run_all.sh`.

---

## 4. Data

### 4.1 The panel: `data/tax_reconciliation_panel_shareable.csv`

One row per **filing × reconciliation year**. A 10-K reports the ETR
reconciliation for up to three fiscal years, so a filing contributes up to three
rows; the same fiscal year of a firm can appear in several filings (the analysis
code deduplicates, see `analysis/build_etr_figures.py`).

* 227,835 rows; 9,790 firms (gvkey); reconciliation years 1998–2024.
* Key: `filing_id` × `recon_year`. Firm link: `gvkey` (Compustat) and `sec_cik` (SEC).
* The reconciliation itself: `statutory_tax`, up to 15 reconciling items
  (`category_i_label`, `category_i_val`), `effective_tax`, `israte`, `recon_check`.
  * `israte = 1`: a **rate** reconciliation, everything in percentage points
    (statutory 35.0 → ... → effective 21.3).
  * `israte = 0`: a **dollar** reconciliation in the filing's own units (usually
    thousands of dollars): expected tax at the statutory rate → items → total
    provision. The analysis converts amounts to rates by dividing by Compustat
    pretax income (`comp_pi`), which is why Compustat is needed.
  * `recon_check` is True when start + items = end within tolerance. The paper
    uses rows with `recon_check = True` (see Appendix A for the exact screens).
* Extraction provenance (`extraction_mode`, `phase1_*`, `source_dir`,
  `overlap_repair_*`): how the row was obtained. Kept for transparency; the
  analysis uses `extraction_mode`/`overlap_repair_*` only to break ties between
  duplicate filings.
* SEC metadata (`sec_*`, `filed_at`, `period_of_report`): from the EDGAR index.
* **Compustat block (41 columns, blank in this file)**: `comp_datadate`, `comp_fyear`,
  36 data items (`comp_pi`, `comp_pidom`, `comp_pifo`, `comp_ni`, `comp_txt`, `comp_txpd`,
  `comp_sale`, `comp_cogs`, `comp_xsga`, `comp_xrd`, `comp_emp`, `comp_at`, `comp_capx`,
  `comp_ppent`, `comp_intan`, `comp_gdwl`, `comp_aqc`, `comp_che`, `comp_dltt`, `comp_dlc`,
  `comp_ceq`, `comp_dvc`, `comp_prstkc`, `comp_mkvalt`, `comp_prcc_f`, `comp_csho`,
  `comp_txndb`, `comp_txndba`, `comp_txndbl`, `comp_txdb`, `comp_txditc`, `comp_tlcf`,
  `comp_txfed`, `comp_txfo`, `comp_oancf`, `comp_sstk`),
  `comp_sic`, `comp_fic`, and `filing_compustat_fyear`. These are the only Compustat
  columns any script reads. The authors' working panel carries the full
  Fundamentals Annual record (981 `comp_*` columns); the rest are not needed.
* `has_compustat_year_match` is shipped as in the authors' panel and is
  **recomputed** by `merge_compustat.py` from your extract.

The full column list with descriptions is `data/codebook_tax_reconciliation_panel.csv`.

### 4.2 Refilling Compustat: `merge_compustat.py`

1. Download Compustat Fundamentals Annual from WRDS as described in
   `data/original/README.md` and save it as `data/original/compustat_data.csv`.
2. `python3 merge_compustat.py` (or let `run_all.sh` do it) writes
   `data/analysis/output/tax_reconciliation_panel_base.csv`, the file every
   script reads. The merge keeps one Compustat row per (gvkey, fiscal year),
   preferring `INDL` over `FS`, `STD`, consolidated, rows with non-missing
   sale/ni/pi, then the latest `datadate`, and left-joins on
   (gvkey, recon_year = fyear). This is the rule in the authors'
   `build_analysis_base.py`.

   The shipped `has_compustat_year_match` flag defines the authors' sample: a
   row the authors could not match stays unmatched even if your extract has a
   record for it (`--ignore-shipped-flag` disables this), and a row the authors
   matched but your extract lacks is reported and left blank.

**Verified**: with the authors' own WRDS extract (downloaded 2026-04-18), the
refilled values of all 36 data items are identical to the authors' panel in
every row, the matched-row count is the authors' 226,292, and every file the
paper loads is reproduced identically (§4.5). Two things are worth knowing:
* One row (Dell Technologies, gvkey 14489, fiscal 2014) has a Compustat record
  in the extract in which every data item is blank; the authors' panel treats
  it as unmatched and so does the helper. Filling it (`--ignore-shipped-flag`)
  changes which of Dell's overlapping filings is kept and moves the Table 8
  coefficients in the third decimal.
* 32 rows appended by a manual rescue pass carry a `comp_datadate` in the
  authors' panel but no financials; the helper reproduces the (empty) data
  items, only the date label differs.

A different WRDS vintage can change values for firms that restated, and firms
added to or removed from Compustat. Expect small differences in the last digit
of some estimates if your extract was downloaded later than April 2026.

### 4.3 Optional inputs: Compustat Segments and Exhibit 21 subsidiaries

Two further licensed files extend the reproduction. `run_all.sh` detects them
and falls back to the authors' aggregate results when they are absent, so the
paper always compiles.

**`data/original/compustat_segments.csv`** (Compustat Historical Segments) is read by
* `analysis/build_geo_revenue_share_outcomes.py`: foreign share of revenue from
  `GEOSEG` rows → `geo_revenue_share_tcja_panel.csv` (firm-year, not shipped),
  `geo_revenue_share_tcja_regression_summary.json` and `tbl_geo_revenue_share.tex`
  (aggregate, shipped as fallback). The paper prints two counts from them
  (`\GeoRevenueShareFirms`, `\GeoRevenueShareFirmYears`, Section 3.2) and the
  foreign-revenue-share predictor is row 4 of Table B.2.
* `analysis/capex_location_test.py`: foreign shares of PP&E, capex and sales →
  `capital_location_summary.json`, `capital_location_event_study.csv` (aggregate,
  shipped as fallback) and `capital_location_panel.csv` (firm-year, not shipped).
  The paper prints the capital-location sentences of Section 3.2 and
  `\CapLocCapxPerPp` from the summary. Without the panel, one slide-only key
  (`capLoc*PreMeanPct`) is left blank.

**`data/original/subsidiaries*.csv`** (WRDS SEC Analytics Suite, Exhibit 21) is read by
`analysis/build_subsidiary_footprint.py` → `subsidiary_footprint_panel.csv`
(firm-year counts of subsidiaries by country class, not shipped). It feeds the
subsidiary-network predictors: rows 5–7 of Table B.2. Without it,
`analyze_profit_shifting_mechanisms.py` skips those predictors and
`build_heterogeneity_table.py` writes those rows blank (a note is printed).
The complete Table B.2 is in `reference/clean_rebuild/paper_inputs/tables/heterogeneity_gamma.csv`.

`analysis/analyze_profit_shifting_deep_dive.py` (slide-only mechanism checks)
runs only when both optional files are present; otherwise its four aggregate
output CSVs are taken from `data/precomputed/`.

### 4.4 Precomputed inputs (`data/precomputed/`, and two files in `paper/Inputs/tables/`)

These summarize the extraction stage over the full universe of SEC filings and
Compustat firm-years. They cannot be regenerated from the panel alone, so they
are shipped as produced by the authors:

| File | Used for | Produced by (not in package) |
|---|---|---|
| `coverage_audit/*.csv`, `metadata.json` | Table 3 (coverage), Figure A.1, Appendix A funnel counts | `analysis/build_coverage_audit.py` on Compustat + Phase 1/2 batch files |
| `pipelinecounts.csv` | Tables A.1–A.5 and in-text counts (123,835 filings, 82,853 extracted, ...) | `emit_pipeline_counts()` in `build_slide_data.py` on Phase 1/2 batch files |
| `appendix_reconciliation.csv` | Appendix Table A.1 reconciliation of row counts | `paper/Inputs/tables/build_appendix_reconciliation.py` |
| `capital_location_summary.json`, `capital_location_event_study.csv` | Fallback for §4.3 (no Segments) | `analysis/capex_location_test.py` on Compustat Segments |
| `geo_revenue_share_tcja_regression_summary.json`, `tbl_geo_revenue_share.tex` | Fallback for §4.3 (no Segments) | `analysis/build_geo_revenue_share_outcomes.py` on Compustat Segments |
| `profit_shifting_deep_dive_{alt_outcome_main,alt_outcome_interactions,tax_wedge_interactions,country_interactions}.csv` | Fallback for §4.3 (slide-only keys) | `analysis/analyze_profit_shifting_deep_dive.py` on both optional inputs |
| `paper/Inputs/tables/reconciliation_extraction_headlines.csv`, `reconciliation_coverage_time_headlines.csv` | 4 in-text numbers in Section 2.2 (123,835; 82,853; 8.3%; 79.5%) | frozen in June 2026; equal to the live `pipelinecounts.csv` / `coverage_by_year.csv` values |

`run_all.sh` copies them to where the scripts expect them (step 2).

### 4.5 Reference outputs (`reference/`)

* `reference/clean_rebuild/paper_inputs/`: the 70 CSV/JSON files under
  `paper/Inputs` exactly as this package produces them, plus the compiled PDF.
  `compare_to_reference.py` diffs against this by default; with the authors'
  Compustat extract every file the draft loads comes out **identical**.
* `reference/nber_draft/`: the same files behind the PDF submitted to NBER on
  17 September 2026. See §6 for the (small) differences.

---

## 5. Running the package

```bash
cd tcja_replication
# 1. put your WRDS extract(s) in data/original/   (see data/original/README.md)
# 2.
bash run_all.sh                 # or: PYTHON=... JULIA=... bash run_all.sh
# 3. read run_all.log; the last block is the comparison with reference/clean_rebuild
```

`run_all.sh` does, in order (all paths relative to the package root):

| Step | Script | What it does | Writes |
|---|---|---|---|
| 1 | `merge_compustat.py` | refills Compustat, recomputes `has_compustat_year_match` | `data/analysis/output/tax_reconciliation_panel_base.csv` |
| 2 | (copy) | stages `data/precomputed/*` | `data/analysis/output/{coverage_audit,slide_data/pipelinecounts.csv}`, `paper/Inputs/tables/appendix_reconciliation.csv` |
| 3.1 | `analysis/build_etr_figures.py` | sample screens, deduplication of overlapping filings, conversion of dollar reconciliations to rates, mapping of free-text labels to categories (`classify()` / `group_map`), aggregate decomposition | `data/analysis/cache/{filtered,deduped,rate_clean,amount_clean,combined,profitable,long,full_sample,driver_panel_full}.pkl`, `data/analysis/tables/etr_drivers_by_year_full.csv` |
| 3.2 | `build_substitution_analysis.py` | channel-by-channel pass-through DiD (the module the other scripts import) | `tables/substitution/tbl_substitution_main.tex`, `tbl_channel_summary.tex`, figures |
| 3.3 | `build_substitution_real_outcomes.py` | foreign-share event study and pooled DiD (Figure 3c, Table 7) | `tables/substitution/tbl_real_outcomes.tex` |
| 3.4 | `build_closure_ratio_by_clip.py` | closure ratios (β_Share/β_Intl etc.) quoted in §1 and §3.2 | `output/robustness/foreign_share_closure_ratio_by_clip.csv` |
| 3.5–3.6 | `run_ratio_sum_treatment_robustness.py`, `run_balance_requirement_robustness.py` | treatment-definition and balance robustness (slides) | `output/robustness/ratio_sum_*`, `balance_ge1_ge1_*` |
| 3.7–3.8 | `build_geo_revenue_share_outcomes.py`, `capex_location_test.py` | foreign revenue share; capital location (need Segments; else fallback) | `output/geo_revenue_share_*`, `output/capital_location_*` |
| 3.9 | `build_subsidiary_footprint.py data/original/subsidiaries*.csv` | subsidiary counts by country class (needs Exhibit 21; else skipped) | `output/subsidiary_footprint_*` |
| 3.10 | `analyze_profit_shifting_mechanisms.py` | heterogeneity of the profit-location response (Table B.2) | `output/profit_shifting_mechanism_*.{csv,json}` |
| 3.11 | `analyze_profit_shifting_deep_dive.py` | further mechanism checks (slides; needs both optional inputs, else fallback) | `output/profit_shifting_deep_dive_*` |
| 3.12 | `build_real_activity_dynamics.py` | real-activity DiD and dynamics (Table 8) | `tables/real_activity_dynamics/*.tex`, `figures/` |
| 3.13 | `build_investment_elasticity.py` | deferred-tax windfall first stage and NI response (§3.2.4) | `tables/investment/*.tex`, `windfall_summary.json` |
| 3.14 | `build_firststage_event_studies.py` | international and ETR event studies (Figure 3a-b, Table 7) | (consumed by 4.3) |
| 3.15 | `build_robustness_ladder.py` | channel-clip robustness | `output/robustness/channel_clip_*` |
| 3.16 | `cash_etr_passthrough.py` | cash-tax pass-through (slides) | `output/cash_etr_passthrough_summary.json` |
| 3.17–3.19 | `build_pre_tcja_placebo_windows.py`, `build_post_window_variants.py`, `build_real_activity_post_window_variants.py` | placebo and post-window variants (Table 7 rows) | `output/robustness/*` |
| 3.20 | `build_real_activity_robustness.py` | real-activity robustness (reads your raw Compustat file) | `output/real_activity_robustness_summary.json` |
| 3.21 | `build_windfall_foreign_exposure_integration.py` | windfall-control robustness (§3.2.4) | `output/robustness/windfall_*` |
| 4.1 | `analysis/build_slide_data.py` | headline regression numbers, sample sizes, worked example | `data/analysis/output/slide_data/*.csv` |
| 4.2 | `paper/Inputs/sync_slide_data.sh` | copies 4 slide-data CSVs into `paper/Inputs/slide_data/` | |
| 4.3 | `paper/Inputs/build_inputs_from_external.py` | Figures 1–3, Tables 3, 4, 5, B.1, coverage by year | `paper/Inputs/{decomp_tikz,decomp_ind_new_tikz,first_*_tikz,coverage_time_tikz,tables/*}` |
| 4.4 | `identification_tikz/build_identification_tikz_data.py` | Table 6 | `paper/Inputs/identification_tikz/` |
| 4.5 | `tables/build_*.py` (11 scripts) | Tables 7, 8, B.2; balance-sheet, capital-location, windfall, case-study and headline CSVs | `paper/Inputs/tables/*.csv` |
| 4.6 | `analysis/build_section4_same_sample.py` | same-sample DiD and 999 firm-bootstrap draws for Section 4 | `paper/Inputs/tables/section4_{same_sample,bootstrap_draws}.csv` |
| 4.7 | `paper/section4/code/theory_calibration.jl` (Julia, run from `paper/`) | inversion of the model: ψ, τ_f, shifted shares, CIs (Table 9, §4 text) | `paper/Inputs/tables/theory_calibration.csv`, `theory_grid.csv` |
| 4.8 | `analysis/build_category_diagnostics.py` | Appendix A category table | `paper/Inputs/tables/category_diagnostics.csv` |
| 4.9 | `paper/Inputs/check_inputs_consistency.py` | 9 cross-file checks (e.g. channel betas sum to β_ETR) | exits non-zero on failure |
| 5 | `pdflatex`, `bibtex` | compiles `paper/tcja_draft5.pdf` | |
| 6 | `compare_to_reference.py` | cell-by-cell diff of the 46 files the draft loads | |

Run `bash run_all.sh --no-pdf` to stop after step 4. To re-run a single step,
call the script from the package root (steps 4.3–4.5 and 4.9 from
`paper/Inputs/`; step 4.7 from `paper/`), as `run_all.sh` does.

The industry classification used everywhere (Figure 2, Table 5, Table 6, the
sample screen that drops financial firms) is the single function in
`analysis/sic_industry.py`.

### Where each object of the paper comes from

| Paper | Loaded from (`paper/Inputs/`) | Built by |
|---|---|---|
| §1 headline β's, pass-through (79%), N | `slide_data/samplesanddescriptive.csv`, `tables/pooled_did_results_table.csv` | 4.1, 4.5 |
| Table 3, Figure A.1 | `tables/coverage_table.csv`, `coverage_time_tikz/` | 4.3 from precomputed `coverage_audit/` |
| Figure 1, Table 4 | `decomp_tikz/`, `tables/contribution_gap_*` | 4.3 from 3.1 |
| Figure 2, Table 5 | `decomp_ind_new_tikz/summary.csv`, `tables/industry_gap_distribution_lookup.csv` | 4.3 from 3.1 + `sic_industry.py` |
| Figure 3 | `first_stage_tikz/`, `first_etr_tikz/`, `first_pishr_tikz/` | 4.3 (imports 3.5, 3.6) |
| Table 6 | `identification_tikz/intl_pre_by_industry.csv` | 4.4 |
| Table 7 | `tables/pooled_did_results_table.csv` | 4.5 from 3.6 + 3.11 |
| Table 8 | `tables/real_activity_log_table.csv` | 4.5 from 3.3 |
| §3.2.4 windfall θ = 0.485, control robustness | `tables/windfall_ni_headline.csv`, `windfall_control_table.csv` | 4.5 from 3.4, 3.14 |
| §3.3 case studies | `tables/case_study_table.csv`, `case_study_litigation.csv` | 4.5 |
| Table 9, §4 ranges and CIs | `tables/theory_calibration.csv` | 4.7 from 4.6 |
| Tables A.1–A.5 | `slide_data/pipelinecounts.csv`, `tables/appendix_reconciliation.csv`, `category_diagnostics.csv` | precomputed, 4.8 |
| Table B.1 | `tables/channel_did_table.csv` | 4.3 |
| Table B.2 | `tables/heterogeneity_gamma.csv` | 4.5 from 3.10 (rows 4–7 need the optional inputs) |
| §3.2 segment counts, closure ratios (§1, §3.2) | `slide_data/samplesanddescriptive.csv`, `regressionheadlines.csv`, `tables/profit_location_headlines.csv` | 4.1, 4.5 from 3.4, 3.7 |

---

## 6. Known differences from the NBER PDF

`reference/nber_draft/` is what was submitted. A clean rebuild
(`reference/clean_rebuild/`, and what you will get) differs in two printed
sentences and in some inputs the paper does not print:

1. **§3.2.4, the windfall-control robustness sentence.** The NBER draft prints
   "0.760 (0.172) without the windfall control and 0.760 (0.173) with it"; the
   rebuild gives 0.749 (0.157) and 0.750 (0.158). The submitted numbers came
   from an intermediate file written before the September 5, 2026 change that
   keeps 2016 in the pooled DiD; re-running the windfall script with 2016
   excluded again reproduces that file exactly. The conclusion ("essentially
   unchanged") is the same, and the rebuild is the correct one.
2. **Section 4, the pre-TCJA shifted-share range.** The NBER draft used a
   post-TCJA macro as the upper bound and printed "25 to 11"; the corrected
   `paper/tcja_draft5.tex` uses `\GShiftedPreStatZero` and prints "25 to 28".
3. `tables/theory_calibration.csv`: five auxiliary keys (`betaPpeShare*`,
   `betaLogRd`) differ; none is printed in the paper.
4. `slide_data/samplesanddescriptive.csv` has one extra key
   (`passthroughLeakagePct`), defined but not printed.
5. `slide_data/cashandcapital.csv` (slides only): the cash-ETR pass-through
   block is refreshed.
6. `slide_data/samplesanddescriptive.csv` (slides only): five keys
   (`balanceRuleBeta*`, `incomeLocPerPp*`) change because two robustness
   files (`balance_ge1_ge1_comparison.csv`, `ratio_sum_treatment_diagnostics.csv`)
   dated July 2026 in the authors' repository are regenerated here on the
   current panel. The paper does not print them.

Items 1 and 3–6 all have the same cause: on September 5, 2026 the pooled-DiD
sample gained 2016 and the real-activity post window became 2019–2022, and
several scripts that import those definitions were not re-run. The authors'
repository was brought up to date on October 23, 2026, so its `paper/Inputs`
now matches `reference/clean_rebuild/` exactly.

What to expect from `compare_to_reference.py` (against `clean_rebuild`):
* with `compustat_data.csv`, `compustat_segments.csv` and the Exhibit 21 files from
  the same WRDS vintage: **44 of 44 files identical**;
* with `compustat_data.csv` only: 42 identical; `tables/heterogeneity_gamma.csv`
  (rows 4–7 blank) and `slide_data/cashandcapital.csv` (blank slide-only keys)
  differ as described in §4.3. The compiled PDF then differs from the full-data
  PDF only in the blank rows of Table B.2 and in the Appendix B sentence that
  quotes the two subsidiary coefficients (it prints empty parentheses).

---

## 7. Modifications made for this package

The analysis and paper scripts are copied verbatim from the authors'
repository (`github.com/apb-umn/TaxReconciliation`, commit `f67633e1`,
2026-09-18). Every edit is marked `# [replication package]` and only adds a
fallback for a file that is not in the package; with all inputs present the
code paths are the originals.

| File | Edit |
|---|---|
| `analysis/build_slide_data.py` | `main()`: `emit_pipeline_counts()` runs only if the extraction-stage files exist, else the precomputed `pipelinecounts.csv` is kept. `emit_cash_and_capital()`: tolerates a missing `capital_location_panel.csv`. `emit_remaining_pipeline()`: two slide-only loops emit blanks for predictors that were not estimated. |
| `analysis/analyze_profit_shifting_mechanisms.py` | `load_geo_pre_predictors()`, `load_subsidiary_pre_predictors()`: return no predictors (with a printed note) when their input file is absent. |
| `paper/Inputs/tables/build_heterogeneity_table.py` | a Table B.2 row whose predictor was not estimated is written blank instead of raising. |
| `analysis/cash_etr_passthrough.py` | reads `comp_txpd` from the base panel when the canonical panel (`tax_reconciliation_panel_v4.csv`, not shipped) is absent. Slide A.31 only. |
| `analysis/build_subsidiary_footprint.py` | output directory set to `data/analysis/output/` (the original wrote to `analysis/output/` and the file was moved by hand). |

One script of the original build is not run because it needs the
extraction-stage files: `paper/Inputs/tables/build_appendix_reconciliation.py`
(its output is shipped). Everything else runs unchanged.

---

## 8. Troubleshooting

* *`merge_compustat.py: lacks required Compustat variables`*: your WRDS export
  is missing a column; the message lists them. Column names must be lower case.
* *Far fewer than 226,292 matched rows*: check the fiscal-year range (1995–2024)
  and that you did not restrict the download to `fic = USA` or to one `indfmt`.
* *`ModuleNotFoundError`*: run the scripts from the package root (or from
  `paper/Inputs` / `paper` for the paper-layer scripts), as `run_all.sh` does;
  they import siblings by relative path.
* *pandas 3 errors*: install pandas 2.x.
* *Table B.2 rows 4–7 are blank / "NOTE: ... not present"*: expected without the
  optional Segments and Exhibit 21 files (§4.3).
* *`build_subsidiary_footprint.py` picks up the wrong files*: it receives every
  `data/original/subsidiaries*.csv`; keep only Exhibit 21 exports under that name.
* *`! LaTeX Error: File 'datatool.sty' not found`*: install the packages listed
  in §3 (TeX Live `collection-latexextra` covers them).
* *Numbers differ in the last digit from the reference*: your Compustat vintage
  differs from April 2026 (restatements, added/removed firms). Differences that
  change a conclusion would indicate a problem; please contact the authors.

---

## 9. Citation and contact

Bhandari, Anmol, and Ellen R. McGrattan (2026). "Foreign Tax Advantages after
the U.S. Tax Reform of 2017." Working paper.

Questions about the package: Anmol Bhandari, bhandari@umn.edu.

The reconciliation data were extracted by the authors from public SEC EDGAR
filings and may be reused with attribution. Compustat data are the property of
S&P Global and must be obtained under your own license.
