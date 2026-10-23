#!/usr/bin/env bash
# run_all.sh -- rebuild every table, figure and number in tcja_draft5.tex from
# data/tax_reconciliation_panel_shareable.csv + your own Compustat extract.
#
#   bash run_all.sh            full run (merge, analysis, paper inputs, PDF, comparison)
#   bash run_all.sh --no-pdf   skip the LaTeX step
#   PYTHON=/path/to/python3 JULIA=/path/to/julia bash run_all.sh
#
# Expected runtime: ~10 minutes on a laptop (the analysis layer is ~5 min).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
PYTHON=${PYTHON:-python3}
JULIA=${JULIA:-julia}
LOG=$ROOT/run_all.log; : > "$LOG"
say(){ echo "==> $*" | tee -a "$LOG"; }
run(){ say "$1"; shift; ( "$@" ) >> "$LOG" 2>&1 || { echo "FAILED: $* (see run_all.log)"; exit 1; }; }

# ---------------------------------------------------------------- 0. checks
say "Python: $($PYTHON -c 'import sys,pandas,numpy,scipy,statsmodels,matplotlib;print(sys.version.split()[0],"pandas",pandas.__version__,"numpy",numpy.__version__,"scipy",scipy.__version__,"statsmodels",statsmodels.__version__)')"
command -v "$JULIA" >/dev/null || { echo "julia not found (set JULIA=...)"; exit 1; }
say "Julia: $($JULIA --version)"
[ -f data/original/compustat_data.csv ] || { echo "Missing data/original/compustat_data.csv -- see data/original/README.md"; exit 1; }
HAVE_SEG=0; [ -f data/original/compustat_segments.csv ] && HAVE_SEG=1
HAVE_SUB=0; ls data/original/subsidiaries*.csv >/dev/null 2>&1 && HAVE_SUB=1
say "Optional inputs: Compustat Segments=$HAVE_SEG, Exhibit 21 subsidiaries=$HAVE_SUB"
mkdir -p data/analysis/{cache,output/slide_data,output/robustness,output/coverage_audit,tables,figures} paper/Inputs/slide_data

# ---------------------------------------------------------------- 1. merge Compustat
if [ ! -f data/analysis/output/tax_reconciliation_panel_base.csv ]; then
  run "Merging your Compustat extract into the panel" $PYTHON merge_compustat.py
else
  say "Using existing data/analysis/output/tax_reconciliation_panel_base.csv (delete it to re-merge)"
fi

# ---------------------------------------------------------------- 2. stage precomputed extraction-stage summaries
cp data/precomputed/coverage_audit/* data/analysis/output/coverage_audit/
cp data/precomputed/pipelinecounts.csv data/analysis/output/slide_data/pipelinecounts.csv
cp data/precomputed/appendix_reconciliation.csv paper/Inputs/tables/appendix_reconciliation.csv
say "Staged precomputed extraction-stage summaries (coverage audit, pipeline counts, appendix A.1-A.5)"

# ---------------------------------------------------------------- 3. analysis layer (panel -> caches -> estimates)
A=analysis
run "3.1 build_etr_figures.py (cleans the panel, writes data/analysis/cache/*.pkl)"      $PYTHON $A/build_etr_figures.py
run "3.2 build_substitution_analysis.py (channel DiD; tbl_substitution_main.tex)"       $PYTHON $A/build_substitution_analysis.py
run "3.3 build_substitution_real_outcomes.py (foreign-share event study, Fig. 3c, Table 7)" $PYTHON $A/build_substitution_real_outcomes.py
run "3.4 build_closure_ratio_by_clip.py (closure ratios quoted in Sec. 1 and 3.2)"       $PYTHON $A/build_closure_ratio_by_clip.py
run "3.5 run_ratio_sum_treatment_robustness.py"                                         $PYTHON $A/run_ratio_sum_treatment_robustness.py
run "3.6 run_balance_requirement_robustness.py"                                         $PYTHON $A/run_balance_requirement_robustness.py
if [ $HAVE_SEG = 1 ]; then
  run "3.7 build_geo_revenue_share_outcomes.py (foreign revenue share; needs compustat_segments.csv)" $PYTHON $A/build_geo_revenue_share_outcomes.py
  run "3.8 capex_location_test.py (capital location; needs compustat_segments.csv)"      $PYTHON $A/capex_location_test.py
else
  cp data/precomputed/geo_revenue_share_tcja_regression_summary.json data/analysis/output/
  mkdir -p data/analysis/tables/substitution && cp data/precomputed/tbl_geo_revenue_share.tex data/analysis/tables/substitution/
  cp data/precomputed/capital_location_summary.json data/precomputed/capital_location_event_study.csv data/analysis/output/
  say "3.7-3.8 geo revenue share + capital location SKIPPED (no data/original/compustat_segments.csv); using precomputed aggregate results"
fi
if [ $HAVE_SUB = 1 ]; then
  run "3.9 build_subsidiary_footprint.py (Exhibit 21 subsidiary counts)"                $PYTHON $A/build_subsidiary_footprint.py data/original/subsidiaries*.csv
else
  say "3.9 build_subsidiary_footprint.py SKIPPED (no data/original/subsidiaries*.csv); Table B.2 rows 5-7 will be blank"
fi
run "3.10 analyze_profit_shifting_mechanisms.py (Table B.2)"                            $PYTHON $A/analyze_profit_shifting_mechanisms.py
if [ $HAVE_SEG = 1 ] && [ $HAVE_SUB = 1 ]; then
  run "3.11 analyze_profit_shifting_deep_dive.py (slide-only mechanism checks)"        $PYTHON $A/analyze_profit_shifting_deep_dive.py
else
  cp data/precomputed/profit_shifting_deep_dive_*.csv data/analysis/output/
  say "3.11 analyze_profit_shifting_deep_dive.py SKIPPED (needs both optional inputs); using precomputed aggregate results"
fi
run "3.12 build_real_activity_dynamics.py (Table 8 inputs)"                             $PYTHON $A/build_real_activity_dynamics.py
run "3.13 build_investment_elasticity.py (deferred-tax windfall, Sec. 3.2.4)"            $PYTHON $A/build_investment_elasticity.py
run "3.14 build_firststage_event_studies.py (Fig. 3a-b, Table 7)"                       $PYTHON $A/build_firststage_event_studies.py
run "3.15 build_robustness_ladder.py"                                                   $PYTHON $A/build_robustness_ladder.py
run "3.16 cash_etr_passthrough.py (slides)"                                             $PYTHON $A/cash_etr_passthrough.py
run "3.17 build_pre_tcja_placebo_windows.py"                                            $PYTHON $A/build_pre_tcja_placebo_windows.py
run "3.18 build_post_window_variants.py (Table 7 post-window rows)"                     $PYTHON $A/build_post_window_variants.py
run "3.19 build_real_activity_post_window_variants.py"                                  $PYTHON $A/build_real_activity_post_window_variants.py
run "3.20 build_real_activity_robustness.py (reads your raw Compustat file)"            $PYTHON $A/build_real_activity_robustness.py
run "3.21 build_windfall_foreign_exposure_integration.py (Sec. 3.2.4 windfall control)" $PYTHON $A/build_windfall_foreign_exposure_integration.py

# ---------------------------------------------------------------- 4. paper inputs (datatool CSVs under paper/Inputs)
run "4.1 build_slide_data.py (headline regression numbers, sample sizes)"         $PYTHON $A/build_slide_data.py
run "4.2 sync_slide_data.sh"                                                       bash paper/Inputs/sync_slide_data.sh
( cd paper/Inputs
  run "4.3 build_inputs_from_external.py (Figs 1-3, Tables 3-5, B.1)"           $PYTHON build_inputs_from_external.py
  run "4.4 identification_tikz/build_identification_tikz_data.py (Table 6)"     $PYTHON identification_tikz/build_identification_tikz_data.py
  for t in build_pooled_did_table build_profit_location_headlines build_clip_robustness_headlines build_real_activity_pooled_did_table build_heterogeneity_table build_real_activity_log_table build_balance_sheet_table build_capital_location_table build_windfall_ni_headline build_windfall_control_table build_case_study_table; do
    run "4.5 tables/$t.py" $PYTHON tables/$t.py
  done
  say "4.5 tables/build_appendix_reconciliation.py SKIPPED (needs extraction-stage files); using precomputed appendix_reconciliation.csv"
)
run "4.6 build_section4_same_sample.py (same-sample DiD + bootstrap draws for Table 9)" $PYTHON $A/build_section4_same_sample.py
( cd paper && run "4.7 section4/code/theory_calibration.jl (Table 9, Julia)" $JULIA section4/code/theory_calibration.jl )
run "4.8 build_category_diagnostics.py (Appendix A category table)"               $PYTHON $A/build_category_diagnostics.py
( cd paper/Inputs && run "4.9 check_inputs_consistency.py" $PYTHON check_inputs_consistency.py )

# ---------------------------------------------------------------- 5. PDF
if [ "${1:-}" != "--no-pdf" ]; then
  if command -v pdflatex >/dev/null; then
    ( cd paper && run "5. pdflatex/bibtex tcja_draft5.tex" bash -c "pdflatex -interaction=nonstopmode tcja_draft5.tex && bibtex tcja_draft5 && pdflatex -interaction=nonstopmode tcja_draft5.tex && pdflatex -interaction=nonstopmode tcja_draft5.tex" )
    say "PDF: paper/tcja_draft5.pdf"
  else
    say "pdflatex not found; skipping the PDF (paper/Inputs is complete)"
  fi
fi

# ---------------------------------------------------------------- 6. compare with the authors' outputs
run "6. compare_to_reference.py" $PYTHON compare_to_reference.py
tail -n 60 "$LOG" | grep -E "^(IDENTICAL|DIFF|MISSING|summary)" || true
say "Done. Full log: run_all.log"
