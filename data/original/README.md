# data/original/ — files you must supply (licensed; not redistributable)

All come from WRDS (wrds.wharton.upenn.edu). Column names must be the
lower-case WRDS mnemonics that WRDS exports by default. Put the files in this
folder with exactly these names, then run `bash run_all.sh` from the package root.

| File | Source | Required? | Enables |
|---|---|---|---|
| `compustat_data.csv` | Compustat Fundamentals Annual (`comp.funda`) | **yes** | everything |
| `compustat_segments.csv` | Compustat Historical Segments (`comp.wrds_segmerged` / `comp.seg_*`) | optional | foreign revenue share (2 in-text counts, Table B.2 row 4) and capital location (§3.2 sentences, `\CapLocCapxPerPp`) |
| `subsidiaries*.csv` | WRDS SEC Analytics Suite, Exhibit 21 subsidiaries (one or several files) | optional | subsidiary-network predictors (Table B.2 rows 5–7) |

Without an optional file, `run_all.sh` substitutes the authors' aggregate
results for that block (`data/precomputed/`) and leaves the affected Table B.2
rows blank; nothing else changes.

## 1. `compustat_data.csv` — Compustat Fundamentals Annual  (required)

* Fiscal years **1995–2024** (the panel's `recon_year` runs 1998–2024; merge is on `fyear`).
  In the WRDS query form, set the date variable to **Fiscal Year (`fyear`)**, not
  Data Date (`datadate`); selecting on data date drops or shifts firm-years
  whose fiscal year does not end in the calendar year.
* Company codes: all. Do not pre-filter by country or industry; the code applies its
  own screens (`fic == 'USA'`, the SIC filter in `analysis/sic_industry.py`).
* Consolidation / format: select **all** (`consol` C and others; `indfmt` INDL and FS;
  `datafmt` STD). `merge_compustat.py` keeps one row per (gvkey, fyear), preferring
  INDL, STD, consolidated, rows with non-missing sale/ni/pi, then the latest datadate.
* Variables (WRDS query-form names). Identifiers and screens:
  `gvkey fyear datadate indfmt datafmt consol fic sic cik conm`
  Data items read from the merged panel:
  `pi pidom pifo ni txt txpd sale cogs xsga xrd emp at capx ppent intan gdwl aqc che dltt dlc ceq dvc prstkc mkvalt prcc_f csho txndb txndba txndbl txdb txditc tlcf txfed txfo oancf sstk`
  Data items read directly from this file by `analysis/build_real_activity_robustness.py`
  and `analysis/capex_location_test.py`: `sppe dp` (plus items above).
  Downloading all variables also works (the authors' file had 980 columns, 1.2 GB).
* CSV with a header row.

The authors' extract (downloaded 2026-04-18) had 398,792 rows (363,031 unique
gvkey-fyear); `merge_compustat.py` matches 226,292 of the 227,835 panel rows with it.

## 2. `compustat_segments.csv` — Compustat Historical Segments  (optional)

* Dates 2013–2024 (`datadate`); only `stype == 'GEOSEG'` rows are used
  (`geotp == 3` marks foreign geographic segments).
* Variables: `stype datadate gvkey snms geotp sid sales capxs ppents` (others ignored).
* The authors' file: 884 MB, 2.83M rows, all segment types.

Used by `analysis/build_geo_revenue_share_outcomes.py` and `analysis/capex_location_test.py`.

## 3. `subsidiaries*.csv` — Exhibit 21 subsidiary lists  (optional)

WRDS SEC Analytics Suite → Subsidiaries (Exhibit 21), exported as CSV. The SEC
Analytics Suite is a separate WRDS product that many institutional subscriptions
(the University of Minnesota's among them) do not include; without it, skip this
file. One file
or several (e.g. `subsidiaries_pre_tcja.csv`, `subsidiaries_post_tcja.csv`);
`run_all.sh` passes every `data/original/subsidiaries*.csv` to
`analysis/build_subsidiary_footprint.py`, which keeps annual filings
(10-K, 10-K/A, 10-KT) for report years 2013–2024 and counts subsidiaries by
country (tax-haven and conduit lists are in the script).

* Variables used (`USECOLS` in the script): `gvkey cik coname accession form fdate rdate
  subsidiary_name clean_company location country_code subdiv_code extype ex21url formurl`.
* The authors' files: three CSVs, 1.8 GB in total, 2013–2024.
