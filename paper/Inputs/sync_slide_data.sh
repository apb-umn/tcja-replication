#!/usr/bin/env bash
# Stage the slide-data CSVs the drafts load into paper/Inputs/slide_data/, so
# every CSV a draft reads lives under one root (\Inputs).  Re-run after
# `make data` in presentation/ regenerates them.
#
# Usage: Inputs/sync_slide_data.sh        (from paper/)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../../data/analysis/output/slide_data"
DST="$HERE/slide_data"
mkdir -p "$DST"
for f in samplesanddescriptive pipelinecounts regressionheadlines cashandcapital; do
  cp "$SRC/$f.csv" "$DST/$f.csv"
  echo "  staged $f.csv"
done
echo "wrote $DST"
