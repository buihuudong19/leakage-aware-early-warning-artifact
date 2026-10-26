#!/usr/bin/env bash
# Full reproduction on Linux or macOS. Extra arguments are passed to run_pipeline.py, e.g. ./run_pipeline.sh --quick
# Without arguments the previous outputs are first moved to _archive/outputs_<date>_<time>, so a full run starts clean;
# ./run_pipeline.sh --resume continues an interrupted full run without archiving (finished robustness tasks are kept).
set -euo pipefail
cd "$(dirname "$0")"
RESUME=0; ARGS=()
for a in "$@"; do
  if [ "$a" = "--resume" ]; then RESUME=1; else ARGS+=("$a"); fi
done
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements-lock.txt --quiet || .venv/bin/python -m pip install -r requirements.txt --quiet
if [ ${#ARGS[@]} -eq 0 ] && [ "$RESUME" -eq 0 ]; then
  echo "Full run: moving the previous outputs to _archive ..."
  .venv/bin/python tools/archive_outputs.py
fi
mkdir -p outputs/logs
TEE=(tee); [ "$RESUME" -eq 1 ] && TEE=(tee -a)
.venv/bin/python -u run_pipeline.py ${ARGS[@]+"${ARGS[@]}"} 2>&1 | "${TEE[@]}" outputs/logs/pipeline.log
