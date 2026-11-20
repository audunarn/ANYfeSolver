#!/usr/bin/env bash
# Sequential S3 runs, one fresh process each (timed runs first, on a machine as quiet as available).
set -u
R=$(cd ../.. && pwd -W)
export PYTHONPATH="$R/src;$R/packages/results/src;$R/tests"
PY=${PY:-/c/Github/ANYsolid/.venv/Scripts/python.exe}
run() { local name=$1; shift; local t0=$(date +%s)
  timeout 2400 "$PY" specimen.py "$@" > "raw/$name.json" 2> "raw/$name.err"; local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s load=$(powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).LoadPercentage')" | tee -a raw/runs.log; }
run timing_coupled_t1      --threads 1 coupled 20 5
run timing_coupled_default --threads default coupled 20 0
run timing_oneway_t1       --threads 1 oneway 20 5
run timing_svd_t1          --threads 1 svd 5
run peak_coupled_t1        --threads 1 peak 20
run r0_coupled_n10         --threads 1 coupled 10 0
for n in 10 20 40 80 160; do run shell_n$n --threads 1 shell $n; done
for n in 20 40; do for l in 2 4; do run solid_n${n}_l$l --threads default solid $n $l; done; done
# Diagnostic only (not gates): the candidate refined 5 times in the measurement process (finding R1-3).
run diag_coupled_n10_ref5 --threads 1 --diagnostic-refinement 5 coupled 10 0
run diag_coupled_n20_ref5 --threads 1 --diagnostic-refinement 5 coupled 20 2
