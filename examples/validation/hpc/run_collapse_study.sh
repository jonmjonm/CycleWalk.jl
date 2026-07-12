#!/usr/bin/env bash
# Weight-collapse study: run AIS with per-sample annealing-path recording for one or
# more cases at several annealing-schedule lengths, then analyze where the importance-
# weight variance is injected (and print a suggested non-linear reschedule). Built to run
# on a big shared machine (e.g. hamilton, 64 cores) but works locally too.
#
#   cd CycleWalk.jl/examples
#   THREADS=60 CASES="ct nc" ANNEAL="4000 16000 24000" SAMPLES=5000 \
#       bash validation/hpc/run_collapse_study.sh
#
# Disconnect-safe launch on a remote host (recommended):
#   ssh hamilton 'cd ~/MergeAnneledTemp/CycleWalk.jl/examples && \
#     tmux new-session -d -s study "THREADS=60 bash validation/hpc/run_collapse_study.sh \
#       > validation/hpc/study.log 2>&1"'
#   ssh hamilton 'tmux attach -t study'      # to watch;  Ctrl-b d to detach
#
# Config via environment variables (defaults shown):
: "${CASES:=ct nc}"          # cases to study (ct, nc, grid, small)
: "${ANNEAL:=4000 16000 24000}"  # annealing steps per sample, per run
: "${SAMPLES:=5000}"         # AIS outer samples (enough for stable variance curves)
: "${BASE_STEPS:=500}"       # base-chain steps between samples
: "${PATH_POINTS:=100}"      # recorded points along each sample's anneal
: "${THREADS:=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 8 )}"
: "${RUN_CYCLEWALK:=0}"      # 1 => also run a cycle-walk baseline per case (for validation)
: "${JULIA:=}"               # override julia binary; else auto-detect below

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1        # -> CycleWalk.jl/examples
mkdir -p validation/hpc/logs output/validation

# locate Julia: explicit $JULIA, then module system, then PATH, then juliaup
if [ -z "$JULIA" ]; then
    if command -v module >/dev/null 2>&1; then module load julia 2>/dev/null || true; fi
    if command -v julia >/dev/null 2>&1; then JULIA=julia
    elif [ -x "$HOME/.juliaup/bin/julia" ]; then JULIA="$HOME/.juliaup/bin/julia"
    else echo "ERROR: julia not found (set JULIA=/path/to/julia)"; exit 1; fi
fi
read -ra JULIA_CMD <<< "$JULIA"   # allow JULIA to carry args, e.g. "julia +1.12.6"
echo "=== collapse study ==="
echo "julia:   $("${JULIA_CMD[@]}" --version 2>/dev/null)  ($JULIA)"
echo "threads: $THREADS   cases: $CASES   anneal: $ANNEAL   samples: $SAMPLES   path-points: $PATH_POINTS"
echo "started: $(date)"

run() {   # run <case> <mode> <extra args...> ; echoes elapsed seconds
    local case=$1 mode=$2; shift 2
    local t0=$(date +%s)
    "${JULIA_CMD[@]}" -t "$THREADS" validation/run_case.jl --case "$case" --mode "$mode" \
        --samples "$SAMPLES" --base-steps "$BASE_STEPS" "$@" \
        > "validation/hpc/logs/${case}_${mode}_$$.log" 2>&1
    local rc=$?
    echo "    (rc=$rc, $(( $(date +%s) - t0 ))s)  log: validation/hpc/logs/${case}_${mode}_$$.log"
    return $rc
}

for case in $CASES; do
    echo ""; echo "########## case: $case ##########  $(date)"
    if [ "$RUN_CYCLEWALK" = "1" ]; then
        echo "  [cyclewalk baseline]"; run "$case" cyclewalk
        cp -f "output/validation/${case}_cyclewalk.jsonl.gz" \
              "output/validation/${case}_cyclewalk.jsonl.gz" 2>/dev/null || true
    fi
    for steps in $ANNEAL; do
        echo "  [ais --record-path --anneal-steps $steps]"
        if run "$case" ais --record-path --anneal-steps "$steps" --path-points "$PATH_POINTS"; then
            cp -f "output/validation/${case}_ais.jsonl.gz" \
                  "output/validation/${case}_ais_path_a${steps}.jsonl.gz"
            echo "    saved output/validation/${case}_ais_path_a${steps}.jsonl.gz"
        else
            echo "    !! run failed; see log"
        fi
    done
    echo "  [analyze path -> variance injection + suggested reschedule]"
    "${JULIA_CMD[@]}" validation/analyze_path.jl --case "$case" \
        2>&1 | tee "validation/hpc/logs/${case}_analyze.log" | sed -n '/=== /,$p'
done

echo ""; echo "=== done: $(date) ==="
echo "figures + reschedule CSVs in output/validation/  (path_<case>_*.svg, path_<case>_reschedule.csv)"
