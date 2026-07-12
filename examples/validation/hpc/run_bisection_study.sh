#!/usr/bin/env bash
# Adaptive bisection convergence study for the STANDARD cycle walk along the (gamma, iso)
# line  gamma = t,  iso = 0.3*t,  t in [0,1].  At each test point we run COPIES independent
# copies (same target, different seeds), then analyze_convergence.jl asks whether the copies
# agree on the rank-ordered per-district log-spanning-tree marginals. We bisect on t to
# locate where agreement breaks down.
#
#   cd CycleWalk.jl/examples
#   JULIA="julia +1.12.6" THREADS=60 CASE=grid bash validation/hpc/run_bisection_study.sh
#
# Disconnect-safe launch on hamilton (recommended):
#   ssh hamilton 'cd ~/MergeAnneledTemp/CycleWalk.jl/examples && \
#     tmux new-session -d -s bis_grid "JULIA=\"julia +1.12.6\" THREADS=60 CASE=grid \
#       bash validation/hpc/run_bisection_study.sh > validation/hpc/logs/bisection_grid.out 2>&1"'
#
# Config (defaults shown):
: "${CASE:=grid}"          # grid, ct, ...
: "${SAMPLES:=20000}"      # recorded cycle-walk samples per copy (>= 20k requested)
: "${SPACING:=1000}"       # MH steps between recorded samples (decorrelation)
: "${COPIES:=8}"           # independent copies per point (one serial chain each)
: "${STEPS:=6}"            # bisection steps AFTER the two endpoints (>= 6 requested)
: "${STRICT:=1}"           # 1 => abort if endpoint sanity fails (t=0 must CONVERGE, t=1 NOT)
: "${THREADS:=$( (command -v nproc >/dev/null && nproc) || echo 8 )}"  # (informational)
: "${JULIA:=}"

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1        # -> CycleWalk.jl/examples
mkdir -p validation/hpc/logs output/validation

# Each copy is a SERIAL MH chain: Julia threads don't help it, so pin BLAS to 1 and run the
# copies concurrently (COPIES cores). Small per-district logdet matrices => BLAS>1 is waste.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

if [ -z "$JULIA" ]; then
    if command -v julia >/dev/null 2>&1; then JULIA=julia
    elif [ -x "$HOME/.juliaup/bin/julia" ]; then JULIA="$HOME/.juliaup/bin/julia"
    else echo "ERROR: julia not found (set JULIA=/path/to/julia)"; exit 1; fi
fi
read -ra JULIA_CMD <<< "$JULIA"

SUMMARY="validation/hpc/logs/bisection_${CASE}_summary.tsv"
LOG="validation/hpc/logs/bisection_${CASE}.log"
: > "$SUMMARY"; printf "point\tt\tgamma\tiso\tverdict\tseconds\n" >> "$SUMMARY"

echo "=== bisection convergence study ===" | tee -a "$LOG"
echo "julia:   $("${JULIA_CMD[@]}" --version 2>/dev/null)  ($JULIA)" | tee -a "$LOG"
echo "case=$CASE  copies=$COPIES  samples=$SAMPLES  spacing=$SPACING  steps=$STEPS" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"

POINT=0
BASE_SEED=216334338    # = 0x0CE10002 (matches run_case cyclewalk default family)

# run_point <t> ; sets global LAST_VERDICT to CONVERGED|NOT ; echoes nothing
run_point() {
    local t=$1
    POINT=$((POINT + 1))
    # iso = 0.3*t ; tag "t<t>" with fixed precision so the analyzer glob is stable
    local iso ttag
    iso=$(awk -v t="$t" 'BEGIN{printf "%.6f", 0.3*t}')
    ttag=$(awk -v t="$t" 'BEGIN{printf "t%.6f", t}')
    echo "" | tee -a "$LOG"
    echo "########## point $POINT  t=$t  (gamma=$t iso=$iso)  $(date) ##########" | tee -a "$LOG"

    local t0 c seed pids=()
    t0=$(date +%s)
    for ((c = 1; c <= COPIES; c++)); do
        seed=$((BASE_SEED + 10007 * POINT + c))
        "${JULIA_CMD[@]}" -t 1 validation/run_case.jl --case "$CASE" --mode cyclewalk \
            --gamma "$t" --iso "$iso" --std-samples "$SAMPLES" --std-spacing "$SPACING" \
            --seed "$seed" --tag "${ttag}_c${c}" \
            > "validation/hpc/logs/${CASE}_${ttag}_c${c}.log" 2>&1 &
        pids+=($!)
    done
    local rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    local secs=$(( $(date +%s) - t0 ))
    if [ "$rc" != 0 ]; then
        echo "  !! a copy failed at t=$t (rc=$rc) -- see validation/hpc/logs/${CASE}_${ttag}_c*.log" | tee -a "$LOG"
    fi

    # analyze the COPIES copies at this point
    "${JULIA_CMD[@]}" validation/analyze_convergence.jl --case "$CASE" --tag "$ttag" --t "$t" \
        2>&1 | tee -a "$LOG" > "validation/hpc/logs/${CASE}_${ttag}_analyze.log"
    cat "validation/hpc/logs/${CASE}_${ttag}_analyze.log" >> "$LOG"

    LAST_VERDICT=$(grep -h '^VERDICT ' "validation/hpc/logs/${CASE}_${ttag}_analyze.log" \
                   | tail -1 | awk '{print $NF}')
    [ -z "$LAST_VERDICT" ] && LAST_VERDICT="NOT"
    printf "%d\t%s\t%s\t%s\t%s\t%d\n" "$POINT" "$t" "$t" "$iso" "$LAST_VERDICT" "$secs" >> "$SUMMARY"
    echo "  -> t=$t : $LAST_VERDICT  (${secs}s)" | tee -a "$LOG"
}

# --- endpoints: sanity (expect t=0 CONVERGED, t=1 NOT) ---------------------------------
run_point 0; V0=$LAST_VERDICT
run_point 1; V1=$LAST_VERDICT
echo "" | tee -a "$LOG"
echo "endpoint sanity: t=0 -> $V0 (expect CONVERGED),  t=1 -> $V1 (expect NOT)" | tee -a "$LOG"
if { [ "$V0" != "CONVERGED" ] || [ "$V1" != "NOT" ]; }; then
    echo "!! ENDPOINT SANITY FAILED" | tee -a "$LOG"
    if [ "$STRICT" = "1" ]; then
        echo "   STRICT=1 -> aborting before bisection. Inspect, then rerun (or set STRICT=0 FORCE)." | tee -a "$LOG"
        echo "=== stopped: $(date) ===" | tee -a "$LOG"; exit 2
    fi
    echo "   STRICT=0 -> continuing bisection anyway." | tee -a "$LOG"
fi

# --- bisection: lo converges, hi does not; find where it flips -------------------------
lo=0; hi=1
for ((s = 1; s <= STEPS; s++)); do
    mid=$(awk -v a="$lo" -v b="$hi" 'BEGIN{printf "%.6f", (a+b)/2}')
    echo "" | tee -a "$LOG"
    echo "===== bisection step $s/$STEPS : bracket [$lo, $hi] -> test mid=$mid =====" | tee -a "$LOG"
    run_point "$mid"
    if [ "$LAST_VERDICT" = "CONVERGED" ]; then lo=$mid; else hi=$mid; fi
    echo "  new bracket: converged<=$lo   <t*<=   not>=$hi" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== done: $(date) ===" | tee -a "$LOG"
echo "convergence breaks in t* in ($lo, $hi]   (gamma=t, iso=0.3*t)" | tee -a "$LOG"
echo "summary: $SUMMARY" | tee -a "$LOG"
