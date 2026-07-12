#!/usr/bin/env bash
# Self-contained adaptive bisection convergence study for the STANDARD cycle walk along the
# line  gamma = t,  iso = ISO_SLOPE*t,  t in [0,1].  At each test point we run COPIES
# independent chains (same target, different seeds) and a convergence analyzer decides
# CONVERGED vs NOT; we bisect on t to bracket where convergence breaks.
#
# CRITERION=rhat (default): rank-normalized split-R-hat (Gelman-Rubin) per district rank,
#   CONVERGED iff max R-hat < RHAT.  CRITERION=rel: the older between-copy TV-vs-noise gate.
#
# The job is FIRE-AND-FORGET: it runs the whole bisection, dumps the marginals, builds a
# self-contained HTML figure (R-hat-vs-t + overlays) if python3 is present, writes a summary,
# and finally a DONE marker. It is RESUMABLE: a chain whose output already exists is reused,
# so a re-launch after a stall fast-forwards. No interactive ssh needed except to fetch.
#
#   ssh hamilton 'cd ~/MergeAnneledTemp/CycleWalk.jl/examples && tmux new-session -d -s bis_nc \
#     "JULIA=\"julia +1.12.6\" THREADS=60 CASE=nc SAMPLES=50000 COPIES=8 STEPS=8 SPACING=... \
#       bash validation/hpc/run_bisection_study.sh > validation/hpc/logs/bisection_nc.out 2>&1"'
#
: "${CASE:=grid}"
: "${SAMPLES:=20000}"       # recorded samples per chain
: "${SPACING:=1000}"        # MH steps between recorded samples
: "${COPIES:=8}"            # independent chains per point
: "${STEPS:=8}"             # bisection steps after the two endpoints
: "${CRITERION:=rhat}"      # rhat | rel
: "${RHAT:=1.01}"           # R-hat convergence threshold
: "${ISO_SLOPE:=0.3}"       # iso = ISO_SLOPE * t
: "${STRICT:=1}"            # 1 => abort if endpoint sanity fails (t=0 CONVERGED, t=1 NOT)
: "${THREADS:=$( (command -v nproc >/dev/null && nproc) || echo 8 )}"  # (informational)
: "${JULIA:=}"

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1        # -> CycleWalk.jl/examples
RES="validation/hpc/results_${CASE}"
LOGD="validation/hpc/logs"
mkdir -p "$LOGD" "$RES" output/validation
rm -f "$RES/DONE"

# Each chain is a SERIAL MH walk: Julia threads don't help it, so pin BLAS to 1 and run the
# COPIES chains concurrently (COPIES cores). Per-district logdet matrices are small.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

if [ -z "$JULIA" ]; then
    if command -v julia >/dev/null 2>&1; then JULIA=julia
    elif [ -x "$HOME/.juliaup/bin/julia" ]; then JULIA="$HOME/.juliaup/bin/julia"
    else echo "ERROR: julia not found (set JULIA=/path/to/julia)"; exit 1; fi
fi
read -ra JULIA_CMD <<< "$JULIA"

case "$CRITERION" in
    rhat) ANALYZER=(validation/analyze_rhat.jl --rhat "$RHAT") ;;
    rel)  ANALYZER=(validation/analyze_convergence.jl) ;;
    *)    echo "ERROR: CRITERION must be rhat or rel"; exit 1 ;;
esac

SUMMARY="$RES/summary.tsv"
LOG="$RES/run.log"
: > "$SUMMARY"; printf "point\tt\tgamma\tiso\tverdict\trhat\tseconds\n" >> "$SUMMARY"
TAGS=""

echo "=== bisection convergence study ($CRITERION) ===" | tee -a "$LOG"
echo "julia:   $("${JULIA_CMD[@]}" --version 2>/dev/null)  ($JULIA)" | tee -a "$LOG"
echo "case=$CASE copies=$COPIES samples=$SAMPLES spacing=$SPACING steps=$STEPS thr=$RHAT iso_slope=$ISO_SLOPE" | tee -a "$LOG"
echo "started: $(date)" | tee -a "$LOG"

POINT=0
BASE_SEED=216334338    # = 0x0CE10002

# a chain is "done" if its atlas exists and its run log reports collection
chain_done() {
    local ttag=$1 c=$2
    [ -s "output/validation/${CASE}_cyclewalk_${ttag}_c${c}.jsonl.gz" ] &&
        grep -q "collected ~" "$LOGD/${CASE}_${ttag}_c${c}.log" 2>/dev/null
}

# run_point <t> ; sets LAST_VERDICT (CONVERGED|NOT) and LAST_RHAT
run_point() {
    local t=$1
    POINT=$((POINT + 1))
    local iso ttag
    iso=$(awk -v t="$t" -v s="$ISO_SLOPE" 'BEGIN{printf "%.6f", s*t}')
    ttag=$(awk -v t="$t" 'BEGIN{printf "t%.6f", t}')
    TAGS="${TAGS:+$TAGS,}$ttag"
    echo "" | tee -a "$LOG"
    echo "########## point $POINT  t=$t  (gamma=$t iso=$iso)  $(date) ##########" | tee -a "$LOG"

    local t0 c seed pids=() rc=0
    t0=$(date +%s)
    for ((c = 1; c <= COPIES; c++)); do
        if chain_done "$ttag" "$c"; then
            echo "  copy $c: reuse existing" | tee -a "$LOG"; continue
        fi
        seed=$((BASE_SEED + 10007 * POINT + c))
        "${JULIA_CMD[@]}" -t 1 validation/run_case.jl --case "$CASE" --mode cyclewalk \
            --gamma "$t" --iso "$iso" --std-samples "$SAMPLES" --std-spacing "$SPACING" \
            --seed "$seed" --tag "${ttag}_c${c}" --no-districting \
            > "$LOGD/${CASE}_${ttag}_c${c}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]:-}"; do [ -n "${p:-}" ] && { wait "$p" || rc=1; }; done
    local secs=$(( $(date +%s) - t0 ))
    [ "$rc" != 0 ] && echo "  !! a copy failed at t=$t (rc=$rc)" | tee -a "$LOG"

    "${JULIA_CMD[@]}" "${ANALYZER[@]}" --case "$CASE" --tag "$ttag" --t "$t" \
        > "$LOGD/${CASE}_${ttag}_analyze.log" 2>&1
    cat "$LOGD/${CASE}_${ttag}_analyze.log" >> "$LOG"

    local vline
    vline=$(grep -h '^VERDICT ' "$LOGD/${CASE}_${ttag}_analyze.log" | tail -1)
    LAST_VERDICT=$(echo "$vline" | awk '{print $NF}'); [ -z "$LAST_VERDICT" ] && LAST_VERDICT="NOT"
    LAST_RHAT=$(echo "$vline" | grep -o 'rhat=[0-9.]*' | cut -d= -f2); [ -z "$LAST_RHAT" ] && LAST_RHAT="NA"
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%d\n" "$POINT" "$t" "$t" "$iso" "$LAST_VERDICT" "$LAST_RHAT" "$secs" >> "$SUMMARY"
    echo "  -> t=$t : $LAST_VERDICT  (R̂=$LAST_RHAT, ${secs}s)" | tee -a "$LOG"
}

finish() {   # dump marginals, build figure, write DONE (always, even on early abort)
    echo "" | tee -a "$LOG"; echo "=== finalizing: $(date) ===" | tee -a "$LOG"
    "${JULIA_CMD[@]}" validation/dump_marginals.jl --case "$CASE" --tags "$TAGS" --bins 50 \
        > "$RES/marginals.csv" 2>>"$LOG" && echo "  wrote $RES/marginals.csv" | tee -a "$LOG"
    if command -v python3 >/dev/null 2>&1; then
        python3 validation/plot_bisection.py --case "$CASE" --res "$RES" --thr "$RHAT" \
            >> "$LOG" 2>&1 && echo "  wrote $RES/report.html" | tee -a "$LOG" \
            || echo "  (plot_bisection.py failed; CSV+TSV still available)" | tee -a "$LOG"
    else
        echo "  (no python3 on host; fetch $RES/*.csv and plot locally)" | tee -a "$LOG"
    fi
    date > "$RES/DONE"
    echo "=== DONE: $(date) ===  results in $RES/" | tee -a "$LOG"
}
trap finish EXIT

# --- endpoints ------------------------------------------------------------------------
run_point 0; V0=$LAST_VERDICT
run_point 1; V1=$LAST_VERDICT
echo "" | tee -a "$LOG"
echo "endpoint sanity: t=0 -> $V0 (expect CONVERGED),  t=1 -> $V1 (expect NOT)" | tee -a "$LOG"
if { [ "$V0" != "CONVERGED" ] || [ "$V1" != "NOT" ]; }; then
    echo "!! ENDPOINT SANITY FAILED" | tee -a "$LOG"
    if [ "$STRICT" = "1" ]; then
        echo "   STRICT=1 -> aborting before bisection (finalize + DONE still run)." | tee -a "$LOG"
        exit 2
    fi
    echo "   STRICT=0 -> continuing bisection anyway." | tee -a "$LOG"
fi

# --- bisection: lo converges, hi does not ---------------------------------------------
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
echo "convergence breaks at t* in ($lo, $hi]   (gamma=t, iso=${ISO_SLOPE}*t)" | tee -a "$LOG"
echo "BRACKET case=$CASE lo=$lo hi=$hi" | tee -a "$LOG"
# finish() runs on EXIT
