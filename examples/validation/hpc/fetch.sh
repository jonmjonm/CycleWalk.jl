#!/usr/bin/env bash
# Pull the study outputs (atlases, figures, reschedule CSVs, logs) back from the remote.
#
#   REMOTE=hamilton DEST='~/MergeAnneledTemp' bash validation/hpc/fetch.sh
#
# By default the big path atlases are included; set ATLASES=0 to fetch only the small
# figures/CSVs/logs (the atlases can be ~50-100 MB each for CT/NC).
set -euo pipefail
REMOTE=${REMOTE:-hamilton}
DEST=${DEST:-'~/MergeAnneledTemp'}
ATLASES=${ATLASES:-1}
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"        # -> CycleWalk.jl/examples
RDIR="$DEST/CycleWalk.jl/examples/output/validation"

mkdir -p "$LOCAL/output/validation" "$LOCAL/validation/hpc/logs"
echo "fetching study outputs from $REMOTE:$RDIR"
# small artifacts always
rsync -az --include='*/' \
    --include='path_*' --include='*_reschedule.csv' --include='study.log' \
    --exclude='*' \
    "$REMOTE:$RDIR/" "$LOCAL/output/validation/" || true
if [ "$ATLASES" = "1" ]; then
    echo "  + path atlases (large)"
    rsync -az --include='*_ais_path_a*.jsonl.gz' --include='*_cyclewalk.jsonl.gz' \
        --exclude='*' "$REMOTE:$RDIR/" "$LOCAL/output/validation/" || true
fi
rsync -az "$REMOTE:$DEST/CycleWalk.jl/examples/validation/hpc/logs/" \
    "$LOCAL/validation/hpc/logs/" 2>/dev/null || true
echo "done -> $LOCAL/output/validation/"
echo "regenerate figures locally with:  julia validation/analyze_path.jl --case ct   (and --case nc)"
