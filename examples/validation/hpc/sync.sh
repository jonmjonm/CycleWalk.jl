#!/usr/bin/env bash
# Push the CycleWalk stack to a remote host, preserving the sibling layout the Manifest
# expects (CycleWalk.jl next to AtlasIO.jl and LinkCutTreesAugmented.jl). Graph JSONs are
# included; local results (output/) and compressed atlases are NOT (regenerated remotely).
#
#   REMOTE=hamilton DEST='~/MergeAnneledTemp' bash validation/hpc/sync.sh
#
# Requires a live SSH master (see README): run `ssh -fN hamilton` first.
set -euo pipefail
REMOTE=${REMOTE:-hamilton}
DEST=${DEST:-'~/MergeAnneledTemp'}
PARENT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # -> the MergeAnneledTemp checkout
REPOS=(CycleWalk.jl AtlasIO.jl LinkCutTreesAugmented.jl)

echo "syncing ${REPOS[*]}  ->  $REMOTE:$DEST"
ssh "$REMOTE" "mkdir -p $DEST"
for d in "${REPOS[@]}"; do
    [ -d "$PARENT/$d" ] || { echo "  missing $PARENT/$d, skipping"; continue; }
    echo "  $d/"
    rsync -az \
        --exclude '.git' --exclude 'output/' --exclude '*.jsonl.gz' \
        --exclude '.DS_Store' --exclude '*.pdf' \
        "$PARENT/$d/" "$REMOTE:$DEST/$d/"
done
echo "done. Next: remote instantiate, then launch the study (see README)."
