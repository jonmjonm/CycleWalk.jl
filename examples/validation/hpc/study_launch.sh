#!/usr/bin/env bash
# The exact launcher used for the hamilton weight-collapse study (CT 5-dist + NC 14-dist,
# 4k/16k/24k, path recording). Run it inside tmux so it survives disconnect:
#
#   ssh hamilton 'tmux new-session -d -s study \
#     "bash ~/MergeAnneledTemp/CycleWalk.jl/examples/validation/hpc/study_launch.sh"'
#
# JULIA="julia +1.12.6" pins the dev-matched Julia (juliaup channel) so the pinned
# Manifest (julia_version 1.12.6) is not re-resolved. Edit THREADS/SAMPLES/ANNEAL to taste.
cd ~/MergeAnneledTemp/CycleWalk.jl/examples
export JULIA="julia +1.12.6" THREADS=60 CASES="ct nc" ANNEAL="4000 16000 24000" \
       SAMPLES=3000 BASE_STEPS=500 PATH_POINTS=100
bash validation/hpc/run_collapse_study.sh > validation/hpc/study.log 2>&1
