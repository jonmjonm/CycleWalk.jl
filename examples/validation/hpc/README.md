# Running the weight-collapse study on hamilton (or any big machine)

Goal: run AIS with **annealing-path recording** for CT (5 districts) and NC (14
districts) at several schedule lengths (4k / 16k / 24k …) on the 64-core machine, then
analyze where the importance-weight variance is injected and whether a non-linear
schedule can prevent the collapse.

`hamilton.math.duke.edu` = Intel Xeon Gold 6226R, **64 threads**, AVX-512.

## 0. One-time SSH setup (already done on this laptop)

`~/.ssh/config` has a `hamilton` block with connection multiplexing so you authenticate
**once** (password + Duo) and every later `ssh hamilton …` reuses the socket for 12 h:

```
Host hamilton hamilton.math.duke.edu
 HostName 152.3.30.52
 IdentityFile ~/.ssh/id_ed25519
 ControlMaster auto
 ControlPath ~/.ssh/controlmasters/%r@%h:%p
 ControlPersist 12h
 ServerAliveInterval 60
```

## 1. Open the master session (interactive — do this yourself)

```bash
ssh -fN hamilton        # type password, approve Duo; backgrounds and persists 12h
ssh -O check hamilton   # should print: Master running (pid=…)
```

## 2. Ship the code

```bash
cd CycleWalk.jl/examples
bash validation/hpc/sync.sh          # rsyncs CycleWalk.jl + AtlasIO.jl + LinkCutTreesAugmented.jl
                                     # to ~/MergeAnneledTemp/ (graphs included; results excluded)
```

## 3. Instantiate the Julia environment on hamilton (first time / after dep changes)

```bash
ssh hamilton 'cd ~/MergeAnneledTemp/CycleWalk.jl/examples && \
  (module load julia 2>/dev/null; julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()")'
```

If `MetropolizedForestRecom` fails to resolve (it is a registered, not path, dependency),
the machine may be missing the registry that hosts it — report the error and we will add
the registry or `dev` a source checkout.

## 4. Launch the study (disconnect-safe, under tmux)

```bash
ssh hamilton 'cd ~/MergeAnneledTemp/CycleWalk.jl/examples && \
  tmux new-session -d -s study "THREADS=60 CASES=\"ct nc\" ANNEAL=\"4000 16000 24000\" \
    SAMPLES=5000 PATH_POINTS=100 bash validation/hpc/run_collapse_study.sh \
    > validation/hpc/study.log 2>&1"'
```

Watch / detach:

```bash
ssh hamilton 'tmux attach -t study'     # Ctrl-b then d to detach
ssh hamilton 'tail -f ~/MergeAnneledTemp/CycleWalk.jl/examples/validation/hpc/study.log'
ssh hamilton 'tmux ls'                  # is it still running?
```

Knobs (environment variables to `run_collapse_study.sh`): `CASES`, `ANNEAL`, `SAMPLES`,
`BASE_STEPS`, `PATH_POINTS`, `THREADS`, `RUN_CYCLEWALK=1` (adds a cycle-walk baseline per
case). Leave a few cores free on a shared machine (`THREADS=60`, not 64).

### Rough cost

CT 16k was ~19 min on a 10-core laptop; NC (14 districts, 2650 nodes) is much heavier per
step. On 60 Xeon threads expect CT’s three lengths in a couple of hours and NC’s to
dominate — start with `SAMPLES=5000` (plenty for stable variance curves) and a subset of
`ANNEAL` if you want a quick first pass, then scale up.

## 5. Bring results back and analyze locally

```bash
bash validation/hpc/fetch.sh                     # atlases + figures + reschedule CSVs + logs
julia validation/analyze_path.jl --case ct       # variance-injection + suggested reschedule
julia validation/analyze_path.jl --case nc
julia validation/analyze_weights.jl output/validation/{ct,nc}_ais_path_a*.jsonl.gz
```

`analyze_path.jl` writes, per case, into `output/validation/`:
`path_<case>_variance.svg` (where variance is injected), `path_<case>_varaccum.svg`,
`path_<case>_mean.svg`, `path_<case>_reschedule.svg`, and `path_<case>_reschedule.csv`
(the `u → γ/target` warp to drive a non-linear `modify_measure!`).

## Files

- `run_collapse_study.sh` — the batch runner (cases × anneal lengths, path recording, then analysis).
- `sync.sh` / `fetch.sh` — push code / pull results over the persistent SSH socket.
- `../run_case.jl` — the driver; size knobs `--samples --anneal-steps --base-steps --path-points`, case `--case nc`.
- `../analyze_path.jl` — variance-injection analysis + suggested reschedule.
