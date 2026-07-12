# AIS validation

Cross-checks the annealed importance sampler against a standard cycle walk: both
target the **same** measure, and we ask whether they produce the same distribution of
district compactness. The target is per-case:

- `ct`, `grid`: `get_log_spanning_forests` weight `gamma=0.25` **and**
  `get_isoperimetric_score` weight `0.3`, 0.1/0.9 proposal mixture.
- `small`: matches the AIS unit test (`test/test_annealed_importance_sampling.jl`) —
  `get_log_spanning_forests` weight `gamma=0.7`, **no isoperimetric penalty**, 0.5/0.5
  mixture. (The isoperimetric score is still *written* as an observable so the same
  comparison runs; it just isn't in the energy.)

- `run_case.jl` — samples one case with one method and writes an Atlas (per-district
  isoperimetric scores; AIS maps carry the log importance weight, cycle-walk maps
  carry weight 1). AIS anneals both energy weights linearly `0 -> target`.
- `analyze.jl` — reads both Atlases, sorts each plan's district scores into order
  statistics (rank 1 = most compact … rank K = least compact), and per rank compares
  the AIS (importance-weighted) marginal to the cycle-walk marginal: weighted mean,
  weighted variance, and total-variation distance of binned histograms. Each
  between-method TV is judged against a **noise floor** = the TV between the two
  halves of each run.
- `analyze_weights.jl` — importance-weight diagnostics for the AIS Atlases: log-weight
  statistics, ESS, weight concentration (`n50`/`n90`), and ASCII log-weight histograms.
  With no args it reads the preserved `output/validation/*_ais_a{4000,16000}.jsonl.gz`.
- `plot_weights.jl` — renders three self-contained SVG/PNG figures from the same
  weight files (centered log-weight densities, weight-concentration curves, and
  ESS vs σ against the `exp(−σ²)` law) into `output/validation/`.
- `analyze_path.jl` — for runs made with `run_case.jl --record-path`, shows **where along
  the annealing schedule the weight variance is injected** (across the recorded per-sample
  trajectories) and prints/plots a **suggested non-linear reschedule** (`u → γ/target`
  warp, saved to `path_<case>_reschedule.csv`).
- `hpc/` — scripts to run the larger **weight-collapse study** (CT 5-dist and NC 14-dist
  at 4k/16k/24k with path recording) on a big machine over a persistent SSH connection.
  See `hpc/README.md`.

Cases: `small`, `ct`, `grid`, and `nc` (NC 2020 precincts, 14 districts). Size knobs on
`run_case.jl`: `--samples`, `--anneal-steps`, `--base-steps`, `--path-points`,
`--record-path`.

## Results write-up

`RESULTS.md` is the full production report: per-case verdicts, the split-half
noise-floor methodology, runtimes, and the importance-weight analysis (with the figures
above and a pro/con assessment of whether the AIS chain is behaving well). Headline
outcome: **small validates exactly; grid validates once annealed enough (ESS 65%→90%,
gap → noise floor at `--anneal-steps 16000`); CT does not validate at 4k or 16k** — its
weights collapse (ESS ≈ 0.1–0.2%), diagnosing that the γ=0 base and the target are too
far apart for a short linear anneal, not a correctness bug.

## Cases

| `--case` | graph | districts |
|---|---|---|
| `small` | `test/test_graphs/4x4pct_2x2cnty.json` | 4 |
| `ct`    | `data/ct/CT_pct20.json` | 5 |
| `grid`  | `data/grid/grid_graph_10_by_10.json` | 3 |

## Run

Start Julia with threads (AIS anneals `ntasks` samples in parallel):

```bash
cd examples

# smoke test (tiny sizes, --dev): proves the pipeline, numbers are NOT meaningful
julia -t 4 validation/run_case.jl --case small --mode ais       --dev
julia -t 4 validation/run_case.jl --case small --mode cyclewalk  --dev
julia -t 4 validation/analyze.jl  --case small --dev --bins 15

# production (>=10,000 samples, 4000 annealing steps per sample) — drop --dev
julia -t 8 validation/run_case.jl --case ct --mode ais
julia -t 8 validation/run_case.jl --case ct --mode cyclewalk
julia -t 8 validation/analyze.jl  --case ct
```

## Sizes

`--dev` uses 200 samples / 200 annealing steps / 50 base steps between samples.
Production uses 10,000 samples / 4000 annealing steps / 500 base steps, and the cycle
walk collects 10,000 samples 1000 steps apart. Edit the `if dev … else … end` block
in `run_case.jl` to change these.

## Reading the output

- **ESS** (printed by both `run_case.jl` and `analyze.jl`) is the AIS *effective*
  sample size. If it is a small fraction of the sample count the importance weights
  have collapsed — usually too few annealing steps — and the comparison is unreliable
  regardless of the TV verdict.
- A rank is `ok` when its AIS-vs-cyclewalk TV sits within the finite-sample noise
  floor; `**DIFFERS**` flags a real discrepancy to investigate. At `--dev` sizes
  expect low ESS and unstable verdicts — that is why `--dev` is only a plumbing check.
