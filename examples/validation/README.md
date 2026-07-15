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
  With no args it looks for a default set of `output/validation/<case>_ais_a<steps>.jsonl.gz`
  files (produced by tagging `run_case.jl --mode ais` runs with `--tag a<steps>`); pass
  explicit Atlas paths as `ARGS` to analyze a different set.
- `plot_weights.jl` — renders three self-contained SVG/PNG figures from the same
  weight files (centered log-weight densities, weight-concentration curves, and
  ESS vs σ against the `exp(−σ²)` law) into `output/validation/`.
- `analyze_path.jl` — for runs made with `run_case.jl --record-path`, shows **where along
  the annealing schedule the weight variance is injected** (across the recorded per-sample
  trajectories) and prints/plots a **suggested non-linear reschedule** (`u → γ/target`
  warp, saved to `path_<case>_reschedule.csv`).
- `analyze_rhat.jl` / `analyze_convergence.jl` — per-district split-R-hat (Gelman-Rubin)
  and TV-vs-noise convergence checks across multiple independent chains, used to decide
  whether a cycle walk has mixed at a given point on the annealing schedule.
- `dump_marginals.jl` — dumps per-rank marginal histograms across a set of tagged runs to
  a CSV for downstream plotting.

Cases: `small`, `ct`, `grid`, and `nc` (NC 2020 precincts, 14 districts). Size knobs on
`run_case.jl`: `--samples`, `--anneal-steps`, `--base-steps`, `--path-points`,
`--record-path`.

Larger multi-chain studies (bisection convergence sweeps, weight-collapse sweeps across
annealing-schedule lengths) are driven by shell scripts that live outside this repo, since
they aren't part of the CycleWalk.jl package itself — they just call the scripts above.

## Cases

| `--case` | graph | districts |
|---|---|---|
| `small` | `test/test_graphs/4x4pct_2x2cnty.json` | 4 |
| `ct`    | `data/ct/CT_pct20.json` | 5 |
| `grid`  | `data/grid/grid_graph_10_by_10.json` | 3 |
| `nc`    | `test/test_graphs/NC_pct21.json` | 14 |

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
