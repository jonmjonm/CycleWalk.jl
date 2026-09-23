# `run_ais_toml.jl` and its configuration file

[`examples/run_ais_toml.jl`](../examples/run_ais_toml.jl) runs annealed importance
sampling (AIS): a **serial base chain** samples the base measure, and each retained
sample is deep-copied and annealed toward the `[measure]` target on its own task while
its log importance weight is accumulated. It takes every setting from a TOML file, the
same way [`run_cyclewalk_toml.jl`](../examples/run_cyclewalk_toml.jl) and
[`run_asmc_toml.jl`](../examples/run_asmc_toml.jl) do.
[`examples/run_ais_ct.jl`](../examples/run_ais_ct.jl) is the same run expressed
directly in Julia, with no config file.

This document covers what's specific to AIS — the `[ais]` table, output, and the
scaling ceiling. `[plans]` and `[measure]` are read the same way every other TOML
runner reads them — see
[`run_cyclewalk_toml.md`](run_cyclewalk_toml.md#plans--the-map-and-the-districts) for
those, including `[plans.derive]` (computed node columns) and `[[measure.energy]]
context = ["base_graph"]`, both of which this runner now supports too.

- [Quick start](#quick-start)
- [A minimal configuration](#a-minimal-configuration)
- [What the script does with it](#what-the-script-does-with-it)
- [`[ais]` — steps, samples, tempering](#ais--steps-samples-tempering)
- [`linear` and `path` tempering](#linear-and-path-tempering)
- [The scaling ceiling: why `ntasks` has a limit](#the-scaling-ceiling-why-ntasks-has-a-limit)
- [Output: log importance weights](#output-log-importance-weights)
- [Error messages and what they mean](#error-messages-and-what-they-mean)

## Quick start

```bash
cd examples
julia -t 4 run_ais_toml.jl toml/param_ais_ct.toml
julia -t 4 run_ais_toml.jl toml/param_ais_ct.toml --overwrite
```

`-t` threads run concurrent annealing tasks (see
[the scaling ceiling](#the-scaling-ceiling-why-ntasks-has-a-limit) for how many are
actually useful). `--overwrite` is the only flag this runner reads besides the config
path — there is no per-key `--set`/named-flag override here, unlike
`run_cyclewalk_toml.jl` and `run_asmc_toml.jl`.

## A minimal configuration

This is [`examples/toml/param_ais_ct.toml`](../examples/toml/param_ais_ct.toml):

```toml
[plans]
pop_dev = 0.02
num_dists = 5
node_data = ["COUNTY", "NAME", "POP20", "area", "border_length"]
geo_units = ["NAME"]
pop_col = "POP20"
area_col = "area"
node_border_col = "border_length"
edge_perimeter_col = "length"
map_directory = ["data", "ct"]
map_file = "CT_pct20.json"

[measure]
gamma = 1.0
iso_weight = 0.3

[[measure.energy]]
name = "get_log_spanning_forests"
weight = "gamma"
# weight_start = 0.0

[[measure.energy]]
name = "get_isoperimetric_score"
weight = "iso_weight"
# weight_start = 0.0

[ais]
total_steps = 1.0e4
base_steps_per_sample = 500
steps_per_annealing = 2000

[run]
thread_id = 1
atlasNameBase = "ct"
outputDirectory = ["output", "ct"]
two_cycle_walk_frac = 0.1
writer_stats = ["get_isoperimetric_scores"]
compress = "gz"
ntasks = 4
blas_threads = 0
output_districting = true
io_mode = "w"
description = "CT annealed importance sampling example"
```

Three tables, all required: `[plans]`, `[measure]`, `[ais]`. `[run]` carries output and
performance knobs.

## What the script does with it

```
graph JSON ──▶ BaseGraph ──▶ Graph ──▶ base chain, at weight_start   [plans]
   [plans]                            + PopulationConstraint
                                                                     [measure]
      │  base_steps_per_sample MH steps (serial)
      ▼
   sample i ─┬─▶ deep-copy ──▶ anneal steps_per_annealing steps ──▶ log weight  [ais]
             │                 (own task, own rng, own deep-copied measure)
      │  (next sample drawn while sample i is still annealing)
      ▼
   total_steps ÷ base_steps_per_sample samples, in base-chain order   [run]
```

1. **Graph, constraint, proposal** — identical to every other runner (see
   `run_cyclewalk_toml.md`).
2. **Target measure** — built from `[[measure.energy]]` exactly as elsewhere; `weight`
   is each energy's value at the annealing target (t=1), `weight_start` (default 0) its
   value at t=0, which the base chain must be able to mix under.
3. **Tempering path** — `[ais] temper` picks how each energy's weight moves from t=0 to
   t=1; see [below](#linear-and-path-tempering). Built with the same `AnnealPath` seam
   annealed SMC and parallel tempering use (`anneal_path.jl`).
4. **Base chain** — runs on the working `partition` under the base measure for
   `base_steps_per_sample` MH steps between samples, `total_steps` steps in all, so
   `total_steps ÷ base_steps_per_sample` samples are drawn. This chain is **serial**.
5. **Annealing** — each drawn sample is deep-copied and annealed for
   `steps_per_annealing` steps on its own task (own RNG, own deep-copied measure), while
   its log importance weight accumulates (`e_before − e_after` in energy, i.e.
   `log[ν_new/ν_old]`, at every step the annealing schedule advances). Runs on `ntasks`
   concurrent tasks — give the process threads with `-t`.
6. **Write** — one map per sample, `"sample<i>"`, with its log importance weight as the
   map's weight, landing on disk in base-chain order regardless of which annealing task
   finishes first.

## `[ais]` — steps, samples, tempering

| Key | Meaning |
|---|---|
| `total_steps` | Total base-chain steps. Number of samples drawn is `total_steps ÷ base_steps_per_sample`. |
| `base_steps_per_sample` | Base-chain MH steps between retained samples (mixing budget for the base chain). |
| `steps_per_annealing` | Annealing MH steps applied to each retained sample. |
| `temper` | `"linear"` (default) or `"path"` — see below. |

`[run]` carries `ntasks` (default `Threads.nthreads()`, concurrent annealing tasks) and
`blas_threads`. Whenever `ntasks > 1`, `run_annealed_importance_sampling!`
unconditionally pins BLAS to 1 thread for the duration of the run — several tasks each
spawning their own BLAS pool for the spanning-forest energy's per-district
log-determinant would oversubscribe the machine — and restores the prior BLAS thread
count when it finishes; `blas_threads` cannot override this pin while `ntasks > 1`.
What `blas_threads` (default `0`, meaning "leave BLAS's own default alone") actually
controls: the thread count set *before* the run starts, which only matters when
`ntasks == 1` (nothing pins BLAS in that case, so this value holds for the whole run),
or as the value BLAS is left at afterward when `ntasks > 1`.

## `linear` and `path` tempering

Identical mechanism to parallel tempering's `[pt] temper`, with `t` (not β) as the
schedule variable — see
[`run_pt_toml.md`'s "Two tempering modes"](run_pt_toml.md#two-tempering-modes-linear-and-path)
for the full `weight_start` / `weight_path` reference; it is not PT-specific, it is the
shared `AnnealPath` seam every annealing-based runner uses.

`temper = "linear"` ramps each energy from its `weight_start` (default `0.0`) straight
to `weight`. `temper = "path"` instead follows each energy's own `weight_path`
expression in `t`; an energy with no `weight_path` holds constant at `weight` the whole
run.

## The scaling ceiling: why `ntasks` has a limit

The base chain is **serial** — sample `i+1` can't be drawn until sample `i`'s
`base_steps_per_sample` steps finish — so by Amdahl's law the speedup from `ntasks` is
capped at

```
(base_steps_per_sample + steps_per_annealing) / base_steps_per_sample
```

independent of how many threads you give it. With the example config's
`base_steps_per_sample=500` and `steps_per_annealing=2000` that ceiling is 5×, so
`ntasks` beyond 5 buys nothing. Raise `steps_per_annealing` (better annealing *and* more
parallel headroom) or lower `base_steps_per_sample` (cheaper, but more correlated base
samples) to lift it. This is why parallel tempering and annealed SMC — which parallelize
*within* a step across replicas/particles instead — scale further; AIS's structure is
different by design (a single base-chain trajectory).

## Output: log importance weights

Each sample's map carries its log importance weight as the map's weight (`Writer`
constructed with `weight_type=Float64`). The script prints, on completion:

```
collected <N> samples
log weights: min=... max=...
effective sample size: <ESS> of <N>
```

ESS (`(Σw)²/Σw²` on the exponentiated, max-shifted weights) tells you how many of the
`N` samples are doing the work — a small ESS relative to `N` means the target measure
is far from the base and most weight sits on a few samples; consider more annealing
steps or a `path` schedule tuned to spend more time where the weight variance is
highest.

## Error messages and what they mean

- `ais.temper must be "linear" or "path", got "..."` — typo'd or unsupported mode name.
- `[measure] defines a parameter named "t", which is reserved for weight_path
  expressions` — `t` is checked unconditionally while parsing `[measure]`, before
  `[ais] temper` is even read, so this fires under `temper = "linear"` too; rename the
  parameter.
- `energy "..." has both weight_start and weight_path` — pick one tempering mode per
  energy.
- `map file not found: ...` — `[plans] map_directory`/`map_file` don't resolve to a
  real file (paths are relative to wherever you run the script from — normally
  `examples/`).
- `<path> already exists, and this run would truncate it. Pass --overwrite ...` — an
  Atlas of that name already exists; pass `--overwrite`, or change `thread_id` /
  `atlasNameBase` / the measure parameters so the run gets a new name.
