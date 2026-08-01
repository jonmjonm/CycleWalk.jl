# `run_asmc_toml.jl` and its configuration file

[`examples/run_asmc_toml.jl`](../examples/run_asmc_toml.jl) runs annealed sequential
Monte Carlo (SMC): a **population of particles** is jointly tempered from the base
measure to the `[measure]` target, resampling when the population's effective sample
size (ESS) drops and rejuvenating with MH moves between tempering steps — the whole
population moves together, in parallel, unlike AIS's single serial base chain. It takes
every setting from a TOML file, the same way
[`run_cyclewalk_toml.jl`](../examples/run_cyclewalk_toml.jl) and
[`run_pt_toml.jl`](../examples/run_pt_toml.jl) do, plus CLI flags that override it.

This document covers what's specific to annealed SMC — the `[smc]` table, the two
schedules, and post-anneal amplification. `[plans]` and `[measure]` are read the same
way every other TOML runner reads them — see
[`run_cyclewalk_toml.md`](run_cyclewalk_toml.md#plans--the-map-and-the-districts) for
those, including `[plans.derive]` (computed node columns) and `[[measure.energy]]
context = ["base_graph"]`, both of which this runner now supports too.

- [Quick start](#quick-start)
- [A minimal configuration](#a-minimal-configuration)
- [What the script does with it](#what-the-script-does-with-it)
- [`[smc]` — population, schedule, rejuvenation](#smc--population-schedule-rejuvenation)
- [`fixed` and `adaptive` schedules](#fixed-and-adaptive-schedules)
- [`linear` and `path` tempering](#linear-and-path-tempering)
- [Post-anneal amplification: `collect_steps`](#post-anneal-amplification-collect_steps)
- [Output files](#output-files)
- [Overriding on the command line](#overriding-on-the-command-line)
- [Error messages and what they mean](#error-messages-and-what-they-mean)

## Quick start

```bash
cd examples
julia -t 2  run_asmc_toml.jl toml/param_annealed_smc_grid.toml
julia -t 48 run_asmc_toml.jl toml/param_annealed_smc_nc.toml
```

Rejuvenation is parallelized across particles with `Threads.@threads` — give it `-t N`;
unlike AIS there's no serial-chain ceiling on how many threads help (more particles can
always use more threads, up to `particles` itself).

## A minimal configuration

This is
[`examples/toml/param_annealed_smc_grid.toml`](../examples/toml/param_annealed_smc_grid.toml):

```toml
[plans]
pop_dev = 0.02
num_dists = 3
node_data = ["node_name", "population", "area", "border_length", "county"]
geo_units = ["node_name"]
pop_col = "population"
area_col = "area"
node_border_col = "border_length"
edge_perimeter_col = "length"
map_directory = ["data", "grid"]
map_file = "grid_graph_10_by_10.json"

[measure]
gamma = 0.25
iso_weight = 0.3

[[measure.energy]]
name = "get_log_spanning_forests"
weight = "gamma"

[[measure.energy]]
name = "get_isoperimetric_score"
weight = "iso_weight"

[smc]
schedule = "fixed"
particles = 16
blocks = 10
rejuv = 20
init_steps = 50
ess_frac = 0.5
collect_steps = 100
collect_every = 50
resample_before_collect = true

[run]
atlasNameBase = "gridtest"
outputDirectory = ["output", "validation"]
two_cycle_walk_frac = 0.1
writer_stats = ["get_isoperimetric_scores", "get_log_spanning_trees"]
compress = "gz"
seed = 87654321
description = "grid SMC smoke"
```

Three tables, all required: `[plans]`, `[measure]`, `[smc]`. `[run]` carries output and
seeding. [`param_annealed_smc_nc.toml`](../examples/toml/param_annealed_smc_nc.toml) is
a larger (14-district) worked example with every `[smc]` key commented in place.

## What the script does with it

```
graph JSON ──▶ BaseGraph ──▶ Graph ──▶ N particles                     [plans]
   [plans]                            + PopulationConstraint          [measure]
      │
      ├── init_steps: population burns in at the BASE measure (t=0), independently
      │   per particle — this is what makes the population diverse at t=0
      │
      └── while not done(schedule)
            t = next_t(schedule, ...)             fixed grid, or adaptive bisection
            incremental log-weight per particle    (pre-move state)
            resample if ESS below threshold        (fixed: conditional; adaptive: always)
            rejuvenate: `rejuv` MH steps per particle at t, in parallel  [smc]
      │
      └── (optional) collect_steps > 0: keep sampling the TARGET past t=1,
            emit every particle every collect_every steps                [run]
```

1. **Graph, constraint, proposal** — identical to every other runner (see
   `run_cyclewalk_toml.md`).
2. **Target measure** — built from `[[measure.energy]]` exactly as elsewhere; `weight`
   is each energy's value at t=1, `weight_start` (default 0) its value at t=0.
3. **Tempering path** — `[smc] temper` picks how each energy's weight moves from t=0 to
   t=1; see [below](#linear-and-path-tempering). Built on the same `AnnealPath` seam
   parallel tempering and AIS use (`anneal_path.jl`).
4. **Initialization** — `partition` is cloned `particles` times and, if `init_steps >
   0`, each clone burns in independently under the base measure. Without this every
   particle starts as an identical clone (see
   [`AdaptiveTempering`'s requirement](#fixed-and-adaptive-schedules) below).
5. **Schedule** — `FixedSchedule` (precomputed t-grid, resample when ESS drops below
   `ess_frac * N`) or `AdaptiveTempering` (bisect for the next t that holds ESS at
   `ess_target * N`, resampling every step). See
   [below](#fixed-and-adaptive-schedules).
6. **Main loop** — per block: compute the incremental log-weight from the cached
   per-particle potentials, resample if called for, rejuvenate every particle `rejuv`
   MH steps at the new t (parallel across particles via `Threads.@threads`), refresh
   potentials.
7. **Amplification** — with `collect_steps > 0`, after reaching t=1 the sampler keeps
   applying MH moves at the fixed target measure and periodically records every
   particle, yielding more than `particles` output samples. See
   [below](#post-anneal-amplification-collect_steps).

## `[smc]` — population, schedule, rejuvenation

| Key | Default | Meaning |
|---|---|---|
| `schedule` | `"fixed"` | `"fixed"` or `"adaptive"` — see below. |
| `temper` | `"linear"` | `"linear"` or `"path"` — see below. |
| `particles` | 512 | Population size N. |
| `blocks` | 100 | `fixed` schedule only: number of t-grid blocks (0→1, so `blocks+1` grid points). |
| `rejuv` | 300 | MH rejuvenation steps per particle, per block. |
| `init_steps` | 2000 | Base-measure burn-in per particle before annealing starts. **Required (`> 0`) with `schedule = "adaptive"`** — see below. |
| `ess_frac` | 0.5 | `fixed` schedule: resample when ESS `< ess_frac * N`. |
| `ess_target` | 0.5 | `adaptive` schedule: target ESS fraction the bisection holds each step to. |
| `collect_steps` | 0 | Post-t=1 target-measure sampling length; `0` disables amplification (one map per particle). |
| `collect_every` | 100 | Recording cadence within the post-t=1 chains. |
| `resample_before_collect` | `true` | Resample to equal weights at t=1 before amplifying, so no heavy particle dominates its whole collected chain. |

## `fixed` and `adaptive` schedules

**`FixedSchedule`** (`schedule = "fixed"`): a precomputed grid `range(0, 1;
length=blocks+1)`. Resamples whenever ESS drops below `ess_frac * N`. Simple and safe
to use even with a degenerate (unburned) population — it just loses diversity rather
than erroring, so `init_steps` is merely *recommended*, not required.

**`AdaptiveTempering`** (`schedule = "adaptive"`): chooses the next t by bisection so
the population's ESS lands at `ess_target * N`, resampling every block (the "coupled"
form). This makes `init_steps > 0` a hard requirement: with an unburned population every
particle is identical, so every incremental weight is equal, ESS ≡ N regardless of t,
and the bisection jumps straight to t=1 on that false signal — one block, no annealing.
`run_annealed_smc!` raises an `ArgumentError` rather than silently doing this.

## `linear` and `path` tempering

Identical mechanism to parallel tempering's `[pt] temper`, with `t` (not β) as the
schedule variable — see
[`run_pt_toml.md`'s "Two tempering modes"](run_pt_toml.md#two-tempering-modes-linear-and-path)
for the full `weight_start` / `weight_path` reference; it is not PT-specific, it is the
shared `AnnealPath` seam every annealing-based runner uses.

`temper = "linear"` ramps each energy from its `weight_start` (default `0.0`) straight
to `weight`. `temper = "path"` instead follows each energy's own `weight_path`
expression in `t`; an energy with no `weight_path` holds constant at `weight` the whole
run. For a `LinearPath`, incremental weights are computed from each particle's cached
`phi` (path-independent per-term energies) with no fresh energy evaluation — an
arithmetic dot product — which is also what makes `AdaptiveTempering`'s bisection cheap.

## Post-anneal amplification: `collect_steps`

With `collect_steps = 0` (default), the writer records one map per particle — the
annealing-final state at t=1, weighted by its final `logW` (0 after a final resample).

With `collect_steps > 0`, after the schedule finishes:

1. If `resample_before_collect` (default `true`), resample to equal weights first, so
   the collected chains all start unweighted.
2. Configure the measure to the target (t=1) and keep applying MH moves, `collect_every`
   steps at a time, writing every particle's state after each block:
   `"particle<i>_s<step>"`.

This yields `particles * (collect_steps ÷ collect_every)` samples without enlarging the
population — the N particles are independent/well-dispersed by construction, but states
*within* one particle's post-t=1 chain are autocorrelated, so tune `collect_every` the
way you would tune a thinning interval on any MH chain.

## Output files

The filename tag encodes `particles`, `blocks`, `rejuv`, thread count, `temper` mode (if
`"path"`), `collect_steps`/`collect_every` (if amplifying), and every `[measure]`
parameter actually referenced — so two runs differing in any of these never collide, and
a run that would overwrite an existing Atlas stops unless you pass `--overwrite`.

## Overriding on the command line

```bash
julia -t 48 run_asmc_toml.jl toml/param_annealed_smc_nc.toml \
      --particles 2048 --rejuv 3000 --collect-steps 8000 --collect-every 500
```

Flags: `--schedule`, `--temper`, `--particles`, `--blocks`, `--rejuv`, `--init-steps`,
`--collect-steps`, `--collect-every`, `--no-resample-before-collect`, `--ess-frac`,
`--ess-target`, `--gamma`, `--iso-weight`, `--gamma-start`, `--iso-start`, `--seed`,
`--overwrite`. CLI flags win over the TOML file.

## Error messages and what they mean

- `smc.temper must be "linear" or "path", got "..."` — typo'd or unsupported mode name.
- `smc.schedule must be fixed or adaptive` — same, for the schedule.
- `run_annealed_smc! with AdaptiveTempering requires init_steps > 0: ...` — see
  [`fixed` and `adaptive` schedules](#fixed-and-adaptive-schedules); raise `init_steps`
  or switch to `schedule = "fixed"`.
- `[measure] defines a parameter named "t", which is reserved for weight_path
  expressions` — `t` is checked unconditionally while parsing `[measure]`, before
  `[smc] temper` is even read, so this fires under `temper = "linear"` too; rename the
  parameter.
- `energy "..." has both weight_start and weight_path` — pick one tempering mode per
  energy.
- `map file not found: ...` — `[plans] map_directory`/`map_file` don't resolve to a
  real file (paths are relative to wherever you run the script from — normally
  `examples/`).
