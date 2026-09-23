# `run_pt_toml.jl` and its configuration file

[`examples/run_pt_toml.jl`](../examples/run_pt_toml.jl) runs parallel tempering: a
ladder of replicas ("rungs"), one per point on a β-lattice from the untempered base
measure (β=0) to the configured target (β=1), swapping adjacent rungs on a
deterministic even/odd schedule. It takes every setting from a TOML file, the same
way [`run_cyclewalk_toml.jl`](../examples/run_cyclewalk_toml.jl) and
[`run_asmc_toml.jl`](../examples/run_asmc_toml.jl) do.

This document covers what's specific to PT — the `[pt]` table, the two tempering
modes, and output/heat-bath specifics. **`[plans]` and `[measure]` are identical to
every other TOML runner** — see
[`run_cyclewalk_toml.md`](run_cyclewalk_toml.md#plans--the-map-and-the-districts)
for those.

- [Quick start](#quick-start)
- [A minimal configuration](#a-minimal-configuration)
- [What the script does with it](#what-the-script-does-with-it)
- [`[pt]` — the ladder, the schedule, the backend](#pt--the-ladder-the-schedule-the-backend)
- [Two tempering modes: `linear` and `path`](#two-tempering-modes-linear-and-path)
- [`[pt.heat_bath]` — optional independent exchanges](#ptheat_bath--optional-independent-exchanges)
- [Output files](#output-files)
- [Overriding on the command line](#overriding-on-the-command-line)
- [Sizing `n_rungs` to your machine](#sizing-n_rungs-to-your-machine)
- [Error messages and what they mean](#error-messages-and-what-they-mean)

## Quick start

```bash
cd examples
julia -t 8 run_pt_toml.jl toml/param_pt_grid.toml
```

Give it `-t N` matching (or exceeding) `[pt] n_rungs` — the threaded backend spawns
one task per rung each round, so threads beyond `n_rungs` sit idle. See
[Sizing `n_rungs` to your machine](#sizing-n_rungs-to-your-machine).

## A minimal configuration

This is [`examples/toml/param_pt_grid.toml`](../examples/toml/param_pt_grid.toml):

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

[pt]
n_rungs       = 8
lattice       = "linear"
swap_interval = 100
n_rounds      = 200
init_steps    = 200
backend       = "threaded"
temper        = "linear"
write_rungs   = "target"
output_every  = 1

[run]
atlasNameBase = "gridtest"
outputDirectory = ["output", "validation"]
two_cycle_walk_frac = 0.1
writer_stats = ["get_isoperimetric_scores", "get_log_spanning_trees"]
compress = "gz"
seed = 87654321
description = "grid PT smoke"
```

Three tables, all required: `[plans]`, `[measure]`, `[pt]`, `[run]`.

## What the script does with it

```
graph JSON ──▶ BaseGraph ──▶ Graph ──▶ M replicas, one per BetaLattice rung
   [plans]                            + PopulationConstraint          [pt]
                                                                     [measure]
      │
      ├── init_steps: each replica burns in at its OWN rung's measure
      │
      └── for round in 1:n_rounds
            advance M replicas swap_interval MH steps each (parallel)
            swap adjacent rungs (deterministic even/odd)
            [pt.heat_bath], if configured, fires on the rung the swap
              pattern leaves idle
            record diagnostics; emit maps per write_rungs         [run]
```

1. **Graph, constraint, proposal** — identical to every other runner (see
   `run_cyclewalk_toml.md`).
2. **Target measure** — built from `[[measure.energy]]` exactly as elsewhere; `weight`
   is each energy's value at β=1.
3. **Tempering path** — `[pt] temper` picks how each energy's weight moves from β=0 to
   β=1. See [below](#two-tempering-modes-linear-and-path).
4. **β-lattice** — `[pt] lattice` (`"linear"` or `"geometric"`, or an explicit `betas
   = [...]` array) places `n_rungs` points in `[0, 1]`, rung 1 hottest (β=0, the
   untempered base) and rung `n_rungs` coldest (β=1, the target). Downstream analysis
   wants rung `n_rungs`'s samples.
5. **Backend** — `"serial"` (a plain loop) or `"threaded"` (`Threads.@spawn`, one task
   per rung, bit-identical to serial for the same seed — see
   `docs/pt_profiling_notes.md` for scaling data). `"distributed"` is not implemented
   yet.
6. **Run** — `run_parallel_tempering!` with `init_steps` burn-in, then `n_rounds`
   rounds of `swap_interval` MH steps + one swap attempt per adjacent pair. Reports
   swap rates, round trips, and mean straggler gap when it finishes.

## `[pt]` — the ladder, the schedule, the backend

| Key | Default | Meaning |
|---|---|---|
| `n_rungs` | 8 | Rung count. Size this to your thread count — see below. Overridden by an explicit `betas` array, if given. |
| `lattice` | `"linear"` | `"linear"` (evenly spaced) or `"geometric"` (evenly spaced in *temperature* — needs `beta_min`). |
| `beta_min` | 0.05 | Only used when `lattice = "geometric"`: the hottest rung's β (cannot be exactly 0 for a geometric ladder). |
| `betas` | — | Explicit `[0.0, 0.3, 0.7, 1.0]`-style array, overriding `lattice`/`n_rungs`/`beta_min` entirely. Must be strictly increasing, end at exactly `1.0`. |
| `swap_interval` | 500 | MH steps per rung between swap rounds. |
| `n_rounds` | 2000 | Number of swap rounds. Total MH steps per rung is `swap_interval * n_rounds`. |
| `init_steps` | 2000 | Burn-in steps per rung, at that rung's own β, before swapping starts. `0` is legal but warns — the ladder starts fully correlated (every replica an identical clone). |
| `backend` | `"threaded"` | `"serial"` or `"threaded"`. |
| `temper` | `"linear"` | `"linear"` or `"path"` — see next section. |
| `write_rungs` | `"target"` | `"target"` (one Atlas, the coldest rung — what you sample from), `"all"` (one Atlas per rung, diagnostic), or `"none"`. |
| `output_every` | 1 | Emit a map every N swap rounds (`write_rungs != "none"` only). |

## Two tempering modes: `linear` and `path`

Both ramp every energy's weight from β=0 to β=1 along the SAME shared β (the
`BetaLattice` value a rung sits at is exactly the `t` an `AnnealPath` is evaluated
at — one variable, two names for the two contexts it's used in).

**`temper = "linear"`** (the roadmap's `"all"`/`"gamma"`/`"explicit"` modes, unified):
each energy has its own optional `weight_start` (default `0.0`) and moves in a
straight line to `weight`:

```
weight_k(β) = weight_start_k + β · (weight_k − weight_start_k)
```

Leave `weight_start` off every energy and every term ramps from 0 together (what the
roadmap called `"all"`). Set `weight_start = "iso_weight"` (i.e. equal to the target)
on the terms you don't want to move, and leave the rest at the default 0 — that's the
roadmap's `"gamma"` case, spelled out explicitly instead of hardcoding one energy's
name.

```toml
[[measure.energy]]
name = "get_log_spanning_forests"
weight = "gamma"
# weight_start defaults to 0 — ramps with everything else

[[measure.energy]]
name = "get_isoperimetric_score"
weight = "iso_weight"
weight_start = "iso_weight"   # already at target, does not move
```

**`temper = "path"`**: each energy gets its own `weight_path` — an arbitrary
expression in `t` (the same restricted arithmetic grammar as `weight`/`weight_start`:
`+ - * / ^`, numeric literals, and named `[measure]` parameters, never `eval`ed —
`t` is additionally available, and reserved: a `[measure]` parameter literally named
`t` is a config error). An energy with no `weight_path` holds constant at its
`weight` (present in the measure the whole run, untempered) — you can temper only
the energies that need it.

```toml
[[measure.energy]]
name = "get_log_spanning_forests"
weight = "gamma"
weight_path = "gamma*t"          # linear — same as temper="linear" would give it

[[measure.energy]]
name = "get_isoperimetric_score"
weight = "iso_weight"
weight_path = "iso_weight*t^2"   # held back until late (t^2 < t for t in (0,1))
```

`weight_path` is not restricted to monotonic ramps from 0 to the target — an
expression like `"mcd_weight*(1-t)"` (active at β=0, fading to 0 by β=1) is legal.
The only thing this changes is interpretation: the "hot rung samples the untempered
base" reading only holds for the conventional 0→target shape; a path that starts
nonzero means the hot rung's measure isn't literally ν₀ anymore, by design. Nothing
about the sampler requires it to be — `run_metropolis_hastings!` Metropolizes
correctly regardless of the weight values.

`weight_path` expressions are parsed and validated ONCE, when the measure is built
(not re-parsed every round) — a typo or unknown name fails immediately at startup,
not partway through a long run.

Every parameter a `weight_path` reads is tagged into the atlas filename, same as
`weight`/`weight_start` — a parameter referenced only inside a `weight_path` still
can't collide two runs onto the same output path.

## `[pt.heat_bath]` — optional independent exchanges

Omit this table entirely to disable. When present, a rung is exchanged against an
independent-ish draw from a *different*, already-sampled Atlas on the rounds the
swap pattern leaves it idle (free — no extra cost):

```toml
[pt.heat_bath]
source    = "output/reference.jsonl.gz"   # must have been written with a config_file
                                          # (any of these TOML runners does this)
rung      = 1                             # default 1 (the hottest)
burn_in   = 10000                         # maps to skip before drawing samples
n_samples = 1200                          # default: n_rounds÷2 + 20
```

`source`'s header must carry an embedded TOML config (`Writer`'s `config_file`
argument — every runner in this directory sets this automatically), since that's how
the bath's own reference measure is rebuilt. Size `n_samples` to at least
`n_rounds ÷ 2` (a bath at rung 1 fires on roughly every other round); the run errors
immediately, at startup, if the source atlas doesn't have enough maps past `burn_in`
to satisfy that.

## Output files

- `write_rungs = "target"` → one Atlas, the coldest rung's stream — the samples you
  actually want.
- `write_rungs = "all"` → one Atlas per rung, filename suffixed `_beta1` … `_betaN`,
  sharing the same base name/tag. Diagnostic: inspect how the whole ladder is
  behaving, not just the target.
- `write_rungs = "none"` → no Atlas (tuning/benchmark runs; diagnostics are still
  available via the printed swap-rate/round-trip/straggler-gap report).

The filename tag (like every runner here) encodes the settings that define the run —
`n_rungs`, `swap_interval`, `n_rounds`, thread count, backend, temper mode, and every
`[measure]` parameter actually referenced — so two runs that differ in any of them
never collide, and a run that would overwrite an existing Atlas stops unless you pass
`--overwrite`.

## Overriding on the command line

```bash
julia -t 16 run_pt_toml.jl toml/param_pt_grid.toml --n-rungs 16 --swap-interval 300
julia -t 8  run_pt_toml.jl toml/param_pt_grid.toml --temper path --backend serial
```

Flags: `--n-rungs`, `--lattice`, `--beta-min`, `--swap-interval`, `--n-rounds`,
`--init-steps`, `--backend`, `--temper`, `--write-rungs`, `--output-every`,
`--gamma`, `--iso-weight`, `--seed`, `--overwrite`.

## Sizing `n_rungs` to your machine

`ThreadedBackend` spawns exactly `n_rungs` tasks per swap round — one per replica.
Extra threads beyond `n_rungs` do nothing; fewer threads than `n_rungs` just means
some replicas queue behind others within a round (still correct, just not fully
parallel). Profiling on a 64-core machine (`docs/pt_profiling_notes.md`) found no
scaling ceiling up to 64 threads *once `n_rungs` was set to match* — the earlier
"flattens by 8 threads" result was an artifact of a fixed small `n_rungs`, not a
property of the backend. Rule of thumb: **set `n_rungs` to (at least) the thread
count you're running with.**

## Error messages and what they mean

- `pt.backend must be "serial" or "threaded" for now ("distributed" is not
  implemented yet...)` — the Distributed backend (roadmap task 9) isn't built.
- `pt.temper must be "linear" or "path"` / `pt.write_rungs must be "target", "all",
  or "none"` — typo'd or unsupported mode name.
- `pt.lattice must be "linear" or "geometric" (or supply pt.betas explicitly)` —
  same, for the lattice.
- `[measure] defines a parameter named "t", which is reserved for weight_path
  expressions` — rename the parameter; `t` is the tempering fraction, bound
  automatically wherever a `weight_path` expression runs.
- `energy "..." has both weight_start and weight_path` — pick one tempering mode per
  energy; they belong to `"linear"` and `"path"` respectively and don't combine.
- `heat bath measure at ... scores ..., which the ensemble's target measure does
  not` — the `[pt.heat_bath] source` atlas's measure has an energy your target
  measure doesn't; it would have no slot in the cached potentials and be silently
  dropped from the exchange, so this is refused instead. Add that energy to your
  target too, or use a bath source built with a matching (or smaller) measure.
- `heat bath at rung N has no unused samples left` — `n_samples` was too small for
  how long the run went (or the rung fired more often than `n_rounds ÷ 2` implies);
  raise `n_samples` or `burn_in`'s source atlas needs to be longer.
- `pt.n_rungs=N but Julia has T thread(s)` (a warning, not an error) — see
  [Sizing `n_rungs` to your machine](#sizing-n_rungs-to-your-machine).
