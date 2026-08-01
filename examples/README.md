# CycleWalk examples

Runnable scripts, sample graphs, and configuration files for sampling redistricting
plans with CycleWalk. Everything here expects to be run **from this directory**, which
carries its own environment (`Project.toml`, with `CycleWalk` taken from the repository
root):

```bash
cd examples
julia --project=. run_cyclewalk_toml.jl toml/param_grid4x4.toml
```

## The scripts

| Script | What it does |
| --- | --- |
| [`run_cyclewalk_toml.jl`](run_cyclewalk_toml.jl) | **The general runner.** Everything comes from a TOML config file — see below. |
| [`run_cyclewalk_ct.jl`](run_cyclewalk_ct.jl) | The same Connecticut run written directly in Julia: the shortest path from graph to Atlas, for when a config file is not enough. |
| [`run_cyclewalk_ct_metadata.jl`](run_cyclewalk_ct_metadata.jl) | Focused on what lands in the Atlas header, and on registering your own observables with `push_writer!`. |
| [`run_cyclewalk_extend.jl`](run_cyclewalk_extend.jl) | Adds samples to a finished Atlas, restarting from its last recorded plan and reading its config back out of its header. |
| [`run_ais_ct.jl`](run_ais_ct.jl), [`run_ais_toml.jl`](run_ais_toml.jl) | Annealed importance sampling: a base chain plus an annealing pass per retained sample. The base chain is serial, so speedup is capped at `(base_steps_per_sample + steps_per_annealing) / base_steps_per_sample` however many threads you give it. |
| [`run_asmc_toml.jl`](run_asmc_toml.jl) | Annealed sequential Monte Carlo: a particle population tempered from the base measure to the target. Give it `init_steps > 0` — required with `schedule = "adaptive"`, and without it the population starts as N identical clones. |
| [`run_pt_toml.jl`](run_pt_toml.jl) | Parallel tempering: a ladder of replicas tempered from the base measure to the target, swapping adjacent rungs. `ThreadedBackend` spawns one task per rung, so give it `-t N` matching or exceeding `[pt] n_rungs` — see [`docs/run_pt_toml.md`](../docs/run_pt_toml.md) and [`docs/pt_profiling_notes.md`](../docs/pt_profiling_notes.md) for scaling data. |
| [`parameterUtils.jl`](parameterUtils.jl), [`runtimeParameters.jl`](runtimeParameters.jl) | Included by the runners, not run directly: command-line parsing and every quantity derived from a config. |

## The directories

| Directory | Contents |
| --- | --- |
| [`toml/`](toml) | Ready-to-run configurations: 4×4, 8×8 and 10×10 grids, a 10×10 hex lattice, Connecticut precincts, North Carolina precincts (`param_nc.toml`, demonstrates `[plans.derive]`), and the AIS/SMC/PT configs. |
| `data/` | The graphs those configs read (`ct/`, `grid/`, `hex/`, `nc/`, `oh/`). |
| `output/` | Where runs write their Atlas files, under the config's `outputDirectory`. |
| [`validation/`](validation) | Cross-checks of the annealed samplers against a standard cycle walk; has its own README. |
| `src-hold/` | Helpers for generating the sample grid graphs. |

## The configuration-file runner

[`run_cyclewalk_toml.jl`](run_cyclewalk_toml.jl) samples plans by Metropolis-Hastings and
writes them to an Atlas, taking every setting from a TOML file. A configuration has four
tables:

- **`[plans]`** — the graph to load and which of its columns are population, area and
  perimeter; how many districts, and how tightly their populations must balance. An
  optional `[plans.derive]` subtable computes a named column from others (e.g. a
  unique node name joined from columns that aren't unique alone) — see `param_nc.toml`.
- **`[mcmc]`** — how many cycle-walk steps to run, and what fraction of proposals are
  two-tree moves.
- **`[measure]`** — the target distribution: named numeric parameters, plus one
  `[[measure.energy]]` block per energy, each weighted by a number or by arithmetic over
  those parameters (`weight = "2*gamma + 1"`). A weight of zero drops its energy.
- **`[run]`** — where output goes, how it is named, the RNG seed, which observables are
  recorded with each plan, and whether diagnostics are collected.

```toml
[measure]
gamma = 1.0
iso_weight = 0.3

[[measure.energy]]
name = "get_log_spanning_forests"
weight = "gamma"

[[measure.energy]]
name = "get_isoperimetric_score"
weight = "iso_weight"
```

Any key can be overridden per run, either with a named flag or with the repeatable
`--set <table>.<key>=<value>`; the output file name carries every parameter the measure
reads, so the points of a sweep never collide, and a run that would overwrite an
existing Atlas stops unless you pass `--overwrite`.

```bash
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml --thread_id 7 --gamma 0.5
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml --set measure.vra_weight=2.0
```

📖 **[`docs/run_cyclewalk_toml.md`](../docs/run_cyclewalk_toml.md) documents every key,
every energy and observable you can name, how the output file name is built, all the
command-line overrides, and what the error messages mean.** Start there when writing a
configuration of your own.
