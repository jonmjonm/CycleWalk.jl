# `run_cyclewalk_toml.jl` and its configuration file

[`examples/run_cyclewalk_toml.jl`](../examples/run_cyclewalk_toml.jl) is the general
Cycle Walk runner: it samples redistricting plans by Metropolis-Hastings and writes them
to an Atlas file, taking every setting from a TOML configuration file rather than from
edited Julia source. Two runs that differ only in their config are otherwise identical,
and each Atlas embeds the config that produced it, so a run can be read back and
extended later.

This document covers the script, the format of the configuration file, and every key it
reads.

- [Quick start](#quick-start)
- [A minimal configuration, line by line](#a-minimal-configuration-line-by-line)
- [What the script does with it](#what-the-script-does-with-it)
- [Configuration reference](#configuration-reference)
  - [`[plans]` — the map and the districts](#plans--the-map-and-the-districts)
  - [`[mcmc]` — how long, and which moves](#mcmc--how-long-and-which-moves)
  - [`[measure]` — the target distribution](#measure--the-target-distribution)
  - [`[run]` — output, seeding, diagnostics](#run--output-seeding-diagnostics)
- [Energies you can name](#energies-you-can-name)
- [Observables you can record](#observables-you-can-record)
- [Weights, parameters, and expressions](#weights-parameters-and-expressions)
- [The output file and its name](#the-output-file-and-its-name)
- [Overriding on the command line](#overriding-on-the-command-line)
- [Worked examples](#worked-examples)
- [The earlier measure format (deprecated)](#the-earlier-measure-format-deprecated)
- [Extending a finished run](#extending-a-finished-run)
- [Error messages and what they mean](#error-messages-and-what-they-mean)

## Quick start

Run from the `examples` directory — the script activates the environment in the working
directory:

```bash
cd examples
julia --project=. run_cyclewalk_toml.jl toml/param_grid4x4.toml
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml
julia --project=. run_cyclewalk_toml.jl toml/param_hex10x10.toml
```

Any key can be changed without editing the file:

```bash
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml \
      --thread_id 7 --gamma 0.5 --cycle_walk_steps 1e5
```

The configs shipped in [`examples/toml`](../examples/toml) are working starting points:
a 4×4, 8×8 and 10×10 grid, a 10×10 hex lattice, and Connecticut precincts.

## A minimal configuration, line by line

This is [`examples/toml/param_grid4x4.toml`](../examples/toml/param_grid4x4.toml), a
complete configuration — nothing else is required.

```toml
[plans]
pop_dev = 0.0001                    # districts may differ from ideal by ±0.01%
num_dists = 4                       # districts to build
node_data = ["county", "node_name", "population", "area", "border_length"]
                                    # node columns to load from the graph JSON
geo_units = ["node_name"]           # the column that names a node
pop_col = "population"              # which column is population
area_col = "area"                   # node area, for compactness scores
node_border_col = "border_length"   # node perimeter on the outer boundary
edge_perimeter_col = "length"       # shared perimeter of an edge's two nodes
map_directory = ["data", "grid"]    # path components, joined for you
map_file = "grid_graph_4_by_4.json"

[mcmc]
cycle_walk_steps = 5e8              # two-tree cycle-walk steps to perform
two_cycle_walk_frac = 0.3           # fraction of proposals that are two-tree moves

[measure]
gamma = 0.0                         # named parameters: any number here may be
iso_weight = 0.0                    # used by a weight below, and swept from the CLI

[[measure.energy]]                  # one block per energy in the target
name = "get_log_spanning_forests"
weight = "gamma"                    # weight 0 drops the energy: this samples the
                                    # plain spanning-forest measure
[[measure.energy]]
name = "get_isoperimetric_score"
weight = "iso_weight"

[run]
thread_id = 1                       # seeds the RNG and names the file
atlasNameBase = "grid4x4_cycleWalk" # the file name is built from this
outputDirectory = ["output", "grid"]
cycle_walk_out_freq = 100           # record a plan every 100 cycle-walk steps
writer_stats = ["get_log_spanning_trees"]   # extra data recorded with each plan
compress = "gz"                     # write .jsonl.gz rather than .jsonl
run_diagnostics = true              # collect acceptance ratios, cycle lengths, …
```

Four tables, all required: `[plans]`, `[mcmc]`, `[measure]`, `[run]`.

## What the script does with it

```
graph JSON ──▶ BaseGraph ──▶ Graph ──▶ LinkCutPartition ──▶ Metropolis-Hastings ──▶ Atlas
   [plans]                            + PopulationConstraint      [mcmc]           [run]
                                                                 [measure]
```

1. **Graph.** `map_directory` and `map_file` are joined into a path; the file must
   exist or the run stops immediately. `node_data` selects the node columns to load;
   `pop_col`, `area_col`, `node_border_col` and `edge_perimeter_col` say which of them
   mean what. `geo_units` names the node-naming column(s).
2. **Constraint.** One `PopulationConstraint(graph, num_dists, pop_dev)` — the only
   constraint this runner installs. Others (region packing, district caps) need a
   script; see [`run_cyclewalk_ct.jl`](../examples/run_cyclewalk_ct.jl).
3. **Initial plan.** A `LinkCutPartition` of `num_dists` districts drawn with the run's
   RNG, satisfying the population constraint.
4. **Proposals.** A mixture: `two_cycle_walk_frac` of proposals are two-tree cycle walks
   (moves that can change the districting), the rest are one-tree internal forest walks
   (moves that re-draw a district's spanning tree).
5. **Measure.** Built from the `[[measure.energy]]` blocks. Builders that need the graph
   receive it through the runner's context (`graph`, `base_graph`, `num_dists`,
   `pop_col`).
6. **Run.** `steps` Metropolis-Hastings steps, recording a plan every `outfreq` steps,
   both derived below.

Derived quantities, all computed in
[`examples/parameterUtils.jl`](../examples/parameterUtils.jl):

| Quantity | From |
| --- | --- |
| `steps` = `ceil(cycle_walk_steps / two_cycle_walk_frac)` | so that the run performs about `cycle_walk_steps` *two-tree* steps regardless of the mixture |
| `outfreq` = `floor(cycle_walk_out_freq / two_cycle_walk_frac)` | so that a plan is recorded every `cycle_walk_out_freq` two-tree steps |
| `rng_seed` = `rng_seed_base + 15123 * thread_id` | independent streams per thread from one base seed |
| output file name | see [The output file and its name](#the-output-file-and-its-name) |

So `cycle_walk_steps = 5e8` with `two_cycle_walk_frac = 0.3` runs 1.67e9 total steps,
and `cycle_walk_out_freq = 100` records a plan every 333.

## Configuration reference

Types are TOML types. **Required** keys have no default; the run fails without them.

### `[plans]` — the map and the districts

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `map_directory` | array of strings | **required** | Path components of the directory holding the graph, relative to where you run the script. Joined with `joinpath`. |
| `map_file` | string | **required** | Graph JSON file name. Missing file ⇒ the run stops before doing any work. |
| `num_dists` | integer | **required** | Number of districts. |
| `pop_dev` | float | **required** | Allowed fractional deviation from ideal district population — `0.02` is ±2%. |
| `node_data` | array of strings | **required** | Node columns read out of the JSON. Anything an energy, an observable or a column key below refers to must be listed here. |
| `pop_col` | string | **required** | The population column. |
| `geo_units` | array of strings | **required** | The node-naming column(s), finest last. Nearly every config is single-level, e.g. `["NAME"]`. |
| `area_col` | string | `nothing` | Node area. Required by the Polsby-Popper compactness scores. |
| `node_border_col` | string | `nothing` | Length of a node's boundary that lies on the outer boundary of the state. |
| `edge_perimeter_col` | string | `nothing` | Edge column giving the perimeter shared by the edge's two nodes. |

### `[mcmc]` — how long, and which moves

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cycle_walk_steps` | number | **required** | Two-tree cycle-walk steps to perform. Scientific notation (`1e6`) is fine — it is a float, and the total step count is rounded up. |
| `two_cycle_walk_frac` | float in `[0,1]` | **required** | Fraction of proposals drawn as two-tree cycle walks; the remainder are one-tree internal forest walks. Outside `[0,1]` is an assertion error. |

### `[measure]` — the target distribution

The target is a weighted sum of energies. Every **number** in `[measure]` is a *named
parameter* usable by any weight; every `[[measure.energy]]` block adds one energy.

```toml
[measure]
gamma = 1.0                            # named parameters (any numeric key)
iso_weight = 0.3
vra_weight = 2.0

[[measure.energy]]
name   = "get_log_spanning_forests"
weight = "gamma"

[[measure.energy]]
name   = "get_isoperimetric_score"
weight = "iso_weight"

[[measure.energy]]
name   = "get_log_district_trees"
weight = "2*gamma + 1"                 # arithmetic over the parameters above

[[measure.energy]]
name    = "build_get_partisan_seats"   # a builder, called with its arguments
args    = ["G20PREDEM", "G20PREREP"]
weight  = "vra_weight"
desc    = "Dem seats (G20 pres)"       # a built energy is a closure — label it
```

Keys of a `[[measure.energy]]` block:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | **required** | An energy **exported** by `CycleWalk`, or a builder returning one. Unexported names are refused, so a config cannot reach into internals. |
| `weight` | number or string | **required** | The weight in front of this energy. A string is arithmetic over the `[measure]` parameters. **A weight of `0` drops the energy from the measure** (unless it has a nonzero `weight_start`). |
| `weight_start` | number or string | *none* | Only meaningful for annealed runs (AIS/SMC): the weight the schedule starts from, with `weight` as the target. `run_cyclewalk_toml.jl` ignores it. Present with a zero `weight`, it keeps the energy in the measure so a schedule can ramp it. |
| `args` | array | `[]` | Literal arguments for a builder, passed positionally after any `context`. |
| `kwargs` | table | `{}` | Keyword arguments for a builder. |
| `context` | array of strings | `[]` | Values the *runner* supplies because a config cannot write them down, passed before `args`. This runner offers `"graph"`, `"base_graph"`, `"num_dists"`, `"pop_col"`. |
| `desc` | string | the function name | The label recorded in the Atlas header and used as the key for the energy. A builder returns a closure whose automatic name is something like `#7#8`, so label built energies. |

Rules worth knowing:

- Any other key in a block is an **error**, not an ignored typo — a silently different
  target is worse than a stopped run.
- A block with no `args`, `kwargs` or `context` uses the named function directly;
  otherwise the name is treated as a builder and called.
- Two blocks resolving to the *same function* is an error: a `Measure` is keyed by
  function and the second would silently replace the first. Builders are exempt — each
  call returns a distinct closure — so one builder may appear several times with
  different arguments.
- `gamma` and `iso_weight` are set to `0.0` if absent, so they are always available as
  parameters and as `--gamma` / `--iso_weight` on the command line.

### `[run]` — output, seeding, diagnostics

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `atlasNameBase` | string | **required** | Stem of the output file name. |
| `outputDirectory` | array of strings | **required** | Path components of the output directory; created if missing. |
| `thread_id` | integer | **required** | Identifies this chain: it seeds the RNG (`rng_seed_base + 15123*thread_id`) and appears in the file name. Run the same config with different `thread_id`s for independent chains. |
| `cycle_walk_out_freq` | integer | **required** | Record a plan every this many *two-tree* steps. |
| `run_diagnostics` | boolean | **required** | `true` attaches acceptance ratios, cycle lengths and changed-node counts to the cycle-walk proposal; they are written alongside each recorded plan. |
| `writer_stats` | array of strings | `[]` | Observables recorded with every plan — see [Observables you can record](#observables-you-can-record). Read from `[run]`; also accepted in `[plans]` for older configs. |
| `compress` | string | *none* | Appended to `.jsonl` as an extension. `"gz"` writes `.jsonl.gz` (gzip); omit the key for plain `.jsonl`. |
| `output_districting` | boolean | `true` | `false` drops the per-map node→district assignment, leaving only the recorded statistics. Much smaller files, but the plans themselves are gone — and a run cannot be extended from them. |
| `io_mode` | string | `"w"` | `"w"` truncates, `"a"` appends to an existing Atlas (no second header is written). |
| `description` | string | `""` | Free text recorded in the Atlas header. |
| `rng_seed_base` | integer | `454190` | Base RNG seed; see `thread_id`. |
| `blas_threads` | integer | `0` | `0` leaves BLAS alone; a positive value pins BLAS threads. The spanning-forest log-determinant is BLAS-bound, so this can speed a single serial chain on large-district graphs. |

## Energies you can name

An energy must return **one number** per call, with signature
`f(partition, districts; update)`. These are exported and usable in
`[[measure.energy]]`:

| Name | What it scores |
| --- | --- |
| `get_log_spanning_forests` | Log number of spanning forests of the districting — the default Cycle Walk energy. Its weight is conventionally called `gamma`. |
| `get_isoperimetric_score` | Polsby-Popper compactness, summed over districts. Needs `area_col`, `node_border_col`, `edge_perimeter_col`. Its weight is conventionally called `iso_weight`. |
| `get_log_district_trees` | Log number of spanning trees, summed over districts. |
| `get_log_linking_edges` | Log number of edges linking district pairs. |
| `build_get_partisan_seats` | Builder: `args = [votes1_col, votes2_col]` gives the seats won by the first column's party. Both columns must appear in `node_data`. |

Vector-valued functions (`get_isoperimetric_scores`, `get_log_spanning_trees`,
`get_average_degrees`, …) are **observables, not energies** — they belong in
`writer_stats`. Putting one in the measure fails at the first energy evaluation.

Builders whose arguments cannot be written as TOML values are not configurable from a
file. `build_performant_vra_score` and `build_performant_vra_report` take a vector of
tuples-of-tuples of column names; TOML has arrays but no tuples, so those need a Julia
script.

## Observables you can record

`writer_stats` names functions called as `f(partition)` once per recorded plan; each is
stored under its own name in that plan's Atlas record.

| Name | Recorded value |
| --- | --- |
| `get_log_spanning_trees` | Log spanning-tree count, per district |
| `get_isoperimetric_scores` | Polsby-Popper score, per district |
| `get_isoperimetric_score` | The same, summed |
| `get_log_spanning_forests` | Log spanning-forest count of the plan |
| `get_log_district_trees` | Log district-tree count |
| `get_log_linking_edges` | Log linking-edge count |
| `get_cut_edge_sum` | Number of cut (boundary) edges |
| `get_diameters` | Tree diameter, per district |
| `get_average_degrees` | Average node degree, per district |
| `get_center_moments`, `get_center_leaves_moments` | Tree-center moments, per district |
| `get_degree_distributions`, `get_neighbor_lists` | Degree distribution / neighbor list, per district |

`writer_stats` takes **bare names only** — it has no `args`, so builder-based
observables such as `build_get_partisan_margins` cannot be registered from a config
file. Register those with `push_writer!` in a script; see
[`run_cyclewalk_ct_metadata.jl`](../examples/run_cyclewalk_ct_metadata.jl).

## Weights, parameters, and expressions

Every numeric key in `[measure]` is a parameter a weight may be written in terms of:

```toml
[measure]
gamma = 1.5
[[measure.energy]]
name = "get_log_district_trees"
weight = "2*gamma + 1"          # 4.0
```

Only `+ - * / ^`, numeric literals, and parameter names are permitted. Expressions are
parsed and then *interpreted* — never `eval`ed — so a config file cannot run code. That
matters because configs are read back out of Atlas headers written elsewhere. A function
call, an index, a second statement, or an unknown name is rejected with a message naming
what was refused, as is a result that is not finite.

Write the weight as a parameter name whenever you intend to sweep it: `--gamma 0.5`
changes the *parameter*, so an energy whose weight is the literal `1.0` will not move.

## The output file and its name

The name is assembled so that two runs with different settings cannot land on one path:

```
<atlasNameBase>_thread<thread_id>_cyclewalkVS_2treeCycleWalk_<two_cycle_walk_frac>
    [_gamma<gamma>] [_iso<iso_weight>] [_<param><value> …] .jsonl[.gz]
```

- `_gamma…` and `_iso…` appear only when nonzero, and are read **off the built measure**
  (the weights in front of `get_log_spanning_forests` and `get_isoperimetric_score`)
  rather than out of config keys — so the name cannot disagree with the run.
- Every *other* parameter that any weight expression reads is appended as
  `_<name><value>`, skipping zeros, so a sweep over a parameter that is not `gamma` or
  `iso_weight` still writes each point to its own file. A parameter no expression uses
  cannot change the target and is not tagged.
- Values are formatted exactly, so a computed weight can read `_w0.30000000000000004`;
  that is the number the run used, and rounding it could let two different runs share a
  name.

The file lands in `joinpath(outputDirectory..., <name>)`.

As a backstop, a run that would **truncate an existing Atlas stops before starting**.
Pass `--overwrite` to replace it, or set `io_mode = "a"` to append.

The Atlas header records the measure's energies and weights, the constraints, the
CycleWalk version, execution metadata (user, script name, and the script's own source),
the full text of the config file, any command-line overrides, and:
`popdev`, `toml_config_file`, `map_file`, `pop_col`, `geo_units`, `cycle_walk_steps`,
`cycle_walk_out_freq`, `rng_seed_base`, `blas_threads`, `run_diagnostics`.

## Overriding on the command line

```bash
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml --thread_id 7 --gamma 0.5
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml --set measure.vra_weight=2.0
```

Named flags:

| Flag | Sets |
| --- | --- |
| `--thread_id` | `run.thread_id` |
| `--two_cycle_walk_frac`, `-f` | `mcmc.two_cycle_walk_frac` |
| `--cycle_walk_steps`, `-s` | `mcmc.cycle_walk_steps` |
| `--gamma` | `measure.gamma` |
| `--iso_weight` | `measure.iso_weight` |
| `--num_dists`, `-n` | `plans.num_dists` |
| `--pop_dev`, `-p` | `plans.pop_dev` |
| `--run_diagnostics`, `-d` | `run.run_diagnostics` |
| `--overwrite` | allow truncating an existing Atlas |

`--set <table>.<key>=<value>` reaches **any** key, is repeatable, and needs no change to
the runner when a config grows a new key:

```bash
julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml \
      --set measure.vra_weight=2.0 --set run.description='vra sweep, point 3'
```

- Values are parsed by TOML itself, so they get the types they would have had in the
  file: `--set mcmc.cycle_walk_steps=1e6` is a float, `--set plans.num_dists=4` an
  integer, `--set plans.geo_units='["node_name"]'` an array. Text that is not valid TOML
  is taken as a plain string, so a description needs no quoting gymnastics.
- The table must already exist, so `--set measur.gamma=2` is caught as the typo it is.
  Keys within a table are open — supplying a *new* parameter is the point.
- Setting the same value with both a flag and `--set` is an error rather than a silent
  winner: a run whose `gamma` has two answers should not pick one.
- Overrides are recorded in the Atlas header under `cli_overrides`, since the embedded
  config file alone would not show them.

## Worked examples

**Sweep a compactness weight across files and threads.**

```bash
for iso in 0.0 0.25 0.5 1.0; do
  for t in 1 2 3 4; do
    julia --project=. run_cyclewalk_toml.jl toml/param_ct.toml \
          --iso_weight $iso --thread_id $t &
  done
done
```

Each combination writes its own Atlas (`…_thread2…_iso0.25.jsonl.gz`); nothing collides.

**A target with three energies and a shared parameter.**

```toml
[measure]
gamma = 1.0
compact = 0.4

[[measure.energy]]
name = "get_log_spanning_forests"
weight = "gamma"

[[measure.energy]]
name = "get_isoperimetric_score"
weight = "compact"

[[measure.energy]]
name = "get_log_district_trees"
weight = "0.5*gamma"
```

`compact` is not `iso_weight`, so it is tagged into the file name as `_compact0.4` and
swept with `--set measure.compact=…`.

**A partisan energy on the Connecticut graph.** `G20PREDEM`/`G20PREREP` must be listed
in `node_data`:

```toml
[plans]
node_data = ["COUNTY", "NAME", "POP20", "area", "border_length",
             "G20PREDEM", "G20PREREP"]

[measure]
seat_weight = 1.0

[[measure.energy]]
name   = "build_get_partisan_seats"
args   = ["G20PREDEM", "G20PREREP"]
weight = "seat_weight"
desc   = "Dem seats (G20 pres)"
```

## The earlier measure format (deprecated)

Configs written before `[[measure.energy]]` listed the scores by name:

```toml
[measure]
measure_scores = ["get_log_spanning_forests", "get_isoperimetric_score"]
gamma = 1.0
iso_weight = 0.3

[measure.weights]                        # for scores gamma/iso_weight do not cover
get_log_district_trees = "2*gamma + 1"
```

**This still works and existing files need not change.** Every Atlas embeds its config
and `run_cyclewalk_extend.jl` reads those back to resume a chain, so the short form has
to keep running indefinitely; it is *translated* into the equivalent list of energies
rather than handled separately.

It is deprecated for new configs because it cannot express an energy built from
arguments, a label for one, an annealing start, or a weight for a third energy without a
second table. A file may use one form or the other — carrying both is an error.

## Extending a finished run

```bash
julia --project=. run_cyclewalk_extend.jl output/grid/<atlas>.jsonl.gz \
      --add_cycle_walk_steps 1e5
```

Adds samples to an existing Atlas, restarting from the last plan it recorded and reading
the configuration back out of its header — so the extension derives its settings exactly
as the original run did. `--burn_in` discards steps before recording resumes. An Atlas
records a districting rather than a spanning forest, so the forest is redrawn on restart
exactly as a fresh run's initial partition is.

## Error messages and what they mean

| Message | Cause |
| --- | --- |
| `map file not found: …` | `map_directory`/`map_file` do not resolve from the current directory. Run from `examples`. |
| `… already exists, and this run would truncate it` | An Atlas of that name is there. Use `--overwrite`, `io_mode = "a"`, or change `thread_id`. |
| `unknown energy "x": CycleWalk exports no such name` | Typo, or a name that exists but is not exported. |
| `energy "x" has unknown key "y"` | A `[[measure.energy]]` key outside `name`/`weight`/`weight_start`/`args`/`kwargs`/`context`/`desc`. |
| `energy "x" has no weight` | Every energy block needs a weight — there is no default. |
| `the expression "…" refers to "z", which …` | A weight names a parameter that is not a number in `[measure]`. |
| `energies "a" and "b" are the same function` | Two blocks resolve to one function; a measure can weight it once. |
| `[measure] has both an [[measure.energy]] list and a measure_scores list` | Pick one form. |
| `could not build energy "x" from its configured arguments` | A builder rejected the `args`/`kwargs` — often TOML arrays where the builder wants tuples. |
| `--set … the config has no [t] table` | Mistyped table name. |
| `--gamma and --set measure.gamma both set the same value` | One or the other. |
| `AssertionError: 0 ≤ two_cycle_walk_frac ≤ 1` | Exactly that. |

## See also

- [`examples/README.md`](../examples/README.md) — the other example scripts.
- [`examples/run_cyclewalk_ct.jl`](../examples/run_cyclewalk_ct.jl) — the same run
  written directly in Julia, for when a config file is not enough.
- [`examples/run_ais_toml.jl`](../examples/run_ais_toml.jl) and
  [`examples/run_asmc_toml.jl`](../examples/run_asmc_toml.jl) — annealed samplers. They
  read the same `[plans]`/`[measure]`/`[run]` tables, add `[ais]`/`[smc]`, and use
  `weight_start` to define the annealing base.
- [duke.is/cyclewalk](https://duke.is/CycleWalk) — introduction and further examples.
