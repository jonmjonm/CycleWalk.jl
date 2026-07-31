# CycleWalk.jl Public API Reference

This document lists the public API of `CycleWalk.jl`: the functions, types, and
data structures exported by `using CycleWalk`. They are grouped by the role they
play in a typical run (build a graph → build a partition → define constraints and
a target measure → run the MCMC sampler → write an Atlas of samples).

A complete, runnable example using most of this surface lives in
[`examples/run_cyclewalk_ct.jl`](./examples/run_cyclewalk_ct.jl). See the
[README](./README.md) for installation and higher-level documentation.

Conventions used below:

- Arguments after a `;` are keyword arguments.
- `partition` is a [`LinkCutPartition`](#linkcutpartition).
- `districts` defaults to all districts (`collect(1:partition.num_dists)`).
- Energy / score callbacks all share the signature
  `f(partition, districts=...; update=nothing)` and return a `Float64` (or a
  `Vector{Float64}` for the per-district `..._scores` / `..._trees` variants).

---

## Table of Contents

- [Graphs](#graphs)
- [Partition](#partition)
- [Constraints](#constraints)
- [Proposals](#proposals)
- [Measures, Energies & Observables](#measures-energies--observables)
- [Diagnostics](#diagnostics)
- [Writer (Atlas output)](#writer-atlas-output)
- [Running the Sampler](#running-the-sampler)
- [Annealed Importance Sampling & SMC](#annealed-importance-sampling--smc)
- [Chain (convenience wrapper)](#chain-convenience-wrapper)
- [Data Structures & Types Summary](#data-structures--types-summary)

---

## Graphs

CycleWalk operates on dual graphs of geographic units. The graph types are
re-exported from `MetropolizedForestRecom.jl`.

### Types

| Type | Description |
| --- | --- |
| `AbstractGraph` | Abstract supertype for all graph representations. |
| `BaseGraph` | A single-level dual graph with node/edge attributes (populations, areas, perimeters, etc.). |
| `MultiLevelGraph` | A hierarchically-leveled graph (e.g. precinct → county). |
| `Graph` | Alias for `MultiLevelGraph`. |

### `BaseGraph(filepath, pop_col; ...)`

Construct a base graph from a graph file (`.json`).

```julia
BaseGraph(
    filepath::AbstractString,
    pop_col::AbstractString;
    inc_node_data::Set{String}=Set(),
    edge_weights::String="connections",
    bpop_col=nothing, vap_col=nothing, bvap_col=nothing,
    area_col=nothing, node_border_col=nothing,
    edge_perimeter_col=nothing, oriented_nbrs_col=nothing,
    mcd_col=nothing, adjacency::String="rook"
)::BaseGraph
```

### `Graph` / `MultiLevelGraph`

`Graph` is an alias for `MultiLevelGraph`. It can be constructed directly from a
file (loading a `BaseGraph` and a set of hierarchy levels), as in the example
script:

```julia
graph = Graph(pctGraphPath, "POP20", "NAME"; inc_node_data=nodeData,
              area_col="area", node_border_col="border_length",
              edge_perimeter_col="length")
```

### `multi_level_graph(base_graph, levels)`

```julia
multi_level_graph(base_graph::BaseGraph, levels::Vector{String})::MultiLevelGraph
```

Build a `MultiLevelGraph` from a `BaseGraph` and a list of level names (keys in
the node attributes). The levels are reordered from coarsest to finest; the
hierarchy must be strict (each finer value maps to exactly one coarser value).

### `edge_weight(...)`

Accessor for the weight of an edge between two nodes (re-exported from
`MetropolizedForestRecom`).

### `modify_edge_weights!(graph, edge_weight_func)`

```julia
modify_edge_weights!(base_graph::BaseGraph, edge_weight_func::Function)
modify_edge_weights!(graph::MultiLevelGraph, edge_weight_func::Function)
```

Recompute every edge weight in place. `edge_weight_func(graph, src, dst)` is
called for each edge and must return the new weight.

### `build_graph`

Re-exported graph-construction helper from `MetropolizedForestRecom`.

### `cluster_base_graph`

Re-exported helper that clusters a base graph into coarser units.

---

## Partition

### `LinkCutPartition`

The central mutable data structure. It represents a districting plan as a
spanning forest (one spanning tree per district) maintained in an augmented
link/cut tree so that proposals can be applied and reverted in `O(log n)`.

```julia
mutable struct LinkCutPartition <: AbstractPartition
    num_dists::Int64
    cross_district_edges::Dict{Tuple{Int64,Int64}, Set{SimpleWeightedEdge}}
    district_roots::Vector{Int64}
    roots_to_district::Dict{Int64, Int64}
    energy_data::Dict{..., AbstractEnergyData}
    node_to_dist::Vector{Int64}
    node_to_dist_update::Vector{Int64}
    lct::LinkCutTree
    node_col::String
    graph::BaseGraph
    node_pops::Vector{Float64}
    identifier::Int64
end
```

#### Constructors

```julia
# Build from a graph + constraints, drawing an initial balanced plan.
LinkCutPartition(
    graph::MultiLevelGraph,
    constraints::Constraints,
    num_dists::Int;
    rng::AbstractRNG=PCG.PCGStateOneseq(UInt64),
    verbose::Bool=false
)::LinkCutPartition

# Build from an existing MultiLevelPartition.
LinkCutPartition(
    partition::MultiLevelPartition,
    rng::AbstractRNG=PCG.PCGStateOneseq(UInt64)
)::LinkCutPartition
```

The first form is the one used in practice: it draws an initial partition that
satisfies `constraints` and wraps it in a link/cut tree.

### `relabel_districts!(partition, node_to_district)`

Renumber `partition`'s districts so every node carries the district index it has in
`node_to_district` — a `Dict` from each node's one-tuple id to its district, which is
the form an Atlas map records.

```julia
relabel_districts!(
    partition::LinkCutPartition,
    node_to_district::AbstractDict{<:Tuple{Vararg{String}}, <:Integer}
)::LinkCutPartition
```

The constructors number districts by the order their trees are first met in the
link/cut tree, which has nothing to do with the labels of the assignment a plan was
built from. That does not matter to sampling, but it does when a recorded plan is
reloaded and the new samples should line up with the old — as when extending an
existing Atlas (`examples/run_cyclewalk_extend.jl`). `node_to_district` must describe
the same plan, differing only in labels; anything else throws an `ArgumentError`.
Cached `energy_data` is dropped (it is keyed by district) and rebuilt lazily.

### Related re-exported types

| Type | Description |
| --- | --- |
| `AbstractPartition` | Abstract supertype for partitions. |
| `MultiLevelPartition` | The `MetropolizedForestRecom` multi-level partition (used to seed a `LinkCutPartition`). |

---

## Constraints

Constraints restrict the space of valid plans. They are collected into a
`Constraints` container and enforced during sampling.

### `Constraints` / `initialize_constraints`

```julia
mutable struct Constraints
    population_constraint::PopulationConstraint
    constraints::Vector{AbstractConstraint}
    descriptions::Vector{String}
end

Constraints(population_constraint=nothing)::Constraints
const initialize_constraints = Constraints   # alias
```

`initialize_constraints()` creates an empty constraint set (with an unbounded
population constraint by default).

### `add_constraint!` / `push_constraint!`

```julia
add_constraint!(constraints::Constraints, constraint::AbstractConstraint; desc::String="")
const push_constraint! = add_constraint!     # alias
```

Add a constraint. A `PopulationConstraint` replaces the population constraint;
any other constraint is appended to the list.

### `satisfies_constraint` / `satisfies_constraints`

```julia
satisfies_constraints(
    partition::LinkCutPartition,
    constraints::Constraints,
    districts=collect(1:partition.num_dists);
    check_population::Bool=false,
    update::Union{Update, Nothing}=nothing
)::Bool
```

Returns whether the (optionally updated) partition satisfies all constraints.
`satisfies_constraint` is the per-constraint method dispatched on each
constraint type.

### Constraint types

| Type | Constructor (key form) | Description |
| --- | --- | --- |
| `AbstractConstraint` | — | Abstract supertype. |
| `PopulationConstraint` | `PopulationConstraint(graph, num_dists, tolerance)` or `PopulationConstraint(min_pop, max_pop)` | Bounds each district's population to `[min_pop, max_pop]`. |
| `PackRegionConstraint` | `PackRegionConstraint(graph, region; unpack=0, num_dists=0, ideal_pop=0)` | Requires regions to be "packed" with at least their proportional share of whole districts. |
| `CapRegionDistricts` / `CapRegionDistConstraint` | `CapRegionDistricts(graph, region; excess_split=0, num_dists=0, ideal_pop=0)` | Caps how many districts may touch a region. |
| `BudgetedRegionConstraint` | `BudgetedRegionConstraint(graph, region; total_budget, num_dists=0, ideal_pop=0, pack_budget=nothing, cap_budget=nothing, budget_mode=:cap_only, rng=nothing)` | Combined pack/cap constraint with a shared split budget. `budget_mode ∈ (:cap_only, :pack_only, :fixed, :random)`. |

---

## Proposals

A proposal is a function `f(partition, rng; diagnostics=nothing)` returning
`(acceptance_prob_ratio, update)`. The `build_*` functions return such a closure,
captured against a constraint set. They come in two families:

- **One-tree / internal forest walk** — adds an internal edge and cuts the
  resulting cycle within a single tree.
- **Two-tree / lifted cycle walk** — adds two edges between adjacent districts
  and recuts, moving the boundary between two districts.

```julia
build_one_tree_cycle_walk(constraints::Constraints)    # one-tree walk
build_internal_forest_walk   = build_one_tree_cycle_walk    # alias (same function)

build_lifted_tree_cycle_walk(constraints::Constraints) # two-tree walk
build_cycle_walk             = build_lifted_tree_cycle_walk # alias
build_two_tree_cycle_walk    = build_lifted_tree_cycle_walk # alias
```

Combine proposals into a weighted mixture (weights must sum to `1`) for
`run_metropolis_hastings!`:

```julia
proposal = [(twocycle_frac, build_two_tree_cycle_walk(constraints)),
            (1.0 - twocycle_frac, build_one_tree_cycle_walk(constraints))]
```

---

## Measures, Energies & Observables

### `Measure`

```julia
mutable struct Measure
    weights::Dict{Function, Float64}
    scores::Set{Function}
    descriptions::Dict{Function, String}
end

Measure()::Measure
```

The target distribution is `exp(-∑ weightᵢ · scoreᵢ)`. Build it by pushing
weighted energy/score functions.

### `push_energy!`

```julia
push_energy!(measure::Measure, score::Function, weight::Real; desc::String="",
             allow_zero::Bool=false)
```

Add a weighted score function to the measure. A zero weight is ignored.

Pass `allow_zero=true` to keep a zero-weight score anyway. An annealing schedule
needs this: a run that ramps an energy's weight *down to* zero has a zero target
weight, and a score that never entered `measure.scores` can never be annealed —
`get_log_energy` sums over `scores`, so a weight written straight into
`measure.weights` for a score outside that set is silently ignored. Keeping the
score costs nothing per step (`get_log_energy` skips zero weights when it
evaluates), but it does appear in the Atlas header's energy list with weight `0`.

### `evaluate_weight_expression`

```julia
evaluate_weight_expression(expr::AbstractString, parameters::AbstractDict)::Float64
```

Evaluate an arithmetic expression over named parameters, for measure weights written
in a config file rather than in code:

```julia
evaluate_weight_expression("2*gamma + 1", Dict("gamma" => 1.5))   # 4.0
```

Only `+ - * / ^`, numeric literals, and names present in `parameters` are permitted.
Anything else — a function call, an index, a field access, a second statement, an
unknown name — raises an `ArgumentError` naming what was rejected.

The expression is parsed to an AST and then **interpreted**; it is never `eval`ed, so
`"run(\`rm -rf /\`)"` is refused at inspection time rather than run. This matters
because these expressions arrive in config files, which CycleWalk reads back out of
Atlas headers written elsewhere. Expressions nesting deeper than
`CycleWalk.MAX_WEIGHT_EXPRESSION_DEPTH` (16) or larger than
`CycleWalk.MAX_WEIGHT_EXPRESSION_NODES` (100) are rejected, as is any result that is
not a finite real — an `Inf` or `NaN` weight would silently change the target rather
than fail.

### `get_log_energy`

```julia
get_log_energy(partition, measure, districts=...; ) ::Float64
```

Evaluates the total weighted log-energy `∑ weightᵢ · scoreᵢ` of `measure` on
`partition`.

### Energy / score functions

These can be passed to `push_energy!` and/or `push_writer!`. Each has the
signature `f(partition, districts=...; update=nothing)`.

| Function | Returns | Description |
| --- | --- | --- |
| `get_log_spanning_trees` | `Vector{Float64}` | Log spanning-tree count per district. |
| `get_log_spanning_forests` | `Float64` | Sum of log spanning-tree counts (the spanning-forest energy). |
| `get_isoperimetric_scores` | `Vector{Float64}` | Per-district isoperimetric (Polsby–Popper) ratio. |
| `get_isoperimetric_score` | `Float64` | Summed isoperimetric score; kwargs `omit_least_compact`, `pow_on_sum`, `exponent`. |
| `get_log_linking_edges` | `Float64` | Log product of cross-district linking-edge weights. |
| `get_log_district_trees` | `Float64` | Log spanning-tree count of the district adjacency graph. |
| `get_cut_edge_sum` | `Real` | Total weight of cut (cross-district) edges; kwarg `column="connections"`. |

### Geometry / tree observables on a partition

| Function | Description |
| --- | --- |
| `get_diameters(partition)` | Diameter of each district's spanning tree. |
| `get_neighbor_lists(partition)` | Neighbor-list representation of each district tree. |
| `get_degree_distributions(partition)` | Degree distribution per district tree. |
| `get_average_degrees(partition)` | Average degree per district tree. |
| `get_center_moments(partition; p=1)` | `Lᵖ` moment of vertex distances from each tree's center. |
| `get_center_leaves_moments(partition; p=1)` | `Lᵖ` moment of leaf distances from each tree's center. |

### VRA (Voting Rights Act) scores

```julia
build_performant_vra_score(graph::BaseGraph, elections::Vector;
    weights=ones(...), target_districts=nothing, num_dists=nothing,
    total_pop_col=nothing, mino_pop_col=nothing) -> score_function

build_performant_vra_report(graph::BaseGraph, elections::Vector; <same kwargs>) -> report_function

get_target_vra_districts(graph::BaseGraph, num_dists, total_pop_col, mino_pop_col)::Int
```

`build_performant_vra_score` returns an energy callback scoring how many
districts are likely to perform for the minority group; `build_performant_vra_report`
returns a callback producing a count report; `get_target_vra_districts` computes
the target number of opportunity districts.

### Partisan observables

```julia
build_get_partisan_margins(votes1::String, votes2::String) -> observable_function
build_get_partisan_seats(votes1::String, votes2::String)   -> observable_function
```

Each returns a callback `f(partition, districts; update)` giving per-district
margins or the seat count for the two given vote columns. Typically passed to
`push_writer!`.

---

## Diagnostics

Diagnostics record per-proposal statistics during a run. They live in a
`RunDiagnostics` container keyed by proposal function.

### Containers

| Type | Description |
| --- | --- |
| `RunDiagnostics` | `Dict{Function, Tuple{String, ProposalDiagnostics}}` — diagnostics per proposal. |
| `ProposalDiagnostics` | `Dict{Type, AbstractProposalDiagnostics}` — diagnostics of a single proposal. |

### `push_diagnostic!`

```julia
push_diagnostic!(run_diagnostics::RunDiagnostics, proposal::Function,
                 proposal_diagnostic::AbstractProposalDiagnostics;
                 desc::String=string(proposal))
```

Register a diagnostic to be gathered for a given proposal.

### Diagnostic types

Each is constructed with no arguments (e.g. `AcceptanceRatios()`):

| Type | Description |
| --- | --- |
| `AcceptanceRatios` | Acceptance-probability ratios per step. |
| `CycleLengthDiagnostic` | Length of the proposed cycle. |
| `DeltaNodesDiagnostic` | Number of nodes that changed district. |
| `DeltaPopDiagnostic` | Population moved between districts. |
| `CuttableEdgePairsDiagnostic` | Number of valid cuttable edge pairs. |
| `UniqueCuttableEdgesDiagnostic` | Number of unique cuttable edges. |
| `MaxSwappablePopulationDiagnostic` | Max population swappable in a proposal. |
| `AvgSwappablePopulationDiagnostic` | Avg population swappable in a proposal. |

```julia
run_diagnostics = RunDiagnostics()
push_diagnostic!(run_diagnostics, cycle_walk, AcceptanceRatios(), desc="cycle_walk")
push_diagnostic!(run_diagnostics, cycle_walk, CycleLengthDiagnostic())
```

---

## Writer (Atlas output)

The `Writer` serializes accepted plans and per-step data to an
[Atlas](https://github.com/jonmjonm/AtlasIO.jl) file (`.jsonl` or `.jsonl.gz`).

### `Writer`

```julia
Writer(
    measure::Measure,
    constraints::Constraints,
    partition::LinkCutPartition,
    output_file_path::String;
    output_districting=true,
    description::String="",
    time_stamp=string(Dates.now()),
    io_mode::String="w",
    additional_parameters::Dict{String,Any}=Dict{String,Any}(),
    weight_type::DataType=Int64,
    path_target_points::Int=50,
    include_script::Bool=true,
    config_file::Union{String,Nothing}=nothing,
    write_header::Bool=true,
    max_include_bytes::Int=MAX_INCLUDE_BYTES
)::Writer
```

Opens the output file and writes the Atlas header (energies, weights, population
bounds, constraint descriptions, package version, plus any
`additional_parameters`). `weight_type` is the type of each map's sampling weight:
the default `Int64` suits ordinary MCMC runs (every map has weight 1); pass
`Float64` when maps carry real-valued weights, as in
[`run_annealed_importance_sampling!`](#run_annealed_importance_sampling).
`path_target_points` sets the approximate number of points recorded per sample by
[`push_path_writer!`](#push_path_writer).

Execution metadata is stamped automatically: the running `"user"`, the
`"julia_version"`, the `"script_name"`, and — when `include_script=true` (the
default) — the full source of the executing script under `"script"`, followed by
the sources of every file that script `include`s under `"script_includes"` (see
[`collect_included_sources`](#collect_included_sources)). `max_include_bytes` caps
how much included source is embedded; anything past it is listed by name under
`"script_includes_skipped"`. Pass `include_script=false` to embed no source at all.

If `config_file` names an existing file, its full contents are read and appended
under `"toml_config"` as the header's very last key — after `"script"` — so it
can never be overwritten by other header data. If `config_file` is `nothing` or
doesn't point to an existing file, this is silently skipped.

Pass `write_header=false` when appending maps to an Atlas that already carries a
header (`io_mode="a"`): the header is treated as already present and none is
emitted, so the appended maps continue the existing file rather than embedding a
second header block in its middle. Nothing this run assembles reaches the file's
header in that case — see `examples/run_cyclewalk_extend.jl`, which writes a
sidecar lineage file when appending.

### `push_writer!`

```julia
push_writer!(writer::Writer, get_data::Function; desc::Union{String,Nothing}=nothing)
```

Register a per-step observable `get_data(partition)` whose value is written into
each output map under `desc`.

### `push_path_writer!`

```julia
push_path_writer!(writer::Writer, spec::Symbol; desc::Union{String,Nothing}=nothing)
push_path_writer!(writer::Writer, get_data::Function; desc::Union{String,Nothing}=nothing)
```

Register a per-step recorder along an annealing trajectory (AIS or annealed SMC
rejuvenation): roughly `writer.path_target_points` points are recorded per sample
and written into that sample's output map under `desc` (default `"path/"*string(spec)`)
as a vector alongside the ordinary observables. `spec` is either a built-in symbol —
`:log_weight` (running cumulative log importance weight), `:delta_log_weight` (that
step's increment), or `:schedule_frac` (`cur_step/total_steps`) — or a partition
observable `f(partition)` (e.g. `get_isoperimetric_score`, `get_log_spanning_forests`)
evaluated on the intermediate annealing partition. Recording is fully opt-in: with no
recorders registered there is no added cost.

### `close_writer`

```julia
close_writer(writer::Writer)
```

Flush and close the Atlas file. Call once after the run finishes.

### `collect_included_sources`

```julia
collect_included_sources(
    script_path::AbstractString;
    max_bytes::Int=MAX_INCLUDE_BYTES,
    max_depth::Int=MAX_INCLUDE_DEPTH
) -> (sources, unresolved, skipped)
```

Read the sources of every file `script_path` pulls in with `include`, following
`include`s of `include`s. Used by the `Writer` to record a run's full source in the
Atlas header, not just the top-level script — a runner script's behaviour usually
lives mostly in the files it includes.

| Returned | Contents |
| --- | --- |
| `sources` | `OrderedDict` from each file's path (relative to `script_path`'s directory) to its text, in discovery order. |
| `unresolved` | `include(…)` calls with a non-literal (computed) path, and literal includes whose file is missing or unreadable. |
| `skipped` | Files not embedded because `max_bytes` of source had already been collected. |

Each file's `include`s resolve relative to that file's own directory, matching Julia.
Commented-out includes are ignored, each path is visited once (so cycles terminate),
and nothing throws — provenance collection must never take a run down.

`CycleWalk.MAX_INCLUDE_BYTES` (1 MB) and `CycleWalk.MAX_INCLUDE_DEPTH` (8) are the
package-level defaults (unexported; pass `max_include_bytes` to `Writer` to override
per run). The header is a single JSON line that every reader parses, so the byte cap
keeps a runner that includes something large from bloating every atlas it writes.

---

## Running the Sampler

### `run_metropolis_hastings!`

```julia
run_metropolis_hastings!(
    partition::LinkCutPartition,
    proposal::Union{Function, Vector{Tuple{<:Real, Function}}},
    measure::Measure,
    steps::Union{Int, Tuple{Int,Int}},
    rng::AbstractRNG;
    writer::Union{Writer, Nothing}=nothing,
    output_freq::Int=250,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    prestepf::Function=(x...)->nothing,
    prestepargs::Tuple=(),
    output_initial::Bool=true,
    weight::Union{Real, MutableFloat}=1
)
```

Run the Metropolis–Hastings sampler in place on `partition`. `proposal` is either
a single proposal closure or a weighted mixture (weights summing to `1`).
`steps` is either a step count or an `(initial, final)` range. Every
`output_freq` steps the current plan and registered observables/diagnostics are
written to `writer`; set `output_initial=false` to suppress the map otherwise
written before the first step. `prestepf(step, prestepargs...)` is called before
each proposal — this is the hook [`run_annealed_importance_sampling!`](#run_annealed_importance_sampling)
uses to anneal `measure` and accumulate the log importance weight. `weight` is
recorded as each output map's sampling weight; ordinary runs can leave it at the
default `1` (a plain MCMC sample). `prestepf`/`prestepargs`/`weight` are the
low-level hooks AIS is built on — most callers won't need them directly.

---

## Annealed Importance Sampling & SMC

Two alternatives to sampling the target measure directly with
`run_metropolis_hastings!`: both anneal from a base measure toward the target
along a schedule, accumulating an importance weight/normalizer estimate as they go.

### `run_annealed_importance_sampling!`

```julia
run_annealed_importance_sampling!(
    partition::LinkCutPartition,
    proposal::Union{Function, Vector{Tuple{<:Real, Function}}},
    measure::Measure,
    modify_measure!::Function,
    total_steps::Int,
    base_steps_per_sample::Int,
    steps_per_annealing::Int,
    rng::AbstractRNG;
    writer::Union{Writer, Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    ntasks::Int=1
)::Vector{Float64}
```

Run annealed importance sampling (AIS) and return the vector of log importance
weights, one per annealing run. `modify_measure!(measure, cur_step, total_steps)`
defines the annealing schedule: called with `(0, 1)` it must set `measure` to the
base measure, and stepping `cur_step` from 1 to `total_steps` it must interpolate
toward the target measure. The base chain runs on `partition` under the base
measure for `base_steps_per_sample` Metropolis–Hastings steps between samples;
each sample is then deep-copied and annealed toward the target for
`steps_per_annealing` steps while its log weight is accumulated. The base chain
takes `total_steps` steps in all, so `total_steps ÷ base_steps_per_sample`
annealing runs are performed.

Annealing runs execute on `ntasks` concurrent tasks (use `julia -t N` to give them
threads); each run gets an independent, reproducible RNG. If a `writer` is
supplied — construct it with `weight_type=Float64` — sample `i` is written as map
`"sample<i>"` with its log importance weight as the map's weight.

### `run_annealed_smc!`

```julia
run_annealed_smc!(
    partition::LinkCutPartition,
    proposal::Union{Function, Vector{Tuple{<:Real, Function}}},
    measure::Measure,
    schedule::AnnealSchedule,
    n_particles::Int,
    rejuv_steps::Int,
    rng::AbstractRNG;
    path::Union{AnnealPath, Function, Nothing}=nothing,
    init_steps::Int=0,
    collect_steps::Int=0,
    collect_every::Int=100,
    resample_before_collect::Bool=true,
    writer::Union{Writer, Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics()
) -> (particles, logZ, trace)
```

Run an annealed sequential Monte Carlo (SMC) sampler: a population of
`n_particles` is jointly tempered from the base measure to the target, resampling
and rejuvenating (`rejuv_steps` Metropolis-Hastings steps per particle) as needed.
`schedule` (a [`FixedSchedule`](#fixedschedule) or [`AdaptiveTempering`](#adaptivetempering))
is the sole thing distinguishing the two variants; `path` (default:
[`linear_path`](#linear_path) built from `measure`'s target weights) is the
swappable path geometry. Returns the final (near-equally-weighted) particles, the
log-normalizer estimate `logZ`, and a per-block trace of `(t, ess, resampled)`.

With `collect_steps > 0`, after the population reaches `t=1` the sampler keeps
applying MH moves at the target measure and, to `writer`, emits every particle's
state once per `collect_every` steps — yielding far more than `n_particles`
samples without enlarging the population. With the default `collect_steps=0` the
writer gets one map per particle (the annealing-final state).
`resample_before_collect` (default `true`) resamples to equal weights at `t=1`
before amplifying, so every collected sample is unweighted; `logZ` is finalized
first, so this does not affect it.

### `FixedSchedule`

```julia
FixedSchedule(grid; ess_frac=0.5)
```

A precomputed increasing `t`-grid ending at `1.0` (e.g. `range(0, 1, length=K+1)`).
Resamples adaptively when ESS drops below `ess_frac * n_particles`.

### `AdaptiveTempering`

```julia
AdaptiveTempering(; ess_target=0.5)
```

Chooses each `t_next` online so the population ESS drops to `ess_target *
n_particles`, resampling every step.

### `LinearPath` / `linear_path`

```julia
linear_path(target_w::NTuple{K,Float64}) -> LinearPath
```

The default path geometry: energy term weights vary linearly in `t`,
`t ↦ t .* target_w`, where `target_w` is the target measure's per-term weights
(see `annealed_smc_scores_and_targets`). `LinearPath` is the underlying type; most
callers only need `linear_path`, or can pass a bare `t -> NTuple{K,Float64}`
function as `run_annealed_smc!`'s `path` keyword directly.

### `annealed_smc_scores_and_targets`

```julia
annealed_smc_scores_and_targets(measure::Measure) -> (scores, target_w)
```

Freeze a target `measure` into an ordered tuple of energy functions (`scores`)
and their target weights (`target_w`), the inputs `linear_path` expects.

---

## Chain (convenience wrapper)

`Chain` bundles a proposal, measure, writer, and RNG so a run can be launched
with a single call.

```julia
mutable struct Chain{T <: Real}
    proposal::Union{Function, Vector{Tuple{T, Function}}}
    measure::Measure
    writer::Union{Writer, Nothing}
    rng::AbstractRNG
end

Chain(proposal::Function, measure::Measure, writer::Union{Writer,Nothing}, rng::AbstractRNG)

run_chain!(partition, chain::Chain, steps::Union{Int, Tuple{Int,Int}})
```

`run_chain!` wraps `run_metropolis_hastings!` in a `try`/`catch`, returning
`(0, measure)` on success and `(1, measure)` on error.

---

## Data Structures & Types Summary

| Name | Kind | Module of origin |
| --- | --- | --- |
| `AbstractGraph` | abstract type | re-exported (MFR) |
| `BaseGraph` | struct | re-exported (MFR) |
| `MultiLevelGraph` / `Graph` | struct / alias | re-exported (MFR) |
| `AbstractPartition` | abstract type | re-exported (MFR) |
| `MultiLevelPartition` | struct | re-exported (MFR) |
| `LinkCutPartition` | mutable struct | CycleWalk |
| `Constraints` | mutable struct | CycleWalk |
| `AbstractConstraint` | abstract type | re-exported (MFR) |
| `PopulationConstraint` | struct | re-exported (MFR) |
| `PackRegionConstraint` | struct | CycleWalk |
| `CapRegionDistricts` / `CapRegionDistConstraint` | struct / alias | CycleWalk |
| `BudgetedRegionConstraint` | struct | CycleWalk |
| `Measure` | mutable struct | CycleWalk |
| `Writer` | mutable struct | CycleWalk |
| `FixedSchedule` | mutable struct | CycleWalk |
| `AdaptiveTempering` | mutable struct | CycleWalk |
| `LinearPath{K,F}` | struct | CycleWalk |
| `Chain{T}` | mutable struct | CycleWalk |
| `RunDiagnostics` | type alias (`Dict`) | CycleWalk |
| `ProposalDiagnostics` | type alias (`Dict`) | CycleWalk |
| `AcceptanceRatios` | struct | CycleWalk |
| `CycleLengthDiagnostic` | struct | CycleWalk |
| `DeltaNodesDiagnostic` | struct | CycleWalk |
| `DeltaPopDiagnostic` | struct | CycleWalk |
| `CuttableEdgePairsDiagnostic` | struct | CycleWalk |
| `UniqueCuttableEdgesDiagnostic` | struct | CycleWalk |
| `MaxSwappablePopulationDiagnostic` | struct | CycleWalk |
| `AvgSwappablePopulationDiagnostic` | struct | CycleWalk |
