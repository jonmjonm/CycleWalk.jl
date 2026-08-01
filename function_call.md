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
- [Parallel Tempering](#parallel-tempering)
- [Run Metadata & Provenance](#run-metadata--provenance)
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

### `derive_node_columns!(base_graph, specs)`

```julia
derive_node_columns!(base_graph::BaseGraph, specs) -> base_graph
```

Compute new node attributes from existing ones — e.g. a unique node name joined from
two columns that aren't unique alone, or a population adjusted by another column —
before `MultiLevelGraph` is built. `specs` is a `name => expr` mapping (an
`OrderedDict` or `Vector{Pair}` if evaluation order matters); each `expr` is evaluated
against every node's raw attributes with
[`evaluate_column_expression`](#weight_path_closure--evaluate_column_expression) and
stored under `name`, which then behaves as an ordinary column (usable as
`geo_units`, `pop_col`, or an energy's column argument). `expr` may only reference raw
columns, not another entry of `specs`. Errors if `name` collides with an existing
column. This is what `run_cyclewalk_toml.jl`'s `[plans.derive]` table calls — the
AIS/SMC/PT TOML runners do not call it, so `[plans.derive]` has no effect there.

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

### `name_proposal!` / `proposal_name` / `describe_proposal`

```julia
name_proposal!(f::Function, name::AbstractString) -> f
proposal_name(f::Function)::String
describe_proposal(proposal::Function)::String
describe_proposal(proposal::Vector{Tuple{<:Real,Function}})::Vector{Dict}
```

Optional human-readable naming for a proposal closure, so run metadata reads e.g.
`two_tree_cycle_walk` rather than an anonymous closure name. The `build_*` proposal
constructors already tag their returned closures; call `name_proposal!` yourself to
name a custom proposal. `proposal_name` falls back to `string(f)` when nothing was
registered. `describe_proposal` is what the run-metadata functions (below) call to
serialize a proposal or weighted-mixture into a run's Atlas header.

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

### `build_measure` / `energy_specs` / `measure_parameters`

Build a `Measure` from a config file rather than from code.

```julia
build_measure(measure_config::AbstractDict; context=(;))::Measure
build_measure(specs::AbstractVector{EnergySpec}, parameters::AbstractDict;
              context=(;))::Measure

energy_specs(measure_config::AbstractDict)::Vector{EnergySpec}
measure_parameters(measure_config::AbstractDict)::Dict{String, Any}
energy_weight(specs, name, parameters)::Float64
energy_weight_start(specs, name, parameters)::Float64
annealing_weights(specs, parameters)::Dict{String, Tuple{Float64, Float64}}

build_annealed_measure(specs, parameters; context=(;),
                       default_start=0.0)::Tuple{Measure, Dict}
```

`build_annealed_measure` returns the target measure *and* the schedule to ramp along:
a `Dict` from each energy **function** in the measure to its `(start, target)` weight.
Keyed by function because that is what a schedule writes into `measure.weights`, and
because a built energy has no name to look up — re-resolving a builder later would
produce a different closure than the one the measure holds. An energy with no
`weight_start` starts from `default_start` (zero: the spanning-forest base). This is
what `run_ais_toml.jl` and `run_asmc_toml.jl` build their ramp and `LinearPath` from.

`energy_weight` names a weight by what it *is* rather than by the config key it came
from. `gamma` is the weight in front of the log spanning-forest energy and
`iso_weight` the one in front of the Polsby-Popper score, so

```julia
gamma      = energy_weight(specs, "get_log_spanning_forests", parameters)
iso_weight = energy_weight(specs, "get_isoperimetric_score",  parameters)
```

reports the real thing whichever form the config was written in. The example runners
tag the atlas filename with these (`_gamma1.0`, `_iso0.3`); reading a `[measure]` key
called `gamma` instead would let that tag disagree with the measure the run actually
sampled. Returns `0.0` when the measure has no such energy.

The explicit form names each energy and how it is weighted:

```toml
[measure]
gamma = 1.0
vra_weight = 2.0                 # any numeric scalar here is a named parameter

[[measure.energy]]
name   = "get_log_spanning_forests"
weight = "gamma"                 # a number, or arithmetic over the parameters

[[measure.energy]]
name   = "get_log_district_trees"
weight = "2*gamma + 1"

[[measure.energy]]
name    = "build_get_partisan_margins"
args    = ["PRES16_D", "PRES16_R"]
weight  = "vra_weight"
desc    = "partisan margins (PRES16)"
```

| Key | Meaning |
| --- | --- |
| `name` | An energy **exported** by `CycleWalk`, or a builder returning one. |
| `weight` | A number, or an expression over the parameters (see `evaluate_weight_expression`). |
| `weight_start` | The weight at the start of an annealing schedule; absent means the energy does not anneal. |
| `args` / `kwargs` | Literal arguments for a builder. |
| `context` | Names of values the *runner* supplies (`graph`, `num_dists`, …), prepended to `args`. A config cannot write a graph down. |
| `desc` | The label in the Atlas header. Matters for builders: a built energy is a closure, whose automatic label is `#7#8`. |

The older short form is **translated** into the explicit one rather than handled
separately, so both run down one path and existing configs — including those read
back out of Atlas headers by `run_cyclewalk_extend.jl` — keep working:

```toml
[measure]
measure_scores = ["get_log_spanning_forests", "get_isoperimetric_score"]
gamma = 1.0
iso_weight = 0.3
[measure.weights]                 # for scores gamma/iso_weight do not cover
get_log_district_trees = "2*gamma + 1"
```

`"get_log_spanning_forests"` becomes a spec weighted by the expression `"gamma"`. A
config carrying both forms is an error rather than a guess.

Refused with a message naming the cause: an unexported or unknown `name`, an unknown
key (a typo would otherwise silently change the target), a missing `name`/`weight`,
a builder whose arguments do not fit, context the run does not provide, and two specs
resolving to the same function — `Measure` is keyed by function, so the second would
replace the first. Builders are exempt from that last one, since each call returns a
distinct closure.

A spec whose target `weight` is zero but whose `weight_start` is not is kept in the
measure (`push_energy!`'s `allow_zero`), because an energy annealed *down* to zero
must be present for a schedule to ramp it.

### `build_path_measure`

```julia
build_path_measure(specs::AbstractVector{EnergySpec}, parameters::AbstractDict;
                   context=(;)) -> (measure::Measure, path::AnnealPath)
```

The `weight_path` counterpart to `build_annealed_measure`: builds the target
`Measure` and, alongside it, a `LinearPath` (`AnnealPath` subtype) tempering it. For
each spec with a `weight_path` expression, the path evaluates that expression in `t`
(parsed and validated once, via `weight_path_closure`, not re-parsed per round); for
one without, the path holds constant at `weight`. This is what `[ais]`/`[smc]`/`[pt]
temper = "path"` build from. `weight_path` and `weight_start` are mutually exclusive
per spec — pick one tempering mode for the whole run, not mixed per energy.

### `weight_expression_parameters` / `referenced_parameters`

```julia
weight_expression_parameters(expr::AbstractString)::Vector{String}
referenced_parameters(specs::AbstractVector{EnergySpec})::Vector{String}
```

The names an expression reads, and the union of those across a measure's specs —
sorted, without repeats. A weight written as a plain number reads nothing.

`weight_expression_parameters` is the *same walk over the same parsed tree* as
`evaluate_weight_expression`, with names collected instead of looked up, so it refuses
exactly what evaluating refuses. It deliberately does not require the names to exist,
since it is called on a config before a run's parameters are assembled.

This is what decides a run's file name: a parameter no expression names cannot change
the target, while one that is named must appear in the name, or two runs differing
only in it would compute the same output path and the second would overwrite the
first.

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

### `weight_path_closure` / `evaluate_column_expression`

```julia
weight_path_closure(expr::AbstractString, parameters::AbstractDict)::Function   # t::Float64 -> Float64
evaluate_column_expression(expr::AbstractString, columns::AbstractDict)::Union{Float64, String}
```

`weight_path_closure` parses a `weight_path` expression once (same restricted grammar
as `evaluate_weight_expression`, plus a bound `t`) and returns a closure evaluating it
at any `t` — what `build_path_measure` calls per energy so the parse/validate cost is
paid once at startup, not once per round.

`evaluate_column_expression` is the same grammar applied at the graph layer instead of
the measure layer: it evaluates an expression against a node's raw column values
(allowing string-valued columns and string concatenation, unlike
`evaluate_weight_expression`, since composing names/ids is its whole point) and is
what `derive_node_columns!` calls per node, per `[plans.derive]` entry.

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
| `ProposalDiagnostics` | `Dict{Type, AbstractProposalDiagnostics}` — diagnostics of a single proposal. **Not exported** — this alias is internal; refer to it as `CycleWalk.ProposalDiagnostics` if needed. `push_diagnostic!`/`RunDiagnostics` are the exported surface for using it. |

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

### `stamp_execution_metadata!`

```julia
stamp_execution_metadata!(writer::Writer; include_script::Bool=true,
                          max_include_bytes::Int=MAX_INCLUDE_BYTES)
```

Fill in provenance fields on `writer`'s Atlas header that aren't already set: the OS
user (`"user"`, `"user_full_name"`), `"julia_version"`, and — when running as a
script (`PROGRAM_FILE` non-empty) — `"script_name"` and, with `include_script=true`,
the script's own source text (`"script"`, capped at `max_include_bytes`) and any
files it `include`s (`"script_includes"`), so the exact code that produced an Atlas
travels with it. This is what lets `run_cyclewalk_extend.jl` warn when the includes
on disk have drifted from what a parent run actually used. Existing keys are never
overwritten — call it before or after a run-metadata builder merges its own keys in,
either order is safe.

### `close_writer`

```julia
close_writer(writer::Writer)
```

Flush and close the Atlas file. Call once after the run finishes.

### `write_header!`

```julia
write_header!(writer::Writer)
```

Write the Atlas header — the file's first three lines — to disk, exactly once. This is
deferred from the `Writer` constructor so a `run_*!` sampler can merge its run metadata
into `atlasParam` first, and the samplers, `output` and `close_writer` all call it, so
an ordinary run never needs to. Call it by hand only to force the header out early:
`examples/run_cyclewalk_extend.jl` does so in its rewrite mode, to get the parent
Atlas' maps copied in behind a freshly written header. Idempotent.

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
    weight::Union{Real, MutableFloat}=1,
    seed=nothing
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
low-level hooks AIS is built on — most callers won't need them directly. `seed`, if
given, is stamped into the run metadata written to `writer`'s Atlas header (via
[`metropolis_hastings_run_metadata`](#run-metadata--provenance)) — it does not seed
`rng` itself, which the caller already constructed.

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
    total_steps::Int,
    base_steps_per_sample::Int,
    steps_per_annealing::Int,
    rng::AbstractRNG;
    path::Union{AnnealPath, Function, Nothing}=nothing,
    writer::Union{Writer, Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    ntasks::Int=1,
    seed=nothing
)::Vector{Float64}
```

Run annealed importance sampling (AIS) and return the vector of log importance
weights, one per annealing run. `path` (default: [`linear_path`](#linear_path) built
from `measure`'s target weights — the same convention `run_annealed_smc!` uses)
defines the annealing schedule: it must give the base measure at `t=0` and the
target measure at `t=1`; a bare `t ↦ NTuple` function is wrapped as a `LinearPath`
automatically. The base chain runs on `partition` under the base measure for
`base_steps_per_sample` Metropolis–Hastings steps between samples; each sample is
then deep-copied and annealed toward the target for `steps_per_annealing` steps
while its log weight is accumulated. The base chain takes `total_steps` steps in
all, so `total_steps ÷ base_steps_per_sample` annealing runs are performed.

Annealing runs execute on `ntasks` concurrent tasks (use `julia -t N` to give them
threads); each run gets an independent RNG seeded sequentially off `rng`, so log
weights are reproducible and identical for every `ntasks`. **Scaling ceiling**: the
base chain is serial, so by Amdahl's law the speedup from `ntasks` is capped at
`(base_steps_per_sample + steps_per_annealing) / base_steps_per_sample`, regardless
of how many threads are given. Whenever `ntasks > 1`, BLAS is pinned to 1 thread for
the duration of the run (several tasks each spawning their own BLAS pool for the
spanning-forest energy's log-determinant would oversubscribe the machine) and
restored to its prior value on exit. If a `writer` is supplied — construct it with
`weight_type=Float64` — sample `i` is written as map `"sample<i>"` with its log
importance weight as the map's weight, in base-chain order regardless of which task
finishes first. `seed`, if given, is stamped into the run metadata written to
`writer`'s Atlas header.

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
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    seed=nothing
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
n_particles`, resampling every step. Requires `init_steps > 0` in
`run_annealed_smc!` — an unburned population is `n_particles` identical clones, so
every incremental weight is equal, ESS stays at `n_particles` regardless of `t`, and
the bisection jumps straight to `t=1` on that false signal (`run_annealed_smc!`
raises an `ArgumentError` rather than silently doing this).

Both schedule types carry a cursor (`FixedSchedule`'s grid index, `AdaptiveTempering`'s
`t_prev`) that a finished run leaves exhausted. `reset_schedule!(schedule)` returns it
to the start so the same object can drive another run; `run_annealed_smc!` calls this
on entry, so most callers never need it directly.

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

## Parallel Tempering

A ladder of replicas ("rungs"), one per point on a `BetaLattice` from the untempered
base measure (rung 1, β=0) to the target (rung `M`, β=1), periodically proposing swaps
between adjacent rungs so samples move between temperatures instead of annealing
alone. Shares the same [`AnnealPath`](#linearpath--linear_path) seam annealed SMC uses
— PT calls its schedule variable β where SMC calls it `t`, but it is the same slot.
See [`docs/run_pt_toml.md`](docs/run_pt_toml.md) for the full `[pt]` config reference.

### `BetaLattice` / `linear_betas` / `geometric_betas`

```julia
BetaLattice(betas::AbstractVector{<:Real})
linear_betas(M::Int) -> BetaLattice
geometric_betas(M::Int; beta_min::Float64=0.05) -> BetaLattice
```

The inverse-temperature lattice: rung `k` sits at `betas[k]`, strictly increasing,
`betas[1] >= 0`, `betas[end] == 1.0`. `linear_betas` spaces `M` rungs evenly on
`[0,1]`; `geometric_betas` spaces them geometrically in temperature (`1/β`) between
`beta_min` and `1` (cannot include β=0 — prepend it explicitly, or use `linear_betas`,
for the exact base measure).

### `run_parallel_tempering!`

```julia
run_parallel_tempering!(
    partition::LinkCutPartition,
    proposal::Union{Function, Vector{Tuple{<:Real, Function}}},
    measure::Measure,
    lattice::BetaLattice,
    swap_interval::Int,
    n_rounds::Int,
    rng::AbstractRNG;
    path::Union{AnnealPath, Function, Nothing}=nothing,
    backend::PTBackend=SerialBackend(proposal),
    heat_bath::Union{HeatBath, Nothing}=nothing,
    init_steps::Int=0,
    write_rungs::Symbol=:target,          # :target | :all | :none
    output_every::Int=1,
    writers::Union{Vector{Writer}, Writer, Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    diagnostics::Union{PTDiagnostics, Nothing}=nothing,
    seed=nothing,
) -> (ensemble, diagnostics)
```

`M = length(lattice)` replicas are cloned from `partition`, one per rung, each with
its own RNG (seeded sequentially from `rng`), diagnostics, and private `work_measure`.
After an optional `init_steps` per-rung burn-in, each round advances every replica
`swap_interval` MH steps at its own rung (via `backend`), then attempts one swap per
adjacent rung pair on a deterministic even/odd schedule, for `n_rounds` rounds. An
optional `heat_bath` exchanges the rung the swap pattern leaves idle against an
independent draw from a reference Atlas. `write_rungs` controls which rungs' states
reach `writers` (`:target` — rung `M` only; `:all` — every rung; `:none`). Returns the
final `PTEnsemble` and the `PTDiagnostics` gathered (swap rates, round trips,
straggler gaps).

### `SerialBackend` / `ThreadedBackend`

```julia
SerialBackend(proposal)
ThreadedBackend(proposal)
```

The two `PTBackend`s `run_parallel_tempering!` can advance replicas with.
`SerialBackend` is a plain loop; `ThreadedBackend` spawns one task per rung via
`Threads.@spawn` and is verified bit-identical to `SerialBackend` for the same seed
regardless of thread count (`test/test_parallel_tempering.jl`) — give it `-t N`
matching or exceeding `M`, since extra threads beyond `M` sit idle and fewer just
queue replicas within a round. No `DistributedBackend` exists yet.

### `HeatBath` / `parse_bath_measure` / `parse_bath_samples` / `try_heat_bath!`

```julia
HeatBath(source_path::String, target_scores::NTuple{K,Function}, rng::AbstractRNG;
         rung::Int=1, burn_in::Int=0, n_samples::Union{Int,Nothing}=nothing,
         context=(;)) -> HeatBath

parse_bath_measure(source_path::String; context=(;)) -> Measure
parse_bath_samples(source_path::String, burn_in::Int, n::Int, rng::AbstractRNG) -> Vector{Dict}
try_heat_bath!(ensemble::PTEnsemble, hb::HeatBath, round::Int, rng::AbstractRNG,
               diag::PTDiagnostics)
```

`HeatBath` holds independent draws from a reference measure, read from a stored
Atlas (which must have been written with an embedded `config_file`, so its own
reference measure can be rebuilt), exchanged against `rung` (default 1, the hottest)
on rounds the swap pattern leaves it idle. Each stored sample is consumed at most
once — the run errors up front if the source doesn't have `n_samples` unused maps
past `burn_in`, rather than wrapping around and correlating exchanges. `parse_bath_measure`/
`parse_bath_samples` do the construction; `try_heat_bath!` is the exchange move itself,
called internally by `run_parallel_tempering!`.

### `PTDiagnostics` / `reset_pt_diagnostics!` / `swap_rate`

```julia
PTDiagnostics(M::Int)
reset_pt_diagnostics!(d::PTDiagnostics) -> d
swap_rate(d::PTDiagnostics) -> Vector{Float64}
```

Per-run PT diagnostics: `attempts`/`accepts`/`accept_prob_sum` per adjacent rung pair,
`occupancy[w,k]` (blocks walker `w` spent at rung `k`), `round_trips[w]` (completed
rung-`M` → rung-1 → rung-`M` journeys — the single most informative PT diagnostic),
`bath_attempts`/`bath_accepts`, and `straggler_gap` (per-block max−mean replica advance
time, for sizing `swap_interval`). Stateful — `run_parallel_tempering!` calls
`reset_pt_diagnostics!` on entry, so one object may drive several runs.
`swap_rate(d)` gives the empirical acceptance rate per adjacent pair; a well-placed
lattice has these roughly *equal* across pairs, not maximized.

---

## Run Metadata & Provenance

Each sampler entry point has a matching builder here returning a `Dict{String,Any}`
describing the run: a human-readable `"chain.run"` tag plus a nested
`"chain.parameters"` dict (schedule, walker/particle/rung count, chain lengths,
resampling, annealing endpoints, seed, proposal, …). Every `run_*!` function calls its
builder automatically and stamps the result into `writer`'s Atlas header before the
header is written — call one directly only to precompute the dict yourself. The
target measure itself (energies, weights, population bounds, constraints) is already
recorded by `Writer`, so these builders add only run/annealing-specific information.

```julia
metropolis_hastings_run_metadata(proposal, steps; seed=nothing, output_freq=250,
                                 extra=Dict()) -> Dict{String,Any}

annealed_importance_sampling_run_metadata(measure, total_steps,
    base_steps_per_sample, steps_per_annealing; seed=nothing, path=nothing,
    proposal=nothing, ntasks=1, extra=Dict()) -> Dict{String,Any}

annealed_smc_run_metadata(measure, schedule, n_particles, rejuv_steps;
    seed=nothing, path=nothing, proposal=nothing, init_steps=0, collect_steps=0,
    collect_every=100, resample_before_collect=true, extra=Dict()) -> Dict{String,Any}

parallel_tempering_run_metadata(measure, lattice, swap_interval, n_rounds;
    seed=nothing, path=nothing, proposal=nothing, backend=SerialBackend(proposal),
    init_steps=0, write_rungs=:target, output_every=1, heat_bath=nothing,
    extra=Dict()) -> Dict{String,Any}
```

Each records the annealing-path endpoints (`weights_at(path, 0)`/`weights_at(path, 1)`,
labelled by score) directly from the `AnnealPath` actually used — the same
introspectable seam `run_annealed_smc!` and `run_parallel_tempering!` share, and (for
AIS) a replacement for the older `modify_measure!` closure, which had no way to be
inspected or logged this way. `proposal` is recorded via
[`describe_proposal`](#name_proposal--proposal_name--describe_proposal). `extra` is
merged into the parameters dict for a caller's own additions; `seed` is recorded only
when given, since a run function sees only the `rng` object, not its original seed.

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
| `AnnealPath` | abstract type | CycleWalk, **not exported** — `FixedSchedule`/`AdaptiveTempering`/`LinearPath` are the exported surface; refer to it as `CycleWalk.AnnealPath` if needed. |
| `LinearPath{K,F}` | struct | CycleWalk |
| `BetaLattice` | struct | CycleWalk |
| `HeatBath` | struct | CycleWalk |
| `PTDiagnostics` | mutable struct | CycleWalk |
| `SerialBackend{P}` / `ThreadedBackend{P}` | struct / mutable struct | CycleWalk |
| `Chain{T}` | mutable struct | CycleWalk |
| `RunDiagnostics` | type alias (`Dict`) | CycleWalk |
| `ProposalDiagnostics` | type alias (`Dict`) | CycleWalk, **not exported** — refer to it as `CycleWalk.ProposalDiagnostics` if needed; use `RunDiagnostics`/`push_diagnostic!` directly instead. |
| `AcceptanceRatios` | struct | CycleWalk |
| `CycleLengthDiagnostic` | struct | CycleWalk |
| `DeltaNodesDiagnostic` | struct | CycleWalk |
| `DeltaPopDiagnostic` | struct | CycleWalk |
| `CuttableEdgePairsDiagnostic` | struct | CycleWalk |
| `UniqueCuttableEdgesDiagnostic` | struct | CycleWalk |
| `MaxSwappablePopulationDiagnostic` | struct | CycleWalk |
| `AvgSwappablePopulationDiagnostic` | struct | CycleWalk |
