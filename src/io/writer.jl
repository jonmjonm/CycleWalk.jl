
AtlasParam=Dict{String, Any}
MapParam=Dict{String, Any}

"""
    PathRecorder(desc, kind, f)

One entry of a [`Writer`](@ref)'s annealing-path recorder list (see
[`push_path_writer!`](@ref)). During annealed importance sampling, every `path_stride`
steps its value is appended to the sample's path buffer under key `desc`. `kind` is one
of the built-in symbols `:log_weight` (running cumulative log importance weight),
`:delta_log_weight` (that step's increment), `:schedule_frac` (`cur_step/total_steps`),
or `:observable`, in which case `f(partition)` is evaluated on the intermediate
annealing partition. Only registered recorders are ever evaluated, so recording adds no
cost when none are registered.
"""
struct PathRecorder
    desc::String
    kind::Symbol
    f::Union{Function, Nothing}
end

"""
    Writer

Serializes accepted plans and per-step data to an
[Atlas](https://github.com/jonmjonm/AtlasIO.jl) file. `atlas` is the open output
handle; `map_output_data` maps each registered observable's description to its getter
(see [`push_writer!`](@ref)); `map_param` buffers the current step's observable
values; `output_districting` selects whether each map records the full node→district
assignment; `node_map`/`node_field` describe how nodes are keyed in the output.
`path_recorders`/`path_target_points` configure per-sample annealing-path recording for
annealed importance sampling (see [`push_path_writer!`](@ref)); an empty
`path_recorders` (the default) records nothing and adds no cost.
"""
mutable struct Writer
    atlas::Atlas{AtlasParam}
    map_param::MapParam
    map_output_data::Dict{String, Function}
    output_districting::Bool
    node_map::Dict{Tuple{Vararg{String}}, Int}
    node_field::String
    path_recorders::Vector{PathRecorder}
    path_target_points::Int
end

"""
    Writer(measure, constraints, partition, output_file_path; output_districting=true,
           description="", time_stamp=string(Dates.now()), io_mode="w",
           additional_parameters=Dict{String,Any}(), weight_type=Int64)

Open `output_file_path` (`.jsonl` or `.jsonl.gz`) and write the Atlas header,
recording the measure's energies and weights, the population bounds and constraint
descriptions, the CycleWalk package version, and any `additional_parameters`. Creates
the output directory if needed. Register observables with [`push_writer!`](@ref) and
close with [`close_writer`](@ref). `weight_type` is the type of each map's sampling
weight, recorded in the Atlas header: the default `Int64` suits ordinary MCMC runs
(every map has weight 1); pass `Float64` when maps carry real-valued weights, as in
[`run_annealed_importance_sampling!`](@ref).
"""
function Writer(
    measure::Measure,
    constraints::Constraints,
    partition::LinkCutPartition,
    output_file_path::String;
    output_districting=true,
    description::String="",
    time_stamp=string(Dates.now()),
    io_mode::String="w",
    additional_parameters::Dict{String, Any}=Dict{String,Any}(),
    weight_type::DataType=Int64,
    path_target_points::Int=50
    # proposal_diagnostics::Dict=Dict()
)
    graph = partition.graph
    scores = collect(measure.scores)
    energies = [measure.descriptions[e] for e in scores]
    weights = [measure.weights[e] for e in scores]
    atlasParam=AtlasParam("energies"=>energies, "energy weights"=>weights,
                          "districts"=>partition.num_dists)

    min_pop = constraints.population_constraint.min_pop
    max_pop = constraints.population_constraint.max_pop
    atlasParam["population bounds"] = [min_pop, max_pop]

    for ii = 1:length(constraints.constraints)
        desc = constraints.descriptions[ii]
        push!(get!(atlasParam, "constraints", String[]), desc)
    end

    f = @__FILE__

    versionNumber = pkgversion(CycleWalk)
    versionString = string(versionNumber)
    atlasParam["package.version"] = "CycleWalk v"*versionString

    for (key,val) in additional_parameters
        atlasParam[key] = val
    end
    
    # to add to atlasParam
    # other constraints

    dir = dirname(output_file_path)
    # split_path = split(output_file_path, "/")
    # dir = join(split_path[1:length(split_path)-1], "/")
    if !isdir(dir)
        mkpath(dir)
    end

    atlasHeader = AtlasHeader(description, time_stamp, AtlasParam, MapParam;
                              weightType=weight_type)
    io = smartOpen(output_file_path, io_mode)
    newAtlas(io, atlasHeader, atlasParam)

    atlas = Atlas{AtlasParam}(io, description, time_stamp, atlasParam, MapParam,
                              weight_type)
    map_output_data = Dict{String, Function}()

    node_map = get_node_map(partition.node_col, partition)

    return Writer(atlas, MapParam(), map_output_data, output_districting,
                  node_map, partition.node_col,
                  PathRecorder[], path_target_points)#, proposal_diagnostics)
end

"""
    push_writer!(writer, get_data; desc=nothing)

Register a per-step observable `get_data(partition)` whose value is written into each
output map under the key `desc` (defaulting to the function's name).
"""
function push_writer!(
    writer::Writer,
    get_data::Function;
    desc::Union{String, Nothing}=nothing
)
    if desc == nothing
        desc = string(get_data)
    end
    writer.map_output_data[desc] = get_data
end

"""
    push_path_writer!(writer, spec; desc=nothing)

Register an annealing-path recorder on `writer`, the path-time analogue of
[`push_writer!`](@ref). During [`run_annealed_importance_sampling!`](@ref) each
registered recorder's value is sampled every `path_stride` steps (chosen so each
annealed sample yields about `writer.path_target_points` points) and written to that
sample's output map under `desc`, as a vector alongside the ordinary observables.

`spec` is either a built-in symbol — `:log_weight` (running cumulative log importance
weight), `:delta_log_weight` (that step's increment), or `:schedule_frac`
(`cur_step/total_steps`) — or a partition observable `f(partition)` (e.g.
`get_isoperimetric_score`, `get_log_spanning_forests`) evaluated on the intermediate
annealing partition. `desc` defaults to `"path/"*string(spec)`. Recording is fully
opt-in: with no recorders registered, annealing runs the original hook with no added
cost; a partition-observable recorder costs one evaluation per stride only when set.
"""
function push_path_writer!(
    writer::Writer,
    spec::Symbol;
    desc::Union{String, Nothing}=nothing
)
    spec in (:log_weight, :delta_log_weight, :schedule_frac) ||
        throw(ArgumentError("unknown path recorder symbol :$spec (expected " *
              ":log_weight, :delta_log_weight, or :schedule_frac)"))
    d = desc === nothing ? "path/"*string(spec) : desc
    push!(writer.path_recorders, PathRecorder(d, spec, nothing))
end

function push_path_writer!(
    writer::Writer,
    get_data::Function;
    desc::Union{String, Nothing}=nothing
)
    d = desc === nothing ? "path/"*string(get_data) : desc
    push!(writer.path_recorders, PathRecorder(d, :observable, get_data))
end

"""
    close_writer(writer)

Flush and close the underlying Atlas file. Call once after the run finishes.
"""
function close_writer(writer::Writer)
    close(writer.atlas.io)
end

"""
    get_node_map(node_field, partition, node_map=nothing)

Build (or refill) a map from each node's `node_field` key to its current district
index, over all base nodes of `partition`. Reuses `node_map` if provided.
"""
function get_node_map(
    node_field::String,
    partition::LinkCutPartition,
    node_map::Union{Nothing, Dict{Tuple{Vararg{String}}, Int}} = nothing
)
    if node_map === nothing
        node_map = Dict{Tuple{Vararg{String}}, Int}()
    end

    graph = partition.graph
    for ni = 1:graph.num_nodes
        node_name = Tuple([graph.node_attributes[ni][node_field]])
        node_map[node_name] = partition.node_to_dist[ni]
    end
    return node_map
end


"""
    get_node_map!(writer, partition)

Refresh the writer's node→district map in place from the current `partition` and
return it.
"""
function get_node_map!(
    writer::Writer,
    partition::LinkCutPartition
)
    return get_node_map(writer.node_field, partition, writer.node_map)
end

"""
    fill_map_param!(map_param, writer, partition, run_diagnostics)

Evaluate every observable registered on `writer` and every recorded diagnostic into
`map_param` (a `Dict{String, Any}`) for the current `partition`, and return it.
"""
function fill_map_param!(
    map_param::MapParam,
    writer::Writer,
    partition::LinkCutPartition,
    run_diagnostics::RunDiagnostics
)
    for (desc, f) in writer.map_output_data
        map_param[desc] = f(partition)
    end
    for (rd_desc, proposal_diagnostics) in values(run_diagnostics)
        for (pd_desc, proposal_diagnostic) in proposal_diagnostics
            desc = "("*join([rd_desc, pd_desc], ",")*")"
            map_param[desc] = proposal_diagnostic.data_vec
        end
    end
    return map_param
end

"""
    build_output_map(writer, partition, name, weight=1,
                     run_diagnostics=RunDiagnostics())

Build and return the Atlas `Map` that [`output`](@ref) would write for the current
`partition`, named `name` and carrying the sampling weight `weight`, without
touching the writer's file or its reusable buffers. Every container in the returned
map is freshly allocated, so this is safe to call concurrently from several tasks
sharing one `Writer` (each with its own `partition` and `run_diagnostics`); only
the eventual `addMap` call must be serialized.
"""
function build_output_map(
    writer::Writer,
    partition::LinkCutPartition,
    name::String,
    weight::Real=1,
    run_diagnostics::RunDiagnostics=RunDiagnostics();
    extra_data::Union{Nothing, AbstractDict}=nothing
)
    map_param = fill_map_param!(MapParam(), writer, partition, run_diagnostics)
    extra_data !== nothing && merge!(map_param, extra_data)
    if !writer.output_districting
        districting = Dict{Tuple{Vararg{String}}, Int}()
    else
        districting = get_node_map(writer.node_field, partition)
    end
    return Map{MapParam}(name, districting, weight, map_param)
end

"""
    output(partition, measure, step, count, writer, run_diagnostics=RunDiagnostics();
           weight=1)

Write one Atlas map for the current `partition`: evaluate every registered observable
and diagnostic into the map's parameters, attach the node→district assignment (unless
`output_districting` is false), append the map to the Atlas, and reset the
diagnostics. No-op if `writer === nothing`. `weight` is recorded as the map's
sampling weight (a [`MutableFloat`](@ref) is unboxed); its type should match the
writer's `weight_type`.
"""
function output(
    partition::LinkCutPartition,
    measure::Measure,
    step::Integer,
    count::Int,
    writer::Union{Writer, Nothing},
    run_diagnostics::RunDiagnostics=RunDiagnostics();
    weight::Union{Real, MutableFloat}=1
)
    if writer == nothing
        return
    end
    if weight isa MutableFloat
        weight = weight.value
    end

    fill_map_param!(writer.map_param, writer, partition, run_diagnostics)

    # @show writer.map_param
########## get_map() or something
    if !writer.output_districting
        d = Dict{Tuple{Vararg{String}}, Int}()
        map = Map{MapParam}("step"*string(step-count), d, weight,
                            writer.map_param)
    else
        map = Map{MapParam}("step"*string(step-count),
                            get_node_map!(writer, partition),
                            weight, writer.map_param)
    end
##########

    try
        addMap(writer.atlas.io, map)
    catch e
        @show writer.map_param
        @show count
        # println("partition.node_to_district: ", partition.node_to_district)
        println("Could not add map to atlas")
        @assert false
    end
    reset_diagnostics!(run_diagnostics)
end