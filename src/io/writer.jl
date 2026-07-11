
AtlasParam=Dict{String, Any}
MapParam=Dict{String, Any}

"""
    Writer

Serializes accepted plans and per-step data to an
[Atlas](https://github.com/jonmjonm/AtlasIO.jl) file. `atlas` is the open output
handle; `map_output_data` maps each registered observable's description to its getter
(see [`push_writer!`](@ref)); `map_param` buffers the current step's observable
values; `output_districting` selects whether each map records the full node→district
assignment; `node_map`/`node_field` describe how nodes are keyed in the output.
"""
mutable struct Writer
    atlas::Atlas{AtlasParam}
    map_param::MapParam
    map_output_data::Dict{String, Function}
    output_districting::Bool
    node_map::Dict{Tuple{Vararg{String}}, Int}
    node_field::String
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
    weight_type::DataType=Int64
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
                  node_map, partition.node_col)#, proposal_diagnostics)
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

    for (desc, f) in writer.map_output_data
        writer.map_param[desc] = f(partition)
    end
    for (rd_desc, proposal_diagnostics) in values(run_diagnostics)
        for (pd_desc, proposal_diagnostic) in proposal_diagnostics
            desc = "("*join([rd_desc, pd_desc], ",")*")"
            writer.map_param[desc] = proposal_diagnostic.data_vec
        end
    end

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