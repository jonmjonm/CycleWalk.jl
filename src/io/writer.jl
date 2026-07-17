
# AtlasParam is insertion-ordered (OrderedDict) so the header's third line
# serializes its keys in a stable, deterministic order. This lets execution
# metadata such as the full "script" blob be appended last (see
# [`stamp_execution_metadata!`](@ref) and [`write_header!`](@ref)). MapParam stays a
# plain Dict; per-map key order is irrelevant.
AtlasParam=OrderedDict{String, Any}
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
`path_recorders` (the default) records nothing and adds no cost. `script_text` holds the
executing script's source (see [`stamp_execution_metadata!`](@ref)), deferred so
[`write_header!`](@ref) can append it as the header's last key, or `nothing` when script
embedding is disabled or no script is running.
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
    header_written::Bool
    script_text::Union{String, Nothing}
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

Execution metadata is also stamped automatically (see
[`stamp_execution_metadata!`](@ref)): the running `"user"`, the `"script_name"`, and —
when `include_script=true` (the default) — the full source of the executing script under
`"script"`, appended as the header's final key. Pass `include_script=false` to omit the
embedded source.
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
    path_target_points::Int=50,
    include_script::Bool=true
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

    # The Atlas header (the file's first three lines) is NOT written here: it is
    # deferred to `write_header!` so that a `run_*!` sampler can stamp its run
    # metadata into `atlasParam` first (see [`stamp_run_metadata!`](@ref)). The header
    # is written exactly once, at the top of the run — before any map — with any manual
    # write and `close_writer` guarding it as a safety net.
    io = smartOpen(output_file_path, io_mode)

    atlas = Atlas{AtlasParam}(io, description, time_stamp, atlasParam, MapParam,
                              weight_type)
    map_output_data = Dict{String, Function}()

    node_map = get_node_map(partition.node_col, partition)

    writer = Writer(atlas, MapParam(), map_output_data, output_districting,
                    node_map, partition.node_col,
                    PathRecorder[], path_target_points, false,
                    nothing)#, proposal_diagnostics)
    stamp_execution_metadata!(writer; include_script=include_script)
    return writer
end

"""
    write_header!(writer)

Write the Atlas header (the file's first three lines: format banner, header record,
and the `atlasParam` dictionary) to disk, exactly once. Deferred from the `Writer`
constructor so a `run_*!` sampler can merge its run metadata into `atlasParam` first
(see [`stamp_run_metadata!`](@ref)). Idempotent: no-op once the header is written, so it
is safe to call from every map-writing entry point.
"""
function write_header!(writer::Writer)
    writer.header_written && return
    # Append the captured script source as the header's final key, right before
    # serialization, so it lands last on the third line regardless of any run
    # metadata stamped in between (AtlasParam is an OrderedDict). A user-supplied
    # `additional_parameters["script"]` keeps its own position and wins.
    if writer.script_text !== nothing && !haskey(writer.atlas.atlasParam, "script")
        writer.atlas.atlasParam["script"] = writer.script_text
    end
    atlasHeader = AtlasHeader(writer.atlas.description, writer.atlas.date,
                              AtlasParam, MapParam;
                              weightType=writer.atlas.weightType)
    newAtlas(writer.atlas.io, atlasHeader, writer.atlas.atlasParam)
    writer.header_written = true
    return
end

"""
    real_name() -> String

Best-effort lookup of the full ("real") name of the OS user running this process,
from the OS user database rather than the environment. Returns an empty string when
no full name is available (a common case on servers, containers, and CI, where the
field is unset). Platform sources:

- macOS: `id -F` (the Directory Services full name).
- Linux: the first comma-separated component of the GECOS field of the `getent passwd`
  entry (extra office/phone components are dropped).
- Windows: the account's `FullName` via WMI.

The name is free-form, user/admin-set text — fine for provenance labeling, but not an
authenticated identity. Any failure (missing tool, empty database entry) is swallowed
and yields `""`.
"""
function real_name()
    try
        if Sys.isapple()
            return strip(read(`id -F`, String))
        elseif Sys.islinux()
            user = get(ENV, "USER", "")
            isempty(user) && return ""
            line = strip(read(`getent passwd $user`, String))
            fields = split(line, ':')
            length(fields) < 5 && return ""
            return strip(split(fields[5], ',')[1])
        elseif Sys.iswindows()
            name = get(ENV, "USERNAME", "")
            isempty(name) && return ""
            cmd = "(Get-CimInstance Win32_UserAccount -Filter \"Name='$name'\").FullName"
            return strip(read(`powershell -NoProfile -Command $cmd`, String))
        end
    catch
    end
    return ""
end

"""
    stamp_execution_metadata!(writer; include_script=true)

Stamp who and what is producing this Atlas into the header's `atlasParam`: the running
`"user"` (from `ENV["USER"]`/`ENV["USERNAME"]`), the `"user_full_name"` (from
[`real_name`](@ref), omitted when unavailable), and, when a script is being run
(`PROGRAM_FILE` is non-empty), its `"script_name"`. When `include_script=true` (the
default) the executing script's full source is captured and, by
[`write_header!`](@ref), appended as the header's **last** key under `"script"`; pass
`include_script=false` to omit the embedded source. Called automatically by the
[`Writer`](@ref) constructor, but also safe to call by hand on an existing writer before
its header is written. Each field is set only if absent, so a user's
`additional_parameters` win; a no-op in the REPL for the script fields (no
`PROGRAM_FILE`), and an unreadable script is skipped rather than raised.
"""
function stamp_execution_metadata!(writer::Writer; include_script::Bool=true)
    ap = writer.atlas.atlasParam
    haskey(ap, "user") ||
        (ap["user"] = get(ENV, "USER", get(ENV, "USERNAME", "unknown")))
    if !haskey(ap, "user_full_name")
        name = real_name()
        isempty(name) || (ap["user_full_name"] = name)
    end
    if !isempty(PROGRAM_FILE)
        haskey(ap, "script_name") || (ap["script_name"] = basename(PROGRAM_FILE))
        if include_script && !haskey(ap, "script") && isfile(PROGRAM_FILE)
            try
                writer.script_text = read(PROGRAM_FILE, String)
            catch
                writer.script_text = nothing
            end
        end
    end
    return writer
end

"""
    stamp_run_metadata!(writer, metadata)

Merge a sampler's run-metadata dict (see the `*_run_metadata` builders) into the
writer's `atlasParam` and then write the header. Existing keys are preserved — any
`additional_parameters` the caller passed to [`Writer`](@ref) win over the auto-stamped
fields — so a user's own entries are never clobbered. No-op if `writer === nothing` or
the header is already written (metadata must be stamped before the first map). Called at
the top of each `run_*!` sampler.
"""
function stamp_run_metadata!(writer::Union{Writer, Nothing}, metadata::AbstractDict)
    writer === nothing && return
    if !writer.header_written
        for (k, v) in metadata
            haskey(writer.atlas.atlasParam, k) || (writer.atlas.atlasParam[k] = v)
        end
    end
    write_header!(writer)
    return
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
    write_header!(writer)   # ensure a valid file even if no map was ever written
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
    write_header!(writer)   # safety net for direct map writes (no-op after the first)
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