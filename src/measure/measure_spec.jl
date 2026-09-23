# Building a Measure from a config file rather than from code.
#
# A run's target is a weighted sum of energies. Written in Julia that is a handful of
# `push_energy!` calls; written in a TOML config it has to survive a round trip through
# strings — which is what these types do. A config names each energy, says how it is
# weighted (a number, or arithmetic over the run's named parameters), and, for the
# energies that are built rather than called directly, supplies the arguments its
# builder needs.
#
# Everything here treats the config as data from an untrusted file: names resolve only
# against CycleWalk's exported bindings, weights are interpreted and never `eval`ed
# (see `evaluate_weight_expression`), and anything unrecognized is an error naming what
# was rejected rather than a silently different target.

"""
    EnergySpec(name; weight, weight_start=nothing, args=[], kwargs=Dict(),
               context=String[], desc="")

One energy of a [`Measure`](@ref), as a config file describes it. Built from a config
by [`energy_specs`](@ref) and turned into a `Measure` by [`build_measure`](@ref).

- `name`: an energy function exported by `CycleWalk`, e.g. `"get_log_spanning_forests"`,
  or a builder that returns one, e.g. `"build_get_partisan_margins"`.
- `weight`: a number, or a string of arithmetic over the run's named parameters
  (`"2*gamma + 1"` — see [`evaluate_weight_expression`](@ref)).
- `weight_start`: the weight at the start of an annealing schedule, in the same two
  forms, or `nothing` for a run that does not anneal. A target `weight` of zero with a
  nonzero `weight_start` is what an energy annealed *down* to zero looks like, and is
  the case [`push_energy!`](@ref)'s `allow_zero` exists for.
- `args` / `kwargs`: literal arguments for a builder, from the config.
- `context`: names of values the *runner* supplies — the graph, the district count —
  prepended to `args` in the order given. A config cannot write a graph down, so a
  builder that needs one names it here (see [`build_measure`](@ref)'s `context`).
- `desc`: the label recorded in the Atlas header. Matters most for builders: a built
  energy is a closure, whose automatic label is something like `#7#8`.
"""
struct EnergySpec
    name::String
    weight::Any
    weight_start::Any
    weight_path::Union{String, Nothing}
    args::Vector{Any}
    kwargs::Dict{Symbol, Any}
    context::Vector{String}
    desc::String
end

function EnergySpec(name::AbstractString; weight,
                    weight_start=nothing,
                    weight_path::Union{AbstractString, Nothing}=nothing,
                    args::AbstractVector=Any[],
                    kwargs::AbstractDict=Dict{Symbol, Any}(),
                    context::AbstractVector=String[],
                    desc::AbstractString="")
    weight_start === nothing || weight_path === nothing ||
        throw(ArgumentError("energy \"$name\" has both weight_start and " *
                            "weight_path; they belong to different tempering modes " *
                            "(\"linear\" vs \"path\") — use one or the other"))
    return EnergySpec(String(name), weight, weight_start,
                      weight_path === nothing ? nothing : String(weight_path),
                      collect(Any, args),
                      Dict{Symbol, Any}(Symbol(k) => v for (k, v) in kwargs),
                      String[String(c) for c in context], String(desc))
end

"""
    measure_parameters(measure_config) -> Dict{String, Any}

The named parameters a weight expression may use: every numeric scalar in the config's
`[measure]` table. `gamma` and `iso_weight` are therefore always available to an
expression without being special-cased anywhere.

`t` may not be one of them: it is reserved for `weight_path` expressions (the
tempering fraction — see [`weight_path_closure`](@ref)), so a `[measure]` table that
defines its own `t` is rejected here, before it could silently collide.
"""
function measure_parameters(measure_config::AbstractDict)
    params = Dict{String, Any}(String(k) => v for (k, v) in measure_config
                               if v isa Real && !(v isa Bool))
    haskey(params, "t") &&
        throw(ArgumentError("[measure] defines a parameter named \"t\", which is " *
                            "reserved for weight_path expressions (the tempering " *
                            "fraction); rename it"))
    return params
end

"""
    energy_specs(measure_config) -> Vector{EnergySpec}

Read a config's `[measure]` table into a list of [`EnergySpec`](@ref)s, accepting
either the explicit form or the older short form.

The explicit form names each energy and its weight:

```toml
[[measure.energy]]
name   = "get_log_spanning_forests"
weight = "gamma"

[[measure.energy]]
name    = "build_get_partisan_margins"
args    = ["PRES16_D", "PRES16_R"]
weight  = 2.0
desc    = "partisan margins (PRES16)"
```

The short form lists score names and weights them with `gamma`, `iso_weight`, and a
`[measure.weights]` table:

```toml
[measure]
measure_scores = ["get_log_spanning_forests", "get_log_district_trees"]
gamma = 1.0

[measure.weights]
get_log_district_trees = "2*gamma + 1"
```

The short form is *translated* into the explicit one rather than handled separately —
`"get_log_spanning_forests"` becomes a spec weighted by the expression `"gamma"` — so
both run down one path and old configs keep working. That matters beyond politeness:
`examples/run_cyclewalk_extend.jl` reads configs back out of Atlas headers written
before this schema existed, and could not extend those runs otherwise.

A config carrying both forms is an error rather than a guess at which was meant.
"""
function energy_specs(measure_config::AbstractDict)
    explicit = get(measure_config, "energy", nothing)
    scores = get(measure_config, "measure_scores", nothing)

    if explicit !== nothing && scores !== nothing
        throw(ArgumentError("[measure] has both an [[measure.energy]] list and a " *
                            "measure_scores list; use one or the other"))
    elseif explicit !== nothing
        return _explicit_specs(explicit)
    elseif scores !== nothing
        return _short_form_specs(measure_config, scores)
    else
        return EnergySpec[]
    end
end

const _ENERGY_SPEC_KEYS = Set(["name", "weight", "weight_start", "weight_path",
                               "args", "kwargs", "context", "desc"])

function _explicit_specs(entries)
    entries isa AbstractVector ||
        throw(ArgumentError("[measure.energy] must be a list of tables " *
                            "([[measure.energy]]), got $(typeof(entries))"))
    specs = EnergySpec[]
    for (ii, entry) in enumerate(entries)
        entry isa AbstractDict ||
            throw(ArgumentError("entry $ii of [[measure.energy]] is not a table"))
        haskey(entry, "name") ||
            throw(ArgumentError("entry $ii of [[measure.energy]] has no name"))
        name = String(entry["name"])
        haskey(entry, "weight") ||
            throw(ArgumentError("energy \"$name\" has no weight"))
        # An unknown key is far more likely a typo than an intention, and silently
        # ignoring it would sample something other than what was written down.
        for key in keys(entry)
            String(key) in _ENERGY_SPEC_KEYS ||
                throw(ArgumentError("energy \"$name\" has unknown key \"$key\" " *
                                    "(expected: $(join(sort!(collect(_ENERGY_SPEC_KEYS)), ", ")))"))
        end
        weight_path = get(entry, "weight_path", nothing)
        weight_path === nothing || weight_path isa AbstractString ||
            throw(ArgumentError("energy \"$name\"'s weight_path is " *
                                "$(repr(weight_path)), not a string"))
        push!(specs, EnergySpec(name;
                                weight       = entry["weight"],
                                weight_start = get(entry, "weight_start", nothing),
                                weight_path  = weight_path,
                                args         = get(entry, "args", Any[]),
                                kwargs       = get(entry, "kwargs", Dict{String, Any}()),
                                context      = get(entry, "context", String[]),
                                desc         = get(entry, "desc", "")))
    end
    return specs
end

# The short form's two named scores are weighted by the parameters they have always
# been weighted by; everything else takes its weight from [measure.weights]. It exists
# for the configs of released runs — every Atlas the sampler has written embeds one,
# and `examples/run_cyclewalk_extend.jl` reads them back to extend those runs — so it
# only ever describes a target. An annealing base is a `weight_start` in the explicit
# form; the short form has no way to say one.
const _SHORT_FORM_PARAMETERS = Dict("get_log_spanning_forests" => "gamma",
                                    "get_isoperimetric_score"  => "iso_weight")

function _short_form_specs(measure_config::AbstractDict, scores)
    weights = get(measure_config, "weights", Dict{String, Any}())
    specs = EnergySpec[]
    for score in scores
        name = String(score)
        weight = if haskey(_SHORT_FORM_PARAMETERS, name)
            _SHORT_FORM_PARAMETERS[name]        # the expression "gamma"/"iso_weight"
        elseif haskey(weights, name)
            weights[name]
        else
            throw(ArgumentError("no weight for measure score \"$name\": add it " *
                                "under [measure.weights], or use the " *
                                "[[measure.energy]] form"))
        end
        push!(specs, EnergySpec(name; weight=weight))
    end
    return specs
end

"""
    resolve_energy_weight(value, parameters, name) -> Float64

Turn a configured weight into a number: a number is taken as written, a string is
arithmetic over `parameters` (see [`evaluate_weight_expression`](@ref)). `name` only
identifies the energy in error messages.
"""
function resolve_energy_weight(value, parameters::AbstractDict, name::AbstractString)
    value isa Real && !(value isa Bool) && return Float64(value)
    if value isa AbstractString
        try
            return evaluate_weight_expression(value, parameters)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("weight of energy \"$name\": $(err.msg)"))
        end
    end
    throw(ArgumentError("weight of energy \"$name\" is $(repr(value)); expected a " *
                        "number or an expression over the [measure] parameters"))
end

"""
    resolve_energy_function(spec, context) -> (f, label)

Resolve a spec's `name` to the energy function to push, and the label to record for
it. The name must be **exported** by `CycleWalk` — resolving against every binding in
the module would let a config reach internals and non-energies alike.

With no `args`, `kwargs` or `context`, the named function is the energy. Otherwise it
is treated as a builder and called: values named in `spec.context` are looked up in
`context` and passed first, then `spec.args`, then `spec.kwargs` as keywords. Each
call returns its own closure, so the same builder may appear several times in one
measure with different arguments.
"""
function resolve_energy_function(spec::EnergySpec, context)
    sym = Symbol(spec.name)
    sym in names(CycleWalk) ||
        throw(ArgumentError("unknown energy \"$(spec.name)\": CycleWalk exports no " *
                            "such name"))
    f = getfield(CycleWalk, sym)
    f isa Function ||
        throw(ArgumentError("energy \"$(spec.name)\" is not a function"))

    built = !isempty(spec.args) || !isempty(spec.kwargs) || !isempty(spec.context)
    built || return f, spec.desc

    positional = Any[]
    for cname in spec.context
        haskey(context, Symbol(cname)) ||
            throw(ArgumentError("energy \"$(spec.name)\" asks for context " *
                                "\"$cname\", which this run does not provide " *
                                "(available: $(join(sort!(collect(String.(keys(context)))), ", ")))"))
        push!(positional, context[Symbol(cname)])
    end
    append!(positional, spec.args)

    energy = try
        f(positional...; spec.kwargs...)
    catch err
        throw(ArgumentError("could not build energy \"$(spec.name)\" from its " *
                            "configured arguments: $(sprint(showerror, err))"))
    end
    energy isa Function ||
        throw(ArgumentError("\"$(spec.name)\" returned $(typeof(energy)), not an " *
                            "energy function"))
    # A built energy is a closure, whose automatic label is something like "#7#8", so
    # fall back to the builder's name rather than leaving that in the Atlas header.
    return energy, isempty(spec.desc) ? spec.name : spec.desc
end

"""
    build_measure(specs, parameters; context=(;)) -> Measure

Assemble a [`Measure`](@ref) from a config's [`EnergySpec`](@ref)s and its named
`parameters` (see [`measure_parameters`](@ref)). `context` supplies the values a
config cannot write down — typically `(graph=..., num_dists=...)` — for builders that
name them (see [`resolve_energy_function`](@ref)).

Each spec's weight is resolved, and the energy pushed with
[`push_energy!`](@ref). A spec whose target weight is zero but whose `weight_start` is
not is kept in the measure anyway (`allow_zero`), since an energy annealed *down* to
zero must be present for a schedule to ramp it.

Two specs resolving to the same function is an error: `Measure` is keyed by function,
so the second would silently replace the first. Builders are exempt — each call
returns a distinct closure — so one builder may appear repeatedly with different
arguments.
"""
function build_measure(specs::AbstractVector{EnergySpec},
                       parameters::AbstractDict;
                       context=(;))
    measure, _ = _build_measure(specs, parameters, context, nothing)
    return measure
end

"""
    build_annealed_measure(specs, parameters; context=(;), default_start=0.0)
        -> (measure, ramp)

Build the target [`Measure`](@ref) as [`build_measure`](@ref) does, and alongside it
the schedule an annealing run ramps along: `ramp` maps each energy *function* in the
measure to its `(start, target)` weight.

Keyed by function rather than by name because that is what a schedule has to write
into `measure.weights`, and because a built energy has no name to look up — two specs
naming one builder resolve to two distinct closures, and re-resolving them later would
produce two more that the measure has never seen.

An energy with no `weight_start` starts from `default_start`, zero by default: for
[`run_annealed_importance_sampling!`](@ref) and [`run_annealed_smc!`](@ref) that is the
base measure the base chain samples, and zero recovers the spanning-forest base. An
energy whose target weight is zero but whose start is not is kept in the measure (see
[`push_energy!`](@ref)'s `allow_zero`), since a schedule cannot ramp an energy the
measure does not have.
"""
function build_annealed_measure(specs::AbstractVector{EnergySpec},
                                parameters::AbstractDict;
                                context=(;),
                                default_start::Real=0.0)
    return _build_measure(specs, parameters, context, Float64(default_start))
end

# Resolve every spec once. `default_start === nothing` means "not an annealing run":
# no ramp is collected and only an explicit weight_start affects `allow_zero`.
function _build_measure(specs::AbstractVector{EnergySpec},
                        parameters::AbstractDict,
                        context,
                        default_start::Union{Float64, Nothing})
    measure = Measure()
    seen = Dict{Function, String}()
    ramp = Dict{Function, Tuple{Float64, Float64}}()
    for spec in specs
        energy, label = resolve_energy_function(spec, context)
        if haskey(seen, energy)
            throw(ArgumentError("energies \"$(seen[energy])\" and \"$(spec.name)\" " *
                                "are the same function; a measure can weight it once"))
        end
        seen[energy] = spec.name

        weight = resolve_energy_weight(spec.weight, parameters, spec.name)
        start = if spec.weight_start !== nothing
            resolve_energy_weight(spec.weight_start, parameters, spec.name)
        else
            default_start
        end
        allow_zero = start !== nothing && start != 0
        push_energy!(measure, energy, weight; desc=label, allow_zero=allow_zero)
        # a score push_energy! dropped is not in the measure, so nothing may ramp it
        default_start === nothing || energy in measure.scores || continue
        default_start === nothing || (ramp[energy] = (start, weight))
    end
    return measure, ramp
end

"""
    build_path_measure(specs, parameters; context=(;)) -> (measure, path)

Build the target [`Measure`](@ref) as [`build_measure`](@ref) does, and alongside it
an [`AnnealPath`](@ref) (a `LinearPath` — see its docs; "linear" names the
energy-in-phi relationship, not the shape of this schedule) tempering it: for each
spec with a `weight_path` expression, that expression (parsed and validated ONCE,
here — see [`weight_path_closure`](@ref) — not re-parsed per round); for one without,
a constant at its `weight` (present in the measure the whole run, untempered).

Like [`build_annealed_measure`](@ref), a spec whose `weight` happens to be zero but
that carries a `weight_path` is kept in the measure anyway (`allow_zero`) — the
schedule may still make it nonzero at some `t`, and it must have a slot in `phi` for
that to reach the sampler at all.

`weight_path` and `weight_start` are mutually exclusive per spec ([`EnergySpec`](@ref)
rejects both at once) — "linear" and "path" are different tempering modes for the
whole run, not something mixed per energy.
"""
function build_path_measure(specs::AbstractVector{EnergySpec},
                            parameters::AbstractDict;
                            context=(;))
    measure = Measure()
    seen = Dict{Function, String}()
    weight_fns = Dict{Function, Function}()   # energy -> (t::Float64 -> Float64)
    for spec in specs
        energy, label = resolve_energy_function(spec, context)
        if haskey(seen, energy)
            throw(ArgumentError("energies \"$(seen[energy])\" and \"$(spec.name)\" " *
                                "are the same function; a measure can weight it once"))
        end
        seen[energy] = spec.name

        weight = resolve_energy_weight(spec.weight, parameters, spec.name)
        push_energy!(measure, energy, weight; desc=label,
                    allow_zero=spec.weight_path !== nothing)
        weight_fns[energy] = spec.weight_path === nothing ?
            (_t -> weight) : weight_path_closure(spec.weight_path, parameters)
    end
    scores = Tuple(measure.scores)
    K = length(scores)
    fns = ntuple(k -> weight_fns[scores[k]], K)
    path = LinearPath{K}(t -> ntuple(k -> fns[k](t), K))
    return measure, path
end

"""
    build_measure(measure_config; context=(;)) -> Measure

Build a [`Measure`](@ref) straight from a config's `[measure]` table, reading its
energies with [`energy_specs`](@ref) and its parameters with
[`measure_parameters`](@ref).
"""
build_measure(measure_config::AbstractDict; context=(;)) =
    build_measure(energy_specs(measure_config), measure_parameters(measure_config);
                  context=context)

"""
    energy_weight(specs, name, parameters) -> Float64

The weight the energy called `name` carries, or `0.0` if the measure has no such
energy. Lets a caller name a weight by what it *is* rather than by the config key it
happened to come from: `gamma` is the weight in front of the log spanning-forest
energy, so

```julia
gamma = energy_weight(specs, "get_log_spanning_forests", parameters)
```

reports the real thing whichever form the config was written in — and cannot disagree
with it, as reading a `[measure]` key called `gamma` could.
"""
function energy_weight(specs::AbstractVector{EnergySpec}, name::AbstractString,
                       parameters::AbstractDict)
    for spec in specs
        spec.name == name &&
            return resolve_energy_weight(spec.weight, parameters, spec.name)
    end
    return 0.0
end

"""
    energy_weight_start(specs, name, parameters) -> Float64

The weight the energy called `name` starts an annealing schedule from, or `0.0` when
it has no `weight_start` (or the measure has no such energy) — the base an annealing
run ramps up from. The counterpart of [`energy_weight`](@ref), which reports where the
same energy ends up.
"""
function energy_weight_start(specs::AbstractVector{EnergySpec}, name::AbstractString,
                             parameters::AbstractDict)
    for spec in specs
        if spec.name == name
            spec.weight_start === nothing && return 0.0
            return resolve_energy_weight(spec.weight_start, parameters, spec.name)
        end
    end
    return 0.0
end

"""
    referenced_parameters(specs) -> Vector{String}

Every named parameter the measure described by `specs` actually reads, sorted and
without repeats — the union over each spec's `weight`, `weight_start`, and
`weight_path` of [`weight_expression_parameters`](@ref) (`weight_path`'s own `t` is
excluded — it is the tempering fraction, not a run parameter, and is never part of a
run's identity). A weight written as a plain number reads nothing and contributes no
names.

This is what a run must put in its output file name: a parameter no expression names
cannot change the target, while one that is named must appear, or two runs differing
only in it would compute the same path and the second would overwrite the first. A
parameter referenced ONLY inside a `weight_path` expression still has to appear here,
or two runs differing only in it would collide.
"""
function referenced_parameters(specs::AbstractVector{EnergySpec})
    found = String[]
    for spec in specs, value in (spec.weight, spec.weight_start, spec.weight_path)
        value isa AbstractString || continue
        for name in weight_expression_parameters(value)
            name == "t" && continue
            name in found || push!(found, name)
        end
    end
    return sort!(found)
end

"""
    annealing_weights(specs, parameters) -> Dict{String, Tuple{Float64, Float64}}

The `(start, target)` weight of each spec that carries a `weight_start`, keyed by
energy name — the schedule an annealing run ramps along. A spec without a
`weight_start` does not anneal and is absent.
"""
function annealing_weights(specs::AbstractVector{EnergySpec},
                           parameters::AbstractDict)
    ramps = Dict{String, Tuple{Float64, Float64}}()
    for spec in specs
        spec.weight_start === nothing && continue
        ramps[spec.name] =
            (resolve_energy_weight(spec.weight_start, parameters, spec.name),
             resolve_energy_weight(spec.weight, parameters, spec.name))
    end
    return ramps
end
