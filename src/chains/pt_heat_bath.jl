# =============================================================================
# Parallel tempering — heat bath exchange
#
# See docs/parallel_tempering_implementation.md (task 8) and §2.4 of the design
# rationale. `HeatBath` itself is defined in parallel_tempering_types.jl; this file
# builds one from a stored Atlas and implements the exchange move.
# =============================================================================

"""
    parse_bath_measure(source_path; context=(;)) -> Measure

Rebuild the reference `Measure` an Atlas was sampled under, from its embedded
`"toml_config"` header field — the same route `examples/run_cyclewalk_extend.jl` uses
to read a config back out of a header (`energy_specs`/`measure_parameters`/
`build_measure` over the config's `[measure]` table), not the header's plain
`"energies"`/`"energy weights"` arrays, which record only human-readable labels and
cannot be resolved back to actual energy functions.

`context` supplies whatever a builder-based energy needs (e.g. `(graph=...,
num_dists=...)`); pass the SAME graph/constraints the caller's own PT run already has
loaded rather than rebuilding one from the config, since the bath must score states
on that same graph.
"""
function parse_bath_measure(source_path::String; context=(;))
    io = smartOpen(source_path, "r")
    atlas = openAtlas(io)
    close(io)
    haskey(atlas.atlasParam, "toml_config") ||
        throw(ArgumentError("$source_path has no \"toml_config\" in its header; a " *
                            "heat bath source must have been written by a config-" *
                            "driven runner that embeds one (see Writer's config_file " *
                            "argument)"))
    config = TOML.parse(String(atlas.atlasParam["toml_config"]))
    haskey(config, "measure") ||
        throw(ArgumentError("$source_path's embedded config has no [measure] table"))
    return build_measure(config["measure"]; context=context)
end

"""
    parse_bath_samples(source_path, burn_in, n, rng) -> Vector{Dict{Tuple{Vararg{String}},Int}}

Pre-draw `n` distinct map positions after `burn_in` from the Atlas at `source_path`
(uniformly, without replacement), then stream the file once more, in increasing
position order, collecting exactly those maps' districtings — every other map is
`skipMap`ped, so the file's maps are never held in memory beyond the `n` actually
drawn (the MSMS approach at `parallel_tempering.jl:170`).

Errors if `source_path` holds fewer than `burn_in + n` maps: sample pools must be
sized `>= n_rounds ÷ 2` (a bath attached to rung 1 fires on roughly every other
round). Draws are only APPROXIMATELY independent (one MCMC atlas, not fresh draws
from the reference measure) — burn `burn_in` in and keep `n` small relative to the
chain's length so the implied stride covers enough autocorrelation time (see
[`HeatBath`](@ref)'s docs for this as a modelling assumption, not a guarantee).
"""
function parse_bath_samples(source_path::String, burn_in::Int, n::Int, rng::AbstractRNG)
    burn_in >= 0 || throw(ArgumentError("burn_in must be >= 0, got $burn_in"))
    n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))

    io = smartOpen(source_path, "r")
    atlas = openAtlas(io)
    total = 0
    while !eof(atlas)
        skipMap(atlas)
        total += 1
    end
    close(io)

    available = total - burn_in
    available >= n ||
        throw(ArgumentError("$source_path has $total maps, only $available past " *
                            "burn_in=$burn_in; need >= $n samples for this heat bath " *
                            "(size the pool >= n_rounds ÷ 2)"))

    chosen = Set{Int}()
    while length(chosen) < n
        push!(chosen, rand(rng, 1:available))
    end
    wanted = Set(p + burn_in for p in chosen)

    io2 = smartOpen(source_path, "r")
    atlas2 = openAtlas(io2)
    samples = Dict{Tuple{Vararg{String}}, Int}[]
    idx = 0
    while !eof(atlas2) && length(samples) < n
        idx += 1
        if idx in wanted
            m = nextMap(atlas2)
            push!(samples, Dict{Tuple{Vararg{String}}, Int}(m.districting))
        else
            skipMap(atlas2)
        end
    end
    close(io2)
    return samples
end

"""
    HeatBath(source_path, target_scores, graph; rung=1, burn_in=0, n_samples, rng)

Build a [`HeatBath`](@ref) from a stored Atlas: reads its reference measure
([`parse_bath_measure`](@ref)) and pre-draws `n_samples` independent-ish districtings
past `burn_in` ([`parse_bath_samples`](@ref)). `graph` is the SAME `MultiLevelGraph`
the running PT ensemble uses — a sample's districting is rebuilt against it
(`MultiLevelPartition(graph, assignment)`) at exchange time.

Validates that the bath measure's energies are a SUBSET of `target_scores` (the
ensemble's `scores`, in order) — an energy the bath measure has but the target
doesn't would have no slot in `φ` and be silently dropped from the exchange (see the
module docs' §2.4); the reverse (target has an energy the bath lacks) is fine, that
energy's bath-side weight is simply 0.
"""
function HeatBath(source_path::String, target_scores::NTuple{K,Function},
                  graph::MultiLevelGraph; rung::Int=1, burn_in::Int=0, n_samples::Int,
                  rng::AbstractRNG, context=(;)) where {K}
    measure = parse_bath_measure(source_path; context=context)
    bath_only = setdiff(Set(keys(measure.weights)), Set(target_scores))
    isempty(bath_only) ||
        throw(ArgumentError("heat bath measure at $source_path scores " *
                            "$(join(string.(collect(bath_only)), ", ")), which the " *
                            "ensemble's target measure does not — it would be " *
                            "silently dropped from the exchange"))
    samples = parse_bath_samples(source_path, burn_in, n_samples, rng)
    return HeatBath(source_path, measure, samples, rung, graph)
end

"""
    try_heat_bath!(ensemble, hb, round, rng, diag)

Attempt a heat-bath exchange at `hb.rung` against the next unused independent-ish
draw from `hb.measure`, on rounds the even/odd swap pattern leaves that rung idle (so
the move costs nothing extra — design §1.9). No-op on any other round.

The exchange is a swap against a virtual rung (§2.4): identical algebra to
[`swap_logratio`](@ref), called here with the bath's own weight tuple (looked up in
`ensemble.scores` order, defaulting an energy the bath lacks to weight 0) rather than
a second path point. On accept, the rung's walker's state and `phi` are replaced,
`bath_swaps` increments, and `last_end_visited` resets to 0 — an accepted bath swap
breaks that walker's round-trip lineage, so counting through it would inflate the
diagnostic.
"""
function try_heat_bath!(ensemble::PTEnsemble{K}, hb::HeatBath, round::Int,
                        rng::AbstractRNG, diag::PTDiagnostics) where {K}
    M = nrungs(ensemble)
    hb.rung in idle_rungs(M, round) || return nothing

    isempty(hb.samples) &&
        throw(ArgumentError("heat bath at rung $(hb.rung) has no unused samples " *
                            "left (size the pool >= n_rounds ÷ 2)"))
    assignment = popfirst!(hb.samples)

    replica = ensemble.replicas[ensemble.walker_at_rung[hb.rung]]
    mlp = MultiLevelPartition(hb.graph, assignment)
    y = LinkCutPartition(mlp, rng)
    y.num_dists == replica.state.num_dists ||
        throw(ArgumentError("heat bath sample at rung $(hb.rung) has $(y.num_dists) " *
                            "districts, the running chain has $(replica.state.num_dists)"))
    phi_y = map(e -> e(y)::Float64, ensemble.scores)

    w_r = weights_at(ensemble.path, ensemble.lattice[hb.rung])
    w_b = map(e -> Float64(get(hb.measure.weights, e, 0.0)), ensemble.scores)

    logα = swap_logratio(w_r, w_b, replica.phi, phi_y)
    diag.bath_attempts += 1
    if log(rand(rng)) < logα
        replica.state = y
        replica.phi = phi_y
        replica.bath_swaps += 1
        replica.last_end_visited = 0
        diag.bath_accepts += 1
    end
    return nothing
end
