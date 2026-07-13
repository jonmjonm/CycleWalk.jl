# Compare an AIS run against a cycle-walk run for one case: are they sampling the
# same target distribution?
#
#   julia -t auto analyze.jl --case ct [--dev] [--bins 40]
#
# For every plan we read the per-district isoperimetric scores and SORT them ascending,
# turning each plan into order statistics: rank 1 = most compact district ... rank K =
# least compact. For each rank we compare the AIS marginal (importance-weighted) to the
# cycle-walk marginal (unweighted): weighted mean, weighted variance, and the total
# variation distance between binned histograms on shared bins.
#
# There is no analytic answer here, so each between-method TV is judged against a
# NOISE FLOOR: the TV between the two halves of each run on its own. If the two
# samplers agree, the AIS-vs-cyclewalk TV should sit near that finite-sample noise.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using CycleWalk
using Printf
using Statistics

const AIO = CycleWalk.AtlasIO

function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    (i == length(ARGS)) && error("missing value after $flag")
    return ARGS[i+1]
end
hasflag(flag) = flag in ARGS

case = argval("--case")
case === nothing && error("usage: julia analyze.jl --case {small,ct,grid} [--dev] [--bins N]")
dev  = hasflag("--dev")
nbins = parse(Int, argval("--bins", "40"))

const EXAMPLES = normpath(joinpath(@__DIR__, ".."))
suffix = dev ? "_dev" : ""
outdir = joinpath(EXAMPLES, "output", "validation")
# --smc compares the annealed-SMC atlas (run_asmc.jl) instead of AIS; both use the
# same LOG-importance-weight map convention, so the importance-arm reader is shared.
smc = hasflag("--smc")
imp_label = smc ? "SMC" : "AIS"
ais_path = joinpath(outdir, "$(case)_$(smc ? "smc" : "ais")$(suffix).jsonl.gz")
cw_path  = joinpath(outdir, "$(case)_cyclewalk$(suffix).jsonl.gz")

# label-invariant key for a districting: the set of districts as node-sets, so two
# plans that are the same partition under a relabeling of district numbers collapse to
# one state. This removes ONLY the arbitrary district numbering -- it does NOT fold
# together geometrically symmetric-but-distinct partitions (unlike the exactly-solvable
# suite's symmetry counts), so every concrete partition is its own state.
function canonical_state(districting)
    groups = Dict{Int, Vector{Any}}()
    for (node, d) in districting
        push!(get!(groups, d, Any[]), node)
    end
    parts = sort!([Tuple(sort!(collect(v))) for v in values(groups)])
    return Tuple(parts)
end

# ---------------------------------------------------------------------------
# read an Atlas -> (K x N sorted-score matrix, weights, per-plan canonical states)
# is_ais: map weights are LOG importance weights and must be exponentiated.
# ---------------------------------------------------------------------------
function read_case(path, is_ais)
    isfile(path) || error("missing $path -- run run_case.jl for this case/mode first")
    io = AIO.smartOpen(path, "r")
    atlas = AIO.openAtlas(io)
    maps = AIO.nextMaps(atlas)
    close(io)
    isempty(maps) && error("no maps in $path")

    K = length(maps[1].data["get_isoperimetric_scores"])
    N = length(maps)
    ranks = Matrix{Float64}(undef, K, N)   # ranks[k, i] = k-th smallest score of plan i
    states = Vector{Any}(undef, N)
    for (i, m) in enumerate(maps)
        v = sort!(Float64.(m.data["get_isoperimetric_scores"]))
        @assert length(v) == K "ragged district count in $path"
        ranks[:, i] = v
        states[i] = isempty(m.districting) ? nothing : canonical_state(m.districting)
    end

    if is_ais
        lw = Float64[m.weight for m in maps]
        w = exp.(lw .- maximum(lw))     # unnormalized importance weights
    else
        w = ones(N)
    end
    return ranks, w, states
end

wmean(x, w) = sum(w .* x) / sum(w)
function wvar(x, w)
    m = wmean(x, w)
    sum(w .* (x .- m) .^ 2) / sum(w)
end
ess(w) = sum(w)^2 / sum(w .^ 2)

# weighted, normalized histogram of x on the given edges
function whist(x, w, edges)
    nb = length(edges) - 1
    h = zeros(nb)
    for (xi, wi) in zip(x, w)
        b = searchsortedlast(edges, xi)
        b = clamp(b, 1, nb)
        h[b] += wi
    end
    s = sum(h)
    s == 0 ? h : h ./ s
end
tv(p, q) = 0.5 * sum(abs.(p .- q))

# TV between the first and second halves of one (x, w) sample -> finite-sample floor
function split_half_tv(x, w, edges)
    n = length(x)
    n < 4 && return NaN
    h = n ÷ 2
    tv(whist(x[1:h], w[1:h], edges), whist(x[h+1:end], w[h+1:end], edges))
end

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
ais_ranks, ais_w, ais_states = read_case(ais_path, true)
cw_ranks,  cw_w,  cw_states  = read_case(cw_path,  false)
K = size(ais_ranks, 1)
@assert size(cw_ranks, 1) == K "district counts differ between runs"

println("="^96)
println("case=$case  dev=$dev  bins=$nbins")
@printf("%-11s %d samples, ESS=%.1f\n", imp_label*":", size(ais_ranks, 2), ess(ais_w))
@printf("cycle walk: %d samples\n", size(cw_ranks, 2))
println("rank 1 = most compact district ... rank K = least compact  (score = perimeter^2 / area)")
println("="^96)
@printf("%-5s | %-19s | %-19s | %-8s | %-8s | %s\n",
        "rank", "mean ($imp_label / CW)", "var ($imp_label / CW)", "TV", "noise", "verdict")
println("-"^96)

function compare_ranks(ais_ranks, ais_w, cw_ranks, cw_w, K, nbins)
    pass = true
    for k in 1:K
        a = ais_ranks[k, :]; b = cw_ranks[k, :]
        lo = min(minimum(a), minimum(b)); hi = max(maximum(a), maximum(b))
        hi <= lo && (hi = lo + 1.0)
        edges = range(lo, hi; length = nbins + 1)

        ma, mb = wmean(a, ais_w), wmean(b, cw_w)
        va, vb = wvar(a, ais_w),  wvar(b, cw_w)
        tv_between = tv(whist(a, ais_w, edges), whist(b, cw_w, edges))
        noise = max(split_half_tv(a, ais_w, edges), split_half_tv(b, cw_w, edges))

        ok = isnan(noise) ? true : tv_between <= 2 * noise + 0.02
        pass &= ok
        @printf("%-5d | %8.3f / %8.3f | %8.3f / %8.3f | %-8.4f | %-8.4f | %s\n",
                k, ma, mb, va, vb, tv_between, noise, ok ? "ok" : "**DIFFERS**")
    end
    return pass
end
pass = compare_ranks(ais_ranks, ais_w, cw_ranks, cw_w, K, nbins)
println("-"^96)
println(pass ? "PASS: every rank's AIS-vs-cyclewalk TV is within the finite-sample noise floor" :
               "REVIEW: one or more ranks exceed the noise floor -- inspect above")

# ---------------------------------------------------------------------------
# small case only: direct check over concrete partitions (mixing + agreement).
# Small enough to enumerate the states actually visited, so instead of the binned
# score marginals we compare the full weighted distribution over distinct partitions
# and see whether each chain even mixes across more than one state.
# ---------------------------------------------------------------------------
function state_distribution(states, w)
    d = Dict{Any, Float64}()
    for (s, wi) in zip(states, w)
        s === nothing && continue
        d[s] = get(d, s, 0.0) + wi
    end
    tot = sum(values(d))
    tot > 0 && (for k in keys(d); d[k] /= tot; end)
    return d
end

# TV between two normalized state distributions (dicts state -> prob)
function state_tv(ad, cd)
    ks = union(keys(ad), keys(cd))
    0.5 * sum(abs(get(ad, k, 0.0) - get(cd, k, 0.0)) for k in ks)
end

# finite-sample floor: TV between the state distributions of the two halves of one run
function split_half_state_tv(states, w)
    n = length(states)
    n < 4 && return NaN
    h = n ÷ 2
    state_tv(state_distribution(states[1:h], w[1:h]),
             state_distribution(states[h+1:end], w[h+1:end]))
end

function small_state_check(ais_states, ais_w, cw_states, cw_w)
    if any(s -> s === nothing, ais_states) || any(s -> s === nothing, cw_states)
        println("\n(skipping direct state check: run_case.jl was run with output_districting=false)")
        return
    end
    ad = state_distribution(ais_states, ais_w)
    cd = state_distribution(cw_states, cw_w)
    keys_all = union(keys(ad), keys(cd))
    tv_states = state_tv(ad, cd)
    # split-half noise floor: how much two halves of one run disagree by chance, given
    # this many samples over this many states. The AIS-vs-CW TV is only meaningful
    # relative to this (with ~100 states and few hundred samples the floor is large).
    noise = max(split_half_state_tv(ais_states, ais_w),
                split_half_state_tv(cw_states, cw_w))
    state_ok = isnan(noise) ? true : tv_states <= 2 * noise + 0.02

    println("\n" * "="^96)
    println("small case: direct distribution over distinct partitions (district labels canonicalized;")
    println("no graph-symmetry folding, so each concrete partition is its own state)")
    println("="^96)
    @printf("distinct states visited:  AIS=%d   cycle walk=%d   (union=%d)\n",
            length(ad), length(cd), length(keys_all))
    @printf("max single-state mass:    AIS=%.3f   cycle walk=%.3f   (1.0 => frozen / not mixing)\n",
            isempty(ad) ? NaN : maximum(values(ad)),
            isempty(cd) ? NaN : maximum(values(cd)))
    @printf("state TV (AIS vs cycle walk): %.4f   split-half noise floor: %.4f   => %s\n",
            tv_states, noise, state_ok ? "ok" : "**DIFFERS**")
    println("-"^96)
    @printf("%-10s | %-10s | %s\n", "AIS prob", "CW prob", "state (district node-sets)")
    println("-"^96)
    for k in sort(collect(keys_all); by = k -> -max(get(ad, k, 0.0), get(cd, k, 0.0)))[1:min(end, 12)]
        pa = get(ad, k, 0.0); pc = get(cd, k, 0.0)
        label = join(["{" * join(first(part, 3), ",") * (length(part) > 3 ? ",…" : "") * "}"
                      for part in k], " ")
        @printf("%-10.4f | %-10.4f | %s\n", pa, pc, label)
    end
    println("-"^96)
    if length(cd) == 1
        println("NOTE: cycle walk sat in a single partition -- it did not mix at this size/schedule")
    elseif length(ad) == 1
        println("NOTE: AIS mass collapsed onto a single partition (ESS/annealing issue)")
    elseif state_ok
        println("PASS: both chains mixed and their state distributions agree within the noise floor")
    else
        println("REVIEW: state TV exceeds the noise floor -- the chains may differ (or need more samples)")
    end
end

if case == "small"
    small_state_check(ais_states, ais_w, cw_states, cw_w)
end
