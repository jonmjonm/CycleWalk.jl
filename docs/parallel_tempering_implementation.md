# Parallel tempering in CycleWalk.jl — implementation roadmap

**Audience: an implementing agent or developer with no prior context on this feature.**

This is the *build* document. Its companion
[`plan_parallel_tempering.md`](plan_parallel_tempering.md) is the *design* document —
it records which options were considered and why they were rejected. Read this file to
build; read that one when you want to know why something is the way it is. Section
references like "(design §1.5)" point there.

Nothing in this feature is implemented yet. Everything below is to be written.

---

## 0. How to use this document

Work through §5 **in order**. Each task states its goal, the files it touches, a code
sketch, and an acceptance check you must run before moving on. Do not skip ahead — task
4 (the serial driver) is what establishes correctness, and the parallel backends in
tasks 7–9 are a performance layer on top of an already-correct sampler.

Three rules that override any instinct to the contrary:

1. **Never write a test that restates the implementation.** This codebase already
   shipped an inverted-sign bug for exactly that reason: a test asserted
   `incremental_logweight!` equalled `energy_at(t) - energy_at(t_prev)`, which is what
   the function computes, so no sign error could fail it. Every claim of correctness
   must be anchored to a number derived *outside* the code — §7 gives you those numbers.
2. **Stateful objects must be resettable.** `FixedSchedule` was silently single-use for
   months because it carried a cursor nobody reset; a second run returned `logZ = 0.0`
   with no error. `BetaLattice` and `PTDiagnostics` carry state. Give them
   `reset_pt_diagnostics!` and call it at driver entry, and test reuse.
3. **Get the sign right and pin it.** See §2. The swap acceptance is the one place a
   sign error would produce a plausible-looking but wrong sampler.

---

## 1. Orientation — read these first

| File | Why it matters |
|---|---|
| `src/chains/annealed_smc.jl` | The model to follow. Defines `AnnealPath`, `LinearPath`, `weights_at`, `energy_at`, `configure_measure!`, `Particle{K}`, and the BLAS-pinning pattern. PT reuses all of it. |
| `src/chains/mcmc.jl` | `run_metropolis_hastings!` — the inner loop every replica runs. Note it takes `steps` as either `Int` or `(initial, final)`. |
| `src/measure/measure.jl` | `Measure`, `get_log_energy`, `get_delta_energy`. Note `scores::Set{Function}` — untyped, which is why ASMC freezes it to an `NTuple` (`annealed_smc_scores_and_targets`). |
| `src/partition/link_cut_partition.jl` | `clone_for_annealing` — the cheap per-replica copy. **Shares** the graph; copies everything sampling touches. |
| `src/io/writer.jl` | `Writer`, `build_output_map` (documented thread-safe), `addMap` (must be serialized), `push_writer!`, `stamp_run_metadata!`. |
| `src/io/run_metadata.jl` | The three existing metadata builders. Yours must match their shape exactly. |
| `test/test_annealed_smc.jl` | The testing style to copy: pure-helper unit tests, then driver invariants, then ground truth. |
| `test/test_cases/small_square_p88_unweighted.jl` | **The ground truth.** Enumerated distributions for the 4×4 graph. §7 depends on it. |

### Invariants you must not break

* **A replica's `state`, `rng`, and `diagnostics` are never shared.** `run_metropolis_hastings!`
  mutates its `RunDiagnostics`, so a shared object races. This is why
  `rejuvenate!` in `annealed_smc.jl` builds one `RunDiagnostics` per particle.
* **Never index anything by `Threads.threadid()`.** Under `@spawn` (and `@threads
  :dynamic`, the default since 1.8) a task may migrate between threads mid-execution.
  Index by replica id. (design §1.5)
* **Pin BLAS to 1 thread** for the duration of a multi-worker run and restore it in a
  `finally`. The spanning-forest energy takes a log-det per district via LAPACK; several
  Julia threads each spawning a BLAS pool oversubscribes badly. Copy the pattern from
  `annealed_smc.jl` verbatim.
* **Seed every replica from a draw made *sequentially* on the driver's `rng`**, as
  `run_annealed_importance_sampling!` does. This is what makes results independent of
  worker count and lets §7's cross-backend test exist.

---

## 2. The math, stated once

### 2.1 Sign convention

`get_log_energy` returns an **energy**, not a log-density:

```
ν(τ) ∝ exp( −get_log_energy(τ, measure) )
```

This is settled, not a matter of taste. §6 of the Cycle Walk paper
(`~/Git/Greg/CycleWalk/Accepted/cycleWalk-siam.tex`) defines the measure on spanning
forests as `ν_γ(τ) ∝ π(ξ_τ)/Tree(ξ_τ)^γ ∝ exp(−γ·J_Tree − J)`, and that exponent is
exactly what `get_log_energy` sums, term for term. `run_metropolis_hastings!` already
assumes it via `get_delta_energy`'s `exp(E_cur − E_prop)`.

*(Historical note so you don't "fix" it back: AIS and ASMC had this inverted and were
corrected in commit `02c304c`. If you see `energy_at(t_prev) − energy_at(t)` and think
the operands look backwards, they are not.)*

### 2.2 The rung measure

Rung `k` sits at path point `β_k` and targets

```
ν_k(x) ∝ exp( −E_k(x) ),      E_k(x) = ⟨ weights_at(path, β_k), φ(x) ⟩
```

where `φ(x) = (e₁(x), …, e_K(x))` are the cached raw per-term energies (`Particle.phi`
in ASMC). `E_k` is arithmetic on cached values — **no energy evaluation**.

### 2.3 Swap acceptance — derive it, don't copy it

Swapping the states held at rungs `i` and `j`:

```
α  = [ ν_i(x_j) ν_j(x_i) ] / [ ν_i(x_i) ν_j(x_j) ]

log α = −E_i(x_j) − E_j(x_i) + E_i(x_i) + E_j(x_j)
      = −⟨w_i, φ_j⟩ − ⟨w_j, φ_i⟩ + ⟨w_i, φ_i⟩ + ⟨w_j, φ_j⟩
      = ⟨ w_i − w_j , φ_i − φ_j ⟩
```

with `w_i = weights_at(path, β_i)`, `φ_i = φ(x_i)`. Accept with probability
`min(1, exp(log α))`.

**Two independent cross-checks that this sign is right** — reproduce both as tests:

1. For a pure β-ladder (`w_k = β_k · w`) it collapses to
   `log α = (β_i − β_j)(E_i − E_j)` with `E = ⟨w, φ⟩`. That is exactly
   `ParallelTempering.java:210` in `~/Git/Greg/NC_StateLeg/src/`, an independent
   implementation written for `π ∝ exp(−βE)`.
2. **Direction check.** Let rung `i` be *hotter* (`β_i < β_j`) and let it hold the
   higher-energy state (`E_i > E_j`). Then `(β_i − β_j) < 0` and `(E_i − E_j) > 0`, so
   `log α < 0` — the swap is discouraged. Correct: hot-chain-holds-high-energy is
   already the natural arrangement. If your implementation *encourages* that swap, the
   sign is flipped.

### 2.4 Heat bath is the same formula

A heat-bath exchange at rung `r` against an independent draw `y ~ ν_bath` (weights
`w_b`) is a swap with a virtual rung:

```
log α = ⟨ w_r − w_b , φ(x) − φ(y) ⟩
```

Identical algebra, so **implement it by calling the same `swap_logratio` function**
with the bath as the partner.

**Requirement that is easy to miss:** `φ` is a tuple ordered by the ensemble's `scores`,
so `w_b` must be expressed in *that same order*. The bath measure must therefore be over
the same score set as the target — an energy the target lacks has no slot in `φ`, and an
energy the bath lacks is simply weight 0. Validate this when the `HeatBath` is
constructed (compare `Set(keys(bath.measure.weights))` against the ensemble's scores) and
throw a message naming the offending energy. Silently misaligning the two tuples
produces a sampler that runs and is wrong.

Validity also requires `y` to be an *independent* draw from `ν_bath`; drawn from a stored
MCMC atlas it is only approximately so — document that, and never reuse a stored sample.

### 2.5 β and the lattice

`β = 0` gives all-zero weights, i.e. `ν_0` — uniform on balanced spanning forests, the
proposal's own measure, which the 1-Tree walk samples without Metropolization (paper
§8). `β = 1` gives the configured target. Note `β = 1` does **not** mean `γ = 1`; it
means "whatever `[measure]` configures".

A scalar β is the *diagonal* path and tempers every term together. The per-term
`LinearPath` also lets you ramp `γ` alone with the compactness weight held fixed, which
is usually the ladder you want, because paper §7 identifies γ as the hard direction for
convergence. (design §1.1.1)

---

## 3. Architecture in one picture

```
run_parallel_tempering!(partition, proposal, measure, lattice, ...)
  │
  ├── build PTEnsemble: M replicas, each a clone_for_annealing(partition),
  │     own rng (seeded serially), own RunDiagnostics, own phi
  │
  ├── init_steps: advance every replica at its own β (decorrelate)
  │
  └── for round in 1:n_rounds
        ├── advance!(backend, ensemble, swap_interval)   ← PARALLEL, barrier at end
        ├── refresh_potentials!(ensemble)                 ← phi after the moves
        ├── pairs = even_odd_pairs(M, round)              ← SERIAL from here
        ├── for (i,j) in pairs: maybe swap rungs i,j
        ├── heat_bath !== nothing && try_heat_bath!(...)
        ├── record diagnostics (swap matrix, occupancy, round trips)
        └── emit maps per write_rungs
```

### The central data-structure decision

Replicas are indexed by **walker id and never permuted**; each carries the rung it
currently occupies. A swap exchanges *rung labels*, not array slots.

```julia
replicas :: Vector{Replica}   # indexed by walker id — NEVER reordered
walker_at_rung :: Vector{Int} # rung k -> walker id currently at rung k
```

with the invariant

```
replicas[walker_at_rung[k]].beta_index == k     for every k
```

Why this and not "swap the array entries" (which is what the MSMS reference does): under
`Distributed` a worker owns one walker for the life of the run, so the walker→process
binding must be fixed and only the label can move (design §1.6). Using the same
representation for both backends means **one swap implementation**, and it makes
round-trip diagnostics natural because a walker keeps its identity.

---

## 4. File manifest

| File | Status | Contents |
|---|---|---|
| `src/chains/anneal_path.jl` | **new** (task 1) | `AnnealPath`, `LinearPath`, `linear_path`, `weights_at`, `energy_at`, `configure_measure!`, `set_measure_weights!`, `measure_scores_and_targets`, `_dot`, `_sub`, `ess_from_logw` — moved out of `annealed_smc.jl` |
| `src/chains/annealed_smc.jl` | edit (task 1) | delete the moved block; no behaviour change |
| `src/chains/parallel_tempering_types.jl` | **new** (task 2) | `BetaLattice`, `Replica`, `PTEnsemble`, `HeatBath`, `PTDiagnostics` |
| `src/chains/parallel_tempering.jl` | **new** (tasks 3–6) | `even_odd_pairs`, `swap_logratio`, `try_swap!`, `swap_round!`, `try_heat_bath!`, `run_parallel_tempering!` |
| `src/chains/pt_backends.jl` | **new** (tasks 7, 9) | `PTBackend`, `SerialBackend`, `ThreadedBackend`, `DistributedBackend` |
| `src/chains/pt_heat_bath.jl` | **new** (task 8) | `parse_bath_measure`, `parse_bath_samples`, bath state reconstruction |
| `src/io/run_metadata.jl` | edit (task 6) | `parallel_tempering_run_metadata` |
| `src/CycleWalk.jl` | edit | `include` the new files; fill the `# parallel tempering` export block (it is already stubbed at the bottom of the export list) |
| `test/test_parallel_tempering.jl` | **new** (tasks 3–9) | everything in §7 |
| `test/runtests.jl` | edit | `include` the new test file |
| `examples/run_pt_toml.jl` | **new** (task 10) | model on `examples/run_asmc_toml.jl` |
| `examples/toml/param_pt_grid.toml` | **new** (task 10) | model on `param_annealed_smc_grid.toml` |
| `examples/validation/smoke_pt.jl` | **new** (task 10) | model on `smoke_annealed_smc.jl` |
| `docs/run_pt_toml.md` | **new** (task 10) | model on `docs/run_cyclewalk_toml.md` |

`Project.toml`: add `Distributed` and `Serialization` (both stdlibs, no compat churn) in
task 9 only.

---

## 5. Tasks

### Task 1 — Extract the path/seam code

**Goal.** PT must not `include` the SMC file to get `AnnealPath`.

Move from `annealed_smc.jl` to a new `src/chains/anneal_path.jl`, unchanged:
`_dot`, `_sub`, `ess_from_logw`, `AnnealPath`, `LinearPath`, `linear_path`,
`weights_at`, `energy_at`, `configure_measure!`, `set_measure_weights!`,
`annealed_smc_scores_and_targets`.

Rename `annealed_smc_scores_and_targets` → `measure_scores_and_targets` and keep the old
name as an exported alias:

```julia
const annealed_smc_scores_and_targets = measure_scores_and_targets
```

It is exported and used in `test/test_run_metadata.jl`, `examples/run_asmc_toml.jl`, and
`examples/validation/*.jl`; removing it breaks all of them.

`energy_at(path, p, t)` is currently typed on `::Particle`. Loosen it to accept anything
with a `phi` field, or add a method for `Replica`:

```julia
@inline energy_at(path::LinearPath, phi::NTuple{K,Float64}, t::Float64) where {K} =
    _dot(weights_at(path, t), phi)
```

and have the `Particle`/`Replica` methods forward to it. That keeps one implementation.

In `src/CycleWalk.jl`, `include("./chains/anneal_path.jl")` **before** both
`annealed_smc.jl` and `parallel_tempering.jl`.

> **Acceptance.** `julia --project=. test/runtests.jl` exits 0 with no test changes.
> This task must be behaviour-neutral.

---

### Task 2 — Types

Create `src/chains/parallel_tempering_types.jl`.

```julia
"""
    BetaLattice(betas)

The inverse-temperature lattice: rung `k` sits at path point `betas[k]`. Strictly
increasing, `betas[1] >= 0`, `betas[end] == 1.0` (rung `M` is the target — note that is
"the configured target", not γ = 1; see the module docs).

Rung 1 is the HOTTEST (most tempered, easiest to sample); rung `M` is the COLDEST
(the target). Samples for downstream analysis come from rung `M`.
"""
struct BetaLattice
    betas::Vector{Float64}
    function BetaLattice(betas::AbstractVector{<:Real})
        b = collect(Float64, betas)
        length(b) >= 1 || throw(ArgumentError("a lattice needs at least one rung"))
        issorted(b) && allunique(b) ||
            throw(ArgumentError("betas must be strictly increasing, got $b"))
        b[1] >= 0 || throw(ArgumentError("betas must be >= 0, got $(b[1])"))
        b[end] == 1.0 ||
            throw(ArgumentError("the last rung must be the target (beta = 1.0), got $(b[end])"))
        return new(b)
    end
end

Base.length(l::BetaLattice) = length(l.betas)
Base.getindex(l::BetaLattice, k::Int) = l.betas[k]

"""
    linear_betas(M) -> BetaLattice

`M` rungs evenly spaced on [0, 1].
"""
linear_betas(M::Int) = BetaLattice(range(0, 1; length=M))

"""
    geometric_betas(M; beta_min=0.05) -> BetaLattice

`M` rungs geometrically spaced in TEMPERATURE (= 1/β) between `beta_min` and 1 — the
ladder shape of `ParallelTempering.java:37`, which builds the same set of values in the
opposite order (it runs cold-to-hot; we run hot-to-cold, so rung 1 is always the most
tempered). A test comparing the two must `sort` or `reverse` one of them.

Cannot include β = 0 (infinite temperature); use `linear_betas`, or prepend 0
explicitly, if you want the exact ν_0 end.
"""
function geometric_betas(M::Int; beta_min::Float64=0.05)
    0 < beta_min < 1 || throw(ArgumentError("beta_min must be in (0,1), got $beta_min"))
    M >= 2 || return BetaLattice([1.0])
    logT = range(log(1/beta_min), log(1.0); length=M)   # temperature high -> low
    return BetaLattice([1/exp(x) for x in logT])
end
```

```julia
"""
    Replica{K}

One PT walker. `state`/`rng`/`diagnostics` are private to this replica and must never be
shared (see the module invariants). `beta_index` is the rung it currently occupies;
`phi` is the cached raw per-term energies of `state`, refreshed once per block.

`replica_id` is fixed for the life of the run and follows the walker up and down the
ladder — it is what makes round-trip diagnostics possible. `bath_swaps` counts heat-bath
replacements, which break the walker's lineage.
"""
mutable struct Replica{K}
    state::LinkCutPartition
    beta_index::Int
    phi::NTuple{K,Float64}
    rng::PCG.PCGStateOneseq
    diagnostics::RunDiagnostics
    replica_id::Int
    bath_swaps::Int
    last_end_visited::Int      # 0 = neither end yet; else 1 or M — for round trips
end
```

```julia
"""
    PTEnsemble{K}

`replicas` is indexed by WALKER id and is never permuted; `walker_at_rung[k]` is the
walker currently at rung `k`. The invariant

    replicas[walker_at_rung[k]].beta_index == k    for every k

is checked by [`check_ensemble`](@ref) and must hold on entry to and exit from every
swap round. A swap exchanges rung labels, never array slots — see the module docs for
why (it is what lets one swap implementation serve both the threaded and distributed
backends).
"""
mutable struct PTEnsemble{K}
    replicas::Vector{Replica{K}}
    walker_at_rung::Vector{Int}
    lattice::BetaLattice
    path::AnnealPath
    scores::NTuple{K,Function}
end

nrungs(e::PTEnsemble) = length(e.lattice)

"""
    check_ensemble(e) -> Bool

Assert the rung/walker bijection. Call it in tests and under `@debug`; it is O(M).
"""
function check_ensemble(e::PTEnsemble)
    M = nrungs(e)
    length(e.walker_at_rung) == M || return false
    sort(e.walker_at_rung) == collect(1:M) || return false
    return all(k -> e.replicas[e.walker_at_rung[k]].beta_index == k, 1:M)
end
```

```julia
"""
    PTDiagnostics(M)

Per-run PT diagnostics. STATEFUL — `run_parallel_tempering!` resets it on entry (see
[`reset_pt_diagnostics!`](@ref)), so one object may drive several runs. This mirrors the
`reset_schedule!` fix in `annealed_smc.jl`, which exists because a stateful schedule
silently made a second run a no-op.

- `attempts` / `accepts`: per adjacent pair `(k, k+1)`, length `M-1`
- `accept_prob_sum`: running sum of min(1, α) per pair, so the mean can be reported
- `occupancy[w, k]`: blocks walker `w` spent at rung `k`. Flat rows mean a well-placed
  lattice; a walker stuck in one band means the lattice is too coarse there.
- `round_trips[w]`: completed rung-M -> rung-1 -> rung-M journeys. **The single most
  informative PT diagnostic** and the one neither reference implementation computes.
- `bath_attempts` / `bath_accepts`
- `straggler_gap`: per block, `max - mean` of the per-replica advance time. Use it to
  size `swap_interval`.
"""
mutable struct PTDiagnostics
    attempts::Vector{Int}
    accepts::Vector{Int}
    accept_prob_sum::Vector{Float64}
    occupancy::Matrix{Int}
    round_trips::Vector{Int}
    bath_attempts::Int
    bath_accepts::Int
    straggler_gap::Vector{Float64}
end

PTDiagnostics(M::Int) = PTDiagnostics(zeros(Int, max(M-1,0)), zeros(Int, max(M-1,0)),
                                      zeros(Float64, max(M-1,0)), zeros(Int, M, M),
                                      zeros(Int, M), 0, 0, Float64[])

function reset_pt_diagnostics!(d::PTDiagnostics)
    fill!(d.attempts, 0); fill!(d.accepts, 0); fill!(d.accept_prob_sum, 0.0)
    fill!(d.occupancy, 0); fill!(d.round_trips, 0)
    d.bath_attempts = 0; d.bath_accepts = 0
    empty!(d.straggler_gap)
    return d
end

"""
    swap_rate(d) -> Vector{Float64}

Empirical acceptance rate per adjacent pair. A well-placed lattice has these roughly
EQUAL across pairs — that is the tuning target (design §1.10), not "as high as possible".
"""
swap_rate(d::PTDiagnostics) =
    [d.attempts[k] == 0 ? NaN : d.accepts[k]/d.attempts[k] for k in eachindex(d.attempts)]
```

> **Acceptance.** Unit-test `BetaLattice` validation (rejects unsorted, duplicated,
> negative, and `betas[end] != 1`), `linear_betas(5).betas == [0,0.25,0.5,0.75,1]`, and
> that `geometric_betas` reproduces the Java ladder for matching endpoints.

---

### Task 3 — Swap mechanics (pure functions, no driver)

Create `src/chains/parallel_tempering.jl`.

```julia
"""
    even_odd_pairs(M, round) -> Vector{Tuple{Int,Int}}

Adjacent rung pairs to attempt this round, alternating between the even and odd
pairings:

    round odd  ->  (1,2) (3,4) (5,6) ...
    round even ->  (2,3) (4,5) (6,7) ...

Pairs within a round are disjoint, so the whole round is one serial pass with no
conflicts. This is the deterministic even-odd (DEO) / non-reversible scheme of Syed,
Bouchard-Cote, Deligiannidis & Doucet (2022), whose round-trip rate is O(1) in the
number of rungs rather than the diffusive O(1/M^2) of random-pair schemes. Do NOT
replace it with "pick one random adjacent pair", which is what
`ParallelTempering.java:184` does and which performs M times less exchange per round.
"""
function even_odd_pairs(M::Int, round::Int)
    start = isodd(round) ? 1 : 2
    return [(k, k+1) for k in start:2:(M-1)]
end

"""
    idle_rungs(M, round) -> Vector{Int}

Rungs that appear in no pair this round. The count depends on both parities and is 0, 1,
or 2 — do NOT hard-code a case analysis, derive it from [`even_odd_pairs`](@ref):

    M=4, odd round : (1,2)(3,4)      -> idle = []
    M=4, even round: (2,3)           -> idle = [1, 4]
    M=5, odd round : (1,2)(3,4)      -> idle = [5]
    M=5, even round: (2,3)(4,5)      -> idle = [1]

The heat bath fires only when its rung is idle, so it costs nothing (design §1.9).
"""
idle_rungs(M::Int, round::Int) =
    setdiff(1:M, Iterators.flatten(even_odd_pairs(M, round)))
```

Note this means a bath attached to rung 1 fires on roughly every *other* round, so size
the sample pool at `>= n_rounds ÷ 2`. **Test `idle_rungs` against `even_odd_pairs`
directly** rather than against a hand-written table — assert no returned rung appears in
any pair of that round, and that pairs ∪ idle covers `1:M`.

```julia
"""
    swap_logratio(path, beta_i, beta_j, phi_i, phi_j) -> Float64

Log acceptance ratio for exchanging the states whose cached potentials are `phi_i`,
`phi_j` between path points `beta_i`, `beta_j`:

    log α = ⟨ w(beta_i) − w(beta_j) , phi_i − phi_j ⟩

Arithmetic only — no energy evaluation. Derived in the module docs; note the operand
order, which follows from ν ∝ exp(−energy) (NOT exp(+energy)). For a pure β-ladder this
equals `(beta_i − beta_j) * (E_i − E_j)`, which is the independent cross-check.
"""
@inline function swap_logratio(path::AnnealPath, beta_i::Float64, beta_j::Float64,
                               phi_i::NTuple{K,Float64}, phi_j::NTuple{K,Float64}) where {K}
    return _dot(_sub(weights_at(path, beta_i), weights_at(path, beta_j)),
                _sub(phi_i, phi_j))
end
```

```julia
"""
    swap_round!(ensemble, round, rng, diagnostics) -> Int

Attempt every pair from [`even_odd_pairs`](@ref) and return the number accepted. Runs
SERIALLY on the calling thread: once `phi` is cached each swap is a handful of flops, so
spawning a task per pair (as the MSMS reference does) costs orders of magnitude more
than the work (design §1.5).

A swap exchanges the two rungs' `beta_index` and the `walker_at_rung` entries. States
never move.
"""
function swap_round!(ensemble::PTEnsemble{K}, round::Int, rng::AbstractRNG,
                     diag::PTDiagnostics) where {K}
    M = nrungs(ensemble)
    accepted = 0
    for (i, j) in even_odd_pairs(M, round)
        wi, wj = ensemble.walker_at_rung[i], ensemble.walker_at_rung[j]
        logα = swap_logratio(ensemble.path, ensemble.lattice[i], ensemble.lattice[j],
                             ensemble.replicas[wi].phi, ensemble.replicas[wj].phi)
        diag.attempts[i] += 1
        diag.accept_prob_sum[i] += min(1.0, exp(logα))
        if log(rand(rng)) < logα
            ensemble.walker_at_rung[i], ensemble.walker_at_rung[j] = wj, wi
            ensemble.replicas[wi].beta_index = j
            ensemble.replicas[wj].beta_index = i
            diag.accepts[i] += 1
            accepted += 1
        end
    end
    return accepted
end
```

Use `log(rand(rng)) < logα` rather than `rand(rng) < exp(logα)` — `exp` overflows for
large positive `logα`, and the log form is exact where it matters.

```julia
"""
    record_round!(ensemble, diagnostics)

Update occupancy and round-trip counts after a swap round. A round trip is a walker
travelling rung M -> rung 1 -> rung M; `last_end_visited` tracks which end it saw last,
and reaching rung M having last seen rung 1 completes one.
"""
function record_round!(ensemble::PTEnsemble, diag::PTDiagnostics)
    M = nrungs(ensemble)
    for r in ensemble.replicas
        diag.occupancy[r.replica_id, r.beta_index] += 1
        if r.beta_index == 1
            r.last_end_visited = 1
        elseif r.beta_index == M
            r.last_end_visited == 1 && (diag.round_trips[r.replica_id] += 1)
            r.last_end_visited = M
        end
    end
end
```

> **Acceptance (all cheap unit tests, no sampling):**
> - `even_odd_pairs`: pairs disjoint; alternates; over two consecutive rounds every
>   adjacent pair `(k,k+1)` appears exactly once; `idle_rungs` returns exactly the unpaired rungs, and pairs ∪ idle covers 1:M.
> - `swap_logratio`: for `w = t·w_target`, equals `(β_i−β_j)*(E_i−E_j)` computed by
>   hand. Include the **direction check** from §2.3.
> - `swap_round!`: with all `β` equal, `logα == 0` for every pair, so acceptance is 1
>   and `check_ensemble` still holds. With `φ_i == φ_j`, likewise.
> - `record_round!`: drive a walker M→1→M by hand and assert exactly one round trip.

---

### Task 4 — Serial driver

Still in `parallel_tempering.jl`. **Write this before any parallelism.** Tests in §7.1
and §7.2 must pass against the serial driver; the backends of tasks 7 and 9 are then
required to reproduce it bit for bit.

```julia
function run_parallel_tempering!(
    partition::LinkCutPartition,
    proposal::Union{Function,Vector{Tuple{T,Function}}},
    measure::Measure,
    lattice::BetaLattice,
    swap_interval::Int,
    n_rounds::Int,
    rng::AbstractRNG;
    path::Union{AnnealPath,Function,Nothing}=nothing,
    backend::PTBackend=SerialBackend(),
    heat_bath::Union{HeatBath,Nothing}=nothing,
    init_steps::Int=0,
    write_rungs::Symbol=:target,          # :target | :all | :none
    output_every::Int=1,                  # emit every N swap rounds
    writers::Union{Vector{Writer},Writer,Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    diagnostics::Union{PTDiagnostics,Nothing}=nothing,
    seed=nothing,
) where {T<:Real}
```

Body, in order:

1. **Validate.** `swap_interval >= 1`, `n_rounds >= 1`, `write_rungs in (:target,:all,:none)`.
   Follow the `init_steps` precedent from `run_annealed_smc!`: if `init_steps <= 0` and
   `length(lattice) > 1`, **warn** (not throw — unlike adaptive SMC it still works, the
   ladder just starts fully correlated).
2. **Reset** `diagnostics` (`reset_pt_diagnostics!`) — see rule 2 in §0.
3. **Stamp metadata** on the writer(s) before any map is written (task 6).
4. **BLAS pin**, with `try`/`finally` restore. Copy from `annealed_smc.jl`.
5. **Build the ensemble.** For `w in 1:M`: `clone_for_annealing(partition)`, a
   `PCG.PCGStateOneseq(UInt64, rand(rng, UInt64))` **drawn sequentially**,
   `deepcopy(run_diagnostics)`, `phi = map(e -> e(st)::Float64, scores)`,
   `beta_index = w`, `replica_id = w`, `last_end_visited = 0`.
   `walker_at_rung = collect(1:M)`.
6. **`init_steps`**: `advance!(backend, ensemble, init_steps)` then
   `refresh_potentials!(ensemble)`.
7. **Main loop** `for round in 1:n_rounds`:
   ```
   t0 = time()
   advance!(backend, ensemble, swap_interval)
   push!(diag.straggler_gap, backend_straggler_gap(backend))
   refresh_potentials!(ensemble)
   swap_round!(ensemble, round, rng, diag)
   heat_bath !== nothing && try_heat_bath!(ensemble, heat_bath, round, rng, diag)
   record_round!(ensemble, diag)
   round % output_every == 0 && emit_maps!(writers, ensemble, write_rungs, round, ...)
   ```
8. **Return** `(ensemble, diagnostics)`.

`advance!` must run each replica at **its own** β:

```julia
function advance_replica!(replica, ensemble, proposal, work_measure, steps)
    configure_measure!(ensemble.path, work_measure, ensemble.scores,
                       ensemble.lattice[replica.beta_index])
    run_metropolis_hastings!(replica.state, proposal, work_measure, steps, replica.rng;
                             output_initial=false, run_diagnostics=replica.diagnostics)
end
```

**Each worker needs its own `work_measure`** (`deepcopy(measure)` per replica, built
once at ensemble construction and stored on the `Replica`, or one per backend worker
slot). `configure_measure!` mutates it, so a shared one races. This is the PT analogue
of the per-particle `RunDiagnostics` rule — put the `work_measure` on the `Replica`.

> **Acceptance.** §7.1 tests 1–4 pass.

---

### Task 5 — Output

Rung-indexed and walker-indexed views are both needed and neither substitutes for the
other: **rung `M` is the sample stream; walker identity is the mixing diagnostic.**

* `write_rungs = :target` → one `Writer`, receiving rung `M`'s state each output round.
* `write_rungs = :all` → `M` writers, one per rung, file suffix `_beta<k>`. Per-rung
  files keep the write path lock-free and match what the distributed backend wants.
* `write_rungs = :none` → no output (for tests and tuning runs).

Every emitted map must carry, in its `MapParam`:

| key | value |
|---|---|
| `pt/replica_id` | which walker |
| `pt/beta_index` | which rung |
| `pt/beta` | `lattice[beta_index]` |
| `pt/bath_swaps` | that walker's bath-replacement count |
| `pt/round` | swap round index |

Use `build_output_map(writer, state, name, weight, diagnostics; extra_data=...)` — it is
documented safe to call concurrently; only `addMap` must be serialized. Name maps
`"round<r>_beta<k>"`.

Weights are 1 (PT is an MCMC sampler, not importance sampling), so build writers with
the default `weight_type=Int64` — unlike AIS/ASMC, which need `Float64`.

---

### Task 6 — Run metadata

Add to `src/io/run_metadata.jl`, matching the existing builders' shape exactly:

```julia
Dict{String,Any}(
  "chain.run"        => "parallel tempering CycleWalk",
  "chain.parameters" => Dict{String,Any}(
     "function"          => "run_parallel_tempering!",
     "betas"             => copy(lattice.betas),
     "n_rungs"           => length(lattice),
     "swap_interval"     => swap_interval,
     "n_rounds"          => n_rounds,
     "total_steps"       => swap_interval * n_rounds,
     "swap_scheme"       => "deterministic even/odd (non-reversible, DEO)",
     "backend"           => backend_name(backend),   # "serial"|"threaded"|"distributed"
     "workers"           => backend_workers(backend),
     "init_steps"        => init_steps,
     "write_rungs"       => String(write_rungs),
     "output_every"      => output_every,
     "heat_bath"         => heat_bath === nothing ? nothing :
                            Dict("source"=>hb.source_path, "rung"=>hb.rung,
                                 "n_samples"=>length(hb.samples),
                                 "weights"=>_weights_dict(hb.measure)),
     "threads"           => Threads.nthreads(),
     "weights.hot"       => _label_weights(scores, weights_at(path, lattice[1]),   measure),
     "weights.target"    => _label_weights(scores, weights_at(path, lattice[end]), measure),
     "proposal"          => describe_proposal(proposal),
  ))
```

`_label_weights`, `_weights_dict`, `_maybe_seed!`, `describe_proposal` already exist in
that file. Record `seed` via `_maybe_seed!`.

**Also record the achieved diagnostics at close**, since they are what makes a run
auditable: per-pair swap rates, per-walker round trips, and the mean straggler gap. Put
them under `"chain.results"` so they are clearly distinguishable from inputs.

> **Acceptance.** Extend `test/test_run_metadata.jl` in its existing style: run a tiny
> PT job with a writer, read the atlas header back, assert every key.

---

### Task 7 — Threaded backend

Create `src/chains/pt_backends.jl`.

```julia
abstract type PTBackend end

"""
    advance!(backend, ensemble, steps)

Advance every replica `steps` MH steps at its current β. Returns nothing. Implementations
must leave `ensemble` otherwise untouched.
"""
function advance! end

backend_name(::PTBackend) = "unknown"
backend_workers(::PTBackend) = 1
backend_straggler_gap(::PTBackend) = 0.0
```

`SerialBackend` is a plain loop (task 4 already needs it).

`ThreadedBackend`: **one `Threads.@spawn` task per replica**, not `Threads.@threads`.

```julia
function advance!(b::ThreadedBackend, e::PTEnsemble, steps::Int)
    times = b.times                      # preallocated Vector{Float64}, length M
    tasks = map(1:length(e.replicas)) do w
        Threads.@spawn begin
            t0 = time()
            advance_replica!(e.replicas[w], e, b.proposal, e.replicas[w].work_measure, steps)
            times[w] = time() - t0        # indexed by REPLICA, never threadid()
        end
    end
    foreach(wait, tasks)
    b.last_gap = maximum(times) - sum(times)/length(times)
    return nothing
end
```

Why dynamic rather than static: spawn overhead is ~50–100 ns and a replica-block is
≥10⁴× that, so dynamic scheduling is free, and it absorbs the load imbalance between
rungs (accepted moves cost more than rejected ones, and acceptance varies with β). Since
the swap barrier makes the straggler gap the dominant cost, dynamic wins. (design §1.5)

> **Acceptance.** §7.1 test 5 — `ThreadedBackend` with `-t 1` and `-t 4` produces
> **bit-identical** output to `SerialBackend` for the same seed. This works because
> replica RNGs are seeded sequentially from the driver rng (§1 invariants). If it fails,
> something is sharing state — find it, do not weaken the test.

---

### Task 8 — Heat bath

Create `src/chains/pt_heat_bath.jl`.

```julia
"""
    HeatBath(source_path, measure, samples, rung)

Independent draws from a reference `measure`, read from a stored Atlas, exchanged
against `rung` (default: the hottest, rung 1) on the rounds the even/odd pattern leaves
it idle.

Each stored sample is consumed AT MOST ONCE — sampling with replacement from a small
pool would correlate the exchanges and break the independence the move relies on. The
run errors rather than wrapping around when the pool is exhausted, so size
`length(samples) >= n_rounds ÷ 2`.

Validity note: the move is an exact MH step only if each `y` is an independent draw from
`measure`. Drawn from a stored MCMC atlas they are approximately independent at best —
use a burn-in and a wide stride, and treat the resulting bias as a modelling assumption.
"""
struct HeatBath
    source_path::String
    measure::Measure
    samples::Vector{Dict{Tuple{Vararg{String}},Int}}   # districting assignments
    rung::Int
end
```

`parse_bath_measure(path)` — read the reference measure back from the stored atlas
header. **Reuse `examples/run_cyclewalk_extend.jl`'s reader**, which already pulls a
config out of an Atlas header and rebuilds it with `energy_specs`/`build_measure`; do
not re-derive from the raw `"energies"`/`"energy weights"` arrays.

`parse_bath_samples(path, burn_in, n, rng)` — pre-draw `n` sorted indices past
`burn_in`, then stream the atlas once collecting them in order (the MSMS approach at
`parallel_tempering.jl:170`), so the file is never held in memory.

`try_heat_bath!(ensemble, hb, round, rng, diag)`:

1. Only fire when `hb.rung in idle_rungs(M, round)`.
2. Take the next unused sample; rebuild a `LinkCutPartition` from the districting (via
   `MultiLevelPartition` then `LinkCutPartition(mlp, rng)` — same route
   `run_cyclewalk_extend.jl` uses to restart from a stored plan).
3. `φ_y = map(e -> e(y)::Float64, scores)`.
4. `logα = swap_logratio(path, lattice[hb.rung], BATH_POINT, φ_x, φ_y)` — see §2.4. The
   bath's weights come from `hb.measure`, so add a `swap_logratio` method taking two
   explicit weight tuples rather than two path points.
5. On accept: replace the rung's state and `phi`, `bath_swaps += 1`, and **reset that
   walker's `last_end_visited = 0`** — its round-trip lineage is broken and continuing
   to count would inflate the diagnostic.

> **Acceptance.** §7.3.

---

### Task 9 — Distributed backend

Only after tasks 1–8 are green. Read the Julia manual's *Distributed Computing* page,
particularly its "Global Variables" section, before starting.

The three traps, in order of how much damage they do:

1. **Never let the graph into a per-round message.** `remotecall` serializes the closure
   *with everything it captures*, so a closure capturing a `Replica` drags the whole
   `BaseGraph` to the worker every round. Hold replica state in a `const Ref` **declared
   inside the `CycleWalk` module** (not a `Main`-level `@everywhere const`, which is
   subject to the manual's binding-caching rules), populate it once at init from the
   graph's **filesystem path**, and make the per-round call carry `(steps, beta_index)`
   and return `K` floats.
2. **Issue the gather concurrently.** `M` sequential `remotecall_fetch`es cost `M × 1 ms`;
   wrapped in `@sync` + `@async` they cost ~1 ms total. At M = 64 that is 64 ms vs 1 ms
   per round.
3. **Do not use `RemoteChannel` for the replica state**, despite it being the manual's
   featured idiom for persistent per-worker state. Its example does `take!`/`put!` *from
   the master*, round-tripping the whole state object through the master every round —
   fine for a 100-element vector, catastrophic for a partition. `RemoteChannel` is for
   output streaming and coordination only.

Also: assignment is **static and pinned** (a walker lives on one rank for the run), so
`pmap` is wrong here even though it is the better-load-balanced choice in general —
migrating work would migrate state, which is the thing this design exists to avoid.
`SharedArrays` cannot hold the state at all (isbits only).

Sizing: a distributed message is ~1 ms, so `swap_interval × cost_per_MH_step` must be
≥10× that. Measure `cost_per_MH_step` on the real graph (task 10 gives you the harness)
and **warn at startup** if the product is under the floor.

Keep `addprocs`/`ClusterManagers` in the *example runner*, never in the package — the
package accepts an existing worker pool.

> **Acceptance.** §7.1 test 5 again, with `DistributedBackend` added.

---

### Task 10 — Runner, config, docs

`examples/run_pt_toml.jl`, modelled closely on `run_asmc_toml.jl` (same CLI-override
pattern, same `parameter_tag`/`ensure_writable` helpers, same writer construction).

```toml
[pt]
n_rungs       = 16
lattice       = "linear"      # "linear" | "geometric" | explicit `betas = [...]`
beta_min      = 0.05          # geometric only
swap_interval = 500           # MH steps between swap rounds
n_rounds      = 2000
init_steps    = 2000
backend       = "threaded"    # "serial" | "threaded" | "distributed"
temper        = "all"         # "all" (scalar beta, diagonal) | "gamma" | "explicit"
write_rungs   = "target"      # "target" | "all" | "none"
output_every  = 1

[pt.heat_bath]                # omit the table entirely to disable
source   = "output/reference.jsonl.gz"
rung     = 1
burn_in  = 10000
```

`temper` maps to the path (design §1.1.1):

* `"all"` → `base_w = 0`, i.e. `linear_path(target_w)`. β = 0 is ν_0.
* `"gamma"` → `base_w = target_w` **except** the `get_log_spanning_forests` entry, which
  starts at 0. Ramps γ alone with every other weight fixed. This is usually the ladder
  you want.
* `"explicit"` → use each energy's `weight_start` from `[[measure.energy]]`, via
  `build_annealed_measure`, exactly as `run_asmc_toml.jl` does.

Also write `examples/validation/smoke_pt.jl` (model: `smoke_annealed_smc.jl`) reporting
swap rates per pair, round trips, and the straggler gap; and `docs/run_pt_toml.md`.

Add a row to `examples/README.md`'s script table.

---

## 6. Ordering and definition of done

| Task | Done when |
|---|---|
| 1 Extract path | Existing suite exits 0, no test edits |
| 2 Types | Lattice/diagnostics unit tests pass |
| 3 Swap mechanics | §7.1 tests 1–3 pass |
| 4 Serial driver | §7.1 test 4 and §7.2 pass |
| 5 Output | Maps carry all five `pt/` keys |
| 6 Metadata | Header round-trip test passes |
| 7 Threaded | §7.1 test 5 (serial ≡ threaded, bitwise) |
| 8 Heat bath | §7.3 passes |
| 9 Distributed | §7.1 test 5 extended; no graph in the per-round payload |
| 10 Runner/docs | `smoke_pt.jl` runs end to end |

---

## 7. Test plan

New file `test/test_parallel_tempering.jl`, included from `test/runtests.jl`. Follow
`test_annealed_smc.jl`'s structure. Budget ≲ 60 s — CI runs the suite 4× for stochastic
variance.

Standard fixture (matches every other annealing test file):

```julia
json  = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
graph = MultiLevelGraph(BaseGraph(json, "pop",
            inc_node_data=Set(["county","pct","pop","area","border_length"]),
            area_col="area", node_border_col="border_length",
            edge_perimeter_col="length"), ["pct"])
constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(4, 4))   # the p88 hard constraint
proposal = [(0.1, build_lifted_tree_cycle_walk(constraints)),
            (0.9, build_internal_forest_walk(constraints))]
```

### 7.1 Structural — fast, deterministic, no sampling

1. **`even_odd_pairs`** — disjointness; alternation; over rounds `r` and `r+1` every
   adjacent pair appears exactly once; `idle_rungs` returns exactly the rungs in no pair, and pairs ∪ idle covers 1:M.
2. **`swap_logratio`** — equals `(β_i−β_j)*(E_i−E_j)` computed independently for a
   diagonal path; **direction check** of §2.3 (hotter rung holding higher energy ⇒
   `logα < 0`).
3. **`swap_round!` degenerate cases** — all β equal ⇒ every `logα == 0`, acceptance
   probability 1, `check_ensemble` holds after. Identical `φ` ⇒ same.
4. **Single rung** — `M == 1` reduces to `run_metropolis_hastings!`: identical final
   state for the same seed. And `n_rounds × swap_interval` total steps were taken.
5. **Backend equivalence** — `SerialBackend` vs `ThreadedBackend` (and later
   `DistributedBackend`), same seed ⇒ identical `walker_at_rung`, `round_trips`, and
   final states. Run the threaded case under `-t 1` and `-t 4`.
6. **Ensemble invariant** — `check_ensemble` true after every round of a short run.
7. **Diagnostics reuse** — run twice with the same `PTDiagnostics` object; results must
   agree (this is the `reset_schedule!` lesson; see §0 rule 2).
8. **`BetaLattice` validation** — rejects unsorted, duplicate, negative, and
   `betas[end] != 1`.

### 7.2 Correctness against ground truth — the tests that matter

The 4×4 graph with `PopulationConstraint(4,4)` is enumerated in
`test/test_cases/small_square_p88_unweighted.jl`. With `ν_γ(ξ) ∝ Tree(ξ)^(1−γ)`:

| cut edges | γ = 0 (β = 0) | γ = 1 (β = 1) |
|---|---|---|
| 8 | 256/654 = 0.3914 | 1/117 = 0.0085 |
| 10 | 224/654 = 0.3425 | 14/117 = 0.1197 |
| 11 | 96/654 = 0.1468 | 24/117 = 0.2051 |
| 12 | 78/654 = 0.1193 | 78/117 = 0.6667 |

9. **Target rung samples the target.** Ladder from β=0 to β=1 with
   `push_energy!(measure, get_log_spanning_forests, 1.0)`. Histogram
   `get_cut_edge_sum(state, column="connections")` over rung M's emitted maps and
   compare with the γ=1 column. **This is the headline test** — everything else is
   scaffolding. Suggested tolerance: L1 < 0.10 over the four bins, with enough rounds
   that the histogram is stable (calibrate over ≥5 seeds before fixing the number, as
   was done for the ASMC logZ test).
10. **Hot rung samples ν_0.** Same run, rung 1's maps must match the γ=0 column. This
    catches a swap that corrupts the *ladder* while leaving the target accidentally
    right, and it is nearly free once test 9's harness exists.
11. **Degenerate ladder ≡ independent chains.** All β = 1 (M rungs, same point): swap
    acceptance is identically 1, and each rung's marginal must still match the γ=1
    column.

### 7.3 Heat bath

12. **Matched bath.** Bath measure == rung 1's measure ⇒ high acceptance, and rung 1's
    marginal unchanged (still the γ=0 column, within tolerance).
13. **Exhaustion errors.** A bath with fewer samples than needed must throw a clear
    error, not silently reuse. Assert the error type and that the message names the
    shortfall.
14. **Lineage.** An accepted bath swap increments `bath_swaps` and resets that walker's
    `last_end_visited`.

### 7.4 Metadata

15. Round-trip every key of §6 through a written atlas header, in
    `test/test_run_metadata.jl`'s existing style.

---

## 8. Reference

* **Design rationale for every choice above**: [`plan_parallel_tempering.md`](plan_parallel_tempering.md).
* **DeFord, Herschlag, Mattingly**, *A Cycle Walk for Sampling Measures on Spanning
  Forests for Redistricting*, `~/Git/Greg/CycleWalk/Accepted/cycleWalk-siam.tex`. §2 for
  the lift of π from partitions to forests; §6 for `ν_γ ∝ e^{−γJ_Tree − J}` (the sign
  convention); §7 for γ being the hard direction; §8 for `P_1Tree = Q_1Tree`.
* **Syed, Bouchard-Côté, Deligiannidis, Doucet** (2022), *Non-reversible parallel
  tempering*, JRSS-B — the DEO scheme, the O(1) round-trip result, and the
  equal-rejection lattice-tuning rule.
* **Port sources**: `~/Git/Greg/multiscalemapsampler-public/src/parallel_tempering.jl`
  (structure, heat bath); `~/Git/Greg/NC_StateLeg/src/ParallelTempering.java`
  (β ladder, swap diagnostics).
* [Julia manual — Distributed Computing](https://docs.julialang.org/en/v1/manual/distributed-computing/)
  for task 9.
* `~/Git/Greg/BatchParallelTemprering/` — Greg's batch-tempering convergence notes; read
  before attempting the adaptive lattice (design §1.10, deferred).

---

## 9. Explicitly out of scope

Do not build these unless asked; they are recorded so nobody wonders whether they were
forgotten.

* **Adaptive β-lattice tuning** (design §1.10). Ship a fixed lattice first; add the
  equal-rejection refit once the diagnostics of task 2 show it is needed.
* **Independent PT ensembles** (`n_ensembles`, design §1.7) for the case M < workers.
  Useful, but orthogonal to correctness.
* **Unifying AIS onto `AnnealPath`.** Task 1 makes it possible and it would fix AIS's
  unverifiable `schedule` label, but it is an API change to a separate sampler.
* **MPI.** Considered and deferred; `Distributed` suffices at M ≲ 256 with a concurrent
  gather.
* **Micro-optimising the energy path.** `get_log_energy` allocates ~880 B/call and
  `measure.scores::Set{Function}` forces dynamic dispatch, but a full-measure evaluation
  measured 1.96 µs against a 44.1 µs MH step on the CT precinct graph. It is 4%. Leave
  it alone.
