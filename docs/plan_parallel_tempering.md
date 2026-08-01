# Implementation plan: parallel tempering in CycleWalk.jl

Status: **plan only — nothing implemented yet.** Written to be picked up by a fresh
session. Reference implementation being ported:
`/Users/jonm/Git/Greg/multiscalemapsampler-public/src/parallel_tempering.jl` (219 lines,
Julia, `MultiScaleMapSampler`). A second, independent implementation with a good
diagnostic set lives at `/Users/jonm/Git/Greg/NC_StateLeg/src/ParallelTempering.java`.

The deliverable is a new parallel chain in `src/chains/`, alongside `mcmc.jl`,
`annealed_importance_sampling.jl`, and `annealed_smc.jl`.

---

## 0. Requirements

1. A parallel-tempering chain in `src/chains/`.
2. The **inverse-temperature (β) lattice is specifiable** by the caller / TOML config.
3. A **heat bath** option: the end rung exchanges against draws from a reference
   measure held in a stored Atlas file.
4. **Different threads or processes run the different replicas.**
5. Must run on: `hamilton.math.duke.edu` (large shared-memory SMP), `grid.math.duke.edu`
   (Grid Engine, multi-box), and the Duke DCC (Slurm).

---

## 1. Key design decisions, and why

### 1.1 A β-lattice is a set of points on the existing `AnnealPath`

`src/chains/annealed_smc.jl` already carries the abstraction PT needs. It defines an
`AnnealPath` with two seams:

```julia
weights_at(path, t)              -> NTuple{K,Float64}   # per-energy weight at t
energy_at(path, particle, t)     -> ⟨weights_at(t), φ⟩  # φ = cached per-term potentials
configure_measure!(path, m, scores, t)                  # write weights into a Measure
```

A PT replica at inverse temperature β is exactly a particle **pinned** at `t = β`. So:

* the β-lattice is a `Vector{Float64}` of path points, `0 ≤ β_1 < … < β_M = 1`;
* `LinearPath` with a nonzero base gives **per-term** tempering — e.g. hold γ fixed and
  temper only the isoperimetric weight — which a scalar β cannot express;
* the potentials cache `φ = (e₁(state), …, e_K(state))` makes every swap decision pure
  arithmetic, with **zero energy evaluations**.

This is strictly more general than the MSMS implementation, which calls `log_measure`
four times per swap attempt.

**Required refactor (task 2 below).** `AnnealPath`, `LinearPath`, `linear_path`,
`weights_at`, `energy_at`, `configure_measure!`, `set_measure_weights!`,
`ess_from_logw`, and `annealed_smc_scores_and_targets` currently live inside
`annealed_smc.jl`. Move them to a new `src/chains/anneal_path.jl`, included *before*
both `annealed_smc.jl` and `parallel_tempering.jl`, so PT does not depend on the SMC
file. Keep `annealed_smc_scores_and_targets` as an exported alias of a newly named
`measure_scores_and_targets` — it is exported and appears in
`test/test_run_metadata.jl`, so removing the name would break tests.

### 1.1.1 What β should temper — and why it must not be a scalar

In the paper's notation (§1.3) the target is `ν(τ) ∝ exp(−[γ·J_Tree + J])`. A **scalar**
inverse temperature β scales the whole exponent:

```
ν_β(τ) ∝ exp( −β[γ·J_Tree + J] ) = exp( −[(βγ)·J_Tree + βJ] )
```

so a scalar β-ladder is the *diagonal* path `t ↦ t·(γ, w_Compact)` — it tempers γ and
the compactness weight together along one ray, with β = 0 giving `ν_0`, uniform on
balanced forests. That is a legitimate and useful ladder, and it is what
`linear_path(target_w)` already produces.

**But the diagonal is probably not the ladder you want**, because the two directions are
not equally hard. §7 of the paper is explicit that γ is the difficult direction —
convergence degrades as γ grows, and Figure `convergenceWGammaStudy` is the evidence
(*"Metropolized Forest Recombination has difficulty converging at much smaller γ than
Cycle Walk"*). γ controls how far the target sits from the proposal's natural forest
measure; `J_Compact` is an ordinary soft score. So the ladder that buys the most is
often

```
base_w   = (0,  w_Compact)      # γ = 0: ν_0, forest-uniform, proposal-aligned
target_w = (γ,  w_Compact)      # compactness held FIXED across the ladder
```

which is not expressible as a scalar β at all. This is the concrete justification for
the per-term `LinearPath` of §1.1 over a scalar β, and it should be the documented
default in the TOML runner (`temper = "gamma"` | `"all"` | explicit per-term
`weight_start`s), with the scalar diagonal available as `"all"`.

Note also that γ need not stop at 1 — the paper allows γ ∈ ℝ while noting γ ∈ [0,1] is
typical. β = 1 in the lattice means "the configured target", whatever γ_target is; it
does not mean γ = 1.

**A free efficiency property worth knowing.** Per §8, the 1-Tree (internal) Cycle Walk
needs **no Metropolization at any γ** — `m^{(α)}_ξ` is invariant for `Q_1Tree`, so
`P_1Tree = Q_1Tree` and its acceptance is identically 1 for every ν_γ^{(α)}. Since the
production proposal mixture is ~90% internal walk (`(0.1, cycle_walk),
(0.9, internal_walk)` throughout the examples and tests), temperature affects only the
~10% of steps that actually change the partition. Two consequences for PT:

* per-rung cost is more uniform than it would otherwise be, which helps the §1.5 barrier;
* the hot rungs are not "cheap" in wall-clock terms — do not expect a ladder to cost
  much less than M× a single chain.

### 1.2 Swap acceptance in φ form

For neighbouring rungs `i, j` holding states `x_i, x_j` with cached potentials
`φ_i, φ_j`, writing `w_i = weights_at(path, β_i)`:

```
log α = ⟨w_i − w_j, φ_i − φ_j⟩
```

(sign convention pending §1.3). For a pure β-ladder, `w_i = β_i · w`, this collapses to

```
log α = (β_i − β_j) · (E_i − E_j),      E = ⟨w, φ⟩
```

which is exactly `ParallelTempering.java:210`. That agreement is the algebraic
cross-check to reproduce in a unit test.

Cost per swap round: `O(M·K)` flops, K = number of energies (2–5). Nanoseconds. It
should run **serially on one thread**, not as spawned tasks.

### 1.3 ✅ RESOLVED — the energy sign convention

**`get_log_energy` returns an ENERGY — it is exactly the paper's score exponent, and
the log-density is its negative. MH (`src/chains/mcmc.jl`) is correct; AIS and ASMC are
sign-inverted.**

**The measure lives on spanning FORESTS, not on partitions.** This matters for getting
the statement right. From DeFord–Herschlag–Mattingly, *A Cycle Walk for Sampling
Measures on Spanning Forests for Redistricting* (`~/Git/Greg/CycleWalk/Accepted/
cycleWalk-siam.tex`, §2 and §6), with `π(ξ) ∝ e^{−J(ξ)}` on partitions
(eq. `pi_def`) and `Tree(ξ)` the number of spanning forests inducing ξ:

```
ν_γ(τ)  ∝  π(ξ_τ) / Tree(ξ_τ)^γ  ∝  exp( −γ·J_Tree(ξ_τ) − J(ξ_τ) )     (§6)
```

where `J_Tree(ξ) = log Tree(ξ)`. Matching term by term against the code:

| paper | code |
|---|---|
| `J_Tree = log Tree` | `get_log_spanning_forests` |
| `γ` | its weight in `push_energy!` |
| `J = J_Compact = w·Σ perim²/area` (eq. `compactscore`) | `get_isoperimetric_score` × `iso_weight` |
| `γ·J_Tree + J` — the whole exponent | **`get_log_energy`** |

So the target on the chain's actual state space is

```
ν(τ) ∝ exp( −get_log_energy(ξ_τ, measure) )
```

with no extra factor. `get_log_energy` returns the paper's score `J`-sum, and
`get_delta_energy`'s `exp(E_cur − E_prop)` is the correct Metropolis ratio for it.

**The `Tree` factor is a fiber size, not a proposal artifact.** Projecting ν_γ down to
partitions multiplies by the number of forests over each partition (ν_γ is constant on
a fiber when α ≡ 1):

```
ν_γ(ξ) = Σ_{τ∈τ_ξ} ν_γ(τ) = Tree(ξ) · π(ξ)/Tree(ξ)^γ = π(ξ) · Tree(ξ)^{1−γ}
```

which is precisely what `test/test_cases/small_square_p88_weighted.jl:87` writes as
`sfs^(1.0 - gamma) * sym_count`. That CI test and the paper agree, and both pin the
sign the same way.

**Numerical confirmation** (4×4 test graph, `PopulationConstraint(4,4)`). Plain MH
reproduces `π ∝ F^(1−γ)` at both endpoints:

| cut edges | γ=0 observed | F¹ predicts | γ=1 observed | F⁰ predicts |
|---|---|---|---|---|
|  8 | 0.3834 | 0.3914 | 0.0085 | 0.0085 |
| 10 | 0.3523 | 0.3425 | 0.1171 | 0.1197 |
| 11 | 0.1526 | 0.1468 | 0.1982 | 0.2051 |
| 12 | 0.1117 | 0.1193 | 0.6762 | 0.6667 |

— i.e. `ν_γ(ξ) ∝ Tree(ξ)^{1−γ}` with `J ≡ 0`, exactly as §6 predicts (`ν_0` uniform on
forests, `ν_1` the lift of the uniform measure on partitions).

The sharp test is the normalizing constant, known exactly on this graph:
`Z(0) = 256+224+96+78 = 654`, `Z(1) = 1+14+24+78 = 117`, so
**`log Z(1)/Z(0) = log(117/654) = −1.72093`**. This is precisely what AIS and ASMC
estimate:

| sampler | as written | negated | truth |
|---|---|---|---|
| AIS (`run_annealed_importance_sampling!`) | **+2.379** (err +4.100) | −1.742 (err −0.021) | −1.72093 |
| ASMC (`run_annealed_smc!`) | **+1.917** (err +3.638) | −1.917 (err −0.196) | −1.72093 |

Both return a positive number where the answer is negative — they are estimating
`Z(0)/Z(1)`.

**⚠️ Negating the output is NOT the fix, and the two samplers differ in how badly.** In
AIS the weights only enter at the end, so negating them recovers the right answer
almost exactly (err −0.021 ≈ Monte Carlo noise). In ASMC the weights *also drive
resampling* (`resample!`, and `should_resample`/ESS), so an inverted sign kills the
wrong particles and degrades the population itself — negating `logZ` after the fact
leaves a residual −0.196 error that no amount of sampling removes. **ASMC must be fixed
inside the algorithm.**

**Sites to fix** (all are the same transposition):

| file | line | current | correct |
|---|---|---|---|
| `annealed_importance_sampling.jl` | 21 | `weight.value += e2 - e1` | `e1 - e2` |
| `annealed_importance_sampling.jl` | 49 | `delta = e2 - e1` | `e1 - e2` |
| `annealed_smc.jl` | 106 | `p.logW + energy_at(…,t) - energy_at(…,t_prev)` | operands swapped |
| `annealed_smc.jl` | 252 | `p.logW += energy_at(…,t) - energy_at(…,t_prev)` | operands swapped |
| `annealed_smc.jl` | 260 | `dw = _sub(weights_at(path,t), weights_at(path,t_prev))` | operands swapped |

Line 106 is inside `next_t(::AdaptiveTempering, …)`, so the adaptive schedule is
currently choosing its `t` steps off inverted ESS as well. Docstrings at
`annealed_smc.jl:245` and the module header need the same correction.

**Consequence for PT.** The swap acceptance of §1.2 is written against
`log π = −get_log_energy`, giving

```
log α = ⟨w_i − w_j, φ_i − φ_j⟩          # note: NOT negated
```

Check: for a pure β-ladder this is `(β_i − β_j)(E_i − E_j)` with `E = ⟨w, φ⟩` an
*energy*, which is exactly `ParallelTempering.java:210` — and that implementation is
written for `π ∝ exp(−βE)`. The two references agree, and PT should follow them rather
than the AIS/ASMC precedent in this repo. State this at the top of
`parallel_tempering.jl` so the next reader does not "fix" it into consistency with the
neighbouring files.

### 1.4 Swap scheme: deterministic even/odd (non-reversible)

Alternate the pairing each round:

```
round odd :  (1,2) (3,4) (5,6) …
round even:  (2,3) (4,5) (6,7) …
```

All pairs in a round are attempted; the pairs are disjoint, so the whole round is one
serial pass. This is what MSMS does (`pair_start` toggling at
`parallel_tempering.jl:73`), and it is the **non-reversible / DEO** scheme of Syed,
Bouchard-Côté, Deligiannidis & Doucet (2022), whose round-trip rate is O(1) in the
number of rungs rather than the diffusive O(1/M²) of random-pair schemes. Do *not* copy
the Java approach of one random adjacent pair per attempt — with a long ladder it does
M× less exchange work per round.

The idle end rung alternates between rung 1 and rung M. §1.6 puts the heat bath there.

### 1.5 Parallelization — what actually to do with many cores

The MSMS structure should **not** be ported as-is. Three specific problems:

1. `parallel_tempering.jl:12` asserts `length(replicas) == Threads.nthreads()`. This
   ties ladder length to core count. The number of rungs is a *statistical* choice
   (set by the round-trip rate); the core count is a hardware fact. On a 128-core
   node this forces a 128-rung ladder; on a 16-core DCC allocation it forbids a
   64-rung one.
2. `parallel_tempering.jl:66` spawns a task per swap pair. Once φ is cached each swap
   is a handful of flops; task spawn dominates by orders of magnitude.
3. `parallel_tempering.jl:53` prints to stdout every swap round from inside the loop.
   Fine at 8 replicas, an I/O bottleneck at 128.

**Design instead:**

* **Decouple M (rungs) from T (workers).** Chunk the M replicas across T workers,
  `⌈M/T⌉` each. `M ≥ T` is the normal case; `M < T` is handled by §1.7.
* **Barrier once per swap interval**, not per step. Each block costs `max` over
  replicas rather than `mean`, so amortize it — see §1.5.1 for how to size it against
  the backend's communication floor. Report the straggler gap as a diagnostic.
* **Swap round is serial** on the driver thread: gather φ, decide all pairs, apply.
* **One `Threads.@spawn` task per replica**, not `Threads.@threads`. Spawn overhead is
  ~50–100 ns and a replica-block is ≥ 10⁴× that, so dynamic scheduling is free here,
  and it absorbs the load imbalance between rungs (accepted moves cost more than
  rejected ones — `update_partition!` does link/cut edits plus energy-cache rolls — so
  rungs with different acceptance rates finish at different times). Since the barrier
  makes the straggler gap the dominant cost, dynamic wins. This is a deliberate
  divergence from `rejuvenate!` (`annealed_smc.jl:342`), which uses static
  `Threads.@threads`; ASMC has the same barrier and may want the same change.
* **Never index anything by `Threads.threadid()`.** With `@spawn` (and with
  `@threads :dynamic`, the default since 1.8) a task may migrate between threads at a
  yield point, so `threadid()` can change *mid-task* and thread-indexed scratch
  silently corrupts. Index every per-worker buffer by **replica index**. CycleWalk
  already does this correctly — `rejuvenate!` indexes `diags[i]` by particle — but the
  older `cache[threadid()]` idiom is still widely published (including in the SciML
  notes' thread-local-storage section) and must not be copied in.
* **Pin BLAS to 1 thread** for the duration of a multi-worker run and restore on exit.
  The spanning-forest energy takes a log-det per district via LAPACK; several Julia
  threads each spawning a BLAS pool oversubscribes the machine badly. This is already
  the established pattern — copy it verbatim from `annealed_smc.jl:421-422` and
  `annealed_importance_sampling.jl:175-176`, including the `try`/`finally` restore.
* **Thread safety is already established.** `rejuvenate!` (`annealed_smc.jl:339`)
  threads `run_metropolis_hastings!` across particles given (a) an independent
  partition per worker, (b) an independent RNG per worker, and (c) an independent
  `RunDiagnostics` per worker (the MH loop mutates diagnostics — a shared object
  races). PT inherits the same three requirements; do not share any of the three.

### 1.5.1 Sizing the swap interval against the communication floor

The SciML notes (§9) give the latency hierarchy that decides this:

| mechanism | per-operation overhead | minimum work to amortize |
|---|---|---|
| `Threads.@spawn` | 50–100 ns | ~1 µs |
| CPU↔GPU transfer | 20–50 µs each way | ~200 µs–1 ms |
| `Distributed` message | hardware-dependent | **~1 ms** |

PT pays one round of this per swap interval. So:

```
block wall time ≈ swap_interval × cost_per_MH_step
```

and the requirement is `block time ≫ backend floor`, with a target of ≥ 10× so
communication stays under ~10% of runtime.

* **Threads**: floor ~1 µs. Even `swap_interval = 10` is comfortably amortized. The
  interval is effectively a free parameter, set by statistics alone.
* **Distributed**: floor ~1 ms. At an estimated 10–100 µs per MH step on a precinct-
  scale graph, `swap_interval = 100` gives 1–10 ms — *marginal*. `swap_interval ≥ 1000`
  gives 10–100 ms, comfortably 10–100× the floor. **Measure the real per-step cost on
  the target graph before choosing** rather than trusting that estimate.

**The consequence is a decision rule, not just a tuning tip.** `swap_interval` is a
*sampler* parameter — changing it changes the chain, not merely its speed. So it cannot
be tuned per-backend. If the statistically desirable interval (short, for frequent
exchange and good round-trip rates) is below the ~1 ms Distributed floor, **that is
itself the argument for the threaded backend on that problem**. Frequent-exchange
ladders belong on one shared-memory node.

Add a startup check: time a few hundred MH steps, and warn if
`swap_interval × measured_step_cost` is under 10× the backend's floor.

### 1.6 Distributed: swap the label, not the state

Under `Distributed`, the decisive design property is that **partition state never moves
between ranks.** Two mathematically equivalent formulations of a PT swap:

* swap the *states* between two rungs (what MSMS and the Java code do — they exchange
  the objects); or
* swap the *β labels* between two chains, each keeping its own state.

The second is what a distributed implementation wants. Each worker rank owns one
replica for the life of the run. Per swap round it sends the master its K cached
potentials; the master decides every swap from the φ's alone and sends back a new β
index per rank. **Communication per round is M×K Float64 regardless of graph size** —
for M = 64, K = 3, that is 1.5 kB.

This keeps the master/worker boundary narrow enough that both backends fit one
interface:

```julia
abstract type PTBackend end

# advance every replica `steps` MH steps at its current β; return nothing
advance!(backend, ensemble, steps)
# current per-replica potentials, in rung order
gather_potentials(backend, ensemble) -> Vector{NTuple{K,Float64}}
# assign rung k the inverse temperature betas[k] (after a swap round)
set_betas!(backend, ensemble, betas)
```

`ThreadedBackend` implements these with `Threads.@spawn` over chunks and direct array
access. `DistributedBackend` implements them with `remotecall`/`fetch` over a
`WorkerPool`, plus a `RemoteChannel` per rank for output.

**Recommendation: build the seam now, ship `ThreadedBackend` first, add
`DistributedBackend` second.** Three reasons, in order of weight:

1. **A ~10⁴× communication gap.** ~50–100 ns per thread spawn against ~1 ms per
   distributed message (§1.5.1). PT pays this every swap round, so on a single node the
   threaded backend permits swap intervals two to three orders of magnitude shorter —
   i.e. far more exchange per unit compute, which is precisely what drives the
   round-trip rate PT exists to maximize.
2. **Memory.** `clone_for_annealing` (`src/partition/link_cut_partition.jl:57`)
   deliberately *shares* the `BaseGraph` between replicas; its docstring notes the graph
   is ~75% of a partition's bytes. Threads pay for the graph once, Distributed pays P
   times. An NC precinct graph × 64 ranks is real.
3. **The hazards are already solved for threads here** — BLAS pinning and per-worker
   diagnostics both have working precedent in `annealed_smc.jl` and
   `annealed_importance_sampling.jl`.

Where Distributed still earns its keep: ladders too long for one node, `grid.math`
(heterogeneous boxes, no shared memory), and escaping GC contention on very long runs.
The seam is worth building in phase 3 even though the implementation lands in phase 5 —
retrofitting it later would mean rewriting the driver.

**Distributed-specific work items** (defer to phase 5):

* `Distributed` and `Serialization` are stdlibs — add to `[deps]`, no version churn.

* **The single most important detail: never let the graph into a message.**
  `remotecall` serializes the closure *together with everything it captures*. A naive
  `remotecall(w, () -> advance!(replica, steps))` captures the replica, which holds the
  `BaseGraph` — so every block would serialize the entire graph to every rank. That
  turns a working design into a 100× slowdown, and it will not show up on the 4×4 test
  graph.

  Instead: hold replica state in a `const Ref` **declared inside the `CycleWalk`
  module**, populated once at init, and make the per-block call carry only scalars:

  ```julia
  # in src/chains/pt_backends.jl — module-scoped, NOT a Main-level global
  const _PT_REPLICA = Ref{Any}(nothing)

  # once, at init — each rank constructs its own replica from the shared path
  remotecall_wait(w, graph_path, cfg, rung, seed) do path, cfg, rung, seed
      CycleWalk._PT_REPLICA[] = CycleWalk.build_replica(path, cfg, rung, seed)
      nothing
  end

  # per block — argument payload is (Int, Int); return payload is K Float64
  remotecall(CycleWalk.advance_local!, w, steps, beta_index)
  ```

  Ranks load the graph **from the shared filesystem path**, never by serializing a
  `MultiLevelGraph` from the master. All three target clusters have a shared home
  directory, so every rank reads the same JSON. This is the manual's own
  "construct it on the worker" idiom (`@spawnat :any rand(1000,1000)^2` rather than
  building locally and sending).

  **Module-scoped, not `@everywhere const` at `Main` level.** The manual's rule that
  new global bindings are created on destination workers and cached until their value
  changes applies *"under `Main` module only"*. Declaring the holder inside `CycleWalk`
  sidesteps that machinery entirely — no implicit binding creation, no cached-global
  lifetime questions. Workers need `@everywhere using CycleWalk` (or `julia -p n -L
  setup.jl`) before any remote call, since all remotely executed code must already be
  loaded on the target.

* **⚠️ Do NOT use `RemoteChannel` to hold the replica state**, despite it being the
  manual's featured idiom for persistent per-worker state across rounds. Its worked
  example does `take!(state_ref)` / `put!(state_ref)` **from the master**, which
  serializes the whole state object to the master and back every round. That is fine
  for the manual's 100-elem vector and catastrophic for a `LinkCutPartition` carrying a
  graph — it reintroduces exactly the cost the label-swapping design exists to avoid.
  `RemoteChannel` is the right tool here only for *output* (a worker streaming finished
  maps to a writer task) and for coordination, never for the hot state.

* If a closure ever does have to capture something non-trivial, use a
  **`CachingPool`** rather than a bare `WorkerPool`: it caches the serialized closure on
  the workers instead of re-sending it per call. Prefer restructuring so nothing large
  is captured at all.

* **Issue the gather concurrently, not in a loop.** M sequential `remotecall_fetch`es
  cost M × 1 ms; issued concurrently they cost ~1 ms total. Use the notes' §5 pattern:

  ```julia
  @sync for (k, w) in enumerate(pool)
      @async phis[k] = remotecall_fetch(advance_and_report, w, steps, betas[k])
  end
  ```

  With M = 64 that is the difference between 64 ms and 1 ms of communication per swap
  round. Getting this wrong makes distributed PT look inherently slow when it isn't.

* **Static, pinned assignment — `pmap` is actively wrong here.** The notes present
  `pmap` (dynamic) as the better-load-balanced choice over `@distributed` (static), and
  in general it is. But PT's whole architecture rests on each replica living on one
  rank for the life of the run (§1.6): if work migrated to whichever rank were free,
  the replica's state would have to migrate with it, and state migration is exactly what
  the label-swapping design exists to avoid. So the distributed backend is static and
  pinned, by construction. The load imbalance that `pmap` would fix is instead absorbed
  by choosing a swap interval long enough that per-rung cost differences average out.

  Note the asymmetry with the threaded backend (§1.5), which *does* use dynamic
  `@spawn`: under shared memory a task can move between threads without the state moving
  at all, so dynamic scheduling is free there and impossible here.

* **`SharedArrays` cannot hold the state at all** — the manual is explicit that only
  `isbits` types are supported, and a `LinkCutPartition` is a pointer-linked tree of
  mutable nodes. Same for `DistributedArrays` / `Elemental`: they parallelize *arrays*,
  which is not the shape of this problem. Don't spend time on them.

* **MPI was considered and deferred.** The DEO exchange in §1.4 is a textbook SPMD
  nearest-neighbour pattern (odd–even transposition), and paired even/odd send/recv is
  deadlock-free, so `MPI.jl` would fit PT better than it fits most problems, and would
  remove the master bottleneck. It is still out of scope: the notes rate MPI a last
  resort on safety grounds, `Distributed`'s master–worker model is far easier to debug,
  and the master is not a bottleneck at M ≲ 256 when the gather is concurrent. Revisit
  only if ladders reach the hundreds of rungs across many nodes.

* Cluster launch: `addprocs(SlurmManager(n))` from `ClusterManagers.jl` for DCC;
  `addprocs([("host", n), …])` or `--machine-file` for grid.math. Keep this in the
  *example runner*, not in the package — the package should accept an existing worker
  pool and never call `addprocs` itself.

* Output: one Atlas shard per rank is far simpler than funnelling through the master.
  See §1.8.

### 1.7 Using cores when M < T: independent ensembles

If the ladder wants 16 rungs and the node has 128 cores, 112 cores idle. Support
`n_ensembles = R`: R independent PT ladders run concurrently, M×R ≤ T, each with its
own seed. This is a better use of cores than lengthening the ladder past its useful
size, and it produces between-ensemble variance for free — feed it straight to the
existing `examples/validation/analyze_rhat.jl`. Tag every emitted map with its
ensemble index.

### 1.8 Output

PT produces samples of the target only at the β = 1 rung; the other rungs are
machinery. But rung-wise output is what the diagnostics need. Plan:

* Default `write_rungs = :target` — one Atlas holding the β = 1 rung's trajectory.
* `write_rungs = :all` — one Atlas **per rung**, suffixed `_beta<k>`. Per-rung files
  (rather than one file with a rung field) keep the write path lock-free and match
  what `DistributedBackend` wants anyway.
* Every map records **both** `replica_id` (which physical walker) and `beta_index`
  (which rung), plus `bath_swaps`. MSMS carries exactly these as partition extensions
  (`parallel_tempering.jl:20-23`) and they are what round-trip diagnostics need:
  rung-indexed output is the sample stream, replica-indexed output is the mixing
  diagnostic. Losing either makes the run unauditable.
* If a single shared `Writer` is ever needed, reuse AIS's pattern: `build_output_map`
  is documented as safe to call concurrently, and a lone consumer task drains a
  `Channel` and is the only caller of `addMap`
  (`annealed_importance_sampling.jl:195-206`). Do not call `addMap` from workers.

### 1.9 Heat bath

MSMS's second `try_swap_replicas!` method (`parallel_tempering.jl:103`) exchanges the
end rung against a plan drawn from a stored Atlas of samples from a *reference*
measure. This is a valid MH-within-PT move because the reference draws are independent
of the chain. Port it, with these details:

* `parse_base_measure` (`:139`) reconstructs the reference `Measure` from the stored
  Atlas header (`gamma`, `energies`, `energy weights`). CycleWalk's equivalent is
  richer — `energy_specs` / `build_measure` (`src/measure/measure_spec.jl`) already
  reads a `[measure]` table, and `Writer` stamps the full config into the header.
  Read the header's config back rather than re-deriving from `energies` + weights.
  `examples/run_cyclewalk_extend.jl` already reads a config out of an Atlas header —
  reuse that reader.
* `parse_base_samples` (`:170`) pre-draws sample indices past a burn-in, sorts them,
  and streams the Atlas once collecting them in order. Keep this — it avoids holding
  the whole file. It costs one full pass to count lines first; consider recording the
  map count in the header instead.
* Reconstructing a partition from a stored plan: MSMS does
  `MultiLevelPartition(graph, base_map)`. CycleWalk has
  `LinkCutPartition(partition::MultiLevelPartition, rng)`
  (`src/partition/link_cut_partition.jl`) and `run_cyclewalk_extend.jl` restarts from
  an Atlas's last plan — reuse that path.
* **Each stored sample must be consumed at most once.** MSMS indexes
  `base_samples[swap_ind]`, one per swap round — correct. Do not draw with replacement
  from a small pool.
* **Attach the bath to the rung the even/odd pattern leaves idle.** MSMS gates the
  bath swap on `pair_start == 2` (`:57`), so it fills the end rung on exactly the
  rounds it would otherwise be unpaired. Free work. Replicate.
* A bath swap replaces the end rung's state, breaking that walker's round-trip
  lineage. Increment `bath_swaps` and carry `replica_id` forward, as MSMS does
  (`:126-131`), so diagnostics can see it.

### 1.10 Adaptive β-lattice (phase 6, optional)

The DEO scheme comes with a placement rule: choose the lattice so the *rejection rate
is equal between every adjacent pair* (equivalently, equal increments of the
communication barrier `Λ`). Implement as an optional warm-up phase — run with the
initial lattice, accumulate per-pair rejection rates, refit the lattice by
interpolating the cumulative barrier, then freeze it before the production run. This
mirrors `AdaptiveTempering` in `annealed_smc.jl:74`, and the frozen lattice must be
recorded in the run metadata so the run is reproducible.

---

## 2. File-by-file work plan

| # | File | Action |
|---|------|--------|
| 1 | ✅ *done* — §1.3 sign convention resolved: `π ∝ exp(−get_log_energy)`; AIS/ASMC inverted | Fixing AIS/ASMC is a **separate decision** (changes prior results); PT does not depend on it |
| 2 | `src/chains/anneal_path.jl` | **New.** Move the path/seam code out of `annealed_smc.jl` (§1.1). |
| 3 | `src/chains/annealed_smc.jl` | Delete the moved block; no behaviour change. Tests must stay green. |
| 4 | `src/chains/parallel_tempering.jl` | **New.** Types + driver (§3). |
| 5 | `src/chains/pt_backends.jl` | **New.** `PTBackend`, `ThreadedBackend`; `DistributedBackend` in phase 5. |
| 6 | `src/chains/pt_diagnostics.jl` | **New.** Swap matrix, round trips, rung occupancy (§4). |
| 7 | `src/io/run_metadata.jl` | Add `parallel_tempering_run_metadata` (§5). |
| 8 | `src/CycleWalk.jl` | `include` the four new files; uncomment and fill the `# parallel tempering` export block at lines 193–196. |
| 9 | `Project.toml` | Add `Distributed`, `Serialization` (stdlibs) in phase 5. |
| 10 | `test/test_parallel_tempering.jl` | **New** (§6). Add to `test/runtests.jl`. |
| 11 | `examples/run_pt_toml.jl` | **New.** Model on `examples/run_asmc_toml.jl`. |
| 12 | `examples/toml/param_pt_grid.toml` | **New.** Model on `param_annealed_smc_grid.toml`. |
| 13 | `examples/validation/smoke_pt.jl` | **New.** Model on `smoke_annealed_smc.jl`. |
| 14 | `docs/run_pt_toml.md` | **New.** Model on `docs/run_cyclewalk_toml.md`. |

Note: `test/test_docs_coverage.jl` and `test/test_run_metadata.jl` are strict about
docstrings and header contents. Every new exported name needs a docstring in the house
style (see `annealed_smc.jl` for the register — it explains *why*, not just *what*).

---

## 3. Core types and driver

```julia
"""
    Replica{K}

One PT walker: its `state`, the rung index `beta_index` it currently sits at, the
cached path-independent per-term potentials `phi`, an independent RNG, its own
`RunDiagnostics`, and identity/lineage counters (`replica_id`, `bath_swaps`) that
follow the walker as it moves up and down the ladder.
"""
mutable struct Replica{K}
    state::LinkCutPartition
    beta_index::Int
    phi::NTuple{K,Float64}
    rng::PCG.PCGStateOneseq
    diagnostics::RunDiagnostics
    replica_id::Int
    bath_swaps::Int
end

"""
    BetaLattice(betas)

The inverse-temperature lattice: a strictly increasing vector of path points with
`betas[end] == 1.0` (the target). Constructors for the two standard shapes,
`geometric_betas` and `linear_betas`.
"""
struct BetaLattice
    betas::Vector{Float64}
end

"""
    HeatBath(measure, samples, rung)

Independent draws from a reference `measure`, read from a stored Atlas, exchanged
against the ladder's `rung` end on the rounds the even/odd pattern leaves it idle.
"""
struct HeatBath ... end

run_parallel_tempering!(
    partition, proposal, measure, lattice, swap_interval, n_swaps, rng;
    path              = nothing,             # defaults to linear_path(target_w)
    backend           = ThreadedBackend(),
    heat_bath         = nothing,
    n_ensembles       = 1,
    init_steps        = 0,
    write_rungs       = :target,             # :target | :all | :none
    writer            = nothing,
    run_diagnostics   = RunDiagnostics(),
    seed              = nothing,
) -> (replicas, diagnostics, trace)
```

Driver outline:

```
stamp_run_metadata!(writer, parallel_tempering_run_metadata(...))
prev_blas = BLAS.get_num_threads(); nworkers > 1 && BLAS.set_num_threads(1)
try
    build M replicas: clone_for_annealing(partition), independent rng seeded
        SERIALLY from `rng`, independent RunDiagnostics, phi computed
    init_steps > 0 && advance!(backend, ensemble, init_steps)   # decorrelate rungs
    for swap_round in 1:n_swaps
        advance!(backend, ensemble, swap_interval)      # parallel, barrier at end
        phis = gather_potentials(backend, ensemble)     # cheap
        pairs = even_odd_pairs(M, swap_round)           # serial from here
        for (i,j) in pairs
            accept swap by ⟨w_i − w_j, φ_i − φ_j⟩; on accept exchange beta_index
        end
        heat_bath !== nothing && try_heat_bath!(...)    # on the idle end rung
        set_betas!(backend, ensemble, lattice)
        record(diagnostics, swap_round, pairs, accepts)
        emit maps per write_rungs
    end
finally
    nworkers > 1 && BLAS.set_num_threads(prev_blas)
end
```

**Reproducibility requirement, copied from AIS.** Seed each replica from a draw made
*sequentially* on the driver's `rng`
(`annealed_importance_sampling.jl:231`), so results are identical for every thread
count and identical between `ThreadedBackend` and `DistributedBackend`. This is what
makes the §6 cross-backend test possible, and it is worth the small serial cost.

---

## 4. Diagnostics

The Java implementation is the better model here — it is much richer than the Julia one
and the fields are the right ones. Port:

* per-adjacent-pair attempt / accept counts and mean acceptance probability
  (`ParallelTempering.java:26-28`, the `numTry`/`numSucceed`/`acceptProbSum` matrices);
* per-replica rung-occupancy histogram (`iterSpentOnBeta`), which is flat iff the
  lattice is well placed;
* **round-trip count** — the number of times a walker travels β_M → β_1 → β_M. This is
  the single most informative PT diagnostic and neither reference implementation
  computes it. Track a per-replica "last end visited" flag and count transitions.
* straggler gap per block (`max − mean` worker time), to size `swap_interval`.

Emit these into the Atlas header at close, and expose a `PTDiagnostics` struct so a
runner can print the swap matrix as the Java version does at
`ParallelTempering.java:130-136`.

---

## 5. Run metadata

Add to `src/io/run_metadata.jl`, matching the existing three builders exactly in shape:

```julia
Dict("chain.run"        => "parallel tempering CycleWalk",
     "chain.parameters" => Dict(
        "function"        => "run_parallel_tempering!",
        "betas"           => lattice.betas,
        "n_rungs"         => M,
        "swap_interval"   => swap_interval,
        "n_swaps"         => n_swaps,
        "swap_scheme"     => "deterministic even/odd (non-reversible)",
        "backend"         => "threaded" | "distributed",
        "workers"         => T,
        "n_ensembles"     => R,
        "init_steps"      => init_steps,
        "heat_bath"       => nothing | Dict("source"=>path, "rung"=>k, "n_samples"=>n),
        "weights.base"    => _label_weights(scores, weights_at(path, betas[1]),   measure),
        "weights.target"  => _label_weights(scores, weights_at(path, betas[end]), measure),
        "threads"         => Threads.nthreads(),
        "proposal"        => describe_proposal(proposal),
        "seed"            => seed))
```

`test/test_run_metadata.jl` tests the other three builders thoroughly; extend it in the
same style. If the lattice is adaptively refit (§1.10), the **frozen final** lattice is
what gets recorded, and record the initial one too under `"betas.initial"`.

---

## 6. Testing

Follow `test/test_annealed_smc.jl`: unit-test the pure helpers deterministically, then
drive the whole sampler on invariants. All on the 4×4 grid / 2×2 county test graph.

Unit:
1. `even_odd_pairs` — disjoint, alternating, covers every adjacent pair over two rounds,
   correct idle end rung.
2. Swap acceptance — for a pure β-ladder, `⟨w_i − w_j, φ_i − φ_j⟩` equals
   `(β_i − β_j)(E_i − E_j)` computed independently. This is the algebraic
   cross-check against the Java formula.
3. `BetaLattice` validation — rejects non-monotone lattices and `betas[end] != 1`.
4. `geometric_betas` reproduces the Java `fillBetas` ladder
   (`ParallelTempering.java:37-59`) for the same endpoints.

Driver invariants:
5. **Degenerate ladder.** All β equal ⇒ every swap has log α = 0 ⇒ acceptance 1 ⇒ the
   run is M independent chains. Check the accept rate is exactly 1 and each rung's
   marginal matches a plain MH run.
6. **Single rung.** M = 1 reduces to `run_metropolis_hastings!` — identical output for
   an identical seed.
7. **Correctness at the target.** The β = 1 rung's cut-edge distribution matches the
   theoretical distribution the existing tests check (`get_observed_cut_edges` in
   `test/runtests.jl:20` is the harness). This is the real test.
8. **Thread-count invariance.** Same seed, `T = 1` and `T = 4` ⇒ bitwise-identical
   output. Guaranteed by the serial per-replica seeding in §3.
9. **Backend invariance** (phase 5). `ThreadedBackend` and `DistributedBackend`, same
   seed ⇒ identical output.
10. **Heat bath.** With a bath whose reference measure equals the end rung's measure,
    acceptance should be high and the target marginal unchanged; with the bath source
    exhausted, the run must error clearly rather than silently reuse samples.
11. Metadata round-trip through the Atlas header, per `test/test_run_metadata.jl`.

CI runs the suite 4× for stochastic variance (`.github/workflows/ci.yml`); keep PT's
statistical tests within that budget — target ≲ 60 s.

---

## 7. Phasing

| Phase | Content | Gate |
|-------|---------|------|
| 1 | §1.3 sign investigation | Written conclusion; AIS/ASMC fixed or documented |
| 2 | `anneal_path.jl` refactor | Existing ASMC/AIS tests still green |
| 3 | Types, `PTBackend` seam, serial driver (`M` replicas, no threading), even/odd swaps, β lattice | Tests 1–7 pass |
| 4 | `ThreadedBackend` (`@spawn` per replica), BLAS pinning, diagnostics, metadata, TOML runner | Test 8 passes; **measured per-MH-step cost on the production graph** (the input to §1.5.1); scaling measured on hamilton |
| 5 | `DistributedBackend`, cluster launch recipes for grid.math + DCC | Test 9 passes; **verified no graph in the per-block payload** (§1.6) and gather issued concurrently; swap interval clears the 10× floor |
| 6 | Heat bath | Test 10 passes |
| 7 | Adaptive β lattice (optional) | Round-trip rate improves vs. fixed geometric |

Phase 3 being *serial* is deliberate: it makes the swap algebra debuggable without any
concurrency in the picture, and tests 5–7 are the ones that actually establish
correctness. Parallelism is a performance change layered on a correct sampler, not part
of its definition.

---

## 8. Cluster notes

* **hamilton.math.duke.edu** — shared-memory SMP. `julia -t N --project=.`,
  `ThreadedBackend`. This is the primary target and phase 4 finishes it.
* **DCC (Slurm)** — a single fat node with `-t N` covers most runs and is simpler and
  faster than multi-node. For multi-node, `ClusterManagers.jl`'s `SlurmManager`, in the
  *runner script*, not the package.
* **grid.math.duke.edu** — Grid Engine, heterogeneous boxes, no shared memory ⇒
  `DistributedBackend` with an explicit host list. Note that heterogeneous core speeds
  make the per-block barrier worse; size `swap_interval` from the slowest box.
* On every target: pin BLAS to 1 thread (§1.5), and set `JULIA_NUM_THREADS` /
  `--threads` explicitly rather than relying on `auto`, which reads the machine's core
  count and not the scheduler's allocation.

---

## 9. Reference reading

* **DeFord, Herschlag, Mattingly, *A Cycle Walk for Sampling Measures on Spanning
  Forests for Redistricting*** —
  `~/Git/Greg/CycleWalk/Accepted/cycleWalk-siam.tex`. The definitional source for what
  the sampler targets. §2 (eq. `pi_def`, `nu_0_1`, `nudefinition`, `mAlpha`) for the
  lift of π from partitions to forests; **§6 for the score-function form
  `ν_γ ∝ e^{−γJ_Tree − J}` that fixes the sign convention** (§1.3); §7 for γ being the
  hard direction (§1.1.1); §8 for `P_1Tree = Q_1Tree` and the α-weighted measure. Write
  the PT docstrings in this notation so the authors can review them against the paper.
* [Julia manual — *Distributed Computing*](https://docs.julialang.org/en/v1/manual/distributed-computing/):
  the API reference for phase 5. `remotecall`/`remotecall_fetch`/`@spawnat`/`Future`;
  **the "Global Variables" section** on how closures implicitly serialize what they
  capture and how `Main`-level bindings are created and cached on workers (§1.6);
  `@everywhere` and `-L` for code loading; `WorkerPool`/`CachingPool`; `RemoteChannel`
  vs `Channel`; `ClusterManager`'s `launch`/`manage` interface; and the
  "construct it on the worker" data-movement idiom. Note that its featured
  `RemoteChannel` persistent-state example is a trap at our state size — see §1.6.
* SciML Book, [ch. 5 — *The Basics of Single Node Parallel
  Computing*](https://book.sciml.ai/notes/05-The_Basics_of_Single_Node_Parallel_Computing/):
  static (`@threads`) vs dynamic (`@spawn`) scheduling, the ~50–100 ns spawn cost, the
  `@sync`/`@async` concurrent-issue pattern, and thread-local storage hazards. Drives
  §1.5 and §1.5.1.
* SciML Book, [ch. 6 — *The Different Flavors of
  Parallelism*](https://book.sciml.ai/notes/06-The_Different_Flavors_of_Parallelism/):
  the latency hierarchy (threads ~1 µs / GPU ~200 µs / distributed ~1 ms minimum
  amortizable work), master–worker vs SPMD, `pmap` vs `@distributed`, and the
  `SharedArrays`/`DistributedArrays`/`Elemental` ladder. Drives §1.5.1 and §1.6.
* Syed, Bouchard-Côté, Deligiannidis, Doucet, *Non-reversible parallel tempering: a
  scalable highly parallel MCMC scheme*, JRSS-B 2022 — the DEO scheme, the O(1)
  round-trip result, and the equal-rejection lattice rule (§1.4, §1.10).
* `/Users/jonm/Git/Greg/BatchParallelTemprering/` — Greg's own batch-tempering
  convergence notes (`batchTemporingConvergence.tex`, `efficiency_proof.tex`). Read
  before phase 7; the heat-bath construction in §1.9 is the practical side of this.
* `/Users/jonm/Git/Greg/multiscalemapsampler-public/src/parallel_tempering.jl` — the
  port source.
* `/Users/jonm/Git/Greg/NC_StateLeg/src/ParallelTempering.java` — the better
  diagnostics and the geometric β ladder.
