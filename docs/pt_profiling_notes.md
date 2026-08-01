# Parallel tempering profiling notes

Working notes from profiling `ThreadedBackend` scaling, to decide whether the
Distributed backend (roadmap task 9) is actually needed before building it. Updated
as runs complete — treat earlier numbers as superseded when a methodology fix is
noted below them.

**Bottom line**: `ThreadedBackend` scales to at least 64 cores with no sign of an
absolute throughput ceiling, *provided* the rung count (M) is set high enough to
match the thread count — it was never algorithm-limited, an earlier test just had M
fixed too low. The one nuance added since: the *relative* 64-thread speedup (vs.
1 thread) shrinks as districts get bigger — from ~5-6x at small district sizes down
to ~2.8x at the largest tested (hex50, 500 nodes/district) — plausibly because the
dominant per-step cost (a dense Cholesky sized to district, not graph, dimension)
competes harder for memory bandwidth/cache as M concurrent replicas' matrices grow.
See "Design implication for task 9" at the bottom.

Script: `examples/profile_pt_threaded.jl` (not part of the package; a profiling
tool). Usage: `julia --project=examples -t N profile_pt_threaded.jl [backend]
[graph] [M]`.

## Methodology

- Fixed workload per run: `M * swap_interval(200) * n_rounds(50)` total MH steps,
  `write_rungs=:none` (no atlas I/O — isolates compute scaling from write
  overhead), `init_steps=0` (steady-state throughput only, not correctness).
  One small untimed warm-up call absorbs JIT/precompilation before the timed run.
- **Correctness proven separately, already**: `test/test_parallel_tempering.jl`'s
  "ThreadedBackend reproduces SerialBackend bitwise" test (subprocess at `-t1`/`-t4`)
  already confirms `ThreadedBackend` is bit-identical to `SerialBackend` regardless
  of thread count. This document is about *throughput*, not correctness.
- **Methodology fix (important)**: `ThreadedBackend.advance!` spawns exactly `M`
  tasks per round — one per replica — so a run with `M` fixed below the thread
  count cannot show scaling past `M`, no matter how many cores are available. The
  script now defaults `M = max(nthreads, 8)` rather than a fixed `M=8`. Any table
  below marked "(M=8 fixed)" predates this fix and undercounts scaling at high
  thread counts — read it as "8-rung scaling," not general scaling.
- Graphs, small to large: `hex10`/`hex20`/`hex30`/`hex40`/`hex50` (N×N hex
  lattices, 100–2500 nodes; `hex` is an alias for `hex10`), `ct` (Connecticut
  precincts, ~600 nodes), `nc` (North Carolina precincts, 2650 nodes), `oh` (Ohio
  precincts, 7404 nodes). Real-map/hex-size JSON pulled from
  https://quantifyinggerrymandering.pages.oit.duke.edu/codedoc/geographic.html
  (`Geo/Adjacency/{NC_pct21,OHpct20,hex_graph_N_by_N}.json`).
- **Methodology fix (nc)**: `nc`'s `uid` (`county_prec_id`, needed because raw
  `prec_id` isn't unique across counties) is a *derived* column
  (`derive_node_columns!`), not one already present in the downloaded JSON. The
  script previously built `Graph` directly from the raw JSON asking for `"uid"`
  outright, which throws `KeyError: key "uid" not found` — so `nc` was never
  actually runnable through this script before now; there is no earlier `nc` data
  to treat as superseded. Fixed by building a `BaseGraph` first, deriving `uid`,
  then wrapping it (`Graph(base_graph, "uid")`), matching how
  `run_cyclewalk_toml.jl` handles `[plans.derive]`.

## Local Mac (10 cores)

### Corrected (M = max(nthreads, 8))

| graph | 1 thread | 2 | 4 | 6 | 8 | 10 (M=10) |
|---|---|---|---|---|---|---|
| hex | 170,567 steps/s | 246,580 (1.45x) | 483,369 (2.83x) | 554,289 (3.25x) | 709,301 (4.16x) | 682,724 |
| ct  | 25,037 | 41,379 (1.65x) | 63,425 (2.53x) | 80,877 (3.23x) | 97,898 (3.91x) | 107,703 |
| nc  | 15,354 | 23,377 (1.52x) | 35,239 (2.30x) | 42,057 (2.74x) | 46,104 (3.00x) | 50,217 |
| oh  | 3,625 | 4,126 (1.14x) | 6,205 (1.71x) | 7,008 (1.93x) | 8,052 (2.22x) | 8,773 |

## Observations so far

1. **Scaling degrades as the graph gets bigger.** hex (smallest) scales best
   (~4.2x at 8 threads), oh (largest) worst (~2.2x at 8 threads). This is the
   opposite of what pure compute-bound parallelism would predict (more work per
   task should amortize fixed overhead *better*) — points at something
   graph-size-dependent capping the gain, not just thread/scheduling overhead.
2. Candidate explanations to check against Hamilton's higher core count, not yet
   confirmed:
   - Memory bandwidth / cache contention: `clone_for_annealing` shares the base
     graph but each replica has its own link-cut tree + energy caches; a bigger
     graph means bigger per-replica state, more memory traffic when threads run
     concurrently, less of it fitting in per-core cache.
   - BLAS pinning: the driver pins BLAS to 1 thread for the whole run when
     `nthreads()>1`; if OH's spanning-forest energy leans harder on the LAPACK
     log-det than the smaller graphs, single-threaded BLAS could be a bigger
     relative cost on OH specifically.
   - GC pressure: bigger graphs → bigger per-replica allocations per MH step;
     Julia's GC can serialize across threads.
3. `mean_straggler_gap` stays small everywhere (< 0.09s) — load imbalance between
   rungs is not the bottleneck at these M; consistent across corrected and
   uncorrected runs.
4. **Answered by the Hamilton run below**: scaling continues past 8-10 threads all
   the way to 64, once M scales too — no early plateau. See "Design implication for
   task 9."

## Hamilton (64 cores)

`ct` and `oh`, nthreads in {1,2,4,8,16,32,48,64}, M=max(nthreads,8) (so M=8 for the
1/2/4/8 points, then M tracks nthreads exactly from 16 up). Julia 1.12.6, matching
the Mac, so not a version confound. Hamilton is a shared, general-access server (no
SLURM/job scheduler found — jobs just run directly), not a dedicated/exclusive
compute node, which matters for reading the middle of these tables. Confirmed with
`uptime` right after the sweep: load average 8.18/12.66/6.79 (1/5/15 min) and 8
other logged-in users — real background contention on the same 64 cores, not a
clean/dedicated benchmark environment.

### ct (Connecticut precincts, ~600 nodes)

| threads | M | steps/s | speedup vs 1-thread |
|---|---|---|---|
| 1 | 8 | 12,576 | 1.00x |
| 2 | 8 | 22,432 | 1.78x |
| 4 | 8 | 22,937 | 1.82x |
| 8 | 8 | 22,287 | 1.77x |
| 16 | 16 | 33,045 | 2.63x |
| 32 | 32 | 46,230 | 3.68x |
| 48 | 48 | 56,651 | 4.50x |
| 64 | 64 | 64,218 | **5.11x** |

### oh (Ohio precincts, 7404 nodes)

| threads | M | steps/s | speedup vs 1-thread |
|---|---|---|---|
| 1 | 8 | 2,058 | 1.00x |
| 2 | 8 | 2,286 | 1.11x |
| 4 | 8 | 2,904 | 1.41x |
| 8 | 8 | 3,156 | 1.53x |
| 16 | 16 | 4,365 | 2.12x |
| 32 | 32 | 6,206 | 3.02x |
| 48 | 48 | 6,969 | 3.39x |
| 64 | 64 | 7,662 | **3.72x** |

### nc (North Carolina precincts, 2650 nodes)

| threads | M | steps/s | speedup vs 1-thread |
|---|---|---|---|
| 1 | 8 | 7,568 | 1.00x |
| 2 | 8 | 12,299 | 1.63x |
| 4 | 8 | 14,423 | 1.91x |
| 8 | 8 | 14,467 | 1.91x |
| 16 | 16 | 22,146 | 2.93x |
| 32 | 32 | 31,499 | 4.16x |
| 48 | 48 | 36,642 | 4.84x |
| 64 | 64 | 49,325 | **6.52x** |

### Hex lattices, 10×10 to 50×50 (100–2,500 nodes, 5 districts fixed)

Same recipe (M=max(nthreads,8)), run across five sizes to isolate graph/district
size as a variable on its own, separate from real-map irregularity. Cells are
steps/s (speedup vs. that size's own 1-thread point).

| threads | hex10 (100 nodes) | hex20 (400) | hex30 (900) | hex40 (1,600) | hex50 (2,500) |
|---|---|---|---|---|---|
| 1 | 102,732 | 37,286 | 13,394 | 5,587 | 3,223 |
| 2 | 146,994 (1.43x) | 52,138 (1.40x) | 21,160 (1.58x) | 8,493 (1.52x) | 3,439 (1.07x) |
| 4 | 156,401 (1.52x) | 58,039 (1.56x) | 23,026 (1.72x) | 9,884 (1.77x) | 4,155 (1.29x) |
| 8 | 150,749 (1.47x) | 58,475 (1.57x) | 20,586 (1.54x) | 9,097 (1.63x) | 4,772 (1.48x) |
| 16 | 272,988 (2.66x) | 87,717 (2.35x) | 32,794 (2.45x) | 12,894 (2.31x) | 5,635 (1.75x) |
| 32 | 367,168 (3.57x) | 116,433 (3.12x) | 43,186 (3.23x) | 15,728 (2.82x) | 7,145 (2.22x) |
| 48 | 427,871 (4.16x) | 146,177 (3.92x) | 51,002 (3.81x) | 18,278 (3.27x) | 7,920 (2.46x) |
| 64 | 536,716 (**5.22x**) | 174,425 (**4.68x**) | 57,644 (**4.30x**) | 19,579 (**3.50x**) | 8,954 (**2.78x**) |

### Reading this

1. **No plateau up to 64 threads, once M scales with the thread count.** Every
   graph — hex at every size, and all three real maps — keeps gaining throughput
   all the way out to 64 threads; there's no sign of an intrinsic ceiling within
   the tested range. This is the main answer to the design question:
   **`ThreadedBackend` productively uses a large single-machine core count, as
   long as M (rung count) is set high enough to give it that many independent
   replicas to schedule.** It is not capped at some small thread count by the
   algorithm itself.
2. **The 2-8 thread region is flat/noisy for `ct`** (22.4k -> 22.9k -> 22.3k,
   essentially no gain across two more doublings), unlike the Mac's smooth ramp
   over the same thread range. Since M=8 is fixed across that whole region (so
   parallelism-to-schedule isn't the limiter there), the likely explanation is
   Hamilton being shared/multi-tenant — contention from other users' processes for
   memory bandwidth/cache, not a property of the algorithm. The Mac numbers (a
   dedicated, single-user machine) are the cleaner signal for *shape*; Hamilton is
   the better signal for *does it keep scaling at high core counts*, and the answer
   there is yes.
3. **`mean_straggler_gap` grows with thread/M count** (ct: 0.007s -> 0.04s; oh:
   0.06s -> 0.26s; hex50: 0.037s -> 0.28s at 64 threads) — expected: more replicas
   means more variance in per-replica round time, and dynamic scheduling absorbs it
   rather than eliminating it. Still small relative to total round wall time
   (`swap_interval` MH steps), so not currently a correctness or efficiency
   concern, just something to watch if M grows much further.
4. **64-thread speedup tracks per-district size more than total node count, and
   degrades noticeably at the high end.** Ranking the six graphs by nodes ÷
   districts: hex10 (20/dist, 5.22x) > hex20 (80/dist, 4.68x) ≈ ct (120/dist,
   5.11x) ≈ hex30 (180/dist, 4.30x) ≈ nc (189/dist, 6.52x) > hex40 (320/dist,
   3.50x) ≈ oh (493/dist, 3.72x) > hex50 (500/dist, **2.78x**, the worst of
   anything tested). It's a loose, noisy correlation, not a clean law — nc's
   6.52x breaks the pattern outright, and Hamilton's shared-machine contention
   (point 2) adds real noise — but the two extremes are suggestive: the biggest
   individual districts (hex50, oh) give the worst speedup, the smallest (hex10)
   the best. Plausible mechanism, tying back to the hotspot profiling below: the
   dominant per-step cost (`potrf!`, a dense Cholesky over the *district-induced*
   subgraph) scales with district size, not graph size, so bigger districts mean
   bigger per-replica dense-linear-algebra working sets competing for memory
   bandwidth/cache when M replicas run concurrently — more of a shared-resource
   problem than a scheduling one, which is consistent with speedup degrading
   rather than throughput plateauing outright.

## Hotspot profiling (serial, `Profile.jl`)

Script: `examples/profile_pt_hotspots.jl`. Always uses `SerialBackend` (never
threaded) so profiler samples attribute to one call stack instead of being split
across `Threads.@spawn`'d replica tasks; prints a flat, count-sorted text table via
stdlib `Profile.print` (no extra dependency). Run on `ct` (small) and `oh` (large),
`M=8`, `swap_interval=200`, `n_rounds=50` (80,000 total MH steps), to see whether
hotspots shift with graph size.

**Same dominant hotspot at both sizes, growing faster than the workload does**:
`potrf!` (LAPACK's dense Cholesky, called from `get_log_spanning_trees`'s
matrix-tree-theorem log-determinant) — self-samples 1,694 (ct) -> 13,142 (oh), a
~7.8x jump for the same 80,000-step workload. `polsby_popper.jl`'s
`set_areas_and_perimeters!` is the other consistent hotspot in both. Both are
**general CycleWalk energy-function costs, not PT-specific**: the profiled call
stack is `run_parallel_tempering! → advance! → advance_replica! →
run_metropolis_hastings! → get_log_energy → get_delta_energy → {get_log_spanning_trees,
get_isoperimetric_scores}` — the same `run_metropolis_hastings!` every sampler
(plain CycleWalk, annealed SMC/AIS, PT) calls per step. Nothing PT-specific (swap
mechanics, heat bath, backend dispatch) showed up as a hotspot; PT just runs this
same per-step cost M times per round, making it more visible in aggregate without
being its cause.

Status of the three follow-ups identified here:

1. `set_areas_and_perimeters!`'s O(N) full-graph scan per changed district —
   **attempted twice on `perf/cholesky-sparse-vs-dense`, both reverted.** A
   link-cut-tree walk (visiting only a district's own nodes, `O(|di|)` instead of
   `O(N)`) came out a wash to slightly worse in direct timing on `oh` (2,073-2,098
   before vs. 1,887-2,034 steps/s after) — the walk's pointer-chasing overhead
   (`expose!`, growing a `Vector` via `push!`/`pop!`) ate the savings from visiting
   fewer nodes; a plain sequential array scan is too cache-friendly to beat that
   way. A combined single pass (all changed districts in one O(N) scan instead of
   one O(N) scan per changed district) was also a wash (2,084-2,111 steps/s) —
   *why*, in hindsight: `potrf!` alone was ~35% of real profiler samples on `oh` at
   the time, so even fully eliminating this scan's redundancy only touched a small
   slice of total cost. **Worth re-benchmarking now** (not yet done): with the
   Cholesky fix below in place, `set_areas_and_perimeters!` accounts for ~11% of
   real work on `nc`'s profile (comparable to `potrf!` alone) — a fix that was
   noise-level against the old baseline might be measurable against the new one.
2. `get_log_spanning_trees`'s dense-vs-sparse Cholesky — **done, see below.**
3. Reusing a scratch buffer for `get_log_spanning_trees`'s per-call `m×m`
   allocation — still open, lower priority than (1) now.

## Dense vs. sparse Cholesky for `get_log_spanning_trees` (done)

Benchmarked before implementing anything (full plan at the top of this repo's
`perf/cholesky-sparse-vs-dense` branch history): dense `cholesky!` (current code)
vs. `SparseArrays`'s CHOLMOD `cholesky` (no new dependency — already reachable),
timing assembly *and* factorization together (not factorization alone, which
undercounts sparse's extra assembly cost), with `BLAS.set_num_threads(1)` matching
how `run_parallel_tempering!` actually runs it. Real induced-subgraph data, not
synthetic random sparse matrices — three different connectivity structures (6-
neighbor hex, 4-neighbor grid, irregular real precinct adjacency: ct/nc/oh) —
**all agree on a crossover around district size m≈150-190**:

| graph | avg district size (m) | sparse/dense ratio |
|---|---|---|
| grid10 | ~32 | 4.14x **slower** |
| hex10 | ~20 | 1.79x slower |
| hex20 | ~80 | 1.72x slower |
| ct | ~120-146 | 1.27x slower |
| nc | ~189 avg (125-286 range) | 1.33x **faster** |
| hex30 | ~180 | 1.77x faster |
| grid30 | ~300 | 2.02x faster |
| hex40 | ~320 | 1.94x faster |
| oh | ~350-594 | 3.40x faster |
| hex50 | ~499 | 3.98x faster |
| grid50 | ~830 | 9.20x faster |

Also checked within a single real map (`oh` at 8/15/25 districts, isolating
district-count from graph identity): 924 avg → 0.135 ratio, 493 avg → 0.302, 295
avg (range down to 120) → 0.538 — same trend, confirming it isn't an artifact of
comparing different maps. (`oh` at 40 districts failed to construct — a population-
balance infeasibility at that district count/seed, unrelated to this question.)

**Implemented** as `DENSE_CHOLESKY_MAX_M = 160` in
`src/measure/energy/log_forest_count.jl`: a plain `if m < DENSE_CHOLESKY_MAX_M`
inside `get_log_spanning_trees`, checked fresh on *every call* using that call's
own `m` — not decided once for the whole graph/run. Two reasons that matters: the
check itself is a single comparison against a value already being computed, so
there's no cost to save by deciding it "in advance"; and district sizes drift
across a run (nodes move between districts on accepted moves) and vary a lot
*within* one map (nc: 125-286 nodes), so a cached decision could go stale as a
district's size crosses the threshold mid-run.

**Correctness**: full suite passes (164/164 PT tests, including "ThreadedBackend
reproduces SerialBackend bitwise"). Added `test/test_log_forest_count.jl`: both
branches checked against an independent reference (`Graphs.jl`'s own
`induced_subgraph` + dense `logdet` — a different code path from CycleWalk's
hand-rolled kernel) on both a connected and a disconnected district, on both sides
of the threshold.

**Real measured win** (serial backend, same 80,000-step workload used
throughout this document, before/after via `git stash`):

| graph | before | after | speedup |
|---|---|---|---|
| oh | 2,133 steps/s | 3,448 steps/s | **1.62x** |
| hex50 | 3,183 | 5,284 | **1.66x** |
| nc | 7,467 | 8,484 | **1.14x** (mixed — only districts ≥160 nodes cross over) |
| ct | 12,878 | 12,751 | 0.99x (no regression — every ct district stays on the dense path, as it should) |

Hotspot profile confirms the mechanism directly: `potrf!` (dense LAPACK)
disappears *entirely* from `oh`'s profile post-fix (its smallest district is 260
nodes, comfortably above the threshold) — replaced by cheaper CHOLMOD calls, and
total real (non-idle) profiler samples for the same workload dropped from ~39,000
to ~22,500. `nc`'s profile (mixed district sizes straddling the threshold) shows
`potrf!` **and** CHOLMOD together in the same run, at comparable magnitude to each
other — direct confirmation the per-call dispatch is splitting work correctly
within one run, not just switching wholesale between maps.

## Design implication for task 9 (Distributed backend)

The data argues for **deprioritizing, not abandoning**, task 9. `ThreadedBackend`
already scales productively to 64 cores on one machine (Hamilton), which covers the
realistic range for M (rung count) in most redistricting-analysis use cases — the
design doc's own out-of-scope section already anticipated this ("`Distributed`
suffices at M ≲ 256 with a concurrent gather"). Distributed is still worth having
for the case that genuinely needs it (M or per-replica memory beyond one machine,
or spreading across a real multi-node cluster), but it is not fixing a scaling
problem `ThreadedBackend` actually has — the earlier concern that motivated
building it *right after* the threaded backend (roadmap task ordering) is not borne
out by this data. Reasonable to finish task 10 (runner/config/docs) first, as
already decided, and revisit task 9 with a concrete M/cluster-size target in mind
rather than as a default next step.

One thing worth re-checking if this comes up again: the shrinking relative speedup
at large district sizes (hex50: 2.78x) is still a *positive* speedup, not a
plateau — nothing here argues Distributed would help where Threaded doesn't, since
the bottleneck (dense per-district linear algebra contending for memory
bandwidth/cache) would burden separate machines too, just without the shared-cache
contention. It's a reason to look at the energy-function hotspots (see below)
before reaching for Distributed on a very-large-district map, not a reason to
build Distributed sooner.

One methodology note to carry into task 10's defaults: **pick `n_rungs` with the
target core count in mind**, not a fixed default like 8 — the config's
`backend = "threaded"` default is only as good as M lets it be.
