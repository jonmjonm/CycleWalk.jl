# Parallel tempering profiling notes

Working notes from profiling `ThreadedBackend` scaling, to decide whether the
Distributed backend (roadmap task 9) is actually needed before building it. Updated
as runs complete — treat earlier numbers as superseded when a methodology fix is
noted below them.

**Bottom line**: `ThreadedBackend` scales to at least 64 cores with no sign of a
ceiling, *provided* the rung count (M) is set high enough to match the thread count
— it was never algorithm-limited, an earlier test just had M fixed too low. See
"Design implication for task 9" at the bottom.

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
- Graphs, small to large: `hex` (10x10 hex lattice, ~100 nodes), `ct` (Connecticut
  precincts, ~600 nodes), `nc` (North Carolina precincts, 2650 nodes), `oh` (Ohio
  precincts, 7404 nodes). NC/OH pulled from
  https://quantifyinggerrymandering.pages.oit.duke.edu/codedoc/geographic.html
  (`Geo/Adjacency/{NC_pct21,OHpct20}.json`); NC's JSON has a synthetic `uid`
  (`county_prec_id`) column added locally since `prec_id` alone isn't unique
  across counties and the raw `id` column is an integer (`MultiLevelGraph` needs a
  unique **string** level column).

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

### Reading this

1. **No plateau up to 64 threads, once M scales with the thread count.** Both
   graphs keep gaining throughput all the way out to 64 threads — there's no sign
   of an intrinsic ceiling within the tested range. This is the main answer to the
   design question: **`ThreadedBackend` productively uses a large single-machine
   core count, as long as M (rung count) is set high enough to give it that many
   independent replicas to schedule.** It is not capped at some small thread count
   by the algorithm itself.
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
   0.06s -> 0.26s at 64 threads) — expected: more replicas means more variance in
   per-replica round time, and dynamic scheduling absorbs it rather than
   eliminating it. Still small relative to total round wall time (`swap_interval`
   MH steps), so not currently a correctness or efficiency concern, just something
   to watch if M grows much further.

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
out by this data. Task 10 (runner/config/docs) has since shipped as
`examples/run_pt_toml.jl` / `docs/run_pt_toml.md`; task 9 remains deprioritized per
the data above, to be revisited only with a concrete M/cluster-size target in mind,
not as a default next step.

The methodology note this data motivated: **pick `n_rungs` with the target core
count in mind**, not a fixed default like 8. `run_pt_toml.md` ships `n_rungs = 8` as
its default (a reasonable smoke-test/fallback value, not a scaling recommendation)
and separately documents this in its own "Sizing `n_rungs` to your machine"
section — raise it to at least your thread count for a production run.
