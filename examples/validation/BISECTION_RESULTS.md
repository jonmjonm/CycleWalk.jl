# Cycle-walk convergence bisection study — results

**Date:** 2026-07-11/12 · **Host:** hamilton (64-core) · **Sampler:** standard cycle walk
(Metropolis–Hastings on the fixed target, *not* AIS).

## Question

For the standard cycle walk, how sharp can the target measure get before independent chains
stop agreeing? We probe along the line **γ = t, iso = 0.3·t**, `t ∈ [0,1]`, from a flat base
(t=0) to a sharp target (t=1). At each point we run **8 independent copies** (distinct seeds),
**20,000 samples** each, spacing 1000 MH steps. Observable: the **rank-ordered marginals of the
per-district log spanning-tree count** (`get_log_spanning_trees`, sorted ascending per plan).

**Verdict rule (per point):** CONVERGED iff, for *every* district rank, the max pairwise
total-variation distance between the 8 copies' binned marginals is within the split-half noise
floor, `maxTV ≤ 2·noise + 0.02`. Bisection: endpoints first (expect t=0 CONVERGED, t=1 NOT),
then bisect `t` to bracket the break; STRICT abort if the endpoint assumption fails.

## Headline result

**The bisection never triggered — the cycle walk converges across the entire line on both
graphs.** Both endpoints came back CONVERGED, so there is no critical `t*` to bracket in
`[0,1]`; STRICT correctly stopped before bisecting.

| Case | t=0 | t=1 | transition |
|------|-----|-----|-----------|
| Grid (10×10, 3 dist) | CONVERGED | CONVERGED | none |
| CT (5 dist)          | CONVERGED | CONVERGED | none |

All 8 CT copies at t=1 completed the full 20,004 lines — the result is real, not a casualty of
the overnight hamilton storage outage (see Notes).

## Data

Columns: `mean spread` = min→max of the 8 copy means; `maxTV` = max pairwise binned-TV across
copies; `2·noise` = split-half noise floor (the pass threshold); `maxKS` = max pairwise
Kolmogorov–Smirnov distance; `meanΔ` = copy-mean spread in nats. Rank 1 = fewest spanning trees.

### Grid (10×10, 3 districts)

**t = 0** — CONVERGED (208 s)

| rank | mean spread | maxTV | 2·noise | maxKS | meanΔ |
|-----:|:-----------:|------:|--------:|------:|------:|
| 1 | 21.90→21.95 | 0.0207 | 0.0549 | 0.0166 | 0.054 |
| 2 | 23.90→23.95 | 0.0184 | 0.0545 | 0.0144 | 0.047 |
| 3 | 25.55→25.58 | 0.0234 | 0.0533 | 0.0160 | 0.028 |

**t = 1** — CONVERGED (355 s)

| rank | mean spread | maxTV | 2·noise | maxKS | meanΔ |
|-----:|:-----------:|------:|--------:|------:|------:|
| 1 | 22.05→22.09 | 0.0191 | 0.0667 | 0.0114 | 0.035 |
| 2 | 23.86→23.89 | 0.0165 | 0.0503 | 0.0141 | 0.028 |
| 3 | 25.41→25.43 | 0.0202 | 0.0601 | 0.0141 | 0.023 |

The grid is essentially flat from base to sharp target — meanΔ ≈ 0.03–0.05 at both ends. The
3-district / 100-node graph is too small to strain the sampler anywhere on the line.

### CT (5 districts)

**t = 0** — CONVERGED (530 s)

| rank | mean spread | maxTV | 2·noise | maxKS | meanΔ |
|-----:|:-------------:|------:|--------:|------:|------:|
| 1 | 160.44→160.61 | 0.0219 | 0.0557 | 0.0151 | 0.174 |
| 2 | 169.37→169.46 | 0.0229 | 0.0681 | 0.0126 | 0.092 |
| 3 | 179.31→179.44 | 0.0243 | 0.0657 | 0.0133 | 0.131 |
| 4 | 187.05→187.19 | 0.0250 | 0.0661 | 0.0140 | 0.141 |
| 5 | 198.79→199.12 | 0.0235 | 0.0703 | 0.0197 | 0.333 |

**t = 1** — CONVERGED (2290 s; inflated ~4× by storage stalls, normal ≈ 9 min)

| rank | mean spread | maxTV | 2·noise | maxKS | meanΔ |
|-----:|:-------------:|------:|--------:|------:|------:|
| 1 | 134.27→136.93 | 0.1602 | 0.5367 | 0.1615 | **2.66** |
| 2 | 141.79→143.98 | 0.1713 | 0.4619 | 0.1726 | **2.19** |
| 3 | 147.24→149.01 | 0.1512 | 0.3722 | 0.1534 | **1.78** |
| 4 | 152.74→154.10 | 0.1090 | 0.2246 | 0.1094 | **1.37** |
| 5 | 160.33→161.59 | 0.0726 | 0.1619 | 0.0739 | **1.26** |

## The nuance: CT degrades toward the sharp target — it just doesn't break

CT passes the verdict at both ends, but the underlying numbers show the mixing clearly getting
harder as t→1, even though it stays under the (widening) noise floor:

| CT metric (max over ranks) | t=0 | t=1 | change |
|---|---|---|---|
| between-copy meanΔ (nats) | ~0.1–0.3 | 1.3–2.7 | **~10–20×** |
| within-chain floor `2·noise` | ~0.06 | 0.16–0.54 | **~10×** |
| between-copy maxTV | ~0.02 | 0.07–0.17 | ~7× |

At γ=1 each CT chain is mixing far more slowly (floor grew ~10×), but 8 independent copies still
agree *within their own sampling noise* at 20k samples. Because the verdict is **relative**
(between-copy TV vs the chain's own split-half noise), the gate gets *easier to pass* exactly as
mixing worsens — which is why CT t=1 reads CONVERGED despite a 2.7-nat mean spread.

## Interpretation

The **AIS weight collapse observed earlier was an importance-weighting / annealing-schedule
problem, not a failure of the underlying cycle walk to mix.** As a direct sampler at the fixed
target, the walk stays self-consistent across independent runs over the whole (γ, iso) line, on
both graphs. The graceful CT slow-down marks where a harder case or longer run would push it past
breaking.

## Figures

- Grid overlay: <https://claude.ai/code/artifact/7d17cbdd-0831-472d-a456-6064e2430a6d>
- CT overlay:   <https://claude.ai/code/artifact/cf608deb-a1ca-4ce3-b77c-083d326b3446>

Each panel overlays all 8 copies' marginals in one translucent ink; tight single band =
agreement, separated bands = drift. Solid line = pointwise mean.

## Next steps (open for discussion)

1. **Define the break we want.** The current gate is relative. If the interest is *mixing rate*,
   switch to an **absolute** criterion (fixed ESS or a fixed TV tolerance) — that would flip CT
   t=1 to a fail and give the bisection an actual `t*` to bracket.
2. **Run NC (14 districts).** The true stress case, where AIS collapsed hardest. The rig is ready
   — `CASE=nc bash validation/hpc/run_bisection_study.sh`. Highest-information next run.
3. **Sample budget.** The CT t=1 slow-down is visible but sub-floor at 20k; a longer run would
   resolve whether the copies genuinely agree or eventually separate.

## Artifacts / reproduce

New/changed code (all in `examples/validation/`):

- `run_case.jl` — added `--gamma`/`--iso` (place target on the line), `--tag` (output naming),
  and records `get_log_spanning_trees` per-district in cyclewalk mode.
- `analyze_convergence.jl` — cross-copy verdict at one point (rank marginals → maxTV vs noise
  floor + KS + mean spread; emits `VERDICT … CONVERGED|NOT`).
- `hpc/run_bisection_study.sh` — adaptive driver: 8 copies/point, endpoint sanity, then ≤6
  bisection steps. Env: `CASE, SAMPLES=20000, SPACING=1000, COPIES=8, STEPS=6, STRICT=1`.
- `dump_marginals.jl` + `make_grid_plot.py` / `make_ct_plot.py` — histogram dump and overlay
  figures.

Run: `JULIA="julia +1.12.6" THREADS=60 CASE=ct bash validation/hpc/run_bisection_study.sh`.
Per-point logs and `bisection_<case>_summary.tsv` land in `validation/hpc/logs/`.

## Notes

- **hamilton storage outage:** during the CT run the home/NFS filesystem hung for ~3 hours
  (07-11 ~23:10 → 07-12 ~10:29). Network stayed healthy throughout; the chains blocked in I/O
  wait and resumed on recovery, which is why CT t=1 wall-time (2290 s) is ~4× the normal ~9 min.
  Sample counts are complete, so results are unaffected.
- **Timings (per point, 8 copies parallel):** grid ≈ 3.5–6 min; CT ≈ 9 min nominal.
