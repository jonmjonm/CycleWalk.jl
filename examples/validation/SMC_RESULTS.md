# Annealed SMC — validation results

_Follow-through on the AIS collapse study (`RESULTS.md`), which recommended replacing
single-path annealed importance sampling with **annealed SMC + adaptive resampling** as
the primary fix (essential for NC). This report covers the SMC sampler and its results._

## What this tests

The annealed-SMC sampler (`src/chains/annealed_smc.jl`, branch `annealed-smc`) evolves a
**population of `N` particles** base→target along the same γ+iso path used by AIS, but
each block it **resamples** when the running ESS degrades and **rejuvenates** with MH
moves at the current measure. Turning one intractable importance weight into a product of
well-conditioned, resampled increments is the textbook cure for the steady weight
degeneration that collapsed AIS on CT/NC (see `RESULTS.md`).

Two schedules are available and differ only in how each next temperature is chosen:

- **`FixedSchedule`** — a precomputed t-grid; resample when `ESS < ess_frac·N`.
- **`AdaptiveTempering`** — choose each t online so ESS drops to a target; resample every step.

Validation reuses the AIS study's method exactly: each plan's per-district isoperimetric
scores are sorted (rank 1 = most compact … rank K = least compact), and each rank's
SMC-vs-cycle-walk total-variation distance is judged against the **split-half noise
floor** (see `RESULTS.md` for the methodology). Runners: `examples/run_asmc_toml.jl`
(TOML config + CLI overrides) and `examples/validation/run_asmc.jl`; comparison via
`analyze.jl --case <c> --smc` against the same `<case>_cyclewalk` baseline.

## Headline: SMC removes the weight collapse that sank AIS

Where single-path AIS gave ESS ~0.1–0.2% on CT/NC, SMC holds **52% (CT)** and
**66–99% (NC)**. The resampling works exactly as predicted — the weight degeneracy is
gone.

## CT (5 districts): validated

`p512 b100 r300` — final ESS 52%, 2 resamples, logZ 292.4. `analyze.jl --smc` **PASSES**:
every rank's SMC-vs-cyclewalk TV sits inside the split-half noise floor. SMC reproduces
the target *marginals*, not merely a healthy ESS.

## NC (14 districts): two hurdles

NC exposes a second hurdle AIS never reached — its weights collapsed first. All SMC runs
compared to the 10,001-sample `nc_cyclewalk` baseline (γ=0.25, iso=0.3); the tabulated
value is the per-rank sorted isoperimetric-score mean (rank 14 = least compact):

| rank | cyclewalk | SMC p512·r300 | SMC p512·r1500 | SMC p1024·r1500 |
|-----:|----------:|--------------:|---------------:|----------------:|
| 1    | 29.5      | 34.8          | 29.9           | 29.9            |
| 8    | 42.4      | 51.2          | 44.4           | 45.7            |
| 14   | 57.3      | 78.6          | 61.0           | 67.2            |
| **ranks within noise** | — | **0 / 14** | **5 / 14** | **1 / 14** |
| logZ | —         | 1157.1        | 1123.5         | 1125.0          |

(Notation: `p` = particles, `b` = t-blocks, `r` = rejuvenation MH steps per block.)

Two levers were tested at fixed schedule, 100 blocks:

- **Rejuvenation (mixing budget).** r300 → r1500 (5×) cut the bias ~5× (rank-14 error
  +21.3 → +3.8; TVs 0.6–0.9 → 0.14–0.32) and moved logZ 1157 → 1124. Every signal
  improved. The means remain systematically **high** (biased toward the less-compact base)
  and the bias **grows with rank** — the fingerprint of particles that lag the anneal and
  keep sprawling, base-like districts the iso penalty has not yet compactified.
- **Particle count (resolution).** Doubling N (512 → 1024) at fixed r1500 did **not** close
  the gap — it **sharpened the failure** (5/14 → 1/14 ranks pass), because more particles
  measure the *same biased distribution* more precisely, so the real bias clears the (now
  tighter) noise floor unambiguously. logZ is flat across N (1123.5 vs 1125.0), confirming
  the population converges to the *same* under-mixed law regardless of N.

## Conclusion: the remaining bottleneck is mixing, not resolution or pacing

The residual NC discrepancy is a **state (mixing) bias**, not estimator variance: it is
insensitive to particle count and sensitive to rejuvenation. Resampling solved the *weight*
problem; reaching the target *marginals* additionally needs enough MH mixing per block for
the particles to track the moving distribution.

- **High ESS is necessary but not sufficient.** ESS 66–99% coexisted with clearly biased
  marginals.
- **`logZ` is the honest bias tell** — inflated when under-mixed, falling toward truth as
  mixing improves (1157 → 1124 with 5× rejuvenation; flat across particle count).
- **Next step for NC: more rejuvenation** (r ≈ 3000–4000), optionally with more particles
  to tighten the (then-correct) estimates — not more particles alone, and not schedule
  repacing (already ruled out in `RESULTS.md`).

## Reproduce / artifacts

```
julia -t <threads> run_asmc_toml.jl toml/param_annealed_smc_nc.toml --rejuv 3000
julia --project=examples validation/analyze.jl --case nc --smc
```

Run atlases and per-run `analyze` tables are archived under
`output/validation/<case>_smc_p<N>_b<blocks>_r<rejuv>_t<threads>[_c<cs>e<ce>].jsonl.gz`
and `validation/hpc/logs_archive/`. The sampler additionally supports post-t=1 sample
amplification (`collect_steps`/`collect_every`, `resample_before_collect`) to emit
`N × collect_steps/collect_every` target samples per run for tighter marginal estimates.
