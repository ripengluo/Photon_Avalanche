# Production workflow: fixed-power preparation for n4 convergence

## Objective

Use one self-contained production entry point, `tm_npt_kmc_production.py`, to
improve the initial excited-state population before each target-power run. The
workflow must generate its own geometry and use only configuration and scripts
stored in this folder; it does not require an earlier run directory.

## Smooth preparation schedule

Preparation always runs at 50k W/cm2. The number of preparation steps is a
deliberately approximate choice intended to provide a better target starting
state rather than an optimized one. A smooth, monotonic, endpoint-constrained
saturating curve was fitted to the original estimates. It gives zero steps at
and below 10k, 30M steps at and above 50k, and a calculated value for any power
between them, including homogeneous and centered-Gaussian sweep points.

| Default centered-Gaussian target (W/cm2) | Fitted preparation steps at 50k |
| ---: | ---: |
| 3,000 | 0 |
| 4,614.84 | 0 |
| 7,098.92 | 0 |
| 10,390.7 | 0.251M |
| 11,732.9 | 1.740M |
| 12,535.5 | 2.818M |
| 13,000 | 3.476M |
| 14,112.3 | 5.108M |
| 15,504.8 | 7.197M |
| 21,130.5 | 15.075M |
| 32,503.8 | 25.170M |
| 50,000 | 30M |

The target phase defaults to 70M steps for every power. Powers with zero
preparation retain the normal cold-start production path.

## Prepared-power mechanics

For each target with nonzero preparation:

1. Build fresh 50k rate and state databases from the generated geometry.
2. Run all requested seeds to that target's fitted preparation cutoff and
   validate `interrupt_state` and `interrupt_cutoff`.
3. Build fresh target-power rates.
4. Copy each seed's complete preparation state into an independent target
   `initial_state.sqlite`; validate site IDs and degrees of freedom.
5. Run the target trajectories concurrently, one NPMC invocation per seed.
6. Replay and summarize only the target-power trajectories, then merge them
   into the normal `npt_run_summary.json` schema.
7. Preserve target checkpoints so `--resume` extends the target phase without
   repeating preparation.

The per-seed target databases are required because NPMC's `initial_state`
table does not contain a seed column. The shared target `np.sqlite` is linked
into each seed directory to avoid duplicating the rate database.

## Output layout

```text
runN/
  generated_geometry/
  power_XX_<target>/
    preparation/
      np.sqlite
      initial_state.sqlite
      npt_interaction_manifest.json
    restart/
      seed_1000/{np.sqlite, initial_state.sqlite, stdout, stderr}
      ...
    np.sqlite
    npt_interaction_manifest.json
    preparation_protocol.json
    npt_run_summary.json
    n4_trajectory.png
```

## Validation

- Database-only dry run on one prepared target.
- Small real NPMC preparation/target run that verifies checkpoint transplant.
- Resume smoke test that extends all per-seed target checkpoints.
- Confirm target summaries initialize replay from the transplanted state.
- Confirm convergence plotting reads the split per-seed trajectory layout.
- Production validation should start with 50k, then 12.5k, 13k, and 14.1k.

No full 30M/70M production run should be launched as a quick validation check.
