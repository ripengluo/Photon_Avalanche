# Photon Avalanche kMC

This repository contains the Tm avalanche kMC workflow used to generate
NanoParticleTools-based production sweeps and post-process completed runs into
percolation and trajectory visualizations.

## What lives here

| File | Purpose |
| --- | --- |
| `tm_npt_rates.py` | Shared NPT-based rate-generation helpers used by the production workflow. |
| `tm_npt_kmc_production.py` | Builds a production sweep, writes per-power `np.sqlite` / `initial_state.sqlite`, runs NPMC, and summarizes each power point. |
| `tm_npt_kmc_percolation.py` | Replays completed runs and computes the percolation-style order parameter, susceptibility, and derivative plots. |
| `tm_npt_kmc_trajectory_3d.py` | Visualizes one trajectory, generates a GIF, and summarizes cluster statistics. |
| `simulate_kmc_production.sh` | Example shell wrapper for a hardcoded production run. |
| `SK_input.json` | Geometry defaults, spectral-kinetics defaults, and NPT production defaults. |

## Requirements

- Python 3.10 or newer.
- Python packages: `numpy`, `scipy`, `matplotlib`, and `pillow` for GIF export.
- A sibling checkout of `NanoParticleTools` at `../NanoParticleTools/src` relative to this repo, or an equivalent source path on `PYTHONPATH`.
- An NPMC binary for production runs. The default path in `tm_npt_kmc_production.py` is local and usually needs to be overridden with `--npmc-command`.

## Recommended checkout layout

```text
project_UCNP/
  Avalanche_kmc/
  NanoParticleTools/
```

The scripts automatically add `../NanoParticleTools/src` to `sys.path` when it
exists.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pillow
```

If you keep `NanoParticleTools` as a sibling checkout, you can also install it
in editable mode:

```bash
pip install -e ../NanoParticleTools
```

## Common workflows

### 1. Run a production sweep

```bash
python tm_npt_kmc_production.py \
 --output-root run1 \
 --npmc-command /path/to/NPMC \
 --sigma-esa-scale 600 \
  --s12-scale 30 \
  --power-sampling-mode centered-gaussian \
  --power-center 1.1e4 \
  --power-min 3.0e3 \
  --power-max 3.0e4 \
  --power-count 20 \
  --num-sims 8 \
  --thread-count 8
```

Use `--dry-run` first if you only want the databases and manifests.

Production sweeps always use the NanoParticleTools rate model. The main knobs
for matching prior behavior are the fixed empirical scales such as
`--sigma-esa-scale`, `--s12-scale`, and `--em-scale`.

The helper script `simulate_kmc_production.sh` is a convenience wrapper, but
the `NPMC` path in `tm_npt_kmc_production.py` may need to be updated for
your machine.

### 2. Run an adaptive two-stage sweep

`--workflow-mode adaptive-two-stage` runs a coarse log-spaced pilot scan,
detects the avalanche transition from the steepest log-log slope between
adjacent pilot points, then runs a denser center-weighted refinement grid
around that bracket. Refinement points use an adaptive terminal-block
convergence test: unconverged runs are resumed from NPMC checkpoints in
`--checkpoint-extension-steps` increments up to `--max-step-cutoff`.

Example production command (`em_scale=1.0`):

```bash
python -B ./tm_npt_kmc_production.py \
  --workflow-mode adaptive-two-stage \
  --npmc-command /path/to/NPMC \
  --core-radius-a 50 \
  --shell-thickness-a 25 \
  --tm-fraction 0.08 \
  --surface-quench-mode outer_layer \
  --sigma-esa-scale 1 \
  --q21-scale 1 \
  --s54-scale 1 \
  --s45-scale 1 \
  --s12-scale 1 \
  --em-mode all \
  --em-scale 1.0 \
  --pilot-power-min 3000 \
  --pilot-power-max 50000 \
  --pilot-power-count 12 \
  --pilot-step-cutoff 10000000 \
  --pilot-num-sims 8 \
  --refine-power-count 12 \
  --refine-num-sims 16 \
  --checkpoint-extension-steps 5000000 \
  --max-step-cutoff 100000000 \
  --convergence-block-count 4 \
  --convergence-min-events-per-block 200 \
  --convergence-min-block-time-s 0.1 \
  --convergence-relative-drift 0.10 \
  --convergence-poisson-z 3.0 \
  --convergence-required-passes 2 \
  --num-sims 8 \
  --thread-count 8 \
  --output-root r50-8p0-EM1-adaptive
```

How the convergence test works (algorithm `terminal-blocks-v2`): for each
seed, the analyzer walks backward from the final physical time and builds
`--convergence-block-count` (even, at least 4) non-overlapping terminal
blocks, each spanning at least `--convergence-min-block-time-s` and
containing at least `--convergence-min-events-per-block` Rad-800 events. The
older and newer block halves must agree within
`--convergence-relative-drift` and, for the rate observable, within the
Poisson z limit `--convergence-poisson-z`. A run is converged only after
every seed passes `--convergence-required-passes` consecutive checkpoints.
Whole-run averages and low-count auto-acceptance are never used.

Every checkpoint statistic comes from index-only range probes against a
covering SQLite index on `trajectories(seed, interaction_id, time, step)`,
created lazily on first use (a one-time cost on large databases) and recorded
in `adaptive_run_state.json`. Per-checkpoint analysis cost is therefore
bounded by the terminal window size, not the trajectory length — no full
table scans and no trajectory replay per checkpoint. The N4 population
observable is computed once at finalization by streaming the full trajectory
in append (`rowid`) order with independent per-seed state. This avoids a
production-scale SQLite sort even when seed extension chunks are interleaved.
N4 is reported under `validation` in the summary and never gates the stopping
decision. Consequently `--convergence-observables` must include at least one
rate stopping observable (`rad800` or `rad700`); `n4` alone is rejected.

`--convergence-semantics` records how the result may be read: `branch`
(default) means each seed converges in its own basin and the aggregate is not
an equilibrium average (a mixed-basin warning is emitted when per-seed
terminal rates span two or more decades); `equilibrium` requires
`--convergence-mode-threshold` so dark/bright basins are explicit, observation
of both basins in the terminal blocks, and at least
`--convergence-min-switches` transitions. A stationary single-basin seed may
pass `branch` convergence but is censored under `equilibrium` semantics.

Per-run convergence statuses recorded in `npt_run_summary.json`:

- `converged`: all seeds passed the required consecutive checkpoints.
- `capped`: `--max-step-cutoff` was reached without full convergence.
- `insufficient_history`: the trajectory is too short to build the blocks.
- `insufficient_counts`: history is long enough but Rad-800 counts stay below
  the block threshold; a Poisson upper rate bound is reported instead.
- `metastable_censored`: under `equilibrium` semantics, both basins were not
  observed or fewer than `--convergence-min-switches` transitions occurred.
- `failed`: the power point raised an exception (recorded in the manifest;
  re-run the same command to retry).

Summaries carry `schema_version: 2` with explicit `whole_run` and
`terminal_estimate` sections; the legacy top-level proxy keys are unchanged
whole-run values, and plotting prefers `terminal_estimate`. N4 validation
results appear under `validation.n4`. Each stage writes its own summary
(`npt_run_summary_pilot.json` / `npt_run_summary_refinement.json`); the
canonical `npt_run_summary.json` mirrors the latest stage.

Resume semantics: re-running the exact same command against the same output
root loads `adaptive_sweep_manifest.json`, validates that the physics-affecting
configuration is identical, and continues at the first incomplete stage, power,
or checkpoint extension. Identity is content-based: the parameter file and the
NPMC binary are SHA-256 hashed, the nanoparticle geometry is hashed over its
logical site/species rows, and the interaction manifest is hashed over its
canonical payload — editing any of them mid-workflow is refused with the
mismatched component named. If the physics configuration changed, pick a new
output root. Powers shared by pilot and refinement run only once; refinement
requirements (more seeds/steps) are met by resuming and extending the existing
run rather than rerunning it.

Crash recovery: `adaptive_run_state.json` records a `pending_extension`
before every NPMC call and commits `current_step_cutoff` only after the
database is verified to have reached it. After a SIGTERM (NPMC checkpoints
cleanly) the interrupted extension is rerun idempotently; trajectory rows
beyond the last NPMC checkpoint (the SIGKILL signature) are refused rather
than resumed into duplicate steps. A convergence pass streak increments only
for a strictly newer per-seed checkpoint identity, so re-evaluating the same
data after an interruption never counts a duplicate pass. Failed power points
are retried on the next invocation and their manifest failure records are
marked `resolved` on success; the workflow status is `complete` only when no
unresolved failures remain. The detection bracket/center is computed once
from pilot signals frozen at completion time (needing at least 3 valid pilot
points) and never recomputed on resume, so retries cannot shift the
refinement grid.

Adaptive-mode notes and current limitations:

- Power directories stay flat (`power_<value>`) with a full-precision power ID
  in the manifests; no pilot/refinement subdirectories are created.
- Adaptive mode does not use node-local SQLite staging, so interrupted runs
  never depend on scratch state that is gone after a crash.
- The refinement initial step cutoff is `max(--simulation-length,
  --pilot-step-cutoff)`; `--max-step-cutoff` must not be below it.
- NPMC resume extends every seed of a power point together; per-seed-only
  extension is not supported by the NPMC CLI, so all seeds are resumed.
- If the pilot scan shows no slope above `--transition-min-slope`, refinement
  is skipped unless `--refine-center` is given.

Unit tests for the power selection, transition detection, block statistics,
metastability classification, and resume validation live in
`tests/test_adaptive_sweep.py` and use only tiny synthetic SQLite databases.

### 3. Analyze a completed sweep

```bash
python tm_npt_kmc_percolation.py run1
```

If you already have a summary JSON, replot it with:

```bash
python tm_npt_kmc_percolation.py \
  --summary-input run1/percolation_order_parameter_susceptibility_n4_plus_n5.json
```

### 4. Visualize one trajectory

```bash
python tm_npt_kmc_trajectory_3d.py run1/power_00_3000
```

Use `--seed all` to overlay every seed, or `--gif-output` / `--summary-output`
to customize filenames.

## What gets written

- Production root: `npt_production_config.json`, `npt_power_sweep_summary.json`, `npt_avalanche_curve.png`, and `generated_geometry/` if geometry is created on the fly. Adaptive two-stage runs also write `adaptive_sweep_manifest.json` (stage/resume state) and `npt_avalanche_curve.json` (plotted data).
- Each power directory: `np.sqlite`, `initial_state.sqlite`, `npt_interaction_manifest.json`, `npt_run_summary.json` (plus per-stage `npt_run_summary_<stage>.json` in adaptive mode), plus `stdout` and `stderr` from the NPMC run. Adaptive points also keep `adaptive_run_state.json` (schema 2: pass streaks, counted checkpoint identities, extension history, pending-extension crash marker, and the geometry/manifest content hashes).
- Older archived runs may still contain `dre_5level_*` artifact names, and the trajectory loader still accepts them.
- Percolation analysis: a main order-parameter PNG, companion fragment/derivative plots, and a JSON summary.
- Trajectory analysis: an overview PNG, a criticality GIF, a cluster-size PNG, and a JSON summary.

## Version control

Commit the source files and parameter JSON. Leave generated `run*/` directories,
temporary logs, and other large outputs out of GitHub.

## Troubleshooting

- `No module named NanoParticleTools`: clone `NanoParticleTools` next to this repo or install it so `../NanoParticleTools/src` is visible.
- `NPMC` not found: pass `--npmc-command /path/to/NPMC` to the production script.
- `Could not resolve host: github.com`: that is a network or DNS problem, not a Git authentication problem.
