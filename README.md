# Photon Avalanche kMC

This repository contains the five-level Tm avalanche kMC workflow used to
calibrate DRE rates, generate production sweeps, and post-process completed
runs into percolation and trajectory visualizations.

## What lives here

| File | Purpose |
| --- | --- |
| `dre_kmc_rate_calibration.py` | Converts Table S3 DRE constants into pairwise kMC/NPMC rates and compares them with NanoParticleTools spectral-kinetics values. |
| `tm_dre_5level_kmc_production.py` | Builds a production sweep, writes per-power `np.sqlite` / `initial_state.sqlite`, runs NPMC, and summarizes each power point. |
| `tm_dre_5level_kmc_percolation.py` | Replays completed runs and computes the percolation-style order parameter, susceptibility, and derivative plots. |
| `tm_dre_5level_kmc_trajectory_3d.py` | Visualizes one trajectory, generates a GIF, and summarizes cluster statistics. |
| `simulate_kmc_production.sh` | Example shell wrapper for a hardcoded production run. |
| `table_s3_4p5_0nN.json` | DRE/Table S3 parameter set used by the calibration and production scripts. |

## Requirements

- Python 3.10 or newer.
- Python packages: `numpy`, `scipy`, `matplotlib`, and `pillow` for GIF export.
- A sibling checkout of `NanoParticleTools` at `../NanoParticleTools/src` relative to this repo, or an equivalent source path on `PYTHONPATH`.
- An NPMC binary for production runs. The default path in `tm_dre_5level_kmc_production.py` is local and usually needs to be overridden with `--npmc-command`.

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

### 1. Inspect or compare the calibration

```bash
python dre_kmc_rate_calibration.py \
  --params table_s3_4p5_0nN.json \
  --np-db /path/to/np.sqlite
```

Add `--json` or `--json-out report.json` if you want machine-readable output.

### 2. Run a production sweep

```bash
python tm_dre_5level_kmc_production.py \
  --interaction-mode calibrated \
  --output-root run1 \
  --npmc-command /path/to/NPMC \
  --power-sampling-mode centered-gaussian \
  --power-center 1.1e4 \
  --power-min 3.0e3 \
  --power-max 3.0e4 \
  --power-count 20 \
  --num-sims 8 \
  --thread-count 8
```

Use `--dry-run` first if you only want the databases and manifests.

The helper script `simulate_kmc_production.sh` is a convenience wrapper, but
the `NPMC` path in `tm_dre_5level_kmc_production.py` may need to be updated for
your machine.

### 3. Analyze a completed sweep

```bash
python tm_dre_5level_kmc_percolation.py run1
```

If you already have a summary JSON, replot it with:

```bash
python tm_dre_5level_kmc_percolation.py \
  --summary-input run1/percolation_order_parameter_susceptibility_n4_plus_n5.json
```

### 4. Visualize one trajectory

```bash
python tm_dre_5level_kmc_trajectory_3d.py run1/power_00_3000
```

Use `--seed all` to overlay every seed, or `--gif-output` / `--summary-output`
to customize filenames.

## What gets written

- Production root: `dre_5level_production_config.json`, `dre_5level_power_sweep_summary.json`, `dre_5level_avalanche_curve.png`, and `generated_geometry/` if geometry is created on the fly.
- Each power directory: `np.sqlite`, `initial_state.sqlite`, `dre_5level_interaction_manifest.json`, `dre_5level_run_summary.json`, plus `stdout` and `stderr` from the NPMC run.
- Percolation analysis: a main order-parameter PNG, companion fragment/derivative plots, and a JSON summary.
- Trajectory analysis: an overview PNG, a criticality GIF, a cluster-size PNG, and a JSON summary.

## Version control

Commit the source files and parameter JSON. Leave generated `run*/` directories,
temporary logs, and other large outputs out of GitHub.

## Troubleshooting

- `No module named NanoParticleTools`: clone `NanoParticleTools` next to this repo or install it so `../NanoParticleTools/src` is visible.
- `NPMC` not found: pass `--npmc-command /path/to/NPMC` to the production script.
- `Could not resolve host: github.com`: that is a network or DNS problem, not a Git authentication problem.
