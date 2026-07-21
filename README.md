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
| `SK_input.json` | SpectralKinetics inputs, geometry defaults, and NPT production defaults. |

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
  --npt-cr-mode exported \
  --sigma-esa-scale 1.0 \
  --beta-s12 0.0023 \
  --s12-scale 1 \
  --power-sampling-mode centered-gaussian \
  --power-center 1.1e4 \
  --power-min 3.0e3 \
  --power-max 3.0e4 \
  --power-count 20 \
  --simulation-length 2000000 \
  --max-simulation-length 10000000 \
  --num-sims 8 \
  --thread-count 8
```

Use `--dry-run` first if you only want the databases and manifests.

Production sweeps always use the NanoParticleTools rate model. The main knobs
for matching prior behavior are the fixed empirical scales such as
`--sigma-esa-scale`, residual `--s12-scale`, and `--em-scale`.

### s12,42 cross-relaxation correction

The PA-loop cross relaxation

```text
3H4 + 3H6 -> 3F4 + 3F4
```

is labeled `s12,42` in the production manifests. The default correction is now
`--beta-s12 0.0023`, stored as `beta_s12_cm` in `SK_input.json`. This replaces
the previous baseline use of `--s12-scale 30` with a channel-specific
multiphonon beta for this one CR path; `--s12-scale` remains available as a
residual multiplier and defaults to `1`.

The physical interpretation is a Stark- and phonon-sideband-resolved overlap
between the donor emission and acceptor absorption bands:

```text
J_s12 = sum_{i,j,k,l,m} p_i |V_{ijkl}|^2 S_m
        integral E_3H4_i->3F4_k(lambda)
                 A_3H6_j->3F4_l(lambda - m hbar omega_ph) d lambda
```

where `E_3H4->3F4` is the donor emission line shape,
`A_3H6->3F4` is the acceptor absorption line shape, `p_i` is the thermal
population of the donor Stark sublevel, and `S_m` weights the `m`-phonon
sideband. In the current reduced NPT workflow this detailed overlap is
represented by

```text
K_s12(corrected) = K_s12(NPT) exp[(mpr_beta - beta_s12) |Delta_eff|]
```

with `Delta_eff` computed from the NPT multiplet energies plus the configured
Stokes shift. The default `beta_s12 = 0.0023 cm` is therefore a spectral-overlap
surrogate for the `3H4 -> 3F4` / `3H6 -> 3F4` CR pair, not a global MPR change.

### Inhomogeneous simulation length

Step-cutoff runs can assign longer trajectories near the PA build-up region and
shorter trajectories at the low/high power limits. Set `--max-simulation-length`
to enable the centered-Gaussian scheduler. In this mode `--simulation-length`
is the floor and `--max-simulation-length` is the central value:

```text
L(P) = L_floor + (L_max - L_floor)
       exp[-0.5 ((log10(P) - log10(P_center)) / sigma_decades)^2]
```

`P_center` and `sigma_decades` reuse `--power-center` and
`--power-gaussian-sigma-decades`, so the length profile follows the same center
as `--power-sampling-mode centered-gaussian`. If `--max-simulation-length` is
omitted, all powers use the homogeneous `--simulation-length` cutoff. The exact
per-power cutoffs are written to `npt_production_config.json` and each power
manifest.

The helper script `simulate_kmc_production.sh` is a convenience wrapper, but
the `NPMC` path in `tm_npt_kmc_production.py` may need to be updated for
your machine.

### 2. Analyze a completed sweep

```bash
python tm_npt_kmc_percolation.py run1
```

If you already have a summary JSON, replot it with:

```bash
python tm_npt_kmc_percolation.py \
  --summary-input run1/percolation_order_parameter_susceptibility_n4_plus_n5.json
```

### 3. Visualize one trajectory

```bash
python tm_npt_kmc_trajectory_3d.py run1/power_00_3000
```

Use `--seed all` to overlay every seed, or `--gif-output` / `--summary-output`
to customize filenames.

## What gets written

- Production root: `npt_production_config.json`, `npt_power_sweep_summary.json`, `npt_avalanche_curve.png`, and `generated_geometry/` if geometry is created on the fly.
- Each power directory: `np.sqlite`, `initial_state.sqlite`, `npt_interaction_manifest.json`, `npt_run_summary.json`, `npt_emission_spectrum.json`, `npt_emission_spectrum.png`, plus `stdout` and `stderr` from the NPMC run.
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
