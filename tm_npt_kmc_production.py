"""Production runner for the Tm avalanche kMC model built on NPT defaults."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import matplotlib
import numpy as np

# This is a headless batch runner, and power points may finalize on worker
# threads. GUI backends such as Tk can abort when destroyed off the main
# thread, so select the non-interactive backend before importing pyplot.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import tm_npt_rates as rates
from NanoParticleTools.analysis.util import get_spectrum_wavelength_from_dndt
from NanoParticleTools.core import NPMCInput
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle, SphericalConstraint


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NPMC_COMMAND = "/home/rpluo/Desktop/project_MFML_UCNP/RNMC/build/NPMC"
FALLBACK_NPMC_COMMAND = "/global/home/users/rluo/project_UCNP/RNMC/build/NPMC"
DEFAULT_POWER_MIN = 3.0e3
DEFAULT_POWER_MAX = 3.0e4
DEFAULT_POWER_COUNT = 8
DEFAULT_SIMULATION_LENGTH = 2000000
DEFAULT_POWER_SAMPLING_MODE = "homogeneous"
DEFAULT_POWER_GAUSSIAN_CENTER = 1.0e4
DEFAULT_POWER_GAUSSIAN_SIGMA_DECADES = 0.18
DEFAULT_TRAJECTORY_ARCHIVE_ROOT = Path(
    "/home/rpluo/Desktop/hdd_large/KMC_trajectories/Tm_4p5-NPT"
)
SURFACE_LAYER_THICKNESS_NM = 0.5
SURFACE_LAYER_THICKNESS_A = SURFACE_LAYER_THICKNESS_NM * 10
DEFAULT_SURFACE_QUENCH_MODE = "off"
DEFAULT_SURFACE_SPECIES = "Surface"
DEFAULT_SURFACE_FRACTION = 0.20
TM_SPECIES_ID = 0
SURFACE_SPECIES_ID = 1

Q21_CHANNEL_NAME = "Q21,24"
S12_CHANNEL_NAME = "s12,42"
S54_CHANNEL_NAME = "s54,23"
S45_CHANNEL_NAME = "s45,32"
N4_LEVEL = 3

# ---------------------------------------------------------------------------
# Adaptive two-stage workflow and terminal-block convergence defaults.
# Every adaptive/convergence default lives here as a named constant instead
# of an unexplained literal inside a function (see r50-8p0-EM0p05/AGENT.md).
# ---------------------------------------------------------------------------
DEFAULT_WORKFLOW_MODE = "single-stage"

# Stage 1 pilot scan. In this project "homogeneous" means homogeneous in
# log10(power), i.e. a geometric sweep, consistent with np.geomspace.
DEFAULT_PILOT_POWER_MIN = 3.0e3
DEFAULT_PILOT_POWER_MAX = 5.0e4
DEFAULT_PILOT_POWER_COUNT = 12
DEFAULT_PILOT_STEP_CUTOFF = 10_000_000
DEFAULT_PILOT_NUM_SIMS = 8
# Fraction of each pilot trajectory's terminal physical time used for the
# transition-detection observables. Recorded in every pilot summary.
DEFAULT_PILOT_TERMINAL_FRACTION = 0.5

# Stage 2 transition detection and center-weighted refinement.
DEFAULT_REFINE_POWER_COUNT = 12
DEFAULT_REFINE_HALF_WIDTH_DECADES = 0.12
DEFAULT_REFINE_MIN_POWER_GAP_FRACTION = 0.005
DEFAULT_REFINE_NUM_SIMS = 16
# Minimum local log-log slope of the 800 nm signal that counts as a
# photon-avalanche transition. Flatter maxima do not claim a transition.
DEFAULT_TRANSITION_MIN_SLOPE = 2.0

# Minimum valid pilot points required before transition detection runs.
MIN_DETECTION_PILOT_POINTS = 3

# Terminal-block convergence ("terminal-blocks" mode, steps cutoff only).
DEFAULT_CHECKPOINT_EXTENSION_STEPS = 5_000_000
DEFAULT_MAX_STEP_CUTOFF = 100_000_000
DEFAULT_CONVERGENCE_BLOCK_COUNT = 4
DEFAULT_CONVERGENCE_MIN_EVENTS_PER_BLOCK = 200
DEFAULT_CONVERGENCE_MIN_BLOCK_TIME_S = 0.1
DEFAULT_CONVERGENCE_RELATIVE_DRIFT = 0.10
DEFAULT_CONVERGENCE_POISSON_Z = 3.0
DEFAULT_CONVERGENCE_REQUIRED_PASSES = 2
DEFAULT_CONVERGENCE_OBSERVABLES = "rad800,n4"
DEFAULT_CONVERGENCE_MIN_SWITCHES = 2
# Metastability interpretation when no dark/bright threshold is supplied:
# "branch" convergence means each seed reached a locally stationary dark or
# bright branch; it is NOT an equilibrium basin-weighted statement.
# "equilibrium" additionally requires --convergence-mode-threshold and the
# observed switching minimum, censoring seeds that never mix.
DEFAULT_CONVERGENCE_SEMANTICS = "branch"
CONVERGENCE_SEMANTICS_CHOICES = ("branch", "equilibrium")
MATPLOTLIB_LOCK = threading.RLock()
# Per-seed terminal rad800 rates spanning at least this ratio trigger a
# mixed-basin heterogeneity warning (2 decades).
SEED_HETEROGENEITY_RATIO = 100.0

# Documented internal floors (not CLI knobs). They only regularize division
# and log transforms; they never decide which data is analyzed.
CONVERGENCE_RATE_FLOOR_PER_S = 1.0e-12
CONVERGENCE_POPULATION_FLOOR = 1.0e-12
# terminal-blocks-v2: stopping decisions use index-only Rad-band terminal
# block statistics (no trajectory replay per checkpoint); N4 is computed
# once at finalization and reported as a validation observable.
CONVERGENCE_ALGORITHM_ID = "terminal-blocks-v2"
# Covering index that makes every convergence query index-only. Created
# lazily at the first analysis and recorded in the run state; it adds a
# small per-insert cost to NPMC extensions (measured, acceptable).
TRAJECTORY_ANALYSIS_INDEX = "idx_tm_npt_traj_seed_id_time"
SUMMARY_SCHEMA_VERSION = 2
ADAPTIVE_MANIFEST_SCHEMA_VERSION = 2
TM_CHANNEL_TUPLE_TO_NAME = {
    (3, 1, 0, 1): S12_CHANNEL_NAME,
    (1, 0, 1, 3): Q21_CHANNEL_NAME,
    (1, 0, 1, 2): "Q21,23",
    (4, 3, 1, 2): S54_CHANNEL_NAME,
    (2, 1, 3, 4): S45_CHANNEL_NAME,
}

# Table S1 / run1 geometry approximation used previously:
# 4.56% Tm core minor/major axes = 20.7 / 32.5 nm, shell thickness = 5.5 nm.
CORE_MEAN_DIAMETER_NM = (20.7 + 32.5) / 2
CORE_RADIUS_A = CORE_MEAN_DIAMETER_NM * 5
AVERAGE_SHELL_THICKNESS_NM = 5.5
AVERAGE_SHELL_THICKNESS_A = AVERAGE_SHELL_THICKNESS_NM * 10

FALLBACK_PRODUCTION_DEFAULTS = {
    "sigma_esa_scale": 1185.7978647623052,
    "q21_scale": 1.0,
    "s54_scale": 1.0,
    "s45_scale": 1.0,
    "s12_scale": 21.148836746821555,
    "em_mode": "off",
    "em_scale": 1.0,
    "surface_quench_mode": DEFAULT_SURFACE_QUENCH_MODE,
    "surface_species": DEFAULT_SURFACE_SPECIES,
    "surface_fraction": DEFAULT_SURFACE_FRACTION,
}


def json_safe(value: Any) -> Any:
    """Convert NumPy values and tuples into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def format_power_tick(value: float, _pos: int | None = None) -> str:
    if value >= 1000.0:
        scaled = value / 1000.0
        if abs(scaled - round(scaled)) < 1e-8:
            return f"{int(round(scaled))}k"
        text = f"{scaled:.2f}".rstrip("0").rstrip(".")
        return f"{text}k"
    if abs(value - round(value)) < 1e-8:
        return f"{int(round(value))}"
    return f"{value:g}"


def build_centered_gaussian_power_sweep(
    power_min: float,
    power_max: float,
    power_count: int,
    center: float,
    sigma_decades: float,
) -> np.ndarray:
    """Build a deterministic, center-weighted sweep in log10(power) space."""
    if power_count < 2:
        raise ValueError("power_count must be at least 2")
    if power_min <= 0 or power_max <= power_min:
        raise ValueError("power_min must be positive and smaller than power_max")
    if not (power_min <= center <= power_max):
        raise ValueError("power_center must lie within the sampled power range")
    if sigma_decades <= 0:
        raise ValueError("power_gaussian_sigma_decades must be positive")

    x_min = float(np.log10(power_min))
    x_max = float(np.log10(power_max))
    x_center = float(np.log10(center))
    x_dense = np.linspace(x_min, x_max, max(4000, int(power_count) * 200))
    centered = (x_dense - x_center) / float(sigma_decades)
    weights = np.exp(-0.5 * centered**2) + 0.15
    dx = np.diff(x_dense)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (weights[:-1] + weights[1:]) * dx)))
    cdf /= float(cdf[-1])
    quantiles = np.linspace(0.0, 1.0, int(power_count))
    x_samples = np.interp(quantiles, cdf, x_dense)
    powers = np.power(10.0, x_samples)
    powers[0] = float(power_min)
    powers[-1] = float(power_max)
    if power_count >= 3:
        center_index = int(np.argmin(np.abs(powers - float(center))))
        powers[center_index] = float(center)
        powers = np.maximum.accumulate(powers)
    return powers


def parse_power_sweep(args: argparse.Namespace) -> np.ndarray:
    """Parse either explicit powers or a geometric sweep."""
    if args.powers:
        return np.asarray(
            [float(item) for item in args.powers.replace(",", " ").split()],
            dtype=float,
        )
    if args.power_sampling_mode == "centered-gaussian":
        return build_centered_gaussian_power_sweep(
            power_min=float(args.power_min),
            power_max=float(args.power_max),
            power_count=int(args.power_count),
            center=float(args.power_center),
            sigma_decades=float(args.power_gaussian_sigma_decades),
        )
    return np.geomspace(args.power_min, args.power_max, args.power_count)


def stable_power_id(power: float) -> str:
    """Full-precision, collision-safe identifier for one power value."""
    return repr(float(power))


def build_pilot_power_grid(
    power_min: float,
    power_max: float,
    power_count: int,
) -> np.ndarray:
    """Stage-1 pilot grid: geometrically spaced (homogeneous in log power)."""
    if power_count < 3:
        raise ValueError("pilot_power_count must be at least 3 for transition detection")
    if power_min <= 0 or power_max <= power_min:
        raise ValueError("pilot_power_min must be positive and smaller than pilot_power_max")
    return np.geomspace(float(power_min), float(power_max), int(power_count))


def aggregate_pilot_power_signal(
    seed_rates: list[float],
    seed_durations: list[float],
    mode_threshold_log10: float | None = None,
) -> dict[str, Any]:
    """Robust per-power pilot signal from terminal-window per-seed rates.

    Each seed contributes log10(rate + floor) with floor = 1/duration, i.e.
    one resolvable event over the seed's own exposure, so log10(0) never
    occurs. The transition signal is the median over seeds; mean, standard
    deviation, min/max, and the fraction of seeds above an optional mode
    threshold are recorded alongside.
    """
    if len(seed_rates) != len(seed_durations):
        raise ValueError("seed_rates and seed_durations must have the same length")
    log_signals: list[float] = []
    for rate, duration in zip(seed_rates, seed_durations):
        if duration <= 0 or not math.isfinite(duration):
            continue
        floor = 1.0 / float(duration)
        value = math.log10(max(float(rate), 0.0) + floor)
        if math.isfinite(value):
            log_signals.append(value)
    if not log_signals:
        return {
            "n_seeds": 0,
            "median_log10": None,
            "mean_log10": None,
            "std_log10": None,
            "min_log10": None,
            "max_log10": None,
            "fraction_above_mode_threshold": None,
        }
    arr = np.asarray(log_signals, dtype=float)
    fraction_above = None
    if mode_threshold_log10 is not None:
        fraction_above = float(np.mean(arr > float(mode_threshold_log10)))
    return {
        "n_seeds": int(arr.size),
        "median_log10": float(np.median(arr)),
        "mean_log10": float(np.mean(arr)),
        "std_log10": float(np.std(arr)),
        "min_log10": float(np.min(arr)),
        "max_log10": float(np.max(arr)),
        "fraction_above_mode_threshold": fraction_above,
    }


def detect_avalanche_transition(
    powers: Iterable[float],
    median_log10_signals: Iterable[float | None],
    *,
    min_slope: float = DEFAULT_TRANSITION_MIN_SLOPE,
    manual_center: float | None = None,
) -> dict[str, Any]:
    """Detect the photon-avalanche transition from adjacent pilot points.

    For each adjacent pair of valid pilot points the local log-log slope of
    the per-seed-median 800 nm signal is computed. The interval with the
    largest positive finite slope is the primary transition bracket, and the
    default refinement center is the geometric midpoint sqrt(P_i * P_{i+1}).
    """
    if min_slope <= 0 or not math.isfinite(min_slope):
        raise ValueError("transition min_slope must be positive and finite")
    power_list = [float(p) for p in powers]
    signal_list = [
        None if s is None else float(s) for s in median_log10_signals
    ]
    if len(power_list) != len(signal_list):
        raise ValueError("powers and signals must have the same length")
    order = np.argsort(power_list)
    sorted_powers = [power_list[i] for i in order]
    sorted_signals = [signal_list[i] for i in order]
    # Drop non-positive or non-finite intensities (here: missing signals).
    valid = [
        (p, s)
        for p, s in zip(sorted_powers, sorted_signals)
        if s is not None and math.isfinite(s) and p > 0
    ]
    result: dict[str, Any] = {
        "n_pilot_points": len(power_list),
        "n_valid_points": len(valid),
        "min_slope_required": float(min_slope),
        "manual_center": None if manual_center is None else float(manual_center),
        "slopes": [],
        "max_slope": None,
        "bracket_powers": None,
        "geometric_center": None,
        "center": None,
        "edge_detected": False,
        "transition_detected": False,
        "reason": None,
    }
    if manual_center is not None and manual_center <= 0:
        raise ValueError("manual refine center must be positive")
    if len(valid) < 3:
        result["reason"] = "insufficient_pilot_points"
        if manual_center is not None:
            result["center"] = float(manual_center)
        return result

    log_powers = [math.log10(p) for p, _s in valid]
    slopes: list[dict[str, Any]] = []
    for i in range(len(valid) - 1):
        denom = log_powers[i + 1] - log_powers[i]
        slope = float("nan")
        if denom > 0:
            slope = (valid[i + 1][1] - valid[i][1]) / denom
        slopes.append(
            {
                "interval_index": i,
                "power_low": valid[i][0],
                "power_high": valid[i + 1][0],
                "signal_low": valid[i][1],
                "signal_high": valid[i + 1][1],
                "slope": float(slope),
            }
        )
    result["slopes"] = slopes
    finite_slopes = [
        (i, entry["slope"])
        for i, entry in enumerate(slopes)
        if math.isfinite(entry["slope"])
    ]
    if not finite_slopes:
        result["reason"] = "no_finite_slopes"
        if manual_center is not None:
            result["center"] = float(manual_center)
        return result

    best_index, max_slope = max(finite_slopes, key=lambda item: item[1])
    result["max_slope"] = float(max_slope)
    result["edge_detected"] = bool(best_index == 0 or best_index == len(slopes) - 1)
    bracket = (valid[best_index][0], valid[best_index + 1][0])
    result["bracket_powers"] = [float(bracket[0]), float(bracket[1])]
    geometric_center = math.sqrt(bracket[0] * bracket[1])
    result["geometric_center"] = float(geometric_center)

    if max_slope < min_slope:
        result["reason"] = "max_slope_below_threshold"
        # Do not silently invent a threshold: refinement only proceeds when
        # the user supplied an explicit manual center.
        if manual_center is not None:
            result["center"] = float(manual_center)
        return result

    result["transition_detected"] = True
    result["center"] = float(
        manual_center if manual_center is not None else geometric_center
    )
    return result


def build_refinement_power_grid(
    *,
    center: float,
    half_width_decades: float,
    power_count: int,
    pilot_min: float,
    pilot_max: float,
    bracket_powers: tuple[float, float] | list[float] | None,
    min_gap_fraction: float,
    existing_powers: Iterable[float] = (),
) -> dict[str, Any]:
    """Deterministic center-weighted refinement grid in log10(power).

    The grid spans log10(center) +/- half_width_decades, clipped to the
    global pilot range, sampled with the centered-Gaussian quantile sampler.
    Both bracket endpoints and the center are always included. Points closer
    than min_gap_fraction (relative to the smaller power) are merged; a
    merged point that matches an existing (pilot) power adopts that exact
    power value so the pilot result can be reused instead of rerun.
    """
    if power_count < 2:
        raise ValueError("refine_power_count must be at least 2")
    if half_width_decades <= 0:
        raise ValueError("refine_half_width_decades must be positive")
    if not (0.0 < min_gap_fraction < 1.0):
        raise ValueError("refine_min_power_gap_fraction must lie in (0, 1)")
    if pilot_min <= 0 or pilot_max <= pilot_min:
        raise ValueError("pilot range must be positive and non-degenerate")
    if center <= 0 or not math.isfinite(center):
        raise ValueError("refinement center must be positive and finite")

    clipped_center = min(max(float(center), float(pilot_min)), float(pilot_max))
    center_clipped = not math.isclose(
        clipped_center, float(center), rel_tol=0.0, abs_tol=0.0
    )
    x_center = math.log10(clipped_center)
    range_low = max(10.0 ** (x_center - half_width_decades), float(pilot_min))
    range_high = min(10.0 ** (x_center + half_width_decades), float(pilot_max))
    if range_high <= range_low:
        raise ValueError(
            "refinement window collapsed after clipping to the pilot range; "
            "widen the pilot range or choose a different center"
        )

    grid_powers = build_centered_gaussian_power_sweep(
        power_min=range_low,
        power_max=range_high,
        power_count=int(power_count),
        center=clipped_center,
        sigma_decades=float(half_width_decades),
    )

    # Candidate entries: (power, kind). Lower sort rank wins a merge.
    candidates: list[dict[str, Any]] = [
        {"power": float(p), "kind": "grid", "rank": 3} for p in grid_powers
    ]
    if bracket_powers is not None:
        for p in bracket_powers:
            candidates.append({"power": float(p), "kind": "bracket", "rank": 1})
    candidates.append({"power": float(clipped_center), "kind": "center", "rank": 2})

    existing_sorted = sorted(float(p) for p in existing_powers if float(p) > 0)

    def matching_existing(power: float) -> float | None:
        best: float | None = None
        best_gap = float("inf")
        for other in existing_sorted:
            if other <= 0:
                continue
            gap = abs(other - power) / min(other, power)
            if gap < min_gap_fraction and gap < best_gap:
                best = other
                best_gap = gap
        return best

    for entry in candidates:
        entry["reused_power"] = matching_existing(entry["power"])

    candidates.sort(key=lambda entry: entry["power"])
    # Cluster adjacent candidates whose relative gap is below the merge
    # threshold, then keep one representative per cluster. Representatives
    # that match an existing power adopt its exact full-precision value.
    merged: list[dict[str, Any]] = []
    for entry in candidates:
        if merged:
            last = merged[-1]
            gap = (entry["power"] - last["power"]) / last["power"]
            if gap < min_gap_fraction:
                last["members"].append(entry)
                # Prefer a reused (existing) power, then the lowest rank.
                champion = min(
                    last["members"],
                    key=lambda e: (e["reused_power"] is None, e["rank"]),
                )
                last["representative"] = champion
                last["power"] = float(
                    champion["reused_power"]
                    if champion["reused_power"] is not None
                    else champion["power"]
                )
                continue
        merged.append(
            {
                "members": [entry],
                "representative": entry,
                "power": float(
                    entry["reused_power"]
                    if entry["reused_power"] is not None
                    else entry["power"]
                ),
            }
        )

    points = []
    for cluster in merged:
        rep = cluster["representative"]
        points.append(
            {
                "power": float(cluster["power"]),
                "kind": str(rep["kind"]),
                "reused_from_existing": rep["reused_power"] is not None,
                "merged_kinds": sorted({m["kind"] for m in cluster["members"]}),
            }
        )
    points.sort(key=lambda row: row["power"])
    # Exact duplicates after adopting existing values collapse to one point.
    deduped: list[dict[str, Any]] = []
    for point in points:
        if deduped and stable_power_id(deduped[-1]["power"]) == stable_power_id(
            point["power"]
        ):
            deduped[-1]["merged_kinds"] = sorted(
                set(deduped[-1]["merged_kinds"]) | set(point["merged_kinds"])
            )
            deduped[-1]["reused_from_existing"] = bool(
                deduped[-1]["reused_from_existing"] or point["reused_from_existing"]
            )
            continue
        deduped.append(point)

    return {
        "powers": [float(point["power"]) for point in deduped],
        "points": deduped,
        "requested_power_count": int(power_count),
        "center": float(clipped_center),
        "center_clipped_to_pilot_range": bool(center_clipped),
        "range": [float(range_low), float(range_high)],
        "half_width_decades": float(half_width_decades),
        "min_gap_fraction": float(min_gap_fraction),
    }


def resolve_simulation_cutoff(
    args: argparse.Namespace,
) -> tuple[str, int | None, float | None]:
    """Resolve the requested simulation cutoff mode and validate its inputs."""
    mode = args.cutoff_mode
    if mode is None:
        mode = "physical-time" if args.simulation_time is not None else "steps"

    if mode == "steps":
        if args.simulation_time is not None and args.cutoff_mode == "steps":
            raise ValueError(
                "--simulation-time cannot be combined with --cutoff-mode steps; "
                "use --cutoff-mode physical-time instead"
            )
        step_cutoff = int(args.simulation_length)
        if step_cutoff <= 0:
            raise ValueError("--simulation-length must be positive")
        return mode, step_cutoff, None

    if args.simulation_time is None:
        raise ValueError(
            "--cutoff-mode physical-time requires --simulation-time to be set"
        )
    time_cutoff = float(args.simulation_time)
    if time_cutoff <= 0:
        raise ValueError("--simulation-time must be positive")
    return mode, None, time_cutoff


def next_run_dir(parent: Path = SCRIPT_DIR) -> Path:
    """Return the first available runN directory."""
    index = 1
    while (parent / f"run{index}").exists():
        index += 1
    return parent / f"run{index}"


def default_power_parallel_total_slots() -> int | None:
    """Infer the total CPU slots available for parallel power processing."""
    ntasks = os.environ.get("SLURM_NTASKS")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    if ntasks is not None:
        total = int(ntasks)
        if cpus_per_task is not None:
            total *= int(cpus_per_task)
        return total
    if cpus_on_node is not None:
        return int(cpus_on_node)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return int(cpu_count)
    return None


def resolve_npmc_command(npmc_command: str) -> str:
    """Resolve the NPMC binary path with a built-in HPC fallback."""
    requested = Path(npmc_command).expanduser()
    if requested.exists():
        return str(requested)
    if npmc_command == DEFAULT_NPMC_COMMAND:
        fallback = Path(FALLBACK_NPMC_COMMAND)
        if fallback.exists():
            return str(fallback)
    return npmc_command


def resolve_local_db_staging_root() -> Path | None:
    """Return a node-local staging root for heavy SQLite writes when available."""
    override = os.environ.get("KMC_LOCAL_STAGING_ROOT")
    candidate_roots: list[Path] = []
    if override:
        candidate_roots.append(Path(override).expanduser())
    if os.environ.get("SLURM_JOB_ID") is not None:
        candidate_roots.extend(
            Path(raw)
            for raw in (
                "/dev/shm",
                os.environ.get("SLURM_TMPDIR"),
                os.environ.get("TMPDIR"),
                "/tmp",
            )
            if raw
        )
    for candidate in candidate_roots:
        if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
            return candidate.resolve()
    return None


def prepare_power_db_work_dir(
    final_output_dir: Path,
    *,
    local_db_staging_root: Path | None,
    dry_run: bool,
) -> Path:
    """Choose where the per-power SQLite databases should be built and mutated."""
    final_output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run or local_db_staging_root is None:
        return final_output_dir
    staging_parent = local_db_staging_root / "tm_npt_kmc_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{final_output_dir.parent.name}_{final_output_dir.name}_"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=staging_parent))


def move_output_file(source_path: Path, destination_path: Path) -> Path:
    """Move one file into its final output location, replacing any stale file."""
    if not source_path.exists():
        raise FileNotFoundError(f"Missing output artifact: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        return destination_path.resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_symlink() or destination_path.exists():
        destination_path.unlink()
    shutil.move(str(source_path), str(destination_path))
    return destination_path.resolve()


def resolve_source_np_db(
    args: argparse.Namespace,
    params: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
) -> Path:
    """Return an existing source geometry DB or generate one self-contained."""
    if args.source_np_db is not None:
        source_np_db_path = Path(args.source_np_db)
        if not source_np_db_path.exists():
            raise FileNotFoundError(f"Geometry database not found: {source_np_db_path}")
        return source_np_db_path

    geometry_dir = output_root / "generated_geometry"
    source_np_db_path = geometry_dir / "source_geometry_np.sqlite"
    if source_np_db_path.exists() and not args.regenerate_geometry:
        return source_np_db_path

    geometry_dir.mkdir(parents=True, exist_ok=True)
    if source_np_db_path.exists():
        source_np_db_path.unlink()

    tm_fraction = (
        float(args.tm_fraction)
        if args.tm_fraction is not None
        else float(params["simulation_defaults"]["tm_fraction_for_semi_empirical"])
    )
    shell_thickness_a = float(args.shell_thickness_a)
    outer_radius_a = float(args.core_radius_a) + shell_thickness_a
    surface_enabled = (
        config["surface_quench_mode"] == "outer_layer"
        and float(config["surface_fraction"]) > 0.0
    )
    surface_inner_radius_a = max(
        float(args.core_radius_a),
        outer_radius_a - SURFACE_LAYER_THICKNESS_A,
    )
    if surface_enabled:
        constraints = [
            SphericalConstraint(float(args.core_radius_a)),
            SphericalConstraint(surface_inner_radius_a),
            SphericalConstraint(outer_radius_a),
        ]
        dopant_specification = [
            (0, tm_fraction, "Tm", "Y"),
            (
                2,
                float(config["surface_fraction"]),
                str(config["surface_species"]),
                "Y",
            ),
        ]
    else:
        constraints = [
            SphericalConstraint(float(args.core_radius_a)),
            SphericalConstraint(outer_radius_a),
        ]
        dopant_specification = [(0, tm_fraction, "Tm", "Y")]
    nanoparticle = DopedNanoparticle(
        constraints=constraints,
        dopant_specification=dopant_specification,
        seed=int(args.doping_seed),
        prune_hosts=True,
    )
    nanoparticle.generate()

    dopant, sk = rates.build_spectral_kinetics(
        params,
        excitation_power_w_cm2=1.0,
        tm_fraction=tm_fraction,
        surface_species=(
            str(config["surface_species"]) if surface_enabled else None
        ),
        surface_fraction=(
            float(config["surface_fraction"]) if surface_enabled else 0.0
        ),
    )
    _ = dopant
    npmc_input = NPMCInput(sk, nanoparticle, initial_states=None)
    npmc_input.generate_nano_particle_database(str(source_np_db_path))
    site_counts_by_species = rates.count_sites_by_species(source_np_db_path)

    metadata = {
        "source": "generated_by_tm_npt_kmc_production.py",
        "doping_seed": int(args.doping_seed),
        "tm_fraction": tm_fraction,
        "core_radius_A": float(args.core_radius_a),
        "shell_thickness_A": shell_thickness_a,
        "outer_radius_A": outer_radius_a,
        "n_dopant_sites": len(nanoparticle.dopant_sites),
        "surface_quench_mode": str(config["surface_quench_mode"]),
        "surface_species": str(config["surface_species"]),
        "surface_fraction": float(config["surface_fraction"]),
        "surface_layer_thickness_A": SURFACE_LAYER_THICKNESS_A,
        "surface_inner_radius_A": (
            float(surface_inner_radius_a) if surface_enabled else None
        ),
        "site_counts_by_species": json_safe(site_counts_by_species),
    }
    with open(geometry_dir / "source_geometry_metadata.json", "w") as f:
        json.dump(json_safe(metadata), f, indent=2)
    return source_np_db_path


def interaction_row(
    *,
    number_of_sites: int,
    left_state_1: int,
    right_state_1: int,
    rate: float,
    interaction_type: str,
    label: str,
    source: str,
    species_id_1: int = 0,
    species_id_2: int = -1,
    left_state_2: int = -1,
    right_state_2: int = -1,
) -> dict[str, Any]:
    """Create one custom NPMC interaction row with extra manifest metadata."""
    return {
        "interaction_id": None,
        "number_of_sites": int(number_of_sites),
        "species_id_1": int(species_id_1),
        "species_id_2": int(species_id_2),
        "left_state_1": int(left_state_1),
        "left_state_2": int(left_state_2),
        "right_state_1": int(right_state_1),
        "right_state_2": int(right_state_2),
        "rate": float(rate),
        "interaction_type": interaction_type,
        "label": label,
        "source": source,
    }


def validate_species_interactions(
    interactions: list[dict[str, Any]],
    species_degrees: dict[int, int],
) -> None:
    """Ensure every interaction state falls within the exported species manifolds."""
    for row in interactions:
        species_1 = int(row["species_id_1"])
        max_state_1 = int(species_degrees[species_1]) - 1
        states_1 = [int(row["left_state_1"]), int(row["right_state_1"])]
        bad_states_1 = [state for state in states_1 if state < 0 or state > max_state_1]
        if bad_states_1:
            raise ValueError(
                f"Interaction {row['label']} exceeds species {species_1} states: {bad_states_1}"
            )

        if int(row["number_of_sites"]) != 2:
            continue
        species_2 = int(row["species_id_2"])
        max_state_2 = int(species_degrees[species_2]) - 1
        states_2 = [int(row["left_state_2"]), int(row["right_state_2"])]
        bad_states_2 = [state for state in states_2 if state < 0 or state > max_state_2]
        if bad_states_2:
            raise ValueError(
                f"Interaction {row['label']} exceeds species {species_2} states: {bad_states_2}"
            )


def tm_pair_label(local_tuple: tuple[int, int, int, int]) -> str:
    """Return a stable label for one exported Tm-Tm ET row."""
    channel_name = TM_CHANNEL_TUPLE_TO_NAME.get(local_tuple)
    if channel_name is not None:
        return channel_name
    if local_tuple[1] == local_tuple[2] and local_tuple[3] == local_tuple[0]:
        left_1, right_1, left_2, right_2 = local_tuple
        return f"EM {left_1 + 1}+{left_2 + 1}->{right_1 + 1}+{right_2 + 1}"
    return (
        f"ET {local_tuple[0] + 1}+{local_tuple[2] + 1}->"
        f"{local_tuple[1] + 1}+{local_tuple[3] + 1}"
    )


def tm_pair_description(
    dopant: Any,
    local_tuple: tuple[int, int, int, int],
) -> str:
    """Describe one exported Tm-Tm ET row using NPT's local level labels."""
    di, dj, ai, aj = local_tuple
    return (
        f"({dopant.energy_levels[di].label} ; {dopant.energy_levels[ai].label}) -> "
        f"({dopant.energy_levels[dj].label} ; {dopant.energy_levels[aj].label})"
    )


def one_site_label(
    interaction_type: str,
    left_state: int,
    right_state: int,
) -> str:
    """Return a stable 1-based label for one local one-site transition."""
    return f"{interaction_type} {left_state + 1}->{right_state + 1}"


def resonant_migration_metadata(
    local_tuple: tuple[int, int, int, int],
    dopant: Any,
) -> dict[str, Any] | None:
    """Classify one resonant Tm-Tm ET row for EM-mode filtering and scaling."""
    if local_tuple[1] != local_tuple[2] or local_tuple[3] != local_tuple[0]:
        return None

    ground_mediated_tuples = {
        (1, 0, 0, 1),
        (3, 0, 0, 3),
        (4, 0, 0, 4),
    }
    in_loop_only_tuples = {
        (3, 1, 1, 3),
        (4, 1, 1, 4),
        (4, 3, 3, 4),
    }
    if local_tuple in ground_mediated_tuples:
        migration_family = "ground_mediated"
        enabled_modes = ["all", "ground_mediated", "in_loop"]
    elif local_tuple in in_loop_only_tuples:
        migration_family = "in_loop_only"
        enabled_modes = ["all", "in_loop"]
    else:
        migration_family = "other_resonant_migration"
        enabled_modes = ["all"]
    return {
        "channel_name": tm_pair_label(local_tuple),
        "description": tm_pair_description(dopant, local_tuple),
        "migration_family": migration_family,
        "enabled_modes": enabled_modes,
    }


def load_mode_defaults(params: dict[str, Any]) -> dict[str, Any]:
    """Load the NPT production defaults from JSON with fallback values."""
    defaults = copy.deepcopy(FALLBACK_PRODUCTION_DEFAULTS)
    configured = params.get("production_defaults", {}).get("npt", {})
    if not isinstance(configured, dict):
        raise ValueError("Expected production_defaults['npt'] to be a JSON object")
    for key in defaults:
        if key in configured:
            defaults[key] = configured[key]
    return defaults


def resolve_production_config(
    args: argparse.Namespace,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the final NPT production configuration from defaults plus CLI overrides."""
    defaults = load_mode_defaults(params)

    def choose(name: str, cli_value: Any) -> Any:
        return defaults[name] if cli_value is None else cli_value

    config = {
        "rate_model": "npt",
        "npt_cr_mode": "exported",
        "sigma_esa_scale": float(choose("sigma_esa_scale", args.sigma_esa_scale)),
        "q21_scale": float(choose("q21_scale", args.q21_scale)),
        "s54_scale": float(choose("s54_scale", args.s54_scale)),
        "s45_scale": float(choose("s45_scale", args.s45_scale)),
        "s12_scale": float(choose("s12_scale", args.s12_scale)),
        "em_mode": str(choose("em_mode", args.em_mode)),
        "em_scale": float(choose("em_scale", args.em_scale)),
        "surface_quench_mode": str(
            choose("surface_quench_mode", args.surface_quench_mode)
        ),
        "surface_species": str(choose("surface_species", args.surface_species)),
        "surface_fraction": float(choose("surface_fraction", args.surface_fraction)),
        "surface_layer_thickness_a": float(SURFACE_LAYER_THICKNESS_A),
        "mode_defaults": defaults,
    }
    if config["em_mode"] not in ("off", "all", "ground_mediated", "in_loop"):
        raise ValueError(f"Unsupported em_mode: {config['em_mode']!r}")
    if config["surface_quench_mode"] not in ("off", "outer_layer"):
        raise ValueError(
            f"Unsupported surface_quench_mode: {config['surface_quench_mode']!r}"
        )
    if not (0.0 <= config["surface_fraction"] <= 1.0):
        raise ValueError("surface_fraction must lie between 0 and 1")
    return config


def pair_scale_for_channel(channel_name: str, config: dict[str, Any]) -> float:
    """Return the direct fixed scale for one named two-site channel."""
    if channel_name == Q21_CHANNEL_NAME:
        return float(config["q21_scale"])
    if channel_name == S54_CHANNEL_NAME:
        return float(config["s54_scale"])
    if channel_name == S45_CHANNEL_NAME:
        return float(config["s45_scale"])
    if channel_name == S12_CHANNEL_NAME:
        return float(config["s12_scale"])
    return 1.0


def build_npt_raw_vs_npmc_readin(
    one_site_report: list[dict[str, Any]],
    two_site_report: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize raw NPT rates beside the final rates written for NPMC."""
    def one_site_row(row: dict[str, Any]) -> dict[str, Any]:
        included = bool(row["included"])
        return {
            "label": row["label"],
            "transition": row["transition"],
            "interaction_type": row["interaction_type"],
            "species_id": int(row["species_id"]),
            "species_name": str(row["species_name"]),
            "npt_raw_rate_s^-1": float(row["base_rate_s^-1"]),
            "rate_scale_factor": float(row["rate_scale_factor"]),
            "npmc_readin_rate_s^-1": float(row["rate_s^-1"]) if included else None,
            "included_in_npmc": included,
            "filter_reason": row["filter_reason"],
            "rate_source": row["rate_source"],
        }

    def two_site_row(row: dict[str, Any]) -> dict[str, Any]:
        included = bool(row["included"])
        return {
            "channel_name": row["channel_name"],
            "description": row["description"],
            "species_id_1": int(row.get("species_id_1", TM_SPECIES_ID)),
            "species_id_2": int(row.get("species_id_2", TM_SPECIES_ID)),
            "kmc_tuple": row["kmc_tuple"],
            "npt_raw_rate_constant_nm6_s": float(row["base_kmc_rate"]),
            "rate_scale_factor": float(row["rate_scale_factor"]),
            "npmc_readin_rate_constant_nm6_s": (
                float(row["effective_kmc_rate"]) if included else None
            ),
            "included_in_npmc": included,
            "filter_reason": row["filter_reason"],
            "npt_raw_dre_equivalent_rate_s^-1": row[
                "base_dre_equivalent_rate_s^-1"
            ],
            "npmc_readin_dre_equivalent_rate_s^-1": (
                row["effective_dre_equivalent_rate_s^-1"] if included else None
            ),
            "rate_source": row["base_rate_source"],
        }

    return {
        "one_site_rates": {
            "units": "s^-1",
            "left_column": "npt_raw_rate_s^-1",
            "right_column": "npmc_readin_rate_s^-1",
            "rows": [one_site_row(row) for row in one_site_report],
        },
        "two_site_rate_constants": {
            "units": "nm^6 s^-1",
            "left_column": "npt_raw_rate_constant_nm6_s",
            "right_column": "npmc_readin_rate_constant_nm6_s",
            "rows": [two_site_row(row) for row in two_site_report],
        },
    }


def load_manifest_raw_vs_npmc_readin(manifest_path: str | Path) -> dict[str, Any]:
    """Load or reconstruct the raw-NPT vs NPMC-readin table from a manifest."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if "npt_raw_vs_npmc_readin" in manifest:
        return manifest["npt_raw_vs_npmc_readin"]
    return build_npt_raw_vs_npmc_readin(
        manifest["one_site"],
        manifest["two_site"],
    )


def build_power_rate_tables(build_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect per-power raw-NPT vs NPMC-readin rate tables for the root config."""
    tables = []
    for record in sorted(build_records, key=lambda row: int(row["power_index"])):
        manifest_path = Path(record["manifest_path"])
        tables.append(
            {
                "power_index": int(record["power_index"]),
                "excitation_power_w_cm2": float(record["excitation_power_w_cm2"]),
                "manifest_path": str(manifest_path.resolve()),
                "npt_raw_vs_npmc_readin": load_manifest_raw_vs_npmc_readin(
                    manifest_path
                ),
            }
        )
    return tables


def build_custom_interactions(
    params: dict[str, Any],
    source_np_db_path: Path,
    excitation_power: float,
    include_zero_rates: bool,
    tm_fraction: float | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one full-NPT interaction network for a single excitation power."""
    # Source DB provides geometry/species; rates are rebuilt below.
    sim_defaults = params["simulation_defaults"]
    surface_enabled = (
        config["surface_quench_mode"] == "outer_layer"
        and float(config["surface_fraction"]) > 0.0
    )
    geometry = rates.compute_geometry_factor(
        source_np_db_path,
        interaction_radius_bound_nm=float(sim_defaults["interaction_radius_bound_nm"]),
        distance_factor_type=sim_defaults["distance_factor_type"],
        species_ids={TM_SPECIES_ID},
    )
    site_counts_by_species = rates.count_sites_by_species(source_np_db_path)

    # NPT baseline for this power.
    semi_dopant, semi_sk = rates.build_spectral_kinetics(
        params,
        excitation_power_w_cm2=excitation_power,
        tm_fraction=tm_fraction,
        surface_species=(
            str(config["surface_species"]) if surface_enabled else None
        ),
        surface_fraction=(
            float(config["surface_fraction"]) if surface_enabled else 0.0
        ),
    )
    species_degrees = {
        species_id: int(dopant.n_levels)
        for species_id, dopant in enumerate(semi_sk.dopants)
    }
    species_slices = rates.species_level_slices(semi_sk)
    tm_slice = species_slices[TM_SPECIES_ID]

    kmc_default_absorption_cross_sections = rates.build_kmc_default_absorption_cross_sections(
        params,
        tm_fraction=tm_fraction,
    )
    base_rate_source_label = "NPT exported"

    interactions: list[dict[str, Any]] = []
    one_site_report: list[dict[str, Any]] = []
    two_site_report: list[dict[str, Any]] = []
    total_rad = (
        semi_sk.radiative_rate_matrix + semi_sk.magnetic_dipole_rate_matrix
    )[tm_slice, tm_slice]
    nr = semi_sk.non_radiative_rate_matrix[tm_slice, tm_slice]

    # One-site radiative rates from NPT.
    for left_state in range(total_rad.shape[0]):
        for right_state in range(total_rad.shape[1]):
            base_rate = float(total_rad[left_state, right_state])
            if base_rate == 0.0:
                continue
            included = include_zero_rates or base_rate != 0.0
            report = {
                "label": one_site_label("Rad", left_state, right_state),
                "transition": f"{left_state + 1}->{right_state + 1}",
                "species_id": TM_SPECIES_ID,
                "species_name": "Tm",
                "left_level": int(left_state),
                "right_level": int(right_state),
                "base_rate_s^-1": base_rate,
                "rate_scale_factor": 1.0,
                "rate_s^-1": base_rate,
                "included": included,
                "filter_reason": None if included else "zero_rate",
                "interaction_type": "Rad",
                "sigma_esa_scale": None,
                "pump_cross_section_source": None,
                "rate_source": "NPT radiative",
            }
            one_site_report.append(report)
            if not included:
                continue
            interactions.append(
                interaction_row(
                    number_of_sites=1,
                    left_state_1=int(left_state),
                    right_state_1=int(right_state),
                    rate=base_rate,
                    interaction_type="Rad",
                    label=report["label"],
                    source=str(report["rate_source"]),
                )
            )

    # One-site nonradiative rates from NPT.
    for left_state in range(nr.shape[0]):
        for right_state in range(nr.shape[1]):
            base_rate = float(nr[left_state, right_state])
            if base_rate == 0.0:
                continue
            rate_scale_factor = 1.0
            effective_rate = base_rate * rate_scale_factor
            included = include_zero_rates or effective_rate != 0.0
            filter_reason = None if included else "scale_zero_rate"
            report = {
                "label": one_site_label("NR", left_state, right_state),
                "transition": f"{left_state + 1}->{right_state + 1}",
                "species_id": TM_SPECIES_ID,
                "species_name": "Tm",
                "left_level": int(left_state),
                "right_level": int(right_state),
                "base_rate_s^-1": base_rate,
                "rate_scale_factor": rate_scale_factor,
                "rate_s^-1": effective_rate,
                "included": included,
                "filter_reason": filter_reason,
                "interaction_type": "NR",
                "sigma_esa_scale": None,
                "pump_cross_section_source": None,
                "rate_source": "NPT nonradiative",
            }
            one_site_report.append(report)
            if not included:
                continue
            interactions.append(
                interaction_row(
                    number_of_sites=1,
                    left_state_1=int(left_state),
                    right_state_1=int(right_state),
                    rate=effective_rate,
                    interaction_type="NR",
                    label=report["label"],
                    source=str(report["rate_source"]),
                )
            )

    # Pump rates from NPT effective sigma times photon flux; ESA may be scaled.
    incident_flux = float(semi_sk.incident_photon_flux)
    pump_rows = [
        {
            "left_state": 0,
            "right_state": 2,
            "base_rate_s^-1": float(
                kmc_default_absorption_cross_sections["sigma_GSA"] * incident_flux
            ),
            "rate_scale_factor": 1.0,
        },
        {
            "left_state": 1,
            "right_state": 5,
            "base_rate_s^-1": float(
                kmc_default_absorption_cross_sections["sigma_ESA"] * incident_flux
            ),
            "rate_scale_factor": float(config["sigma_esa_scale"]),
        },
    ]
    for row in pump_rows:
        effective_rate = float(row["base_rate_s^-1"]) * float(row["rate_scale_factor"])
        included = include_zero_rates or effective_rate != 0.0
        filter_reason = None if included else "scale_zero_rate"
        report = {
            "label": one_site_label("Pump", row["left_state"], row["right_state"]),
            "transition": f"{row['left_state'] + 1}->{row['right_state'] + 1}",
            "species_id": TM_SPECIES_ID,
            "species_name": "Tm",
            "left_level": int(row["left_state"]),
            "right_level": int(row["right_state"]),
            "base_rate_s^-1": float(row["base_rate_s^-1"]),
            "rate_scale_factor": float(row["rate_scale_factor"]),
            "rate_s^-1": effective_rate,
            "included": included,
            "filter_reason": filter_reason,
            "interaction_type": "Pump",
            "sigma_esa_scale": float(config["sigma_esa_scale"]),
            "pump_cross_section_source": "kmc-default",
            "rate_source": "NPT effective absorption cross section",
        }
        one_site_report.append(report)
        if not included:
            continue
        interactions.append(
            interaction_row(
                number_of_sites=1,
                left_state_1=int(row["left_state"]),
                right_state_1=int(row["right_state"]),
                rate=effective_rate,
                interaction_type="Pump",
                label=report["label"],
                source=str(report["rate_source"]),
            )
        )

    # Two-site ET/CR constants exported by NPT; selected channels may be scaled.
    exported_tm_pair_rows = sorted(
        (
            row
            for row in rates.build_full_npt_interactions(semi_sk)
            if row["interaction_type"] == "ET"
            and int(row["species_id_1"]) == TM_SPECIES_ID
            and int(row["species_id_2"]) == TM_SPECIES_ID
        ),
        key=lambda row: (
            int(row["left_state_1"]),
            int(row["right_state_1"]),
            int(row["left_state_2"]),
            int(row["right_state_2"]),
        ),
    )
    exported_tm_pair_tuples = {
        (
            int(row["left_state_1"]),
            int(row["right_state_1"]),
            int(row["left_state_2"]),
            int(row["right_state_2"]),
        )
        for row in exported_tm_pair_rows
    }
    for row in exported_tm_pair_rows:
        kmc_tuple = (
            int(row["left_state_1"]),
            int(row["right_state_1"]),
            int(row["left_state_2"]),
            int(row["right_state_2"]),
        )
        channel_name = tm_pair_label(kmc_tuple)
        description = tm_pair_description(semi_dopant, kmc_tuple)
        base_rate = float(row["rate"])
        same_initial_state = kmc_tuple[0] == kmc_tuple[2]
        degeneracy_factor = 2.0 if same_initial_state else 1.0
        base_dre_equivalent = (
            base_rate * geometry.ordered_factor_sum * degeneracy_factor / geometry.ion_count
        )

        migration = resonant_migration_metadata(kmc_tuple, semi_dopant)
        if migration is not None:
            rate_scale_factor = float(config["em_scale"])
            effective_rate = base_rate * rate_scale_factor
            effective_dre_equivalent = (
                base_dre_equivalent * rate_scale_factor
            )
            if config["em_mode"] not in migration["enabled_modes"]:
                included = False
                filter_reason = "em_mode_disabled"
            elif include_zero_rates or effective_rate != 0.0:
                included = True
                filter_reason = None
            else:
                included = False
                filter_reason = "scale_zero_rate"
            report = {
                "channel_name": channel_name,
                "description": description,
                "dre_rate_s^-1": None,
                "npt_selected_kmc_rate": base_rate,
                "npt_selected_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "npt_exported_kmc_rate": base_rate,
                "npt_exported_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "semi_empirical_exported_nm6_s": base_rate,
                "semi_empirical_exported": True,
                "semi_empirical_branch": "resonant_migration",
                "energy_gap_cm": None,
                "effective_energy_gap_cm": None,
                "base_rate_source": "NPT resonant migration",
                "base_kmc_rate": base_rate,
                "base_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "rate_scale_factor": rate_scale_factor,
                "effective_kmc_rate": effective_rate,
                "effective_dre_equivalent_rate_s^-1": effective_dre_equivalent,
                "same_initial_state": same_initial_state,
                "degeneracy_factor": degeneracy_factor,
                "kmc_tuple": list(kmc_tuple),
                "included": included,
                "filter_reason": filter_reason,
                "source": "NPT resonant migration",
                "is_resonant_migration": True,
                "migration_family": migration["migration_family"],
                "enabled_modes": migration["enabled_modes"],
            }
        else:
            rate_scale_factor = float(pair_scale_for_channel(channel_name, config))
            effective_rate = base_rate * rate_scale_factor
            effective_dre_equivalent = (
                base_dre_equivalent * rate_scale_factor
            )
            included = include_zero_rates or effective_rate != 0.0
            filter_reason = None if included else "scale_zero_rate"
            report = {
                "channel_name": channel_name,
                "description": description,
                "dre_rate_s^-1": None,
                "npt_selected_kmc_rate": base_rate,
                "npt_selected_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "npt_exported_kmc_rate": base_rate,
                "npt_exported_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "semi_empirical_exported_nm6_s": base_rate,
                "semi_empirical_exported": True,
                "semi_empirical_branch": "exported_npt",
                "energy_gap_cm": None,
                "effective_energy_gap_cm": None,
                "base_rate_source": base_rate_source_label,
                "base_kmc_rate": base_rate,
                "base_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "rate_scale_factor": rate_scale_factor,
                "effective_kmc_rate": effective_rate,
                "effective_dre_equivalent_rate_s^-1": effective_dre_equivalent,
                "same_initial_state": same_initial_state,
                "degeneracy_factor": degeneracy_factor,
                "kmc_tuple": list(kmc_tuple),
                "included": included,
                "filter_reason": filter_reason,
                "source": base_rate_source_label,
                "is_resonant_migration": False,
            }
        two_site_report.append(report)
        if not included:
            continue
        interactions.append(
            interaction_row(
                number_of_sites=2,
                species_id_1=TM_SPECIES_ID,
                species_id_2=TM_SPECIES_ID,
                left_state_1=kmc_tuple[0],
                left_state_2=kmc_tuple[2],
                right_state_1=kmc_tuple[1],
                right_state_2=kmc_tuple[3],
                rate=effective_rate,
                interaction_type="ET",
                label=channel_name,
                source=str(report["source"]),
            )
        )

   # Optional outer-layer surface quenching.
    if surface_enabled:
        for row in rates.build_surface_one_site_rates(
            semi_sk,
            species_id=SURFACE_SPECIES_ID,
        ):
            rate = float(row["dre_rate_s"])
            base_rate = float(row.get("base_dre_rate_s", rate))
            included = include_zero_rates or rate != 0.0
            report = {
                "label": (
                    f"Surface {row['type']} "
                    f"{int(row['left']) + 1}->{int(row['right']) + 1}"
                ),
                "transition": f"{int(row['left']) + 1}->{int(row['right']) + 1}",
                "species_id": int(row["species_id"]),
                "species_name": str(row["species_name"]),
                "left_level": int(row["left"]),
                "right_level": int(row["right"]),
                "base_rate_s^-1": base_rate,
                "rate_scale_factor": 1.0,
                "rate_s^-1": rate,
                "included": included,
                "filter_reason": None if included else "zero_rate",
                "interaction_type": row["type"],
                "sigma_esa_scale": None,
                "pump_cross_section_source": None,
                "rate_source": "NPT Surface NR",
            }
            one_site_report.append(report)
            if not included:
                continue
            interactions.append(
                interaction_row(
                    number_of_sites=1,
                    species_id_1=SURFACE_SPECIES_ID,
                    left_state_1=int(row["left"]),
                    right_state_1=int(row["right"]),
                    rate=rate,
                    interaction_type=row["type"],
                    label=report["label"],
                    source=str(report["rate_source"]),
                )
            )

        for surface_row in rates.build_tm_surface_energy_transfer_rates(
            semi_sk,
            tm_species_id=TM_SPECIES_ID,
            surface_species_id=SURFACE_SPECIES_ID,
        ):
            effective_rate = float(surface_row["rate_nm6_s"])
            included = include_zero_rates or effective_rate != 0.0
            filter_reason = None if included else "zero_rate"
            report = {
                "channel_name": surface_row["channel_name"],
                "description": "Surface quenching channel",
                "dre_rate_s^-1": None,
                "npt_selected_kmc_rate": effective_rate,
                "npt_selected_dre_equivalent_rate_s^-1": None,
                "npt_exported_kmc_rate": effective_rate,
                "npt_exported_dre_equivalent_rate_s^-1": None,
                "semi_empirical_exported_nm6_s": effective_rate,
                "semi_empirical_exported": True,
                "semi_empirical_branch": "surface_quenching",
                "energy_gap_cm": None,
                "effective_energy_gap_cm": None,
                "base_rate_source": surface_row["source"],
                "base_kmc_rate": effective_rate,
                "base_dre_equivalent_rate_s^-1": None,
                "rate_scale_factor": 1.0,
                "effective_kmc_rate": effective_rate,
                "effective_dre_equivalent_rate_s^-1": None,
                "same_initial_state": False,
                "degeneracy_factor": 1.0,
                "kmc_tuple": list(surface_row["kmc_tuple"]),
                "included": included,
                "filter_reason": filter_reason,
                "source": surface_row["source"],
                "is_resonant_migration": False,
                "species_id_1": int(surface_row["species_id_1"]),
                "species_id_2": int(surface_row["species_id_2"]),
                "species_name_1": str(surface_row["species_name_1"]),
                "species_name_2": str(surface_row["species_name_2"]),
                "is_surface_quench": True,
            }
            two_site_report.append(report)
            if not included:
                continue
            interactions.append(
                interaction_row(
                    number_of_sites=2,
                    species_id_1=int(surface_row["species_id_1"]),
                    species_id_2=int(surface_row["species_id_2"]),
                    left_state_1=int(surface_row["kmc_tuple"][0]),
                    left_state_2=int(surface_row["kmc_tuple"][2]),
                    right_state_1=int(surface_row["kmc_tuple"][1]),
                    right_state_2=int(surface_row["kmc_tuple"][3]),
                    rate=effective_rate,
                    interaction_type="ET",
                    label=surface_row["channel_name"],
                    source=surface_row["source"],
                )
            )

    # Final interaction table for np.sqlite.
    for interaction_id, interaction in enumerate(interactions):
        interaction["interaction_id"] = interaction_id

    validate_species_interactions(interactions, species_degrees)
    manifest = {
        "profile": params["profile"],
        "excitation_power_w_cm2": float(excitation_power),
        "include_zero_rates": include_zero_rates,
        "rate_model": config["rate_model"],
        "pump_cross_section_source": "kmc-default",
        "npt_cr_mode": config["npt_cr_mode"],
        "sigma_esa_scale": float(config["sigma_esa_scale"]),
        "em_mode": config["em_mode"],
        "em_scale": float(config["em_scale"]),
        "surface_quench_mode": str(config["surface_quench_mode"]),
        "surface_species": str(config["surface_species"]),
        "surface_fraction": float(config["surface_fraction"]),
        "surface_layer_thickness_a": float(config["surface_layer_thickness_a"]),
        "q21_scale": float(config["q21_scale"]),
        "s54_scale": float(config["s54_scale"]),
        "s45_scale": float(config["s45_scale"]),
        "s12_scale": float(config["s12_scale"]),
        "absorption_cross_sections_cm^2": {
            "npt_effective": json_safe(kmc_default_absorption_cross_sections),
        },
        "npt_raw_vs_npmc_readin": build_npt_raw_vs_npmc_readin(
            one_site_report,
            two_site_report,
        ),
        "geometry": {
            **geometry.__dict__,
            "total_site_count": int(sum(site_counts_by_species.values())),
            "site_counts_by_species": json_safe(site_counts_by_species),
            "tm_site_count": int(site_counts_by_species.get(TM_SPECIES_ID, 0)),
            "surface_site_count": int(site_counts_by_species.get(SURFACE_SPECIES_ID, 0)),
        },
        "interaction_count": len(interactions),
        "one_site": one_site_report,
        "two_site": two_site_report,
        "interactions": interactions,
    }
    return interactions, manifest


def write_custom_npmc_databases(
    *,
    source_np_db_path: Path,
    output_dir: Path,
    interactions: list[dict[str, Any]],
    interaction_radius_bound_nm: float,
    distance_factor_type: str,
) -> tuple[Path, Path]:
    """Write custom NPMC databases for one excitation power."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np_db_path = output_dir / "np.sqlite"
    initial_state_db_path = output_dir / "initial_state.sqlite"
    for path in (np_db_path, initial_state_db_path):
        if path.exists():
            path.unlink()

    site_records = rates.load_site_records_from_np_db(source_np_db_path)
    species_records = rates.load_species_records_from_np_db(source_np_db_path)
    with sqlite3.connect(np_db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE species (
                species_id          INTEGER NOT NULL PRIMARY KEY,
                degrees_of_freedom  INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE sites (
                site_id             INTEGER NOT NULL PRIMARY KEY,
                x                   REAL NOT NULL,
                y                   REAL NOT NULL,
                z                   REAL NOT NULL,
                species_id          INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE interactions (
                interaction_id      INTEGER NOT NULL PRIMARY KEY,
                number_of_sites     INTEGER NOT NULL,
                species_id_1        INTEGER NOT NULL,
                species_id_2        INTEGER NOT NULL,
                left_state_1        INTEGER NOT NULL,
                left_state_2        INTEGER NOT NULL,
                right_state_1       INTEGER NOT NULL,
                right_state_2       INTEGER NOT NULL,
                rate                REAL NOT NULL,
                interaction_type    TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE metadata (
                number_of_species                   INTEGER NOT NULL,
                number_of_sites                     INTEGER NOT NULL,
                number_of_interactions              INTEGER NOT NULL
            );
            """
        )
        cur.executemany(
            "INSERT INTO species VALUES (?, ?)",
            [
                (int(row.species_id), int(row.degrees_of_freedom))
                for row in species_records
            ],
        )
        cur.executemany(
            "INSERT INTO sites VALUES (?, ?, ?, ?, ?)",
            [
                (
                    int(row.site_id),
                    float(row.x),
                    float(row.y),
                    float(row.z),
                    int(row.species_id),
                )
                for row in site_records
            ],
        )
        cur.executemany(
            "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["interaction_id"],
                    row["number_of_sites"],
                    row["species_id_1"],
                    row["species_id_2"],
                    row["left_state_1"],
                    row["left_state_2"],
                    row["right_state_1"],
                    row["right_state_2"],
                    row["rate"],
                    row["interaction_type"],
                )
                for row in interactions
            ],
        )
        cur.execute(
            "INSERT INTO metadata VALUES (?, ?, ?)",
            (len(species_records), len(site_records), len(interactions)),
        )
        con.commit()

    with sqlite3.connect(initial_state_db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE initial_state (
                site_id            INTEGER NOT NULL PRIMARY KEY,
                degree_of_freedom  INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE trajectories (
                seed               INTEGER NOT NULL,
                step               INTEGER NOT NULL,
                time               REAL NOT NULL,
                site_id_1          INTEGER NOT NULL,
                site_id_2          INTEGER NOT NULL,
                interaction_id     INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE factors (
                one_site_interaction_factor      REAL NOT NULL,
                two_site_interaction_factor      REAL NOT NULL,
                interaction_radius_bound         REAL NOT NULL,
                distance_factor_type             TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE interrupt_state (
                seed                INTEGER NOT NULL,
                site_id             INTEGER NOT NULL,
                degree_of_freedom  INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE interrupt_cutoff (
                seed                INTEGER NOT NULL,
                step                INTEGER NOT NULL,
                time                REAL NOT NULL
            );
            """
        )
        cur.executemany(
            "INSERT INTO initial_state VALUES (?, ?)",
            [(int(row.site_id), 0) for row in site_records],
        )
        cur.execute(
            "INSERT INTO factors VALUES (?, ?, ?, ?)",
            (1.0, 1.0, float(interaction_radius_bound_nm), distance_factor_type),
        )
        con.commit()

    return np_db_path, initial_state_db_path


# ---------------------------------------------------------------------------
# Terminal-block convergence analysis ("terminal-blocks" mode).
#
# Design rules enforced here (replacing the old final-window/whole-run test):
# - Convergence is judged only from terminal, non-overlapping trajectory
#   blocks; no terminal block is ever compared with whole-run averages.
# - Low-count ("dark") seeds are never auto-accepted: a seed that cannot
#   supply the requested per-block statistics is reported as
#   insufficient_history or insufficient_counts, never as converged.
# - Every stopping decision is reconstructible from the JSON block
#   statistics written into the run summary / run_state.json.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceParameters:
    """Tunable knobs of the terminal-blocks convergence algorithm."""

    block_count: int = DEFAULT_CONVERGENCE_BLOCK_COUNT
    min_events_per_block: int = DEFAULT_CONVERGENCE_MIN_EVENTS_PER_BLOCK
    min_block_time_s: float = DEFAULT_CONVERGENCE_MIN_BLOCK_TIME_S
    relative_drift: float = DEFAULT_CONVERGENCE_RELATIVE_DRIFT
    poisson_z: float = DEFAULT_CONVERGENCE_POISSON_Z
    required_passes: int = DEFAULT_CONVERGENCE_REQUIRED_PASSES
    observables: tuple[str, ...] = ("rad800", "n4")
    mode_threshold_log10: float | None = None
    min_switches: int = DEFAULT_CONVERGENCE_MIN_SWITCHES
    semantics: str = DEFAULT_CONVERGENCE_SEMANTICS
    extension_steps: int = DEFAULT_CHECKPOINT_EXTENSION_STEPS
    max_step_cutoff: int = DEFAULT_MAX_STEP_CUTOFF
    algorithm: str = CONVERGENCE_ALGORITHM_ID

    def to_json(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "block_count": int(self.block_count),
            "min_events_per_block": int(self.min_events_per_block),
            "min_block_time_s": float(self.min_block_time_s),
            "relative_drift": float(self.relative_drift),
            "poisson_z": float(self.poisson_z),
            "required_passes": int(self.required_passes),
            "observables": list(self.observables),
            "mode_threshold_log10": (
                None
                if self.mode_threshold_log10 is None
                else float(self.mode_threshold_log10)
            ),
            "min_switches": int(self.min_switches),
            "semantics": str(self.semantics),
            "extension_steps": int(self.extension_steps),
            "max_step_cutoff": int(self.max_step_cutoff),
            "rate_floor_per_s": CONVERGENCE_RATE_FLOOR_PER_S,
            "population_floor": CONVERGENCE_POPULATION_FLOOR,
        }


CONVERGENCE_OBSERVABLE_CHOICES = ("rad800", "rad700", "n4")


def parse_convergence_observables(text: str) -> tuple[str, ...]:
    """Parse the comma-separated --convergence-observables value."""
    observables = tuple(
        item.strip() for item in str(text).split(",") if item.strip()
    )
    if not observables:
        raise ValueError("--convergence-observables must name at least one observable")
    unknown = [
        item for item in observables if item not in CONVERGENCE_OBSERVABLE_CHOICES
    ]
    if unknown:
        raise ValueError(
            f"Unsupported convergence observables: {unknown}; "
            f"choose from {CONVERGENCE_OBSERVABLE_CHOICES}"
        )
    if len(set(observables)) != len(observables):
        raise ValueError("--convergence-observables must not contain duplicates")
    return observables


def validate_convergence_parameters(params: ConvergenceParameters) -> None:
    """Fail fast on invalid convergence configuration (called early in main)."""
    if params.block_count < 4 or params.block_count % 2 != 0:
        raise ValueError("convergence block_count must be even and at least 4")
    if params.min_events_per_block < 1:
        raise ValueError("convergence min_events_per_block must be at least 1")
    if params.min_block_time_s <= 0:
        raise ValueError("convergence min_block_time_s must be positive")
    if params.relative_drift <= 0:
        raise ValueError("convergence relative_drift must be positive")
    if params.poisson_z <= 0:
        raise ValueError("convergence poisson_z must be positive")
    if params.required_passes < 1:
        raise ValueError("convergence required_passes must be at least 1")
    if params.min_switches < 1:
        raise ValueError("convergence min_switches must be at least 1")
    if params.semantics not in CONVERGENCE_SEMANTICS_CHOICES:
        raise ValueError(
            f"convergence semantics must be one of {CONVERGENCE_SEMANTICS_CHOICES}"
        )
    if params.semantics == "equilibrium" and params.mode_threshold_log10 is None:
        raise ValueError(
            "equilibrium convergence semantics require "
            "--convergence-mode-threshold so dark/bright basins are explicit"
        )
    if params.extension_steps < 1:
        raise ValueError("checkpoint extension_steps must be at least 1")
    if params.max_step_cutoff < 1:
        raise ValueError("max_step_cutoff must be at least 1")
    unknown = [
        item
        for item in params.observables
        if item not in CONVERGENCE_OBSERVABLE_CHOICES
    ]
    if unknown:
        raise ValueError(f"Unsupported convergence observables: {unknown}")
    if not ({"rad800", "rad700"} & set(params.observables)):
        raise ValueError(
            "terminal-blocks-v2 requires at least one rate stopping observable "
            "('rad800' or 'rad700'); 'n4' is validation-only"
        )


def resolve_convergence_parameters(args: argparse.Namespace) -> ConvergenceParameters:
    """Build and validate the convergence configuration from CLI arguments."""
    params = ConvergenceParameters(
        block_count=int(args.convergence_block_count),
        min_events_per_block=int(args.convergence_min_events_per_block),
        min_block_time_s=float(args.convergence_min_block_time_s),
        relative_drift=float(args.convergence_relative_drift),
        poisson_z=float(args.convergence_poisson_z),
        required_passes=int(args.convergence_required_passes),
        observables=parse_convergence_observables(args.convergence_observables),
        mode_threshold_log10=(
            None
            if args.convergence_mode_threshold is None
            else float(args.convergence_mode_threshold)
        ),
        min_switches=int(args.convergence_min_switches),
        semantics=str(args.convergence_semantics),
        extension_steps=int(args.checkpoint_extension_steps),
        max_step_cutoff=int(args.max_step_cutoff),
    )
    validate_convergence_parameters(params)
    return params


@dataclass
class BlockStatistics:
    """Statistics of one terminal, non-overlapping trajectory block."""

    seed: int
    block_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    rad800_count: int
    rad700_count: int
    rad800_rate_s: float
    rad700_rate_s: float
    n4_time_average: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "block_index": int(self.block_index),
            "start_time_s": float(self.start_time_s),
            "end_time_s": float(self.end_time_s),
            "duration_s": float(self.duration_s),
            "rad800_count": int(self.rad800_count),
            "rad700_count": int(self.rad700_count),
            "rad800_rate_s": float(self.rad800_rate_s),
            "rad700_rate_s": float(self.rad700_rate_s),
            "n4_time_average": (
                None
                if self.n4_time_average is None
                else float(self.n4_time_average)
            ),
        }


def rad_band_interaction_ids(
    interactions: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Return (rad800, rad700) interaction IDs.

    rad800 is the 800 nm 3H4 band: Rad rows with left_state_1 == N4_LEVEL.
    rad700 is the 700 nm 3F3 band: Rad rows with left_state_1 == 4.
    """
    rad800_ids = sorted(
        int(row["interaction_id"])
        for row in interactions
        if row["interaction_type"] == "Rad"
        and int(row["left_state_1"]) == N4_LEVEL
    )
    rad700_ids = sorted(
        int(row["interaction_id"])
        for row in interactions
        if row["interaction_type"] == "Rad" and int(row["left_state_1"]) == 4
    )
    return rad800_ids, rad700_ids


def ensure_trajectory_analysis_index(con: sqlite3.Connection) -> bool:
    """Create the covering analysis index on trajectories when missing.

    Returns True when this call built the index. Every convergence query is
    index-only once it exists; without it each per-seed query degenerates
    into a full table scan (the v1 production blocker). Building it on a
    multi-hundred-million-row table is a one-time cost of a few minutes and
    adds a small per-insert overhead to later NPMC extensions; both are
    recorded in the run state by the caller.
    """
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (TRAJECTORY_ANALYSIS_INDEX,),
    ).fetchone()
    if exists:
        return False
    con.execute(
        f"CREATE INDEX {TRAJECTORY_ANALYSIS_INDEX} "
        "ON trajectories(seed, interaction_id, time, step)"
    )
    con.commit()
    return True


def load_interrupt_cutoffs(con: sqlite3.Connection) -> dict[int, tuple[int, float]]:
    """Latest (step, time) per seed recorded by NPMC in interrupt_cutoff.

    NPMC appends one row per seed per invocation; the latest row wins.
    Returns an empty dict when the table is missing (checkpoint=0 runs and
    hand-built databases).
    """
    try:
        rows = con.execute("SELECT seed, step, time FROM interrupt_cutoff").fetchall()
    except sqlite3.OperationalError:
        return {}
    latest: dict[int, tuple[int, float]] = {}
    for seed, step, event_time in rows:
        key = (int(step), float(event_time))
        seed = int(seed)
        if seed not in latest or key > latest[seed]:
            latest[seed] = key
    return latest


def load_seed_checkpoint_identities(
    con: sqlite3.Connection,
    interaction_ids: list[int],
    analysis: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-seed checkpoint identity (max step/time) from trajectory contents.

    One index-only MAX probe per (seed, interaction_id) range, so the cost
    is O(seeds x interactions) index seeks regardless of trajectory length.
    Within a seed, step and time are monotonically coupled, so the row with
    the greatest (time, step) is also the greatest-step row. This immutable
    identity prevents counting the same database state as two convergence
    passes. The seed list comes from interrupt_cutoff when available (tiny)
    and falls back to a DISTINCT scan for hand-built databases.
    """
    seeds = sorted(load_interrupt_cutoffs(con))
    if not seeds:
        seeds = [
            int(row[0]) for row in con.execute("SELECT DISTINCT seed FROM trajectories")
        ]
    identities: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        best: tuple[float, int] | None = None
        for interaction_id in interaction_ids:
            row = con.execute(
                "SELECT time, step FROM trajectories "
                "WHERE seed=? AND interaction_id=? "
                "ORDER BY time DESC, step DESC LIMIT 1",
                (seed, int(interaction_id)),
            ).fetchone()
            if row is not None and (best is None or (row[0], row[1]) > best):
                best = (float(row[0]), int(row[1]))
        if analysis is not None:
            analysis["aggregate_queries"] = (
                int(analysis.get("aggregate_queries", 0)) + len(interaction_ids)
            )
        if best is not None:
            identities[seed] = {"max_step": best[1], "max_time_s": best[0]}
    return identities


def count_band_events(
    con: sqlite3.Connection,
    seed: int,
    interaction_ids: list[int],
    t_min: float | None = None,
    t_max: float | None = None,
    closed_end: bool = False,
) -> int:
    """Index-only event count for one seed/band, optionally time-windowed.

    Windows are half-open [t_min, t_max), closed at the right end only when
    closed_end is set (the terminal block ending exactly at the final time).
    """
    if not interaction_ids:
        return 0
    total = 0
    for interaction_id in interaction_ids:
        sql = "SELECT COUNT(*) FROM trajectories WHERE seed=? AND interaction_id=?"
        bind: list[Any] = [int(seed), int(interaction_id)]
        if t_min is not None:
            sql += " AND time >= ?"
            bind.append(float(t_min))
        if t_max is not None:
            sql += " AND time <= ?" if closed_end else " AND time < ?"
            bind.append(float(t_max))
        total += int(con.execute(sql, bind).fetchone()[0])
    return total


def fetch_latest_band_events(
    con: sqlite3.Connection,
    seed: int,
    interaction_ids: list[int],
    limit: int,
) -> list[tuple[float, int]]:
    """Newest `limit` events (time, step) for the band, ascending by time.

    Fetches the newest `limit` rows per interaction id through the covering
    index (never a sort of the whole band) and merges in Python.
    """
    if limit <= 0 or not interaction_ids:
        return []
    merged: list[tuple[float, int]] = []
    for interaction_id in interaction_ids:
        merged.extend(
            (float(row[0]), int(row[1]))
            for row in con.execute(
                "SELECT time, step FROM trajectories "
                "WHERE seed=? AND interaction_id=? "
                "ORDER BY time DESC, step DESC LIMIT ?",
                (seed, int(interaction_id), int(limit)),
            )
        )
    merged.sort(key=lambda item: (item[0], item[1]))
    return merged[-limit:]


def build_terminal_blocks(
    event_times: Any,
    final_time_s: float,
    block_count: int,
    min_events_per_block: int,
    min_block_time_s: float,
) -> tuple[list[tuple[float, float]] | None, str | None, dict[str, Any]]:
    """Build terminal, non-overlapping blocks walking back from the final time.

    Each block spans at least min_block_time_s and contains at least
    min_events_per_block Rad-800 events, taking the smallest terminal window
    that satisfies both so that older blocks keep as much history as
    possible. Returns (blocks, None, detail) with blocks sorted oldest to
    newest (the last block ends exactly at final_time_s), or
    (None, reason, detail) with reason "insufficient_history" when not
    enough physical time is available or "insufficient_counts" when there
    is enough time but too few events. Neither failure is convergence.
    """
    times = np.asarray(event_times, dtype=float)
    total_events = int(times.size)
    detail: dict[str, Any] = {
        "final_time_s": float(final_time_s),
        "total_rad800_events": total_events,
        "requested_block_count": int(block_count),
        "requested_min_events_per_block": int(min_events_per_block),
        "requested_min_block_time_s": float(min_block_time_s),
    }
    if not math.isfinite(final_time_s) or final_time_s <= 0:
        return None, "insufficient_history", detail
    if final_time_s < block_count * min_block_time_s:
        # The run is shorter than the analysis window: not enough physical
        # history no matter how many events were recorded.
        return None, "insufficient_history", detail
    boundaries = [float(final_time_s)]
    for _ in range(int(block_count)):
        end = boundaries[-1]
        if end < min_block_time_s:
            return None, "insufficient_history", detail
        # Events with time < end are still available to this block; events
        # exactly at end belong to the newer block, except at the final
        # time, which closes the newest block.
        side = "right" if end == float(final_time_s) else "left"
        available = int(np.searchsorted(times, end, side=side))
        if available < min_events_per_block:
            return None, "insufficient_counts", detail
        start_by_events = float(times[available - min_events_per_block])
        boundaries.append(min(start_by_events, end - min_block_time_s))
    blocks = [
        (boundaries[index + 1], boundaries[index])
        for index in range(int(block_count))
    ]
    blocks.reverse()  # oldest first; the last block ends at final_time_s
    detail["block_boundaries"] = [float(value) for value in reversed(boundaries)]
    return blocks, None, detail


def matching_interval_indices(
    intervals: list[tuple[float, float]],
    event_time: float,
    closed_end_time_s: float | None,
) -> list[int]:
    """Indices of the intervals containing event_time.

    Intervals are half-open [start, end), except that an interval ending
    exactly at closed_end_time_s (the seed's final time) also includes its
    right endpoint. Intervals may overlap and need not be sorted; every
    interval containing the event is returned.
    """
    hits: list[int] = []
    for index, (start, end) in enumerate(intervals):
        if event_time < start:
            continue
        if event_time < end or (
            closed_end_time_s is not None
            and event_time == end
            and end == closed_end_time_s
        ):
            hits.append(index)
    return hits


def compare_block_halves_rate(
    n1: int,
    t1: float,
    n2: int,
    t2: float,
    drift_tolerance: float,
    z_max: float,
) -> dict[str, Any]:
    """Older-half vs newer-half Poisson-rate comparison for one observable.

    Passes only when the relative drift is within tolerance AND the Poisson
    z-score of the rate difference is within z_max.
    """
    r1 = float(n1) / float(t1) if t1 > 0 else 0.0
    r2 = float(n2) / float(t2) if t2 > 0 else 0.0
    mean_rate = max(0.5 * (r1 + r2), CONVERGENCE_RATE_FLOOR_PER_S)
    relative_drift = abs(r2 - r1) / mean_rate
    variance = (float(n1) / t1**2 if t1 > 0 else 0.0) + (
        float(n2) / t2**2 if t2 > 0 else 0.0
    )
    if variance > 0:
        poisson_z = abs(r2 - r1) / math.sqrt(variance)
    else:
        poisson_z = 0.0 if r1 == r2 else float("inf")
    passed = bool(relative_drift <= drift_tolerance and poisson_z <= z_max)
    return {
        "n1": int(n1),
        "t1_s": float(t1),
        "n2": int(n2),
        "t2_s": float(t2),
        "r1_per_s": float(r1),
        "r2_per_s": float(r2),
        "relative_drift": float(relative_drift),
        "relative_drift_limit": float(drift_tolerance),
        "poisson_z": float(poisson_z),
        "poisson_z_limit": float(z_max),
        "passed": passed,
    }


def compare_block_halves_n4(
    block_means: list[float],
    drift_tolerance: float,
) -> dict[str, Any]:
    """Older-half vs newer-half comparison for N4 time-averaged population.

    Population averages are not Poisson counts, so only the relative-drift
    limit applies; the spread among block means estimates the uncertainty.
    """
    means = [float(value) for value in block_means]
    half = len(means) // 2
    older = means[:half]
    newer = means[half:]
    m1 = float(np.mean(older))
    m2 = float(np.mean(newer))
    center = max(0.5 * (m1 + m2), CONVERGENCE_POPULATION_FLOOR)
    relative_drift = abs(m2 - m1) / center
    if len(means) > 1:
        block_sem = float(np.std(means, ddof=1) / math.sqrt(len(means)))
    else:
        block_sem = 0.0
    passed = bool(relative_drift <= drift_tolerance)
    return {
        "block_means": means,
        "mean_older_half": m1,
        "mean_newer_half": m2,
        "relative_drift": float(relative_drift),
        "relative_drift_limit": float(drift_tolerance),
        "block_standard_error": block_sem,
        "passed": passed,
    }


def classify_block_modes(
    blocks: list[BlockStatistics],
    params: ConvergenceParameters,
) -> dict[str, Any]:
    """Per-block dark/bright mode classification for metastability checks.

    The per-block indicator is log10(rad800_rate + 1/duration), i.e. the
    exposure-derived one-event floor. Classification only happens when the
    user supplied --convergence-mode-threshold; without it the indicators
    are recorded as a diagnostic and never control stopping.
    """
    indicators = [
        math.log10(max(block.rad800_rate_s, 0.0) + 1.0 / block.duration_s)
        if block.duration_s > 0
        else float("nan")
        for block in blocks
    ]
    info: dict[str, Any] = {
        "classified": params.mode_threshold_log10 is not None,
        "threshold_log10": params.mode_threshold_log10,
        "indicator_log10": [float(value) for value in indicators],
        "sequence": None,
        "switches": None,
        "both_modes_present": False,
    }
    if params.mode_threshold_log10 is None:
        return info
    sequence = [
        "bright" if value >= params.mode_threshold_log10 else "dark"
        for value in indicators
    ]
    switches = sum(
        1 for older, newer in zip(sequence, sequence[1:]) if older != newer
    )
    info["sequence"] = sequence
    info["switches"] = int(switches)
    info["both_modes_present"] = len(set(sequence)) > 1
    return info


def poisson_upper_rate_bound(
    count: int,
    exposure_s: float,
    confidence: float = 0.95,
) -> float | None:
    """One-sided upper Poisson rate bound for an insufficient-count seed.

    Uses -log(1 - confidence)/T for zero events and the exact chi-square
    (Garwood) bound for nonzero counts when SciPy is available; otherwise
    falls back to the conservative (count - log(1 - confidence))/T.
    """
    if exposure_s <= 0:
        return None
    if count <= 0:
        return float(-math.log(1.0 - confidence) / exposure_s)
    try:
        from scipy.stats import chi2

        return float(0.5 * chi2.ppf(confidence, 2 * (count + 1)) / exposure_s)
    except Exception:
        return float((count - math.log(1.0 - confidence)) / exposure_s)


def classify_seed_convergence(
    seed: int,
    blocks: list[BlockStatistics] | None,
    failure_reason: str | None,
    block_detail: dict[str, Any],
    params: ConvergenceParameters,
) -> dict[str, Any]:
    """Classify one seed from its terminal blocks (or its failure reason).

    Statuses: "passed", "failed_drift", "metastable_censored",
    "insufficient_history", "insufficient_counts". Only "passed" counts as
    converged; nothing here ever auto-accepts a low-count seed.
    """
    report: dict[str, Any] = {
        "seed": int(seed),
        "passed": False,
        "status": None,
        "block_construction": json_safe(block_detail),
        "blocks": None,
        "tests": {},
        "mode": None,
        "poisson_upper_rate_95_per_s": None,
    }
    if failure_reason is not None or blocks is None:
        report["status"] = failure_reason or "insufficient_history"
        if report["status"] == "insufficient_counts":
            report["poisson_upper_rate_95_per_s"] = poisson_upper_rate_bound(
                int(block_detail.get("total_rad800_events", 0)),
                float(block_detail.get("final_time_s", 0.0)),
            )
        return report

    half = len(blocks) // 2
    older = blocks[:half]
    newer = blocks[half:]
    tests: dict[str, Any] = {}
    if "rad800" in params.observables:
        tests["rad800"] = compare_block_halves_rate(
            sum(block.rad800_count for block in older),
            sum(block.duration_s for block in older),
            sum(block.rad800_count for block in newer),
            sum(block.duration_s for block in newer),
            params.relative_drift,
            params.poisson_z,
        )
    if "rad700" in params.observables:
        tests["rad700"] = compare_block_halves_rate(
            sum(block.rad700_count for block in older),
            sum(block.duration_s for block in older),
            sum(block.rad700_count for block in newer),
            sum(block.duration_s for block in newer),
            params.relative_drift,
            params.poisson_z,
        )
    # N4 is never a stopping observable in terminal-blocks-v2: population
    # integrals need a full replay, so they are attached once at
    # finalization under report["validation"] instead of gating here.

    mode_info = classify_block_modes(blocks, params)
    report["blocks"] = [block.to_json() for block in blocks]
    report["tests"] = tests
    report["mode"] = mode_info

    # ``bool(tests)`` is a defense against a programmatic caller bypassing
    # validate_convergence_parameters with an N4-only configuration.  N4 is
    # deliberately validation-only in v2, so an empty set of stopping tests
    # must never pass through Python's vacuous all([]) == True behavior.
    all_passed = bool(tests) and all(test["passed"] for test in tests.values())
    if params.semantics == "equilibrium" and (
        not mode_info["classified"]
        or not mode_info["both_modes_present"]
        or int(mode_info["switches"]) < params.min_switches
    ):
        # An equilibrium claim requires observed occupation of both basins
        # and the requested minimum number of transitions.  A stationary
        # single-basin trajectory is branch-converged, not equilibrium-
        # converged.  Branch semantics leave mode classification diagnostic.
        report["status"] = "metastable_censored"
        report["passed"] = False
        return report
    report["status"] = "passed" if all_passed else "failed_drift"
    report["passed"] = bool(all_passed)
    return report


def evaluate_seed_terminal_blocks(
    con: sqlite3.Connection,
    seed: int,
    identity: dict[str, Any],
    rad800_ids: list[int],
    rad700_ids: list[int],
    params: ConvergenceParameters,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Terminal-block convergence report for one seed, index-only queries.

    Fetches just enough of the newest Rad-800 events to construct the
    terminal blocks (doubling the fetch window only while the time
    constraint reaches past it), then counts each block's Rad-800/Rad-700
    events with index range probes. The fetched window is provably covered:
    every event inside the final blocks is part of the fetched set, so the
    block boundaries match a full-history evaluation exactly.
    """
    final_time = float(identity["max_time_s"])
    total_rad800 = count_band_events(con, seed, rad800_ids)
    analysis["aggregate_queries"] = int(analysis.get("aggregate_queries", 0)) + len(
        rad800_ids
    )
    limit = min(params.block_count * params.min_events_per_block, total_rad800)
    blocks: list[tuple[float, float]] | None = None
    reason: str | None = "insufficient_counts"
    detail: dict[str, Any] = {}
    while True:
        events = fetch_latest_band_events(con, seed, rad800_ids, limit)
        analysis["fetched_event_rows"] = int(analysis.get("fetched_event_rows", 0)) + len(
            events
        )
        times = np.asarray([event[0] for event in events], dtype=float)
        blocks, reason, detail = build_terminal_blocks(
            times,
            final_time,
            params.block_count,
            params.min_events_per_block,
            params.min_block_time_s,
        )
        covered = (
            blocks is not None and (times.size == 0 or blocks[0][0] >= float(times[0]))
        )
        if covered or reason == "insufficient_history" or limit >= total_rad800:
            break
        limit = min(max(2 * limit, limit + 1), total_rad800)
    # The true per-seed total comes from the COUNT probe, not the fetch.
    detail["total_rad800_events"] = int(total_rad800)

    if blocks is None:
        report = classify_seed_convergence(seed, None, reason, detail, params)
    else:
        statistics: list[BlockStatistics] = []
        for block_index, (start, end) in enumerate(blocks):
            closed = bool(end == final_time)
            duration = float(end) - float(start)
            n800 = count_band_events(
                con, seed, rad800_ids, t_min=start, t_max=end, closed_end=closed
            )
            n700 = count_band_events(
                con, seed, rad700_ids, t_min=start, t_max=end, closed_end=closed
            )
            analysis["aggregate_queries"] = int(
                analysis.get("aggregate_queries", 0)
            ) + len(rad800_ids) + len(rad700_ids)
            statistics.append(
                BlockStatistics(
                    seed=int(seed),
                    block_index=block_index,
                    start_time_s=float(start),
                    end_time_s=float(end),
                    duration_s=duration,
                    rad800_count=n800,
                    rad700_count=n700,
                    rad800_rate_s=(n800 / duration if duration > 0 else 0.0),
                    rad700_rate_s=(n700 / duration if duration > 0 else 0.0),
                    n4_time_average=None,
                )
            )
        report = classify_seed_convergence(seed, statistics, None, detail, params)
    report["checkpoint_identity"] = {
        "max_step": int(identity["max_step"]),
        "max_time_s": float(identity["max_time_s"]),
    }
    if "n4" in params.observables:
        report["n4_evaluation"] = "deferred_to_finalization"
    return report


def evaluate_run_convergence(
    initial_state_db_path: Path,
    interactions: list[dict[str, Any]],
    params: ConvergenceParameters,
) -> dict[str, Any]:
    """Evaluate per-seed convergence with index-only terminal-block queries.

    terminal-blocks-v2: every statistic comes from range probes against
    TRAJECTORY_ANALYSIS_INDEX (created on first use and reported under
    analysis.index_created). There is no trajectory replay and no full
    table scan, so the per-checkpoint cost is bounded by the terminal
    window size instead of the total trajectory length; analysis counters
    report exactly how many rows were consumed. N4 is computed once at
    finalization as a validation observable, never as a stopping criterion.
    """
    rad800_ids, rad700_ids = rad_band_interaction_ids(interactions)
    all_ids = sorted({int(row["interaction_id"]) for row in interactions})
    analysis: dict[str, Any] = {
        "index_created": False,
        "fetched_event_rows": 0,
        "aggregate_queries": 0,
    }
    con = sqlite3.connect(initial_state_db_path)
    try:
        analysis["index_created"] = bool(ensure_trajectory_analysis_index(con))
        identities = load_seed_checkpoint_identities(con, all_ids, analysis)
        seed_reports = {
            seed: evaluate_seed_terminal_blocks(
                con, seed, identity, rad800_ids, rad700_ids, params, analysis
            )
            for seed, identity in sorted(identities.items())
        }
    finally:
        con.close()
    all_passed = bool(seed_reports) and all(
        report["passed"] for report in seed_reports.values()
    )
    return {
        "algorithm": params.algorithm,
        "parameters": params.to_json(),
        "seeds": seed_reports,
        "all_passed": bool(all_passed),
        "checkpoint_identities": {
            seed: dict(report["checkpoint_identity"])
            for seed, report in seed_reports.items()
        },
        "analysis": analysis,
    }


def update_pass_streaks(
    streaks: dict[int, int] | None,
    evaluation: dict[str, Any],
    required_passes: int,
    last_counted: dict[int, tuple[int, float]] | None = None,
) -> tuple[dict[int, int], dict[int, tuple[int, float]], bool]:
    """Advance consecutive-pass streaks from one checkpoint evaluation.

    A pass increments the streak only when the seed's checkpoint identity
    (max step/time of its trajectory) is strictly newer than the identity
    last counted for that seed, so re-evaluating identical data after an
    interruption can never produce a duplicate pass. A failing seed resets
    to zero. Returns (streaks, last_counted, all_converged).
    """
    previous = dict(streaks or {})
    new_counted = {int(seed): tuple(value) for seed, value in (last_counted or {}).items()}
    new_streaks: dict[int, int] = {}
    for seed, report in evaluation["seeds"].items():
        seed = int(seed)
        if not report["passed"]:
            new_streaks[seed] = 0
            continue
        identity = report.get("checkpoint_identity") or {}
        id_tuple = (
            (int(identity["max_step"]), float(identity["max_time_s"]))
            if "max_step" in identity
            else None
        )
        if id_tuple is not None and new_counted.get(seed) is not None:
            if id_tuple <= new_counted[seed]:
                # Same (or older) database state as the last counted pass:
                # keep the streak, do not count again.
                new_streaks[seed] = previous.get(seed, 0)
                continue
        new_streaks[seed] = previous.get(seed, 0) + 1
        if id_tuple is not None:
            new_counted[seed] = id_tuple
    all_converged = bool(new_streaks) and all(
        streak >= required_passes for streak in new_streaks.values()
    )
    return new_streaks, new_counted, all_converged


def derive_run_convergence_status(
    evaluation: dict[str, Any],
    all_converged: bool,
    hit_cap: bool,
) -> str:
    """Roll per-seed statuses up into one run-level status string."""
    if all_converged:
        return "converged"
    statuses = {
        str(report["status"]) for report in evaluation["seeds"].values()
    }
    for priority in (
        "metastable_censored",
        "insufficient_counts",
        "insufficient_history",
    ):
        if priority in statuses:
            return priority
    return "capped" if hit_cap else "running"


def summarize_mode_fractions(evaluation: dict[str, Any]) -> dict[str, Any] | None:
    """Dark/bright/switching/censored fractions when mode classification ran."""
    classified = [
        report
        for report in evaluation["seeds"].values()
        if report.get("mode") and report["mode"].get("classified")
    ]
    if not classified:
        return None
    dark = bright = switching = censored = 0
    for report in classified:
        mode = report["mode"]
        if report["status"] == "metastable_censored":
            censored += 1
        elif mode["both_modes_present"]:
            switching += 1
        elif mode["sequence"] and mode["sequence"][-1] == "dark":
            dark += 1
        else:
            bright += 1
    total = len(classified)
    return {
        "n_classified_seeds": int(total),
        "dark_fraction": dark / total,
        "bright_fraction": bright / total,
        "switching_fraction": switching / total,
        "censored_fraction": censored / total,
    }


def run_npmc(
    np_db_path: Path,
    initial_state_db_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    step_cutoff: int | None = None,
    num_sims: int | None = None,
) -> None:
    """Run NPMC on the custom databases.

    num_sims defaults to args.num_sims; the adaptive refinement passes a
    larger value when a reused pilot point needs additional seeds. NPMC
    seeds are base_seed .. base_seed + number_of_simulations - 1, so a
    larger count with the same base seed resumes existing seeds from their
    checkpoints and starts the new seeds fresh.
    """
    run_args = [
        args.npmc_command,
        f"--nano_particle_database={np_db_path}",
        f"--initial_state_database={initial_state_db_path}",
        f"--number_of_simulations={int(num_sims) if num_sims is not None else int(args.num_sims)}",
        f"--base_seed={args.base_seed}",
        f"--thread_count={args.thread_count}",
    ]
    if args.resolved_cutoff_mode == "steps":
        if step_cutoff is None:
            step_cutoff = int(args.resolved_simulation_length)
        run_args.append(f"--step_cutoff={step_cutoff}")
        # checkpoint=1 lets a later invocation with a larger absolute
        # step_cutoff resume each seed from interrupt_state/interrupt_cutoff
        # instead of replaying the trajectory table.
        run_args.append("--checkpoint=1")
    elif args.resolved_cutoff_mode == "physical-time":
        run_args.append(f"--time_cutoff={args.resolved_simulation_time}")
        run_args.append("--checkpoint=0")
    else:
        raise ValueError(
            f"Unsupported simulation cutoff mode: {args.resolved_cutoff_mode!r}"
        )

    print(f'Running NPMC using the command: "{" ".join(run_args)}"', flush=True)
    with open(output_dir / "stdout", "a") as f_std, open(output_dir / "stderr", "a") as f_err:
        subprocess.run(run_args, stdout=f_std, stderr=f_err, check=True)


def run_power_point(
    *,
    power_index: int,
    power_count: int,
    power: float,
    output_root: Path,
    params: dict[str, Any],
    source_np_db_path: Path,
    include_zero_rates: bool,
    tm_fraction: float | None,
    config: dict[str, Any],
    args: argparse.Namespace,
    trajectory_archive_root: Path,
    local_db_staging_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build and optionally run one power point."""
    output_dir = output_root / f"power_{power_index:02d}_{power:.6g}"
    db_work_dir = prepare_power_db_work_dir(
        output_dir,
        local_db_staging_root=local_db_staging_root,
        dry_run=bool(args.dry_run),
    )
    print(
        f"[power {power_index + 1}/{power_count}] building {power:.6g} W cm^-2 "
        f"for the NPT rate model",
        flush=True,
    )
    if db_work_dir != output_dir:
        print(
            f"[power {power_index + 1}/{power_count}] staging SQLite writes in {db_work_dir}",
            flush=True,
        )
    interactions, manifest = build_custom_interactions(
        params=params,
        source_np_db_path=source_np_db_path,
        excitation_power=float(power),
        include_zero_rates=include_zero_rates,
        tm_fraction=tm_fraction,
        config=config,
    )
    np_db_path, initial_state_db_path = write_custom_npmc_databases(
        source_np_db_path=source_np_db_path,
        output_dir=db_work_dir,
        interactions=interactions,
        interaction_radius_bound_nm=float(
            params["simulation_defaults"]["interaction_radius_bound_nm"]
        ),
        distance_factor_type=str(
            params["simulation_defaults"]["distance_factor_type"]
        ),
    )
    manifest_path = output_dir / "npt_interaction_manifest.json"
    write_json_atomic(manifest_path, manifest)

    build_record = {
        "power_index": int(power_index),
        "excitation_power_w_cm2": float(power),
        "output_dir": str(output_dir.resolve()),
        "interaction_count": int(len(interactions)),
        "manifest_path": str(manifest_path.resolve()),
        "np_db_path": str((output_dir / "np.sqlite").resolve()),
        "initial_state_db_path": str((output_dir / "initial_state.sqlite").resolve()),
    }
    if args.dry_run:
        return build_record, None

    current_step_cutoff = (
        int(args.resolved_simulation_length)
        if args.resolved_cutoff_mode == "steps"
        else None
    )
    initial_step_cutoff = current_step_cutoff
    run_npmc(
        np_db_path,
        initial_state_db_path,
        output_dir,
        args,
        step_cutoff=current_step_cutoff,
    )
    convergence: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    if current_step_cutoff is not None:
        conv_params = resolve_convergence_parameters(args)
        streaks: dict[int, int] = {}
        last_counted: dict[int, tuple[int, float]] = {}
        extension_count = 0
        checkpoint_history: list[dict[str, Any]] = []
        status = "running"
        while True:
            evaluation = evaluate_run_convergence(
                initial_state_db_path, interactions, conv_params
            )
            streaks, last_counted, all_converged = update_pass_streaks(
                streaks, evaluation, conv_params.required_passes, last_counted
            )
            checkpoint_history.append(
                {
                    "step_cutoff": int(current_step_cutoff),
                    "seed_statuses": {
                        str(seed): report["status"]
                        for seed, report in sorted(evaluation["seeds"].items())
                    },
                    "seed_pass_streaks": {
                        str(seed): int(streak)
                        for seed, streak in sorted(streaks.items())
                    },
                    "checkpoint_identities": {
                        str(seed): dict(identity)
                        for seed, identity in sorted(
                            evaluation["checkpoint_identities"].items()
                        )
                    },
                    "all_converged": bool(all_converged),
                }
            )
            if all_converged:
                status = "converged"
                break
            if current_step_cutoff >= conv_params.max_step_cutoff:
                status = derive_run_convergence_status(
                    evaluation, all_converged=False, hit_cap=True
                )
                break
            n_not_passed = sum(
                1 for report in evaluation["seeds"].values() if not report["passed"]
            )
            current_step_cutoff = min(
                current_step_cutoff + conv_params.extension_steps,
                conv_params.max_step_cutoff,
            )
            extension_count += 1
            print(
                f"[power {power_index + 1}/{power_count}] "
                f"{n_not_passed}/{len(evaluation['seeds'])} seed(s) not "
                f"converged; extending to {current_step_cutoff} steps",
                flush=True,
            )
            run_npmc(
                np_db_path,
                initial_state_db_path,
                output_dir,
                args,
                step_cutoff=current_step_cutoff,
            )
        print(
            f"[power {power_index + 1}/{power_count}] convergence "
            f"{status} after {extension_count} extension(s) "
            f"at {current_step_cutoff} steps",
            flush=True,
        )
    # One replay pass at finalization for the whole-run summary, the N4
    # validation attachment, and the terminal estimate (never per checkpoint).
    estimate_intervals = (
        block_intervals_from_evaluation(evaluation) if evaluation is not None else None
    )
    replay = replay_trajectories(
        initial_state_db_path,
        interactions,
        intervals_by_seed=estimate_intervals or None,
    )
    terminal_estimate = None
    if evaluation is not None:
        attach_n4_validation(evaluation, replay, conv_params)
        terminal_estimate = terminal_estimate_from_interval_stats(
            replay.get("interval_stats", {}),
            int(replay["tm_site_count"]),
            {"type": "terminal_blocks", "algorithm": evaluation["algorithm"]},
        )
        convergence = {
            "algorithm": conv_params.algorithm,
            "semantics": conv_params.semantics,
            "status": status,
            "extensions_disabled": False,
            "initial_step_cutoff": int(initial_step_cutoff),
            "final_step_cutoff": int(current_step_cutoff),
            "extension_count": int(extension_count),
            "parameters": conv_params.to_json(),
            "seeds": {
                str(seed): report
                for seed, report in sorted(evaluation["seeds"].items())
            },
            "seed_pass_streaks": {
                str(seed): int(streak) for seed, streak in sorted(streaks.items())
            },
            "checkpoints": checkpoint_history,
            "mode_fractions": summarize_mode_fractions(evaluation),
            "seed_heterogeneity": seed_heterogeneity_warning(terminal_estimate),
        }
    summary = summarize_run(
        replay=replay,
        interactions=interactions,
        n_sites=int(manifest["geometry"]["ion_count"]),
        excitation_power=float(power),
        manifest=manifest,
        simulation_cutoff_mode=args.resolved_cutoff_mode,
        simulation_step_cutoff=current_step_cutoff,
        simulation_time_cutoff_s=args.resolved_simulation_time,
        stage="single",
        terminal_estimate=terminal_estimate,
    )
    if convergence is not None:
        summary["convergence"] = convergence
    write_json_atomic(output_dir / "npt_run_summary.json", summary)
    try:
        with MATPLOTLIB_LOCK:
            plot_power_spectrum(
                params=params,
                excitation_power=float(power),
                tm_fraction=tm_fraction,
                config=config,
                summary=summary,
                interactions=interactions,
                output_dir=output_dir,
            )
    except Exception as exc:
        print(
            f"[power {power_index + 1}/{power_count}] WARNING: emission spectrum "
            f"failed: {exc}",
            flush=True,
        )
    try:
        from plot_mechanism_diagrams import plot_run as plot_mechanism_run

        with MATPLOTLIB_LOCK:
            plot_mechanism_run(
                output_dir,
                "mechanism_diagram.png",
                "events_per_ion_s",
                1e-3,
                {"ET": 16, "Rad": 18, "NR": 12, "Pump": 8, "SQ": 16},
                220,
            )
    except Exception as exc:
        print(
            f"[power {power_index + 1}/{power_count}] WARNING: mechanism diagram "
            f"failed: {exc}",
            flush=True,
        )

    move_output_file(np_db_path, output_dir / "np.sqlite")
    archived_initial_state_db_path = finalize_initial_state_database(
        initial_state_db_path=initial_state_db_path,
        output_dir=output_dir,
        archive_root=trajectory_archive_root,
    )
    build_record["np_db_path"] = str((output_dir / "np.sqlite").resolve())
    build_record["initial_state_db_path"] = str(
        (output_dir / "initial_state.sqlite").resolve()
    )
    build_record["archived_initial_state_db_path"] = str(archived_initial_state_db_path)
    if db_work_dir != output_dir and db_work_dir.exists():
        db_work_dir.rmdir()
    return build_record, summary


def build_npt_dndt_from_summary(
    summary: dict[str, Any],
    interactions: list[dict[str, Any]],
) -> list[list[Any]]:
    """Reconstruct the NPT dN/dt row format expected by its spectrum helper."""
    flux_by_id = {
        int(row["interaction_id"]): float(row["events_per_particle_s"])
        for row in summary.get("per_interaction", [])
    }
    dndt_rows: list[list[Any]] = []
    for row in interactions:
        interaction_id = int(row["interaction_id"])
        dndt_rows.append(
            [
                interaction_id,
                int(row["number_of_sites"]),
                int(row["species_id_1"]),
                int(row["species_id_2"]),
                int(row["left_state_1"]),
                int(row["left_state_2"]),
                int(row["right_state_1"]),
                int(row["right_state_2"]),
                str(row["interaction_type"]),
                float(row["rate"]),
                float(flux_by_id.get(interaction_id, 0.0)),
            ]
        )
    return dndt_rows


def plot_power_spectrum(
    *,
    params: dict[str, Any],
    excitation_power: float,
    tm_fraction: float | None,
    config: dict[str, Any],
    summary: dict[str, Any],
    interactions: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save a per-power emission spectrum using NPT's dN/dt spectrum helper."""
    surface_enabled = (
        config["surface_quench_mode"] == "outer_layer"
        and float(config["surface_fraction"]) > 0.0
    )
    _, sk = rates.build_spectral_kinetics(
        params,
        excitation_power_w_cm2=excitation_power,
        tm_fraction=tm_fraction,
        surface_species=(
            str(config["surface_species"]) if surface_enabled else None
        ),
        surface_fraction=(
            float(config["surface_fraction"]) if surface_enabled else 0.0
        ),
    )
    dndt_rows = build_npt_dndt_from_summary(summary, interactions)
    wavelength_signed, spectrum = get_spectrum_wavelength_from_dndt(
        dndt_rows,
        sk.dopants,
        lower_bound=-2000,
        upper_bound=-300,
        step=2,
    )
    wavelength_nm = np.abs(wavelength_signed)
    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    spectrum = np.asarray(spectrum, dtype=float)[order]

    spectrum_data = {
        "source": "NanoParticleTools.analysis.util.get_spectrum_wavelength_from_dndt",
        "excitation_power_w_cm2": float(excitation_power),
        "x_axis": "emission_wavelength_nm",
        "y_axis": "radiative_events_per_particle_s",
        "wavelength_nm": [float(value) for value in wavelength_nm],
        "spectrum": [float(value) for value in spectrum],
    }
    with open(output_dir / "npt_emission_spectrum.json", "w") as f:
        json.dump(json_safe(spectrum_data), f, indent=2)

    fig, ax = plt.subplots(dpi=300, figsize=(6.2, 4.2))
    ax.plot(wavelength_nm, spectrum, linewidth=1.6)
    ax.set_xlabel("Emission wavelength (nm)", fontsize=12)
    ax.set_ylabel("Radiative events per particle per s", fontsize=12)
    ax.set_title(
        f"NPT spectrum | {format_power_tick(float(excitation_power))} W cm$^{{-2}}$",
        fontsize=12,
    )
    ax.set_xlim(float(wavelength_nm[0]), float(wavelength_nm[-1]))
    if np.any(spectrum > 0):
        ax.set_ylim(bottom=0.0, top=float(np.max(spectrum)) * 1.08)
    ax.grid(True, alpha=0.25)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.90)
    fig.savefig(output_dir / "npt_emission_spectrum.png")
    plt.close(fig)


def finalize_initial_state_database(
    initial_state_db_path: Path,
    output_dir: Path,
    archive_root: Path,
) -> Path:
    """Materialize the completed trajectory DB in the final output location."""
    if not initial_state_db_path.exists():
        raise FileNotFoundError(
            f"Missing completed trajectory database: {initial_state_db_path}"
        )
    final_initial_state_db_path = output_dir / "initial_state.sqlite"
    if initial_state_db_path.is_symlink():
        archived_path = initial_state_db_path.resolve()
        if final_initial_state_db_path != initial_state_db_path:
            if final_initial_state_db_path.is_symlink() or final_initial_state_db_path.exists():
                final_initial_state_db_path.unlink()
            final_initial_state_db_path.symlink_to(archived_path)
        return archived_path
    if not archive_root.exists():
        return move_output_file(initial_state_db_path, final_initial_state_db_path)

    now = datetime.now()
    archive_day_dir = archive_root / now.strftime("%Y-%m-%d")
    archive_day_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    archive_name = f"{output_dir.parent.name}_{output_dir.name}_{timestamp}.sqlite"
    archived_path = archive_day_dir / archive_name
    suffix = 1
    while archived_path.exists():
        archived_path = archive_day_dir / (
            f"{output_dir.parent.name}_{output_dir.name}_{timestamp}_{suffix}.sqlite"
        )
        suffix += 1

    shutil.move(str(initial_state_db_path), str(archived_path))
    if final_initial_state_db_path.is_symlink() or final_initial_state_db_path.exists():
        final_initial_state_db_path.unlink()
    final_initial_state_db_path.symlink_to(archived_path.resolve())
    return archived_path.resolve()


def replay_trajectories(
    initial_state_db_path: Path,
    interactions: list[dict[str, Any]],
    intervals_by_seed: dict[int, list[tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """Read event counts, final simulated times, and time-averaged n4 occupancy.

    When intervals_by_seed is given, the same single streaming pass also
    accumulates per-interval statistics (Rad-800/Rad-700 event counts and
    the N4 time integral) for arbitrary [start_time, end_time] intervals,
    replaying from the initial state so every block boundary has the
    correct site states. Intervals may overlap; each accumulates
    independently, and per-seed interval order is preserved in the
    returned interval_stats. Intervals are half-open [start, end), except
    that any interval ending at the seed's final time also includes its
    right endpoint.
    """
    simulation_time: dict[int, float] = {}
    event_counts: dict[int, Counter] = defaultdict(Counter)
    n4_time_integral: dict[int, float] = defaultdict(float)
    n4_population_per_seed: dict[int, float] = {}
    q24_total_count = 0
    s12_total_count = 0
    q24_after_s12_same_pair_count = 0
    s12_after_q24_same_pair_count = 0

    np_db_path = initial_state_db_path.with_name("np.sqlite")
    with sqlite3.connect(np_db_path) as con:
        site_rows = con.execute(
            "SELECT site_id, species_id FROM sites ORDER BY site_id"
        ).fetchall()
    site_count = len(site_rows)
    site_species = np.asarray(
        [int(species_id) for _site_id, species_id in site_rows],
        dtype=np.int8,
    )
    tm_site_count = int(np.sum(site_species == TM_SPECIES_ID))

    interactions_by_id = {
        int(row["interaction_id"]): {
            "number_of_sites": int(row["number_of_sites"]),
            "left_state_1": int(row["left_state_1"]),
            "left_state_2": int(row["left_state_2"]),
            "right_state_1": int(row["right_state_1"]),
            "right_state_2": int(row["right_state_2"]),
            "label": row["label"],
        }
        for row in interactions
    }
    s12_interaction_id = next(
        (
            int(row["interaction_id"])
            for row in interactions
            if row["interaction_type"] == "ET" and row["label"] == S12_CHANNEL_NAME
        ),
        None,
    )
    q24_interaction_id = next(
        (
            int(row["interaction_id"])
            for row in interactions
            if row["interaction_type"] == "ET" and row["label"] == Q21_CHANNEL_NAME
        ),
        None,
    )
    rad800_ids, rad700_ids = rad_band_interaction_ids(interactions)
    rad800_id_set = set(rad800_ids)
    rad700_id_set = set(rad700_ids)

    interval_lists: dict[int, list[tuple[float, float]]] = {}
    if intervals_by_seed:
        for seed, intervals in intervals_by_seed.items():
            # Caller order is preserved so interval_stats stay aligned with
            # the caller's own block/interval lists.
            normalized = [(float(start), float(end)) for start, end in intervals]
            if normalized:
                interval_lists[int(seed)] = normalized
    interval_stats: dict[int, list[dict[str, Any]]] = {}

    with sqlite3.connect(initial_state_db_path) as con:
        rows = con.execute(
            """
            SELECT seed, step, time, site_id_1, site_id_2, interaction_id
            FROM trajectories
            ORDER BY rowid
            """
        )
        # NPMC appends each seed's events in step/time order, but extensions
        # may interleave seed chunks.  Keep replay state per seed so the table
        # can be streamed in rowid order.  This avoids the production-scale
        # temporary B-tree formerly required by ORDER BY seed, step.
        previous_time_by_seed: dict[int, float] = {}
        previous_step_by_seed: dict[int, int] = {}
        site_states_by_seed: dict[int, np.ndarray] = {}
        n4_count_by_seed: dict[int, int] = {}
        previous_event_by_seed: dict[int, tuple[int, tuple[int, int] | None]] = {}
        active_interval_acc_by_seed: dict[int, list[dict[str, float]]] = {}
        active_interval_closed_end_by_seed: dict[int, float] = {}

        def finalize_seed(seed: int) -> None:
            final_time = previous_time_by_seed[seed]
            simulation_time[seed] = float(final_time)
            if final_time > 0 and tm_site_count > 0:
                n4_population_per_seed[seed] = (
                    n4_time_integral[seed] / (float(tm_site_count) * float(final_time))
                )
            else:
                n4_population_per_seed[seed] = 0.0
            active_intervals = interval_lists.get(seed)
            active_interval_acc = active_interval_acc_by_seed.get(seed)
            if active_intervals is not None and active_interval_acc is not None:
                stats = []
                for (start, end), acc in zip(active_intervals, active_interval_acc):
                    duration = float(end) - float(start)
                    stats.append(
                        {
                            "start_time_s": float(start),
                            "end_time_s": float(end),
                            "duration_s": duration,
                            "rad800_count": int(acc["rad800_count"]),
                            "rad700_count": int(acc["rad700_count"]),
                            "n4_time_integral": float(acc["n4_integral"]),
                            "n4_time_average": (
                                acc["n4_integral"] / (float(tm_site_count) * duration)
                                if duration > 0 and tm_site_count > 0
                                else None
                            ),
                        }
                    )
                interval_stats[seed] = stats

        for seed, step, event_time, site_id_1, site_id_2, interaction_id in rows:
            seed = int(seed)
            step = int(step)
            event_time = float(event_time)
            interaction_id = int(interaction_id)
            site_id_1 = int(site_id_1)
            site_id_2 = int(site_id_2)

            if seed not in site_states_by_seed:
                previous_time_by_seed[seed] = 0.0
                previous_step_by_seed[seed] = -1
                site_states_by_seed[seed] = np.zeros(site_count, dtype=np.int8)
                n4_count_by_seed[seed] = 0
                active_intervals = interval_lists.get(seed)
                if active_intervals is not None:
                    active_interval_acc_by_seed[seed] = [
                        {"rad800_count": 0, "rad700_count": 0, "n4_integral": 0.0}
                        for _interval in active_intervals
                    ]
                    active_interval_closed_end_by_seed[seed] = max(
                        end for _start, end in active_intervals
                    )

            previous_step = previous_step_by_seed[seed]
            previous_time = previous_time_by_seed[seed]
            site_states = site_states_by_seed[seed]
            current_n4_count = n4_count_by_seed[seed]
            active_intervals = interval_lists.get(seed)
            active_interval_acc = active_interval_acc_by_seed.get(seed)
            active_interval_closed_end = active_interval_closed_end_by_seed.get(seed)

            if step <= previous_step:
                raise ValueError(
                    f"Trajectory step did not increase in append order for seed {seed}: "
                    f"{step} <= {previous_step}"
                )

            dt = event_time - previous_time
            if dt < -1e-12:
                raise ValueError(
                    f"Trajectory time decreased for seed {seed} step {step}: "
                    f"{event_time} < {previous_time}"
                )
            n4_time_integral[seed] += current_n4_count * max(dt, 0.0)
            simulation_time[seed] = float(event_time)
            event_counts[seed][interaction_id] += 1

            if active_intervals is not None and active_interval_acc is not None:
                for (start, end), acc in zip(active_intervals, active_interval_acc):
                    overlap = min(event_time, end) - max(previous_time, start)
                    if overlap > 0:
                        acc["n4_integral"] += current_n4_count * overlap
                if (
                    interaction_id in rad800_id_set
                    or interaction_id in rad700_id_set
                ):
                    for interval_index in matching_interval_indices(
                        active_intervals, event_time, active_interval_closed_end
                    ):
                        acc = active_interval_acc[interval_index]
                        if interaction_id in rad800_id_set:
                            acc["rad800_count"] += 1
                        else:
                            acc["rad700_count"] += 1

            interaction = interactions_by_id[interaction_id]
            pair_key: tuple[int, int] | None = None
            if interaction["number_of_sites"] == 2:
                pair_key = (min(site_id_1, site_id_2), max(site_id_1, site_id_2))

            previous_event = previous_event_by_seed.get(seed)
            if q24_interaction_id is not None and interaction_id == q24_interaction_id:
                q24_total_count += 1
                if (
                    previous_event is not None
                    and s12_interaction_id is not None
                    and previous_event[0] == s12_interaction_id
                    and pair_key is not None
                    and previous_event[1] == pair_key
                ):
                    q24_after_s12_same_pair_count += 1
            if s12_interaction_id is not None and interaction_id == s12_interaction_id:
                s12_total_count += 1
                if (
                    previous_event is not None
                    and q24_interaction_id is not None
                    and previous_event[0] == q24_interaction_id
                    and pair_key is not None
                    and previous_event[1] == pair_key
                ):
                    s12_after_q24_same_pair_count += 1

            current_state_1 = int(site_states[site_id_1])
            if current_state_1 != interaction["left_state_1"]:
                raise ValueError(
                    "Trajectory replay mismatch for seed "
                    f"{seed} step {step} site {site_id_1}: "
                    f"expected state {interaction['left_state_1']}, found {current_state_1}"
                )
            if int(site_species[site_id_1]) == TM_SPECIES_ID:
                current_n4_count += int(interaction["right_state_1"] == N4_LEVEL)
                current_n4_count -= int(current_state_1 == N4_LEVEL)
            site_states[site_id_1] = interaction["right_state_1"]

            if interaction["number_of_sites"] == 2:
                current_state_2 = int(site_states[site_id_2])
                if current_state_2 != interaction["left_state_2"]:
                    raise ValueError(
                        "Trajectory replay mismatch for seed "
                        f"{seed} step {step} site {site_id_2}: "
                        f"expected state {interaction['left_state_2']}, found {current_state_2}"
                    )
                if int(site_species[site_id_2]) == TM_SPECIES_ID:
                    current_n4_count += int(interaction["right_state_2"] == N4_LEVEL)
                    current_n4_count -= int(current_state_2 == N4_LEVEL)
                site_states[site_id_2] = interaction["right_state_2"]

            previous_event_by_seed[seed] = (interaction_id, pair_key)
            previous_step_by_seed[seed] = step
            previous_time_by_seed[seed] = event_time
            n4_count_by_seed[seed] = current_n4_count

        for seed in sorted(previous_time_by_seed):
            finalize_seed(seed)

    return {
        "simulation_time": simulation_time,
        "event_counts": event_counts,
        "n4_time_integral": n4_time_integral,
        "n4_population_per_seed": n4_population_per_seed,
        "total_site_count": int(site_count),
        "tm_site_count": int(tm_site_count),
        "q24_total_count": q24_total_count,
        "s12_total_count": s12_total_count,
        "q24_after_s12_same_pair_count": q24_after_s12_same_pair_count,
        "s12_after_q24_same_pair_count": s12_after_q24_same_pair_count,
        "interval_stats": interval_stats,
    }


def combine_seed_interval_stats(interval_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-interval accumulators into one terminal-window accumulator."""
    return {
        "duration_s": float(sum(float(iv["duration_s"]) for iv in interval_stats)),
        "rad800_count": int(sum(int(iv["rad800_count"]) for iv in interval_stats)),
        "rad700_count": int(sum(int(iv["rad700_count"]) for iv in interval_stats)),
        "n4_time_integral": float(
            sum(float(iv["n4_time_integral"]) for iv in interval_stats)
        ),
    }


def block_intervals_from_evaluation(
    evaluation: dict[str, Any],
) -> dict[int, list[tuple[float, float]]]:
    """Terminal-block [start, end) intervals per seed from an evaluation."""
    intervals: dict[int, list[tuple[float, float]]] = {}
    for seed, report in evaluation["seeds"].items():
        blocks = report.get("blocks") or []
        if blocks:
            intervals[int(seed)] = [
                (float(block["start_time_s"]), float(block["end_time_s"]))
                for block in blocks
            ]
    return intervals


def attach_n4_validation(
    evaluation: dict[str, Any],
    replay: dict[str, Any],
    params: ConvergenceParameters,
) -> None:
    """Attach the once-per-run N4 validation to each seed report.

    Replayed per-block N4 time averages fill the report blocks and the
    older/newer-half drift comparison is recorded under
    report["validation"]["n4"]. In terminal-blocks-v2 this never changes
    report["passed"]: N4 is a validation observable, not a stopping one.
    """
    if "n4" not in params.observables:
        return
    interval_stats = replay.get("interval_stats", {})
    for seed, report in evaluation["seeds"].items():
        blocks = report.get("blocks") or []
        if not blocks:
            continue
        seed_stats = interval_stats.get(int(seed), [])[: len(blocks)]
        means: list[float] = []
        for block, acc in zip(blocks, seed_stats):
            average = acc.get("n4_time_average")
            block["n4_time_average"] = (
                None if average is None else float(average)
            )
            means.append(float(average) if average is not None else 0.0)
        if len(means) == len(blocks):
            result = compare_block_halves_n4(means, params.relative_drift)
            result["validation_only"] = True
            report.setdefault("validation", {})["n4"] = result
            report["n4_evaluation"] = "validation_only"


def terminal_estimate_from_interval_stats(
    interval_stats_by_seed: dict[int, list[dict[str, Any]]],
    tm_site_count: int,
    window: dict[str, Any],
) -> dict[str, Any] | None:
    """Terminal estimate from per-seed replay interval accumulators.

    Each seed's intervals (terminal blocks or the pilot terminal-fraction
    window) are combined into per-seed rates/populations. Returns None when
    no seed produced analyzable intervals.
    """
    per_seed: dict[int, dict[str, Any]] = {}
    for seed, stats in interval_stats_by_seed.items():
        if stats:
            per_seed[int(seed)] = combine_seed_interval_stats(stats)
    if not per_seed:
        return None
    return build_terminal_estimate(per_seed, tm_site_count, window)


def seed_heterogeneity_warning(
    terminal_estimate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Flag points whose per-seed terminal rad800 rates span >= 2 decades.

    A wide spread means the point likely contains seeds sitting in
    different (dark/bright) basins; under branch convergence semantics the
    aggregate must not be read as an equilibrium basin-weighted result.
    """
    if not terminal_estimate:
        return None
    seed_values = terminal_estimate.get("seed_values") or {}
    rates = [
        float(value)
        for value in (seed_values.get("rad800_events_per_particle_s") or {}).values()
        if value is not None and float(value) > 0
    ]
    if len(rates) < 2:
        return None
    ratio = max(rates) / min(rates)
    return {
        "max_min_seed_rate_ratio": float(ratio),
        "mixed_basin_warning": bool(ratio >= SEED_HETEROGENEITY_RATIO),
    }


def build_terminal_estimate(
    per_seed: dict[int, dict[str, Any]],
    tm_site_count: int,
    window: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate per-seed terminal-analysis observables for one power point.

    per_seed maps seed -> combine_seed_interval_stats output. Rates are
    events per second per nanoparticle (the "per_particle" convention of the
    whole-run proxy fields). The flat mean/median/standard_error/min/max
    keys describe the per-seed Rad-800 rates; per-observable statistics are
    nested under "per_observable".
    """

    def stats_block(values: list[float]) -> dict[str, Any]:
        if not values:
            return {
                "mean": None,
                "median": None,
                "standard_error": None,
                "min": None,
                "max": None,
            }
        arr = np.asarray(values, dtype=float)
        standard_error = (
            float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        )
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "standard_error": standard_error,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    seed_rad800: dict[str, float] = {}
    seed_rad700: dict[str, float] = {}
    seed_n4: dict[str, float] = {}
    seed_exposure: dict[str, float] = {}
    total_duration = 0.0
    total_rad800 = 0
    total_rad700 = 0
    total_n4_integral = 0.0
    for seed, combined in sorted(per_seed.items()):
        duration = float(combined["duration_s"])
        if duration <= 0:
            continue
        seed_rad800[str(seed)] = float(combined["rad800_count"]) / duration
        seed_rad700[str(seed)] = float(combined["rad700_count"]) / duration
        if tm_site_count > 0:
            seed_n4[str(seed)] = float(combined["n4_time_integral"]) / (
                float(tm_site_count) * duration
            )
        seed_exposure[str(seed)] = duration
        total_duration += duration
        total_rad800 += int(combined["rad800_count"])
        total_rad700 += int(combined["rad700_count"])
        total_n4_integral += float(combined["n4_time_integral"])

    rad800_stats = stats_block(list(seed_rad800.values()))
    return {
        "window": json_safe(window),
        "n_seeds": len(seed_rad800),
        "rad_800_proxy_events_per_particle_s": (
            total_rad800 / total_duration if total_duration > 0 else 0.0
        ),
        "rad_700_proxy_events_per_particle_s": (
            total_rad700 / total_duration if total_duration > 0 else 0.0
        ),
        "n4_time_averaged_population": (
            total_n4_integral / (float(tm_site_count) * total_duration)
            if total_duration > 0 and tm_site_count > 0
            else 0.0
        ),
        "seed_values": {
            "rad800_events_per_particle_s": seed_rad800,
            "rad700_events_per_particle_s": seed_rad700,
            "n4_time_averaged_population": seed_n4,
            "exposure_s": seed_exposure,
        },
        "mean": rad800_stats["mean"],
        "median": rad800_stats["median"],
        "standard_error": rad800_stats["standard_error"],
        "min": rad800_stats["min"],
        "max": rad800_stats["max"],
        "per_observable": {
            "rad800": rad800_stats,
            "rad700": stats_block(list(seed_rad700.values())),
            "n4": stats_block(list(seed_n4.values())),
        },
    }


def summarize_run(
    replay: dict[str, Any],
    interactions: list[dict[str, Any]],
    n_sites: int,
    excitation_power: float,
    manifest: dict[str, Any],
    simulation_cutoff_mode: str,
    simulation_step_cutoff: int | None,
    simulation_time_cutoff_s: float | None,
    stage: str = "single",
    terminal_estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact summary for a completed power point.

    Schema version 2 keeps every pre-existing top-level key as a whole-run
    value, mirrors the three primary observables under "whole_run", and
    adds "stage" plus the terminal-analysis "terminal_estimate". Plotting
    and detection prefer terminal_estimate; nothing redefines the old keys.
    """
    interactions_by_id = {row["interaction_id"]: row for row in interactions}
    total_time = float(sum(replay["simulation_time"].values()))
    total_counts = Counter()
    for seed_counts in replay["event_counts"].values():
        total_counts.update(seed_counts)

    per_interaction = []
    for interaction_id, count in sorted(total_counts.items()):
        row = interactions_by_id[interaction_id]
        per_interaction.append(
            {
                "interaction_id": interaction_id,
                "label": row["label"],
                "interaction_type": row["interaction_type"],
                "count": int(count),
                "events_per_particle_s": count / total_time if total_time > 0 else 0.0,
                "events_per_ion_s": count / total_time / n_sites if total_time > 0 else 0.0,
            }
        )

    rad_800_count = sum(
        count
        for interaction_id, count in total_counts.items()
        if interactions_by_id[interaction_id]["interaction_type"] == "Rad"
        and interactions_by_id[interaction_id]["left_state_1"] == 3
    )
    rad_700_count = sum(
        count
        for interaction_id, count in total_counts.items()
        if interactions_by_id[interaction_id]["interaction_type"] == "Rad"
        and interactions_by_id[interaction_id]["left_state_1"] == 4
    )
    w4r_total = sum(
        float(row["rate"])
        for row in interactions
        if row["interaction_type"] == "Rad" and row["left_state_1"] == N4_LEVEL
    )
    n4_time_averaged_population = (
        float(sum(replay["n4_time_integral"].values())) / (float(n_sites) * total_time)
        if total_time > 0
        else 0.0
    )
    n4_from_rad_800_proxy = (
        (rad_800_count / total_time) / (w4r_total * float(n_sites))
        if total_time > 0 and w4r_total > 0 and n_sites > 0
        else None
    )

    two_site_by_name = {
        row["channel_name"]: row
        for row in manifest["two_site"]
        if not row.get("is_resonant_migration", False)
    }
    interaction_flux_by_label = {
        row["label"]: row["events_per_ion_s"]
        for row in per_interaction
    }

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage": str(stage),
        "excitation_power_w_cm2": float(excitation_power),
        "rate_model": str(manifest["rate_model"]),
        "pump_cross_section_source": str(manifest["pump_cross_section_source"]),
        "npt_cr_mode": str(manifest["npt_cr_mode"]),
        "sigma_esa_scale": float(manifest["sigma_esa_scale"]),
        "q21_scale": float(manifest["q21_scale"]),
        "s54_scale": float(manifest["s54_scale"]),
        "s45_scale": float(manifest["s45_scale"]),
        "s12_scale": float(manifest["s12_scale"]),
        "em_mode": str(manifest["em_mode"]),
        "em_scale": float(manifest["em_scale"]),
        "surface_quench_mode": str(manifest["surface_quench_mode"]),
        "surface_species": str(manifest["surface_species"]),
        "surface_fraction": float(manifest["surface_fraction"]),
        "surface_layer_thickness_a": float(manifest["surface_layer_thickness_a"]),
        "whole_run": {
            "rad_800_proxy_events_per_particle_s": (
                rad_800_count / total_time if total_time > 0 else 0.0
            ),
            "rad_700_proxy_events_per_particle_s": (
                rad_700_count / total_time if total_time > 0 else 0.0
            ),
            "n4_time_averaged_population": n4_time_averaged_population,
        },
        "terminal_estimate": json_safe(terminal_estimate),
        "simulation_cutoff_mode": simulation_cutoff_mode,
        "simulation_step_cutoff": (
            None if simulation_step_cutoff is None else int(simulation_step_cutoff)
        ),
        "simulation_time_cutoff_s": (
            None if simulation_time_cutoff_s is None else float(simulation_time_cutoff_s)
        ),
        "n_sites": int(n_sites),
        "tm_site_count": int(replay.get("tm_site_count", n_sites)),
        "total_site_count": int(replay.get("total_site_count", n_sites)),
        "surface_site_count": int(manifest["geometry"].get("surface_site_count", 0)),
        "num_completed_sims": len(replay["simulation_time"]),
        "total_simulation_time_s": total_time,
        "n4_time_averaged_population": n4_time_averaged_population,
        "n4_population_per_seed": {
            str(seed): float(value)
            for seed, value in sorted(replay["n4_population_per_seed"].items())
        },
        "n4_from_rad_800_proxy": n4_from_rad_800_proxy,
        "n4_rad_consistency_ratio": (
            n4_from_rad_800_proxy / n4_time_averaged_population
            if n4_from_rad_800_proxy is not None and n4_time_averaged_population > 0
            else None
        ),
        "rad_800_proxy_events_per_particle_s": (
            rad_800_count / total_time if total_time > 0 else 0.0
        ),
        "rad_700_proxy_events_per_particle_s": (
            rad_700_count / total_time if total_time > 0 else 0.0
        ),
        "q21_base_kmc_rate": (
            float(two_site_by_name[Q21_CHANNEL_NAME]["base_kmc_rate"])
            if Q21_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "q21_effective_kmc_rate": (
            float(two_site_by_name[Q21_CHANNEL_NAME]["effective_kmc_rate"])
            if Q21_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "q21_base_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[Q21_CHANNEL_NAME]["base_dre_equivalent_rate_s^-1"])
            if Q21_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "q21_effective_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[Q21_CHANNEL_NAME]["effective_dre_equivalent_rate_s^-1"])
            if Q21_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s54_base_kmc_rate": (
            float(two_site_by_name[S54_CHANNEL_NAME]["base_kmc_rate"])
            if S54_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s54_effective_kmc_rate": (
            float(two_site_by_name[S54_CHANNEL_NAME]["effective_kmc_rate"])
            if S54_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s54_base_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[S54_CHANNEL_NAME]["base_dre_equivalent_rate_s^-1"])
            if S54_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s54_effective_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[S54_CHANNEL_NAME]["effective_dre_equivalent_rate_s^-1"])
            if S54_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s45_base_kmc_rate": (
            float(two_site_by_name[S45_CHANNEL_NAME]["base_kmc_rate"])
            if S45_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s45_effective_kmc_rate": (
            float(two_site_by_name[S45_CHANNEL_NAME]["effective_kmc_rate"])
            if S45_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s45_base_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[S45_CHANNEL_NAME]["base_dre_equivalent_rate_s^-1"])
            if S45_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s45_effective_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[S45_CHANNEL_NAME]["effective_dre_equivalent_rate_s^-1"])
            if S45_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s12_base_kmc_rate": (
            float(two_site_by_name[S12_CHANNEL_NAME]["base_kmc_rate"])
            if S12_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s12_effective_kmc_rate": (
            float(two_site_by_name[S12_CHANNEL_NAME]["effective_kmc_rate"])
            if S12_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s12_base_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[S12_CHANNEL_NAME]["base_dre_equivalent_rate_s^-1"])
            if S12_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s12_effective_dre_equivalent_rate_s^-1": (
            float(two_site_by_name[S12_CHANNEL_NAME]["effective_dre_equivalent_rate_s^-1"])
            if S12_CHANNEL_NAME in two_site_by_name
            else None
        ),
        "s12_events_per_ion_s": float(interaction_flux_by_label.get(S12_CHANNEL_NAME, 0.0)),
        "s54_events_per_ion_s": float(interaction_flux_by_label.get(S54_CHANNEL_NAME, 0.0)),
        "s45_events_per_ion_s": float(interaction_flux_by_label.get(S45_CHANNEL_NAME, 0.0)),
        "q24_total_count": int(replay["q24_total_count"]),
        "s12_total_count": int(replay["s12_total_count"]),
        "q24_after_s12_same_pair_count": int(replay["q24_after_s12_same_pair_count"]),
        "s12_after_q24_same_pair_count": int(replay["s12_after_q24_same_pair_count"]),
        "q24_after_s12_same_pair_fraction": (
            replay["q24_after_s12_same_pair_count"] / replay["q24_total_count"]
            if replay["q24_total_count"] > 0
            else None
        ),
        "s12_after_q24_same_pair_fraction": (
            replay["s12_after_q24_same_pair_count"] / replay["s12_total_count"]
            if replay["s12_total_count"] > 0
            else None
        ),
        "per_interaction": per_interaction,
    }


# ---------------------------------------------------------------------------
# Adaptive two-stage orchestration (pilot scan -> transition detection ->
# center-weighted refinement with adaptive terminal-block convergence).
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically (temporary file then os.replace) so an
    interrupted run keeps a readable, resumable file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(json_safe(payload), handle, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_json_file(path: Path) -> Any | None:
    """Load a JSON file, returning None when it does not exist."""
    if not path.exists():
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def resolve_power_parallel_total_slots(args: argparse.Namespace) -> int | None:
    """Effective CPU-slot budget for power-parallel scheduling.

    A SLURM allocation always wins over --power-parallel-total-slots: the
    scheduler, not the CLI flag, knows the real CPU budget on a cluster node.
    """
    if (
        os.environ.get("SLURM_NTASKS") is not None
        or os.environ.get("SLURM_CPUS_ON_NODE") is not None
    ):
        return default_power_parallel_total_slots()
    if args.power_parallel_total_slots is not None:
        return int(args.power_parallel_total_slots)
    return default_power_parallel_total_slots()


def resolve_power_parallel_workers(args: argparse.Namespace, n_jobs: int) -> int:
    """Worker count for concurrent power points (same rule as single-stage)."""
    total_slots = resolve_power_parallel_total_slots(args)
    if args.power_parallel_workers is not None:
        workers = max(1, int(args.power_parallel_workers))
    elif total_slots is not None:
        workers = max(1, int(total_slots) // int(args.thread_count))
    else:
        workers = 1
    return max(1, min(workers, max(1, int(n_jobs))))


def resolve_tm_fraction(args: argparse.Namespace, params: dict[str, Any]) -> float:
    """Tm fraction used for geometry and spectral kinetics alike."""
    if args.tm_fraction is not None:
        return float(args.tm_fraction)
    return float(params["simulation_defaults"]["tm_fraction_for_semi_empirical"])


def sha256_file_bytes(path: Path) -> str:
    """SHA-256 of a file's raw bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_volatile_identity_keys(payload: Any) -> Any:
    """Drop keys that carry non-physical content (paths, timestamps).

    Identity hashes must change only when the physics changes; absolute
    paths (keys containing "path") and timestamps (keys ending "_at")
    would otherwise poison the hash on every machine or rewrite.
    """
    if isinstance(payload, dict):
        return {
            key: strip_volatile_identity_keys(value)
            for key, value in payload.items()
            if "path" not in str(key).lower() and not str(key).lower().endswith("_at")
        }
    if isinstance(payload, list):
        return [strip_volatile_identity_keys(item) for item in payload]
    return payload


def canonical_payload_sha256(payload: Any) -> str:
    """Deterministic SHA-256 over a payload with volatile keys stripped."""
    canonical = json.dumps(
        json_safe(strip_volatile_identity_keys(payload)),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def geometry_logical_sha256(np_db_path: Path) -> str:
    """Logical geometry identity from ordered species/site rows.

    Hashes the physical content (species degrees of freedom, site
    coordinates, site species assignment) rather than the SQLite file
    bytes, because harmless encoding/layout differences across rebuilds
    may alter the file hash without changing the geometry.
    """
    con = sqlite3.connect(np_db_path)
    try:
        species = con.execute(
            "SELECT species_id, degrees_of_freedom FROM species ORDER BY species_id"
        ).fetchall()
        sites = con.execute(
            "SELECT site_id, x, y, z, species_id FROM sites ORDER BY site_id"
        ).fetchall()
    finally:
        con.close()
    return canonical_payload_sha256({"species": species, "sites": sites})


def adaptive_physics_identity(
    args: argparse.Namespace,
    params: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Everything that must match exactly for an adaptive resume to be safe."""
    npmc_path = Path(args.npmc_command)
    return {
        "resolved_config": {
            key: value for key, value in config.items() if key != "mode_defaults"
        },
        # Content hashes: editing the parameter JSON or swapping the NPMC
        # binary is rejected even when every CLI value is unchanged.
        "params_sha256": sha256_file_bytes(Path(args.params)),
        "npmc_sha256": sha256_file_bytes(npmc_path) if npmc_path.is_file() else None,
        "tm_fraction": resolve_tm_fraction(args, params),
        "include_zero_rates": bool(args.include_zero_rates),
        "base_seed": int(args.base_seed),
        "doping_seed": int(args.doping_seed),
        "core_radius_a": float(args.core_radius_a),
        "shell_thickness_a": float(args.shell_thickness_a),
        "pilot": {
            "power_min": float(args.pilot_power_min),
            "power_max": float(args.pilot_power_max),
            "power_count": int(args.pilot_power_count),
            "step_cutoff": int(args.pilot_step_cutoff),
            "num_sims": int(args.pilot_num_sims),
            "terminal_fraction": float(args.pilot_terminal_fraction),
        },
        "refinement": {
            "power_count": int(args.refine_power_count),
            "half_width_decades": float(args.refine_half_width_decades),
            "min_power_gap_fraction": float(args.refine_min_power_gap_fraction),
            "num_sims": int(args.refine_num_sims),
            "center": (
                None if args.refine_center is None else float(args.refine_center)
            ),
            "transition_min_slope": float(args.transition_min_slope),
            "initial_step_cutoff": int(args.adaptive_refine_initial_step_cutoff),
        },
        # Stopping-policy knobs (extension_steps, max_step_cutoff) are
        # excluded from the identity: they bound the search but do not
        # change any recorded verdict, so raising the cap after a "capped"
        # outcome is a valid resume. They are still recorded in summaries.
        "convergence": {
            key: value
            for key, value in resolve_convergence_parameters(args).to_json().items()
            if key not in ("extension_steps", "max_step_cutoff")
        },
        "thread_count": int(args.thread_count),
    }


def new_adaptive_manifest(
    args: argparse.Namespace,
    params: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fresh adaptive_sweep_manifest.json content."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": ADAPTIVE_MANIFEST_SCHEMA_VERSION,
        "workflow_mode": "adaptive-two-stage",
        "created_at": now,
        "updated_at": now,
        "command_line": list(sys.argv),
        "identity": adaptive_physics_identity(args, params, config),
        "pilot": {
            "status": "in_progress",
            "points": {},
        },
        "detection": None,
        "refinement": {
            "status": "pending",
            "grid": None,
            "points": {},
        },
        "status": "in_progress",
        "failures": [],
        "established_geometry_sha256": None,
    }


def assign_power_dir_name(manifest: dict[str, Any], power: float) -> str:
    """Deterministic flat power directory name with collision disambiguation.

    Names stay rounded for readability; the stable full-precision power ID
    is always stored in the manifest alongside, so rounding never causes
    confusion between two distinct powers.
    """
    power_id = stable_power_id(power)
    used: dict[str, str] = {}
    for stage_points in (
        manifest["pilot"]["points"],
        manifest["refinement"]["points"],
    ):
        for pid, record in stage_points.items():
            used[record["dir"]] = pid
    base = f"power_{power:.6g}"
    if base not in used or used[base] == power_id:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used and used[f"{base}_{suffix}"] != power_id:
        suffix += 1
    return f"{base}_{suffix}"


def find_point_record(
    manifest: dict[str, Any], power_id: str
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Locate a power point record across stages."""
    for stage in ("pilot", "refinement"):
        record = manifest[stage]["points"].get(power_id)
        if record is not None:
            return stage, record
    return None, None


def run_adaptive_power_point(
    *,
    power: float,
    power_id: str,
    stage: str,
    num_sims: int,
    initial_step_cutoff: int,
    adaptive_convergence: bool,
    dir_name: str,
    output_root: Path,
    params: dict[str, Any],
    source_np_db_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    trajectory_archive_root: Path,
) -> dict[str, Any]:
    """Run (or resume) one adaptive-mode power point.

    Resume rules: an existing adaptive_run_state.json must match the
    requested power, base seed, physics identity (including parameter-file
    and NPMC-binary content hashes), stored logical geometry hash, and
    stored interaction-manifest hash; the requested seed count must not
    shrink. Completed trajectory databases are never deleted or rewritten —
    a larger absolute step cutoff resumes via NPMC checkpoint=1, and a
    larger seed count appends new seeds (NPMC seeds are
    base_seed .. base_seed + number_of_simulations - 1). Per-seed-only
    extension is not supported by the NPMC CLI, so all seeds are resumed;
    this is a documented, deliberate limitation.

    Crash recovery: pending_extension records an unfinished NPMC call. On
    resume, trajectory maxima are compared against NPMC's interrupt_cutoff
    checkpoints; rows beyond the last checkpoint (SIGKILL) are refused,
    otherwise the extension is rerun idempotently. current_step_cutoff is
    committed only after the database is verified to have reached it, and
    a convergence pass streak increments only for a strictly newer
    per-seed checkpoint identity, so identical data never counts twice.

    Adaptive mode keeps the trajectory DB inside the power directory (no
    node-local staging) so interrupted runs resume without depending on
    node-local scratch state.
    """
    label = f"[{stage} {power:.6g}]"
    result: dict[str, Any] = {
        "power": float(power),
        "power_id": power_id,
        "stage": stage,
        "status": "failed",
        "summary": None,
        "build_record": None,
        "error": None,
    }
    output_dir = output_root / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = output_dir / "adaptive_run_state.json"
    np_db_path = output_dir / "np.sqlite"
    initial_state_db_path = output_dir / "initial_state.sqlite"
    manifest_path = output_dir / "npt_interaction_manifest.json"
    summary_path = output_dir / "npt_run_summary.json"
    conv_params = resolve_convergence_parameters(args)
    physics_identity = adaptive_physics_identity(args, params, config)

    try:
        run_state = load_json_file(run_state_path)
        if run_state is not None:
            if stable_power_id(run_state["power"]) != power_id:
                raise ValueError(
                    f"{label} run state power {run_state['power']!r} does not match "
                    f"requested power {power!r}; refusing to resume"
                )
            if int(run_state["base_seed"]) != int(args.base_seed):
                raise ValueError(
                    f"{label} base seed mismatch on resume "
                    f"({run_state['base_seed']} != {args.base_seed})"
                )
            if json_safe(run_state["physics_identity"]) != json_safe(physics_identity):
                raise ValueError(
                    f"{label} physics-affecting parameters differ from the stored "
                    "run state; use a new output location instead of resuming"
                )
            if int(num_sims) < int(run_state["num_sims"]):
                raise ValueError(
                    f"{label} requested num_sims {num_sims} is below the stored "
                    f"{run_state['num_sims']}; seed count must not shrink"
                )
            interactions_manifest = load_json_file(manifest_path)
            if interactions_manifest is None:
                raise FileNotFoundError(
                    f"{label} missing rate manifest {manifest_path} for resume"
                )
            manifest = interactions_manifest
            interactions = manifest["interactions"]
            stored_geometry_hash = run_state.get("geometry_sha256")
            if stored_geometry_hash is not None and stored_geometry_hash != (
                geometry_logical_sha256(np_db_path)
            ):
                raise ValueError(
                    f"{label} identity component changed: geometry content "
                    "(geometry_sha256 mismatch); use a new output location"
                )
            stored_manifest_hash = run_state.get("interaction_manifest_sha256")
            if stored_manifest_hash is not None and stored_manifest_hash != (
                canonical_payload_sha256(manifest)
            ):
                raise ValueError(
                    f"{label} identity component changed: interaction manifest "
                    "content (interaction_manifest_sha256 mismatch); use a new "
                    "output location"
                )
        else:
            print(f"{label} building interaction network", flush=True)
            interactions, manifest = build_custom_interactions(
                params=params,
                source_np_db_path=source_np_db_path,
                excitation_power=float(power),
                include_zero_rates=bool(args.include_zero_rates),
                tm_fraction=(
                    None if args.tm_fraction is None else float(args.tm_fraction)
                ),
                config=config,
            )
            write_json_atomic(manifest_path, manifest)
            write_custom_npmc_databases(
                source_np_db_path=source_np_db_path,
                output_dir=output_dir,
                interactions=interactions,
                interaction_radius_bound_nm=float(
                    params["simulation_defaults"]["interaction_radius_bound_nm"]
                ),
                distance_factor_type=str(
                    params["simulation_defaults"]["distance_factor_type"]
                ),
            )
            run_state = {
                "schema_version": 2,
                "power": float(power),
                "power_id": power_id,
                "stages": [],
                "status": "in_progress",
                "base_seed": int(args.base_seed),
                "num_sims": 0,
                "physics_identity": physics_identity,
                "geometry_sha256": geometry_logical_sha256(np_db_path),
                "interaction_manifest_sha256": canonical_payload_sha256(manifest),
                "initial_step_cutoff": int(initial_step_cutoff),
                "current_step_cutoff": None,
                "pending_extension": None,
                "analysis_index": None,
                "extension_count": 0,
                "seed_pass_streaks": {},
                "seed_last_counted_checkpoint": {},
                "checkpoints": [],
                "convergence": None,
                "error": None,
            }
            write_json_atomic(run_state_path, run_state)

        if stage not in run_state["stages"]:
            run_state["stages"].append(stage)
        run_state["status"] = "in_progress"
        run_state["error"] = None

        build_record = {
            "excitation_power_w_cm2": float(power),
            "power_id": power_id,
            "output_dir": str(output_dir.resolve()),
            "interaction_count": int(len(interactions)),
            "manifest_path": str(manifest_path.resolve()),
            "np_db_path": str(np_db_path.resolve()),
            "initial_state_db_path": str(initial_state_db_path.resolve()),
            "geometry_sha256": run_state.get("geometry_sha256"),
        }
        if args.dry_run:
            run_state["status"] = "dry_run"
            write_json_atomic(run_state_path, run_state)
            result.update(
                {"status": "dry_run", "build_record": build_record}
            )
            return result

        all_ids = sorted({int(row["interaction_id"]) for row in interactions})

        def verify_and_commit_extension(target_cutoff: int, target_sims: int) -> None:
            """Commit a cutoff only after the DB provably reached it.

            current_step_cutoff is updated solely on this path: NPMC must
            have returned successfully AND every expected seed's trajectory
            must contain a step >= the absolute target cutoff.
            """
            con = sqlite3.connect(initial_state_db_path)
            try:
                identities = load_seed_checkpoint_identities(con, all_ids)
            finally:
                con.close()
            missing = [
                seed
                for seed in range(int(args.base_seed), int(args.base_seed) + target_sims)
                if identities.get(seed) is None
                or int(identities[seed]["max_step"]) < int(target_cutoff)
            ]
            if missing:
                raise RuntimeError(
                    f"{label} NPMC returned but seed(s) {missing} did not reach "
                    f"{target_cutoff} steps; refusing to commit the cutoff"
                )
            run_state["pending_extension"] = None
            run_state["current_step_cutoff"] = int(target_cutoff)
            run_state["num_sims"] = int(target_sims)
            write_json_atomic(run_state_path, run_state)

        def execute_extension(target_cutoff: int, target_sims: int) -> None:
            run_state["pending_extension"] = {
                "target_step_cutoff": int(target_cutoff),
                "num_sims": int(target_sims),
                "state": "extension_running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_json_atomic(run_state_path, run_state)
            run_npmc(
                np_db_path,
                initial_state_db_path,
                output_dir,
                args,
                step_cutoff=int(target_cutoff),
                num_sims=int(target_sims),
            )
            verify_and_commit_extension(target_cutoff, target_sims)

        if run_state["pending_extension"] is not None:
            # Crash recovery: a previous invocation died during NPMC. A
            # SIGTERM checkpoints each seed cleanly; a SIGKILL can leave
            # trajectory rows beyond the last checkpoint, which NPMC would
            # resume into duplicate step ranges. Detect and refuse instead.
            pending = run_state["pending_extension"]
            pending_cutoff = int(pending["target_step_cutoff"])
            pending_sims = int(pending["num_sims"])
            con = sqlite3.connect(initial_state_db_path)
            try:
                identities = load_seed_checkpoint_identities(con, all_ids)
                interrupt = load_interrupt_cutoffs(con)
            finally:
                con.close()
            if not interrupt and run_state["current_step_cutoff"] is not None:
                raise RuntimeError(
                    f"{label} interrupt_cutoff is empty although a verified "
                    "cutoff exists; cannot assess the interrupted NPMC state"
                )
            orphans = [
                seed
                for seed, identity in identities.items()
                if (seed in interrupt and int(identity["max_step"]) > interrupt[seed][0])
                or (seed not in interrupt and bool(interrupt))
            ]
            if orphans:
                raise RuntimeError(
                    f"{label} seed(s) {orphans} have trajectory rows beyond the "
                    "last NPMC checkpoint (SIGKILL mid-extension). Resuming "
                    "would duplicate steps; refusing. Remove the orphan rows "
                    "manually or use a new output location."
                )
            missing = [
                seed
                for seed in range(int(args.base_seed), int(args.base_seed) + pending_sims)
                if identities.get(seed) is None
                or int(identities[seed]["max_step"]) < pending_cutoff
            ]
            if missing:
                print(
                    f"{label} resuming interrupted NPMC extension to "
                    f"{pending_cutoff} steps ({len(missing)} seed(s) behind)",
                    flush=True,
                )
                run_npmc(
                    np_db_path,
                    initial_state_db_path,
                    output_dir,
                    args,
                    step_cutoff=pending_cutoff,
                    num_sims=pending_sims,
                )
            verify_and_commit_extension(pending_cutoff, pending_sims)

        target_num_sims = max(int(num_sims), int(run_state["num_sims"]))
        needed_cutoff = max(
            int(run_state["current_step_cutoff"] or 0), int(initial_step_cutoff)
        )
        if (
            run_state["current_step_cutoff"] is None
            or needed_cutoff > int(run_state["current_step_cutoff"])
            or target_num_sims > int(run_state["num_sims"])
        ):
            print(
                f"{label} running NPMC to {needed_cutoff} steps with "
                f"{target_num_sims} seed(s)",
                flush=True,
            )
            execute_extension(needed_cutoff, target_num_sims)

        current_step_cutoff = int(run_state["current_step_cutoff"])

        def evaluate_and_record() -> dict[str, Any]:
            evaluation = evaluate_run_convergence(
                initial_state_db_path, interactions, conv_params
            )
            if (
                evaluation["analysis"]["index_created"]
                and run_state["analysis_index"] is None
            ):
                # Record that the analysis index was built after trajectory
                # data already existed (review requirement).
                run_state["analysis_index"] = {
                    "name": TRAJECTORY_ANALYSIS_INDEX,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "created_at_step_cutoff": int(current_step_cutoff),
                }
            return evaluation

        streaks = {
            int(seed): int(streak)
            for seed, streak in run_state["seed_pass_streaks"].items()
        }
        last_counted = {
            int(seed): (int(value[0]), float(value[1]))
            for seed, value in run_state["seed_last_counted_checkpoint"].items()
        }
        extension_count = int(run_state["extension_count"])

        if adaptive_convergence:
            # Refinement stage: adaptive terminal-block convergence with
            # checkpoint extensions, resumable via adaptive_run_state.json.
            status = "running"
            while True:
                evaluation = evaluate_and_record()
                streaks, last_counted, all_converged = update_pass_streaks(
                    streaks, evaluation, conv_params.required_passes, last_counted
                )
                run_state["seed_pass_streaks"] = {
                    str(seed): int(streak) for seed, streak in sorted(streaks.items())
                }
                run_state["seed_last_counted_checkpoint"] = {
                    str(seed): [int(value[0]), float(value[1])]
                    for seed, value in sorted(last_counted.items())
                }
                run_state["checkpoints"].append(
                    {
                        "stage": stage,
                        "step_cutoff": int(current_step_cutoff),
                        "seed_statuses": {
                            str(seed): report["status"]
                            for seed, report in sorted(evaluation["seeds"].items())
                        },
                        "seed_pass_streaks": dict(run_state["seed_pass_streaks"]),
                        "checkpoint_identities": {
                            str(seed): dict(identity)
                            for seed, identity in sorted(
                                evaluation["checkpoint_identities"].items()
                            )
                        },
                        "all_converged": bool(all_converged),
                    }
                )
                write_json_atomic(run_state_path, run_state)
                if all_converged:
                    status = "converged"
                    break
                if current_step_cutoff >= conv_params.max_step_cutoff:
                    status = derive_run_convergence_status(
                        evaluation, all_converged=False, hit_cap=True
                    )
                    break
                n_not_passed = sum(
                    1
                    for report in evaluation["seeds"].values()
                    if not report["passed"]
                )
                next_cutoff = min(
                    current_step_cutoff + conv_params.extension_steps,
                    conv_params.max_step_cutoff,
                )
                extension_count += 1
                run_state["extension_count"] = int(extension_count)
                print(
                    f"{label} {n_not_passed}/{len(evaluation['seeds'])} seed(s) "
                    f"not converged; extending to {next_cutoff} steps",
                    flush=True,
                )
                execute_extension(next_cutoff, int(run_state["num_sims"]))
                current_step_cutoff = int(next_cutoff)
            estimate_window = {
                "type": "terminal_blocks",
                "algorithm": evaluation["algorithm"],
            }
            estimate_intervals = block_intervals_from_evaluation(evaluation)
        else:
            # Pilot stage: fixed cutoff, no adaptive extensions. The block
            # analysis runs once for diagnostics; detection observables come
            # from the terminal fraction of each trajectory.
            evaluation = evaluate_and_record()
            if not evaluation["seeds"]:
                raise RuntimeError(f"{label} produced no trajectory rows")
            run_state["checkpoints"].append(
                {
                    "stage": stage,
                    "step_cutoff": int(current_step_cutoff),
                    "seed_statuses": {
                        str(seed): report["status"]
                        for seed, report in sorted(evaluation["seeds"].items())
                    },
                    "seed_pass_streaks": {},
                    "checkpoint_identities": {
                        str(seed): dict(identity)
                        for seed, identity in sorted(
                            evaluation["checkpoint_identities"].items()
                        )
                    },
                    "all_converged": bool(evaluation["all_passed"]),
                    "extensions_disabled": True,
                }
            )
            write_json_atomic(run_state_path, run_state)
            status = derive_run_convergence_status(
                evaluation,
                all_converged=bool(evaluation["all_passed"]),
                hit_cap=False,
            )
            fraction = float(args.pilot_terminal_fraction)
            estimate_window = {"type": "terminal_fraction", "fraction": fraction}
            estimate_intervals = {
                seed: [
                    (
                        float(identity["max_time_s"]) * (1.0 - fraction),
                        float(identity["max_time_s"]),
                    )
                ]
                for seed, identity in evaluation["checkpoint_identities"].items()
            }

        convergence = {
            "algorithm": conv_params.algorithm,
            "semantics": conv_params.semantics,
            "status": status,
            "extensions_disabled": not adaptive_convergence,
            "initial_step_cutoff": int(initial_step_cutoff),
            "final_step_cutoff": int(current_step_cutoff),
            "extension_count": int(extension_count),
            "parameters": conv_params.to_json(),
            "seeds": {
                str(seed): report
                for seed, report in sorted(evaluation["seeds"].items())
            },
            "seed_pass_streaks": {
                str(seed): int(streak) for seed, streak in sorted(streaks.items())
            },
            "checkpoints": list(run_state["checkpoints"]),
            "mode_fractions": summarize_mode_fractions(evaluation),
        }

        # Finalization: one replay pass over the full trajectory for the
        # whole-run summary, the once-per-run N4 validation attachment, and
        # the terminal estimate. This is the only replay in v2; it never
        # runs per checkpoint.
        replay = replay_trajectories(
            initial_state_db_path,
            interactions,
            intervals_by_seed=estimate_intervals or None,
        )
        attach_n4_validation(evaluation, replay, conv_params)
        terminal_estimate = terminal_estimate_from_interval_stats(
            replay.get("interval_stats", {}),
            int(replay["tm_site_count"]),
            estimate_window,
        )
        convergence["seed_heterogeneity"] = seed_heterogeneity_warning(
            terminal_estimate
        )
        heterogeneity = convergence["seed_heterogeneity"] or {}
        if heterogeneity.get("mixed_basin_warning"):
            print(
                f"{label} WARNING: per-seed terminal rad800 rates span "
                f"{heterogeneity['max_min_seed_rate_ratio']:.1f}x; likely mixed "
                "dark/bright basins. Convergence semantics are '"
                f"{conv_params.semantics}', not an equilibrium average.",
                flush=True,
            )

        summary = summarize_run(
            replay=replay,
            interactions=interactions,
            n_sites=int(manifest["geometry"]["ion_count"]),
            excitation_power=float(power),
            manifest=manifest,
            simulation_cutoff_mode="steps",
            simulation_step_cutoff=int(current_step_cutoff),
            simulation_time_cutoff_s=None,
            stage=stage,
            terminal_estimate=terminal_estimate,
        )
        summary["stages"] = list(run_state["stages"])
        summary["power_id"] = power_id
        summary["convergence"] = convergence
        # Stage-specific summaries stay immutable once written; the shared
        # canonical file mirrors the latest stage for downstream tooling.
        write_json_atomic(output_dir / f"npt_run_summary_{stage}.json", summary)
        write_json_atomic(summary_path, summary)

        try:
            with MATPLOTLIB_LOCK:
                plot_power_spectrum(
                    params=params,
                    excitation_power=float(power),
                    tm_fraction=(
                        None if args.tm_fraction is None else float(args.tm_fraction)
                    ),
                    config=config,
                    summary=summary,
                    interactions=interactions,
                    output_dir=output_dir,
                )
        except Exception as exc:
            print(f"{label} WARNING: emission spectrum failed: {exc}", flush=True)

        archived_path = finalize_initial_state_database(
            initial_state_db_path=initial_state_db_path,
            output_dir=output_dir,
            archive_root=trajectory_archive_root,
        )
        build_record["archived_initial_state_db_path"] = str(archived_path)
        run_state["convergence"] = {
            "status": convergence["status"],
            "final_step_cutoff": int(current_step_cutoff),
            "extension_count": int(convergence["extension_count"]),
        }
        run_state["status"] = "done"
        write_json_atomic(run_state_path, run_state)
        print(
            f"{label} finished with convergence status "
            f"{convergence['status']} at {current_step_cutoff} steps",
            flush=True,
        )
        result.update(
            {
                "status": "done",
                "summary": summary,
                "build_record": build_record,
            }
        )
        return result
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"{label} FAILED: {error_text}", flush=True)
        failed_state = load_json_file(run_state_path)
        if failed_state is not None:
            failed_state["status"] = "failed"
            failed_state["error"] = error_text
            write_json_atomic(run_state_path, failed_state)
        result["error"] = error_text
        return result


def pilot_point_signal(
    summary: dict[str, Any],
    mode_threshold_log10: float | None,
) -> dict[str, Any]:
    """Robust per-power transition signal from a pilot summary.

    Includes the raw per-seed rates/exposures alongside the aggregate so
    the manifest alone can reproduce every detection input later.
    """
    estimate = summary.get("terminal_estimate") or {}
    seed_values = estimate.get("seed_values") or {}
    rates = seed_values.get("rad800_events_per_particle_s") or {}
    exposures = seed_values.get("exposure_s") or {}
    seed_rates: list[float] = []
    seed_durations: list[float] = []
    for seed, rate in rates.items():
        duration = exposures.get(seed)
        if duration is None:
            continue
        seed_rates.append(float(rate))
        seed_durations.append(float(duration))
    signal = aggregate_pilot_power_signal(
        seed_rates, seed_durations, mode_threshold_log10
    )
    signal["seed_rates_per_particle_s"] = {
        str(seed): float(rate) for seed, rate in rates.items()
    }
    signal["seed_exposures_s"] = {
        str(seed): float(duration) for seed, duration in exposures.items()
    }
    return signal


def run_adaptive_points_parallel(
    jobs: list[dict[str, Any]],
    *,
    workers: int,
    kwargs: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Run adaptive power points concurrently, yielding (job, result)."""
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(run_adaptive_power_point, **{**kwargs, **job}): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                yield future_to_job[future], future.result()
    else:
        for job in jobs:
            result = run_adaptive_power_point(**{**kwargs, **job})
            yield job, result


def derive_stage_status(points: dict[str, Any]) -> str:
    """Stage status derived from its point records (never unconditional).

    "done" only when every point is done; "failed" when failures remain
    after the current attempt; otherwise "in_progress".
    """
    statuses = {str(record["status"]) for record in points.values()}
    if statuses and statuses <= {"done"}:
        return "done"
    if "failed" in statuses:
        return "failed"
    return "in_progress"


def record_point_failure(
    manifest: dict[str, Any],
    stage: str,
    power_id: str,
    record: dict[str, Any],
    error: str,
) -> None:
    """Mark a point failed and append an unresolved failure record."""
    record["status"] = "failed"
    record["error"] = error
    manifest["failures"].append(
        {
            "stage": stage,
            "power_id": power_id,
            "power": float(record["power"]),
            "error": error,
            "resolved": False,
        }
    )


def record_point_success(
    manifest: dict[str, Any],
    stage: str,
    power_id: str,
    record: dict[str, Any],
) -> None:
    """Mark a point done and resolve its earlier failure records."""
    record["status"] = "done"
    record["error"] = None
    for failure in manifest["failures"]:
        if (
            failure["stage"] == stage
            and failure["power_id"] == power_id
            and not failure["resolved"]
        ):
            failure["resolved"] = True


def stored_run_state_num_sims(output_root: Path, dir_name: str) -> int:
    """Seed count recorded in a point's run state (0 when absent)."""
    run_state = load_json_file(output_root / dir_name / "adaptive_run_state.json")
    if not run_state:
        return 0
    return int(run_state.get("num_sims") or 0)


def note_established_geometry(
    manifest: dict[str, Any], result: dict[str, Any]
) -> None:
    """Pin the shared geometry identity from the first built power point.

    Every later point must reproduce the same logical geometry hash; a
    mismatch means the source geometry changed mid-workflow.
    """
    build_record = result.get("build_record") or {}
    geometry_hash = build_record.get("geometry_sha256")
    if not geometry_hash:
        return
    established = manifest.get("established_geometry_sha256")
    if established is None:
        manifest["established_geometry_sha256"] = geometry_hash
    elif established != geometry_hash:
        raise ValueError(
            "identity component changed: geometry_sha256 differs between "
            "power points; the source geometry changed mid-workflow"
        )


def run_adaptive_two_stage(
    args: argparse.Namespace,
    params: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    trajectory_archive_root: Path,
    source_np_db_path: Path,
) -> None:
    """Integrated pilot + detection + refinement workflow (one command)."""
    conv_params = resolve_convergence_parameters(args)
    manifest_path = output_root / "adaptive_sweep_manifest.json"
    manifest = load_json_file(manifest_path)
    identity = adaptive_physics_identity(args, params, config)
    if manifest is not None:
        if json_safe(manifest.get("identity")) != json_safe(identity):
            raise ValueError(
                "adaptive_sweep_manifest.json was created with different "
                "physics-affecting parameters; refusing to resume. "
                "Choose a new --output-root for a different configuration."
            )
        print("Resuming adaptive two-stage workflow from manifest", flush=True)
    else:
        manifest = new_adaptive_manifest(args, params, config)

    def save_manifest() -> None:
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(manifest_path, manifest)

    save_manifest()

    # ---- Stage 1: pilot scan ---------------------------------------------
    pilot_powers = build_pilot_power_grid(
        float(args.pilot_power_min),
        float(args.pilot_power_max),
        int(args.pilot_power_count),
    )
    pilot_points = manifest["pilot"]["points"]
    for power in pilot_powers:
        power_id = stable_power_id(power)
        if power_id not in pilot_points:
            pilot_points[power_id] = {
                "power": float(power),
                "dir": assign_power_dir_name(manifest, float(power)),
                "status": "pending",
                "summary_path": None,
                "signal": None,
                "error": None,
            }
    save_manifest()

    shared_kwargs = {
        "output_root": output_root,
        "params": params,
        "source_np_db_path": source_np_db_path,
        "config": config,
        "args": args,
        "trajectory_archive_root": trajectory_archive_root,
    }

    if manifest["pilot"]["status"] != "done":
        jobs = []
        # Highest power first: bright points finish fast, giving early signal.
        for power_id, record in sorted(
            pilot_points.items(), key=lambda item: item[1]["power"], reverse=True
        ):
            # Schedule every pending or failed point whose configuration
            # identity is valid; only frozen, summarized results are skipped.
            if (
                record["status"] == "done"
                and record.get("signal") is not None
                and record.get("summary_path")
                and Path(record["summary_path"]).exists()
            ):
                continue
            jobs.append(
                {
                    "power": float(record["power"]),
                    "power_id": power_id,
                    "stage": "pilot",
                    # A shared power directory may already hold more seeds
                    # from the refinement stage; seed count must not shrink.
                    "num_sims": max(
                        int(args.pilot_num_sims),
                        stored_run_state_num_sims(output_root, record["dir"]),
                    ),
                    "initial_step_cutoff": int(args.pilot_step_cutoff),
                    "adaptive_convergence": False,
                    "dir_name": record["dir"],
                }
            )
        workers = resolve_power_parallel_workers(args, max(1, len(jobs)))
        for job, point_result in run_adaptive_points_parallel(
            jobs, workers=workers, kwargs=shared_kwargs
        ):
            record = pilot_points[job["power_id"]]
            if point_result["status"] in ("done", "dry_run"):
                record_point_success(manifest, "pilot", job["power_id"], record)
                summary = point_result.get("summary")
                if summary is not None:
                    summary_path = (
                        output_root / record["dir"] / "npt_run_summary_pilot.json"
                    ).resolve()
                    record["summary_path"] = str(summary_path)
                    # Freeze the detection input at completion time: the
                    # signal is never recomputed from mutable summaries.
                    record["signal"] = pilot_point_signal(
                        summary, conv_params.mode_threshold_log10
                    )
                    record["summary_sha256"] = sha256_file_bytes(summary_path)
                note_established_geometry(manifest, point_result)
            else:
                record_point_failure(
                    manifest, "pilot", job["power_id"], record, point_result["error"]
                )
            save_manifest()
        manifest["pilot"]["status"] = derive_stage_status(pilot_points)
        save_manifest()

    if args.dry_run:
        print(
            "Dry run: pilot databases and manifests written; skipping "
            "detection, refinement, and plotting.",
            flush=True,
        )
        save_manifest()
        return

    # ---- Stage 2 center detection -----------------------------------------
    # Detection runs at most once and is frozen in the manifest afterwards;
    # it consumes the per-point signals frozen at pilot completion, never
    # re-read mutable summaries, so resume cannot shift the bracket/center.
    detection = manifest.get("detection")
    if detection is None:
        detection_powers: list[float] = []
        detection_signals: list[float | None] = []
        detection_inputs: dict[str, Any] = {}
        for power_id, record in sorted(
            pilot_points.items(), key=lambda item: item[1]["power"]
        ):
            signal = record.get("signal")
            if record["status"] != "done" or signal is None:
                continue
            detection_powers.append(float(record["power"]))
            detection_signals.append(signal["median_log10"])
            detection_inputs[power_id] = {
                "power": float(record["power"]),
                "signal": signal,
                "summary_sha256": record.get("summary_sha256"),
                "window": {
                    "type": "terminal_fraction",
                    "fraction": float(args.pilot_terminal_fraction),
                },
            }
        n_valid = len(detection_powers)
        if n_valid >= MIN_DETECTION_PILOT_POINTS:
            detection = detect_avalanche_transition(
                detection_powers,
                detection_signals,
                min_slope=float(args.transition_min_slope),
                manual_center=(
                    None if args.refine_center is None else float(args.refine_center)
                ),
            )
            detection["partial_pilot"] = bool(n_valid < len(pilot_points))
            if detection["partial_pilot"]:
                print(
                    f"WARNING: detection uses {n_valid}/{len(pilot_points)} "
                    "pilot points (partial pilot after failures).",
                    flush=True,
                )
        else:
            detection = {
                "n_pilot_points": len(pilot_points),
                "n_valid_points": n_valid,
                "transition_detected": False,
                "reason": "insufficient_valid_pilot_points",
                "center": None,
                "geometric_center": None,
                "bracket_powers": None,
                "max_slope": None,
                "edge_detected": False,
                "partial_pilot": bool(n_valid < len(pilot_points)),
            }
        detection["inputs"] = detection_inputs
        manifest["detection"] = detection
        save_manifest()
    if detection.get("edge_detected"):
        print(
            "WARNING: maximum pilot slope is at the edge of the pilot range; "
            "consider widening --pilot-power-min/--pilot-power-max.",
            flush=True,
        )
    if not detection.get("transition_detected"):
        print(
            f"WARNING: no avalanche transition detected "
            f"(reason={detection.get('reason')}, "
            f"max slope={detection.get('max_slope')}).",
            flush=True,
        )

    # ---- Stage 2: refinement -----------------------------------------------
    refinement_points = manifest["refinement"]["points"]
    refine_initial_cutoff = int(args.adaptive_refine_initial_step_cutoff)
    if detection.get("center") is not None and manifest["refinement"]["status"] not in (
        "done",
        "skipped",
    ):
        if manifest["refinement"]["grid"] is None:
            # Build the grid once; it is immutable on resume afterwards.
            done_pilot_powers = [
                float(record["power"])
                for power_id, record in pilot_points.items()
                if record["status"] == "done"
            ]
            grid = build_refinement_power_grid(
                center=float(detection["center"]),
                half_width_decades=float(args.refine_half_width_decades),
                power_count=int(args.refine_power_count),
                pilot_min=float(args.pilot_power_min),
                pilot_max=float(args.pilot_power_max),
                bracket_powers=detection.get("bracket_powers"),
                min_gap_fraction=float(args.refine_min_power_gap_fraction),
                existing_powers=done_pilot_powers,
            )
            manifest["refinement"]["grid"] = grid
            for point in grid["points"]:
                power = float(point["power"])
                power_id = stable_power_id(power)
                _stage, existing = find_point_record(manifest, power_id)
                if power_id not in refinement_points:
                    if existing is not None:
                        dir_name = existing["dir"]
                        reused_from = power_id
                    else:
                        dir_name = assign_power_dir_name(manifest, power)
                        reused_from = None
                    refinement_points[power_id] = {
                        "power": power,
                        "dir": dir_name,
                        "kind": point["kind"],
                        "merged_kinds": point["merged_kinds"],
                        "reused_from_pilot_power_id": reused_from,
                        "status": "pending",
                        "summary_path": None,
                        "error": None,
                    }
            manifest["refinement"]["status"] = "in_progress"
            save_manifest()

        jobs = []
        # Highest power first, matching the pilot scheduling order.
        for power_id, record in sorted(
            refinement_points.items(), key=lambda item: item[1]["power"], reverse=True
        ):
            if record["status"] == "done" and (
                record.get("summary_path")
                and Path(record["summary_path"]).exists()
            ):
                continue
            jobs.append(
                {
                    "power": float(record["power"]),
                    "power_id": power_id,
                    "stage": "refinement",
                    # A shared power directory may already hold more seeds
                    # from the pilot stage; seed count must not shrink.
                    "num_sims": max(
                        int(args.refine_num_sims),
                        stored_run_state_num_sims(output_root, record["dir"]),
                    ),
                    "initial_step_cutoff": refine_initial_cutoff,
                    "adaptive_convergence": True,
                    "dir_name": record["dir"],
                }
            )
        workers = resolve_power_parallel_workers(args, max(1, len(jobs)))
        for job, point_result in run_adaptive_points_parallel(
            jobs, workers=workers, kwargs=shared_kwargs
        ):
            record = refinement_points[job["power_id"]]
            if point_result["status"] == "done":
                record_point_success(manifest, "refinement", job["power_id"], record)
                record["summary_path"] = str(
                    (
                        output_root / record["dir"] / "npt_run_summary_refinement.json"
                    ).resolve()
                )
                note_established_geometry(manifest, point_result)
            else:
                record_point_failure(
                    manifest,
                    "refinement",
                    job["power_id"],
                    record,
                    point_result["error"],
                )
            save_manifest()
        manifest["refinement"]["status"] = derive_stage_status(refinement_points)
        save_manifest()
    elif detection.get("center") is None:
        manifest["refinement"]["status"] = "skipped"
        save_manifest()
        print(
            "Refinement skipped: no transition center available. Provide "
            "--refine-center to force a manual center.",
            flush=True,
        )

    # ---- Combined summary, config, and plot --------------------------------
    summaries: list[dict[str, Any]] = []
    build_records: list[dict[str, Any]] = []
    seen_power_ids: set[str] = set()
    all_records = []
    for power_id, record in refinement_points.items():
        all_records.append((power_id, record, "refinement"))
    for power_id, record in pilot_points.items():
        if power_id in {pid for pid, _r, _s in all_records}:
            continue
        all_records.append((power_id, record, "pilot"))
    for power_id, record, stage in sorted(
        all_records, key=lambda item: item[1]["power"]
    ):
        if power_id in seen_power_ids:
            continue
        seen_power_ids.add(power_id)
        if record["status"] != "done" or not record.get("summary_path"):
            continue
        summary = load_json_file(Path(record["summary_path"]))
        if summary is None:
            continue
        summaries.append(summary)
        build_records.append(
            {
                "power_index": len(build_records),
                "excitation_power_w_cm2": float(record["power"]),
                "power_id": power_id,
                "stage": stage,
                "output_dir": str((output_root / record["dir"]).resolve()),
                "manifest_path": str(
                    (output_root / record["dir"] / "npt_interaction_manifest.json").resolve()
                ),
            }
        )

    root_config = {
        "schema_version": ADAPTIVE_MANIFEST_SCHEMA_VERSION,
        "workflow_mode": "adaptive-two-stage",
        "profile": params["profile"],
        "params_path": str(Path(args.params).resolve()),
        "source_np_db": str(source_np_db_path.resolve()),
        "command_line": list(sys.argv),
        "identity": identity,
        "established_geometry_sha256": manifest.get("established_geometry_sha256"),
        "pilot": {
            "powers_w_cm2": [float(p) for p in pilot_powers],
            "step_cutoff": int(args.pilot_step_cutoff),
            "num_sims": int(args.pilot_num_sims),
            "terminal_fraction": float(args.pilot_terminal_fraction),
        },
        "detection": detection,
        "refinement": {
            "requested_power_count": int(args.refine_power_count),
            "num_sims": int(args.refine_num_sims),
            "initial_step_cutoff": refine_initial_cutoff,
            "grid": manifest["refinement"]["grid"],
            "reused_power_ids": [
                power_id
                for power_id, record in refinement_points.items()
                if record.get("reused_from_pilot_power_id")
            ],
        },
        "convergence_parameters": conv_params.to_json(),
        "convergence_semantics": conv_params.semantics,
        "failures": manifest["failures"],
    }
    write_json_atomic(output_root / "npt_production_config.json", root_config)

    sweep_summary = {
        "workflow_mode": "adaptive-two-stage",
        "detection": detection,
        "max_local_log_slope_800": detection.get("max_slope"),
        "max_local_log_slope_800_power_w_cm2": (
            detection["bracket_powers"][0] if detection.get("bracket_powers") else None
        ),
        **root_config,
        "build_records": build_records,
        "power_points": summaries,
    }
    write_json_atomic(output_root / "npt_power_sweep_summary.json", sweep_summary)
    plot_avalanche_curve(summaries, output_root, detection=detection)

    # Completion is derived, never asserted: every stage must have reached
    # its terminal state with no unresolved failures left.
    unresolved_failures = [
        failure for failure in manifest["failures"] if not failure.get("resolved")
    ]
    stages_settled = manifest["pilot"]["status"] == "done" and manifest[
        "refinement"
    ]["status"] in ("done", "skipped")
    manifest["status"] = (
        "complete" if stages_settled and not unresolved_failures else "incomplete"
    )
    save_manifest()
    if unresolved_failures:
        print(
            f"Adaptive workflow finished with {len(unresolved_failures)} "
            "unresolved failed power point(s); see "
            "adaptive_sweep_manifest.json. Re-run the same command to retry.",
            flush=True,
        )
    else:
        print("Adaptive two-stage workflow complete.", flush=True)



def plot_avalanche_curve(
    summaries: list[dict[str, Any]],
    output_root: Path,
    detection: dict[str, Any] | None = None,
) -> None:
    """Plot 700 nm and 800 nm avalanche proxies versus excitation power.

    Terminal estimates are preferred whenever a summary provides them
    (schema version 2); older summaries fall back to the whole-run proxy
    fields so old JSON still plots. Pilot/refinement points get
    distinguishable markers, per-seed values are shown as faint points,
    per-power error bars show the seed standard error, and points whose
    convergence status is not "converged" (capped, insufficient_counts,
    insufficient_history, metastable_censored) use open markers. Solid
    segments are only drawn between consecutive points with positive
    values. The detected transition bracket/center is recorded in the
    plot-data JSON and shaded subtly when available.
    """
    if not summaries:
        return
    summaries = sorted(summaries, key=lambda row: float(row["excitation_power_w_cm2"]))

    def terminal_or_whole(row: dict[str, Any], key_800: bool) -> float:
        estimate = row.get("terminal_estimate") or {}
        field = (
            "rad_800_proxy_events_per_particle_s"
            if key_800
            else "rad_700_proxy_events_per_particle_s"
        )
        if estimate.get(field) is not None:
            return float(estimate[field])
        return float(row.get(field, 0.0))

    records: list[dict[str, Any]] = []
    for row in summaries:
        estimate = row.get("terminal_estimate") or {}
        convergence = row.get("convergence") or {}
        status = convergence.get("status")
        seed_values = (estimate.get("seed_values") or {}).get(
            "rad800_events_per_particle_s"
        ) or {}
        records.append(
            {
                "power": float(row["excitation_power_w_cm2"]),
                "rad800": terminal_or_whole(row, True),
                "rad700": terminal_or_whole(row, False),
                "standard_error": estimate.get("standard_error"),
                "status": status,
                "stage": str(row.get("stage", "single")),
                "seed_rad800": {
                    str(seed): float(value) for seed, value in seed_values.items()
                },
            }
        )

    powers = np.asarray([rec["power"] for rec in records], dtype=float)
    emission_800 = np.asarray([rec["rad800"] for rec in records], dtype=float)
    emission_700 = np.asarray([rec["rad700"] for rec in records], dtype=float)
    valid_800 = emission_800 > 0
    valid_700 = emission_700 > 0

    fig, ax = plt.subplots(dpi=300, figsize=(6.4, 4.6))

    def plot_curve(
        values: np.ndarray,
        valid: np.ndarray,
        color: str,
    ) -> None:
        # Solid segments only between consecutive valid points.
        segment_start: int | None = None
        for index in range(len(values) + 1):
            in_segment = index < len(values) and valid[index]
            if in_segment and segment_start is None:
                segment_start = index
            elif not in_segment and segment_start is not None:
                sl = slice(segment_start, index)
                ax.loglog(
                    powers[sl],
                    values[sl],
                    color=color,
                    linewidth=1.8,
                    marker="none",
                )
                segment_start = None
        # Markers: filled for converged/unknown, open for flagged statuses.
        for index, rec in enumerate(records):
            if not valid[index]:
                continue
            flagged = rec["status"] not in (None, "converged")
            marker = "o" if rec["stage"] in ("single", "pilot") else "D"
            ax.loglog(
                [powers[index]],
                [values[index]],
                marker=marker,
                markersize=5.0,
                markerfacecolor="none" if flagged else color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                linewidth=0,
            )

    plot_curve(emission_800, valid_800, "C0")
    plot_curve(emission_700, valid_700, "C1")

    # Per-power uncertainty (seed standard error of the 800 nm proxy).
    sem = np.asarray(
        [
            rec["standard_error"] if rec["standard_error"] is not None else np.nan
            for rec in records
        ],
        dtype=float,
    )
    has_sem = valid_800 & np.isfinite(sem) & (sem > 0)
    if np.any(has_sem):
        lower = np.minimum(sem[has_sem], emission_800[has_sem] * (1.0 - 1e-6))
        ax.errorbar(
            powers[has_sem],
            emission_800[has_sem],
            yerr=np.vstack([lower, sem[has_sem]]),
            fmt="none",
            ecolor="C0",
            elinewidth=1.0,
            alpha=0.55,
            capsize=2.5,
        )

    # Faint per-seed points keep seed heterogeneity visible.
    for index, rec in enumerate(records):
        for seed_value in rec["seed_rad800"].values():
            if seed_value > 0:
                ax.loglog(
                    [powers[index]],
                    [seed_value],
                    marker=".",
                    markersize=3.0,
                    color="C0",
                    alpha=0.30,
                    linewidth=0,
                )

    # Legend proxies for the custom marker drawing above.
    ax.loglog([], [], color="C0", marker="o", linewidth=1.8, markersize=5.0,
              label="800 nm proxy (3H4 radiative)")
    ax.loglog([], [], color="C1", marker="o", linewidth=1.8, markersize=5.0,
              label="700 nm proxy (3F3 radiative)")
    ax.loglog([], [], color="0.4", marker="D", linewidth=0, markersize=5.0,
              label="refinement point")
    ax.loglog([], [], color="0.4", marker="o", linewidth=0, markersize=5.0,
              markerfacecolor="none", label="flagged (non-converged/capped)")

    bracket = detection.get("bracket_powers") if detection else None
    center = detection.get("center") if detection else None
    if bracket and len(bracket) == 2 and all(p > 0 for p in bracket):
        ax.axvspan(
            float(bracket[0]), float(bracket[1]), color="0.85", alpha=0.35, zorder=0
        )
    if center and center > 0:
        ax.axvline(
            float(center), color="0.5", linewidth=0.9, linestyle="--", alpha=0.7, zorder=0
        )

    ax.set_xlabel("Excitation power at 1064 nm (W cm$^{-2}$)", fontsize=13)
    ax.set_ylabel("Radiative events per particle per s", fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.grid(True, which="both", alpha=0.25)

    preferred_ticks = np.asarray(
        [
            3000.0,
            4000.0,
            6000.0,
            8000.0,
            10000.0,
            12000.0,
            14000.0,
            16000.0,
            18000.0,
            20000.0,
            30000.0,
            40000.0,
            50000.0,
        ],
        dtype=float,
    )
    tick_positions = preferred_ticks[
        (preferred_ticks >= powers[0] * 0.98) & (preferred_ticks <= powers[-1] * 1.02)
    ]
    tick_positions = np.unique(np.concatenate(([powers[0]], tick_positions, [powers[-1]])))
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(FuncFormatter(format_power_tick))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)

    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="0.85",
        fontsize=8.6,
        borderpad=0.65,
        labelspacing=0.45,
        handlelength=1.8,
        handletextpad=0.6,
        markerscale=1.05,
    )
    fig.subplots_adjust(bottom=0.18, left=0.14, right=0.97, top=0.96)
    fig.savefig(output_root / "npt_avalanche_curve.png")
    plt.close(fig)

    plot_data = {
        "x_axis": "excitation_power_w_cm2",
        "y_axis": "radiative_events_per_particle_s",
        "convergence_semantics": next(
            (
                str(row["convergence"]["semantics"])
                for row in summaries
                if (row.get("convergence") or {}).get("semantics")
            ),
            None,
        ),
        "detection": json_safe(detection),
        "points": [
            {
                "power": rec["power"],
                "rad_800_proxy_events_per_particle_s": rec["rad800"],
                "rad_700_proxy_events_per_particle_s": rec["rad700"],
                "rad800_seed_standard_error": rec["standard_error"],
                "convergence_status": rec["status"],
                "stage": rec["stage"],
                "seed_rad800_events_per_particle_s": rec["seed_rad800"],
            }
            for rec in records
        ],
    }
    write_json_atomic(output_root / "npt_avalanche_curve.json", plot_data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run production NPT-based Tm avalanche kMC sweeps with a simplified "
            "fixed-scale interface."
        )
    )
    parser.add_argument("--params", default=str(rates.DEFAULT_PARAMS_PATH))
    parser.add_argument(
        "--source-np-db",
        default=None,
        help=(
            "Optional existing geometry np.sqlite. If omitted, geometry is "
            "generated inside --output-root."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory. Defaults to the first available runN directory.",
    )
    parser.add_argument(
        "--npmc-command",
        default=DEFAULT_NPMC_COMMAND,
        help=(
            "Path to the NPMC binary. If the default desktop path is missing, "
            "the script falls back automatically to the HPC path."
        ),
    )
    parser.add_argument(
        "--trajectory-archive-root",
        default=str(DEFAULT_TRAJECTORY_ARCHIVE_ROOT),
        help=(
            "Archive root for completed initial_state.sqlite files. Each finished "
            "trajectory DB is moved into a dated subdirectory and replaced by an "
            "absolute symlink in the power directory. If the archive root does "
            "not exist, the trajectory DB is left in the local power directory."
        ),
   )
    parser.add_argument(
        "--sigma-esa-scale",
       type=float,
        default=None,
        help="Fixed multiplicative factor for the 2->6 ESA pump cross section.",
    )
    parser.add_argument(
        "--q21-scale",
        type=float,
        default=None,
        help="Fixed multiplicative factor for Q21,24.",
    )
    parser.add_argument(
        "--s54-scale",
        type=float,
        default=None,
        help="Fixed multiplicative factor for s54,23.",
    )
    parser.add_argument(
        "--s45-scale",
        type=float,
        default=None,
        help="Fixed multiplicative factor for s45,32.",
    )
    parser.add_argument(
        "--s12-scale",
        type=float,
        default=None,
        help="Fixed multiplicative factor for s12,42.",
    )
    parser.add_argument(
        "--em-mode",
        choices=("off", "all", "ground_mediated", "in_loop"),
        default=None,
        help=(
            "Which NPT-derived resonant migration subset to append. Defaults "
            "come from the NPT production settings in the parameter JSON."
        ),
    )
    parser.add_argument(
        "--em-scale",
        type=float,
        default=None,
        help="Fixed multiplicative factor for all enabled EM rows.",
    )
    parser.add_argument(
        "--surface-quench-mode",
        choices=("off", "outer_layer"),
        default=None,
        help=(
            "Enable surface quenching by placing a Surface trap species only in "
            "the outermost shell layer."
        ),
    )
    parser.add_argument(
        "--surface-fraction",
        type=float,
        default=None,
        help="Fraction of outer-layer Y sites replaced by the Surface trap species.",
    )
    parser.add_argument(
        "--surface-species",
        default=None,
        help="NPT surface species name to use for quenching. Defaults to Surface.",
    )
    parser.add_argument("--include-zero-rates", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build DBs and manifests but skip NPMC.")
    parser.add_argument("--powers", default=None, help="Comma/space separated powers in W cm^-2.")
    parser.add_argument(
        "--power-sampling-mode",
        choices=("homogeneous", "centered-gaussian"),
        default=DEFAULT_POWER_SAMPLING_MODE,
        help=(
            "Power sampling strategy when --powers is not provided. "
            "'homogeneous' keeps the geometric sweep; 'centered-gaussian' "
            "densifies the sweep around --power-center in log-power space."
        ),
    )
    parser.add_argument(
        "--power-center",
        type=float,
        default=DEFAULT_POWER_GAUSSIAN_CENTER,
        help="Center power in W cm^-2 for centered-gaussian sampling.",
    )
    parser.add_argument(
        "--power-gaussian-sigma-decades",
        type=float,
        default=DEFAULT_POWER_GAUSSIAN_SIGMA_DECADES,
        help="Gaussian width in log10(power) decades for centered-gaussian sampling.",
    )
    parser.add_argument("--power-min", type=float, default=DEFAULT_POWER_MIN)
    parser.add_argument("--power-max", type=float, default=DEFAULT_POWER_MAX)
    parser.add_argument("--power-count", type=int, default=DEFAULT_POWER_COUNT)
    parser.add_argument("--num-sims", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument(
        "--power-parallel-workers",
        type=int,
        default=None,
        help=(
            "Explicit number of power points to process concurrently. If omitted, "
            "the worker count is derived automatically from available CPU slots "
            "and --thread-count."
        ),
    )
    parser.add_argument(
        "--power-parallel-total-slots",
        type=int,
        default=None,
        help=(
            "Total CPU slots available for the power-parallel scheduler. Defaults "
            "to Slurm environment variables when present."
        ),
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=("steps", "physical-time"),
        default=None,
        help=(
            "Simulation cutoff mode. Defaults to 'steps' unless "
            "--simulation-time is provided, in which case 'physical-time' "
            "is selected automatically."
        ),
    )
    parser.add_argument(
        "--simulation-length",
        type=int,
        default=DEFAULT_SIMULATION_LENGTH,
        help="Per-seed event cutoff used when cutoff mode is 'steps'.",
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=None,
        help="Per-seed physical-time cutoff in seconds used in physical-time mode.",
    )
    parser.add_argument(
        "--tm-fraction",
        type=float,
        default=None,
        help="Tm dopant fraction used consistently for both geometry generation and spectral kinetics.",
    )
    parser.add_argument("--doping-seed", type=int, default=23)
    parser.add_argument("--core-radius-a", type=float, default=CORE_RADIUS_A)
    parser.add_argument(
        "--shell-thickness-a",
        type=float,
        default=AVERAGE_SHELL_THICKNESS_A,
        help="Shell thickness in Angstrom; the outer radius is derived as core radius plus shell thickness.",
    )
    parser.add_argument(
        "--regenerate-geometry",
        action="store_true",
        help="Regenerate the self-contained source geometry database.",
    )

    # --- Integrated adaptive two-stage workflow ---------------------------
    parser.add_argument(
        "--workflow-mode",
        choices=("single-stage", "adaptive-two-stage"),
        default=DEFAULT_WORKFLOW_MODE,
        help=(
            "'single-stage' keeps the manual sweep behavior (--powers or "
            "--power-sampling-mode). 'adaptive-two-stage' runs a log-spaced "
            "pilot scan, detects the avalanche transition, and refines "
            "around it with adaptive terminal-block convergence."
        ),
    )
    # Pilot scan
    parser.add_argument("--pilot-power-min", type=float, default=DEFAULT_PILOT_POWER_MIN)
    parser.add_argument("--pilot-power-max", type=float, default=DEFAULT_PILOT_POWER_MAX)
    parser.add_argument("--pilot-power-count", type=int, default=DEFAULT_PILOT_POWER_COUNT)
    parser.add_argument("--pilot-step-cutoff", type=int, default=DEFAULT_PILOT_STEP_CUTOFF)
    parser.add_argument("--pilot-num-sims", type=int, default=DEFAULT_PILOT_NUM_SIMS)
    parser.add_argument(
        "--pilot-terminal-fraction",
        type=float,
        default=DEFAULT_PILOT_TERMINAL_FRACTION,
        help=(
            "Terminal fraction of each pilot trajectory used for the "
            "transition-detection observables (recorded in every summary)."
        ),
    )
    # Automatic transition detection and refinement
    parser.add_argument("--refine-power-count", type=int, default=DEFAULT_REFINE_POWER_COUNT)
    parser.add_argument(
        "--refine-half-width-decades",
        type=float,
        default=DEFAULT_REFINE_HALF_WIDTH_DECADES,
        help="Refinement spans log10(center) +/- this many decades.",
    )
    parser.add_argument(
        "--refine-min-power-gap-fraction",
        type=float,
        default=DEFAULT_REFINE_MIN_POWER_GAP_FRACTION,
        help="Refinement points closer than this relative gap are merged.",
    )
    parser.add_argument("--refine-num-sims", type=int, default=DEFAULT_REFINE_NUM_SIMS)
    parser.add_argument(
        "--refine-center",
        type=float,
        default=None,
        help="Optional manual refinement center override (W cm^-2).",
    )
    parser.add_argument(
        "--transition-min-slope",
        type=float,
        default=DEFAULT_TRANSITION_MIN_SLOPE,
        help=(
            "Minimum local log-log slope of the 800 nm signal that counts "
            "as a photon-avalanche transition."
        ),
    )
    # Adaptive convergence (terminal-blocks-v2; steps cutoff mode only)
    parser.add_argument(
        "--checkpoint-extension-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_EXTENSION_STEPS,
        help="Absolute step-cutoff increment per convergence extension round.",
    )
    parser.add_argument(
        "--max-step-cutoff",
        type=int,
        default=DEFAULT_MAX_STEP_CUTOFF,
        help="Absolute cap for the per-seed step cutoff.",
    )
    parser.add_argument("--convergence-block-count", type=int, default=DEFAULT_CONVERGENCE_BLOCK_COUNT)
    parser.add_argument(
        "--convergence-min-events-per-block",
        type=int,
        default=DEFAULT_CONVERGENCE_MIN_EVENTS_PER_BLOCK,
    )
    parser.add_argument(
        "--convergence-min-block-time-s",
        type=float,
        default=DEFAULT_CONVERGENCE_MIN_BLOCK_TIME_S,
    )
    parser.add_argument(
        "--convergence-relative-drift",
        type=float,
        default=DEFAULT_CONVERGENCE_RELATIVE_DRIFT,
    )
    parser.add_argument(
        "--convergence-poisson-z",
        type=float,
        default=DEFAULT_CONVERGENCE_POISSON_Z,
    )
    parser.add_argument(
        "--convergence-required-passes",
        type=int,
        default=DEFAULT_CONVERGENCE_REQUIRED_PASSES,
        help="Consecutive checkpoint passes every seed must reach.",
    )
    parser.add_argument(
        "--convergence-observables",
        default=DEFAULT_CONVERGENCE_OBSERVABLES,
        help="Comma-separated subset of rad800,rad700,n4.",
    )
    parser.add_argument(
        "--convergence-mode-threshold",
        type=float,
        default=None,
        help=(
            "Optional log10 threshold on the per-block rad800 rate "
            "indicator for dark/bright mode classification (metastability)."
        ),
    )
    parser.add_argument(
        "--convergence-min-switches",
        type=int,
        default=DEFAULT_CONVERGENCE_MIN_SWITCHES,
        help="Minimum dark/bright transitions before a seed is not censored.",
    )
    parser.add_argument(
        "--convergence-semantics",
        choices=CONVERGENCE_SEMANTICS_CHOICES,
        default=DEFAULT_CONVERGENCE_SEMANTICS,
        help=(
            "Metastability interpretation: 'branch' (default) certifies a "
            "locally stationary dark/bright branch per seed, never an "
            "equilibrium basin weight; 'equilibrium' requires "
            "--convergence-mode-threshold and the switching minimum."
        ),
    )
    return parser


def validate_adaptive_arguments(args: argparse.Namespace) -> None:
    """Early validation for the adaptive two-stage workflow (fail fast)."""
    if args.resolved_cutoff_mode != "steps":
        raise ValueError(
            "--workflow-mode adaptive-two-stage requires the steps cutoff mode "
            "(adaptive extensions are step-based); do not set --simulation-time"
        )
    if args.pilot_power_min <= 0 or args.pilot_power_max <= args.pilot_power_min:
        raise ValueError(
            "--pilot-power-min must be positive and smaller than --pilot-power-max"
        )
    if args.pilot_power_count < 3:
        raise ValueError("--pilot-power-count must be at least 3")
    if args.pilot_step_cutoff < 1:
        raise ValueError("--pilot-step-cutoff must be at least 1")
    if args.pilot_num_sims < 1:
        raise ValueError("--pilot-num-sims must be at least 1")
    if not (0.0 < args.pilot_terminal_fraction <= 1.0):
        raise ValueError("--pilot-terminal-fraction must lie in (0, 1]")
    if args.refine_power_count < 2:
        raise ValueError("--refine-power-count must be at least 2")
    if args.refine_half_width_decades <= 0:
        raise ValueError("--refine-half-width-decades must be positive")
    if not (0.0 < args.refine_min_power_gap_fraction < 1.0):
        raise ValueError("--refine-min-power-gap-fraction must lie in (0, 1)")
    if args.refine_num_sims < 1:
        raise ValueError("--refine-num-sims must be at least 1")
    if args.transition_min_slope <= 0:
        raise ValueError("--transition-min-slope must be positive")
    if args.refine_center is not None and args.refine_center <= 0:
        raise ValueError("--refine-center must be positive")
    conv_params = resolve_convergence_parameters(args)
    if conv_params.max_step_cutoff < int(args.pilot_step_cutoff):
        raise ValueError("--max-step-cutoff must not be below --pilot-step-cutoff")
    if conv_params.max_step_cutoff < int(args.resolved_simulation_length):
        raise ValueError("--max-step-cutoff must not be below --simulation-length")


def main() -> None:
    args = build_arg_parser().parse_args()
    args.npmc_command = resolve_npmc_command(args.npmc_command)
    (
        args.resolved_cutoff_mode,
        args.resolved_simulation_length,
        args.resolved_simulation_time,
    ) = resolve_simulation_cutoff(args)
    params = rates.load_dre_parameters(args.params)
    config = resolve_production_config(args, params)
    output_root = Path(args.output_root) if args.output_root else next_run_dir()
    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_archive_root = Path(args.trajectory_archive_root).expanduser()

    if args.workflow_mode == "adaptive-two-stage":
        validate_adaptive_arguments(args)
        # The refinement stage starts from at least the pilot cutoff so a
        # reused pilot point never loses depth; --simulation-length may
        # raise it further.
        args.adaptive_refine_initial_step_cutoff = max(
            int(args.resolved_simulation_length), int(args.pilot_step_cutoff)
        )
        source_np_db_path = resolve_source_np_db(args, params, output_root, config)
        print(f"Using source geometry database: {source_np_db_path}", flush=True)
        run_adaptive_two_stage(
            args=args,
            params=params,
            config=config,
            output_root=output_root,
            trajectory_archive_root=trajectory_archive_root,
            source_np_db_path=source_np_db_path,
        )
        return

    # Single-stage/manual sweep below; convergence uses the same
    # terminal-blocks algorithm as the adaptive workflow.
    resolve_convergence_parameters(args)
    source_np_db_path = resolve_source_np_db(args, params, output_root, config)
    print(f"Using source geometry database: {source_np_db_path}", flush=True)

    powers = parse_power_sweep(args)
    build_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    power_parallel_workers = resolve_power_parallel_workers(args, len(powers))
    power_parallel_total_slots = resolve_power_parallel_total_slots(args)
    auto_power_parallel = power_parallel_workers > 1
    local_db_staging_root = (
        None if args.dry_run else resolve_local_db_staging_root()
    )
    if local_db_staging_root is not None:
        print(
            "Using node-local staging for per-power SQLite writes: "
            f"{local_db_staging_root}",
            flush=True,
        )

    root_config = {
        "profile": params["profile"],
        "params_path": str(Path(args.params).resolve()),
        "source_np_db": str(source_np_db_path.resolve()),
        "rate_model": config["rate_model"],
        "resolved_config": json_safe(
            {
                key: value
                for key, value in config.items()
                if key != "mode_defaults"
            }
        ),
        "powers_w_cm2": [float(power) for power in powers],
        "dry_run": bool(args.dry_run),
        "num_sims": int(args.num_sims),
        "base_seed": int(args.base_seed),
        "thread_count": int(args.thread_count),
        "power_parallel": bool(auto_power_parallel),
        "power_parallel_workers": int(power_parallel_workers),
        "power_parallel_total_slots": (
            None
            if power_parallel_total_slots is None
            else int(power_parallel_total_slots)
        ),
        "local_db_staging_root": (
            None
            if local_db_staging_root is None
            else str(local_db_staging_root.resolve())
        ),
        "trajectory_archive_root": str(trajectory_archive_root.resolve()),
        "simulation_cutoff_mode": args.resolved_cutoff_mode,
        "simulation_step_cutoff": (
            None
            if args.resolved_simulation_length is None
            else int(args.resolved_simulation_length)
        ),
        "simulation_time_cutoff_s": (
            None
            if args.resolved_simulation_time is None
            else float(args.resolved_simulation_time)
        ),
    }
    write_json_atomic(output_root / "npt_production_config.json", root_config)

    power_jobs = [
        (int(power_index), float(power))
        for power_index, power in enumerate(powers)
    ]
    if auto_power_parallel:
        print(
            f"Running {len(power_jobs)} power points with {power_parallel_workers} concurrent workers "
            f"and {args.thread_count} NPMC threads per power.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=power_parallel_workers) as executor:
            future_to_job = {
                executor.submit(
                    run_power_point,
                    power_index=power_index,
                    power_count=len(power_jobs),
                    power=power,
                    output_root=output_root,
                    params=params,
                    source_np_db_path=source_np_db_path,
                    include_zero_rates=bool(args.include_zero_rates),
                    tm_fraction=args.tm_fraction,
                    config=config,
                    args=args,
                    trajectory_archive_root=trajectory_archive_root,
                    local_db_staging_root=local_db_staging_root,
                ): (power_index, power)
                for power_index, power in power_jobs
            }
            for future in as_completed(future_to_job):
                build_record, summary = future.result()
                build_records.append(build_record)
                if summary is not None:
                    summaries.append(summary)
    else:
        for power_index, power in power_jobs:
            build_record, summary = run_power_point(
                power_index=power_index,
                power_count=len(power_jobs),
                power=power,
                output_root=output_root,
                params=params,
                source_np_db_path=source_np_db_path,
                include_zero_rates=bool(args.include_zero_rates),
                tm_fraction=args.tm_fraction,
                config=config,
                args=args,
                trajectory_archive_root=trajectory_archive_root,
                local_db_staging_root=local_db_staging_root,
            )
            build_records.append(build_record)
            if summary is not None:
                summaries.append(summary)

    build_records.sort(key=lambda row: int(row["power_index"]))
    root_config["npt_raw_vs_npmc_readin_by_power"] = build_power_rate_tables(
        build_records
    )
    write_json_atomic(output_root / "npt_production_config.json", root_config)

    max_local_log_slope_800 = None
    max_local_log_slope_800_power_w_cm2 = None
    if summaries:
        ordered_summaries = sorted(
            summaries,
            key=lambda row: float(row["excitation_power_w_cm2"]),
        )
        powers_arr = np.asarray(
            [row["excitation_power_w_cm2"] for row in ordered_summaries],
            dtype=float,
        )
        emission_800 = np.asarray(
            [row["rad_800_proxy_events_per_particle_s"] for row in ordered_summaries],
            dtype=float,
        )
        if powers_arr.size > 1:
            slopes_800 = np.gradient(
                np.log10(np.maximum(emission_800, 1.0e-300)),
                np.log10(np.maximum(powers_arr, 1.0e-300)),
            )
            if np.any(np.isfinite(slopes_800)):
                slope_index = int(np.nanargmax(slopes_800))
                max_local_log_slope_800 = float(slopes_800[slope_index])
                max_local_log_slope_800_power_w_cm2 = float(powers_arr[slope_index])

    sweep_summary = {
        "max_local_log_slope_800": max_local_log_slope_800,
        "max_local_log_slope_800_power_w_cm2": max_local_log_slope_800_power_w_cm2,
        **root_config,
        "build_records": build_records,
        "power_points": summaries,
    }
    write_json_atomic(output_root / "npt_power_sweep_summary.json", sweep_summary)

    if not args.dry_run:
        plot_avalanche_curve(summaries, output_root)


if __name__ == "__main__":
    main()
