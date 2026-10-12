"""Production runner for the Tm avalanche kMC model built on NPT defaults."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import copy
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

# Matplotlib requires a writable cache directory; default locations can be
# read-only on HPC filesystems, so force a writable tmpfs path before importing
# any matplotlib submodules that rely on it (e.g., mathtext font caching).
_mpl_cache_dir = Path("/dev/shm") / "matplotlib-cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_cache_dir)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import tm_npt_rates as rates
import plot_mechanism_diagrams
import plot_n4_convergence
from convergence_diagnostic import (
    compute_n4_late_window_averages,
    replay_trajectories as _replay_trajectories,
    TM_SPECIES_ID,
    N4_LEVEL,
    Q21_CHANNEL_NAME,
    S12_CHANNEL_NAME,
)
from NanoParticleTools.core import NPMCInput
from NanoParticleTools.analysis.util import get_spectrum_wavelength_from_dndt
from NanoParticleTools.inputs.nanoparticle import (
    DopedNanoparticle,
    SphericalConstraint,
    get_nayf4_structure_v2,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NPMC_COMMAND = "/home/rpluo/Desktop/project_MFML_UCNP/RNMC/build/NPMC"
FALLBACK_NPMC_COMMAND = "/global/home/users/rluo/project_UCNP/RNMC/build/NPMC"
DEFAULT_POWER_MIN = 3.0e3
DEFAULT_POWER_MAX = 5.0e4
DEFAULT_POWER_COUNT = 12
DEFAULT_SIMULATION_LENGTH = 70_000_000
DEFAULT_POWER_SAMPLING_MODE = "centered-gaussian"
DEFAULT_POWER_GAUSSIAN_CENTER = 1.3e4
DEFAULT_POWER_GAUSSIAN_SIGMA_DECADES = 0.05
PREPARATION_POWER_W_CM2 = 50_000.0
PREPARATION_CURVE_MIN_POWER_W_CM2 = 10_000.0
PREPARATION_CURVE_MAX_STEPS = 30_000_000
# Endpoint-constrained least-squares fit to the approximate preparation schedule:
# f(x) = (1 - exp(-a*x**b)) / (1 - exp(-a)), x = (power - 10k) / 40k.
PREPARATION_CURVE_RATE = 3.612222
PREPARATION_CURVE_EXPONENT = 1.315627
FIXED_PRODUCTION_POWERS_W_CM2 = (
    3_000.0,
    4_614.84,
    7_098.92,
    10_390.7,
    11_732.9,
    12_535.5,
    13_000.0,
    14_112.3,
    15_504.8,
    21_130.5,
    32_503.8,
    50_000.0,
)
DEFAULT_TRAJECTORY_ARCHIVE_ROOT = Path(
    "/home/rpluo/Desktop/hdd_large/KMC_trajectories/accelerate_convergence"
)
SURFACE_LAYER_THICKNESS_NM = 0.5
SURFACE_LAYER_THICKNESS_A = SURFACE_LAYER_THICKNESS_NM * 10
DEFAULT_SURFACE_QUENCH_MODE = "off"
DEFAULT_SURFACE_SPECIES = "Surface"
DEFAULT_SURFACE_FRACTION = 0.20
SURFACE_SPECIES_ID = 1
S54_CHANNEL_NAME = "s54,23"
S45_CHANNEL_NAME = "s45,32"
TM_CHANNEL_TUPLE_TO_NAME = {
    (3, 1, 0, 1): S12_CHANNEL_NAME,
    (1, 0, 1, 3): Q21_CHANNEL_NAME,
    (1, 0, 1, 2): "Q21,23",
    (4, 3, 1, 2): S54_CHANNEL_NAME,
    (2, 1, 3, 4): S45_CHANNEL_NAME,
}
N4_TRAJECTORY_TRACE_SAMPLES = 241
NPMC_STDERR_ERROR_MARKERS = {
    "sql_error": "SQL error:",
    "database_locked": "database is locked",
    "transaction_state": "cannot start a transaction within a transaction",
    "commit_state": "cannot commit transaction - SQL statements in progress",
    "traceback": "Traceback",
    "killed": "Killed",
}

# Table S1 / run1 geometry approximation used previously:
# 4.56% Tm core minor/major axes = 20.7 / 32.5 nm, shell thickness = 5.5 nm.
CORE_MEAN_DIAMETER_NM = (20.7 + 32.5) / 2
CORE_RADIUS_A = CORE_MEAN_DIAMETER_NM * 5
AVERAGE_SHELL_THICKNESS_NM = 5.5
AVERAGE_SHELL_THICKNESS_A = AVERAGE_SHELL_THICKNESS_NM * 10
OUTER_RADIUS_A = CORE_RADIUS_A + AVERAGE_SHELL_THICKNESS_A

FALLBACK_PRODUCTION_DEFAULTS = {
    "npt_cr_mode": "exported",
    "sigma_esa_scale": 1.0,
    "q21_scale": 1.0,
    "s54_scale": 1.0,
    "s45_scale": 1.0,
    "s12_scale": 1.0,
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


def utc_now_iso() -> str:
    """Return an unambiguous timestamp for run-history records."""
    return datetime.now(timezone.utc).isoformat()


_CHECKPOINT_REQUESTED = False
_CONTINUE_RUN = False


class CheckpointRequested(Exception):
    """A clean walltime pause, distinct from simulation or analysis failures."""


def checkpoint_boundary() -> None:
    if _CHECKPOINT_REQUESTED:
        raise CheckpointRequested()


def request_checkpoint(_signum: int, _frame: Any) -> None:
    global _CHECKPOINT_REQUESTED
    _CHECKPOINT_REQUESTED = True


def checkpoint_futures(futures: Iterable[Future]) -> Iterator[Future]:
    """Drain workers on pause, but never hide a real error behind that pause."""
    error: BaseException | None = None
    for future in as_completed(futures):
        if not _CONTINUE_RUN:
            yield future
        elif future.exception() is not None:
            request_checkpoint(signal.SIGUSR1, None)
            if error is None or isinstance(error, CheckpointRequested):
                error = future.exception()
        else:
            yield future
    if error is not None:
        raise error


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(json_safe(data), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def replay_trajectories(*args: Any, **kwargs: Any) -> dict[str, Any]:
    checkpoint_boundary()
    if _CONTINUE_RUN:
        kwargs["stop_requested"] = lambda: _CHECKPOINT_REQUESTED
    try:
        result = _replay_trajectories(*args, **kwargs)
    except (InterruptedError, sqlite3.OperationalError):
        checkpoint_boundary()
        raise
    checkpoint_boundary()
    return result


def resume_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Return the resume-critical configuration fingerprint for one power point."""
    keys = (
        "excitation_power_w_cm2",
        "rate_model",
        "pump_cross_section_source",
        "npt_cr_mode",
        "sigma_esa_scale",
        "q21_scale",
        "s54_scale",
        "s45_scale",
        "s12_scale",
        "em_mode",
        "em_scale",
        "surface_quench_mode",
        "surface_species",
        "surface_fraction",
        "surface_layer_thickness_a",
        "simulation_cutoff_mode",
        "geometry",
        "interactions",
    )
    payload = {key: manifest.get(key) for key in keys}
    serialized = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def collect_run_debug(
    *,
    output_dir: Path,
    initial_state_db_path: Path,
    resume: bool,
    previous_step_cutoff: int | None,
    requested_extension_steps: int | None,
    final_step_cutoff: int | None,
    summary: dict[str, Any],
    npmc_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect cheap diagnostics for this completed NPMC attempt."""
    stderr_path = output_dir / "stderr"
    stderr_size_bytes = None
    if stderr_path.exists():
        try:
            stderr_size_bytes = int(stderr_path.stat().st_size)
        except OSError:
            pass

    trajectory_path = initial_state_db_path.resolve()
    trajectory_size_bytes = None
    try:
        trajectory_size_bytes = int(trajectory_path.stat().st_size)
    except OSError:
        pass

    return {
        "recorded_at_utc": utc_now_iso(),
        "resume": bool(resume),
        "checkpoint_requested": 1,
        "previous_step_cutoff": previous_step_cutoff,
        "requested_extension_steps": requested_extension_steps,
        "final_step_cutoff": final_step_cutoff,
        "trajectory_db_path": str(trajectory_path),
        "trajectory_db_size_bytes": trajectory_size_bytes,
        "stderr_path": str(stderr_path.resolve()),
        "stderr_size_bytes": stderr_size_bytes,
        "npmc_attempt": npmc_attempt,
        "replay_result": {
            "num_completed_sims": summary.get("num_completed_sims"),
            "total_simulation_time_s": summary.get("total_simulation_time_s"),
            "n4_time_averaged_population": summary.get("n4_time_averaged_population"),
            "rad_800_proxy_events_per_particle_s": summary.get(
                "rad_800_proxy_events_per_particle_s"
            ),
            "rad_700_proxy_events_per_particle_s": summary.get(
                "rad_700_proxy_events_per_particle_s"
            ),
            "n4_rad_consistency_ratio": summary.get("n4_rad_consistency_ratio"),
        },
    }


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
    if args.power_sampling_mode == "fixed-production":
        return np.asarray(FIXED_PRODUCTION_POWERS_W_CM2, dtype=float)
    if args.power_sampling_mode == "centered-gaussian":
        return build_centered_gaussian_power_sweep(
            power_min=float(args.power_min),
            power_max=float(args.power_max),
            power_count=int(args.power_count),
            center=float(args.power_center),
            sigma_decades=float(args.power_gaussian_sigma_decades),
        )
    return np.geomspace(args.power_min, args.power_max, args.power_count)


def preparation_steps_for_power(power: float, *, enabled: bool = True) -> int:
    """Return the smooth, fitted 50k preparation length for a target power."""
    target_power = float(power)
    if not enabled or target_power <= PREPARATION_CURVE_MIN_POWER_W_CM2:
        return 0
    if target_power >= PREPARATION_POWER_W_CM2:
        return PREPARATION_CURVE_MAX_STEPS

    scaled_power = (
        (target_power - PREPARATION_CURVE_MIN_POWER_W_CM2)
        / (PREPARATION_POWER_W_CM2 - PREPARATION_CURVE_MIN_POWER_W_CM2)
    )
    fitted_fraction = (
        -np.expm1(
            -PREPARATION_CURVE_RATE
            * scaled_power**PREPARATION_CURVE_EXPONENT
        )
        / -np.expm1(-PREPARATION_CURVE_RATE)
    )
    return int(round(PREPARATION_CURVE_MAX_STEPS * fitted_fraction))


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


def simulation_step_cutoff_for_power(args: argparse.Namespace) -> int | None:
    """Return the per-seed step length requested by the user.

    For a fresh run this is the total cutoff. For a resume run it is the
    extension length to add to the previous cutoff.
    """
    if args.resolved_cutoff_mode != "steps":
        return None
    return int(args.resolved_simulation_length)


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


@contextlib.contextmanager
def _power_lock(output_dir: Path, power_index: int, power_count: int):
    """Exclusive per-power lock to prevent concurrent NPMC runs on the same DB."""
    lock_path = output_dir / ".npmc_lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(
            f"[power {power_index + 1}/{power_count}] another NPMC process is "
            f"already running for {output_dir.name}. If it is stale, remove "
            f"{lock_path} and retry."
        )
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


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
    host_structure = get_nayf4_structure_v2()
    if surface_enabled:
        constraints = [
            SphericalConstraint(float(args.core_radius_a), host_structure),
            SphericalConstraint(surface_inner_radius_a, host_structure),
            SphericalConstraint(outer_radius_a, host_structure),
        ]
        dopant_specification = [
            (0, 0.25, "Na", "Y"),
            (0, tm_fraction, "Tm", "Y"),
            (2, 0.25, "Na", "Y"),
            (
                2,
                float(config["surface_fraction"]),
                str(config["surface_species"]),
                "Y",
            ),
        ]
    else:
        constraints = [
            SphericalConstraint(float(args.core_radius_a), host_structure),
            SphericalConstraint(outer_radius_a, host_structure),
        ]
        dopant_specification = [
            (0, 0.25, "Na", "Y"),
            (0, tm_fraction, "Tm", "Y"),
        ]
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
        "npt_cr_mode": str(choose("npt_cr_mode", args.npt_cr_mode)),
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
    if config["npt_cr_mode"] not in ("all", "exported"):
        raise ValueError(f"Unsupported npt_cr_mode: {config['npt_cr_mode']!r}")
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


def build_power_rate_tables(
    build_records: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Collect per-power raw-NPT vs NPMC-readin rate tables for the root config.

    Manifest paths stored in build records may be absolute paths from an earlier
    run location (e.g. after moving or resuming a sweep on a different machine).
    Reconstruct the path relative to the current output_root so resume runs do
    not fail with stale absolute paths.
    """
    tables = []
    for record in sorted(build_records, key=lambda row: int(row["power_index"])):
        power_dir_name = Path(record["output_dir"]).name
        manifest_path = output_root / power_dir_name / "npt_interaction_manifest.json"
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
    state_to_level = (
        rates.dre_state_to_level_map(params)
        if config["npt_cr_mode"] == "all"
        else {}
    )

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

    # Optional CR completion using NPT semi-empirical constants.
    if config["npt_cr_mode"] == "all":
        for channel in rates.load_dre_channels(params):
            ordered = rates.order_channel(channel, semi_dopant, state_to_level)
            kmc_tuple = (
                ordered.donor_initial_level,
                ordered.donor_final_level,
                ordered.acceptor_initial_level,
                ordered.acceptor_final_level,
            )
            if kmc_tuple in exported_tm_pair_tuples:
                continue
            semi = rates.semi_empirical_pair_rate(ordered, semi_dopant, semi_sk)
            base_rate = float(semi["selected_nm6_s"])
            same_initial_state = kmc_tuple[0] == kmc_tuple[2]
            degeneracy_factor = 2.0 if same_initial_state else 1.0
            base_dre_equivalent = (
                base_rate
                * geometry.ordered_factor_sum
                * degeneracy_factor
                / geometry.ion_count
            )
            rate_scale_factor = float(pair_scale_for_channel(channel.name, config))
            effective_rate = base_rate * rate_scale_factor
            effective_dre_equivalent = base_dre_equivalent * rate_scale_factor
            included = include_zero_rates or effective_rate != 0.0
            filter_reason = None if included else "scale_zero_rate"
            report = {
                "channel_name": channel.name,
                "description": channel.description,
                "dre_rate_s^-1": float(channel.dre_rate_s),
                "npt_selected_kmc_rate": base_rate,
                "npt_selected_dre_equivalent_rate_s^-1": base_dre_equivalent,
                "npt_exported_kmc_rate": None,
                "npt_exported_dre_equivalent_rate_s^-1": None,
                "semi_empirical_exported_nm6_s": semi["exported_nm6_s"],
                "semi_empirical_exported": bool(semi["exported"]),
                "semi_empirical_branch": semi["branch"],
                "energy_gap_cm": semi["energy_gap_cm"],
                "effective_energy_gap_cm": semi["effective_energy_gap_cm"],
                "base_rate_source": "NPT semi-empirical selected",
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
                "source": "NPT semi-empirical selected",
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
                    label=channel.name,
                    source="NPT semi-empirical selected",
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
    reuse_existing: bool = False,
) -> tuple[Path, Path]:
    """Write custom NPMC databases for one excitation power."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np_db_path = output_dir / "np.sqlite"
    initial_state_db_path = output_dir / "initial_state.sqlite"
    ready_path = output_dir / ".database_ready.json"
    if reuse_existing and any(
        path.exists() for path in (ready_path, np_db_path, initial_state_db_path)
    ):
        if not all(
            path.exists() for path in (ready_path, np_db_path, initial_state_db_path)
        ):
            raise RuntimeError(f"Incomplete database initialization in {output_dir}")
        return np_db_path, initial_state_db_path
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

    if reuse_existing:
        atomic_json(ready_path, {"ready": True})
    return np_db_path, initial_state_db_path


def run_npmc(
    np_db_path: Path,
    initial_state_db_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    simulation_cutoff_mode: str,
    simulation_step_cutoff: int | None,
    simulation_time_cutoff_s: float | None,
) -> dict[str, Any]:
    """Run NPMC on the custom databases."""
    checkpoint_boundary()
    continuing = bool(getattr(args, "continue_run", False))
    checkpoint_path = output_dir / ".npmc_checkpoint.json"
    completed_seeds: list[int] = []
    saved_states: list[tuple] = []
    saved_cutoffs: list[tuple] = []
    last_rowid = 0
    if continuing:
        if checkpoint_path.exists():
            if json.loads(checkpoint_path.read_text())["status"] != "ready":
                raise RuntimeError(
                    f"Unclean NPMC exit in {output_dir}; refusing an unverified "
                    "checkpoint. Inspect the database before restarting."
                )
            progress = validate_npmc_checkpoint(
                initial_state_db_path,
                final_step_cutoff=int(simulation_step_cutoff),
                base_seed=int(args.base_seed),
                num_sims=int(args.num_sims),
                allow_partial=True,
            )
            completed_seeds = [
                seed for seed, step in progress.items()
                if step == int(simulation_step_cutoff) + 1
            ]
            if len(completed_seeds) == int(args.num_sims):
                return {"reused_checkpoint": True}
        else:
            with sqlite3.connect(initial_state_db_path) as con:
                if any(
                    con.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                    for table in ("trajectories", "interrupt_state", "interrupt_cutoff")
                ):
                    raise RuntimeError(f"Unverified existing trajectory in {output_dir}")

        # NPMC executes one event before testing the cutoff, even for finished
        # seeds. Preserve their endpoints while the other seeds catch up.
        if completed_seeds:
            placeholders = ",".join("?" for _ in completed_seeds)
            with sqlite3.connect(initial_state_db_path) as con:
                saved_states = con.execute(
                    f"SELECT * FROM interrupt_state WHERE seed IN ({placeholders})",
                    completed_seeds,
                ).fetchall()
                saved_cutoffs = con.execute(
                    f"SELECT * FROM interrupt_cutoff WHERE seed IN ({placeholders})",
                    completed_seeds,
                ).fetchall()
                last_rowid = con.execute(
                    "SELECT COALESCE(MAX(rowid), 0) FROM trajectories"
                ).fetchone()[0]

    run_args = [
        args.npmc_command,
        f"--nano_particle_database={np_db_path}",
        f"--initial_state_database={initial_state_db_path}",
        f"--number_of_simulations={args.num_sims}",
        f"--base_seed={args.base_seed}",
        f"--thread_count={args.thread_count}",
    ]
    if simulation_cutoff_mode == "steps":
        run_args.append(f"--step_cutoff={int(simulation_step_cutoff)}")
    elif simulation_cutoff_mode == "physical-time":
        run_args.append(f"--time_cutoff={float(simulation_time_cutoff_s)}")
    else:
        raise ValueError(
            f"Unsupported simulation cutoff mode: {simulation_cutoff_mode!r}"
        )
    # Always write checkpoint tables so future resumes can use them directly.
    run_args.append("--checkpoint=1")

    stderr_path = output_dir / "stderr"
    try:
        stderr_start_offset = int(stderr_path.stat().st_size)
    except FileNotFoundError:
        stderr_start_offset = 0

    print(f'Running NPMC using the command: "{" ".join(run_args)}"', flush=True)
    if continuing:
        atomic_json(checkpoint_path, {"status": "running"})
    with open(output_dir / "stdout", "a") as f_std, open(stderr_path, "a") as f_err:
        with subprocess.Popen(run_args, stdout=f_std, stderr=f_err) as process:
            signalled = False
            while process.poll() is None:
                if _CHECKPOINT_REQUESTED and not signalled:
                    # NPMC installs its handler after loading the model. Sending
                    # SIGTERM before then would kill it without a checkpoint.
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text()
                        caught = next(
                            line.split()[1] for line in status.splitlines()
                            if line.startswith("SigCgt:")
                        )
                        if int(caught, 16) & (1 << (signal.SIGTERM - 1)):
                            process.send_signal(signal.SIGTERM)
                            signalled = True
                    except ProcessLookupError:
                        pass
                    except FileNotFoundError:
                        pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, run_args)

    with open(stderr_path, "r", errors="replace") as f_err:
        f_err.seek(stderr_start_offset)
        stderr_text = f_err.read()
    stderr_counts = {
        name: stderr_text.count(marker)
        for name, marker in NPMC_STDERR_ERROR_MARKERS.items()
    }
    attempt = {
        "stderr_start_offset": stderr_start_offset,
        "stderr_end_offset": int(stderr_path.stat().st_size),
        "stderr_counts": stderr_counts,
    }
    if any(stderr_counts.values()):
        details = ", ".join(
            f"{name}={count}" for name, count in stderr_counts.items() if count
        )
        raise RuntimeError(
            "NPMC completed with new stderr error markers "
            f"({details}); refusing to summarize an untrusted trajectory. "
            f"See {stderr_path} from byte {stderr_start_offset}."
        )
    if continuing:
        if completed_seeds:
            with sqlite3.connect(initial_state_db_path) as con:
                con.execute(
                    f"DELETE FROM trajectories WHERE rowid > ? "
                    f"AND seed IN ({placeholders})",
                    [last_rowid, *completed_seeds],
                )
                for table in ("interrupt_state", "interrupt_cutoff"):
                    con.execute(
                        f"DELETE FROM {table} WHERE seed IN ({placeholders})",
                        completed_seeds,
                    )
                con.executemany("INSERT INTO interrupt_state VALUES (?, ?, ?)", saved_states)
                con.executemany("INSERT INTO interrupt_cutoff VALUES (?, ?, ?)", saved_cutoffs)
        validate_npmc_checkpoint(
            initial_state_db_path,
            final_step_cutoff=int(simulation_step_cutoff),
            base_seed=int(args.base_seed),
            num_sims=int(args.num_sims),
            allow_partial=_CHECKPOINT_REQUESTED,
        )
        atomic_json(checkpoint_path, {"status": "ready", "attempt": attempt})
        checkpoint_boundary()
    return attempt


def validate_npmc_checkpoint(
    initial_state_db_path: Path,
    *,
    final_step_cutoff: int,
    base_seed: int,
    num_sims: int,
    allow_partial: bool = False,
) -> dict[int, int]:
    """Verify that NPMC checkpoint tables are complete for a step cutoff.

    This is used both immediately after an NPMC run and before resuming one.
    The resume path deliberately uses the state NPMC stored in these tables;
    it does not reconstruct or rewrite that state in Python.
    """
    expected_seeds = list(range(base_seed, base_seed + num_sims))
    placeholders = ",".join("?" * len(expected_seeds))
    np_db_path = initial_state_db_path.with_name("np.sqlite").resolve()
    with sqlite3.connect(np_db_path) as con:
        site_count = int(con.execute("SELECT COUNT(*) FROM sites").fetchone()[0])
    with sqlite3.connect(initial_state_db_path.resolve()) as con:
        cutoff_rows = con.execute(
            "SELECT seed, step FROM interrupt_cutoff WHERE seed IN (%s)"
            % placeholders,
            expected_seeds,
        ).fetchall()
        state_rows = con.execute(
            "SELECT seed, COUNT(*), COUNT(DISTINCT site_id) "
            "FROM interrupt_state WHERE seed IN (%s) "
            "GROUP BY seed" % placeholders,
            expected_seeds,
        ).fetchall()

    cutoff_by_seed = {int(seed): int(step) for seed, step in cutoff_rows}
    state_counts = {
        int(seed): (int(total_count), int(distinct_site_count))
        for seed, total_count, distinct_site_count in state_rows
    }
    expected_cutoff_step = int(final_step_cutoff) + 1
    if (
        set(cutoff_by_seed) != set(expected_seeds)
        or len(cutoff_rows) != len(expected_seeds)
        or any(
            not (0 <= step <= expected_cutoff_step)
            if allow_partial else step != expected_cutoff_step
            for step in cutoff_by_seed.values()
        )
        or set(state_counts) != set(expected_seeds)
        or any(
            total_count != site_count or distinct_site_count != site_count
            for total_count, distinct_site_count in state_counts.values()
        )
    ):
        raise RuntimeError(
            "NPMC checkpoint is incomplete for one or more expected seeds; "
            "refusing to use an untrusted trajectory."
        )
    return cutoff_by_seed


def validate_replay_completion(
    replay: dict[str, Any],
    *,
    final_step_cutoff: int | None,
    base_seed: int,
    num_sims: int,
) -> None:
    """Require the already-completed full replay to reach every expected endpoint."""
    expected_seeds = set(range(base_seed, base_seed + num_sims))
    final_steps = {
        int(seed): int(step) for seed, step in replay.get("final_step", {}).items()
    }
    if set(final_steps) != expected_seeds:
        raise RuntimeError(
            "Trajectory replay did not contain every expected seed; refusing to "
            "summarize an incomplete trajectory."
        )
    if final_step_cutoff is not None and any(
        step != int(final_step_cutoff) for step in final_steps.values()
    ):
        raise RuntimeError(
            "Trajectory replay did not reach the requested final step for every "
            "seed; refusing to summarize an incomplete trajectory."
        )


def load_preparation_checkpoint_state(
    initial_state_db_path: Path,
    seed: int,
) -> tuple[list[tuple[int, int]], int, float]:
    """Load one complete state from a validated 50k preparation checkpoint."""
    with sqlite3.connect(initial_state_db_path.resolve()) as con:
        cutoff_rows = con.execute(
            "SELECT step, time FROM interrupt_cutoff WHERE seed = ?",
            (int(seed),),
        ).fetchall()
        state_rows = con.execute(
            "SELECT site_id, degree_of_freedom FROM interrupt_state "
            "WHERE seed = ? ORDER BY site_id",
            (int(seed),),
        ).fetchall()
    if len(cutoff_rows) != 1:
        raise RuntimeError(
            f"Expected one preparation cutoff for seed {seed}, "
            f"found {len(cutoff_rows)}"
        )
    checkpoint_step, checkpoint_time_s = cutoff_rows[0]
    return (
        [(int(site_id), int(state)) for site_id, state in state_rows],
        int(checkpoint_step),
        float(checkpoint_time_s),
    )


def transplant_preparation_state(
    *,
    initial_state_db_path: Path,
    np_db_path: Path,
    state_rows: list[tuple[int, int]],
) -> dict[str, int]:
    """Replace a fresh target initial state with one prepared checkpoint state."""
    with sqlite3.connect(np_db_path.resolve()) as con:
        target_rows = con.execute(
            "SELECT sites.site_id, species.degrees_of_freedom "
            "FROM sites JOIN species USING (species_id) ORDER BY sites.site_id"
        ).fetchall()
    degrees_by_site = {
        int(site_id): int(degrees) for site_id, degrees in target_rows
    }
    state_by_site = {int(site_id): int(state) for site_id, state in state_rows}
    if len(state_by_site) != len(state_rows):
        raise ValueError("Preparation checkpoint contains duplicate site IDs")
    if set(state_by_site) != set(degrees_by_site):
        missing = sorted(set(degrees_by_site) - set(state_by_site))[:8]
        extra = sorted(set(state_by_site) - set(degrees_by_site))[:8]
        raise ValueError(
            "Preparation and target geometries have different site IDs "
            f"(missing={missing}, extra={extra})"
        )
    invalid = [
        (site_id, state, degrees_by_site[site_id])
        for site_id, state in state_by_site.items()
        if state < 0 or state >= degrees_by_site[site_id]
    ]
    if invalid:
        raise ValueError(
            "Preparation checkpoint contains target-incompatible states: "
            f"{invalid[:8]}"
        )

    with sqlite3.connect(initial_state_db_path.resolve()) as con:
        initial_ids = {
            int(site_id)
            for (site_id,) in con.execute("SELECT site_id FROM initial_state")
        }
        if initial_ids != set(degrees_by_site):
            raise ValueError(
                "Fresh initial-state database does not match the target geometry"
            )
        con.executemany(
            "UPDATE initial_state SET degree_of_freedom = ? WHERE site_id = ?",
            [(state, site_id) for site_id, state in sorted(state_by_site.items())],
        )
        con.commit()

    counts = Counter(state_by_site.values())
    return {str(state): int(count) for state, count in sorted(counts.items())}


def merge_single_seed_replays(replays: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge disjoint single-seed target replays for ``summarize_run``."""
    if not replays:
        raise ValueError("No target replays were provided")
    total_site_count = int(replays[0]["total_site_count"])
    tm_site_count = int(replays[0]["tm_site_count"])
    merged: dict[str, Any] = {
        "simulation_time": {},
        "final_step": {},
        "event_counts": {},
        "n4_time_integral": {},
        "n4_population_per_seed": {},
        "n4_late_window_integrals": {},
        "total_site_count": total_site_count,
        "tm_site_count": tm_site_count,
        "q24_total_count": 0,
        "s12_total_count": 0,
        "q24_after_s12_same_pair_count": 0,
        "s12_after_q24_same_pair_count": 0,
        "n4_trace_by_seed": {},
    }
    mapping_keys = (
        "simulation_time",
        "final_step",
        "event_counts",
        "n4_time_integral",
        "n4_population_per_seed",
        "n4_late_window_integrals",
        "n4_trace_by_seed",
    )
    scalar_keys = (
        "q24_total_count",
        "s12_total_count",
        "q24_after_s12_same_pair_count",
        "s12_after_q24_same_pair_count",
    )
    seen_seeds: set[int] = set()
    for replay in replays:
        if (
            int(replay["total_site_count"]) != total_site_count
            or int(replay["tm_site_count"]) != tm_site_count
        ):
            raise ValueError("Cannot merge target replays from different geometries")
        replay_seeds = {int(seed) for seed in replay["simulation_time"]}
        overlap = seen_seeds & replay_seeds
        if overlap:
            raise ValueError(f"Duplicate seeds while merging target replays: {overlap}")
        seen_seeds.update(replay_seeds)
        for key in mapping_keys:
            merged[key].update(replay[key])
        for key in scalar_keys:
            merged[key] += int(replay[key])
    return merged


def single_seed_npmc_args(
    args: argparse.Namespace,
    seed: int,
) -> argparse.Namespace:
    """Copy production arguments for one independent target trajectory."""
    seed_args = copy.copy(args)
    seed_args.num_sims = 1
    seed_args.base_seed = int(seed)
    seed_args.thread_count = 1
    return seed_args


def link_target_np_database(np_link_path: Path, target_np_path: Path) -> None:
    """Point one seed directory at the shared, read-only target rate database."""
    if np_link_path.is_symlink() or np_link_path.exists():
        np_link_path.unlink()
    np_link_path.symlink_to(target_np_path.resolve())


def run_prepared_target_seed(
    seed_record: dict[str, Any],
    *,
    args: argparse.Namespace,
    final_step_cutoff: int,
) -> dict[str, Any]:
    """Run and validate one prepared target-power trajectory."""
    seed = int(seed_record["seed"])
    attempt = run_npmc(
        Path(seed_record["np_db_path"]),
        Path(seed_record["initial_state_db_path"]),
        Path(seed_record["seed_dir"]),
        single_seed_npmc_args(args, seed),
        simulation_cutoff_mode="steps",
        simulation_step_cutoff=int(final_step_cutoff),
        simulation_time_cutoff_s=None,
    )
    validate_npmc_checkpoint(
        Path(seed_record["initial_state_db_path"]),
        final_step_cutoff=int(final_step_cutoff),
        base_seed=seed,
        num_sims=1,
    )
    seed_record["npmc_attempt"] = attempt
    return seed_record


def replay_prepared_target_seeds(
    seed_records: list[dict[str, Any]],
    *,
    interactions: list[dict[str, Any]],
    final_step_cutoff: int,
) -> dict[str, Any]:
    """Replay independent prepared trajectories sequentially and merge them."""
    replays: list[dict[str, Any]] = []
    for record in sorted(seed_records, key=lambda row: int(row["seed"])):
        seed = int(record["seed"])
        print(f"Replaying target trajectory for seed {seed}...", flush=True)
        replay = replay_trajectories(
            Path(record["initial_state_db_path"]),
            interactions,
            n4_trace_samples=N4_TRAJECTORY_TRACE_SAMPLES,
            n4_trace_final_step=int(final_step_cutoff),
        )
        validate_replay_completion(
            replay,
            final_step_cutoff=int(final_step_cutoff),
            base_seed=seed,
            num_sims=1,
        )
        replays.append(replay)
    return merge_single_seed_replays(replays)


def finalize_prepared_seed_databases(
    seed_records: list[dict[str, Any]],
    *,
    output_dir: Path,
    archive_root: Path | None,
) -> dict[str, str]:
    """Archive target trajectories and retain per-seed links in the run tree."""
    paths: dict[str, str] = {}
    target_np_path = output_dir / "np.sqlite"
    for record in seed_records:
        seed = int(record["seed"])
        seed_dir = Path(record["seed_dir"])
        work_dir = Path(record["work_dir"])
        link_target_np_database(seed_dir / "np.sqlite", target_np_path)
        archived_path = finalize_initial_state_database(
            initial_state_db_path=Path(record["initial_state_db_path"]),
            output_dir=seed_dir,
            archive_root=archive_root,
        )
        paths[str(seed)] = str(archived_path)
        if work_dir != seed_dir and work_dir.exists():
            work_np_path = work_dir / "np.sqlite"
            if work_np_path.is_symlink() or work_np_path.exists():
                work_np_path.unlink()
            work_dir.rmdir()
    return paths


def summarize_prepared_power_point(
    *,
    replay: dict[str, Any],
    interactions: list[dict[str, Any]],
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    power: float,
    final_step_cutoff: int,
    params: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    npmc_attempts: dict[str, Any],
    resume_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the standard summary and plots for a prepared target run."""
    summary = summarize_run(
        replay=replay,
        interactions=interactions,
        n_sites=int(manifest["geometry"]["ion_count"]),
        excitation_power=float(power),
        manifest=manifest,
        simulation_cutoff_mode="steps",
        simulation_step_cutoff=int(final_step_cutoff),
        simulation_time_cutoff_s=None,
    )
    summary["preparation_protocol"] = protocol
    summary["run_debug"] = {
        "recorded_at_utc": utc_now_iso(),
        "checkpoint_requested": 1,
        "restart_npmc_attempts": npmc_attempts,
        **(resume_debug or {}),
    }
    with open(output_dir / "npt_run_summary.json", "w") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    if replay["n4_trace_by_seed"]:
        plot_n4_convergence.plot_n4_trajectory(
            replay["n4_trace_by_seed"],
            output_dir / "n4_trajectory.png",
            excitation_power=float(power),
        )
    plot_npt_style_spectrum(
        params=params,
        excitation_power=float(power),
        tm_fraction=args.tm_fraction,
        config=config,
        summary=summary,
        interactions=interactions,
        output_dir=output_dir,
    )
    plot_mechanism_diagrams.plot_run(
        output_dir,
        output_name="mechanism_diagram.png",
        rate_key="events_per_ion_s",
        min_rate=1e-3,
        max_by_kind={"ET": 16, "Rad": 18, "NR": 12, "Pump": 8, "SQ": 16},
        dpi=220,
    )
    return summary


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
    trajectory_archive_root: Path | None,
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
    simulation_cutoff_mode = str(args.resolved_cutoff_mode)
    # User-specified length: total cutoff for fresh runs, extension length for resumes.
    base_simulation_step_cutoff = simulation_step_cutoff_for_power(args)
    base_simulation_time_cutoff_s = args.resolved_simulation_time
    simulation_step_cutoff = base_simulation_step_cutoff
    simulation_time_cutoff_s = base_simulation_time_cutoff_s
    resume_previous_step_cutoff: int | None = None
    interactions, manifest = build_custom_interactions(
        params=params,
        source_np_db_path=source_np_db_path,
        excitation_power=float(power),
        include_zero_rates=include_zero_rates,
        tm_fraction=tm_fraction,
        config=config,
    )
    manifest["simulation_cutoff_mode"] = simulation_cutoff_mode
    final_np_db_path = output_dir / "np.sqlite"
    final_initial_state_db_path = output_dir / "initial_state.sqlite"
    npmc_attempt: dict[str, Any] | None = None

    with _power_lock(output_dir, power_index, power_count):
        # --resume means: extend an existing run by the user-specified length.
        # The previous cutoff is read from the existing npt_run_summary.json and the
        # new cutoff becomes previous + extension. This is not a shortcut for
        # re-summarizing finished runs or creating new points.
        resume_this_power = bool(args.resume)
        if bool(args.resume):
            if not final_np_db_path.exists() or not final_initial_state_db_path.exists():
                raise FileNotFoundError(
                    f"[power {power_index + 1}/{power_count}] resume requested but "
                    f"required databases are missing in {output_dir}"
                )
            np_db_path = final_np_db_path
            initial_state_db_path = final_initial_state_db_path
            existing_manifest_path = output_dir / "npt_interaction_manifest.json"
            if not existing_manifest_path.exists():
                raise FileNotFoundError(
                    f"[power {power_index + 1}/{power_count}] resume requires an "
                    f"existing interaction manifest: {existing_manifest_path}"
                )
            with open(existing_manifest_path, "r") as f:
                existing_manifest = json.load(f)
            expected_fingerprint = resume_manifest_fingerprint(manifest)
            existing_fingerprint = resume_manifest_fingerprint(existing_manifest)
            if existing_fingerprint != expected_fingerprint:
                raise ValueError(
                    f"[power {power_index + 1}/{power_count}] resume configuration "
                    "does not match the existing NPMC database; start a new run "
                    "directory or restore the original settings."
                )
            # The existing DB is authoritative for resumed trajectories. Retain
            # its manifest after proving the requested configuration is identical.
            manifest = existing_manifest
            interactions = manifest["interactions"]
            manifest["resume_manifest_fingerprint"] = existing_fingerprint
            summary_path = output_dir / "npt_run_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"[power {power_index + 1}/{power_count}] resume requires an "
                    f"existing npt_run_summary.json to determine the previous cutoff: "
                    f"{summary_path}"
                )
            with open(summary_path, "r") as f:
                prev_summary = json.load(f)
            expected_seeds = set(
                range(int(args.base_seed), int(args.base_seed) + int(args.num_sims))
            )
            previous_seed_data = prev_summary.get("n4_population_per_seed", {})
            previous_seeds = {int(seed) for seed in previous_seed_data}
            if previous_seeds != expected_seeds:
                raise ValueError(
                    f"[power {power_index + 1}/{power_count}] requested seeds do "
                    "not match the existing trajectory summary."
                )
            if simulation_cutoff_mode == "steps":
                prev_cutoff = prev_summary.get("simulation_step_cutoff")
                if prev_cutoff is None:
                    raise ValueError(
                        f"[power {power_index + 1}/{power_count}] previous "
                        f"simulation_step_cutoff not found in summary"
                    )
                resume_previous_step_cutoff = int(prev_cutoff)
                simulation_step_cutoff = int(prev_cutoff) + int(base_simulation_step_cutoff)
                print(
                    f"[power {power_index + 1}/{power_count}] resuming: extending from {prev_cutoff} steps by {base_simulation_step_cutoff} steps to {simulation_step_cutoff}",
                    flush=True,
                )
                if args.skip_npmc:
                    simulation_step_cutoff = resume_previous_step_cutoff
                    print(
                        f"[power {power_index + 1}/{power_count}] --skip-npmc: using existing cutoff {simulation_step_cutoff}",
                        flush=True,
                    )
                if not args.skip_npmc:
                    validate_npmc_checkpoint(
                        initial_state_db_path,
                        final_step_cutoff=int(prev_cutoff),
                        base_seed=int(args.base_seed),
                        num_sims=int(args.num_sims),
                    )
                    print(
                        f"[power {power_index + 1}/{power_count}] using the "
                        "stored NPMC checkpoint without rebuilding state.",
                        flush=True,
                    )
            elif simulation_cutoff_mode == "physical-time":
                prev_cutoff = prev_summary.get("simulation_time_cutoff_s")
                if prev_cutoff is None:
                    raise ValueError(
                        f"[power {power_index + 1}/{power_count}] previous "
                        f"simulation_time_cutoff_s not found in summary"
                    )
                simulation_time_cutoff_s = float(prev_cutoff) + float(base_simulation_time_cutoff_s)
                print(
                    f"[power {power_index + 1}/{power_count}] resuming: extending from {prev_cutoff} s by {base_simulation_time_cutoff_s} s to {simulation_time_cutoff_s}",
                    flush=True,
                )
                # Step-based checkpoint helper cannot be used for physical-time
                # resumes; physical-time resume is not yet supported.
                raise NotImplementedError(
                    f"[power {power_index + 1}/{power_count}] physical-time resume "
                    f"is not supported"
                )
            run_npmc_needed = True
            if args.skip_npmc:
                run_npmc_needed = False
        else:
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
                reuse_existing=args.continue_run,
            )
            simulation_step_cutoff = base_simulation_step_cutoff
            simulation_time_cutoff_s = base_simulation_time_cutoff_s
            run_npmc_needed = True
        cutoff_label = (
            f"{simulation_step_cutoff} steps"
            if simulation_cutoff_mode == "steps"
            else f"{simulation_time_cutoff_s} s"
        )
        print(
            f"[power {power_index + 1}/{power_count}] simulation cutoff: {cutoff_label}",
            flush=True,
        )

        manifest["simulation_step_cutoff"] = (
            None if simulation_step_cutoff is None else int(simulation_step_cutoff)
        )
        manifest["simulation_time_cutoff_s"] = (
            None if simulation_time_cutoff_s is None else float(simulation_time_cutoff_s)
        )

        manifest_path = output_dir / "npt_interaction_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(json_safe(manifest), f, indent=2)

        build_record = {
            "power_index": int(power_index),
            "excitation_power_w_cm2": float(power),
            "output_dir": str(output_dir.resolve()),
            "interaction_count": int(len(interactions)),
            "manifest_path": str(manifest_path.resolve()),
            "np_db_path": str((output_dir / "np.sqlite").resolve()),
            "initial_state_db_path": str((output_dir / "initial_state.sqlite").resolve()),
            "simulation_cutoff_mode": simulation_cutoff_mode,
            "simulation_step_cutoff": (
                None if simulation_step_cutoff is None else int(simulation_step_cutoff)
            ),
            "simulation_time_cutoff_s": (
                None if simulation_time_cutoff_s is None else float(simulation_time_cutoff_s)
            ),
            "resume": bool(args.resume),
            "checkpoint_requested": 1,
            "resume_previous_step_cutoff": resume_previous_step_cutoff,
            "resume_extension_steps": (
                None
                if not args.resume or base_simulation_step_cutoff is None
                else 0 if args.skip_npmc
                else int(base_simulation_step_cutoff)
            ),
        }
        if args.dry_run:
            return build_record, None

        if run_npmc_needed:
            npmc_attempt = run_npmc(
                np_db_path,
                initial_state_db_path,
                output_dir,
                args,
                simulation_cutoff_mode=simulation_cutoff_mode,
                simulation_step_cutoff=simulation_step_cutoff,
                simulation_time_cutoff_s=simulation_time_cutoff_s,
            )
            if simulation_cutoff_mode == "steps":
                validate_npmc_checkpoint(
                    initial_state_db_path,
                    final_step_cutoff=int(simulation_step_cutoff),
                    base_seed=int(args.base_seed),
                    num_sims=int(args.num_sims),
                )
    trace_replay_kwargs: dict[str, int] = {}
    if simulation_step_cutoff is not None:
        trace_replay_kwargs = {
            "n4_trace_samples": N4_TRAJECTORY_TRACE_SAMPLES,
            "n4_trace_final_step": int(simulation_step_cutoff),
        }
    replay = replay_trajectories(
        initial_state_db_path,
        interactions,
        **trace_replay_kwargs,
    )
    validate_replay_completion(
        replay,
        final_step_cutoff=simulation_step_cutoff,
        base_seed=int(args.base_seed),
        num_sims=int(args.num_sims),
    )
    summary = summarize_run(
        replay=replay,
        interactions=interactions,
        n_sites=int(manifest["geometry"]["ion_count"]),
        excitation_power=float(power),
        manifest=manifest,
        simulation_cutoff_mode=simulation_cutoff_mode,
        simulation_step_cutoff=simulation_step_cutoff,
        simulation_time_cutoff_s=simulation_time_cutoff_s,
    )
    summary["run_debug"] = collect_run_debug(
        output_dir=output_dir,
        initial_state_db_path=initial_state_db_path,
        resume=bool(args.resume),
        previous_step_cutoff=resume_previous_step_cutoff,
        requested_extension_steps=(
            None
            if not args.resume or base_simulation_step_cutoff is None
            else int(base_simulation_step_cutoff)
        ),
        final_step_cutoff=simulation_step_cutoff,
        summary=summary,
        npmc_attempt=npmc_attempt,
    )
    with open(output_dir / "npt_run_summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    build_record["run_debug"] = summary["run_debug"]
    plot_npt_style_spectrum(
        params=params,
        excitation_power=float(power),
        tm_fraction=tm_fraction,
        config=config,
        summary=summary,
        interactions=interactions,
        output_dir=output_dir,
    )
    if replay["n4_trace_by_seed"]:
        plot_n4_convergence.plot_n4_trajectory(
            replay["n4_trace_by_seed"],
            output_dir / "n4_trajectory.png",
            excitation_power=float(power),
        )
    plot_mechanism_diagrams.plot_run(
        output_dir,
        output_name="mechanism_diagram.png",
        rate_key="events_per_ion_s",
        min_rate=1e-3,
        max_by_kind={"ET": 16, "Rad": 18, "NR": 12, "Pump": 8, "SQ": 16},
        dpi=220,
    )

    move_output_file(np_db_path, output_dir / "np.sqlite")
    if resume_this_power:
        archived_initial_state_db_path = final_initial_state_db_path.resolve()
    else:
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


def run_fresh_prepared_power_point(
    *,
    power_index: int,
    power_count: int,
    power: float,
    preparation_steps: int,
    output_root: Path,
    params: dict[str, Any],
    source_np_db_path: Path,
    include_zero_rates: bool,
    tm_fraction: float | None,
    config: dict[str, Any],
    args: argparse.Namespace,
    trajectory_archive_root: Path | None,
    local_db_staging_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run a fixed-50k preparation followed by target-only trajectories."""
    output_dir = output_root / f"power_{power_index:02d}_{power:.6g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_step_cutoff = int(simulation_step_cutoff_for_power(args))
    seeds = list(range(int(args.base_seed), int(args.base_seed) + int(args.num_sims)))
    print(
        f"[power {power_index + 1}/{power_count}] target={power:.6g} W cm^-2, "
        f"preparation={PREPARATION_POWER_W_CM2:.6g} W cm^-2 for "
        f"{preparation_steps:,} steps",
        flush=True,
    )

    target_interactions, target_manifest = build_custom_interactions(
        params=params,
        source_np_db_path=source_np_db_path,
        excitation_power=float(power),
        include_zero_rates=include_zero_rates,
        tm_fraction=tm_fraction,
        config=config,
    )
    target_manifest.update(
        {
            "simulation_cutoff_mode": "steps",
            "simulation_step_cutoff": target_step_cutoff,
            "simulation_time_cutoff_s": None,
        }
    )
    target_manifest_path = output_dir / "npt_interaction_manifest.json"
    with open(target_manifest_path, "w") as handle:
        json.dump(json_safe(target_manifest), handle, indent=2)

    preparation_dir = output_dir / "preparation"
    preparation_work_dir = prepare_power_db_work_dir(
        preparation_dir,
        local_db_staging_root=local_db_staging_root,
        dry_run=bool(args.dry_run),
    )
    preparation_interactions, preparation_manifest = build_custom_interactions(
        params=params,
        source_np_db_path=source_np_db_path,
        excitation_power=PREPARATION_POWER_W_CM2,
        include_zero_rates=include_zero_rates,
        tm_fraction=tm_fraction,
        config=config,
    )
    preparation_manifest.update(
        {
            "simulation_cutoff_mode": "steps",
            "simulation_step_cutoff": int(preparation_steps),
            "simulation_time_cutoff_s": None,
        }
    )
    preparation_np_path, preparation_state_path = write_custom_npmc_databases(
        source_np_db_path=source_np_db_path,
        output_dir=preparation_work_dir,
        interactions=preparation_interactions,
        interaction_radius_bound_nm=float(
            params["simulation_defaults"]["interaction_radius_bound_nm"]
        ),
        distance_factor_type=str(
            params["simulation_defaults"]["distance_factor_type"]
        ),
        reuse_existing=args.continue_run,
    )
    with open(preparation_dir / "npt_interaction_manifest.json", "w") as handle:
        json.dump(json_safe(preparation_manifest), handle, indent=2)
    preparation_attempt: dict[str, Any] | None = None
    if not args.dry_run:
        preparation_attempt = run_npmc(
            preparation_np_path,
            preparation_state_path,
            preparation_dir,
            args,
            simulation_cutoff_mode="steps",
            simulation_step_cutoff=int(preparation_steps),
            simulation_time_cutoff_s=None,
        )
        validate_npmc_checkpoint(
            preparation_state_path,
            final_step_cutoff=int(preparation_steps),
            base_seed=int(args.base_seed),
            num_sims=int(args.num_sims),
        )

    template_dir = output_dir / "restart" / "_template"
    template_np_path, template_state_path = write_custom_npmc_databases(
        source_np_db_path=source_np_db_path,
        output_dir=template_dir,
        interactions=target_interactions,
        interaction_radius_bound_nm=float(
            params["simulation_defaults"]["interaction_radius_bound_nm"]
        ),
        distance_factor_type=str(
            params["simulation_defaults"]["distance_factor_type"]
        ),
    )
    move_output_file(template_np_path, output_dir / "np.sqlite")
    seed_records: list[dict[str, Any]] = []
    seed_preparation: dict[str, Any] = {}
    for seed in seeds:
        seed_dir = output_dir / "restart" / f"seed_{seed}"
        work_dir = prepare_power_db_work_dir(
            seed_dir,
            local_db_staging_root=local_db_staging_root,
            dry_run=bool(args.dry_run),
        )
        initial_state_db_path = work_dir / "initial_state.sqlite"
        np_db_path = work_dir / "np.sqlite"
        preparation_path = seed_dir / "preparation_state.json"
        if args.continue_run and initial_state_db_path.exists():
            if not preparation_path.exists() or not np_db_path.exists():
                raise RuntimeError(f"Incomplete seed initialization in {seed_dir}")
            seed_preparation[str(seed)] = json.loads(preparation_path.read_text())
            seed_records.append(
                {
                    "seed": seed,
                    "seed_dir": seed_dir,
                    "work_dir": work_dir,
                    "np_db_path": np_db_path,
                    "initial_state_db_path": initial_state_db_path,
                }
            )
            continue
        shutil.copy2(template_state_path, initial_state_db_path)
        link_target_np_database(np_db_path, output_dir / "np.sqlite")
        preparation_record: dict[str, Any] = {
            "seed": int(seed),
            "status": "database-only" if args.dry_run else "transplanted",
        }
        if not args.dry_run:
            state_rows, checkpoint_step, checkpoint_time_s = (
                load_preparation_checkpoint_state(preparation_state_path, seed)
            )
            state_counts = transplant_preparation_state(
                initial_state_db_path=initial_state_db_path,
                np_db_path=np_db_path,
                state_rows=state_rows,
            )
            preparation_record.update(
                {
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_time_s": checkpoint_time_s,
                    "site_count": len(state_rows),
                    "state_counts": state_counts,
                }
            )
        seed_preparation[str(seed)] = preparation_record
        atomic_json(preparation_path, preparation_record)
        seed_records.append(
            {
                "seed": seed,
                "seed_dir": seed_dir,
                "work_dir": work_dir,
                "np_db_path": np_db_path,
                "initial_state_db_path": initial_state_db_path,
            }
        )
    if template_state_path.exists():
        template_state_path.unlink()
    if template_dir.exists():
        template_dir.rmdir()

    protocol = {
        "protocol": "fixed-power-checkpoint-preparation",
        "preparation_power_w_cm2": PREPARATION_POWER_W_CM2,
        "preparation_step_cutoff": int(preparation_steps),
        "target_power_w_cm2": float(power),
        "target_step_cutoff": target_step_cutoff,
        "analyzed_phase": "target-only",
        "base_seed": int(args.base_seed),
        "num_sims": int(args.num_sims),
        "seed_preparation": seed_preparation,
        "resume_extensions": [],
    }
    with open(output_dir / "preparation_protocol.json", "w") as handle:
        json.dump(json_safe(protocol), handle, indent=2)
    build_record = {
        "power_index": int(power_index),
        "excitation_power_w_cm2": float(power),
        "preparation_power_w_cm2": PREPARATION_POWER_W_CM2,
        "preparation_step_cutoff": int(preparation_steps),
        "output_dir": str(output_dir.resolve()),
        "interaction_count": len(target_interactions),
        "manifest_path": str(target_manifest_path.resolve()),
        "np_db_path": str((output_dir / "np.sqlite").resolve()),
        "simulation_cutoff_mode": "steps",
        "simulation_step_cutoff": target_step_cutoff,
        "resume": False,
        "checkpoint_requested": 1,
    }
    if args.dry_run:
        return build_record, None

    move_output_file(preparation_np_path, preparation_dir / "np.sqlite")
    finalize_initial_state_database(
        initial_state_db_path=preparation_state_path,
        output_dir=preparation_dir,
        archive_root=trajectory_archive_root,
    )
    if preparation_work_dir != preparation_dir and preparation_work_dir.exists():
        preparation_work_dir.rmdir()
    with ThreadPoolExecutor(max_workers=int(args.restart_workers)) as executor:
        futures = {
            executor.submit(
                run_prepared_target_seed,
                record,
                args=args,
                final_step_cutoff=target_step_cutoff,
            ): int(record["seed"])
            for record in seed_records
        }
        for future in checkpoint_futures(futures):
            seed = futures[future]
            future.result()
            print(f"Target restart complete for seed {seed}", flush=True)
    replay = replay_prepared_target_seeds(
        seed_records,
        interactions=target_interactions,
        final_step_cutoff=target_step_cutoff,
    )
    attempts = {
        str(record["seed"]): record.get("npmc_attempt") for record in seed_records
    }
    summary = summarize_prepared_power_point(
        replay=replay,
        interactions=target_interactions,
        manifest=target_manifest,
        protocol=protocol,
        power=power,
        final_step_cutoff=target_step_cutoff,
        params=params,
        config=config,
        args=args,
        output_dir=output_dir,
        npmc_attempts=attempts,
        resume_debug={
            "resume": False,
            "preparation_npmc_attempt": preparation_attempt,
        },
    )
    build_record["initial_state_db_paths_by_seed"] = (
        finalize_prepared_seed_databases(
            seed_records,
            output_dir=output_dir,
            archive_root=trajectory_archive_root,
        )
    )
    return build_record, summary


def run_resumed_prepared_power_point(
    *,
    power_index: int,
    power_count: int,
    power: float,
    preparation_steps: int,
    output_root: Path,
    params: dict[str, Any],
    source_np_db_path: Path,
    include_zero_rates: bool,
    tm_fraction: float | None,
    config: dict[str, Any],
    args: argparse.Namespace,
    trajectory_archive_root: Path | None,
    local_db_staging_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extend existing per-seed target checkpoints without rerunning preparation."""
    _ = power_count, local_db_staging_root
    output_dir = output_root / f"power_{power_index:02d}_{power:.6g}"
    summary_path = output_dir / "npt_run_summary.json"
    manifest_path = output_dir / "npt_interaction_manifest.json"
    protocol_path = output_dir / "preparation_protocol.json"
    for path in (summary_path, manifest_path, protocol_path, output_dir / "np.sqlite"):
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume prepared target; missing {path}")

    with open(summary_path) as handle:
        previous_summary = json.load(handle)
    previous_cutoff_value = previous_summary.get("simulation_step_cutoff")
    if previous_cutoff_value is None:
        raise ValueError(f"Previous target cutoff missing from {summary_path}")
    previous_cutoff = int(previous_cutoff_value)
    extension_steps = int(simulation_step_cutoff_for_power(args))
    final_step_cutoff = (
        previous_cutoff if args.skip_npmc else previous_cutoff + extension_steps
    )
    with open(manifest_path) as handle:
        stored_manifest = json.load(handle)
    _expected_interactions, expected_manifest = build_custom_interactions(
        params=params,
        source_np_db_path=source_np_db_path,
        excitation_power=float(power),
        include_zero_rates=include_zero_rates,
        tm_fraction=tm_fraction,
        config=config,
    )
    expected_manifest["simulation_cutoff_mode"] = "steps"
    if resume_manifest_fingerprint(stored_manifest) != resume_manifest_fingerprint(
        expected_manifest
    ):
        raise ValueError(
            f"Requested target configuration does not match {manifest_path}"
        )
    with open(protocol_path) as handle:
        protocol = json.load(handle)
    if (
        float(protocol.get("preparation_power_w_cm2", -1.0))
        != PREPARATION_POWER_W_CM2
        or int(protocol.get("preparation_step_cutoff", -1))
        != int(preparation_steps)
    ):
        raise ValueError(
            "Existing preparation protocol does not match the hardcoded schedule"
        )

    seeds = list(range(int(args.base_seed), int(args.base_seed) + int(args.num_sims)))
    if {
        int(seed) for seed in previous_summary.get("n4_population_per_seed", {})
    } != set(seeds):
        raise ValueError(
            "Requested seeds do not match the existing prepared target summary"
        )
    seed_records: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_dir / "restart" / f"seed_{seed}"
        np_db_path = seed_dir / "np.sqlite"
        initial_state_db_path = seed_dir / "initial_state.sqlite"
        for path in (np_db_path, initial_state_db_path):
            if not path.exists():
                raise FileNotFoundError(f"Cannot resume seed {seed}; missing {path}")
        if not args.skip_npmc:
            validate_npmc_checkpoint(
                initial_state_db_path,
                final_step_cutoff=previous_cutoff,
                base_seed=seed,
                num_sims=1,
            )
        seed_records.append(
            {
                "seed": seed,
                "seed_dir": seed_dir,
                "work_dir": seed_dir,
                "np_db_path": np_db_path,
                "initial_state_db_path": initial_state_db_path,
            }
        )
    if not args.skip_npmc:
        with ThreadPoolExecutor(max_workers=int(args.restart_workers)) as executor:
            futures = {
                executor.submit(
                    run_prepared_target_seed,
                    record,
                    args=args,
                    final_step_cutoff=final_step_cutoff,
                ): int(record["seed"])
                for record in seed_records
            }
            for future in checkpoint_futures(futures):
                seed = futures[future]
                future.result()
                print(
                    f"Extended target seed {seed} to {final_step_cutoff:,} steps",
                    flush=True,
                )
    replay = replay_prepared_target_seeds(
        seed_records,
        interactions=stored_manifest["interactions"],
        final_step_cutoff=final_step_cutoff,
    )
    stored_manifest["simulation_step_cutoff"] = final_step_cutoff
    with open(manifest_path, "w") as handle:
        json.dump(json_safe(stored_manifest), handle, indent=2)
    extensions = protocol.get("resume_extensions", [])
    if not isinstance(extensions, list):
        extensions = []
    extensions.append(
        {
            "recorded_at_utc": utc_now_iso(),
            "previous_step_cutoff": previous_cutoff,
            "extension_steps": 0 if args.skip_npmc else extension_steps,
            "final_step_cutoff": final_step_cutoff,
            "skip_npmc": bool(args.skip_npmc),
        }
    )
    protocol["resume_extensions"] = extensions
    protocol["target_step_cutoff"] = final_step_cutoff
    with open(protocol_path, "w") as handle:
        json.dump(json_safe(protocol), handle, indent=2)
    attempts = {
        str(record["seed"]): record.get("npmc_attempt") for record in seed_records
    }
    summary = summarize_prepared_power_point(
        replay=replay,
        interactions=stored_manifest["interactions"],
        manifest=stored_manifest,
        protocol=protocol,
        power=power,
        final_step_cutoff=final_step_cutoff,
        params=params,
        config=config,
        args=args,
        output_dir=output_dir,
        npmc_attempts=attempts,
        resume_debug={
            "resume": True,
            "previous_step_cutoff": previous_cutoff,
            "requested_extension_steps": 0 if args.skip_npmc else extension_steps,
            "final_step_cutoff": final_step_cutoff,
        },
    )
    archived_paths = finalize_prepared_seed_databases(
        seed_records,
        output_dir=output_dir,
        archive_root=trajectory_archive_root,
    )
    build_record = {
        "power_index": int(power_index),
        "excitation_power_w_cm2": float(power),
        "preparation_power_w_cm2": PREPARATION_POWER_W_CM2,
        "preparation_step_cutoff": int(preparation_steps),
        "output_dir": str(output_dir.resolve()),
        "interaction_count": len(stored_manifest["interactions"]),
        "manifest_path": str(manifest_path.resolve()),
        "np_db_path": str((output_dir / "np.sqlite").resolve()),
        "initial_state_db_paths_by_seed": archived_paths,
        "simulation_cutoff_mode": "steps",
        "simulation_step_cutoff": final_step_cutoff,
        "resume": True,
        "resume_previous_step_cutoff": previous_cutoff,
        "resume_extension_steps": 0 if args.skip_npmc else extension_steps,
        "checkpoint_requested": 1,
    }
    return build_record, summary


def run_configured_power_point(
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    checkpoint_boundary()
    output_dir = Path(kwargs["output_root"]) / (
        f"power_{kwargs['power_index']:02d}_{kwargs['power']:.6g}"
    )
    done_path = output_dir / ".power_complete.json"
    if kwargs["args"].continue_run and done_path.exists():
        record = json.loads(done_path.read_text())
        return record["build_record"], record["summary"]
    result = _run_configured_power_point(**kwargs)
    checkpoint_boundary()
    if kwargs["args"].continue_run:
        atomic_json(done_path, {"build_record": result[0], "summary": result[1]})
    return result


def _run_configured_power_point(
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Select the cold-start or fixed-preparation production path."""
    args: argparse.Namespace = kwargs["args"]
    power = float(kwargs["power"])
    preparation_steps = preparation_steps_for_power(
        power,
        enabled=not bool(args.disable_preparation),
    )
    if preparation_steps == 0:
        return run_power_point(**kwargs)
    power_index = int(kwargs["power_index"])
    power_count = int(kwargs["power_count"])
    output_dir = Path(kwargs["output_root"]) / (
        f"power_{power_index:02d}_{power:.6g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with _power_lock(output_dir, power_index, power_count):
        if args.resume:
            return run_resumed_prepared_power_point(
                **kwargs,
                preparation_steps=preparation_steps,
            )
        return run_fresh_prepared_power_point(
            **kwargs,
            preparation_steps=preparation_steps,
        )


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


NPT_STYLE_SPECTRUM_REFERENCE = "NanoParticleTools NPMC spectrum convention"


def plot_npt_style_spectrum(
    *,
    params: dict[str, Any],
    excitation_power: float,
    tm_fraction: float | None,
    config: dict[str, Any],
    summary: dict[str, Any],
    interactions: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save an NPT-style signed-wavelength log-intensity spectrum plot.

    This reproduces the visualization from the reference notebook
    RNMC/examples/NPMC/nano_particle_analysis.ipynb.
    """
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
        upper_bound=1000,
        step=5,
    )
    wavelength_signed = np.asarray(wavelength_signed, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    order = np.argsort(wavelength_signed)
    wavelength_signed = wavelength_signed[order]
    spectrum = spectrum[order]

    spectrum_data = {
        "source": "NanoParticleTools.analysis.util.get_spectrum_wavelength_from_dndt",
        "style_reference": NPT_STYLE_SPECTRUM_REFERENCE,
        "excitation_power_w_cm2": float(excitation_power),
        "x_axis": "signed_wavelength_nm_negative_emission_positive_absorption",
        "y_axis": "radiative_events_per_particle_s",
        "wavelength_nm": [float(value) for value in wavelength_signed],
        "spectrum": [float(value) for value in spectrum],
    }
    if "analysis_window" in summary:
        spectrum_data["analysis_window"] = summary["analysis_window"]
    with open(output_dir / "npt_emission_spectrum.json", "w") as f:
        json.dump(json_safe(spectrum_data), f, indent=2)

    fig, ax = plt.subplots(dpi=300, figsize=(5.0, 4.0))
    ax.plot(wavelength_signed, np.log10(np.add(spectrum, 1)), linewidth=0.5)

    # Emission/absorption arrow bar exactly as in the reference notebook.
    ax.arrow(
        50,
        -0.5,
        900,
        0,
        width=0.01,
        head_width=0.2,
        head_length=50,
        lw=12,
        color="red",
        length_includes_head=True,
    )
    ax.arrow(
        -50,
        -0.5,
        -1900,
        0,
        width=0.01,
        head_width=0.2,
        head_length=50,
        lw=12,
        color="green",
        length_includes_head=True,
    )
    ax.text(200, -0.57, "absorption", color="white")
    ax.text(-1200, -0.57, "emissions", color="white")

    ax.set_ylim(-1, 6)
    ax.set_xticks([-2000, -1000, 0, 1000])
    ax.set_xticklabels(["-2000", "-1000", "0", "1000"], fontsize=13)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["0", "1", "10", "100", "1k", "10k"], fontsize=13)
    ax.set_xlabel("wavelength (nm)", fontsize=20)
    ax.set_ylabel("Log10(Intensity)", fontsize=20)
    ax.set_title(
        f"{format_power_tick(float(excitation_power))} W/cm2",
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "npt_emission_spectrum.png")
    plt.close(fig)


def finalize_initial_state_database(
    initial_state_db_path: Path,
    output_dir: Path,
    archive_root: Path | None,
) -> Path:
    """Materialize the completed trajectory DB in the final output location."""
    if not initial_state_db_path.exists():
        raise FileNotFoundError(
            f"Missing completed trajectory database: {initial_state_db_path}"
        )
    final_initial_state_db_path = output_dir / "initial_state.sqlite"
    if archive_root is None:
        return move_output_file(initial_state_db_path, final_initial_state_db_path)
    if initial_state_db_path.is_symlink():
        archived_path = initial_state_db_path.resolve()
        if final_initial_state_db_path != initial_state_db_path:
            if final_initial_state_db_path.is_symlink() or final_initial_state_db_path.exists():
                final_initial_state_db_path.unlink()
            final_initial_state_db_path.symlink_to(archived_path)
        return archived_path
    if not archive_root.exists():
        # If the HDD mount is missing entirely, fall back to local storage.
        # If only the project-specific subfolder is missing, create it so
        # completed trajectories are archived instead of filling local disk.
        if not archive_root.parent.exists():
            return move_output_file(
                initial_state_db_path, final_initial_state_db_path
            )
        archive_root.mkdir(parents=True, exist_ok=True)

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



def summarize_run(
    replay: dict[str, Any],
    interactions: list[dict[str, Any]],
    n_sites: int,
    excitation_power: float,
    manifest: dict[str, Any],
    simulation_cutoff_mode: str,
    simulation_step_cutoff: int | None,
    simulation_time_cutoff_s: float | None,
) -> dict[str, Any]:
    """Build one compact summary for a completed power point."""
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
    w5_nr_report = next(
        (row for row in manifest["one_site"] if row["label"] == "NR 5->4"),
        None,
    )
    w3_nr_report = next(
        (row for row in manifest["one_site"] if row["label"] == "NR 3->2"),
        None,
    )
    interaction_flux_by_label = {
        row["label"]: row["events_per_ion_s"]
        for row in per_interaction
    }

    return {
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
        "n4_late_window_averages": compute_n4_late_window_averages(replay),
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
        "w3_nr_base_rate_s^-1": (
            float(w3_nr_report["base_rate_s^-1"]) if w3_nr_report is not None else None
        ),
        "w3_nr_rate_scale_factor": (
            float(w3_nr_report["rate_scale_factor"]) if w3_nr_report is not None else None
        ),
        "w3_nr_rate_s^-1": (
            float(w3_nr_report["rate_s^-1"]) if w3_nr_report is not None else None
        ),
        "w5_nr_base_rate_s^-1": (
            float(w5_nr_report["base_rate_s^-1"]) if w5_nr_report is not None else None
        ),
        "w5_nr_rate_scale_factor": (
            float(w5_nr_report["rate_scale_factor"]) if w5_nr_report is not None else None
        ),
        "w5_nr_rate_s^-1": (
            float(w5_nr_report["rate_s^-1"]) if w5_nr_report is not None else None
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


def _load_existing_power_summaries(output_root: Path) -> list[dict[str, Any]]:
    """Load the latest summary for every existing power directory."""
    summaries: list[dict[str, Any]] = []
    if not output_root.exists():
        return summaries
    for power_dir in sorted(output_root.glob("power_*")):
        summary_path = power_dir / "npt_run_summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r") as f:
                summaries.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def _load_existing_sweep_summary(output_root: Path) -> dict[str, Any] | None:
    """Load the prior aggregate summary so resume runs can append to it."""
    summary_path = output_root / "npt_power_sweep_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r") as f:
            value = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def merge_build_records(
    existing_sweep_summary: dict[str, Any] | None,
    updated_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge resumed records into the full, canonical sweep build record list."""
    merged: dict[int, dict[str, Any]] = {}
    if existing_sweep_summary is not None:
        for row in existing_sweep_summary.get("build_records", []):
            if isinstance(row, dict) and "power_index" in row:
                merged[int(row["power_index"])] = row
    for row in updated_records:
        merged[int(row["power_index"])] = row
    return [merged[index] for index in sorted(merged)]


def _next_resume_plot_path(output_root: Path) -> Path:
    """Return the first available npt_avalanche_curve_resume{N}.png path."""
    for i in range(10):
        candidate = output_root / f"npt_avalanche_curve_resume{i}.png"
        if not candidate.exists():
            return candidate
    return output_root / "npt_avalanche_curve_resume9.png"


def plot_avalanche_curve(
    summaries: list[dict[str, Any]],
    output_root: Path,
    resume: bool = False,
) -> None:
    """Plot 700 nm and 800 nm avalanche proxies versus excitation power."""
    if not summaries:
        return
    summaries = sorted(summaries, key=lambda row: float(row["excitation_power_w_cm2"]))
    powers = np.asarray([row["excitation_power_w_cm2"] for row in summaries], dtype=float)
    emission_800 = np.asarray(
        [row["rad_800_proxy_events_per_particle_s"] for row in summaries],
        dtype=float,
    )
    emission_700 = np.asarray(
        [row["rad_700_proxy_events_per_particle_s"] for row in summaries],
        dtype=float,
    )

    fig, ax = plt.subplots(dpi=300, figsize=(6.2, 4.4))
    ax.loglog(
        powers,
        np.maximum(emission_800, 1e-300),
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label="800 nm proxy (3H4 radiative)",
    )
    ax.loglog(
        powers,
        np.maximum(emission_700, 1e-300),
        marker="s",
        linewidth=1.8,
        markersize=4.5,
        label="700 nm proxy (3F3 radiative)",
    )
    ax.set_xlabel("Excitation power at 1064 nm (W/cm2)", fontsize=13)
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
        fontsize=9.2,
        borderpad=0.65,
        labelspacing=0.45,
        handlelength=1.8,
        handletextpad=0.6,
        markerscale=1.05,
    )
    fig.subplots_adjust(bottom=0.18, left=0.14, right=0.97, top=0.96)
    if resume:
        resume_path = _next_resume_plot_path(output_root)
        fig.savefig(resume_path)
        print(f"Wrote resume avalanche curve: {resume_path}", flush=True)
    else:
        fig.savefig(output_root / "npt_avalanche_curve.png")
    plt.close(fig)


def _aggregate_restart_step_window(
    initial_state_db_path: Path,
    window_steps: int,
) -> dict[str, Any]:
    """Aggregate one single-seed restart DB over its final event window.

    Restart DBs are written in step order with ``rowid == step + 1``.  Using
    that invariant keeps this post-processing read bounded to the requested
    late window instead of replaying the full trajectory.
    """
    with sqlite3.connect(f"file:{initial_state_db_path.resolve()}?mode=ro", uri=True) as con:
        cutoff_rows = con.execute(
            "SELECT seed, step FROM interrupt_cutoff"
        ).fetchall()
        if len(cutoff_rows) != 1:
            raise ValueError(
                f"Expected one restart seed in {initial_state_db_path}, "
                f"found {len(cutoff_rows)}"
            )
        seed, checkpoint_step = map(int, cutoff_rows[0])
        final_step = checkpoint_step - 1
        start_step = final_step - int(window_steps) + 1
        if start_step < 1:
            raise ValueError(
                f"Trajectory {initial_state_db_path} has fewer than "
                f"{window_steps:,} steps"
            )

        boundary = con.execute(
            "SELECT seed, step, time FROM trajectories WHERE rowid = ?",
            (start_step,),
        ).fetchone()
        first = con.execute(
            "SELECT seed, step, time FROM trajectories WHERE rowid = ?",
            (start_step + 1,),
        ).fetchone()
        last = con.execute(
            "SELECT seed, step, time FROM trajectories WHERE rowid = ?",
            (final_step + 1,),
        ).fetchone()
        expected = (
            (seed, start_step - 1),
            (seed, start_step),
            (seed, final_step),
        )
        observed = tuple(
            None if row is None else (int(row[0]), int(row[1]))
            for row in (boundary, first, last)
        )
        if observed != expected:
            raise ValueError(
                "Restart trajectory row ordering is not step-contiguous in "
                f"{initial_state_db_path}: expected {expected}, found {observed}"
            )

        counts = {
            int(interaction_id): int(count)
            for interaction_id, count in con.execute(
                "SELECT interaction_id, COUNT(*) FROM trajectories "
                "WHERE rowid BETWEEN ? AND ? GROUP BY interaction_id",
                (start_step + 1, final_step + 1),
            )
        }
        if sum(counts.values()) != int(window_steps):
            raise RuntimeError(
                f"Late-window count mismatch in {initial_state_db_path}: "
                f"expected {window_steps:,}, found {sum(counts.values()):,}"
            )
        elapsed_time = float(last[2]) - float(boundary[2])
        if elapsed_time <= 0.0:
            raise ValueError(f"Non-positive late-window duration in {initial_state_db_path}")
    return {
        "seed": seed,
        "start_step": start_step,
        "final_step": final_step,
        "elapsed_time_s": elapsed_time,
        "event_counts": counts,
    }


def _aggregate_combined_step_windows(
    initial_state_db_path: Path,
    window_steps: int,
    chunk_rows: int = 2_000_000,
) -> list[dict[str, Any]]:
    """Aggregate final per-seed windows from an interleaved trajectory DB.

    Older production points store all seeds in one table. Rows from different
    seeds are interleaved, so this scans a bounded tail in rowid chunks and
    aggregates each chunk in SQLite. The extra boundary event supplies the
    exact elapsed-time origin and is removed from the event counts afterward.
    """
    uri = f"file:{initial_state_db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        cutoff_rows = [
            (int(seed), int(step))
            for seed, step in con.execute(
                "SELECT seed, step FROM interrupt_cutoff ORDER BY seed"
            )
        ]
        if not cutoff_rows:
            raise ValueError(f"No completed seeds in {initial_state_db_path}")
        final_steps = {seed: checkpoint_step - 1 for seed, checkpoint_step in cutoff_rows}
        start_steps = {
            seed: final_step - int(window_steps) + 1
            for seed, final_step in final_steps.items()
        }
        if min(start_steps.values()) < 1:
            raise ValueError(
                f"Trajectory {initial_state_db_path} has fewer than "
                f"{window_steps:,} steps per seed"
            )
        boundary_steps = {start_step - 1 for start_step in start_steps.values()}
        if len(boundary_steps) != 1:
            raise ValueError(
                "Combined trajectory seeds have unequal final steps; cannot use "
                "one bounded tail scan"
            )
        boundary_step = boundary_steps.pop()
        max_rowid = int(
            con.execute("SELECT MAX(rowid) FROM trajectories").fetchone()[0]
        )

        # Normal interleaving needs N*window rows. Keep a two-window margin for
        # seed scheduling imbalance, then expand and retry only if validation
        # shows that a seed's boundary was not reached.
        tail_rows = int(window_steps) * len(cutoff_rows) + max(
            10_000_000, 2 * int(window_steps)
        )
        while True:
            lower_rowid = max(0, max_rowid - tail_rows)
            counts_by_seed: dict[int, Counter] = defaultdict(Counter)
            min_by_seed_interaction: dict[tuple[int, int], tuple[int, float]] = {}
            max_time_by_seed: dict[int, float] = {}
            for first_rowid in range(lower_rowid + 1, max_rowid + 1, int(chunk_rows)):
                last_rowid = min(max_rowid, first_rowid + int(chunk_rows) - 1)
                rows = con.execute(
                    "SELECT seed, interaction_id, COUNT(*), MIN(step), "
                    "MIN(time), MAX(time) FROM trajectories "
                    "WHERE rowid BETWEEN ? AND ? AND step >= ? "
                    "GROUP BY seed, interaction_id",
                    (first_rowid, last_rowid, boundary_step),
                )
                for seed, interaction_id, count, min_step, min_time, max_time in rows:
                    seed = int(seed)
                    interaction_id = int(interaction_id)
                    counts_by_seed[seed][interaction_id] += int(count)
                    key = (seed, interaction_id)
                    candidate = (int(min_step), float(min_time))
                    if key not in min_by_seed_interaction or candidate < min_by_seed_interaction[key]:
                        min_by_seed_interaction[key] = candidate
                    max_time_by_seed[seed] = max(
                        max_time_by_seed.get(seed, float("-inf")), float(max_time)
                    )

            expected_with_boundary = int(window_steps) + 1
            complete = all(
                sum(counts_by_seed[seed].values()) == expected_with_boundary
                for seed, _checkpoint_step in cutoff_rows
            )
            if complete:
                break
            if lower_rowid == 0:
                observed = {
                    seed: sum(counts_by_seed[seed].values())
                    for seed, _checkpoint_step in cutoff_rows
                }
                raise RuntimeError(
                    f"Combined late-window count mismatch in {initial_state_db_path}: "
                    f"expected {expected_with_boundary:,} including boundary, "
                    f"found {observed}"
                )
            tail_rows = min(max_rowid, tail_rows + int(window_steps) * len(cutoff_rows))

    windows: list[dict[str, Any]] = []
    for seed, _checkpoint_step in cutoff_rows:
        boundary_matches = [
            (interaction_id, time_s)
            for (candidate_seed, interaction_id), (step, time_s) in min_by_seed_interaction.items()
            if candidate_seed == seed and step == boundary_step
        ]
        if len(boundary_matches) != 1:
            raise RuntimeError(
                f"Expected one boundary event for seed {seed} in "
                f"{initial_state_db_path}, found {boundary_matches}"
            )
        boundary_interaction_id, boundary_time = boundary_matches[0]
        counts_by_seed[seed][boundary_interaction_id] -= 1
        if counts_by_seed[seed][boundary_interaction_id] == 0:
            del counts_by_seed[seed][boundary_interaction_id]
        if sum(counts_by_seed[seed].values()) != int(window_steps):
            raise RuntimeError(f"Final combined-window count mismatch for seed {seed}")
        elapsed_time = max_time_by_seed[seed] - boundary_time
        if elapsed_time <= 0.0:
            raise ValueError(f"Non-positive late-window duration for seed {seed}")
        windows.append(
            {
                "seed": seed,
                "start_step": start_steps[seed],
                "final_step": final_steps[seed],
                "elapsed_time_s": elapsed_time,
                "event_counts": dict(counts_by_seed[seed]),
            }
        )
    return windows


def replot_restart_step_window(
    *,
    output_root: Path,
    window_steps: int,
    params: dict[str, Any],
    config: dict[str, Any],
    tm_fraction: float | None,
    workers: int,
) -> None:
    """Regenerate plots from final N steps, preferring restart trajectories."""
    if window_steps <= 0:
        raise ValueError("--replot-last-steps must be positive")
    power_dirs = sorted(
        (
            path
            for path in output_root.glob("power_*")
            if (path / "npt_interaction_manifest.json").exists()
            and (path / "npt_run_summary.json").exists()
            and (
                list(path.glob("restart/seed_*/initial_state.sqlite"))
                or (path / "initial_state.sqlite").exists()
            )
        ),
        key=plot_mechanism_diagrams.power_sort_key,
    )
    if not power_dirs:
        raise FileNotFoundError(f"No completed trajectories found below {output_root}")

    summaries: list[dict[str, Any]] = []
    summary_name = f"npt_run_summary_last_{int(window_steps)}_steps.json"
    for power_dir in power_dirs:
        manifest = json.loads(
            (power_dir / "npt_interaction_manifest.json").read_text()
        )
        interactions = manifest["interactions"]
        restart_paths = sorted(power_dir.glob("restart/seed_*/initial_state.sqlite"))
        if restart_paths:
            source = "restart/seed_*/initial_state.sqlite"
            print(
                f"Aggregating last {window_steps:,} steps: {power_dir.name} "
                f"({len(restart_paths)} restart seeds)",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=min(max(1, workers), len(restart_paths))) as executor:
                windows = list(
                    executor.map(
                        lambda path: _aggregate_restart_step_window(path, window_steps),
                        restart_paths,
                    )
                )
        else:
            source = "initial_state.sqlite"
            print(
                f"Aggregating last {window_steps:,} steps: {power_dir.name} "
                "(combined trajectory fallback)",
                flush=True,
            )
            windows = _aggregate_combined_step_windows(
                power_dir / "initial_state.sqlite", window_steps
            )

        replay: dict[str, Any] = {
            "simulation_time": {
                row["seed"]: row["elapsed_time_s"] for row in windows
            },
            "final_step": {row["seed"]: row["final_step"] for row in windows},
            "event_counts": {
                row["seed"]: Counter(row["event_counts"]) for row in windows
            },
            # These occupancy observables require state replay and are not used
            # by any of the three requested event-rate plots.
            "n4_time_integral": {row["seed"]: 0.0 for row in windows},
            "n4_population_per_seed": {row["seed"]: 0.0 for row in windows},
            "n4_late_window_integrals": {
                row["seed"]: (0.0, 0.0) for row in windows
            },
            "total_site_count": int(manifest["geometry"]["total_site_count"]),
            "tm_site_count": int(manifest["geometry"]["tm_site_count"]),
            "q24_total_count": 0,
            "s12_total_count": 0,
            "q24_after_s12_same_pair_count": 0,
            "s12_after_q24_same_pair_count": 0,
            "n4_trace_by_seed": {},
        }
        summary = summarize_run(
            replay=replay,
            interactions=interactions,
            n_sites=int(manifest["geometry"]["ion_count"]),
            excitation_power=float(manifest["excitation_power_w_cm2"]),
            manifest=manifest,
            simulation_cutoff_mode="steps",
            simulation_step_cutoff=max(row["final_step"] for row in windows),
            simulation_time_cutoff_s=None,
        )
        summary["analysis_window"] = {
            "source": source,
            "selection": "last_steps_per_seed",
            "window_steps_per_seed": int(window_steps),
            "start_step_by_seed": {
                str(row["seed"]): int(row["start_step"]) for row in windows
            },
            "final_step_by_seed": {
                str(row["seed"]): int(row["final_step"]) for row in windows
            },
            "elapsed_time_s_by_seed": {
                str(row["seed"]): float(row["elapsed_time_s"]) for row in windows
            },
        }
        (power_dir / summary_name).write_text(
            json.dumps(json_safe(summary), indent=2) + "\n"
        )
        canonical_summary_path = power_dir / "npt_run_summary.json"
        canonical_summary = json.loads(canonical_summary_path.read_text())
        reports = canonical_summary.get("late_window_reports", [])
        if not isinstance(reports, list):
            reports = []
        report = {
            "analysis_window": summary["analysis_window"],
            "summary_file": summary_name,
            "num_completed_sims": summary["num_completed_sims"],
            "total_simulation_time_s": summary["total_simulation_time_s"],
            "rad_800_proxy_events_per_particle_s": summary[
                "rad_800_proxy_events_per_particle_s"
            ],
            "rad_700_proxy_events_per_particle_s": summary[
                "rad_700_proxy_events_per_particle_s"
            ],
            "per_interaction": summary["per_interaction"],
        }
        reports = [
            old
            for old in reports
            if not (
                isinstance(old, dict)
                and old.get("analysis_window", {}).get("selection")
                == "last_steps_per_seed"
                and old.get("analysis_window", {}).get("window_steps_per_seed")
                == int(window_steps)
            )
        ]
        reports.append(report)
        canonical_summary["late_window_reports"] = reports
        canonical_summary_path.write_text(
            json.dumps(json_safe(canonical_summary), indent=2) + "\n"
        )
        plot_npt_style_spectrum(
            params=params,
            excitation_power=float(summary["excitation_power_w_cm2"]),
            tm_fraction=tm_fraction,
            config=config,
            summary=summary,
            interactions=interactions,
            output_dir=power_dir,
        )
        plot_mechanism_diagrams.plot_run(
            power_dir,
            output_name="mechanism_diagram.png",
            rate_key="events_per_ion_s",
            min_rate=1e-3,
            max_by_kind={"ET": 16, "Rad": 18, "NR": 12, "Pump": 8, "SQ": 16},
            dpi=220,
            summary_name=summary_name,
        )
        summaries.append(summary)

    plot_avalanche_curve(summaries, output_root, resume=False)
    print(
        f"Replotted {len(summaries)} powers from their final "
        f"{window_steps:,} steps per seed.",
        flush=True,
    )


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
        "--npt-cr-mode",
        choices=("all", "exported"),
        default=None,
        help=(
            "'exported' keeps only ET rows exported by NPT. 'all' also restores "
            "the missing low-level DRE-mapped rows using NPT's semi-empirical "
            "pair constants."
        ),
    )
    parser.add_argument(
        "--sigma-esa-scale",
        type=float,
        default=None,
        help="Fixed multiplicative factor for the 2->5 ESA pump cross section.",
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
    parser.add_argument(
        "--continue-run",
        action="store_true",
        help=(
            "Start or continue a checkpointed sweep to the same absolute cutoff. "
            "Requires its own output root; SIGUSR1 requests a clean pause (exit 75). "
            "Keeps databases in that root and retries unfinished analysis."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Extend existing trajectories by --simulation-length (or "
            "--simulation-time), then replay and summarize the full trajectories."
        ),
    )
    parser.add_argument(
        "--resume-power-indices",
        default=None,
        help=(
            "Comma-separated list of power indices to process when using --resume. "
            "If omitted, all power points are processed."
        ),
    )
    parser.add_argument(
        "--skip-npmc",
        action="store_true",
        help=(
            "Skip the NPMC executable and only replay/summarize existing trajectories. "
            "Only meaningful with --resume; useful for finishing post-processing after "
            "a resume run fails during summary/plot generation."
        ),
    )
    parser.add_argument("--powers", default=None, help="Comma/space separated powers in W cm^-2.")
    parser.add_argument(
        "--power-sampling-mode",
        choices=("fixed-production", "homogeneous", "centered-gaussian"),
        default=DEFAULT_POWER_SAMPLING_MODE,
        help=(
            "Power sampling strategy when --powers is not provided. "
            "'fixed-production' uses the self-contained 12-power preparation "
            "schedule; 'homogeneous' keeps the geometric sweep; 'centered-gaussian' "
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
        "--restart-workers",
        type=int,
        default=None,
        help=(
            "Concurrent single-seed target runs after preparation. Defaults "
            "to min(--num-sims, --thread-count)."
        ),
    )
    parser.add_argument(
        "--disable-preparation",
        action="store_true",
        help=(
            "Disable the fixed 50k checkpoint preparation schedule and cold-start "
            "all requested powers."
        ),
    )
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
        default=24,
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
        help=(
            "Per-seed target-power cutoff used when cutoff mode is 'steps' "
            "(default: 70M). On resume, this is the extension length."
        ),
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
    parser.add_argument(
        "--replot-last-steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Post-process existing single-seed restart trajectories using only "
            "their final N steps, replace the spectrum/mechanism/avalanche PNGs, "
            "and append late-window reports to the run summaries."
        ),
    )
    return parser


def run_production(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.replot_last_steps is not None:
        if args.replot_last_steps <= 0:
            parser.error("--replot-last-steps must be positive")
        params = rates.load_sk_parameters(args.params)
        config = resolve_production_config(args, params)
        output_root = Path(args.output_root) if args.output_root else next_run_dir()
        if not output_root.is_dir():
            parser.error(f"--replot-last-steps requires existing output root: {output_root}")
        replot_restart_step_window(
            output_root=output_root,
            window_steps=int(args.replot_last_steps),
            params=params,
            config=config,
            tm_fraction=args.tm_fraction,
            workers=(
                int(args.restart_workers)
                if args.restart_workers is not None
                else min(int(args.num_sims), int(args.thread_count))
            ),
        )
        return
    args.npmc_command = resolve_npmc_command(args.npmc_command)
    (
        args.resolved_cutoff_mode,
        args.resolved_simulation_length,
        args.resolved_simulation_time,
    ) = resolve_simulation_cutoff(args)
    if args.num_sims <= 0 or args.thread_count <= 0:
        parser.error("--num-sims and --thread-count must be positive")
    if args.restart_workers is None:
        args.restart_workers = min(int(args.num_sims), int(args.thread_count))
    if args.restart_workers <= 0:
        parser.error("--restart-workers must be positive")
    args.restart_workers = min(int(args.restart_workers), int(args.num_sims))
    if not args.disable_preparation and args.resolved_cutoff_mode != "steps":
        parser.error(
            "The fixed checkpoint preparation workflow requires step cutoffs; "
            "use --disable-preparation for physical-time cold starts."
        )
    params = rates.load_sk_parameters(args.params)
    config = resolve_production_config(args, params)
    output_root = Path(args.output_root) if args.output_root else next_run_dir()
    if not args.continue_run and (output_root / ".chain_state.json").exists():
        parser.error("This is a checkpoint-chain output root; use --continue-run.")
    if args.resume and not output_root.is_dir():
        parser.error(f"--resume requires existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_started_at_utc = utc_now_iso()
    existing_sweep_summary = (
        _load_existing_sweep_summary(output_root) if args.resume else None
    )
    trajectory_archive_root = (
        None if args.continue_run else Path(args.trajectory_archive_root).expanduser()
    )
    source_np_db_path = resolve_source_np_db(args, params, output_root, config)
    print(f"Using source geometry database: {source_np_db_path}", flush=True)

    powers = parse_power_sweep(args)
    try:
        preparation_steps_by_power = {
            float(power): preparation_steps_for_power(
                float(power),
                enabled=not bool(args.disable_preparation),
            )
            for power in powers
        }
    except ValueError as exc:
        parser.error(str(exc))
    build_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    power_parallel_total_slots = (
        int(args.power_parallel_total_slots)
        if args.power_parallel_total_slots is not None
        else default_power_parallel_total_slots()
    )
    if args.power_parallel_workers is not None:
        power_parallel_workers = max(1, int(args.power_parallel_workers))
    elif power_parallel_total_slots is not None:
        derived_slots = (
            int(power_parallel_total_slots)
        )
        power_parallel_workers = max(1, derived_slots // int(args.thread_count))
    else:
        power_parallel_workers = 1
    power_parallel_workers = min(power_parallel_workers, len(powers))
    auto_power_parallel = power_parallel_workers > 1
    local_db_staging_root = (
        None if args.dry_run or args.continue_run else resolve_local_db_staging_root()
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
        "preparation_power_w_cm2": (
            None if args.disable_preparation else PREPARATION_POWER_W_CM2
        ),
        "preparation_steps_by_power": [
            {
                "power_w_cm2": float(power),
                "preparation_step_cutoff": int(
                    preparation_steps_by_power[float(power)]
                ),
            }
            for power in powers
        ],
        "preparation_disabled": bool(args.disable_preparation),
        "simulation_step_cutoffs_by_power": [
            {
                "power_w_cm2": float(power),
                "simulation_step_cutoff": simulation_step_cutoff_for_power(args),
            }
            for power in powers
        ],
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
        "trajectory_archive_root": (
            None if trajectory_archive_root is None
            else str(trajectory_archive_root.resolve())
        ),
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
    if not args.dry_run:
        with open(output_root / "npt_production_config.json", "w") as f:
            json.dump(root_config, f, indent=2)

    power_jobs = [
        (int(power_index), float(power))
        for power_index, power in sorted(
            enumerate(powers), key=lambda item: item[1], reverse=True
        )
    ]
    if args.resume and args.resume_power_indices is None:
        print(
            "Auto-selecting unconverged power indices for resume extension...",
            flush=True,
        )
        from convergence_diagnostic import find_unconverged_power_indices

        unconverged_indices = find_unconverged_power_indices(
            output_root,
            drift_tol=0.10,
            min_pass_frac=0.75,
        )
        if not unconverged_indices:
            print(
                "All existing power points are converged; nothing to extend.",
                flush=True,
            )
            return
        args.resume_power_indices = ",".join(str(i) for i in unconverged_indices)
        print(
            f"Extending power indices: {args.resume_power_indices}",
            flush=True,
        )

    if args.resume_power_indices is not None:
        if not args.resume:
            parser.error("--resume-power-indices requires --resume")
        allowed = {int(x.strip()) for x in args.resume_power_indices.split(",")}
        power_jobs = [(idx, pwr) for idx, pwr in power_jobs if idx in allowed]
        if not power_jobs:
            parser.error("--resume-power-indices did not match any power index")

    if args.skip_npmc and not args.resume:
        parser.error("--skip-npmc requires --resume")

    if auto_power_parallel:
        print(
            f"Running {len(power_jobs)} power points with {power_parallel_workers} concurrent workers "
            f"and {args.thread_count} NPMC threads per power.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=power_parallel_workers) as executor:
            future_to_job = {
                executor.submit(
                    run_configured_power_point,
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
            for future in checkpoint_futures(future_to_job):
                build_record, summary = future.result()
                build_records.append(build_record)
                if summary is not None:
                    summaries.append(summary)
    else:
        for power_index, power in power_jobs:
            build_record, summary = run_configured_power_point(
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
    sweep_build_records = (
        merge_build_records(existing_sweep_summary, build_records)
        if args.resume
        else build_records
    )
    root_config["npt_raw_vs_npmc_readin_by_power"] = build_power_rate_tables(
        sweep_build_records, output_root
    )
    if not args.dry_run:
        with open(output_root / "npt_production_config.json", "w") as f:
            json.dump(json_safe(root_config), f, indent=2)

    # For resume runs, merge with existing power summaries so the sweep summary
    # and avalanche curve reflect the full sweep, not just the resumed subset.
    if args.resume:
        existing_summaries = _load_existing_power_summaries(output_root)
        merged_by_power: dict[float, dict[str, Any]] = {
            float(row["excitation_power_w_cm2"]): row
            for row in existing_summaries
        }
        for row in summaries:
            merged_by_power[float(row["excitation_power_w_cm2"])] = row
        sweep_summaries = list(merged_by_power.values())
    else:
        sweep_summaries = summaries

    max_local_log_slope_800 = None
    max_local_log_slope_800_power_w_cm2 = None
    if sweep_summaries:
        ordered_summaries = sorted(
            sweep_summaries,
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

    if args.resume and existing_sweep_summary is not None:
        # Archive this extension while keeping the canonical aggregate fields
        # synchronized with the full, current sweep.
        sweep_summary = copy.deepcopy(existing_sweep_summary)
        resume_runs = sweep_summary.get("resume_runs", [])
        if not isinstance(resume_runs, list):
            resume_runs = []
        resume_record = {
            "run_id": utc_now_iso(),
            "started_at_utc": run_started_at_utc,
            "finished_at_utc": utc_now_iso(),
            "command": list(sys.argv),
            "power_indices": [int(index) for index, _power in power_jobs],
            "requested_extension_steps": (
                None
                if args.resolved_simulation_length is None
                else int(args.resolved_simulation_length)
            ),
            "checkpoint_requested": 1,
            "build_records": build_records,
            "power_points": sorted(
                summaries,
                key=lambda row: float(row["excitation_power_w_cm2"]),
            ),
        }
        resume_runs.append(resume_record)
        sweep_summary["resume_runs"] = resume_runs
        sweep_summary["power_points"] = sorted(
            sweep_summaries,
            key=lambda row: float(row["excitation_power_w_cm2"]),
        )
        sweep_summary["build_records"] = sweep_build_records
        sweep_summary["npt_raw_vs_npmc_readin_by_power"] = root_config[
            "npt_raw_vs_npmc_readin_by_power"
        ]
        sweep_summary["max_local_log_slope_800"] = max_local_log_slope_800
        sweep_summary[
            "max_local_log_slope_800_power_w_cm2"
        ] = max_local_log_slope_800_power_w_cm2
        sweep_summary["latest_power_points"] = sorted(
            sweep_summaries,
            key=lambda row: float(row["excitation_power_w_cm2"]),
        )
        sweep_summary["latest_build_records"] = sweep_build_records
        sweep_summary["latest_npt_raw_vs_npmc_readin_by_power"] = root_config.get(
            "npt_raw_vs_npmc_readin_by_power", []
        )
        sweep_summary["latest_max_local_log_slope_800"] = max_local_log_slope_800
        sweep_summary[
            "latest_max_local_log_slope_800_power_w_cm2"
        ] = max_local_log_slope_800_power_w_cm2
        sweep_summary["latest_generated_at_utc"] = utc_now_iso()
    else:
        sweep_summary = {
            "max_local_log_slope_800": max_local_log_slope_800,
            "max_local_log_slope_800_power_w_cm2": max_local_log_slope_800_power_w_cm2,
            **root_config,
            "build_records": build_records,
            "power_points": sweep_summaries,
        }
    with open(output_root / "npt_power_sweep_summary.json", "w") as f:
        json.dump(json_safe(sweep_summary), f, indent=2)

    if not args.dry_run:
        plot_avalanche_curve(sweep_summaries, output_root, resume=args.resume)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.continue_run:
        run_production(args, parser)
        return 0
    if (
        not args.output_root or args.resume or args.skip_npmc or args.dry_run
        or args.regenerate_geometry or args.resume_power_indices is not None
        or args.replot_last_steps is not None
        or resolve_simulation_cutoff(args)[0] != "steps"
    ):
        parser.error(
            "--continue-run requires --output-root and step cutoffs; it cannot "
            "be combined with resume, dry-run, skip, replot, or geometry regeneration."
        )
    output_root = Path(args.output_root).expanduser().resolve()
    args.output_root = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Never unlink this lock: all jobs must lock the same inode.
    with open(output_root / ".chain.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output_root / ".chain_state.json"
        settings = vars(args).copy()
        for key in (
            "thread_count", "restart_workers", "power_parallel_workers",
            "power_parallel_total_slots",
        ):
            settings.pop(key)
        settings["params_contents"] = rates.load_sk_parameters(args.params)
        if args.source_np_db:
            settings["source_np_db_sha256"] = hashlib.sha256(
                Path(args.source_np_db).read_bytes()
            ).hexdigest()
        fingerprint = hashlib.sha256(
            json.dumps(json_safe(settings), sort_keys=True).encode()
        ).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["fingerprint"] != fingerprint:
                parser.error("Continuation settings changed; use the original settings.")
            if state["status"] == "complete":
                print("Sweep and post-analysis are already complete.", flush=True)
                return 0
        else:
            if any(output_root.glob("power_*")):
                parser.error("--continue-run requires a new, dedicated output root.")
            state = {"fingerprint": fingerprint, "status": "pending"}
            atomic_json(state_path, state)
        global _CONTINUE_RUN, _CHECKPOINT_REQUESTED
        _CONTINUE_RUN = True
        _CHECKPOINT_REQUESTED = False
        handlers = {
            sig: signal.signal(sig, request_checkpoint)
            for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)
        }
        try:
            run_production(args, parser)
            checkpoint_boundary()
            state["status"] = "complete"
            atomic_json(state_path, state)
        except CheckpointRequested:
            print("Checkpoint saved; simulation or post-analysis will continue next job.", flush=True)
            return 75
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            _CONTINUE_RUN = False
            _CHECKPOINT_REQUESTED = False
    return 0


if __name__ == "__main__":
    sys.exit(main())
