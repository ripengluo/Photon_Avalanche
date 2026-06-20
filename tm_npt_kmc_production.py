"""Production runner for the Tm avalanche kMC model built on NPT defaults."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import tm_npt_rates as rates
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
TM_W3_NR_TRANSITION = (2, 1)
TM_W5_NR_TRANSITION = (4, 3)
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
OUTER_RADIUS_A = CORE_RADIUS_A + AVERAGE_SHELL_THICKNESS_A

FALLBACK_PRODUCTION_DEFAULTS = {
    "npt_cr_mode": "all",
    "sigma_esa_scale": 1185.7978647623052,
    "q21_scale": 1.0,
    "s54_scale": 1.0,
    "s45_scale": 1.0,
    "s12_scale": 21.148836746821555,
    "w3_nr_scale": 1.0,
    "w5_nr_scale": 1.0,
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
        "npt_cr_mode": str(choose("npt_cr_mode", args.npt_cr_mode)),
        "sigma_esa_scale": float(choose("sigma_esa_scale", args.sigma_esa_scale)),
        "q21_scale": float(choose("q21_scale", args.q21_scale)),
        "s54_scale": float(choose("s54_scale", args.s54_scale)),
        "s45_scale": float(choose("s45_scale", args.s45_scale)),
        "s12_scale": float(choose("s12_scale", args.s12_scale)),
        "w3_nr_scale": float(choose("w3_nr_scale", args.fixed_w3_nr_scale)),
        "w5_nr_scale": float(choose("w5_nr_scale", args.fixed_w5_nr_scale)),
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


def build_custom_interactions(
    params: dict[str, Any],
    source_np_db_path: Path,
    excitation_power: float,
    include_zero_rates: bool,
    tm_fraction: float | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one full-NPT interaction network for a single excitation power."""
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
    state_to_level = rates.dre_state_to_level_map(params)
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

    table_s3_reference_cross_sections = params["absorption_cross_sections_cm^2"]
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

    for left_state in range(nr.shape[0]):
        for right_state in range(nr.shape[1]):
            base_rate = float(nr[left_state, right_state])
            if base_rate == 0.0:
                continue
            if (left_state, right_state) == TM_W3_NR_TRANSITION:
                rate_scale_factor = float(config["w3_nr_scale"])
            elif (left_state, right_state) == TM_W5_NR_TRANSITION:
                rate_scale_factor = float(config["w5_nr_scale"])
            else:
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
            "right_state": 4,
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
        "w3_nr_scale": float(config["w3_nr_scale"]),
        "w5_nr_scale": float(config["w5_nr_scale"]),
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
            "table_s3_reference": json_safe(table_s3_reference_cross_sections),
            "npt_effective": json_safe(kmc_default_absorption_cross_sections),
        },
        "mode_defaults": json_safe(config["mode_defaults"]),
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


def run_npmc(
    np_db_path: Path,
    initial_state_db_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Run NPMC on the custom databases."""
    run_args = [
        args.npmc_command,
        f"--nano_particle_database={np_db_path}",
        f"--initial_state_database={initial_state_db_path}",
        f"--number_of_simulations={args.num_sims}",
        f"--base_seed={args.base_seed}",
        f"--thread_count={args.thread_count}",
    ]
    if args.resolved_cutoff_mode == "steps":
        run_args.append(f"--step_cutoff={args.resolved_simulation_length}")
    elif args.resolved_cutoff_mode == "physical-time":
        run_args.append(f"--time_cutoff={args.resolved_simulation_time}")
    else:
        raise ValueError(
            f"Unsupported simulation cutoff mode: {args.resolved_cutoff_mode!r}"
        )
    run_args.append("--checkpoint=0")

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
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build and optionally run one power point."""
    output_dir = output_root / f"power_{power_index:02d}_{power:.6g}"
    print(
        f"[power {power_index + 1}/{power_count}] building {power:.6g} W cm^-2 "
        f"for the NPT rate model",
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
        output_dir=output_dir,
        interactions=interactions,
        interaction_radius_bound_nm=float(
            params["simulation_defaults"]["interaction_radius_bound_nm"]
        ),
        distance_factor_type=str(
            params["simulation_defaults"]["distance_factor_type"]
        ),
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
        "np_db_path": str(np_db_path.resolve()),
        "initial_state_db_path": str(initial_state_db_path.resolve()),
    }
    if args.dry_run:
        return build_record, None

    run_npmc(np_db_path, initial_state_db_path, output_dir, args)
    archived_initial_state_db_path = archive_initial_state_database(
        initial_state_db_path=initial_state_db_path,
        output_dir=output_dir,
        archive_root=trajectory_archive_root,
    )
    build_record["archived_initial_state_db_path"] = str(
        archived_initial_state_db_path
    )
    replay = replay_trajectories(initial_state_db_path, interactions)
    summary = summarize_run(
        replay=replay,
        interactions=interactions,
        n_sites=int(manifest["geometry"]["ion_count"]),
        excitation_power=float(power),
        manifest=manifest,
        simulation_cutoff_mode=args.resolved_cutoff_mode,
        simulation_step_cutoff=args.resolved_simulation_length,
        simulation_time_cutoff_s=args.resolved_simulation_time,
    )
    with open(output_dir / "npt_run_summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    return build_record, summary


def archive_initial_state_database(
    initial_state_db_path: Path,
    output_dir: Path,
    archive_root: Path,
) -> Path:
    """Archive a completed trajectory DB when the configured archive root exists."""
    if not initial_state_db_path.exists():
        raise FileNotFoundError(
            f"Missing completed trajectory database: {initial_state_db_path}"
        )
    if initial_state_db_path.is_symlink():
        return initial_state_db_path.resolve()
    if not archive_root.exists():
        return initial_state_db_path.resolve()

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
    initial_state_db_path.symlink_to(archived_path.resolve())
    return archived_path.resolve()


def replay_trajectories(
    initial_state_db_path: Path,
    interactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read event counts, final simulated times, and time-averaged n4 occupancy."""
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

    with sqlite3.connect(initial_state_db_path) as con:
        rows = con.execute(
            """
            SELECT seed, step, time, site_id_1, site_id_2, interaction_id
            FROM trajectories
            ORDER BY seed, step
            """
        )
        current_seed: int | None = None
        previous_time = 0.0
        site_states: np.ndarray | None = None
        current_n4_count = 0
        previous_event_by_seed: dict[int, tuple[int, tuple[int, int] | None]] = {}

        def finalize_seed(seed: int | None, final_time: float) -> None:
            if seed is None:
                return
            simulation_time[seed] = float(final_time)
            if final_time > 0 and tm_site_count > 0:
                n4_population_per_seed[seed] = (
                    n4_time_integral[seed] / (float(tm_site_count) * float(final_time))
                )
            else:
                n4_population_per_seed[seed] = 0.0

        for seed, step, event_time, site_id_1, site_id_2, interaction_id in rows:
            seed = int(seed)
            step = int(step)
            event_time = float(event_time)
            interaction_id = int(interaction_id)
            site_id_1 = int(site_id_1)
            site_id_2 = int(site_id_2)

            if current_seed != seed:
                finalize_seed(current_seed, previous_time)
                current_seed = seed
                previous_time = 0.0
                site_states = np.zeros(site_count, dtype=np.int8)
                current_n4_count = 0

            if site_states is None:
                raise RuntimeError("site state replay was not initialized")

            dt = event_time - previous_time
            if dt < -1e-12:
                raise ValueError(
                    f"Trajectory time decreased for seed {seed} step {step}: "
                    f"{event_time} < {previous_time}"
                )
            n4_time_integral[seed] += current_n4_count * max(dt, 0.0)
            simulation_time[seed] = float(event_time)
            event_counts[seed][interaction_id] += 1

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
            previous_time = event_time

        finalize_seed(current_seed, previous_time)

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
        "w3_nr_scale": float(manifest["w3_nr_scale"]),
        "w5_nr_scale": float(manifest["w5_nr_scale"]),
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


def plot_avalanche_curve(summaries: list[dict[str, Any]], output_root: Path) -> None:
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
        fontsize=9.2,
        borderpad=0.65,
        labelspacing=0.45,
        handlelength=1.8,
        handletextpad=0.6,
        markerscale=1.05,
    )
    fig.subplots_adjust(bottom=0.18, left=0.14, right=0.97, top=0.96)
    fig.savefig(output_root / "npt_avalanche_curve.png")
    plt.close(fig)


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
    parser.add_argument(
        "--fixed-W3_NR-scale",
        "--fixed-w3-nr-scale",
        dest="fixed_w3_nr_scale",
        type=float,
        default=None,
        help=(
            "Global multiplicative factor for the selected W3NR one-site "
            "nonradiative rate (NR 3->2, 3H5->3F4)."
        ),
    )
    parser.add_argument(
        "--fixed-W5_NR-scale",
        "--fixed-w5-nr-scale",
        dest="fixed_w5_nr_scale",
        type=float,
        default=None,
        help=(
            "Global multiplicative factor for W5NR "
            "(NR 5->4, 3F3->3H4)."
        ),
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
    return parser


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
    source_np_db_path = resolve_source_np_db(args, params, output_root, config)
    print(f"Using source geometry database: {source_np_db_path}", flush=True)

    powers = parse_power_sweep(args)
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
        "mode_defaults": json_safe(config["mode_defaults"]),
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
    with open(output_root / "npt_production_config.json", "w") as f:
        json.dump(root_config, f, indent=2)

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
            )
            build_records.append(build_record)
            if summary is not None:
                summaries.append(summary)

    build_records.sort(key=lambda row: int(row["power_index"]))

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
    with open(output_root / "npt_power_sweep_summary.json", "w") as f:
        json.dump(json_safe(sweep_summary), f, indent=2)

    if not args.dry_run:
        plot_avalanche_curve(summaries, output_root)


if __name__ == "__main__":
    main()
