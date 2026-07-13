"""Percolation-style order parameter and susceptibility versus power.

This script treats the active Tm network as a finite percolation problem.
For each power point it replays all seeds in the corresponding kMC run,
builds connected clusters of active sites, and reports:

* order parameter: largest active-cluster fraction, P_infty = S_max / N_active
* susceptibility: mean finite-cluster size, chi = sum(s^2) / sum(s)
* optional reference metrics: n4 and 800 nm event rate

The default active mode is the legacy token ``n4+n5``, now interpreted as the
positive-gain set ``n2+n4+n5`` (3F4, 3H4, and 3F3). The script also supports
an ``n4``-only mode for checking whether connectivity is being overestimated by
including the loop-participating bridge manifolds.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.spatial import cKDTree

from tm_npt_kmc_trajectory_3d import (
    iter_trajectory_rows,
    load_interactions,
    load_manifest,
    load_site_count,
)


DEFAULT_ACTIVE_MODE = "n4+n5"
DEFAULT_SNAPSHOT_COUNT = 100
DEFAULT_WINDOW_FRACTION = 0.5
DEFAULT_OUTPUT_PNG = "percolation_order_parameter_susceptibility.png"
DEFAULT_OUTPUT_JSON = "percolation_order_parameter_susceptibility.json"
DEFAULT_FRAGMENT_OUTPUT_PNG = "percolation_fragment_count.png"
DEFAULT_OP_DERIVATIVE_OUTPUT_PNG = "percolation_order_parameter_derivative.png"
DEFAULT_ACTIVE_CLUSTER_DERIVATIVE_OUTPUT_PNG = "percolation_active_cluster_derivatives.png"
ACTIVE_MODE_CHOICES = ("n4+n5", "n4")


@dataclass
class PowerPercolationResult:
    power_w_cm2: float
    seed_count: int
    effective_snapshot_count: int
    candidate_snapshot_count: int
    order_parameter_mean: float
    order_parameter_std: float
    susceptibility_mean: float
    susceptibility_std: float
    susceptibility_fluctuation: float
    n4_time_averaged_population_mean: float
    n4_time_averaged_population_std: float
    rad_800_events_per_ion_s_mean: float
    rad_800_events_per_ion_s_std: float
    active_site_count_mean: float
    active_site_count_std: float
    largest_cluster_size_mean: float
    largest_cluster_size_std: float
    largest_cluster_fraction_active_mean: float
    largest_cluster_fraction_active_std: float
    fragment_count_mean: float
    fragment_count_std: float
    selected_seed: int | None = None


def normalize_active_mode(active_mode: str) -> str:
    mode = str(active_mode).strip().lower()
    if mode not in ACTIVE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported active mode {active_mode!r}. Choose from {ACTIVE_MODE_CHOICES}."
        )
    return mode


def active_mode_states(active_mode: str) -> tuple[int, ...]:
    mode = normalize_active_mode(active_mode)
    if mode == "n4+n5":
        # Keep the existing CLI token for compatibility, but include n2 because
        # 3F4 participates directly in the local positive-gain loop.
        return (1, 3, 4)
    if mode == "n4":
        return (3,)
    raise ValueError(f"Unsupported active mode: {active_mode}")


def active_mode_slug(active_mode: str) -> str:
    return normalize_active_mode(active_mode).replace("+", "_plus_")


def active_mode_label(active_mode: str) -> str:
    mode = normalize_active_mode(active_mode)
    if mode == "n4+n5":
        return "n2+n4+n5 (3F4 + 3H4 + 3F3)"
    if mode == "n4":
        return "n4 only (3H4)"
    raise ValueError(f"Unsupported active mode: {active_mode}")


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


def load_summary_results(summary_path: Path) -> tuple[dict[str, Any], list[PowerPercolationResult]]:
    with open(summary_path, "r") as fh:
        summary = json.load(fh)

    results: list[PowerPercolationResult] = []
    default_snapshot_count = int(summary.get("snapshot_count", 0))
    for row in summary.get("power_results", []):
        results.append(
            PowerPercolationResult(
                power_w_cm2=float(row["power_w_cm2"]),
                seed_count=int(row["seed_count"]),
                effective_snapshot_count=int(
                    row.get("effective_snapshot_count", default_snapshot_count)
                ),
                candidate_snapshot_count=int(
                    row.get("candidate_snapshot_count", default_snapshot_count)
                ),
                order_parameter_mean=float(row["order_parameter_mean"]),
                order_parameter_std=float(row["order_parameter_std"]),
                susceptibility_mean=float(row["susceptibility_mean"]),
                susceptibility_std=float(row["susceptibility_std"]),
                susceptibility_fluctuation=float(row["susceptibility_fluctuation"]),
                n4_time_averaged_population_mean=float(row["n4_time_averaged_population_mean"]),
                n4_time_averaged_population_std=float(row["n4_time_averaged_population_std"]),
                rad_800_events_per_ion_s_mean=float(row["rad_800_events_per_ion_s_mean"]),
                rad_800_events_per_ion_s_std=float(row["rad_800_events_per_ion_s_std"]),
                active_site_count_mean=float(row.get("active_site_count_mean", float("nan"))),
                active_site_count_std=float(row.get("active_site_count_std", float("nan"))),
                largest_cluster_size_mean=float(row.get("largest_cluster_size_mean", float("nan"))),
                largest_cluster_size_std=float(row.get("largest_cluster_size_std", float("nan"))),
                largest_cluster_fraction_active_mean=float(row["largest_cluster_fraction_active_mean"]),
                largest_cluster_fraction_active_std=float(row["largest_cluster_fraction_active_std"]),
                fragment_count_mean=float(row.get("fragment_count_mean", float("nan"))),
                fragment_count_std=float(row.get("fragment_count_std", float("nan"))),
                selected_seed=(
                    int(row["selected_seed"])
                    if row.get("selected_seed") is not None
                    else None
                ),
            )
        )
    return summary, results


def build_active_mask(site_states: np.ndarray, active_mode: str) -> np.ndarray:
    mode = normalize_active_mode(active_mode)
    site_states = np.asarray(site_states, dtype=np.int8)
    return np.isin(site_states, active_mode_states(mode))


def load_site_positions(np_db_path: Path) -> np.ndarray:
    with sqlite3.connect(np_db_path) as con:
        rows = con.execute("SELECT x, y, z FROM sites ORDER BY site_id").fetchall()
    if not rows:
        raise ValueError(f"No site coordinates found in {np_db_path}")
    return np.asarray(rows, dtype=float)


def list_seeds(initial_state_db_path: Path) -> list[int]:
    with sqlite3.connect(initial_state_db_path) as con:
        rows = con.execute("SELECT DISTINCT seed FROM trajectories ORDER BY seed").fetchall()
    return [int(row[0]) for row in rows]


def parse_power_from_name(name: str) -> float:
    match = re.search(r"power_\d+_([0-9.]+)$", name)
    if not match:
        raise ValueError(f"Cannot parse power from directory name: {name}")
    return float(match.group(1))


def resolve_power_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    if (root / "initial_state.sqlite").exists() and (root / "np.sqlite").exists():
        return [root]

    power_dirs = sorted(
        [
            path
            for path in root.rglob("power_*")
            if path.is_dir()
            and (path / "initial_state.sqlite").exists()
            and (path / "np.sqlite").exists()
        ],
        key=lambda path: (parse_power_from_name(path.name), path.as_posix()),
    )
    if not power_dirs:
        raise ValueError(
            f"No power directories found under {root}. Pass a sweep root like "
            "'run10/s12_scale_20' or a single power directory."
        )
    return power_dirs


def resolve_geometry_db_path(run_root: Path, power_dirs: list[Path]) -> Path:
    """Resolve a shared geometry database for a run root if one exists."""
    candidate = run_root / "generated_geometry" / "source_geometry_np.sqlite"
    if candidate.exists():
        return candidate
    if (run_root / "np.sqlite").exists():
        return run_root / "np.sqlite"
    if not power_dirs:
        raise ValueError(f"No power directories available under {run_root}")
    first_np = power_dirs[0] / "np.sqlite"
    if first_np.exists():
        return first_np
    raise FileNotFoundError(
        f"Could not resolve a geometry database under {run_root} or {power_dirs[0]}"
    )


def build_neighbor_pairs(positions: np.ndarray, cutoff_nm: float) -> np.ndarray:
    if cutoff_nm <= 0:
        return np.zeros((0, 2), dtype=int)
    pairs = cKDTree(positions).query_pairs(float(cutoff_nm))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(list(pairs), dtype=int)


def cluster_metrics_fast(
    site_states: np.ndarray,
    neighbor_pairs: np.ndarray,
    active_mode: str,
) -> dict[str, float]:
    """Vectorized active-edge filtering with the same union-find semantics."""
    active_mask = build_active_mask(site_states=site_states, active_mode=active_mode)
    active_indices = np.flatnonzero(active_mask)
    active_count = int(active_indices.size)
    active_fraction = float(active_count) / float(site_states.size)

    if active_count == 0:
        return {
            "order_parameter": 0.0,
            "susceptibility": 0.0,
            "active_fraction": active_fraction,
            "largest_cluster_fraction_active": 0.0,
            "fragment_count": 0.0,
        }

    parent = np.arange(site_states.size, dtype=np.int32)
    rank = np.zeros(site_states.size, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    if neighbor_pairs.size:
        edge_mask = active_mask[neighbor_pairs[:, 0]] & active_mask[neighbor_pairs[:, 1]]
        if np.any(edge_mask):
            for ii, jj in neighbor_pairs[edge_mask]:
                union(int(ii), int(jj))

    roots = np.asarray([find(int(idx)) for idx in active_indices], dtype=np.int32)
    component_sizes = np.bincount(roots, minlength=site_states.size)
    component_sizes = component_sizes[component_sizes > 0].astype(float, copy=False)
    fragment_count = float(component_sizes.size)
    largest_size = float(component_sizes.max()) if len(component_sizes) else 0.0
    largest_cluster_fraction_active = largest_size / float(active_count) if active_count else 0.0
    order_parameter = largest_cluster_fraction_active

    finite_sizes = component_sizes[component_sizes < largest_size]
    if finite_sizes.size == 0:
        susceptibility = 0.0
    else:
        susceptibility = float(np.sum(finite_sizes**2) / np.sum(finite_sizes))

    return {
        "order_parameter": float(order_parameter),
        "susceptibility": float(susceptibility),
        "active_fraction": float(active_fraction),
        "largest_cluster_fraction_active": float(largest_cluster_fraction_active),
        "fragment_count": float(fragment_count),
    }


def replay_seed_tail_metrics(
    initial_state_db_path: Path,
    np_db_path: Path,
    site_count: int,
    interactions: dict[int, dict[str, Any]],
    rad_800_id: int,
    seed: int,
    neighbor_pairs: np.ndarray,
    active_mode: str,
    window_fraction: float,
    snapshot_count: int,
) -> dict[str, Any]:
    rows = list(iter_trajectory_rows(initial_state_db_path, seed=seed))
    if not rows:
        raise ValueError(f"Seed {seed} not found in {initial_state_db_path}")

    total_time = float(rows[-1][2])
    start_time = max(0.0, total_time * (1.0 - float(window_fraction)))
    snapshot_count = max(2, int(snapshot_count))
    target_times = np.linspace(start_time, total_time, snapshot_count, dtype=float)
    target_idx = 0

    site_states = np.zeros(site_count, dtype=np.int8)
    counts = np.zeros(5, dtype=np.int64)
    counts[0] = site_count
    previous_time = 0.0
    n4_time_integral = 0.0
    rad_800_count = 0
    snapshots: list[dict[str, float]] = []

    def record_snapshot(snapshot_time: float) -> None:
        metrics = cluster_metrics_fast(
            site_states=site_states,
            neighbor_pairs=neighbor_pairs,
            active_mode=active_mode,
        )
        active_site_count = float(metrics["active_fraction"] * float(site_count))
        largest_cluster_size = float(
            metrics["largest_cluster_fraction_active"] * active_site_count
        )
        snapshots.append(
            {
                "time": float(snapshot_time),
                "n4": float(counts[3] / site_count),
                "rad_800_count": float(rad_800_count),
                "active_site_count": active_site_count,
                "largest_cluster_size": largest_cluster_size,
                **metrics,
            }
        )

    while target_idx < len(target_times) and target_times[target_idx] <= 0.0:
        record_snapshot(float(target_times[target_idx]))
        target_idx += 1

    for row_seed, step, event_time, site_id_1, site_id_2, interaction_id in rows:
        row_seed = int(row_seed)
        step = int(step)
        event_time = float(event_time)
        site_id_1 = int(site_id_1)
        site_id_2 = int(site_id_2)
        interaction_id = int(interaction_id)
        interaction = interactions[interaction_id]

        dt = event_time - previous_time
        if dt < -1e-12:
            raise ValueError(
                f"Trajectory time decreased for seed {row_seed} step {step}: {event_time} < {previous_time}"
            )
        n4_time_integral += float(counts[3]) * max(dt, 0.0)

        if interaction_id == rad_800_id:
            rad_800_count += 1

        while target_idx < len(target_times) and target_times[target_idx] <= event_time:
            record_snapshot(float(target_times[target_idx]))
            target_idx += 1

        current_state_1 = int(site_states[site_id_1])
        if current_state_1 != int(interaction["left_state_1"]):
            raise ValueError(
                f"Replay mismatch at seed {row_seed} step {step} site {site_id_1}: "
                f"expected {interaction['left_state_1']}, found {current_state_1}"
            )
        counts[current_state_1] -= 1
        counts[int(interaction["right_state_1"])] += 1
        site_states[site_id_1] = int(interaction["right_state_1"])

        if int(interaction["number_of_sites"]) == 2:
            current_state_2 = int(site_states[site_id_2])
            if current_state_2 != int(interaction["left_state_2"]):
                raise ValueError(
                    f"Replay mismatch at seed {row_seed} step {step} site {site_id_2}: "
                    f"expected {interaction['left_state_2']}, found {current_state_2}"
                )
            counts[current_state_2] -= 1
            counts[int(interaction["right_state_2"])] += 1
            site_states[site_id_2] = int(interaction["right_state_2"])

        previous_time = event_time

    while target_idx < len(target_times):
        record_snapshot(float(target_times[target_idx]))
        target_idx += 1

    snapshot_order = np.asarray([frame["order_parameter"] for frame in snapshots], dtype=float)
    snapshot_sus = np.asarray([frame["susceptibility"] for frame in snapshots], dtype=float)
    snapshot_n4 = np.asarray([frame["n4"] for frame in snapshots], dtype=float)
    snapshot_active_count = np.asarray(
        [frame["active_site_count"] for frame in snapshots],
        dtype=float,
    )
    snapshot_largest_cluster_size = np.asarray(
        [frame["largest_cluster_size"] for frame in snapshots],
        dtype=float,
    )

    return {
        "seed": int(seed),
        "site_count": int(site_count),
        "total_time": float(total_time),
        "n4_time_averaged_population": float(
            n4_time_integral / (float(site_count) * total_time) if total_time > 0 else 0.0
        ),
        "rad_800_count": int(rad_800_count),
        "rad_800_events_per_ion_s": float(
            rad_800_count / (float(site_count) * total_time) if total_time > 0 else 0.0
        ),
        "op_mean": float(np.mean(snapshot_order)),
        "op_std": float(np.std(snapshot_order, ddof=1)) if len(snapshot_order) > 1 else 0.0,
        "chi_mean": float(np.mean(snapshot_sus)),
        "chi_std": float(np.std(snapshot_sus, ddof=1)) if len(snapshot_sus) > 1 else 0.0,
        "n4_mean": float(np.mean(snapshot_n4)),
        "n4_std": float(np.std(snapshot_n4, ddof=1)) if len(snapshot_n4) > 1 else 0.0,
        "active_site_count_mean": float(np.mean(snapshot_active_count)),
        "active_site_count_std": (
            float(np.std(snapshot_active_count, ddof=1))
            if len(snapshot_active_count) > 1
            else 0.0
        ),
        "largest_cluster_size_mean": float(np.mean(snapshot_largest_cluster_size)),
        "largest_cluster_size_std": (
            float(np.std(snapshot_largest_cluster_size, ddof=1))
            if len(snapshot_largest_cluster_size) > 1
            else 0.0
        ),
        "snapshots": snapshots,
        "active_mode": normalize_active_mode(active_mode),
        "active_states": list(active_mode_states(active_mode)),
        "window_fraction": float(window_fraction),
        "effective_snapshot_count": int(len(snapshots)),
        "candidate_snapshot_count": int(len(snapshots)),
    }


def analyze_power_dir(
    power_dir: Path,
    active_mode: str,
    snapshot_count: int,
    window_fraction: float,
    cluster_cutoff_nm: float | None,
    site_count: int | None = None,
    neighbor_pairs: np.ndarray | None = None,
    geometry_db_path: Path | None = None,
) -> PowerPercolationResult:
    initial_state_db_path = power_dir / "initial_state.sqlite"
    manifest = load_manifest(power_dir)
    interactions, rad_800_id = load_interactions(manifest)

    if geometry_db_path is None:
        geometry_db_path = power_dir / "np.sqlite"

    if cluster_cutoff_nm is None:
        cluster_cutoff_nm = float(manifest["geometry"]["interaction_radius_bound_nm"])

    if site_count is None:
        site_count = load_site_count(geometry_db_path)
    if neighbor_pairs is None:
        positions = load_site_positions(geometry_db_path)
        if len(positions) != site_count:
            raise ValueError(
                f"Coordinate count {len(positions)} does not match metadata.number_of_sites {site_count}"
            )
        neighbor_pairs = build_neighbor_pairs(positions, float(cluster_cutoff_nm))
    seeds = list_seeds(initial_state_db_path)
    if not seeds:
        raise ValueError(f"No seeds found in {initial_state_db_path}")

    seed_metrics = [
        replay_seed_tail_metrics(
            initial_state_db_path=initial_state_db_path,
            np_db_path=geometry_db_path,
            site_count=int(site_count),
            interactions=interactions,
            rad_800_id=rad_800_id,
            seed=seed,
            neighbor_pairs=neighbor_pairs,
            active_mode=active_mode,
            window_fraction=float(window_fraction),
            snapshot_count=int(snapshot_count),
        )
        for seed in seeds
    ]

    op_vals = np.asarray([item["op_mean"] for item in seed_metrics], dtype=float)
    chi_vals = np.asarray([item["chi_mean"] for item in seed_metrics], dtype=float)
    n4_vals = np.asarray([item["n4_time_averaged_population"] for item in seed_metrics], dtype=float)
    rad_vals = np.asarray([item["rad_800_events_per_ion_s"] for item in seed_metrics], dtype=float)
    active_count_vals = np.asarray(
        [item["active_site_count_mean"] for item in seed_metrics],
        dtype=float,
    )
    largest_cluster_size_vals = np.asarray(
        [item["largest_cluster_size_mean"] for item in seed_metrics],
        dtype=float,
    )
    largest_active_vals = np.asarray(
        [np.mean([frame["largest_cluster_fraction_active"] for frame in item["snapshots"]]) for item in seed_metrics],
        dtype=float,
    )
    fragment_count_vals = np.asarray(
        [np.mean([frame["fragment_count"] for frame in item["snapshots"]]) for item in seed_metrics],
        dtype=float,
    )

    site_count = int(seed_metrics[0]["site_count"])
    susceptibility_fluctuation = (
        float(site_count * np.var(op_vals, ddof=1)) if len(op_vals) > 1 else 0.0
    )

    return PowerPercolationResult(
        power_w_cm2=parse_power_from_name(power_dir.name),
        seed_count=int(len(seeds)),
        effective_snapshot_count=int(sum(int(item["effective_snapshot_count"]) for item in seed_metrics)),
        candidate_snapshot_count=int(sum(int(item["candidate_snapshot_count"]) for item in seed_metrics)),
        order_parameter_mean=float(np.mean(op_vals)),
        order_parameter_std=float(np.std(op_vals, ddof=1)) if len(op_vals) > 1 else 0.0,
        susceptibility_mean=float(np.mean(chi_vals)),
        susceptibility_std=float(np.std(chi_vals, ddof=1)) if len(chi_vals) > 1 else 0.0,
        susceptibility_fluctuation=susceptibility_fluctuation,
        n4_time_averaged_population_mean=float(np.mean(n4_vals)),
        n4_time_averaged_population_std=float(np.std(n4_vals, ddof=1)) if len(n4_vals) > 1 else 0.0,
        rad_800_events_per_ion_s_mean=float(np.mean(rad_vals)),
        rad_800_events_per_ion_s_std=float(np.std(rad_vals, ddof=1)) if len(rad_vals) > 1 else 0.0,
        active_site_count_mean=float(np.mean(active_count_vals)),
        active_site_count_std=float(np.std(active_count_vals, ddof=1))
        if len(active_count_vals) > 1
        else 0.0,
        largest_cluster_size_mean=float(np.mean(largest_cluster_size_vals)),
        largest_cluster_size_std=float(np.std(largest_cluster_size_vals, ddof=1))
        if len(largest_cluster_size_vals) > 1
        else 0.0,
        largest_cluster_fraction_active_mean=float(np.mean(largest_active_vals)),
        largest_cluster_fraction_active_std=float(np.std(largest_active_vals, ddof=1))
        if len(largest_active_vals) > 1
        else 0.0,
        fragment_count_mean=float(np.mean(fragment_count_vals)),
        fragment_count_std=float(np.std(fragment_count_vals, ddof=1))
        if len(fragment_count_vals) > 1
        else 0.0,
        selected_seed=int(
            seeds[
                int(np.argmax(np.asarray([item["op_mean"] for item in seed_metrics], dtype=float)))
            ]
        ),
    )


def plot_sweep(
    results: list[PowerPercolationResult],
    output_path: Path,
    active_mode: str,
    snapshot_count: int,
    window_fraction: float,
) -> None:
    results = sorted(results, key=lambda item: item.power_w_cm2)
    powers = np.asarray([item.power_w_cm2 for item in results], dtype=float)
    op = np.asarray([item.order_parameter_mean for item in results], dtype=float)
    op_err = np.asarray([item.order_parameter_std for item in results], dtype=float)
    chi = np.asarray([item.susceptibility_mean for item in results], dtype=float)
    chi_err = np.asarray([item.susceptibility_std for item in results], dtype=float)
    n4 = np.asarray([item.n4_time_averaged_population_mean for item in results], dtype=float)
    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.05], wspace=0.25)
    ax_op = fig.add_subplot(gs[0, 0])
    ax_chi = fig.add_subplot(gs[0, 1])
    ax_op2 = ax_op.twinx()

    ax_op.errorbar(
        powers,
        op,
        yerr=op_err,
        fmt="o-",
        color="tab:blue",
        linewidth=1.8,
        markersize=4.5,
        capsize=2.5,
        label=r"$P_\infty = S_{\max}/N_{\mathrm{active}}$",
    )
    ax_op2.plot(
        powers,
        n4,
        "^:",
        color="tab:green",
        linewidth=1.2,
        markersize=4.0,
        label=r"$\langle n_4 \rangle$",
    )
    ax_op.set_xscale("log")
    ax_op.set_xlabel("Excitation power (W cm$^{-2}$)", fontsize=14)
    ax_op.set_ylabel(r"Order parameter $P_\infty$", fontsize=14)
    ax_op.tick_params(axis="both", which="major", labelsize=12)
    ax_op.tick_params(axis="both", which="minor", labelsize=11)
    ax_op2.set_ylabel(r"$\langle n_4 \rangle$", color="tab:green", fontsize=14)
    ax_op2.tick_params(axis="y", colors="tab:green", labelsize=12)
    ax_op.grid(True, alpha=0.25)

    ax_chi.errorbar(
        powers,
        chi,
        yerr=chi_err,
        fmt="o-",
        color="tab:red",
        linewidth=1.8,
        markersize=4.5,
        capsize=2.5,
        label=r"$\chi$",
    )
    ax_chi.set_xscale("log")
    ax_chi.set_xlabel("Excitation power (W cm$^{-2}$)", fontsize=14)
    ax_chi.set_ylabel(r"Susceptibility $\chi$", fontsize=14)
    ax_chi.tick_params(axis="both", which="major", labelsize=12)
    ax_chi.tick_params(axis="both", which="minor", labelsize=11)
    ax_chi.grid(True, alpha=0.25)

    h1, l1 = ax_op.get_legend_handles_labels()
    h2, l2 = ax_op2.get_legend_handles_labels()
    legend_kwargs = dict(
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
    ax_op.legend(h1 + h2, l1 + l2, **legend_kwargs)
    ax_chi.legend(**legend_kwargs)

    preferred_ticks = np.asarray(
        [3000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0, 30000.0, 40000.0, 50000.0],
        dtype=float,
    )
    tick_positions = preferred_ticks[(preferred_ticks >= powers[0] * 0.98) & (preferred_ticks <= powers[-1] * 1.02)]
    tick_positions = np.unique(np.concatenate(([powers[0]], tick_positions, [powers[-1]])))
    for ax in (ax_op, ax_chi):
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        ax.xaxis.set_major_formatter(FuncFormatter(format_power_tick))
        ax.tick_params(axis="x", labelrotation=45, labelsize=11)

    op_text = (
        r"$P_\infty = S_{\max}/N_{\mathrm{active}}$" "\n"
        r"$N_{\mathrm{active}}$: active Tm sites" "\n"
        r"$\langle n_4 \rangle$"
    )
    chi_text = (
        r"$\chi = \left\langle \frac{\sum_{s<S_{\max}} s^2}{\sum_{s<S_{\max}} s} \right\rangle$" "\n"
        r"$s$: finite active-cluster size"
    )
    ax_op.text(
        0.03,
        0.42,
        op_text,
        transform=ax_op.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax_chi.text(
        0.03,
        0.42,
        chi_text,
        transform=ax_chi.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    fig.suptitle(
        f"Percolation-style order parameter and susceptibility | active mode={active_mode_label(active_mode)} | "
        f"snapshots={snapshot_count} | window fraction={window_fraction:.2f}",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.88, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fragment_count(
    results: list[PowerPercolationResult],
    output_path: Path,
    active_mode: str,
    snapshot_count: int,
    window_fraction: float,
) -> None:
    results = sorted(results, key=lambda item: item.power_w_cm2)
    powers = np.asarray([item.power_w_cm2 for item in results], dtype=float)
    fragment_mean = np.asarray([item.fragment_count_mean for item in results], dtype=float)
    fragment_std = np.asarray([item.fragment_count_std for item in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.errorbar(
        powers,
        fragment_mean,
        yerr=fragment_std,
        fmt="o-",
        color="tab:purple",
        linewidth=1.9,
        markersize=4.8,
        capsize=2.5,
        label=r"Fragments $N_{\mathrm{frag}}$",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Excitation power (W cm$^{-2}$)", fontsize=14)
    ax.set_ylabel(r"Number of fragments $N_{\mathrm{frag}}$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, alpha=0.25)

    preferred_ticks = np.asarray(
        [3000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0, 30000.0, 40000.0, 50000.0],
        dtype=float,
    )
    tick_positions = preferred_ticks[(preferred_ticks >= powers[0] * 0.98) & (preferred_ticks <= powers[-1] * 1.02)]
    tick_positions = np.unique(np.concatenate(([powers[0]], tick_positions, [powers[-1]])))
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(FuncFormatter(format_power_tick))
    ax.tick_params(axis="x", labelrotation=45, labelsize=11)

    fragment_text = (
        r"$N_{\mathrm{frag}}$: number of disconnected active clusters" "\n"
        r"active sites defined by " + active_mode_label(active_mode)
    )
    ax.text(
        0.03,
        0.42,
        fragment_text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
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

    fig.suptitle(
        f"Fragment count versus power | active mode={active_mode_label(active_mode)} | "
        f"snapshots={snapshot_count} | window fraction={window_fraction:.2f}",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def derive_fragment_output_path(main_output_path: Path) -> Path:
    stem = main_output_path.stem
    if "percolation_order_parameter_susceptibility" in stem:
        new_stem = stem.replace(
            "percolation_order_parameter_susceptibility",
            "percolation_fragment_count",
        )
    else:
        new_stem = f"{stem}_fragment_count"
    return main_output_path.with_name(f"{new_stem}{main_output_path.suffix}")


def apply_symlog_yaxis(ax: plt.Axes, *arrays: np.ndarray) -> None:
    """Use a sign-preserving log scale with a small linear region near zero."""
    finite_chunks = [
        np.abs(np.asarray(values, dtype=float))
        for values in arrays
        if np.asarray(values, dtype=float).size > 0
    ]
    if not finite_chunks:
        return
    magnitudes = np.concatenate(finite_chunks)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    magnitudes = magnitudes[magnitudes > 0.0]
    if magnitudes.size == 0:
        return

    # Anchor the linear window near the smallest nonzero derivative magnitude
    # so weaker low-power features are still expanded on the plotted scale.
    linthresh = max(float(np.min(magnitudes)) * 0.5, np.finfo(float).tiny)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)


def plot_order_parameter_derivative(
    results: list[PowerPercolationResult],
    output_path: Path,
    active_mode: str,
    snapshot_count: int,
    window_fraction: float,
) -> None:
    results = sorted(results, key=lambda item: item.power_w_cm2)
    powers = np.asarray([item.power_w_cm2 for item in results], dtype=float)
    op = np.asarray([item.order_parameter_mean for item in results], dtype=float)
    if powers.size < 2:
        raise ValueError("Need at least two power points to compute dP_infty/dP.")
    derivative = np.gradient(op, powers)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.plot(
        powers,
        derivative,
        "o-",
        color="tab:orange",
        linewidth=1.9,
        markersize=4.8,
        label=r"$d\mathrm{OP} / dP$",
    )
    ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xscale("log")
    apply_symlog_yaxis(ax, derivative)
    ax.set_xlabel("Excitation power (W cm$^{-2}$)", fontsize=14)
    ax.set_ylabel(r"Order-parameter derivative $d\mathrm{OP} / dP$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, which="both", alpha=0.25)

    preferred_ticks = np.asarray(
        [3000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0, 30000.0, 40000.0, 50000.0],
        dtype=float,
    )
    tick_positions = preferred_ticks[(preferred_ticks >= powers[0] * 0.98) & (preferred_ticks <= powers[-1] * 1.02)]
    tick_positions = np.unique(np.concatenate(([powers[0]], tick_positions, [powers[-1]])))
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(FuncFormatter(format_power_tick))
    ax.tick_params(axis="x", labelrotation=45, labelsize=11)

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

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_active_cluster_derivatives(
    results: list[PowerPercolationResult],
    output_path: Path,
    active_mode: str,
    snapshot_count: int,
    window_fraction: float,
) -> None:
    """Plot relative growth rates (1/N)dN/dP and (1/S)dS/dP versus power."""
    results = sorted(results, key=lambda item: item.power_w_cm2)
    powers = np.asarray([item.power_w_cm2 for item in results], dtype=float)
    active_counts = np.asarray([item.active_site_count_mean for item in results], dtype=float)
    largest_cluster_sizes = np.asarray(
        [item.largest_cluster_size_mean for item in results],
        dtype=float,
    )
    if powers.size < 2:
        raise ValueError("Need at least two power points to compute dN_active/dP and dS_max/dP.")
    if not np.all(np.isfinite(active_counts)) or not np.all(np.isfinite(largest_cluster_sizes)):
        raise ValueError(
            "Percolation summary is missing N_active/S_max sweep means. "
            "Rerun the analysis with the updated script to generate them."
        )

    d_active_d_power = np.gradient(active_counts, powers)
    d_largest_d_power = np.gradient(largest_cluster_sizes, powers)
    relative_active_growth = np.divide(
        d_active_d_power,
        active_counts,
        out=np.full_like(d_active_d_power, np.nan),
        where=active_counts > 0.0,
    )
    relative_largest_growth = np.divide(
        d_largest_d_power,
        largest_cluster_sizes,
        out=np.full_like(d_largest_d_power, np.nan),
        where=largest_cluster_sizes > 0.0,
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.plot(
        powers,
        relative_active_growth,
        "o-",
        color="tab:blue",
        linewidth=1.9,
        markersize=4.8,
        label=r"$\left(1/N_{\mathrm{active}}\right)\, dN_{\mathrm{active}} / dP$",
    )
    ax.plot(
        powers,
        relative_largest_growth,
        "s-",
        color="tab:red",
        linewidth=1.9,
        markersize=4.8,
        label=r"$\left(1/S_{\max}\right)\, dS_{\max} / dP$",
    )
    ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xscale("log")
    apply_symlog_yaxis(ax, relative_active_growth, relative_largest_growth)
    ax.set_xlabel("Excitation power (W cm$^{-2}$)", fontsize=14)
    ax.set_ylabel("Relative cluster-growth derivatives", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, which="both", alpha=0.25)

    preferred_ticks = np.asarray(
        [3000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0, 30000.0, 40000.0, 50000.0],
        dtype=float,
    )
    tick_positions = preferred_ticks[(preferred_ticks >= powers[0] * 0.98) & (preferred_ticks <= powers[-1] * 1.02)]
    tick_positions = np.unique(np.concatenate(([powers[0]], tick_positions, [powers[-1]])))
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(FuncFormatter(format_power_tick))
    ax.tick_params(axis="x", labelrotation=45, labelsize=11)

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

    fig.suptitle(
        f"Relative active-cluster growth versus power | active mode={active_mode_label(active_mode)} | "
        f"snapshots={snapshot_count} | window fraction={window_fraction:.2f}",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def derive_order_parameter_derivative_output_path(main_output_path: Path) -> Path:
    stem = main_output_path.stem
    if "percolation_order_parameter_susceptibility" in stem:
        new_stem = stem.replace(
            "percolation_order_parameter_susceptibility",
            "percolation_order_parameter_derivative",
        )
    else:
        new_stem = f"{stem}_order_parameter_derivative"
    return main_output_path.with_name(f"{new_stem}{main_output_path.suffix}")


def derive_active_cluster_derivative_output_path(main_output_path: Path) -> Path:
    stem = main_output_path.stem
    if "percolation_order_parameter_susceptibility" in stem:
        new_stem = stem.replace(
            "percolation_order_parameter_susceptibility",
            "percolation_active_cluster_derivatives",
        )
    else:
        new_stem = f"{stem}_active_cluster_derivatives"
    return main_output_path.with_name(f"{new_stem}{main_output_path.suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a percolation-style order parameter and susceptibility versus power."
    )
    parser.add_argument(
        "run_root",
        nargs="?",
        default="run10/s12_scale_20",
        help="Sweep root containing power_* subdirectories, or a single power directory.",
    )
    parser.add_argument(
        "--summary-input",
        default=None,
        help="Load an existing percolation summary JSON and replot it without recomputing trajectories.",
    )
    parser.add_argument(
        "--active-mode",
        choices=ACTIVE_MODE_CHOICES,
        default=DEFAULT_ACTIVE_MODE,
        help=(
            "Active-site definition for cluster metrics: legacy `n4+n5` "
            "(interpreted as n2+n4+n5) or `n4`."
        ),
    )
    parser.add_argument(
        "--active-threshold",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=DEFAULT_SNAPSHOT_COUNT,
        help="Number of time snapshots sampled from the tail of each trajectory.",
    )
    parser.add_argument(
        "--window-fraction",
        type=float,
        default=DEFAULT_WINDOW_FRACTION,
        help="Fraction of the trajectory tail used for snapshot sampling.",
    )
    parser.add_argument(
        "--cluster-cutoff-nm",
        type=float,
        default=None,
        help=(
            "Neighbor cutoff in nm for connectivity. Defaults to the interaction radius bound "
            "from each run's manifest."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output PNG path. Defaults to {DEFAULT_OUTPUT_PNG} in the run root.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help=f"Output JSON path. Defaults to {DEFAULT_OUTPUT_JSON} in the run root.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.summary_input is not None:
        summary_input = Path(args.summary_input)
        summary, results = load_summary_results(summary_input)
        summary_active_mode = summary.get("active_mode")
        if summary_active_mode is None:
            active_mode = DEFAULT_ACTIVE_MODE
        else:
            active_mode = normalize_active_mode(summary_active_mode)
        snapshot_count = int(summary["snapshot_count"])
        window_fraction = float(summary["window_fraction"])
        output_path = (
            Path(args.output)
            if args.output is not None
            else summary_input.with_suffix(".png")
        )
        fragment_output_path = derive_fragment_output_path(output_path)
        derivative_output_path = derive_order_parameter_derivative_output_path(output_path)
        active_cluster_derivative_output_path = derive_active_cluster_derivative_output_path(
            output_path
        )
        plot_sweep(
            results=results,
            output_path=output_path,
            active_mode=active_mode,
            snapshot_count=snapshot_count,
            window_fraction=window_fraction,
        )
        plot_fragment_count(
            results=results,
            output_path=fragment_output_path,
            active_mode=active_mode,
            snapshot_count=snapshot_count,
            window_fraction=window_fraction,
        )
        plot_order_parameter_derivative(
            results=results,
            output_path=derivative_output_path,
            active_mode=active_mode,
            snapshot_count=snapshot_count,
            window_fraction=window_fraction,
        )
        active_cluster_derivative_written = False
        active_cluster_derivative_error: ValueError | None = None
        try:
            plot_active_cluster_derivatives(
                results=results,
                output_path=active_cluster_derivative_output_path,
                active_mode=active_mode,
                snapshot_count=snapshot_count,
                window_fraction=window_fraction,
            )
            active_cluster_derivative_written = True
        except ValueError as exc:
            active_cluster_derivative_error = exc
        print(f"Wrote {output_path}")
        print(f"Wrote {fragment_output_path}")
        print(f"Wrote {derivative_output_path}")
        if active_cluster_derivative_written:
            print(f"Wrote {active_cluster_derivative_output_path}")
        elif active_cluster_derivative_error is not None:
            print(
                f"Skipped {active_cluster_derivative_output_path}: "
                f"{active_cluster_derivative_error}"
            )
        print("Per-power snapshot counts:")
        for item in sorted(results, key=lambda result: result.power_w_cm2):
            print(
                "  "
                f"{item.power_w_cm2:.6g} W/cm^2: "
                f"snapshots={item.effective_snapshot_count}"
            )
        return

    if args.active_threshold is not None:
        if int(args.active_threshold) != 3:
            raise ValueError(
                "--active-threshold is deprecated. Use --active-mode n4 or --active-mode 'n4+n5'. "
                "Only the legacy threshold 3 is accepted, and it now maps to the "
                "default n2+n4+n5 mode."
            )
        active_mode = DEFAULT_ACTIVE_MODE
    else:
        active_mode = normalize_active_mode(args.active_mode)

    run_root = Path(args.run_root)
    power_dirs = resolve_power_dirs(run_root)
    geometry_db_path = resolve_geometry_db_path(run_root, power_dirs)
    if args.cluster_cutoff_nm is None:
        first_manifest = load_manifest(power_dirs[0])
        cluster_cutoff_nm = float(first_manifest["geometry"]["interaction_radius_bound_nm"])
    else:
        cluster_cutoff_nm = float(args.cluster_cutoff_nm)
    site_count = load_site_count(geometry_db_path)
    positions = load_site_positions(geometry_db_path)
    if len(positions) != site_count:
        raise ValueError(
            f"Coordinate count {len(positions)} does not match metadata.number_of_sites {site_count}"
        )
    neighbor_pairs = build_neighbor_pairs(positions, cluster_cutoff_nm)

    results: list[PowerPercolationResult] = []
    for power_dir in power_dirs:
        results.append(
            analyze_power_dir(
                power_dir=power_dir,
                active_mode=active_mode,
                snapshot_count=int(args.snapshot_count),
                window_fraction=float(args.window_fraction),
                cluster_cutoff_nm=cluster_cutoff_nm,
                site_count=int(site_count),
                neighbor_pairs=neighbor_pairs,
                geometry_db_path=geometry_db_path,
            )
        )

    mode_suffix = active_mode_slug(active_mode)
    output_path = (
        Path(args.output)
        if args.output is not None
        else run_root / f"percolation_order_parameter_susceptibility_{mode_suffix}.png"
    )
    fragment_output_path = derive_fragment_output_path(output_path)
    derivative_output_path = derive_order_parameter_derivative_output_path(output_path)
    active_cluster_derivative_output_path = derive_active_cluster_derivative_output_path(
        output_path
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output is not None
        else run_root / f"percolation_order_parameter_susceptibility_{mode_suffix}.json"
    )

    plot_sweep(
        results=results,
        output_path=output_path,
        active_mode=active_mode,
        snapshot_count=int(args.snapshot_count),
        window_fraction=float(args.window_fraction),
    )
    plot_fragment_count(
        results=results,
        output_path=fragment_output_path,
        active_mode=active_mode,
        snapshot_count=int(args.snapshot_count),
        window_fraction=float(args.window_fraction),
    )
    plot_order_parameter_derivative(
        results=results,
        output_path=derivative_output_path,
        active_mode=active_mode,
        snapshot_count=int(args.snapshot_count),
        window_fraction=float(args.window_fraction),
    )
    plot_active_cluster_derivatives(
        results=results,
        output_path=active_cluster_derivative_output_path,
        active_mode=active_mode,
        snapshot_count=int(args.snapshot_count),
        window_fraction=float(args.window_fraction),
    )

    summary = {
        "run_root": run_root.as_posix(),
        "active_mode": active_mode,
        "active_states": list(active_mode_states(active_mode)),
        "active_threshold": None,
        "snapshot_count": int(args.snapshot_count),
        "window_fraction": float(args.window_fraction),
        "cluster_cutoff_nm": float(args.cluster_cutoff_nm) if args.cluster_cutoff_nm is not None else None,
        "power_results": [
            {
                "power_w_cm2": float(item.power_w_cm2),
                "seed_count": int(item.seed_count),
                "candidate_snapshot_count": int(item.candidate_snapshot_count),
                "effective_snapshot_count": int(item.effective_snapshot_count),
                "selected_seed": int(item.selected_seed) if item.selected_seed is not None else None,
                "order_parameter_mean": float(item.order_parameter_mean),
                "order_parameter_std": float(item.order_parameter_std),
                "susceptibility_mean": float(item.susceptibility_mean),
                "susceptibility_std": float(item.susceptibility_std),
                "susceptibility_fluctuation": float(item.susceptibility_fluctuation),
                "n4_time_averaged_population_mean": float(item.n4_time_averaged_population_mean),
                "n4_time_averaged_population_std": float(item.n4_time_averaged_population_std),
                "rad_800_events_per_ion_s_mean": float(item.rad_800_events_per_ion_s_mean),
                "rad_800_events_per_ion_s_std": float(item.rad_800_events_per_ion_s_std),
                "largest_cluster_fraction_active_mean": float(item.largest_cluster_fraction_active_mean),
                "largest_cluster_fraction_active_std": float(item.largest_cluster_fraction_active_std),
                "active_site_count_mean": float(item.active_site_count_mean),
                "active_site_count_std": float(item.active_site_count_std),
                "largest_cluster_size_mean": float(item.largest_cluster_size_mean),
                "largest_cluster_size_std": float(item.largest_cluster_size_std),
                "fragment_count_mean": float(item.fragment_count_mean),
                "fragment_count_std": float(item.fragment_count_std),
            }
            for item in results
        ],
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {output_path}")
    print(f"Wrote {fragment_output_path}")
    print(f"Wrote {derivative_output_path}")
    print(f"Wrote {active_cluster_derivative_output_path}")
    print(f"Wrote {summary_path}")
    print("Per-power snapshot counts:")
    for item in sorted(results, key=lambda result: result.power_w_cm2):
        print(
            "  "
            f"{item.power_w_cm2:.6g} W/cm^2: "
            f"snapshots={item.effective_snapshot_count}"
        )


if __name__ == "__main__":
    main()
