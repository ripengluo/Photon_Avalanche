"""Plot one kMC trajectory plus a criticality GIF and cluster-size distribution."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree


N2_LEVEL = 1
N4_LEVEL = 3
N5_LEVEL = 4
RAD_800_LABEL = "Rad 4->1"
DEFAULT_MAX_POINTS = 6000
DEFAULT_MAX_MARKERS = 4000
DEFAULT_GIF_FRAMES = 120
DEFAULT_GIF_FPS = 12
ACTIVE_CLUSTER_STATES = (N2_LEVEL, N4_LEVEL, N5_LEVEL)
ACTIVE_CLUSTER_STATE_LABEL = "n2+n4+n5 (3F4 + 3H4 + 3F3)"
DEFAULT_EXACT_CLUSTER_SIZE_LIMIT = 50
DEFAULT_PSEUDO_ZERO_HEIGHT = 2e-4

STATE_PALETTE = np.asarray(
    [
        to_rgba("#D9D9D9"),
        to_rgba("#F4A261"),
        to_rgba("#2A9D8F"),
        to_rgba("#1D4ED8"),
        to_rgba("#7E57C2"),
    ],
    dtype=float,
)


@dataclass
class SeedSummary:
    seed: int
    final_time: float
    n4_avg: float
    rad_800_count: int


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def load_site_count(np_db_path: Path) -> int:
    with sqlite3.connect(np_db_path) as con:
        row = con.execute("SELECT number_of_sites FROM metadata").fetchone()
    if row is None:
        raise ValueError(f"Missing metadata.number_of_sites in {np_db_path}")
    return int(row[0])


def load_site_positions(np_db_path: Path) -> np.ndarray:
    with sqlite3.connect(np_db_path) as con:
        rows = con.execute("SELECT x, y, z FROM sites ORDER BY site_id").fetchall()
    if not rows:
        raise ValueError(f"No site coordinates in {np_db_path}")
    return np.asarray(rows, dtype=float)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "dre_5level_interaction_manifest.json")


def load_interactions(manifest: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], int]:
    interactions = {
        int(row["interaction_id"]): dict(row)
        for row in manifest["interactions"]
    }
    rad_800_id = next(
        (interaction_id for interaction_id, row in interactions.items() if row["label"] == RAD_800_LABEL),
        None,
    )
    if rad_800_id is None:
        raise ValueError(f"Could not find {RAD_800_LABEL!r}")
    return interactions, int(rad_800_id)


def iter_trajectory_rows(initial_state_db_path: Path, seed: int | None = None) -> Iterator[tuple[Any, ...]]:
    query = """
        SELECT seed, step, time, site_id_1, site_id_2, interaction_id
        FROM trajectories
    """
    params: tuple[Any, ...] = ()
    if seed is not None:
        query += " WHERE seed = ?"
        params = (int(seed),)
    query += " ORDER BY seed, step"
    with sqlite3.connect(initial_state_db_path) as con:
        yield from con.execute(query, params)


def count_seed_events(initial_state_db_path: Path, seed: int) -> int:
    with sqlite3.connect(initial_state_db_path) as con:
        row = con.execute(
            "SELECT COUNT(*) FROM trajectories WHERE seed = ?",
            (int(seed),),
        ).fetchone()
    return int(row[0]) if row is not None else 0


def choose_seed(
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    rad_800_id: int,
    seed_arg: str,
) -> tuple[int, list[SeedSummary]]:
    with sqlite3.connect(initial_state_db_path) as con:
        seed_values = sorted(int(row[0]) for row in con.execute("SELECT DISTINCT seed FROM trajectories"))
    summaries = [
        replay_seed_summary(initial_state_db_path, np_db_path, interactions, rad_800_id, seed)
        for seed in seed_values
    ]
    if not summaries:
        raise ValueError(f"No seeds found in {initial_state_db_path}")
    if seed_arg in {"auto", "all"}:
        best = max(summaries, key=lambda item: (item.n4_avg, item.rad_800_count))
        return best.seed, summaries
    requested = int(seed_arg)
    if requested not in seed_values:
        raise ValueError(f"Seed {requested} not found; available seeds: {seed_values}")
    return requested, summaries


def replay_seed_summary(
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    rad_800_id: int,
    seed: int,
) -> SeedSummary:
    site_count = load_site_count(np_db_path)
    site_states = np.zeros(site_count, dtype=np.int8)
    counts = np.zeros(5, dtype=np.int64)
    counts[0] = site_count
    previous_time = 0.0
    n4_integral = 0.0
    rad_800_count = 0
    have_rows = False

    for row_seed, step, event_time, site_id_1, site_id_2, interaction_id in iter_trajectory_rows(
        initial_state_db_path, seed=seed
    ):
        have_rows = True
        event_time = float(event_time)
        site_id_1 = int(site_id_1)
        site_id_2 = int(site_id_2)
        interaction_id = int(interaction_id)
        interaction = interactions[interaction_id]

        dt = event_time - previous_time
        if dt < -1e-12:
            raise ValueError(f"Time decreased for seed {row_seed} step {step}")
        n4_integral += float(counts[N4_LEVEL]) * max(dt, 0.0)

        apply_interaction(site_states, counts, interaction, site_id_1, site_id_2, row_seed, int(step))
        if interaction_id == rad_800_id:
            rad_800_count += 1
        previous_time = event_time

    if not have_rows:
        raise ValueError(f"Seed {seed} does not exist in {initial_state_db_path}")

    total_time = previous_time
    n4_avg = n4_integral / (float(site_count) * total_time) if total_time > 0 else 0.0
    return SeedSummary(int(seed), float(total_time), float(n4_avg), int(rad_800_count))


def apply_interaction(
    site_states: np.ndarray,
    counts: np.ndarray,
    interaction: dict[str, Any],
    site_id_1: int,
    site_id_2: int,
    row_seed: int,
    step: int,
) -> None:
    current_state_1 = int(site_states[site_id_1])
    if current_state_1 != int(interaction["left_state_1"]):
        raise ValueError(
            f"Replay mismatch at seed {row_seed} step {step} site {site_id_1}: "
            f"expected {interaction['left_state_1']}, found {current_state_1}"
        )
    counts[current_state_1] -= 1
    counts[int(interaction["right_state_1"])] += 1
    site_states[site_id_1] = int(interaction["right_state_1"])

    if int(interaction["number_of_sites"]) != 2:
        return

    current_state_2 = int(site_states[site_id_2])
    if current_state_2 != int(interaction["left_state_2"]):
        raise ValueError(
            f"Replay mismatch at seed {row_seed} step {step} site {site_id_2}: "
            f"expected {interaction['left_state_2']}, found {current_state_2}"
        )
    counts[current_state_2] -= 1
    counts[int(interaction["right_state_2"])] += 1
    site_states[site_id_2] = int(interaction["right_state_2"])


def extract_seed_trajectory(
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    rad_800_id: int,
    seed: int,
) -> dict[str, Any]:
    site_count = load_site_count(np_db_path)
    site_states = np.zeros(site_count, dtype=np.int8)
    counts = np.zeros(5, dtype=np.int64)
    counts[0] = site_count

    times = [0.0]
    n2 = [0.0]
    n4 = [0.0]
    n5 = [0.0]
    cumulative_800 = [0]
    event_times_800: list[float] = []
    event_points_800: list[tuple[float, float, float]] = []
    previous_time = 0.0
    n4_integral = 0.0
    rad_800_count = 0

    for row_seed, step, event_time, site_id_1, site_id_2, interaction_id in iter_trajectory_rows(
        initial_state_db_path, seed=seed
    ):
        event_time = float(event_time)
        site_id_1 = int(site_id_1)
        site_id_2 = int(site_id_2)
        interaction_id = int(interaction_id)
        interaction = interactions[interaction_id]

        dt = event_time - previous_time
        if dt < -1e-12:
            raise ValueError(f"Time decreased for seed {row_seed} step {step}")
        n4_integral += float(counts[N4_LEVEL]) * max(dt, 0.0)

        apply_interaction(site_states, counts, interaction, site_id_1, site_id_2, int(row_seed), int(step))
        if interaction_id == rad_800_id:
            rad_800_count += 1
            event_times_800.append(event_time)
            event_points_800.append(
                (
                    counts[N2_LEVEL] / site_count,
                    counts[N4_LEVEL] / site_count,
                    counts[N5_LEVEL] / site_count,
                )
            )

        times.append(event_time)
        n2.append(counts[N2_LEVEL] / site_count)
        n4.append(counts[N4_LEVEL] / site_count)
        n5.append(counts[N5_LEVEL] / site_count)
        cumulative_800.append(rad_800_count)
        previous_time = event_time

    total_time = previous_time
    return {
        "seed": int(seed),
        "site_count": int(site_count),
        "total_time": float(total_time),
        "n4_avg": float(n4_integral / (float(site_count) * total_time) if total_time > 0 else 0.0),
        "times": np.asarray(times, dtype=float),
        "n2": np.asarray(n2, dtype=float),
        "n4": np.asarray(n4, dtype=float),
        "n5": np.asarray(n5, dtype=float),
        "cumulative_800": np.asarray(cumulative_800, dtype=int),
        "event_times_800": np.asarray(event_times_800, dtype=float),
        "event_points_800": np.asarray(event_points_800, dtype=float) if event_points_800 else np.zeros((0, 3), dtype=float),
        "rad_800_count": int(rad_800_count),
    }


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def segment_collection(times: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([xs, ys, zs])
    return np.stack([points[:-1], points[1:]], axis=1), times[:-1]


def state_to_rgba(states: np.ndarray) -> np.ndarray:
    return STATE_PALETTE[np.asarray(states, dtype=np.intp)]


def build_neighbor_pairs(positions: np.ndarray, cutoff_nm: float) -> np.ndarray:
    if cutoff_nm <= 0:
        return np.zeros((0, 2), dtype=int)
    pairs = cKDTree(positions).query_pairs(float(cutoff_nm))
    return np.asarray(list(pairs), dtype=int) if pairs else np.zeros((0, 2), dtype=int)


def build_cluster_active_mask(site_states: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(site_states, dtype=np.int8), np.asarray(ACTIVE_CLUSTER_STATES, dtype=np.int8))


def compute_active_cluster_statistics(
    site_states: np.ndarray,
    neighbor_pairs: np.ndarray,
) -> dict[str, Any]:
    active_mask = build_cluster_active_mask(site_states)
    active_indices = np.flatnonzero(active_mask)
    active_fraction = float(active_indices.size) / float(site_states.size)
    if active_indices.size == 0:
        return {
            "largest_mask": np.zeros_like(active_mask, dtype=bool),
            "active_fraction": active_fraction,
            "largest_fraction_all": 0.0,
            "largest_fraction_active": 0.0,
            "cluster_sizes": np.zeros(0, dtype=int),
        }

    parent = np.arange(site_states.size, dtype=np.int32)
    rank = np.zeros(site_states.size, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

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

    for i, j in neighbor_pairs:
        if active_mask[int(i)] and active_mask[int(j)]:
            union(int(i), int(j))

    active_roots = np.fromiter((find(int(i)) for i in active_indices), dtype=np.int32)
    unique_roots, counts = np.unique(active_roots, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique_roots = unique_roots[order]
    counts = counts[order]
    largest_root = int(unique_roots[0])
    largest_mask = np.zeros_like(active_mask, dtype=bool)
    largest_mask[active_indices[active_roots == largest_root]] = True
    largest_size = int(counts[0])
    return {
        "largest_mask": largest_mask,
        "active_fraction": active_fraction,
        "largest_fraction_all": float(largest_size) / float(site_states.size),
        "largest_fraction_active": float(largest_size) / float(active_indices.size),
        "cluster_sizes": counts.astype(int),
    }


def build_movie_frames(
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    rad_800_id: int,
    seed: int,
    cluster_cutoff_nm: float,
    frame_count: int,
) -> dict[str, Any]:
    site_count = load_site_count(np_db_path)
    positions = load_site_positions(np_db_path)
    if len(positions) != site_count:
        raise ValueError("Site position count mismatch")

    n_events = count_seed_events(initial_state_db_path, seed)
    if n_events <= 0:
        raise ValueError(f"Seed {seed} not found")

    event_indices = set(downsample_indices(n_events, max(1, max(2, int(frame_count)) - 1)).tolist())
    neighbor_pairs = build_neighbor_pairs(positions, float(cluster_cutoff_nm))

    site_states = np.zeros(site_count, dtype=np.int8)
    counts = np.zeros(5, dtype=np.int64)
    counts[0] = site_count
    previous_time = 0.0
    n4_integral = 0.0
    rad_800_count = 0

    frames: list[dict[str, Any]] = []
    initial_stats = compute_active_cluster_statistics(site_states, neighbor_pairs)
    frames.append(
        {
            "time": 0.0,
            "step": 0,
            "states": site_states.copy(),
            "n4": 0.0,
            "cumulative_800": 0,
            "rad_800_sites": np.zeros(0, dtype=int),
            "active_fraction": float(initial_stats["active_fraction"]),
            "largest_fraction_all": float(initial_stats["largest_fraction_all"]),
            "largest_fraction_active": float(initial_stats["largest_fraction_active"]),
            "largest_mask": np.asarray(initial_stats["largest_mask"], dtype=bool),
            "cluster_sizes": np.asarray(initial_stats["cluster_sizes"], dtype=int),
        }
    )

    for event_index, row in enumerate(iter_trajectory_rows(initial_state_db_path, seed=seed)):
        row_seed, step, event_time, site_id_1, site_id_2, interaction_id = row
        event_time = float(event_time)
        site_id_1 = int(site_id_1)
        site_id_2 = int(site_id_2)
        interaction_id = int(interaction_id)
        interaction = interactions[interaction_id]

        dt = event_time - previous_time
        if dt < -1e-12:
            raise ValueError(f"Time decreased for seed {row_seed} step {step}")
        n4_integral += float(counts[N4_LEVEL]) * max(dt, 0.0)

        apply_interaction(site_states, counts, interaction, site_id_1, site_id_2, int(row_seed), int(step))
        if interaction_id == rad_800_id:
            rad_800_count += 1

        if event_index in event_indices:
            stats = compute_active_cluster_statistics(site_states, neighbor_pairs)
            frames.append(
                {
                    "time": float(event_time),
                    "step": int(step),
                    "states": site_states.copy(),
                    "n4": float(counts[N4_LEVEL] / site_count),
                    "cumulative_800": int(rad_800_count),
                    "rad_800_sites": np.asarray([site_id_1], dtype=int) if interaction_id == rad_800_id else np.zeros(0, dtype=int),
                    "active_fraction": float(stats["active_fraction"]),
                    "largest_fraction_all": float(stats["largest_fraction_all"]),
                    "largest_fraction_active": float(stats["largest_fraction_active"]),
                    "largest_mask": np.asarray(stats["largest_mask"], dtype=bool),
                    "cluster_sizes": np.asarray(stats["cluster_sizes"], dtype=int),
                }
            )
        previous_time = event_time

    total_time = previous_time
    n4_avg = n4_integral / (float(site_count) * total_time) if total_time > 0 else 0.0
    return {
        "seed": int(seed),
        "positions": positions,
        "frames": frames,
        "event_count": int(n_events),
        "rad_800_count": int(rad_800_count),
        "n4_avg": float(n4_avg),
        "cluster_cutoff_nm": float(cluster_cutoff_nm),
        "active_states": [int(v) for v in ACTIVE_CLUSTER_STATES],
        "active_state_label": ACTIVE_CLUSTER_STATE_LABEL,
    }


def build_criticality_gif(run_dir: Path, movie: dict[str, Any], output_path: Path, fps: int) -> None:
    positions = np.asarray(movie["positions"], dtype=float)
    frames = movie["frames"]
    x, y, z = positions.T
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(np.maximum(maxs - mins, 1e-6))) * 1.005, 1e-3)

    fig = plt.figure(figsize=(8.4, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22, azim=35)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")

    first_states = np.asarray(frames[0]["states"], dtype=np.int8)
    base = ax.scatter(
        x,
        y,
        z,
        c=state_to_rgba(first_states),
        s=np.where(build_cluster_active_mask(first_states), 13.0, 5.8),
        depthshade=False,
        linewidths=0.0,
    )
    cluster = ax.scatter([], [], [], s=42, facecolors="none", edgecolors="gold", linewidths=1.0, depthshade=False)
    flash = ax.scatter([], [], [], s=120, marker="*", c="crimson", depthshade=False)

    state_legend = ax.legend(
        handles=[
            Patch(facecolor=STATE_PALETTE[0], edgecolor="0.35", label="3H6"),
            Patch(facecolor=STATE_PALETTE[1], edgecolor="0.35", label="3F4"),
            Patch(facecolor=STATE_PALETTE[2], edgecolor="0.35", label="3H5"),
            Patch(facecolor=STATE_PALETTE[3], edgecolor="0.35", label="3H4"),
            Patch(facecolor=STATE_PALETTE[4], edgecolor="0.35", label="3F3"),
        ],
        title="Site states",
        loc="upper left",
        frameon=True,
        framealpha=0.88,
        fontsize=9,
        title_fontsize=9,
    )
    ax.add_artist(state_legend)


    def set_offsets(scatter: Any, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        scatter._offsets3d = (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float))

    def update(frame_index: int):
        frame = frames[frame_index]
        states = np.asarray(frame["states"], dtype=np.int8)
        colors = state_to_rgba(states)
        sizes = np.where(build_cluster_active_mask(states), 13.0, 5.8)
        base.set_facecolors(colors)
        base.set_edgecolors(colors)
        base.set_sizes(sizes)
        giant_mask = np.asarray(frame["largest_mask"], dtype=bool)
        flash_sites = np.asarray(frame["rad_800_sites"], dtype=int)
        set_offsets(cluster, x[giant_mask], y[giant_mask], z[giant_mask])
        set_offsets(flash, x[flash_sites], y[flash_sites], z[flash_sites])
        event_count = int(movie.get("event_count", 0))
        event_step = int(frame.get("step", 0))
        ax.set_title(
            f"t = {frame['time']:.3e} s | sampled frame = {frame_index}/{len(frames) - 1} | kMC step = {event_step}/{event_count} | largest cluster = {frame['largest_fraction_all']:.3e}",
            fontsize=10,
            pad=12,
        )
        return base, cluster, flash

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False, interval=1000 / max(int(fps), 1))
    anim.save(output_path, writer=animation.PillowWriter(fps=int(fps)), dpi=180)
    plt.close(fig)


def plot_cluster_size_distribution(run_dir: Path, movie: dict[str, Any], output_path: Path) -> dict[str, Any]:
    all_sizes: list[int] = []
    for frame in movie["frames"][1:]:
        sizes = np.asarray(frame.get("cluster_sizes", np.zeros(0, dtype=int)), dtype=int)
        if sizes.size:
            all_sizes.extend(int(v) for v in sizes)

    size_array = np.asarray(all_sizes if all_sizes else [1], dtype=int)
    unique_sizes, counts = np.unique(size_array, return_counts=True)
    total_count = int(np.sum(counts))
    probabilities = counts.astype(float) / float(total_count)
    max_size = int(np.max(unique_sizes))
    exact_limit = int(DEFAULT_EXACT_CLUSTER_SIZE_LIMIT)

    exact_sizes = np.arange(1, min(max_size, exact_limit - 1) + 1, dtype=float)
    exact_prob = np.zeros_like(exact_sizes, dtype=float)
    exact_lookup = {int(size): float(prob) for size, prob in zip(unique_sizes, probabilities) if int(size) < exact_limit}
    for idx, size in enumerate(exact_sizes.astype(int)):
        exact_prob[idx] = exact_lookup.get(int(size), 0.0)
    exact_positive = exact_prob > 0
    exact_prob_visible = exact_prob.copy()
    exact_prob_visible[(exact_sizes >= 10) & (~exact_positive)] = DEFAULT_PSEUDO_ZERO_HEIGHT
    exact_visible_mask = exact_positive | (exact_sizes >= 10)

    tail_bin_count = 0
    tail_edges = np.zeros(0, dtype=float)
    tail_prob = np.zeros(0, dtype=float)
    tail_prob_visible = np.zeros(0, dtype=float)
    tail_centers = np.zeros(0, dtype=float)
    tail_nonzero = np.zeros(0, dtype=bool)
    if np.any(size_array >= exact_limit):
        tail_sizes = size_array[size_array >= exact_limit]
        if max_size > exact_limit:
            tail_bin_count = int(min(16, max(4, math.ceil(math.log10((max_size + 1.0) / float(exact_limit)) * 6.0))))
            tail_edges = np.geomspace(float(exact_limit), float(max_size) + 1.0, num=tail_bin_count + 1)
            tail_edges[0] = float(exact_limit)
            tail_edges[-1] = float(max_size) + 1.0
            tail_edges = np.unique(tail_edges)
        if tail_edges.size < 2:
            tail_edges = np.asarray([float(exact_limit), float(max_size) + 1.0], dtype=float)
        tail_hist, tail_edges = np.histogram(tail_sizes, bins=tail_edges)
        tail_bin_count = int(max(tail_edges.size - 1, 0))
        tail_prob = tail_hist.astype(float) / float(total_count)
        tail_prob_visible = np.where(tail_prob > 0, tail_prob, DEFAULT_PSEUDO_ZERO_HEIGHT)
        tail_centers = np.sqrt(tail_edges[:-1] * tail_edges[1:])
        tail_nonzero = tail_hist > 0

    fig, ax = plt.subplots(figsize=(8.7, 6.1), dpi=220)

    x_ticks = np.asarray([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000], dtype=float)
    x_ticks = x_ticks[x_ticks <= max_size]
    if x_ticks.size == 0:
        x_ticks = np.asarray([1.0], dtype=float)

    plot_exact_sizes = exact_sizes[exact_visible_mask]
    plot_exact_prob = exact_prob_visible[exact_visible_mask]
    if plot_exact_sizes.size > 1:
        ax.plot(plot_exact_sizes, plot_exact_prob, color="#1D4ED8", linewidth=1.1, alpha=0.82, zorder=3)
    if plot_exact_sizes.size:
        ax.scatter(plot_exact_sizes, plot_exact_prob, color="#1D4ED8", s=26, linewidths=0.0, zorder=4)
    if tail_prob_visible.size:
        ax.plot(tail_centers, tail_prob_visible, color="#D97706", linewidth=1.4, alpha=0.90, zorder=3)
        ax.scatter(tail_centers, tail_prob_visible, color="#D97706", marker="s", s=24, linewidths=0.0, zorder=4)

    ymax = max(
        float(np.max(exact_prob_visible)) if exact_prob_visible.size else 0.0,
        float(np.max(tail_prob_visible)) if tail_prob_visible.size else 0.0,
        DEFAULT_PSEUDO_ZERO_HEIGHT,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.82, max_size * 1.15)
    y_upper = max(1.25, ymax * 1.12, DEFAULT_PSEUDO_ZERO_HEIGHT * 1.2)
    ax.set_ylim(DEFAULT_PSEUDO_ZERO_HEIGHT, y_upper)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}" if value >= 1 else ""))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=200))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=3.0, width=0.6)
    ax.set_xlabel("Cluster size s", fontsize=16)
    ax.set_ylabel("Fraction of sampled clusters", fontsize=16)
    ax.grid(True, which="both", alpha=0.25)
    y_major = [DEFAULT_PSEUDO_ZERO_HEIGHT]
    decade_ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for tick in decade_ticks:
        if tick >= DEFAULT_PSEUDO_ZERO_HEIGHT and tick <= y_upper * 1.001:
            y_major.append(float(tick))
    y_major = np.asarray(sorted(set(y_major)), dtype=float)
    ax.yaxis.set_major_locator(FixedLocator(y_major))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: "0"
            if abs(value - DEFAULT_PSEUDO_ZERO_HEIGHT) < 1e-12
            else ("1" if abs(value - 1.0) < 1e-12 else f"1e{int(round(math.log10(value)))}")
        )
    )
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=3.0, width=0.6)
    ax.text(-0.02, -0.08, "0", transform=ax.transAxes, ha="center", va="top", fontsize=11, color="0.25", clip_on=False)
    ax.plot([0.0, 0.0], [0.0, -0.03], transform=ax.transAxes, color="0.45", clip_on=False, lw=0.9)
    fig.suptitle(
        f"{run_dir.as_posix()}\nseed={movie['seed']} | aggregated active-cluster size distribution "
        f"(exact PMF for s < {exact_limit}, log-binned tail for s >= {exact_limit})",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return {
        "output": output_path.as_posix(),
        "distribution_mode": "aggregated_small_exact_tail_log_binned_single_axis",
        "bin_count": int(tail_bin_count),
        "exact_pmf_size_limit_exclusive": int(exact_limit),
        "tail_bin_start_inclusive": int(exact_limit),
        "cluster_sample_count": int(size_array.size),
        "max_cluster_size": int(max_size),
        "mean_cluster_size": float(np.mean(size_array)),
        "median_cluster_size": float(np.median(size_array)),
    }


def plot_trajectory(
    run_dir: Path,
    trajectory: dict[str, Any],
    selected_seed: int,
    summaries: list[SeedSummary],
    overlay_all_seeds: bool,
    output_path: Path,
    max_points: int,
    max_markers: int,
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    rad_800_id: int,
) -> None:
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_ts2 = ax_ts.twinx()

    if overlay_all_seeds:
        for summary in summaries:
            traj = extract_seed_trajectory(initial_state_db_path, np_db_path, interactions, rad_800_id, summary.seed)
            idx = downsample_indices(len(traj["times"]), max(1000, max_points // 4))
            ax3d.plot(np.asarray(traj["n2"])[idx], np.asarray(traj["n4"])[idx], np.asarray(traj["n5"])[idx], color="0.7", alpha=0.15, linewidth=0.8)

    times = np.asarray(trajectory["times"], dtype=float)
    n2 = np.asarray(trajectory["n2"], dtype=float)
    n4 = np.asarray(trajectory["n4"], dtype=float)
    n5 = np.asarray(trajectory["n5"], dtype=float)
    cumulative_800 = np.asarray(trajectory["cumulative_800"], dtype=int)
    event_points_800 = np.asarray(trajectory["event_points_800"], dtype=float)
    idx = downsample_indices(len(times), max_points)
    times_ds = times[idx]
    n2_ds = n2[idx]
    n4_ds = n4[idx]
    n5_ds = n5[idx]
    cumulative_800_ds = cumulative_800[idx]

    if len(times_ds) >= 2:
        segments, segment_times = segment_collection(times_ds, n2_ds, n4_ds, n5_ds)
        lc = Line3DCollection(segments, cmap="viridis", norm=Normalize(vmin=float(times_ds[0]), vmax=float(times_ds[-1])), linewidth=1.4, alpha=0.9)
        lc.set_array(segment_times)
        ax3d.add_collection3d(lc)
        cbar = fig.colorbar(lc, ax=ax3d, pad=0.06, fraction=0.04)
        cbar.set_label("Time (s)")
    else:
        ax3d.plot(n2_ds, n4_ds, n5_ds, color="tab:blue", linewidth=1.4)

    ax3d.scatter(n2_ds[0], n4_ds[0], n5_ds[0], color="limegreen", s=45)
    ax3d.scatter(n2_ds[-1], n4_ds[-1], n5_ds[-1], color="black", s=45)
    if len(event_points_800):
        marker_idx = downsample_indices(len(event_points_800), max_markers)
        pts = event_points_800[marker_idx]
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="crimson", s=10, alpha=0.55)
    ax3d.set_title(
        f"3D n2-n4-n5 trajectory, {'all seeds overlayed, highlighted ' if overlay_all_seeds else ''}seed {selected_seed}",
        fontsize=11,
    )
    ax3d.set_xlabel(r"$n_2$ (3F4 fraction)")
    ax3d.set_ylabel(r"$n_4$ (3H4 fraction)")
    ax3d.set_zlabel(r"$n_5$ (3F3 fraction)")
    ax3d.view_init(elev=22, azim=35)
    ax3d.grid(True, alpha=0.2)

    ax_ts.set_title("Population and 800 nm emission vs time", fontsize=11)
    ax_ts.plot(times_ds, n2_ds, color="tab:orange", linewidth=1.2, label=r"$n_2$")
    ax_ts.plot(times_ds, n4_ds, color="tab:blue", linewidth=1.8, label=r"$n_4$")
    ax_ts.plot(times_ds, n5_ds, color="tab:green", linewidth=1.2, label=r"$n_5$")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Population fraction")
    ax_ts.grid(True, alpha=0.25)
    ax_ts.set_ylim(bottom=0.0)
    ax_ts2.plot(times_ds, cumulative_800_ds, color="crimson", linewidth=1.4, label="cumulative 800 nm events")
    ax_ts2.set_ylabel("Cumulative 800 nm events", color="crimson")
    ax_ts2.tick_params(axis="y", colors="crimson")
    h1, l1 = ax_ts.get_legend_handles_labels()
    h2, l2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)

    best = max(summaries, key=lambda item: (item.n4_avg, item.rad_800_count))
    fig.suptitle(
        f"{run_dir.as_posix()}\nselected seed={selected_seed}, n4_avg={trajectory['n4_avg']:.3e}, "
        f"800nm events={trajectory['rad_800_count']}, best seed={best.seed}, best n4_avg={best.n4_avg:.3e}",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.84, wspace=0.24)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_summary_json(
    manifest: dict[str, Any],
    summaries: list[SeedSummary],
    selected_seed: int,
    trajectory: dict[str, Any],
    movie: dict[str, Any],
    cluster_distribution: dict[str, Any],
    gif_output: Path,
    cluster_output: Path,
) -> dict[str, Any]:
    best = max(summaries, key=lambda item: (item.n4_avg, item.rad_800_count))
    return {
        "run_dir": manifest.get("condition_root"),
        "selected_seed": int(selected_seed),
        "best_seed": int(best.seed),
        "selected_seed_n4_avg": float(trajectory["n4_avg"]),
        "selected_seed_rad_800_count": int(trajectory["rad_800_count"]),
        "selected_seed_total_time_s": float(trajectory["total_time"]),
        "seed_summaries": [
            {
                "seed": int(item.seed),
                "final_time": float(item.final_time),
                "n4_avg": float(item.n4_avg),
                "rad_800_count": int(item.rad_800_count),
            }
            for item in summaries
        ],
        "gif_output": gif_output.as_posix(),
        "cluster_distribution_output": cluster_output.as_posix(),
        "movie": {
            "gif_frames": int(len(movie["frames"])),
            "gif_fps": None,
            "gif_active_threshold": None,
            "gif_active_states": [int(v) for v in movie["active_states"]],
            "gif_active_state_label": str(movie["active_state_label"]),
            "gif_cluster_cutoff_nm": float(movie["cluster_cutoff_nm"]),
            "movie_n4_avg": float(movie["n4_avg"]),
            "movie_rad_800_count": int(movie["rad_800_count"]),
            "movie_peak_active_fraction": float(max(frame["active_fraction"] for frame in movie["frames"])),
            "movie_peak_largest_cluster_fraction_all": float(max(frame["largest_fraction_all"] for frame in movie["frames"])),
            "movie_peak_largest_cluster_fraction_active": float(max(frame["largest_fraction_active"] for frame in movie["frames"])),
        },
        "cluster_distribution": cluster_distribution,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one kMC trajectory plus criticality outputs.")
    parser.add_argument("run_dir", help="Path to the power run directory.")
    parser.add_argument("--seed", default="auto", help="Integer seed, 'auto', or 'all'.")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--max-markers", type=int, default=DEFAULT_MAX_MARKERS)
    parser.add_argument("--output", default=None, help="Output PNG path for the overview plot.")
    parser.add_argument("--overlay-all-seeds", action="store_true")
    parser.add_argument("--gif-output", default=None)
    parser.add_argument("--gif-frames", type=int, default=DEFAULT_GIF_FRAMES)
    parser.add_argument("--gif-fps", type=int, default=DEFAULT_GIF_FPS)
    parser.add_argument(
        "--gif-active-threshold",
        type=int,
        default=None,
        help="Deprecated and ignored. Active cluster sites are fixed to n2+n4+n5.",
    )
    parser.add_argument("--gif-cluster-cutoff-nm", type=float, default=None)
    parser.add_argument("--cluster-output", default=None, help="Output PNG path for cluster-size distribution.")
    parser.add_argument("--summary-output", default=None, help="Output JSON summary path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    initial_state_db_path = run_dir / "initial_state.sqlite"
    np_db_path = run_dir / "np.sqlite"
    manifest = load_manifest(run_dir)
    interactions, rad_800_id = load_interactions(manifest)

    selected_seed, summaries = choose_seed(
        initial_state_db_path=initial_state_db_path,
        np_db_path=np_db_path,
        interactions=interactions,
        rad_800_id=rad_800_id,
        seed_arg=str(args.seed),
    )

    trajectory = extract_seed_trajectory(
        initial_state_db_path=initial_state_db_path,
        np_db_path=np_db_path,
        interactions=interactions,
        rad_800_id=rad_800_id,
        seed=selected_seed,
    )

    overview_output = Path(args.output) if args.output is not None else run_dir / "trajectory_3d_overview.png"
    gif_output = Path(args.gif_output) if args.gif_output is not None else run_dir / "trajectory_criticality.gif"
    cluster_output = Path(args.cluster_output) if args.cluster_output is not None else run_dir / "cluster_size_distribution.png"
    summary_output = Path(args.summary_output) if args.summary_output is not None else run_dir / "trajectory_3d_overview_summary.json"

    plot_trajectory(
        run_dir=run_dir,
        trajectory=trajectory,
        selected_seed=selected_seed,
        summaries=summaries,
        overlay_all_seeds=bool(args.overlay_all_seeds or str(args.seed) == "all"),
        output_path=overview_output,
        max_points=int(args.max_points),
        max_markers=int(args.max_markers),
        initial_state_db_path=initial_state_db_path,
        np_db_path=np_db_path,
        interactions=interactions,
        rad_800_id=rad_800_id,
    )

    movie = build_movie_frames(
        initial_state_db_path=initial_state_db_path,
        np_db_path=np_db_path,
        interactions=interactions,
        rad_800_id=rad_800_id,
        seed=selected_seed,
        cluster_cutoff_nm=float(
            args.gif_cluster_cutoff_nm
            if args.gif_cluster_cutoff_nm is not None
            else float(manifest["geometry"]["interaction_radius_bound_nm"])
        ),
        frame_count=int(args.gif_frames),
    )
    build_criticality_gif(run_dir=run_dir, movie=movie, output_path=gif_output, fps=int(args.gif_fps))
    cluster_distribution = plot_cluster_size_distribution(run_dir=run_dir, movie=movie, output_path=cluster_output)

    summary = build_summary_json(
        manifest=manifest,
        summaries=summaries,
        selected_seed=selected_seed,
        trajectory=trajectory,
        movie=movie,
        cluster_distribution=cluster_distribution,
        gif_output=gif_output,
        cluster_output=cluster_output,
    )
    summary["movie"]["gif_fps"] = int(args.gif_fps)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {overview_output}")
    print(f"Wrote {gif_output}")
    print(f"Wrote {cluster_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
