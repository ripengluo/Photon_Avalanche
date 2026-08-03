#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter
N1_LEVEL = 0
N2_LEVEL = 1
N3_LEVEL = 2
N4_LEVEL = 3
N5_LEVEL = 4
N6_LEVEL = 5
N5PLUS_LEVELS = (N5_LEVEL, N6_LEVEL)
BOUNDARY_OFFSET_NM = 1.0
SNAPSHOT_TAIL_FRACTION = 1.0
ISO_SECTION_HALF_THICKNESS_NM = 2.0
CIRCLE_BOUNDARY_POINTS = 361
STATE_VISUAL_N2_ONLY = "n2-only"
STATE_VISUAL_N2_N4_N5PLUS = "n2-n4-n5plus"
GROUPED_AXES_FACE_COLOR = "#515051"
DEFAULT_AXES_FACE_COLOR = "#fbfaf7"
LEVEL_COUNT_LABELS = ("3F4", "3H5", "3H4", "3F2+3F3")
BALL_DIAMETER_SCALE = 2.0
# Matplotlib scatter sizes are area-like, so doubling marker diameter needs 4x area.
BALL_AREA_SCALE = BALL_DIAMETER_SCALE ** 2

STATE_COLORS = {
    0: "#D9D9D9",
    1: "#F97316",
    2: "#C7CDD3",
    3: "#A9B1BA",
    4: "#7B8390",
}
GROUND_RGBA_FRONT = np.asarray(to_rgba("#EAEAEA", alpha=0.12), dtype=float)
GROUND_RGBA_BACK = np.asarray(to_rgba("#8D949D", alpha=0.30), dtype=float)
OTHER_RGBA_FRONT = np.asarray(to_rgba("#D8DDE3", alpha=0.18), dtype=float)
OTHER_RGBA_BACK = np.asarray(to_rgba("#757D88", alpha=0.36), dtype=float)
N2_FACE_RGBA = np.asarray(to_rgba(STATE_COLORS[N2_LEVEL], alpha=0.0), dtype=float)
N2_EDGE_RGBA = np.asarray(to_rgba("#161616", alpha=0.82), dtype=float)
N2_GROUPED_FACE_RGBA = np.asarray(to_rgba("#FACC15", alpha=0.92), dtype=float)
N2_GROUPED_EDGE_RGBA = np.asarray(to_rgba("#713F12", alpha=0.95), dtype=float)
N3_OPTIONAL_FACE_RGBA = np.asarray(to_rgba("#F97316", alpha=0.92), dtype=float)
N3_OPTIONAL_EDGE_RGBA = np.asarray(to_rgba("#7C2D12", alpha=0.96), dtype=float)
N4_GROUPED_FACE_RGBA = np.asarray(to_rgba("#DC2626", alpha=0.92), dtype=float)
N4_GROUPED_EDGE_RGBA = np.asarray(to_rgba("#450A0A", alpha=0.96), dtype=float)
N5PLUS_FACE_RGBA = np.asarray(to_rgba("#7C3AED", alpha=0.88), dtype=float)
N5PLUS_EDGE_RGBA = np.asarray(to_rgba("#2E1065", alpha=0.94), dtype=float)
GROUND_SIZE = 7.2 * BALL_AREA_SCALE
OTHER_SIZE = 12.0 * BALL_AREA_SCALE
N2_SIZE = 24.0 * BALL_AREA_SCALE
N2_GROUPED_SIZE = 26.0 * BALL_AREA_SCALE
N3_OPTIONAL_SIZE = 24.0 * BALL_AREA_SCALE
N4_GROUPED_SIZE = 23.0 * BALL_AREA_SCALE
N5PLUS_SIZE = 20.0 * BALL_AREA_SCALE


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    for name in ("npt_interaction_manifest.json", "dre_5level_interaction_manifest.json"):
        path = run_dir / name
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(f"No interaction manifest found in {run_dir}")


def load_interactions(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["interaction_id"]): dict(row) for row in manifest["interactions"]}


def load_site_count(np_db_path: Path) -> int:
    with sqlite3.connect(np_db_path) as con:
        row = con.execute("SELECT number_of_sites FROM metadata").fetchone()
    if row is None:
        raise ValueError(f"Missing metadata.number_of_sites in {np_db_path}")
    return int(row[0])


def load_site_geometry(np_db_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with sqlite3.connect(np_db_path) as con:
        rows = con.execute("SELECT x, y, z, species_id FROM sites ORDER BY site_id").fetchall()
    if not rows:
        raise ValueError(f"No site coordinates found in {np_db_path}")
    positions = np.asarray([row[:3] for row in rows], dtype=float)
    species_ids = np.asarray([row[3] for row in rows], dtype=np.int32)
    return positions, species_ids


def iter_trajectory_rows(initial_state_db_path: Path, seed: int) -> Iterator[tuple[Any, ...]]:
    query = """
        SELECT seed, step, time, site_id_1, site_id_2, interaction_id
        FROM trajectories
        WHERE seed = ?
        ORDER BY step
    """
    with sqlite3.connect(initial_state_db_path) as con:
        yield from con.execute(query, (int(seed),))


def load_selected_seed(run_dir: Path, initial_state_db_path: Path, seed_arg: str) -> int:
    if seed_arg != "auto":
        return int(seed_arg)

    summary_path = run_dir / "trajectory_3d_overview_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        if "selected_seed" in summary:
            return int(summary["selected_seed"])
        if "best_seed" in summary:
            return int(summary["best_seed"])

    with sqlite3.connect(initial_state_db_path) as con:
        row = con.execute(
            "SELECT seed FROM trajectories ORDER BY seed, step LIMIT 1"
        ).fetchone()
    if row is None:
        raise ValueError(f"No trajectory rows found in {initial_state_db_path}")
    return int(row[0])


def load_total_time(
    run_dir: Path,
    initial_state_db_path: Path,
    seed: int,
) -> float:
    summary_path = run_dir / "trajectory_3d_overview_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        if int(summary.get("selected_seed", -1)) == int(seed):
            if "selected_seed_total_time_s" in summary:
                return float(summary["selected_seed_total_time_s"])

    with sqlite3.connect(initial_state_db_path) as con:
        row = con.execute(
            "SELECT MAX(time) FROM trajectories WHERE seed = ?",
            (int(seed),),
        ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"Could not determine total time for seed {seed}")
    return float(row[0])


def load_total_steps(initial_state_db_path: Path, seed: int) -> int:
    with sqlite3.connect(initial_state_db_path) as con:
        row = con.execute(
            "SELECT MAX(step) FROM trajectories WHERE seed = ?",
            (int(seed),),
        ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"Could not determine total steps for seed {seed}")
    return int(row[0])


def load_excitation_power(run_dir: Path, manifest: dict[str, Any]) -> float:
    summary_path = run_dir / "npt_run_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        if "excitation_power_w_cm2" in summary:
            return float(summary["excitation_power_w_cm2"])

    if "excitation_power_w_cm2" in manifest:
        return float(manifest["excitation_power_w_cm2"])

    raise ValueError(f"Could not determine excitation power for {run_dir}")


def format_power_label(power_w_cm2: float) -> str:
    power_w_cm2 = float(power_w_cm2)
    if power_w_cm2 >= 1000.0:
        return f"{int(math.floor(power_w_cm2 / 1000.0 + 0.5))}k"
    if math.isclose(power_w_cm2, round(power_w_cm2)):
        return f"{int(round(power_w_cm2))}"
    return f"{power_w_cm2:g}"


def visual_mode_output_suffix(visual_mode: str, show_n3: bool = False) -> str:
    visual_mode = str(visual_mode)
    if visual_mode == STATE_VISUAL_N2_ONLY:
        return ""
    suffix = f"_{visual_mode.replace('-', '')}"
    if bool(show_n3):
        suffix += "_withn3"
    return suffix


def default_snapshot_output_path(
    run_dir: Path,
    power_label: str,
    projection: str,
    visual_mode: str = STATE_VISUAL_N2_ONLY,
    show_n3: bool = False,
) -> Path:
    suffix = visual_mode_output_suffix(visual_mode, show_n3=show_n3)
    return run_dir / f"{power_label}_{projection_output_slug(projection)}{suffix}.png"


def default_animation_output_path(
    run_dir: Path,
    power_label: str,
    projection: str,
    visual_mode: str = STATE_VISUAL_N2_ONLY,
    show_n3: bool = False,
) -> Path:
    suffix = visual_mode_output_suffix(visual_mode, show_n3=show_n3)
    return run_dir / f"{power_label}_{projection_output_slug(projection)}{suffix}.mp4"


def normalize_projection_name(projection: str) -> str:
    projection = str(projection)
    if projection in ("iso_section", "iso_cross_section"):
        return "slice"
    return projection


def projection_output_slug(projection: str) -> str:
    return normalize_projection_name(projection)


def apply_interaction(
    site_states: np.ndarray,
    interaction: dict[str, Any],
    site_id_1: int,
    site_id_2: int,
    row_seed: int,
    step: int,
) -> list[tuple[int, int, int]]:
    changes: list[tuple[int, int, int]] = []

    current_state_1 = int(site_states[site_id_1])
    expected_state_1 = int(interaction["left_state_1"])
    if current_state_1 != expected_state_1:
        raise ValueError(
            f"Replay mismatch at seed {row_seed} step {step} site {site_id_1}: "
            f"expected {expected_state_1}, found {current_state_1}"
        )
    next_state_1 = int(interaction["right_state_1"])
    site_states[site_id_1] = next_state_1
    changes.append((site_id_1, current_state_1, next_state_1))

    if int(interaction["number_of_sites"]) == 2:
        current_state_2 = int(site_states[site_id_2])
        expected_state_2 = int(interaction["left_state_2"])
        if current_state_2 != expected_state_2:
            raise ValueError(
                f"Replay mismatch at seed {row_seed} step {step} site {site_id_2}: "
                f"expected {expected_state_2}, found {current_state_2}"
            )
        next_state_2 = int(interaction["right_state_2"])
        site_states[site_id_2] = next_state_2
        changes.append((site_id_2, current_state_2, next_state_2))

    return changes


def project_positions_with_depth(positions: np.ndarray, projection: str) -> tuple[np.ndarray, np.ndarray]:
    projection = normalize_projection_name(projection)
    if projection == "xy":
        positions = np.asarray(positions, dtype=float)
        return positions[:, [0, 1]], positions[:, 2]
    if projection == "xz":
        positions = np.asarray(positions, dtype=float)
        return positions[:, [0, 2]], positions[:, 1]
    if projection == "yz":
        positions = np.asarray(positions, dtype=float)
        return positions[:, [1, 2]], positions[:, 0]
    if projection in ("iso", "slice"):
        theta = math.radians(45.0)
        phi = math.radians(35.264389682754654)

        rot_z = np.asarray(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        rot_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(phi), -math.sin(phi)],
                [0.0, math.sin(phi), math.cos(phi)],
            ],
            dtype=float,
        )
        rotated = np.asarray(positions, dtype=float) @ rot_z.T @ rot_x.T
        return rotated[:, :2], rotated[:, 2]
    raise ValueError(f"Unknown projection: {projection}")


def project_positions(positions: np.ndarray, projection: str) -> np.ndarray:
    projected_positions, _ = project_positions_with_depth(positions, projection)
    return projected_positions


def projection_axis_labels(projection: str) -> tuple[str, str]:
    projection = normalize_projection_name(projection)
    if projection == "iso":
        return "u (nm)", "v (nm)"
    if projection == "slice":
        return "x (nm)", "y (nm)"
    if projection == "xy":
        return "x (nm)", "y (nm)"
    if projection == "xz":
        return "x (nm)", "z (nm)"
    if projection == "yz":
        return "y (nm)", "z (nm)"
    return "x (nm)", "y (nm)"


def build_iso_section_mask(depth_values: np.ndarray, depth_center: float) -> np.ndarray:
    """Keep a 2 nm-thick slab centered on the isometric depth axis."""
    depths = np.asarray(depth_values, dtype=float)
    if depths.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.abs(depths - float(depth_center)) <= float(ISO_SECTION_HALF_THICKNESS_NM)


def build_display_mask(
    positions: np.ndarray,
    species_ids: np.ndarray,
    projection: str,
    all_sites: bool,
) -> np.ndarray:
    projection = normalize_projection_name(projection)
    if all_sites:
        base_mask = np.ones_like(species_ids, dtype=bool)
    else:
        base_mask = np.asarray(species_ids, dtype=np.int32) == 0

    if projection != "slice":
        return base_mask

    _projected_positions, depth_values = project_positions_with_depth(positions, projection)
    visible_depth_values = depth_values[base_mask]
    if visible_depth_values.size == 0:
        return base_mask
    depth_center = 0.5 * (float(np.min(visible_depth_values)) + float(np.max(visible_depth_values)))
    return base_mask & build_iso_section_mask(depth_values, depth_center)


def normalize_depth(depths: np.ndarray) -> np.ndarray:
    depths = np.asarray(depths, dtype=float)
    if depths.size == 0:
        return depths
    depth_min = float(np.min(depths))
    depth_max = float(np.max(depths))
    if math.isclose(depth_min, depth_max):
        return np.full(depths.shape, 0.5, dtype=float)
    return (depths - depth_min) / (depth_max - depth_min)


def replay_snapshot(
    run_dir: Path,
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    seed: int,
    mode: str,
    time_fraction: float,
    snapshot_step: int,
    projection_mask: np.ndarray,
    total_time: float,
) -> dict[str, Any]:
    site_count = load_site_count(np_db_path)
    site_states = np.zeros(site_count, dtype=np.int8)

    rows = iter_trajectory_rows(initial_state_db_path, seed=seed)
    display_count = int(np.count_nonzero(projection_mask))
    display_level_counts = empty_level_counts()

    last_time = 0.0
    last_step = 0

    best_state = site_states.copy()
    best_time = 0.0
    best_step = 0
    best_n2_count = 0
    best_level_counts = copy_level_counts(display_level_counts)
    found_tail_candidate = False

    tail_start_time = total_time * (1.0 - SNAPSHOT_TAIL_FRACTION)
    tail_duration = max(float(total_time) - float(tail_start_time), 0.0)
    tail_fraction = min(max(float(time_fraction), 0.0), 1.0)
    target_time = (
        float(tail_start_time) + tail_duration * tail_fraction if mode == "fraction" else None
    )
    target_step = int(snapshot_step) if mode == "step" else None

    for row_seed, step, event_time, site_id_1, site_id_2, interaction_id in rows:
        row_seed = int(row_seed)
        step = int(step)
        event_time = float(event_time)
        site_id_1 = int(site_id_1)
        site_id_2 = int(site_id_2)
        interaction = interactions[int(interaction_id)]

        if mode == "fraction" and event_time > float(target_time):
            break
        if mode == "step" and step > int(target_step):
            break

        changes = apply_interaction(
            site_states=site_states,
            interaction=interaction,
            site_id_1=site_id_1,
            site_id_2=site_id_2,
            row_seed=row_seed,
            step=step,
        )

        for site_id, old_state, new_state in changes:
            if projection_mask[site_id]:
                update_level_counts(display_level_counts, old_state, new_state)

        last_time = event_time
        last_step = step

        if mode == "peak-n2" and float(event_time) >= float(tail_start_time):
            current_n2_count = int(display_level_counts["3F4"])
            if not found_tail_candidate:
                found_tail_candidate = True
                best_n2_count = current_n2_count
                best_level_counts = copy_level_counts(display_level_counts)
                best_state = site_states.copy()
                best_time = event_time
                best_step = step
            elif current_n2_count > best_n2_count:
                best_n2_count = current_n2_count
                best_level_counts = copy_level_counts(display_level_counts)
                best_state = site_states.copy()
                best_time = event_time
                best_step = step

    if mode == "peak-n2":
        if found_tail_candidate:
            snapshot_states = best_state
            snapshot_time = best_time
            snapshot_step_out = best_step
            snapshot_n2_count = best_n2_count
            snapshot_level_counts = best_level_counts
        else:
            snapshot_states = site_states
            snapshot_time = last_time
            snapshot_step_out = last_step
            snapshot_level_counts = copy_level_counts(display_level_counts)
            snapshot_n2_count = int(snapshot_level_counts["3F4"])
    else:
        snapshot_states = site_states
        snapshot_time = float(target_time) if mode == "fraction" else last_time
        snapshot_step_out = int(target_step) if mode == "step" else last_step
        snapshot_level_counts = copy_level_counts(display_level_counts)
        snapshot_n2_count = int(snapshot_level_counts["3F4"])

    if mode == "fraction":
        snapshot_time = min(max(float(target_time), float(tail_start_time)), total_time)
    if mode == "step":
        snapshot_step_out = min(int(target_step), last_step) if last_step >= 0 else int(target_step)

    return {
        "seed": int(seed),
        "mode": str(mode),
        "site_states": snapshot_states,
        "snapshot_time": float(snapshot_time),
        "snapshot_step": int(snapshot_step_out),
        "n2_count": int(snapshot_n2_count),
        "level_counts": snapshot_level_counts,
        "display_count": int(display_count),
        "n2_fraction": float(snapshot_n2_count / display_count) if display_count else 0.0,
        "last_time": float(last_time),
        "last_step": int(last_step),
    }


def state_color(state: int) -> str:
    state = int(state)
    if state == N1_LEVEL:
        return STATE_COLORS[N1_LEVEL]
    if state == N2_LEVEL:
        return STATE_COLORS[N2_LEVEL]
    return "#BFC5CC"


def build_particle_boundary(positions_2d: np.ndarray, offset_nm: float = 0.0) -> np.ndarray | None:
    positions_2d = np.asarray(positions_2d, dtype=float)
    if positions_2d.shape[0] < 3:
        return None

    center = np.mean(positions_2d, axis=0)
    radial_distances = np.linalg.norm(positions_2d - center[None, :], axis=1)
    radius = float(np.max(radial_distances)) + float(offset_nm)
    if not np.isfinite(radius) or radius <= 0.0:
        return None

    angles = np.linspace(0.0, 2.0 * math.pi, int(CIRCLE_BOUNDARY_POINTS), endpoint=True, dtype=float)
    boundary = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ]
    )
    return boundary


def draw_particle_boundary(ax: Any, positions_2d: np.ndarray, offset_nm: float = 0.0) -> np.ndarray | None:
    boundary = build_particle_boundary(positions_2d, offset_nm=offset_nm)
    if boundary is None:
        return None
    ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        color="white",
        linewidth=3.2,
        alpha=0.88,
        zorder=4.05,
    )
    ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        color="#1F2937",
        linewidth=1.45,
        alpha=0.92,
        zorder=4.1,
    )
    return boundary


def background_styles(states: np.ndarray, depth_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.int8)
    depth_norm = np.clip(np.asarray(depth_norm, dtype=float), 0.0, 1.0)
    colors = np.empty((states.size, 4), dtype=float)
    sizes = np.empty(states.size, dtype=float)
    if states.size == 0:
        return colors.reshape((0, 4)), sizes

    ground_mask = states == N1_LEVEL
    if np.any(ground_mask):
        ground_depth = depth_norm[ground_mask][:, None]
        colors[ground_mask] = GROUND_RGBA_FRONT * (1.0 - ground_depth) + GROUND_RGBA_BACK * ground_depth
        sizes[ground_mask] = GROUND_SIZE

    other_mask = ~ground_mask
    if np.any(other_mask):
        other_depth = depth_norm[other_mask][:, None]
        colors[other_mask] = OTHER_RGBA_FRONT * (1.0 - other_depth) + OTHER_RGBA_BACK * other_depth
        sizes[other_mask] = OTHER_SIZE
    return colors, sizes


def empty_level_counts() -> dict[str, int]:
    return {label: 0 for label in LEVEL_COUNT_LABELS}


def copy_level_counts(level_counts: dict[str, int]) -> dict[str, int]:
    return {label: int(level_counts.get(label, 0)) for label in LEVEL_COUNT_LABELS}


def level_count_bucket(state: int) -> str | None:
    state = int(state)
    if state == N2_LEVEL:
        return "3F4"
    if state == N3_LEVEL:
        return "3H5"
    if state == N4_LEVEL:
        return "3H4"
    if state in N5PLUS_LEVELS:
        return "3F2+3F3"
    return None


def update_level_counts(level_counts: dict[str, int], old_state: int, new_state: int) -> None:
    old_bucket = level_count_bucket(old_state)
    if old_bucket is not None:
        level_counts[old_bucket] -= 1

    new_bucket = level_count_bucket(new_state)
    if new_bucket is not None:
        level_counts[new_bucket] += 1


def format_level_count_summary(level_counts: dict[str, int]) -> str:
    counts = copy_level_counts(level_counts)
    return (
        "3F4/3H5/3H4/(3F2+3F3) = "
        f"{counts['3F4']}/{counts['3H5']}/{counts['3H4']}/{counts['3F2+3F3']}"
    )


def state_visual_specs(visual_mode: str, show_n3: bool = False) -> list[dict[str, Any]]:
    visual_mode = str(visual_mode)
    if visual_mode == STATE_VISUAL_N2_N4_N5PLUS:
        specs = [
            {
                "name": "n2",
                "states": (N2_LEVEL,),
                "face_rgba": N2_GROUPED_FACE_RGBA,
                "edge_rgba": N2_GROUPED_EDGE_RGBA,
                "size": N2_GROUPED_SIZE,
                "linewidth": 0.46,
                "zorder": 5.0,
            },
        ]
        if bool(show_n3):
            specs.append(
                {
                    "name": "n3",
                    "states": (N3_LEVEL,),
                    "face_rgba": N3_OPTIONAL_FACE_RGBA,
                    "edge_rgba": N3_OPTIONAL_EDGE_RGBA,
                    "size": N3_OPTIONAL_SIZE,
                    "linewidth": 0.45,
                    "zorder": 5.05,
                }
            )
        specs.extend(
            [
                {
                "name": "n4",
                "states": (N4_LEVEL,),
                "face_rgba": N4_GROUPED_FACE_RGBA,
                "edge_rgba": N4_GROUPED_EDGE_RGBA,
                "size": N4_GROUPED_SIZE,
                "linewidth": 0.44,
                "zorder": 5.1,
            },
            {
                "name": "n5plus",
                "states": N5PLUS_LEVELS,
                "face_rgba": N5PLUS_FACE_RGBA,
                "edge_rgba": N5PLUS_EDGE_RGBA,
                "size": N5PLUS_SIZE,
                "linewidth": 0.40,
                "zorder": 5.2,
            },
            ]
        )
        return specs
    return [
        {
            "name": "n2",
            "states": (N2_LEVEL,),
            "face_rgba": N2_FACE_RGBA,
            "edge_rgba": N2_EDGE_RGBA,
            "size": N2_SIZE,
            "linewidth": 0.36,
            "zorder": 5.0,
        }
    ]


def highlight_mask_for_spec(states: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    states = np.asarray(states, dtype=np.int16)
    if "states" in spec:
        return np.isin(states, np.asarray(spec["states"], dtype=np.int16))
    if "min_state" in spec:
        return states >= int(spec["min_state"])
    raise ValueError(f"Invalid state visual spec: {spec}")


def combined_highlight_mask(states: np.ndarray, visual_mode: str, show_n3: bool = False) -> np.ndarray:
    mask = np.zeros(np.asarray(states).shape, dtype=bool)
    for spec in state_visual_specs(visual_mode, show_n3=show_n3):
        mask |= highlight_mask_for_spec(states, spec)
    return mask


def background_mask(states: np.ndarray, visual_mode: str, show_n3: bool = False) -> np.ndarray:
    states = np.asarray(states)
    if str(visual_mode) == STATE_VISUAL_N2_N4_N5PLUS:
        return np.ones(states.shape, dtype=bool)
    return ~combined_highlight_mask(states, visual_mode, show_n3=show_n3)


def state_heatmap_enabled(visual_mode: str) -> bool:
    return str(visual_mode) == STATE_VISUAL_N2_ONLY


def axes_face_color(visual_mode: str) -> str:
    if str(visual_mode) == STATE_VISUAL_N2_N4_N5PLUS:
        return GROUPED_AXES_FACE_COLOR
    return DEFAULT_AXES_FACE_COLOR


def make_heatmap(
    n2_positions: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    bins: int,
    sigma: float,
) -> np.ndarray:
    """Build a normalized, Gaussian-smoothed 2D occupancy map for n2 sites.

    The values are computed by binning the projected n2 coordinates onto a
    regular grid covering the plotted domain, smoothing the binned counts with
    a Gaussian kernel whose width is `sigma` grid cells, and normalizing the
    result so the maximum value is 1.0.
    """
    n2_positions = np.asarray(n2_positions, dtype=float)
    x_vals = n2_positions[:, 0] if n2_positions.size else np.asarray([], dtype=float)
    y_vals = n2_positions[:, 1] if n2_positions.size else np.asarray([], dtype=float)
    hist, _, _ = np.histogram2d(
        x_vals,
        y_vals,
        bins=max(20, int(bins)),
        range=[[x0, x1], [y0, y1]],
    )
    heat = gaussian_filter(hist.T, sigma=float(sigma))
    peak = float(np.max(heat)) if heat.size else 0.0
    if peak > 0:
        heat = heat / peak
    return heat


def format_time_natural(time_seconds: float) -> str:
    time_seconds = float(time_seconds)
    abs_time = abs(time_seconds)
    if math.isclose(abs_time, 0.0):
        return "0 s"
    if abs_time >= 1.0:
        return f"{time_seconds:.6f} s"
    if abs_time >= 1e-3:
        return f"{time_seconds * 1e3:.3f} ms"
    if abs_time >= 1e-6:
        return f"{time_seconds * 1e6:.3f} us"
    if abs_time >= 1e-9:
        return f"{time_seconds * 1e9:.3f} ns"
    return f"{time_seconds * 1e12:.3f} ps"


def format_status_line(
    frame_index: int,
    frame_total: int,
    time_seconds: float,
    level_counts: dict[str, int],
) -> str:
    return (
        f"frame {int(frame_index)}/{int(frame_total)} | "
        f"t = {format_time_natural(time_seconds)} | "
        f"{format_level_count_summary(level_counts)}"
    )


def add_vertical_colorbar(fig: Any, ax: Any, mappable: Any) -> Any:
    bbox = ax.get_position()
    cbar_width = 0.024
    cbar_pad = 0.032
    cax = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_width, bbox.height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.set_ticks([])
    cbar.ax.text(
        1.85,
        0.5,
        "Normalized n2 Occupancy",
        rotation=270,
        ha="center",
        va="center",
        transform=cbar.ax.transAxes,
        fontsize=15,
    )
    cbar.ax.text(
        0.5,
        1.02,
        "highest PA intensity",
        ha="center",
        va="bottom",
        transform=cbar.ax.transAxes,
        fontsize=12,
    )
    cbar.ax.text(
        0.5,
        -0.04,
        "no PA activity",
        ha="center",
        va="top",
        transform=cbar.ax.transAxes,
        fontsize=12,
    )
    cbar.ax.text(
        1.18,
        1.0,
        "1.0",
        ha="left",
        va="center",
        transform=cbar.ax.transAxes,
        fontsize=9,
    )
    cbar.ax.text(
        1.18,
        0.0,
        "0.0",
        ha="left",
        va="center",
        transform=cbar.ax.transAxes,
        fontsize=9,
    )
    return cbar


def build_animation_frames(
    initial_state_db_path: Path,
    np_db_path: Path,
    interactions: dict[int, dict[str, Any]],
    seed: int,
    projection_mask: np.ndarray,
    total_steps: int,
    step_span: int,
    step_interval: int,
    start_fraction: float = 0.5,
    use_full_trajectory: bool = False,
) -> dict[str, Any]:
    site_count = load_site_count(np_db_path)
    site_states = np.zeros(site_count, dtype=np.int8)
    display_count = int(np.count_nonzero(projection_mask))
    display_level_counts = empty_level_counts()
    if step_span <= 0 and not use_full_trajectory:
        raise ValueError("Animation step span must be positive")
    if step_interval <= 0:
        raise ValueError("Animation step interval must be positive")
    final_step = int(total_steps)
    if use_full_trajectory:
        start_step = 0
        end_step = final_step
    else:
        start_fraction = min(max(float(start_fraction), 0.0), 1.0)
        start_step = min(final_step, int(round(float(final_step) * start_fraction)))
        end_step = min(final_step, int(step_span))
        end_step = min(final_step, start_step + int(step_span))
    target_steps = list(range(start_step, end_step + 1, int(step_interval)))
    if not target_steps or target_steps[-1] != end_step:
        target_steps.append(end_step)

    rows = iter_trajectory_rows(initial_state_db_path, seed=seed)
    current_row = next(rows, None)
    last_step = -1
    last_time = 0.0
    frames: list[dict[str, Any]] = []

    for frame_index, target_step in enumerate(target_steps):
        while current_row is not None and int(current_row[1]) <= int(target_step):
            row_seed, step, event_time, site_id_1, site_id_2, interaction_id = current_row
            row_seed = int(row_seed)
            step = int(step)
            event_time = float(event_time)
            site_id_1 = int(site_id_1)
            site_id_2 = int(site_id_2)
            interaction = interactions[int(interaction_id)]

            changes = apply_interaction(
                site_states=site_states,
                interaction=interaction,
                site_id_1=site_id_1,
                site_id_2=site_id_2,
                row_seed=row_seed,
                step=step,
            )
            for site_id, old_state, new_state in changes:
                if projection_mask[site_id]:
                    update_level_counts(display_level_counts, old_state, new_state)

            last_step = step
            last_time = event_time
            current_row = next(rows, None)

        frame_level_counts = copy_level_counts(display_level_counts)
        frames.append(
            {
                "time": float(last_time),
                "step": int(target_step),
                "site_states": site_states.copy(),
                "n2_count": int(frame_level_counts["3F4"]),
                "level_counts": frame_level_counts,
                "display_count": int(display_count),
                "n2_fraction": float(frame_level_counts["3F4"] / display_count) if display_count else 0.0,
                "frame_index": int(frame_index),
                "frame_total": int(len(target_steps)),
                "last_event_time": float(last_time),
            }
        )

    return {
        "seed": int(seed),
        "frames": frames,
        "frame_count": int(len(frames)),
        "display_count": int(display_count),
        "total_steps": int(total_steps),
        "step_span": int(step_span),
        "step_interval": int(step_interval),
        "start_step": int(start_step),
        "final_step": int(end_step),
        "start_fraction": float(start_fraction if not use_full_trajectory else 0.0),
        "use_full_trajectory": bool(use_full_trajectory),
        "total_time": float(last_time),
    }


def render_snapshot(
    output_path: Path,
    positions: np.ndarray,
    species_ids: np.ndarray,
    snapshot: dict[str, Any],
    projection: str,
    display_mask: np.ndarray,
    visual_mode: str,
    show_n3: bool,
    bins: int,
    sigma: float,
    heatmap_alpha: float,
    dpi: int,
) -> None:
    projected_positions, depth_values = project_positions_with_depth(positions, projection)
    display_mask = np.asarray(display_mask, dtype=bool)

    plot_positions = projected_positions[display_mask]
    plot_states = np.asarray(snapshot["site_states"], dtype=np.int8)[display_mask]
    plot_depths = normalize_depth(depth_values[display_mask])
    sort_order = np.argsort(plot_depths)[::-1]
    plot_positions = plot_positions[sort_order]
    plot_states = plot_states[sort_order]
    plot_depths = plot_depths[sort_order]

    n2_mask = plot_states == N2_LEVEL
    n2_positions = plot_positions[n2_mask]

    core_mask = display_mask & (np.asarray(species_ids, dtype=np.int32) == 0)
    core_boundary = build_particle_boundary(projected_positions[core_mask], offset_nm=BOUNDARY_OFFSET_NM)

    extent_positions = plot_positions
    if core_boundary is not None:
        extent_positions = np.vstack([extent_positions, core_boundary])
    xmin, ymin = np.min(extent_positions, axis=0)
    xmax, ymax = np.max(extent_positions, axis=0)
    xpad = max(0.03 * (xmax - xmin), 0.25)
    ypad = max(0.03 * (ymax - ymin), 0.25)
    x0, x1 = xmin - xpad, xmax + xpad
    y0, y1 = ymin - ypad, ymax + ypad

    fig, ax = plt.subplots(figsize=(8.7, 8.1), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(axes_face_color(visual_mode))
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.08, top=0.88)

    if state_heatmap_enabled(visual_mode) and n2_positions.size:
        hist, xedges, yedges = np.histogram2d(
            n2_positions[:, 0],
            n2_positions[:, 1],
            bins=max(20, int(bins)),
            range=[[x0, x1], [y0, y1]],
        )
        hist = gaussian_filter(hist.T, sigma=float(sigma))
        if np.max(hist) > 0:
            heat = hist / float(np.max(hist))
            im = ax.imshow(
                heat,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
                alpha=float(heatmap_alpha),
                interpolation="bilinear",
                zorder=1,
            )
            # add_vertical_colorbar(fig, ax, im)

    draw_particle_boundary(ax, projected_positions[core_mask], offset_nm=BOUNDARY_OFFSET_NM)

    bg_mask = background_mask(plot_states, visual_mode, show_n3=show_n3)
    bg_positions = plot_positions[bg_mask]
    bg_colors, bg_sizes = background_styles(plot_states[bg_mask], plot_depths[bg_mask])
    ax.scatter(
        bg_positions[:, 0],
        bg_positions[:, 1],
        c=bg_colors,
        s=bg_sizes,
        linewidths=0.0,
        zorder=2,
    )

    for spec in state_visual_specs(visual_mode, show_n3=show_n3):
        spec_mask = highlight_mask_for_spec(plot_states, spec)
        if not np.any(spec_mask):
            continue
        spec_positions = plot_positions[spec_mask]
        spec_depths = plot_depths[spec_mask]
        spec_order = np.argsort(spec_depths)[::-1]
        spec_positions = spec_positions[spec_order]
        spec_count = int(spec_positions.shape[0])
        ax.scatter(
            spec_positions[:, 0],
            spec_positions[:, 1],
            facecolors=np.repeat(np.asarray(spec["face_rgba"], dtype=float)[None, :], spec_count, axis=0),
            s=np.full(spec_count, float(spec["size"]), dtype=float),
            edgecolors=np.repeat(np.asarray(spec["edge_rgba"], dtype=float)[None, :], spec_count, axis=0),
            linewidths=float(spec["linewidth"]),
            zorder=float(spec["zorder"]),
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=9)

    xlabel, ylabel = projection_axis_labels(projection)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    fig.text(
        0.5,
        0.958,
        format_status_line(1, 1, snapshot["snapshot_time"], snapshot["level_counts"]),
        ha="center",
        va="top",
        fontsize=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_animation(
    output_path: Path,
    positions: np.ndarray,
    species_ids: np.ndarray,
    frames: list[dict[str, Any]],
    seed: int,
    projection: str,
    display_mask: np.ndarray,
    visual_mode: str,
    show_n3: bool,
    bins: int,
    sigma: float,
    heatmap_alpha: float,
    fps: int,
    dpi: int,
    time_origin_override: float | None = None,
    progress_label: str = "render_animation",
) -> None:
    projected_positions, depth_values = project_positions_with_depth(positions, projection)
    display_mask = np.asarray(display_mask, dtype=bool)

    plot_positions = projected_positions[display_mask]
    plot_depths = normalize_depth(depth_values[display_mask])
    sort_order = np.argsort(plot_depths)[::-1]
    plot_positions = plot_positions[sort_order]
    plot_depths = plot_depths[sort_order]

    core_mask = display_mask & (np.asarray(species_ids, dtype=np.int32) == 0)
    core_boundary = build_particle_boundary(projected_positions[core_mask], offset_nm=BOUNDARY_OFFSET_NM)

    extent_positions = plot_positions
    if core_boundary is not None:
        extent_positions = np.vstack([extent_positions, core_boundary])
    xmin, ymin = np.min(extent_positions, axis=0)
    xmax, ymax = np.max(extent_positions, axis=0)
    xpad = max(0.03 * (xmax - xmin), 0.25)
    ypad = max(0.03 * (ymax - ymin), 0.25)
    x0, x1 = xmin - xpad, xmax + xpad
    y0, y1 = ymin - ypad, ymax + ypad

    fig, ax = plt.subplots(figsize=(8.7, 8.1), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(axes_face_color(visual_mode))
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.08, top=0.88)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=12)

    xlabel, ylabel = projection_axis_labels(projection)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    draw_particle_boundary(ax, projected_positions[core_mask], offset_nm=BOUNDARY_OFFSET_NM)

    first_states = np.asarray(frames[0]["site_states"], dtype=np.int8)[display_mask][sort_order]
    first_n2_mask = first_states == N2_LEVEL
    first_bg_mask = background_mask(first_states, visual_mode, show_n3=show_n3)
    heat_im = ax.imshow(
        make_heatmap(plot_positions[first_n2_mask], x0, x1, y0, y1, bins=bins, sigma=sigma),
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        alpha=float(heatmap_alpha) if state_heatmap_enabled(visual_mode) else 0.0,
        interpolation="bilinear",
        zorder=1,
    )
    # add_vertical_colorbar(fig, ax, heat_im)

    first_bg_colors, first_bg_sizes = background_styles(first_states[first_bg_mask], plot_depths[first_bg_mask])
    bg_scatter = ax.scatter(
        plot_positions[first_bg_mask, 0],
        plot_positions[first_bg_mask, 1],
        c=first_bg_colors,
        s=first_bg_sizes,
        linewidths=0.0,
        zorder=2,
    )
    highlight_scatters: list[Any] = []
    for spec in state_visual_specs(visual_mode, show_n3=show_n3):
        spec_mask = highlight_mask_for_spec(first_states, spec)
        spec_count = int(np.count_nonzero(spec_mask))
        scatter = ax.scatter(
            plot_positions[spec_mask, 0],
            plot_positions[spec_mask, 1],
            facecolors=np.repeat(np.asarray(spec["face_rgba"], dtype=float)[None, :], spec_count, axis=0),
            s=np.full(spec_count, float(spec["size"]), dtype=float),
            edgecolors=np.repeat(np.asarray(spec["edge_rgba"], dtype=float)[None, :], spec_count, axis=0),
            linewidths=float(spec["linewidth"]),
            zorder=float(spec["zorder"]),
        )
        highlight_scatters.append(scatter)

    status_text = fig.text(0.5, 0.958, "", ha="center", va="top", fontsize=15)
    time_origin = float(time_origin_override) if time_origin_override is not None else (float(frames[0]["time"]) if frames else 0.0)
    total_frames = int(len(frames))
    progress_start = time.perf_counter()
    progress_every = max(100, total_frames // 100) if total_frames > 0 else 100

    def update(frame_index: int):
        if total_frames > 0 and (
            frame_index == 0
            or frame_index + 1 == total_frames
            or ((frame_index + 1) % progress_every == 0)
        ):
            elapsed = time.perf_counter() - progress_start
            percent = 100.0 * float(frame_index + 1) / float(total_frames)
            print(
                f"[{progress_label}] frame {frame_index + 1}/{total_frames} "
                f"({percent:.1f}%) elapsed={elapsed:.1f}s",
                flush=True,
            )
        frame = frames[frame_index]
        states = np.asarray(frame["site_states"], dtype=np.int8)[display_mask][sort_order]

        n2_mask = states == N2_LEVEL
        bg_mask = background_mask(states, visual_mode, show_n3=show_n3)
        bg_positions = plot_positions[bg_mask]
        bg_states = states[bg_mask]
        bg_colors, bg_sizes = background_styles(bg_states, plot_depths[bg_mask])
        if bg_positions.size:
            bg_scatter.set_offsets(bg_positions)
            bg_scatter.set_facecolors(bg_colors)
            bg_scatter.set_sizes(bg_sizes)
        else:
            bg_scatter.set_offsets(np.zeros((0, 2), dtype=float))
            bg_scatter.set_facecolors(np.zeros((0, 4), dtype=float))
            bg_scatter.set_sizes(np.zeros(0, dtype=float))

        n2_positions = plot_positions[n2_mask]
        for spec, scatter in zip(state_visual_specs(visual_mode, show_n3=show_n3), highlight_scatters):
            spec_mask = highlight_mask_for_spec(states, spec)
            spec_positions = plot_positions[spec_mask]
            spec_count = int(spec_positions.shape[0])
            if spec_positions.size:
                scatter.set_offsets(spec_positions)
                scatter.set_facecolors(
                    np.repeat(np.asarray(spec["face_rgba"], dtype=float)[None, :], spec_count, axis=0)
                )
                scatter.set_edgecolors(
                    np.repeat(np.asarray(spec["edge_rgba"], dtype=float)[None, :], spec_count, axis=0)
                )
                scatter.set_sizes(np.full(spec_count, float(spec["size"]), dtype=float))
            else:
                scatter.set_offsets(np.zeros((0, 2), dtype=float))
                scatter.set_facecolors(np.zeros((0, 4), dtype=float))
                scatter.set_edgecolors(np.zeros((0, 4), dtype=float))
                scatter.set_sizes(np.zeros(0, dtype=float))

        if state_heatmap_enabled(visual_mode):
            heat_im.set_data(make_heatmap(n2_positions, x0, x1, y0, y1, bins=bins, sigma=sigma))
            heat_im.set_alpha(float(heatmap_alpha))
        else:
            heat_im.set_data(np.zeros_like(np.asarray(heat_im.get_array())))
            heat_im.set_alpha(0.0)
        status_text.set_text(
            format_status_line(
                int(frame["frame_index"]) + 1,
                int(frame["frame_total"]),
                max(float(frame["time"]) - time_origin, 0.0),
                dict(frame["level_counts"]),
            )
        )
        return (bg_scatter, *highlight_scatters, heat_im, status_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False, interval=1000 / max(int(fps), 1))
    try:
        writer = animation.FFMpegWriter(
            fps=max(int(fps), 1),
            codec="libx264",
            bitrate=2400,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    except RuntimeError as exc:
        plt.close(fig)
        raise RuntimeError(
            "FFmpeg is required for MP4 animation export. Install ffmpeg and rerun the command."
        ) from exc
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def split_frames_into_segments(frames: list[dict[str, Any]], segment_count: int) -> list[list[dict[str, Any]]]:
    if segment_count <= 1 or len(frames) <= 1:
        return [frames]
    segment_count = max(1, min(int(segment_count), len(frames)))
    boundaries = np.linspace(0, len(frames), segment_count + 1, dtype=int)
    segments: list[list[dict[str, Any]]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append(frames[start:end])
    return segments or [frames]


def render_animation_segment_worker(
    *,
    output_path: str,
    positions: np.ndarray,
    species_ids: np.ndarray,
    frames: list[dict[str, Any]],
    seed: int,
    projection: str,
    display_mask: np.ndarray,
    visual_mode: str,
    show_n3: bool,
    bins: int,
    sigma: float,
    heatmap_alpha: float,
    fps: int,
    dpi: int,
    time_origin_override: float,
    progress_label: str,
) -> str:
    render_animation(
        output_path=Path(output_path),
        positions=positions,
        species_ids=species_ids,
        frames=frames,
        seed=seed,
        projection=projection,
        display_mask=display_mask,
        visual_mode=visual_mode,
        show_n3=show_n3,
        bins=bins,
        sigma=sigma,
        heatmap_alpha=heatmap_alpha,
        fps=fps,
        dpi=dpi,
        time_origin_override=time_origin_override,
        progress_label=progress_label,
    )
    return output_path


def concat_mp4_segments(segment_paths: list[Path], output_path: Path) -> None:
    if not segment_paths:
        raise ValueError("No MP4 segments provided for concatenation")
    concat_list_path = output_path.parent / f"{output_path.stem}_concat_list.txt"
    concat_lines = [f"file '{path.resolve().as_posix()}'" for path in segment_paths]
    concat_list_path.write_text("\n".join(concat_lines) + "\n")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg concat failed for {output_path}: {exc.stderr.strip() or exc.stdout.strip()}"
        ) from exc
    finally:
        try:
            concat_list_path.unlink()
        except OSError:
            pass


def render_animation_parallel(
    output_path: Path,
    positions: np.ndarray,
    species_ids: np.ndarray,
    frames: list[dict[str, Any]],
    seed: int,
    projection: str,
    display_mask: np.ndarray,
    visual_mode: str,
    show_n3: bool,
    bins: int,
    sigma: float,
    heatmap_alpha: float,
    fps: int,
    dpi: int,
    parallel_segments: int,
) -> None:
    segments = split_frames_into_segments(frames, parallel_segments)
    if len(segments) <= 1:
        render_animation(
            output_path=output_path,
            positions=positions,
            species_ids=species_ids,
            frames=frames,
            seed=seed,
            projection=projection,
            display_mask=display_mask,
            visual_mode=visual_mode,
            show_n3=show_n3,
            bins=bins,
            sigma=sigma,
            heatmap_alpha=heatmap_alpha,
            fps=fps,
            dpi=dpi,
        )
        return

    time_origin = float(frames[0]["time"]) if frames else 0.0
    segment_dir = output_path.parent / f"{output_path.stem}_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_paths = [segment_dir / f"{output_path.stem}_part_{idx:02d}.mp4" for idx in range(len(segments))]

    with ProcessPoolExecutor(max_workers=len(segments)) as executor:
        futures = []
        for idx, (segment_frames, segment_path) in enumerate(zip(segments, segment_paths), start=1):
            futures.append(
                executor.submit(
                    render_animation_segment_worker,
                    output_path=str(segment_path),
                    positions=positions,
                    species_ids=species_ids,
                    frames=segment_frames,
                    seed=seed,
                    projection=projection,
                    display_mask=display_mask,
                    visual_mode=visual_mode,
                    show_n3=show_n3,
                    bins=bins,
                    sigma=sigma,
                    heatmap_alpha=heatmap_alpha,
                    fps=fps,
                    dpi=dpi,
                    time_origin_override=time_origin,
                    progress_label=f"segment {idx}/{len(segments)}",
                )
            )
        for future in as_completed(futures):
            future.result()

    concat_mp4_segments(segment_paths, output_path)

    for segment_path in segment_paths:
        try:
            segment_path.unlink()
        except OSError:
            pass
    try:
        segment_dir.rmdir()
    except OSError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one trajectory snapshot as an EDS-like n2 map."
    )
    parser.add_argument("run_dir", help="Path to the run directory, for example baseline/power_06_10000")
    parser.add_argument(
        "--seed",
        default="auto",
        help="Trajectory seed to plot, or auto to use trajectory_3d_overview_summary.json when present.",
    )
    parser.add_argument(
        "--mode",
        choices=("peak-n2", "fraction", "step", "final"),
        default="peak-n2",
        help="How to choose the snapshot from the trajectory. peak-n2 and fraction are restricted to the last 30% of the trajectory.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Render an MP4 across a fixed step window instead of a single snapshot.",
    )
    parser.add_argument(
        "--step-span",
        type=int,
        default=None,
        help="Total trajectory step span to include for --animate, starting from step 0. Overrides --video-duration-min when provided.",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=10,
        help="Step interval between adjacent animation frames for --animate.",
    )
    parser.add_argument(
        "--video-duration-min",
        type=float,
        default=5.0,
        help="Target animation duration in minutes for --animate when --step-span is not provided. Default: 5 min.",
    )
    parser.add_argument(
        "--animation-start-fraction",
        type=float,
        default=0.0,
        help="Start the animation window at this fraction of the selected seed trajectory when not using --full-trajectory-animation. Default: 0.5.",
    )
    parser.add_argument(
        "--full-trajectory-animation",
        action="store_true",
        help="Use the full trajectory from step 0 to the final step for --animate.",
    )
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for --animate.")
    parser.add_argument(
        "--parallel-segments",
        type=int,
        default=24,
        help="Number of MP4 segments to render in parallel for --animate. Use 1 to disable parallel rendering. Default: 8.",
    )
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=0.55,
        help="Used only with --mode fraction. Pick the state at this fraction of the last 30% of the selected seed's total time.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Used only with --mode step. Pick the state after this KMC step.",
    )
    parser.add_argument(
        "--projection",
        choices=("xy", "xz", "yz", "iso", "slice", "iso_section", "iso_cross_section"),
        default="xy",
        help="2D projection for the spatial map. xy is the most EDS-like. slice keeps only sites within 1 nm of the central isometric cut plane.",
    )
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="Plot all sites instead of only Tm sites (species_id = 0).",
    )
    parser.add_argument(
        "--state-visual-mode",
        choices=(STATE_VISUAL_N2_ONLY, STATE_VISUAL_N2_N4_N5PLUS),
        default=STATE_VISUAL_N2_ONLY,
        help="Rendering style: keep the original n2-only view or group states into 3F4 (yellow), 3H4 (red), and 3F2+3F3 (purple) highlight layers.",
    )
    parser.add_argument(
        "--show-n3",
        action="store_true",
        help="Only for the grouped visual mode: highlight 3H5 (n3) in orange instead of leaving it in the background layer.",
    )
    parser.add_argument("--bins", type=int, default=180, help="Histogram bins for the n2 heatmap.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur for the heatmap.")
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.68,
        help="Opacity for the smoothed n2 intensity layer.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <run_dir>/<power>_<projection>.png, for example 15k_xy.png.",
    )
    parser.add_argument(
        "--animation-output",
        default=None,
        help="Output MP4 path for --animate. Defaults to <run_dir>/<power>_<projection>.mp4, for example 15k_xy.mp4.",
    )
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    np_db_path = run_dir / "np.sqlite"
    initial_state_db_path = run_dir / "initial_state.sqlite"
    manifest = load_manifest(run_dir)
    interactions = load_interactions(manifest)
    positions, species_ids = load_site_geometry(np_db_path)
    excitation_power = load_excitation_power(run_dir, manifest)
    power_label = format_power_label(excitation_power)

    if len(positions) != load_site_count(np_db_path):
        raise ValueError("Site geometry count does not match metadata.number_of_sites")

    seed = load_selected_seed(run_dir, initial_state_db_path, str(args.seed))
    total_time = load_total_time(run_dir, initial_state_db_path, seed)
    total_steps = load_total_steps(initial_state_db_path, seed)

    display_mask = build_display_mask(
        positions=positions,
        species_ids=species_ids,
        projection=str(args.projection),
        all_sites=bool(args.all_sites),
    )

    if bool(args.animate):
        if bool(args.full_trajectory_animation):
            step_span = int(total_steps)
        elif args.step_span is not None:
            step_span = int(args.step_span)
        else:
            target_frames = max(2, int(round(float(args.video_duration_min) * 60.0 * float(args.fps))))
            step_span = max(int(args.step_interval), (target_frames - 1) * int(args.step_interval))
        movie = build_animation_frames(
            initial_state_db_path=initial_state_db_path,
            np_db_path=np_db_path,
            interactions=interactions,
            seed=seed,
            projection_mask=display_mask,
            total_steps=int(total_steps),
            step_span=int(step_span),
            step_interval=int(args.step_interval),
            start_fraction=float(args.animation_start_fraction),
            use_full_trajectory=bool(args.full_trajectory_animation),
        )
        visual_mode = str(args.state_visual_mode)
        show_n3 = bool(args.show_n3)
        output_path = (
            Path(args.animation_output)
            if args.animation_output
            else default_animation_output_path(
                run_dir,
                power_label,
                str(args.projection),
                visual_mode=visual_mode,
                show_n3=show_n3,
            )
        )
        render_animation_parallel(
            output_path=output_path,
            positions=positions,
            species_ids=species_ids,
            frames=list(movie["frames"]),
            seed=seed,
            projection=str(args.projection),
            display_mask=display_mask,
            visual_mode=visual_mode,
            show_n3=show_n3,
            bins=int(args.bins),
            sigma=float(args.sigma),
            heatmap_alpha=float(args.heatmap_alpha),
            fps=int(args.fps),
            dpi=int(args.dpi),
            parallel_segments=max(1, int(args.parallel_segments)),
        )

        print(f"Wrote {output_path}")
        print(
            f"seed={seed} frames={movie['frame_count']} steps={movie['start_step']}->{movie['final_step']} "
            f"interval={movie['step_interval']} span={movie['step_span']} start_fraction={movie['start_fraction']:.3f} "
            f"state_visual_mode={visual_mode} show_n3={show_n3} "
            f"parallel_segments={max(1, int(args.parallel_segments))} "
            f"last_event_t={movie['total_time']:.6e} s "
            f"sites_shown={movie['display_count']}"
        )
    else:
        snapshot = replay_snapshot(
            run_dir=run_dir,
            initial_state_db_path=initial_state_db_path,
            np_db_path=np_db_path,
            interactions=interactions,
            seed=seed,
            mode=str(args.mode),
            time_fraction=float(args.time_fraction),
            snapshot_step=int(args.step),
            projection_mask=display_mask,
            total_time=float(total_time),
        )

        visual_mode = str(args.state_visual_mode)
        show_n3 = bool(args.show_n3)
        output_path = (
            Path(args.output)
            if args.output
            else default_snapshot_output_path(
                run_dir,
                power_label,
                str(args.projection),
                visual_mode=visual_mode,
                show_n3=show_n3,
            )
        )
        render_snapshot(
            output_path=output_path,
            positions=positions,
            species_ids=species_ids,
            snapshot=snapshot,
            projection=str(args.projection),
            display_mask=display_mask,
            visual_mode=visual_mode,
            show_n3=show_n3,
            bins=int(args.bins),
            sigma=float(args.sigma),
            heatmap_alpha=float(args.heatmap_alpha),
            dpi=int(args.dpi),
        )

        print(f"Wrote {output_path}")
        print(
            f"seed={seed} mode={args.mode} snapshot_t={snapshot['snapshot_time']:.6e} s "
            f"step={snapshot['snapshot_step']} state_visual_mode={visual_mode} show_n3={show_n3} "
            f"counts[{format_level_count_summary(snapshot['level_counts'])}] "
            f"sites_shown={snapshot['display_count']}"
        )


if __name__ == "__main__":
    main()
