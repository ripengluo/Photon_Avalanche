#!/usr/bin/env python3
"""Plot saved late-window N4 convergence diagnostics for a production sweep."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Collection

# Keep Matplotlib cache files out of the run directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DRIFT_TOLERANCE = 0.10
DEFAULT_TRACE_SAMPLES = 241


def load_records(output_root: Path) -> list[dict[str, object]]:
    """Load saved N4 late-window averages without replaying trajectory SQLite."""
    records: list[dict[str, object]] = []
    for power_dir in sorted(output_root.glob("power_*")):
        summary_path = power_dir / "npt_run_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as fh:
            summary = json.load(fh)
        try:
            power_index = int(power_dir.name.split("_")[1])
            power = float(summary["excitation_power_w_cm2"])
        except (IndexError, KeyError, ValueError):
            continue

        averages = summary.get("n4_late_window_averages", {})
        seeds: list[dict[str, float]] = []
        for seed, values in averages.items():
            a1 = float(values["A1"])
            a2 = float(values["A2"])
            drift = abs(a2 - a1) / abs(a2) if a2 else float("inf")
            seeds.append({"seed": int(seed), "a1": a1, "a2": a2, "drift": drift})
        records.append(
            {
                "power_index": power_index,
                "power": power,
                "directory": power_dir.name,
                "seeds": sorted(seeds, key=lambda row: row["seed"]),
            }
        )
    return sorted(records, key=lambda row: float(row["power"]))


def format_power(power: float) -> str:
    return f"{power / 1000:.1f}k" if power >= 1000 else f"{power:g}"


def plot_convergence(records: list[dict[str, object]], output_path: Path) -> None:
    """Write a compact N4 convergence figure from saved diagnostic values."""
    available = [record for record in records if record["seeds"]]
    if not available:
        raise ValueError("No saved n4_late_window_averages found in this sweep")

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 0.90), hspace=0.36, wspace=0.28)
    ax_n4 = fig.add_subplot(grid[0, 0])
    ax_drift = fig.add_subplot(grid[0, 1])
    ax_status = fig.add_subplot(grid[1, :])

    for record in available:
        power = float(record["power"])
        seeds = record["seeds"]
        a1 = np.asarray([float(row["a1"]) for row in seeds], dtype=float)
        a2 = np.asarray([float(row["a2"]) for row in seeds], dtype=float)
        drifts = np.asarray([float(row["drift"]) for row in seeds], dtype=float)

        ax_n4.scatter(
            np.full(a1.size, power),
            a1,
            s=28,
            facecolors="none",
            edgecolors="#2563eb",
            linewidths=1.0,
            alpha=0.85,
        )
        ax_n4.scatter(
            np.full(a2.size, power),
            a2,
            s=24,
            color="#dc2626",
            alpha=0.82,
        )
        ax_n4.plot(
            [power, power],
            [np.mean(a1), np.mean(a2)],
            color="#64748b",
            linewidth=1.1,
            alpha=0.8,
        )
        ax_n4.plot(power, np.mean(a1), marker="o", markersize=5, color="#1d4ed8")
        ax_n4.plot(power, np.mean(a2), marker="o", markersize=5, color="#b91c1c")

        pass_mask = drifts < DRIFT_TOLERANCE
        ax_drift.scatter(
            np.full(np.count_nonzero(pass_mask), power),
            drifts[pass_mask],
            s=32,
            color="#15803d",
            alpha=0.9,
        )
        ax_drift.scatter(
            np.full(np.count_nonzero(~pass_mask), power),
            drifts[~pass_mask],
            s=36,
            color="#dc2626",
            marker="x",
            linewidths=1.5,
        )

    powers = np.asarray([float(record["power"]) for record in records], dtype=float)
    overview_ticks = np.asarray([3000.0, 7000.0, 10000.0, 13000.0, 20000.0, 50000.0])
    for axis in (ax_n4, ax_drift):
        axis.set_xscale("log")
        axis.set_xlim(float(powers.min()) * 0.90, float(powers.max()) * 1.12)
        axis.set_xticks(overview_ticks)
        axis.set_xticklabels([format_power(value) for value in overview_ticks], rotation=0)
        axis.grid(True, which="major", color="#cbd5e1", linewidth=0.75, alpha=0.8)

    ax_n4.set_yscale("log")
    ax_n4.set_title("Late-window N4 population by seed")
    ax_n4.set_xlabel("Excitation power (W cm$^{-2}$)")
    ax_n4.set_ylabel("Time-averaged N4 population")
    ax_n4.plot([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#2563eb", label="A1: [0.6T, 0.8T]")
    ax_n4.plot([], [], marker="o", linestyle="none", color="#dc2626", label="A2: [0.6T, T]")
    ax_n4.legend(frameon=False, fontsize=9, loc="upper left")

    ax_drift.axhline(DRIFT_TOLERANCE, color="#0f172a", linestyle="--", linewidth=1.2)
    ax_drift.text(
        0.98,
        DRIFT_TOLERANCE,
        "10% threshold",
        transform=ax_drift.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#0f172a",
    )
    ax_drift.set_ylim(bottom=0.0)
    ax_drift.set_title("Per-seed late-window drift")
    ax_drift.set_xlabel("Excitation power (W cm$^{-2}$)")
    ax_drift.set_ylabel(r"$|A2-A1|/|A2|$")
    ax_drift.plot([], [], marker="o", linestyle="none", color="#15803d", label="passes")
    ax_drift.plot([], [], marker="x", linestyle="none", color="#dc2626", label="fails")
    ax_drift.legend(frameon=False, fontsize=9, loc="upper left")

    all_seeds = sorted(
        {int(row["seed"]) for record in available for row in record["seeds"]}
    )
    status = np.full((len(all_seeds), len(records)), np.nan)
    seed_positions = {seed: index for index, seed in enumerate(all_seeds)}
    for col, record in enumerate(records):
        for row in record["seeds"]:
            status[seed_positions[int(row["seed"])], col] = float(row["drift"]) < DRIFT_TOLERANCE

    image = ax_status.imshow(
        status,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#dc2626", "#15803d"]),
        vmin=0,
        vmax=1,
    )
    image.cmap.set_bad("#cbd5e1")
    ax_status.set_title("Convergence status by seed and power")
    ax_status.set_xlabel("Excitation power (W cm$^{-2}$)")
    ax_status.set_ylabel("Seed")
    ax_status.set_xticks(np.arange(len(records)))
    ax_status.set_xticklabels([format_power(float(record["power"])) for record in records])
    ax_status.set_yticks(np.arange(len(all_seeds)))
    ax_status.set_yticklabels([str(seed) for seed in all_seeds])
    for col, record in enumerate(records):
        if not record["seeds"]:
            ax_status.text(col, len(all_seeds) / 2, "pending", rotation=90, ha="center", va="center", fontsize=8, color="#334155")

    fig.suptitle("N4 late-window convergence status", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.01,
        "Derived from saved summary diagnostics; no trajectory SQLite replay was performed.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_n4_trajectory(
    n4_trace_by_seed: dict[int, dict[str, list[float]]],
    output_path: Path,
    *,
    excitation_power: float,
    dpi: int = 220,
) -> None:
    """Plot cumulative time-averaged N4 versus time and KMC step for one power."""
    traces = [
        (int(seed), values)
        for seed, values in sorted(n4_trace_by_seed.items())
        if len(values.get("step", [])) >= 2
    ]
    if not traces:
        raise ValueError("No sampled N4 traces were provided")

    sample_count = min(len(values["step"]) for _seed, values in traces)
    steps = np.asarray(traces[0][1]["step"][:sample_count], dtype=float)
    values = np.vstack(
        [
            np.asarray(trace["n4_running_average"][:sample_count], dtype=float)
            for _seed, trace in traces
        ]
    )
    times = np.vstack(
        [np.asarray(trace["time_s"][:sample_count], dtype=float) for _seed, trace in traces]
    )
    mean_values = np.mean(values, axis=0)
    mean_times = np.mean(times, axis=0)
    y_max = max(float(np.max(values)) * 1.08, 1.0e-12)

    fig, (ax_time, ax_step) = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=dpi)
    colors = plt.get_cmap("tab10")
    for index, (seed, _trace) in enumerate(traces):
        color = colors(index % 10)
        ax_time.plot(times[index], values[index], color=color, alpha=0.40, linewidth=0.85)
        ax_step.plot(steps / 1.0e6, values[index], color=color, alpha=0.40, linewidth=0.85)

    ax_time.plot(mean_times, mean_values, color="#111827", linewidth=1.8, label="equal-step mean")
    ax_step.plot(steps / 1.0e6, mean_values, color="#111827", linewidth=1.8, label="equal-step mean")
    for axis in (ax_time, ax_step):
        axis.set_ylim(0.0, y_max)
        axis.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.85)
        axis.set_ylabel("Cumulative time-averaged N4 per Tm ion")
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    ax_time.set_xlabel("Physical KMC time (s)")
    ax_time.set_title("N4 convergence versus physical time")
    ax_step.set_xlabel("KMC step (millions)")
    ax_step.set_title("N4 convergence versus KMC step")
    fig.suptitle(
        f"N4 trajectory | {format_power(float(excitation_power))} W cm$^{{-2}}$ | {len(traces)} seeds",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def sample_n4_trajectory(
    initial_state_db_path: Path,
    interactions: list[dict[str, Any]],
    *,
    final_step: int,
    trace_samples: int = DEFAULT_TRACE_SAMPLES,
    seeds: Collection[int] | None = None,
) -> dict[int, dict[str, list[float]]]:
    """Return bounded N4 convergence traces without sorting archived events.

    Sample times are fetched through the archived ``(seed, step)`` index. One
    sequential, read-only scan then processes only N4-changing events and
    asserts that each seed remains time-ordered in the insertion stream.
    """
    if final_step < 1 or trace_samples < 2:
        raise ValueError("N4 trace sampling requires a positive cutoff and two samples")
    tm_species_id = 0
    n4_level = 3
    n4_delta_by_id: dict[int, int] = {}
    for interaction in interactions:
        delta = 0
        if int(interaction["species_id_1"]) == tm_species_id:
            delta += int(int(interaction["right_state_1"]) == n4_level)
            delta -= int(int(interaction["left_state_1"]) == n4_level)
        if (
            int(interaction["number_of_sites"]) == 2
            and int(interaction["species_id_2"]) == tm_species_id
        ):
            delta += int(int(interaction["right_state_2"]) == n4_level)
            delta -= int(int(interaction["left_state_2"]) == n4_level)
        if delta:
            n4_delta_by_id[int(interaction["interaction_id"])] = delta
    if not n4_delta_by_id:
        raise ValueError("No interactions change the N4 occupancy")

    interaction_ids = sorted(n4_delta_by_id)
    n4_placeholders = ",".join("?" * len(interaction_ids))
    resolved_db_path = initial_state_db_path.resolve()
    np_db_path = initial_state_db_path.with_name("np.sqlite").resolve()
    with sqlite3.connect(np_db_path) as con:
        tm_site_ids = [
            int(site_id)
            for (site_id,) in con.execute(
                "SELECT site_id FROM sites WHERE species_id = ?", (tm_species_id,)
            )
        ]
    tm_site_count = len(tm_site_ids)
    if tm_site_count == 0:
        raise ValueError(f"No Tm sites found in {np_db_path}")

    with sqlite3.connect(resolved_db_path) as con:
        con.execute("PRAGMA query_only = ON")
        con.execute("PRAGMA cache_size = -32768")
        initial_n4_count = int(
            con.execute(
                "SELECT COUNT(*) FROM initial_state "
                "WHERE degree_of_freedom = ? AND site_id IN (%s)"
                % ",".join("?" * tm_site_count),
                (n4_level, *tm_site_ids),
            ).fetchone()[0]
        )
        checkpoint_time_by_seed = {
            int(seed): float(time_s)
            for seed, time_s in con.execute(
                "SELECT seed, time FROM interrupt_cutoff WHERE step = ?",
                (int(final_step) + 1,),
            )
        }
        trace_final_step = int(final_step)
        if not checkpoint_time_by_seed:
            terminal_row = con.execute(
                "SELECT step FROM trajectories ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if terminal_row is None:
                raise ValueError(f"No trajectory rows found in {resolved_db_path}")
            trace_final_step = min(trace_final_step, int(terminal_row[0]))
            if trace_final_step < int(final_step):
                print(
                    f"Using archived final step {trace_final_step:,} instead of "
                    f"requested {int(final_step):,} for {resolved_db_path.name}",
                    flush=True,
                )

        target_steps = np.unique(
            np.rint(np.linspace(0, trace_final_step, int(trace_samples))).astype(np.int64)
        )
        target_positions = {int(step): index for index, step in enumerate(target_steps)}
        target_placeholders = ",".join("?" * len(target_steps))
        seed_ids = sorted(
            {int(seed) for seed in seeds}
            if seeds is not None
            else checkpoint_time_by_seed.keys()
        )
        if not seed_ids:
            seed_ids = [
                int(seed)
                for (seed,) in con.execute(
                    "SELECT DISTINCT seed FROM trajectories "
                    "INDEXED BY idx_trajectories_seed_step ORDER BY seed"
                )
            ]
        if not seed_ids:
            raise ValueError(f"No trajectory seeds found in {resolved_db_path}")

        time_by_seed: dict[int, np.ndarray] = {}
        for seed in seed_ids:
            times = np.full(target_steps.size, np.nan, dtype=float)
            for step, time_s in con.execute(
                "SELECT step, time FROM trajectories "
                "WHERE seed = ? AND step IN (%s)" % target_placeholders,
                (seed, *(int(step) for step in target_steps)),
            ):
                times[target_positions[int(step)]] = float(time_s)
            time_by_seed[seed] = times

        # Older trajectory archives did not preserve interrupt_cutoff rows.
        # Their final trajectory-step time is the best stored endpoint.
        final_time_by_seed = dict(checkpoint_time_by_seed)
        for seed, times in time_by_seed.items():
            if np.isfinite(times[-1]):
                final_time_by_seed.setdefault(seed, float(times[-1]))
        if not final_time_by_seed:
            raise ValueError(
                f"No checkpoint or final-step times found in {resolved_db_path}"
            )

        for seed, final_time in final_time_by_seed.items():
            if seed in time_by_seed and not np.isfinite(time_by_seed[seed][-1]):
                time_by_seed[seed][-1] = final_time

        # Some legacy extension archives omit a contiguous range of stored
        # steps. Keep only target steps that exist for every plotted seed;
        # filling an internal gap with the final timestamp would create a false
        # backwards-time segment in the physical-time plot.
        shared_samples = np.logical_and.reduce(
            [np.isfinite(times) for times in time_by_seed.values()]
        )
        if int(np.count_nonzero(shared_samples)) < 2:
            raise ValueError(
                f"Fewer than two shared sampled trajectory steps found in {resolved_db_path}"
            )
        missing_sample_count = int(shared_samples.size - np.count_nonzero(shared_samples))
        if missing_sample_count:
            print(
                f"Skipping {missing_sample_count} absent sampled steps in "
                f"{resolved_db_path.name}",
                flush=True,
            )
            target_steps = target_steps[shared_samples]
            time_by_seed = {
                seed: times[shared_samples] for seed, times in time_by_seed.items()
            }

        state_by_seed: dict[int, dict[str, Any]] = {}
        for seed, final_time in final_time_by_seed.items():
            times = time_by_seed.get(seed)
            if times is None:
                continue
            times[0] = 0.0
            if np.any(np.diff(times) < -1e-12):
                raise ValueError(f"Sampled trajectory times decrease for seed {seed}")
            state_by_seed[seed] = {
                "times": times,
                "sample_index": 0,
                "last_time": 0.0,
                "n4_count": initial_n4_count,
                "integral": 0.0,
                "population": np.zeros(target_steps.size, dtype=float),
                "running_average": np.zeros(target_steps.size, dtype=float),
            }

        def record_samples_before(state: dict[str, Any], event_time: float) -> None:
            while (
                int(state["sample_index"]) < target_steps.size
                and float(state["times"][int(state["sample_index"])]) < event_time
            ):
                index = int(state["sample_index"])
                target_time = float(state["times"][index])
                integral = float(state["integral"]) + int(state["n4_count"]) * (
                    target_time - float(state["last_time"])
                )
                state["population"][index] = int(state["n4_count"]) / float(tm_site_count)
                state["running_average"][index] = (
                    integral / (target_time * float(tm_site_count)) if target_time > 0.0 else 0.0
                )
                state["sample_index"] = index + 1

        for seed, step, event_time, interaction_id in con.execute(
            "SELECT seed, step, time, interaction_id FROM trajectories "
            "WHERE step <= ? AND interaction_id IN (%s)" % n4_placeholders,
            (trace_final_step, *interaction_ids),
        ):
            seed = int(seed)
            if seed not in state_by_seed:
                continue
            state = state_by_seed[seed]
            event_time = float(event_time)
            if event_time < float(state["last_time"]) - 1e-12:
                raise ValueError(
                    f"N4-changing events are not time-ordered for seed {seed}; "
                    "cannot sample this unindexed trajectory safely"
                )
            record_samples_before(state, event_time)
            state["integral"] = float(state["integral"]) + int(state["n4_count"]) * (
                event_time - float(state["last_time"])
            )
            state["last_time"] = event_time
            state["n4_count"] = int(state["n4_count"]) + n4_delta_by_id[int(interaction_id)]

        traces: dict[int, dict[str, list[float]]] = {}
        for seed, state in state_by_seed.items():
            record_samples_before(state, float("inf"))
            traces[seed] = {
                "step": [float(step) for step in target_steps],
                "time_s": [float(value) for value in state["times"]],
                "n4_population": [float(value) for value in state["population"]],
                "n4_running_average": [
                    float(value) for value in state["running_average"]
                ],
            }
    return traces


def plot_n4_trajectory_sweep(
    output_root: Path,
    *,
    trace_samples: int = DEFAULT_TRACE_SAMPLES,
    dpi: int = 220,
    force: bool = False,
) -> list[Path]:
    """Scan power points and write missing or stale N4 trace plots."""

    output_paths: list[Path] = []
    for record in load_records(output_root):
        power_dir = output_root / str(record["directory"])
        summary_path = power_dir / "npt_run_summary.json"
        manifest_path = power_dir / "npt_interaction_manifest.json"
        combined_trajectory_path = power_dir / "initial_state.sqlite"
        if combined_trajectory_path.exists():
            trajectory_db_paths = [combined_trajectory_path]
        else:
            trajectory_db_paths = sorted(
                power_dir.glob("restart/seed_*/initial_state.sqlite")
            )
        if not trajectory_db_paths:
            print(f"Skipping {power_dir.name}: no trajectory databases")
            continue
        output_path = power_dir / "n4_trajectory.png"
        input_paths = (
            summary_path,
            manifest_path,
            *(path.resolve() for path in trajectory_db_paths),
        )
        if (
            not force
            and output_path.exists()
            and all(output_path.stat().st_mtime >= path.stat().st_mtime for path in input_paths)
        ):
            print(f"Keeping up-to-date N4 trace plot in {power_dir.name}", flush=True)
            output_paths.append(output_path)
            continue
        with open(summary_path) as fh:
            summary: dict[str, Any] = json.load(fh)
        final_step = summary.get("simulation_step_cutoff")
        if final_step is None:
            print(f"Skipping {power_dir.name}: no step cutoff in summary")
            continue
        with open(manifest_path) as fh:
            manifest: dict[str, Any] = json.load(fh)
        print(f"Sampling N4 trace from {power_dir.name}...", flush=True)
        requested_seeds = {
            int(seed) for seed in summary.get("n4_late_window_averages", {})
        }
        n4_trace_by_seed: dict[int, dict[str, list[float]]] = {}
        for trajectory_db_path in trajectory_db_paths:
            seeds_for_database: Collection[int] | None = requested_seeds
            if len(trajectory_db_paths) > 1:
                try:
                    seed = int(trajectory_db_path.parent.name.removeprefix("seed_"))
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot parse seed from {trajectory_db_path.parent.name}"
                    ) from exc
                if requested_seeds and seed not in requested_seeds:
                    continue
                seeds_for_database = [seed]
            n4_trace_by_seed.update(
                sample_n4_trajectory(
                    trajectory_db_path,
                    manifest["interactions"],
                    final_step=int(final_step),
                    trace_samples=trace_samples,
                    seeds=seeds_for_database,
                )
            )
        plot_n4_trajectory(
            n4_trace_by_seed,
            output_path,
            excitation_power=float(record["power"]),
            dpi=dpi,
        )
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved N4 late-window convergence diagnostics."
    )
    parser.add_argument("output_root", type=Path, help="Sweep output directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figure path; defaults to <output_root>/n4_convergence_status.png",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Write only the saved-diagnostic status plot; skip trajectory archive replays",
    )
    parser.add_argument(
        "--trace-samples",
        type=int,
        default=DEFAULT_TRACE_SAMPLES,
        help="Bounded samples per seed in each N4 trajectory trace",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate trajectory plots even when their inputs have not changed",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_path = args.output or output_root / "n4_convergence_status.png"
    records = load_records(output_root)
    plot_convergence(records, output_path)
    available = sum(bool(record["seeds"]) for record in records)
    print(f"Wrote {output_path} using saved diagnostics for {available}/{len(records)} power points")
    if not args.status_only:
        paths = plot_n4_trajectory_sweep(
            output_root,
            trace_samples=int(args.trace_samples),
            force=bool(args.force),
        )
        print(f"Wrote {len(paths)} per-power N4 trajectory plots")


if __name__ == "__main__":
    main()
