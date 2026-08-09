#!/usr/bin/env python3
"""Fast convergence diagnostic from a run summary JSON.

Reads the n4_late_window_averages written by tm_npt_kmc_production.py, or
replays the trajectory database to compute them if they are missing, and
judges whether the run is converged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_sqlite_temp_dir() -> None:
    """Make sure SQLite has a writable temp directory for large sorts."""
    for var in ("SQLITE_TMPDIR", "TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(var)
        if value:
            candidate = Path(value)
            if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
                return
    for fallback in (Path("/dev/shm"), Path("/tmp")):
        if fallback.is_dir() and os.access(fallback, os.W_OK | os.X_OK):
            # Set TMPDIR before sqlite3 is imported so SQLite picks it up.
            os.environ["SQLITE_TMPDIR"] = str(fallback)
            os.environ["TMPDIR"] = str(fallback)
            return


_ensure_sqlite_temp_dir()

import sqlite3

TM_SPECIES_ID = 0
N4_LEVEL = 3
Q21_CHANNEL_NAME = "Q21,24"
S12_CHANNEL_NAME = "s12,42"

def replay_trajectories(
    initial_state_db_path: Path,
    interactions: list[dict[str, Any]],
    tm_species_id: int = TM_SPECIES_ID,
    n4_level: int = N4_LEVEL,
    s12_channel_name: str = S12_CHANNEL_NAME,
    q21_channel_name: str = Q21_CHANNEL_NAME,
) -> dict[str, Any]:
    """Read event counts, final simulated times, and time-averaged n4 occupancy."""
    simulation_time: dict[int, float] = {}
    event_counts: dict[int, Counter] = defaultdict(Counter)
    n4_time_integral: dict[int, float] = defaultdict(float)
    n4_population_per_seed: dict[int, float] = {}
    n4_late_window_integrals: dict[int, tuple[float, float]] = {}
    q24_total_count = 0
    s12_total_count = 0
    q24_after_s12_same_pair_count = 0
    s12_after_q24_same_pair_count = 0

    # Resolve symlinks and use a memory journal so SQLite does not try to write
    # journal files next to the (possibly symlinked) archive database.
    resolved_initial_state_db_path = initial_state_db_path.resolve()
    np_db_path = initial_state_db_path.with_name("np.sqlite").resolve()
    with sqlite3.connect(np_db_path) as con:
        con.execute("PRAGMA journal_mode = MEMORY")
        site_rows = con.execute(
            "SELECT site_id, species_id FROM sites ORDER BY site_id"
        ).fetchall()
    site_count = len(site_rows)
    site_species = np.asarray(
        [int(species_id) for _site_id, species_id in site_rows],
        dtype=np.int8,
    )
    tm_site_count = int(np.sum(site_species == tm_species_id))

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
            if row["interaction_type"] == "ET" and row["label"] == s12_channel_name
        ),
        None,
    )
    q24_interaction_id = next(
        (
            int(row["interaction_id"])
            for row in interactions
            if row["interaction_type"] == "ET" and row["label"] == q21_channel_name
        ),
        None,
    )

    with sqlite3.connect(resolved_initial_state_db_path) as con:
        con.execute("PRAGMA journal_mode = MEMORY")
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
        seed_times: list[float] = []
        seed_n4: list[int] = []

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

            # Late-window n4 integrals for convergence diagnostics.
            # Window 1: [0.6 T, 0.8 T]; Window 2: [0.6 T, T].
            late_start_frac = 0.6
            late_mid_frac = 0.8
            t_start = late_start_frac * final_time
            t_mid = late_mid_frac * final_time
            integral_1 = 0.0
            integral_2 = 0.0
            prev = 0.0
            for t, n in zip(seed_times, seed_n4):
                seg_a = max(prev, t_start)
                seg_b_full = min(t, final_time)
                seg_b_half = min(t, t_mid)
                if seg_b_half > seg_a:
                    integral_1 += n * (seg_b_half - seg_a)
                if seg_b_full > seg_a:
                    integral_2 += n * (seg_b_full - seg_a)
                prev = t
                if prev >= final_time:
                    break
            n4_late_window_integrals[seed] = (integral_1, integral_2)
            seed_times.clear()
            seed_n4.clear()

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

            # Buffer piecewise-constant n4 trajectory for late-window integration.
            seed_times.append(event_time)
            seed_n4.append(current_n4_count)

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
            if int(site_species[site_id_1]) == tm_species_id:
                current_n4_count += int(interaction["right_state_1"] == n4_level)
                current_n4_count -= int(current_state_1 == n4_level)
            site_states[site_id_1] = interaction["right_state_1"]

            if interaction["number_of_sites"] == 2:
                current_state_2 = int(site_states[site_id_2])
                if current_state_2 != interaction["left_state_2"]:
                    raise ValueError(
                        "Trajectory replay mismatch for seed "
                        f"{seed} step {step} site {site_id_2}: "
                        f"expected state {interaction['left_state_2']}, found {current_state_2}"
                    )
                if int(site_species[site_id_2]) == tm_species_id:
                    current_n4_count += int(interaction["right_state_2"] == n4_level)
                    current_n4_count -= int(current_state_2 == n4_level)
                site_states[site_id_2] = interaction["right_state_2"]

            previous_event_by_seed[seed] = (interaction_id, pair_key)
            previous_time = event_time

        finalize_seed(current_seed, previous_time)

    return {
        "simulation_time": simulation_time,
        "event_counts": event_counts,
        "n4_time_integral": n4_time_integral,
        "n4_population_per_seed": n4_population_per_seed,
        "n4_late_window_integrals": n4_late_window_integrals,
        "total_site_count": int(site_count),
        "tm_site_count": int(tm_site_count),
        "q24_total_count": q24_total_count,
        "s12_total_count": s12_total_count,
        "q24_after_s12_same_pair_count": q24_after_s12_same_pair_count,
        "s12_after_q24_same_pair_count": s12_after_q24_same_pair_count,
    }

def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def compute_n4_late_window_averages(
    replay: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Convert n4_late_window_integrals to the averages dict used in the summary."""
    averages: dict[str, dict[str, float]] = {}
    tm_site_count = float(replay.get("tm_site_count", 1))
    for seed, (integral_1, integral_2) in replay["n4_late_window_integrals"].items():
        T = float(replay["simulation_time"].get(seed, 0.0))
        if T > 0 and tm_site_count > 0:
            A1 = float(integral_1) / (0.2 * T * tm_site_count)
            A2 = float(integral_2) / (0.4 * T * tm_site_count)
        else:
            A1 = 0.0
            A2 = 0.0
        averages[str(seed)] = {"A1": A1, "A2": A2, "T": T}
    return averages


def ensure_n4_late_window_averages(
    run_dir: Path,
    summary: dict[str, Any] | None = None,
    write_back: bool = True,
) -> dict[str, dict[str, float]]:
    """Return n4_late_window_averages from a summary, computing them if missing.

    If the averages are not present in ``npt_run_summary.json``, this function
    reads the interaction manifest and the trajectory SQLite database, replays
    the trajectories, and writes the computed averages back into the summary.
    """
    if summary is None:
        summary_path = run_dir / "npt_run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary found: {summary_path}")
        summary = load_json(summary_path)

    averages = summary.get("n4_late_window_averages")
    if averages:
        return averages

    manifest_path = run_dir / "npt_interaction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Need manifest to replay trajectories: {manifest_path}"
        )
    manifest = load_json(manifest_path)
    interactions = manifest.get("interactions")
    if not interactions:
        raise ValueError(f"No interactions found in manifest: {manifest_path}")

    initial_state_db_path = run_dir / "initial_state.sqlite"
    if not initial_state_db_path.exists():
        raise FileNotFoundError(
            f"Missing trajectory database: {initial_state_db_path}"
        )
    np_db_path = run_dir / "np.sqlite"
    if not np_db_path.exists():
        raise FileNotFoundError(f"Missing geometry database: {np_db_path}")

    replay = replay_trajectories(initial_state_db_path, interactions)
    averages = compute_n4_late_window_averages(replay)

    if write_back:
        summary_path = run_dir / "npt_run_summary.json"
        summary["n4_late_window_averages"] = averages
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    return averages


def diagnose_summary(
    summary: dict,
    drift_tol: float = 0.10,
    min_pass_frac: float = 0.75,
    averages: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Return pass/fail and per-seed drift from a npt_run_summary.json dict."""
    power_w_cm2 = summary.get("excitation_power_w_cm2")
    if averages is None:
        averages = summary.get("n4_late_window_averages", {})

    if not averages:
        return {
            "power_w_cm2": power_w_cm2,
            "seed_count": 0,
            "pass_count": 0,
            "pass_fraction": 0.0,
            "converged": False,
            "error": "No n4_late_window_averages found; could not generate them.",
            "seeds": [],
            "ensemble_stats": {},
        }

    rows = []
    for seed_str, vals in sorted(averages.items(), key=lambda kv: int(kv[0])):
        A1 = float(vals.get("A1", float("nan")))
        A2 = float(vals.get("A2", float("nan")))
        T = float(vals.get("T", float("nan")))
        drift = abs(A2 - A1) / abs(A2) if A2 != 0 else float("inf")
        rows.append(
            {
                "seed": int(seed_str),
                "A1": A1,
                "A2": A2,
                "T": T,
                "drift": drift,
                "passed": drift < drift_tol,
            }
        )

    A2_vals = np.array([r["A2"] for r in rows if np.isfinite(r["A2"])])
    pass_count = sum(r["passed"] for r in rows)
    pass_fraction = pass_count / len(rows) if rows else 0.0
    converged = pass_fraction >= min_pass_frac

    stats = {}
    if A2_vals.size > 0:
        stats = {
            "mean": float(np.mean(A2_vals)),
            "std": float(np.std(A2_vals, ddof=1)) if A2_vals.size > 1 else 0.0,
            "min": float(np.min(A2_vals)),
            "max": float(np.max(A2_vals)),
            "span_ratio": float(np.max(A2_vals) / max(np.min(A2_vals), 1e-300)),
        }

    return {
        "power_w_cm2": power_w_cm2,
        "drift_tol": drift_tol,
        "min_pass_frac": min_pass_frac,
        "seed_count": len(rows),
        "pass_count": pass_count,
        "pass_fraction": pass_fraction,
        "converged": converged,
        "ensemble_stats": stats,
        "seeds": rows,
    }


def diagnose_run_dir(
    run_dir: Path,
    drift_tol: float = 0.10,
    min_pass_frac: float = 0.75,
    ensure_averages: bool = True,
) -> dict:
    """Diagnose a single power_* directory."""
    summary_path = run_dir / "npt_run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary found: {summary_path}")
    summary = load_json(summary_path)
    if ensure_averages:
        averages = ensure_n4_late_window_averages(run_dir, summary=summary, write_back=True)
    else:
        averages = summary.get("n4_late_window_averages", {})
    result = diagnose_summary(
        summary,
        averages=averages,
        drift_tol=drift_tol,
        min_pass_frac=min_pass_frac,
    )
    result["run_dir"] = str(run_dir)
    return result


def find_unconverged_power_indices(
    output_root: Path,
    drift_tol: float = 0.10,
    min_pass_frac: float = 0.75,
) -> list[int]:
    """Return 0-based power indices that are not converged.

    Missing n4_late_window_averages are generated on the fly so the diagnostic
    can be used directly on older run folders.
    """
    indices: list[int] = []
    for power_dir in sorted(output_root.glob("power_*")):
        name = power_dir.name
        parts = name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        try:
            result = diagnose_run_dir(
                power_dir,
                drift_tol=drift_tol,
                min_pass_frac=min_pass_frac,
                ensure_averages=True,
            )
            if not result["converged"]:
                indices.append(idx)
        except Exception as exc:
            # If a finished directory cannot be diagnosed, include it so it
            # gets reprocessed rather than silently skipped.
            print(
                f"[convergence check] including {name} due to diagnostic error: {exc}",
                file=sys.stderr,
            )
            indices.append(idx)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast late-window convergence diagnostic for one kMC power point"
    )
    parser.add_argument("run_dir", type=Path, help="Path to a power_* directory")
    parser.add_argument(
        "--drift-tol", type=float, default=0.10, help="Relative drift tolerance"
    )
    parser.add_argument(
        "--min-pass-frac",
        type=float,
        default=0.75,
        help="Minimum fraction of seeds that must pass",
    )
    parser.add_argument(
        "--ensure-averages",
        action="store_true",
        default=True,
        help="Compute n4_late_window_averages from trajectories if missing (default)",
    )
    parser.add_argument(
        "--no-ensure-averages",
        action="store_false",
        dest="ensure_averages",
        help="Skip generating missing n4_late_window_averages",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON output path"
    )
    args = parser.parse_args()

    result = diagnose_run_dir(
        args.run_dir,
        drift_tol=args.drift_tol,
        min_pass_frac=args.min_pass_frac,
        ensure_averages=args.ensure_averages,
    )

    verdict = "CONVERGED" if result["converged"] else "NOT CONVERGED"
    print(f"Run: {result['run_dir']}")
    if result["power_w_cm2"] is not None:
        print(f"Power: {result['power_w_cm2']:.3e} W/cm2")
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    print(f"Windows: [0.6T, 0.8T] vs [0.6T, T]")
    print(
        f"Seeds: {result['seed_count']}  Passed: {result['pass_count']} "
        f"({result['pass_fraction']:.2%})"
    )
    print(f"Verdict: {verdict}")
    print()
    print("Ensemble A2 stats:")
    for key, value in result["ensemble_stats"].items():
        print(
            f"  {key}: {value:.4e}"
            if isinstance(value, float)
            else f"  {key}: {value}"
        )
    print()
    print("Per-seed details:")
    print(f"{'seed':>6} {'A1':>14} {'A2':>14} {'drift':>10} {'status':>8}")
    for row in result["seeds"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{row['seed']:6d} {row['A1']:14.4e} {row['A2']:14.4e} "
            f"{row['drift']:10.3f} {status:>8}"
        )

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
