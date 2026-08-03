#!/usr/bin/env python3
"""Plot per-power Tm3+ mechanism diagrams from NPT KMC run outputs.

The run folders already contain the needed direct outputs:
  - npt_interaction_manifest.json: interaction definitions and state changes
  - npt_run_summary.json: realized counts and event rates by interaction_id

This script joins those files, draws the 12 Tm energy levels, and overlays the
largest observed radiative, pump, non-radiative, and energy-transfer events.

Usage examples:
  python3 plot_mechanism_diagrams.py
  python3 plot_mechanism_diagrams.py run1
  python3 plot_mechanism_diagrams.py run1 --rate-key events_per_particle_s
  python3 plot_mechanism_diagrams.py --min-rate 1e-2 --max-et 20 --dpi 300

Outputs are written into each discovered power_* folder:
  - mechanism_diagram.png
  - mechanism_diagram_labels.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


TM_LABELS = [
    "3H6",
    "3F4",
    "3H5",
    "3H4",
    "3F3",
    "3F2",
    "1G4",
    "1D2",
    "3P0",
    "1I6",
    "3P1",
    "3P2",
]

TM_ENERGIES_CM = [
    153.0,
    5828.0,
    8396.0,
    12735.0,
    14598.0,
    15180.0,
    21352.0,
    28028.0,
    34886.0,
    35621.0,
    36603.0,
    38344.0,
]

COLORS = {
    "ET": "#3158a6",
    "EM": "#3158a6",
    "SQ": "#8e44ad",
    "Rad": "#e41a1c",
    "Pump": "#1aaf4b",
    "NR": "#9a9a9a",
}


@dataclass(frozen=True)
class Process:
    interaction_ids: tuple[int, ...]
    label: str
    kind: str
    components: tuple[tuple[int, int], ...]
    rate: float
    color_group: str


def color_group(kind: str) -> str:
    if kind in {"ET", "EM"}:
        return "ET"
    return kind


def is_surface_quench(interaction: dict, label: str, source: str) -> bool:
    return (
        bool(interaction.get("is_surface_quench"))
        or "Surface ET" in source
        or (label.startswith("SQ ") and "Surface" in label)
    )


def load_species_data() -> tuple[list[str], list[float]]:
    """Use NanoParticleTools' Tm species database when available."""
    try:
        import NanoParticleTools

        species_path = (
            Path(NanoParticleTools.__file__).resolve().parent
            / "species_data"
            / "data"
            / "Tm.json"
        )
        data = json.loads(species_path.read_text())
        labels = data.get("EnergyLevelLabels")
        energies = data.get("EnergyLevels")
        if labels and energies and len(labels) == len(energies):
            return [str(x) for x in labels], [float(x) for x in energies]
    except Exception:
        pass
    return TM_LABELS, TM_ENERGIES_CM


def discover_power_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in base_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("power_")
        and (p / "npt_interaction_manifest.json").exists()
        and (p / "npt_run_summary.json").exists()
    )


def power_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.match(r"power_(\d+)_([0-9.]+)$", path.name)
    if not match:
        return (9999, math.inf, path.name)
    return (int(match.group(1)), float(match.group(2)), path.name)


def event_rate(row: dict, rate_key: str) -> float:
    value = row.get(rate_key)
    if value is None:
        value = row.get("events_per_ion_s")
    return float(value or 0.0)


def extract_arrows(
    run_dir: Path,
    rate_key: str,
    min_rate: float,
    max_by_kind: dict[str, int],
) -> tuple[list[Process], dict]:
    manifest = json.loads((run_dir / "npt_interaction_manifest.json").read_text())
    summary = json.loads((run_dir / "npt_run_summary.json").read_text())

    interactions = {int(row["interaction_id"]): row for row in manifest["interactions"]}
    per_interaction = summary.get("per_interaction", [])
    candidates: list[Process] = []

    for row in per_interaction:
        iid = int(row["interaction_id"])
        interaction = interactions.get(iid)
        if not interaction:
            continue
        rate = event_rate(row, rate_key)
        if rate < min_rate:
            continue

        kind = str(interaction.get("interaction_type") or row.get("interaction_type"))
        label = str(row.get("label") or interaction.get("label") or iid)
        source = str(interaction.get("source") or "")
        surface_quench = is_surface_quench(interaction, label, source)
        if kind == "ET" and surface_quench:
            kind = "SQ"
        if kind == "ET" and (label.startswith("EM ") or "resonant migration" in source):
            continue
        if kind == "ET" and (
            int(interaction.get("species_id_1", -1)) != 0
            or int(interaction.get("species_id_2", -1)) != 0
        ):
            continue

        raw_components = [
            (
                int(interaction.get("species_id_1", -1)),
                int(interaction.get("left_state_1", -1)),
                int(interaction.get("right_state_1", -1)),
            )
        ]
        if int(interaction.get("number_of_sites", 1)) == 2:
            raw_components.append(
                (
                    int(interaction.get("species_id_2", -1)),
                    int(interaction.get("left_state_2", -1)),
                    int(interaction.get("right_state_2", -1)),
                )
            )

        tm_components = tuple(
            (src, dst)
            for species_id, src, dst in raw_components
            if species_id == 0 and src >= 0 and dst >= 0 and src != dst
        )
        if not tm_components:
            continue
        candidates.append(Process((iid,), label, kind, tm_components, rate, color_group(kind)))

    pump_transitions = {
        process.components[0]
        for process in candidates
        if process.kind == "Pump" and len(process.components) == 1
    }
    if pump_transitions:
        candidates = [
            process
            for process in candidates
            if not (
                process.kind == "Rad"
                and len(process.components) == 1
                and process.components[0] in pump_transitions
            )
        ]

    deduped: dict[tuple[str, tuple[tuple[int, int], ...]], Process] = {}
    for process in candidates:
        key = (process.color_group, process.components)
        current = deduped.get(key)
        if current is None:
            deduped[key] = process
            continue
        ids = tuple(sorted(set(current.interaction_ids + process.interaction_ids)))
        labels = current.label if current.rate >= process.rate else process.label
        kind = current.kind
        if current.color_group == "ET" and "ET" in {current.kind, process.kind}:
            kind = "ET"
        if current.color_group == "SQ":
            kind = "SQ"
        best = current if current.rate >= process.rate else process
        deduped[key] = Process(ids, labels, kind, best.components, max(current.rate, process.rate), best.color_group)

    candidates = list(deduped.values())

    selected: list[Process] = []
    for kind, limit in max_by_kind.items():
        rows = [a for a in candidates if a.color_group == kind]
        selected.extend(sorted(rows, key=lambda a: a.rate, reverse=True)[:limit])
    selected = list({(a.color_group, a.components): a for a in selected}.values())
    selected.sort(key=lambda a: (a.color_group, a.components, a.interaction_ids))
    return selected, summary


def line_width(rate: float, min_rate: float, max_rate: float, n_in_group: int) -> float:
    """Scale linewidth within one color group, with a lower cap for crowded groups."""
    n_in_group = max(n_in_group, 1)
    max_width = min(5.0, max(2.7, 8.5 / math.sqrt(n_in_group)))
    min_width = min(0.55, max_width * 0.16)
    if max_rate <= min_rate:
        return (min_width + max_width) / 2
    lo = math.log10(max(min_rate, 1e-30))
    hi = math.log10(max(max_rate, min_rate))
    frac = (math.log10(max(rate, min_rate)) - lo) / max(hi - lo, 1e-9)
    frac = frac**1.45
    return min_width + (max_width - min_width) * frac


def format_power(power: float | None, run_name: str) -> str:
    if power is None:
        return run_name
    if power >= 1000:
        return f"{power:,.0f} W cm$^{{-2}}$"
    return f"{power:g} W cm$^{{-2}}$"


def superscript_label(label: str) -> str:
    match = re.match(r"^(\d+)([A-Z])(\d+)$", label)
    if not match:
        return label
    mult, letter, j = match.groups()
    return rf"$^{{{mult}}}{letter}_{{{j}}}$"


def draw_arrow(ax, x: float, y0: float, y1: float, color: str, dashed: bool, lw: float):
    patch = FancyArrowPatch(
        (x, y0),
        (x, y1),
        arrowstyle="-|>",
        mutation_scale=7.5 + 1.6 * lw,
        linewidth=lw,
        facecolor="none",
        edgecolor=color,
        alpha=0.9,
    )
    ax.add_patch(patch)
    if dashed:
        ax.plot([x, x], [y0, y1], color=color, lw=max(0.75, lw * 0.75), ls=":", alpha=0.75)


def process_sort_key(process: Process) -> tuple:
    levels = [level for component in process.components for level in component]
    return (-process.rate, min(levels), max(levels), process.components)


def assign_lanes(groups: dict[str, list[Process]], spans: dict[str, tuple[float, float]]) -> dict[Process, float]:
    lanes: dict[Process, float] = {}
    for group_name, rows in groups.items():
        left, right = spans[group_name]
        rows.sort(key=process_sort_key)
        n = len(rows)
        for j, process in enumerate(rows):
            x = (left + right) / 2 if n == 1 else left + (right - left) * j / (n - 1)
            lanes[process] = x
    return lanes


def assign_plot_labels(groups: dict[str, list[Process]], lanes: dict[Process, float]) -> dict[Process, int]:
    labels: dict[Process, int] = {}
    for rows in groups.values():
        for i, process in enumerate(sorted(rows, key=lambda a: (lanes[a], process_sort_key(a)))):
            labels[process] = i
    return labels


def label_y_position(y0: float, y1: float, plot_label: int) -> float:
    low = min(y0, y1) + 520.0
    high = max(y0, y1) - 520.0
    base = (y0 + y1) / 2
    if high <= low:
        return base
    stagger = ((plot_label % 5) - 2) * 260.0
    return min(max(base + stagger, low), high)


def group_rate_bounds(groups: dict[str, list[Process]], min_rate: float) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for group_name, rows in groups.items():
        rates = [process.rate for process in rows]
        if rates:
            bounds[group_name] = (max(min(rates), min_rate), max(rates))
    return bounds


def plot_run(
    run_dir: Path,
    output_name: str,
    rate_key: str,
    min_rate: float,
    max_by_kind: dict[str, int],
    dpi: int,
) -> bool:
    labels, energies = load_species_data()
    y_positions = [float(energy) for energy in energies]
    arrows, summary = extract_arrows(run_dir, rate_key, min_rate, max_by_kind)
    if not arrows:
        print(f"SKIP {run_dir.name}: no arrows above min-rate")
        return False
    fig, ax = plt.subplots(figsize=(17.0, 13.5))
    fig.subplots_adjust(top=0.88)
    ax.set_xlim(0.0, 1.52)
    ax.set_ylim(min(y_positions) - 900, max(y_positions) + 1800)
    ax.axis("off")

    x0, x1 = 0.06, 1.32
    for i, (label, y) in enumerate(zip(labels, y_positions)):
        ax.hlines(y, x0, x1, color="black", lw=1.8)
        ax.text(1.37, y, superscript_label(label), va="center", ha="left", fontsize=15)
        ax.text(0.025, y, str(i), va="center", ha="right", fontsize=9, color="#555555")

    groups = {
        "Pump": [process for process in arrows if process.color_group == "Pump"],
        "Rad": [process for process in arrows if process.color_group == "Rad"],
        "NR": [process for process in arrows if process.color_group == "NR"],
        "ET": [process for process in arrows if process.color_group == "ET"],
        "SQ": [process for process in arrows if process.color_group == "SQ"],
    }
    spans = {
        "Pump": (0.105, 0.155),
        "Rad": (0.245, 0.47),
        "NR": (0.52, 0.66),
        "ET": (0.69, 1.10),
        "SQ": (1.17, 1.30),
    }
    lanes = assign_lanes(groups, spans)
    plot_labels = assign_plot_labels(groups, lanes)
    rate_bounds = group_rate_bounds(groups, min_rate)
    label_map: list[dict] = []

    for group_name, rows in groups.items():
        if not rows:
            continue
        for process in sorted(rows, key=lambda a: (lanes[a], process_sort_key(a))):
            x = lanes[process]
            color = COLORS.get(process.kind, "#333333")
            group_min_rate, group_max_rate = rate_bounds[group_name]
            lw = line_width(process.rate, group_min_rate, group_max_rate, len(rows))
            component_offsets = [0.0]
            if len(process.components) == 2:
                component_offsets = [-0.0024, 0.0024]
            elif len(process.components) > 2:
                center = (len(process.components) - 1) / 2
                component_offsets = [(i - center) * 0.0024 for i in range(len(process.components))]

            component_ys = []
            for offset, (source, target) in zip(component_offsets, process.components):
                y0 = y_positions[source]
                y1 = y_positions[target]
                component_ys.extend([y0, y1])
                draw_arrow(ax, x + offset, y0, y1, color, dashed=process.kind == "EM", lw=lw)

            plot_label = plot_labels[process]
            y_label = label_y_position(min(component_ys), max(component_ys), plot_label)
            ax.text(
                x,
                y_label,
                str(plot_label),
                ha="center",
                va="center",
                fontsize=7.0,
                color="white",
                bbox=dict(boxstyle="round,pad=0.11", fc=color, ec="none", alpha=0.95),
                zorder=5,
            )
            label_map.append(
                {
                    "plot_label": plot_label,
                    "color_group": group_name,
                    "color": color,
                    "interaction_ids": list(process.interaction_ids),
                    "representative_label": process.label,
                    "components": [
                        {
                            "source_level_index": source,
                            "target_level_index": target,
                            "source_level": labels[source],
                            "target_level": labels[target],
                        }
                        for source, target in process.components
                    ],
                    "line_width": lw,
                    rate_key: process.rate,
                }
            )

    power = summary.get("excitation_power_w_cm2")
    ax.text(0.5, 1.02, format_power(power, run_dir.name), transform=ax.transAxes, ha="center", fontsize=18)
    ax.text(0.5, -0.035, r"Tm$^{3+}$", transform=ax.transAxes, ha="center", fontsize=28, weight="bold")
    fig.text(
        0.075,
        0.975,
        "Arrow labels count left-to-right by decreasing rate within each color; duplicate transitions collapsed; EM excluded",
        ha="left",
        va="center",
        fontsize=9,
        color="#555555",
    )

    handles = [
        Line2D([0], [0], color=COLORS["ET"], lw=4, label="Energy transfer"),
        Line2D([0], [0], color=COLORS["SQ"], lw=4, label="Surface quenching"),
        Line2D([0], [0], color=COLORS["Pump"], lw=4, label="GSA / ESA pump"),
        Line2D([0], [0], color=COLORS["Rad"], lw=4, label="Radiative"),
        Line2D([0], [0], color=COLORS["NR"], lw=4, label="Non-radiative"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.96), frameon=True, fontsize=10)

    out = run_dir / output_name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    map_out = out.with_name(out.stem + "_labels.json")
    map_out.write_text(json.dumps(sorted(label_map, key=lambda r: (r["color_group"], r["plot_label"])), indent=2) + "\n")
    print(f"OK   {run_dir.name}: {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", nargs="?", type=Path, default=None, help="Directory containing power_* runs")
    parser.add_argument("--base-dir", dest="base_dir_flag", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-name", default="mechanism_diagram.png", help="Filename to save in each run folder")
    parser.add_argument("--rate-key", default="events_per_ion_s", choices=["events_per_ion_s", "events_per_particle_s"])
    parser.add_argument("--min-rate", type=float, default=1e-3, help="Minimum plotted event rate")
    parser.add_argument("--max-et", type=int, default=16, help="Max energy-transfer arrows")
    parser.add_argument("--max-rad", type=int, default=18, help="Max radiative arrows")
    parser.add_argument("--max-nr", type=int, default=12, help="Max non-radiative arrows")
    parser.add_argument("--max-pump", type=int, default=8, help="Max pump arrows")
    parser.add_argument("--max-sq", type=int, default=16, help="Max surface-quenching arrows")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    max_by_kind = {
        "ET": args.max_et,
        "Rad": args.max_rad,
        "NR": args.max_nr,
        "Pump": args.max_pump,
        "SQ": args.max_sq,
    }
    base_dir = args.base_dir if args.base_dir is not None else args.base_dir_flag
    if base_dir is None:
        base_dir = Path(".")
    power_dirs = sorted(discover_power_dirs(base_dir), key=power_sort_key)
    print(f"Found {len(power_dirs)} power folders")
    done = 0
    for run_dir in power_dirs:
        done += int(plot_run(run_dir, args.output_name, args.rate_key, args.min_rate, max_by_kind, args.dpi))
    print(f"Done: saved {done}/{len(power_dirs)} mechanism diagrams")


if __name__ == "__main__":
    main()
