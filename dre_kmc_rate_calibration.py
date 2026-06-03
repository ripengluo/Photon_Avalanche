"""Calibrate Table S3 DRE constants into NPMC/kMC rate constants.

The functions in this file are intended to be imported by future simulation
scripts. Running this file directly prints a comparison between:

1. DRE-calibrated NPMC pair coefficients for the 4.5%, 0 nN Table S3 profile.
2. NanoParticleTools' semi-empirical SpectralKinetics ET coefficients.

The key conversion is:

    k_pair = N_ions * K_DRE / sum_ordered_pairs(f(r_ab))

where NPMC uses f(r) = r^-6 for distance_factor_type="inverse_cubic".
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
NPT_SRC = PROJECT_ROOT / "NanoParticleTools" / "src"
if NPT_SRC.exists() and str(NPT_SRC) not in sys.path:
    sys.path.insert(0, str(NPT_SRC))

from NanoParticleTools.inputs.photo_physics import (  # noqa: E402
    energy_transfer_constant,
    gaussian,
    gaussian_overlap_integral,
    get_absorption_cross_section_from_line_strength,
    get_critical_energy_gap,
    phonon_assisted_energy_transfer_constant,
)
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics  # noqa: E402
from NanoParticleTools.species_data.species import Dopant  # noqa: E402
from NanoParticleTools.util.constants import c_CGS, h_CGS  # noqa: E402


DEFAULT_PARAMS_PATH = ROOT / "table_s3_4p5_0nN.json"
DEFAULT_NP_DB_PATH = (
    ROOT / "run1" / ".scratch_tm_4p56_no_force" / "power_00" / "np.sqlite"
)


@dataclass(frozen=True)
class DREChannel:
    """A two-ion DRE channel from Table S3 notation."""

    name: str
    transition_a: tuple[int, int]
    transition_b: tuple[int, int]
    description: str
    dre_rate_s: float


@dataclass(frozen=True)
class OrderedChannel:
    """The donor/acceptor ordering used by SpectralKinetics ET rows."""

    donor_initial_state: int
    donor_final_state: int
    acceptor_initial_state: int
    acceptor_final_state: int
    donor_initial_level: int
    donor_final_level: int
    acceptor_initial_level: int
    acceptor_final_level: int


@dataclass(frozen=True)
class GeometryFactor:
    """Geometry normalization for converting mean-field DRE rates to pairs."""

    np_db_path: str
    ion_count: int
    interaction_radius_bound_nm: float
    distance_factor_type: str
    unordered_pair_count: int
    unordered_factor_sum: float
    ordered_factor_sum: float


@dataclass(frozen=True)
class PairRateComparison:
    """Comparison for one two-ion channel."""

    channel_name: str
    description: str
    dre_rate_s: float
    calibrated_kmc_rate_nm6_s: float
    semi_empirical_selected_nm6_s: float
    semi_empirical_exported_nm6_s: float | None
    semi_empirical_exported: bool
    semi_empirical_equivalent_dre_rate_s: float
    selected_over_calibrated: float | None
    exported_over_calibrated: float | None
    kmc_tuple: tuple[int, int, int, int]
    branch: str
    energy_gap_cm: float
    effective_energy_gap_cm: float


def load_dre_parameters(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load the JSON DRE profile."""
    with open(path) as f:
        return json.load(f)


def dre_state_to_level_map(params: dict[str, Any]) -> dict[int, int]:
    """Map DRE's 1-based state numbering onto 0-based Tm kMC levels."""
    return {
        int(row["dre_state"]): int(row["kmc_level"])
        for row in params["states"]
    }


def load_dre_channels(params: dict[str, Any]) -> list[DREChannel]:
    """Build channel objects from the JSON profile."""
    rates = params["cross_relaxation_rates_s^-1"]
    channels = []
    for row in params["dre_channels"]:
        name = row["name"]
        channels.append(
            DREChannel(
                name=name,
                transition_a=tuple(row["transition_a"]),
                transition_b=tuple(row["transition_b"]),
                description=row["description"],
                dre_rate_s=float(rates[name]),
            )
        )
    return channels


def build_resonant_migration_pair_rates(
    params: dict[str, Any],
    sk: SpectralKinetics,
) -> list[dict[str, Any]]:
    """Return five-level resonant ET swap rows exported by SpectralKinetics.

    Resonant migration is identified as a two-site swap, e.g. ``2 + 1 -> 1 + 2``,
    where the donor final level equals the acceptor initial level and the acceptor
    final level equals the donor initial level. Only rows that remain inside the
    configured DRE five-level subspace are returned.
    """
    valid_levels = set(dre_state_to_level_map(params).values())
    level_to_state = {
        int(row["kmc_level"]): int(row["dre_state"])
        for row in params["states"]
    }
    level_to_label = {
        int(row["kmc_level"]): str(row["label"])
        for row in params["states"]
    }
    ground_mediated_tuples = {
        (1, 0, 0, 1),
        (3, 0, 0, 3),
        (4, 0, 0, 4),
    }
    ground_mediated_targets = {
        (1, 0, 0, 1): "em21",
        (3, 0, 0, 3): "em41",
        (4, 0, 0, 4): "em51",
    }
    in_loop_only_tuples = {
        (3, 1, 1, 3),
        (4, 1, 1, 4),
        (4, 3, 3, 4),
    }

    resonant_rows: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in np.asarray(sk.energy_transfer_rate_matrix):
        di, dj, ai, aj = (int(value) for value in row[:4])
        kmc_tuple = (di, dj, ai, aj)
        if not all(level in valid_levels for level in kmc_tuple):
            continue
        if dj != ai or aj != di:
            continue

        dre_tuple = tuple(level_to_state[level] for level in kmc_tuple)
        channel_name = (
            f"EM {dre_tuple[0]}+{dre_tuple[2]}->{dre_tuple[1]}+{dre_tuple[3]}"
        )
        description = (
            f"({level_to_label[di]} ; {level_to_label[ai]}) -> "
            f"({level_to_label[dj]} ; {level_to_label[aj]})"
        )
        if kmc_tuple in ground_mediated_tuples:
            migration_family = "ground_mediated"
            enabled_modes = ["all", "ground_mediated", "in_loop"]
            scan_target = ground_mediated_targets[kmc_tuple]
        elif kmc_tuple in in_loop_only_tuples:
            migration_family = "in_loop_only"
            enabled_modes = ["all", "in_loop"]
            scan_target = None
        else:
            migration_family = "level3_background"
            enabled_modes = ["all"]
            scan_target = None
        resonant_rows[kmc_tuple] = {
            "channel_name": channel_name,
            "description": description,
            "kmc_tuple": list(kmc_tuple),
            "dre_tuple": list(dre_tuple),
            "rate_nm6_s": float(row[4]) * 1.0e42,
            "source": "NPT resonant migration",
            "migration_family": migration_family,
            "enabled_modes": enabled_modes,
            "scan_target": scan_target,
        }

    return [resonant_rows[key] for key in sorted(resonant_rows)]


def load_sites_from_np_db(np_db_path: str | Path) -> np.ndarray:
    """Load NPMC site coordinates from a nanoparticle SQLite database."""
    with sqlite3.connect(np_db_path) as con:
        rows = con.execute(
            "SELECT x, y, z FROM sites ORDER BY site_id"
        ).fetchall()
    if not rows:
        raise ValueError(f"No sites found in {np_db_path}")
    return np.asarray(rows, dtype=float)


def distance_factor(distance_nm: np.ndarray, factor_type: str, cutoff_nm: float) -> np.ndarray:
    """Evaluate the same distance factor family used by NPMC."""
    if factor_type == "inverse_cubic":
        return np.power(distance_nm, -6)
    if factor_type == "linear":
        return 1.0 - distance_nm / cutoff_nm
    raise ValueError(f"Unsupported distance_factor_type: {factor_type!r}")


def compute_geometry_factor(
    np_db_path: str | Path,
    interaction_radius_bound_nm: float,
    distance_factor_type: str,
) -> GeometryFactor:
    """Compute the ordered pair normalization used for DRE-to-kMC conversion."""
    sites = load_sites_from_np_db(np_db_path)
    tree = cKDTree(sites)
    pairs = tree.query_pairs(interaction_radius_bound_nm, output_type="ndarray")

    if len(pairs):
        distances = np.linalg.norm(sites[pairs[:, 0]] - sites[pairs[:, 1]], axis=1)
        values = distance_factor(
            distances,
            factor_type=distance_factor_type,
            cutoff_nm=interaction_radius_bound_nm,
        )
        unordered_sum = float(np.sum(values))
    else:
        unordered_sum = 0.0

    return GeometryFactor(
        np_db_path=str(np_db_path),
        ion_count=int(len(sites)),
        interaction_radius_bound_nm=float(interaction_radius_bound_nm),
        distance_factor_type=distance_factor_type,
        unordered_pair_count=int(len(pairs)),
        unordered_factor_sum=unordered_sum,
        ordered_factor_sum=2.0 * unordered_sum,
    )


def calibrate_dre_pair_rate_nm6_s(
    dre_rate_s: float,
    geometry: GeometryFactor,
) -> float:
    """Convert a Table S3 DRE two-ion rate into an NPMC interaction rate row."""
    if geometry.ordered_factor_sum <= 0:
        raise ValueError("ordered_factor_sum must be positive")
    return geometry.ion_count * dre_rate_s / geometry.ordered_factor_sum


def build_spectral_kinetics(
    params: dict[str, Any],
    excitation_power_w_cm2: float,
    tm_fraction: float | None = None,
) -> tuple[Dopant, SpectralKinetics]:
    """Build NanoParticleTools SpectralKinetics using JSON defaults."""
    sim_defaults = params["simulation_defaults"]
    sk_defaults = params["spectral_kinetics_defaults"]
    if tm_fraction is None:
        tm_fraction = float(sim_defaults["tm_fraction_for_semi_empirical"])

    dopant = Dopant("Tm", tm_fraction)
    sk = SpectralKinetics(
        [dopant],
        excitation_wavelength=float(sim_defaults["excitation_wavelength_nm"]),
        excitation_power=float(excitation_power_w_cm2),
        phonon_energy=float(sk_defaults["phonon_energy_cm^-1"]),
        zero_phonon_rate=float(sk_defaults["zero_phonon_rate_s^-1"]),
        mpr_alpha=float(sk_defaults["mpr_alpha_cm"]),
        n_refract=float(sk_defaults["n_refract"]),
        stokes_shift=float(sk_defaults["stokes_shift_cm^-1"]),
        energy_transfer_rate_threshold=float(
            sk_defaults["energy_transfer_rate_threshold_s^-1"]
        ),
    )
    return dopant, sk


def build_kmc_default_absorption_cross_sections(
    params: dict[str, Any],
    tm_fraction: float | None = None,
) -> dict[str, float]:
    """Compute NPT-style effective absorption cross sections for the pump channels."""
    dopant, sk = build_spectral_kinetics(params, excitation_power_w_cm2=1.0, tm_fraction=tm_fraction)
    line_strengths = dopant.get_line_strength_matrix()

    def effective_sigma(initial_level: int, final_level: int) -> float:
        energy_gap = (
            dopant.energy_levels[final_level].energy
            - dopant.energy_levels[initial_level].energy
        )
        absfwhm = dopant.absFWHM[final_level]
        abs_sigma = absfwhm / (2 * np.sqrt(2 * np.log(2)))
        absorption_cross_section = get_absorption_cross_section_from_line_strength(
            energy_gap,
            line_strengths[initial_level][final_level],
            dopant.slj[initial_level][2],
            sk.n_refract,
        )
        critical_energy_gap = get_critical_energy_gap(sk.mpr_alpha, absfwhm)
        if abs(energy_gap - sk.incident_wavenumber) > critical_energy_gap:
            return float(
                absorption_cross_section
                * gaussian(energy_gap, energy_gap, abs_sigma)
                * np.exp(-sk.mpr_alpha * abs(energy_gap - sk.incident_wavenumber))
            )
        return float(
            absorption_cross_section
            * gaussian(energy_gap, sk.incident_wavenumber, abs_sigma)
        )

    return {
        "sigma_GSA": effective_sigma(0, 2),
        "sigma_ESA": effective_sigma(1, 4),
    }


def order_channel(
    channel: DREChannel,
    dopant: Dopant,
    state_to_level: dict[int, int],
) -> OrderedChannel:
    """Order a DRE two-ion channel as donor downhill plus acceptor uphill."""
    candidates = []
    for initial_state, final_state in (channel.transition_a, channel.transition_b):
        initial_level = state_to_level[int(initial_state)]
        final_level = state_to_level[int(final_state)]
        energy_change = (
            dopant.energy_levels[final_level].energy
            - dopant.energy_levels[initial_level].energy
        )
        candidates.append(
            {
                "initial_state": int(initial_state),
                "final_state": int(final_state),
                "initial_level": initial_level,
                "final_level": final_level,
                "energy_change": energy_change,
            }
        )

    downhill = [item for item in candidates if item["energy_change"] < 0]
    uphill = [item for item in candidates if item["energy_change"] > 0]
    if len(downhill) != 1 or len(uphill) != 1:
        raise ValueError(f"{channel.name} is not a one-donor/one-acceptor channel")

    donor = downhill[0]
    acceptor = uphill[0]
    return OrderedChannel(
        donor_initial_state=donor["initial_state"],
        donor_final_state=donor["final_state"],
        acceptor_initial_state=acceptor["initial_state"],
        acceptor_final_state=acceptor["final_state"],
        donor_initial_level=donor["initial_level"],
        donor_final_level=donor["final_level"],
        acceptor_initial_level=acceptor["initial_level"],
        acceptor_final_level=acceptor["final_level"],
    )


def semi_empirical_pair_rate(
    ordered: OrderedChannel,
    dopant: Dopant,
    sk: SpectralKinetics,
) -> dict[str, float | str | tuple[int, int, int, int] | bool | None]:
    """Compute NanoParticleTools' semi-empirical pair coefficient for a channel."""
    line_strengths = dopant.get_line_strength_matrix()
    di = ordered.donor_initial_level
    dj = ordered.donor_final_level
    ai = ordered.acceptor_initial_level
    aj = ordered.acceptor_final_level

    donor_energy_change = dopant.energy_levels[dj].energy - dopant.energy_levels[di].energy
    acceptor_energy_change = dopant.energy_levels[aj].energy - dopant.energy_levels[ai].energy
    energy_gap = donor_energy_change + acceptor_energy_change
    effective_energy_gap = energy_gap + sk.stokes_shift
    critical_energy_gap = get_critical_energy_gap(sk.mpr_beta, dopant.absFWHM[di])
    line_width = max(dopant.absFWHM[di], dopant.absFWHM[aj])

    donor_line_strength = line_strengths[di][dj]
    acceptor_line_strength = line_strengths[ai][aj]
    donor_j = dopant.slj[di, 2]
    acceptor_j = dopant.slj[ai, 2]

    direct_overlap = gaussian_overlap_integral(abs(effective_energy_gap), line_width)
    zero_gap_overlap = gaussian_overlap_integral(0, line_width)
    direct_rate_cm6_s = energy_transfer_constant(
        donor_line_strength,
        acceptor_line_strength,
        direct_overlap,
        sk.n_refract,
        donor_j,
        acceptor_j,
    )
    phonon_rate_cm6_s = phonon_assisted_energy_transfer_constant(
        donor_line_strength,
        acceptor_line_strength,
        zero_gap_overlap,
        sk.n_refract,
        donor_j,
        acceptor_j,
        abs(effective_energy_gap),
        sk.mpr_beta,
    )

    branch = "direct" if effective_energy_gap > -critical_energy_gap else "phonon_assisted"
    selected_cm6_s = direct_rate_cm6_s if branch == "direct" else phonon_rate_cm6_s
    selected_nm6_s = selected_cm6_s * 1.0e42

    target_tuple = (di, dj, ai, aj)
    exported_nm6_s = None
    exported = False
    for row in sk.energy_transfer_rate_matrix:
        if tuple(int(value) for value in row[:4]) == target_tuple:
            exported = True
            exported_nm6_s = float(row[4]) * 1.0e42
            break

    return {
        "kmc_tuple": target_tuple,
        "branch": branch,
        "energy_gap_cm": float(energy_gap),
        "effective_energy_gap_cm": float(effective_energy_gap),
        "selected_nm6_s": float(selected_nm6_s),
        "exported_nm6_s": exported_nm6_s,
        "exported": exported,
    }


def equivalent_dre_rate_s(k_pair_nm6_s: float, geometry: GeometryFactor) -> float:
    """Convert an NPMC pair coefficient back to its mean-field DRE equivalent."""
    return k_pair_nm6_s * geometry.ordered_factor_sum / geometry.ion_count


def compare_with_semi_empirical(
    params: dict[str, Any],
    np_db_path: str | Path,
    excitation_power_w_cm2: float = 1.0e4,
    tm_fraction: float | None = None,
) -> dict[str, Any]:
    """Return a complete DRE-converted vs semi-empirical comparison report."""
    sim_defaults = params["simulation_defaults"]
    geometry = compute_geometry_factor(
        np_db_path=np_db_path,
        interaction_radius_bound_nm=float(sim_defaults["interaction_radius_bound_nm"]),
        distance_factor_type=sim_defaults["distance_factor_type"],
    )
    dopant, sk = build_spectral_kinetics(
        params,
        excitation_power_w_cm2=excitation_power_w_cm2,
        tm_fraction=tm_fraction,
    )
    state_to_level = dre_state_to_level_map(params)

    pair_comparisons = []
    for channel in load_dre_channels(params):
        ordered = order_channel(channel, dopant, state_to_level)
        calibrated = calibrate_dre_pair_rate_nm6_s(channel.dre_rate_s, geometry)
        semi = semi_empirical_pair_rate(ordered, dopant, sk)
        selected = float(semi["selected_nm6_s"])
        exported = semi["exported_nm6_s"]

        pair_comparisons.append(
            PairRateComparison(
                channel_name=channel.name,
                description=channel.description,
                dre_rate_s=channel.dre_rate_s,
                calibrated_kmc_rate_nm6_s=calibrated,
                semi_empirical_selected_nm6_s=selected,
                semi_empirical_exported_nm6_s=exported,
                semi_empirical_exported=bool(semi["exported"]),
                semi_empirical_equivalent_dre_rate_s=equivalent_dre_rate_s(
                    selected,
                    geometry,
                ),
                selected_over_calibrated=(
                    selected / calibrated if calibrated > 0 else None
                ),
                exported_over_calibrated=(
                    exported / calibrated
                    if exported is not None and calibrated > 0
                    else None
                ),
                kmc_tuple=semi["kmc_tuple"],
                branch=str(semi["branch"]),
                energy_gap_cm=float(semi["energy_gap_cm"]),
                effective_energy_gap_cm=float(semi["effective_energy_gap_cm"]),
            )
        )

    return {
        "profile": params["profile"],
        "excitation_power_w_cm2": float(excitation_power_w_cm2),
        "tm_fraction_for_semi_empirical": float(
            tm_fraction
            if tm_fraction is not None
            else sim_defaults["tm_fraction_for_semi_empirical"]
        ),
        "geometry": asdict(geometry),
        "pair_rate_comparisons": [asdict(item) for item in pair_comparisons],
        "one_site_comparisons": build_one_site_comparisons(params, dopant, sk),
    }


def photon_flux_cm2_s(excitation_power_w_cm2: float, wavelength_nm: float) -> float:
    """Photon flux density in photons / (s cm^2)."""
    incident_wavenumber = 1.0e7 / wavelength_nm
    return excitation_power_w_cm2 * 1.0e7 / (h_CGS * c_CGS * incident_wavenumber)


def build_dre_one_site_rates(
    params: dict[str, Any],
    excitation_power_w_cm2: float,
    absorption_cross_sections: dict[str, float] | None = None,
    sigma_esa_scale: float = 1.0,
    w3_one_site_source: str = "table-s3",
    spectral_kinetics: SpectralKinetics | None = None,
    w3_nr_scale: float = 1.0,
    w5_nr_scale: float = 1.0,
    one_site_source: str | None = None,
) -> list[dict[str, Any]]:
    """Expand DRE one-site rates into explicit transition rows."""
    branching = params["branching_ratios"]
    radiative = params["radiative_rates_s^-1"]
    nonradiative = params["nonradiative_rates_s^-1"]
    wavelength_nm = float(params["simulation_defaults"]["excitation_wavelength_nm"])
    flux = photon_flux_cm2_s(excitation_power_w_cm2, wavelength_nm)
    cross_sections = (
        params["absorption_cross_sections_cm^2"]
        if absorption_cross_sections is None
        else absorption_cross_sections
    )
    if one_site_source is None:
        if w3_one_site_source not in {"table-s3", "npt"}:
            raise ValueError(f"Unsupported W3 one-site source: {w3_one_site_source!r}")
        resolved_one_site_source = "table-s3"
        legacy_w3_only_override = w3_one_site_source == "npt"
    else:
        if one_site_source not in {"table-s3", "npt"}:
            raise ValueError(f"Unsupported one-site source: {one_site_source!r}")
        resolved_one_site_source = one_site_source
        legacy_w3_only_override = False

    if (
        resolved_one_site_source == "npt" or legacy_w3_only_override
    ) and spectral_kinetics is None:
        raise ValueError(
            "spectral_kinetics is required when one-site rates use NPT values"
        )

    total_rad = None
    nr = None
    if spectral_kinetics is not None:
        total_rad = (
            spectral_kinetics.radiative_rate_matrix
            + spectral_kinetics.magnetic_dipole_rate_matrix
        )
        nr = spectral_kinetics.non_radiative_rate_matrix

    rows: list[dict[str, Any]] = []
    if resolved_one_site_source == "npt":
        if total_rad is None:
            raise ValueError("Missing SpectralKinetics radiative rates for NPT one-site source")
        w2r_rate = float(total_rad[1, 0])
    else:
        w2r_rate = float(radiative["W2R"])
    rows.append(
        {
            "type": "Rad",
            "left": 2,
            "right": 1,
            "base_dre_rate_s": w2r_rate,
            "rate_scale_factor": 1.0,
            "dre_rate_s": w2r_rate,
        }
    )
    for initial in (3, 4, 5):
        for final in range(1, initial):
            if resolved_one_site_source == "npt":
                if total_rad is None:
                    raise ValueError("Missing SpectralKinetics radiative rates for NPT one-site source")
                base_rate = float(total_rad[initial - 1, final - 1])
            elif initial == 3 and legacy_w3_only_override:
                if total_rad is None:
                    raise ValueError("Missing SpectralKinetics radiative rates for W3 source override")
                base_rate = float(total_rad[2, final - 1])
            else:
                total = float(radiative[f"W{initial}R"])
                key = f"b{initial}{final}"
                if key not in branching or branching[key] == 0:
                    continue
                base_rate = total * float(branching[key])
            rows.append(
                {
                    "type": "Rad",
                    "left": initial,
                    "right": final,
                    "base_dre_rate_s": base_rate,
                    "rate_scale_factor": 1.0,
                    "dre_rate_s": base_rate,
                }
            )

    for initial in (2, 3, 4, 5):
        if resolved_one_site_source == "npt":
            if nr is None:
                raise ValueError("Missing SpectralKinetics nonradiative rates for NPT one-site source")
            base_rate = float(nr[initial - 1, initial - 2])
        elif initial == 3 and legacy_w3_only_override:
            if nr is None:
                raise ValueError("Missing SpectralKinetics nonradiative rates for W3 source override")
            base_rate = float(nr[2, 1])
        else:
            base_rate = float(nonradiative[f"W{initial}NR"])
        if initial == 3:
            rate_scale_factor = float(w3_nr_scale)
        elif initial == 5:
            rate_scale_factor = float(w5_nr_scale)
        else:
            rate_scale_factor = 1.0
        rows.append(
            {
                "type": "NR",
                "left": initial,
                "right": initial - 1,
                "base_dre_rate_s": base_rate,
                "rate_scale_factor": rate_scale_factor,
                "dre_rate_s": base_rate * rate_scale_factor,
            }
        )

    gsa_rate = float(cross_sections["sigma_GSA"] * flux)
    rows.append(
        {
            "type": "Pump",
            "left": 1,
            "right": 3,
            "base_dre_rate_s": gsa_rate,
            "rate_scale_factor": 1.0,
            "dre_rate_s": gsa_rate,
        }
    )
    esa_rate = float(cross_sections["sigma_ESA"] * sigma_esa_scale * flux)
    rows.append(
        {
            "type": "Pump",
            "left": 2,
            "right": 5,
            "base_dre_rate_s": esa_rate,
            "rate_scale_factor": 1.0,
            "dre_rate_s": esa_rate,
        }
    )
    return rows


def build_one_site_comparisons(
    params: dict[str, Any],
    dopant: Dopant,
    sk: SpectralKinetics,
) -> list[dict[str, Any]]:
    """Compare DRE one-site transition rates with SpectralKinetics rates."""
    total_rad = sk.radiative_rate_matrix + sk.magnetic_dipole_rate_matrix
    nr = sk.non_radiative_rate_matrix
    comparisons = []
    for row in build_dre_one_site_rates(params, sk.excitation_power):
        left_level = int(row["left"]) - 1
        right_level = int(row["right"]) - 1
        if row["type"] == "NR":
            semi_rate = float(nr[left_level, right_level])
        else:
            semi_rate = float(total_rad[left_level, right_level])
        dre_rate = float(row["dre_rate_s"])
        comparisons.append(
            {
                "type": row["type"],
                "transition": f"{row['left']}->{row['right']}",
                "labels": (
                    f"{dopant.energy_levels[left_level].label}->"
                    f"{dopant.energy_levels[right_level].label}"
                ),
                "dre_rate_s": dre_rate,
                "semi_empirical_rate_s": semi_rate,
                "semi_over_dre": semi_rate / dre_rate if dre_rate > 0 else None,
            }
        )
    return comparisons


def _json_safe(value: Any) -> Any:
    """Convert tuples and NumPy scalars for JSON serialization."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"


def print_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable comparison report."""
    geometry = report["geometry"]
    print(f"DRE to kMC calibration report: {report['profile']}")
    print(f"np.sqlite: {geometry['np_db_path']}")
    print(
        "geometry: "
        f"N={geometry['ion_count']}, "
        f"cutoff={geometry['interaction_radius_bound_nm']} nm, "
        f"distance_factor={geometry['distance_factor_type']}, "
        f"ordered_sum={geometry['ordered_factor_sum']:.6g}"
    )
    print(
        "semi-empirical settings: "
        f"Tm={report['tm_fraction_for_semi_empirical']:.4f}, "
        f"excitation_power={report['excitation_power_w_cm2']:.6g} W cm^-2"
    )
    print()
    print("Two-site ET / CR rates")
    print(
        f"{'channel':<10} {'K_DRE(s^-1)':>13} {'DRE kMC(nm^6/s)':>18} "
        f"{'semi selected':>15} {'semi exported':>15} {'selected/DRE':>13}"
    )
    for row in report["pair_rate_comparisons"]:
        exported = (
            f"{row['semi_empirical_exported_nm6_s']:.6g}"
            if row["semi_empirical_exported_nm6_s"] is not None
            else "not exported"
        )
        print(
            f"{row['channel_name']:<10} "
            f"{row['dre_rate_s']:>13.6g} "
            f"{row['calibrated_kmc_rate_nm6_s']:>18.6g} "
            f"{row['semi_empirical_selected_nm6_s']:>15.6g} "
            f"{exported:>15} "
            f"{_format_ratio(row['selected_over_calibrated']):>13}"
        )
    print()
    print("One-site rates")
    print(
        f"{'type':<6} {'transition':<10} {'DRE(s^-1)':>13} "
        f"{'semi(s^-1)':>13} {'semi/DRE':>10} labels"
    )
    for row in report["one_site_comparisons"]:
        print(
            f"{row['type']:<6} "
            f"{row['transition']:<10} "
            f"{row['dre_rate_s']:>13.6g} "
            f"{row['semi_empirical_rate_s']:>13.6g} "
            f"{_format_ratio(row['semi_over_dre']):>10} "
            f"{row['labels']}"
        )
    print()
    print(
        "Note: 'semi selected' is the semi-empirical pair coefficient before "
        "NanoParticleTools filtering; 'semi exported' is present in the actual "
        "SpectralKinetics ET matrix only if it passes the built-in filters."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Table S3 DRE-calibrated kMC rates with semi-empirical NanoParticleTools rates."
    )
    parser.add_argument(
        "--params",
        default=str(DEFAULT_PARAMS_PATH),
        help="Path to a Table S3 DRE JSON profile.",
    )
    parser.add_argument(
        "--np-db",
        default=str(DEFAULT_NP_DB_PATH),
        help="Path to an NPMC np.sqlite database containing site coordinates.",
    )
    parser.add_argument(
        "--excitation-power",
        type=float,
        default=1.0e4,
        help="Excitation power density in W cm^-2 for pump-rate comparisons.",
    )
    parser.add_argument(
        "--tm-fraction",
        type=float,
        default=None,
        help="Override Tm molar fraction for semi-empirical SpectralKinetics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full comparison report as JSON.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full comparison report as JSON.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    params = load_dre_parameters(args.params)
    report = compare_with_semi_empirical(
        params=params,
        np_db_path=args.np_db,
        excitation_power_w_cm2=args.excitation_power,
        tm_fraction=args.tm_fraction,
    )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(_json_safe(report), f, indent=2)

    if args.json:
        print(json.dumps(_json_safe(report), indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
