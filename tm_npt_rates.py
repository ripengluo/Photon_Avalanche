"""Shared NPT rate helpers for the Tm avalanche kMC workflow."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
from NanoParticleTools.inputs.util import get_all_interactions  # noqa: E402
from NanoParticleTools.species_data.species import Dopant  # noqa: E402


DEFAULT_PARAMS_PATH = ROOT / "SK_input.json"


@dataclass(frozen=True)
class DREChannel:
    """Legacy low-level channel metadata used for compatibility scaling."""

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
    """Geometry normalization for converting pair rates to mean-field-equivalent form."""

    np_db_path: str
    ion_count: int
    interaction_radius_bound_nm: float
    distance_factor_type: str
    unordered_pair_count: int
    unordered_factor_sum: float
    ordered_factor_sum: float


@dataclass(frozen=True)
class SiteRecord:
    """One dopant site loaded from an NPMC nanoparticle database."""

    site_id: int
    x: float
    y: float
    z: float
    species_id: int


@dataclass(frozen=True)
class SpeciesRecord:
    """One species definition loaded from an NPMC nanoparticle database."""

    species_id: int
    degrees_of_freedom: int


def load_sk_parameters(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load the JSON parameter/profile file."""
    with open(path) as f:
        return json.load(f)


def load_dre_parameters(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Backward-compatible alias for older production entrypoints."""
    return load_sk_parameters(path)


def dre_state_to_level_map(params: dict[str, Any]) -> dict[int, int]:
    """Map legacy 1-based state numbering onto 0-based Tm local levels."""
    return {
        int(row["dre_state"]): int(row["kmc_level"])
        for row in params["states"]
    }


def load_dre_channels(params: dict[str, Any]) -> list[DREChannel]:
    """Build channel objects from the JSON profile."""
    rates = params["cross_relaxation_rates_s^-1"]
    channels = []
    for row in params["dre_channels"]:
        channels.append(
            DREChannel(
                name=str(row["name"]),
                transition_a=tuple(row["transition_a"]),
                transition_b=tuple(row["transition_b"]),
                description=str(row["description"]),
                dre_rate_s=float(rates[row["name"]]),
            )
        )
    return channels


def load_site_records_from_np_db(np_db_path: str | Path) -> list[SiteRecord]:
    """Load site coordinates and species IDs from a nanoparticle SQLite database."""
    with sqlite3.connect(np_db_path) as con:
        rows = con.execute(
            "SELECT site_id, x, y, z, species_id FROM sites ORDER BY site_id"
        ).fetchall()
    if not rows:
        raise ValueError(f"No sites found in {np_db_path}")
    return [
        SiteRecord(
            site_id=int(site_id),
            x=float(x),
            y=float(y),
            z=float(z),
            species_id=int(species_id),
        )
        for site_id, x, y, z, species_id in rows
    ]


def load_species_records_from_np_db(np_db_path: str | Path) -> list[SpeciesRecord]:
    """Load species degrees of freedom from a nanoparticle SQLite database."""
    with sqlite3.connect(np_db_path) as con:
        rows = con.execute(
            "SELECT species_id, degrees_of_freedom FROM species ORDER BY species_id"
        ).fetchall()
    if not rows:
        raise ValueError(f"No species found in {np_db_path}")
    return [
        SpeciesRecord(
            species_id=int(species_id),
            degrees_of_freedom=int(degrees_of_freedom),
        )
        for species_id, degrees_of_freedom in rows
    ]


def count_sites_by_species(np_db_path: str | Path) -> dict[int, int]:
    """Return the number of sites assigned to each species ID."""
    counts: dict[int, int] = {}
    for row in load_site_records_from_np_db(np_db_path):
        counts[row.species_id] = counts.get(row.species_id, 0) + 1
    return counts


def distance_factor(
    distance_nm: np.ndarray,
    factor_type: str,
    cutoff_nm: float,
) -> np.ndarray:
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
    species_ids: Iterable[int] | None = None,
) -> GeometryFactor:
    """Compute the ordered pair normalization used for mean-field-equivalent reporting."""
    site_records = load_site_records_from_np_db(np_db_path)
    if species_ids is not None:
        allowed = {int(species_id) for species_id in species_ids}
        site_records = [
            row for row in site_records if row.species_id in allowed
        ]
    if not site_records:
        raise ValueError("No sites remain after applying the geometry species filter")
    sites = np.asarray([(row.x, row.y, row.z) for row in site_records], dtype=float)
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


def build_spectral_kinetics(
    params: dict[str, Any],
    excitation_power_w_cm2: float,
    tm_fraction: float | None = None,
    surface_species: str | None = None,
    surface_fraction: float = 0.0,
    surface_n_levels: int | None = None,
) -> tuple[Dopant, SpectralKinetics]:
    """Build NanoParticleTools SpectralKinetics using JSON defaults."""
    sim_defaults = params["simulation_defaults"]
    sk_defaults = params["spectral_kinetics_defaults"]
    if tm_fraction is None:
        tm_fraction = float(sim_defaults["tm_fraction_for_semi_empirical"])

    dopant = Dopant("Tm", tm_fraction)
    dopants = [dopant]
    if surface_species and surface_fraction > 0:
        if surface_n_levels is None:
            dopants.append(Dopant(str(surface_species), float(surface_fraction)))
        else:
            dopants.append(
                Dopant(
                    str(surface_species),
                    float(surface_fraction),
                    n_levels=int(surface_n_levels),
                )
            )
    sk = SpectralKinetics(
        dopants,
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


def species_level_slices(sk: SpectralKinetics) -> dict[int, slice]:
    """Return the combined-matrix slice occupied by each species."""
    slices: dict[int, slice] = {}
    start = 0
    for species_id, dopant in enumerate(sk.dopants):
        stop = start + int(dopant.n_levels)
        slices[species_id] = slice(start, stop)
        start = stop
    return slices


def combined_level_to_species_local(
    sk: SpectralKinetics,
    combined_level: int,
) -> tuple[int, int]:
    """Map one combined SpectralKinetics level index to species/local indices."""
    index = int(combined_level)
    for species_id, species_slice in species_level_slices(sk).items():
        if species_slice.start <= index < species_slice.stop:
            return species_id, index - int(species_slice.start)
    raise ValueError(f"Combined level {combined_level} is outside the SpectralKinetics map")


def build_kmc_default_absorption_cross_sections(
    params: dict[str, Any],
    tm_fraction: float | None = None,
) -> dict[str, float]:
    """Compute NPT-style effective absorption cross sections for the pump channels."""
    dopant, sk = build_spectral_kinetics(
        params,
        excitation_power_w_cm2=1.0,
        tm_fraction=tm_fraction,
    )
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
        "sigma_ESA": effective_sigma(1, 5),
    }


def order_channel(
    channel: DREChannel,
    dopant: Dopant,
    state_to_level: dict[int, int],
) -> OrderedChannel:
    """Order a two-ion channel as donor downhill plus acceptor uphill."""
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
    """Compute the NPT semi-empirical pair coefficient for a channel."""
    line_strengths = dopant.get_line_strength_matrix()
    di = ordered.donor_initial_level
    dj = ordered.donor_final_level
    ai = ordered.acceptor_initial_level
    aj = ordered.acceptor_final_level

    donor_energy_change = (
        dopant.energy_levels[dj].energy - dopant.energy_levels[di].energy
    )
    acceptor_energy_change = (
        dopant.energy_levels[aj].energy - dopant.energy_levels[ai].energy
    )
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


def build_full_npt_interactions(
    spectral_kinetics: SpectralKinetics,
) -> list[dict[str, Any]]:
    """Return the full exported NPT interaction list."""
    return list(get_all_interactions(spectral_kinetics).values())


def build_surface_one_site_rates(
    spectral_kinetics: SpectralKinetics,
    species_id: int,
) -> list[dict[str, Any]]:
    """Build sink-only one-site rows for the exported Surface species."""
    species_slice = species_level_slices(spectral_kinetics)[int(species_id)]
    nr = spectral_kinetics.non_radiative_rate_matrix[species_slice, species_slice]
    rows: list[dict[str, Any]] = []
    for initial in range(1, min(nr.shape[0], nr.shape[1])):
        base_rate = float(nr[initial, initial - 1])
        rows.append(
            {
                "type": "NR",
                "left": int(initial),
                "right": int(initial - 1),
                "base_dre_rate_s": base_rate,
                "rate_scale_factor": 1.0,
                "dre_rate_s": base_rate,
                "species_id": int(species_id),
                "species_name": str(spectral_kinetics.dopants[species_id].symbol),
            }
        )
    return rows


def build_tm_surface_energy_transfer_rates(
    spectral_kinetics: SpectralKinetics,
    tm_species_id: int = 0,
    surface_species_id: int = 1,
    max_local_state: int | None = None,
) -> list[dict[str, Any]]:
    """Extract NPT-exported Tm->Surface ET rows using local species-level indices."""
    rows: list[dict[str, Any]] = []
    for combined_di, combined_dj, combined_ai, combined_aj, rate in np.asarray(
        spectral_kinetics.energy_transfer_rate_matrix
    ):
        donor_species_id, donor_initial = combined_level_to_species_local(
            spectral_kinetics, int(combined_di)
        )
        donor_species_id_final, donor_final = combined_level_to_species_local(
            spectral_kinetics, int(combined_dj)
        )
        acceptor_species_id, acceptor_initial = combined_level_to_species_local(
            spectral_kinetics, int(combined_ai)
        )
        acceptor_species_id_final, acceptor_final = combined_level_to_species_local(
            spectral_kinetics, int(combined_aj)
        )
        if donor_species_id != int(tm_species_id):
            continue
        if donor_species_id_final != int(tm_species_id):
            continue
        if acceptor_species_id != int(surface_species_id):
            continue
        if acceptor_species_id_final != int(surface_species_id):
            continue
        if max_local_state is not None:
            if max(donor_initial, donor_final, acceptor_initial, acceptor_final) > int(
                max_local_state
            ):
                continue
        rows.append(
            {
                "channel_name": (
                    f"SQ Tm({donor_initial}->{donor_final}) "
                    f"Surface({acceptor_initial}->{acceptor_final})"
                ),
                "kmc_tuple": (
                    int(donor_initial),
                    int(donor_final),
                    int(acceptor_initial),
                    int(acceptor_final),
                ),
                "rate_nm6_s": float(rate) * 1.0e42,
                "species_id_1": int(tm_species_id),
                "species_id_2": int(surface_species_id),
                "species_name_1": str(spectral_kinetics.dopants[tm_species_id].symbol),
                "species_name_2": str(spectral_kinetics.dopants[surface_species_id].symbol),
                "source": "NPT Surface ET",
            }
        )
    rows.sort(key=lambda row: row["kmc_tuple"])
    return rows
