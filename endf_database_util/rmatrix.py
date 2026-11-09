"""R-matrix (LRF=7) resolved resonance reconstruction.

Some nuclides -- notably copper-63 and copper-65 -- are evaluated in the
R-matrix limited formalism rather than the Breit-Wigner family, so they
cannot be treated with the MLBW or Reich-Moore formulas directly.

This module converts the R-matrix parameters into an equivalent set of
Breit-Wigner levels and reconstructs the cross-section from them. The
conversion uses the partial widths given in the R-matrix evaluation
directly, with the neutron width made energy dependent through the channel
penetrability, exactly as in the multi-level Breit-Wigner formalism
(ENDF-102 equation D.7).

The approximation is that interference *between* resonances of the same
spin group is neglected, i.e. the R-matrix is treated as a sum of
independent Breit-Wigner resonances. For the copper isotopes this is
accurate in the region of interest, where the relevant resonances are
separated by many times their widths.

Reference
---------
ENDF-102, sections D.2-D.14 (Breit-Wigner formalism) and D.29-D.33
(R-matrix formalism).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .local_endf import (
    LGroup,
    ResolvedResonance,
    ResonanceParameters,
    RMatrixParameters,
)
from .resonance import CrossSectionResult, _reconstruct_mlbw_from_levels

__all__ = ["rmatrix_resonance_energies", "rmatrix_total_cross_section"]

# CODATA 2014.
_NEUTRON_MASS_ENERGY = 939.5654133e6  # eV/c^2
_HBAR_C = 197.3269788e5  # eV * b^0.5


@dataclass
class _RMLevel:
    """A resonance expressed in Breit-Wigner terms."""

    energy: float
    j: float
    l: int
    gn: float
    gg: float


def _find_channel(
    channels: np.ndarray, particle_pairs: np.ndarray, predicate
) -> int | None:
    """Index of the first *channel* whose particle pair satisfies ``predicate``.

    The width arrays are indexed by channel, while each channel record refers
    to a particle pair through its PPI (particle pair index) field, so the
    two must not be confused.
    """
    for c, channel in enumerate(channels):
        pair = particle_pairs[int(channel[0]) - 1]
        if predicate(pair):
            return c
    return None


def _target_info(params: RMatrixParameters) -> tuple[float, float]:
    """Return the target atomic weight ratio and intrinsic spin.

    In the R-matrix particle-pair record the neutron is the particle with
    ``MA == 1``. For that pair, ``MB`` is the target's atomic weight ratio
    and ``IB`` is the target spin with its parity sign.
    """
    a_ratio = 1.0
    target_spin = 0.0
    for pair in params.particle_pairs:
        if pair[0] == 1:
            a_ratio = float(pair[1])
            target_spin = abs(float(pair[5]))
            break
    return a_ratio, target_spin


def rmatrix_levels(params: RMatrixParameters) -> tuple[list[_RMLevel], dict[int, dict]]:
    """Convert R-matrix parameters into Breit-Wigner levels.

    Returns
    -------
    (list, dict)
        The levels and, for each :math:`l`, the channel radius, scattering
        radius and atomic weight ratio to use.
    """
    pairs = params.particle_pairs
    a_ratio, _ = _target_info(params)

    levels: list[_RMLevel] = []
    radii: dict[int, dict] = {}

    for group in params.spin_groups:
        channels = group["channels"]
        if len(channels) == 0:
            continue

        # The elastic channel is the one whose particle pair has MA == 1
        # (the neutron); the capture channel is the one whose pair has the
        # capture reaction MT == 102.
        neutron_channel = _find_channel(channels, pairs, lambda p: p[0] == 1)
        capture_channel = _find_channel(channels, pairs, lambda p: p[9] == 102)
        if neutron_channel is None:
            continue

        neutron = channels[neutron_channel]
        l_value = int(neutron[1])
        channel_radius = float(neutron[4])  # APE, effective radius
        scattering_radius = float(neutron[5])  # APT, true radius
        radii.setdefault(
            l_value,
            {
                "channel_radius": channel_radius,
                "scattering_radius": scattering_radius,
                "atomic_weight_ratio": a_ratio,
            },
        )

        # The spin group gives the total angular momentum. A value of zero
        # means only the parity is given; for a half-integer target spin the
        # lowest allowed J is then 1/2.
        j_value = float(group["J"])
        if j_value == 0.0:
            j_value = 0.5

        energies = np.asarray(group["energies"], dtype=float)
        widths = np.asarray(group["widths"], dtype=float)

        for r, resonance_energy in enumerate(energies):
            if resonance_energy <= 0.0:
                continue
            g_n = float(widths[r, neutron_channel])
            g_c = (
                float(widths[r, capture_channel])
                if capture_channel is not None
                else 0.0
            )
            if g_n <= 0.0 and g_c <= 0.0:
                continue
            levels.append(
                _RMLevel(
                    energy=float(resonance_energy),
                    j=abs(j_value),
                    l=l_value,
                    gn=g_n,
                    gg=g_c,
                )
            )

    return levels, radii


def rmatrix_total_cross_section(
    params: RMatrixParameters, energies: np.ndarray
) -> CrossSectionResult:
    """Reconstruct the total cross-section from R-matrix parameters.

    Parameters
    ----------
    params : RMatrixParameters
        R-matrix resonance parameters, as returned by
        :meth:`endf_database_util.local_endf.LocalENDFDatabase.resonance_parameters`.
    energies : numpy.ndarray
        Energies in eV at which to evaluate the cross-section.

    Returns
    -------
    CrossSectionResult
        The reconstructed partial and total cross-sections in barns.
    """
    levels, radii = rmatrix_levels(params)
    a_ratio, target_spin = _target_info(params)

    resonances = [
        ResolvedResonance(
            energy=level.energy,
            j=level.j,
            l=level.l,
            widths={
                "n": level.gn,
                "g": level.gg,
                "f": 0.0,
                "t": level.gn + level.gg,
            },
        )
        for level in levels
    ]

    l_groups = {
        l_value: LGroup(
            l=l_value,
            atomic_weight_ratio=values["atomic_weight_ratio"],
            scattering_radius=values["scattering_radius"],
            nro=0,
            naps=1,
            channel_radius=values["channel_radius"],
        )
        for l_value, values in radii.items()
    }

    equivalent = ResonanceParameters(
        symbol=params.symbol,
        formalism=7,
        target_spin=target_spin,
        scattering_radius=0.0,
        atomic_weight_ratio=a_ratio,
        resonances=resonances,
        l_groups=l_groups,
    )
    return _reconstruct_mlbw_from_levels(equivalent, energies)


def rmatrix_resonance_energies(
    params: RMatrixParameters, energy_min: float = 0.0, energy_max: float = 1e9
) -> np.ndarray:
    """Return the R-matrix resonance energies within a range."""
    levels, _ = rmatrix_levels(params)
    energies = np.array([level.energy for level in levels])
    keep = (energies > energy_min) & (energies <= energy_max)
    return np.sort(energies[keep])
