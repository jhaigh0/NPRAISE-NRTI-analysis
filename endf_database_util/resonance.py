"""Reconstruct neutron total cross-sections from ENDF resolved resonance data.

The tabulated cross-sections in ENDF MF=3 are far too coarse in the
resonance region to model the sharp dips seen in the nGEM spectra: for most
nuclides there are only zero to thirty tabulated points between 1 and 500 eV.
The resolved resonance parameters in MF=2, MT=151 *do* resolve the
resonances, so the cross-section is reconstructed from them here.

The implementation follows ENDF-102 and the formulation in LA-UR-12-27079,
matching the reference implementation in OpenMC
(``openmc/data/reconstruct.pyx``, v0.14.0).

Supported formalisms
---------------------
- LRF=1 single-level Breit-Wigner
- LRF=2 multi-level Breit-Wigner
- LRF=3 Reich-Moore

The R-matrix formalism (LRF=7) is parsed by
:mod:`endf_database_util.local_endf` but is not reconstructed here; the
R-matrix resonance *energies* are still available for template matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .local_endf import ResonanceParameters

__all__ = [
    "CrossSectionResult",
    "penetration_shift",
    "phaseshift",
    "resonance_energies",
    "total_cross_section",
    "wave_number",
]


# CODATA 2014, as used by ENDF-102 / OpenMC.
_NEUTRON_MASS_ENERGY = 939.5654133e6  # eV/c^2
_NEUTRON_MASS_AMU = 1.00866491595  # amu
_HBAR_C = 197.3269788e5  # eV * b^0.5


def wave_number(a_ratio: float, energy: float) -> float:
    """Neutron wave number in the centre-of-mass system.

    ENDF-102 equation D.10.

    Parameters
    ----------
    a_ratio : float
        Ratio of target mass to neutron mass.
    energy : float
        Energy in eV.

    Returns
    -------
    float
        Wave number in :math:`\\mathrm{b}^{-1/2}`.
    """
    return (
        a_ratio
        / (a_ratio + 1.0)
        * math.sqrt(2.0 * _NEUTRON_MASS_ENERGY * abs(energy))
        / _HBAR_C
    )


def penetration_shift(l_value: int, rho: float) -> tuple[float, float]:
    """Penetration and shift factors (ENDF-102 equations D.11 and D.12).

    Parameters
    ----------
    l_value : int
        Orbital angular momentum quantum number.
    rho : float
        Product of the wave number and the channel radius.

    Returns
    -------
    (float, float)
        Penetration factor and shift factor.
    """
    if l_value == 0:
        return rho, 0.0
    if l_value == 1:
        den = 1.0 + rho**2
        return rho**3 / den, -1.0 / den
    if l_value == 2:
        den = 9.0 + 3.0 * rho**2 + rho**4
        return rho**5 / den, -(18.0 + 3.0 * rho**2) / den
    if l_value == 3:
        den = 225.0 + 45.0 * rho**2 + 6.0 * rho**4 + rho**6
        return rho**7 / den, -(675.0 + 90.0 * rho**2 + 6.0 * rho**4) / den
    if l_value == 4:
        den = 11025.0 + 1575.0 * rho**2 + 135.0 * rho**4 + 10.0 * rho**6 + rho**8
        return rho**9 / den, -(
            44100.0 + 4725.0 * rho**2 + 270.0 * rho**4 + 10.0 * rho**6
        ) / den
    raise ValueError(f"Unsupported l value: {l_value}")


def penetration_shift_array(
    l_value: int, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised form of :func:`penetration_shift`.

    Parameters
    ----------
    l_value : int
        Orbital angular momentum quantum number.
    rho : numpy.ndarray
        Products of the wave number and the channel radius.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Penetration factor and shift factor, element by element.
    """
    rho = np.asarray(rho, dtype=float)
    if l_value == 0:
        return rho, np.zeros_like(rho)
    if l_value == 1:
        den = 1.0 + rho**2
        return rho**3 / den, -1.0 / den
    if l_value == 2:
        den = 9.0 + 3.0 * rho**2 + rho**4
        return rho**5 / den, -(18.0 + 3.0 * rho**2) / den
    if l_value == 3:
        den = 225.0 + 45.0 * rho**2 + 6.0 * rho**4 + rho**6
        return rho**7 / den, -(675.0 + 90.0 * rho**2 + 6.0 * rho**4) / den
    if l_value == 4:
        den = 11025.0 + 1575.0 * rho**2 + 135.0 * rho**4 + 10.0 * rho**6 + rho**8
        return rho**9 / den, -(
            44100.0 + 4725.0 * rho**2 + 270.0 * rho**4 + 10.0 * rho**6
        ) / den
    raise ValueError(f"Unsupported l value: {l_value}")


def phaseshift(l_value: int, rho: float) -> float:
    """Hard-sphere phase shift (ENDF-102 equation D.13)."""
    if l_value == 0:
        return rho
    if l_value == 1:
        return rho - math.atan(rho)
    if l_value == 2:
        return rho - math.atan(3.0 * rho / (3.0 - rho**2))
    if l_value == 3:
        return rho - math.atan((15.0 * rho - rho**3) / (15.0 - 6.0 * rho**2))
    if l_value == 4:
        return rho - math.atan(
            (105.0 * rho - 10.0 * rho**3) / (105.0 - 45.0 * rho**2 + rho**4)
        )
    raise ValueError(f"Unsupported l value: {l_value}")


def phaseshift_array(l_value: int, rho: np.ndarray) -> np.ndarray:
    """Vectorised form of :func:`phaseshift`."""
    rho = np.asarray(rho, dtype=float)
    if l_value == 0:
        return rho
    if l_value == 1:
        return rho - np.arctan(rho)
    if l_value == 2:
        return rho - np.arctan(3.0 * rho / (3.0 - rho**2))
    if l_value == 3:
        return rho - np.arctan((15.0 * rho - rho**3) / (15.0 - 6.0 * rho**2))
    if l_value == 4:
        return rho - np.arctan(
            (105.0 * rho - 10.0 * rho**3) / (105.0 - 45.0 * rho**2 + rho**4)
        )
    raise ValueError(f"Unsupported l value: {l_value}")


@dataclass
class CrossSectionResult:
    """Reconstructed partial cross-sections."""

    energy: np.ndarray
    """Energies in eV."""
    total: np.ndarray
    """Total cross-section in barns."""
    elastic: np.ndarray
    """Elastic scattering cross-section in barns."""
    capture: np.ndarray
    """Radiative capture cross-section in barns."""
    fission: np.ndarray
    """Fission cross-section in barns."""


def _channel_radius(a_ratio: float) -> float:
    """Channel radius from ENDF-102 equation D.14.

    The result is in units of :math:`10^{-13}` m; combined with a wave number
    in :math:`\\mathrm{b}^{-1/2}` this gives a dimensionless :math:`\\rho`.
    """
    return 0.123 * (_NEUTRON_MASS_AMU * a_ratio) ** (1.0 / 3.0) + 0.08


def _k_squared(a_ratio: float, energies: np.ndarray) -> np.ndarray:
    """Return :math:`k^2` for each energy (vectorised)."""
    energies = np.asarray(energies, dtype=float)
    k = (
        a_ratio
        / (a_ratio + 1.0)
        * np.sqrt(2.0 * _NEUTRON_MASS_ENERGY * np.abs(energies))
        / _HBAR_C
    )
    return k * k


@dataclass
class _Level:
    """A prepared resonance, with evaluation-energy-independent quantities."""

    energy: float
    j: float
    l: int
    gn: float
    gg: float
    gf: float
    p_r: float
    s_r: float


def _prepare(params: ResonanceParameters) -> dict[int, dict]:
    """Pre-compute per-resonance penetrability and shift factors.

    Returns a mapping from :math:`l` to the channel/scattering radii and the
    prepared levels belonging to that :math:`l`.
    """
    grouped: dict[int, list] = {}
    for resonance in params.resonances:
        grouped.setdefault(resonance.l, []).append(resonance)

    prepared = {}
    for l_value, resonances in grouped.items():
        group = params.l_groups.get(l_value)
        if group is not None:
            a_ratio = group.atomic_weight_ratio
            scattering_radius = group.scattering_radius
            naps = group.naps
        else:
            a_ratio = params.atomic_weight_ratio
            scattering_radius = params.scattering_radius
            naps = 0

        # ENDF-102 equation D.14 gives the standard channel radius. The NAPS
        # flag selects which radius is used instead.
        standard_radius = _channel_radius(a_ratio)
        if group is not None and group.channel_radius is not None:
            channel_radius = group.channel_radius
        elif naps in (1, 2):
            channel_radius = scattering_radius
        else:
            channel_radius = standard_radius

        levels = []
        for resonance in resonances:
            energy = resonance.energy
            if energy <= 0.0:
                # Bound state: no contribution at positive energies.
                continue
            k = wave_number(a_ratio, energy)
            p_r, s_r = penetration_shift(l_value, k * channel_radius)
            levels.append(
                _Level(
                    energy=energy,
                    j=resonance.j,
                    l=l_value,
                    gn=resonance.widths["n"],
                    gg=resonance.widths["g"],
                    gf=resonance.widths["f"],
                    p_r=p_r,
                    s_r=s_r,
                )
            )
        prepared[l_value] = {
            "a_ratio": a_ratio,
            "channel_radius": channel_radius,
            "scattering_radius": scattering_radius,
            "levels": levels,
        }
    return prepared


def _reconstruct_mlbw(
    params: ResonanceParameters, energies: np.ndarray
) -> CrossSectionResult:
    """Reconstruct cross-sections using the MLBW formalism.

    Follows LA-UR-12-27079 equation 40, as implemented in OpenMC's
    ``reconstruct_mlbw``.
    """
    return _reconstruct_mlbw_from_levels(params, energies)


def _reconstruct_mlbw_from_levels(
    params: ResonanceParameters, energies: np.ndarray
) -> CrossSectionResult:
    """Reconstruct MLBW cross-sections from prepared resonance levels.

    The energy loop is vectorised over the energy axis, and the resonances of
    each :math:`l` are looped over in Python. This keeps the cost linear in
    the number of energies without building an energy-by-resonance array.
    """
    target_spin = params.target_spin
    prepared = _prepare(params)

    energies = np.asarray(energies, dtype=float)
    n = len(energies)
    elastic = np.zeros(n)
    capture = np.zeros(n)
    fission = np.zeros(n)
    a_ratio = params.atomic_weight_ratio

    for l_value, group in prepared.items():
        channel_radius = group["channel_radius"]
        scattering_radius = group["scattering_radius"]
        a_ratio = group["a_ratio"]
        levels = group["levels"]
        if not levels:
            continue

        # Range of total angular momenta allowed for this l (LA-UR-12-27079 eq. 41).
        j_min = abs(abs(target_spin - l_value) - 0.5)
        j_max = target_spin + l_value + 0.5
        n_j = int(j_max - j_min + 1)

        # Statistical weights g_j and the D_l factor (LA-UR-12-27079 eq. 43).
        g_j = np.empty(n_j)
        d_l = 2 * l_value + 1
        for ij in range(n_j):
            j = j_min + ij
            g_j[ij] = (2 * j + 1) / (4 * target_spin + 2)
            d_l -= g_j[ij]

        # Map each level onto its index in the J range.
        indexed = []
        for level in levels:
            ij = int(level.j - j_min)
            if 0 <= ij < n_j:
                indexed.append((level, ij))
        if not indexed:
            continue

        # Wave number and the l-dependent factors, evaluated on the whole
        # energy axis at once.
        k = (
            a_ratio
            / (a_ratio + 1.0)
            * np.sqrt(2.0 * _NEUTRON_MASS_ENERGY * np.abs(energies))
            / _HBAR_C
        )
        positive = energies > 0.0
        k = np.where(positive, k, 0.0)

        p, s = penetration_shift_array(l_value, k * channel_radius)
        phi = phaseshift_array(l_value, k * scattering_radius)
        cos_2phi = np.cos(2 * phi)
        sin_2phi = np.sin(2 * phi)

        # Accumulated terms of LA-UR-12-27079 equation 40, one array per J.
        sum_0 = np.zeros((n_j, n))
        sum_1 = np.zeros((n_j, n))

        for level, ij in indexed:
            g_n = level.gn
            g_g = level.gg
            g_f = level.gf

            # ENDF-102 equation D.7: energy-dependent neutron width.
            g_n_e = p * g_n / level.p_r
            g_t_e = g_n_e + g_g + g_f
            valid = g_t_e > 0.0
            if not np.any(valid):
                continue
            safe_g_t = np.where(valid, g_t_e, 1.0)

            # ENDF-102 equation D.9: shifted resonance energy.
            e_prime = level.energy + (level.s_r - s) / (2 * level.p_r) * g_n
            x = 2 * (energies - e_prime) / safe_g_t
            f = 2 * g_n_e / (safe_g_t * (1 + x * x))
            f = np.where(valid, f, 0.0)

            sum_0[ij] += f
            sum_1[ij] += f * x
            capture += f * g_j[ij] * g_g / safe_g_t
            if g_f > 0.0:
                fission += f * g_j[ij] * g_f / safe_g_t

        # Elastic scattering (LA-UR-12-27079 equation 40).
        elastic += np.sum(
            g_j[:, None]
            * ((1 - cos_2phi[None, :] - sum_0) ** 2 + (sin_2phi[None, :] + sum_1) ** 2),
            axis=0,
        )
        elastic += 2 * d_l * (1 - cos_2phi)

    # Apply the 1/k^2 normalisation once, after summing over all l.
    k_sq = _k_squared(a_ratio, energies)
    elastic *= math.pi / k_sq
    capture *= 2 * math.pi / k_sq
    fission *= 2 * math.pi / k_sq

    total = elastic + capture + fission
    return CrossSectionResult(
        energy=energies,
        total=total,
        elastic=elastic,
        capture=capture,
        fission=fission,
    )


def _invert_symmetric_3x3(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """Invert a stack of symmetric 3x3 matrices.

    The collision matrix ``I - K`` of the Reich-Moore formalism is
    symmetric, so its inverse is computed from the cofactors. This avoids
    the failure of a batched LU inversion when a single energy makes the
    matrix singular.

    Parameters
    ----------
    a, b, c, d, e, f : numpy.ndarray
        The elements of the symmetric matrix
        ``[[a, b, c], [b, d, e], [c, e, f]]``.

    Returns
    -------
    numpy.ndarray
        The inverses, with shape ``(n, 3, 3)``.
    """
    det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    singular = np.abs(det) < 1e-300
    det = np.where(singular, 1.0, det)

    inv = np.empty((a.shape[0], 3, 3), dtype=complex)
    inv[:, 0, 0] = (d * f - e * e) / det
    inv[:, 1, 1] = (a * f - c * c) / det
    inv[:, 2, 2] = (a * d - b * b) / det
    inv[:, 0, 1] = (c * e - b * f) / det
    inv[:, 0, 2] = (b * e - c * d) / det
    inv[:, 1, 2] = (b * c - a * e) / det
    inv[:, 1, 0] = inv[:, 0, 1]
    inv[:, 2, 0] = inv[:, 0, 2]
    inv[:, 2, 1] = inv[:, 1, 2]
    # A singular matrix contributes nothing, matching the behaviour of the
    # reference implementation, which skips such energies.
    inv[singular] = 0.0
    return inv


def _reconstruct_rm(
    params: ResonanceParameters, energies: np.ndarray
) -> CrossSectionResult:
    """Reconstruct cross-sections using the Reich-Moore formalism.

    Follows ENDF-102 equations D.23-D.28, matching the reference
    implementation in OpenMC's ``reconstruct_rm``. The collision matrix
    ``K`` is built from the resonance parameters for each :math:`(l, J)`
    group and inverted to give the scattering matrix ``U``.
    """
    target_spin = params.target_spin
    a_ratio = params.atomic_weight_ratio
    prepared = _prepare(params)

    energies = np.asarray(energies, dtype=float)
    n = len(energies)
    elastic = np.zeros(n)
    capture = np.zeros(n)
    fission = np.zeros(n)
    total = np.zeros(n)

    # Energies at or below zero carry no contribution, but are replaced by a
    # positive value so that the vectorised expressions stay finite.
    positive = energies > 0.0
    safe_energies = np.where(positive, energies, 1.0)

    # The channel spin s ranges over |I - 1/2| < s < I + 1/2.
    s_min = abs(target_spin - 0.5)
    s_max = target_spin + 0.5
    num_s = int(s_max - s_min + 1)

    for l_value, group in prepared.items():
        channel_radius = group["channel_radius"]
        scattering_radius = group["scattering_radius"]
        a_ratio = group["a_ratio"]
        levels = group["levels"]
        if not levels:
            continue

        # Group the levels by |J|; the RM treatment couples the levels
        # within each (l, |J|) group.
        by_j: dict[int, list[_Level]] = {}
        for level in levels:
            by_j.setdefault(int(abs(level.j)), []).append(level)

        k = (
            a_ratio
            / (a_ratio + 1.0)
            * np.sqrt(2.0 * _NEUTRON_MASS_ENERGY * safe_energies)
            / _HBAR_C
        )
        p, _ = penetration_shift_array(l_value, k * channel_radius)
        phi = phaseshift_array(l_value, k * scattering_radius)

        # Common factor on the collision matrix terms (ENDF-102 eq. D.27).
        u_bar = np.exp(-2j * phi)

        for i_s in range(num_s):
            s = i_s + s_min
            j_min = abs(l_value - s)
            j_max = l_value + s
            num_j = int(j_max - j_min + 1)

            for i_j in range(num_j):
                j_total = i_j + j_min
                g_j = (2 * j_total + 1) / (4 * target_spin + 2)

                # The sign of J selects the channel spin.
                selected = []
                for level in by_j.get(int(j_total), []):
                    if l_value > 0 and (
                        (level.j < 0 and s != s_min) or (level.j > 0 and s != s_max)
                    ):
                        continue
                    selected.append(level)

                if not selected:
                    continue

                # Build the upper triangle of the K matrix (ENDF-102 eq. D.28).
                k00 = np.zeros(n, dtype=complex)
                k01 = np.zeros(n, dtype=complex)
                k02 = np.zeros(n, dtype=complex)
                k11 = np.zeros(n, dtype=complex)
                k12 = np.zeros(n, dtype=complex)
                k22 = np.zeros(n, dtype=complex)
                has_fission = False

                for level in selected:
                    g_n = level.gn
                    g_g = level.gg
                    g_fa = level.gf
                    g_fb = level.gf

                    # Neutron width at the evaluation energy.
                    g_n_e = np.sqrt(p * g_n / level.p_r)
                    factor = 0.5j / (level.energy - safe_energies - 0.5j * g_g)

                    k00 += g_n_e * g_n_e * factor
                    if g_fa != 0.0 or g_fb != 0.0:
                        sign_a = -1 if g_fa < 0 else 1
                        sign_b = -1 if g_fb < 0 else 1
                        g_fa = sign_a * math.sqrt(abs(g_fa))
                        g_fb = sign_b * math.sqrt(abs(g_fb))
                        k01 += g_n_e * g_fa * factor
                        k02 += g_n_e * g_fb * factor
                        k11 += g_fa * g_fa * factor
                        k12 += g_fa * g_fb * factor
                        k22 += g_fb * g_fb * factor
                        has_fission = True

                if has_fission:
                    # Collision matrix inversion (ENDF-102 eq. D.27).
                    kmat = np.stack(
                        [
                            np.stack([k00, k01, k02], axis=-1),
                            np.stack([k01, k11, k12], axis=-1),
                            np.stack([k02, k12, k22], axis=-1),
                        ],
                        axis=-2,
                    )
                    imat = _invert_symmetric_3x3(
                        1 - kmat[:, 0, 0],
                        -kmat[:, 0, 1],
                        -kmat[:, 0, 2],
                        1 - kmat[:, 1, 1],
                        -kmat[:, 1, 2],
                        1 - kmat[:, 2, 2],
                    )
                    u_00 = u_bar * (2 * imat[:, 0, 0] - 1)
                    elastic += g_j * np.abs(1 - u_00) ** 2
                    total += 2 * g_j * (1 - u_00.real)
                    fission += (
                        4
                        * g_j
                        * (np.abs(imat[:, 1, 0]) ** 2 + np.abs(imat[:, 2, 0]) ** 2)
                    )
                else:
                    u_ = u_bar * (2 / (1 - k00) - 1)
                    precise = (np.abs(k00.real) < 3e-4) & (np.abs(phi) < 3e-4)
                    if np.any(precise):
                        # Avoid loss of precision when U is close to unity.
                        k_r = k00.real
                        k_i = k00.imag
                        x = (
                            2
                            * (
                                -k_r
                                + (k_r * k_r + k_i * k_i) * (1 - phi * phi)
                                + phi * phi
                                - np.sin(2 * phi) * k_i
                            )
                            / ((1 - k_r) * (1 - k_r) + k_i * k_i)
                        )
                        total += np.where(precise, 2 * g_j * x, 0.0)
                        elastic += np.where(precise, g_j * (x * x + u_.imag**2), 0.0)
                    general = ~precise
                    total += np.where(general, 2 * g_j * (1 - u_.real), 0.0)
                    elastic += np.where(general, g_j * np.abs(1 - u_) ** 2, 0.0)

    # Capture is the difference of the other cross-sections (ENDF-102 eq. D.25).
    capture[:] = total - elastic - fission

    k_sq = _k_squared(a_ratio, energies)
    elastic *= math.pi / k_sq
    capture *= math.pi / k_sq
    fission *= math.pi / k_sq

    return CrossSectionResult(
        energy=energies,
        total=elastic + capture + fission,
        elastic=elastic,
        capture=capture,
        fission=fission,
    )


def total_cross_section(
    params: ResonanceParameters, energies: np.ndarray
) -> CrossSectionResult:
    """Reconstruct the total cross-section from resolved resonance parameters.

    Parameters
    ----------
    params : ResonanceParameters
        Resolved resonance parameters, as returned by
        :meth:`endf_database_util.local_endf.LocalENDFDatabase.resonance_parameters`.
    energies : numpy.ndarray
        Energies in eV at which to evaluate the cross-section.

    Returns
    -------
    CrossSectionResult
        The reconstructed partial and total cross-sections in barns.
    """
    energies = np.asarray(energies, dtype=float)
    if params.formalism == 3:
        return _reconstruct_rm(params, energies)
    return _reconstruct_mlbw(params, energies)


def resonance_energies(
    params: ResonanceParameters, energy_min: float = 0.0, energy_max: float = 1e9
) -> np.ndarray:
    """Return the resolved resonance energies within a range."""
    energies = np.array([r.energy for r in params.resonances])
    keep = (energies > energy_min) & (energies <= energy_max)
    return np.sort(energies[keep])
