"""Fit a measured region against the local ENDF database.

This is the theoretical counterpart of :mod:`data_manipulation.fit_abundances`.
Where that module fits a measured spectrum against a dictionary of *measured*
spectra, this one fits it against cross-section templates reconstructed from
the ENDF evaluation of each element.

The physics is the transmission law. A sample of areal density :math:`N`
atoms per unit area and thickness :math:`t` attenuates the neutron beam by

.. math:: T(E) = \\exp(-N\\,t\\,\\sigma(E))

so a measured profile and a cross-section can be compared directly without
knowing the neutron flux or the detector response: the flux cancels in the
ratio of the sample region to the empty region.

The fit is a non-negative least squares fit of the transmission

.. math:: T(E) = \\exp\\left(-\\sum_i c_i\\,\\sigma_i(E)\\right)

where the coefficients :math:`c_i` are the areal densities of the elements.
The detector's energy resolution is not known exactly, so a Gaussian
broadening of the templates is fitted alongside the compositions.

Example::

    from data_manipulation.fit_endf import fit_region

    result = fit_region(profile, ["Cu", "Zn"])
    print(result.dominant, result.fractions)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from endf_database_util.templates import (
    TemplateSet,
    build_fine_isotope_templates,
    build_fine_templates,
)

from .sample_metadata import SampleMetadata

__all__ = [
    "DEFAULT_BROADENINGS",
    "DEFAULT_REFINEMENT",
    "DEFAULT_RESOLUTION_FACTORS",
    "ENDFfitResult",
    "fit_region",
    "fit_regions",
    "main",
]


# The relative energy resolutions that are tried when fitting. The detector
# resolution is not known exactly, so the best one is found by the fit.
# The values are the fractional widths in energy, which is the same as the
# fractional widths in log energy for the narrow resonances of interest.
DEFAULT_BROADENINGS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.03)

# The factors by which the *calculated* flight-path resolution of the
# instrument are multiplied when the resolution is derived from the
# geometry rather than fitted as a constant.
#
# The time-of-flight binning gives a resolution that depends on the
# energy: the flight time falls as the square root of the energy, so the
# relative resolution dE/E = 2*dt/t grows as the square root of the
# energy. It is 0.0012 at 1 eV, 0.028 at 500 eV and 0.068 at 3000 eV.
# A constant broadening cannot reproduce that, and using one biases the
# fitted amount of material, because the sharp resonances at low energy
# are widened too much.
DEFAULT_RESOLUTION_FACTORS = (0.0, 1.0, 2.0)

# The flight path of the instrument, in metres, and the width of a time of
# flight bin, in seconds. These give the resolution of the measurement.
DEFAULT_FLIGHT_PATH = 22.17
DEFAULT_TIME_BIN = 1.0e-6

# How many times finer than the measurement grid the cross-sections are
# evaluated on. The measurement grid has a step of about 0.5 %, which is
# 1.15 eV at 230 eV, and that badly undersamples the narrow resonances of
# copper and niobium. Sixty times finer resolves every resonance.
DEFAULT_REFINEMENT = 60

# The width, in log energy, of the smoothing that separates the narrow
# resonances of a cross-section from its smooth continuum. A value of 0.25
# corresponds to a Gaussian of about 25 % of the energy, which is far wider
# than any resonance of interest but much narrower than the energy range.
DEFAULT_CONTINUUM_WIDTH = 0.25

# How far above the floor a bin has to be before it is used in a fit that
# leaves out the saturated bins. A bin at the floor carries no information
# about the amount of material, and a bin just above it carries very
# little, because the noise there is large.
_UNSATURATED_THRESHOLD = 3.0


@dataclass
class ENDFfitResult:
    """The result of fitting one region against a set of ENDF templates.

    Attributes:
        symbols: The elements that were fitted.
        areal_densities: The fitted areal density of each element, in
            barns inverse, so that a coefficient multiplies a cross-section
            in barns to give an optical thickness.
        fractions: The composition of the region. The areal densities are
            normalised to sum to one, which is meaningful because the
            elements are scaled to the same total cross-section first.
        dominant: The element that makes up the largest fraction.
        residual: The measured transmission minus the modelled one.
        rms_residual: The root mean square of the residual, in units of
            the transmission.
        broadening: The relative energy resolution that gave the best fit.
        correlations: The correlation of the measured profile with each
            template, which is a quick check on the identification.
        excluded: The elements that were left out of the fit, because they
            have no cross-section over the energy range and so carry no
            information.
        floor: The transmission floor that was assumed, which is the level at
            which the deepest dip saturates.
        saturated: Whether the deepest dip is clipped by the floor. When it
            is, the depth of the dip carries no information about the
            amount of material, so the fitted abundances are not
            quantitative.
        resonance_areal_densities: The areal density of each element that
            comes from its narrow resonances. This is the part of the
            abundance that is specific to the element, because the
            resonances sit at energies where no other element has one.
        continuum_areal_densities: The areal density of each element that
            comes from its smooth continuum. The continua of the elements
            are very similar to each other, so this part is degenerate and
            is not a reliable measure of the abundance.
        continuum_split: Whether the resonance and the continuum were
            fitted separately.
        fixed_composition: The composition that was held fixed, if the fit
            was constrained to a known composition.
        unsaturated_bins: The number of bins that were used in the fit. When
            the dip is saturated, the bins at the bottom of the dip carry
            no information about the amount of material, so they are left
            out of the fit.
        mass_fractions: The mass fraction of each element. This is the
            fraction of the sample by mass, which is what a composition is
            normally quoted as. It is ``None`` when the symbols are not
            real elements, because the atomic masses are then unknown.
        optical_thickness_fractions: The share of the modelled optical
            thickness that each element contributes. This is not the same
            as the mass fraction: an element with a small cross-section
            needs a large areal density to make the same dip, so its share
            of the optical thickness understates it.
        total_areal_density: The total number of atoms per barn of the
            region, when it was held fixed by the metadata.
    """

    symbols: list[str]
    areal_densities: np.ndarray
    fractions: np.ndarray
    residual: np.ndarray
    rms_residual: float
    broadening: float
    correlations: dict[str, float] = field(default_factory=dict)
    excluded: list[str] = field(default_factory=list)
    floor: float = 0.0
    saturated: bool = False
    resonance_areal_densities: np.ndarray | None = None
    continuum_areal_densities: np.ndarray | None = None
    continuum_split: bool = False
    fixed_composition: dict[str, float] | None = None
    unsaturated_bins: int = 0
    mass_fractions: np.ndarray | None = None
    optical_thickness_fractions: np.ndarray | None = None
    total_areal_density: float | None = None
    resolution_factor: float = 0.0
    isotopes: dict[str, int] = field(default_factory=dict)
    """The mass number of each entry, when the entries are isotopes."""
    atomic_fractions: np.ndarray | None = None
    """The number fraction of each entry, which is the share of the atoms.

    This is the usual way an isotopic composition is quoted. It is the
    same as the ``fractions`` when the entries are isotopes, because the
    isotopes of an element have nearly the same cross-section scale, but
    it is not the same in general: the ``fractions`` are shares of the
    modelled optical thickness.
    """

    @property
    def dominant(self) -> str:
        """The element that makes up the largest fraction of the region."""
        if not len(self.symbols):
            raise ValueError("The fit has no elements.")
        return self.symbols[int(np.argmax(self.fractions))]


def _transmission(profile: np.ndarray) -> np.ndarray:
    """Convert a measured profile into a transmission.

    The profile is normalised by its median, so that the transmission is one
    where the sample does not absorb. Bins with no counts are given a very
    small transmission rather than being dropped, so that the arrays stay
    the same length.
    """
    profile = np.asarray(profile, dtype=float)
    positive = profile[profile > 0]
    reference = np.median(positive) if positive.size else 1.0
    if reference <= 0:
        reference = 1.0
    return np.clip(profile / reference, 1e-8, None)


def _estimate_floor(profile: np.ndarray) -> float:
    """Estimate the transmission floor of a measured profile.

    A deep absorption dip does not reach zero counts: the dip is limited by
    the background and by the detector, so the transmission flattens out at
    some value above zero. The floor is estimated as the smallest
    transmission in the profile, which is where the deepest dip bottoms out.

    A dip that is not saturated has a sharp minimum, so the floor is only
    reported when the bottom of the dip is genuinely flat. Otherwise the
    floor is zero, and the fit can use the full depth of the dip.
    """
    transmission = _transmission(profile)
    minimum = float(np.min(transmission))
    if minimum <= 0.0:
        return 0.0

    # A saturated dip has a run of neighbouring bins that all sit at the
    # minimum. A sharp dip has only one or two such bins.
    tolerance = minimum * 1.05 + 1e-12
    at_minimum = transmission <= tolerance
    longest_run = 0
    run = 0
    for flag in at_minimum:
        run = run + 1 if flag else 0
        longest_run = max(longest_run, run)

    # The dip has to be deep enough to be clipped by the floor. A shallow
    # dip that happens to be flat is not saturated.
    if longest_run < 3 or minimum > 0.5:
        return 0.0
    return minimum


def _broaden(
    cross_section: np.ndarray, relative_width: float, log_step: float
) -> np.ndarray:
    """Apply a Gaussian broadening of the templates.

    The broadening is applied in log energy, because the detector resolves
    a fixed fraction of the energy rather than a fixed amount. A relative
    width of ``relative_width`` therefore corresponds to
    ``relative_width / log_step`` bins of the log energy axis.
    """
    if relative_width <= 0:
        return cross_section
    width_bins = relative_width / log_step
    return gaussian_filter1d(cross_section, width_bins, mode="nearest")


def _flight_resolution(
    energies: np.ndarray,
    flight_path: float = DEFAULT_FLIGHT_PATH,
    time_bin: float = DEFAULT_TIME_BIN,
) -> np.ndarray:
    """The relative energy resolution of the time-of-flight binning.

    The flight time from the moderator to the detector is
    :math:`t = L/v`, and the width of a time bin gives a relative energy
    resolution of :math:`\\Delta E/E = 2\\Delta t/t`. The resolution
    therefore grows as the square root of the energy.

    Parameters
    ----------
    energies : numpy.ndarray
        The energies in eV.
    flight_path : float
        The distance from the moderator to the detector, in metres.
    time_bin : float
        The width of a time-of-flight bin, in seconds.

    Returns:
        The relative resolution at each energy.
    """
    energies = np.asarray(energies, dtype=float)
    # The speed of a neutron at the given energy, in metres per second.
    speed = np.sqrt(2.0 * energies * 1.602176634e-19 / 1.67492749804e-27)
    flight_time = flight_path / speed
    return 2.0 * time_bin / flight_time


def _broaden_energy_dependent(
    cross_section: np.ndarray,
    energies: np.ndarray,
    factor: float,
    log_step: float,
) -> np.ndarray:
    """Broaden a cross-section with the energy-dependent resolution.

    The width of the smoothing follows the resolution of the instrument,
    so the sharp resonances at low energy are widened less than the broad
    ones at high energy. A single constant width cannot do that, and using
    one biases the fitted amount of material.

    Parameters
    ----------
    cross_section : numpy.ndarray
        The cross-section to broaden.
    energies : numpy.ndarray
        The energies of the cross-section, in eV.
    factor : float
        The factor by which the calculated resolution is multiplied.
    log_step : float
        The step of the log energy axis of the cross-section.

    Returns:
        The broadened cross-section.
    """
    if factor <= 0:
        return cross_section
    widths = _flight_resolution(energies) * factor
    broadened = np.empty_like(cross_section)
    index = 0
    count = len(energies)
    while index < count:
        width = widths[index]
        # Group the neighbouring energies whose width is nearly the same,
        # so that each block can be smoothed with a single width.
        end = index
        while end + 1 < count and abs(widths[end + 1] - width) < 0.15 * max(
            width, 1.0e-9
        ):
            end += 1
        block = cross_section[index : end + 1]
        width_bins = width / log_step
        if width_bins > 0.5:
            broadened[index : end + 1] = gaussian_filter1d(
                block, width_bins, mode="nearest"
            )
        else:
            broadened[index : end + 1] = block
        index = end + 1
    return broadened


def _atomic_mass(symbol: str) -> float:
    """Return the standard atomic weight of an element.

    Raises:
        ValueError: If the symbol is not a known element.
    """
    import periodictable

    try:
        return float(getattr(periodictable, symbol).mass)
    except AttributeError as exc:
        raise ValueError(f"'{symbol}' is not a known element symbol.") from exc


def _simplex_from_stick(u: np.ndarray) -> np.ndarray:
    """Map stick-breaking parameters onto a point of the simplex.

    The parameters are in the unit cube, and they are turned into weights
    that sum to one by taking a fraction of what is left at each step. The
    last weight is whatever remains. This keeps every weight positive
    without the constraint of a sum, which would make the fit ill
    conditioned.

    Parameters
    ----------
    u : numpy.ndarray
        The stick-breaking parameters, one fewer than the number of
        weights.

    Returns
    -------
    numpy.ndarray
        The weights, which sum to one.
    """
    u = np.asarray(u, dtype=float)
    weights = np.empty(u.size + 1, dtype=float)
    remainder = 1.0
    for index in range(u.size):
        weights[index] = remainder * u[index]
        remainder -= weights[index]
    weights[-1] = remainder
    return weights


def _stick_from_simplex(weights: np.ndarray) -> np.ndarray:
    """Turn weights on the simplex into stick-breaking parameters.

    This is the inverse of :func:`_simplex_from_stick`, and is used to
    build a starting point for the fit.

    Parameters
    ----------
    weights : numpy.ndarray
        The weights, which sum to one.

    Returns:
        The stick-breaking parameters.
    """
    weights = np.asarray(weights, dtype=float)
    parameters = np.empty(max(weights.size - 1, 0), dtype=float)
    remainder = 1.0
    for index in range(parameters.size):
        parameters[index] = weights[index] / remainder if remainder > 0 else 0.0
        remainder -= weights[index]
    return parameters


def _mass_fractions(
    symbols: list[str], areal_densities: np.ndarray
) -> np.ndarray | None:
    """Return the mass fraction of each element.

    The mass fraction is the share of the sample by mass, which is the
    usual way a composition is quoted. It is worked out from the areal
    densities and the atomic masses.

    Returns:
        The mass fractions, or ``None`` when the symbols are not real
        elements and the atomic masses are therefore unknown.
    """
    try:
        masses = np.asarray([_atomic_mass(symbol) for symbol in symbols])
    except ValueError:
        return None
    weighted = masses * np.asarray(areal_densities, dtype=float)
    total = float(weighted.sum())
    if total <= 0:
        return None
    return weighted / total


def _atomic_fractions(areal_densities: np.ndarray) -> np.ndarray | None:
    """Return the number fraction of each entry.

    The number fraction is the share of the atoms, which is the usual way
    an isotopic composition is quoted.

    Returns:
        The number fractions, or ``None`` when there is nothing to
        normalise.
    """
    densities = np.asarray(areal_densities, dtype=float)
    total = float(densities.sum())
    if total <= 0:
        return None
    return densities / total


def _isotope_mass(symbol: str, mass: int) -> float:
    """Return the mass of one isotope, in grams per mole.

    Parameters
    ----------
    symbol : str
        The element symbol.
    mass : int
        The mass number of the isotope.

    Returns:
        The mass of the isotope.

    Raises:
        ValueError: If the symbol is not a known element.
    """
    import periodictable

    try:
        element = getattr(periodictable, symbol)
    except AttributeError as exc:
        raise ValueError(f"'{symbol}' is not a known element symbol.") from exc
    try:
        return float(element[mass].mass)
    except Exception as exc:
        raise ValueError(f"'{symbol}-{mass}' is not a known isotope.") from exc


@dataclass
class _TemplateEntry:
    """One column of the fit: a template and what it is.

    Attributes:
        name: The name of the entry, e.g. ``'Ag'`` or ``'Ag-107'``.
        symbol: The element symbol.
        mass: The mass number, when the entry is an isotope.
        cross_section: The cross-section in barns.
        atomic_mass: The mass of the entry in grams per mole, which is
            the mass of the isotope for an isotope and the standard atomic
            weight for an element.
        is_isotope: Whether the entry is an isotope.
    """

    name: str
    symbol: str
    mass: int | None
    cross_section: np.ndarray
    atomic_mass: float | None
    is_isotope: bool


def _resolve_templates(
    symbols: list[str] | tuple[str, ...], templates: TemplateSet
) -> list[_TemplateEntry]:
    """Find the template of each of the requested entries.

    An entry is either an element, e.g. ``'Ag'``, or an isotope, e.g.
    ``'Ag-107'``. The element templates are used for the elements, and
    the isotope templates for the isotopes.

    Parameters
    ----------
    symbols : list of str
        The entries to resolve.
    templates : TemplateSet
        The templates to resolve them against.

    Returns:
        One entry for each of the requested names.

    Raises:
        ValueError: If one of the names is not in the template set.
    """
    entries: list[_TemplateEntry] = []
    for name in symbols:
        if name in templates.isotope_templates:
            template = templates.isotope_templates[name]
            entries.append(
                _TemplateEntry(
                    name=name,
                    symbol=template.symbol,
                    mass=template.mass,
                    cross_section=np.asarray(template.cross_section, dtype=float),
                    atomic_mass=_isotope_mass(template.symbol, template.mass),
                    is_isotope=True,
                )
            )
        elif name in templates:
            template = templates[name]
            try:
                atomic_mass = _atomic_mass(name)
            except ValueError:
                atomic_mass = None
            entries.append(
                _TemplateEntry(
                    name=name,
                    symbol=name,
                    mass=None,
                    cross_section=np.asarray(template.cross_section, dtype=float),
                    atomic_mass=atomic_mass,
                    is_isotope=False,
                )
            )
        else:
            raise ValueError(f"The templates do not contain '{name}'.")
    return entries


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation of two arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a @ a) * (b @ b))
    if denominator <= 0:
        return 0.0
    return float((a @ b) / denominator)


def _split_continuum(
    cross_section: np.ndarray, log_step: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split a cross-section into its resonance and continuum parts.

    A total cross-section is the sum of a smooth continuum, which falls
    roughly as one over the velocity, and a set of narrow resonances. The
    two are separated by smoothing the cross-section with a Gaussian that
    is much wider than any resonance but much narrower than the energy
    range: what survives the smoothing is the continuum, and what is
    removed is the resonance part.

    The separation matters because the wings of a resonance look exactly
    like a continuum. When the two are fitted together, the amplitude of
    the resonance and the amplitude of the continuum trade off against
    each other, and the fitted abundance of the element that owns the
    resonance becomes unreliable.

    Parameters
    ----------
    cross_section : numpy.ndarray
        The cross-section to split.
    log_step : float
        The step of the log energy axis of the cross-section.
    width : float
        The width of the smoothing, in log energy.

    Returns:
        The resonance part and the continuum part.
    """
    if width <= 0:
        return cross_section, np.zeros_like(cross_section)
    smoothed = gaussian_filter1d(cross_section, width / log_step, mode="nearest")
    return cross_section - smoothed, smoothed


def fit_region(
    profile: np.ndarray,
    symbols: list[str] | tuple[str, ...],
    *,
    energies: np.ndarray | None = None,
    templates: TemplateSet | None = None,
    broadenings: Sequence[float] = DEFAULT_BROADENINGS,
    floor: float | None = None,
    refinement: int = DEFAULT_REFINEMENT,
    separate_continuum: bool = False,
    continuum_width: float = DEFAULT_CONTINUUM_WIDTH,
    fixed_composition: dict[str, float] | None = None,
    unsaturated_only: bool = False,
    fixed_amount: SampleMetadata | None = None,
    resolution_factors: Sequence[float] = DEFAULT_RESOLUTION_FACTORS,
) -> ENDFfitResult:
    """Fit a measured profile against the ENDF cross-sections of elements.

    Parameters
    ----------
    profile : numpy.ndarray
        The measured profile, one value per bin.
    symbols : list of str
        The elements to fit the region against.
    energies : numpy.ndarray, optional
        The energies in eV of the profile bins. Only needed when the
        templates are not given.
    templates : TemplateSet, optional
        Pre-built templates. They are built from the database when not
        given, which is slow but cached.
    broadenings : sequence of float
        The relative energy resolutions to try.
    floor : float, optional
        The transmission floor of the measurement, which is the level at
        which the deepest dip saturates. When it is not given it is
        estimated from the profile. Set it to zero to fit the dip depth
        without a floor.
    refinement : int
        How many times finer than the measurement grid the cross-sections
        are evaluated on, when the templates have to be built. A finer grid
        is needed to resolve the narrow resonances of copper and niobium.
    separate_continuum : bool
        Whether to fit the narrow resonances and the smooth continuum of
        each element with separate coefficients. The wings of a resonance
        look exactly like a continuum, so when the two are fitted together
        the amplitude of the resonance trades off against the continuum,
        and the fitted abundance of the element that owns the resonance
        becomes unreliable. When this is set, the result also reports the
        areal density that comes from the resonances, which is the part of
        the abundance that is specific to the element.
    continuum_width : float
        The width, in log energy, of the smoothing that separates the
        resonances from the continuum.
    fixed_composition : dict, optional
        A known composition of the sample, as the mass fraction of each
        element. When this is given, the fit is constrained to that
        composition and only the overall amount of material is fitted. This
        is the reliable way to get an areal density out of a saturated
        dip, because the shape of the dip is then fixed and only its
        amplitude is fitted.
    unsaturated_only : bool
        Whether to leave the saturated bins out of the fit. The bottom of
        a saturated dip is pinned at the floor, so it carries no
        information about the amount of material, and including it biases
        the fit towards more material than there really is. The wings of
        the dip, where the transmission is still well above the floor, do
        carry the amplitude information.
    fixed_amount: SampleMetadata, optional
        Metadata that pins the amount of material in the sample, from its
        thickness and its composition. When this is given, the fit finds
        the *relative* abundance of the elements, with the total amount of
        material held fixed by the metadata. This answers the question
        "the thickness and the elements are known, but what are the
        ratios?".

        This is more robust than a free fit when the dip is saturated,
        because the depth of the dip is then fixed by the metadata rather
        than fitted, so the proportions are determined by the *shape* of
        the dip rather than by its depth.

        The amount of material is turned into an optical thickness through
        the density of the sample and the mean molar mass of its
        composition. The composition in the metadata is used for that, so
        it fixes the scale; the elements being fitted do not have to be
        the ones in the metadata.
    resolution_factors: sequence of float
        The factors by which the calculated flight-path resolution of the
        instrument are multiplied, and which are tried alongside the
        constant broadenings. The time-of-flight binning gives a
        resolution that grows as the square root of the energy, and a
        constant broadening cannot reproduce that.

    Returns:
        The result of the fit.

    Raises:
        ValueError: If no elements were given, or if the profile and the
            templates do not have the same number of bins.
    """
    profile = np.asarray(profile, dtype=float)
    if not symbols:
        raise ValueError("At least one element is needed for the fit.")
    if fixed_composition is not None and fixed_amount is not None:
        raise ValueError(
            "A fixed composition and a fixed amount cannot both be given: "
            "the first fixes the proportions and the second frees them."
        )

    if templates is None:
        if energies is None:
            raise ValueError("Either the energies or a set of templates must be given.")
        # The entries may be isotopes, in which case the isotope templates
        # are built instead of the element templates.
        if any("-" in str(symbol) for symbol in symbols):
            templates = build_fine_isotope_templates(
                list(symbols), energies, refinement=refinement
            )
        else:
            templates = build_fine_templates(
                list(symbols), energies, refinement=refinement
            )

    entries = _resolve_templates(symbols, templates)

    n_bins = len(profile)
    # The templates may be on a finer grid than the measurement, in which
    # case they are sampled back onto the measurement grid after the
    # broadening. The number of bins to compare against is then the number
    # of measurement energies.
    if templates.measurement_energies is not None:
        expected = len(templates.measurement_energies)
    else:
        expected = len(templates.energies)
    for entry in entries:
        if len(entry.cross_section) != len(templates.energies):
            raise ValueError(
                f"The template for '{entry.name}' has "
                f"{len(entry.cross_section)} bins but the grid "
                f"has {len(templates.energies)}."
            )
    if expected != n_bins:
        raise ValueError(
            f"The profile has {n_bins} bins but the templates have {expected}."
        )

    measured = _transmission(profile)

    # A deep dip does not reach zero transmission: it flattens out at the
    # background level of the measurement. The model therefore has a floor
    # term, so that the saturated part of the dip does not distort the
    # fitted abundances. Without it, the fit compensates for the clipped
    # dip by putting more material in the sample than there really is.
    if floor is None:
        floor = _estimate_floor(profile)
    floor = float(max(0.0, floor))

    # The bottom of a saturated dip is pinned at the floor, so it carries
    # no information about the amount of material. When the dip is
    # saturated, those bins are left out of the fit, and only the wings,
    # where the transmission is still well above the floor, are used.
    if unsaturated_only and floor > 0:
        usable = measured > _UNSATURATED_THRESHOLD * floor
        if usable.sum() < 10:
            usable = np.ones_like(measured, dtype=bool)
    else:
        usable = np.ones_like(measured, dtype=bool)

    # The templates are scaled to unit maximum, so that the coefficients of
    # the fit are of a comparable size and the fit is well conditioned. The
    # scale factors are kept, so that the coefficients can be turned back
    # into areal densities.
    #
    # An element whose cross-section is zero over the whole energy range
    # carries no information: its coefficient is unconstrained, so the fit
    # can put any amount of it there. Such elements are dropped from the
    # fit and reported separately.
    cross_sections = []
    raw_cross_sections = []
    scales = []
    fitted_symbols = []
    fitted_entries: list[_TemplateEntry] = []
    for entry in entries:
        sigma = entry.cross_section
        scale = float(np.max(sigma)) if sigma.size else 0.0
        if scale <= 0.0:
            continue
        fitted_symbols.append(entry.name)
        fitted_entries.append(entry)
        scales.append(scale)
        raw_cross_sections.append(sigma)
        cross_sections.append(sigma / scale)

    if not fitted_symbols:
        raise ValueError(
            "None of the elements has a cross-section over the energy "
            "range, so there is nothing to fit."
        )

    energies = templates.energies
    log_step = float(np.log(energies[1]) - np.log(energies[0]))

    # The templates may be on a finer grid than the measurement, in which
    # case the broadening is applied on that fine grid and the result is
    # sampled back onto the measurement grid. This matters: the measurement
    # grid has a step of about 0.5 %, which is 1.15 eV at 230 eV, and that
    # badly undersamples the narrow resonances of copper and niobium.
    fine = templates.measurement_energies is not None
    fine_energies = energies
    fine_log_step = log_step
    if fine:
        measurement_energies = np.asarray(templates.measurement_energies, dtype=float)
    else:
        measurement_energies = energies
    measurement_log_step = float(
        np.log(measurement_energies[1]) - np.log(measurement_energies[0])
    )

    best = None
    fixed_shape_max = 1.0
    # The total amount of material, when it is fixed by the metadata. It
    # is set inside the loop below, and it is bound as a default argument
    # of the residual function so that the function does not depend on
    # the loop variable.
    total_areal_density: float | None = None
    # The broadening is tried in two ways: as a constant width in log
    # energy, and as the energy-dependent resolution of the instrument.
    # The second is the physically correct one, because the time of
    # flight binning gives a resolution that grows as the square root of
    # the energy.
    broadening_options: list[tuple[float, float]] = [
        (width, 0.0) for width in broadenings
    ]
    for factor in resolution_factors:
        if factor > 0:
            broadening_options.append((0.0, factor))

    for relative_width, resolution_factor in broadening_options:
        if (
            separate_continuum
            or fixed_composition is not None
            or fixed_amount is not None
        ):
            # The split is made on the measurement grid, after the
            # broadening, so the smoothing width is in the units of the
            # grid that the fit is done on. The cross-sections are kept in
            # barns, so that the scale factors below are in barns too and
            # the coefficients can be turned into areal densities.
            #
            # The fixed composition needs the cross-sections in barns as
            # well, because the weighting of the elements depends on the
            # relative size of their cross-sections. The fixed amount
            # needs them in barns too, because the optical thickness is
            # the known amount of material times the weighted sum of the
            # cross-sections.
            source = raw_cross_sections
        else:
            source = cross_sections
        broadened = np.column_stack(
            [
                _broaden_energy_dependent(
                    sigma, fine_energies, resolution_factor, fine_log_step
                )
                if resolution_factor > 0
                else _broaden(sigma, relative_width, fine_log_step)
                for sigma in source
            ]
        )
        if fine:
            # Sample the broadened templates onto the measurement grid.
            sampled = np.column_stack(
                [
                    np.interp(measurement_energies, fine_energies, column)
                    for column in broadened.T
                ]
            )
        else:
            sampled = broadened

        if fixed_composition is not None:
            # The composition of the sample is known, so the shape of the
            # optical thickness is fixed and only the overall amount of
            # material is fitted. This is the reliable way to get an
            # areal density out of a saturated dip, because the depth of
            # the dip carries no information but the wings still do.
            #
            # The number fraction of each element is proportional to its
            # mass fraction divided by its atomic mass.
            numbers = []
            for entry in fitted_entries:
                mass_fraction = fixed_composition.get(entry.name, 0.0)
                if mass_fraction <= 0:
                    continue
                if entry.atomic_mass is None:
                    raise ValueError(
                        f"The mass of '{entry.name}' is not known, so the "
                        "composition cannot be used."
                    )
                numbers.append(mass_fraction / entry.atomic_mass)
            if not numbers:
                raise ValueError(
                    "The fixed composition does not contain any of the "
                    "elements that are being fitted."
                )
            number_fractions = np.asarray(numbers, dtype=float)
            number_fractions = number_fractions / number_fractions.sum()
            # The optical thickness is the overall amount of material times
            # the weighted sum of the cross-sections. Only the elements
            # that are in the fixed composition contribute.
            contributing = [
                index
                for index, entry in enumerate(fitted_entries)
                if fixed_composition.get(entry.name, 0.0) > 0
            ]
            # The cross-sections are in barns, so the shape is the optical
            # thickness per unit areal density.
            shape = sampled[:, contributing] @ number_fractions
            fixed_shape_max = float(shape.max())
            design = (shape / fixed_shape_max).reshape(-1, 1)
            fixed_weights = np.zeros(len(fitted_symbols))
            for position, index in enumerate(contributing):
                fixed_weights[index] = number_fractions[position]
            split_scales = None
        elif fixed_amount is not None:
            # The amount of material is known from the metadata, so the
            # fit only has to decide the proportions. The optical
            # thickness is the known amount of material times the
            # weighted sum of the cross-sections, where the weights are
            # the number fractions that are being fitted.
            #
            # The number fractions are fitted on the simplex, so that they
            # always sum to one and are all positive. The cross-sections
            # are used in barns, so that the optical thickness comes out
            # in the right units.
            total_areal_density = fixed_amount.total_areal_density
            if total_areal_density is None:
                raise ValueError(
                    f"The metadata for '{fixed_amount.name}' does not give "
                    "enough information to work out the amount of material. "
                    "The thickness and the composition are needed."
                )
            design = sampled
            split_scales = None
        elif separate_continuum:
            # Each element contributes two columns: the narrow resonances
            # and the smooth continuum. They are fitted with separate
            # coefficients, so that the amplitude of a resonance is not
            # traded off against the continuum.
            columns = []
            split_scales = []
            for column in sampled.T:
                resonance, continuum = _split_continuum(
                    column, measurement_log_step, continuum_width
                )
                # The two parts are scaled to unit maximum, so that the
                # coefficients of the fit are of a comparable size and the
                # fit is well conditioned. The scale factors are kept, so
                # that the coefficients can be turned back into areal
                # densities.
                resonance_scale = float(resonance.max())
                continuum_scale = float(continuum.max())
                columns.append(
                    resonance / resonance_scale if resonance_scale > 0 else resonance
                )
                columns.append(
                    continuum / continuum_scale if continuum_scale > 0 else continuum
                )
                split_scales.append((resonance_scale, continuum_scale))
            design = np.column_stack(columns)
        else:
            design = sampled
            split_scales = None

        def residual(
            coefficients: np.ndarray,
            design: np.ndarray = design,
            total_areal_density: float | None = total_areal_density,
        ) -> np.ndarray:
            if total_areal_density is not None:
                # The coefficients are the stick-breaking parameters of
                # the number fractions, and the amount of material is
                # fixed by the metadata.
                fractions = _simplex_from_stick(coefficients)
                optical_thickness = total_areal_density * (design @ fractions)
            else:
                optical_thickness = design @ coefficients
            modelled = floor + (1.0 - floor) * np.exp(-optical_thickness)
            return modelled[usable] - measured[usable]

        # The fit is started from several points, because the exponential
        # model has local minima and a single start can settle in the wrong
        # one. The starts are the all-zero vector, the all-ones vector, and
        # one that puts all the weight on a single element in turn.
        n_columns = design.shape[1]
        if fixed_amount is not None:
            # The coefficients are stick-breaking parameters, so they are
            # in the unit cube. The equal split is the obvious start, and
            # the ones that put all the weight on a single element are
            # tried as well, because the fit can settle in a local
            # minimum otherwise.
            n_elements = len(fitted_symbols)
            starts = [
                np.full(n_elements - 1, 1.0 / (n_elements - 1)),
                np.zeros(n_elements - 1),
                np.ones(n_elements - 1),
            ]
            for index in range(n_elements - 1):
                single = np.zeros(n_elements - 1)
                single[index] = 1.0
                starts.append(single)
        elif fixed_composition is not None:
            # The single coefficient is the optical thickness at the peak of
            # the weighted cross-section, which is of the order of the
            # amount of material. The starts are spread over the range
            # that is physically sensible.
            starts = [np.array([value]) for value in (0.1, 1.0, 10.0, 100.0, 1000.0)]
        else:
            starts = [np.zeros(n_columns), np.ones(n_columns)]
            for index in range(len(fitted_symbols)):
                single = np.zeros(n_columns)
                if separate_continuum:
                    single[2 * index] = 1.0
                else:
                    single[index] = 1.0
                starts.append(single)

        if separate_continuum:
            # The split fit has many more parameters than the plain one,
            # and it has a shallow minimum that the optimiser can miss.
            # It is seeded from the plain fit, which is done here on the
            # same broadened templates. The areal densities of the plain
            # fit are then split between the two parts in proportion to
            # the size of each part.
            # The plain fit is done on the same broadened templates, scaled
            # to unit maximum, so that its coefficients can be turned into
            # areal densities and used to seed the split fit.
            plain_maxima = sampled.max(axis=0)
            plain_design = sampled / np.where(plain_maxima > 0, plain_maxima, 1.0)

            def plain_residual(
                coefficients: np.ndarray, plain_design: np.ndarray = plain_design
            ) -> np.ndarray:
                optical_thickness = plain_design @ coefficients
                return floor + (1.0 - floor) * np.exp(-optical_thickness) - measured

            plain_best = None
            for plain_start in (
                np.zeros(len(fitted_symbols)),
                np.ones(len(fitted_symbols)),
            ):
                solution = least_squares(
                    plain_residual,
                    plain_start,
                    bounds=(0.0, np.inf),
                    method="trf",
                    xtol=1e-14,
                    ftol=1e-14,
                    max_nfev=5000,
                )
                rms = float(np.sqrt(np.mean(solution.fun**2)))
                if plain_best is None or rms < plain_best[0]:
                    plain_best = (rms, solution.x)
            assert plain_best is not None
            plain_areal_densities = plain_best[1] / np.where(
                plain_maxima > 0, plain_maxima, 1.0
            )

            seed = np.zeros(n_columns)
            for index in range(len(fitted_symbols)):
                resonance_scale, continuum_scale = split_scales[index]
                seed[2 * index] = plain_areal_densities[index] * resonance_scale
                seed[2 * index + 1] = plain_areal_densities[index] * continuum_scale
            starts.append(seed)

        for start in starts:
            if fixed_amount is not None:
                # The stick-breaking parameters are fractions of what is
                # left, so they lie between zero and one.
                bounds = (0.0, 1.0)
            else:
                bounds = (0.0, np.inf)
            solution = least_squares(
                residual,
                start,
                bounds=bounds,
                method="trf",
                xtol=1e-14,
                ftol=1e-14,
                max_nfev=5000,
            )
            rms = float(np.sqrt(np.mean(solution.fun**2)))
            if best is None or rms < best[0]:
                best = (
                    rms,
                    relative_width,
                    solution,
                    split_scales,
                    fixed_shape_max,
                    resolution_factor,
                )

    assert best is not None
    (
        rms,
        relative_width,
        solution,
        split_scales,
        fixed_shape_max,
        resolution_factor,
    ) = best
    coefficients = solution.x

    # The model is exp(-sum c_i sigma_i), where sigma_i is the true
    # cross-section in barns, so c_i is the areal density of the element in
    # atoms per barn. The fit was done with the scaled templates
    # sigma_i / scale_i, so the areal density is the coefficient divided by
    # the scale.
    scale_array = np.asarray(scales)
    if fixed_composition is not None:
        # Only the overall amount of material was fitted, so the areal
        # densities follow from the known composition. The fitted
        # coefficient is the total optical thickness at the peak of the
        # weighted cross-section, and the peak of the weighted
        # cross-section is fixed_shape_max barns, so the total areal
        # density is the coefficient divided by that.
        peak_thickness = float(coefficients[0])
        total_areal_density = peak_thickness / fixed_shape_max
        areal_densities = total_areal_density * fixed_weights
        resonance_areal_densities = None
        continuum_areal_densities = None
    elif fixed_amount is not None:
        # The amount of material was fixed by the metadata and the fit
        # found the proportions, so the areal density of each element is
        # the known total times its number fraction.
        number_fractions = _simplex_from_stick(coefficients)
        areal_densities = total_areal_density * number_fractions
        resonance_areal_densities = None
        continuum_areal_densities = None
    elif separate_continuum:
        # The coefficients alternate between the resonance part and the
        # continuum part of each element. The two parts are scaled
        # separately, because the resonance part and the continuum part
        # have very different magnitudes.
        resonance_coefficients = coefficients[0::2]
        continuum_coefficients = coefficients[1::2]
        resonance_scales = np.asarray([scales[0] for scales in split_scales])
        continuum_scales = np.asarray([scales[1] for scales in split_scales])
        resonance_areal_densities = resonance_coefficients / np.where(
            resonance_scales > 0, resonance_scales, 1.0
        )
        continuum_areal_densities = continuum_coefficients / np.where(
            continuum_scales > 0, continuum_scales, 1.0
        )
        # The total areal density of an element is the sum of the two
        # parts, because the cross-section is the sum of the two.
        areal_densities = resonance_areal_densities + continuum_areal_densities
    else:
        resonance_areal_densities = None
        continuum_areal_densities = None
        areal_densities = coefficients / scale_array

    # The fractions are the shares of the modelled optical thickness, which
    # is what the fit actually constrains. The areal densities are not used
    # for this, because the elements have very different cross-section
    # magnitudes: an element with a small cross-section needs a large areal
    # density to make the same dip, so normalising the areal densities would
    # overstate it.
    #
    # When the continuum is fitted separately, the coefficients alternate
    # between the resonance part and the continuum part of each element,
    # so they are summed per element before being normalised.
    if fixed_composition is not None:
        # The composition is known, so the fractions are the known ones.
        masses = np.asarray(
            [
                (entry.atomic_mass or 0.0) * fixed_weights[index]
                for index, entry in enumerate(fitted_entries)
            ]
        )
        fractions = masses / masses.sum() if masses.sum() > 0 else masses
    elif fixed_amount is not None:
        # The number fractions are what was fitted, so they are the
        # fractions of the optical thickness.
        fractions = _simplex_from_stick(coefficients)
    elif separate_continuum:
        per_element = coefficients[0::2] + coefficients[1::2]
        total = per_element.sum()
        fractions = per_element / total if total > 0 else np.zeros_like(per_element)
    else:
        total = coefficients.sum()
        fractions = coefficients / total if total > 0 else np.zeros_like(coefficients)

    correlations = {}
    for entry in entries:
        sigma = entry.cross_section
        if fine:
            # The correlation is with the measured profile, so the template
            # has to be sampled onto the measurement grid first.
            sigma = np.interp(measurement_energies, fine_energies, sigma)
        correlations[entry.name] = _correlate(measured, sigma)

    # The mass fraction is the share of the sample by mass, which is the
    # usual way a composition is quoted. It is reported separately from
    # the fractions, which are shares of the modelled optical thickness.
    mass_fractions = _mass_fractions(fitted_symbols, areal_densities)
    if mass_fractions is None and any(entry.is_isotope for entry in fitted_entries):
        # The mass of an isotope is known, so the mass fractions can be
        # worked out even when the entries are isotopes rather than
        # elements.
        masses = np.asarray(
            [
                entry.atomic_mass if entry.atomic_mass is not None else 0.0
                for entry in fitted_entries
            ]
        )
        weighted = masses * np.asarray(areal_densities, dtype=float)
        total = float(weighted.sum())
        if total > 0:
            mass_fractions = weighted / total

    return ENDFfitResult(
        symbols=list(fitted_symbols),
        areal_densities=areal_densities,
        fractions=fractions,
        residual=solution.fun,
        rms_residual=rms,
        broadening=relative_width,
        correlations=correlations,
        excluded=[symbol for symbol in symbols if symbol not in fitted_symbols],
        floor=floor,
        saturated=floor > 0.0,
        resonance_areal_densities=resonance_areal_densities,
        continuum_areal_densities=continuum_areal_densities,
        continuum_split=separate_continuum,
        fixed_composition=fixed_composition,
        unsaturated_bins=int(usable.sum()),
        mass_fractions=mass_fractions,
        optical_thickness_fractions=fractions,
        total_areal_density=(
            float(total_areal_density)
            if fixed_amount is not None or fixed_composition is not None
            else float(np.sum(areal_densities))
        ),
        resolution_factor=resolution_factor,
        isotopes={
            entry.name: entry.mass
            for entry in fitted_entries
            if entry.is_isotope and entry.mass is not None
        },
        atomic_fractions=_atomic_fractions(areal_densities),
    )


def fit_regions(
    profiles: np.ndarray,
    symbols: list[str] | tuple[str, ...],
    *,
    energies: np.ndarray | None = None,
    templates: TemplateSet | None = None,
    broadenings: Sequence[float] = DEFAULT_BROADENINGS,
    floor: float | None = None,
    refinement: int = DEFAULT_REFINEMENT,
    separate_continuum: bool = False,
    continuum_width: float = DEFAULT_CONTINUUM_WIDTH,
    fixed_composition: dict[str, float] | None = None,
    unsaturated_only: bool = False,
    fixed_amount: SampleMetadata | None = None,
    resolution_factors: Sequence[float] = DEFAULT_RESOLUTION_FACTORS,
) -> list[ENDFfitResult]:
    """Fit a stack of profiles against the ENDF cross-sections.

    Parameters
    ----------
    profiles : numpy.ndarray
        The measured profiles, one row per region.
    symbols : list of str
        The elements to fit the regions against.
    energies : numpy.ndarray, optional
        The energies in eV of the profile bins.
    templates : TemplateSet, optional
        Pre-built templates.
    broadenings : sequence of float
        The relative energy resolutions to try.
    floor : float, optional
        The transmission floor of the measurement.
    refinement : int
        How many times finer than the measurement grid the cross-sections
        are evaluated on.
    separate_continuum : bool
        Whether to fit the resonances and the continuum separately.
    continuum_width : float
        The width of the smoothing that separates the two.
    fixed_composition : dict, optional
        A known composition of the sample.
    unsaturated_only : bool
        Whether to leave the saturated bins out of the fit.

    Returns:
        One result per profile.
    """
    profiles = np.atleast_2d(np.asarray(profiles, dtype=float))
    return [
        fit_region(
            profile,
            symbols,
            energies=energies,
            templates=templates,
            broadenings=broadenings,
            floor=floor,
            refinement=refinement,
            separate_continuum=separate_continuum,
            continuum_width=continuum_width,
            fixed_composition=fixed_composition,
            unsaturated_only=unsaturated_only,
            fixed_amount=fixed_amount,
            resolution_factors=resolution_factors,
        )
        for profile in profiles
    ]


def _load_dictionary(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the profiles and the energy grid of a dictionary file."""
    with np.load(path, allow_pickle=True) as data:
        names = [str(name) for name in data["entry_names"]]
        profiles = data["profiles"]
        x_bins = data["x_bins"]
    return names, profiles, x_bins


def _load_workspaces(
    paths: Sequence[str], l2_values: Sequence[float]
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Load raw selection workspaces and turn them into profiles.

    Each workspace is converted from time of flight to energy at every
    sample-detector distance, and the profiles of those conversions are
    averaged together. This is the same conversion that the dictionary
    uses, so the profiles are the same.

    A path may carry a name after an ``=``, e.g.
    ``Havar.nxs=Havar/Havar_5foils``, which is the name the region is
    known by. It is used to look the metadata and the known composition
    up. Without it the name of the file is used.

    Parameters
    ----------
    paths : sequence of str
        The workspace files to load.
    l2_values : sequence of float
        The sample-detector distances to convert at.

    Returns:
        The names of the regions, their profiles, and the bin edges.

    Raises:
        FileNotFoundError: If one of the files does not exist.
        ValueError: If a file cannot be loaded, or if the workspaces do
            not share the same bin edges.
    """
    from mantid.simpleapi import Load

    from data_loading.load_and_normalise import convert_TOF_to_energy

    names: list[str] = []
    profiles: list[np.ndarray] = []
    x_bins: np.ndarray | None = None

    for entry in paths:
        text = str(entry)
        path, separator, name = text.partition("=")
        path = Path(path.strip())
        if not separator:
            name = path.stem
        name = name.strip() or path.stem

        if not path.is_file():
            raise FileNotFoundError(f"Workspace file not found: {path}")

        try:
            workspace = Load(
                str(path), OutputWorkspace=f"{name}_selection", StoreInADS=False
            )
        except Exception as exc:
            raise ValueError(f"Could not load '{path}': {exc}") from exc

        collected = []
        for l2 in l2_values:
            converted = convert_TOF_to_energy(
                workspace, l2=l2, output_name=f"{name}_energy_{l2:g}"
            )
            if x_bins is None:
                x_bins = converted.readX(0).copy()
            elif not np.array_equal(x_bins, converted.readX(0)):
                raise ValueError(
                    f"'{path}' has different bin edges to the other "
                    "workspaces, so the spectra cannot be compared."
                )
            collected.append(converted.extractY().mean(axis=0))

        names.append(name)
        profiles.append(np.mean(collected, axis=0))

    if x_bins is None:
        raise ValueError("No workspaces were given.")
    return names, profiles, x_bins


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the entries of a dictionary against the local ENDF database.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit the measured regions of a dictionary against the "
            "cross-sections reconstructed from the local ENDF database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help=(
            "Path to the dictionary .npz file holding the measured regions. "
            "This is one way of giving the fit something to fit; a raw "
            "selection workspace can be given with --workspace instead."
        ),
    )
    parser.add_argument(
        "--workspace",
        action="append",
        default=None,
        help=(
            "A raw selection workspace to fit, e.g. "
            "'Havar/Havar_5foils_div_Empty_normMonitor_selection.nxs'. "
            "May be given more than once. The name of the region may be "
            "given after an '=', e.g. 'file.nxs=Havar/Havar_5foils', and "
            "is used to look the metadata up."
        ),
    )
    parser.add_argument(
        "--l2",
        type=float,
        action="append",
        default=None,
        help=(
            "The sample-detector distance to convert the workspaces at, in "
            "metres. May be given more than once, and the profiles of the "
            "conversions are averaged. The default is 0.63 m."
        ),
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help=(
            "The elements to fit the regions against. An isotope may be "
            "given instead of an element, e.g. 'Ag-107', in which case the "
            "isotopes are fitted separately."
        ),
    )
    parser.add_argument(
        "--isotopes",
        nargs="+",
        default=None,
        help=(
            "The isotopes to fit the regions against, e.g. 'Ag-107 "
            "Ag-109'. This is the same as --elements, but it makes the "
            "intent clear."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        default=None,
        help=(
            "The name of a region to fit. May be given more than once. "
            "By default every region of the dictionary is fitted."
        ),
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Skip the regions that hold no isotope.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help=(
            "The transmission floor of the measurement, which is the level "
            "at which the deepest dip saturates. By default it is "
            "estimated from each profile."
        ),
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=DEFAULT_REFINEMENT,
        help=(
            "How many times finer than the measurement grid the "
            "cross-sections are evaluated on. A finer grid is needed to "
            "resolve the narrow resonances of copper and niobium."
        ),
    )
    parser.add_argument(
        "--unsaturated-only",
        action="store_true",
        help=(
            "Leave the saturated bins out of the fit. The bottom of a "
            "saturated dip is pinned at the floor, so it carries no "
            "information about the amount of material."
        ),
    )
    parser.add_argument(
        "--composition",
        type=Path,
        default=None,
        help=(
            "A YAML file with the known composition of each region, as "
            "the mass fraction of each element. When this is given, the "
            "fit is constrained to that composition and only the overall "
            "amount of material is fitted."
        ),
    )
    parser.add_argument(
        "--separate-continuum",
        action="store_true",
        help=(
            "Fit the narrow resonances and the smooth continuum of each "
            "element with separate coefficients."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "A YAML file or an Excel spreadsheet with the metadata of each "
            "region: its thickness, its composition and its density. When "
            "this is given, the amount of material is pinned by the "
            "metadata and the fit finds the relative abundance of the "
            "elements."
        ),
    )
    parser.add_argument(
        "--fixed-amount",
        action="store_true",
        help=(
            "Pin the amount of material with the metadata, and fit only the "
            "relative abundance of the elements. This needs --metadata."
        ),
    )
    parser.add_argument(
        "--continuum-width",
        type=float,
        default=DEFAULT_CONTINUUM_WIDTH,
        help=(
            "The width, in log energy, of the smoothing that separates "
            "the resonances from the continuum."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path of the CSV file to write the results to.",
    )
    args = parser.parse_args(argv)

    # The fit needs a profile and an energy grid. They can come from a
    # dictionary of measured regions, or from raw selection workspaces.
    # The dictionary is not needed by the fit itself: it is only one way
    # of holding the measured regions.
    if args.dictionary and args.workspace:
        print(
            "Error: --dictionary and --workspace cannot both be given.",
            file=sys.stderr,
        )
        return 1
    if not args.dictionary and not args.workspace:
        print(
            "Error: either --dictionary or --workspace must be given.",
            file=sys.stderr,
        )
        return 1

    if args.dictionary:
        try:
            names, profile_matrix, x_bins = _load_dictionary(args.dictionary)
        except (OSError, KeyError, ValueError) as exc:
            print(
                f"Error: could not read {args.dictionary}: {exc}",
                file=sys.stderr,
            )
            return 1
        profiles = [profile_matrix[:, index] for index in range(len(names))]
    else:
        l2_values = args.l2 if args.l2 else [0.63]
        try:
            names, profiles, x_bins = _load_workspaces(args.workspace, l2_values)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error: could not read the workspaces: {exc}", file=sys.stderr)
            return 1

    selected = list(range(len(names)))
    if args.region:
        wanted = set(args.region)
        selected = [i for i, name in enumerate(names) if name in wanted]
        missing = wanted - set(names)
        if missing:
            print(
                f"Error: no region named {', '.join(sorted(missing))}.",
                file=sys.stderr,
            )
            return 1
    if args.exclude_background:
        selected = [i for i in selected if "background" not in names[i].lower()]

    if not selected:
        print("Error: no regions to fit.", file=sys.stderr)
        return 1

    # The energy grid of the dictionary is a set of bin edges in meV, which
    # are converted to eV at the bin centres.
    centres = np.sqrt(x_bins[:-1] * x_bins[1:]) / 1000.0

    # The entries to fit are either elements or isotopes. The isotopes are
    # given either with --isotopes, or with --elements when the name
    # carries a mass number.
    if args.isotopes and args.elements:
        print(
            "Error: --elements and --isotopes cannot both be given.",
            file=sys.stderr,
        )
        return 1
    entries = list(args.isotopes) if args.isotopes else list(args.elements or [])
    if not entries:
        print(
            "Error: either --elements or --isotopes must be given.",
            file=sys.stderr,
        )
        return 1

    try:
        if any("-" in name for name in entries):
            templates = build_fine_isotope_templates(
                entries, centres, refinement=args.refinement
            )
        else:
            templates = build_fine_templates(
                entries, centres, refinement=args.refinement
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: could not build the templates: {exc}", file=sys.stderr)
        return 2

    # The known compositions, if they were given, are read from a YAML
    # file that maps the name of a region to the mass fraction of each
    # element.
    compositions: dict[str, dict[str, float]] = {}
    if args.composition is not None:
        try:
            import yaml

            with args.composition.open(encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            for name, composition in loaded.items():
                compositions[str(name)] = {
                    str(symbol): float(value) for symbol, value in composition.items()
                }
        except (OSError, ValueError) as exc:
            print(
                f"Error: could not read {args.composition}: {exc}",
                file=sys.stderr,
            )
            return 1

    # The metadata of each region, when it has been given. It pins the
    # amount of material, and so it needs the thickness and the
    # composition of the sample.
    metadata: dict[str, SampleMetadata] = {}
    if args.metadata is not None:
        try:
            from .sample_metadata import load_metadata

            metadata = load_metadata(args.metadata)
        except (OSError, ValueError) as exc:
            print(f"Error: could not read the metadata: {exc}", file=sys.stderr)
            return 1

    rows = []
    for index in selected:
        # A region that has no metadata cannot be fitted with a fixed
        # amount, so it is reported as skipped rather than being an error.
        region_metadata = metadata.get(names[index]) if args.fixed_amount else None
        if args.fixed_amount and region_metadata is None:
            print(
                f"Skipping '{names[index]}': it has no metadata.",
                file=sys.stderr,
            )
            continue
        if args.fixed_amount and region_metadata.total_areal_density is None:
            print(
                f"Skipping '{names[index]}': its metadata does not give the "
                "amount of material.",
                file=sys.stderr,
            )
            continue
        result = fit_region(
            profiles[index],
            entries,
            templates=templates,
            floor=(
                args.floor
                if args.floor is not None
                else (
                    region_metadata.floor
                    if region_metadata is not None and region_metadata.floor is not None
                    else None
                )
            ),
            separate_continuum=args.separate_continuum,
            continuum_width=args.continuum_width,
            fixed_composition=compositions.get(names[index]),
            unsaturated_only=args.unsaturated_only,
            fixed_amount=region_metadata,
        )
        row: dict = {
            "region": names[index],
            "dominant": result.dominant,
            "rms_residual": result.rms_residual,
            "broadening": result.broadening,
            "floor": result.floor,
            "unsaturated_bins": result.unsaturated_bins,
        }
        for symbol, fraction, density, correlation in zip(
            result.symbols,
            result.fractions,
            result.areal_densities,
            (result.correlations[s] for s in result.symbols),
        ):
            row[f"fraction_{symbol}"] = fraction
            row[f"areal_density_{symbol}"] = density
            row[f"correlation_{symbol}"] = correlation
        if result.mass_fractions is not None:
            for symbol, fraction in zip(result.symbols, result.mass_fractions):
                row[f"mass_fraction_{symbol}"] = fraction
        if result.atomic_fractions is not None:
            for symbol, fraction in zip(result.symbols, result.atomic_fractions):
                row[f"atomic_fraction_{symbol}"] = fraction
        rows.append(row)

    if not rows:
        print("Error: no regions were fitted.", file=sys.stderr)
        return 1

    fieldnames = list(rows[0])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} result(s) to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
