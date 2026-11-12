"""Find the elements and isotopes that a measurement could possibly contain.

The number of nuclides in the ENDF database is far larger than the number
that could be fitted to a measurement, because most of them have no
resonance in the energy range of the measurement at all. This module
scans the database for the nuclides that do, which turns the problem of
fitting an unknown sample into a fit over a manageable number of
candidates.

The scan is the slow part: it reads every ENDF file in the database, which
takes a couple of minutes. The result is cached on disk, so it is only
paid once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from endf_database_util.local_endf import LocalENDFDatabase

# The energy range that the nGEM measurements cover. A nuclide that has no
# resonance in this range leaves no trace in the spectrum, so it cannot be
# found and does not need to be fitted for.
DEFAULT_MIN_ENERGY = 1.0
DEFAULT_MAX_ENERGY = 500.0

# How much of the resonance energy tolerance is used. The resolved
# resonance energies in the database are the energies of the resonance
# centres, and they are given to about half a barn of precision, so a
# small tolerance is needed to catch a resonance that sits just outside
# the range.
DEFAULT_ENERGY_TOLERANCE = 1.0


@dataclass
class NuclideResonances:
    """The resonances of one nuclide that fall in the energy range.

    Attributes:
        symbol: The element symbol.
        mass: The mass number of the isotope.
        energies: The energies of the resonances in eV.
        natural: Whether the isotope has a natural abundance.
    """

    symbol: str
    mass: int
    energies: np.ndarray
    natural: bool

    @property
    def name(self) -> str:
        """The name of the nuclide, e.g. ``'Ag-107'``."""
        return f"{self.symbol}-{self.mass}"


@dataclass
class DiscoveryResult:
    """The nuclides that could be present in a measurement.

    Attributes:
        min_energy: The lower edge of the energy range in eV.
        max_energy: The upper edge of the energy range in eV.
        nuclides: Every nuclide with a resonance in the range.
        elements: The elements that have at least one natural isotope with
            a resonance in the range. These are the candidates for a fit
            at the level of elements.
        isotopes: The natural isotopes that have a resonance in the
            range. These are the candidates for a fit at the level of
            isotopes.
    """

    min_energy: float
    max_energy: float
    nuclides: list[NuclideResonances] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    isotopes: list[str] = field(default_factory=list)

    @property
    def element_count(self) -> int:
        """The number of candidate elements."""
        return len(self.elements)

    @property
    def isotope_count(self) -> int:
        """The number of candidate isotopes."""
        return len(self.isotopes)

    @property
    def resonance_count(self) -> int:
        """The total number of resonances in the range."""
        return int(sum(len(n.energies) for n in self.nuclides))


def _natural_masses(symbol: str) -> list[int]:
    """Return the natural isotopes of an element.

    An element that has no natural isotope at all cannot be in a natural
    sample, so it is of no use as a candidate. The synthetic elements and
    the ones with no stable isotope are the ones that this removes.
    """
    from endf_database_util.templates import element_isotopes

    try:
        return list(element_isotopes(symbol))
    except (KeyError, ValueError):
        return []


def _cache_key(min_energy: float, max_energy: float, tolerance: float) -> str:
    """Return a cache key for a scan."""
    digest = hashlib.sha256()
    digest.update(f"{min_energy!r},{max_energy!r},{tolerance!r}".encode())
    return digest.hexdigest()[:16]


def scan_database(
    min_energy: float = DEFAULT_MIN_ENERGY,
    max_energy: float = DEFAULT_MAX_ENERGY,
    *,
    tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    database: LocalENDFDatabase | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> DiscoveryResult:
    """Find every nuclide with a resonance in an energy range.

    The scan reads every nuclide in the database, which takes a couple of
    minutes. The result is cached on disk, so it is only paid once.

    Parameters
    ----------
    min_energy : float
        The lower edge of the energy range in eV.
    max_energy : float
        The upper edge of the energy range in eV.
    tolerance : float
        How far outside the range a resonance may lie and still be
        counted. The resolved resonance energies are only given to about
        half a barn of precision, so a small tolerance avoids losing a
        resonance that sits just outside the range.
    database : LocalENDFDatabase, optional
        The database to scan. The local database is used by default.
    cache_dir : str or pathlib.Path, optional
        Where to cache the result. Defaults to a ``.cache/discovery``
        directory next to the package.
    use_cache : bool
        Whether to read and write the cache.

    Returns:
        DiscoveryResult
            The nuclides, elements and isotopes that could be present.
    """
    if min_energy <= 0.0:
        raise ValueError("The lower edge of the energy range must be positive.")
    if max_energy <= min_energy:
        raise ValueError(
            "The upper edge of the energy range must be above the lower edge."
        )

    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".cache" / "discovery"
    cache_path = Path(cache_dir)
    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)
        cached = (
            cache_path / f"scan_{_cache_key(min_energy, max_energy, tolerance)}.json"
        )
        if cached.is_file():
            try:
                with cached.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                return _result_from_dict(raw)
            except (OSError, ValueError, KeyError):
                pass

    if database is None:
        database = LocalENDFDatabase()

    lower = min_energy - tolerance
    upper = max_energy + tolerance

    nuclides: list[NuclideResonances] = []
    for entry in database.available_isotopes():
        symbol, _, mass_text = entry.partition("-")
        mass_text = mass_text.split("m")[0]
        try:
            mass = int(mass_text)
        except ValueError:
            continue
        try:
            parameters = database.resonance_parameters(symbol, mass)
        except (FileNotFoundError, ValueError, KeyError, IndexError):
            continue

        energies = _resonance_energies(parameters, lower, upper)
        if energies.size == 0:
            continue

        natural = mass in _natural_masses(symbol)
        nuclides.append(
            NuclideResonances(
                symbol=symbol,
                mass=mass,
                energies=np.sort(energies),
                natural=natural,
            )
        )

    result = _assemble(min_energy, max_energy, nuclides)

    if use_cache:
        try:
            with cached.open("w", encoding="utf-8") as handle:
                json.dump(_result_to_dict(result), handle, indent=1)
        except OSError:
            pass

    return result


def _resonance_energies(parameters, low: float, high: float) -> np.ndarray:
    """Return the energies of the resonances of one nuclide in a range.

    The MLBW and Reich-Moore evaluations keep their resonances in a flat
    list, and the R-matrix evaluations keep them in spin groups. The two
    formalisms need different handling, so the dedicated helpers of the
    package are used for each of them.
    """
    from endf_database_util.local_endf import RMatrixParameters
    from endf_database_util.resonance import resonance_energies
    from endf_database_util.rmatrix import rmatrix_resonance_energies

    if isinstance(parameters, RMatrixParameters):
        return np.asarray(
            rmatrix_resonance_energies(parameters, low, high), dtype=float
        )
    return np.asarray(resonance_energies(parameters, low, high), dtype=float)


def _assemble(
    min_energy: float, max_energy: float, nuclides: list[NuclideResonances]
) -> DiscoveryResult:
    """Build the result from the nuclides that were found."""
    elements = sorted(
        {n.symbol for n in nuclides if n.natural},
    )
    isotopes = sorted(
        {n.name for n in nuclides if n.natural},
        key=lambda name: (name.split("-")[0], int(name.split("-")[1])),
    )
    return DiscoveryResult(
        min_energy=min_energy,
        max_energy=max_energy,
        nuclides=sorted(nuclides, key=lambda n: (n.symbol, n.mass)),
        elements=elements,
        isotopes=isotopes,
    )


def _result_to_dict(result: DiscoveryResult) -> dict:
    """Turn a result into something that can be written to JSON."""
    return {
        "min_energy": result.min_energy,
        "max_energy": result.max_energy,
        "nuclides": [
            {
                "symbol": n.symbol,
                "mass": n.mass,
                "energies": [float(e) for e in n.energies],
                "natural": bool(n.natural),
            }
            for n in result.nuclides
        ],
        "elements": list(result.elements),
        "isotopes": list(result.isotopes),
    }


def _result_from_dict(raw: dict) -> DiscoveryResult:
    """Turn a cached result back into a result."""
    nuclides = [
        NuclideResonances(
            symbol=entry["symbol"],
            mass=int(entry["mass"]),
            energies=np.asarray(entry["energies"], dtype=float),
            natural=bool(entry["natural"]),
        )
        for entry in raw["nuclides"]
    ]
    return DiscoveryResult(
        min_energy=float(raw["min_energy"]),
        max_energy=float(raw["max_energy"]),
        nuclides=nuclides,
        elements=list(raw["elements"]),
        isotopes=list(raw["isotopes"]),
    )
