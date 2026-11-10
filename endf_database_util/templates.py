"""Build cross-section templates for elements from the local ENDF database.

A *template* is the reconstructed neutron total cross-section of an element,
evaluated on the energy grid of the measured spectra. The template is what
the dips in a measured profile are compared against, so it is the theoretical
counterpart of a dictionary entry.

The natural isotopic composition is used, weighted by the isotopic
abundances, because a metal sample contains all its isotopes. The
abundances come from the :mod:`periodictable` package.

Templates are expensive to build (each one needs the resonance parameters of
every isotope of the element to be parsed and reconstructed), so they are
cached on disk. The cache is keyed by the element, the isotopes and the
energy grid, so a template is only rebuilt when one of those changes.

Example::

    from endf_database_util.templates import build_templates

    templates = build_templates(["Cu", "Zn"], energies)
    print(templates["Cu"].cross_section.max())
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .local_endf import LocalENDFDatabase, RMatrixParameters
from .resonance import total_cross_section
from .rmatrix import rmatrix_total_cross_section

__all__ = [
    "ElementTemplate",
    "IsotopeTemplate",
    "TemplateSet",
    "build_fine_isotope_templates",
    "build_fine_templates",
    "build_isotope_templates",
    "build_templates",
    "element_isotopes",
    "isotope_name",
    "isotopic_abundances",
    "natural_cross_section",
    "parse_isotope_name",
]


# The isotopes of each element that have a natural abundance, taken from the
# periodictable package. Only isotopes with a non-zero abundance are used.
def _element(symbol: str):
    """Return the periodictable element for a symbol.

    Raises:
        ValueError: If the symbol is not a known element.
    """
    import periodictable

    try:
        return getattr(periodictable, symbol)
    except AttributeError as exc:
        raise ValueError(f"'{symbol}' is not a known element symbol.") from exc


def element_isotopes(symbol: str) -> list[int]:
    """Return the mass numbers of the naturally abundant isotopes.

    Parameters
    ----------
    symbol : str
        Element symbol, e.g. ``'Cu'``.

    Returns
    -------
    list of int
        The mass numbers with a non-zero natural abundance, in increasing
        order.

    Raises
    ------
    ValueError
        If the element symbol is not known.
    """

    element = _element(symbol)
    masses = []
    for mass in element.isotopes:
        try:
            abundance = element[mass].abundance
        except KeyError:
            # The isotope is not in the table of the element, so it has no
            # abundance to speak of.
            continue
        if abundance:
            masses.append(int(mass))
    return sorted(masses)


def isotopic_abundances(symbol: str, masses: list[int]) -> dict[int, float]:
    """Return the natural abundance of each isotope, as fractions.

    Parameters
    ----------
    symbol : str
        Element symbol.
    masses : list of int
        The mass numbers to look up.

    Returns
    -------
    dict
        A mapping of mass number to abundance, with the abundances summing
        to one over the isotopes that were found.
    """

    element = _element(symbol)
    raw = {}
    for mass in masses:
        try:
            abundance = element[mass].abundance
        except KeyError:
            # The isotope is not in the table of the element, so it has no
            # abundance to speak of.
            continue
        if abundance:
            raw[int(mass)] = float(abundance)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {mass: abundance / total for mass, abundance in raw.items()}


def natural_cross_section(
    database: LocalENDFDatabase,
    symbol: str,
    masses: list[int],
    energies: np.ndarray,
) -> np.ndarray:
    """Reconstruct the total cross-section of a naturally composed element.

    Each isotope is reconstructed separately and combined with its natural
    abundance.

    Parameters
    ----------
    database : LocalENDFDatabase
        The database to read the evaluations from.
    symbol : str
        Element symbol, e.g. ``'Cu'``.
    masses : list of int
        The isotopes to combine.
    energies : numpy.ndarray
        Energies in eV.

    Returns
    -------
    numpy.ndarray
        The cross-section in barns at each energy.

    Raises
    ------
    ValueError
        If none of the isotopes could be evaluated.
    """
    energies = np.asarray(energies, dtype=float)
    abundances = isotopic_abundances(symbol, masses)

    total = np.zeros_like(energies)
    weight = 0.0
    for mass in masses:
        abundance = abundances.get(int(mass))
        if not abundance:
            continue
        params = database.resonance_parameters(symbol, int(mass))
        if isinstance(params, RMatrixParameters):
            result = rmatrix_total_cross_section(params, energies)
        else:
            result = total_cross_section(params, energies)
        total += abundance * result.total
        weight += abundance

    if weight <= 0:
        raise ValueError(
            f"No isotope of {symbol} with a natural abundance could be evaluated."
        )
    return total / weight


@dataclass
class ElementTemplate:
    """The cross-section template of one element.

    Attributes:
        symbol: The element symbol.
        isotopes: The isotopes that make up the template.
        abundances: The natural abundance of each isotope.
        energies: The energies in eV at which the cross-section is given.
        cross_section: The total cross-section in barns.
        resonance_energies: The resolved resonance energies in eV, which
            mark where the dips of this element appear.
    """

    symbol: str
    isotopes: list[int]
    abundances: dict[int, float]
    energies: np.ndarray
    cross_section: np.ndarray
    resonance_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class IsotopeTemplate:
    """The cross-section template of one isotope.

    Attributes:
        symbol: The element symbol.
        mass: The mass number of the isotope.
        name: The name of the isotope, e.g. ``'Ag-107'``.
        energies: The energies in eV at which the cross-section is given.
        cross_section: The total cross-section in barns.
        resonance_energies: The resolved resonance energies in eV, which
            mark where the dips of this isotope appear.
    """

    symbol: str
    mass: int
    name: str
    energies: np.ndarray
    cross_section: np.ndarray
    resonance_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class TemplateSet:
    """A set of element templates sharing one energy grid.

    Attributes:
        templates: The templates, keyed by element symbol.
        energies: The energies in eV shared by every template.
        measurement_energies: The energies of the measurement grid, when the
            templates are on a finer grid than the measurement. The
            templates are sampled back onto this grid after any broadening.
        isotope_templates: The isotope templates, keyed by the name of the
            isotope, e.g. ``'Ag-107'``. They are only present when the set
            was built for isotopes.
    """

    templates: dict[str, ElementTemplate]
    energies: np.ndarray
    measurement_energies: np.ndarray | None = None
    isotope_templates: dict[str, IsotopeTemplate] = field(default_factory=dict)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.templates

    def __len__(self) -> int:
        return len(self.templates)

    def __getitem__(self, symbol: str) -> ElementTemplate:
        return self.templates[symbol]

    @property
    def symbols(self) -> list[str]:
        """The element symbols of the templates, in order."""
        return list(self.templates)

    @property
    def isotope_names(self) -> list[str]:
        """The names of the isotope templates, in order."""
        return list(self.isotope_templates)

    def matrix(self) -> np.ndarray:
        """The templates as a matrix of shape (n_energies, n_templates)."""
        if not self.templates:
            raise ValueError("The template set is empty.")
        return np.column_stack(
            [self.templates[symbol].cross_section for symbol in self.symbols]
        )

    def isotope_matrix(self) -> np.ndarray:
        """The isotope templates as a matrix of shape (n_energies, n_isotopes)."""
        if not self.isotope_templates:
            raise ValueError("The template set has no isotope templates.")
        return np.column_stack(
            [self.isotope_templates[name].cross_section for name in self.isotope_names]
        )


def _cache_key(symbol: str, masses: list[int], energies: np.ndarray) -> str:
    """Return a cache key for one template."""
    digest = hashlib.sha256()
    digest.update(symbol.encode())
    digest.update(json.dumps(masses).encode())
    digest.update(np.asarray(energies, dtype=float).tobytes())
    return digest.hexdigest()[:16]


def build_templates(
    symbols: list[str] | tuple[str, ...],
    energies: np.ndarray,
    *,
    database: LocalENDFDatabase | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> TemplateSet:
    """Build the cross-section templates of several elements.

    Parameters
    ----------
    symbols : list of str
        The elements to build templates for.
    energies : numpy.ndarray
        The energies in eV to evaluate the cross-sections at.
    database : LocalENDFDatabase, optional
        The database to read from. The local database is used by default.
    cache_dir : str or pathlib.Path, optional
        Where to cache the built templates. Defaults to a ``.cache/templates``
        directory next to the package.
    use_cache : bool
        Whether to read and write the cache. Set to ``False`` to rebuild
        every template.

    Returns
    -------
    TemplateSet
        The templates, keyed by element symbol.
    """
    energies = np.asarray(energies, dtype=float)
    if database is None:
        database = LocalENDFDatabase()

    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".cache" / "templates"
    cache_path = Path(cache_dir)
    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)

    templates: dict[str, ElementTemplate] = {}
    for symbol in symbols:
        masses = element_isotopes(symbol)
        if not masses:
            raise ValueError(f"No natural isotopes are known for {symbol}.")

        key = _cache_key(symbol, masses, energies)
        cached = cache_path / f"{symbol}_{key}.npz" if use_cache else None

        cross_section = None
        resonance_energies = None
        if cached is not None and cached.is_file():
            try:
                with np.load(cached) as data:
                    cross_section = data["cross_section"]
                    resonance_energies = data["resonance_energies"]
            except (OSError, KeyError, ValueError):
                cross_section = None
                resonance_energies = None

        if cross_section is None:
            cross_section = natural_cross_section(database, symbol, masses, energies)
            resonance_energies = _resonance_energies(database, symbol, masses, energies)
            if cached is not None:
                try:
                    np.savez_compressed(
                        cached,
                        cross_section=cross_section,
                        resonance_energies=resonance_energies,
                        masses=np.asarray(masses),
                        abundances=np.asarray(
                            [isotopic_abundances(symbol, masses)[m] for m in masses]
                        ),
                    )
                except OSError:
                    pass

        templates[symbol] = ElementTemplate(
            symbol=symbol,
            isotopes=masses,
            abundances=isotopic_abundances(symbol, masses),
            energies=energies,
            cross_section=np.asarray(cross_section, dtype=float),
            resonance_energies=np.asarray(resonance_energies, dtype=float),
        )

    return TemplateSet(templates=templates, energies=energies)


def build_fine_templates(
    symbols: list[str] | tuple[str, ...],
    energies: np.ndarray,
    *,
    refinement: int = 60,
    database: LocalENDFDatabase | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> TemplateSet:
    """Build templates on a fine grid, for use with narrow resonances.

    The measured spectra are rebinned onto a logarithmic grid with a step of
    about 0.5 %, which is 0.005 eV at 1 eV but 1.15 eV at 230 eV. A
    resonance that is narrower than that step is badly undersampled: the
    copper-65 resonance at 230 eV has a peak of 80 barns, but only 15 barns
    are seen when it is evaluated on the measurement grid, and the niobium
    resonance at 378 eV is worse still, losing a factor of nine.

    This function evaluates the cross-sections on a grid that is
    ``refinement`` times finer, so that the peak of every resonance is
    resolved. The result is then sampled back onto the requested grid by
    the caller, after any broadening has been applied.

    Parameters
    ----------
    symbols : list of str
        The elements to build templates for.
    energies : numpy.ndarray
        The energies in eV of the measurement grid. The templates are
        returned on a finer grid that spans the same range.
    refinement : int
        How many times finer the evaluation grid should be.
    database : LocalENDFDatabase, optional
        The database to read from.
    cache_dir : str or pathlib.Path, optional
        Where to cache the built templates.
    use_cache : bool
        Whether to read and write the cache.

    Returns
    -------
    TemplateSet
        The templates, on the fine grid.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.size < 2:
        raise ValueError("At least two energies are needed.")
    if refinement < 1:
        raise ValueError("The refinement factor must be at least one.")

    fine = np.logspace(
        np.log10(energies[0]),
        np.log10(energies[-1]),
        energies.size * refinement,
    )
    templates = build_templates(
        symbols,
        fine,
        database=database,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    templates.measurement_energies = energies
    return templates


def isotope_name(symbol: str, mass: int) -> str:
    """Return the name of an isotope, e.g. ``'Ag-107'``.

    Parameters
    ----------
    symbol : str
        The element symbol.
    mass : int
        The mass number.

    Returns:
        The name of the isotope.
    """
    return f"{symbol}-{int(mass)}"


def parse_isotope_name(name: str) -> tuple[str, int]:
    """Split the name of an isotope into its symbol and its mass number.

    Parameters
    ----------
    name : str
        The name of the isotope, e.g. ``'Ag-107'``.

    Returns:
        The element symbol and the mass number.

    Raises:
        ValueError: If the name is not of the expected form.
    """
    text = str(name).strip()
    if "-" not in text:
        raise ValueError(
            f"'{name}' is not the name of an isotope. It should be of the "
            "form 'Ag-107'."
        )
    symbol, _, mass = text.rpartition("-")
    try:
        return symbol.strip(), int(mass)
    except ValueError as exc:
        raise ValueError(
            f"'{name}' is not the name of an isotope. It should be of the "
            "form 'Ag-107'."
        ) from exc


def build_isotope_templates(
    isotopes: list[str] | tuple[str, ...],
    energies: np.ndarray,
    *,
    database: LocalENDFDatabase | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> TemplateSet:
    """Build the cross-section templates of individual isotopes.

    A sample does not have to hold the natural isotopic composition of an
    element. Enriched material, and material made from separated isotopes,
    are both common, and the resonances of the isotopes of an element sit
    at *different* energies, so they can be told apart.

    This builds one template per isotope, rather than one per element. The
    templates are keyed by the name of the isotope, e.g. ``'Ag-107'``.

    Parameters
    ----------
    isotopes : list of str
        The isotopes to build templates for, e.g. ``['Ag-107', 'Ag-109']``.
    energies : numpy.ndarray
        The energies in eV to evaluate the cross-sections at.
    database : LocalENDFDatabase, optional
        The database to read from. The local database is used by default.
    cache_dir : str or pathlib.Path, optional
        Where to cache the built templates.
    use_cache : bool
        Whether to read and write the cache.

    Returns:
        TemplateSet
            The templates, with the isotope templates in the
            ``isotope_templates`` field.

    Raises:
        ValueError: If no isotopes were given, or if one of them cannot be
            evaluated.
    """
    energies = np.asarray(energies, dtype=float)
    if not isotopes:
        raise ValueError("At least one isotope is needed.")

    if database is None:
        database = LocalENDFDatabase()

    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".cache" / "templates"
    cache_path = Path(cache_dir)
    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)

    templates: dict[str, IsotopeTemplate] = {}
    for name in isotopes:
        symbol, mass = parse_isotope_name(name)
        key = _cache_key(symbol, [mass], energies)
        cached = cache_path / f"{symbol}{mass}_{key}.npz" if use_cache else None

        cross_section = None
        resonance_energies = None
        if cached is not None and cached.is_file():
            try:
                with np.load(cached) as data:
                    cross_section = data["cross_section"]
                    resonance_energies = data["resonance_energies"]
            except (OSError, KeyError, ValueError):
                cross_section = None
                resonance_energies = None

        if cross_section is None:
            params = database.resonance_parameters(symbol, mass)
            if isinstance(params, RMatrixParameters):
                result = rmatrix_total_cross_section(params, energies)
            else:
                result = total_cross_section(params, energies)
            cross_section = result.total
            resonance_energies = _resonance_energies(database, symbol, [mass], energies)
            if cached is not None:
                try:
                    np.savez_compressed(
                        cached,
                        cross_section=cross_section,
                        resonance_energies=resonance_energies,
                    )
                except OSError:
                    pass

        templates[name] = IsotopeTemplate(
            symbol=symbol,
            mass=mass,
            name=name,
            energies=energies,
            cross_section=np.asarray(cross_section, dtype=float),
            resonance_energies=np.asarray(resonance_energies, dtype=float),
        )

    return TemplateSet(templates={}, energies=energies, isotope_templates=templates)


def build_fine_isotope_templates(
    isotopes: list[str] | tuple[str, ...],
    energies: np.ndarray,
    *,
    refinement: int = 60,
    database: LocalENDFDatabase | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> TemplateSet:
    """Build isotope templates on a fine grid, for narrow resonances.

    This is the isotope equivalent of :func:`build_fine_templates`. The
    cross-sections are evaluated on a grid that is ``refinement`` times
    finer than the measurement grid, so that the peak of every resonance
    is resolved, and the result is sampled back onto the measurement grid
    by the caller after any broadening.

    Parameters
    ----------
    isotopes : list of str
        The isotopes to build templates for.
    energies : numpy.ndarray
        The energies in eV of the measurement grid.
    refinement : int
        How many times finer the evaluation grid should be.
    database : LocalENDFDatabase, optional
        The database to read from.
    cache_dir : str or pathlib.Path, optional
        Where to cache the built templates.
    use_cache : bool
        Whether to read and write the cache.

    Returns:
        TemplateSet
            The templates, on the fine grid.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.size < 2:
        raise ValueError("At least two energies are needed.")
    if refinement < 1:
        raise ValueError("The refinement factor must be at least one.")

    fine = np.logspace(
        np.log10(energies[0]),
        np.log10(energies[-1]),
        energies.size * refinement,
    )
    templates = build_isotope_templates(
        isotopes,
        fine,
        database=database,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    templates.measurement_energies = energies
    return templates


def _resonance_energies(
    database: LocalENDFDatabase,
    symbol: str,
    masses: list[int],
    energies: np.ndarray,
) -> np.ndarray:
    """Return the resolved resonance energies of an element's isotopes."""
    from .resonance import resonance_energies
    from .rmatrix import rmatrix_resonance_energies

    low = float(np.min(energies))
    high = float(np.max(energies))

    found = []
    for mass in masses:
        try:
            params = database.resonance_parameters(symbol, int(mass))
        except (FileNotFoundError, ValueError):
            continue
        if isinstance(params, RMatrixParameters):
            values = rmatrix_resonance_energies(params, low, high)
        else:
            values = resonance_energies(params, low, high)
        found.extend(float(value) for value in values)

    return np.sort(np.unique(np.asarray(found, dtype=float)))
