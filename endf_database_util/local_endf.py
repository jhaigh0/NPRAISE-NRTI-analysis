"""Read neutron total cross-sections from a local ENDF/B-VIII.0 database.

This module complements :mod:`endf_database_util.endf_database`, which
downloads a tabulated total cross-section from the KAERI web service. The web
plot data is coarse in the resonance region for some nuclides (and, for
copper, the interesting resonances belong to an isotope whose R-matrix
evaluation is not resolved by that output at all).

Reading the ENDF evaluation files directly lets us use the *resolved
resonance parameters* (MF=2, MT=151) instead of the coarse tabulated data
(MF=3, MT=1), which is what determines the sharp dips observed in the nGEM
spectra.

The ENDF files are parsed with the ``endf`` package
(https://github.com/paulromano/endf-python), except for the R-matrix
formalism (LRF=7), which that package mis-parses; :func:`parse_rmatrix`
implements the ENDF-102 record layout directly.

Expected layout of the database::

    <database>/neutrons/n-029_Cu_063.endf
    <database>/neutrons/n-030_Zn_064.endf
    ...

"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "LocalENDFDatabase",
    "RMatrixParameters",
    "ResolvedResonance",
    "ResonanceParameters",
    "find_database_root",
]


# ENDF stores floats without an exponent character, e.g. ``-1.23481+10``.
_ENDF_FLOAT_RE = re.compile(r"([\s\-+]?\d*\.\d+)([+\-]) ?(\d+)")


def _f(text: str) -> float:
    """Convert an ENDF-6 formatted float field to a Python float."""
    return float(_ENDF_FLOAT_RE.sub(r"\1e\2\3", text))


def _cont(line: str) -> tuple[float, float, int, int, int, int]:
    """Split an ENDF CONT record into its six fields."""
    return (
        _f(line[0:11]),
        _f(line[11:22]),
        int(line[22:33] or 0),
        int(line[33:44] or 0),
        int(line[44:55] or 0),
        int(line[55:66] or 0),
    )


def _read_list(lines: list[str], index: int) -> tuple[tuple, np.ndarray, int]:
    """Read an ENDF LIST record whose CONT header is ``lines[index]``.

    Returns the header tuple, the flattened values, and the index of the next
    line after the record.
    """
    header = _cont(lines[index])
    npl = header[4]
    values: list[float] = []
    cursor = index + 1
    for _ in range((npl - 1) // 6 + 1):
        line = lines[cursor]
        values.extend(_f(line[11 * k : 11 * (k + 1)]) for k in range(6))
        cursor += 1
    return header, np.asarray(values), cursor


@dataclass
class ResolvedResonance:
    """A single resolved resonance from an MLBW/Reich-Moore evaluation."""

    energy: float
    """Resonance energy in eV."""
    j: float
    """Nuclear spin :math:`J`."""
    l: int
    """Orbital angular momentum :math:`l`."""
    widths: dict[str, float]
    """Partial widths in eV, keyed by ``'n'`` (neutron), ``'g'`` (capture),
    ``'f'`` (fission) and ``'t'`` (total)."""


@dataclass
class LGroup:
    """The parameters that ENDF-102 gives per orbital angular momentum."""

    l: int
    """Orbital angular momentum quantum number."""
    atomic_weight_ratio: float
    """Atomic weight ratio for this :math:`l` (the ``AWRI`` field)."""
    scattering_radius: float
    """Scattering radius for this :math:`l` (the ``APL`` field, falling back
    to the range-level ``AP`` when it is zero)."""
    nro: int
    """Flag for an energy-dependent scattering radius."""
    naps: int
    """Flag selecting the channel radius (0 = standard, 1 = scattering
    radius, 2 = per-:math:`l` scattering radius)."""
    channel_radius: float | None = None
    """Explicit channel radius, overriding the ``naps`` selection. Used by the
    R-matrix formalism, which supplies the effective channel radius (APE)
    directly."""


@dataclass
class ResonanceParameters:
    """Resolved resonance parameters for one nuclide."""

    symbol: str
    """Nuclide symbol, e.g. ``'29-Cu- 63'``."""
    formalism: int
    """ENDF resonance formalism (1=SLBW, 2=MLBW, 3=Reich-Moore, 7=R-matrix)."""
    target_spin: float
    """Intrinsic spin :math:`I` of the target."""
    scattering_radius: float
    """Energy-independent scattering radius in fm."""
    atomic_weight_ratio: float
    """Atomic weight ratio of the target."""
    resonances: list[ResolvedResonance] = field(default_factory=list)
    """Resolved resonances, in increasing energy order."""
    l_groups: dict[int, LGroup] = field(default_factory=dict)
    """Per-:math:`l` channel parameters."""


@dataclass
class RMatrixParameters:
    """R-matrix (LRF=7) resolved resonance parameters for one nuclide."""

    symbol: str
    """Nuclide symbol, e.g. ``'29-Cu- 63'``."""
    energy_min: float
    """Lower bound of the resolved range in eV."""
    energy_max: float
    """Upper bound of the resolved range in eV."""
    particle_pairs: np.ndarray
    """Particle-pair table with 12 columns per pair."""
    spin_groups: list[dict]
    """Spin groups; each holds ``J``, ``parity``, ``channels`` and
    ``resonances`` (energies and widths)."""


class LocalENDFDatabase:
    """Reader for a local ENDF/B-VIII.0 neutron sublibrary.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Path to the database root. If omitted, :func:`find_database_root`
        is used to locate it relative to the current working directory.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = find_database_root()
        self.root = Path(root)
        self.neutrons_dir = self.root / "neutrons"

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def find_file(self, symbol: str, mass: int) -> Path:
        """Return the ENDF file for a given element symbol and mass number.

        Parameters
        ----------
        symbol : str
            Element symbol, e.g. ``'Cu'``.
        mass : int
            Mass number, e.g. ``63``.

        Returns
        -------
        pathlib.Path
            Path to the matching ``.endf`` file.

        Raises
        ------
        FileNotFoundError
            If no file matches the requested nuclide.
        """
        z = _atomic_number(symbol)
        candidates = sorted(self.neutrons_dir.glob(f"n-{z:03d}_{symbol}_*.endf"))
        for path in candidates:
            stem = path.stem
            # e.g. n-029_Cu_063 or n-029_Cu_058m1
            suffix = stem.split("_", 2)[2]
            if suffix == f"{mass:03d}" or suffix.startswith(f"{mass:03d}m"):
                return path
        raise FileNotFoundError(
            f"No ENDF file for {symbol}-{mass} in {self.neutrons_dir}"
        )

    def available_isotopes(self) -> list[str]:
        """Return the nuclide symbols available in the neutron sublibrary."""
        out = []
        for path in sorted(self.neutrons_dir.glob("n-*.endf")):
            parts = path.stem.split("_", 2)
            if len(parts) == 3:
                out.append(f"{parts[1]}-{parts[2]}")
        return out

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def read_material(self, symbol: str, mass: int):
        """Return the parsed :class:`endf.Material` for a nuclide."""
        import endf

        path = self.find_file(symbol, mass)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return endf.get_materials(str(path))[0]

    def tabulated_total_cross_section(
        self, symbol: str, mass: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the MF=3/MT=1 tabulated total cross-section.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Energies in eV and cross-sections in barns. The tabulated grid is
            coarse in the resonance region for many nuclides, so this is
            usually *not* the best source of resonance information.
        """
        material = self.read_material(symbol, mass)
        tabulated = material.section_data[(3, 1)]["sigma"]
        return np.asarray(tabulated.x, dtype=float), np.asarray(
            tabulated.y, dtype=float
        )

    def resonance_parameters(
        self, symbol: str, mass: int
    ) -> ResonanceParameters | RMatrixParameters:
        """Return the resolved resonance parameters for a nuclide.

        MLBW (LRF=2) and Reich-Moore (LRF=3) evaluations are handled through
        the ``endf`` package; the R-matrix formalism (LRF=7) is parsed here
        directly because the library mis-parses it.
        """
        material = self.read_material(symbol, mass)
        symbol_text = str(material[1, 451]["ZSYMAM"]).strip()

        raw = material.section_text.get((2, 151))
        if raw is None:
            raise ValueError(f"{symbol_text} has no MF=2 resonance section")

        lines = raw.splitlines()
        formalism = _cont(lines[2])[3]
        if formalism == 7:
            return self._parse_rmatrix(symbol_text, lines)
        return self._parse_mlbw(symbol_text, material, lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_mlbw(
        symbol_text: str, material, lines: list[str]
    ) -> ResonanceParameters:
        """Parse an SLBW/MLBW/Reich-Moore resolved resonance section."""
        target_spin = 0.0
        scattering_radius = 0.0
        atomic_weight_ratio = material[1, 451]["AWR"]
        resonances: list[ResolvedResonance] = []
        l_groups: dict[int, LGroup] = {}

        for isotope in material[2, 151]["isotopes"]:
            for region in isotope["ranges"]:
                if region.get("LRU") != 1:
                    continue
                if "SPI" in region:
                    target_spin = region["SPI"]
                if "AP" in region:
                    scattering_radius = region["AP"]
                nro = int(region.get("NRO", 0) or 0)
                naps = int(region.get("NAPS", 0) or 0)
                for section in region.get("sections", []):
                    ell = int(section["L"])
                    # The per-l scattering radius is the APL field, falling
                    # back to the range-level AP when APL is zero.
                    apl = float(section.get("APL") or 0.0)
                    if apl == 0.0:
                        apl = scattering_radius
                    l_groups[ell] = LGroup(
                        l=ell,
                        atomic_weight_ratio=float(
                            section.get("AWRI") or atomic_weight_ratio
                        ),
                        scattering_radius=apl,
                        nro=nro,
                        naps=naps,
                    )
                    energies = np.asarray(section["ER"], dtype=float)
                    spins = np.asarray(section["AJ"], dtype=float)
                    gamma_n = np.asarray(section["GN"], dtype=float)
                    gamma_g = np.asarray(section["GG"], dtype=float)
                    if "GF" in section:
                        gamma_f = np.asarray(section["GF"], dtype=float)
                    else:
                        gamma_f = np.zeros_like(gamma_n)
                    if "GT" in section:
                        gamma_t = np.asarray(section["GT"], dtype=float)
                    else:
                        gamma_t = gamma_n + gamma_g + gamma_f
                    for i in range(len(energies)):
                        resonances.append(
                            ResolvedResonance(
                                energy=float(energies[i]),
                                j=float(spins[i]),
                                l=ell,
                                widths={
                                    "n": float(gamma_n[i]),
                                    "g": float(gamma_g[i]),
                                    "f": float(gamma_f[i]),
                                    "t": float(gamma_t[i]),
                                },
                            )
                        )

        resonances.sort(key=lambda r: r.energy)
        return ResonanceParameters(
            symbol=symbol_text,
            formalism=_cont(lines[2])[3],
            target_spin=target_spin,
            scattering_radius=scattering_radius,
            atomic_weight_ratio=atomic_weight_ratio,
            resonances=resonances,
            l_groups=l_groups,
        )

    @staticmethod
    def _parse_rmatrix(symbol_text: str, lines: list[str]) -> RMatrixParameters:
        """Parse an R-matrix (LRF=7) resolved resonance section.

        This implements the ENDF-102 record layout directly. The ``endf``
        package reads these sections incorrectly: it uses ``values[::NCH + 1]``
        as the resonance stride, where a resonance row actually occupies
        ``6 * (NCH // 6 + 1)`` values.
        """
        energy_min, energy_max = _cont(lines[2])[0:2]
        n_spin_groups = _cont(lines[3])[4]
        n_pairs = _cont(lines[4])[2]

        # Particle pairs: NPP rows of 12 values, with no LIST header.
        values: list[float] = []
        cursor = 5
        while len(values) < n_pairs * 12:
            line = lines[cursor]
            values.extend(_f(line[11 * k : 11 * (k + 1)]) for k in range(6))
            cursor += 1
        particle_pairs = np.asarray(values[: n_pairs * 12]).reshape(n_pairs, 12)

        spin_groups: list[dict] = []
        for _ in range(n_spin_groups):
            header, channel_values, cursor = _read_list(lines, cursor)
            j_value = header[0]
            parity = header[1]
            kbk = header[2]
            kps = header[3]
            n_channels = header[5]
            channels = channel_values.reshape(-1, 6)

            res_header, res_values, cursor = _read_list(lines, cursor)
            n_resonances = res_header[3]
            stride = 6 * (n_channels // 6 + 1)
            block = res_values.reshape(-1, stride)
            energies = block[:, 0]
            widths = block[:, 1 : 1 + n_channels]

            spin_groups.append(
                {
                    "J": j_value,
                    "parity": parity,
                    "kbk": kbk,
                    "kps": kps,
                    "channels": channels,
                    "energies": energies[:n_resonances],
                    "widths": widths[:n_resonances],
                }
            )

        return RMatrixParameters(
            symbol=symbol_text,
            energy_min=energy_min,
            energy_max=energy_max,
            particle_pairs=particle_pairs,
            spin_groups=spin_groups,
        )


def _atomic_number(symbol: str) -> int:
    """Return the atomic number for an element symbol."""
    import periodictable

    try:
        return getattr(periodictable, symbol).number
    except AttributeError as exc:
        raise ValueError(f"Unknown element symbol: {symbol}") from exc


def find_database_root(start: str | Path | None = None) -> Path:
    """Locate an ENDF/B-VIII.0 database root.

    Searches the current directory and its parents for a directory containing
    a ``neutrons`` subdirectory with ENDF files in it.

    Raises
    ------
    FileNotFoundError
        If no database can be located.
    """
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        # The release ships as <root>/ENDF-B-VIII.0/ENDF-B-VIII.0/neutrons/,
        # so search the candidate itself and a few levels of nesting.
        for base in (candidate, *candidate.glob("*")):
            if not base.is_dir():
                continue
            for level in (base, *base.glob("*")):
                if not level.is_dir():
                    continue
                neutrons = level / "neutrons"
                if neutrons.is_dir() and any(neutrons.glob("n-*.endf")):
                    return level
    raise FileNotFoundError(
        "Could not locate an ENDF/B-VIII.0 database (a directory containing "
        "'neutrons/n-*.endf' files)."
    )
