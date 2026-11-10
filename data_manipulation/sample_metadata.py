"""Metadata about the measured standards: thickness, composition and density.

The ENDF fit can answer two different questions, and the metadata says
which one is possible:

**Discovery** - which elements are in the sample? No metadata is needed.
The fit is free, and the answer is limited by the energy range: only
elements with resolved resonances can be identified.

**Quantification** - how much of each element is there? The metadata is
needed, because the depth of a dip gives the *product* of the amount of
material and the cross-section, and only the metadata breaks that
degeneracy.

This module holds that metadata in one place, in a form that both the YAML
configuration and the Excel spreadsheet of the standards can be read into.

A :class:`SampleMetadata` can be built from

* a YAML file, which is the recommended route, because it can be written to
  match the names of the regions of a dictionary exactly;
* the Excel spreadsheet of the standards, whose ``Details`` sheet lists the
  sample, its mass, its thickness, its area and its elements.

The two derived quantities that the fit needs are

``mass_areal_density``
    the mass of the sample per unit area, in grams per square centimetre.
    It is ``density * thickness``, or ``mass / area`` when the mass and the
    area are known.

``total_areal_density``
    the number of atoms per unit area, and then the number of atoms per
    barn, which is the quantity that multiplies a cross-section in barns
    to give an optical thickness. This needs the mean molar mass of the
    sample, which comes from the composition.

Example::

    from data_manipulation.sample_metadata import load_metadata

    metadata = load_metadata("configs/standards_metadata.yaml")
    hav = metadata["Havar/Havar_5foils"]
    print(hav.thickness_cm, hav.mass_areal_density)
    print(hav.total_areal_density)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ELEMENT_DENSITIES",
    "SampleMetadata",
    "atomic_mass",
    "load_metadata",
    "load_metadata_excel",
    "load_metadata_yaml",
]


# The density of each element in grams per cubic centimetre, at room
# temperature. These are the standard handbook values; they are used to
# turn a thickness into a mass per unit area. The density of the sample as
# a whole is not the average of these, but the reciprocal of the mass
# weighted average of their reciprocals.
ELEMENT_DENSITIES: dict[str, float] = {
    "H": 0.00008988,
    "B": 2.34,
    "C": 2.267,
    "N": 0.0012506,
    "O": 0.001429,
    "F": 0.001696,
    "Na": 0.971,
    "Mg": 1.738,
    "Al": 2.70,
    "Si": 2.329,
    "P": 1.823,
    "S": 2.07,
    "K": 0.862,
    "Ca": 1.55,
    "Ti": 4.506,
    "V": 6.11,
    "Cr": 7.19,
    "Mn": 7.21,
    "Fe": 7.874,
    "Co": 8.90,
    "Ni": 8.908,
    "Cu": 8.96,
    "Zn": 7.14,
    "Ga": 5.91,
    "Ge": 5.323,
    "As": 5.727,
    "Se": 4.81,
    "Br": 3.1028,
    "Rb": 1.532,
    "Sr": 2.64,
    "Y": 4.472,
    "Zr": 6.52,
    "Nb": 8.57,
    "Mo": 10.28,
    "Ru": 12.45,
    "Rh": 12.41,
    "Pd": 12.023,
    "Ag": 10.49,
    "Cd": 8.65,
    "In": 7.31,
    "Sn": 7.265,
    "Sb": 6.685,
    "Te": 6.232,
    "I": 4.933,
    "Xe": 0.005894,
    "Cs": 1.873,
    "Ba": 3.51,
    "La": 6.162,
    "Ce": 6.770,
    "W": 19.25,
    "Ta": 16.69,
    "Hf": 13.31,
    "Re": 21.02,
    "Os": 22.59,
    "Ir": 22.56,
    "Pt": 21.45,
    "Au": 19.30,
    "Hg": 13.534,
    "Tl": 11.85,
    "Pb": 11.34,
    "Bi": 9.78,
    "Th": 11.72,
    "U": 19.1,
}


# The Avogadro constant, in molecules per mole.
AVOGADRO = 6.02214076e23


def atomic_mass(symbol: str) -> float:
    """Return the standard atomic weight of an element.

    Parameters
    ----------
    symbol : str
        Element symbol, e.g. ``'Cu'``.

    Returns
    -------
    float
        The standard atomic weight in grams per mole.

    Raises
    ------
    ValueError
        If the symbol is not a known element.
    """
    import periodictable

    try:
        return float(getattr(periodictable, symbol).mass)
    except AttributeError as exc:
        raise ValueError(f"'{symbol}' is not a known element symbol.") from exc


@dataclass
class SampleMetadata:
    """What is known about one measured sample.

    Attributes:
        name: The name of the sample. It matches the name of a region of a
            dictionary, so that the metadata can be looked up by name.
        thickness_cm: The thickness of the sample in centimetres.
        composition: The composition of the sample, as the mass fraction of
            each element. The values are normalised to sum to one.
        density: The density of the sample in grams per cubic centimetre.
            When it is not given it is worked out from the composition.
        area_cm2: The area of the sample in square centimetres.
        mass_g: The mass of the sample in grams.
        floor: The transmission floor of the measurement of this sample,
            which is the ratio of the sample side to the empty side
            background. It is specific to each measurement.
        notes: Free text notes about the sample.
    """

    name: str
    thickness_cm: float | None = None
    composition: dict[str, float] = field(default_factory=dict)
    density: float | None = None
    area_cm2: float | None = None
    mass_g: float | None = None
    floor: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.thickness_cm is not None and self.thickness_cm <= 0:
            raise ValueError(
                f"The thickness of '{self.name}' must be greater than zero."
            )
        if self.composition:
            total = sum(self.composition.values())
            if total <= 0:
                raise ValueError(
                    f"The composition of '{self.name}' does not add up to anything."
                )
            self.composition = {
                symbol: value / total
                for symbol, value in self.composition.items()
                if value > 0
            }

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def number_fractions(self) -> dict[str, float]:
        """The number fraction of each element."""
        if not self.composition:
            return {}
        numbers = {
            symbol: mass / atomic_mass(symbol)
            for symbol, mass in self.composition.items()
        }
        total = sum(numbers.values())
        if total <= 0:
            return {}
        return {symbol: value / total for symbol, value in numbers.items()}

    @property
    def mean_molar_mass(self) -> float:
        """The mean molar mass of the sample, in grams per mole."""
        numbers = self.number_fractions
        if not numbers:
            raise ValueError(
                f"The composition of '{self.name}' is not known, so its "
                "mean molar mass is not known either."
            )
        return sum(
            fraction * atomic_mass(symbol) for symbol, fraction in numbers.items()
        )

    @property
    def mixture_density(self) -> float | None:
        """The density of the sample, in grams per cubic centimetre.

        For a unit of total mass, the volume of each element is its mass
        fraction divided by its density, so the total volume is the sum of
        those and the density of the mixture is its reciprocal. This is
        the mass weighted harmonic mean of the densities.

        It is only available when the composition is known.
        """
        if self.density is not None:
            return self.density
        if not self.composition:
            return None
        inverse = 0.0
        for symbol, fraction in self.composition.items():
            density = ELEMENT_DENSITIES.get(symbol)
            if density is None:
                return None
            inverse += fraction / density
        if inverse <= 0:
            return None
        return 1.0 / inverse

    @property
    def mass_areal_density(self) -> float | None:
        """The mass of the sample per unit area, in grams per square cm.

        It is worked out from the thickness and the density when they are
        known, and from the mass and the area otherwise.
        """
        if self.thickness_cm is not None:
            density = self.mixture_density
            if density is not None:
                return density * self.thickness_cm
        if self.mass_g is not None and self.area_cm2:
            return self.mass_g / self.area_cm2
        return None

    @property
    def total_areal_density(self) -> float | None:
        """The total number of atoms per barn of the sample.

        This is the quantity that multiplies a cross-section in barns to
        give an optical thickness. It is the number of atoms per square
        centimetre, divided by 1e24.
        """
        areal = self.mass_areal_density
        if areal is None:
            return None
        molar_mass = self.mean_molar_mass
        if molar_mass <= 0:
            return None
        atoms_per_cm2 = areal / molar_mass * AVOGADRO
        return atoms_per_cm2 / 1e24

    def element_areal_densities(self) -> dict[str, float]:
        """The number of atoms per barn of each element.

        The empty dictionary is returned when the composition or the total
        areal density is not known.
        """
        total = self.total_areal_density
        if total is None:
            return {}
        return {
            symbol: total * fraction
            for symbol, fraction in self.number_fractions.items()
        }

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> SampleMetadata:
        """Build a :class:`SampleMetadata` from a dictionary."""
        known = {
            "thickness_cm",
            "composition",
            "density",
            "area_cm2",
            "mass_g",
            "floor",
            "notes",
        }
        unknown = set(raw) - known - {"thickness_mm", "elements"}
        if unknown:
            raise ValueError(
                f"The metadata for '{name}' has unknown keys: "
                f"{', '.join(sorted(unknown))}."
            )

        thickness_cm = raw.get("thickness_cm")
        thickness_mm = raw.get("thickness_mm")
        if thickness_cm is None and thickness_mm is not None:
            thickness_cm = float(thickness_mm) / 10.0

        composition = raw.get("composition") or {}
        if not composition and raw.get("elements"):
            # A list of element names with no fractions, which is only
            # useful for a sample of one element.
            elements = list(raw["elements"])
            if len(elements) != 1:
                raise ValueError(
                    f"The metadata for '{name}' lists several elements but "
                    "no composition, so the fractions are unknown."
                )
            composition = {elements[0]: 1.0}

        return cls(
            name=name,
            thickness_cm=(float(thickness_cm) if thickness_cm is not None else None),
            composition={
                str(symbol): float(value) for symbol, value in composition.items()
            },
            density=(float(raw["density"]) if raw.get("density") is not None else None),
            area_cm2=(
                float(raw["area_cm2"]) if raw.get("area_cm2") is not None else None
            ),
            mass_g=(float(raw["mass_g"]) if raw.get("mass_g") is not None else None),
            floor=float(raw["floor"]) if raw.get("floor") is not None else None,
            notes=str(raw.get("notes", "")),
        )


def load_metadata_yaml(path: str | Path) -> dict[str, SampleMetadata]:
    """Read sample metadata from a YAML file.

    The file maps the name of a region to its metadata::

        Havar/Havar_5foils:
          thickness_mm: 2.5
          composition:
            Co: 42.5
            Cr: 20.0
            Ni: 13.0
            Fe: 19.05
            W: 2.7
            Mo: 2.2
            Mn: 1.6

    Parameters
    ----------
    path : str or pathlib.Path
        The YAML file to read.

    Returns
    -------
    dict
        The metadata, keyed by the name of the region.

    Raises
    ------
    OSError
        If the file cannot be read.
    TypeError
        If the file does not hold a mapping of regions to metadata.
    """
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"'{path}' does not hold a mapping of regions to metadata.")
    return {
        str(name): SampleMetadata.from_dict(str(name), dict(raw or {}))
        for name, raw in loaded.items()
    }


def _parse_composition(text: str) -> dict[str, float]:
    """Parse a composition written as ``Co42.5/Cr20/Ni13/Fe``.

    The names of the standards use this form, for example
    ``Co42.5/Cr20/Ni13/Fe/W/Mo/Mn``. The elements that carry a number get
    that weight, and the ones that do not share what is left equally.
    """
    parts = [part.strip() for part in str(text).split("/") if part.strip()]
    if not parts:
        return {}

    weights: dict[str, float] = {}
    plain: list[str] = []
    for part in parts:
        symbol = "".join(character for character in part if character.isalpha())
        digits = "".join(character for character in part if character.isdigit())
        if not symbol:
            continue
        if digits:
            weights[symbol] = float(digits)
        else:
            plain.append(symbol)
    if plain:
        # The elements with no number share the rest equally. If the
        # numbered ones already add up to more than the whole, they are
        # normalised on their own.
        numbered = sum(weights.values())
        remainder = max(0.0, 100.0 - numbered)
        share = remainder / len(plain) if plain else 0.0
        for symbol in plain:
            weights[symbol] = share
    return {symbol: value for symbol, value in weights.items() if value > 0}


def load_metadata_excel(path: str | Path) -> dict[str, SampleMetadata]:
    """Read sample metadata from the Excel spreadsheet of the standards.

    The ``Details`` sheet of the spreadsheet lists the samples. Its columns
    are read by name, so that the layout of the sheet does not have to be
    exactly the one that is expected.

    The composition of a sample is taken from the name of the sample when
    the name carries one, for example ``Co42.5/Cr20/Ni13/Fe/W/Mo/Mn``.
    The thickness is in millimetres and the mass is in grams.

    Parameters
    ----------
    path : str or pathlib.Path
        The spreadsheet to read.

    Returns
    -------
    dict
        The metadata, keyed by the name of the sample.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file does not have a ``Details`` sheet.
    """
    import openpyxl

    path = Path(path)
    workbook = openpyxl.load_workbook(path, data_only=True)
    if "Details" not in workbook.sheetnames:
        raise ValueError(f"'{path}' has no 'Details' sheet.")
    sheet = workbook["Details"]

    # Find the header row, which is the first row that has a cell
    # containing "Sample".
    header: list[str] = []
    header_row = 0
    for index, row in enumerate(sheet.iter_rows(values_only=True)):
        values = [str(value).strip() for value in row if value is not None]
        if any(value == "Sample" for value in values):
            header = ["" if value is None else str(value).strip() for value in row]
            header_row = index
            break
    if not header:
        raise ValueError(f"The 'Details' sheet of '{path}' has no 'Sample' column.")

    def column(label: str) -> int | None:
        for index, value in enumerate(header):
            if value.lower().startswith(label.lower()):
                return index
        return None

    sample_col = column("Sample")
    mass_col = column("Mass")
    thickness_col = column("Thickness")
    elements_col = column("Elements")
    if sample_col is None or thickness_col is None:
        raise ValueError(
            f"The 'Details' sheet of '{path}' needs a 'Sample' column and a "
            "'Thickness' column."
        )

    metadata: dict[str, SampleMetadata] = {}
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        if not row or sample_col >= len(row):
            continue
        name = row[sample_col]
        if name is None or not str(name).strip():
            continue
        name = str(name).strip()

        thickness = None
        if thickness_col < len(row) and row[thickness_col] is not None:
            try:
                thickness = float(row[thickness_col])
            except (TypeError, ValueError):
                thickness = None

        mass = None
        if mass_col is not None and mass_col < len(row) and row[mass_col] is not None:
            try:
                mass = float(row[mass_col])
            except (TypeError, ValueError):
                mass = None

        elements: list[str] = []
        if (
            elements_col is not None
            and elements_col < len(row)
            and row[elements_col] is not None
        ):
            elements = [
                part.strip()
                for part in str(row[elements_col]).split(",")
                if part.strip()
            ]

        composition = _parse_composition(name)
        if not composition and len(elements) == 1:
            composition = {elements[0]: 1.0}

        if thickness is None and not composition:
            continue

        metadata[name] = SampleMetadata(
            name=name,
            thickness_cm=thickness / 10.0 if thickness is not None else None,
            composition=composition,
            mass_g=mass,
        )
    return metadata


def load_metadata(path: str | Path) -> dict[str, SampleMetadata]:
    """Read sample metadata from a YAML file or an Excel spreadsheet.

    The format is chosen from the extension of the file.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.

    Returns
    -------
    dict
        The metadata, keyed by the name of the region.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return load_metadata_excel(path)
    return load_metadata_yaml(path)
