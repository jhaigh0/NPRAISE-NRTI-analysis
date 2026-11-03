"""A dictionary of known spectra, used to identify the contents of a sample.

The dictionary holds one entry per known component, each of them the average
measured spectrum of a region containing a single isotope (or a background
region). A spectrum is then explained as a non-negative combination of those
entries, which recovers the abundance of each known component.

The entries are built from *measured* data rather than from theory: the
measured spectrum is the differential response of the sample convolved with
the detector response and the background, which the total cross-section from
the ENDF database does not reproduce.

Example::

    dictionary = build_dictionary_from_files(
        ["Ag_Ag_n50_noise0.1_l2x4.npz", "Au_Au_n50_noise0.1_l2x4.npz"]
    )
    dictionary.save("dictionary.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Dictionary",
    "DictionaryEntry",
    "SelectionSpec",
    "build_dictionary_from_files",
    "build_dictionary_from_selections",
]


@dataclass
class DictionaryEntry:
    """One known component of the dictionary.

    Attributes:
        name: Name of the entry, unique within the dictionary.
        profile: The average spectrum of the entry, one value per bin.
        isotopes: The isotopes present in the entry. Empty for a background
            entry.
        is_background: Whether the entry describes a background region.
        n_samples: How many spectra the average was taken over.
        source: The file the entry was built from.
    """

    name: str
    profile: np.ndarray
    isotopes: list[str] = field(default_factory=list)
    is_background: bool = False
    n_samples: int = 0
    source: str = ""


@dataclass
class SelectionSpec:
    """One selection region to build a dictionary entry from.

    Attributes:
        name: Name of the entry, unique within the dictionary. It is usually
            given as ``workspace/selection`` so that the background regions
            of different workspaces stay separate.
        path: Location of the selection workspace file. It holds a subset of
            the detector IDs of the full workspace it came from.
        isotopes: The isotopes present in the region. An empty list means
            the region contains background only.
        workspace: Name of the workspace the region came from. It is used
            for the error messages and to name the loaded workspace.
    """

    name: str
    path: Path
    isotopes: list[str] = field(default_factory=list)
    workspace: str = ""


@dataclass
class Dictionary:
    """A set of known spectra that a measured spectrum can be fitted against.

    Attributes:
        entries: The known components.
        x_bins: The bin edges shared by every entry.
    """

    entries: list[DictionaryEntry]
    x_bins: np.ndarray

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        """The names of the entries, in order."""
        return [entry.name for entry in self.entries]

    @property
    def matrix(self) -> np.ndarray:
        """The dictionary as a matrix of shape (n_bins, n_entries).

        Each column is normalised to unit sum, so that the coefficients of a
        fit are directly proportional to the share of the total counts that
        the entry explains.
        """
        if not self.entries:
            raise ValueError("The dictionary is empty.")
        columns = []
        for entry in self.entries:
            total = entry.profile.sum()
            if total <= 0:
                raise ValueError(
                    f"Dictionary entry '{entry.name}' has no counts, so it "
                    "cannot be used as a dictionary column."
                )
            columns.append(entry.profile / total)
        return np.column_stack(columns)

    @property
    def column_totals(self) -> np.ndarray:
        """The total counts of each entry, used to convert coefficients back
        into the counts that each entry explains."""
        return np.asarray([entry.profile.sum() for entry in self.entries], dtype=float)

    def entry_index(self, name: str) -> int:
        """The index of the entry with the given name.

        Raises:
            KeyError: If no entry has that name.
        """
        for index, entry in enumerate(self.entries):
            if entry.name == name:
                return index
        raise KeyError(f"No dictionary entry named '{name}'.")

    def save(self, path: str | Path) -> Path:
        """Save the dictionary to a ``.npz`` file.

        Args:
            path: Where to write the file.

        Returns:
            The path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {
            "x_bins": np.asarray(self.x_bins),
            "entry_names": np.asarray(self.names, dtype=str),
            "profiles": np.column_stack([entry.profile for entry in self.entries]),
            "entry_isotopes": np.asarray(
                [",".join(entry.isotopes) for entry in self.entries], dtype=str
            ),
            "entry_is_background": np.asarray(
                [entry.is_background for entry in self.entries]
            ),
            "entry_n_samples": np.asarray([entry.n_samples for entry in self.entries]),
            "entry_sources": np.asarray(
                [entry.source for entry in self.entries], dtype=str
            ),
        }
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dictionary:
        """Load a dictionary from a ``.npz`` file written by :meth:`save`.

        Args:
            path: The file to read.

        Returns:
            The loaded dictionary.
        """
        with np.load(Path(path), allow_pickle=True) as data:
            names = [str(name) for name in data["entry_names"]]
            profiles = data["profiles"]
            isotopes = data["entry_isotopes"]
            is_background = data["entry_is_background"]
            n_samples = data["entry_n_samples"]
            sources = data["entry_sources"]
            x_bins = data["x_bins"].copy()

        entries = []
        for index, name in enumerate(names):
            entry_isotopes = str(isotopes[index])
            entries.append(
                DictionaryEntry(
                    name=name,
                    profile=profiles[:, index].copy(),
                    isotopes=[item for item in entry_isotopes.split(",") if item],
                    is_background=bool(is_background[index]),
                    n_samples=int(n_samples[index]),
                    source=str(sources[index]),
                )
            )
        return cls(entries=entries, x_bins=x_bins)


def build_dictionary_from_files(
    paths: list[str | Path], *, name_from: str = "selection_name"
) -> Dictionary:
    """Build a dictionary from generated data files.

    Each file becomes one entry, whose profile is the average of all the
    spectra in that file. Files of different noise levels or L2 values may be
    given for the same component, in which case their spectra are averaged
    together.

    Args:
        paths: The generated ``.npz`` files to build the entries from.
        name_from: Which metadata field gives the name of an entry. The
            default uses the selection name, so that the pure region of a
            workspace keeps its own name.

    Returns:
        The built dictionary.

    Raises:
        ValueError: If no files were given, or if the files do not share the
            same bin edges.
    """
    if not paths:
        raise ValueError("At least one file is needed to build a dictionary.")

    entries: list[DictionaryEntry] = []
    x_bins: np.ndarray | None = None

    for path in paths:
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            if "spectra_samples" not in data:
                raise ValueError(
                    f"'{path}' does not contain 'spectra_samples', so it is "
                    "not a generated data file."
                )
            spectra = data["spectra_samples"]
            file_x_bins = data["x_bins"].copy()

            if x_bins is None:
                x_bins = file_x_bins
            elif not np.array_equal(x_bins, file_x_bins):
                raise ValueError(
                    f"'{path}' has different bin edges to the other files, so "
                    "the spectra cannot be compared."
                )

            isotopes = [str(item) for item in data["isotopes"]]
            is_background = bool(data["is_background_only"])
            name = str(data[name_from]) if name_from in data else path.stem
            workspace = str(data["name"]) if "name" in data else ""

        # Distinguish the entries of different workspaces, so that the
        # background region of one workspace is not merged with another's.
        if workspace:
            name = f"{workspace}/{name}"

        profile = spectra.mean(axis=0)
        existing = next((entry for entry in entries if entry.name == name), None)
        if existing is None:
            entries.append(
                DictionaryEntry(
                    name=name,
                    profile=profile,
                    isotopes=isotopes,
                    is_background=is_background,
                    n_samples=int(spectra.shape[0]),
                    source=str(path),
                )
            )
        else:
            # Average the two profiles, weighted by how many spectra each
            # was built from.
            total = existing.n_samples + spectra.shape[0]
            existing.profile = (
                existing.profile * existing.n_samples + profile * spectra.shape[0]
            ) / total
            existing.n_samples = int(total)
            existing.source = f"{existing.source}, {path}"

    assert x_bins is not None
    return Dictionary(entries=entries, x_bins=x_bins)


def build_dictionary_from_selections(
    selections: list[SelectionSpec],
    *,
    l2_values: list[float] | None = None,
) -> Dictionary:
    """Build a dictionary from raw selection workspace files.

    This is the same idea as :func:`build_dictionary_from_files`, but it
    takes the selection workspaces directly, so a dictionary can be built
    for an element without first generating any training data for it.

    Each selection is converted from TOF to energy at every L2 value, and
    the profiles of those conversions are averaged together. The L2 values
    shift the energy axis by only a fraction of a bin, so pooling them
    gives an entry that is valid at any of them.

    The profile of an entry is the average of every detector of the region,
    which is the spectrum of a single detector. The entries of a dictionary
    built from generated data are instead averages of *summed* spectra, but
    the two give the same answer: the columns are normalised to unit sum
    before the fit, so only the shape of a profile matters.

    Args:
        selections: The selection regions to build the entries from.
        l2_values: The sample-detector distances to convert at. They should
            match the values the data to be fitted was converted at.

    Returns:
        The built dictionary.

    Raises:
        ValueError: If no selections were given, if a selection file cannot
            be read, or if the selections do not share the same bin edges.
    """
    if not selections:
        raise ValueError("At least one selection is needed to build a dictionary.")

    from mantid.simpleapi import Load

    from data_loading.load_and_normalise import convert_TOF_to_energy

    if l2_values is None:
        l2_values = [0.63]
    l2_values = [float(value) for value in l2_values]
    if any(value <= 0 for value in l2_values):
        raise ValueError("The L2 values must all be greater than zero.")

    entries: list[DictionaryEntry] = []
    x_bins: np.ndarray | None = None

    for spec in selections:
        path = Path(spec.path)
        if not path.is_file():
            raise ValueError(f"Selection file not found: {path}")

        workspace_name = spec.workspace or spec.name
        try:
            workspace = Load(
                str(path),
                OutputWorkspace=f"{workspace_name}_selection",
                StoreInADS=False,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not load the selection workspace '{path}': {exc}"
            ) from exc

        # Each L2 value needs its own copy of the workspace, because the
        # conversion moves the instrument components of the workspace it is
        # given.
        profiles = []
        for l2 in l2_values:
            converted = convert_TOF_to_energy(
                workspace,
                l2=l2,
                output_name=f"{workspace_name}_energy_{l2:g}",
            )
            if x_bins is None:
                x_bins = converted.readX(0).copy()
            elif not np.array_equal(x_bins, converted.readX(0)):
                raise ValueError(
                    f"'{path}' has different bin edges to the other "
                    "selections, so the spectra cannot be compared."
                )
            profiles.append(converted.extractY().mean(axis=0))

        profile = np.mean(profiles, axis=0)
        n_detectors = workspace.getNumberHistograms()

        entries.append(
            DictionaryEntry(
                name=spec.name,
                profile=profile,
                isotopes=list(spec.isotopes),
                is_background=not spec.isotopes,
                n_samples=n_detectors * len(l2_values),
                source=str(path),
            )
        )

    assert x_bins is not None
    return Dictionary(entries=entries, x_bins=x_bins)
