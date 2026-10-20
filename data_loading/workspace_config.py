"""Parsing and validation of the training data generation configuration file.

The configuration file is YAML and describes every workspace that training data
should be generated from. Each workspace names:

* ``workspace``  - the full detector workspace (e.g. a Mantid ``.nxs`` file).
* ``selections`` - a list of *selection regions*. Each region is a cropped
  part of the detector, described by a selection workspace file holding a
  subset of the detector IDs present in the full workspace. A region contains
  a single isotope, a mix of isotopes, or no isotope at all (background only).
* ``isotopes``   - the isotope(s) contained in a selection region. This is
  metadata only; it is stored alongside the generated samples so that training
  scripts can build labels from it. An empty list means the region contains
  background only.
* ``label``      - an optional label value (any type) stored with the data.

Paths are resolved relative to the location of the configuration file, so a
config and its data can be moved around together.

Example configuration::

    defaults:
      l2: 0.6
      noise_level: 0.1
      n_sets: 10
      n_samples: [10, 20, 50, 100]

    workspaces:
      - name: Sn_letters_Bronze_sheet
        workspace: data/Sn_letters_Bronze_sheet_normMonitor.nxs
        selections:
          - name: tin
            selection: data/sel_tin.nxs
            isotopes: [Sn-118, Sn-120]
          - name: silver
            selection: data/sel_silver.nxs
            isotopes: [Ag-107, Ag-109]
          - name: background
            selection: data/sel_background.nxs
            isotopes: []
            label: 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "SelectionEntry",
    "TrainingConfig",
    "WorkspaceEntry",
    "first_not_none",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when the training data configuration file is invalid."""


@dataclass
class SelectionEntry:
    """A single selection region within a workspace.

    Attributes:
        selection: Location of the selection workspace file. It holds a
            subset of the detector IDs of the full workspace.
        isotopes: The isotope(s) present in the region. An empty list means
            the region contains background only.
        name: Name for the region, used in the output file name.
        label: Optional label stored with the generated data.
    """

    selection: Path
    isotopes: list[str] = field(default_factory=list)
    name: str = ""
    label: str | int | float | None = None

    @property
    def is_background_only(self) -> bool:
        """Whether the region contains no isotope, i.e. background only."""
        return not self.isotopes


@dataclass
class WorkspaceEntry:
    """A single workspace to generate training data from."""

    name: str
    workspace: Path
    selections: list[SelectionEntry] = field(default_factory=list)
    label: str | int | float | None = None
    l2: float | None = None


@dataclass
class TrainingConfig:
    """The parsed contents of a training data configuration file."""

    workspaces: list[WorkspaceEntry]
    l2: float = 0.6
    noise_level: float = 0.1
    n_sets: int = 10
    n_samples: list[int] = field(default_factory=list)
    source: Path | None = None


def first_not_none(*values: Any) -> Any:
    """Return the first value that is not ``None``, or ``None`` if there is none."""
    for value in values:
        if value is not None:
            return value
    return None


def load_config(path: str | Path) -> TrainingConfig:
    """Load and validate a YAML training data configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed :class:`TrainingConfig`.

    Raises:
        ConfigError: If the file is missing, malformed, or semantically invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Could not parse YAML configuration {config_path}: {exc}"
        ) from exc

    if raw is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration file must contain a mapping at the top level: {config_path}"
        )

    base_dir = config_path.resolve().parent
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ConfigError("'defaults' must be a mapping.")

    workspaces_raw = raw.get("workspaces")
    if workspaces_raw is None:
        raise ConfigError("Configuration must define a 'workspaces' list.")
    if not isinstance(workspaces_raw, list) or not workspaces_raw:
        raise ConfigError("'workspaces' must be a non-empty list.")

    workspaces = [
        _parse_entry(entry, base_dir, index)
        for index, entry in enumerate(workspaces_raw)
    ]
    _check_duplicate_names(workspaces)

    return TrainingConfig(
        workspaces=workspaces,
        l2=_as_float(defaults.get("l2", 0.6), "defaults.l2"),
        noise_level=_as_float(defaults.get("noise_level", 0.1), "defaults.noise_level"),
        n_sets=_as_int(defaults.get("n_sets", 10), "defaults.n_sets"),
        n_samples=_as_int_list(defaults.get("n_samples", []), "defaults.n_samples"),
        source=config_path.resolve(),
    )


def _parse_entry(entry: Any, base_dir: Path, index: int) -> WorkspaceEntry:
    """Validate and convert a single ``workspaces`` entry."""
    context = f"workspaces[{index}]"
    if not isinstance(entry, dict):
        raise ConfigError(f"{context} must be a mapping.")

    workspace_raw = entry.get("workspace")
    if not workspace_raw:
        raise ConfigError(f"{context} must define a 'workspace' file location.")
    workspace = _resolve_path(workspace_raw, base_dir, f"{context}.workspace")

    name = entry.get("name") or workspace.stem
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{context}.name must be a non-empty string.")
    name = name.strip()

    label = entry.get("label")
    if label is not None and not isinstance(label, (str, int, float)):
        raise ConfigError(f"{context}.label must be a string, number or null.")

    l2 = entry.get("l2")
    l2 = _as_float(l2, f"{context}.l2") if l2 is not None else None

    selections = _parse_selections(entry, base_dir, context, label)

    return WorkspaceEntry(
        name=name,
        workspace=workspace,
        selections=selections,
        label=label,
        l2=l2,
    )


def _parse_selections(
    entry: dict, base_dir: Path, context: str, workspace_label: Any
) -> list[SelectionEntry]:
    """Validate and convert the selection regions of a workspace.

    Two forms are supported. The ``selections`` key holds a list of regions,
    which is the normal form. The ``selection`` key holds a single region, for
    workspaces that only have one; its isotopes are then given by the
    workspace level ``isotopes`` key.
    """
    has_selections = "selections" in entry
    has_selection = "selection" in entry

    if has_selections and has_selection:
        raise ConfigError(
            f"{context} must define either 'selections' (a list of regions) or "
            "'selection' (a single region), not both."
        )

    if has_selection:
        selection = _resolve_path(entry["selection"], base_dir, f"{context}.selection")
        isotopes = _parse_isotopes(entry.get("isotopes", []), f"{context}.isotopes")
        return [
            SelectionEntry(
                selection=selection,
                isotopes=isotopes,
                name=selection.stem,
                label=workspace_label,
            )
        ]

    if not has_selections:
        raise ConfigError(
            f"{context} must define a 'selections' list or a 'selection' file location."
        )

    raw_selections = entry["selections"]
    if not isinstance(raw_selections, list) or not raw_selections:
        raise ConfigError(f"{context}.selections must be a non-empty list.")

    selections = [
        _parse_selection(
            raw, base_dir, f"{context}.selections[{index}]", workspace_label
        )
        for index, raw in enumerate(raw_selections)
    ]
    _check_duplicate_selection_names(selections, context)
    return selections


def _parse_selection(
    raw: Any, base_dir: Path, context: str, workspace_label: Any
) -> SelectionEntry:
    """Validate and convert a single selection region."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{context} must be a mapping.")

    selection_raw = raw.get("selection") or raw.get("file")
    if not selection_raw:
        raise ConfigError(f"{context} must define a 'selection' file location.")
    selection = _resolve_path(selection_raw, base_dir, f"{context}.selection")

    isotopes = _parse_isotopes(raw.get("isotopes", []), f"{context}.isotopes")

    name = raw.get("name") or selection.stem
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{context}.name must be a non-empty string.")

    # An absent label inherits the workspace label, while an explicit null
    # means "no label for this region".
    label = raw.get("label", workspace_label)
    if label is not None and not isinstance(label, (str, int, float)):
        raise ConfigError(f"{context}.label must be a string, number or null.")

    return SelectionEntry(
        selection=selection,
        isotopes=isotopes,
        name=name.strip(),
        label=label,
    )


def _parse_isotopes(value: Any, context: str) -> list[str]:
    """Coerce the isotopes of a region to a list of strings.

    An empty list is valid and means the region contains background only.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{context} must be a list of strings.")
    return list(value)


def _check_duplicate_names(workspaces: list[WorkspaceEntry]) -> None:
    """Ensure every entry has a unique name so outputs do not collide."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in workspaces:
        if entry.name in seen:
            duplicates.append(entry.name)
        seen.add(entry.name)
    if duplicates:
        raise ConfigError(
            "Duplicate workspace name(s) in configuration: "
            + ", ".join(sorted(set(duplicates)))
        )


def _check_duplicate_selection_names(
    selections: list[SelectionEntry], context: str
) -> None:
    """Ensure every region within a workspace has a unique name."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for selection in selections:
        if selection.name in seen:
            duplicates.append(selection.name)
        seen.add(selection.name)
    if duplicates:
        raise ConfigError(
            f"{context} has duplicate selection name(s): "
            + ", ".join(sorted(set(duplicates)))
        )


def _resolve_path(value: Any, base_dir: Path, context: str) -> Path:
    """Resolve a path from the config, relative to the config file's directory."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty path string.")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _as_float(value: Any, context: str) -> float:
    """Coerce a value to a float with a helpful error message."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a number, got {value!r}.")
    return float(value)


def _as_int(value: Any, context: str) -> int:
    """Coerce a value to an int with a helpful error message."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer, got {value!r}.")
    return value


def _as_int_list(value: Any, context: str) -> list[int]:
    """Coerce a value to a list of positive ints with a helpful error message."""
    if value is None:
        return []
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be an integer or list of integers.")
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, list):
        raise ConfigError(f"{context} must be an integer or list of integers.")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{context} must contain only integers, got {item!r}.")
        if item <= 0:
            raise ConfigError(f"{context} must contain only positive integers.")
        result.append(item)
    return result
