"""Build a dictionary of known spectra from a YAML configuration file.

A dictionary holds one entry per known component, each of them the average
measured spectrum of a region containing a single isotope or a background.
The entries can be built from either of two sources:

* **selection workspaces** - the raw ``.nxs`` files holding a cropped region
  of the detector. This is the usual route, and it means a dictionary can be
  built for an element without first generating any training data for it.
* **generated data files** - the ``.npz`` files written by
  ``generate_training_data.py``.

Both routes give the same dictionary, because both use measured data.

Example configuration::

    l2_values: [0.62, 0.63, 0.64, 0.65]

    selections:
      - name: Ag/Ag
        selection: data/Ag_normMonitor_selection.nxs
        isotopes: [Ag-107, Ag-109]
        workspace: Ag
      - name: Au/Au
        selection: data/Au_1foil_normMonitor_selection_new.nxs
        isotopes: [Au-197]
        workspace: Au
      - name: Ag/background
        selection: data/Ag_normMonitor_empty_selection.nxs
        isotopes: []
        workspace: Ag

Usage::

    python -m data_manipulation.build_dictionary \\
        --config dictionary.yaml \\
        --output dictionary.npz
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from data_manipulation.dictionary import (
    Dictionary,
    SelectionSpec,
    build_dictionary_from_files,
    build_dictionary_from_selections,
)

__all__ = [
    "DictionaryConfig",
    "build_configured_dictionary",
    "load_dictionary_config",
    "main",
]


class ConfigError(ValueError):
    """Raised when the dictionary configuration file is invalid."""


@dataclass
class DictionaryConfig:
    """The parsed contents of a dictionary configuration file.

    Attributes:
        selections: The selection regions to build entries from.
        files: The generated data files to build entries from.
        l2_values: The sample-detector distances to convert at.
    """

    selections: list[SelectionSpec] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    l2_values: list[float] = field(default_factory=list)


def load_dictionary_config(path: str | Path) -> DictionaryConfig:
    """Load and validate a dictionary configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: If the file is missing, malformed or invalid.
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
    l2_values = _as_float_list(raw.get("l2_values", []), "l2_values")

    selections = [
        _parse_selection(item, base_dir, index)
        for index, item in enumerate(raw.get("selections") or [])
    ]

    files = []
    for index, item in enumerate(raw.get("files") or []):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"files[{index}] must be a non-empty path string.")
        file_path = Path(item.strip()).expanduser()
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        files.append(file_path.resolve())

    if not selections and not files:
        raise ConfigError(
            "The configuration must define either a 'selections' list or a "
            "'files' list."
        )

    return DictionaryConfig(
        selections=selections,
        files=files,
        l2_values=l2_values,
    )


def _parse_selection(raw: Any, base_dir: Path, index: int) -> SelectionSpec:
    """Validate and convert one selection region of the configuration."""
    context = f"selections[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{context} must be a mapping.")

    selection_raw = raw.get("selection") or raw.get("file")
    if not selection_raw:
        raise ConfigError(f"{context} must define a 'selection' file location.")
    if not isinstance(selection_raw, str) or not selection_raw.strip():
        raise ConfigError(f"{context}.selection must be a non-empty path string.")
    selection_path = Path(selection_raw.strip()).expanduser()
    if not selection_path.is_absolute():
        selection_path = base_dir / selection_path
    selection_path = selection_path.resolve()

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{context}.name must be a non-empty string.")

    isotopes = raw.get("isotopes", [])
    if isotopes is None:
        isotopes = []
    if isinstance(isotopes, str):
        isotopes = [isotopes]
    if not isinstance(isotopes, list) or not all(
        isinstance(item, str) for item in isotopes
    ):
        raise ConfigError(f"{context}.isotopes must be a list of strings.")

    workspace = raw.get("workspace", "")
    if not isinstance(workspace, str):
        raise ConfigError(f"{context}.workspace must be a string.")

    return SelectionSpec(
        name=name.strip(),
        path=selection_path,
        isotopes=list(isotopes),
        workspace=workspace.strip(),
    )


def _as_float_list(value: Any, context: str) -> list[float]:
    """Coerce a value to a list of positive floats."""
    if value is None:
        return []
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be a number or list of numbers.")
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, list):
        raise ConfigError(f"{context} must be a number or list of numbers.")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{context} must contain only numbers, got {item!r}.")
        if item <= 0:
            raise ConfigError(f"{context} must contain only positive numbers.")
        result.append(float(item))
    if len(set(result)) != len(result):
        raise ConfigError(f"{context} must not contain duplicate values.")
    return result


def build_configured_dictionary(config: DictionaryConfig) -> Dictionary:
    """Build a dictionary from a parsed configuration.

    Args:
        config: The parsed configuration.

    Returns:
        The built dictionary.
    """
    entries = []
    x_bins = None

    if config.selections:
        selection_dictionary = build_dictionary_from_selections(
            config.selections, l2_values=config.l2_values or None
        )
        entries.extend(selection_dictionary.entries)
        x_bins = selection_dictionary.x_bins

    if config.files:
        file_dictionary = build_dictionary_from_files(config.files)
        for entry in file_dictionary.entries:
            if any(existing.name == entry.name for existing in entries):
                continue
            entries.append(entry)
        if x_bins is None:
            x_bins = file_dictionary.x_bins

    assert x_bins is not None
    return Dictionary(entries=entries, x_bins=x_bins)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a dictionary from a configuration file.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a dictionary of known spectra from a YAML file listing "
            "selection workspaces or generated data files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path of the dictionary .npz file to write.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_dictionary_config(args.config)
        dictionary = build_configured_dictionary(config)
    except (ConfigError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    dictionary.save(args.output)
    print(f"Wrote {len(dictionary)} entry(ies) to {args.output}")
    for entry in dictionary.entries:
        isotopes = ", ".join(entry.isotopes) if entry.isotopes else "background only"
        print(f"  {entry.name:24s} {isotopes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
