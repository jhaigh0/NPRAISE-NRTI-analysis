#!/usr/bin/env python3
"""Generate training data for the 1D CNN from a YAML list of workspaces.

For every workspace listed in the configuration file this script:

1. Loads the full detector workspace and converts its spectra from TOF to
   energy using
   :func:`data_loading.load_and_normalise.convert_TOF_to_energy`.
2. For each *selection region* of the workspace, loads the selection
   workspace (a cropped region of the detector holding a single isotope, a mix
   of isotopes, or background only) and samples summed spectra from it with
   :func:`data_manipulation.generate_test_data.generate_test_data_from_selection_workspace`,
   once for every requested ``n_samples`` value.
3. Saves the result to a ``.npz`` file ready for training.

Example::

    python generate_training_data.py \\
        --config configs/example_training_data.yaml \\
        --n-sets 10 \\
        --n-samples 10 20 50 100 \\
        --noise 0.1 \\
        --l2 0.6 \\
        --output-dir training_data
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mantid.simpleapi import Load

from data_loading.export import save_sampled_data_to_np_format
from data_loading.load_and_normalise import convert_TOF_to_energy
from data_loading.workspace_config import (
    ConfigError,
    SelectionEntry,
    TrainingConfig,
    WorkspaceEntry,
    first_not_none,
    load_config,
)
from data_manipulation.generate_test_data import (
    generate_test_data_from_selection_workspace,
)

DEFAULT_L2 = 0.6
DEFAULT_NOISE = 0.1
DEFAULT_N_SETS = 10


@dataclass
class RunSettings:
    """The effective settings for a run, after merging CLI args and config."""

    l2: float
    noise_level: float
    n_sets: int
    n_samples: list[int]
    output_dir: Path
    seed: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate 1D CNN training data from a YAML file listing Mantid "
            "workspaces and their selection workspaces."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file listing the workspaces.",
    )
    parser.add_argument(
        "--n-sets",
        type=int,
        default=None,
        help=(
            "Number of summed spectra generated for each n_samples value. "
            "Falls back to the 'n_sets' value in the config file."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more values for the number of spectra summed per sample. "
            "Looped over, so samples with different signal-to-noise ratios are "
            "produced. Falls back to 'n_samples' in the config file."
        ),
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help=(
            "Noise level: the fraction of each sample drawn from outside the "
            "selection region. Falls back to 'noise_level' in the config file."
        ),
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=None,
        help=(
            "Sample-detector distance in metres used for the TOF to energy "
            "conversion. Falls back to 'l2' in the config file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data"),
        help="Directory the generated .npz files are written to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator, for reproducible data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace, config: TrainingConfig) -> RunSettings:
    """Merge the command line arguments with the configuration file.

    Command line arguments take precedence over the config file, which in
    turn takes precedence over the built in defaults.
    """
    n_samples = args.n_samples if args.n_samples is not None else config.n_samples
    if not n_samples:
        raise ConfigError(
            "No 'n_samples' values given. Pass --n-samples on the command line "
            "or set 'n_samples' in the configuration file."
        )

    noise_level = first_not_none(args.noise, config.noise_level, DEFAULT_NOISE)
    l2 = first_not_none(args.l2, config.l2, DEFAULT_L2)
    n_sets = first_not_none(args.n_sets, config.n_sets, DEFAULT_N_SETS)

    if noise_level < 0:
        raise ConfigError("--noise must be greater than or equal to zero.")
    if l2 <= 0:
        raise ConfigError("--l2 must be greater than zero.")
    if n_sets <= 0:
        raise ConfigError("--n-sets must be greater than zero.")

    return RunSettings(
        l2=float(l2),
        noise_level=float(noise_level),
        n_sets=int(n_sets),
        n_samples=[int(value) for value in n_samples],
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


def load_workspace(path: Path, name: str):
    """Load a workspace from a file, without adding it to the ADS.

    Args:
        path: Location of the workspace file (e.g. a Mantid ``.nxs`` file).
        name: Name to give the workspace. Used for the output file name.

    Returns:
        The loaded workspace.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be loaded as a workspace.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Workspace file not found: {path}")

    try:
        return Load(str(path), OutputWorkspace=name, StoreInADS=False)
    except Exception as exc:  # pragma: no cover - depends on Mantid internals
        raise RuntimeError(f"Could not load workspace from {path}: {exc}") from exc


def build_output_filename(
    entry: WorkspaceEntry,
    selection: SelectionEntry,
    n_samples: int,
    settings: RunSettings,
) -> str:
    """Build the output file name for one selection region and n_samples value.

    The name combines the workspace, the selection region and the settings
    used, so that no two generated files collide.
    """
    return (
        f"{entry.name}_{selection.name}_n{n_samples}"
        f"_noise{settings.noise_level:g}_l2{settings.l2:g}.npz"
    )


def _encode_label(label: str | float | None) -> np.ndarray:
    """Encode a label so that it can be stored in a ``.npz`` file.

    Numeric labels keep their numeric type so that they can be used directly
    as class indices. A missing label is stored as an empty string rather than
    ``None``, because ``np.asarray(None)`` produces an object array that cannot
    be read back without ``allow_pickle=True``. Use the ``label_is_null``
    metadata to tell the two apart.
    """
    if label is None:
        return np.asarray("")
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return np.asarray(label)
    return np.asarray(str(label))


def check_noise_is_possible(
    energy_ws, selection_ws, name: str, settings: RunSettings
) -> None:
    """Check that background spectra are available to draw noise from.

    Noise spectra are drawn from the detectors *outside* the selection region,
    so a selection that covers every detector of the full workspace leaves
    nothing to draw from.

    Args:
        energy_ws: The full workspace, converted to energy.
        selection_ws: The selection workspace.
        name: Name of the selection region, used for error messages.
        settings: The effective run settings, used to check the noise level.

    Raises:
        ValueError: If noise is requested but the selection leaves no
            detectors outside it.
    """
    if settings.noise_level <= 0:
        return

    n_selected = selection_ws.getNumberHistograms()
    n_total = energy_ws.getNumberHistograms()
    if n_selected >= n_total:
        raise ValueError(
            f"Selection workspace '{name}' covers every detector in the "
            "full workspace, so no background spectra are available to draw "
            "noise from. Use --noise 0 or select a smaller region."
        )


def generate_for_workspace(
    entry: WorkspaceEntry,
    settings: RunSettings,
    overwrite: bool = False,
) -> list[Path]:
    """Generate and save the training data for a single workspace.

    The full workspace is loaded and converted to energy once, then each
    selection region of the workspace is sampled in turn.

    Args:
        entry: The workspace entry from the configuration file.
        settings: The effective run settings.
        overwrite: Whether to overwrite existing output files.

    Returns:
        The list of files written.
    """
    l2 = entry.l2 if entry.l2 is not None else settings.l2

    print(f"[{entry.name}] loading workspace")
    sample_ws = load_workspace(entry.workspace, f"{entry.name}_sample")

    print(f"[{entry.name}] converting TOF to energy (l2={l2})")
    energy_ws = convert_TOF_to_energy(sample_ws, l2=l2)

    written: list[Path] = []
    for selection in entry.selections:
        written.extend(
            generate_for_selection(entry, selection, energy_ws, settings, overwrite)
        )
    return written


def generate_for_selection(
    entry: WorkspaceEntry,
    selection: SelectionEntry,
    energy_ws,
    settings: RunSettings,
    overwrite: bool = False,
) -> list[Path]:
    """Generate and save the training data for one selection region.

    Args:
        entry: The workspace entry from the configuration file.
        selection: The selection region to sample from.
        energy_ws: The full workspace, converted to energy.
        settings: The effective run settings.
        overwrite: Whether to overwrite existing output files.

    Returns:
        The list of files written.
    """
    name = f"{entry.name}/{selection.name}"
    selection_ws = load_workspace(selection.selection, f"{entry.name}_selection")

    check_noise_is_possible(energy_ws, selection_ws, name, settings)
    isotopes = (
        ", ".join(selection.isotopes) if selection.isotopes else "background only"
    )
    print(
        f"[{name}] {selection_ws.getNumberHistograms()} detector(s), "
        f"isotopes: {isotopes}"
    )

    written: list[Path] = []
    for n_samples in settings.n_samples:
        output_path = settings.output_dir / build_output_filename(
            entry, selection, n_samples, settings
        )
        if output_path.exists() and not overwrite:
            print(f"[{name}] skipping existing {output_path.name}")
            continue

        print(
            f"[{name}] sampling n_samples={n_samples}, "
            f"n_sets={settings.n_sets}, noise={settings.noise_level}"
        )
        samples = generate_test_data_from_selection_workspace(
            energy_ws,
            selection_ws,
            settings.n_sets,
            n_samples,
            settings.noise_level,
        )

        save_sampled_data_to_np_format(
            samples,
            energy_ws.readX(0),
            output_path,
            metadata={
                "name": entry.name,
                "selection_name": selection.name,
                "isotopes": np.asarray(selection.isotopes, dtype=str),
                "is_background_only": np.asarray(selection.is_background_only),
                "label": _encode_label(selection.label),
                "label_is_null": np.asarray(selection.label is None),
                "n_samples": np.asarray(n_samples),
                "n_sets": np.asarray(settings.n_sets),
                "noise_level": np.asarray(settings.noise_level),
                "l2": np.asarray(entry.l2 if entry.l2 is not None else settings.l2),
            },
        )
        print(f"[{name}] wrote {output_path}")
        written.append(output_path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training data generation.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    args = parse_args(argv)

    try:
        config = load_config(args.config)
        settings = build_settings(args, config)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if settings.seed is not None:
        np.random.seed(settings.seed)
        print(f"Using random seed {settings.seed}")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating training data for {len(config.workspaces)} workspace(s) "
        f"into {settings.output_dir}"
    )

    failures = 0
    for entry in config.workspaces:
        try:
            generate_for_workspace(entry, settings, overwrite=args.overwrite)
        except Exception as exc:
            failures += 1
            print(f"Error processing {entry.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"Finished with {failures} failure(s).", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
