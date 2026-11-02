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
   once for every requested ``n_samples`` value. A region may instead be a
   mixture of several other regions, given by a ``mix`` list.
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
from data_loading.load_and_normalise import convert_TOF_to_energy_at_l2_values
from data_loading.workspace_config import (
    ConfigError,
    SelectionEntry,
    TrainingConfig,
    WorkspaceEntry,
    first_not_none,
    load_config,
)
from data_manipulation.generate_test_data import (
    generate_test_data_from_mixed_parts,
)

DEFAULT_L2 = 0.6
DEFAULT_NOISE = 0.1
DEFAULT_N_SETS = 10


@dataclass
class RunSettings:
    """The effective settings for a run, after merging CLI args and config."""

    l2: float
    l2_values: list[float]
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
            "conversion. This applies to every workspace, since a mixed "
            "region may take parts from several of them and they have to "
            "share the same energy axis. Falls back to 'l2' in the config "
            "file, and is ignored when --l2-values is given."
        ),
    )
    parser.add_argument(
        "--l2-values",
        type=float,
        nargs="+",
        default=None,
        help=(
            "One or more sample-detector distances to spread the samples over. "
            "Each selection region then gets samples taken at every distance, "
            "so that the training data covers the variation between "
            "experiments. This applies to every workspace. Falls back to "
            "'l2_values' in the config file."
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

    # The L2 values to spread the samples over. An explicit list of values
    # takes precedence over the single reference distance.
    l2_values = first_not_none(args.l2_values, config.l2_values, None)
    if l2_values:
        l2_values = [float(value) for value in l2_values]
    else:
        l2_values = [float(l2)]

    if noise_level < 0:
        raise ConfigError("--noise must be greater than or equal to zero.")
    if l2 <= 0:
        raise ConfigError("--l2 must be greater than zero.")
    if n_sets <= 0:
        raise ConfigError("--n-sets must be greater than zero.")
    if any(value <= 0 for value in l2_values):
        raise ConfigError("--l2-values must all be greater than zero.")
    if len(set(l2_values)) != len(l2_values):
        raise ConfigError("--l2-values must not contain duplicates.")

    return RunSettings(
        l2=float(l2),
        l2_values=l2_values,
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
    used, so that no two generated files collide. The L2 part is only included
    when a single L2 value is used, since a file with several L2 values has no
    single L2 to name it after.
    """
    l2_part = (
        f"_l2{settings.l2:g}"
        if len(settings.l2_values) <= 1
        else f"_l2x{len(settings.l2_values)}"
    )
    return (
        f"{entry.name}_{selection.name}_n{n_samples}"
        f"_noise{settings.noise_level:g}{l2_part}.npz"
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
    parts: list[tuple], name: str, settings: RunSettings
) -> None:
    """Check that background spectra are available to draw noise from.

    Noise spectra are drawn from the detectors of a workspace that are
    outside the region, so a region that covers every detector of one of its
    workspaces leaves nothing to draw from.

    Args:
        parts: The parts of the region. Each is a tuple of the full
            workspace, its selection workspace and its fraction.
        name: Name of the selection region, used for error messages.
        settings: The effective run settings, used to check the noise level.

    Raises:
        ValueError: If noise is requested but the region leaves no detectors
            outside it.
    """
    if settings.noise_level <= 0:
        return

    # The detectors used by each workspace, so that a workspace is only
    # checked against the parts that come from it.
    used: dict[int, set[int]] = {}
    for full_ws, selection_ws, _ in parts:
        key = id(full_ws)
        used.setdefault(key, set()).update(
            selection_ws.getDetectorIDToWorkspaceIndexMap(False, False).keys()
        )

    for full_ws, _, _ in parts:
        key = id(full_ws)
        n_total = full_ws.getNumberHistograms()
        if len(used[key]) >= n_total:
            raise ValueError(
                f"Selection region '{name}' covers every detector of one of "
                "its workspaces, so no background spectra are available to "
                "draw noise from. Use --noise 0 or select a smaller region."
            )


def load_selection_parts(
    selection: SelectionEntry, workspace_name: str
) -> list[tuple[str, object, float]]:
    """Load the selection workspaces that make up a region.

    A plain region is a single part with a fraction of one. A mixed region
    is made up of one part per entry of its ``mix`` list, and each part may
    come from a different workspace.

    Args:
        selection: The region to load.
        workspace_name: Name of the workspace holding the region.

    Returns:
        A list of parts. Each part is a tuple of the name of the workspace
        it comes from, its selection workspace, and the fraction of the
        region it covers.
    """
    if not selection.is_mixed:
        name = f"{workspace_name}_selection_{selection.name}"
        selection_ws = load_workspace(selection.selection, name)
        return [(workspace_name, selection_ws, 1.0)]

    parts = []
    for index, part in enumerate(selection.mix):
        part_workspace = part.workspace or workspace_name
        name = f"{workspace_name}_selection_{selection.name}_{index}_{part.name}"
        selection_ws = load_workspace(part.selection, name)
        parts.append((part_workspace, selection_ws, part.fraction))
    return parts


def parts_at_l2(
    parts: list[tuple],
    energy_workspaces: dict[str, list],
    l2_index: int,
) -> list[tuple]:
    """Build the parts of a region for one L2 value.

    Args:
        parts: The parts of the region, as returned by
            :func:`load_selection_parts`.
        energy_workspaces: The converted workspaces of every workspace that
            takes part, keyed by workspace name.
        l2_index: The index of the L2 value to use.

    Returns:
        The parts, each a tuple of its full workspace, its selection
        workspace and its fraction.
    """
    return [
        (
            energy_workspaces[workspace_name][l2_index],
            selection_ws,
            fraction,
        )
        for workspace_name, selection_ws, fraction in parts
    ]


def sample_over_l2_values(
    energy_workspaces: dict[str, list],
    l2_values: list[float],
    parts: list[tuple],
    n_samples: int,
    settings: RunSettings,
) -> tuple[list[np.ndarray], list[float]]:
    """Sample spectra from a selection region at each of the L2 values.

    The samples are spread evenly over the L2 values, so that the region ends
    up with the same number of samples at every sample-detector distance. All
    the workspaces share the same fixed energy grid, so the samples can be
    stored together directly, with the L2 variation showing up as a small
    shift of the spectra.

    Args:
        energy_workspaces: The converted workspaces of every workspace that
            takes part, keyed by workspace name. Each holds one workspace per
            L2 value.
        l2_values: The L2 value of each workspace in ``energy_workspaces``.
        parts: The parts of the region. Each is a tuple of the name of the
            workspace it comes from, its selection workspace, and the
            fraction of the region it covers.
        n_samples: The number of spectra summed together to make one sample.
        settings: The effective run settings.

    Returns:
        A tuple of the sampled spectra and the L2 value of each sample.
    """
    n_l2 = len(l2_values)
    n_sets = settings.n_sets

    # Spread the sets evenly over the L2 values, so every L2 gets the same
    # number of samples.
    sets_per_l2 = [n_sets // n_l2] * n_l2
    for i in range(n_sets % n_l2):
        sets_per_l2[i] += 1

    samples: list[np.ndarray] = []
    sample_l2: list[float] = []

    for l2_index, (l2, n_sets_here) in enumerate(zip(l2_values, sets_per_l2)):
        if n_sets_here == 0:
            continue

        l2_samples = generate_test_data_from_mixed_parts(
            parts_at_l2(parts, energy_workspaces, l2_index),
            n_sets_here,
            n_samples,
            settings.noise_level,
        )
        samples.extend(l2_samples)
        sample_l2.extend([l2] * n_sets_here)

    return samples, sample_l2


def load_and_convert_workspace(entry: WorkspaceEntry, settings: RunSettings) -> list:
    """Load a workspace and convert it to energy at each L2 value.

    Every workspace is converted at the same set of L2 values, because the
    sample-detector distance is a global setting.

    Args:
        entry: The workspace entry from the configuration file.
        settings: The effective run settings.

    Returns:
        The converted workspaces, one per L2 value.
    """
    l2_values = settings.l2_values

    print(f"[{entry.name}] loading workspace")
    sample_ws = load_workspace(entry.workspace, f"{entry.name}_sample")

    print(
        f"[{entry.name}] converting TOF to energy "
        f"(l2 values: {', '.join(f'{v:g}' for v in l2_values)})"
    )
    return convert_TOF_to_energy_at_l2_values(
        sample_ws, l2_values, name_prefix=f"{entry.name}_l2"
    )


def generate_for_workspace(
    entry: WorkspaceEntry,
    energy_workspaces: dict[str, list],
    settings: RunSettings,
    overwrite: bool = False,
) -> list[Path]:
    """Generate and save the training data for a single workspace.

    Each selection region of the workspace is sampled in turn. A region may
    mix parts taken from other workspaces, which are sampled from the
    workspaces they belong to.

    Args:
        entry: The workspace entry from the configuration file.
        energy_workspaces: The converted workspaces of every workspace that
            takes part, keyed by workspace name.
        settings: The effective run settings.
        overwrite: Whether to overwrite existing output files.

    Returns:
        The list of files written.
    """
    written: list[Path] = []
    for selection in entry.selections:
        written.extend(
            generate_for_selection(
                entry,
                selection,
                energy_workspaces,
                settings,
                overwrite,
            )
        )
    return written


def generate_for_selection(
    entry: WorkspaceEntry,
    selection: SelectionEntry,
    energy_workspaces: dict[str, list],
    settings: RunSettings,
    overwrite: bool = False,
) -> list[Path]:
    """Generate and save the training data for one selection region.

    The samples are spread over the given L2 values, so that the region has
    data at each of the sample-detector distances seen between experiments.
    All the samples share one common energy axis, so that they can be stored
    together in a single file.

    Args:
        entry: The workspace entry from the configuration file.
        selection: The selection region to sample from.
        energy_workspaces: The converted workspaces of every workspace that
            takes part, keyed by workspace name. Each holds one workspace per
            L2 value.
        settings: The effective run settings.
        overwrite: Whether to overwrite existing output files.

    Returns:
        The list of files written.
    """
    name = f"{entry.name}/{selection.name}"
    parts = load_selection_parts(selection, entry.name)

    n_detectors = sum(ws.getNumberHistograms() for _, ws, _ in parts)
    check_noise_is_possible(parts_at_l2(parts, energy_workspaces, 0), name, settings)
    isotopes = (
        ", ".join(selection.isotopes) if selection.isotopes else "background only"
    )
    if selection.is_mixed:
        described = ", ".join(
            f"{part.name} ({part.workspace}) at {fraction:g}"
            for part, (_, _, fraction) in zip(selection.mix, parts)
        )
        print(
            f"[{name}] mixed region of {n_detectors} detector(s) "
            f"({described}), isotopes: {isotopes}"
        )
    else:
        print(f"[{name}] {n_detectors} detector(s), isotopes: {isotopes}")

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
        samples, sample_l2 = sample_over_l2_values(
            energy_workspaces,
            settings.l2_values,
            parts,
            n_samples,
            settings,
        )

        save_sampled_data_to_np_format(
            samples,
            energy_workspaces[entry.name][0].readX(0),
            output_path,
            metadata={
                "name": entry.name,
                "selection_name": selection.name,
                "isotopes": np.asarray(selection.isotopes, dtype=str),
                "is_background_only": np.asarray(selection.is_background_only),
                "is_mixed": np.asarray(selection.is_mixed),
                "mix_fractions": np.asarray([part[2] for part in parts], dtype=float),
                "mix_names": np.asarray(
                    [part.name for part in selection.mix], dtype=str
                ),
                "mix_workspaces": np.asarray(
                    [part.workspace or entry.name for part in selection.mix],
                    dtype=str,
                ),
                "label": _encode_label(selection.label),
                "label_is_null": np.asarray(selection.label is None),
                "n_samples": np.asarray(n_samples),
                "n_sets": np.asarray(settings.n_sets),
                "noise_level": np.asarray(settings.noise_level),
                "l2_values": np.asarray(settings.l2_values),
                "spectra_l2": np.asarray(sample_l2),
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

    # Every workspace is loaded and converted before any data is generated,
    # because a mixed region may take parts from several workspaces.
    energy_workspaces: dict[str, list] = {}
    failures = 0
    for entry in config.workspaces:
        try:
            converted = load_and_convert_workspace(entry, settings)
        except Exception as exc:
            failures += 1
            print(f"Error loading {entry.name}: {exc}", file=sys.stderr)
            continue
        energy_workspaces[entry.name] = converted

    for entry in config.workspaces:
        if entry.name not in energy_workspaces:
            continue
        try:
            generate_for_workspace(
                entry,
                energy_workspaces,
                settings,
                overwrite=args.overwrite,
            )
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
