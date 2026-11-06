#!/usr/bin/env python3
"""Plot the reference spectra held in a dictionary, for visual inspection.

The dictionary holds one entry per known component. This script draws
them so that the shape of each one can be inspected, in particular the
Bragg peaks that distinguish one element from another.

Several views are available, because the interesting features are not
always visible in a single plot:

* ``overlay`` - every entry on one pair of axes, so that they can be
  compared directly.
* ``grid`` - one panel per entry, so that each one can be seen on its
  own scale.
* ``peaks`` - the same as ``grid``, but with a smooth baseline removed
  and the axis zoomed to the interior of the energy range, which is where
  the Bragg peaks live. This is the view that shows whether the entries
  can actually be told apart.
* ``difference`` - every entry minus the first one (or a chosen
  reference), which makes the small differences between two elements
  obvious.

Example::

    python plot_dictionary.py dictionary_all_elements.npz \\
        --view grid --output dictionary_grid.png

    python plot_dictionary.py dictionary_all_elements.npz \\
        --view peaks --x-min 1000 --x-max 30000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from data_manipulation.dictionary import Dictionary

__all__ = ["main", "plot_dictionary"]

# A moving-average window, in bins, used to estimate the smooth baseline.
BASELINE_WINDOW = 51


def _baseline_removed(profile: np.ndarray, window: int = BASELINE_WINDOW) -> np.ndarray:
    """Subtract a moving average to leave the peak structure.

    Args:
        profile: The spectrum to process, one value per bin.
        window: The width of the moving average, in bins.

    Returns:
        The spectrum with the smooth baseline removed.
    """
    if profile.size < window:
        return profile - profile.mean()
    kernel = np.ones(window) / window
    smooth = np.convolve(profile, kernel, mode="same")
    return profile - smooth


def _bin_centres(x_bins: np.ndarray) -> np.ndarray:
    """The centre of each bin, for plotting against the energy."""
    return 0.5 * (x_bins[:-1] + x_bins[1:])


def _normalise(profile: np.ndarray, mode: str) -> np.ndarray:
    """Rescale a profile so that entries can be compared.

    The entries are measured on very different count scales, so plotting
    them raw hides the smaller ones completely. Normalising to the peak
    of the smoothed baseline makes the shape of each one visible.

    Args:
        profile: The spectrum to rescale.
        mode: How to rescale it. ``none`` leaves it alone, ``sum``
            divides by the total counts, and ``baseline`` divides by the
            mean of the smoothed profile, which keeps the absorption dips
            and Bragg peaks at their true relative depth.

    Returns:
        The rescaled spectrum.
    """
    if mode == "none":
        return profile
    if mode == "sum":
        total = profile.sum()
        return profile / total if total else profile
    if mode == "baseline":
        window = BASELINE_WINDOW
        if profile.size < window:
            return profile / profile.mean() if profile.mean() else profile
        kernel = np.ones(window) / window
        smooth = np.convolve(profile, kernel, mode="same")
        scale = np.median(smooth)
        return profile / scale if scale else profile
    raise ValueError(
        f"Unknown normalisation '{mode}'. Choose one of: none, sum, baseline."
    )


def _select_entries(
    dictionary: Dictionary, names: Sequence[str], exclude_background: bool = False
) -> list[tuple[int, str]]:
    """The indices and names of the entries to plot.

    Args:
        dictionary: The dictionary to plot.
        names: The names of the entries wanted. An empty sequence means
            every entry.
        exclude_background: Whether to leave out the entries that hold no
            isotope.

    Returns:
        A list of (index, name) pairs.

    Raises:
        ValueError: If one of the requested names is not in the
            dictionary, or if nothing is left to plot.
    """
    if names:
        index_of = {name: index for index, name in enumerate(dictionary.names)}
        missing = [name for name in names if name not in index_of]
        if missing:
            raise ValueError(
                f"These entries are not in the dictionary: {', '.join(missing)}. "
                f"Available: {', '.join(dictionary.names)}"
            )
        selected = [(index_of[name], name) for name in names]
    else:
        selected = list(enumerate(dictionary.names))

    if exclude_background:
        selected = [
            (index, name)
            for index, name in selected
            if not dictionary.entries[index].is_background
        ]

    if not selected:
        raise ValueError("There are no entries left to plot.")

    return selected


def _apply_range(
    x: np.ndarray, y: np.ndarray, x_min: float | None, x_max: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the data to an energy range."""
    mask = np.ones(x.shape, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    return x[mask], y[mask]


def _save(fig: plt.Figure, output: Path | None) -> None:
    """Save the figure, or show it if no output file was given."""
    if output is None:
        plt.show()
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def _plot_overlay(
    dictionary: Dictionary,
    selected: list[tuple[int, str]],
    x_min: float | None,
    x_max: float | None,
    output: Path | None,
    normalise: str = "none",
) -> None:
    """Every entry on one pair of axes."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = _bin_centres(dictionary.x_bins)
    for index, name in selected:
        profile = _normalise(dictionary.entries[index].profile, normalise)
        px, py = _apply_range(x, profile, x_min, x_max)
        ax.plot(px, py, linewidth=0.8, alpha=0.85, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Energy (meV)")
    ax.set_ylabel("Counts per bin" if normalise == "none" else "Normalised counts")
    ax.set_title("Dictionary entries")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def _plot_grid(
    dictionary: Dictionary,
    selected: list[tuple[int, str]],
    x_min: float | None,
    x_max: float | None,
    output: Path | None,
    normalise: str = "none",
) -> None:
    """One panel per entry."""
    n = len(selected)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    x = _bin_centres(dictionary.x_bins)
    for ax, (index, name) in zip(axes, selected):
        entry = dictionary.entries[index]
        profile = _normalise(entry.profile, normalise)
        px, py = _apply_range(x, profile, x_min, x_max)
        ax.plot(px, py, linewidth=0.8, color="C0")
        ax.set_xscale("log")
        ax.set_title(
            f"{name}\n{', '.join(entry.isotopes) if entry.isotopes else 'background only'}",
            fontsize=9,
        )
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("Counts per bin" if normalise == "none" else "Normalised counts")
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes[len(selected) :]:
        ax.set_visible(False)

    fig.suptitle("Dictionary entries", fontsize=13)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def _plot_peaks(
    dictionary: Dictionary,
    selected: list[tuple[int, str]],
    x_min: float | None,
    x_max: float | None,
    output: Path | None,
    normalise: str = "none",
) -> None:
    """One panel per entry, with the baseline removed.

    This is the view that shows whether the entries can be told apart.
    The vertical axis is scaled to each entry's own strongest peak, so
    that weak structure is still visible.
    """
    n = len(selected)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    x = _bin_centres(dictionary.x_bins)
    for ax, (index, name) in zip(axes, selected):
        entry = dictionary.entries[index]
        peaks = _baseline_removed(_normalise(entry.profile, normalise))
        px, py = _apply_range(x, peaks, x_min, x_max)
        ax.plot(px, py, linewidth=0.8, color="C3")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xscale("log")
        ax.set_title(f"{name} (baseline removed)", fontsize=9)
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("Counts above baseline")
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes[len(selected) :]:
        ax.set_visible(False)

    fig.suptitle("Dictionary entries, baseline removed", fontsize=13)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def _plot_difference(
    dictionary: Dictionary,
    selected: list[tuple[int, str]],
    x_min: float | None,
    x_max: float | None,
    output: Path | None,
    normalise: str = "none",
) -> None:
    """Every entry minus a reference entry.

    The reference is the first entry, or the one named by ``--reference``.
    """
    if not selected:
        raise ValueError("There are no entries to plot.")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = _bin_centres(dictionary.x_bins)
    reference_index, reference_name = selected[0]
    reference = _normalise(dictionary.entries[reference_index].profile, normalise)

    for index, name in selected:
        profile = _normalise(dictionary.entries[index].profile, normalise)
        px, py = _apply_range(x, profile - reference, x_min, x_max)
        ax.plot(px, py, linewidth=0.8, alpha=0.85, label=name)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Energy (meV)")
    ax.set_ylabel(f"Counts per bin minus {reference_name}")
    ax.set_title("Difference from the reference entry")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def plot_dictionary(
    path: str | Path,
    *,
    view: str = "grid",
    names: Sequence[str] = (),
    x_min: float | None = None,
    x_max: float | None = None,
    output: Path | None = None,
    normalise: str = "none",
    exclude_background: bool = False,
) -> None:
    """Plot the reference spectra of a dictionary.

    Args:
        path: The dictionary ``.npz`` file to plot.
        view: Which view to draw. One of ``overlay``, ``grid``, ``peaks``
            or ``difference``.
        names: The names of the entries to plot. Every entry is plotted
            when this is empty.
        x_min: The lowest energy to show, in meV.
        x_max: The highest energy to show, in meV.
        output: Where to write the figure. The figure is shown instead
            when this is None.
        normalise: How to rescale the entries so that they can be
            compared. One of ``none``, ``sum`` or ``baseline``.
        exclude_background: Whether to leave out the entries that hold
            no isotope.

    Raises:
        ValueError: If the dictionary cannot be loaded, if the view is
            not one of the known ones, if a requested entry is not in
            the dictionary, or if the normalisation is not one of the
            known ones.
    """
    dictionary = Dictionary.load(path)
    if not dictionary.entries:
        raise ValueError(f"The dictionary '{path}' has no entries.")

    if normalise not in {"none", "sum", "baseline"}:
        raise ValueError(
            f"Unknown normalisation '{normalise}'. Choose one of: none, sum, baseline."
        )

    selected = _select_entries(dictionary, names, exclude_background)

    if view == "overlay":
        _plot_overlay(dictionary, selected, x_min, x_max, output, normalise)
    elif view == "grid":
        _plot_grid(dictionary, selected, x_min, x_max, output, normalise)
    elif view == "peaks":
        _plot_peaks(dictionary, selected, x_min, x_max, output, normalise)
    elif view == "difference":
        _plot_difference(dictionary, selected, x_min, x_max, output, normalise)
    else:
        raise ValueError(
            f"Unknown view '{view}'. Choose one of: overlay, grid, peaks, difference."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Plot the reference spectra of a dictionary.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot the reference spectra held in a dictionary, so that they "
            "can be inspected visually."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dictionary",
        type=Path,
        help="Path to the dictionary .npz file.",
    )
    parser.add_argument(
        "--view",
        default="grid",
        choices=["overlay", "grid", "peaks", "difference"],
        help="Which view to draw.",
    )
    parser.add_argument(
        "--entries",
        nargs="*",
        default=[],
        metavar="NAME",
        help="The entries to plot. Every entry is plotted when omitted.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="The lowest energy to show, in meV.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="The highest energy to show, in meV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path of the image file to write. The figure is shown instead when omitted."
        ),
    )
    parser.add_argument(
        "--normalise",
        default="none",
        choices=["none", "sum", "baseline"],
        help=(
            "How to rescale the entries so that they can be compared. "
            "'baseline' divides by the smoothed profile, which keeps the "
            "absorption dips and Bragg peaks at their true depth."
        ),
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Leave out the entries that hold no isotope.",
    )
    args = parser.parse_args(argv)

    try:
        plot_dictionary(
            args.dictionary,
            view=args.view,
            names=args.entries,
            x_min=args.x_min,
            x_max=args.x_max,
            output=args.output,
            normalise=args.normalise,
            exclude_background=args.exclude_background,
        )
    except (OSError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
