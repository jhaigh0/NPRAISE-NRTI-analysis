"""Fit the abundances of known components in a measured spectrum.

Given a :class:`~data_manipulation.dictionary.Dictionary` of known spectra, a
measured spectrum is explained as a non-negative combination of those entries.
The coefficients are the abundances of each known component, and the part of
the spectrum the dictionary cannot explain is the residual.

Because the mixture of two isotopes is exactly linear in their spectra, the
non-negative least squares solution recovers the mixture fractions directly,
rather than having to be learned.

NMF is only used for the optional unknown component: when the residual is
larger than a threshold, the part of the spectrum the dictionary does not
explain is decomposed with one free non-negative component.

Example::

    dictionary = Dictionary.load("dictionary.npz")
    result = fit_spectrum(spectrum, dictionary, detect_unknown=True)
    print(result.dominant, result.fractions)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from data_manipulation.dictionary import Dictionary

__all__ = [
    "FitResult",
    "build_dictionary_from_files",
    "fit_spectra",
    "fit_spectrum",
    "main",
]

# A spectrum whose residual is a larger fraction of its counts than this is
# treated as containing something the dictionary does not describe.
DEFAULT_UNKNOWN_THRESHOLD = 0.003


@dataclass
class FitResult:
    """The result of fitting one spectrum against a dictionary.

    Attributes:
        abundances: The non-negative coefficient of each dictionary entry.
            Since the columns of the dictionary are normalised to unit sum,
            this is the number of counts that each entry explains. The
            coefficients together account for nearly all the counts of the
            spectrum.
        fractions: The composition of the region. A region that is a
            fraction ``f`` of one entry and ``1 - f`` of another has a
            spectrum of ``f * p_a + (1 - f) * p_b``, so the coefficient of an
            entry divided by the total counts of that entry gives its
            fraction of the region. The fractions together sum to one.
        residual: The part of the spectrum the dictionary cannot explain.
        residual_norm: The Euclidean norm of the residual.
        residual_fraction: The residual norm as a fraction of the total
            counts of the spectrum.
        dominant: The name of the entry that makes up the largest fraction
            of the region.
        unknown_detected: Whether the residual was large enough to look for
            an unknown component.
        unknown_component: The unexplained component, if one was found.
    """

    abundances: np.ndarray
    fractions: np.ndarray
    residual: np.ndarray
    residual_norm: float
    residual_fraction: float
    dominant: str
    unknown_detected: bool = False
    unknown_component: np.ndarray | None = None


def fit_spectrum(
    spectrum: np.ndarray,
    dictionary: Dictionary,
    *,
    detect_unknown: bool = False,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    n_unknown_components: int = 1,
) -> FitResult:
    """Fit one spectrum against a dictionary.

    Args:
        spectrum: The measured spectrum, one value per bin.
        dictionary: The known components to fit against.
        detect_unknown: Whether to look for an unknown component in the
            residual.
        unknown_threshold: The residual fraction above which an unknown
            component is looked for.
        n_unknown_components: How many unknown components to look for.

    Returns:
        The result of the fit.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    matrix = dictionary.matrix

    if spectrum.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"The spectrum has {spectrum.shape[0]} bins but the dictionary "
            f"has {matrix.shape[0]}."
        )

    abundances, _ = nnls(matrix, spectrum)
    residual = spectrum - matrix @ abundances

    total = spectrum.sum()
    residual_norm = float(np.linalg.norm(residual))
    residual_fraction = residual_norm / total if total > 0 else 0.0

    # The columns of the matrix are normalised to unit sum, so the
    # coefficient of an entry is the number of counts that entry explains.
    # A region that is a fraction f of an entry has a spectrum of
    # f * p_entry, so dividing the coefficient by the total counts of the
    # entry gives the fraction of the region that the entry covers.
    weights = abundances / dictionary.column_totals
    weights_total = weights.sum()
    fractions = weights / weights_total if weights_total > 0 else np.zeros_like(weights)

    dominant = dictionary.names[int(np.argmax(weights))]

    unknown_detected = residual_fraction > unknown_threshold
    unknown_component = None
    if detect_unknown and unknown_detected:
        unknown_component = _extract_unknown_component(residual, n_unknown_components)

    return FitResult(
        abundances=abundances,
        fractions=fractions,
        residual=residual,
        residual_norm=residual_norm,
        residual_fraction=residual_fraction,
        dominant=dominant,
        unknown_detected=unknown_detected,
        unknown_component=unknown_component,
    )


def fit_spectra(
    spectra: np.ndarray,
    dictionary: Dictionary,
    *,
    detect_unknown: bool = False,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    n_unknown_components: int = 1,
) -> list[FitResult]:
    """Fit a stack of spectra against a dictionary.

    Args:
        spectra: The measured spectra, one row per spectrum.
        dictionary: The known components to fit against.
        detect_unknown: Whether to look for an unknown component in each
            residual.
        unknown_threshold: The residual fraction above which an unknown
            component is looked for.
        n_unknown_components: How many unknown components to look for.

    Returns:
        One result per spectrum.
    """
    spectra = np.asarray(spectra, dtype=float)
    return [
        fit_spectrum(
            spectrum,
            dictionary,
            detect_unknown=detect_unknown,
            unknown_threshold=unknown_threshold,
            n_unknown_components=n_unknown_components,
        )
        for spectrum in spectra
    ]


def _extract_unknown_component(
    residual: np.ndarray, n_components: int
) -> np.ndarray | None:
    """Extract the components that the dictionary does not explain.

    The residual is clipped at zero, since a negative residual only reflects
    the non-negativity constraint of the fit rather than a missing
    component.
    """
    from sklearn.decomposition import NMF

    positive = np.maximum(residual, 0.0)
    if positive.sum() <= 0:
        return None

    n_components = max(1, min(n_components, positive.shape[0]))
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=0,
        max_iter=1000,
    )
    model.fit_transform(positive.reshape(1, -1))
    components = model.components_[0]
    if components.max() <= 0:
        return None
    return components / components.sum()


def build_dictionary_from_files(paths: list[str | Path]) -> Dictionary:
    """Build a dictionary from generated data files.

    Args:
        paths: The generated ``.npz`` files to build the entries from.

    Returns:
        The built dictionary.
    """
    from data_manipulation.dictionary import build_dictionary_from_files as build

    return build(paths)


def _load_spectra(path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Load the spectra and the L2 values of a generated data file."""
    with np.load(path, allow_pickle=True) as data:
        spectra = data["spectra_samples"]
        x_bins = data["x_bins"]
        if "spectra_l2" in data:
            spectra_l2 = data["spectra_l2"]
        else:
            spectra_l2 = np.zeros(len(spectra))
    return spectra, x_bins, [float(value) for value in spectra_l2]


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the contents of generated data files against a dictionary.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit the abundances of known components in generated spectra "
            "against a dictionary of known spectra."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        type=Path,
        help="Path to the dictionary .npz file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        nargs="+",
        help="The generated data files to fit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path of the CSV file to write the results to.",
    )
    parser.add_argument(
        "--detect-unknown",
        action="store_true",
        help="Look for an unknown component when the residual is large.",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
        help="Residual fraction above which an unknown component is looked for.",
    )
    args = parser.parse_args(argv)

    try:
        dictionary = Dictionary.load(args.dictionary)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Error: could not load dictionary: {exc}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for path in args.input:
        try:
            spectra, x_bins, spectra_l2 = _load_spectra(path)
        except (OSError, KeyError, ValueError) as exc:
            print(f"Error: could not read {path}: {exc}", file=sys.stderr)
            return 1

        if not np.array_equal(x_bins, dictionary.x_bins):
            print(
                f"Error: {path} has different bin edges to the dictionary.",
                file=sys.stderr,
            )
            return 1

        results = fit_spectra(
            spectra,
            dictionary,
            detect_unknown=args.detect_unknown,
            unknown_threshold=args.unknown_threshold,
        )
        for index, result in enumerate(results):
            row: dict = {
                "file": path.name,
                "sample": index,
                "l2": spectra_l2[index],
                "dominant": result.dominant,
                "residual_fraction": result.residual_fraction,
                "unknown_detected": result.unknown_detected,
            }
            for name, fraction, abundance in zip(
                dictionary.names, result.fractions, result.abundances
            ):
                row[f"fraction_{name}"] = fraction
                row[f"counts_{name}"] = abundance
            rows.append(row)

    if not rows:
        print("Error: no spectra to fit.", file=sys.stderr)
        return 1

    fieldnames = list(rows[0])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} result(s) to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
