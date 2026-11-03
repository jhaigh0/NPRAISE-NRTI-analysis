"""Tests for the dictionary and the abundance fitting.

The tests use synthetic spectra, so that the expected answer is known
exactly and the tests do not depend on any large data files.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_manipulation.dictionary import (
    Dictionary,
    DictionaryEntry,
    SelectionSpec,
    build_dictionary_from_files,
    build_dictionary_from_selections,
)
from data_manipulation.fit_abundances import fit_spectra, fit_spectrum

N_BINS = 64


def make_profile(peaks: dict[int, float], total: float = 1000.0) -> np.ndarray:
    """Build a spectrum with the given peaks, scaled to the given total."""
    profile = np.full(N_BINS, 1.0)
    for bin_index, height in peaks.items():
        profile[bin_index] += height
    return profile * (total / profile.sum())


def make_dictionary() -> Dictionary:
    """A dictionary with two known entries and one background entry."""
    entries = [
        DictionaryEntry(
            name="gold",
            profile=make_profile({10: 500.0, 30: 200.0}, total=1000.0),
            isotopes=["Au-197"],
        ),
        DictionaryEntry(
            name="silver",
            profile=make_profile({10: 100.0, 40: 500.0}, total=1000.0),
            isotopes=["Ag-107", "Ag-109"],
        ),
        DictionaryEntry(
            name="background",
            profile=make_profile({}, total=500.0),
            is_background=True,
        ),
    ]
    x_bins = np.linspace(1000.0, 500000.0, N_BINS + 1)
    return Dictionary(entries=entries, x_bins=x_bins)


class DictionaryTests(unittest.TestCase):
    """Tests for building and storing the dictionary."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def _write_data_file(
        self,
        name: str,
        profile: np.ndarray,
        *,
        isotopes: list[str],
        is_background: bool,
        workspace: str = "sample",
        selection: str = "region",
    ) -> Path:
        """Write a file in the format the generation script produces."""
        path = self.tmp_path / name
        np.savez_compressed(
            path,
            spectra_samples=np.tile(profile, (4, 1)),
            x_bins=np.linspace(1000.0, 500000.0, N_BINS + 1),
            name=np.asarray(workspace),
            selection_name=np.asarray(selection),
            isotopes=np.asarray(isotopes, dtype=str),
            is_background_only=np.asarray(is_background),
        )
        return path

    def test_matrix_columns_are_normalised(self):
        dictionary = make_dictionary()
        matrix = dictionary.matrix
        self.assertEqual(matrix.shape, (N_BINS, 3))
        np.testing.assert_allclose(matrix.sum(axis=0), np.ones(3))

    def test_matrix_rejects_an_empty_entry(self):
        dictionary = make_dictionary()
        dictionary.entries[0].profile = np.zeros(N_BINS)
        with self.assertRaises(ValueError):
            _ = dictionary.matrix

    def test_build_from_files_names_entries_by_workspace_and_selection(self):
        gold = make_profile({10: 500.0}, total=1000.0)
        silver = make_profile({40: 500.0}, total=1000.0)
        paths = [
            self._write_data_file(
                "gold.npz", gold, isotopes=["Au-197"], is_background=False
            ),
            self._write_data_file(
                "silver.npz",
                silver,
                isotopes=["Ag-107"],
                is_background=False,
                selection="other",
            ),
        ]
        dictionary = build_dictionary_from_files(paths)
        self.assertEqual(dictionary.names, ["sample/region", "sample/other"])

    def test_build_from_files_merges_files_of_the_same_entry(self):
        gold = make_profile({10: 500.0}, total=1000.0)
        paths = [
            self._write_data_file(
                "a.npz", gold, isotopes=["Au-197"], is_background=False
            ),
            self._write_data_file(
                "b.npz", gold, isotopes=["Au-197"], is_background=False
            ),
        ]
        dictionary = build_dictionary_from_files(paths)
        self.assertEqual(len(dictionary), 1)
        self.assertEqual(dictionary.entries[0].n_samples, 8)

    def test_build_from_files_rejects_mismatched_bin_edges(self):
        first = self._write_data_file(
            "a.npz", make_profile({}), isotopes=["Au-197"], is_background=False
        )
        second = self._write_data_file(
            "b.npz", make_profile({}), isotopes=["Ag-107"], is_background=False
        )
        with np.load(second, allow_pickle=True) as data:
            contents = {key: data[key] for key in data.files}
        contents["x_bins"] = contents["x_bins"] * 2.0
        np.savez_compressed(second, **contents)

        with self.assertRaises(ValueError):
            build_dictionary_from_files([first, second])

    def test_build_from_files_requires_at_least_one_file(self):
        with self.assertRaises(ValueError):
            build_dictionary_from_files([])

    def test_build_from_selections_requires_at_least_one_selection(self):
        with self.assertRaises(ValueError):
            build_dictionary_from_selections([])

    def test_build_from_selections_rejects_a_missing_file(self):
        with self.assertRaises(ValueError):
            build_dictionary_from_selections(
                [
                    SelectionSpec(
                        name="gold",
                        path=self.tmp_path / "does_not_exist.nxs",
                        isotopes=["Au-197"],
                    )
                ]
            )

    def test_build_from_selections_rejects_a_non_positive_l2(self):
        spec = SelectionSpec(
            name="gold",
            path=self.tmp_path / "missing.nxs",
            isotopes=["Au-197"],
        )
        with self.assertRaises(ValueError):
            build_dictionary_from_selections([spec], l2_values=[0.0])

    def test_save_and_load_round_trip(self):
        dictionary = make_dictionary()
        path = dictionary.save(self.tmp_path / "dictionary.npz")
        loaded = Dictionary.load(path)

        self.assertEqual(loaded.names, dictionary.names)
        np.testing.assert_allclose(loaded.x_bins, dictionary.x_bins)
        np.testing.assert_allclose(loaded.matrix, dictionary.matrix)
        self.assertEqual(loaded.entries[1].isotopes, ["Ag-107", "Ag-109"])
        self.assertTrue(loaded.entries[2].is_background)


class FitTests(unittest.TestCase):
    """Tests for fitting a spectrum against a dictionary."""

    def setUp(self):
        self.dictionary = make_dictionary()
        self.gold = self.dictionary.entries[0].profile
        self.silver = self.dictionary.entries[1].profile
        self.background = self.dictionary.entries[2].profile

    def test_pure_spectrum_recovers_a_full_abundance(self):
        result = fit_spectrum(self.gold, self.dictionary)
        self.assertEqual(result.dominant, "gold")
        self.assertAlmostEqual(result.fractions[0], 1.0, places=6)
        self.assertAlmostEqual(result.residual_fraction, 0.0, places=10)

    def test_mixture_fractions_are_recovered(self):
        for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            with self.subTest(fraction=fraction):
                mixture = fraction * self.gold + (1.0 - fraction) * self.silver
                result = fit_spectrum(mixture, self.dictionary)
                self.assertAlmostEqual(result.fractions[0], fraction, places=6)
                self.assertAlmostEqual(result.fractions[1], 1.0 - fraction, places=6)

    def test_mixture_with_background_is_recovered(self):
        mixture = 0.6 * self.gold + 0.4 * self.silver
        result = fit_spectrum(mixture, self.dictionary)
        self.assertAlmostEqual(result.fractions[0], 0.6, places=6)
        self.assertAlmostEqual(result.fractions[1], 0.4, places=6)

    def test_abundances_account_for_the_counts(self):
        result = fit_spectrum(self.gold, self.dictionary)
        self.assertAlmostEqual(result.abundances[0], self.gold.sum(), places=4)

    def test_an_unknown_component_is_detected(self):
        # A spectrum with a peak that no dictionary entry explains.
        unknown = np.full(N_BINS, 1.0)
        unknown[55] += 800.0
        spectrum = 0.5 * self.gold + unknown

        result = fit_spectrum(
            spectrum, self.dictionary, detect_unknown=True, unknown_threshold=0.01
        )
        self.assertTrue(result.unknown_detected)
        self.assertIsNotNone(result.unknown_component)
        self.assertEqual(int(np.argmax(result.unknown_component)), 55)

    def test_a_known_spectrum_does_not_trigger_unknown_detection(self):
        result = fit_spectrum(
            self.gold,
            self.dictionary,
            detect_unknown=True,
            unknown_threshold=0.01,
        )
        self.assertFalse(result.unknown_detected)

    def test_fitting_many_spectra(self):
        spectra = np.stack([self.gold, self.silver, 0.5 * self.gold])
        results = fit_spectra(spectra, self.dictionary)
        self.assertEqual(len(results), 3)
        self.assertEqual(
            [result.dominant for result in results], ["gold", "silver", "gold"]
        )

    def test_a_mismatched_spectrum_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_spectrum(np.ones(N_BINS + 1), self.dictionary)


if __name__ == "__main__":
    unittest.main()
