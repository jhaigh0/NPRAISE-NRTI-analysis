"""Tests for building a dictionary from a YAML configuration file."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from mantid.simpleapi import (
    CreateWorkspace,
    ExtractSpectra,
    LoadInstrument,
    SaveNexus,
)

from data_manipulation.build_dictionary import (
    ConfigError,
    build_configured_dictionary,
    load_dictionary_config,
)
from data_manipulation.dictionary import (
    Dictionary,
    SelectionSpec,
    build_dictionary_from_selections,
)

N_PIXELS = 16
N_BINS = 40
# The converted data is rebinned onto the fixed energy grid, so the number of
# output bins does not depend on the number of bins of the input workspace.
N_OUTPUT_BINS = 1246


class LoadDictionaryConfigTests(unittest.TestCase):
    """Tests for parsing the dictionary configuration file."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def _write(self, content: str) -> Path:
        """Write a config file into the temporary directory and return its path."""
        config_path = self.tmp_path / "dictionary.yaml"
        config_path.write_text(content, encoding="utf-8")
        return config_path

    def test_loads_selections(self):
        config = load_dictionary_config(
            self._write(
                """
l2_values: [0.62, 0.63, 0.64, 0.65]

selections:
  - name: Ag/Ag
    selection: sel_ag.nxs
    isotopes: [Ag-107, Ag-109]
    workspace: Ag
  - name: Ag/background
    selection: sel_bg.nxs
    isotopes: []
    workspace: Ag
""",
            )
        )

        self.assertEqual(config.l2_values, [0.62, 0.63, 0.64, 0.65])
        self.assertEqual(len(config.selections), 2)
        self.assertEqual(config.selections[0].name, "Ag/Ag")
        self.assertEqual(config.selections[0].isotopes, ["Ag-107", "Ag-109"])
        self.assertEqual(config.selections[0].workspace, "Ag")
        self.assertEqual(config.selections[1].isotopes, [])

    def test_loads_generated_data_files(self):
        config = load_dictionary_config(
            self._write(
                """
files:
  - a.npz
  - b.npz
""",
            )
        )
        self.assertEqual(len(config.files), 2)
        self.assertEqual(config.files[0].name, "a.npz")

    def test_resolves_paths_relative_to_the_config(self):
        config = load_dictionary_config(
            self._write(
                """
selections:
  - name: Ag/Ag
    selection: sel.nxs
    isotopes: [Ag-107]
""",
            )
        )
        self.assertEqual(
            config.selections[0].path, (self.tmp_path / "sel.nxs").resolve()
        )

    def test_a_single_isotope_string_is_accepted(self):
        config = load_dictionary_config(
            self._write(
                """
selections:
  - name: Ag/Ag
    selection: sel.nxs
    isotopes: Ag-107
""",
            )
        )
        self.assertEqual(config.selections[0].isotopes, ["Ag-107"])

    def test_a_config_with_neither_selections_nor_files_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(self._write("l2_values: [0.63]\n"))

    def test_a_selection_without_a_file_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(
                self._write(
                    """
selections:
  - name: Ag/Ag
    isotopes: [Ag-107]
""",
                )
            )

    def test_a_selection_without_a_name_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(
                self._write(
                    """
selections:
  - selection: sel.nxs
    isotopes: [Ag-107]
""",
                )
            )

    def test_duplicate_l2_values_raise(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(self._write("l2_values: [0.63, 0.63]\n"))

    def test_a_non_positive_l2_value_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(self._write("l2_values: [-0.63]\n"))

    def test_a_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(self.tmp_path / "does_not_exist.yaml")

    def test_an_empty_file_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(self._write(""))

    def test_a_file_that_is_not_a_mapping_raises(self):
        with self.assertRaises(ConfigError):
            load_dictionary_config(self._write("- one\n- two\n"))


class BuildFromSelectionsTests(unittest.TestCase):
    """Tests that build a dictionary from real Mantid selection workspaces."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

        edges = np.arange(100.0, 100.0 + (N_BINS + 1) * 10.0, 10.0)
        rng = np.random.default_rng(0)
        y = rng.poisson(10.0, size=(N_PIXELS * N_PIXELS, N_BINS)).astype(float)

        full_ws = CreateWorkspace(
            DataX=edges, DataY=y, NSpec=N_PIXELS * N_PIXELS, UnitX="TOF"
        )
        LoadInstrument(full_ws, InstrumentName="NGEM", RewriteSpectraMap=True)

        self.gold_path = self._save_selection(
            full_ws, list(range(100, 140)), "gold.nxs"
        )
        self.silver_path = self._save_selection(
            full_ws, list(range(140, 180)), "silver.nxs"
        )

    def _save_selection(self, full_ws, det_ids, name: str) -> Path:
        """Extract a detector region and save it as a selection workspace."""
        path = self.tmp_path / name
        selection_ws = ExtractSpectra(
            full_ws,
            DetectorList=",".join(map(str, det_ids)),
            StoreInADS=False,
        )
        SaveNexus(selection_ws, Filename=str(path))
        return path

    def test_builds_one_entry_per_selection(self):
        dictionary = build_dictionary_from_selections(
            [
                SelectionSpec(
                    name="gold",
                    path=self.gold_path,
                    isotopes=["Au-197"],
                    workspace="gold_workspace",
                ),
                SelectionSpec(
                    name="silver",
                    path=self.silver_path,
                    isotopes=["Ag-107", "Ag-109"],
                ),
            ],
            l2_values=[0.63],
        )

        self.assertEqual(dictionary.names, ["gold", "silver"])
        self.assertEqual(dictionary.entries[0].isotopes, ["Au-197"])
        self.assertEqual(dictionary.entries[1].isotopes, ["Ag-107", "Ag-109"])
        self.assertFalse(dictionary.entries[0].is_background)
        self.assertFalse(dictionary.entries[1].is_background)

    def test_a_selection_without_isotopes_is_background(self):
        dictionary = build_dictionary_from_selections(
            [SelectionSpec(name="background", path=self.gold_path)],
            l2_values=[0.63],
        )
        self.assertTrue(dictionary.entries[0].is_background)
        self.assertEqual(dictionary.entries[0].isotopes, [])

    def test_the_matrix_is_normalised(self):
        dictionary = build_dictionary_from_selections(
            [
                SelectionSpec(name="gold", path=self.gold_path, isotopes=["Au-197"]),
                SelectionSpec(
                    name="silver", path=self.silver_path, isotopes=["Ag-107"]
                ),
            ],
            l2_values=[0.63],
        )
        matrix = dictionary.matrix
        self.assertEqual(matrix.shape[0], N_OUTPUT_BINS)
        self.assertEqual(matrix.shape[1], 2)
        np.testing.assert_allclose(matrix.sum(axis=0), np.ones(2))

    def test_n_samples_counts_the_detectors_and_l2_values(self):
        dictionary = build_dictionary_from_selections(
            [SelectionSpec(name="gold", path=self.gold_path, isotopes=["Au-197"])],
            l2_values=[0.62, 0.63],
        )
        self.assertEqual(dictionary.entries[0].n_samples, 40 * 2)

    def test_the_source_is_recorded(self):
        dictionary = build_dictionary_from_selections(
            [SelectionSpec(name="gold", path=self.gold_path, isotopes=["Au-197"])],
            l2_values=[0.63],
        )
        self.assertEqual(dictionary.entries[0].source, str(self.gold_path))

    def test_a_saved_dictionary_can_be_reloaded(self):
        dictionary = build_dictionary_from_selections(
            [
                SelectionSpec(name="gold", path=self.gold_path, isotopes=["Au-197"]),
                SelectionSpec(name="background", path=self.silver_path, isotopes=[]),
            ],
            l2_values=[0.63],
        )
        path = dictionary.save(self.tmp_path / "dictionary.npz")
        reloaded = Dictionary.load(path)

        self.assertEqual(reloaded.names, ["gold", "background"])
        self.assertEqual(reloaded.entries[0].isotopes, ["Au-197"])
        self.assertTrue(reloaded.entries[1].is_background)
        np.testing.assert_allclose(reloaded.matrix, dictionary.matrix)


class BuildConfiguredDictionaryTests(unittest.TestCase):
    """Tests for building a dictionary from a parsed configuration."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def test_builds_from_generated_data_files(self):
        profile = np.linspace(1.0, 100.0, 1246)
        data_path = self.tmp_path / "gold.npz"
        np.savez_compressed(
            data_path,
            spectra_samples=np.tile(profile, (4, 1)),
            x_bins=np.linspace(1000.0, 500000.0, 1247),
            name=np.asarray("sample"),
            selection_name=np.asarray("gold"),
            isotopes=np.asarray(["Au-197"]),
            is_background_only=np.asarray(False),
        )

        config_path = self.tmp_path / "dictionary.yaml"
        config_path.write_text(f"files:\n  - {data_path.name}\n", encoding="utf-8")
        config = load_dictionary_config(config_path)
        dictionary = build_configured_dictionary(config)

        self.assertEqual(dictionary.names, ["sample/gold"])
        self.assertEqual(dictionary.entries[0].isotopes, ["Au-197"])
        self.assertFalse(dictionary.entries[0].is_background)
        np.testing.assert_allclose(dictionary.matrix.sum(axis=0), np.ones(1))

    def test_selections_and_files_can_be_combined(self):
        profile = np.linspace(1.0, 100.0, 1246)
        data_path = self.tmp_path / "gold.npz"
        np.savez_compressed(
            data_path,
            spectra_samples=np.tile(profile, (4, 1)),
            x_bins=np.linspace(1000.0, 500000.0, 1247),
            name=np.asarray("sample"),
            selection_name=np.asarray("from_file"),
            isotopes=np.asarray(["Au-197"]),
            is_background_only=np.asarray(False),
        )

        config_path = self.tmp_path / "dictionary.yaml"
        config_path.write_text(f"files:\n  - {data_path.name}\n", encoding="utf-8")
        config = load_dictionary_config(config_path)
        dictionary = build_configured_dictionary(config)

        self.assertEqual(dictionary.names, ["sample/from_file"])


if __name__ == "__main__":
    unittest.main()
