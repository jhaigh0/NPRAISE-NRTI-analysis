"""Tests for reading the local ENDF/B-VIII.0 database.

The tests that need the database are skipped when it is not present, so the
suite still runs on a machine without the ENDF files.
"""

import unittest
from pathlib import Path

import numpy as np

from endf_database_util.local_endf import (
    LocalENDFDatabase,
    ResonanceParameters,
    RMatrixParameters,
    find_database_root,
)

try:
    DATABASE_ROOT = find_database_root()
except FileNotFoundError:
    DATABASE_ROOT = None


def _database():
    if DATABASE_ROOT is None:
        raise unittest.SkipTest("ENDF/B-VIII.0 database not available")
    return LocalENDFDatabase(DATABASE_ROOT)


class LocalENDFDatabaseTests(unittest.TestCase):
    def test_find_file(self):
        db = _database()
        path = db.find_file("Cu", 63)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "n-029_Cu_063.endf")

    def test_find_file_metastable(self):
        db = _database()
        path = db.find_file("Ag", 110)
        self.assertTrue(path.exists())
        self.assertIn("110m1", path.name)

    def test_find_file_missing(self):
        db = _database()
        with self.assertRaises(FileNotFoundError):
            db.find_file("Cu", 999)

    def test_available_isotopes(self):
        db = _database()
        isotopes = db.available_isotopes()
        self.assertIn("Cu-063", isotopes)
        self.assertIn("Zn-064", isotopes)
        self.assertGreater(len(isotopes), 100)

    def test_tabulated_cross_section(self):
        db = _database()
        x, y = db.tabulated_total_cross_section("Zn", 64)
        self.assertGreater(len(x), 100)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.all(x > 0))
        self.assertTrue(np.all(y >= 0))

    def test_mlbw_resonance_parameters(self):
        db = _database()
        params = db.resonance_parameters("Zn", 64)
        self.assertIsInstance(params, ResonanceParameters)
        self.assertEqual(params.formalism, 2)
        self.assertGreater(len(params.resonances), 100)

        energies = np.array([r.energy for r in params.resonances])
        # Zinc-64 has a strong resonance at 281 eV, which is visible in the
        # zinc selection, so it must appear here.
        self.assertTrue(
            np.any(np.abs(energies - 281.0) < 1.0),
            "no resonance near 281 eV",
        )

    def test_reich_moore_resonance_parameters(self):
        db = _database()
        params = db.resonance_parameters("Au", 197)
        self.assertIsInstance(params, ResonanceParameters)
        self.assertEqual(params.formalism, 3)
        energies = np.array([r.energy for r in params.resonances])
        # The 4.9 eV gold resonance is prominent in the measured spectrum.
        self.assertTrue(np.any(np.abs(energies - 4.8997) < 0.01))

    def test_rmatrix_resonance_parameters(self):
        db = _database()
        params = db.resonance_parameters("Cu", 65)
        self.assertIsInstance(params, RMatrixParameters)
        self.assertEqual(len(params.spin_groups), 6)
        self.assertEqual(params.particle_pairs.shape, (2, 12))

        energies = np.concatenate([group["energies"] for group in params.spin_groups])
        # Copper-65 has a resonance at 230.04 eV, which is the source of the
        # dip observed in the copper selection.
        self.assertTrue(np.any(np.abs(energies - 230.0444) < 0.01))

    def test_rmatrix_copper_63(self):
        db = _database()
        params = db.resonance_parameters("Cu", 63)
        self.assertIsInstance(params, RMatrixParameters)
        energies = np.concatenate([group["energies"] for group in params.spin_groups])
        # Copper-63 has no resonance below 400 eV, which is why the copper
        # dip comes from copper-65 rather than copper-63.
        self.assertFalse(np.any((energies > 0.05) & (energies < 400.0)))


class FindDatabaseRootTests(unittest.TestCase):
    def test_raises_when_missing(self):
        import tempfile

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaises(FileNotFoundError),
        ):
            find_database_root(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
