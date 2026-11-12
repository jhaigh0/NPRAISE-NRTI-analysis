"""Tests for finding the nuclides that a measurement could contain."""

import unittest

from endf_database_util.discovery import (
    DiscoveryResult,
    NuclideResonances,
    _result_from_dict,
    _result_to_dict,
    scan_database,
)
from endf_database_util.local_endf import find_database_root

try:
    DATABASE_ROOT = find_database_root()
except FileNotFoundError:
    DATABASE_ROOT = None


def _requires_database(test):
    """Skip a test when the local ENDF database is not available."""
    return unittest.skipIf(
        DATABASE_ROOT is None,
        "The local ENDF database is not available.",
    )(test)


class DiscoveryResultTests(unittest.TestCase):
    """Tests for the result of a scan."""

    def _result(self):
        return DiscoveryResult(
            min_energy=1.0,
            max_energy=500.0,
            nuclides=[
                NuclideResonances(
                    symbol="Ag",
                    mass=107,
                    energies=[16.3, 51.6],
                    natural=True,
                ),
                NuclideResonances(
                    symbol="Na",
                    mass=22,
                    energies=[145.0],
                    natural=False,
                ),
            ],
            elements=["Ag"],
            isotopes=["Ag-107"],
        )

    def test_the_counts_are_reported(self):
        result = self._result()
        self.assertEqual(result.element_count, 1)
        self.assertEqual(result.isotope_count, 1)
        self.assertEqual(result.resonance_count, 3)

    def test_a_result_survives_a_round_trip_through_json(self):
        """The result can be cached and read back unchanged."""
        result = self._result()
        restored = _result_from_dict(_result_to_dict(result))
        self.assertEqual(restored.elements, result.elements)
        self.assertEqual(restored.isotopes, result.isotopes)
        self.assertEqual(restored.min_energy, result.min_energy)
        self.assertEqual(restored.max_energy, result.max_energy)
        self.assertEqual(len(restored.nuclides), len(result.nuclides))
        self.assertEqual(restored.nuclides[0].name, "Ag-107")
        self.assertTrue(restored.nuclides[0].natural)
        self.assertFalse(restored.nuclides[1].natural)


class ScanTests(unittest.TestCase):
    """Tests for the scan of the database."""

    def test_a_bad_energy_range_raises(self):
        with self.assertRaises(ValueError):
            scan_database(0.0, 500.0)
        with self.assertRaises(ValueError):
            scan_database(500.0, 1.0)

    @_requires_database
    def test_the_scan_finds_the_copper_resonance(self):
        """The copper resonance at 230 eV puts copper in the candidates."""
        result = scan_database(1.0, 500.0)
        self.assertIn("Cu", result.elements)
        self.assertIn("Cu-65", result.isotopes)

    @_requires_database
    def test_the_scan_excludes_the_radioactive_isotopes(self):
        """An isotope with no natural abundance is not a candidate.

        Sodium-22 and nickel-59 have resonances in the measured range,
        but neither has a natural abundance, so neither can be in a
        natural sample.
        """
        result = scan_database(1.0, 500.0)
        self.assertNotIn("Na", result.elements)
        self.assertNotIn("Ni", result.elements)
        self.assertNotIn("Y", result.elements)

    @_requires_database
    def test_the_scan_is_cached(self):
        """A second scan of the same range gives the same answer."""
        first = scan_database(1.0, 500.0)
        second = scan_database(1.0, 500.0)
        self.assertEqual(first.elements, second.elements)
        self.assertEqual(first.isotopes, second.isotopes)

    @_requires_database
    def test_a_narrow_range_gives_fewer_candidates(self):
        """A narrower energy range cannot give more candidates."""
        wide = scan_database(1.0, 500.0)
        narrow = scan_database(200.0, 500.0)
        self.assertLessEqual(set(narrow.elements), set(wide.elements))


if __name__ == "__main__":
    unittest.main()
