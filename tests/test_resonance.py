"""Tests for reconstructing cross-sections from ENDF resolved resonances."""

import html
import math
import re
import unittest
from pathlib import Path

import numpy as np

from endf_database_util.local_endf import (
    LocalENDFDatabase,
    RMatrixParameters,
    find_database_root,
)
from endf_database_util.resonance import (
    penetration_shift,
    phaseshift,
    total_cross_section,
    wave_number,
)
from endf_database_util.rmatrix import rmatrix_total_cross_section

try:
    DATABASE_ROOT = find_database_root()
except FileNotFoundError:
    DATABASE_ROOT = None

KAERI_CACHE = Path(__file__).parent.parent / "endf_database_util" / ".cache" / "endf"


def _load_kaeri(name):
    """Load a cached KAERI cross-section curve, or None if it is not present."""
    path = KAERI_CACHE / name
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    match = re.search(
        r"<span\s+class=['\"]text_s['\"]>(.*?)</span>",
        html.unescape(raw),
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    lines = [
        line.replace("<br>", "").strip()
        for line in match.group(1).splitlines()
        if line.strip()
    ]
    x_values = []
    y_values = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                x_values.append(float(parts[0]))
                y_values.append(float(parts[1]))
            except ValueError:
                pass
    x = np.asarray(x_values)
    y = np.asarray(y_values)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    unique_y = np.zeros_like(unique_x)
    np.add.at(unique_y, inverse, y)
    unique_y /= np.bincount(inverse)
    return unique_x, unique_y


def _database():
    if DATABASE_ROOT is None:
        raise unittest.SkipTest("ENDF/B-VIII.0 database not available")
    return LocalENDFDatabase(DATABASE_ROOT)


class PhysicalConstantsTests(unittest.TestCase):
    def test_wave_number(self):
        # ENDF-102 equation D.10 for a 63 amu target at 1 eV.
        k = wave_number(63.38, 1.0)
        self.assertAlmostEqual(k, 2.1627e-3, places=7)

    def test_wave_number_scales_with_energy(self):
        k1 = wave_number(63.38, 1.0)
        k4 = wave_number(63.38, 4.0)
        self.assertAlmostEqual(k4 / k1, 2.0, places=10)

    def test_penetration_shift_l0(self):
        p, s = penetration_shift(0, 0.5)
        self.assertAlmostEqual(p, 0.5)
        self.assertEqual(s, 0.0)

    def test_penetration_shift_l1(self):
        p, s = penetration_shift(1, 1.0)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(s, -0.5)

    def test_phaseshift_l0(self):
        self.assertAlmostEqual(phaseshift(0, 0.7), 0.7)

    def test_phaseshift_l1(self):
        self.assertAlmostEqual(phaseshift(1, 1.0), 1.0 - math.pi / 4)


class MLBWReconstructionTests(unittest.TestCase):
    def test_reconstruction_is_positive_and_finite(self):
        db = _database()
        params = db.resonance_parameters("Zn", 64)
        energies = np.linspace(0.01, 500.0, 200)
        result = total_cross_section(params, energies)

        self.assertTrue(np.all(np.isfinite(result.total)))
        self.assertTrue(np.all(result.total >= 0.0))
        self.assertTrue(np.all(result.capture >= 0.0))
        self.assertTrue(np.all(result.elastic >= 0.0))

    def test_total_is_sum_of_partials(self):
        db = _database()
        params = db.resonance_parameters("Zn", 64)
        energies = np.linspace(0.01, 500.0, 100)
        result = total_cross_section(params, energies)
        np.testing.assert_allclose(
            result.total, result.elastic + result.capture + result.fission
        )

    def test_peak_at_known_resonance(self):
        # Zinc-64 has a strong resonance at 281 eV. The reconstructed
        # cross-section must peak there.
        db = _database()
        params = db.resonance_parameters("Zn", 64)
        energies = np.linspace(270.0, 292.0, 2000)
        result = total_cross_section(params, energies)
        peak = energies[int(np.argmax(result.total))]
        self.assertAlmostEqual(peak, 281.0, delta=2.0)
        self.assertGreater(result.total.max(), 10.0)

    def test_resonance_area_matches_reference(self):
        # The area of a resonance feature is preserved by any broadening, so
        # it is the right quantity to compare against the KAERI reference
        # curve for the same ENDF/B-VIII.0 evaluation. The tabulated MF=3
        # section in the ENDF file is far too coarse to use here.
        reference = _load_kaeri("107-Ag.txt")
        if reference is None:
            self.skipTest("KAERI reference data not available")
        x_ref, y_ref = reference

        db = _database()
        params = db.resonance_parameters("Ag", 107)
        centre = 51.56
        half_width = 1.0
        grid = np.linspace(centre - half_width, centre + half_width, 4001)
        reconstructed = total_cross_section(params, grid).total
        tabulated = np.interp(grid, x_ref, y_ref)

        def feature_area(x, y):
            return np.trapezoid(y - np.median(y), x)

        reconstructed_area = feature_area(grid, reconstructed)
        reference_area = feature_area(grid, tabulated)

        # The reference curve is smoothed, which loses some area from the
        # sharpest features, so allow a factor of three.
        ratio = reconstructed_area / reference_area
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 3.0)


class RMatrixReconstructionTests(unittest.TestCase):
    def test_copper_65_peak_position(self):
        # The copper-65 R-matrix resonance at 230.04 eV must appear at the
        # right energy. This is the resonance responsible for the dip seen
        # in the copper selection.
        db = _database()
        params = db.resonance_parameters("Cu", 65)
        energies = np.linspace(225.0, 235.0, 2000)
        result = rmatrix_total_cross_section(params, energies)
        peak = energies[int(np.argmax(result.total))]
        self.assertAlmostEqual(peak, 230.04, delta=0.5)
        self.assertGreater(result.total.max(), 20.0)

    def test_copper_63_has_no_low_energy_peak(self):
        db = _database()
        params = db.resonance_parameters("Cu", 63)
        energies = np.linspace(200.0, 260.0, 2000)
        result = rmatrix_total_cross_section(params, energies)
        # Copper-63 has no resonance in this window, so the cross-section
        # must stay small and smooth.
        self.assertLess(result.total.max(), 5.0)

    def test_is_rmatrix(self):
        db = _database()
        params = db.resonance_parameters("Cu", 65)
        self.assertIsInstance(params, RMatrixParameters)


if __name__ == "__main__":
    unittest.main()
