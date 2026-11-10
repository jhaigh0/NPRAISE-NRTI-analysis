"""Tests for the isotope templates and the isotope fits."""

import unittest

import numpy as np

from data_manipulation.fit_endf import fit_region
from endf_database_util.local_endf import find_database_root
from endf_database_util.templates import (
    TemplateSet,
    build_fine_isotope_templates,
    build_isotope_templates,
    isotope_name,
    parse_isotope_name,
)

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


class IsotopeNameTests(unittest.TestCase):
    """Tests for the names of the isotopes."""

    def test_the_name_is_built_from_the_symbol_and_the_mass(self):
        self.assertEqual(isotope_name("Ag", 107), "Ag-107")

    def test_the_name_is_split_again(self):
        self.assertEqual(parse_isotope_name("Ag-107"), ("Ag", 107))
        self.assertEqual(parse_isotope_name("Pb-208"), ("Pb", 208))

    def test_a_name_without_a_mass_raises(self):
        with self.assertRaises(ValueError):
            parse_isotope_name("Ag")

    def test_a_name_with_a_bad_mass_raises(self):
        with self.assertRaises(ValueError):
            parse_isotope_name("Ag-abc")


class IsotopeTemplateTests(unittest.TestCase):
    """Tests for building the templates of the isotopes."""

    @_requires_database
    def test_builds_a_template_per_isotope(self):
        energies = np.geomspace(1.0, 500.0, 300)
        templates = build_isotope_templates(["Ag-107", "Ag-109"], energies)
        self.assertEqual(templates.isotope_names, ["Ag-107", "Ag-109"])
        for name in templates.isotope_names:
            template = templates.isotope_templates[name]
            self.assertEqual(len(template.cross_section), len(energies))
            self.assertTrue(np.all(template.cross_section >= 0))

    @_requires_database
    def test_the_isotopes_have_different_resonances(self):
        """The isotopes of silver have their resonances at different energies."""
        energies = np.geomspace(1.0, 500.0, 2000)
        templates = build_isotope_templates(["Ag-107", "Ag-109"], energies)
        resonances_107 = set(
            np.round(templates.isotope_templates["Ag-107"].resonance_energies, 1)
        )
        resonances_109 = set(
            np.round(templates.isotope_templates["Ag-109"].resonance_energies, 1)
        )
        # The two sets of resonances do not overlap, which is what makes
        # the isotopes tell-able-apart.
        self.assertEqual(resonances_107 & resonances_109, set())

    @_requires_database
    def test_the_fine_grid_is_used(self):
        energies = np.geomspace(1.0, 500.0, 100)
        templates = build_fine_isotope_templates(
            ["Ag-107", "Ag-109"], energies, refinement=10
        )
        self.assertEqual(len(templates.energies), 1000)
        self.assertIsNotNone(templates.measurement_energies)
        self.assertEqual(len(templates.measurement_energies), 100)

    @_requires_database
    def test_an_unknown_isotope_raises(self):
        energies = np.geomspace(1.0, 500.0, 50)
        with self.assertRaises((FileNotFoundError, ValueError)):
            build_isotope_templates(["Ag-999"], energies)


class IsotopeFitTests(unittest.TestCase):
    """Tests for fitting a profile against the isotopes of an element."""

    def _templates(self):
        """A synthetic template set with two isotopes of one element."""
        energies = np.geomspace(1.0, 500.0, 400)
        sigma_107 = np.exp(-((energies - 100.0) ** 2) / (2 * 2.0**2))
        sigma_109 = np.exp(-((energies - 250.0) ** 2) / (2 * 3.0**2))

        def make(name, mass, sigma):
            return type(
                "T",
                (),
                {
                    "symbol": name.split("-")[0],
                    "mass": mass,
                    "name": name,
                    "isotopes": [mass],
                    "abundances": {mass: 1.0},
                    "energies": energies,
                    "cross_section": sigma,
                    "resonance_energies": np.array([100.0]),
                },
            )()

        return TemplateSet(
            templates={},
            energies=energies,
            isotope_templates={
                "Ag-107": make("Ag-107", 107, sigma_107),
                "Ag-109": make("Ag-109", 109, sigma_109),
            },
        )

    def _profile(self, templates, fraction_107, amount=0.003):
        sigma_107 = templates.isotope_templates["Ag-107"].cross_section
        sigma_109 = templates.isotope_templates["Ag-109"].cross_section
        return np.exp(
            -amount * (fraction_107 * sigma_107 + (1 - fraction_107) * sigma_109)
        )

    def test_a_pure_isotope_is_found(self):
        templates = self._templates()
        profile = self._profile(templates, 1.0)
        result = fit_region(profile, ["Ag-107", "Ag-109"], templates=templates)
        self.assertAlmostEqual(result.atomic_fractions[0], 1.0, places=3)
        self.assertAlmostEqual(result.atomic_fractions[1], 0.0, places=3)

    def test_the_other_pure_isotope_is_found(self):
        templates = self._templates()
        profile = self._profile(templates, 0.0)
        result = fit_region(profile, ["Ag-107", "Ag-109"], templates=templates)
        self.assertAlmostEqual(result.atomic_fractions[0], 0.0, places=3)
        self.assertAlmostEqual(result.atomic_fractions[1], 1.0, places=3)

    def test_a_mixture_is_found(self):
        templates = self._templates()
        profile = self._profile(templates, 0.3)
        result = fit_region(profile, ["Ag-107", "Ag-109"], templates=templates)
        self.assertAlmostEqual(result.atomic_fractions[0], 0.3, places=2)
        self.assertAlmostEqual(result.atomic_fractions[1], 0.7, places=2)

    def test_the_isotope_numbers_are_reported(self):
        templates = self._templates()
        profile = self._profile(templates, 0.5)
        result = fit_region(profile, ["Ag-107", "Ag-109"], templates=templates)
        self.assertEqual(result.isotopes, {"Ag-107": 107, "Ag-109": 109})

    def test_the_mass_fractions_use_the_isotope_masses(self):
        templates = self._templates()
        profile = self._profile(templates, 0.5)
        result = fit_region(profile, ["Ag-107", "Ag-109"], templates=templates)
        self.assertIsNotNone(result.mass_fractions)
        # Silver-107 is the lighter of the two, so a 50-50 mix of atoms
        # is a little under 50 % by mass.
        self.assertLess(result.mass_fractions[0], 0.5)

    def test_an_unknown_isotope_raises(self):
        templates = self._templates()
        profile = self._profile(templates, 0.5)
        with self.assertRaises(ValueError):
            fit_region(profile, ["Ag-111"], templates=templates)


if __name__ == "__main__":
    unittest.main()
