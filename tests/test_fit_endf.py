"""Tests for fitting measured regions against the local ENDF database."""

import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import ClassVar

import numpy as np

from data_manipulation.fit_endf import (
    ENDFfitResult,
    _atomic_mass,
    _scaled_start,
    _simplex_from_stick,
    _stick_jacobian,
    fit_region,
    fit_regions,
    main,
)
from endf_database_util.local_endf import find_database_root
from endf_database_util.templates import (
    ElementTemplate,
    TemplateSet,
    build_fine_templates,
    build_templates,
    element_isotopes,
    isotopic_abundances,
)

try:
    DATABASE_ROOT = find_database_root()
except FileNotFoundError:
    DATABASE_ROOT = None

DICTIONARY = Path(__file__).parent.parent / "dictionary_all_elements.npz"


def _requires_database(test):
    """Skip a test when the local ENDF database is not available."""
    return unittest.skipIf(
        DATABASE_ROOT is None,
        "The local ENDF database is not available.",
    )(test)


class ElementIsotopeTests(unittest.TestCase):
    """Tests for the isotope lists and abundances."""

    def test_natural_isotopes(self):
        self.assertEqual(element_isotopes("Cu"), [63, 65])
        self.assertEqual(element_isotopes("Au"), [197])
        self.assertEqual(element_isotopes("Nb"), [93])
        self.assertEqual(element_isotopes("Zn"), [64, 66, 67, 68, 70])

    def test_abundances_sum_to_one(self):
        abundances = isotopic_abundances("Cu", [63, 65])
        self.assertAlmostEqual(sum(abundances.values()), 1.0)
        self.assertAlmostEqual(abundances[63], 0.6917, places=3)
        self.assertAlmostEqual(abundances[65], 0.3083, places=3)

    def test_unknown_element_raises(self):
        with self.assertRaises(ValueError):
            element_isotopes("Xx")


class TemplateTests(unittest.TestCase):
    """Tests for building the cross-section templates."""

    @_requires_database
    def test_builds_templates(self):
        energies = np.geomspace(1.0, 500.0, 500)
        templates = build_templates(
            ["Cu", "Zn"], energies, cache_dir=Path(self._cache_dir())
        )
        self.assertEqual(len(templates), 2)
        self.assertIn("Cu", templates)
        for symbol in templates.symbols:
            template = templates[symbol]
            self.assertEqual(len(template.cross_section), len(energies))
            self.assertTrue(np.all(template.cross_section >= 0))
            self.assertTrue(np.all(np.isfinite(template.cross_section)))

    @_requires_database
    def test_copper_resonance_at_230_eV(self):
        energies = np.geomspace(1.0, 500.0, 2000)
        templates = build_templates(["Cu"], energies, cache_dir=Path(self._cache_dir()))
        template = templates["Cu"]
        peak = template.cross_section.argmax()
        self.assertAlmostEqual(energies[peak], 230.0, delta=1.0)
        # The resonance energy itself must be reported.
        self.assertTrue(np.any(np.abs(template.resonance_energies - 230.04) < 0.1))

    @_requires_database
    def test_cache_is_reused(self):
        energies = np.geomspace(1.0, 500.0, 200)
        cache = Path(self._cache_dir())
        first = build_templates(["Au"], energies, cache_dir=cache)
        second = build_templates(["Au"], energies, cache_dir=cache)
        self.assertTrue(
            np.array_equal(first["Au"].cross_section, second["Au"].cross_section)
        )

    def test_unknown_element_raises(self):
        with self.assertRaises(ValueError):
            build_templates(["Xx"], np.geomspace(1.0, 500.0, 10))

    @_requires_database
    def test_fine_grid_resolves_the_narrow_resonances(self):
        """The measurement grid undersamples the narrow resonances.

        The measured spectra are rebinned onto a grid with a step of about
        0.5 %, which is 1.15 eV at 230 eV. The copper-65 resonance at
        230 eV is much narrower than that, so evaluating it on the
        measurement grid loses most of its peak. The fine grid has to
        recover it.
        """
        energies = 1000.0 * 1.005 ** np.arange(1246) / 1000.0
        coarse = build_templates(
            ["Cu", "Nb"], energies, cache_dir=Path(self._cache_dir())
        )
        fine = build_fine_templates(
            ["Cu", "Nb"], energies, cache_dir=Path(self._cache_dir())
        )
        self.assertGreater(len(fine.energies), len(coarse.energies))
        self.assertEqual(len(fine.measurement_energies), len(energies))
        for symbol in ["Cu", "Nb"]:
            self.assertGreater(
                fine[symbol].cross_section.max(),
                1.5 * coarse[symbol].cross_section.max(),
            )

    @staticmethod
    def _cache_dir():
        import tempfile

        return tempfile.mkdtemp(prefix="endf_templates_test_")


class FitRegionTests(unittest.TestCase):
    """Tests for the fitting itself, using synthetic profiles."""

    def _templates(self):
        """A small synthetic template set, with no database needed."""
        energies = np.geomspace(1.0, 500.0, 400)
        sigma_a = np.exp(-((energies - 100.0) ** 2) / (2 * 2.0**2))
        sigma_b = np.exp(-((energies - 250.0) ** 2) / (2 * 3.0**2))
        return TemplateSet(
            templates={
                "A": type(
                    "T",
                    (),
                    {
                        "symbol": "A",
                        "isotopes": [1],
                        "abundances": {1: 1.0},
                        "energies": energies,
                        "cross_section": sigma_a,
                        "resonance_energies": np.array([100.0]),
                    },
                )(),
                "B": type(
                    "T",
                    (),
                    {
                        "symbol": "B",
                        "isotopes": [2],
                        "abundances": {2: 1.0},
                        "energies": energies,
                        "cross_section": sigma_b,
                        "resonance_energies": np.array([250.0]),
                    },
                )(),
            },
            energies=energies,
        )

    def test_finds_a_single_element(self):
        templates = self._templates()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertIsInstance(result, ENDFfitResult)
        self.assertEqual(result.dominant, "A")
        self.assertGreater(result.fractions[0], 0.9)

    def test_finds_a_mixture(self):
        templates = self._templates()
        sigma_a = templates["A"].cross_section
        sigma_b = templates["B"].cross_section
        profile = np.exp(
            -(1.0 * sigma_a / sigma_a.max() + 0.5 * sigma_b / sigma_b.max())
        )
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertEqual(result.dominant, "A")
        # The shares of the modelled optical thickness are 1 and 0.5.
        self.assertAlmostEqual(result.fractions[0], 2.0 / 3.0, places=2)
        self.assertAlmostEqual(result.fractions[1], 1.0 / 3.0, places=2)

    def test_residual_is_small_for_a_good_fit(self):
        templates = self._templates()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertLess(result.rms_residual, 1e-3)
        self.assertEqual(len(result.residual), len(profile))

    def test_areal_densities_are_positive(self):
        templates = self._templates()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertTrue(np.all(result.areal_densities >= 0))

    def test_mismatched_length_raises(self):
        templates = self._templates()
        with self.assertRaises(ValueError):
            fit_region(np.ones(10), ["A"], templates=templates)

    def test_missing_element_raises(self):
        templates = self._templates()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        with self.assertRaises(ValueError):
            fit_region(profile, ["C"], templates=templates)

    def test_no_elements_raises(self):
        templates = self._templates()
        with self.assertRaises(ValueError):
            fit_region(np.ones(10), [], templates=templates)

    def test_element_without_cross_section_is_excluded(self):
        """An element with no cross-section carries no information.

        Its coefficient would be unconstrained, so the fit could put any
        amount of it there. Such an element is dropped from the fit and
        reported separately.
        """
        templates = self._templates()
        # Add an element whose cross-section is zero everywhere.
        templates.templates["C"] = type(
            "T",
            (),
            {
                "symbol": "C",
                "isotopes": [12],
                "abundances": {12: 1.0},
                "energies": templates.energies,
                "cross_section": np.zeros_like(templates.energies),
                "resonance_energies": np.zeros(0),
            },
        )()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B", "C"], templates=templates)
        self.assertEqual(result.excluded, ["C"])
        self.assertNotIn("C", result.symbols)
        self.assertEqual(result.dominant, "A")

    def test_all_elements_without_cross_section_raises(self):
        templates = self._templates()
        templates.templates["A"] = type(
            "T",
            (),
            {
                "symbol": "A",
                "isotopes": [1],
                "abundances": {1: 1.0},
                "energies": templates.energies,
                "cross_section": np.zeros_like(templates.energies),
                "resonance_energies": np.zeros(0),
            },
        )()
        with self.assertRaises(ValueError):
            fit_region(np.ones(len(templates.energies)), ["A"], templates=templates)

    def test_fit_regions_returns_one_result_each(self):
        templates = self._templates()
        sigma = templates["A"].cross_section
        profiles = np.vstack([np.exp(-2.0 * sigma / sigma.max())] * 3)

        results = fit_regions(profiles, ["A", "B"], templates=templates)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.dominant, "A")

    def test_a_fine_grid_template_set_is_accepted(self):
        """Templates on a finer grid are sampled onto the measurement grid."""
        templates = self._templates()
        measurement = templates.energies
        fine = np.linspace(1.0, 500.0, measurement.size * 10)
        fine_templates = TemplateSet(
            templates={
                symbol: type(
                    "T",
                    (),
                    {
                        "symbol": symbol,
                        "isotopes": [1],
                        "abundances": {1: 1.0},
                        "energies": fine,
                        "cross_section": np.interp(
                            fine, measurement, template.cross_section
                        ),
                        "resonance_energies": template.resonance_energies,
                    },
                )()
                for symbol, template in templates.templates.items()
            },
            energies=fine,
            measurement_energies=measurement,
        )
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=fine_templates)
        self.assertEqual(result.dominant, "A")
        self.assertGreater(result.fractions[0], 0.9)

    def test_a_mismatched_measurement_grid_raises(self):
        """A template set whose measurement grid does not match raises."""
        templates = self._templates()
        templates.measurement_energies = templates.energies[:-1]
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        with self.assertRaises(ValueError):
            fit_region(profile, ["A", "B"], templates=templates)


class OptimiserTests(unittest.TestCase):
    """Tests for the optimiser, which has to cope with many candidates."""

    def test_the_stick_jacobian_matches_the_stick_parameters(self):
        """The analytic stick Jacobian agrees with a numerical one."""
        parameters = np.array([0.2, 0.5, 0.3, 0.8])
        analytic = _stick_jacobian(parameters)
        numerical = np.zeros_like(analytic)
        for index in range(parameters.size):
            step = 1e-7
            higher = parameters.copy()
            higher[index] += step
            lower = parameters.copy()
            lower[index] -= step
            numerical[:, index] = (
                _simplex_from_stick(higher) - _simplex_from_stick(lower)
            ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, atol=1e-6)

    def test_the_scaled_start_has_a_sane_optical_thickness(self):
        """The generic start does not saturate the transmission."""
        design = np.ones((10, 50))
        start = _scaled_start(design)
        self.assertAlmostEqual(float(np.max(design @ start)), 1.0)

    def test_a_fit_with_many_candidates_finishes(self):
        """A fit with more candidates than the start-point limit works.

        The single-element starts are only used for a small number of
        elements, because each one costs a full fit. The generic starts
        have to be enough on their own for a large number of candidates.

        The cross-sections are narrow bumps in the upper half of the
        energy range, so that the profile has a median of one and the
        normalisation of the measured transmission does not change it.
        """
        energies = np.linspace(1.0, 500.0, 120)
        templates = TemplateSet(
            energies=energies,
            templates={},
            measurement_energies=None,
        )
        symbols = [f"X{index}" for index in range(40)]
        for index, symbol in enumerate(symbols):
            centre = 260.0 + 6.0 * index
            sigma = np.exp(-(((energies - centre) / 4.0) ** 2))
            sigma[energies < 250.0] = 0.0
            templates.templates[symbol] = ElementTemplate(
                symbol=symbol,
                isotopes=[],
                abundances={},
                energies=energies,
                cross_section=sigma,
            )
        truth = np.zeros(len(symbols))
        truth[3] = 1.0
        truth[17] = 0.5
        matrix = np.column_stack(
            [templates.templates[s].cross_section for s in symbols]
        )
        profile = np.exp(-(matrix @ truth))
        result = fit_region(profile, symbols, templates=templates)
        self.assertLess(result.rms_residual, 1e-3)
        self.assertGreater(result.fractions[3], 0.5)
        self.assertGreater(result.fractions[17], 0.1)


class FloorTests(unittest.TestCase):
    """Tests for the treatment of a saturated dip."""

    def _templates(self):
        energies = np.linspace(1.0, 500.0, 200)
        sigma_a = np.exp(-((energies - 250.0) ** 2) / 50.0)
        sigma_b = np.exp(-((energies - 100.0) ** 2) / 50.0)
        return TemplateSet(
            templates={
                "A": type(
                    "T",
                    (),
                    {
                        "symbol": "A",
                        "isotopes": [1],
                        "abundances": {1: 1.0},
                        "energies": energies,
                        "cross_section": sigma_a,
                        "resonance_energies": np.array([250.0]),
                    },
                )(),
                "B": type(
                    "T",
                    (),
                    {
                        "symbol": "B",
                        "isotopes": [2],
                        "abundances": {2: 1.0},
                        "energies": energies,
                        "cross_section": sigma_b,
                        "resonance_energies": np.array([100.0]),
                    },
                )(),
            },
            energies=energies,
        )

    def test_an_unsaturated_dip_has_no_floor(self):
        """A dip that reaches zero transmission is not saturated."""
        templates = self._templates()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertEqual(result.floor, 0.0)
        self.assertFalse(result.saturated)

    def test_a_saturated_dip_is_detected(self):
        """A dip that flattens out at a floor is reported as saturated."""
        templates = self._templates()
        sigma = templates["A"].cross_section
        floor = 0.05
        profile = floor + (1.0 - floor) * np.exp(-20.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertAlmostEqual(result.floor, floor, places=3)
        self.assertTrue(result.saturated)

    def test_the_floor_can_be_set_by_hand(self):
        """The floor can be given explicitly, which overrides the estimate."""
        templates = self._templates()
        sigma = templates["A"].cross_section
        profile = np.exp(-2.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates, floor=0.2)
        self.assertEqual(result.floor, 0.2)
        self.assertTrue(result.saturated)

    def test_a_saturated_dip_still_finds_the_right_element(self):
        """The floor does not change which element is identified."""
        templates = self._templates()
        sigma = templates["A"].cross_section
        floor = 0.05
        profile = floor + (1.0 - floor) * np.exp(-20.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertEqual(result.dominant, "A")

    def test_the_saturated_bins_can_be_left_out(self):
        """The saturated bins are left out when they are asked for."""
        templates = self._templates()
        sigma = templates["A"].cross_section
        floor = 0.05
        profile = floor + (1.0 - floor) * np.exp(-20.0 * sigma / sigma.max())
        result = fit_region(
            profile, ["A", "B"], templates=templates, unsaturated_only=True
        )
        self.assertLess(result.unsaturated_bins, len(profile))
        self.assertEqual(result.dominant, "A")

    def test_all_the_bins_are_used_by_default(self):
        """Every bin is used when the saturated bins are not asked for."""
        templates = self._templates()
        sigma = templates["A"].cross_section
        floor = 0.05
        profile = floor + (1.0 - floor) * np.exp(-20.0 * sigma / sigma.max())
        result = fit_region(profile, ["A", "B"], templates=templates)
        self.assertEqual(result.unsaturated_bins, len(profile))


class FixedCompositionTests(unittest.TestCase):
    """Tests for fitting with a known composition."""

    def _templates(self):
        energies = np.linspace(1.0, 500.0, 200)
        sigma_a = np.exp(-((energies - 250.0) ** 2) / 50.0)
        sigma_b = np.exp(-((energies - 100.0) ** 2) / 50.0)
        return TemplateSet(
            templates={
                symbol: type(
                    "T",
                    (),
                    {
                        "symbol": symbol,
                        "isotopes": [1],
                        "abundances": {1: 1.0},
                        "energies": energies,
                        "cross_section": sigma,
                        "resonance_energies": np.array([250.0]),
                    },
                )()
                for symbol, sigma in (("Cu", sigma_a), ("Zn", sigma_b))
            },
            energies=energies,
        )

    def test_a_known_composition_recovers_the_amount_of_material(self):
        """With a known composition, the amount of material is recovered."""
        templates = self._templates()
        sigma_a = templates["Cu"].cross_section
        sigma_b = templates["Zn"].cross_section
        # The composition is given as a mass fraction, so the number
        # fractions have to be worked out from the atomic masses.
        composition = {"Cu": 2.0, "Zn": 1.0}
        numbers = {
            symbol: mass / _atomic_mass(symbol) for symbol, mass in composition.items()
        }
        total = sum(numbers.values())
        numbers = {k: v / total for k, v in numbers.items()}
        # A known amount of material, in atoms per barn.
        amount = 0.003
        profile = np.exp(-amount * (numbers["Cu"] * sigma_a + numbers["Zn"] * sigma_b))
        result = fit_region(
            profile,
            ["Cu", "Zn"],
            templates=templates,
            fixed_composition=composition,
        )
        self.assertIsNotNone(result.fixed_composition)
        self.assertAlmostEqual(
            result.areal_densities[0], amount * numbers["Cu"], places=6
        )
        self.assertAlmostEqual(
            result.areal_densities[1], amount * numbers["Zn"], places=6
        )

    def test_a_known_composition_survives_a_saturated_dip(self):
        """The amount of material is recovered even from a clipped dip."""
        templates = self._templates()
        sigma_a = templates["Cu"].cross_section
        sigma_b = templates["Zn"].cross_section
        composition = {"Cu": 2.0, "Zn": 1.0}
        numbers = {
            symbol: mass / _atomic_mass(symbol) for symbol, mass in composition.items()
        }
        total = sum(numbers.values())
        numbers = {k: v / total for k, v in numbers.items()}
        amount = 30.0
        floor = 0.05
        profile = floor + (1.0 - floor) * np.exp(
            -amount * (numbers["Cu"] * sigma_a + numbers["Zn"] * sigma_b)
        )
        result = fit_region(
            profile,
            ["Cu", "Zn"],
            templates=templates,
            fixed_composition=composition,
        )
        self.assertAlmostEqual(
            result.areal_densities[0], amount * numbers["Cu"], places=2
        )
        self.assertAlmostEqual(
            result.areal_densities[1], amount * numbers["Zn"], places=2
        )

    def test_a_composition_with_no_matching_elements_raises(self):
        """A composition that does not match the elements raises."""
        templates = self._templates()
        profile = np.ones(len(templates.energies))
        with self.assertRaises(ValueError):
            fit_region(
                profile,
                ["Cu", "Zn"],
                templates=templates,
                fixed_composition={"Fe": 1.0},
            )


@unittest.skipIf(
    not DICTIONARY.exists(),
    "The dictionary of measured regions is not available.",
)
@_requires_database
class MeasuredRegionTests(unittest.TestCase):
    """Tests against the measured regions of the dictionary."""

    ELEMENTS: ClassVar[list[str]] = ["Ag", "Au", "In", "Zn", "Cu", "Nb", "Fe"]

    @classmethod
    def setUpClass(cls):
        with np.load(DICTIONARY, allow_pickle=True) as data:
            cls.names = [str(name) for name in data["entry_names"]]
            cls.profiles = data["profiles"]
            cls.x_bins = data["x_bins"]
        centres = np.sqrt(cls.x_bins[:-1] * cls.x_bins[1:]) / 1000.0
        cls.templates = build_templates(cls.ELEMENTS, centres)

    def test_identifies_each_region(self):
        """Every measured region is identified as its own element.

        The iron region is excluded, because the iron standard was found to
        hold only noise, so there is nothing there to identify.
        """
        expected = {
            "Ag/Ag": "Ag",
            "Au/Au_1foil": "Au",
            "In/In": "In",
            "Zn/Zn": "Zn",
            "Cu/Cu": "Cu",
            "Nb/Nb_disk_right": "Nb",
        }
        for name, element in expected.items():
            with self.subTest(region=name):
                index = self.names.index(name)
                result = fit_region(
                    self.profiles[:, index],
                    self.ELEMENTS,
                    templates=self.templates,
                )
                self.assertEqual(result.dominant, element)
                self.assertGreater(result.fractions[result.symbols.index(element)], 0.8)

    def test_copper_is_identified(self):
        """The copper dip is the Cu-65 resonance at 230 eV.

        Copper-63 has no resonance below 400 eV, so a natural copper sample
        is only recognisable through Cu-65.
        """
        index = self.names.index("Cu/Cu")
        result = fit_region(
            self.profiles[:, index], self.ELEMENTS, templates=self.templates
        )
        self.assertEqual(result.dominant, "Cu")
        self.assertGreater(result.fractions[self.ELEMENTS.index("Cu")], 0.8)


class CommandLineTests(unittest.TestCase):
    """Tests for the command line interface."""

    def test_no_input_is_an_error(self):
        self.assertEqual(main([]), 1)

    def test_both_inputs_is_an_error(self):
        self.assertEqual(
            main(["--dictionary", str(DICTIONARY), "--workspace", "x.nxs"]), 1
        )

    def test_no_elements_is_an_error(self):
        self.assertEqual(main(["--dictionary", str(DICTIONARY)]), 1)

    def test_discover_cannot_be_combined_with_elements(self):
        """The candidates and an explicit list cannot both be given."""
        self.assertEqual(
            main(
                [
                    "--dictionary",
                    str(DICTIONARY),
                    "--discover",
                    "--elements",
                    "Cu",
                ]
            ),
            1,
        )

    def test_discover_and_discover_isotopes_cannot_both_be_given(self):
        """The element and isotope routes are separate."""
        self.assertEqual(
            main(
                [
                    "--dictionary",
                    str(DICTIONARY),
                    "--discover",
                    "--discover-isotopes",
                ]
            ),
            1,
        )

    def test_a_missing_dictionary_is_an_error(self):
        self.assertEqual(
            main(["--dictionary", "does_not_exist.npz", "--elements", "Cu"]), 1
        )

    @unittest.skipIf(
        not DICTIONARY.exists(),
        "The dictionary of measured regions is not available.",
    )
    def test_the_dictionary_route_fits_the_regions(self):
        """The dictionary is one way of giving the fit something to fit."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "--dictionary",
                    str(DICTIONARY),
                    "--elements",
                    "Ag",
                    "Au",
                    "In",
                    "Zn",
                    "Cu",
                    "Nb",
                    "Fe",
                    "--exclude-background",
                ]
            )
        self.assertEqual(code, 0)
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertIn("dominant", row)
            self.assertIn("rms_residual", row)


if __name__ == "__main__":
    unittest.main()
