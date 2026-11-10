"""Tests for the sample metadata and the metadata-driven fit modes."""

import unittest
from pathlib import Path

import numpy as np

from data_manipulation.fit_endf import (
    _simplex_from_stick,
    _stick_from_simplex,
    fit_region,
)
from data_manipulation.sample_metadata import (
    ELEMENT_DENSITIES,
    SampleMetadata,
    atomic_mass,
    load_metadata,
    load_metadata_yaml,
)


class AtomicMassTests(unittest.TestCase):
    """Tests for the atomic masses and the element densities."""

    def test_atomic_mass_of_a_known_element(self):
        self.assertAlmostEqual(atomic_mass("Cu"), 63.546, places=3)
        self.assertAlmostEqual(atomic_mass("Co"), 58.933, places=3)

    def test_unknown_element_raises(self):
        with self.assertRaises(ValueError):
            atomic_mass("Xx")

    def test_the_densities_are_physical(self):
        for symbol, density in ELEMENT_DENSITIES.items():
            self.assertGreater(density, 0.0, symbol)
        # Gold and lead are the dense ones, hydrogen is the light one.
        self.assertGreater(ELEMENT_DENSITIES["Au"], 19.0)
        self.assertLess(ELEMENT_DENSITIES["H"], 0.001)


class SampleMetadataTests(unittest.TestCase):
    """Tests for the metadata of one sample."""

    def test_the_composition_is_normalised(self):
        metadata = SampleMetadata(
            name="test", thickness_cm=0.1, composition={"Cu": 70.0, "Zn": 30.0}
        )
        self.assertAlmostEqual(sum(metadata.composition.values()), 1.0)
        self.assertAlmostEqual(metadata.composition["Cu"], 0.7)

    def test_a_negative_thickness_raises(self):
        with self.assertRaises(ValueError):
            SampleMetadata(name="test", thickness_cm=-0.1)

    def test_the_number_fractions_use_the_atomic_masses(self):
        metadata = SampleMetadata(
            name="test", thickness_cm=0.1, composition={"Cu": 70.0, "Zn": 30.0}
        )
        numbers = metadata.number_fractions
        self.assertAlmostEqual(sum(numbers.values()), 1.0)
        # Copper is lighter than zinc, so it has more atoms.
        self.assertGreater(numbers["Cu"], numbers["Zn"])

    def test_the_mixture_density_is_the_harmonic_mean(self):
        metadata = SampleMetadata(
            name="test", thickness_cm=0.1, composition={"Cu": 50.0, "Zn": 50.0}
        )
        expected = 2.0 / (1.0 / ELEMENT_DENSITIES["Cu"] + 1.0 / ELEMENT_DENSITIES["Zn"])
        self.assertAlmostEqual(metadata.mixture_density, expected)

    def test_the_mass_areal_density_follows_the_thickness(self):
        metadata = SampleMetadata(
            name="test", thickness_cm=0.25, composition={"Cu": 100.0}
        )
        self.assertAlmostEqual(
            metadata.mass_areal_density,
            ELEMENT_DENSITIES["Cu"] * 0.25,
        )

    def test_the_mass_areal_density_can_come_from_the_mass_and_area(self):
        metadata = SampleMetadata(
            name="test", mass_g=1.0, area_cm2=0.25, composition={"Cu": 100.0}
        )
        self.assertAlmostEqual(metadata.mass_areal_density, 4.0)

    def test_the_total_areal_density_is_atoms_per_barn(self):
        metadata = SampleMetadata(
            name="test", thickness_cm=0.25, composition={"Cu": 100.0}
        )
        expected = (
            ELEMENT_DENSITIES["Cu"] * 0.25 / atomic_mass("Cu") * 6.02214076e23
        ) / 1e24
        self.assertAlmostEqual(metadata.total_areal_density, expected)

    def test_the_element_areal_densities_add_up(self):
        metadata = SampleMetadata(
            name="test", thickness_cm=0.25, composition={"Cu": 70.0, "Zn": 30.0}
        )
        densities = metadata.element_areal_densities()
        self.assertAlmostEqual(sum(densities.values()), metadata.total_areal_density)

    def test_an_unknown_composition_has_no_areal_density(self):
        metadata = SampleMetadata(name="test", thickness_cm=0.25)
        self.assertIsNone(metadata.total_areal_density)
        self.assertEqual(metadata.element_areal_densities(), {})


class MetadataFileTests(unittest.TestCase):
    """Tests for reading the metadata from a file."""

    def test_reads_a_yaml_file(self):
        import tempfile

        text = (
            "Havar/Havar_5foils:\n"
            "  thickness_mm: 2.5\n"
            "  composition:\n"
            "    Co: 42.5\n"
            "    Cr: 20.0\n"
            "  notes: a cobalt alloy\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.yaml"
            path.write_text(text, encoding="utf-8")
            metadata = load_metadata_yaml(path)

        self.assertIn("Havar/Havar_5foils", metadata)
        hav = metadata["Havar/Havar_5foils"]
        self.assertAlmostEqual(hav.thickness_cm, 0.25)
        self.assertAlmostEqual(hav.composition["Co"], 42.5 / 62.5)
        self.assertEqual(hav.notes, "a cobalt alloy")

    def test_load_metadata_picks_the_reader_from_the_extension(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.yaml"
            path.write_text("a:\n  thickness_mm: 1.0\n", encoding="utf-8")
            metadata = load_metadata(path)
        self.assertIn("a", metadata)

    def test_a_yaml_file_that_is_not_a_mapping_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_metadata_yaml(path)


class SimplexTests(unittest.TestCase):
    """Tests for the stick-breaking parameters."""

    def test_the_weights_sum_to_one(self):
        weights = _simplex_from_stick(np.array([0.3, 0.5, 0.2]))
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(weights[0], 0.3)
        self.assertAlmostEqual(weights[1], 0.35)

    def test_the_weights_are_positive(self):
        weights = _simplex_from_stick(np.array([0.0, 0.0, 0.0]))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(weights[-1], 1.0)

    def test_the_transform_is_invertible(self):
        weights = np.array([0.2, 0.3, 0.5])
        recovered = _simplex_from_stick(_stick_from_simplex(weights))
        self.assertTrue(np.allclose(weights, recovered))


class FixedAmountFitTests(unittest.TestCase):
    """Tests for fitting the relative abundance with a known amount."""

    def _templates(self):
        energies = np.geomspace(1.0, 500.0, 400)
        sigma_a = np.exp(-((energies - 100.0) ** 2) / (2 * 2.0**2))
        sigma_b = np.exp(-((energies - 250.0) ** 2) / (2 * 3.0**2))

        def make(symbol, sigma):
            return type(
                "T",
                (),
                {
                    "symbol": symbol,
                    "isotopes": [1],
                    "abundances": {1: 1.0},
                    "energies": energies,
                    "cross_section": sigma,
                    "resonance_energies": np.array([100.0]),
                },
            )()

        from endf_database_util.templates import TemplateSet

        return TemplateSet(
            templates={"Cu": make("Cu", sigma_a), "Zn": make("Zn", sigma_b)},
            energies=energies,
        )

    def _metadata(self):
        return SampleMetadata(
            name="test",
            thickness_cm=0.1,
            composition={"Cu": 70.0, "Zn": 30.0},
        )

    def _profile(self, templates, metadata, number_fractions):
        """Build a profile with a known composition."""
        total = metadata.total_areal_density
        shape = (
            number_fractions["Cu"] * templates["Cu"].cross_section
            + number_fractions["Zn"] * templates["Zn"].cross_section
        )
        return np.exp(-total * shape)

    def test_the_relative_abundance_is_recovered(self):
        templates = self._templates()
        metadata = self._metadata()
        fractions = {"Cu": 0.4, "Zn": 0.6}
        profile = self._profile(templates, metadata, fractions)

        result = fit_region(
            profile, ["Cu", "Zn"], templates=templates, fixed_amount=metadata
        )
        self.assertAlmostEqual(result.fractions[0], 0.4, places=4)
        self.assertAlmostEqual(result.fractions[1], 0.6, places=4)

    def test_the_mass_fraction_is_reported(self):
        templates = self._templates()
        metadata = self._metadata()
        profile = self._profile(templates, metadata, {"Cu": 0.4, "Zn": 0.6})

        result = fit_region(
            profile, ["Cu", "Zn"], templates=templates, fixed_amount=metadata
        )
        self.assertIsNotNone(result.mass_fractions)
        # The number fractions are 0.4 and 0.6, but copper is the lighter
        # of the two, so its mass fraction is smaller.
        masses = atomic_mass("Cu") * 0.4, atomic_mass("Zn") * 0.6
        self.assertAlmostEqual(
            result.mass_fractions[0], masses[0] / (masses[0] + masses[1]), places=4
        )

    def test_the_total_areal_density_comes_from_the_metadata(self):
        templates = self._templates()
        metadata = self._metadata()
        profile = self._profile(templates, metadata, {"Cu": 0.4, "Zn": 0.6})

        result = fit_region(
            profile, ["Cu", "Zn"], templates=templates, fixed_amount=metadata
        )
        self.assertAlmostEqual(result.total_areal_density, metadata.total_areal_density)

    def test_a_fixed_composition_and_a_fixed_amount_together_raise(self):
        templates = self._templates()
        metadata = self._metadata()
        profile = self._profile(templates, metadata, {"Cu": 0.4, "Zn": 0.6})
        with self.assertRaises(ValueError):
            fit_region(
                profile,
                ["Cu", "Zn"],
                templates=templates,
                fixed_amount=metadata,
                fixed_composition={"Cu": 70.0, "Zn": 30.0},
            )

    def test_metadata_without_a_thickness_raises(self):
        templates = self._templates()
        metadata = SampleMetadata(name="test", composition={"Cu": 100.0})
        profile = np.ones(len(templates.energies))
        with self.assertRaises(ValueError):
            fit_region(
                profile, ["Cu", "Zn"], templates=templates, fixed_amount=metadata
            )

    def test_the_fractions_are_the_number_fractions(self):
        """The fractions are shares of the optical thickness, not of the mass."""
        templates = self._templates()
        metadata = self._metadata()
        profile = self._profile(templates, metadata, {"Cu": 0.4, "Zn": 0.6})

        result = fit_region(
            profile, ["Cu", "Zn"], templates=templates, fixed_amount=metadata
        )
        self.assertAlmostEqual(result.fractions[0], 0.4, places=4)
        self.assertNotAlmostEqual(
            result.fractions[0], float(result.mass_fractions[0]), places=2
        )


if __name__ == "__main__":
    unittest.main()
