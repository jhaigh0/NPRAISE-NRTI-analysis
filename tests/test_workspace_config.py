import tempfile
import unittest
from pathlib import Path

from data_loading.workspace_config import (
    ConfigError,
    load_config,
)


class LoadConfigTests(unittest.TestCase):
    """Tests for parsing the YAML training data configuration."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    def _write(self, content: str) -> Path:
        """Write a config file into the temporary directory and return its path."""
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(content, encoding="utf-8")
        return config_path

    def test_loads_workspaces_and_defaults(self):
        config = load_config(
            self._write(
                """
defaults:
  l2: 0.63
  noise_level: 0.2
  n_sets: 5
  n_samples: [10, 20]

workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: region
        selection: sel.nxs
        isotopes: [Sn-118]
        label: 1
""",
            )
        )

        self.assertEqual(len(config.workspaces), 1)
        entry = config.workspaces[0]
        self.assertEqual(entry.name, "sample")

        self.assertAlmostEqual(config.l2, 0.63)
        self.assertAlmostEqual(config.noise_level, 0.2)
        self.assertEqual(config.n_sets, 5)
        self.assertEqual(config.n_samples, [10, 20])

    def test_loads_multiple_selections_per_workspace(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: tin
        selection: sel_tin.nxs
        isotopes: [Sn-118, Sn-120]
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
""",
            )
        )
        entry = config.workspaces[0]
        self.assertEqual(len(entry.selections), 2)

        first, second = entry.selections
        self.assertEqual(first.name, "tin")
        self.assertEqual(first.isotopes, ["Sn-118", "Sn-120"])
        self.assertEqual(first.selection.name, "sel_tin.nxs")
        self.assertEqual(second.name, "silver")
        self.assertEqual(second.isotopes, ["Ag-107", "Ag-109"])

    def test_background_only_selection_has_no_isotopes(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: background
        selection: sel_background.nxs
        isotopes: []
""",
            )
        )
        selection = config.workspaces[0].selections[0]
        self.assertEqual(selection.isotopes, [])
        self.assertTrue(selection.is_background_only)

    def test_omitted_isotopes_is_treated_as_background_only(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: background
        selection: sel_background.nxs
""",
            )
        )
        selection = config.workspaces[0].selections[0]
        self.assertEqual(selection.isotopes, [])
        self.assertTrue(selection.is_background_only)

    def test_selection_label_inherits_workspace_label(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    label: 7
    selections:
      - name: a
        selection: sel_a.nxs
      - name: b
        selection: sel_b.nxs
        label: 9
""",
            )
        )
        first, second = config.workspaces[0].selections
        self.assertEqual(first.label, 7)
        self.assertEqual(second.label, 9)

    def test_explicit_null_label_overrides_workspace_label(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    label: 7
    selections:
      - name: a
        selection: sel_a.nxs
        label: null
""",
            )
        )
        self.assertIsNone(config.workspaces[0].selections[0].label)

    def test_single_selection_form_is_supported(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selection: sel.nxs
    isotopes: [Sn-118]
""",
            )
        )
        entry = config.workspaces[0]
        self.assertEqual(len(entry.selections), 1)
        self.assertEqual(entry.selections[0].isotopes, ["Sn-118"])
        self.assertEqual(entry.selections[0].name, "sel")

    def test_both_selection_forms_raise(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selection: sel.nxs
    selections:
      - selection: sel2.nxs
""",
                )
            )

    def test_resolves_paths_relative_to_config(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: data/full.nxs
    selections:
      - name: region
        selection: data/sel.nxs
""",
            )
        )
        entry = config.workspaces[0]
        self.assertTrue(entry.workspace.is_absolute())
        self.assertTrue(entry.selections[0].selection.is_absolute())
        self.assertEqual(entry.workspace.name, "full.nxs")
        self.assertEqual(entry.workspace.parent.name, "data")
        self.assertEqual(entry.selections[0].selection.parent.name, "data")

    def test_name_defaults_to_workspace_stem(self):
        config = load_config(
            self._write(
                """
workspaces:
  - workspace: my_full_workspace.nxs
    selections:
      - selection: sel.nxs
""",
            )
        )
        self.assertEqual(config.workspaces[0].name, "my_full_workspace")

    def test_selection_name_defaults_to_file_stem(self):
        config = load_config(
            self._write(
                """
workspaces:
  - workspace: full.nxs
    selections:
      - selection: my_region.nxs
""",
            )
        )
        self.assertEqual(config.workspaces[0].selections[0].name, "my_region")

    def test_isotopes_accepts_a_single_string(self):
        config = load_config(
            self._write(
                """
workspaces:
  - workspace: full.nxs
    selections:
      - selection: sel.nxs
        isotopes: Sn-118
""",
            )
        )
        self.assertEqual(config.workspaces[0].selections[0].isotopes, ["Sn-118"])

    def test_per_entry_l2_override_is_rejected(self):
        # The sample-detector distance is a global setting, because a mixed
        # region may take parts from several workspaces.
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
defaults:
  l2: 0.6
workspaces:
  - workspace: full.nxs
    selections:
      - selection: sel.nxs
    l2: 0.65
""",
                )
            )

    def test_per_entry_l2_values_override_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
defaults:
  l2_values: [0.62, 0.63]
workspaces:
  - workspace: full.nxs
    selections:
      - selection: sel.nxs
    l2_values: [0.64, 0.65]
""",
                )
            )

    def test_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            load_config(self.tmp_path / "does_not_exist.yaml")

    def test_empty_file_raises(self):
        with self.assertRaises(ConfigError):
            load_config(self._write(""))

    def test_missing_workspaces_raises(self):
        with self.assertRaises(ConfigError):
            load_config(self._write("defaults:\n  l2: 0.6\n"))

    def test_missing_workspace_key_raises(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    "workspaces:\n  - selections:\n      - selection: sel.nxs\n",
                )
            )

    def test_missing_selection_key_raises(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    "workspaces:\n  - workspace: full.nxs\n",
                )
            )

    def test_empty_selections_list_raises(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - workspace: full.nxs
    selections: []
""",
                )
            )

    def test_selection_without_file_raises(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - workspace: full.nxs
    selections:
      - name: region
        isotopes: [Sn-118]
""",
                )
            )

    def test_duplicate_names_raise(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: same
    workspace: a.nxs
    selections:
      - selection: sel.nxs
  - name: same
    workspace: b.nxs
    selections:
      - selection: sel.nxs
""",
                )
            )

    def test_loads_a_mixed_region(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: gold
        selection: sel_gold.nxs
        isotopes: [Au-197]
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
      - name: alloy
        mix:
          - region: gold
            fraction: 0.7
          - region: silver
            fraction: 0.3
""",
            )
        )

        selections = config.workspaces[0].selections
        alloy = selections[2]
        self.assertTrue(alloy.is_mixed)
        self.assertIsNone(alloy.selection)
        self.assertEqual(len(alloy.mix), 2)
        self.assertAlmostEqual(alloy.mix[0].fraction, 0.7)
        self.assertAlmostEqual(alloy.mix[1].fraction, 0.3)
        self.assertEqual(alloy.mix[0].name, "gold")
        # The parts resolve to the selection files of the regions they refer to.
        self.assertEqual(alloy.mix[0].selection, selections[0].selection)
        self.assertEqual(alloy.mix[1].selection, selections[1].selection)

    def test_mixed_region_derives_its_isotopes(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: gold
        selection: sel_gold.nxs
        isotopes: [Au-197]
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
      - name: alloy
        mix:
          - region: gold
            fraction: 0.7
          - region: silver
            fraction: 0.3
""",
            )
        )

        alloy = config.workspaces[0].selections[2]
        self.assertEqual(alloy.isotopes, ["Au-197", "Ag-107", "Ag-109"])

    def test_mixed_region_keeps_explicit_isotopes(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: gold
        selection: sel_gold.nxs
        isotopes: [Au-197]
      - name: alloy
        isotopes: [Au-197, Pd-108]
        mix:
          - region: gold
            fraction: 1.0
          - selection: sel_other.nxs
            fraction: 1.0
""",
            )
        )

        alloy = config.workspaces[0].selections[1]
        self.assertEqual(alloy.isotopes, ["Au-197", "Pd-108"])

    def test_mixed_region_needs_at_least_two_parts(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: alloy
        mix:
          - selection: sel.nxs
            fraction: 1.0
""",
                )
            )

    def test_mixed_region_cannot_have_a_selection_as_well(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: alloy
        selection: sel.nxs
        mix:
          - selection: a.nxs
          - selection: b.nxs
""",
                )
            )

    def test_mixed_region_rejects_an_unknown_region(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: alloy
        mix:
          - region: missing
            fraction: 1.0
          - selection: other.nxs
            fraction: 1.0
""",
                )
            )

    def test_mixed_region_rejects_a_non_positive_fraction(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: alloy
        mix:
          - selection: a.nxs
            fraction: 0.0
          - selection: b.nxs
            fraction: 1.0
""",
                )
            )

    def test_mixed_region_can_span_two_workspaces(self):
        config = load_config(
            self._write(
                """
workspaces:
  - name: first
    workspace: full_a.nxs
    selections:
      - name: gold
        selection: sel_gold.nxs
        isotopes: [Au-197]
      - name: alloy
        mix:
          - region: gold
            fraction: 0.5
          - workspace: second
            region: silver
            fraction: 0.5
  - name: second
    workspace: full_b.nxs
    selections:
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
""",
            )
        )

        first = config.workspaces[0]
        second = config.workspaces[1]
        alloy = first.selections[1]

        self.assertTrue(alloy.is_mixed)
        self.assertEqual(len(alloy.mix), 2)
        # The part that names another workspace resolves to that workspace.
        self.assertEqual(alloy.mix[0].workspace, "first")
        self.assertEqual(alloy.mix[1].workspace, "second")
        self.assertEqual(alloy.mix[0].selection, first.selections[0].selection)
        self.assertEqual(alloy.mix[1].selection, second.selections[0].selection)
        # The isotopes come from the parts of both workspaces.
        self.assertEqual(alloy.isotopes, ["Au-197", "Ag-107", "Ag-109"])

    def test_mixed_region_rejects_an_unknown_workspace(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: alloy
        mix:
          - region: missing
            fraction: 0.5
          - workspace: nowhere
            region: other
            fraction: 0.5
""",
                )
            )

    def test_mixed_region_rejects_an_unknown_region_of_another_workspace(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: full.nxs
    selections:
      - name: alloy
        mix:
          - selection: a.nxs
            fraction: 0.5
          - workspace: other
            region: missing
            fraction: 0.5
  - name: other
    workspace: full_b.nxs
    selections:
      - name: present
        selection: sel.nxs
""",
                )
            )

    def test_duplicate_selection_names_raise(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
workspaces:
  - name: sample
    workspace: a.nxs
    selections:
      - name: region
        selection: sel_a.nxs
      - name: region
        selection: sel_b.nxs
""",
                )
            )

    def test_negative_n_samples_raises(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
defaults:
  n_samples: [-5]
workspaces:
  - workspace: full.nxs
    selections:
      - selection: sel.nxs
""",
                )
            )

    def test_non_numeric_l2_raises(self):
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    """
defaults:
  l2: "not a number"
workspaces:
  - workspace: full.nxs
    selections:
      - selection: sel.nxs
""",
                )
            )


if __name__ == "__main__":
    unittest.main()
