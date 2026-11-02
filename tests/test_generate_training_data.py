"""End to end tests for the training data generation script.

The test workspaces are built programmatically with the NGEM instrument
definition so that no large binary fixtures need to be stored in the repo.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from mantid.simpleapi import (
    CreateWorkspace,
    ExtractSpectra,
    LoadInstrument,
    SaveNexus,
)

import generate_training_data as gtd

N_PIXELS = 16
N_BINS = 40
# The generated data is rebinned onto the fixed energy grid, so the number of
# output bins does not depend on the number of bins of the input workspace.
N_OUTPUT_BINS = 1246
SELECTION_DETECTOR_IDS = list(range(100, 140))
SECOND_SELECTION_DETECTOR_IDS = list(range(140, 180))
BACKGROUND_DETECTOR_IDS = list(range(180, 220))


def build_full_workspace() -> "object":
    """Build a small NGEM workspace with Poisson counts."""
    edges = np.arange(100.0, 100.0 + (N_BINS + 1) * 10.0, 10.0)
    rng = np.random.default_rng(0)
    y = rng.poisson(10.0, size=(N_PIXELS * N_PIXELS, N_BINS)).astype(float)

    ws = CreateWorkspace(DataX=edges, DataY=y, NSpec=N_PIXELS * N_PIXELS, UnitX="TOF")
    LoadInstrument(ws, InstrumentName="NGEM", RewriteSpectraMap=True)
    return ws


def save_selection(full_ws, det_ids: list[int], path: Path) -> Path:
    """Extract a region of the detector and save it as a selection workspace."""
    selection_ws = ExtractSpectra(
        full_ws,
        DetectorList=",".join(map(str, det_ids)),
        StoreInADS=False,
    )
    SaveNexus(selection_ws, Filename=str(path))
    return path


class TrainingDataGenerationTests(unittest.TestCase):
    """Tests that run the full script against generated workspaces."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

        full_ws = build_full_workspace()
        self.full_path = self.tmp_path / "full_workspace.nxs"
        SaveNexus(full_ws, Filename=str(self.full_path))

        self.selection_path = save_selection(
            full_ws, SELECTION_DETECTOR_IDS, self.tmp_path / "sel_tin.nxs"
        )
        self.second_selection_path = save_selection(
            full_ws, SECOND_SELECTION_DETECTOR_IDS, self.tmp_path / "sel_silver.nxs"
        )
        self.background_path = save_selection(
            full_ws, BACKGROUND_DETECTOR_IDS, self.tmp_path / "sel_background.nxs"
        )

    def _write_config(self, **overrides) -> Path:
        """Write a config file pointing at the generated workspaces."""
        settings = {
            "l2": 0.6,
            "noise_level": 0.1,
            "n_sets": 3,
            "n_samples": "[5, 10]",
        }
        settings.update(overrides)

        config = f"""
defaults:
  l2: {settings["l2"]}
  noise_level: {settings["noise_level"]}
  n_sets: {settings["n_sets"]}
  n_samples: {settings["n_samples"]}

workspaces:
  - name: test_workspace
    workspace: full_workspace.nxs
    selections:
      - name: tin
        selection: sel_tin.nxs
        isotopes: [Sn-118, Sn-120]
        label: 0
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
        label: 1
      - name: background
        selection: sel_background.nxs
        isotopes: []
        label: 2
"""
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(config, encoding="utf-8")
        return config_path

    def _run(self, *extra_args) -> tuple[int, str]:
        """Run the script, returning the exit code and captured output."""
        output_dir = self.tmp_path / "out"
        argv = [
            "--config",
            str(self._write_config()),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
            *extra_args,
        ]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = gtd.main(argv)
        return code, buffer.getvalue()

    def test_generates_one_file_per_selection_and_n_samples(self):
        code, _ = self._run()
        self.assertEqual(code, 0)

        output_dir = self.tmp_path / "out"
        files = sorted(p.name for p in output_dir.iterdir())
        self.assertEqual(
            files,
            [
                "test_workspace_background_n10_noise0.1_l20.6.npz",
                "test_workspace_background_n5_noise0.1_l20.6.npz",
                "test_workspace_silver_n10_noise0.1_l20.6.npz",
                "test_workspace_silver_n5_noise0.1_l20.6.npz",
                "test_workspace_tin_n10_noise0.1_l20.6.npz",
                "test_workspace_tin_n5_noise0.1_l20.6.npz",
            ],
        )

    def test_output_contains_spectra_and_metadata(self):
        self._run()
        output_path = self.tmp_path / "out" / "test_workspace_tin_n5_noise0.1_l20.6.npz"

        with np.load(output_path) as data:
            self.assertEqual(data["spectra_samples"].shape, (3, N_OUTPUT_BINS))
            self.assertEqual(data["x_bins"].shape, (N_OUTPUT_BINS + 1,))
            self.assertEqual(str(data["name"]), "test_workspace")
            self.assertEqual(str(data["selection_name"]), "tin")
            self.assertEqual(list(data["isotopes"]), ["Sn-118", "Sn-120"])
            self.assertFalse(bool(data["is_background_only"]))
            self.assertFalse(bool(data["is_mixed"]))
            self.assertEqual(int(data["label"]), 0)
            self.assertFalse(bool(data["label_is_null"]))
            self.assertEqual(int(data["n_samples"]), 5)
            self.assertEqual(int(data["n_sets"]), 3)
            self.assertAlmostEqual(float(data["noise_level"]), 0.1)
            np.testing.assert_allclose(data["l2_values"], [0.6])
            self.assertEqual(data["spectra_l2"].shape, (3,))

    def test_generates_a_mixed_region(self):
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(
            """
defaults:
  l2: 0.6
  noise_level: 0.1
  n_sets: 3
  n_samples: [5]

workspaces:
  - name: test_workspace
    workspace: full_workspace.nxs
    selections:
      - name: tin
        selection: sel_tin.nxs
        isotopes: [Sn-118]
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
      - name: alloy
        mix:
          - region: tin
            fraction: 0.7
          - region: silver
            fraction: 0.3
""",
            encoding="utf-8",
        )
        output_dir = self.tmp_path / "out"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = gtd.main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "42",
                ]
            )
        self.assertEqual(code, 0)

        output_path = output_dir / "test_workspace_alloy_n5_noise0.1_l20.6.npz"
        self.assertTrue(output_path.exists())

        with np.load(output_path) as data:
            self.assertEqual(data["spectra_samples"].shape, (3, N_OUTPUT_BINS))
            self.assertTrue(bool(data["is_mixed"]))
            self.assertEqual(list(data["mix_names"]), ["tin", "silver"])
            np.testing.assert_allclose(data["mix_fractions"], [0.7, 0.3])
            # The isotopes of a mixed region are those of its parts.
            self.assertEqual(list(data["isotopes"]), ["Sn-118", "Ag-107", "Ag-109"])

    def test_a_mixed_region_differs_from_its_parts(self):
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(
            """
defaults:
  l2: 0.6
  noise_level: 0.0
  n_sets: 3
  n_samples: [20]

workspaces:
  - name: test_workspace
    workspace: full_workspace.nxs
    selections:
      - name: tin
        selection: sel_tin.nxs
        isotopes: [Sn-118]
      - name: silver
        selection: sel_silver.nxs
        isotopes: [Ag-107]
      - name: alloy
        mix:
          - region: tin
            fraction: 0.5
          - region: silver
            fraction: 0.5
""",
            encoding="utf-8",
        )
        output_dir = self.tmp_path / "out"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = gtd.main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "42",
                ]
            )
        self.assertEqual(code, 0)

        with np.load(output_dir / "test_workspace_alloy_n20_noise0_l20.6.npz") as d:
            alloy = d["spectra_samples"].copy()
        with np.load(output_dir / "test_workspace_tin_n20_noise0_l20.6.npz") as d:
            tin = d["spectra_samples"].copy()

        self.assertFalse(np.array_equal(alloy, tin))

    def test_generates_a_region_that_mixes_two_workspaces(self):
        # A second, independent workspace to mix with.
        other_ws = build_full_workspace()
        other_full_path = self.tmp_path / "other_workspace.nxs"
        SaveNexus(other_ws, Filename=str(other_full_path))
        save_selection(
            other_ws, SECOND_SELECTION_DETECTOR_IDS, self.tmp_path / "other_sel.nxs"
        )

        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(
            """
defaults:
  l2: 0.6
  noise_level: 0.0
  n_sets: 3
  n_samples: [20]

workspaces:
  - name: first
    workspace: full_workspace.nxs
    selections:
      - name: tin
        selection: sel_tin.nxs
        isotopes: [Sn-118]
      - name: alloy
        mix:
          - region: tin
            fraction: 0.5
          - workspace: second
            region: silver
            fraction: 0.5
  - name: second
    workspace: other_workspace.nxs
    selections:
      - name: silver
        selection: other_sel.nxs
        isotopes: [Ag-107, Ag-109]
""",
            encoding="utf-8",
        )
        output_dir = self.tmp_path / "out"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = gtd.main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "42",
                ]
            )
        self.assertEqual(code, 0)

        output_path = output_dir / "first_alloy_n20_noise0_l20.6.npz"
        self.assertTrue(output_path.exists())

        with np.load(output_path) as data:
            self.assertEqual(data["spectra_samples"].shape, (3, N_OUTPUT_BINS))
            self.assertTrue(bool(data["is_mixed"]))
            self.assertEqual(list(data["mix_names"]), ["tin", "silver"])
            self.assertEqual(list(data["mix_workspaces"]), ["first", "second"])
            np.testing.assert_allclose(data["mix_fractions"], [0.5, 0.5])
            # The isotopes come from the parts of both workspaces.
            self.assertEqual(list(data["isotopes"]), ["Sn-118", "Ag-107", "Ag-109"])

    def test_a_mixture_across_workspaces_is_a_weighted_combination(self):
        other_ws = build_full_workspace()
        other_full_path = self.tmp_path / "other_workspace.nxs"
        SaveNexus(other_ws, Filename=str(other_full_path))
        save_selection(
            other_ws, SECOND_SELECTION_DETECTOR_IDS, self.tmp_path / "other_sel.nxs"
        )

        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(
            """
defaults:
  l2: 0.6
  noise_level: 0.0
  n_sets: 4
  n_samples: [200]

workspaces:
  - name: first
    workspace: full_workspace.nxs
    selections:
      - name: tin
        selection: sel_tin.nxs
        isotopes: [Sn-118]
      - name: alloy
        mix:
          - region: tin
            fraction: 0.25
          - workspace: second
            region: silver
            fraction: 0.75
  - name: second
    workspace: other_workspace.nxs
    selections:
      - name: silver
        selection: other_sel.nxs
        isotopes: [Ag-107, Ag-109]
""",
            encoding="utf-8",
        )
        output_dir = self.tmp_path / "out"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = gtd.main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "42",
                ]
            )
        self.assertEqual(code, 0)

        with np.load(output_dir / "first_alloy_n200_noise0_l20.6.npz") as d:
            alloy = d["spectra_samples"].mean(axis=0)
        with np.load(output_dir / "first_tin_n200_noise0_l20.6.npz") as d:
            tin = d["spectra_samples"].mean(axis=0)
        with np.load(output_dir / "second_silver_n200_noise0_l20.6.npz") as d:
            silver = d["spectra_samples"].mean(axis=0)

        # The mixture is the weighted combination of its parts, up to the
        # counting noise of the sampled spectra.
        expected = 0.25 * tin + 0.75 * silver
        relative_error = np.abs(alloy - expected).sum() / expected.sum()
        self.assertLess(relative_error, 0.1)

    def test_background_selection_is_marked_as_such(self):
        self._run()
        output_path = (
            self.tmp_path / "out" / "test_workspace_background_n5_noise0.1_l20.6.npz"
        )

        with np.load(output_path) as data:
            self.assertEqual(str(data["selection_name"]), "background")
            self.assertEqual(list(data["isotopes"]), [])
            self.assertTrue(bool(data["is_background_only"]))
            self.assertEqual(int(data["label"]), 2)

    def test_selections_get_independent_samples(self):
        self._run()
        output_dir = self.tmp_path / "out"

        with np.load(output_dir / "test_workspace_tin_n5_noise0.1_l20.6.npz") as data:
            tin = data["spectra_samples"].copy()
        with np.load(
            output_dir / "test_workspace_silver_n5_noise0.1_l20.6.npz"
        ) as data:
            silver = data["spectra_samples"].copy()

        # The regions hold different detectors, so the samples must differ.
        self.assertFalse(np.array_equal(tin, silver))

    def test_command_line_overrides_config(self):
        code, _ = self._run("--noise", "0.5", "--n-sets", "2", "--n-samples", "7")
        self.assertEqual(code, 0)

        output_path = self.tmp_path / "out" / "test_workspace_tin_n7_noise0.5_l20.6.npz"
        self.assertTrue(output_path.exists())

        with np.load(output_path) as data:
            self.assertEqual(data["spectra_samples"].shape, (2, N_OUTPUT_BINS))
            self.assertEqual(int(data["n_samples"]), 7)
            self.assertEqual(int(data["n_sets"]), 2)
            self.assertAlmostEqual(float(data["noise_level"]), 0.5)

    def test_seed_makes_output_reproducible(self):
        self._run()
        first = self.tmp_path / "out" / "test_workspace_tin_n5_noise0.1_l20.6.npz"
        with np.load(first) as data:
            first_spectra = data["spectra_samples"].copy()

        # Remove the outputs and regenerate with the same seed.
        for path in (self.tmp_path / "out").iterdir():
            path.unlink()
        self._run()

        second = self.tmp_path / "out" / "test_workspace_tin_n5_noise0.1_l20.6.npz"
        with np.load(second) as data:
            np.testing.assert_array_equal(data["spectra_samples"], first_spectra)

    def test_existing_files_are_skipped(self):
        self._run()
        output_path = self.tmp_path / "out" / "test_workspace_tin_n5_noise0.1_l20.6.npz"
        first_modified = output_path.stat().st_mtime_ns

        code, output = self._run()
        self.assertEqual(code, 0)
        self.assertIn("skipping existing", output)
        self.assertEqual(output_path.stat().st_mtime_ns, first_modified)

    def test_overwrite_replaces_existing_files(self):
        self._run()
        output_path = self.tmp_path / "out" / "test_workspace_tin_n5_noise0.1_l20.6.npz"
        first_modified = output_path.stat().st_mtime_ns

        code, output = self._run("--overwrite")
        self.assertEqual(code, 0)
        self.assertNotIn("skipping existing", output)
        self.assertNotEqual(output_path.stat().st_mtime_ns, first_modified)

    def test_missing_workspace_file_reports_failure(self):
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(
            """
workspaces:
  - name: missing
    workspace: does_not_exist.nxs
    selections:
      - selection: sel_tin.nxs
""",
            encoding="utf-8",
        )
        code = gtd.main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(self.tmp_path / "out"),
                "--n-samples",
                "5",
            ]
        )
        self.assertEqual(code, 1)

    def test_no_n_samples_reports_error(self):
        config_path = self.tmp_path / "config.yaml"
        config_path.write_text(
            """
workspaces:
  - name: sample
    workspace: full_workspace.nxs
    selections:
      - selection: sel_tin.nxs
""",
            encoding="utf-8",
        )
        code = gtd.main(
            ["--config", str(config_path), "--output-dir", str(self.tmp_path / "out")]
        )
        self.assertEqual(code, 2)

    def test_invalid_config_reports_error(self):
        code = gtd.main(
            [
                "--config",
                str(self.tmp_path / "missing.yaml"),
                "--output-dir",
                str(self.tmp_path / "out"),
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
