import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from data_loading.export import export_data_to_np_format


class MockWorkspace:
    """Mock Mantid workspace for testing export_data_to_np_format."""

    def __init__(
        self, name, x_pixels, y_pixels, n_bins, detector_ids_flat, counts_2d, edges
    ):
        self._name = name
        self._x = x_pixels
        self._y = y_pixels
        self._n_bins = n_bins
        self._detector_ids_flat = detector_ids_flat
        self._counts_2d = counts_2d
        self._edges = edges

        self._instrument = MagicMock()
        component = MagicMock()
        component.xpixels.return_value = x_pixels
        component.ypixels.return_value = y_pixels
        self._instrument.getComponentByName.return_value = component

    def getNumberBins(self):
        return self._n_bins

    def getInstrument(self):
        return self._instrument

    def getDetectorIDToWorkspaceIndexMap(self, *args, **kwargs):
        return {det_id: idx for idx, det_id in enumerate(self._detector_ids_flat)}

    def extractY(self):
        return self._counts_2d

    def readX(self, idx):
        return self._edges

    def name(self):
        return self._name


def make_test_workspace(name: str = "test_ws") -> MockWorkspace:
    """Create a standard test workspace with a 3x2 detector grid and 5 bins.

    Detector layout (after reshape to (3, 2)):
        [[101, 102],
         [103, 104],
         [105, 106]]
    """
    x_pixels = 3
    y_pixels = 2
    n_bins = 5
    detector_ids_flat = [101, 102, 103, 104, 105, 106]
    counts_2d = np.array(
        [
            [1010, 1011, 1012, 1013, 1014],
            [1020, 1021, 1022, 1023, 1024],
            [1030, 1031, 1032, 1033, 1034],
            [1040, 1041, 1042, 1043, 1044],
            [1050, 1051, 1052, 1053, 1054],
            [1060, 1061, 1062, 1063, 1064],
        ]
    )
    edges = np.linspace(0, 1, n_bins + 1)
    return MockWorkspace(
        name, x_pixels, y_pixels, n_bins, detector_ids_flat, counts_2d, edges
    )


class ExportDataTests(unittest.TestCase):
    def test_export_without_roi(self):
        """Export the full workspace without any ROI selection."""
        ws = make_test_workspace("test_ws_1")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir)
            npz_path = Path(tmpdir) / "test_ws_1.npz"
            self.assertTrue(npz_path.exists())

            data = np.load(npz_path)
            self.assertIn("counts", data)
            self.assertIn("detector_ids", data)
            self.assertIn("edges", data)

            # Check shapes
            self.assertEqual(data["counts"].shape, (3, 2, 5))
            self.assertEqual(data["detector_ids"].shape, (3, 2))
            self.assertEqual(data["edges"].shape, (6,))

            # Check detector_ids
            expected_ids = np.array([[101, 102], [103, 104], [105, 106]])
            np.testing.assert_array_equal(data["detector_ids"], expected_ids)

            # Check counts
            expected_counts = np.array(
                [
                    [[1010, 1011, 1012, 1013, 1014], [1020, 1021, 1022, 1023, 1024]],
                    [[1030, 1031, 1032, 1033, 1034], [1040, 1041, 1042, 1043, 1044]],
                    [[1050, 1051, 1052, 1053, 1054], [1060, 1061, 1062, 1063, 1064]],
                ]
            )
            np.testing.assert_array_equal(data["counts"], expected_counts)

            # Check edges
            np.testing.assert_array_equal(data["edges"], np.linspace(0, 1, 6))

    def test_export_with_roi_top_row(self):
        """Export with ROI selecting the top row of detectors."""
        ws = make_test_workspace("test_ws_2")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[101, 102])
            npz_path = Path(tmpdir) / "test_ws_2.npz"
            data = np.load(npz_path)

            # ROI [101, 102] is at positions (0,0) and (0,1)
            # Bounding box: rows 0-0, cols 0-1
            self.assertEqual(data["counts"].shape, (1, 2, 5))
            self.assertEqual(data["detector_ids"].shape, (1, 2))
            np.testing.assert_array_equal(data["detector_ids"], np.array([[101, 102]]))
            np.testing.assert_array_equal(
                data["counts"],
                np.array(
                    [[[1010, 1011, 1012, 1013, 1014], [1020, 1021, 1022, 1023, 1024]]]
                ),
            )

    def test_export_with_roi_middle_row(self):
        """Export with ROI selecting the middle row of detectors."""
        ws = make_test_workspace("test_ws_3")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[103, 104])
            npz_path = Path(tmpdir) / "test_ws_3.npz"
            data = np.load(npz_path)

            # ROI [103, 104] is at positions (1,0) and (1,1)
            # Bounding box: rows 1-1, cols 0-1
            self.assertEqual(data["counts"].shape, (1, 2, 5))
            self.assertEqual(data["detector_ids"].shape, (1, 2))
            np.testing.assert_array_equal(data["detector_ids"], np.array([[103, 104]]))
            np.testing.assert_array_equal(
                data["counts"],
                np.array(
                    [[[1030, 1031, 1032, 1033, 1034], [1040, 1041, 1042, 1043, 1044]]]
                ),
            )

    def test_export_with_roi_2x2_block(self):
        """Export with ROI selecting a 2x2 block of detectors."""
        ws = make_test_workspace("test_ws_4")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[101, 102, 103, 104])
            npz_path = Path(tmpdir) / "test_ws_4.npz"
            data = np.load(npz_path)

            # ROI [101, 102, 103, 104] forms a 2x2 block at top-left
            self.assertEqual(data["counts"].shape, (2, 2, 5))
            self.assertEqual(data["detector_ids"].shape, (2, 2))
            np.testing.assert_array_equal(
                data["detector_ids"], np.array([[101, 102], [103, 104]])
            )
            np.testing.assert_array_equal(
                data["counts"],
                np.array(
                    [
                        [
                            [1010, 1011, 1012, 1013, 1014],
                            [1020, 1021, 1022, 1023, 1024],
                        ],
                        [
                            [1030, 1031, 1032, 1033, 1034],
                            [1040, 1041, 1042, 1043, 1044],
                        ],
                    ]
                ),
            )

    def test_export_with_roi_single_detector(self):
        """Export with ROI selecting a single detector."""
        ws = make_test_workspace("test_ws_5")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[101])
            npz_path = Path(tmpdir) / "test_ws_5.npz"
            data = np.load(npz_path)

            # Single detector at position (0,0)
            self.assertEqual(data["counts"].shape, (1, 1, 5))
            self.assertEqual(data["detector_ids"].shape, (1, 1))
            np.testing.assert_array_equal(data["detector_ids"], np.array([[101]]))
            np.testing.assert_array_equal(
                data["counts"], np.array([[[1010, 1011, 1012, 1013, 1014]]])
            )

    def test_export_with_roi_non_contiguous(self):
        """Export with ROI selecting non-contiguous detectors.

        The bounding box should include all detectors within the
        rectangular region spanned by the selected detectors, even
        if some of those detectors were not explicitly selected.
        """
        ws = make_test_workspace("test_ws_6")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[101, 104])
            npz_path = Path(tmpdir) / "test_ws_6.npz"
            data = np.load(npz_path)

            # det_ids [101, 104] are at positions (0,0) and (1,1)
            # Bounding box: rows 0-1, cols 0-1 (includes 102 and 103)
            self.assertEqual(data["counts"].shape, (2, 2, 5))
            self.assertEqual(data["detector_ids"].shape, (2, 2))
            np.testing.assert_array_equal(
                data["detector_ids"], np.array([[101, 102], [103, 104]])
            )
            np.testing.assert_array_equal(
                data["counts"],
                np.array(
                    [
                        [
                            [1010, 1011, 1012, 1013, 1014],
                            [1020, 1021, 1022, 1023, 1024],
                        ],
                        [
                            [1030, 1031, 1032, 1033, 1034],
                            [1040, 1041, 1042, 1043, 1044],
                        ],
                    ]
                ),
            )

    def test_export_creates_save_dir(self):
        """Export should create the save directory if it does not exist."""
        ws = make_test_workspace("test_ws_7")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "nested" / "subdir"
            export_data_to_np_format(ws, save_dir)
            self.assertTrue(save_dir.exists())
            npz_path = save_dir / "test_ws_7.npz"
            self.assertTrue(npz_path.exists())

    def test_export_filename_uses_workspace_name(self):
        """The output file should be named after the workspace."""
        ws = MockWorkspace(
            "my_custom_name",
            3,
            2,
            5,
            [101, 102, 103, 104, 105, 106],
            np.zeros((6, 5)),
            np.linspace(0, 1, 6),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir)
            npz_path = Path(tmpdir) / "my_custom_name.npz"
            self.assertTrue(npz_path.exists())

    def test_export_with_path_object(self):
        """Export should accept a Path object as save_dir."""
        ws = make_test_workspace("test_ws_8")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, Path(tmpdir))
            npz_path = Path(tmpdir) / "test_ws_8.npz"
            self.assertTrue(npz_path.exists())

    def test_export_with_roi_bottom_row(self):
        """Export with ROI selecting the bottom row of detectors."""
        ws = make_test_workspace("test_ws_9")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[105, 106])
            npz_path = Path(tmpdir) / "test_ws_9.npz"
            data = np.load(npz_path)

            # ROI [105, 106] is at positions (2,0) and (2,1)
            # Bounding box: rows 2-2, cols 0-1
            self.assertEqual(data["counts"].shape, (1, 2, 5))
            self.assertEqual(data["detector_ids"].shape, (1, 2))
            np.testing.assert_array_equal(data["detector_ids"], np.array([[105, 106]]))
            np.testing.assert_array_equal(
                data["counts"],
                np.array(
                    [[[1050, 1051, 1052, 1053, 1054], [1060, 1061, 1062, 1063, 1064]]]
                ),
            )

    def test_export_with_roi_single_column(self):
        """Export with ROI selecting a single column of detectors."""
        ws = make_test_workspace("test_ws_10")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_data_to_np_format(ws, tmpdir, det_ids=[101, 103, 105])
            npz_path = Path(tmpdir) / "test_ws_10.npz"
            data = np.load(npz_path)

            # det_ids [101, 103, 105] are at positions (0,0), (1,0), (2,0)
            # Bounding box: rows 0-2, cols 0-0
            self.assertEqual(data["counts"].shape, (3, 1, 5))
            self.assertEqual(data["detector_ids"].shape, (3, 1))
            np.testing.assert_array_equal(
                data["detector_ids"], np.array([[101], [103], [105]])
            )
            np.testing.assert_array_equal(
                data["counts"],
                np.array(
                    [
                        [[1010, 1011, 1012, 1013, 1014]],
                        [[1030, 1031, 1032, 1033, 1034]],
                        [[1050, 1051, 1052, 1053, 1054]],
                    ]
                ),
            )


if __name__ == "__main__":
    unittest.main()
