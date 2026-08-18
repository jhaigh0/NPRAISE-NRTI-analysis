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


def make_test_workspace() -> MockWorkspace:
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
        "test_ws", x_pixels, y_pixels, n_bins, detector_ids_flat, counts_2d, edges
    )


class ExportDataTests(unittest.TestCase):
    def test_export_without_roi(self):
        """Export the full workspace without any ROI selection."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path)
            self.assertTrue(npz_path.exists())

            with np.load(npz_path) as data:
                self.assertIn("counts", data)
                self.assertIn("detector_ids", data)
                self.assertIn("edges", data)

                # Check shapes
                self.assertEqual(data["counts"].shape, (2, 3, 5))
                self.assertEqual(data["detector_ids"].shape, (2, 3))
                self.assertEqual(data["edges"].shape, (6,))

                # Check detector_ids
                expected_ids = np.array([[101, 102, 103], [104, 105, 106]])
                np.testing.assert_array_equal(data["detector_ids"], expected_ids)

                # Check counts
                expected_counts = np.array(
                    [
                        [
                            [1010, 1011, 1012, 1013, 1014],
                            [1020, 1021, 1022, 1023, 1024],
                            [1030, 1031, 1032, 1033, 1034],
                        ],
                        [
                            [1040, 1041, 1042, 1043, 1044],
                            [1050, 1051, 1052, 1053, 1054],
                            [1060, 1061, 1062, 1063, 1064],
                        ],
                    ]
                )
                np.testing.assert_array_equal(data["counts"], expected_counts)

                # Check edges
                np.testing.assert_array_equal(data["edges"], np.linspace(0, 1, 6))

    def test_export_with_roi_top_row(self):
        """Export with ROI selecting the top row of detectors."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path, det_ids=[101, 102, 103])
            with np.load(npz_path) as data:
                # ROI [101, 102, 103] is at positions (0,0), (0,1), and (0,2)
                # Bounding box: rows 0-0, cols 0-2
                self.assertEqual(data["counts"].shape, (1, 3, 5))
                self.assertEqual(data["detector_ids"].shape, (1, 3))
                np.testing.assert_array_equal(
                    data["detector_ids"], np.array([[101, 102, 103]])
                )
                np.testing.assert_array_equal(
                    data["counts"],
                    np.array(
                        [
                            [
                                [1010, 1011, 1012, 1013, 1014],
                                [1020, 1021, 1022, 1023, 1024],
                                [1030, 1031, 1032, 1033, 1034],
                            ]
                        ]
                    ),
                )

    def test_export_with_roi_bottom_row(self):
        """Export with ROI selecting the bottom row of detectors."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path, det_ids=[104, 105, 106])
            with np.load(npz_path) as data:
                # ROI [104, 105, 106] is at positions (1,0), (1,1), and (1,2)
                # Bounding box: rows 1-1, cols 0-2
                self.assertEqual(data["counts"].shape, (1, 3, 5))
                self.assertEqual(data["detector_ids"].shape, (1, 3))
                np.testing.assert_array_equal(
                    data["detector_ids"], np.array([[104, 105, 106]])
                )
                np.testing.assert_array_equal(
                    data["counts"],
                    np.array(
                        [
                            [
                                [1040, 1041, 1042, 1043, 1044],
                                [1050, 1051, 1052, 1053, 1054],
                                [1060, 1061, 1062, 1063, 1064],
                            ]
                        ]
                    ),
                )

    def test_export_with_roi_2x2_block(self):
        """Export with ROI selecting a 2x2 block of detectors."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path, det_ids=[101, 102, 104, 105])
            with np.load(npz_path) as data:
                # ROI [101, 102, 104, 105] forms a 2x2 block at top-left
                self.assertEqual(data["counts"].shape, (2, 2, 5))
                self.assertEqual(data["detector_ids"].shape, (2, 2))
                np.testing.assert_array_equal(
                    data["detector_ids"], np.array([[101, 102], [104, 105]])
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
                                [1040, 1041, 1042, 1043, 1044],
                                [1050, 1051, 1052, 1053, 1054],
                            ],
                        ]
                    ),
                )

    def test_export_with_roi_single_detector(self):
        """Export with ROI selecting a single detector."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path, det_ids=[101])
            with np.load(npz_path) as data:
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
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path, det_ids=[101, 105])
            with np.load(npz_path) as data:
                # det_ids [101, 105] are at positions (0,0) and (1,1)
                # Bounding box: rows 0-1, cols 0-1 (includes 102 and 104)
                self.assertEqual(data["counts"].shape, (2, 2, 5))
                self.assertEqual(data["detector_ids"].shape, (2, 2))
                np.testing.assert_array_equal(
                    data["detector_ids"], np.array([[101, 102], [104, 105]])
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
                                [1040, 1041, 1042, 1043, 1044],
                                [1050, 1051, 1052, 1053, 1054],
                            ],
                        ]
                    ),
                )

    def test_export_creates_save_dir(self):
        """Export should create the save directory if it does not exist."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "nested" / "subdir" / "test_ws.npz"
            export_data_to_np_format(ws, npz_path)
            self.assertTrue(npz_path.parent.exists())
            self.assertTrue(npz_path.exists())

    def test_export_with_path_object(self):
        """Export should accept a Path object as save_dir."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path)
            self.assertTrue(npz_path.exists())

    def test_export_with_roi_single_column(self):
        """Export with ROI selecting a single column of detectors."""
        ws = make_test_workspace()
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "test_ws.npz"
            export_data_to_np_format(ws, npz_path, det_ids=[101, 104])
            with np.load(npz_path) as data:
                # det_ids [101, 104] are at positions (0,0), (1,0)
                # Bounding box: rows 0-2, cols 0-0
                self.assertEqual(data["counts"].shape, (2, 1, 5))
                self.assertEqual(data["detector_ids"].shape, (2, 1))
                np.testing.assert_array_equal(
                    data["detector_ids"], np.array([[101], [104]])
                )
                np.testing.assert_array_equal(
                    data["counts"],
                    np.array(
                        [
                            [[1010, 1011, 1012, 1013, 1014]],
                            [[1040, 1041, 1042, 1043, 1044]],
                        ]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
