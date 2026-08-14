from pathlib import Path

import numpy as np


def export_data_to_np_format(
    ws, filename: str | Path, det_ids: list[int] | None = None
):
    """Export data from a workspace to NumPy format (.npz).

    Args:
        ws : workspace to be saved
        filename (str | Path): Path where the .npz file will be saved
        det_ids (List[int], optional): Optional list of detector IDs describing a rectangular ROI. If this is provided then just this region will be saved. Defaults to None.
    """

    nBins = ws.getNumberBins()
    inst = ws.getInstrument()
    x = inst.getComponentByName("main-detector-bank").xpixels()
    y = inst.getComponentByName("main-detector-bank").ypixels()
    detid_to_wsi = ws.getDetectorIDToWorkspaceIndexMap(True, True)
    detector_ids = np.array(
        [det_id for det_id, _ in sorted(detid_to_wsi.items(), key=lambda item: item[1])]
    ).reshape(x, y)
    counts = ws.extractY().reshape(x, y, nBins)
    if det_ids is not None:
        mask = np.isin(detector_ids, det_ids)
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        detector_ids = detector_ids[r0 : r1 + 1, c0 : c1 + 1]
        counts = counts[r0 : r1 + 1, c0 : c1 + 1, :]

    edges = ws.readX(0)
    if filename:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
    else:
        filename = f"{ws.name()}.npz"
    np.savez_compressed(
        filename,
        counts=counts,
        detector_ids=detector_ids,
        edges=edges,
    )
