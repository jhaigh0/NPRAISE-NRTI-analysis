import numpy as np
from mantid.simpleapi import AppendSpectra, ExtractSpectra, Scale


def _combine_workspaces(ws1, ws2, split_index: int, axis: str):
    """Combine two workspaces by splitting along the given axis ('vertical' or 'horizontal')."""
    factor = np.sum(ws1.extractY()) / np.sum(ws2.extractY())
    nrows, ncols = get_grid_dimensions(ws1)
    first_ids, second_ids = _split_det_ids(nrows, ncols, split_index, axis)
    first_region_ws = ExtractSpectra(
        ws1, DetectorList=",".join(map(str, first_ids)), StoreInADS=False
    )
    second_region_ws = ExtractSpectra(
        ws2, DetectorList=",".join(map(str, second_ids)), StoreInADS=False
    )
    second_region_ws = Scale(second_region_ws, factor, StoreInADS=False)
    result = AppendSpectra(first_region_ws, second_region_ws)
    return result


def combine_workspaces_vertical_split(ws1, ws2, split_index: int):
    return _combine_workspaces(ws1, ws2, split_index, "vertical")


def combine_workspaces_horizontal_split(ws1, ws2, split_index: int):
    return _combine_workspaces(ws1, ws2, split_index, "horizontal")


def _split_det_ids(nrows: int, ncols: int, split_index: int, axis: str):
    """Split detector IDs along the given axis ('vertical' or 'horizontal')."""
    det_ids = np.arange(1, nrows * ncols + 1).reshape(nrows, ncols)
    if axis == "vertical":
        first_ids = det_ids[:, :split_index].ravel()
        second_ids = det_ids[:, split_index:].ravel()
    else:
        first_ids = det_ids[:split_index, :].ravel()
        second_ids = det_ids[split_index:, :].ravel()
    return first_ids, second_ids


def split_det_ids_vertically(nrows: int, ncols: int, split_index: int):
    return _split_det_ids(nrows, ncols, split_index, "vertical")


def split_det_ids_horizontally(nrows: int, ncols: int, split_index: int):
    return _split_det_ids(nrows, ncols, split_index, "horizontal")


def get_grid_dimensions(ws):
    main_bank = ws.getInstrument().getComponentByName("main-detector-bank")
    ncols = main_bank.xpixels()
    nrows = main_bank.ypixels()
    return nrows, ncols
