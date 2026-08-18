import numpy as np
from mantid.simpleapi import AppendSpectra, ExtractSpectra


def combine_workspaces_vertical_split(ws1, ws2, split_index: int):
    nrows, ncols = get_get_grid_dimensions(ws1)
    left_ids, right_ids = split_det_ids_vertically(nrows, ncols, split_index)
    left_region_ws = ExtractSpectra(
        ws1, DetectorList=",".join(map(str, left_ids)), StoreInADS=False
    )
    right_region_ws = ExtractSpectra(
        ws2, DetectorList=",".join(map(str, right_ids)), StoreInADS=False
    )
    result = AppendSpectra(left_region_ws, right_region_ws)
    return result


def combine_workspaces_horizontal_split(ws1, ws2, split_index: int):
    nrows, ncols = get_get_grid_dimensions(ws1)
    top_ids, bottom_ids = split_det_ids_horizontally(nrows, ncols, split_index)
    top_region_ws = ExtractSpectra(
        ws1, DetectorList=",".join(map(str, top_ids)), StoreInADS=False
    )
    bottom_region_ws = ExtractSpectra(
        ws2, DetectorList=",".join(map(str, bottom_ids)), StoreInADS=False
    )
    result = AppendSpectra(top_region_ws, bottom_region_ws)
    return result


def split_det_ids_vertically(nrows: int, ncols: int, split_index: int):
    det_ids = np.arange(1, nrows * ncols + 1).reshape(nrows, ncols)
    left_ids = det_ids[:, :split_index].ravel()
    right_ids = det_ids[:, split_index:].ravel()
    return left_ids, right_ids


def split_det_ids_horizontally(nrows: int, ncols: int, split_index: int):
    det_ids = np.arange(1, nrows * ncols + 1).reshape(nrows, ncols)
    top_ids = det_ids[:split_index, :].ravel()
    bottom_ids = det_ids[split_index:, :].ravel()
    return top_ids, bottom_ids


def get_get_grid_dimensions(ws):
    main_bank = ws.getInstrument().getComponentByName("main-detector-bank")
    ncols = main_bank.xpixels()
    nrows = main_bank.ypixels()
    return nrows, ncols
