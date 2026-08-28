import numpy as np
from sklearn.decomposition import PCA

from data_manipulation.mix_workspaces import get_grid_dimensions


def generate_threshhold_mask_percentage(ws, threshold: float):
    nspec = ws.getNumberHistograms()
    int_counts = np.array(
        ws.getIntegratedCountsForWorkspaceIndicies(
            range(nspec), nspec, None, None, True
        )
    )
    threshold_value = threshold * np.max(int_counts.max())
    mask = (int_counts >= threshold_value).astype(float)
    return mask


def generate_threshhold_mask_absolute(ws, threshold: float):
    nspec = ws.getNumberHistograms()
    int_counts = np.array(
        ws.getIntegratedCountsForWorkspaceIndicies(
            range(nspec), nspec, None, None, True
        )
    )
    mask = (int_counts >= threshold).astype(float)
    return mask


def perform_pca_on_workspace(ws, n_components):
    pca = PCA(n_components=n_components)
    y_data = np.nan_to_num(ws.extractY(), nan=0.0)
    scores = pca.fit_transform(y_data)
    return scores, pca


def get_dimensions_of_roi_workspace(ws):
    nrows, ncols = get_grid_dimensions(ws)
    all_detectors = np.arange(1, nrows * ncols + 1).reshape(nrows, ncols)
    det_ids = list(ws.getDetectorIDToWorkspaceIndexMap(True, True).keys())
    mask = np.isin(all_detectors, det_ids)
    rows, cols = np.where(mask)
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    return height, width
