import numpy as np
from sklearn.decomposition import NMF, PCA

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
    components = pca.components_
    return scores, components


def perform_nmf_on_workspace(ws, n_components, mask_limit=0.0):
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=0)
    y_data = np.nan_to_num(ws.extractY(), nan=0.0)
    W = nmf.fit_transform(y_data)
    H = nmf.components_
    background_mask = np.max(W, axis=1) <= mask_limit
    return W, H, background_mask


def get_workspace_indexes_from_W_array(W, min_score=0.0):
    masks = W > min_score
    x, y = np.where(masks)
    indices_by_column = [x[y == i] for i in range(masks.shape[1])]
    return indices_by_column


def get_dimensions_of_roi_workspace(ws):
    nrows, ncols = get_grid_dimensions(ws)
    all_detectors = np.arange(1, nrows * ncols + 1).reshape(nrows, ncols)
    det_ids = list(ws.getDetectorIDToWorkspaceIndexMap(True, True).keys())
    mask = np.isin(all_detectors, det_ids)
    rows, cols = np.where(mask)
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    return height, width
