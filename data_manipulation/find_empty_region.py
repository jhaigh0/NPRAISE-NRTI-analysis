import numpy as np


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
