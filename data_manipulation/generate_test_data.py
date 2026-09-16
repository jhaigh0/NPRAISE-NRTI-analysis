import numpy as np
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from data_loading.load_and_normalise import load_and_process
from data_manipulation.find_empty_region import perform_nmf_on_workspace
from data_manipulation.mix_workspaces import mix_random_samples_from_workspace_indexes


def generate_test_data(
    run_no: int, name: str, n_components: int | None = None, noise_level: float = 0.05
):
    ws = load_and_process(run_no, name)
    W = None
    if n_components is not None:
        W, _, _, _ = perform_nmf_on_workspace(ws, n_components)
    else:
        W = find_optimal_n_components(ws)
    # TODO: analytical way to determine min_score
    indexes_by_component = get_workspace_indexes_from_W_array_min_score(
        W, min_score=1.0
    )
    for indexes in indexes_by_component:
        # TODO: analytical way to determine n_sets
        sample_spectra = mix_random_samples_from_workspace_indexes(
            ws, indexes, n_sets=10, noise_level=noise_level
        )

    # TODO: output/save with xbins etc
    return sample_spectra


def find_optimal_n_components(ws, max_components: int = 10):
    errors = {}
    matrices = {}
    for n_components in range(1, max_components + 1):
        W, _, reconstruction_error, _ = perform_nmf_on_workspace(ws, n_components)
        errors[n_components] = reconstruction_error
        matrices[n_components] = W

    optimal_n_components = find_elbow_point(errors)
    return matrices[optimal_n_components]


def find_elbow_point(errors: dict[int, float]):
    x = np.array(sorted(errors.keys()))
    y = np.array([errors[k] for k in x])

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line = p2 - p1
    line = line / np.linalg.norm(line)

    distances = []

    for xi, yi in zip(x, y):
        point = np.array([xi, yi])

        v = point - p1
        projection = np.dot(v, line) * line
        perpendicular = v - projection

        distances.append(np.linalg.norm(perpendicular))

    return x[np.argmax(distances)]


def get_workspace_indexes_from_W_array_min_score(W, min_score=0.0):
    masks = W > min_score
    x, y = np.where(masks)
    indices_by_column = [x[y == i] for i in range(masks.shape[1])]
    return indices_by_column


def get_workspace_indexes_from_W_array_relative_threshold(W, threshold=0.2):
    indices_by_column = [
        np.where(W[:, i] > threshold * W[:, i].max())[0] for i in range(W.shape[1])
    ]
    return indices_by_column


def get_workspace_indexes_from_W_array_otsu_threshold(W):
    indices_by_column = []
    for i in range(W.shape[1]):
        thresh = threshold_otsu(W[:, i])
        indices = np.where(W[:, i] > thresh)[0]
        indices_by_column.append(indices)
    return indices_by_column


def get_workspace_indexes_from_W_array_gaussian_mixture(W, n_components=2):
    indices_by_column = []
    for i in range(W.shape[1]):
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        values = W[:, i].reshape(-1, 1)
        labels = gmm.fit_predict(values)
        signal_component = np.argmax(gmm.means_.flatten())
        indices = np.where(labels == signal_component)[0]
        indices_by_column.append(indices)
    return indices_by_column
