import numpy as np
from mantid.simpleapi import ExtractSpectra
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


def generate_test_data_from_selection_workspace(
    ws, selection_ws, n_sets, n_samples, noise=0.0
):
    selection_detid_map = selection_ws.getDetectorIDToWorkspaceIndexMap(False, False)
    selection_detids = selection_detid_map.keys()
    full_detid_map = ws.getDetectorIDToWorkspaceIndexMap(False, False)
    selection_ws_indexes = [full_detid_map[k] for k in selection_detids]
    samples = mix_random_samples_from_workspace_indexes(
        ws, selection_ws_indexes, n_sets, n_samples, noise, replace=True
    )
    return samples


def generate_test_data_from_mixed_parts(parts, n_sets, n_samples, noise=0.0):
    """Sample spectra from a region made up of parts of several workspaces.

    A part is a selection region of one workspace. The region is treated as
    a single detector region made up of the parts, where a part covers a
    fraction of it. Each spectrum of a summed sample is therefore drawn from
    the whole region, landing in a part with the probability given by that
    part's fraction. This keeps the total counts of a summed sample at the
    same scale as an unmixed region, and it makes parts of very different
    sizes comparable.

    The parts may come from different workspaces, which is what allows a
    mixture to be made from, say, an Ag sample and an Au sample that were
    measured separately. Each part is read from its own workspace, and the
    parts must all share the same energy binning.

    Noise spectra are drawn from the detectors of a part's own workspace
    that are not part of the region.

    Args:
        parts: The parts of the region. Each is a tuple of the full
            workspace of the part, its selection workspace, and the fraction
            of the region it covers.
        n_sets: The number of samples to generate.
        n_samples: The number of spectra summed together to make one sample.
        noise: The fraction of each sample drawn from outside the region.

    Returns:
        The list of summed spectra.
    """
    if not parts:
        raise ValueError("A mixed region must have at least one part.")

    weights = np.asarray([part[2] for part in parts], dtype=float)
    if np.any(weights <= 0):
        raise ValueError("The fraction of every part must be greater than zero.")
    weights = weights / weights.sum()

    n_parts = len(parts)
    n_noise = int(noise * n_samples)

    # Decide which part each spectrum of each sample comes from. The slots
    # are flattened so that a part owns whole slots of the samples.
    part_choice = np.random.choice(n_parts, size=(n_sets, n_samples), p=weights)
    if n_noise:
        noise_choice = np.random.choice(n_parts, size=(n_sets, n_noise), p=weights)
    else:
        noise_choice = np.zeros((n_sets, 0), dtype=int)

    # The detector IDs of every part, and the workspace each part belongs
    # to, so that the noise of a workspace avoids every part of that
    # workspace and not just the part being drawn from.
    selection_ids: list[list[int]] = []
    workspace_of_part: list[int] = []
    workspace_ids: list[set[int]] = []
    for part_index, (full_ws, selection_ws, _) in enumerate(parts):
        selection_ids.append(
            list(selection_ws.getDetectorIDToWorkspaceIndexMap(False, False))
        )
        # Parts that share a workspace share the noise pool of that
        # workspace, so they are identified by the index of the workspace
        # rather than by its contents.
        same = [index for index, (other, _, _) in enumerate(parts) if other is full_ws]
        workspace_of_part.append(same[0])
        workspace_ids.append(
            set(full_ws.getDetectorIDToWorkspaceIndexMap(False, False).keys())
        )

    n_bins = parts[0][0].getNumberBins()
    spectra = np.zeros((n_sets, n_bins), dtype=float)

    # The slot of a spectrum tells which sample it belongs to: the slots of
    # a sample are laid out one after the other.
    signal_rows = np.repeat(np.arange(n_sets), n_samples)

    for part_index, (full_ws, _, _) in enumerate(parts):
        signal_slots = np.flatnonzero(part_choice.ravel() == part_index)
        n_signal = signal_slots.size
        if n_noise:
            noise_slots = np.flatnonzero(noise_choice.ravel() == part_index)
            noise_rows = np.repeat(np.arange(n_sets), n_noise)
        else:
            noise_slots = np.zeros(0, dtype=int)
            noise_rows = np.zeros(0, dtype=int)
        n_noise_here = noise_slots.size
        if n_signal == 0 and n_noise_here == 0:
            continue

        detids = selection_ids[part_index]
        if not detids:
            raise ValueError("A part of a mixed region contains no detectors.")

        # The detectors of this workspace that are not part of the region.
        excluded: set[int] = set()
        for other_index, other_workspace in enumerate(workspace_of_part):
            if other_workspace == workspace_of_part[part_index]:
                excluded.update(selection_ids[other_index])
        outside = sorted(workspace_ids[part_index] - excluded)

        picks: list[int] = []
        if n_signal:
            picks.extend(np.random.choice(detids, size=n_signal, replace=True))
        if n_noise_here:
            if not outside:
                raise ValueError(
                    "There are no detectors outside the region to draw noise "
                    "from. Use a noise level of zero, or a smaller region."
                )
            picks.extend(np.random.choice(outside, size=n_noise_here, replace=True))

        # Extract only the detectors that are needed, rather than the whole
        # workspace, which would be far too much memory.
        needed = sorted(set(picks))
        extracted = ExtractSpectra(
            full_ws,
            DetectorList=",".join(map(str, needed)),
            StoreInADS=False,
        )
        extracted_map = extracted.getDetectorIDToWorkspaceIndexMap(False, False)
        values = extracted.extractY()

        rows = np.asarray([extracted_map[detid] for detid in picks])
        contributions = values[rows]

        if n_signal:
            np.add.at(
                spectra,
                signal_rows[signal_slots],
                contributions[:n_signal],
            )
        if n_noise_here:
            np.add.at(
                spectra,
                noise_rows[noise_slots],
                contributions[n_signal:],
            )

    return [spectra[index] for index in range(n_sets)]
