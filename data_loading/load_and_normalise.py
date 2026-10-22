from pathlib import Path
from sys import platform

from mantid.simpleapi import *


def load_and_normalise(
    sample_run_no: int,
    normalise_run_no: int,
    name: str,
    smooth_normalisation: bool = False,
):
    name = name if name else str(sample_run_no)
    sample_ws = load_and_process(sample_run_no, name)
    normalise_ws = load_and_process(normalise_run_no, f"normalise_{name}")
    if smooth_normalisation:
        normalise_ws = create_smooth_normalisation_workspace(normalise_ws)
    Divide(sample_ws, normalise_ws, OutputWorkspace=f"{name}_normalised")
    ReplaceSpecialValues(
        InputWorkspace=f"{name}_normalised",
        NanValue=0.0,
        InfinityValue=0.0,
        OutputWorkspace=f"{name}_normalised",
    )
    SumSpectra(
        InputWorkspace=f"{name}_normalised", OutputWorkspace=f"{name}_normalised_summed"
    )


def load_and_process(run_no: int, name: str):
    filenames = get_ngem_from_INES_run_number(run_no)
    ws = LoadNGEM(Filename=",".join(filenames), OutputWorkspace=name)
    ConvertUnits(InputWorkspace=ws, OutputWorkspace=ws, Target="Energy")
    CropWorkspace(InputWorkspace=ws, OutputWorkspace=ws, XMin=1000, XMax=1000000)
    Rebin(InputWorkspace=ws, OutputWorkspace=ws, Params="1000,-0.005,1e+06")
    return ws


def convert_TOF_to_energy(ws, l2=0.6, output_name: str | None = None):
    ENERGY_REBIN_PARAMS = "1000,-0.005,500000"
    new_ws_title = f"{ws.name()}_energy" if output_name is None else output_name
    MoveInstrumentComponent(
        Workspace=ws, ComponentName="source", Z=22.8, RelativePosition=False
    )
    MoveInstrumentComponent(
        Workspace=ws, ComponentName="main-detector-bank", Z=l2, RelativePosition=False
    )
    ConvertUnits(InputWorkspace=ws, OutputWorkspace=new_ws_title, Target="Energy")
    # Rebin onto a fixed log grid, so that every workspace has the same X
    # binning regardless of the L2 used to convert it.
    return Rebin(
        InputWorkspace=new_ws_title,
        OutputWorkspace=new_ws_title,
        Params=ENERGY_REBIN_PARAMS,
    )


def convert_TOF_to_energy_at_l2_values(ws, l2_values, name_prefix: str = "l2"):
    """Convert a TOF workspace to energy at several sample-detector distances.

    The whole detector bank is moved for each distance, so that every pixel's
    geometry is taken into account by the conversion, exactly as
    :func:`convert_TOF_to_energy` does for a single distance.

    Note that :func:`convert_TOF_to_energy` derives its output name from the
    name of the input workspace, so it can only be used once per input. This
    function therefore performs the conversion itself, giving each distance
    its own output name.

    All the converted workspaces are rebinned onto the same fixed grid, so
    they all share exactly the same X axis and their counts can be compared
    and stored together directly.

    Args:
        ws: The TOF workspace to convert.
        l2_values: The sample-detector distances, in metres.
        name_prefix: Prefix for the names of the generated workspaces.

    Returns:
        A list of workspaces, in the same order as ``l2_values``.
    """
    workspaces = []
    for l2 in l2_values:
        output_name = f"{name_prefix}_{l2:g}"
        workspaces.append(convert_TOF_to_energy(ws, l2=l2, output_name=output_name))
    return workspaces


def get_ngem_from_INES_run_number(run_no: int):
    if platform.startswith("win"):
        data_base_dir = Path("//isis.cclrc.ac.uk/Shares/nGEM-Imaging/DATA/")
    elif platform.startswith("linux"):
        # assumption is we're on idaaas
        data_base_dir = Path("/home/h1121412/nGEM-data/DATA/")

    run_dir = next(data_base_dir.glob(f"*/INES{run_no}"))
    data_files = run_dir.glob("**/*.edb")
    return [str(f) for f in data_files]


def create_smooth_normalisation_workspace(ws):
    y_data = np.nan_to_num(ws.extractY(), nan=0.0)
    summed_spectrum = np.sum(y_data, axis=0)
    spectra_totals = np.sum(y_data, axis=1)
    summed_norm = summed_spectrum / np.sum(summed_spectrum)
    new_y = spectra_totals[:, np.newaxis] * summed_norm[np.newaxis, :]
    new_ws = CreateWorkspace(
        DataX=ws.readX(0),
        DataY=new_y.ravel(),
        NSpec=ws.getNumberHistograms(),
        UnitX=ws.getAxis(0).getUnit().name(),
        ParentWorkspace=ws,
        OutputWorkspace=f"{ws.name()}_smoothed",
    )
    return new_ws


def normalise_via_empty_region_mask(ws, mask_array):
    y_data = np.nan_to_num(ws.extractY(), nan=0.0)
    new_y = np.copy(y_data)
    empty_region_y_data = y_data[mask_array, :]
    sample_region_y_data = y_data[~mask_array, :]
    empty_region_spectrum = np.sum(empty_region_y_data, axis=0) / np.sum(mask_array)
    sample_region_y_data /= empty_region_spectrum[np.newaxis, :]
    new_y[~mask_array, :] = np.nan_to_num(sample_region_y_data, nan=0.0)
    new_ws = CreateWorkspace(
        DataX=ws.readX(0),
        DataY=new_y.ravel(),
        NSpec=ws.getNumberHistograms(),
        UnitX=ws.getAxis(0).getUnit().name(),
        ParentWorkspace=ws,
        OutputWorkspace=f"{ws.name()}_normalised_by_mask",
    )
    return new_ws


def find_first_non_zero_bin(ws):
    y_data = np.nan_to_num(ws.extractY(), nan=0.0)
    first_non_zero_index = np.any(y_data != 0, axis=0).argmax()
    return first_non_zero_index


def crop_workspace_to_first_non_zero_bin(ws):
    first_non_zero_index = find_first_non_zero_bin(ws)
    x_data = ws.readX(0)
    bin_value = x_data[first_non_zero_index]
    CropWorkspace(InputWorkspace=ws, OutputWorkspace=ws, XMin=float(bin_value))
    return ws
