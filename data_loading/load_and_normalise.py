from pathlib import Path
from sys import platform

from mantid.simpleapi import *


def load_and_normalise(sample_run_no: int, normalise_run_no: int, name: str):
    name = name if name else str(sample_run_no)
    sample_ws = load_and_process(sample_run_no, name)
    normalise_ws = load_and_process(normalise_run_no, "normalise")
    Divide(sample_ws, normalise_ws, OutputWorkspace=f"{name}_normalised")


def load_and_process(run_no: int, name: str):
    filenames = load_ngem_from_INES_run_number(run_no)
    ws = LoadNGEM(Filename=",".join(filenames), OutputWorkspace=name)
    ConvertUnits(InputWorkspace=ws, OutputWorkspace=ws, Target="Energy")
    CropWorkspace(InputWorkspace=ws, OutputWorkspace=ws, XMin=1000, XMax=1000000)
    Rebin(InputWorkspace=ws, OutputWorkspace=ws, Params="100,-0.01,1e+06")
    ws_summed = SumSpectra(InputWorkspace=ws, OutputWorkspace=f"{name}_summed")
    return ws_summed


def load_ngem_from_INES_run_number(run_no: int):
    if platform == "windows":
        data_base_dir = Path("//isis.cclrc.ac.uk/Shares/nGEM-Imaging/DATA/")
    elif platform == "linux":
        # assumption is we're on idaaas
        data_base_dir = Path("/home/h1121412/nGEM-data/DATA/")

    run_dir = next(data_base_dir.glob(f"*/INES{run_no}"))
    data_files = run_dir.glob("**/*.edb")
    return [str(f) for f in data_files]
