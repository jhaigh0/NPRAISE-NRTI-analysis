# NPRAISE-NRTI-analysis

## Accessing data on IDAaaS

To run the loading code on IDAaaS,

1. Enter `smb://isis.cclrc.ac.uk/shares/` into the file browser search bar to mount the shares dir
2. Create a symlink (mantid does not cope with the path) to `~/nGEM-data/DATA` with the command `ln -s ~/.gvfs/'smb-share:server=isis.cclrc.ac.uk,share=shares'/nGEM-Imaging/DATA ~/nGEM-data/DATA`

You will have to repeat step 1 with each new IDAaaS instance.

## Generating training data

`generate_training_data.py` builds training data for the 1D CNN from a YAML
file listing the workspaces to process. For every workspace it loads the full
detector workspace and its *selection workspace* (a cropped region holding a
single isotope or a mix of isotopes), converts the spectra from TOF to energy,
samples summed spectra from the selection region, and saves them to `.npz`
files.

### Configuration file

Paths are resolved relative to the configuration file, so the config can live
alongside the data it describes. See
[`configs/example_training_data.yaml`](configs/example_training_data.yaml).

Each workspace has a `workspace` file and a list of `selections`. A selection
is a cropped region of the detector, given by a selection workspace file that
holds a subset of the detector IDs of the full workspace. Each selection lists
the `isotopes` present in it; an empty list means the region contains
background only.

```yaml
defaults:
  l2: 0.6          # sample-detector distance used for the TOF -> energy conversion
  noise_level: 0.1 # fraction of each sample drawn from outside the selection region
  n_sets: 10       # number of summed spectra per configuration
  n_samples: [10, 20, 50, 100]

workspaces:
  - name: Sn_letters_Bronze_sheet
    workspace: data/Sn_letters_Bronze_sheet_normMonitor.nxs
    selections:
      - name: tin
        selection: data/sel_tin.nxs
        isotopes: [Sn-118, Sn-120]
        label: 0
      - name: background
        selection: data/sel_background.nxs
        isotopes: []   # background only, no isotope present
        label: 1
```

A workspace with only one region can use the shorter `selection:` form
instead of `selections:`, with the `isotopes` given at the workspace level.

A selection inherits the workspace `label` unless it defines its own. Use
`label: null` to explicitly clear it.

### Usage

```bash
python generate_training_data.py \
    --config configs/example_training_data.yaml \
    --n-sets 10 \
    --n-samples 10 20 50 100 \
    --noise 0.1 \
    --l2 0.6 \
    --output-dir training_data
```

Command line arguments take precedence over the configuration file, which in
turn takes precedence over the built in defaults. Add `--seed` for
reproducible data and `--overwrite` to regenerate files that already exist.

### Output

One file is written per selection and per `n_samples` value, named
`<workspace>_<selection>_n<n_samples>_noise<noise>_l2<l2>.npz`:

| Key | Description |
| --- | --- |
| `spectra_samples` | Array of shape `(n_sets, n_bins)` of summed spectra |
| `x_bins` | Energy bin edges |
| `name` | Workspace name from the config |
| `selection_name` | Name of the selection region |
| `isotopes` | Isotopes contained in the selection region |
| `is_background_only` | `True` when the region contains no isotope |
| `label` | Label for the region (numeric labels keep their type) |
| `label_is_null` | `True` when no label was given |
| `n_samples`, `n_sets`, `noise_level`, `l2` | Settings used for the file |

> [!NOTE]
> `generate_training_data_poc.py` is the original proof of concept, kept for
> reference. It requires workspaces to already be loaded in Mantid.
