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
  l2_values: [0.62, 0.63, 0.64, 0.65] # optional: spread the samples over several distances
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

> [!NOTE]
> The sample-detector distance is a **global** setting, given in `defaults`
> or with `--l2` / `--l2-values`. A workspace may not override it, because a
> mixed region can take parts from several workspaces and they all have to
> share the same energy axis.

### Mixed regions

A region may be a mixture of other regions, given by a `mix` list instead of
a `selection` file. Each part either names a selection file directly, or
refers to another region by its `name`:

```yaml
    selections:
      - name: gold
        selection: data/sel_gold.nxs
        isotopes: [Au-197]
      - name: silver
        selection: data/sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
      - name: alloy
        mix:
          - region: gold
            fraction: 0.7
          - region: silver
            fraction: 0.3
```

The fractions give the share of the region covered by each part. The mixture
is sampled as a single detector region, so each spectrum of a summed sample
lands in a part with the probability given by that part's fraction. This
keeps the total counts of a summed sample at the same scale as an unmixed
region, and makes parts of very different sizes comparable.

The isotopes of a mixed region are derived from its parts, unless they are
given explicitly.

### Mixtures across workspaces

A part may come from a **different workspace**, by naming it with the
`workspace` key. This allows a mixture to be made from samples that were
measured separately, which is what is needed to build training data for
mixtures of isotopes that were never mounted together:

```yaml
workspaces:
  - name: gold_run
    workspace: data/gold.nxs
    selections:
      - name: gold
        selection: data/sel_gold.nxs
        isotopes: [Au-197]
      - name: alloy
        mix:
          - region: gold
            fraction: 0.5
          - workspace: silver_run
            region: silver
            fraction: 0.5

  - name: silver_run
    workspace: data/silver.nxs
    selections:
      - name: silver
        selection: data/sel_silver.nxs
        isotopes: [Ag-107, Ag-109]
```

Each part is read from its own workspace, and the parts must all share the
same energy binning, which they do because every workspace is converted onto
the same fixed grid. The noise of a part is drawn from the detectors of that
part's own workspace that lie outside the region.

> [!NOTE]
> The sample-detector distance is a global setting, so every workspace is
> converted at the same L2 values and the parts of a mixture always share the
> same energy axis.

### Usage

```bash
python generate_training_data.py \
    --config configs/example_training_data.yaml \
    --n-sets 10 \
    --n-samples 10 20 50 100 \
    --noise 0.1 \
    --l2-values 0.62 0.63 0.64 0.65 \
    --output-dir training_data
```

Command line arguments take precedence over the configuration file, which in
turn takes precedence over the built in defaults. Add `--seed` for
reproducible data and `--overwrite` to regenerate files that already exist.

### Output

One file is written per selection and per `n_samples` value, named
`<workspace>_<selection>_n<n_samples>_noise<noise>_l2<l2>.npz`. When several
L2 values are used the name ends in `_l2x<n>` instead:

| Key | Description |
| --- | --- |
| `spectra_samples` | Array of shape `(n_sets, n_bins)` of summed spectra |
| `x_bins` | Energy bin edges |
| `name` | Workspace name from the config |
| `selection_name` | Name of the selection region |
| `isotopes` | Isotopes contained in the selection region |
| `is_background_only` | `True` when the region contains no isotope |
| `is_mixed` | `True` when the region mixes several other regions |
| `mix_fractions`, `mix_names`, `mix_workspaces` | The parts of a mixed region |
| `label` | Label for the region (numeric labels keep their type) |
| `label_is_null` | `True` when no label was given |
| `n_samples`, `n_sets`, `noise_level` | Settings used for the file |
| `l2_values` | The sample-detector distances used |
| `spectra_l2` | The distance each sample was taken at |

## Identifying samples with a dictionary

Rather than classifying a spectrum with a neural network, the contents of a
sample can be found by fitting it against a dictionary of known spectra.
Because the mixture of two isotopes is exactly linear in their spectra, this
recovers the abundance of each known component directly.

`data_manipulation/dictionary.py` builds the dictionary from the generated
data files, and `data_manipulation/fit_abundances.py` fits spectra against
it.

```python
from data_manipulation.dictionary import build_dictionary_from_files
from data_manipulation.fit_abundances import fit_spectrum

dictionary = build_dictionary_from_files(
    ["Ag_Ag_n50_noise0.1_l2x4.npz", "Au_Au_n50_noise0.1_l2x4.npz"]
)
dictionary.save("dictionary.npz")

result = fit_spectrum(spectrum, dictionary)
print(result.dominant)            # the entry that makes up the largest fraction
print(result.fractions)           # the fraction of the region each entry covers
print(result.residual_fraction)   # how much the dictionary cannot explain
```

The dictionary entries are built from *measured* data, not from the ENDF
total cross-sections, since the measured spectrum is the differential
response convolved with the detector response and the background.

From the command line:

```bash
python -m data_manipulation.fit_abundances \
    --dictionary dictionary.npz \
    --input Ag_Ag_n50_noise0.1_l2x4.npz \
    --detect-unknown \
    --output fits.csv
```

Pass `--detect-unknown` to look for a component the dictionary does not
describe. The residual is decomposed with NMF, and is only reported when it
is larger than `--unknown-threshold` (0.003 by default).

### Building the dictionary from raw selection files

A new element can be added to the dictionary without first generating any
training data for it. The dictionary can instead be built directly from the
raw `.nxs` selection workspaces, using the same TOF-to-energy conversion and
rebinning as the training data:

```bash
python -m data_manipulation.build_dictionary \
    --config configs/example_dictionary.yaml \
    --output dictionary.npz
```

The configuration lists the selection workspaces to use, and which isotopes
each one contains:

```yaml
l2_values: [0.62, 0.63, 0.64, 0.65]

selections:
  - name: Ag/Ag
    selection: Ag_Ag_selection.nxs
    isotopes: [Ag-107, Ag-109]
    workspace: Ag
  - name: Ag/background
    selection: Ag_bg_selection.nxs
    isotopes: []
    workspace: Ag
```

Each selection becomes one dictionary entry. A selection with no isotopes is
treated as a background entry, so background should be listed once per
background region rather than once for the whole experiment.

The same configuration can also list generated data files, which are built
with the `files:` key and use the same format as the training data output.
The two sources can be combined in one configuration.

> [!NOTE]
> On the generated `29-10` data, a dictionary built from the raw selection
> files identified **360/360** spectra correctly, and one built from the
> generated data files identified **359/360**. The two routes are therefore
> equivalent in practice, and the selection route is preferable when adding
> a new element to the dictionary.

> [!NOTE]
> `generate_training_data_poc.py` is the original proof of concept, kept for
> reference. It requires workspaces to already be loaded in Mantid.

### Plotting the dictionary entries

The reference spectra held in a dictionary can be plotted to inspect their
shape, and in particular the Bragg peaks and absorption dips that tell one
element from another:

```bash
python plot_dictionary.py dictionary.npz --view grid --output dictionary.png
```

Four views are available:

| View | What it shows |
| --- | --- |
| `overlay` | Every entry on one pair of axes, for direct comparison |
| `grid` | One panel per entry, each on its own scale |
| `peaks` | One panel per entry, with a smooth baseline removed |
| `difference` | Every entry minus a reference entry |

The entries are measured on very different count scales, so plotting them
raw hides the smaller ones. The `--normalise` option rescales them so
they can be compared:

| Normalisation | Effect |
| --- | --- |
| `none` | The raw counts are plotted |
| `sum` | Each entry is divided by its total counts |
| `baseline` | Each entry is divided by its smoothed profile, which keeps the absorption dips and Bragg peaks at their true relative depth |

Other options are `--entries` to plot only some of the entries,
`--x-min` and `--x-max` to restrict the energy range, and
`--exclude-background` to leave out the entries that hold no isotope.

For example, to compare the element entries on a common scale:

```bash
python plot_dictionary.py dictionary.npz \
    --view overlay --normalise baseline --exclude-background \
    --output elements.png
```

## Reading the local ENDF database

The project can read an ENDF/B-VIII.0 evaluation directly, which gives access
to the resolved resonance parameters behind the absorption dips. Three modules
work together:

| Module | Purpose |
| --- | --- |
| `endf_database_util/local_endf.py` | Finds the database and parses the MF=2 resonance files |
| `endf_database_util/resonance.py` | Reconstructs the cross-section (ENDF-102) from the resonance parameters |
| `endf_database_util/rmatrix.py` | Handles the R-matrix (LRF=7) evaluations, e.g. Cu |

The `endf` library is already a declared dependency in `pixi.toml`:

````toml
[pypi-dependencies]
endf = { git = "https://github.com/paulromano/endf-python.git" }
````

### Locating the database

`find_database_root()` searches the working directory and its parents for a
directory containing `neutrons/n-*.endf`, so the usual layout

````text
../ENDF-B-VIII.0/ENDF-B-VIII.0/neutrons/n-063063.endf
````

is found automatically. An explicit path can always be given instead:

````python
from endf_database_util.local_endf import LocalENDFDatabase

db = LocalENDFDatabase(root=r"..\ENDF-B-VIII.0\ENDF-B-VIII.0")
print(db.available_isotopes()[:5])
````

### Reconstructing a cross-section

The MF=3 tabulated cross-sections have only a handful of points in the
resonance region, so the dips cannot be resolved from them. The MF=2
resolved resonance parameters are therefore reconstructed according to
ENDF-102 (SLBW/MLBW, Reich-Moore and R-matrix):

````python
import numpy as np

from endf_database_util.local_endf import LocalENDFDatabase
from endf_database_util.resonance import total_cross_section
from endf_database_util.rmatrix import RMatrixParameters, rmatrix_total_cross_section

db = LocalENDFDatabase()
energies = np.linspace(1.0, 500.0, 20000)

params = db.resonance_parameters("Cu", 65)
if isinstance(params, RMatrixParameters):
    result = rmatrix_total_cross_section(params, energies)
else:
    result = total_cross_section(params, energies)

print(result.total.max(), "barns")
````

### Why the resonance energies matter

The dips in a measured profile are at the **resonance** energies, so they can
be matched directly against the ENDF evaluation:

| Element | Resonance energies (eV) |
| --- | --- |
| Ag | 5.190, 16.300, 30.600, 51.560, 55.800, 71.000, 87.700 |
| Au | 4.900, 60.291 |
| In | 1.457, 3.850, 9.070, 12.040, 15.025, 22.730, 24.990, 39.610 |
| Zn | 223.100 (Zn-67), 281.000 (Zn-64), 448.200 (Zn-67) |
| Nb | 194.250, 378.470 |
| Cu | 230.044 (Cu-65) |

> [!NOTE]
> Use the **natural isotope mixture** when simulating a metal sample, not a
> single isotope. The copper dip at ~230 eV is the Cu-65 R-matrix resonance;
> Cu-63 has no resonance below 400 eV.

### Comparing a measured profile with a cross-section

Measured profiles are *transmissions* (dips), while cross-sections are *peaks*.
Convert the measured profile to optical thickness before comparing:

````python
import numpy as np
from scipy.ndimage import median_filter

positive = profile[profile > 0]
reference = np.median(positive)
optical_thickness = -np.log(np.maximum(profile / reference, 1e-12))

residuals = optical_thickness - median_filter(optical_thickness, size=51)
````

The residual of the measured profile can then be correlated with the
reconstructed cross-section to identify the element. Doing this with the local
ENDF database alone identifies Ag, Au, In, Zn, Cu and Nb correctly; the Fe
entry in the dictionary is known to be noise and is not identified.
