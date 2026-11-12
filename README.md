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

### Fitting a region against the ENDF database

`data_manipulation/fit_endf.py` fits a measured region against the
cross-sections of a list of elements, without needing a dictionary of
measured spectra at all. It is the theoretical counterpart of
`fit_abundances.py`.

The model is the transmission law

````text
T(E) = exp(-sum_i c_i * sigma_i(E))
````

where `sigma_i` is the reconstructed cross-section of element `i` and `c_i`
is its areal density. The fit is a non-negative least squares fit of this
model to the measured transmission, with a Gaussian broadening of the
templates fitted alongside it, because the detector's energy resolution is
not known exactly.

**The fit needs only a profile and an energy grid.** It does not need a
dictionary: the dictionary is just one way of holding the measured
regions, and it is the *measured dictionary* route that needs one.

```python
import numpy as np

from data_manipulation.fit_endf import fit_region
from endf_database_util.templates import build_fine_templates

# A profile is one transmission spectrum, and the energies are the bin
# centres in eV.
energies = np.sqrt(x_bins[:-1] * x_bins[1:]) / 1000.0
templates = build_fine_templates(["Ag", "Au", "In", "Zn", "Cu", "Nb", "Fe"], energies)

result = fit_region(
    profile,
    ["Ag", "Au", "In", "Zn", "Cu", "Nb", "Fe"],
    energies=energies,
    templates=templates,
)
print(result.dominant)             # 'Cu'
print(dict(zip(result.symbols, result.fractions)))
```

From the command line, a raw selection workspace can be fitted directly:

````bash
python -m data_manipulation.fit_endf \
    --workspace "path/to/Havar_5foils_div_Empty_normMonitor_selection.nxs=Havar/Havar_5foils" \
    --elements Co Cr Ni W Mo Mn Fe \
    --output endf_fits.csv
````

The name after the `=` is the name of the region, and it is used to look
the metadata and the known composition up. `--l2` may be given more than
once, and the profiles of the conversions are averaged.

The regions of a dictionary can be fitted instead, which is convenient
when the dictionary has already been built:

````bash
python -m data_manipulation.fit_endf \
    --dictionary dictionary_all_elements.npz \
    --elements Ag Au In Zn Cu Nb Fe \
    --exclude-background \
    --output endf_fits.csv
````

Use `--region` to fit only some of the regions, and `--output` to write the
results to a CSV file. The CSV holds the dominant element, the residual, the
fitted broadening, and the fraction, areal density and correlation of each
element.

> [!NOTE]
> The templates are cached on disk under
> `endf_database_util/.cache/templates/`, so only the first run is slow.
> Building the templates for all seven elements takes about fifteen seconds;
> a cached run takes about a tenth of a second.

> [!NOTE]
> The **fractions** are the shares of the modelled optical thickness, not
> the shares of the areal density. The elements have very different
> cross-section magnitudes, so an element with a small cross-section needs a
> large areal density to make the same dip. The areal densities are
> reported separately.

> [!IMPORTANT]
> The cross-sections are evaluated on a grid **sixty times finer** than the
> measurement grid, and are sampled back onto it after the broadening. This
> matters: the measurement grid has a step of about 0.5 %, which is 1.15 eV
> at 230 eV, and that badly undersamples the narrow resonances. The
> copper-65 resonance at 230 eV has a peak of 81 barns, but only 15 barns
> are seen when it is evaluated on the measurement grid, and the niobium
> resonance at 378 eV loses a factor of nine. Use
> `build_fine_templates` rather than `build_templates` when the templates
> are to be compared with a measured spectrum.

### Identifying the measured regions

Fitting every measured region of the dictionary against the local ENDF
database gives the correct element for six of the seven regions:

| Region | Dominant | Fraction | Broadening |
| --- | --- | --- | --- |
| `Ag/Ag` | Ag | 0.976 | 1.0% |
| `Au/Au_1foil` | Au | 1.000 | 1.0% |
| `In/In` | In | 1.000 | 1.0% |
| `Zn/Zn` | Zn | 1.000 | 1.0% |
| `Cu/Cu` | Cu | 0.982 | 0.2% |
| `Nb/Nb_disk_right` | Nb | 1.000 | 0.5% |
| `Fe/Fe_7f` | Ag | 0.680 | 3.0% |

The iron region is the one that was already known to hold only noise, so
there is nothing there for the fit to find.

### Multi-element standards

The same fit can be applied to the multi-element standards. The Havar
standard is a Co/Cr/Ni/W/Mo/Mn/C/Be/Fe alloy, and the SrTiO standard is
strontium titanate. Their selection workspaces are not part of the
single-element dictionary, so they are fitted directly:

````python
import numpy as np
from pathlib import Path
from mantid.simpleapi import Load

from data_loading.load_and_normalise import convert_TOF_to_energy
from data_manipulation.fit_endf import fit_region
from endf_database_util.templates import build_templates

base = Path("../NPRAISE - standards/NPRAISE - standards/Normalised nGEM data/2Dmaps")
workspace = Load(
    str(base / "Havar/Havar_5foils_div_Empty_normMonitor_selection.nxs"),
    OutputWorkspace="Havar_selection",
    StoreInADS=False,
)

profiles = []
for l2 in [0.62, 0.63, 0.64, 0.65]:
    converted = convert_TOF_to_energy(workspace, l2=l2, output_name=f"havar_{l2:g}")
    profiles.append(converted.extractY().mean(axis=0))
    x_bins = converted.readX(0).copy()

profile = np.mean(profiles, axis=0)
energies = np.sqrt(x_bins[:-1] * x_bins[1:]) / 1000.0
elements = ["Co", "Cr", "Ni", "W", "Mo", "Mn", "C", "Be", "Fe"]
templates = build_templates(elements, energies)

result = fit_region(profile, elements, templates=templates)
print(result.dominant)          # 'Co'
print(result.excluded)          # ['C', 'Be']
````

The result for Havar is:

| Element | Fitted mass % | Wikipedia mass % |
| --- | --- | --- |
| Co | 87.4 | 42.0 |
| W | 6.1 | 2.7 |
| Mo | 3.3 | 2.2 |
| Mn | 3.2 | 1.6 |
| Cr | 0.0 | 19.5 |
| Ni | 0.0 | 12.7 |
| Fe | 0.0 | 19.05 |
| C | not identifiable | 0.2 |
| Be | not identifiable | 0.05 |

Cobalt is found, and it dominates because the **Co-59 resonance at 132 eV is
very wide** (Γt = 5.74 eV), so it produces the broad absorption band that
peaks at ~130 eV in the measured profile.

The other elements are not found because **they have no resolved resonances
between 1 and 500 eV**. Chromium, nickel and iron have only a smooth
continuum there (about 2.1, 0.75 and 0.67 barns, nearly flat), and their
templates are too similar to each other to be told apart. Carbon and
beryllium have a cross-section of exactly zero, so they carry no information
at all; the fit leaves them out and reports them separately.

The same applies to SrTiO: the **Sr resonance at 3.52 eV** matches the
observed dip at 3.52 eV exactly, so strontium is found, but titanium has no
resonances in this range and oxygen has a zero cross-section.

### Saturated dips

A deep dip does not reach zero transmission: it is limited by the background
and by the detector, so it flattens out at some level above zero. The fit
therefore has a floor term,

````text
T(E) = floor + (1 - floor) * exp(-sum_i c_i * sigma_i(E))
````

The floor is estimated from the profile: it is the smallest transmission,
but only when the bottom of the dip is genuinely flat, so that a sharp
unsaturated dip is not mistaken for a saturated one. It can also be set
explicitly with the `floor` argument or the `--floor` option.

The result carries a `saturated` flag, which is set when the floor is
non-zero. **When the dip is saturated, the depth of the dip carries no
information about the amount of material**, so the fitted abundances are
not quantitative. Only the wings of the dip, where the transmission is
still well above the floor, carry the amplitude information.

There are three ways to deal with this, in increasing order of reliability.

**Leave the saturated bins out.** The `unsaturated_only` argument (the
`--unsaturated-only` option) leaves out the bins that are pinned at the
floor, and fits only the wings. The result reports how many bins were used
in `unsaturated_bins`.

**Fit the resonances and the continuum separately.** The `separate_continuum`
argument (the `--separate-continuum` option) splits each cross-section into
its narrow resonances and its smooth continuum, and fits them with separate
coefficients. This is because the wings of a resonance look exactly like a
continuum, so when the two are fitted together the amplitude of the
resonance trades off against the continuum. The result then reports
`resonance_areal_densities` and `continuum_areal_densities` separately.

**Constrain the composition.** The `fixed_composition` argument (the
`--composition` option) holds the composition of the sample fixed and fits
only the overall amount of material. This is the most reliable of the three,
because the shape of the dip is then fixed and only its amplitude is
fitted, so the saturated bottom cannot bias the answer.

> [!IMPORTANT]
> The Havar Co dip is saturated: the profile bottoms out at a transmission
> of 0.0405 over 119–144 eV, and the Co resonance implies a peak optical
> thickness of about 96, far above the 3.2 that the measurement can
> resolve. The 87 % cobalt figure from a free fit is therefore **not** a
> measurement of the cobalt content.
>
> With the composition held fixed at the known values, the fit recovers
> the cobalt areal density exactly: **N·t_Co = 0.007140** atoms per barn,
> against 0.007140 from the reference fit. The total areal density comes
> out at 0.017152 against 0.01715. The free fit gives 0.007676, which is
> 7.5 % high.
>
> The resonance-plus-continuum split is *not* the answer here. It is
> unbiased on synthetic data, but on the real Havar data it is 32 % off,
> because the measured band is about four times broader than the ENDF
> cobalt resonance. The split model has a genuine degeneracy that the
> free fit exploits.

### Sample metadata: how much of each element

The free fit answers *which elements are present*. The metadata is what
turns it into *how much of each element*, because the depth of a dip gives
the product of the amount of material and the cross-section, and only the
metadata breaks that degeneracy.

`data_manipulation/sample_metadata.py` holds that metadata: the thickness
of each sample, its composition, its density and its measured floor. It
can be read from a YAML file or from the Excel spreadsheet of the
standards, and `configs/standards_metadata.yaml` holds the metadata of the
standards as it is currently known.

The amount of material is worked out from the thickness, the density of
the sample and the mean molar mass of its composition:

````text
mass per unit area  = density * thickness
mean molar mass      = sum_i (number fraction_i * atomic mass_i)
atoms per barn       = mass per unit area / mean molar mass * N_A / 1e24
````

The density of a mixture is the mass weighted harmonic mean of the
densities of its elements, which is what the code uses when the density is
not given.

#### Three modes

| Mode | Needs | Answers | Frees |
| --- | --- | --- | --- |
| **Discovery** (the default) | nothing | which elements | everything |
| **Fixed amount** | thickness + composition | the relative abundance | the proportions |
| **Fixed composition** | the composition | the amount of material | the amount |

The **fixed amount** mode is the one that answers "the thickness and the
elements are known, but what are the ratios?". The amount of material is
pinned by the metadata, and the fit finds the proportions on the simplex,
so they always sum to one and are all positive. Because the depth of the
dip is fixed rather than fitted, the proportions are determined by the
*shape* of the dip, which is far more robust when the dip is saturated.

````python
from data_manipulation.fit_endf import fit_region
from data_manipulation.sample_metadata import load_metadata

metadata = load_metadata("configs/standards_metadata.yaml")
havar = metadata["Havar/Havar_5foils"]

result = fit_region(
    profile, ["Co", "Cr", "Ni", "W", "Mo", "Mn", "Fe"],
    templates=templates,
    fixed_amount=havar,
)
print(dict(zip(result.symbols, result.mass_fractions)))
````

From the command line:

````bash
python -m data_manipulation.fit_endf \
    --dictionary dictionary_all_elements.npz \
    --elements Ag Au In Zn Cu Nb Fe \
    --metadata configs/standards_metadata.yaml \
    --fixed-amount \
    --output endf_fits.csv
````

Regions with no metadata are skipped rather than being an error. The CSV
gains a `mass_fraction_<element>` column for each element.

> [!NOTE]
> The **fractions** are the shares of the modelled optical thickness. The
> **mass fractions** are the shares by mass, which is the usual way a
> composition is quoted. They are different things, because the elements
> have very different cross-sections. Both are reported.

> [!IMPORTANT]
> The metadata is only as good as the sample it describes, and some of the
> standards do not agree with themselves. The gold foil is the clearest
> case: its mass and area imply a thickness of 0.27 mm, but the stated
> thickness is 0.02 mm, a factor of 13.7 apart. The `Measured std` sheet
> also marks `Phosp. Bronze` and `SnPbSb` as **"not realiable"**, so no
> metadata is given for them.
>
> The absolute calibration is still not good enough to quote an areal
> density to better than about 10–20 %. On the single-element foils the
> fitted areal density is 0.84 to 0.95 of the value the metadata
> predicts, and the shortfall is worst for the thickest samples. The
> dominant cause is the energy resolution: the time-of-flight binning
> gives a resolution that grows as the square root of the energy, and the
> fit has to approximate it. The code tries the calculated flight-path
> resolution alongside the constant broadenings, which improves the
> calibration, but it does not remove it.

### Isotopes

A sample does not have to hold the natural isotopic composition of an
element. Enriched material, and material made from separated isotopes, are
both common, and **the resonances of the isotopes of an element sit at
different energies**, so they can be told apart.

This is what makes the technique more than an elemental analysis. The
resonances of the two stable isotopes of silver, for example, do not
overlap at all:

| Isotope | Strongest resonances (eV) | Peak cross-section |
| --- | --- | --- |
| Ag-107 | 16.3, 51.6, 202.4 | 5 558 barns |
| Ag-109 | 5.19, 30.6, 71.0, 133.9 | 34 197 barns |

The isotopes are fitted by giving their names instead of the element name:

```python
from endf_database_util.templates import build_fine_isotope_templates
from data_manipulation.fit_endf import fit_region

templates = build_fine_isotope_templates(["Ag-107", "Ag-109"], energies)

result = fit_region(
    profile, ["Ag-107", "Ag-109"], templates=templates
)
print(result.atomic_fractions)   # the share of the atoms
print(result.mass_fractions)     # the share by mass
print(result.isotopes)           # {'Ag-107': 107, 'Ag-109': 109}
```

From the command line:

````bash
python -m data_manipulation.fit_endf \
    --dictionary dictionary_all_elements.npz \
    --isotopes Ag-107 Ag-109 \
    --output isotope_fits.csv
````

The CSV gains an `atomic_fraction_<isotope>` column for each isotope.

> [!NOTE]
> The **atomic fractions** are the share of the atoms, which is the usual
> way an isotopic composition is quoted. The **mass fractions** are the
> share by mass. The `fractions` are shares of the modelled optical
> thickness, which is a third thing and is usually not what you want for
> isotopes.

#### The silver flake test

The `Ag_Pb_Isotopes_normMonitor.nxs` standard holds flakes of the two
separated silver isotopes. Fitting the two connected regions of the
sample gives:

| Region | Ag-107 (atomic) | Ag-109 (atomic) |
| --- | --- | --- |
| the larger flake | 92.2 % | 7.8 % |
| the smaller flake | 0 % | 100 % |

The two flakes are cleanly separated, which is the point of the test: the
same element, the same energy range, but told apart by the position of
the resonances rather than by the element itself.

> [!IMPORTANT]
> The isotopes can only be told apart when their resonances do not
> overlap. That is a property of the element, not of the code, and it
> does not hold for every element. The isotopes of silver are an
> unusually good case.

### The sample metadata

The file `Standard with nGEM.xlsx`, next to the data, records the
thickness and the composition of every standard. The `Details` sheet gives
the mass, the thickness, the area and the stable isotopes of each sample,
and the `Measured std` sheet gives the thickness in millimetres and the
number of foils. This is what makes the fits quantitative, because the
expected areal density of each element can be worked out from the
thickness and the density.

| Sample | Composition | Thickness |
| --- | --- | --- |
| Au foil | Au-197 (100%) | 0.02 mm (1 foil), 0.04 mm (2 foils) |
| Brass | Cu-63 (69.15%), Cu-65 (30.85%) | 0.1, 0.6, 1.2, 2.0 mm |
| Bronze sheet | Cu 94%, Sn 6% | 2 mm |
| Havar | Co 42.5%, Cr 20%, Ni 13%, Fe, W, Mo, Mn | 2.5 mm (5 foils) |
| Fe | Fe 99.5% | 0.25 mm (1 foil), 1.75 mm (7 foils) |
| Zn | Zn | 2 mm |
| Nb | Nb 99.9% | 0.25 mm (1 disk), 0.75 mm (3 disks) |
| In | In-113 (4.29%), In-115 (95.71%) | 0.2 mm |
| SrTiO | Sr, Ti, O | 2.14 g of powder |
| SnPbSb | Sn 60%, Pb 39%, Sb 1% | 0.1 mm |

The fit reproduces the known composition of the phosphor bronze sheet to
within a few per cent: **Cu 94.0% and Sn 6.0%**, against the nominal
Cu 94 / Sn 6.

> [!NOTE]
> The thickness series are the way to deal with a saturated dip. A dip
> that is clipped at the floor carries no information about the amount of
> material, but the same sample measured at a thinner setting does. The
> gold foils are the clearest case: the one-foil measurement bottoms out
> at a transmission of 0.071, and the two-foil measurement at 0.038, so
> the floor can be solved for from the pair.

> [!WARNING]
> Two of the files are duplicates of each other: `Brass_5f_normMonitor_
> selection.nxs` and `Brass_20f_normMonitor_selection.nxs` hold exactly
> the same data, so the 20-foil brass is not actually measured.

### What the energy range can and cannot do

The energy range of the data limits which elements can be identified, but
**extending it above 500 eV does not help**. The 500 eV limit is only the
rebin parameter (`"1000,-0.005,500000"`); the raw data spans 0.718 eV to
2.87 MeV with good statistics. Fitting the Havar standard at 500, 1000,
2000, 3000 eV and at 2.0 and 2.9 MeV gives the same answer every time:
cobalt at about 88 %, and no chromium, nickel or iron at all.

The reason is that **chromium, nickel and iron have no resonances at all
between 500 and 1000 eV**. Their first resonances are at 1151 eV (Fe),
1290 eV (Ni) and 1626 eV (Cr), and by then the detector's energy
resolution has degraded too far to resolve them. The nGEM flight path is
22.17 m with 1 µs time bins, so the relative resolution is

````text
dE/E = 2 * dt / t = 2 * dt * sqrt(E / (2 * m * L))
````

which is 0.0012 at 1 eV but 0.028 at 500 eV and 0.050 at 1626 eV. The
visibility of a resonance, the ratio of its width to the resolution, is
therefore:

| Element | Resonance | Γt/E | dE/E | Visibility |
| --- | --- | --- | --- | --- |
| Ag | 5.2 eV | 0.359 | 0.0012 | 300 |
| Au | 4.5 eV | 0.323 | 0.0012 | 270 |
| Mn | 340.8 eV | 3.068 | 0.023 | 133 |
| Co | 132.0 eV | 3.033 | 0.013 | 233 |
| Fe | 1151 eV | 0.013 | 0.042 | 0.31 |
| Ni | 1290 eV | 0.009 | 0.044 | 0.20 |
| Cr | 1626 eV | 0.004 | 0.050 | 0.08 |

The resolution gets *worse* as the energy rises, because the flight time
grows as the square root of the energy while the time bins stay fixed. The
chromium, nickel and iron resonances above 500 eV are 25 to 250 times too
narrow to be seen.

The elements with usable resonances between 1 and 500 eV are Ag, Au, In,
Zn, Cu, Nb, Co, W, Mo, Mn and Sr. The others need a different measurement
rather than a different energy range: a better-resolution instrument, a
different modality, or the measured-dictionary route of
`fit_abundances.py`.

> [!NOTE]
> The low-energy region does carry different information, because the
> resonances of tungsten and molybdenum are dense there. Fitting the Havar
> standard over 1–80 eV gives cobalt 66 %, tungsten 20 % and molybdenum
> 15 %, where the full range gives cobalt 88 %. Chromium, nickel and iron
> remain at zero in every case, because their 1/v continua are too similar
> to each other to be told apart: the chromium and nickel templates
> correlate at +0.57 over the whole range and at +0.999 below 20 eV.

### Finding what is in an unknown sample

The last sections assume that somebody already knows which elements are in
the sample. When the sample is genuinely unknown, the fit has to decide
that for itself.

**The energy range does the pruning.** A nuclide that has no resonance
between 1 and 500 eV leaves no trace in the spectrum, so it cannot be
found and does not need to be fitted for. Scanning the ENDF database for
the nuclides that do have a resonance in the range leaves **60 elements
with a natural abundance** (and 191 isotopes). This is the whole candidate
list, and it is small enough to fit all at once.

`endf_database_util/discovery.py` performs the scan:

````python
from endf_database_util.discovery import scan_database

result = scan_database(1.0, 500.0)
print(result.element_count)    # 60
print(result.isotope_count)    # 191
print(result.resonance_count)  # 12045
print(result.elements[:10])
````

The scan reads every file in the database and takes about ninety seconds.
It is cached on disk, so it is only paid once, and afterwards the same
call takes about twenty milliseconds.

The candidates are the elements that have a resonance in the range **and**
a natural abundance. The second condition removes the elements that could
never be in a natural sample: americium, berkelium, californium,
curium, neptunium, promethium, polonium, plutonium, radium and
technetium. It also removes sodium, nickel and yttrium, whose only
resonances in the range belong to the radioactive isotopes sodium-22,
nickel-59 and yttrium-90.

The command line does the whole thing in one step:

````shell
python -m data_manipulation.fit_endf \
    --workspace "Havar/Havar_5foils_div_Empty_normMonitor_selection.nxs" \
    --discover \
    --output havar_discovery.csv
````

`--discover` fits every candidate element, and `--discover-isotopes` fits
every candidate isotope instead. On the Havar standard the element route
finds cobalt at 92 %, with tungsten, manganese and molybdenum as the
smaller components, and finds nothing for chromium, nickel or iron, which
is the correct answer: none of them has a resonance in the range.

> [!NOTE]
> The full isotope route is also practical. Fitting all 191 candidate
> isotopes at once takes about ten seconds, and all 366 nuclides with a
> resonance in the range take about twenty. The templates of the
> candidates are well separated from each other, so the fit is not
> degenerate.

> [!IMPORTANT]
> An element that has no resonance in the measured range cannot be found,
> and it cannot be ruled out either. Its smooth 1/v continuum can hide
> behind that of another element, and the fit gives it a share of zero
> rather than saying that it is absent. This is a limitation of the
> measurement, not of the fit, and it is why the spectrum below 500 eV
> cannot tell chromium, nickel and iron apart.

#### Having a resonance is necessary, but not sufficient

A resonance in the measured range is only the *first* requirement. The
resonance also has to survive the resolution of the instrument, and it
must not be blended with the resonance of another element.

The nGEM flight path is 22.17 m with 1 µs time bins, so the relative
resolution is 0.00125 at 1 eV, 0.0189 at 230 eV and 0.0274 at 480 eV.
A resonance much narrower than that is smeared away. The useful measure
is the **modulation of the resonance above the local continuum, after the
instrument has smeared it**:

| Element | Resonance | Peak σ | Modulation | Findable? |
| --- | --- | --- | --- | --- |
| Sn | 38.8 eV | 1915 b | 75 % | yes |
| Ag | 5.2 eV | 16472 b | 73 % | yes |
| Cu | 230.0 eV | 80.8 b | 67 % | yes, but see below |
| Au | 4.9 eV | 36821 b | 63 % | yes |
| Zn | 223.1 eV | 199 b | 49 % | yes |
| Fe | 230.0 eV | 1.6 b | 33 % | marginally |
| Co | 132.0 eV | 10537 b | 3 % | yes |
| Pb | 480.0 eV | 10.9 b | 3 % | **no** |
| Mn | 340.8 eV | 3254 b | 1 % | **no** |

Lead is the clearest example of a resonance that is present but useless.
Lead-204 has one resonance in the range, at 480 eV, but the cross-section
there is 10.87 b against a continuum of 10.18 b, so the resonance only
modulates the cross-section by 6.8 %. The feature is 0.40 eV wide and the
resolution at 480 eV is 13.1 eV, so after smearing the modulation is
**0.1 %**. A simulated sample of pure lead cannot be recovered at all.

Copper is the interesting middle case. Its resonance at 230.04 eV is
strongly visible, and the copper-bronze standard shows a clean dip at
230.19 eV that is 12.8 % deep, which the fit identifies as 12.6 % copper.
But in the brass standard the copper is *not* found, because zinc-67 has
a resonance at 223.11 eV and the two are only 6.93 eV apart, against a
resolution of 4.35 eV. They merge into a single feature that bottoms out
at 226 eV, and the fit cannot separate them. Fitting the resonances and
the continuum separately does recover copper in brass, with a resonance
areal density of 0.0157 against 0.0049 for zinc.

> [!WARNING]
> The separated fit reports a large lead abundance in every sample. That
> is an artefact: the "resonance part" of the lead cross-section is
> essentially flat, so a large coefficient on it produces almost no
> optical thickness and is left unconstrained. The plain fit does not
> have this problem.
