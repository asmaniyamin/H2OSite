<!--
Copyright 2026 Asmani Yamin
SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0
-->

# H2OSite

Mapping binding-site water positions and energetics to guide ligand design and optimization.

H2OSite estimates the thermodynamic properties of water in protein binding
pockets. It combines molecular dynamics, hydration-site clustering, and
per-site analysis of water interactions, entropy, and hydrogen bonding.
Results can be exported for visualization or used to score how a ligand pose
overlaps hydration sites.

## How it works

1. Prepare the system. Load a protein and ligand, identify the binding
   pocket, handle existing waters, and optionally truncate the protein. Build
   a solvated OpenMM system with the configured force fields.
2. Equilibrate and sample water. Run restrained equilibration and
   GCMC pocket solvation, followed by production MD. Track water molecules
   within the selected pocket.
3. Identify hydration sites. Build a water-density grid and cluster sampled
   water positions into occupied sites.
4. Analyze and export. Estimate interaction energies and entropy, calculate
   hydrogen-bond statistics, and write site tables, structures, density maps,
   and visualization scripts.

Ligands define the pocket in both run modes. Holo retains the ligand during
simulation; apo excludes it after preparation. The default force-field
settings use AMBER14 and TIP3P, with OpenFF parameters for retained ligands.

## Requirements and setup

Use Python 3.10 or later with a compatible scientific environment containing:

- NumPy, SciPy, scikit-learn, Click, tqdm, and PyYAML.
- OpenMM, MDAnalysis, MDTraj, ParmEd, and RDKit.
- OpenFF Toolkit, OpenFF force fields, and `openmmforcefields` for ligand
  parameterization.
- Compatible loch, Sire, BioSimSpace, and GPU runtime packages for the full
  GCMC backend.
- Pytest to run the test suite.

The source directory is the Python package. Clone or place it in a directory
named `H2OSite`, activate the scientific environment, and run commands from
its parent directory. If currently inside the repository, run `cd ..` first.
Alternatively, add that parent directory to `PYTHONPATH`.

```sh
python -m H2OSite.cli --help
python -m H2OSite.cli run --help
```

## Quick start: 5HXB example

The included example provides a prepared protein with an embedded ligand,
production settings, and residue-specific histidine assignments.

Validate input paths and inspect settings without starting MD:

```sh
python -m H2OSite.cli run \
  --config H2OSite/docs/examples/configs/5hxb_production.yaml --dry-run
```

Run the configuration by removing `--dry-run`:

```sh
python -m H2OSite.cli run \
  --config H2OSite/docs/examples/configs/5hxb_production.yaml
```

This example uses 2 ns production and writes holo results to
`results/5hxb_production_holo/`. A dry run checks configuration and input paths;
it does not validate force-field parameterization or execute the scientific
runtime. See the [example guide](docs/README.md) for input details and additional
commands.

## Run another protein

Provide a prepared protein in PDB or mmCIF format and a ligand in the same
coordinate frame, such as an SDF file:

```sh
python -m H2OSite.cli run \
  --protein protein.pdb --ligand ligand.sdf --mode holo \
  --jobname example --output-dir results/example --sim-time 5 --dry-run
```

For a ligand already embedded in the protein structure, replace `--ligand`
with its residue name:

```sh
python -m H2OSite.cli run \
  --protein complex.cif --ligand-resname UNK --mode apo \
  --jobname example --output-dir results/example --dry-run
```

Remove `--dry-run` when ready to execute. Inputs should have the required atoms,
protonation states, and terminal treatment for the chosen force field.

For reusable settings, copy an example YAML and change its input path, ligand
identifier, preparation options, and output directory. The shared workflow
handles setup and execution without custom Python functions. Command-line
options override YAML settings.

Input paths in YAML are relative to the YAML file. Output paths and explicit
command-line paths are relative to the working directory. The `{mode}`
placeholder in YAML output paths separates apo and holo runs.

Useful run options include:

| Option | Purpose |
| --- | --- |
| `--mode apo` / `--mode holo` | Exclude or retain the ligand during simulation |
| `--sim-time 5` | Set production duration in nanoseconds |
| `--preset production` | Use configured simulation stage lengths |
| `--preset short` | Cap production at 20 ps and reduce GCMC/post-GCMC stages |
| `--preset smoke` | Cap production at 2 ps and shorten default stages for execution checks |
| `--restraint-k 10` | Set shared restraint strength in kcal/mol/A^2 |
| `--gpu 0`, `--num-cpus 2` | Select compute resources |
| `--seed 137` | Set the random seed |
| `--skip-gcmc` | Bypass GCMC pocket solvation |

Short and smoke presets cap production duration even if `--sim-time` 
requests more time. They are intended for execution checks. Use separate 
output directories for independent runs; the pipeline can resume from checkpoints in an existing run directory.

## Inspect results

For a run with `--jobname example`, the output directory contains:

| Output | Contents |
| --- | --- |
| `example_wm.csv` | Site coordinates, occupancy, thermodynamic estimates, and hydrogen-bond statistics |
| `example_wm.pdb`, `example_wm.cif` | Hydration-site structures |
| `example_wm.pml` | PyMOL visualization script |
| `example-water.cns` | Water-density grid |
| `example-continuous_cavity.cns` | Cavity map, when available |
| `prep/`, `checkpoints/`, `production/` | Prepared inputs, restart data, and production outputs |

Site CSVs include the water model, bulk-water reference energy, analysis
temperature, and separate entropy components. Scoring and format conversion
use these recorded values. CSVs without model metadata retain their saved
enthalpy and entropy totals without assuming a model or entropy decomposition.

After a completed run, generate centered structures, trajectory views, and a
`fixed/` visualization bundle:

```sh
python -m H2OSite.postprocessing results/example --tracking-center 0 0 0
```

Use the actual production tracking center, in Angstrom. Add `--reanalyze` to
repeat clustering, thermodynamic analysis, and hydrogen-bond analysis from the
saved trajectory. Reanalysis preserves the original analysis files and writes
`*_fixed` outputs. Use `--help` for checkpoint, trajectory, and clustering options.

For reanalysis, specify the water model used during production:

```sh
python -m H2OSite.postprocessing results/example --reanalyze --water-model tip3p
```

Use `tip4p` or `tip4pew` when appropriate. Reanalysis recomputes the analysis
and bulk reference from the trajectory; converting an existing CSV preserves
its recorded thermodynamic values.

## Score a ligand or convert sites

The displacement scorer estimates how much of each hydration site overlaps a
ligand pose and weights that overlap by the site's thermodynamic properties.
Keep the ligand and site coordinates in the same frame.

```sh
python -m H2OSite.cli score \
  --sites results/example/example_wm.csv --ligand ligand.sdf

python -m H2OSite.cli convert \
  --sites results/example/example_wm.csv --output sites.cif --format cif
```

## Python API

Load and run the same YAML configuration from Python:

```python
from H2OSite.workflow import load_workflow, run_workflow

config = load_workflow("H2OSite/docs/examples/configs/5hxb_production.yaml")
run_workflow(config, dry_run=True)
# Set dry_run=False to execute and return the hydration sites.
```

For programmatic configuration, use `WaterMapConfig` from `H2OSite.core.config`
and pass it through `configure_run` in `H2OSite.workflow` before execution.

## Tests

From the directory containing the `H2OSite` package:

```sh
python -m pytest H2OSite/tests -m 'not slow and not integration'
```

Tests requiring unavailable scientific dependencies are skipped where marked.
Full simulation tests require the corresponding runtime and compute hardware.

## License and attribution

Copyright 2026 Asmani Yamin.

H2OSite is source-available under the [H2OSite Noncommercial Research License
1.0](LICENSE). Unmodified noncommercial use and distribution are permitted
with the required notices. Private modifications are permitted solely for
noncommercial research within a controlled research environment.

Commercial use and distribution, publication, or disclosure of modified
versions require Asmani Yamin's prior written permission. Retain [NOTICE](NOTICE)
and the license with distributed copies. Publicly shared results must
acknowledge H2OSite by Asmani Yamin; publications must cite the Software using
[CITATION.cff](CITATION.cff).

Users are responsible for independently validating structural preparation,
force fields, water models, sampling, thermodynamic references, and outputs
before relying on results. Computed hydration sites and scores do not by
themselves establish binding affinity, biological activity, safety, or efficacy.
The license sets out the research-use conditions, warranty exclusions,
liability limitations, indemnification, and termination provisions.

Third-party dependencies and example data retain their applicable terms.
Statutory rights and valid earlier license grants remain unaffected. This
summary does not expand or replace the full [LICENSE](LICENSE).
