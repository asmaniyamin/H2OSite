<!--
Copyright 2026 Asmani Yamin
SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0
-->

# 5HXB production workflow

Activate a Python 3.10+ environment with the scientific dependencies listed in
the [project README](../README.md).

The commands below assume the repository was cloned into a directory named
`H2OSite` and are run from its parent directory. If you are currently in the
repository root, run `cd ..` first so Python can import the `H2OSite` package.

Input paths in YAML are relative to the YAML file. Output paths are relative
to the directory where you run the command.

Inspect the configuration without creating outputs or starting MD:

```sh
python -m H2OSite.cli run --config H2OSite/docs/examples/configs/5hxb_production.yaml --dry-run
```

Run production (2 ns by default):

```sh
python -m H2OSite.cli run --config H2OSite/docs/examples/configs/5hxb_production.yaml
```

The example runner provides a shortcut to the same CLI:

```sh
python -m H2OSite.docs.examples.run_5hxb_production --dry-run
```

## Inputs and settings

- `examples/inputs/5hxb_prepped.pdb`: original prepared receptor with embedded
  ligand `85C`.
- `examples/configs/5hxb_production.yaml`: production and preparation settings.
- `examples/configs/5hxb_histidines.yaml`: residue-specific histidine assignments.

Apo/holo mode, restraint setup, cache paths, presets, and execution are handled
by `H2OSite/workflow.py`. No custom functions are needed in the example.
Command-line options override YAML settings:

```sh
python -m H2OSite.cli run --config H2OSite/docs/examples/configs/5hxb_production.yaml \
  --mode apo --sim-time 5 --seed 137 --output-dir results/5hxb_apo_seed137
```

The `{mode}` output-path placeholder separates apo and holo outputs. The default
holo output is `results/5hxb_production_holo`; the job stem is
`5hxb_holo_production`.

For Python use:

```python
from H2OSite.workflow import load_workflow, run_workflow

config = load_workflow("H2OSite/docs/examples/configs/5hxb_production.yaml")
run_workflow(config, dry_run=True)
```

## Postprocessing and scoring

After a completed production run:

```sh
python -m H2OSite.postprocessing results/5hxb_production_holo --tracking-center 0 0 0
```

Use the actual tracking center reported by the run. Add `--reanalyze` to repeat
analysis from saved MD into fixed outputs.

When reanalyzing this configuration, use `--water-model tip3p` to retain its
production water model. Site CSVs record the model and bulk-water reference
used for thermodynamic analysis and scoring.

```sh
python -m H2OSite.cli score \
  --sites results/5hxb_production_holo/5hxb_holo_production_wm.csv \
  --ligand results/5hxb_production_holo/prep/5hxb_holo_production_ligand.sdf
```
