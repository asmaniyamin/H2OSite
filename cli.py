# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
CLI entry point for H2OSite.

Usage:
    python -m H2OSite.cli run --protein protein.pdb --ligand ligand.sdf [options]
    python -m H2OSite.cli analyze --water-positions water.npy --n-frames 1333 [options]
    python -m H2OSite.cli score --sites sites.csv --ligand ligand.sdf
    python -m H2OSite.cli convert --sites sites.csv --output output.pdb
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from H2OSite.core.config import WaterMapConfig


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def main(verbose: bool):
    """H2OSite - thermodynamic water profiling for noncommercial research."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--protein", "-p", type=click.Path(exists=True, dir_okay=False))
@click.option("--ligand", "-l", type=click.Path(exists=True, dir_okay=False))
@click.option("--ligand-resname", multiple=True, help="Embedded ligand residue name; repeat for alternatives.")
@click.option("--output-dir", "-o", default=None)
@click.option("--jobname", "-j", default=None)
@click.option("--sim-time", type=float, default=None, help="Production MD time in ns")
@click.option("--ligand-distance", type=float, default=None, help="Active site radius in Angstrom")
@click.option("--water-model", default=None)
@click.option("--gpu", default=None, help="GPU device index")
@click.option("--num-cpus", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--mode", type=click.Choice(["apo", "holo"]), default=None)
@click.option("--preset", type=click.Choice(["production", "short", "smoke"]), default=None)
@click.option("--restraint-k", type=float, default=None, help="Restraint strength in kcal/mol/A^2")
@click.option("--skip-gcmc", is_flag=True, default=None, help="Skip GCMC pocket solvation")
@click.option("--twobody", is_flag=True, default=None, help="Compute two-body entropy")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False), help="Run YAML file")
@click.option("--dry-run", is_flag=True, help="Validate and show settings without starting MD")
def run(protein, ligand, ligand_resname, output_dir, jobname, sim_time, ligand_distance,
        water_model, gpu, num_cpus, seed, mode, preset, restraint_k, skip_gcmc,
        twobody, config, dry_run):
    """Run a protein from inputs or a reusable YAML configuration."""
    import yaml
    from H2OSite.workflow import configure_run, load_workflow, run_workflow

    overrides = dict(
        protein_path=protein, ligand_path=ligand, output_dir=output_dir,
        jobname=jobname, simulation_time_ns=sim_time, ligand_distance_A=ligand_distance,
        water_model=water_model, gpu_index=gpu, num_cpus=num_cpus, random_seed=seed,
        skip_gcmc=skip_gcmc, twobody_entropy=twobody,
    )
    if ligand_resname:
        overrides["ligand_resnames"] = list(ligand_resname)
    options = dict(mode=mode, preset=preset, restraint_k_kcal=restraint_k)
    try:
        if config:
            cfg = load_workflow(config, overrides=overrides, workflow_overrides=options)
        else:
            values = {key: value for key, value in overrides.items() if value is not None}
            values.setdefault("simulation_time_ns", 2.0)
            cfg = configure_run(
                WaterMapConfig(**values),
                **{key: value for key, value in options.items() if value is not None},
            )
        run_workflow(cfg, dry_run=dry_run)
    except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option("--water-positions", "-w", required=True, type=click.Path(exists=True),
              help=".npy file of water O positions")
@click.option("--n-frames", "-n", type=int, default=None)
@click.option("--topology", "-t", type=click.Path(exists=True))
@click.option("--trajectory", "-r", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="watermap_analysis")
@click.option("--jobname", "-j", default="watermap")
@click.option("--hs-radius", default=1.0, help="Hydration site radius (Å)")
@click.option("--cutoff", default=2.0, help="Clustering distance cutoff (Å)")
@click.option("--water-model", default="tip3p", help="Water model used for the saved data")
def analyze(water_positions, n_frames, topology, trajectory,
            output_dir, jobname, hs_radius, cutoff, water_model):
    """Analyze pre-computed water positions (analysis-only mode)."""
    cfg = WaterMapConfig(
        output_dir=output_dir,
        jobname=jobname,
        hs_radius_A=hs_radius,
        clustering_cutoff_A=cutoff,
        water_model=water_model,
    )

    from H2OSite.pipeline import WaterMapPipeline
    pipeline = WaterMapPipeline(cfg)
    sites = pipeline.run_analysis_only(
        water_positions, n_frames, topology, trajectory
    )
    click.echo(f"Done: {len(sites)} hydration sites → {output_dir}/")


@main.command()
@click.option("--sites", "-s", required=True, type=click.Path(exists=True),
              help="CSV with hydration site data")
@click.option("--ligand", "-l", required=True, type=click.Path(exists=True))
@click.option("--scale-factor", default=0.8)
def score(sites, ligand, scale_factor):
    """Score a ligand against pre-computed hydration sites."""
    import numpy as np
    from H2OSite.core.structure import load_structure
    from H2OSite.scoring.displacement import DisplacementScorer
    from H2OSite.analysis.clustering import HydrationSite

    lig = load_structure(ligand)
    site_list = _load_sites_csv(sites)

    scorer = DisplacementScorer(scale_factor=scale_factor)
    results = scorer.score(site_list, lig.positions, lig.elements)

    total_dG = sum(r.dG for r in results)
    click.echo(f"Total displacement dG: {total_dG:.2f} kcal/mol")
    for r in results:
        if abs(r.overlap) > 0.01:
            click.echo(
                f"  Site {r.site_index}: overlap={r.overlap:.2f} "
                f"dG={r.dG:.2f} dH={r.dH:.2f} -TdS={r.minus_TdS:.2f}"
            )


@main.command()
@click.option("--sites", "-s", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
@click.option("--format", "-f", "fmt", type=click.Choice(["pdb", "cif", "pymol", "csv"]), default="pdb")
@click.option("--protein", "-p", type=click.Path(exists=True))
@click.option("--ligand", "-l", type=click.Path(exists=True))
def convert(sites, output, fmt, protein, ligand):
    """Convert hydration site data to various output formats."""
    from H2OSite.io.converters import (
        watermap_to_cif,
        watermap_to_csv,
        watermap_to_pdb,
        watermap_to_pymol,
    )

    site_list = _load_sites_csv(sites)

    if fmt == "pdb":
        watermap_to_pdb(site_list, output, protein, ligand)
    elif fmt == "cif":
        watermap_to_cif(site_list, output)
    elif fmt == "pymol":
        watermap_to_pymol(site_list, output, protein, ligand)
    elif fmt == "csv":
        watermap_to_csv(site_list, output)

    click.echo(f"Written to {output}")


def _load_sites_csv(path: str) -> list:
    """Load hydration sites for scoring and format conversion."""
    from H2OSite.io.converters import load_sites_csv

    return load_sites_csv(path)


@main.command(name="reanalyze-trajectory")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Checkpoint .npz with OpenMM topology/system (e.g. post_gcmc_equil.npz)")
@click.option("--trajectory", "-r", required=True, type=click.Path(exists=True),
              help="Production trajectory (DCD/XTC/...)")
@click.option("--production-checkpoint", type=click.Path(exists=True),
              help="Optional final production checkpoint.npz to append the last frame")
@click.option("--output-dir", "-o", default="watermap_reanalysis")
@click.option("--jobname", "-j", default="watermap_reanalysis")
@click.option("--center", nargs=3, type=float, default=(0.0, 0.0, 0.0),
              help="Active-site center in Angstrom")
@click.option("--radius", default=10.0, help="Active-site radius in Angstrom")
@click.option("--hs-radius", default=1.0, help="Hydration site radius (Angstrom)")
@click.option("--cutoff", default=2.0, help="Clustering distance cutoff (Angstrom)")
@click.option("--water-model", default="tip3p", help="Water model used during production")
def reanalyze_trajectory(
    checkpoint,
    trajectory,
    production_checkpoint,
    output_dir,
    jobname,
    center,
    radius,
    hs_radius,
    cutoff,
    water_model,
):
    """Re-run analysis stages from an existing trajectory and checkpoint."""
    import numpy as np

    cfg = WaterMapConfig(
        output_dir=output_dir,
        jobname=jobname,
        ligand_distance_A=radius,
        hs_radius_A=hs_radius,
        clustering_cutoff_A=cutoff,
        water_model=water_model,
    )

    from H2OSite.pipeline import WaterMapPipeline
    pipeline = WaterMapPipeline(cfg)
    sites = pipeline.run_analysis_from_trajectory(
        checkpoint_file=checkpoint,
        trajectory_file=trajectory,
        production_checkpoint_file=production_checkpoint,
        active_site_center_A=np.array(center, dtype=float),
        active_site_radius_A=radius,
    )
    click.echo(f"Done: {len(sites)} hydration sites → {output_dir}/")


if __name__ == "__main__":
    main()
