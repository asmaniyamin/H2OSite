# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Shared setup and execution for protein hydration workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from H2OSite.core.config import WaterMapConfig


def _set_restraints(stages: list[dict], strength: float) -> list[dict]:
    result = deepcopy(stages)
    for stage in result:
        if stage.get("restrain"):
            stage["restraint_k"] = strength
    return result


def _set_stage_times(stages: list[dict], limits: dict[str, float]) -> list[dict]:
    result = deepcopy(stages)
    for stage in result:
        if stage.get("name") in limits:
            stage["time_ps"] = limits[stage["name"]]
    return result


def configure_run(
    config: WaterMapConfig,
    *,
    mode: str | None = None,
    preset: str = "production",
    restraint_k_kcal: float | None = None,
    cache: bool = True,
    normalize_residues: bool | None = None,
) -> WaterMapConfig:
    """Apply shared run settings to an independent copy of a configuration.

    Production preserves configured stage lengths. Short uses
    20 ps example; smoke reduces default stages and uses 2 ps production.
    """
    cfg = deepcopy(config)
    if mode not in (None, "apo", "holo"):
        raise ValueError("mode must be apo or holo")
    if preset not in ("production", "short", "smoke"):
        raise ValueError("preset must be production, short, or smoke")
    if mode is not None:
        cfg.retain_ligand = mode == "holo"
    selected_mode = "holo" if cfg.retain_ligand else "apo"
    cfg.output_dir = Path(str(cfg.output_dir).replace("{mode}", selected_mode))
    cfg.jobname = cfg.jobname.replace("{mode}", selected_mode)
    if normalize_residues is not None:
        cfg.normalize_schrodinger_residues = normalize_residues
    if restraint_k_kcal is not None:
        if restraint_k_kcal < 0:
            raise ValueError("Restraint strength must be nonnegative")
        cfg.gcmc_restraint_k_kcal = restraint_k_kcal
        cfg.production_restraint_k_kcal = restraint_k_kcal
        for field in ("equilibration_stages", "post_gcmc_equil_stages",
                      "gcmc_handoff_relaxation_stages"):
            setattr(cfg, field, _set_restraints(getattr(cfg, field), restraint_k_kcal))
    if cache:
        checkpoints = cfg.output_dir / "checkpoints"
        if cfg.gcmc_cache_dir is None:
            cfg.gcmc_cache_dir = checkpoints / "loch_sampler_cache"
        if cfg.ligand_parameter_cache is None:
            cfg.ligand_parameter_cache = checkpoints / "openff_ligand_cache.json"

    if preset == "short":
        cfg.simulation_time_ns = min(cfg.simulation_time_ns, 0.02)
        cfg.gcmc_n_moves = min(cfg.gcmc_n_moves, 5000)
        cfg.gcmc_n_cycles = min(cfg.gcmc_n_cycles, 25)
        cfg.gcmc_md_steps_per_cycle = 250
        cfg.gcmc_batch_size = 100
        cfg.post_gcmc_equil_stages = [
            dict(name="post_gcmc_relax", ensemble="NVT", integrator="Brownian",
                 temperature_K=10.0, time_ps=0.05, timestep_ps=0.001,
                 restraint_k=5.0, restrain="non_water_heavy", delta_max=0.05),
            dict(name="post_gcmc_fast", ensemble="NPT", integrator="Langevin",
                 temperature_K=300.0, time_ps=0.02, thermostat_tau_ps=0.1,
                 barostat_tau_ps=50.0, restraint_k=5.0, restrain="non_water_heavy"),
        ]
    elif preset == "smoke":
        cfg.simulation_time_ns = min(cfg.simulation_time_ns, 0.002)
        cfg.gcmc_n_moves = min(cfg.gcmc_n_moves, 40)
        cfg.gcmc_n_cycles = min(cfg.gcmc_n_cycles, 4)
        cfg.gcmc_md_steps_per_cycle = 2000
        cfg.generate_display_outputs = False
        cfg.equilibration_stages = _set_stage_times(cfg.equilibration_stages, {
            "brownian_nvt": 2.0, "langevin_nvt": 1.0,
            "langevin_npt_cold": 1.0, "mtk_npt": 2.0,
        })
        cfg.gcmc_handoff_relaxation_stages = _set_stage_times(
            cfg.gcmc_handoff_relaxation_stages, {"gcmc_handoff_relax": 0.2}
        )
        cfg.post_gcmc_equil_stages = _set_stage_times(
            cfg.post_gcmc_equil_stages, {"langevin_npt_post_gcmc": 2.0}
        )
    return cfg


def load_workflow(
    path: str | Path,
    *,
    overrides: dict | None = None,
    workflow_overrides: dict | None = None,
) -> WaterMapConfig:
    """Read a run YAML, resolving input paths relative to that file.

    Output paths are relative to the current directory. Explicit command-line
    input overrides are also relative to the current directory.
    """
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Run configuration must be a YAML mapping")
    options = data.pop("workflow", {})
    if not isinstance(options, dict):
        raise ValueError("workflow must be a YAML mapping")
    histidine_file = options.pop("histidine_states_file", None)
    if histidine_file:
        states_path = Path(histidine_file).expanduser()
        if not states_path.is_absolute():
            states_path = path.parent / states_path
        with states_path.open() as handle:
            states = yaml.safe_load(handle)
        if not isinstance(states, dict):
            raise ValueError("Histidine states file must contain a YAML mapping")
        if "manual_histidine_states" in data:
            raise ValueError("Use either manual_histidine_states or histidine_states_file")
        data["manual_histidine_states"] = states
    for field in ("protein_path", "ligand_path", "gcmc_cache_dir", "ligand_parameter_cache"):
        if data.get(field):
            value = Path(data[field]).expanduser()
            data[field] = value if value.is_absolute() else (path.parent / value).resolve()
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    options.update({k: v for k, v in (workflow_overrides or {}).items() if v is not None})
    return configure_run(WaterMapConfig(**data), **options)


def run_workflow(config: WaterMapConfig, *, dry_run: bool = False) -> list | None:
    """Validate, summarize, and run a configuration. Dry runs create no outputs."""
    cfg = config
    if not cfg.protein_path.is_file():
        raise ValueError(f"Protein input not found: {cfg.protein_path}")
    if cfg.ligand_path != Path():
        if not cfg.ligand_path.is_file():
            raise ValueError(f"Ligand input not found: {cfg.ligand_path}")
    elif not cfg.ligand_resnames:
        raise ValueError("Provide a ligand file or at least one embedded ligand residue name")
    if cfg.simulation_time_ns <= 0:
        raise ValueError("Production time must be positive")
    print(f"Job: {cfg.jobname}")
    print(f"Protein: {cfg.protein_path}")
    print(f"Ligand: {cfg.ligand_path if cfg.ligand_path != Path() else ', '.join(cfg.ligand_resnames)}")
    print(f"Mode: {'holo' if cfg.retain_ligand else 'apo'}")
    print(f"Output: {cfg.output_dir}")
    print(f"Production: {cfg.simulation_time_ns} ns; seed: {cfg.random_seed}")
    print(f"GCMC: {cfg.gcmc_n_moves} moves / {cfg.gcmc_n_cycles} cycles; platform: {cfg.gcmc_platform}")
    if dry_run:
        return None
    from H2OSite.pipeline import WaterMapPipeline

    sites = WaterMapPipeline(cfg).run()
    print(f"Complete: {len(sites)} hydration sites written to {cfg.output_dir}")
    return sites
