# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Shared workflow configuration and execution regression tests."""
from copy import deepcopy
from pathlib import Path

import pytest
from click.testing import CliRunner

from H2OSite.cli import main
from H2OSite.core.config import WaterMapConfig
from H2OSite.workflow import configure_run, load_workflow, run_workflow

EXAMPLES = Path(__file__).resolve().parents[1] / "docs/examples/configs"


def test_restraints_do_not_mutate_defaults():
    original = WaterMapConfig()
    before = deepcopy(original.equilibration_stages)
    cfg = configure_run(original, restraint_k_kcal=10.0)
    assert original.equilibration_stages == before
    assert WaterMapConfig().equilibration_stages == before
    for stages in (cfg.equilibration_stages, cfg.post_gcmc_equil_stages, cfg.gcmc_handoff_relaxation_stages):
        assert all(s["restraint_k"] == 10.0 for s in stages if s.get("restrain"))


@pytest.mark.parametrize("mode", ["apo", "holo"])
def test_5hxb_short_preset(mode):
    cfg = load_workflow(EXAMPLES / "5hxb_production.yaml", workflow_overrides={"mode": mode, "preset": "short", "cache": True})
    assert cfg.retain_ligand == (mode == "holo")
    assert mode in cfg.output_dir.name
    assert cfg.manual_histidine_states["X:526"] == "HID"
    assert len(cfg.manual_histidine_states) == 41
    assert cfg.simulation_time_ns == 0.02
    assert (cfg.gcmc_n_moves, cfg.gcmc_n_cycles, cfg.gcmc_md_steps_per_cycle) == (5000, 25, 250)
    assert [s["time_ps"] for s in cfg.post_gcmc_equil_stages] == [0.05, 0.02]
    assert all(s["restraint_k"] == 5.0 for s in cfg.post_gcmc_equil_stages)
    assert cfg.gcmc_cache_dir == cfg.output_dir / "checkpoints/loch_sampler_cache"


def test_smoke_preset():
    cfg = load_workflow(EXAMPLES / "5hxb_production.yaml", workflow_overrides={"preset": "smoke"})
    assert cfg.simulation_time_ns == 0.002
    assert (cfg.gcmc_n_moves, cfg.gcmc_n_cycles, cfg.gcmc_md_steps_per_cycle) == (40, 4, 2000)
    assert [s["time_ps"] for s in cfg.equilibration_stages] == [2.0, 1.0, 1.0, 2.0]
    assert cfg.gcmc_handoff_relaxation_stages[0]["time_ps"] == 0.2
    assert cfg.post_gcmc_equil_stages[0]["time_ps"] == 2.0
    assert not cfg.generate_display_outputs


def test_custom_stages_and_caches_are_preserved():
    original = WaterMapConfig(equilibration_stages=[dict(name="custom", time_ps=7.0)],
                              gcmc_cache_dir="custom-cache")
    cfg = configure_run(original)
    assert cfg.equilibration_stages == original.equilibration_stages
    assert cfg.gcmc_cache_dir == Path("custom-cache")


def test_yaml_relative_inputs_and_output_templates(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text('protein_path: protein.pdb\nligand_resnames: [ABC]\noutput_dir: results/{mode}\nworkflow:\n  mode: holo\n')
    cfg = load_workflow(config)
    assert cfg.protein_path == tmp_path / "protein.pdb"
    assert cfg.output_dir == Path("results/holo")


def test_cli_config_overrides_and_dry_run(monkeypatch, tmp_path):
    from H2OSite.pipeline import WaterMapPipeline
    def fail(*args, **kwargs):
        raise AssertionError("Dry run constructed a simulation")
    monkeypatch.setattr(WaterMapPipeline, "__init__", fail)
    out = tmp_path / "output"
    result = CliRunner().invoke(main, ["run", "--config", str(EXAMPLES / "5hxb_production.yaml"),
        "--mode", "apo", "--preset", "smoke", "--seed", "137", "--output-dir", str(out), "--dry-run"])
    assert result.exit_code == 0, result.output
    for text in ["Mode: apo", "seed: 137", "Production: 0.002 ns", "GCMC: 40 moves / 4 cycles", str(out)]:
        assert text in result.output
    assert not out.exists()


def test_cli_new_protein_needs_no_custom_functions(tmp_path):
    protein = tmp_path / "protein.pdb"
    protein.touch()
    result = CliRunner().invoke(main, ["run", "--protein", str(protein),
        "--ligand-resname", "ABC", "--mode", "holo", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Ligand: ABC" in result.output


def test_missing_ligand_creates_no_outputs(tmp_path):
    protein = tmp_path / "protein.pdb"
    protein.touch()
    cfg = configure_run(WaterMapConfig(protein_path=protein, output_dir=tmp_path / "run"))
    with pytest.raises(ValueError, match="ligand file"):
        run_workflow(cfg, dry_run=True)
    assert not cfg.output_dir.exists()


def test_runner_passes_prepared_configuration(monkeypatch, tmp_path):
    from H2OSite.pipeline import WaterMapPipeline
    protein = tmp_path / "protein.pdb"
    protein.touch()
    cfg = configure_run(WaterMapConfig(protein_path=protein, ligand_resnames=["ABC"]))
    seen = []
    monkeypatch.setattr(WaterMapPipeline, "__init__", lambda self, config: seen.append(config))
    monkeypatch.setattr(WaterMapPipeline, "run", lambda self: ["site"])
    assert run_workflow(cfg) == ["site"]
    assert seen == [cfg]


@pytest.mark.parametrize("contents", ["[]", "workflow: []", "workflow:\n  preset: unknown\n"])
def test_invalid_yaml_is_rejected(tmp_path, contents):
    path = tmp_path / "run.yaml"
    path.write_text(contents)
    result = CliRunner().invoke(main, ["run", "--config", str(path), "--dry-run"])
    assert result.exit_code != 0
    assert "Error:" in result.output
