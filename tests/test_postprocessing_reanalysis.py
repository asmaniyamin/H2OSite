# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Regression coverage for saved-run reanalysis and display-mode selection."""
import numpy as np
import pytest

from H2OSite import postprocessing as pp
from H2OSite.pipeline import WaterMapPipeline

CSV = (
    "site_index,x,y,z,occupancy,dH_total,dG,dipole_x,dipole_y,dipole_z\n"
    "0,1,2,3,0.8,-10,-9,1,0,0\n"
)


def saved_run(root):
    (root / "checkpoints").mkdir()
    (root / "production").mkdir()
    checkpoint = root / "checkpoints" / "post_gcmc_equil.npz"
    checkpoint.touch()
    trajectory = root / "production" / "production.dcd"
    trajectory.touch()
    return checkpoint, trajectory


@pytest.mark.parametrize("cavity", [True, False])
def test_reanalysis_preserves_originals_and_exports_new_results(monkeypatch, tmp_path, cavity):
    checkpoint, trajectory = saved_run(tmp_path)
    original = tmp_path / "mini_wm.csv"
    original.write_text("original results")
    (tmp_path / "mini-water.cns").write_text("original grid")
    stale = tmp_path / "mini-continuous_cavity_fixed.cns"
    stale.write_text("old fixed cavity")
    (tmp_path / "fixed").mkdir()
    (tmp_path / "fixed" / stale.name).write_text("old bundled cavity")
    prior_temp = tmp_path / "_postprocess_reanalysis"
    prior_temp.mkdir()
    (prior_temp / "keep").write_text("unrelated work")
    captured = {}

    def analyze(self, **kwargs):
        captured.update(kwargs)
        assert self.config.water_model == "tip4pew"
        assert self.config.hs_radius_A == 1.5
        assert self.config.clustering_cutoff_A == 2.5
        out = self.output_dir
        (out / f"{self.config.jobname}_wm.csv").write_text(CSV)
        (out / f"{self.config.jobname}-water.cns").write_text("new grid")
        if cavity:
            (out / f"{self.config.jobname}-continuous_cavity.cns").write_text("new cavity")
        np.save(out / "retracked_water_positions.npy", [[1, 2, 3]])
        return []

    monkeypatch.setattr(WaterMapPipeline, "run_analysis_from_trajectory", analyze)
    artifacts = pp.run_corrected_reanalysis(
        tmp_path, "mini", np.array([4, 5, 6]),
        site_radius_A=8, hs_radius_A=1.5, cutoff_A=2.5, water_model="tip4pew",
    )
    assert captured["checkpoint_file"] == checkpoint
    assert captured["trajectory_file"] == trajectory
    assert captured["production_checkpoint_file"] is None
    np.testing.assert_array_equal(captured["active_site_center_A"], [4, 5, 6])
    assert captured["active_site_radius_A"] == 8
    assert original.read_text() == "original results"
    assert (tmp_path / "mini-water.cns").read_text() == "original grid"
    assert artifacts["fixed_csv"].read_text() == CSV
    assert artifacts["fixed_water_cns"].read_text() == "new grid"
    assert artifacts["fixed_pdb"].is_file()
    assert artifacts["fixed_cif"].is_file()
    np.testing.assert_array_equal(np.load(artifacts["fixed_positions"]), [[1, 2, 3]])
    assert stale.exists() == cavity
    assert ("fixed_cavity_cns" in artifacts) == cavity
    if not cavity:
        assert not (tmp_path / "fixed" / stale.name).exists()
    assert (prior_temp / "keep").read_text() == "unrelated work"
    assert not list(tmp_path.glob(".watermap-reanalysis-*"))


def test_failed_reanalysis_keeps_prior_fixed_outputs(monkeypatch, tmp_path):
    saved_run(tmp_path)
    fixed = tmp_path / "mini_wm_fixed.csv"
    fixed.write_text("previous successful results")

    def analyze(self, **kwargs):
        (self.output_dir / f"{self.config.jobname}_wm.csv").write_text(CSV)
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(WaterMapPipeline, "run_analysis_from_trajectory", analyze)
    with pytest.raises(RuntimeError, match="analysis failed"):
        pp.run_corrected_reanalysis(tmp_path, "mini", np.zeros(3))
    assert fixed.read_text() == "previous successful results"
    assert not list(tmp_path.glob(".watermap-reanalysis-*"))


@pytest.mark.parametrize("reanalyze,has_raw", [(False, True), (True, True), (True, False)])
def test_display_exports_use_selected_analysis(monkeypatch, tmp_path, reanalyze, has_raw):
    _, trajectory = saved_run(tmp_path)
    raw_csv = tmp_path / "mini_wm.csv"
    if has_raw:
        raw_csv.write_text(CSV)
    (tmp_path / "mini-water.cns").write_text("raw grid")
    np.save(tmp_path / "production" / "water_positions.npy", [[0, 0, 0]])
    calls = []

    def analyze(output_dir, jobname, tracking_center_A, **kwargs):
        calls.append(kwargs)
        (output_dir / f"{jobname}_wm_fixed.csv").write_text(CSV.replace(",1,2,3,", ",7,8,9,"))
        (output_dir / f"{jobname}-water_fixed.cns").write_text("recomputed grid")
        np.save(output_dir / "production" / "water_positions_fixed.npy", [[7, 8, 9]])
        return {}

    monkeypatch.setattr(pp, "run_corrected_reanalysis", analyze)
    monkeypatch.setattr(pp, "write_centered_trajectory", lambda **kwargs: np.array([10, 10, 10]))
    monkeypatch.setattr(pp, "write_centered_final_with_sites_cif", lambda **kwargs: None)
    artifacts = pp.generate_display_artifacts(
        output_dir=tmp_path, jobname="mini", topology=None,
        trajectory_file=trajectory, tracking_center_A=np.zeros(3), site_radius_A=10,
        reanalyze=reanalyze,
    )
    assert len(calls) == int(reanalyze)
    expected_csv = CSV.replace(",1,2,3,", ",7,8,9,") if reanalyze else CSV
    assert artifacts["fixed_csv"].read_text() == expected_csv
    assert (artifacts["fixed_dir"] / "mini_wm_fixed.csv").read_text() == expected_csv
    assert artifacts["fixed_water_cns"].read_text() == ("recomputed grid" if reanalyze else "raw grid")
    np.testing.assert_array_equal(np.load(artifacts["fixed_positions"]),
                                  [[7, 8, 9]] if reanalyze else [[0, 0, 0]])
    assert raw_csv.exists() == has_raw
    if has_raw:
        assert raw_csv.read_text() == CSV


def test_saved_run_cli_dispatches_reanalysis_options(monkeypatch, tmp_path, capsys):
    checkpoint, trajectory = saved_run(tmp_path)
    (tmp_path / "prep").mkdir()
    (tmp_path / "prep" / "mini_protein.cif").touch()
    captured = {}
    monkeypatch.setattr(WaterMapPipeline, "_deserialize_checkpoint", lambda path: {"topology": "top"})

    def generate(**kwargs):
        captured.update(kwargs)
        return {"fixed_dir": tmp_path / "fixed"}

    monkeypatch.setattr(pp, "generate_display_artifacts", generate)
    pp.main([str(tmp_path), "--reanalyze", "--tracking-center", "1", "2", "3",
             "--active-site", "4", "5", "6", "--hs-radius", "1.5", "--water-model", "tip4p"])
    assert captured["reanalyze"] is True
    assert captured["checkpoint_file"] == checkpoint
    assert captured["trajectory_file"] == trajectory
    assert captured["jobname"] == "mini"
    assert captured["topology"] == "top"
    assert captured["hs_radius_A"] == 1.5
    assert captured["water_model"] == "tip4p"
    np.testing.assert_array_equal(captured["tracking_center_A"], [1, 2, 3])
    np.testing.assert_array_equal(captured["centered_active_site_A"], [4, 5, 6])
    assert "Fixed bundle:" in capsys.readouterr().out
