# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for cli.py - CLI entry points."""

import numpy as np
import pytest
from click.testing import CliRunner

from H2OSite.cli import main, _load_sites_csv


class TestCLIHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "H2OSite" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--protein" in result.output

    def test_analyze_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--water-positions" in result.output

    def test_score_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--help"])
        assert result.exit_code == 0

    def test_convert_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0


class TestLoadSitesCSV:
    def test_load_valid(self, tmp_path):
        csv = tmp_path / "sites.csv"
        csv.write_text(
            "site_index,x,y,z,occupancy,dH_pw,dH_ww,dH_total,minus_TdS,dG,hbond_pw,hbond_ww,hbond_lw,dipole_x,dipole_y,dipole_z\n"
            "0,1.0,2.0,3.0,0.9,-15.0,-5.0,0.67,-0.9,-0.23,2.1,3.5,0.5,0.1,0.2,0.3\n"
            "1,4.0,5.0,6.0,0.5,-10.0,-3.0,6.67,-1.05,5.62,1.0,2.0,0.0,0.0,0.0,0.0\n"
        )
        sites = _load_sites_csv(str(csv))
        assert len(sites) == 2
        assert sites[0].index == 0
        np.testing.assert_allclose(sites[0].center, [1, 2, 3])
        assert sites[0].occupancy == 0.9

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("site_index,x,y,z,occupancy,dH_pw,dH_ww,dH_total,minus_TdS,dG\n")
        sites = _load_sites_csv(str(csv))
        assert sites == []


def test_reanalysis_command_dispatches_to_integrated_pipeline(monkeypatch, tmp_path):
    from H2OSite.pipeline import WaterMapPipeline

    checkpoint = tmp_path / "stage.npz"
    trajectory = tmp_path / "production.dcd"
    checkpoint.touch()
    trajectory.touch()
    captured = {}

    def reanalyze(self, **kwargs):
        captured.update(kwargs)
        captured["water_model"] = self.config.water_model
        return []

    monkeypatch.setattr(WaterMapPipeline, "run_analysis_from_trajectory", reanalyze)
    result = CliRunner().invoke(main, [
        "reanalyze-trajectory", "--checkpoint", str(checkpoint),
        "--trajectory", str(trajectory), "--output-dir", str(tmp_path / "results"),
        "--center", "1", "2", "3", "--radius", "8", "--water-model", "tip4pew",
    ])
    assert result.exit_code == 0, result.output
    assert captured["checkpoint_file"] == str(checkpoint)
    assert captured["trajectory_file"] == str(trajectory)
    np.testing.assert_allclose(captured["active_site_center_A"], [1, 2, 3])
    assert captured["active_site_radius_A"] == 8.0
    assert captured["water_model"] == "tip4pew"
