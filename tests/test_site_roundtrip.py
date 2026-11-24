# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Thermodynamic references and CLI CSV round-trip regressions."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from H2OSite.analysis.clustering import HydrationSite, HydrationSiteClustering
from H2OSite.analysis.thermodynamics import ThermodynamicCalculator
from H2OSite.cli import main, _load_sites_csv
from H2OSite.core.config import get_water_model_params
from H2OSite.io.converters import (
    load_sites_csv, watermap_to_csv, watermap_to_pdb, watermap_to_cif,
)
from H2OSite.scoring.displacement import DisplacementScorer
from H2OSite.scoring.kinetics import KineticsClassifier


def make_site(model="tip3p"):
    return HydrationSite(
        index=0, center=np.zeros(3), occupancy=0.9, n_water_frames=9,
        waters_per_frame=np.ones(10, dtype=int), enthalpy_pw=-10.0,
        enthalpy_ww=-5.0, translational_entropy=-7.25,
        rotational_entropy=-2.75, dipole=np.array([0.1, 0.2, 0.3]),
        hbond_pw=1.2, hbond_ww=2.3, hbond_lw=0.4, water_model=model,
    )


@pytest.mark.parametrize("model", ["tip3p", "tip4p", "tip4pew"])
def test_csv_preserves_scoring_and_all_site_metadata(model, tmp_path):
    original = make_site(model)
    original.temperature_K = 310.0
    path = tmp_path / "sites.csv"
    watermap_to_csv([original], path)
    loaded = _load_sites_csv(str(path))[0]
    assert loaded.water_model == model
    assert loaded.bulk_reference_energy == original.bulk_reference_energy
    assert loaded.temperature_K == original.temperature_K
    assert loaded.translational_entropy == original.translational_entropy
    assert loaded.rotational_entropy == original.rotational_entropy
    assert loaded.entropy == original.entropy
    assert loaded.free_energy == original.free_energy
    np.testing.assert_array_equal(loaded.dipole, original.dipole)
    assert (loaded.hbond_pw, loaded.hbond_ww, loaded.hbond_lw) == (1.2, 2.3, 0.4)
    scorer = DisplacementScorer()
    before = scorer.score([original], np.zeros((1, 3)), ["C"])[0]
    after = scorer.score([loaded], np.zeros((1, 3)), ["C"])[0]
    assert before == after


@pytest.mark.parametrize("model", ["tip3p", "tip4p", "tip4pew"])
def test_calculation_exports_and_scoring_use_selected_water_model(model, tmp_path):
    site = make_site()
    calc = ThermodynamicCalculator(water_model=model)
    calc.compute_all([site], [np.zeros((1, 3))])
    expected_dh = site.potential_energy - get_water_model_params(model)["bulk_energy_kcal"]
    expected_dg = expected_dh + site.entropy
    assert site.water_model == model
    assert site.relative_enthalpy == pytest.approx(expected_dh)
    assert site.free_energy == pytest.approx(expected_dg)
    assert ThermodynamicCalculator.compute_free_energy(site) == pytest.approx(expected_dg)
    path = tmp_path / "sites.csv"
    watermap_to_csv([site], path)
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["dH_total"]) == pytest.approx(expected_dh)
    assert float(row["dG"]) == pytest.approx(expected_dg)
    pdb = tmp_path / "sites.pdb"
    watermap_to_pdb([site], pdb)
    atom = next(line for line in pdb.read_text().splitlines() if line.startswith("HETATM"))
    assert float(atom[54:60]) == pytest.approx(expected_dh, abs=0.005)
    assert float(atom[60:66]) == pytest.approx(expected_dg, abs=0.005)
    cif = tmp_path / "sites.cif"
    watermap_to_cif([site], cif)
    atom = next(line.split() for line in cif.read_text().splitlines() if line.startswith("HETATM"))
    assert float(atom[13]) == pytest.approx(expected_dh, abs=0.005)
    assert float(atom[14]) == pytest.approx(expected_dg, abs=0.005)
    score = DisplacementScorer().score([site], np.zeros((1, 3)), ["C"])[0]
    assert score.overlap == 1.0
    assert score.dH == pytest.approx(expected_dh)
    assert score.dG == pytest.approx(expected_dg)
    result = KineticsClassifier().classify([site], [score], np.zeros((1, 3)), ["C"])
    assert result.per_site[0]["dH"] == pytest.approx(expected_dh)


def test_legacy_csv_preserves_totals_without_inventing_model_or_entropy_split(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text(
        "site_index,x,y,z,occupancy,dH_pw,dH_ww,dH_total,minus_TdS,dG\n"
        "0,0,0,0,0.9,-10,-5,4.3,3,7.3\n"
    )
    site = load_sites_csv(path)[0]
    assert site.water_model is None
    assert not site.properties["entropy_components_available"]
    assert site.bulk_reference_energy == pytest.approx(-19.3)
    assert site.entropy == pytest.approx(3.0)
    assert site.free_energy == pytest.approx(7.3)
    second = tmp_path / "second.csv"
    watermap_to_csv([site], second)
    reloaded = load_sites_csv(second)[0]
    assert reloaded.entropy == site.entropy
    assert reloaded.free_energy == site.free_energy
    assert reloaded.water_model is None
    assert not reloaded.properties["entropy_components_available"]


def test_legacy_archived_values_keep_recorded_free_energies(tmp_path):
    path = tmp_path / "archived.csv"
    path.write_text(
        "site_index,x,y,z,occupancy,dH_pw,dH_ww,dH_total,minus_TdS,dG\n"
        "0,-9.724,0.188,-3.011,0.8571,-31.313,-18.289,-29.933,2.939,-26.994\n"
        "1,6.285,-4.009,-1.535,1,16.675,-44.639,-8.294,2.811,-5.483\n"
        "2,6.898,1.270,-4.749,0.4286,3.207,-37.235,-14.358,0.669,-13.689\n"
    )
    sites = load_sites_csv(path)
    assert [site.free_energy for site in sites] == pytest.approx([-26.994, -5.483, -13.689])


def test_named_columns_and_missing_optional_fields(tmp_path):
    path = tmp_path / "reordered.csv"
    path.write_text(
        "dG,minus_TdS,dH_total,dH_ww,dH_pw,occupancy,z,y,x,site_index\n"
        "7.3,3,4.3,-5,-10,0.9,3,2,1,0\n"
    )
    site = load_sites_csv(path)[0]
    assert site.free_energy == pytest.approx(7.3)
    np.testing.assert_array_equal(site.center, [1, 2, 3])
    assert site.hbond_pw == 0.0


@pytest.mark.parametrize("optional", ["hbond_pw", "hbond_pw,hbond_ww"])
def test_partial_hbond_columns_are_supported(tmp_path, optional):
    path = tmp_path / "sites.csv"
    values = ",".join("1.2" for _ in optional.split(","))
    path.write_text(f"site_index,x,y,z,occupancy,dH_pw,dH_ww,dH_total,minus_TdS,dG,{optional}\n"
                    f"0,0,0,0,1,-10,-5,4.3,3,7.3,{values}\n")
    assert load_sites_csv(path)[0].hbond_pw == 1.2


def test_cli_score_and_conversion_preserve_saved_free_energy(tmp_path, monkeypatch):
    site = make_site()
    path = tmp_path / "sites.csv"
    watermap_to_csv([site], path)
    ligand = tmp_path / "ligand.sdf"
    ligand.touch()
    monkeypatch.setattr("H2OSite.core.structure.load_structure",
                        lambda path: SimpleNamespace(positions=np.zeros((1, 3)), elements=["C"]))
    result = CliRunner().invoke(main, ["score", "--sites", str(path), "--ligand", str(ligand)])
    assert result.exit_code == 0, result.output
    assert "Total displacement dG: 7.30 kcal/mol" in result.output
    converted = tmp_path / "converted.csv"
    result = CliRunner().invoke(main, ["convert", "--sites", str(path), "--output", str(converted), "--format", "csv"])
    assert result.exit_code == 0, result.output
    assert load_sites_csv(converted)[0].free_energy == pytest.approx(7.3)


def test_clustering_uses_model_density_and_labels_sites():
    clusterer = HydrationSiteClustering(water_model="tip3p")
    assert clusterer.bulk_density == get_water_model_params("tip3p")["bulk_density"]
    sites = clusterer._assign_waters_to_sites([np.zeros(3)], np.zeros((1, 3)), 1, [np.zeros((1, 3))])
    assert sites[0].water_model == "tip3p"


def test_explicit_calculator_override_uses_requested_reference():
    site = make_site("tip3p")
    expected = site.potential_energy - get_water_model_params("tip4pew")["bulk_energy_kcal"] + site.entropy
    assert ThermodynamicCalculator.compute_free_energy(site, "tip4pew") == pytest.approx(expected)


@pytest.mark.parametrize("model", ["tip3p", "tip4p", "tip4pew"])
def test_analysis_pipeline_persists_selected_model_in_outputs(model, tmp_path):
    from H2OSite.core.config import WaterMapConfig
    from H2OSite.pipeline import WaterMapPipeline

    frames = np.random.default_rng(23).normal(0.0, 0.2, (20, 15, 3))
    positions = tmp_path / "water.npy"
    np.save(positions, frames)
    pipeline = WaterMapPipeline(WaterMapConfig(
        output_dir=tmp_path / "analysis", jobname="check", water_model=model,
    ))
    sites = pipeline.run_analysis_only(positions)
    assert sites
    loaded = load_sites_csv(tmp_path / "analysis" / "check_wm.csv")
    assert len(loaded) == len(sites)
    for original, saved in zip(sites, loaded):
        assert original.water_model == saved.water_model == model
        assert saved.bulk_reference_energy == get_water_model_params(model)["bulk_energy_kcal"]
        assert saved.entropy == original.entropy
        assert saved.free_energy == original.free_energy
