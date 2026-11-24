# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for generic postproduction display artifacts."""

from __future__ import annotations

import numpy as np

from H2OSite.tests.conftest import requires_mdanalysis, requires_openmm


def test_wrap_point_into_box_keeps_centered_sites_in_primary_image():
    from H2OSite.postprocessing import _wrap_point_into_box

    wrapped = _wrap_point_into_box(
        np.array([87.714569, -108.921796, -121.752284], dtype=float),
        np.array([198.654, 198.654, 198.654], dtype=float),
    )

    np.testing.assert_allclose(
        wrapped,
        [87.714569, 89.732204, 76.901716],
        atol=1e-6,
    )


@requires_openmm
def test_get_protein_component_indices_splits_disconnected_subunits():
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    from H2OSite.postprocessing import _get_protein_component_indices

    topology = app.Topology()
    chain = topology.addChain("1")

    res1 = topology.addResidue("GLY", chain)
    atom1 = topology.addAtom("N", app.element.nitrogen, res1)
    atom2 = topology.addAtom("CA", app.element.carbon, res1)
    topology.addBond(atom1, atom2)

    res2 = topology.addResidue("GLY", chain)
    atom3 = topology.addAtom("N", app.element.nitrogen, res2)
    atom4 = topology.addAtom("CA", app.element.carbon, res2)
    topology.addBond(atom3, atom4)

    topology.setPeriodicBoxVectors(
        (
            openmm.Vec3(2.0, 0.0, 0.0),
            openmm.Vec3(0.0, 2.0, 0.0),
            openmm.Vec3(0.0, 0.0, 2.0),
        ) * unit.nanometer
    )

    components = _get_protein_component_indices(
        topology,
        np.array([0, 1, 2, 3], dtype=int),
    )

    assert list(components) == ["protein_component_1", "protein_component_2"]
    np.testing.assert_array_equal(components["protein_component_1"], [0, 1])
    np.testing.assert_array_equal(components["protein_component_2"], [2, 3])


def _make_solvated_system():
    """Build a small solvated alanine dipeptide using openmmtools."""
    from openmmtools.testsystems import AlanineDipeptideExplicit
    import openmm.app as app

    ala = AlanineDipeptideExplicit(constraints=app.HBonds, rigid_water=True)
    return ala.topology, ala.positions, ala.system


@requires_openmm
def test_water_tracker_collect_uses_minimum_image():
    from H2OSite.simulation.production import WaterTracker

    tracker = WaterTracker(center=np.zeros(3), radius=0.25)
    selected = tracker.collect(
        positions_nm=np.array([[0.92, 0.0, 0.0]], dtype=float),
        water_oxygen_indices=[0],
        water_residue_ids=[7],
        box_lengths_nm=np.array([1.0, 1.0, 1.0], dtype=float),
    )

    assert selected.tolist() == [0]
    np.testing.assert_allclose(tracker.frames[0][0], [-0.08, 0.0, 0.0], atol=1e-6)
    assert tracker.frame_resids[0].tolist() == [7]


@requires_openmm
def test_water_tracker_reports_drift_event():
    from H2OSite.simulation.production import WaterTracker

    tracker = WaterTracker(center=np.zeros(3), radius=0.25)
    records = [
        {
            "label": "HOH:A:101",
            "topology_resid": 12,
            "residue_id": "101",
            "chain_id": "A",
            "residue_name": "HOH",
        }
    ]

    tracker.collect(
        positions_nm=np.array([[0.10, 0.0, 0.0]], dtype=float),
        water_oxygen_indices=[0],
        water_residue_ids=[12],
        box_lengths_nm=np.array([2.0, 2.0, 2.0], dtype=float),
        water_records=records,
    )
    tracker.collect(
        positions_nm=np.array([[0.40, 0.0, 0.0]], dtype=float),
        water_oxygen_indices=[0],
        water_residue_ids=[12],
        box_lengths_nm=np.array([2.0, 2.0, 2.0], dtype=float),
        water_records=records,
    )

    assert len(tracker.drift_events) == 1
    event = tracker.drift_events[0]
    assert event["label"] == "HOH:A:101"
    np.testing.assert_allclose(event["position_nm"], [0.40, 0.0, 0.0])


@requires_openmm
@requires_mdanalysis
def test_generate_display_artifacts_smoke(tmp_path):
    import openmm.app as app

    from H2OSite.postprocessing import generate_display_artifacts
    from H2OSite.simulation.production import (
        ProductionMD,
        _find_water_residue_records,
    )

    topology, positions, system = _make_solvated_system()
    output_dir = tmp_path / "display_out"
    production_dir = output_dir / "production"
    prep_dir = output_dir / "prep"
    production_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    jobname = "mini"

    prod = ProductionMD(
        topology,
        system,
        positions,
        output_dir=production_dir,
    )
    tracker = prod.run(
        time_ps=0.03,
        trajectory_interval_ps=0.01,
        active_site_center_nm=np.zeros(3),
        active_site_radius_nm=2.0,
        save_trajectory=True,
    )

    assert tracker.trajectory_path is not None and tracker.trajectory_path.exists()

    wm_csv = output_dir / f"{jobname}_wm.csv"
    wm_csv.write_text(
        "site_index,x,y,z,occupancy,dH_pw,dH_ww,dH_total,minus_TdS,dG,"
        "hbond_pw,hbond_ww,hbond_lw,dipole_x,dipole_y,dipole_z\n"
        "0,0.000,0.000,0.000,0.5000,-5.000,-5.000,-10.000,1.000,-9.000,"
        "0.00,0.00,0.00,1.0000,0.0000,0.0000\n"
    )
    (output_dir / f"{jobname}-water.cns").write_text("dummy water cns\n")
    (output_dir / f"{jobname}-continuous_cavity.cns").write_text("dummy cavity cns\n")

    with open(prep_dir / f"{jobname}_protein.pdb", "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)
    (prep_dir / f"{jobname}_ligand.pdb").write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    np.savez(prep_dir / f"{jobname}_meta.npz", translation_vector=np.array([1.0, 2.0, 3.0]))

    with open(output_dir / "input_protein.pdb", "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)

    water_records = _find_water_residue_records(topology)
    drift_report = output_dir / f"{jobname}_drifted_waters.csv"
    drift_report.write_text(
        "transition,event,label,topology_resid,residue_id,chain_id,residue_name,"
        "position_x_A,position_y_A,position_z_A,distance_A,"
        "last_inside_x_A,last_inside_y_A,last_inside_z_A,last_inside_distance_A\n"
        f"Stage 8,left_stage_sphere,{water_records[0]['label']},{water_records[0]['topology_resid']},"
        f"{water_records[0]['residue_id']},{water_records[0]['chain_id']},{water_records[0]['residue_name']},"
        "1.0,2.0,3.0,12.0,0.0,0.0,0.0,5.0\n"
    )

    artifacts = generate_display_artifacts(
        output_dir=output_dir,
        jobname=jobname,
        topology=topology,
        trajectory_file=tracker.trajectory_path,
        tracking_center_A=np.zeros(3),
        site_radius_A=10.0,
        nearby_radius_A=5.0,
    )

    expected = [
        output_dir / "solvated.pdb",
        output_dir / "solvated.cif",
        output_dir / "solvated_final.pdb",
        output_dir / "solvated_final.cif",
        output_dir / f"{jobname}_solvated_final_with_wm_fixed.pdb",
        output_dir / f"{jobname}_solvated_final_with_wm_fixed.cif",
        production_dir / "trajectory_centered.dcd",
        output_dir / f"{jobname}_wm_fixed.csv",
        output_dir / f"{jobname}_wm_fixed.pdb",
        output_dir / f"{jobname}_wm_fixed.cif",
        output_dir / f"{jobname}_wm_fixed_centered.pdb",
        output_dir / f"{jobname}_wm_fixed_centered.cif",
        output_dir / f"{jobname}_wm_fixed_input_frame.pdb",
        output_dir / f"{jobname}_wm_fixed_input_frame.cif",
        output_dir / f"{jobname}_drifted_waters_final.pdb",
        output_dir / f"{jobname}_drifted_waters_final.cif",
        output_dir / "view_watermap_traj_fixed.pml",
        production_dir / "water_positions_fixed.npy",
        output_dir / "fixed",
    ]
    for path in expected:
        assert path.exists(), path

    assert "merged_original_protein" in artifacts
    assert artifacts["merged_original_protein"].exists()
    assert "centered_dcd" in artifacts
    assert artifacts["centered_dcd"].exists()
    assert "drifted_waters_final_pdb" in artifacts
    assert "solvated_final_with_wm_fixed_pdb" in artifacts

    fixed_pml_text = (output_dir / f"{jobname}_wm_fixed.pml").read_text()
    assert "load solvated_final.cif, final_system" in fixed_pml_text
    assert "load mini_wm_fixed_centered.cif, wm_sites_fixed" in fixed_pml_text
    assert "select ligand_sel" in fixed_pml_text

    view_pml_text = (output_dir / "view_watermap_traj_fixed.pml").read_text()
    assert "load solvated.cif, system" in view_pml_text
    assert "select ligand_sel" in view_pml_text

    merged_text = (output_dir / f"{jobname}_solvated_final_with_wm_fixed.pdb").read_text()
    assert "WMP" in merged_text
