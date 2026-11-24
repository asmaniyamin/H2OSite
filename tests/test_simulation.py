# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for simulation/ - protocol, production, GCMC.

All tests use ultra-short timescales (0.01-0.03 ps) on CPU.
Uses openmmtools.testsystems for properly parameterized alanine dipeptide.
"""

import numpy as np
import pytest

from H2OSite.tests.conftest import requires_mdanalysis, requires_openmm


def _make_solvated_system():
    """Build a small solvated alanine dipeptide using openmmtools."""
    from openmmtools.testsystems import AlanineDipeptideExplicit
    import openmm.app as app

    ala = AlanineDipeptideExplicit(constraints=app.HBonds, rigid_water=True)
    return ala.topology, ala.positions, ala.system


class TestWaterTracker:
    def test_collect_uses_minimum_image(self):
        from H2OSite.simulation.production import WaterTracker

        tracker = WaterTracker(center=np.zeros(3), radius=1.0)
        positions = np.array([
            [12.85, 0.00, 0.00],  # same pocket across +x boundary
            [0.20, 0.00, 0.00],   # directly near the center
            [6.00, 0.00, 0.00],   # far away
        ])
        selected = tracker.collect(
            positions_nm=positions,
            water_oxygen_indices=[0, 1, 2],
            water_residue_ids=[10, 11, 12],
            box_lengths_nm=np.array([12.9, 12.9, 12.9]),
        )

        assert selected.tolist() == [0, 1]
        assert tracker.frame_resids[0].tolist() == [10, 11]
        assert tracker.frames[0].shape == (2, 3)
        assert np.allclose(tracker.frames[0][0], [-0.05, 0.0, 0.0], atol=1e-6)
        assert np.allclose(tracker.frames[0][1], [0.20, 0.0, 0.0], atol=1e-6)


@requires_openmm
class TestProtocolSmall:
    @pytest.fixture
    def tiny_system(self):
        return _make_solvated_system()

    def test_brownian_nvt_desmond_custom(self, tiny_system, tmp_path):
        """Brownian NVT with CustomIntegrator + velocity clipping."""
        from H2OSite.simulation.protocol import SimulationProtocol

        topology, positions, system = tiny_system
        heavy = [
            a.index for a in topology.atoms()
            if a.element is not None and a.element.symbol != "H"
        ][:5]

        protocol = SimulationProtocol(
            topology, positions, system,
            output_dir=tmp_path / "brownian_custom",
        )
        stages = [{
            "name": "brownian_custom",
            "ensemble": "NVT",
            "integrator": "Brownian",
            "temperature_K": 10.0,
            "time_ps": 0.02,
            "timestep_ps": 0.001,
            "thermostat_tau_ps": 0.1,
            "delta_max": 0.1,         # Å
            "restraint_k": 50.0,
            "restrain": "solute_heavy",
        }]
        pos, vel, box = protocol.run_equilibration(
            stages, heavy_atom_indices=heavy,
        )
        assert pos is not None
        assert vel is not None

    def test_langevin_npt_stage(self, tiny_system, tmp_path):
        from H2OSite.simulation.protocol import SimulationProtocol

        topology, positions, system = tiny_system
        protocol = SimulationProtocol(
            topology, positions, system,
            output_dir=tmp_path / "equil2",
        )
        stages = [{
            "name": "langevin_npt_fast",
            "ensemble": "NPT",
            "integrator": "Langevin",
            "temperature_K": 300.0,
            "time_ps": 0.01,
            "thermostat_tau_ps": 1.0,
            "barostat_tau_ps": 50.0,
            "restraint_k": 0.0,
        }]
        pos, vel, box = protocol.run_equilibration(stages)
        assert pos is not None

    def test_with_restraints(self, tiny_system, tmp_path):
        from H2OSite.simulation.protocol import SimulationProtocol

        topology, positions, system = tiny_system
        heavy_indices = [
            a.index for a in topology.atoms()
            if a.element is not None and a.element.symbol != "H"
        ][:5]

        protocol = SimulationProtocol(
            topology, positions, system,
            output_dir=tmp_path / "equil3",
        )
        stages = [{
            "name": "restrained",
            "ensemble": "NVT",
            "integrator": "Langevin",
            "temperature_K": 10.0,
            "time_ps": 0.01,
            "thermostat_tau_ps": 1.0,
            "restraint_k": 50.0,
            "restrain": "solute_heavy",
        }]
        pos, vel, box = protocol.run_equilibration(
            stages, heavy_atom_indices=heavy_indices,
        )
        assert pos is not None


@requires_openmm
class TestProductionSmall:
    @pytest.fixture
    def solvated_system(self):
        return _make_solvated_system()

    def test_production_collects_water(self, solvated_system, tmp_path):
        from H2OSite.simulation.production import ProductionMD

        topology, positions, system = solvated_system
        prod = ProductionMD(
            topology, system, positions,
            output_dir=tmp_path / "prod",
        )
        tracker = prod.run(
            temperature_K=300.0,
            time_ps=0.03,
            trajectory_interval_ps=0.01,
            active_site_center_nm=np.array([0.1, 0.0, 0.0]),
            active_site_radius_nm=2.0,
            save_trajectory=False,
        )
        assert tracker.n_frames > 0
        all_pos = tracker.get_all_positions()
        assert all_pos.ndim == 2
        assert all_pos.shape[1] == 3

    def test_water_positions_saved(self, solvated_system, tmp_path):
        from H2OSite.simulation.production import ProductionMD

        topology, positions, system = solvated_system
        prod = ProductionMD(
            topology, system, positions,
            output_dir=tmp_path / "prod2",
        )
        prod.run(
            time_ps=0.02,
            trajectory_interval_ps=0.01,
            active_site_center_nm=np.zeros(3),
            active_site_radius_nm=5.0,
            save_trajectory=False,
        )
        npy_path = tmp_path / "prod2" / "water_positions.npy"
        assert npy_path.exists()

    @requires_mdanalysis
    def test_replay_water_tracking_matches_saved_trajectory(
        self,
        solvated_system,
        tmp_path,
    ):
        from H2OSite.simulation.production import (
            ProductionMD,
            replay_water_tracking,
        )

        topology, positions, system = solvated_system
        prod = ProductionMD(
            topology, system, positions,
            output_dir=tmp_path / "prod_replay",
        )
        tracker = prod.run(
            time_ps=0.03,
            trajectory_interval_ps=0.01,
            active_site_center_nm=np.zeros(3),
            active_site_radius_nm=2.0,
            save_trajectory=True,
        )

        replayed = replay_water_tracking(
            topology=topology,
            trajectory_file=tracker.trajectory_path,
            center_nm=np.zeros(3),
            radius_nm=2.0,
            checkpoint_file=tracker.checkpoint_path,
        )

        assert replayed.n_frames == tracker.n_frames
        assert len(replayed.frame_resids) == len(tracker.frame_resids)
        for got, expected in zip(replayed.frame_resids, tracker.frame_resids):
            assert np.array_equal(got, expected)
        assert np.allclose(replayed.get_all_positions(), tracker.get_all_positions())

    @requires_mdanalysis
    def test_stage11b_replay_extracts_dipoles_and_energies(
        self,
        solvated_system,
        tmp_path,
    ):
        from H2OSite.analysis.trajectory_properties import (
            ProductionTrajectoryAnalyzer,
        )
        from H2OSite.simulation.production import ProductionMD

        topology, positions, system = solvated_system
        prod = ProductionMD(
            topology, system, positions,
            output_dir=tmp_path / "prod_stage11b",
        )
        tracker = prod.run(
            time_ps=0.03,
            trajectory_interval_ps=0.01,
            active_site_center_nm=np.zeros(3),
            active_site_radius_nm=2.0,
            save_trajectory=True,
        )

        assert tracker.trajectory_path is not None and tracker.trajectory_path.exists()
        assert tracker.checkpoint_path is not None and tracker.checkpoint_path.exists()
        assert len(tracker.frame_resids) == tracker.n_frames

        analyzer = ProductionTrajectoryAnalyzer(
            topology=topology,
            system=system,
            trajectory_file=tracker.trajectory_path,
            checkpoint_file=tracker.checkpoint_path,
        )
        dipoles, energies = analyzer.compute_tracked_water_properties(
            tracker.frame_resids,
            active_site_center_A=np.zeros(3),
            active_site_radius_A=20.0,
        )

        assert len(dipoles) == tracker.n_frames
        assert len(energies["pw"]) == tracker.n_frames
        assert len(energies["ww"]) == tracker.n_frames
        for frame_resids, frame_dipoles, frame_pw, frame_ww in zip(
            tracker.frame_resids,
            dipoles,
            energies["pw"],
            energies["ww"],
        ):
            assert len(frame_dipoles) == len(frame_resids)
            assert len(frame_pw) == len(frame_resids)
            assert len(frame_ww) == len(frame_resids)


@requires_openmm
class TestGCMCSmall:
    def test_simplified_gcmc(self):
        topology, positions, system = _make_solvated_system()

        from H2OSite.simulation.gcmc import GCMCSolvation

        gcmc = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=np.array([0.1, 0.0, 0.0]),
            radius_nm=1.0,
        )
        final_topology, final_pos, stats = gcmc._run_simplified(
            n_gcmc_moves=10, n_md_steps=5, n_cycles=2,
        )
        assert final_topology is topology
        assert final_pos is not None
        assert stats["mode"] == "simplified"
