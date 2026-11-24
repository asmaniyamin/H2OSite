# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for the loch-backed GCMC Stage 7 implementation.

The fast tests validate the OpenMM -> AMBER -> Sire helper path and the
reference-selection logic without needing a GPU.

The ultra-short smoke tests execute the loch-backed GCMC workflow on a
small solvated alanine dipeptide with reduced settings. Those tests are
skipped unless a usable CUDA/OpenCL device is available.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from H2OSite.tests.conftest import requires_openmm


def _check_loch_runtime() -> bool:
    try:
        from H2OSite.simulation.gcmc import _prepare_openbiosim_runtime

        _prepare_openbiosim_runtime()
        return True
    except Exception:
        return False


def _has_cuda_device() -> bool:
    try:
        import pycuda.driver as cuda

        cuda.init()
        return cuda.Device.count() > 0
    except Exception:
        return False


def _has_opencl_device() -> bool:
    try:
        import pyopencl as cl

        return any(platform.get_devices() for platform in cl.get_platforms())
    except Exception:
        return False


def _preferred_loch_platform() -> str | None:
    if _has_cuda_device() and (
        shutil.which("nvcc") is not None or os.environ.get("PYCUDA_NVCC")
    ):
        return "cuda"
    if _has_opencl_device():
        return "opencl"
    return None


requires_loch_runtime = pytest.mark.skipif(
    not _check_loch_runtime(),
    reason="loch/BioSimSpace/Sire runtime is not importable",
)
requires_loch_gpu = pytest.mark.skipif(
    _preferred_loch_platform() is None,
    reason="No usable CUDA/OpenCL device available for loch GCMC",
)


def _make_solvated_system():
    """Build a small solvated alanine dipeptide system plus export FF."""
    from openmmtools.testsystems import AlanineDipeptideExplicit
    import openmm.app as app

    ala = AlanineDipeptideExplicit(constraints=app.HBonds, rigid_water=True)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    return ala.topology, ala.positions, ala.system, ff


def _heavy_atom_center_nm(topology, positions) -> np.ndarray:
    import openmm.unit as unit

    pos_nm = np.asarray(positions.value_in_unit(unit.nanometer))
    heavy = [
        atom.index
        for atom in topology.atoms()
        if atom.element is not None and atom.element.symbol != "H"
    ]
    return np.mean(pos_nm[heavy[:3]], axis=0)


@requires_openmm
@requires_loch_runtime
class TestLochGCMCHelpers:
    def test_sampler_fingerprint_reuses_cache_across_coordinate_changes(self):
        import openmm.unit as unit

        from H2OSite.simulation.gcmc import GCMCSolvation

        topology, positions, system, ff = _make_solvated_system()
        center_nm = _heavy_atom_center_nm(topology, positions)
        shifted_positions = unit.Quantity(
            np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
            + np.array([0.05, 0.0, 0.0], dtype=float),
            unit.nanometer,
        )

        kwargs = {
            "reference_selection": "atomnum 1",
            "cutoff_type": "pme",
            "cutoff": "10.000000 A",
            "moves_per_cycle": 2,
            "batch_size": 1,
        }
        gcmc_a = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            log_level="error",
        )
        gcmc_b = GCMCSolvation(
            topology=topology,
            system=system,
            positions=shifted_positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            log_level="error",
        )

        assert (
            gcmc_a._compute_sampler_fingerprint(**kwargs)
            == gcmc_b._compute_sampler_fingerprint(**kwargs)
        )

    def test_build_loch_positional_restraints_uses_valid_indices(self):
        from H2OSite.simulation.gcmc import GCMCSolvation

        topology, positions, system, ff = _make_solvated_system()
        center_nm = _heavy_atom_center_nm(topology, positions)

        gcmc = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            restrained_atom_indices=[0, 2, 999],
            restraint_k_kcal=5.0,
            log_level="error",
        )

        captured = {}

        class DummyRestraintsModule:
            @staticmethod
            def positional(mols, atoms, k=None, name=None):
                captured["atoms"] = atoms
                captured["k"] = k
                captured["name"] = name
                return "restraints"

        class DummySire:
            restraints = DummyRestraintsModule()

        class DummySystem:
            @staticmethod
            def atoms():
                return [object(), object(), object(), object()]

        restraints = gcmc._build_loch_positional_restraints(
            DummySire(),
            DummySystem(),
        )

        assert restraints == "restraints"
        assert captured["atoms"] == [0, 2]
        assert captured["k"] == "5.000000 kcal mol-1 A-2"

    def test_auto_reference_selection_resolves_to_unique_atoms(self):
        from H2OSite.simulation.gcmc import (
            GCMCSolvation,
            _prepare_openbiosim_runtime,
        )

        topology, positions, system, ff = _make_solvated_system()
        center_nm = _heavy_atom_center_nm(topology, positions)

        gcmc = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            log_level="error",
        )
        selection = gcmc._get_reference_selection()

        assert "atomnum" in selection

        with tempfile.TemporaryDirectory(prefix="gcmc_test_") as tmp:
            prmtop, inpcrd = gcmc._write_openmm_amber_files(tmp)
            sr, _ = _prepare_openbiosim_runtime()
            mols = sr.load(str(prmtop), str(inpcrd))

            selected = mols[selection].atoms()
            n_selected = len(selected)

        assert 1 <= n_selected <= 4
        assert n_selected == selection.count("atomnum")

    def test_openmm_amber_roundtrip_back_to_openmm_preserves_particle_count(self):
        from H2OSite.simulation.gcmc import (
            GCMCSolvation,
            _prepare_openbiosim_runtime,
        )

        topology, positions, system, ff = _make_solvated_system()
        center_nm = _heavy_atom_center_nm(topology, positions)

        gcmc = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            log_level="error",
        )

        with tempfile.TemporaryDirectory(prefix="gcmc_roundtrip_") as tmp:
            prmtop, inpcrd = gcmc._write_openmm_amber_files(tmp)
            sr, _ = _prepare_openbiosim_runtime()
            mols = sr.load(str(prmtop), str(inpcrd))
            new_topology, new_positions = gcmc._roundtrip_back_to_openmm(
                sr, mols, tmp
            )

        assert sum(1 for _ in new_topology.atoms()) == sum(1 for _ in topology.atoms())
        assert len(new_positions) == sum(1 for _ in topology.atoms())

    def test_refresh_cached_prepared_system_updates_coordinates(self):
        import openmm.unit as unit

        from H2OSite.simulation.gcmc import (
            GCMCSolvation,
            _prepare_openbiosim_runtime,
        )

        topology, positions, system, ff = _make_solvated_system()
        center_nm = _heavy_atom_center_nm(topology, positions)
        shifted_positions = unit.Quantity(
            np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
            + np.array([0.1, 0.0, 0.0], dtype=float),
            unit.nanometer,
        )

        gcmc = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            log_level="error",
        )
        refresher = GCMCSolvation(
            topology=topology,
            system=system,
            positions=shifted_positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            log_level="error",
        )

        with tempfile.TemporaryDirectory(prefix="gcmc_refresh_") as tmp:
            prmtop, inpcrd = gcmc._write_openmm_amber_files(tmp)
            sr, _ = _prepare_openbiosim_runtime()
            mols = sr.load(str(prmtop), str(inpcrd))
            refreshed = refresher._refresh_cached_prepared_system(sr, mols)

        before = mols.atoms()[0].coordinates().x().value()
        after = refreshed.atoms()[0].coordinates().x().value()
        assert after == pytest.approx(before + 1.0, abs=1.0e-6)


@requires_openmm
@requires_loch_runtime
@requires_loch_gpu
class TestLochGCMCUltraShort:
    @pytest.mark.slow
    def test_run_loch_ultra_short_smoke(self):
        from H2OSite.simulation.gcmc import GCMCSolvation

        topology, positions, system, ff = _make_solvated_system()
        center_nm = _heavy_atom_center_nm(topology, positions)

        gcmc = GCMCSolvation(
            topology=topology,
            system=system,
            positions=positions,
            center_nm=center_nm,
            radius_nm=0.4,
            forcefield=ff,
            n_ghosts=4,
            batch_size=1,
            platform=_preferred_loch_platform(),
            log_level="error",
        )

        final_topology, final_positions, stats = gcmc.run(
            n_gcmc_moves=4,
            n_md_steps_per_cycle=1,
            n_cycles=2,
        )

        assert final_topology is not None
        assert final_positions is not None
        assert stats["backend"] == "loch"
        assert stats["n_equil_cycles"] == 1
        assert stats["n_prod_cycles"] == 1
        assert stats["n_moves"] == 2
        assert "reference_selection" in stats
        assert sum(1 for _ in final_topology.atoms()) == len(final_positions)

    @pytest.mark.slow
    def test_pipeline_stage7_ultra_short_smoke(self, tmp_path):
        from H2OSite.core.config import WaterMapConfig
        from H2OSite.pipeline import WaterMapPipeline

        topology, positions, system, ff = _make_solvated_system()
        box_vectors = topology.getPeriodicBoxVectors()

        cfg = WaterMapConfig(
            output_dir=str(tmp_path / "gcmc_stage7"),
            jobname="gcmc_stage7",
            water_model="tip3p",
            ligand_distance_A=4.0,
            gcmc_n_moves=4,
            gcmc_n_cycles=2,
            gcmc_md_steps_per_cycle=1,
            gcmc_n_ghosts=4,
            gcmc_batch_size=1,
            gcmc_platform=_preferred_loch_platform(),
        )

        pipeline = WaterMapPipeline(cfg)
        new_topology, new_system, new_positions = pipeline._run_gcmc(
            topology, system, positions, ff, box_vectors
        )

        assert new_topology is not None
        assert new_system is not None
        assert new_positions is not None
        assert new_system.getNumParticles() == sum(1 for _ in new_topology.atoms())
        assert len(new_positions) == new_system.getNumParticles()
