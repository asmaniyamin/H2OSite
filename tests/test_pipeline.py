# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Fast tests for pipeline orchestration."""

from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
import openmm.unit as unit

from H2OSite.core.config import WaterMapConfig
from H2OSite.pipeline import WaterMapPipeline


class _DummyTracker:
    def __init__(self):
        self.n_frames = 1
        self.frame_resids = [np.empty(0, dtype=int)]
        self.trajectory_path = None
        self.checkpoint_path = None
        self.center = np.zeros(3, dtype=float)

    def get_per_frame_positions(self):
        return [np.empty((0, 3))]

    def get_all_positions(self):
        return np.empty((0, 3))


class _FakeResidue:
    def __init__(self, name):
        self.name = name


class _FakeElement:
    def __init__(self, symbol):
        self.symbol = symbol


class _FakeAtom:
    def __init__(self, index, residue_name, atom_name, element_symbol):
        self.index = index
        self.residue = _FakeResidue(residue_name)
        self.name = atom_name
        self.element = None if element_symbol is None else _FakeElement(element_symbol)


class _FakeTopology:
    def __init__(self, atoms):
        self._atoms = list(atoms)

    def atoms(self):
        return iter(self._atoms)


def _patch_lightweight_run(monkeypatch, tmp_path):
    protocol_calls = []

    class FakeProtocol:
        def __init__(self, topology, positions, system, output_dir, gpu_index):
            self.output_dir = output_dir

        def run_equilibration(self, *args, **kwargs):
            protocol_calls.append(str(self.output_dir))
            return np.zeros((4, 3), dtype=float), "vel", "box"

    class FakeClusterer:
        def __init__(self, *args, **kwargs):
            pass

        def cluster(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        "H2OSite.simulation.protocol.SimulationProtocol",
        FakeProtocol,
    )
    monkeypatch.setattr(
        "H2OSite.analysis.clustering.HydrationSiteClustering",
        FakeClusterer,
    )
    monkeypatch.setattr(
        WaterMapPipeline,
        "_prepare_and_build",
        lambda self: ("top", "pos", "system", "ff"),
    )
    monkeypatch.setattr(
        WaterMapPipeline,
        "_classify_atoms",
        lambda self, topology: ([], []),
    )
    monkeypatch.setattr(
        WaterMapPipeline,
        "_run_production",
        lambda self, topology, system, positions, velocities, **kwargs: _DummyTracker(),
    )
    monkeypatch.setattr(
        WaterMapPipeline,
        "_compute_thermodynamics",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        WaterMapPipeline,
        "_compute_hbonds",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        WaterMapPipeline,
        "_generate_output",
        lambda self, *args, **kwargs: None,
    )
    return protocol_calls


def test_post_gcmc_equil_skipped_when_gcmc_skipped(monkeypatch, tmp_path):
    protocol_calls = _patch_lightweight_run(monkeypatch, tmp_path)

    cfg = WaterMapConfig(
        output_dir=str(tmp_path / "out"),
        skip_gcmc=True,
    )
    pipeline = WaterMapPipeline(cfg)
    pipeline.run()

    assert protocol_calls == [str(tmp_path / "out" / "equilibration")]


def test_post_gcmc_equil_can_be_forced_without_gcmc(monkeypatch, tmp_path):
    protocol_calls = _patch_lightweight_run(monkeypatch, tmp_path)

    cfg = WaterMapConfig(
        output_dir=str(tmp_path / "out_force"),
        skip_gcmc=True,
        run_post_gcmc_without_gcmc=True,
    )
    pipeline = WaterMapPipeline(cfg)
    pipeline.run()

    assert protocol_calls == [
        str(tmp_path / "out_force" / "equilibration"),
        str(tmp_path / "out_force" / "post_gcmc_equil"),
    ]


def test_tracking_center_prefers_gcmc_reference_indices(tmp_path):
    cfg = WaterMapConfig(output_dir=str(tmp_path / "tracking"))
    pipeline = WaterMapPipeline(cfg)
    pipeline._last_gcmc_stats = {
        "tracking_reference_indices": [1, 3],
        "reference_indices": [1, 3],
        "reference_selection": "atomnum 2 or atomnum 4",
    }

    positions_nm = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.2, 0.0],
        ],
        dtype=float,
    )

    center_nm, source = pipeline._resolve_tracking_center_nm(positions_nm)

    assert source == "gcmc_tracking_reference_indices"
    np.testing.assert_allclose(center_nm, [0.3, 0.1, 0.0])


def test_tracking_center_is_canonicalized_under_pbc(tmp_path):
    cfg = WaterMapConfig(output_dir=str(tmp_path / "tracking_pbc"))
    pipeline = WaterMapPipeline(cfg)
    pipeline._last_gcmc_stats = {
        "tracking_reference_indices": [0, 1, 2, 3],
        "reference_indices": [0, 1, 2, 3],
        "reference_selection": "atomnum 1 or atomnum 2 or atomnum 3 or atomnum 4",
    }

    positions_nm = np.array(
        [
            [0.0746559, 0.27178147, 12.65062484],
            [-0.1086597, -0.12938072, 13.19475608],
            [0.14987203, 0.40153375, 12.81182617],
            [-0.06327229, -0.32157211, 12.90821982],
        ],
        dtype=float,
    )

    box_vectors = unit.Quantity(
        (
            openmm.Vec3(12.89962516, 0.0, 0.0),
            openmm.Vec3(0.0, 12.89962516, 0.0),
            openmm.Vec3(0.0, 0.0, 12.89962516),
        ),
        unit.nanometer,
    )

    center_nm, source = pipeline._resolve_tracking_center_nm(
        positions_nm,
        box_vectors=box_vectors,
    )

    assert source == "gcmc_tracking_reference_indices"
    np.testing.assert_allclose(
        center_nm,
        [0.01314898, 0.0555906, -0.00826843],
        atol=1e-6,
    )


def test_generate_display_outputs_uses_tracker_center(monkeypatch, tmp_path):
    cfg = WaterMapConfig(output_dir=str(tmp_path / "display"))
    pipeline = WaterMapPipeline(cfg)

    captured = {}

    def fake_generate_display_artifacts(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "H2OSite.postprocessing.generate_display_artifacts",
        fake_generate_display_artifacts,
    )

    class DummyTracker:
        trajectory_path = tmp_path / "display" / "production.dcd"
        center = np.array([0.12, -0.03, 0.05], dtype=float)

    DummyTracker.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    DummyTracker.trajectory_path.write_text("")

    pipeline._generate_display_outputs(topology="top", tracker=DummyTracker())

    np.testing.assert_allclose(
        captured["tracking_center_A"],
        [1.2, -0.3, 0.5],
    )


def test_extract_stage11b_data_uses_tracker_center(monkeypatch, tmp_path):
    cfg = WaterMapConfig(output_dir=str(tmp_path / "stage11"))
    pipeline = WaterMapPipeline(cfg)

    captured = {}

    class FakeAnalyzer:
        def __init__(self, topology, system, trajectory_file, checkpoint_file):
            pass

        def compute_tracked_water_properties(
            self,
            frame_resids,
            *,
            active_site_center_A,
            active_site_radius_A,
        ):
            captured["center_A"] = np.asarray(active_site_center_A, dtype=float)
            captured["radius_A"] = float(active_site_radius_A)
            return [], {"pw": [], "ww": []}

    monkeypatch.setattr(
        "H2OSite.analysis.trajectory_properties.ProductionTrajectoryAnalyzer",
        FakeAnalyzer,
    )

    class DummyTracker:
        trajectory_path = tmp_path / "stage11" / "production.dcd"
        checkpoint_path = None
        frame_resids = [np.array([1, 2], dtype=int)]
        center = np.array([0.08, 0.01, -0.04], dtype=float)

    DummyTracker.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    DummyTracker.trajectory_path.write_text("")

    per_frame_dipoles, per_frame_energies = pipeline._extract_stage11b_data(
        topology="top",
        system="sys",
        tracker=DummyTracker(),
    )

    assert per_frame_dipoles == []
    assert per_frame_energies == {"pw": [], "ww": []}
    np.testing.assert_allclose(captured["center_A"], [0.8, 0.1, -0.4])
    assert captured["radius_A"] == cfg.ligand_distance_A


def test_prepare_and_build_passes_ligand_files_and_terminal_treatment(
    monkeypatch,
    tmp_path,
):
    cfg = WaterMapConfig(
        output_dir=str(tmp_path / "prep_build"),
        retain_ligand=True,
        terminal_treatment="add_hydrogens",
        ligand_forcefield="openff-2.1.0.offxml",
    )
    pipeline = WaterMapPipeline(cfg)
    captured = {}

    class FakeInput:
        def __init__(self, config):
            self.prepared_ligand_sdf_path = tmp_path / "prep_build" / "prep" / "lig.sdf"
            self.prepared_protein_cif_path = tmp_path / "prep_build" / "prep" / "prot.cif"
            self.prepared_ligand_cif_path = tmp_path / "prep_build" / "prep" / "lig.cif"

        def prepare(self):
            return None

        def write_prepared(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            prot = output_dir / "prot.pdb"
            lig = output_dir / "lig.pdb"
            self.prepared_ligand_sdf_path.write_text("")
            self.prepared_protein_cif_path.write_text("")
            self.prepared_ligand_cif_path.write_text("")
            prot.write_text("")
            lig.write_text("")
            return prot, lig

    class FakeBuilder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def build(self):
            return "top", "pos", "sys", "ff"

    monkeypatch.setattr(
        "H2OSite.prep.input_preparation.WaterMapInput",
        FakeInput,
    )
    monkeypatch.setattr(
        "H2OSite.prep.system_builder.SystemBuilder",
        FakeBuilder,
    )

    result = pipeline._prepare_and_build()

    assert result == ("top", "pos", "sys", "ff")
    assert captured["terminal_treatment"] == "add_hydrogens"
    assert captured["ligand_forcefield"] == "openff-2.1.0.offxml"
    assert captured["ligand_sdf"] == tmp_path / "prep_build" / "prep" / "lig.sdf"
    assert captured["protein_pdb"] == tmp_path / "prep_build" / "prep" / "prot.cif"
    assert captured["ligand_pdb"] == tmp_path / "prep_build" / "prep" / "lig.cif"


def test_run_production_passes_configured_restraint_strength(monkeypatch, tmp_path):
    cfg = WaterMapConfig(
        output_dir=str(tmp_path / "prod"),
        production_restraint_k_kcal=10.0,
    )
    pipeline = WaterMapPipeline(cfg)
    captured = {}

    class FakeProduction:
        def __init__(
            self,
            topology,
            system,
            positions,
            velocities=None,
            box_vectors=None,
            output_dir=None,
            gpu_index=None,
        ):
            captured["box_vectors"] = box_vectors

        def run(self, **kwargs):
            captured.update(kwargs)
            return _DummyTracker()

    monkeypatch.setattr(
        "H2OSite.simulation.production.ProductionMD",
        FakeProduction,
    )

    tracker = pipeline._run_production(
        topology="top",
        system="sys",
        positions=np.zeros((3, 3), dtype=float),
        velocities="vel",
        restrain_indices=[0, 2],
        active_site_center_nm=np.array([0.1, 0.2, 0.3], dtype=float),
        active_site_center_source="test",
        box_vectors="box",
    )

    assert isinstance(tracker, _DummyTracker)
    assert captured["restraint_k_kcal"] == 10.0
    assert captured["box_vectors"] == "box"


def test_non_water_heavy_atom_indices_excludes_waters_and_hydrogens(tmp_path):
    cfg = WaterMapConfig(output_dir=str(tmp_path / "gcmc_indices"))
    pipeline = WaterMapPipeline(cfg)
    topology = _FakeTopology(
        [
            _FakeAtom(0, "ALA", "CA", "C"),
            _FakeAtom(1, "HOH", "O", "O"),
            _FakeAtom(2, "ALA", "H", "H"),
            _FakeAtom(3, "85C", "C1", "C"),
            _FakeAtom(4, "SOL", "OW", "O"),
            _FakeAtom(5, "ZN", "ZN", "Zn"),
        ]
    )

    indices = pipeline._non_water_heavy_atom_indices(topology)

    assert indices == [0, 3, 5]


def test_run_gcmc_passes_stage7_restraint_configuration(monkeypatch, tmp_path):
    cfg = WaterMapConfig(
        output_dir=str(tmp_path / "gcmc"),
        gcmc_restraint_k_kcal=7.5,
        run_gcmc_handoff_relaxation=False,
    )
    pipeline = WaterMapPipeline(cfg)
    captured = {}

    topology = _FakeTopology(
        [
            _FakeAtom(0, "ALA", "CA", "C"),
            _FakeAtom(1, "HOH", "O", "O"),
            _FakeAtom(2, "ALA", "H", "H"),
        ]
    )

    class FakeGCMC:
        def __init__(self, topology, system, positions, **kwargs):
            captured.update(kwargs)

        def run(self, n_gcmc_moves, n_md_steps_per_cycle, n_cycles):
            return topology, np.zeros((3, 3), dtype=float), {"backend": "loch"}

    class FakeFF:
        def createSystem(self, topology, **kwargs):
            captured["rebuild_kwargs"] = kwargs
            return "new_system"

    monkeypatch.setattr(
        "H2OSite.simulation.gcmc.GCMCSolvation",
        FakeGCMC,
    )
    monkeypatch.setattr(
        "H2OSite.simulation.gcmc.extract_nonbonded_forcefield_kwargs",
        lambda system: {},
    )

    new_topology, new_system, new_positions = pipeline._run_gcmc(
        topology=topology,
        system="system",
        positions=np.zeros((3, 3), dtype=float),
        ff=FakeFF(),
        box_vectors="box",
    )

    assert new_topology is topology
    assert new_system == "new_system"
    assert isinstance(new_positions, np.ndarray)
    assert captured["restrained_atom_indices"] == [0]
    assert captured["restraint_k_kcal"] == 7.5
