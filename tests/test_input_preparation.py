# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for prep/input_preparation.py - structure preparation pipeline."""


import numpy as np
import pytest

from H2OSite.core.config import WaterMapConfig
from H2OSite.core.structure import Structure
from H2OSite.prep.input_preparation import WaterMapInput, _write_pdb


# ---- Fast: unit tests with synthetic structures ----

class TestWritePDB:
    def test_writes_valid_pdb(self, tmp_path):
        s = Structure(
            positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            elements=["C", "N"],
            atom_names=["CA", "N"],
            residue_names=["ALA", "ALA"],
            residue_ids=[1, 1],
            chain_ids=["A", "A"],
            is_water=np.array([False, False]),
            is_protein=np.array([True, True]),
            is_ligand=np.array([False, False]),
        )
        path = tmp_path / "test.pdb"
        _write_pdb(s, path)
        text = path.read_text()
        assert "ATOM" in text
        assert "END" in text
        lines = [l for l in text.splitlines() if l.startswith("ATOM")]
        assert len(lines) == 2


class TestInputPreparationMethods:
    @pytest.fixture
    def protein(self):
        # 5 protein residue atoms + 2 water atoms
        return Structure(
            positions=np.array([
                [0, 0, 0], [1, 0, 0], [2, 0, 0],  # res 1 near origin
                [50, 50, 50], [51, 50, 50],  # res 2 far away
                [0.5, 0.5, 0], [0.6, 0.6, 0],  # waters near
            ], dtype=float),
            elements=["C", "N", "O", "C", "N", "O", "H"],
            atom_names=["CA", "N", "O", "CA", "N", "OW", "H1"],
            residue_names=["ALA", "ALA", "ALA", "GLY", "GLY", "HOH", "HOH"],
            residue_ids=[1, 1, 1, 2, 2, 3, 3],
            chain_ids=["A"] * 7,
            is_water=np.array([0, 0, 0, 0, 0, 1, 1], dtype=bool),
            is_protein=np.array([1, 1, 1, 1, 1, 0, 0], dtype=bool),
            is_ligand=np.array([0, 0, 0, 0, 0, 0, 0], dtype=bool),
        )

    def test_truncate_removes_distant(self, protein):
        cfg = WaterMapConfig(truncate_distance_A=10.0)
        inp = WaterMapInput(cfg)
        inp.protein = protein
        inp._truncate_protein(np.array([0.0, 0.0, 0.0]))
        # res 2 at (50,50,50) should be removed, waters kept
        assert inp.protein.n_atoms == 5  # 3 from res1 + 2 waters

    def test_handle_crystal_waters_delete(self, protein):
        cfg = WaterMapConfig(treat_solvent="delete")
        inp = WaterMapInput(cfg)
        inp.protein = protein
        inp._handle_crystal_waters(np.array([0, 0, 0]))
        assert np.sum(inp.protein.is_water) == 0
        assert inp.protein.n_atoms == 5

    def test_handle_crystal_waters_as_solvent(self, protein):
        cfg = WaterMapConfig(treat_solvent="as_solvent")
        inp = WaterMapInput(cfg)
        inp.protein = protein
        inp._handle_crystal_waters(np.array([0, 0, 0]))
        assert inp.crystal_waters is not None
        assert len(inp.crystal_waters) == 2
        assert np.sum(inp.protein.is_water) == 0

    def test_handle_crystal_waters_as_solute(self, protein):
        cfg = WaterMapConfig(treat_solvent="as_solute")
        inp = WaterMapInput(cfg)
        inp.protein = protein
        inp._handle_crystal_waters(np.array([0, 0, 0]))
        assert np.sum(inp.protein.is_water) == 2

    def test_center_on_ligand(self, protein):
        cfg = WaterMapConfig()
        inp = WaterMapInput(cfg)
        inp.protein = protein
        inp.ligand = Structure(
            positions=np.array([[5.0, 5.0, 5.0]]),
            elements=["C"], atom_names=["C1"],
            residue_names=["LIG"], residue_ids=[1], chain_ids=["L"],
            is_water=np.zeros(1, dtype=bool),
            is_protein=np.zeros(1, dtype=bool),
            is_ligand=np.ones(1, dtype=bool),
        )
        center = np.array([5.0, 5.0, 5.0])
        inp._center_on_ligand(center)
        np.testing.assert_allclose(inp.ligand.positions[0], [0, 0, 0])
        np.testing.assert_allclose(inp.translation_vector, [-5, -5, -5])
