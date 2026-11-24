# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for core/structure.py - Structure dataclass and file loading."""

import numpy as np
import pytest

from H2OSite.core.structure import Structure, load_structure
from H2OSite.io.structure_io import write_structure_file
from H2OSite.tests.conftest import requires_mdanalysis, requires_openmm


# ---- Fast: Structure dataclass ----

class TestStructure:
    @pytest.fixture
    def simple_struct(self):
        return Structure(
            positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            elements=["C", "N", "O"],
            atom_names=["CA", "N", "O"],
            residue_names=["ALA", "ALA", "HOH"],
            residue_ids=[1, 1, 2],
            chain_ids=["A", "A", "A"],
            is_water=np.array([False, False, True]),
            is_protein=np.array([True, True, False]),
            is_ligand=np.array([False, False, False]),
        )

    def test_n_atoms(self, simple_struct):
        assert simple_struct.n_atoms == 3

    def test_centroid(self, simple_struct):
        c = simple_struct.centroid
        np.testing.assert_allclose(c, [1 / 3, 1 / 3, 0])

    def test_heavy_atom_mask(self):
        s = Structure(
            positions=np.zeros((4, 3)),
            elements=["C", "H", "N", "H"],
            atom_names=["CA", "H1", "N", "H2"],
            residue_names=["ALA"] * 4,
            residue_ids=[1] * 4,
            chain_ids=["A"] * 4,
            is_water=np.zeros(4, dtype=bool),
            is_protein=np.ones(4, dtype=bool),
            is_ligand=np.zeros(4, dtype=bool),
        )
        mask = s.heavy_atom_mask()
        assert mask.tolist() == [True, False, True, False]

    def test_select_within(self, simple_struct):
        mask = simple_struct.select_within(np.array([0, 0, 0]), 0.5)
        assert mask[0] and not mask[1]

    def test_translate(self, simple_struct):
        simple_struct.translate(np.array([10, 0, 0]))
        assert simple_struct.positions[0, 0] == 10.0

    def test_copy_independent(self, simple_struct):
        c = simple_struct.copy()
        c.positions[0, 0] = 999
        assert simple_struct.positions[0, 0] != 999





@requires_mdanalysis
class TestLoadSmallPDB:
    def test_round_trip_write_load(self, tmp_path):
        """Write a tiny PDB, load it back, verify positions."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      2  N   ALA A   1       4.000   5.000   6.000  1.00  0.00           N\n"
            "HETATM    3  O   HOH A   2       7.000   8.000   9.000  1.00  0.00           O\n"
            "END\n"
        )
        s = load_structure(pdb)
        assert s.n_atoms == 3
        np.testing.assert_allclose(s.positions[0], [1, 2, 3], atol=0.01)
        assert s.is_water[2]
        assert s.is_protein[0]


@requires_openmm
def test_load_small_mmcif_round_trip(tmp_path):
    struct = Structure(
        positions=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=float,
        ),
        elements=["C", "N", "O"],
        atom_names=["CA", "N", "O"],
        residue_names=["ALA", "ALA", "HOH"],
        residue_ids=[1, 1, 2],
        chain_ids=["A", "A", "A"],
        is_water=np.array([False, False, True]),
        is_protein=np.array([True, True, False]),
        is_ligand=np.array([False, False, False]),
    )

    cif_path = tmp_path / "test.cif"
    write_structure_file(struct, cif_path)

    loaded = load_structure(cif_path)
    assert loaded.n_atoms == 3
    np.testing.assert_allclose(loaded.positions[0], [1.0, 2.0, 3.0], atol=0.01)
    assert loaded.is_water[2]
    assert loaded.is_protein[0]
