# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Fast tests for capped-input terminal treatment behavior."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from H2OSite.core.config import WaterMapConfig
from H2OSite.core.structure import Structure
from H2OSite.prep.input_preparation import WaterMapInput
from H2OSite.prep.system_builder import SystemBuilder


def _make_structure(
    residue_names: list[str],
    atom_names: list[str],
    elements: list[str],
    *,
    is_protein: list[bool],
    is_ligand: list[bool],
) -> Structure:
    n_atoms = len(residue_names)
    return Structure(
        positions=np.zeros((n_atoms, 3), dtype=float),
        elements=list(elements),
        atom_names=list(atom_names),
        residue_names=list(residue_names),
        residue_ids=list(range(1, n_atoms + 1)),
        chain_ids=["A"] * n_atoms,
        is_water=np.zeros(n_atoms, dtype=bool),
        is_protein=np.asarray(is_protein, dtype=bool),
        is_ligand=np.asarray(is_ligand, dtype=bool),
    )


def test_preserve_existing_caps_auto_normalizes_nma(monkeypatch, tmp_path):
    cfg = WaterMapConfig(
        protein_path="protein.pdb",
        ligand_path="ligand.sdf",
        output_dir=str(tmp_path / "out"),
        terminal_treatment="preserve_existing_caps",
        normalize_schrodinger_residues=False,
    )
    inp = WaterMapInput(cfg)

    protein = _make_structure(
        ["NMA", "NMA", "ALA", "ALA"],
        ["N", "CH3", "N", "CA"],
        ["N", "C", "N", "C"],
        is_protein=[True, True, True, True],
        is_ligand=[False, False, False, False],
    )
    ligand = _make_structure(
        ["LIG", "LIG"],
        ["C1", "O1"],
        ["C", "O"],
        is_protein=[False, False],
        is_ligand=[True, True],
    )

    calls = {"count": 0}

    def fake_load_structure(path, ligand_resname=None):
        calls["count"] += 1
        return protein.copy() if calls["count"] == 1 else ligand.copy()

    monkeypatch.setattr(
        "H2OSite.prep.input_preparation.load_structure",
        fake_load_structure,
    )
    monkeypatch.setattr(
        WaterMapInput,
        "_load_rdkit_molecule",
        lambda self, path: None,
    )

    inp._load_inputs()

    assert inp.protein is not None
    assert "NMA" not in inp.protein.residue_names
    assert inp.protein.residue_names[:2] == ["NME", "NME"]


def test_system_builder_preserve_existing_caps_skips_terminal_modification():
    class FakeResidue:
        def __init__(self, name):
            self.name = name

    class FakeTopology:
        def residues(self):
            return iter([FakeResidue("ACE"), FakeResidue("NME"), FakeResidue("ALA")])

    class FakeModeller:
        def __init__(self):
            self.topology = FakeTopology()

        def addHydrogens(self, ff):
            raise AssertionError("preserve_existing_caps must not call addHydrogens")

    builder = SystemBuilder(
        protein_pdb="dummy.pdb",
        terminal_treatment="preserve_existing_caps",
    )
    builder._cap_termini = lambda modeller, ff: (_ for _ in ()).throw(
        AssertionError("preserve_existing_caps must not add caps")
    )

    modeller = FakeModeller()
    result = builder._apply_terminal_treatment(modeller, ff=None)

    assert result is modeller
    assert builder._terminal_cap_summary(modeller.topology) == {
        "ACE": 1,
        "NME": 1,
        "NMA": 0,
    }


def test_run_5hxb_build_config_uses_requested_terminal_treatment(tmp_path):
    build_config = pytest.importorskip(
        "testing_on_real_systems.run_5hxb_production",
        reason="System-specific runner was removed from the retained package",
    ).build_config
    args = Namespace(
        protein=Path("/tmp/5hxb_prepped.pdb"),
        ligand_sdf=None,
        mode="holo",
        terminal_treatment="preserve_existing_caps",
        simulation_time_ns=0.1,
        output_dir=tmp_path / "5hxb_out",
        dedup=False,
    )

    cfg = build_config(args)

    assert cfg.terminal_treatment == "preserve_existing_caps"
