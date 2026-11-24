# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for embedded-ligand input handling in the biosim prep layer."""

from __future__ import annotations

from pathlib import Path

from H2OSite.core.config import WaterMapConfig
from H2OSite.prep.input_preparation import WaterMapInput
from H2OSite.tests.conftest import requires_mdanalysis, requires_rdkit


def _write_embedded_ligand_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3 ZN   ZN  Z 501       4.000   0.000   0.000  1.00  0.00          ZN\n"
        "HETATM    4  C1  85C Z 502       5.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    5  O1  85C Z 502       6.200   0.000   0.000  1.00  0.00           O\n"
        "HETATM    6  N1  85C Z 502       5.000   1.200   0.000  1.00  0.00           N\n"
        "CONECT    4    5    6\n"
        "CONECT    5    4\n"
        "CONECT    6    4\n"
        "END\n"
    )


@requires_mdanalysis
@requires_rdkit
def test_prepare_extracts_embedded_ligand_and_writes_parameterizable_files(tmp_path):
    pdb_path = tmp_path / "embedded_85c.pdb"
    _write_embedded_ligand_pdb(pdb_path)

    cfg = WaterMapConfig(
        protein_path=str(pdb_path),
        output_dir=str(tmp_path / "out"),
        jobname="embedded_85c",
        ligand_resnames=["85C"],
        retain_ligand=True,
        truncate_protein=False,
        treat_solvent="delete",
    )

    inp = WaterMapInput(cfg)
    protein, ligand = inp.prepare()
    prot_pdb, lig_pdb = inp.write_prepared(tmp_path / "prep")

    assert protein.n_atoms == 3
    assert ligand.n_atoms == 3
    assert inp.prepared_ligand_sdf_path is not None
    assert inp.prepared_ligand_sdf_path.exists()
    assert prot_pdb.exists()
    assert lig_pdb.exists()
    assert any(res == "ZN" for res in protein.residue_names)
    assert all(res == "85C" for res in ligand.residue_names)
