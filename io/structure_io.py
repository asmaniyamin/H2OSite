# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Shared structure I/O helpers for PDB and mmCIF/PDBx."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def is_cif_path(path: str | Path) -> bool:
    """Return True when the path points to an mmCIF/PDBx file."""
    suffix = Path(path).suffix.lower()
    return suffix in {".cif", ".mmcif", ".pdbx"}


def load_openmm_structure_file(path: str | Path):
    """Load a structure with OpenMM from either PDB or mmCIF/PDBx."""
    import openmm.app as app

    path = Path(path)
    if is_cif_path(path):
        structure = app.PDBxFile(str(path))
    else:
        structure = app.PDBFile(str(path))
    return structure.topology, structure.positions


def write_openmm_structure_file(
    topology,
    positions,
    path: str | Path,
    *,
    keep_ids: bool = True,
) -> Path:
    """Write an OpenMM topology/position pair as either PDB or mmCIF."""
    import openmm.app as app

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        if is_cif_path(path):
            app.PDBxFile.writeFile(topology, positions, handle, keepIds=keep_ids)
        else:
            app.PDBFile.writeFile(topology, positions, handle, keepIds=keep_ids)
    return path


def structure_to_openmm(struct) -> tuple:
    """Convert a lightweight Structure into an OpenMM topology/positions pair."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    # OpenMM requires all residues of a chain to be contiguous, and residues
    # within a chain must appear in sequence order. _reorder_solute can move
    # one residue to index 0, breaking both invariants. Sort by
    # (chain_id, residue_id) to restore correct order; atoms within the same
    # residue keep their original relative order (Python sort is stable).
    sort_order = sorted(
        range(struct.n_atoms),
        key=lambda i: (str(struct.chain_ids[i]), int(struct.residue_ids[i])),
    )
    sort_order = np.array(sort_order)

    topology = app.Topology()
    chain_map: dict[str, app.Topology.Chain] = {}
    residue_map: dict[tuple[str, int, str], app.Topology.Residue] = {}

    for index in sort_order:
        chain_id = str(struct.chain_ids[index] or "A")
        residue_id = int(struct.residue_ids[index])
        residue_name = str(struct.residue_names[index])
        atom_name = str(struct.atom_names[index])
        element_symbol = str(struct.elements[index] or atom_name[:1]).strip().title()

        chain = chain_map.get(chain_id)
        if chain is None:
            chain = topology.addChain(chain_id)
            chain_map[chain_id] = chain

        residue_key = (chain_id, residue_id, residue_name)
        residue = residue_map.get(residue_key)
        if residue is None:
            residue = topology.addResidue(residue_name, chain, id=str(residue_id))
            residue_map[residue_key] = residue

        try:
            element = app.Element.getBySymbol(element_symbol)
        except Exception:
            element = None

        topology.addAtom(atom_name, element, residue)

    if struct.box_vectors is not None:
        box_vectors = np.asarray(struct.box_vectors, dtype=float)
        if box_vectors.shape == (3, 3):
            topology.setPeriodicBoxVectors(
                tuple(openmm.Vec3(*row) for row in box_vectors) * unit.angstrom
            )

    positions = unit.Quantity(
        np.asarray(struct.positions, dtype=float)[sort_order], unit.angstrom
    )
    return topology, positions


def write_structure_file(struct, path: str | Path) -> Path:
    """Write a lightweight Structure to either PDB or mmCIF."""
    topology, positions = structure_to_openmm(struct)
    return write_openmm_structure_file(topology, positions, path)
