# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Structure I/O and manipulation using MDAnalysis and RDKit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from H2OSite.io.structure_io import load_openmm_structure_file

logger = logging.getLogger(__name__)

WATER_RESNAMES = {
    "HOH", "WAT", "TIP3", "TIP4", "T3P", "T4P", "SOL", "TIP4P", "TIP3P",
}

PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "HIE", "HID", "HIP", "CYX", "ACE", "NME", "NMA",
    "ASH", "GLH",
}


@dataclass
class Structure:
    """Lightweight wrapper around molecular structure data.

    Stores positions, topology info, and metadata needed for H2OSite
    without depending on any proprietary SDK.
    """

    positions: np.ndarray  # (N, 3) Å
    elements: list[str]  # element symbols
    atom_names: list[str]
    residue_names: list[str]
    residue_ids: list[int]
    chain_ids: list[str]
    is_water: np.ndarray  # bool mask
    is_protein: np.ndarray  # bool mask
    is_ligand: np.ndarray  # bool mask
    box_vectors: np.ndarray | None = None  # (3, 3) Å or None
    properties: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    def ligand_centroid(self) -> np.ndarray:
        lig_pos = self.positions[self.is_ligand]
        if len(lig_pos) == 0:
            raise ValueError("No ligand atoms found in structure")
        return lig_pos.mean(axis=0)

    def heavy_atom_mask(self) -> np.ndarray:
        return np.array([e != "H" for e in self.elements])

    def select_within(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Return boolean mask of atoms within `radius` Å of `center`."""
        dists = np.linalg.norm(self.positions - center, axis=1)
        return dists <= radius

    def translate(self, vector: np.ndarray) -> None:
        self.positions += vector

    def copy(self) -> Structure:
        import copy
        return copy.deepcopy(self)


def _normalise_ligand_resnames(
    ligand_resname: str | list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if ligand_resname is None:
        return set()
    if isinstance(ligand_resname, str):
        return {ligand_resname}
    return {str(name) for name in ligand_resname}


def reclassify_structure(
    struct: Structure,
    ligand_resnames: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> Structure:
    """Recompute water/protein/ligand masks after residue-name edits."""
    ligand_names = _normalise_ligand_resnames(ligand_resnames)
    residue_names = list(struct.residue_names)

    is_water = np.array([rn in WATER_RESNAMES for rn in residue_names], dtype=bool)
    is_protein = np.array(
        [rn in PROTEIN_RESNAMES for rn in residue_names],
        dtype=bool,
    )
    if ligand_names:
        is_ligand = np.array([rn in ligand_names for rn in residue_names], dtype=bool)
    else:
        is_ligand = ~is_water & ~is_protein

    struct.is_water = is_water
    struct.is_protein = is_protein
    struct.is_ligand = is_ligand
    return struct


def normalize_schrodinger_residue_names(
    struct: Structure,
    manual_histidine_states: dict[str, str] | None = None,
    ligand_resnames: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, int]:
    """Normalize PDB residue names in-place.

    Supported fixes:
    - Generic HIS -> HID / HIE / HIP using an explicit residue-state map or
      the residue's side-chain hydrogen names.
    - NMA -> NME for AMBER/OpenMM compatibility.
    """
    manual_histidine_states = manual_histidine_states or {}
    histidine_map = {
        str(key): value.upper() for key, value in manual_histidine_states.items()
    }

    residue_names = list(struct.residue_names)
    atom_names = list(struct.atom_names)
    residue_ids = list(struct.residue_ids)
    chain_ids = list(struct.chain_ids)

    histidine_updates = 0
    nma_updates = 0

    residue_groups: dict[tuple[str, int], list[int]] = {}
    for idx, (chain_id, residue_id) in enumerate(zip(chain_ids, residue_ids)):
        residue_groups.setdefault((chain_id, int(residue_id)), []).append(idx)

    for (chain_id, residue_id), indices in residue_groups.items():
        current_name = residue_names[indices[0]]

        if current_name == "NMA":
            for idx in indices:
                residue_names[idx] = "NME"
            nma_updates += 1
            continue

        if current_name != "HIS":
            continue

        override_key = f"{chain_id}:{residue_id}"
        override_state = histidine_map.get(override_key)
        if override_state is not None:
            new_name = override_state
        else:
            residue_atom_names = {atom_names[idx] for idx in indices}
            has_hd1 = "HD1" in residue_atom_names
            has_he2 = "HE2" in residue_atom_names
            if has_hd1 and has_he2:
                new_name = "HIP"
            elif has_hd1:
                new_name = "HID"
            elif has_he2:
                new_name = "HIE"
            else:
                new_name = "HIS"

        if new_name != current_name:
            for idx in indices:
                residue_names[idx] = new_name
            histidine_updates += 1

    struct.residue_names = residue_names
    reclassify_structure(struct, ligand_resnames=ligand_resnames)

    return {
        "histidines_renamed": histidine_updates,
        "nma_renamed": nma_updates,
    }


def load_structure(path: str | Path, **kwargs) -> Structure:
    """Load a structure from PDB, GRO, MOL2, or SDF file.

    Uses MDAnalysis for PDB/GRO and RDKit for SDF/MOL2.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".pdb", ".gro", ".pdbqt"):
        return _load_with_mdanalysis(path, **kwargs)
    elif suffix in (".cif", ".mmcif", ".pdbx"):
        return _load_with_openmm(path, **kwargs)
    elif suffix in (".sdf", ".mol2", ".mol"):
        return _load_with_rdkit(path, **kwargs)
    else:
        # Try MDAnalysis first as it handles many formats
        try:
            return _load_with_mdanalysis(path, **kwargs)
        except Exception:
            return _load_with_rdkit(path, **kwargs)


def _load_with_openmm(path: Path, **kwargs) -> Structure:
    import openmm.unit as unit

    topology, positions = load_openmm_structure_file(path)

    positions_A = np.asarray(
        positions.value_in_unit(unit.angstrom),
        dtype=float,
    )

    atom_names: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[int] = []
    chain_ids: list[str] = []
    elements: list[str] = []

    # CIF exports may assign the same label_seq_id to the
    # last amino acid AND its NMA/ACE cap (e.g. VAL 632 and NMA 632 in the
    # same chain). Build a per-chain map from seq_id -> residue_name so we
    # can detect these clashes and give the cap a unique id.
    seq_id_names: dict[tuple[str, int], str] = {}
    id_remaps: dict[int, int] = {}  # topology residue index -> corrected id
    # Pre-collect all seq_ids per chain for clash-free new-id lookup.
    chain_seq_ids: dict[str, set[int]] = {}
    for res in topology.residues():
        cid = str(res.chain.id)
        try:
            sid = int(res.id)
        except Exception:
            sid = res.index
        chain_seq_ids.setdefault(cid, set()).add(sid)

    for res in topology.residues():
        chain_id = str(res.chain.id)
        try:
            seq_id = int(res.id)
        except Exception:
            seq_id = res.index
        key = (chain_id, seq_id)
        if key in seq_id_names and seq_id_names[key] != res.name:
            used = chain_seq_ids[chain_id] | set(id_remaps.values())
            new_id = seq_id + 1
            while new_id in used:
                new_id += 1
            id_remaps[res.index] = new_id
            chain_seq_ids[chain_id].add(new_id)
        else:
            seq_id_names[key] = res.name

    for atom in topology.atoms():
        atom_names.append(atom.name)
        residue_names.append(atom.residue.name)
        try:
            raw_id = int(atom.residue.id)
        except Exception:
            raw_id = len(residue_ids) + 1
        residue_ids.append(id_remaps.get(atom.residue.index, raw_id))
        chain_ids.append(str(atom.residue.chain.id))
        if atom.element is not None:
            elements.append(atom.element.symbol)
        else:
            elements.append(atom.name[:1])

    is_water = np.array([rn in WATER_RESNAMES for rn in residue_names], dtype=bool)
    is_protein = np.array([rn in PROTEIN_RESNAMES for rn in residue_names], dtype=bool)
    is_ligand = ~is_water & ~is_protein

    ligand_resname = kwargs.get("ligand_resname")
    ligand_resnames = _normalise_ligand_resnames(ligand_resname)
    if ligand_resnames:
        is_ligand = np.array([rn in ligand_resnames for rn in residue_names], dtype=bool)

    box_vectors = None
    try:
        periodic_box = topology.getPeriodicBoxVectors()
    except Exception:
        periodic_box = None
    if periodic_box is not None:
        box_vectors = np.asarray(
            [
                [vec.x, vec.y, vec.z]
                for vec in periodic_box.value_in_unit(unit.angstrom)
            ],
            dtype=float,
        )

    return Structure(
        positions=positions_A,
        elements=elements,
        atom_names=atom_names,
        residue_names=residue_names,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        is_water=is_water,
        is_protein=is_protein,
        is_ligand=is_ligand,
        box_vectors=box_vectors,
    )


def _load_with_mdanalysis(path: Path, **kwargs) -> Structure:
    import MDAnalysis as mda

    u = mda.Universe(str(path))
    ag = u.atoms

    # Filter alternate conformations: keep primary conformer only.
    # PDB files may contain alt-loc indicators (A/B/C/...) for residues
    # with multiple observed conformations.  Keeping both causes
    # overlapping atoms -> NaN energies in OpenMM.
    try:
        altlocs = ag.altLocs
        mask = np.array([(a in ('', ' ', 'A')) for a in altlocs])
        if mask.sum() < len(ag):
            n_removed = len(ag) - mask.sum()
            ag = ag[mask]
            logger.info("Removed %d alternate-conformation atoms (kept altLoc A)", n_removed)
    except (AttributeError, mda.exceptions.NoDataError):
        pass

    residue_names = [a.resname for a in ag]
    is_water = np.array([rn in WATER_RESNAMES for rn in residue_names])
    is_protein = np.array([rn in PROTEIN_RESNAMES for rn in residue_names])
    is_ligand = ~is_water & ~is_protein

    # Override ligand assignment if keyword provided
    ligand_resname = kwargs.get("ligand_resname")
    ligand_resnames = _normalise_ligand_resnames(ligand_resname)
    if ligand_resnames:
        is_ligand = np.array([rn in ligand_resnames for rn in residue_names])

    elements = []
    for a in ag:
        try:
            elements.append(a.element)
        except Exception:
            elements.append(a.name[0])

    box_vectors = None
    if u.dimensions is not None and np.any(u.dimensions[:3] > 0):
        from MDAnalysis.lib.mdamath import triclinic_vectors
        box_vectors = triclinic_vectors(u.dimensions)

    return Structure(
        positions=ag.positions.copy(),
        elements=elements,
        atom_names=[a.name for a in ag],
        residue_names=residue_names,
        residue_ids=[a.resid for a in ag],
        chain_ids=[a.segid for a in ag],
        is_water=is_water,
        is_protein=is_protein,
        is_ligand=is_ligand,
        box_vectors=box_vectors,
    )


def _load_with_rdkit(path: Path, **kwargs) -> Structure:
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next(iter(supplier))
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False)
    else:
        raise ValueError(f"Unsupported format for RDKit: {suffix}")

    if mol is None:
        raise ValueError(f"Failed to parse {path}")

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    positions = np.array([conf.GetAtomPosition(i) for i in range(n)])
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    atom_names = [f"{elements[i]}{i + 1}" for i in range(n)]

    ligand_resname = kwargs.get("ligand_resname", "LIG")
    return Structure(
        positions=positions,
        elements=elements,
        atom_names=atom_names,
        residue_names=[ligand_resname] * n,
        residue_ids=[1] * n,
        chain_ids=["L"] * n,
        is_water=np.zeros(n, dtype=bool),
        is_protein=np.zeros(n, dtype=bool),
        is_ligand=np.ones(n, dtype=bool),
    )
