# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""H2OSite input preparation: protein truncation, crystal water handling, and centering."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from H2OSite.core.config import WaterMapConfig
from H2OSite.core.structure import (
    Structure,
    load_structure,
    normalize_schrodinger_residue_names,
)
from H2OSite.io.structure_io import write_structure_file

logger = logging.getLogger(__name__)


class WaterMapInput:
    """Prepare protein and ligand structures for an H2OSite simulation.

    Load structures, optionally truncate the protein, handle crystal waters,
    center on the ligand, and reorder solute atoms by distance from the origin.
    """

    def __init__(self, config: WaterMapConfig):
        self.config = config
        self.protein: Structure | None = None
        self.ligand: Structure | None = None
        self.crystal_waters: np.ndarray | None = None  # positions of extracted waters
        self.translation_vector: np.ndarray = np.zeros(3)
        self.prepared_ligand_sdf_path: Path | None = None
        self.prepared_protein_cif_path: Path | None = None
        self.prepared_ligand_cif_path: Path | None = None
        self._ligand_rdkit_mol = None

    def prepare(self) -> tuple[Structure, Structure]:
        """Run full preparation pipeline. Returns (protein, ligand)."""
        self._load_inputs()

        lig_center = self.ligand.ligand_centroid()
        logger.info("Ligand centroid: %s", lig_center)

        # Step 1: Truncate protein
        if self.config.truncate_protein:
            self._truncate_protein(lig_center)

        # Step 2: Handle crystal waters
        self._handle_crystal_waters(lig_center)

        # Step 3: Center on ligand centroid
        self._center_on_ligand(lig_center)

        # Step 4: Reorder solute atoms
        self._reorder_solute()

        return self.protein, self.ligand

    def _load_inputs(self) -> None:
        """Load protein/ligand from either separate files or an embedded ligand."""
        ligand_resnames = self.config.ligand_resnames or None
        has_external_ligand = str(self.config.ligand_path) not in {"", "."}
        self.protein = load_structure(
            self.config.protein_path,
            ligand_resname=ligand_resnames,
        )

        if self.config.deduplicate_atoms:
            self._deduplicate_atoms()

        auto_normalize_for_caps = (
            self.config.terminal_treatment == "preserve_existing_caps"
        )
        if self.config.normalize_schrodinger_residues or auto_normalize_for_caps:
            if auto_normalize_for_caps and not self.config.normalize_schrodinger_residues:
                logger.info(
                    "Auto-normalizing input residue names because "
                    "terminal_treatment=preserve_existing_caps requires NME-compatible cap names"
                )
            stats = normalize_schrodinger_residue_names(
                self.protein,
                manual_histidine_states=self.config.manual_histidine_states,
                ligand_resnames=ligand_resnames,
            )
            if stats["histidines_renamed"] or stats["nma_renamed"]:
                logger.info(
                    "Normalized input residue names: %d histidines, %d NMA residues",
                    stats["histidines_renamed"],
                    stats["nma_renamed"],
                )

        if has_external_ligand:
            self.ligand = load_structure(self.config.ligand_path)
            self._ligand_rdkit_mol = self._load_rdkit_molecule(self.config.ligand_path)

            if ligand_resnames and np.any(self.protein.is_ligand):
                n_embedded = int(np.sum(self.protein.is_ligand))
                logger.info(
                    "Removing %d embedded ligand atoms from the receptor because a separate ligand file was provided",
                    n_embedded,
                )
                self.protein = self._subset_structure(
                    self.protein,
                    ~self.protein.is_ligand,
                )
            self._add_missing_ligand_hydrogens()
            return

        if not ligand_resnames:
            raise ValueError(
                "No separate ligand file was provided. Set ligand_resnames to extract an embedded ligand from the protein PDB."
            )

        ligand_mask = np.asarray(self.protein.is_ligand, dtype=bool)
        if not np.any(ligand_mask):
            raise ValueError(
                f"Could not find an embedded ligand with residue name(s) {self.config.ligand_resnames!r} in {self.config.protein_path}"
            )

        self.ligand = self._subset_structure(self.protein, ligand_mask)
        self.protein = self._subset_structure(self.protein, ~ligand_mask)
        self._ligand_rdkit_mol = self._extract_embedded_ligand_rdkit_mol()

        ligand_residue_keys = {
            (chain_id, residue_id, residue_name)
            for chain_id, residue_id, residue_name, is_ligand in zip(
                self.ligand.chain_ids,
                self.ligand.residue_ids,
                self.ligand.residue_names,
                self.ligand.is_ligand,
            )
            if is_ligand
        }
        logger.info(
            "Extracted %d ligand atoms across %d embedded ligand residue(s): %s",
            self.ligand.n_atoms,
            len(ligand_residue_keys),
            ", ".join(
                f"{resname}:{chain}:{resid}"
                for chain, resid, resname in sorted(ligand_residue_keys)
            ),
        )

        self._add_missing_ligand_hydrogens()

    def _truncate_protein(self, lig_center: np.ndarray) -> None:
        """Remove protein residues beyond truncate_distance from ligand centroid.

        Mimics system_builder_util.truncateProtein().
        """
        prot = self.protein
        dist = self.config.truncate_distance_A

        # Per-residue minimum distance to ligand centroid
        residue_ids = np.array(prot.residue_ids)
        unique_res = np.unique(residue_ids)

        keep_mask = np.zeros(prot.n_atoms, dtype=bool)
        for rid in unique_res:
            res_mask = residue_ids == rid
            res_positions = prot.positions[res_mask]
            min_dist = np.min(np.linalg.norm(res_positions - lig_center, axis=1))
            if min_dist <= dist:
                keep_mask |= res_mask

        # Also keep all non-protein atoms (waters, ions)
        keep_mask |= ~prot.is_protein

        n_removed = prot.n_atoms - np.sum(keep_mask)
        if n_removed > 0:
            logger.info("Truncated %d atoms beyond %.1f Å from ligand", n_removed, dist)
            self._apply_mask(prot, keep_mask)

    def _handle_crystal_waters(self, lig_center: np.ndarray) -> None:
        """Process crystal waters according to config.treat_solvent."""
        prot = self.protein
        mode = self.config.treat_solvent

        water_mask = prot.is_water
        n_waters = np.sum(water_mask)
        if n_waters == 0:
            return

        if mode == "delete":
            logger.info("Deleting %d crystal waters", n_waters)
            self._apply_mask(prot, ~water_mask)
        elif mode == "as_solvent":
            # Extract crystal water positions for later re-insertion as TIP4P
            self.crystal_waters = prot.positions[water_mask].copy()
            logger.info("Extracted %d crystal waters for TIP4P conversion", n_waters)
            self._apply_mask(prot, ~water_mask)
        elif mode == "as_solute":
            # Mark waters as restrained solute - keep them in place
            logger.info("Keeping %d crystal waters as restrained solute", n_waters)
        else:
            raise ValueError(f"Unknown treat_solvent mode: {mode}")

    def _center_on_ligand(self, lig_center: np.ndarray) -> None:
        """Translate all structures so ligand centroid is at origin.

        Store translation vector for back-transformation after analysis.
        """
        self.translation_vector = -lig_center
        self.protein.translate(self.translation_vector)
        self.ligand.translate(self.translation_vector)
        self._translate_rdkit_ligand(self.translation_vector)
        if self.crystal_waters is not None:
            self.crystal_waters += self.translation_vector
        logger.info("Centered system; translation vector: %s", self.translation_vector)

    def _reorder_solute(self) -> None:
        """Move the residue closest to origin to the start of the structure.

        Prevents periodic image artifacts in OpenMM.  Moves the
        entire residue (not a single atom) so that atoms within each
        residue remain contiguous - required for correct PDB output and
        OpenMM template matching.
        """
        prot = self.protein
        solute_mask = prot.is_protein & prot.heavy_atom_mask()
        if not np.any(solute_mask):
            return

        solute_indices = np.where(solute_mask)[0]
        dists = np.linalg.norm(prot.positions[solute_indices], axis=1)
        closest_idx = solute_indices[np.argmin(dists)]

        # Find all atoms of the closest residue
        target_resid = prot.residue_ids[closest_idx]
        target_chain = prot.chain_ids[closest_idx]
        res_mask = np.array([
            rid == target_resid and cid == target_chain
            for rid, cid in zip(prot.residue_ids, prot.chain_ids)
        ])
        res_indices = np.where(res_mask)[0]

        if res_indices[0] == 0:
            return  # Already at the start

        # Move entire residue to the front, keep everything else in order
        non_res_indices = np.where(~res_mask)[0]
        order = np.concatenate([res_indices, non_res_indices])
        self._reorder(prot, order)

    @staticmethod
    def _apply_mask(struct: Structure, mask: np.ndarray) -> None:
        """Filter structure in-place by boolean mask."""
        struct.positions = struct.positions[mask]
        struct.elements = [e for e, m in zip(struct.elements, mask) if m]
        struct.atom_names = [n for n, m in zip(struct.atom_names, mask) if m]
        struct.residue_names = [r for r, m in zip(struct.residue_names, mask) if m]
        struct.residue_ids = [r for r, m in zip(struct.residue_ids, mask) if m]
        struct.chain_ids = [c for c, m in zip(struct.chain_ids, mask) if m]
        struct.is_water = struct.is_water[mask]
        struct.is_protein = struct.is_protein[mask]
        struct.is_ligand = struct.is_ligand[mask]

    @staticmethod
    def _reorder(struct: Structure, order: np.ndarray) -> None:
        """Reorder atoms in structure by index array."""
        struct.positions = struct.positions[order]
        struct.elements = [struct.elements[i] for i in order]
        struct.atom_names = [struct.atom_names[i] for i in order]
        struct.residue_names = [struct.residue_names[i] for i in order]
        struct.residue_ids = [struct.residue_ids[i] for i in order]
        struct.chain_ids = [struct.chain_ids[i] for i in order]
        struct.is_water = struct.is_water[order]
        struct.is_protein = struct.is_protein[order]
        struct.is_ligand = struct.is_ligand[order]

    def _deduplicate_atoms(self) -> None:
        """Remove duplicate atoms and fix residue-number collisions.

        Handles two common PDB defects:

        1. Residue-number collisions - different residues (e.g. a
           VAL and its NMA cap) sharing the same ``(chain, resid)``.
           OpenMM merges them into one residue and corrupts the bond
           topology.  Fixed by assigning a unique ``resid`` to the
           minority residue.

        2. Duplicate atoms - alternate-conformation atoms that share
           the same ``(chain, resid, atom_name)``.  OpenMM overwrites
           positions but leaves stale bonds.  Fixed by keeping only the
           first occurrence.
        """
        from collections import Counter

        # --- Part 1: fix residue-number collisions ---
        residue_ids = list(self.protein.residue_ids)
        residue_names = list(self.protein.residue_names)
        chain_ids = self.protein.chain_ids

        groups: dict[tuple[str, int], list[int]] = {}
        for i, (ch, rid) in enumerate(zip(chain_ids, residue_ids)):
            groups.setdefault((ch, int(rid)), []).append(i)

        # Collect all resids per chain so we can pick unused neighbours
        chain_resids: dict[str, set[int]] = {}
        for ch, rid in zip(chain_ids, residue_ids):
            chain_resids.setdefault(ch, set()).add(int(rid))

        n_collisions_fixed = 0

        for (ch, rid), indices in groups.items():
            names_in_group = [residue_names[i] for i in indices]
            unique_names = set(names_in_group)
            if len(unique_names) <= 1:
                continue

            name_counts = Counter(names_in_group)
            majority_name = name_counts.most_common(1)[0][0]

            for minority_name in sorted(unique_names - {majority_name}):
                # Use rid+1 to keep the cap adjacent (avoids chain-split
                # in _write_pdb which triggers on resid gaps > 1).  Fall
                # back to rid+2, +3, ... if the slot is already occupied.
                new_rid = rid + 1
                while new_rid in chain_resids[ch]:
                    new_rid += 1
                chain_resids[ch].add(new_rid)

                for i in indices:
                    if residue_names[i] == minority_name:
                        residue_ids[i] = new_rid
                n_collisions_fixed += 1
                logger.info(
                    "Fixed residue-number collision: %s %s:%d -> resid %d "
                    "(was sharing resid with %s)",
                    minority_name, ch, rid, new_rid, majority_name,
                )

        if n_collisions_fixed:
            self.protein.residue_ids = residue_ids

        # --- Part 2: remove duplicate atoms ---
        seen: set[tuple[str, int, str]] = set()
        keep = np.ones(self.protein.n_atoms, dtype=bool)
        for i in range(self.protein.n_atoms):
            key = (
                self.protein.chain_ids[i],
                self.protein.residue_ids[i],
                self.protein.atom_names[i],
            )
            if key in seen:
                keep[i] = False
            else:
                seen.add(key)

        n_removed = int(np.sum(~keep))
        if n_removed:
            logger.info(
                "Deduplicated protein: removed %d duplicate atom(s) "
                "(alternate conformations)",
                n_removed,
            )
            self._apply_mask(self.protein, keep)

        if not n_collisions_fixed and not n_removed:
            logger.info("No duplicate atoms or residue-number collisions found")

    def write_prepared(self, output_dir: Path) -> tuple[Path, Path]:
        """Write prepared protein and ligand to PDB files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        prot_path = output_dir / f"{self.config.jobname}_protein.pdb"
        lig_path = output_dir / f"{self.config.jobname}_ligand.pdb"
        self.prepared_protein_cif_path = output_dir / f"{self.config.jobname}_protein.cif"
        self.prepared_ligand_cif_path = output_dir / f"{self.config.jobname}_ligand.cif"
        self.prepared_ligand_sdf_path = None

        _write_pdb(self.protein, prot_path)
        write_structure_file(self.protein, self.prepared_protein_cif_path)
        if self._ligand_rdkit_mol is not None:
            self.prepared_ligand_sdf_path = (
                output_dir / f"{self.config.jobname}_ligand.sdf"
            )
            self._write_rdkit_ligand_files(
                lig_path,
                self.prepared_ligand_sdf_path,
            )
            write_structure_file(self.ligand, self.prepared_ligand_cif_path)
        else:
            _write_pdb(self.ligand, lig_path)
            write_structure_file(self.ligand, self.prepared_ligand_cif_path)

        # Store translation vector as metadata
        meta_path = output_dir / f"{self.config.jobname}_meta.npz"
        np.savez(meta_path, translation_vector=self.translation_vector)

        return prot_path, lig_path

    @staticmethod
    def _subset_structure(struct: Structure, mask: np.ndarray) -> Structure:
        mask = np.asarray(mask, dtype=bool)
        return Structure(
            positions=struct.positions[mask].copy(),
            elements=[e for e, keep in zip(struct.elements, mask) if keep],
            atom_names=[n for n, keep in zip(struct.atom_names, mask) if keep],
            residue_names=[r for r, keep in zip(struct.residue_names, mask) if keep],
            residue_ids=[r for r, keep in zip(struct.residue_ids, mask) if keep],
            chain_ids=[c for c, keep in zip(struct.chain_ids, mask) if keep],
            is_water=np.asarray(struct.is_water[mask], dtype=bool),
            is_protein=np.asarray(struct.is_protein[mask], dtype=bool),
            is_ligand=np.asarray(struct.is_ligand[mask], dtype=bool),
            box_vectors=None if struct.box_vectors is None else np.array(struct.box_vectors, copy=True),
            properties=dict(struct.properties),
        )

    def _translate_rdkit_ligand(self, vector: np.ndarray) -> None:
        if self._ligand_rdkit_mol is None:
            return
        conf = self._ligand_rdkit_mol.GetConformer()
        dx, dy, dz = [float(v) for v in vector]
        for idx in range(self._ligand_rdkit_mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(idx)
            conf.SetAtomPosition(idx, (pos.x + dx, pos.y + dy, pos.z + dz))

    def _load_rdkit_molecule(self, path: str | Path):
        from rdkit import Chem

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), removeHs=False)
            mol = next((m for m in supplier if m is not None), None)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), removeHs=False)
        elif suffix == ".mol":
            mol = Chem.MolFromMolFile(str(path), removeHs=False)
        elif suffix in {".pdb", ".pdbqt"}:
            mol = Chem.MolFromPDBFile(str(path), removeHs=False)
        elif suffix in {".cif", ".mmcif", ".pdbx"}:
            mol = self._load_rdkit_molecule_from_cif(path)
        else:
            mol = None

        if mol is None:
            logger.warning("Could not load ligand RDKit molecule from %s", path)
        return mol

    def _load_rdkit_molecule_from_cif(self, path: Path):
        from rdkit import Chem

        with tempfile.TemporaryDirectory(prefix="owm_ligand_cif_") as tmpdir:
            temp_pdb = Path(tmpdir) / "ligand_from_cif.pdb"
            struct = load_structure(path)
            write_structure_file(struct, temp_pdb)
            mol = Chem.MolFromPDBFile(str(temp_pdb), removeHs=False)
        return mol

    def _add_missing_ligand_hydrogens(self) -> None:
        """Add explicit hydrogens to the ligand if the RDKit mol has implicit H.

        Prepared PDBs may omit non-polar hydrogens on ligands.
        OpenFF's ``Molecule.from_rdkit()`` makes all implicit H explicit,
        creating an atom-count mismatch between the parameterisation
        template (from SDF) and the topology (from PDB).

        This adds the missing H with 3D coordinates so that both the
        ligand Structure and the RDKit mol/SDF are consistent.
        """
        if self._ligand_rdkit_mol is None:
            return

        from rdkit import Chem

        implicit_h = sum(
            a.GetNumImplicitHs() for a in self._ligand_rdkit_mol.GetAtoms()
        )
        if implicit_h == 0:
            return

        n_before = self._ligand_rdkit_mol.GetNumAtoms()
        mol_h = Chem.AddHs(self._ligand_rdkit_mol, addCoords=True)
        n_after = mol_h.GetNumAtoms()
        n_added = n_after - n_before

        if n_added == 0:
            return

        logger.info(
            "Added %d missing hydrogen(s) to ligand (%d -> %d atoms)",
            n_added, n_before, n_after,
        )

        # Collect existing atom names so we can generate non-colliding names
        # for added H atoms. PDB atom names are 4 chars; RDKit strips whitespace
        # when comparing, so we normalise to stripped names for the check.
        existing_names = {
            mol_h.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
            for i in range(n_before)
            if mol_h.GetAtomWithIdx(i).GetPDBResidueInfo() is not None
        }

        # Copy PDB residue info from parent atoms to new H so that
        # RDKit's MolToPDBFile writes them into the same residue
        # (otherwise they get "UNL 1" and OpenMM splits them out).
        h_counter = 1
        for idx in range(n_before, n_after):
            atom = mol_h.GetAtomWithIdx(idx)
            # Find the heavy-atom neighbour (the parent)
            parent = None
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() < n_before:
                    parent = nbr
                    break
            if parent is not None:
                ref_info = parent.GetPDBResidueInfo()
                if ref_info is not None:
                    # Pick a name that doesn't clash with any existing atom name.
                    while True:
                        candidate = f"Hq{h_counter}"
                        h_counter += 1
                        if candidate not in existing_names:
                            existing_names.add(candidate)
                            break
                    new_info = Chem.AtomPDBResidueInfo()
                    new_info.SetResidueName(ref_info.GetResidueName())
                    new_info.SetResidueNumber(ref_info.GetResidueNumber())
                    new_info.SetChainId(ref_info.GetChainId())
                    new_info.SetName(f"{candidate:<4s}")
                    new_info.SetIsHeteroAtom(ref_info.GetIsHeteroAtom())
                    atom.SetMonomerInfo(new_info)

        self._ligand_rdkit_mol = mol_h

        # Extend the ligand Structure with the new H atoms
        conf = mol_h.GetConformer()
        new_positions = np.array(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z]
             for i in range(n_before, n_after)],
            dtype=np.float32,
        )

        lig = self.ligand
        ref_resname = lig.residue_names[0]
        ref_resid = lig.residue_ids[0]
        ref_chain = lig.chain_ids[0]

        lig.positions = np.concatenate([lig.positions, new_positions], axis=0)
        lig.elements = list(lig.elements) + ["H"] * n_added
        lig.atom_names = list(lig.atom_names) + [
            f"HX{i}" for i in range(1, n_added + 1)
        ]
        lig.residue_names = list(lig.residue_names) + [ref_resname] * n_added
        lig.residue_ids = list(lig.residue_ids) + [ref_resid] * n_added
        lig.chain_ids = list(lig.chain_ids) + [ref_chain] * n_added
        lig.is_water = np.concatenate(
            [lig.is_water, np.zeros(n_added, dtype=bool)]
        )
        lig.is_protein = np.concatenate(
            [lig.is_protein, np.zeros(n_added, dtype=bool)]
        )
        lig.is_ligand = np.concatenate(
            [lig.is_ligand, np.ones(n_added, dtype=bool)]
        )

    def _extract_embedded_ligand_rdkit_mol(self):
        from rdkit import Chem

        suffix = Path(self.config.protein_path).suffix.lower()
        if suffix in {".cif", ".mmcif", ".pdbx"}:
            with tempfile.TemporaryDirectory(prefix="owm_embedded_ligand_") as tmpdir:
                temp_pdb = Path(tmpdir) / "embedded_ligand.pdb"
                write_structure_file(self.ligand, temp_pdb)
                mol = Chem.MolFromPDBFile(str(temp_pdb), removeHs=False)
                if mol is None:
                    raise ValueError(
                        "Failed to build an RDKit molecule for the embedded ligand from mmCIF input. "
                        "Please provide a separate ligand SDF if parameterization fails."
                    )
                return mol

        ligand_resnames = {str(name) for name in self.config.ligand_resnames}
        atom_lines: list[str] = []
        selected_serials: set[int] = set()

        with open(self.config.protein_path) as handle:
            all_lines = handle.readlines()

        for line in all_lines:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[17:20].strip() not in ligand_resnames:
                continue
            atom_lines.append(line.rstrip("\n"))
            selected_serials.add(int(line[6:11]))

        conect_lines: list[str] = []
        for line in all_lines:
            if not line.startswith("CONECT"):
                continue
            try:
                serials = [int(tok) for tok in line.split()[1:]]
            except ValueError:
                continue
            if serials and serials[0] in selected_serials:
                filtered = [serial for serial in serials if serial in selected_serials]
                if len(filtered) > 1:
                    conect_lines.append(
                        "CONECT" + "".join(f"{serial:5d}" for serial in filtered)
                    )

        pdb_block = "\n".join(atom_lines + conect_lines + ["END", ""])
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
        if mol is None:
            raise ValueError(
                "Failed to build an RDKit molecule for the embedded ligand. "
                "Please provide a separate ligand SDF or check the docked ligand connectivity."
            )
        return mol

    def _write_rdkit_ligand_files(self, pdb_path: Path, sdf_path: Path) -> None:
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        mol = Chem.Mol(self._ligand_rdkit_mol)
        conf = mol.GetConformer()
        if mol.GetNumAtoms() != self.ligand.n_atoms:
            raise ValueError(
                "RDKit ligand atom count does not match the prepared ligand coordinates"
            )

        for idx, (x, y, z) in enumerate(np.asarray(self.ligand.positions, dtype=float)):
            conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))

        Chem.MolToPDBFile(mol, str(pdb_path))
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()


def _write_pdb(struct: Structure, path: Path) -> None:
    """Write a Structure to PDB format.

    For truncated proteins, contiguous runs of protein residues within each
    original chain are assigned unique chain IDs (A, B, C, ...) so that
    OpenMM treats each fragment as a separate chain.  This ensures:
      - No phantom peptide bonds across truncation gaps
      - ``_cap_termini`` in SystemBuilder adds ACE/NME to each fragment
      - ``addHydrogens`` adds proper terminal atoms per fragment

    Atoms are sorted by (original_chain, residue_id) to restore backbone
    order after any reordering done during preparation.
    """
    _PROTEIN_RES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL', 'HIE', 'HID', 'HIP', 'CYX', 'ACE', 'NME', 'NMA',
        'ASH', 'GLH',
    }

    # Sort atoms by (chain_id, residue_id, atom_index)
    order = sorted(
        range(struct.n_atoms),
        key=lambda i: (struct.chain_ids[i], struct.residue_ids[i], i),
    )

    # First pass: assign each atom a new chain ID.
    # Each contiguous run of protein residues (no gap in resid > 1) gets
    # its own chain letter so that OpenMM treats each fragment as a
    # separate chain with proper termini.
    _CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    new_chain_ids: dict[int, str] = {}
    chain_idx = -1
    prev_orig_chain = None
    prev_resid = None
    prev_is_protein = False

    for idx in order:
        orig_chain = struct.chain_ids[idx]
        resid = struct.residue_ids[idx]
        resname = struct.residue_names[idx]
        is_protein = resname in _PROTEIN_RES

        need_new_chain = (
            prev_orig_chain is None
            or orig_chain != prev_orig_chain
            or (is_protein and prev_is_protein and abs(resid - prev_resid) > 1)
        )

        if need_new_chain:
            chain_idx = min(chain_idx + 1, len(_CHAIN_LETTERS) - 1)

        new_chain_ids[idx] = _CHAIN_LETTERS[chain_idx]
        prev_orig_chain = orig_chain
        prev_resid = resid
        prev_is_protein = is_protein

    # Second pass: write PDB with new chain IDs and TER records
    with open(path, "w") as f:
        serial = 0
        prev_chain_out = None

        for idx in order:
            chain_out = new_chain_ids[idx]

            if prev_chain_out is not None and chain_out != prev_chain_out:
                f.write(f"TER   {serial + 1:5d}\n")

            serial += 1
            x, y, z = struct.positions[idx]
            f.write(
                f"ATOM  {serial:5d} {struct.atom_names[idx]:4s} "
                f"{struct.residue_names[idx]:3s} {chain_out:1s}"
                f"{struct.residue_ids[idx]:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {struct.elements[idx]:>2s}\n"
            )
            prev_chain_out = chain_out

        f.write("TER\n")
        f.write("END\n")
