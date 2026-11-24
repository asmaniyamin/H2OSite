# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""System building: solvation, box construction, and force-field assignment.

Uses OpenMM Modeller and ForceField with AMBER14 and TIP3P defaults.
System neutralization is disabled by default.

Terminal treatment options:
  caps: add ACE/NME neutral caps for truncated protein chains.
  add_hydrogens: fill missing terminal hydrogens and minimize added atoms.
  preserve_existing_caps: keep supplied caps, normalizing NMA to NME.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from H2OSite.io.structure_io import load_openmm_structure_file

logger = logging.getLogger(__name__)

# Standard amino-acid residue names recognised as protein
_PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
    'TYR', 'VAL', 'HIE', 'HID', 'HIP', 'CYX', 'ASH', 'GLH',
}


class SystemBuilder:
    """Build a solvated simulation system using OpenMM.

    Uses an orthorhombic box with a solute buffer, AMBER14, and TIP3P water
    by default. TIP4P-Ew is available as an alternative water model.
    """

    def __init__(
        self,
        protein_pdb: str | Path,
        ligand_pdb: str | Path | None = None,
        ligand_sdf: str | Path | None = None,
        water_model: str = "tip3p",
        forcefield: str = "amber14-all.xml",
        ligand_forcefield: str = "openff-2.1.0.offxml",
        buffer_A: float = 10.0,

        neutralize: bool = False,
        terminal_treatment: str = "caps",
        ligand_parameter_cache: str | Path | None = None,
    ):
        self.protein_pdb = Path(protein_pdb)
        self.ligand_pdb = Path(ligand_pdb) if ligand_pdb else None
        self.ligand_sdf = Path(ligand_sdf) if ligand_sdf else None
        self.water_model = water_model
        self.forcefield_name = forcefield
        self.ligand_forcefield = ligand_forcefield
        self.buffer_A = buffer_A
        self.neutralize = neutralize
        self.terminal_treatment = terminal_treatment
        self.ligand_parameter_cache = (
            Path(ligand_parameter_cache)
            if ligand_parameter_cache
            else None
        )

    def build(self) -> tuple:
        """Build the solvated system.

        Returns:
            (topology, positions, system, forcefield) - OpenMM objects ready for simulation.
        """
        import openmm.app as app
        import openmm.unit as unit

        logger.info("Building simulation system...")

        # Load force field
        water_ff = self._water_ff_file()
        ff = app.ForceField(self.forcefield_name, water_ff)
        if self.ligand_pdb and self.ligand_pdb.exists():
            self._register_ligand_template_generator(ff)

        # Load protein
        protein_topology, protein_positions = load_openmm_structure_file(
            self.protein_pdb
        )
        modeller = app.Modeller(protein_topology, protein_positions)

        # Terminal treatment
        modeller = self._apply_terminal_treatment(modeller, ff)

        # Add ligand if provided
        if self.ligand_pdb and self.ligand_pdb.exists():
            ligand_topology, ligand_positions = load_openmm_structure_file(
                self.ligand_pdb
            )
            modeller.add(ligand_topology, ligand_positions)
            logger.info("Added ligand from %s", self.ligand_pdb)

        # Add solvent - orthorhombic box with padding
        buffer_nm = self.buffer_A * 0.1 * unit.nanometers
        modeller.addSolvent(
            ff,
            model=self.water_model,
            padding=buffer_nm,
            neutralize=self.neutralize,
        )
        logger.info(
            "Solvated system: %d atoms, %d residues",
            modeller.topology.getNumAtoms(),
            modeller.topology.getNumResidues(),
        )

        # Create system
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
        )

        return modeller.topology, modeller.positions, system, ff

    @staticmethod
    def _terminal_cap_summary(topology) -> dict[str, int]:
        """Count terminal-cap residue names present in a topology."""
        counts = {"ACE": 0, "NME": 0, "NMA": 0}
        for residue in topology.residues():
            if residue.name in counts:
                counts[residue.name] += 1
        return counts

    def _apply_terminal_treatment(self, modeller, ff):
        """Apply the configured terminal treatment to the modeller."""
        cap_counts = self._terminal_cap_summary(modeller.topology)

        if self.terminal_treatment == "caps":
            return self._cap_termini(modeller, ff)

        if self.terminal_treatment == "add_hydrogens":
            n_before = modeller.topology.getNumAtoms()
            modeller.addHydrogens(ff)
            n_after = modeller.topology.getNumAtoms()
            if n_after > n_before:
                logger.info(
                    "Added %d hydrogen atoms, minimizing their positions",
                    n_after - n_before,
                )
                self._minimize_added_atoms(modeller, ff, n_before)
            return modeller

        if self.terminal_treatment == "preserve_existing_caps":
            logger.info(
                "Preserving input terminal treatment as-is "
                "(ACE=%d, NME=%d, NMA=%d); no cap addition or terminal addHydrogens will be performed",
                cap_counts["ACE"],
                cap_counts["NME"],
                cap_counts["NMA"],
            )
            if cap_counts["NMA"]:
                logger.warning(
                    "Detected %d NMA residues after preparation. OpenMM expects NME caps; "
                    "enable residue normalization upstream if this build fails.",
                    cap_counts["NMA"],
                )
            return modeller

        raise ValueError(
            f"Unknown terminal_treatment {self.terminal_treatment!r}. "
            "Expected one of: 'caps', 'add_hydrogens', 'preserve_existing_caps'."
        )

    def _register_ligand_template_generator(self, forcefield) -> None:
        """Register an OpenFF template generator for a retained ligand."""
        from rdkit import Chem
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        ligand_source = self.ligand_sdf or self.ligand_pdb
        if ligand_source is None or not Path(ligand_source).exists():
            raise ValueError(
                "retain_ligand=True requires a ligand file that can be parameterized"
            )

        suffix = Path(ligand_source).suffix.lower()
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(ligand_source), removeHs=False)
            rdkit_mol = next((mol for mol in supplier if mol is not None), None)
        elif suffix == ".mol2":
            rdkit_mol = Chem.MolFromMol2File(str(ligand_source), removeHs=False)
        elif suffix == ".mol":
            rdkit_mol = Chem.MolFromMolFile(str(ligand_source), removeHs=False)
        else:
            rdkit_mol = Chem.MolFromPDBFile(str(ligand_source), removeHs=False)

        if rdkit_mol is None:
            raise ValueError(
                f"Failed to load ligand for OpenFF parameterization from {ligand_source}"
            )

        off_molecule = Molecule.from_rdkit(
            rdkit_mol,
            allow_undefined_stereo=True,
        )

        cache_path = None
        if self.ligand_parameter_cache is not None:
            self.ligand_parameter_cache.parent.mkdir(parents=True, exist_ok=True)
            cache_path = str(self.ligand_parameter_cache)

        template_generator = SMIRNOFFTemplateGenerator(
            molecules=[off_molecule],
            cache=cache_path,
            forcefield=self.ligand_forcefield,
        )
        forcefield.registerTemplateGenerator(template_generator.generator)
        if cache_path is not None:
            logger.info(
                "Registered OpenFF ligand template from %s using %s "
                "(cache=%s)",
                ligand_source,
                self.ligand_forcefield,
                cache_path,
            )
        else:
            logger.info(
                "Registered OpenFF ligand template from %s using %s",
                ligand_source,
                self.ligand_forcefield,
            )

    # ------------------------------------------------------------------
    # Terminal capping
    # ------------------------------------------------------------------

    def _cap_termini(self, modeller, ff):
        """Add ACE/NME caps to protein chain termini.

        Rebuilds the topology with ACE before the first protein residue
        and NME after the last protein residue of each chain.  All cap
        atoms (including CH3 hydrogens) are placed explicitly so that
        ``addHydrogens`` is not required afterward.

        Cap atoms are placed with approximate backbone geometry;
        energy minimisation during equilibration refines positions.
        """
        import openmm.app as app
        import openmm.unit as unit

        old_top = modeller.topology
        old_pos = np.array(
            [v.value_in_unit(unit.nanometers) for v in modeller.positions]
        )

        new_top = app.Topology()
        box = old_top.getPeriodicBoxVectors()
        if box is not None:
            new_top.setPeriodicBoxVectors(box)

        new_pos = []
        atom_map = {}          # old_atom.index -> new_atom
        cap_bonds = []         # deferred peptide bonds (cap <-> terminal res)

        for old_chain in old_top.chains():
            new_chain = new_top.addChain(old_chain.id)
            residues = list(old_chain.residues())

            protein_residues = [r for r in residues if r.name in _PROTEIN_RESIDUES]

            # Non-protein chain - copy as-is
            if not protein_residues:
                for res in residues:
                    self._copy_residue(res, new_top, new_chain, old_pos,
                                       new_pos, atom_map)
                continue

            n_term = protein_residues[0]
            c_term = protein_residues[-1]

            already_ace = any(r.name == 'ACE' for r in residues)
            already_nme = any(r.name in ('NME', 'NMA') for r in residues)

            ace_c_atom = None
            nme_n_atom = None
            first_n_atom = None
            last_c_atom = None

            for res in residues:
                # Insert ACE before the first protein residue
                if res is n_term and not already_ace:
                    ace_c_atom = self._add_ace(new_top, new_chain,
                                               n_term, old_pos, new_pos)

                # Strip terminal-specific atoms when capping:
                #  N-term: H2, H3 are NH3+ extras (keep backbone H only)
                #  C-term: OXT/O2 are carboxylate-specific
                skip_atoms = set()
                if res is n_term and not already_ace:
                    skip_atoms |= {'H2', 'H3', 'HT2', 'HT3'}
                if res is c_term and not already_nme:
                    skip_atoms |= {'OXT', 'O2'}
                n_atom, c_atom = self._copy_residue(
                    res, new_top, new_chain, old_pos, new_pos,
                    atom_map, skip_atoms=skip_atoms,
                )
                if res is n_term and n_atom is not None:
                    first_n_atom = n_atom
                if res is c_term and c_atom is not None:
                    last_c_atom = c_atom

                # Insert NME after the last protein residue
                if res is c_term and not already_nme:
                    nme_n_atom = self._add_nme(new_top, new_chain,
                                                c_term, old_pos, new_pos)

            if ace_c_atom and first_n_atom:
                cap_bonds.append((ace_c_atom, first_n_atom))
            if last_c_atom and nme_n_atom:
                cap_bonds.append((last_c_atom, nme_n_atom))

        # Copy existing bonds
        for bond in old_top.bonds():
            a1, a2 = bond.atom1, bond.atom2
            if a1.index in atom_map and a2.index in atom_map:
                new_top.addBond(atom_map[a1.index], atom_map[a2.index])

        # Add cap <-> residue peptide bonds
        for a1, a2 in cap_bonds:
            new_top.addBond(a1, a2)

        new_positions = np.array(new_pos) * unit.nanometers
        new_modeller = app.Modeller(new_top, new_positions)

        n_ace = 0 if already_ace else sum(
            1 for c in new_top.chains()
            for r in c.residues() if r.name == 'ACE'
        )
        n_nme = 0 if already_nme else sum(
            1 for c in new_top.chains()
            for r in c.residues() if r.name in ('NME', 'NMA')
        )
        logger.info("Terminal caps added: %d ACE, %d NME", n_ace, n_nme)

        return new_modeller

    @staticmethod
    def _copy_residue(res, new_top, new_chain, old_pos, new_pos,
                      atom_map, skip_atoms=None):
        """Copy a residue into the new topology, returning (N_atom, C_atom)."""
        import openmm.app as app
        skip = skip_atoms or set()
        new_res = new_top.addResidue(res.name, new_chain,
                                      res.id, res.insertionCode)
        n_atom = c_atom = None
        for atom in res.atoms():
            if atom.name in skip:
                continue
            new_atom = new_top.addAtom(atom.name, atom.element, new_res)
            atom_map[atom.index] = new_atom
            new_pos.append(old_pos[atom.index])
            if atom.name == 'N':
                n_atom = new_atom
            elif atom.name == 'C':
                c_atom = new_atom
        return n_atom, c_atom

    def _add_ace(self, topology, chain, n_term_res, old_pos, new_pos):
        """Add ACE cap residue (HH31, CH3, HH32, HH33, C, O) before N-terminus.

        Returns the ACE carbon atom (C) for bonding to the first residue's N.
        """
        import openmm.app as app

        atoms_idx = {a.name: a.index for a in n_term_res.atoms()}
        if 'N' not in atoms_idx or 'CA' not in atoms_idx:
            return None

        n_p = old_pos[atoms_idx['N']]
        ca_p = old_pos[atoms_idx['CA']]

        # ACE.C along N<-CA direction, 1.34 Å from N
        d = n_p - ca_p
        d /= np.linalg.norm(d)
        ace_c = n_p + d * 0.134  # nm

        # ACE.O perpendicular, C=O 1.23 Å
        perp = np.cross(d, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 0.01:
            perp = np.cross(d, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        ace_o = ace_c + perp * 0.123

        # ACE.CH3 extends further, C-C 1.52 Å
        ace_ch3 = ace_c + d * 0.152

        # 3 H on CH3 in tetrahedral arrangement
        hh = _tetrahedral_h(ace_ch3, d, bond_length=0.109)

        # Build residue - atom order matches AMBER14 ACE template
        ace_res = topology.addResidue('ACE', chain)
        hh31 = topology.addAtom('HH31', app.element.hydrogen, ace_res)
        new_pos.append(hh[0])
        ch3_atom = topology.addAtom('CH3', app.element.carbon, ace_res)
        new_pos.append(ace_ch3)
        hh32 = topology.addAtom('HH32', app.element.hydrogen, ace_res)
        new_pos.append(hh[1])
        hh33 = topology.addAtom('HH33', app.element.hydrogen, ace_res)
        new_pos.append(hh[2])
        c_atom = topology.addAtom('C', app.element.carbon, ace_res)
        new_pos.append(ace_c)
        o_atom = topology.addAtom('O', app.element.oxygen, ace_res)
        new_pos.append(ace_o)

        # Internal bonds (all required by AMBER14 ACE template)
        topology.addBond(hh31, ch3_atom)
        topology.addBond(ch3_atom, hh32)
        topology.addBond(ch3_atom, hh33)
        topology.addBond(ch3_atom, c_atom)
        topology.addBond(c_atom, o_atom)

        return c_atom

    def _add_nme(self, topology, chain, c_term_res, old_pos, new_pos):
        """Add NME cap residue (N, H, CH3, HH31, HH32, HH33) after C-terminus.

        Returns the NME nitrogen atom (N) for bonding to the last residue's C.
        """
        import openmm.app as app

        atoms_idx = {a.name: a.index for a in c_term_res.atoms()}
        if 'C' not in atoms_idx or 'CA' not in atoms_idx:
            return None

        c_p = old_pos[atoms_idx['C']]
        ca_p = old_pos[atoms_idx['CA']]

        # NME.N along C<-CA direction, 1.34 Å from C
        d = c_p - ca_p
        d /= np.linalg.norm(d)
        nme_n = c_p + d * 0.134

        # NME.H perpendicular, N-H 1.01 Å
        perp = np.cross(d, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 0.01:
            perp = np.cross(d, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        nme_h = nme_n + perp * 0.101

        # NME.CH3 extends further, N-C 1.47 Å
        nme_ch3 = nme_n + d * 0.147

        # 3 H on CH3
        hh = _tetrahedral_h(nme_ch3, d, bond_length=0.109)

        # Build residue - atom order matches AMBER14 NME template
        nme_res = topology.addResidue('NME', chain)
        n_atom = topology.addAtom('N', app.element.nitrogen, nme_res)
        new_pos.append(nme_n)
        h_atom = topology.addAtom('H', app.element.hydrogen, nme_res)
        new_pos.append(nme_h)
        ch3_atom = topology.addAtom('CH3', app.element.carbon, nme_res)
        new_pos.append(nme_ch3)
        hh31 = topology.addAtom('HH31', app.element.hydrogen, nme_res)
        new_pos.append(hh[0])
        hh32 = topology.addAtom('HH32', app.element.hydrogen, nme_res)
        new_pos.append(hh[1])
        hh33 = topology.addAtom('HH33', app.element.hydrogen, nme_res)
        new_pos.append(hh[2])

        # Internal bonds (all required by AMBER14 NME template)
        topology.addBond(n_atom, h_atom)
        topology.addBond(n_atom, ch3_atom)
        topology.addBond(ch3_atom, hh31)
        topology.addBond(ch3_atom, hh32)
        topology.addBond(ch3_atom, hh33)

        return n_atom

    # ------------------------------------------------------------------
    # Constrained H minimization (for add_hydrogens mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _minimize_added_atoms(modeller, ff, n_original):
        """Minimize only atoms added after index n_original.

        All original atoms are held fixed by strong harmonic restraints
        (5000 kJ/mol/nm²).  Only the newly added atoms (typically H)
        are free to move.  Uses gas-phase (NoCutoff) for speed.
        """
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)

        # Restrain all original atoms
        restraint = openmm.CustomExternalForce(
            "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k", 5000.0 * unit.kilojoules_per_mole / unit.nanometers**2
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        positions = modeller.positions
        for i in range(n_original):
            pos = positions[i].value_in_unit(unit.nanometers)
            restraint.addParticle(i, list(pos))
        system.addForce(restraint)

        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName("CPU")
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)

        state = context.getState(getPositions=True)
        modeller.positions = state.getPositions()
        logger.info("Constrained minimization of %d added atoms complete",
                     modeller.topology.getNumAtoms() - n_original)

    # ------------------------------------------------------------------
    # OpenFF ligand path
    # ------------------------------------------------------------------

    def build_with_ligand_params(
        self,
        ligand_sdf: str | Path | None = None,
    ) -> tuple:
        """Build system with OpenFF-parameterized small molecule.

        Uses the OpenFF Sage force field for the ligand and AMBER14 for protein.
        This is the recommended path for drug-like molecules.
        """
        import openmm.app as app
        import openmm.unit as unit

        logger.info("Building system with OpenFF ligand parameters...")

        protein_topology, protein_positions = load_openmm_structure_file(
            self.protein_pdb
        )
        protein_structure = type("OpenMMStructure", (), {
            "topology": protein_topology,
            "positions": protein_positions,
        })()

        if ligand_sdf:
            return self._build_openff_interchange(protein_structure, Path(ligand_sdf))

        # No ligand - standard build
        return self.build()

    def _build_openff_interchange(self, protein_pdb, ligand_sdf: Path) -> tuple:
        """Build combined protein + ligand system using OpenFF Interchange."""
        import openmm.app as app
        import openmm.unit as unit
        from openff.toolkit import Molecule, ForceField as OFFForceField

        # Load ligand with OpenFF
        lig_mol = Molecule.from_file(str(ligand_sdf))
        lig_mol.generate_conformers(n_conformers=1)

        # Parameterize ligand with Sage
        off_ff = OFFForceField("openff-2.1.0.offxml")
        lig_interchange = off_ff.create_interchange(lig_mol.to_topology())

        # For the combined system, use ParmEd to merge
        import parmed

        # Protein
        protein_ff = app.ForceField(self.forcefield_name, self._water_ff_file())
        prot_system = protein_ff.createSystem(
            protein_pdb.topology,
            nonbondedMethod=app.NoCutoff,
        )
        prot_struct = parmed.openmm.load_topology(
            protein_pdb.topology,
            prot_system,
            protein_pdb.positions,
        )

        # Ligand - convert from Interchange to ParmEd
        lig_struct = lig_interchange.to_parmed()

        # Merge
        combined = prot_struct + lig_struct
        combined_system = combined.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
        )

        # Solvate
        modeller = app.Modeller(combined.topology, combined.positions)
        buffer_nm = self.buffer_A * 0.1 * unit.nanometers
        modeller.addSolvent(
            protein_ff,
            model=self.water_model,
            padding=buffer_nm,
            neutralize=self.neutralize,
        )

        final_system = protein_ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
        )

        return modeller.topology, modeller.positions, final_system, protein_ff

    # ------------------------------------------------------------------
    # Water FF mapping
    # ------------------------------------------------------------------

    def _water_ff_file(self) -> str:
        """Map water model name to OpenMM force field XML.

        Note: "tip4p" maps to tip4pew.xml because OpenMM does not ship a
        standalone TIP4P XML - TIP4P-Ew is the OpenMM implementation of the
        4-site model.  The original TIP4P (Jorgensen 1983) was parameterised
        for truncated electrostatics; TIP4P-Ew (Horn 2004) was re-fit for
        Ewald/PME sums.  All GCMC / bulk-energy constants are looked up from
        ``WATER_MODEL_PARAMS`` keyed by the user-facing model name so that
        the thermodynamic references remain self-consistent.

        """
        mapping = {
            "tip3p": "amber14/tip3p.xml",
            "tip4p": "amber14/tip4pew.xml",    # OpenMM TIP4P-Ew is the 4-site model
            "tip4pew": "amber14/tip4pew.xml",
            "tip5p": "amber14/tip5p.xml",
            "tip3pfb": "amber14/tip3pfb.xml",
            "tip4pfb": "amber14/tip4pfb.xml",
        }
        wm = self.water_model.lower()
        if wm not in mapping:
            raise ValueError(
                f"Unknown water model '{self.water_model}'. "
                f"Supported: {list(mapping.keys())}"
            )
        return mapping[wm]


# ======================================================================
# Geometry helpers
# ======================================================================

def _tetrahedral_h(center: np.ndarray, bond_axis: np.ndarray,
                   bond_length: float = 0.109) -> list[np.ndarray]:
    """Place 3 H atoms around an sp3 carbon in tetrahedral arrangement.

    Args:
        center: position of the sp3 carbon (nm)
        bond_axis: direction FROM the carbon TOWARD its bonded partner
        bond_length: C-H bond length in nm (default 1.09 Å)

    Returns:
        List of 3 position arrays (nm)
    """
    d = bond_axis / np.linalg.norm(bond_axis)

    # Two perpendicular basis vectors
    if abs(d[0]) < 0.9:
        perp1 = np.cross(d, [1.0, 0.0, 0.0])
    else:
        perp1 = np.cross(d, [0.0, 1.0, 0.0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(d, perp1)
    perp2 /= np.linalg.norm(perp2)

    # Tetrahedral: H atoms point away from existing bond at ~109.47°
    # cos(109.47°) ≈ -1/3 along -d, sin(109.47°) ≈ √(8/9) perpendicular
    axial = -bond_length / 3.0
    radial = bond_length * np.sqrt(8.0 / 9.0)

    positions = []
    for i in range(3):
        angle = 2.0 * np.pi * i / 3.0
        h_pos = (center
                 + axial * d
                 + radial * (np.cos(angle) * perp1 + np.sin(angle) * perp2))
        positions.append(h_pos)
    return positions
