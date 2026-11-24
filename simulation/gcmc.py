# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
Grand Canonical Monte Carlo (GCMC) pocket solvation.

This Stage 7 implementation keeps the surrounding H2OSite MD pipeline
unchanged, but swaps the old ``grand``-backed GCMC-MD layer for the
``loch``/Sire execution model used by SOMD2:

1. Round-trip the current OpenMM state through AMBER files into Sire.
2. Create a ``loch.GCMCSampler`` on the Sire system.
3. Alternate GCMC moves with short Sire/OpenMM MD blocks.
4. Strip any remaining ghost waters and round-trip back to OpenMM.
"""

from __future__ import annotations

import importlib
import itertools
import hashlib
import gc
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _find_nonbonded_force(system):
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if force.__class__.__name__ == "NonbondedForce":
            return force
    raise RuntimeError("System does not contain a NonbondedForce")


def extract_nonbonded_forcefield_kwargs(system) -> dict[str, object]:
    """Extract the original force-field nonbonded settings from a system."""
    import openmm
    import openmm.app as app

    nonbonded_force = _find_nonbonded_force(system)
    method = nonbonded_force.getNonbondedMethod()

    method_map = {
        openmm.NonbondedForce.NoCutoff: app.NoCutoff,
        openmm.NonbondedForce.CutoffNonPeriodic: app.CutoffNonPeriodic,
        openmm.NonbondedForce.CutoffPeriodic: app.CutoffPeriodic,
        openmm.NonbondedForce.Ewald: app.Ewald,
        openmm.NonbondedForce.PME: app.PME,
    }
    ljpme_force = getattr(openmm.NonbondedForce, "LJPME", None)
    ljpme_app = getattr(app, "LJPME", None)
    if ljpme_force is not None and ljpme_app is not None:
        method_map[ljpme_force] = ljpme_app

    if method not in method_map:
        raise RuntimeError(f"Unsupported NonbondedForce method: {method}")

    kwargs: dict[str, object] = {
        "nonbondedMethod": method_map[method],
    }
    if method != openmm.NonbondedForce.NoCutoff:
        kwargs["nonbondedCutoff"] = nonbonded_force.getCutoffDistance()
        if nonbonded_force.getUseSwitchingFunction():
            kwargs["switchDistance"] = nonbonded_force.getSwitchingDistance()
    return kwargs


def _extract_loch_cutoff_settings(system) -> tuple[str, str]:
    """Map the OpenMM nonbonded settings onto loch/Sire cutoff arguments."""
    import openmm
    import openmm.unit as unit

    nonbonded_force = _find_nonbonded_force(system)
    method = nonbonded_force.getNonbondedMethod()

    pme_methods = {
        openmm.NonbondedForce.PME,
        openmm.NonbondedForce.Ewald,
    }
    ljpme_force = getattr(openmm.NonbondedForce, "LJPME", None)
    if ljpme_force is not None:
        pme_methods.add(ljpme_force)

    rf_methods = {
        openmm.NonbondedForce.CutoffPeriodic,
        openmm.NonbondedForce.CutoffNonPeriodic,
    }

    if method in pme_methods:
        cutoff_type = "pme"
    elif method in rf_methods:
        cutoff_type = "rf"
    else:
        raise RuntimeError(
            "loch-backed GCMC requires PME/Ewald/LJPME or cutoff-based electrostatics"
        )

    cutoff_A = nonbonded_force.getCutoffDistance().value_in_unit(unit.angstrom)
    return cutoff_type, f"{cutoff_A:.6f} A"


def _parse_atomnum_reference_indices(reference_selection: str) -> list[int] | None:
    """Parse a simple ``atomnum`` selection into zero-based OpenMM indices."""
    matches = re.findall(r"atomnum\s+(\d+)", reference_selection, flags=re.IGNORECASE)
    if not matches:
        return None

    stripped = re.sub(r"atomnum\s+\d+", "", reference_selection, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+or\s+", "", stripped, flags=re.IGNORECASE)
    if stripped.strip():
        return None

    return [max(0, int(atomnum) - 1) for atomnum in matches]


def _prepare_openbiosim_runtime():
    """Import sire/BioSimSpace/loch with the compatibility shim this env needs."""
    mpl_dir = Path(tempfile.gettempdir()) / "H2OSite_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import sire as sr

    legacy = importlib.import_module("sire.legacy")
    legacy_io = importlib.import_module("sire.legacy.IO")
    setattr(legacy, "IO", legacy_io)

    # Import BioSimSpace before loch so the mixed sire API initialises cleanly.
    import BioSimSpace  # noqa: F401
    import loch

    return sr, loch


class GCMCSolvation:
    """GCMC solvation of protein binding pockets via loch/SOMD2-style GCMC-MD."""

    def __init__(
        self,
        topology,
        system,
        positions,
        center_nm: np.ndarray,
        radius_nm: float = 1.0,
        mu_excess_kcal: float = -6.058,
        temperature_K: float = 300.0,
        seed: int = 1234,
        forcefield=None,
        n_ghosts: int = 10,
        box_vectors=None,
        water_model: str = "tip3p",
        flexible_water: bool = True,
        timestep_fs: float = 1.0,
        minimize_after_customization: bool = True,
        redraw_velocities_after_initialise: bool = False,
        ghost_min_distance_A: float = 1.5,
        reference_selection: str | None = None,
        batch_size: int | None = None,
        bulk_sampling_probability: float = 0.1,
        standard_volume_A3: float = 30.543,
        platform: str = "auto",
        device: int | None = None,
        restrained_atom_indices: list[int] | None = None,
        restraint_k_kcal: float = 5.0,
        log_level: str = "info",
        cache_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        self.topology = topology
        self.system = system
        self.positions = positions
        self.center_nm = np.asarray(center_nm, dtype=float)
        self.radius_nm = radius_nm
        self.mu_excess_kcal = mu_excess_kcal
        self.temperature_K = temperature_K
        self.seed = seed
        self.forcefield = forcefield
        self.n_ghosts = n_ghosts
        self.box_vectors = box_vectors
        self.water_model = water_model
        self.flexible_water = flexible_water
        self.timestep_fs = timestep_fs
        self.minimize_after_customization = minimize_after_customization
        self.redraw_velocities_after_initialise = (
            redraw_velocities_after_initialise
        )
        self.ghost_min_distance_A = ghost_min_distance_A
        self.reference_selection = reference_selection
        self.batch_size = batch_size
        self.bulk_sampling_probability = bulk_sampling_probability
        self.standard_volume_A3 = standard_volume_A3
        self.platform = platform.lower()
        self.device = device
        self.restrained_atom_indices = (
            None
            if restrained_atom_indices is None
            else [int(idx) for idx in restrained_atom_indices]
        )
        self.restraint_k_kcal = float(restraint_k_kcal)
        self.log_level = log_level
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        n_gcmc_moves: int = 34000,
        n_md_steps_per_cycle: int = 500,
        n_cycles: int = 100,
    ) -> tuple:
        """Run the GCMC solvation protocol."""
        try:
            return self._run_loch(n_gcmc_moves, n_md_steps_per_cycle, n_cycles)
        except ImportError:
            logger.warning(
                "loch/BioSimSpace/Sire is not importable in this environment; "
                "falling back to simplified pocket equilibration"
            )
            return self._run_simplified(
                n_gcmc_moves, n_md_steps_per_cycle, n_cycles
            )

    def _create_simulation(self, app, topology, system, integrator, openmm):
        platform_specs = [
            ("CUDA", {"Precision": "mixed"}),
            ("OpenCL", {}),
            ("CPU", {}),
        ]
        last_error = None

        for platform_name, properties in platform_specs:
            try:
                platform = openmm.Platform.getPlatformByName(platform_name)
                for key, value in properties.items():
                    platform.setPropertyDefaultValue(key, value)
                simulation = app.Simulation(
                    topology, system, integrator, platform
                )
                logger.info("Using %s platform for GCMC", platform_name)
                return simulation
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to initialize %s platform for GCMC: %s",
                    platform_name,
                    exc,
                )

        raise RuntimeError(
            "Unable to create an OpenMM Simulation for GCMC"
        ) from last_error

    def _positions_nm(self) -> np.ndarray:
        import openmm.unit as unit

        return np.asarray(self.positions.value_in_unit(unit.nanometer), dtype=float)

    def _build_loch_positional_restraints(self, sr, gcmc_system):
        """Create Sire positional restraints for Stage 7 retained restraints."""
        if not self.restrained_atom_indices or self.restraint_k_kcal <= 0.0:
            return None

        n_atoms = len(gcmc_system.atoms())
        valid_indices = sorted(
            {
                int(idx)
                for idx in self.restrained_atom_indices
                if 0 <= int(idx) < n_atoms
            }
        )
        if not valid_indices:
            logger.warning(
                "No valid loch restraint atoms remained after AMBER/Sire roundtrip"
            )
            return None

        restraint_k = f"{self.restraint_k_kcal:.6f} kcal mol-1 A-2"
        restraints = sr.restraints.positional(
            gcmc_system,
            valid_indices,
            k=restraint_k,
            name="H2OSite_stage7_positional_restraints",
        )
        logger.info(
            "Applied loch positional restraints to %d atoms at %s",
            len(valid_indices),
            restraint_k,
        )
        return restraints

    @staticmethod
    def _is_hydrogen(atom) -> bool:
        return (
            atom.element is not None and atom.element.symbol == "H"
        ) or atom.name.startswith("H")

    def _auto_reference_selection(self) -> str:
        """Pick a compact Sire selection whose centroid best matches center_nm."""
        positions_nm = self._positions_nm()

        candidates: list[tuple[float, int, object]] = []
        for atom in self.topology.atoms():
            if self._is_hydrogen(atom):
                continue
            resname = atom.residue.name.upper()
            if resname in {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "TIP4", "TIP4P"}:
                continue
            distance = float(np.linalg.norm(positions_nm[atom.index] - self.center_nm))
            candidates.append((distance, atom.index, atom))

        if not candidates:
            raise RuntimeError(
                "Could not auto-generate a loch GCMC reference selection"
            )

        candidates.sort(key=lambda item: item[0])
        search_pool = candidates[: min(16, len(candidates))]

        best_combo: tuple[int, ...] | None = None
        best_error = float("inf")
        best_centroid = None

        for size in range(1, min(4, len(search_pool)) + 1):
            for combo in itertools.combinations(search_pool, size):
                atom_indices = tuple(item[1] for item in combo)
                centroid = np.mean(positions_nm[list(atom_indices)], axis=0)
                error = float(np.linalg.norm(centroid - self.center_nm))
                if error < best_error:
                    best_error = error
                    best_combo = atom_indices
                    best_centroid = centroid

        if best_combo is None or best_centroid is None:
            raise RuntimeError(
                "Failed to determine a loch GCMC reference selection"
            )

        selection = " or ".join(f"atomnum {idx + 1}" for idx in best_combo)
        logger.info(
            "Auto-selected loch GCMC reference %s (centroid offset %.3f A)",
            selection,
            10.0 * best_error,
        )
        return selection

    def _get_reference_selection(self) -> str:
        if self.reference_selection:
            logger.info(
                "Using explicit loch GCMC reference selection: %s",
                self.reference_selection,
            )
            return self.reference_selection
        return self._auto_reference_selection()

    def _write_openmm_amber_files(self, temp_dir: str) -> tuple[Path, Path]:
        """Write the current OpenMM state to AMBER topology/coordinate files."""
        from openmm.app.internal.unitcell import computeLengthsAndAngles
        import parmed as pmd

        prmtop = Path(temp_dir) / "system.prmtop"
        inpcrd = Path(temp_dir) / "system.inpcrd"
        logger.info(
            "Writing loch AMBER input files in %s (particles=%d residues=%d)",
            temp_dir,
            self.system.getNumParticles(),
            self.topology.getNumResidues(),
        )

        export_system = self.system
        if self.forcefield is not None:
            # Export through an unconstrained system so ParmEd has explicit bond
            # parameters to serialise into AMBER format.
            export_started_at = self._log_step_start(
                "Rebuilding unconstrained OpenMM export system"
            )
            export_kwargs = extract_nonbonded_forcefield_kwargs(self.system)
            export_kwargs["constraints"] = None
            export_kwargs["rigidWater"] = False
            export_system = self.forcefield.createSystem(
                self.topology, **export_kwargs
            )
            self._log_step_done(
                "Rebuilding unconstrained OpenMM export system",
                export_started_at,
            )

        parmed_started_at = self._log_step_start(
            "Loading OpenMM topology/system into ParmEd"
        )
        struct = pmd.openmm.load_topology(
            self.topology,
            export_system,
            xyz=self.positions,
        )
        self._log_step_done(
            "Loading OpenMM topology/system into ParmEd",
            parmed_started_at,
        )

        box_vectors = self.box_vectors
        if box_vectors is None:
            box_vectors = self.topology.getPeriodicBoxVectors()

        if box_vectors is not None:
            a, b, c, alpha, beta, gamma = computeLengthsAndAngles(box_vectors)
            struct.box = [
                10.0 * a,
                10.0 * b,
                10.0 * c,
                np.degrees(alpha),
                np.degrees(beta),
                np.degrees(gamma),
            ]

        save_started_at = self._log_step_start("Saving AMBER prmtop/inpcrd for loch")
        struct.save(str(prmtop), overwrite=True)
        struct.save(str(inpcrd), overwrite=True)
        self._log_step_done("Saving AMBER prmtop/inpcrd for loch", save_started_at)
        logger.info(
            "Loch AMBER input files ready: %s (%d bytes), %s (%d bytes)",
            prmtop,
            prmtop.stat().st_size,
            inpcrd,
            inpcrd.stat().st_size,
        )
        return prmtop, inpcrd

    @staticmethod
    def _resnum_to_int(resnum) -> int:
        try:
            return int(resnum.value())
        except Exception:
            text = str(resnum)
            if text.startswith("ResNum(") and text.endswith(")"):
                text = text[7:-1]
            return int(text)

    def _remove_ghost_waters(self, sire_system, ghost_residues):
        """Remove the final ghost waters from a committed Sire system."""
        ghost_ids = {int(x) for x in np.asarray(ghost_residues, dtype=int).tolist()}
        if not ghost_ids:
            return sire_system.clone(), 0

        stripped = sire_system.clone()
        removed = 0
        for mol in list(stripped.molecules()):
            residues = list(mol.residues())
            if len(residues) != 1:
                continue
            resid = self._resnum_to_int(residues[0].number())
            if resid in ghost_ids:
                stripped.remove(mol)
                removed += 1

        return stripped, removed

    def _ghost_atom_mask(self, sire_system, ghost_residues) -> np.ndarray:
        """Return a boolean mask for atoms retained after ghost stripping."""
        ghost_ids = {int(x) for x in np.asarray(ghost_residues, dtype=int).tolist()}
        keep_mask: list[bool] = []

        for mol in list(sire_system.molecules()):
            residues = list(mol.residues())
            is_ghost = False
            if len(residues) == 1:
                resid = self._resnum_to_int(residues[0].number())
                is_ghost = resid in ghost_ids

            atom_count = len(list(mol.atoms()))
            keep_mask.extend([not is_ghost] * atom_count)

        return np.asarray(keep_mask, dtype=bool)

    def _wrapped_openmm_positions_without_ghosts(
        self,
        dynamics_context,
        sire_system,
        ghost_residues,
    ):
        """Get wrapped OpenMM positions and drop atoms belonging to ghost waters."""
        import openmm.unit as unit

        state = dynamics_context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        )
        positions = state.getPositions(asNumpy=True)
        keep_mask = self._ghost_atom_mask(sire_system, ghost_residues)

        positions_nm = np.asarray(
            positions.value_in_unit(unit.nanometer),
            dtype=float,
        )
        if keep_mask.size != positions_nm.shape[0]:
            raise RuntimeError(
                "Ghost-mask atom count does not match OpenMM context positions: "
                f"mask={keep_mask.size}, positions={positions_nm.shape[0]}"
            )

        kept_positions_nm = positions_nm[keep_mask]
        kept_positions = unit.Quantity(kept_positions_nm, unit.nanometer)
        box_vectors = state.getPeriodicBoxVectors()

        max_abs_A = 10.0 * float(np.max(np.abs(kept_positions_nm)))
        logger.info(
            "Prepared wrapped ghost-free OpenMM positions: kept_atoms=%d "
            "removed_atoms=%d max_abs_coordinate=%.3f A",
            kept_positions_nm.shape[0],
            positions_nm.shape[0] - kept_positions_nm.shape[0],
            max_abs_A,
        )
        return kept_positions, box_vectors

    def _wrapped_openmm_positions(self, dynamics_context):
        """Get wrapped OpenMM positions without changing atom ordering."""
        import openmm.unit as unit

        state = dynamics_context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        )
        positions = state.getPositions(asNumpy=True)
        positions_nm = np.asarray(
            positions.value_in_unit(unit.nanometer),
            dtype=float,
        )
        max_abs_A = 10.0 * float(np.max(np.abs(positions_nm)))
        logger.info(
            "Prepared wrapped full OpenMM positions: atoms=%d "
            "max_abs_coordinate=%.3f A",
            positions_nm.shape[0],
            max_abs_A,
        )
        return positions, state.getPeriodicBoxVectors()

    def _topology_from_sire_system(
        self,
        sr,
        sire_system,
        temp_dir: str,
        filename: str = "gcmc_system.prmtop",
    ):
        """Write a Sire system to prmtop and reload just the OpenMM topology."""
        import openmm.app as app

        prmtop_path = Path(temp_dir) / filename
        logger.info("Saving loch topology snapshot to %s", prmtop_path)
        sr.save(sire_system, str(prmtop_path))
        prmtop = app.AmberPrmtopFile(str(prmtop_path))
        return prmtop.topology

    def _remove_ghost_waters_openmm(
        self,
        topology,
        positions,
        box_vectors,
        ghost_residues,
    ) -> tuple:
        """Remove ghost waters from a matching OpenMM topology/position pair."""
        import openmm.app as app

        ghost_ids = {int(x) for x in np.asarray(ghost_residues, dtype=int).tolist()}
        modeller = app.Modeller(topology, positions)
        residues = list(modeller.topology.residues())

        residues_to_delete = []
        for residx, residue in enumerate(residues):
            matches_index = residx in ghost_ids
            matches_resid = False
            try:
                matches_resid = int(residue.id) in ghost_ids
            except Exception:
                matches_resid = False
            if matches_index or matches_resid:
                residues_to_delete.append(residue)

        if residues_to_delete:
            logger.info(
                "Removing %d ghost residues in OpenMM topology space",
                len(residues_to_delete),
            )
            modeller.delete(residues_to_delete)

        if box_vectors is not None:
            modeller.topology.setPeriodicBoxVectors(box_vectors)

        return modeller.topology, modeller.positions, len(residues_to_delete)

    def _roundtrip_back_to_openmm(
        self,
        sr,
        sire_system,
        temp_dir: str,
        positions=None,
        box_vectors=None,
    ) -> tuple:
        """Write the final ghost-free Sire system and reload it into OpenMM."""
        import openmm.app as app

        prmtop_path = Path(temp_dir) / "gcmc_system.prmtop"

        logger.info("Saving ghost-free loch topology to %s", prmtop_path)
        sr.save(sire_system, str(prmtop_path))
        prmtop = app.AmberPrmtopFile(str(prmtop_path))
        topology = prmtop.topology

        if positions is not None:
            if box_vectors is not None:
                topology.setPeriodicBoxVectors(box_vectors)
            logger.info(
                "Reloaded ghost-free topology from %s using wrapped OpenMM positions",
                prmtop_path,
            )
            return topology, positions

        inpcrd_path = Path(temp_dir) / "gcmc_system.inpcrd"
        logger.info("Falling back to Sire-written AmberInpcrdFile: %s", inpcrd_path)
        sr.save(sire_system, str(inpcrd_path))
        inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
        if inpcrd.boxVectors is not None:
            topology.setPeriodicBoxVectors(inpcrd.boxVectors)
        return topology, inpcrd.positions

    @staticmethod
    def _safe_rate(count: float, elapsed_s: float) -> float:
        if elapsed_s <= 0.0:
            return float("inf")
        return float(count) / float(elapsed_s)

    @staticmethod
    def _log_step_start(label: str) -> float:
        logger.info("[Loch step] %s...", label)
        return time.perf_counter()

    @staticmethod
    def _log_step_done(label: str, started_at: float) -> None:
        logger.info(
            "[Loch step] %s done in %.3f s",
            label,
            time.perf_counter() - started_at,
        )

    def _log_runtime_diagnostics(self, loch_module=None) -> None:
        import openmm

        platform_names = [
            openmm.Platform.getPlatform(idx).getName()
            for idx in range(openmm.Platform.getNumPlatforms())
        ]
        has_box = (
            self.box_vectors is not None
            or self.topology.getPeriodicBoxVectors() is not None
        )
        logger.info(
            "Loch runtime diagnostics: particles=%d residues=%d "
            "has_periodic_box=%s forcefield_available=%s water_model=%s",
            self.system.getNumParticles(),
            self.topology.getNumResidues(),
            has_box,
            self.forcefield is not None,
            self.water_model,
        )
        logger.info(
            "Loch environment: CUDA_VISIBLE_DEVICES=%r OPENCL_VISIBLE_DEVICES=%r "
            "PYCUDA_NVCC=%r nvcc=%r",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ.get("OPENCL_VISIBLE_DEVICES"),
            os.environ.get("PYCUDA_NVCC"),
            shutil.which("nvcc"),
        )
        logger.info(
            "OpenMM platforms detected: %s",
            ", ".join(platform_names) if platform_names else "<none>",
        )
        if loch_module is not None:
            logger.info(
                "Loch module path: %s",
                getattr(loch_module, "__file__", "<unknown>"),
            )

    def _get_cache_dir(self) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def _current_box_vectors_nm(self) -> np.ndarray | None:
        import openmm.unit as unit

        box_vectors = self.box_vectors or self.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            return None

        return np.asarray(
            box_vectors.value_in_unit(unit.nanometer),
            dtype=float,
        )

    def _current_sire_space(self, sr):
        box_nm = self._current_box_vectors_nm()
        if box_nm is None:
            return sr.vol.Cartesian()

        box_A = 10.0 * box_nm
        if np.allclose(box_A - np.diag(np.diag(box_A)), 0.0, atol=1.0e-6):
            lengths_A = np.diag(box_A)
            return sr.vol.PeriodicBox(sr.maths.Vector(*lengths_A))

        return sr.vol.TriclinicBox(
            sr.maths.Vector(*box_A[0]),
            sr.maths.Vector(*box_A[1]),
            sr.maths.Vector(*box_A[2]),
        )

    def _compute_sampler_fingerprint(
        self,
        *,
        reference_selection: str,
        cutoff_type: str,
        cutoff: str,
        moves_per_cycle: int,
        batch_size: int,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(self.topology.getNumAtoms()).encode())
        hasher.update(str(self.topology.getNumResidues()).encode())
        hasher.update(str(self.system.getNumParticles()).encode())
        for residue in self.topology.residues():
            chain = getattr(residue.chain, "id", "")
            atom_count = sum(1 for _ in residue.atoms())
            hasher.update(
                f"{chain}|{residue.id}|{residue.name}|{atom_count}".encode()
            )

        payload = "|".join(
            [
                reference_selection,
                cutoff_type,
                cutoff,
                f"{self.radius_nm:.6f}",
                f"{self.mu_excess_kcal:.6f}",
                f"{self.temperature_K:.6f}",
                f"{self.standard_volume_A3:.6f}",
                f"{self.bulk_sampling_probability:.6f}",
                str(self.n_ghosts),
                self.water_model,
                "prepared-cache-v2",
            ]
        )
        hasher.update(payload.encode())
        return hasher.hexdigest()

    def _compute_legacy_sampler_fingerprint(
        self,
        *,
        reference_selection: str,
        cutoff_type: str,
        cutoff: str,
        moves_per_cycle: int,
        batch_size: int,
    ) -> str:
        import openmm.unit as unit

        positions_nm = np.asarray(
            self.positions.value_in_unit(unit.nanometer),
            dtype=np.float32,
        )
        hasher = hashlib.sha256()
        hasher.update(str(self.topology.getNumAtoms()).encode())
        hasher.update(str(self.topology.getNumResidues()).encode())
        hasher.update(str(self.system.getNumParticles()).encode())
        hasher.update(positions_nm.tobytes())

        box_vectors = self.box_vectors or self.topology.getPeriodicBoxVectors()
        if box_vectors is not None:
            box_nm = np.asarray(
                box_vectors.value_in_unit(unit.nanometer),
                dtype=np.float32,
            )
            hasher.update(box_nm.tobytes())

        payload = "|".join(
            [
                reference_selection,
                cutoff_type,
                cutoff,
                f"{self.radius_nm:.6f}",
                f"{self.mu_excess_kcal:.6f}",
                f"{self.temperature_K:.6f}",
                f"{self.standard_volume_A3:.6f}",
                f"{self.bulk_sampling_probability:.6f}",
                str(self.n_ghosts),
                str(self.seed),
                str(moves_per_cycle),
                str(batch_size),
                self.platform,
                str(self.device),
                self.water_model,
                str(self.flexible_water),
                f"{self.timestep_fs:.6f}",
                "v1",
            ]
        )
        hasher.update(payload.encode())
        return hasher.hexdigest()

    def _refresh_cached_prepared_system(self, sr, prepared_system):
        import openmm.unit as unit

        positions_nm = np.asarray(
            self.positions.value_in_unit(unit.nanometer),
            dtype=float,
        )
        system_cursor = prepared_system.cursor()
        atom_cursors = system_cursor.atoms()
        if len(atom_cursors) != len(positions_nm):
            raise ValueError(
                "Cached loch system atom count does not match current OpenMM "
                f"state ({len(atom_cursors)} != {len(positions_nm)})"
            )

        refresh_started_at = self._log_step_start(
            "Refreshing cached loch system coordinates from current OpenMM state"
        )
        atom_cursors["coordinates"] = [
            sr.maths.Vector(*(10.0 * xyz_nm))
            for xyz_nm in positions_nm
        ]
        refreshed_system = system_cursor.commit()
        refreshed_system.set_space(self._current_sire_space(sr))
        self._log_step_done(
            "Refreshing cached loch system coordinates from current OpenMM state",
            refresh_started_at,
        )
        return refreshed_system

    def _sampler_cache_paths(
        self,
        fingerprint: str,
        *,
        create: bool = True,
    ) -> dict[str, Path] | None:
        cache_dir = self._get_cache_dir()
        if cache_dir is None:
            return None

        sampler_dir = cache_dir / fingerprint
        kernel_dir = cache_dir / "kernel_cache"
        if create:
            sampler_dir.mkdir(parents=True, exist_ok=True)
            kernel_dir.mkdir(parents=True, exist_ok=True)
        elif not (sampler_dir.exists() and kernel_dir.exists()):
            return None

        return {
            "root": sampler_dir,
            "prepared_prmtop": sampler_dir / "prepared_system.prmtop",
            "prepared_inpcrd": sampler_dir / "prepared_system.inpcrd",
            "snapshot": sampler_dir / "prepared_sampler.pkl",
            "kernel_dir": kernel_dir,
        }

    @staticmethod
    def _seed_loch_disk_kernel_cache(
        platform: str,
        device: int,
        compiler_optimisations: bool,
        kernel_dir: Path,
    ) -> bool:
        if platform == "cuda":
            module = importlib.import_module("loch._platforms._cuda")
            ext = "cubin"
        elif platform == "opencl":
            module = importlib.import_module("loch._platforms._opencl")
            ext = "bin"
        else:
            return False

        cache_key = (device, compiler_optimisations)
        kernel_path = (
            kernel_dir
            / f"{platform}_device{device}_fastmath{int(compiler_optimisations)}.{ext}"
        )
        if not kernel_path.exists():
            return False

        with open(kernel_path, "rb") as handle:
            module._kernel_cache[cache_key] = handle.read()
        return True

    @staticmethod
    def _persist_loch_disk_kernel_cache(
        sampler,
        platform: str,
        device: int,
        compiler_optimisations: bool,
        kernel_dir: Path,
    ) -> Path | None:
        if platform == "cuda":
            module = importlib.import_module("loch._platforms._cuda")
            ext = "cubin"
        elif platform == "opencl":
            module = importlib.import_module("loch._platforms._opencl")
            ext = "bin"
        else:
            return None

        cache_key = (device, compiler_optimisations)
        blob = module._kernel_cache.get(cache_key)
        if blob is None:
            return None

        kernel_path = (
            kernel_dir
            / f"{platform}_device{device}_fastmath{int(compiler_optimisations)}.{ext}"
        )
        with open(kernel_path, "wb") as handle:
            handle.write(blob)
        return kernel_path

    @staticmethod
    def _pop_loch_cuda_context(sampler) -> bool:
        pop = getattr(sampler, "pop", None)
        if pop is None:
            return False
        pop()
        return True

    @staticmethod
    def _push_loch_cuda_context(sampler) -> bool:
        push = getattr(sampler, "push", None)
        if push is None:
            return False
        push()
        return True

    @staticmethod
    def _release_loch_backend_resources(sampler) -> bool:
        backend = getattr(sampler, "_backend", None)
        if backend is None:
            return False

        rng_manager = getattr(sampler, "_rng_manager", None)
        if rng_manager is not None:
            try:
                rng_manager.shutdown()
            except Exception:
                pass
            sampler._rng_manager = None

        gpu_buffer_names = [
            name
            for name in vars(sampler)
            if name.startswith("_gpu_")
        ]
        gpu_buffer_names.extend(
            [
                "_water_positions",
                "_energy_coul",
                "_energy_lj",
                "_accepted",
                "_energy_change",
                "_probability",
                "_deletion_candidates",
            ]
        )

        for name in gpu_buffer_names:
            if hasattr(sampler, name):
                setattr(sampler, name, None)

        try:
            backend.cleanup()
        finally:
            sampler._backend = None
            sampler._kernels = None
            gc.collect()

        return True

    def _rebuild_loch_backend_resources(self, sampler) -> None:
        create_backend = importlib.import_module(
            "loch._platforms"
        ).create_backend
        rng_manager_cls = importlib.import_module(
            "loch._platforms._rng"
        ).RNGManager

        backend = create_backend(
            platform=sampler._platform,
            device=sampler._device,
            num_points=sampler._num_points,
            num_batch=sampler._batch_size,
            num_waters=sampler._num_waters,
            num_atoms=sampler._num_atoms,
            num_threads=sampler._num_threads,
            nvcc=sampler._nvcc,
            compiler_optimisations=sampler._compiler_optimisations,
        )
        sampler._backend = backend
        sampler._kernels = backend.compile_kernels()
        sampler._rng_manager = rng_manager_cls(
            sampler._batch_size,
            seed=sampler._seed,
        )
        sampler._atom_blocks = sampler._num_atoms // sampler._num_threads + 1
        sampler._batch_blocks = sampler._batch_size // sampler._num_threads + 1
        sampler._water_blocks = sampler._num_waters // sampler._num_threads + 1
        sampler._initialise_gpu_memory()
        sampler.set_box(sampler._system)

    def _save_sampler_checkpoint(
        self,
        *,
        sr,
        sampler,
        cache_paths: dict[str, Path],
        fingerprint: str,
    ) -> None:
        prepared_system = sampler.system()
        sr.save(prepared_system, str(cache_paths["prepared_prmtop"]))
        sr.save(prepared_system, str(cache_paths["prepared_inpcrd"]))

        snapshot = {
            "version": 2,
            "fingerprint": fingerprint,
            "is_fep": getattr(sampler, "_is_fep", False),
            "water_indices": np.asarray(sampler._water_indices, dtype=np.int32),
            "water_residues": np.asarray(sampler._water_residues, dtype=np.int32),
            "reference_indices": (
                None
                if getattr(sampler, "_reference", None) is None
                else np.asarray(sampler._reference_indices, dtype=np.int32)
            ),
        }
        with open(cache_paths["snapshot"], "wb") as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _restore_sampler_checkpoint(
        self,
        *,
        sr,
        loch,
        cache_paths: dict[str, Path],
        fingerprint: str,
        reference_selection: str,
        cutoff_type: str,
        cutoff: str,
        moves_per_cycle: int,
        batch_size: int,
        ghost_file: str,
        log_file: str,
    ):
        snapshot_path = cache_paths["snapshot"]
        prmtop_path = cache_paths["prepared_prmtop"]
        inpcrd_path = cache_paths["prepared_inpcrd"]

        if not (
            snapshot_path.exists()
            and prmtop_path.exists()
            and inpcrd_path.exists()
        ):
            return None

        with open(snapshot_path, "rb") as handle:
            snapshot = pickle.load(handle)

        if snapshot.get("version") not in {1, 2}:
            return None
        if snapshot.get("fingerprint") != fingerprint:
            return None

        prepared_system = sr.load(str(prmtop_path), str(inpcrd_path))
        prepared_system = self._refresh_cached_prepared_system(sr, prepared_system)

        sampler_cls = loch.GCMCSampler
        sampler = sampler_cls.__new__(sampler_cls)
        sampler._system = prepared_system.clone()
        sampler._is_fep = bool(snapshot.get("is_fep", False))
        sampler._reference = reference_selection
        sampler._cutoff_type = cutoff_type
        sampler._is_pme = cutoff_type == "pme"
        sampler._radius = sr.u(f"{10.0 * self.radius_nm:.6f} A")
        sampler._cutoff = sr.u(cutoff)
        sampler._excess_chemical_potential = sr.u(
            f"{self.mu_excess_kcal:.6f} kcal/mol"
        )
        sampler._standard_volume = sr.u(f"{self.standard_volume_A3:.6f} A^3")
        sampler._temperature = sr.u(f"{self.temperature_K:.6f} K")
        sampler._num_ghost_waters = self.n_ghosts
        sampler._adams_shift = 0.0
        sampler._batch_size = batch_size
        sampler._num_attempts = moves_per_cycle
        sampler._num_threads = 1024
        sampler._bulk_sampling_probability = float(
            self.bulk_sampling_probability
        )
        sampler._overwrite = True
        sampler._restart = False
        sampler._ghost_file = ghost_file
        with open(sampler._ghost_file, "w") as handle:
            handle.write("")
        sampler._log_file = log_file
        with open(sampler._log_file, "w") as handle:
            handle.write("")
        sampler._log_level = self.log_level
        sampler._is_debug = self.log_level.lower().replace(" ", "") == "debug"
        sampler._seed = self.seed
        np.random.seed(self.seed)
        sampler._rng = np.random.default_rng(self.seed)
        sampler._nvcc = os.environ.get("PYCUDA_NVCC", shutil.which("nvcc"))
        sampler._tolerance = 0.0
        sampler._lambda_schedule = None
        sampler._lambda_value = 0.0
        sampler._rest2_scale = 1.0
        sampler._rest2_selection = None
        sampler._coulomb_power = 0.0
        sampler._shift_coulomb = sr.u("1 A")
        sampler._shift_delta = sr.u("1.5 A")
        sampler._swap_end_states = False
        sampler._platform = self.platform
        sampler._device = self.device if self.device is not None else 0
        sampler._compiler_optimisations = True
        sampler._water_template = sampler._system[
            "water and not property is_perturbable"
        ][0]
        sampler._num_points = sampler._water_template.num_atoms()
        sampler._reference_indices = (
            None
            if snapshot.get("reference_indices") is None
            else np.asarray(snapshot["reference_indices"], dtype=np.int32)
        )
        sampler._water_indices = np.asarray(
            snapshot["water_indices"], dtype=np.int32
        )
        sampler._water_residues = np.asarray(
            snapshot["water_residues"], dtype=np.int32
        )
        sampler._num_atoms = sampler._system.num_atoms()
        sampler._num_waters = len(sampler._water_indices)

        create_backend = importlib.import_module(
            "loch._platforms"
        ).create_backend
        rng_manager_cls = importlib.import_module(
            "loch._platforms._rng"
        ).RNGManager

        sampler._backend = create_backend(
            platform=sampler._platform,
            device=sampler._device,
            num_points=sampler._num_points,
            num_batch=sampler._batch_size,
            num_waters=sampler._num_waters,
            num_atoms=sampler._num_atoms,
            num_threads=sampler._num_threads,
            nvcc=sampler._nvcc,
            compiler_optimisations=sampler._compiler_optimisations,
        )
        sampler._kernels = sampler._backend.compile_kernels()
        sampler._rng_manager = rng_manager_cls(sampler._batch_size, seed=sampler._seed)
        sampler._atom_blocks = sampler._num_atoms // sampler._num_threads + 1
        sampler._batch_blocks = sampler._batch_size // sampler._num_threads + 1
        sampler._water_blocks = sampler._num_waters // sampler._num_threads + 1
        sampler._initialise_gpu_memory()
        sampler.set_box(sampler._system)

        sampler._beta = 1.0 / (
            sr.units.k_boltz.to("kcal/(mol*kelvin)") * sampler._temperature.value()
        )
        sampler._beta_openmm = 1.0 / (
            importlib.import_module("openmm").unit.BOLTZMANN_CONSTANT_kB
            * importlib.import_module("openmm").unit.AVOGADRO_CONSTANT_NA
            * sampler._temperature.value()
            * importlib.import_module("openmm").unit.kelvin
        )

        volume = sampler._space.volume().value()
        gcmc_volume = (4.0 * np.pi * sampler._radius.value() ** 3) / 3.0
        B = (
            sampler._beta * sampler._excess_chemical_potential.value()
            + np.log(gcmc_volume / sampler._standard_volume.value())
        ) + sampler._adams_shift
        B_bulk = (
            sampler._beta * sampler._excess_chemical_potential.value()
            + np.log(volume / sampler._standard_volume.value())
        ) + sampler._adams_shift
        sampler._exp_B = np.exp(B)
        sampler._exp_minus_B = np.exp(-B)
        sampler._exp_B_bulk = np.exp(B_bulk)
        sampler._exp_minus_B_bulk = np.exp(-B_bulk)
        sampler._prefactor = 1.0 / (4.0 * np.pi * sr.units.epsilon0.value())
        sampler._N = 0
        sampler._num_moves = 0
        sampler._num_accepted = 0
        sampler._num_accepted_attempts = 0
        sampler._num_insertions = 0
        sampler._num_deletions = 0
        sampler._nonbonded_force = None
        sampler._custom_nonbonded_force = None
        sampler._is_bulk = False
        sampler._openmm_context = None
        sampler._is_test = False
        import atexit

        atexit.register(sampler._cleanup)
        return sampler

    def _collect_cycle_metrics(
        self,
        *,
        phase: str,
        cycle_index: int,
        n_cycles: int,
        moves_per_cycle: int,
        sampler,
        cycle_started_at: float,
        phase_started_at: float,
        run_started_at: float,
        previous_counts: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, float | int | str]]:
        cycle_elapsed_s = time.perf_counter() - cycle_started_at
        phase_elapsed_s = time.perf_counter() - phase_started_at
        total_elapsed_s = time.perf_counter() - run_started_at
        pre_phase_overhead_s = max(0.0, phase_started_at - run_started_at)

        current_counts = {
            "accepted": int(sampler.num_accepted_moves()),
            "insertions": int(sampler.num_insertions()),
            "deletions": int(sampler.num_deletions()),
        }
        deltas = {
            key: current_counts[key] - int(previous_counts.get(key, 0))
            for key in current_counts
        }
        attempted_this_phase = (cycle_index + 1) * int(moves_per_cycle)
        accepted_delta = deltas["accepted"]
        insertion_delta = deltas["insertions"]
        deletion_delta = deltas["deletions"]

        metrics: dict[str, float | int | str] = {
            "phase": phase,
            "cycle_index": cycle_index + 1,
            "n_cycles": int(n_cycles),
            "cycle_time_s": cycle_elapsed_s,
            "phase_elapsed_s": phase_elapsed_s,
            "total_elapsed_s": total_elapsed_s,
            "pre_phase_overhead_s": pre_phase_overhead_s,
            "attempts_this_cycle": int(moves_per_cycle),
            "attempts_this_phase": attempted_this_phase,
            "attempts_per_s_cycle": self._safe_rate(
                moves_per_cycle, cycle_elapsed_s
            ),
            "attempts_per_s_phase_avg": self._safe_rate(
                attempted_this_phase, phase_elapsed_s
            ),
            "accepted_total": current_counts["accepted"],
            "accepted_delta": accepted_delta,
            "accepted_per_s_cycle": self._safe_rate(
                accepted_delta, cycle_elapsed_s
            ),
            "insertions_total": current_counts["insertions"],
            "insertions_delta": insertion_delta,
            "insertions_per_s_cycle": self._safe_rate(
                insertion_delta, cycle_elapsed_s
            ),
            "deletions_total": current_counts["deletions"],
            "deletions_delta": deletion_delta,
            "deletions_per_s_cycle": self._safe_rate(
                deletion_delta, cycle_elapsed_s
            ),
            "water_count": int(sampler.num_waters()),
        }
        return current_counts, metrics

    @staticmethod
    def _log_cycle_metrics(metrics: dict[str, float | int | str]) -> None:
        logger.info(
            "  GCMC %s cycle %d/%d: cycle_time=%.3fs phase_elapsed=%.3fs "
            "pre_phase_overhead=%.3fs total_elapsed=%.3fs "
            "attempts=%d rate=%.1f/s avg=%.1f/s "
            "accepted=%d (%+d, %.3f/s) "
            "insertions=%d (%+d, %.3f/s) "
            "deletions=%d (%+d, %.3f/s) "
            "N=%d",
            metrics["phase"],
            metrics["cycle_index"],
            metrics["n_cycles"],
            metrics["cycle_time_s"],
            metrics["phase_elapsed_s"],
            metrics["pre_phase_overhead_s"],
            metrics["total_elapsed_s"],
            metrics["attempts_this_cycle"],
            metrics["attempts_per_s_cycle"],
            metrics["attempts_per_s_phase_avg"],
            metrics["accepted_total"],
            metrics["accepted_delta"],
            metrics["accepted_per_s_cycle"],
            metrics["insertions_total"],
            metrics["insertions_delta"],
            metrics["insertions_per_s_cycle"],
            metrics["deletions_total"],
            metrics["deletions_delta"],
            metrics["deletions_per_s_cycle"],
            metrics["water_count"],
        )

    def _run_loch(self, n_gcmc_moves, n_md_steps, n_cycles) -> tuple:
        """Run loch GCMC-MD using the same Sire dynamics pattern as SOMD2."""
        runtime_started_at = self._log_step_start(
            "Preparing OpenBioSim runtime (sire/BioSimSpace/loch)"
        )
        sr, loch = _prepare_openbiosim_runtime()
        self._log_step_done(
            "Preparing OpenBioSim runtime (sire/BioSimSpace/loch)",
            runtime_started_at,
        )

        if n_cycles <= 0:
            raise ValueError("'n_cycles' must be greater than zero")
        if n_gcmc_moves <= 0:
            logger.info("No GCMC moves requested; returning the input state unchanged")
            stats = {
                "backend": "loch",
                "n_accepted": 0,
                "n_moves": 0,
                "n_insertions": 0,
                "n_deletions": 0,
                "N_waters": 0,
                "n_equil_cycles": 0,
                "n_prod_cycles": 0,
                "reference_selection": None,
                "mode": "no_gcmc_moves",
            }
            return self.topology, self.positions, stats

        if self.minimize_after_customization or self.redraw_velocities_after_initialise:
            logger.info(
                "Ignoring grand-specific post-initialisation controls for loch "
                "(minimize_after_customization=%s, redraw_velocities=%s)",
                self.minimize_after_customization,
                self.redraw_velocities_after_initialise,
            )

        n_cycles = max(1, int(n_cycles))
        n_equil_cycles = max(1, n_cycles // 2)
        n_prod_cycles = max(1, n_cycles - n_equil_cycles)
        moves_per_cycle = max(1, int(n_gcmc_moves) // n_cycles)
        batch_size = moves_per_cycle if self.batch_size is None else int(self.batch_size)
        batch_size = max(1, min(batch_size, moves_per_cycle))
        md_time_ps = max(0.0, float(n_md_steps) * float(self.timestep_fs) * 1.0e-3)
        md_time = f"{md_time_ps:.6f} ps"
        energy_frequency = md_time if md_time_ps > 0 else "0 ps"
        reference_selection = self._get_reference_selection()
        cutoff_type, cutoff = _extract_loch_cutoff_settings(self.system)
        constraint = "none" if self.flexible_water else "h_bonds"

        logger.info("Running GCMC with loch (SOMD2-style Sire dynamics loop)")
        logger.info(
            "GCMC settings: reference=%s radius=%.3f A num_attempts=%d "
            "batch_size=%d cutoff_type=%s cutoff=%s platform=%s device=%s",
            reference_selection,
            10.0 * self.radius_nm,
            moves_per_cycle,
            batch_size,
            cutoff_type,
            cutoff,
            self.platform,
            self.device,
        )
        logger.info(
            "GCMC MD settings: flexible_water=%s timestep=%.3f fs "
            "constraint=%s md_time_per_cycle=%s",
            self.flexible_water,
            self.timestep_fs,
            constraint,
            md_time,
        )
        if self.restrained_atom_indices and self.restraint_k_kcal > 0.0:
            logger.info(
                "GCMC positional restraints: %d non-water heavy atoms at %.3f kcal/mol/A^2",
                len(self.restrained_atom_indices),
                self.restraint_k_kcal,
            )
        self._log_runtime_diagnostics(loch)
        run_started_at = time.perf_counter()

        fingerprint = self._compute_sampler_fingerprint(
            reference_selection=reference_selection,
            cutoff_type=cutoff_type,
            cutoff=cutoff,
            moves_per_cycle=moves_per_cycle,
            batch_size=batch_size,
        )
        legacy_fingerprint = self._compute_legacy_sampler_fingerprint(
            reference_selection=reference_selection,
            cutoff_type=cutoff_type,
            cutoff=cutoff,
            moves_per_cycle=moves_per_cycle,
            batch_size=batch_size,
        )
        cache_paths = self._sampler_cache_paths(fingerprint)
        legacy_cache_paths = None
        if legacy_fingerprint != fingerprint:
            legacy_cache_paths = self._sampler_cache_paths(
                legacy_fingerprint,
                create=False,
            )
        if cache_paths is not None:
            kernel_seeded = self._seed_loch_disk_kernel_cache(
                self.platform,
                self.device if self.device is not None else 0,
                True,
                cache_paths["kernel_dir"],
            )
            logger.info(
                "Loch persistent cache: dir=%s fingerprint=%s kernel_seeded=%s",
                cache_paths["root"],
                fingerprint[:12],
                kernel_seeded,
            )

        with tempfile.TemporaryDirectory(prefix="H2OSite_loch_") as temp_dir:
            logger.info("Loch temporary workspace: %s", temp_dir)
            ghost_file = str(Path(temp_dir) / "gcmc_ghosts.txt")
            log_file = str(Path(temp_dir) / "gcmc.log")

            sampler = None
            if cache_paths is not None:
                restore_started_at = self._log_step_start(
                    "Restoring prepared loch sampler checkpoint"
                )
                sampler = self._restore_sampler_checkpoint(
                    sr=sr,
                    loch=loch,
                    cache_paths=cache_paths,
                    fingerprint=fingerprint,
                    reference_selection=reference_selection,
                    cutoff_type=cutoff_type,
                    cutoff=cutoff,
                    moves_per_cycle=moves_per_cycle,
                    batch_size=batch_size,
                    ghost_file=ghost_file,
                    log_file=log_file,
                )
                if sampler is None and legacy_cache_paths is not None:
                    sampler = self._restore_sampler_checkpoint(
                        sr=sr,
                        loch=loch,
                        cache_paths=legacy_cache_paths,
                        fingerprint=legacy_fingerprint,
                        reference_selection=reference_selection,
                        cutoff_type=cutoff_type,
                        cutoff=cutoff,
                        moves_per_cycle=moves_per_cycle,
                        batch_size=batch_size,
                        ghost_file=ghost_file,
                        log_file=log_file,
                    )
                    if sampler is not None:
                        logger.info(
                            "Restored legacy loch sampler checkpoint; "
                            "refreshing it into the new cache layout"
                        )
                        self._save_sampler_checkpoint(
                            sr=sr,
                            sampler=sampler,
                            cache_paths=cache_paths,
                            fingerprint=fingerprint,
                        )
                if sampler is None:
                    logger.info(
                        "No reusable loch sampler checkpoint found "
                        "(first compatible run will pay the full setup cost)"
                    )
                else:
                    logger.info(
                        "Restored prepared loch sampler checkpoint "
                        "(kernel_cache_hit=%s)",
                        sampler._backend.cache_hit,
                    )
                self._log_step_done(
                    "Restoring prepared loch sampler checkpoint",
                    restore_started_at,
                )

            if sampler is None:
                amber_export_started_at = self._log_step_start(
                    "Exporting current OpenMM state to AMBER for loch"
                )
                prmtop_path, inpcrd_path = self._write_openmm_amber_files(temp_dir)
                self._log_step_done(
                    "Exporting current OpenMM state to AMBER for loch",
                    amber_export_started_at,
                )

                sire_load_started_at = self._log_step_start(
                    "Loading exported AMBER files into Sire"
                )
                mols = sr.load(str(prmtop_path), str(inpcrd_path))
                self._log_step_done(
                    "Loading exported AMBER files into Sire",
                    sire_load_started_at,
                )

                logger.info(
                    "Loch auxiliary files: ghost_file=%s log_file=%s",
                    ghost_file,
                    log_file,
                )

                sampler_started_at = self._log_step_start(
                    "Constructing loch.GCMCSampler"
                )
                sampler = loch.GCMCSampler(
                    mols,
                    reference=reference_selection,
                    num_attempts=moves_per_cycle,
                    batch_size=batch_size,
                    cutoff_type=cutoff_type,
                    cutoff=cutoff,
                    radius=f"{10.0 * self.radius_nm:.6f} A",
                    temperature=f"{self.temperature_K:.6f} K",
                    excess_chemical_potential=f"{self.mu_excess_kcal:.6f} kcal/mol",
                    standard_volume=f"{self.standard_volume_A3:.6f} A^3",
                    num_ghost_waters=self.n_ghosts,
                    bulk_sampling_probability=float(self.bulk_sampling_probability),
                    platform=self.platform,
                    device=self.device,
                    ghost_file=ghost_file,
                    log_file=log_file,
                    log_level=self.log_level,
                    seed=self.seed,
                )
                self._log_step_done("Constructing loch.GCMCSampler", sampler_started_at)
                logger.info(
                    "Loch sampler constructed (kernel_cache_hit=%s)",
                    sampler._backend.cache_hit,
                )
                if cache_paths is not None:
                    kernel_path = self._persist_loch_disk_kernel_cache(
                        sampler,
                        self.platform,
                        self.device if self.device is not None else 0,
                        True,
                        cache_paths["kernel_dir"],
                    )
                    if kernel_path is not None:
                        logger.info(
                            "Saved persistent loch kernel cache to %s",
                            kernel_path,
                        )
                    save_started_at = self._log_step_start(
                        "Saving prepared loch sampler checkpoint"
                    )
                    self._save_sampler_checkpoint(
                        sr=sr,
                        sampler=sampler,
                        cache_paths=cache_paths,
                        fingerprint=fingerprint,
                    )
                    self._log_step_done(
                        "Saving prepared loch sampler checkpoint",
                        save_started_at,
                    )
                    logger.info(
                        "Prepared loch sampler cache is warm; future matching "
                        "reruns should skip the expensive constructor step"
                    )

            system_started_at = self._log_step_start(
                "Materialising loch GCMC system via sampler.system()"
            )
            gcmc_system = sampler.system()
            self._log_step_done(
                "Materialising loch GCMC system via sampler.system()",
                system_started_at,
            )
            loch_restraints = self._build_loch_positional_restraints(
                sr,
                gcmc_system,
            )

            released_loch_cuda_context = False
            released_loch_backend = False
            if self.platform == "cuda":
                try:
                    released_loch_backend = self._release_loch_backend_resources(
                        sampler
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to release loch GPU backend before "
                        "creating the Sire/OpenMM dynamics context: %s",
                        exc,
                    )
                else:
                    if released_loch_backend:
                        logger.info(
                            "Released loch GPU backend before creating "
                            "the Sire/OpenMM dynamics context"
                        )
                    else:
                        try:
                            released_loch_cuda_context = (
                                self._pop_loch_cuda_context(sampler)
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to release loch PyCUDA context before "
                                "creating the Sire/OpenMM dynamics context: %s",
                                exc,
                            )
                        else:
                            if released_loch_cuda_context:
                                logger.info(
                                    "Released loch PyCUDA context before creating "
                                    "the Sire/OpenMM dynamics context"
                                )

            dynamics_started_at = self._log_step_start(
                "Creating Sire dynamics context for loch"
            )
            dynamics = gcmc_system.dynamics(
                integrator="langevin_middle",
                temperature=f"{self.temperature_K:.6f} K",
                pressure=None,
                cutoff_type=cutoff_type,
                cutoff=cutoff,
                constraint=constraint,
                timestep=f"{self.timestep_fs:.6f} fs",
                platform=self.platform,
                device=self.device,
                restraints=loch_restraints,
            )
            self._log_step_done(
                "Creating Sire dynamics context for loch",
                dynamics_started_at,
            )

            bind_started_at = self._log_step_start(
                "Binding loch sampler to the Sire dynamics context"
            )
            sampler.bind_dynamics(dynamics)
            self._log_step_done(
                "Binding loch sampler to the Sire dynamics context",
                bind_started_at,
            )
            if released_loch_backend:
                rebuild_started_at = self._log_step_start(
                    "Rebuilding loch GPU backend after dynamics binding"
                )
                try:
                    self._rebuild_loch_backend_resources(sampler)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to rebuild loch GPU backend after creating "
                        f"the Sire/OpenMM dynamics context: {exc}"
                    ) from exc
                else:
                    self._log_step_done(
                        "Rebuilding loch GPU backend after dynamics binding",
                        rebuild_started_at,
                    )
                    logger.info(
                        "Rebuilt loch GPU backend after binding dynamics"
                    )
            elif released_loch_cuda_context:
                try:
                    self._push_loch_cuda_context(sampler)
                except Exception as exc:
                    logger.warning(
                        "Failed to restore loch PyCUDA context after "
                        "binding dynamics: %s",
                        exc,
                    )
                else:
                    logger.info(
                        "Restored loch PyCUDA context after binding dynamics"
                    )

            logger.info(
                "GCMC equilibration: %d cycles, %d attempts/cycle, %d MD steps/cycle",
                n_equil_cycles,
                moves_per_cycle,
                n_md_steps,
            )
            equil_started_at = time.perf_counter()
            equil_cycle_metrics: list[dict[str, float | int | str]] = []
            previous_counts = {"accepted": 0, "insertions": 0, "deletions": 0}
            for cycle in range(n_equil_cycles):
                cycle_started_at = time.perf_counter()
                if cycle == 0:
                    logger.info(
                        "Starting first loch equilibration move batch "
                        "(startup latency often appears here)"
                    )
                sampler.move(dynamics.context())
                if cycle == 0:
                    logger.info("First loch equilibration move batch returned")
                if not self.flexible_water:
                    dynamics.context().applyConstraints(1.0e-6)
                    if cycle == 0:
                        logger.info(
                            "Applied OpenMM constraints after first loch "
                            "equilibration move batch"
                        )
                if md_time_ps > 0.0:
                    dynamics.run(
                        md_time,
                        energy_frequency=energy_frequency,
                        frame_frequency="0 ps",
                    )
                sampler.write_ghost_residues()
                previous_counts, metrics = self._collect_cycle_metrics(
                    phase="equil",
                    cycle_index=cycle,
                    n_cycles=n_equil_cycles,
                    moves_per_cycle=moves_per_cycle,
                    sampler=sampler,
                    cycle_started_at=cycle_started_at,
                    phase_started_at=equil_started_at,
                    run_started_at=run_started_at,
                    previous_counts=previous_counts,
                )
                equil_cycle_metrics.append(metrics)
                self._log_cycle_metrics(metrics)

            equil_stats = {
                "n_accepted": sampler.num_accepted_moves(),
                "n_insertions": sampler.num_insertions(),
                "n_deletions": sampler.num_deletions(),
                "N_waters": sampler.num_waters(),
                "elapsed_s": time.perf_counter() - equil_started_at,
                "cycle_metrics": equil_cycle_metrics,
            }
            logger.info(
                "GCMC equilibration done: accepted=%d insertions=%d deletions=%d N=%d",
                equil_stats["n_accepted"],
                equil_stats["n_insertions"],
                equil_stats["n_deletions"],
                equil_stats["N_waters"],
            )

            sampler.reset()

            logger.info(
                "GCMC production: %d cycles, %d attempts/cycle, %d MD steps/cycle",
                n_prod_cycles,
                moves_per_cycle,
                n_md_steps,
            )
            prod_started_at = time.perf_counter()
            prod_cycle_metrics: list[dict[str, float | int | str]] = []
            previous_counts = {"accepted": 0, "insertions": 0, "deletions": 0}
            for cycle in range(n_prod_cycles):
                cycle_started_at = time.perf_counter()
                sampler.move(dynamics.context())
                if not self.flexible_water:
                    dynamics.context().applyConstraints(1.0e-6)
                    if cycle == 0:
                        logger.info(
                            "Applied OpenMM constraints after first loch "
                            "production move batch"
                        )
                if md_time_ps > 0.0:
                    dynamics.run(
                        md_time,
                        energy_frequency=energy_frequency,
                        frame_frequency="0 ps",
                    )
                sampler.write_ghost_residues()
                previous_counts, metrics = self._collect_cycle_metrics(
                    phase="prod",
                    cycle_index=cycle,
                    n_cycles=n_prod_cycles,
                    moves_per_cycle=moves_per_cycle,
                    sampler=sampler,
                    cycle_started_at=cycle_started_at,
                    phase_started_at=prod_started_at,
                    run_started_at=run_started_at,
                    previous_counts=previous_counts,
                )
                prod_cycle_metrics.append(metrics)
                self._log_cycle_metrics(metrics)

            total_elapsed_s = time.perf_counter() - run_started_at
            prod_elapsed_s = time.perf_counter() - prod_started_at

            final_loch_state = dynamics.context().getState(getEnergy=True)
            logger.info(
                "Final loch context PE before handoff: %s",
                final_loch_state.getPotentialEnergy(),
            )
            final_mols = dynamics.commit()
            final_ghosts = sampler.ghost_residues()
            full_positions_wrapped, final_box_vectors = self._wrapped_openmm_positions(
                dynamics.context()
            )
            full_openmm_topology = self._topology_from_sire_system(
                sr,
                final_mols,
                temp_dir,
                filename="gcmc_full_system.prmtop",
            )
            ghost_free_topology, ghost_free_positions, n_removed = (
                self._remove_ghost_waters_openmm(
                    full_openmm_topology,
                    full_positions_wrapped,
                    final_box_vectors,
                    final_ghosts,
                )
            )
            if n_removed:
                logger.info(
                    "Removing %d final ghost waters before downstream MD",
                    n_removed,
                )

            tracking_reference_indices = _parse_atomnum_reference_indices(
                reference_selection
            )
            stats = {
                "backend": "loch",
                "reference_selection": reference_selection,
                "tracking_reference_indices": tracking_reference_indices,
                "reference_indices": (
                    None
                    if getattr(sampler, "_reference_indices", None) is None
                    else np.asarray(sampler._reference_indices, dtype=np.int32).tolist()
                ),
                "n_accepted": sampler.num_accepted_moves(),
                "n_moves": n_prod_cycles * moves_per_cycle,
                "n_move_batches": n_prod_cycles,
                "n_insertions": sampler.num_insertions(),
                "n_deletions": sampler.num_deletions(),
                "N_waters": sampler.num_waters(),
                "n_equil_cycles": n_equil_cycles,
                "n_prod_cycles": n_prod_cycles,
                "equil_stats": equil_stats,
                "prod_cycle_metrics": prod_cycle_metrics,
                "num_attempts_per_cycle": moves_per_cycle,
                "batch_size": batch_size,
                "n_final_ghosts_removed": n_removed,
                "timing": {
                    "equilibration_time_s": equil_stats["elapsed_s"],
                    "production_time_s": prod_elapsed_s,
                    "total_time_s": total_elapsed_s,
                    "production_attempts_per_s": self._safe_rate(
                        n_prod_cycles * moves_per_cycle, prod_elapsed_s
                    ),
                    "overall_attempts_per_s": self._safe_rate(
                        n_gcmc_moves, total_elapsed_s
                    ),
                },
            }

        logger.info(
            "GCMC (loch) complete: accepted=%d insertions=%d deletions=%d N=%d "
            "prod_rate=%.1f attempts/s overall_rate=%.1f attempts/s",
            stats["n_accepted"],
            stats["n_insertions"],
            stats["n_deletions"],
            stats["N_waters"],
            stats["timing"]["production_attempts_per_s"],
            stats["timing"]["overall_attempts_per_s"],
        )
        return ghost_free_topology, ghost_free_positions, stats

    def _run_simplified(self, n_gcmc_moves, n_md_steps, n_cycles) -> tuple:
        """Simplified GCMC: NVT MD only (no actual insertion/deletion)."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        logger.info("Running simplified pocket equilibration (no GCMC moves)")

        T = self.temperature_K * unit.kelvin
        integrator = openmm.LangevinMiddleIntegrator(
            T, 10.0 / unit.picoseconds, 0.002 * unit.picoseconds
        )

        simulation = self._create_simulation(
            app, self.topology, self.system, integrator, openmm
        )
        simulation.context.setPositions(self.positions)
        simulation.context.setVelocitiesToTemperature(T)

        total_steps = n_md_steps * n_cycles
        report_interval = max(1, total_steps // 10)
        report_path = self.output_dir / "gcmc_equil.csv"
        simulation.reporters.append(
            app.StateDataReporter(
                str(report_path),
                report_interval,
                step=True,
                temperature=True,
                potentialEnergy=True,
            )
        )

        logger.info("  Running %d steps of NVT equilibration", total_steps)
        simulation.step(total_steps)

        final_state = simulation.context.getState(getPositions=True)
        stats = {
            "backend": "simplified",
            "n_accepted": 0,
            "n_moves": 0,
            "N_waters": 0,
            "n_equil_cycles": 0,
            "n_prod_cycles": n_cycles,
            "mode": "simplified",
        }

        return self.topology, final_state.getPositions(), stats
