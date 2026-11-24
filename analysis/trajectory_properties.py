# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Post-hoc per-water trajectory properties for Stage 11b.

Computes water orientation vectors and protein-water/water-water energies from
saved production trajectories and the OpenMM system definition. Reduced dipole
vectors use OH1 + OH2; energies use the production direct-space nonbonded model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KJ_TO_KCAL = 0.2390057361376673
ONE_4PI_EPS0 = 138.935456
_WATER_RESNAMES = {"HOH", "WAT", "TIP4", "SOL", "T4P", "T3P"}


def _find_nonbonded_force(system):
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if force.__class__.__name__ == "NonbondedForce":
            return force
    raise RuntimeError("System does not contain a NonbondedForce")


@dataclass(frozen=True)
class WaterResidueRecord:
    resid: int
    oxygen_index: int
    hydrogen_indices: tuple[int, ...]
    atom_indices: tuple[int, ...]


class ProductionTrajectoryAnalyzer:
    """Extract per-water dipoles and PW/WW energies from production MD."""

    def __init__(
        self,
        topology,
        system,
        trajectory_file: str | Path,
        checkpoint_file: str | Path | None = None,
        shell_margin_A: float = 5.0,
    ):
        import openmm.unit as unit

        self.topology = topology
        self.system = system
        self.trajectory_file = Path(trajectory_file)
        self.checkpoint_file = (
            Path(checkpoint_file) if checkpoint_file is not None else None
        )
        self.shell_margin_A = shell_margin_A

        nonbonded = _find_nonbonded_force(system)
        self.cutoff_nm = nonbonded.getCutoffDistance().value_in_unit(unit.nanometer)
        self.cutoff_A = self.cutoff_nm * 10.0
        self.switch_nm = (
            nonbonded.getSwitchingDistance().value_in_unit(unit.nanometer)
            if nonbonded.getUseSwitchingFunction()
            else None
        )
        self.use_switch = self.switch_nm is not None

        n_particles = system.getNumParticles()
        self.charges = np.zeros(n_particles, dtype=np.float64)
        self.sigmas_nm = np.zeros(n_particles, dtype=np.float64)
        self.epsilons_kj = np.zeros(n_particles, dtype=np.float64)
        for atom_idx in range(n_particles):
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom_idx)
            self.charges[atom_idx] = charge.value_in_unit(unit.elementary_charge)
            self.sigmas_nm[atom_idx] = sigma.value_in_unit(unit.nanometer)
            self.epsilons_kj[atom_idx] = epsilon.value_in_unit(
                unit.kilojoule_per_mole
            )

        self.exceptions: dict[tuple[int, int], tuple[float, float, float]] = {}
        for exception_idx in range(nonbonded.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(
                exception_idx
            )
            self.exceptions[(min(i, j), max(i, j))] = (
                chargeprod.value_in_unit(unit.elementary_charge**2),
                sigma.value_in_unit(unit.nanometer),
                epsilon.value_in_unit(unit.kilojoule_per_mole),
            )

        self.water_records: dict[int, WaterResidueRecord] = {}
        water_atom_set: set[int] = set()
        for resid, residue in enumerate(topology.residues()):
            if residue.name not in _WATER_RESNAMES:
                continue

            atom_indices = tuple(atom.index for atom in residue.atoms())
            oxygen_index = None
            hydrogen_indices: list[int] = []
            for atom in residue.atoms():
                if oxygen_index is None and (
                    (atom.element is not None and atom.element.symbol == "O")
                    or atom.name in {"O", "OW", "OH2"}
                ):
                    oxygen_index = atom.index
                if (atom.element is not None and atom.element.symbol == "H") or (
                    atom.name.startswith("H")
                ):
                    hydrogen_indices.append(atom.index)

            if oxygen_index is None:
                continue

            self.water_records[resid] = WaterResidueRecord(
                resid=resid,
                oxygen_index=oxygen_index,
                hydrogen_indices=tuple(hydrogen_indices[:2]),
                atom_indices=atom_indices,
            )
            water_atom_set.update(atom_indices)

        self.water_resids = np.array(sorted(self.water_records), dtype=int)
        self.water_oxygen_indices = np.array(
            [self.water_records[resid].oxygen_index for resid in self.water_resids],
            dtype=int,
        )
        self.non_water_atom_indices = np.array(
            [idx for idx in range(n_particles) if idx not in water_atom_set],
            dtype=int,
        )

    def compute_tracked_water_properties(
        self,
        tracked_water_resids: list[np.ndarray],
        *,
        active_site_center_A: np.ndarray,
        active_site_radius_A: float,
    ) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
        """Compute per-frame dipoles and PW/WW energies for tracked waters."""
        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise RuntimeError(
                "MDAnalysis is required for Stage 11b trajectory replay"
            ) from exc

        if not self.trajectory_file.exists():
            raise RuntimeError(
                "Production DCD file is required for Stage 11b replay"
            )

        # Build Universe directly from the OpenMM Topology object (avoids
        # PDB round-trip and hybrid-36 resSeq overflow for >9999 residues).
        universe = mda.Universe(self.topology, str(self.trajectory_file))
        n_target_frames = len(tracked_water_resids)

        per_frame_dipoles: list[np.ndarray] = []
        per_frame_energies = {"pw": [], "ww": []}
        shell_indices: np.ndarray | None = None

        for fi, ts in enumerate(universe.trajectory):
            if fi >= n_target_frames:
                break

            positions_A = universe.atoms.positions.astype(np.float64, copy=True)
            box_A = (
                ts.dimensions[:3].astype(np.float64)
                if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0)
                else None
            )
            frame_resids = np.asarray(tracked_water_resids[fi], dtype=int)

            if shell_indices is None:
                shell_indices = self._select_non_water_shell(
                    positions_A, active_site_center_A, active_site_radius_A, box_A
                )

            per_frame_dipoles.append(
                self._frame_dipoles(frame_resids, positions_A, box_A)
            )
            pw, ww = self._frame_energies(
                frame_resids,
                positions_A,
                box_A,
                shell_indices,
                active_site_center_A,
                active_site_radius_A,
            )
            per_frame_energies["pw"].append(pw)
            per_frame_energies["ww"].append(ww)

        if len(per_frame_dipoles) < n_target_frames and self.checkpoint_file:
            self._append_checkpoint_frame(
                tracked_water_resids,
                per_frame_dipoles,
                per_frame_energies,
                active_site_center_A,
                active_site_radius_A,
                shell_indices,
            )

        while len(per_frame_dipoles) < n_target_frames:
            per_frame_dipoles.append(np.empty((0, 3), dtype=np.float64))
            per_frame_energies["pw"].append(np.empty(0, dtype=np.float64))
            per_frame_energies["ww"].append(np.empty(0, dtype=np.float64))

        logger.info(
            "Stage 11b replay complete for %d production frames",
            len(per_frame_dipoles),
        )
        return per_frame_dipoles, per_frame_energies

    def _append_checkpoint_frame(
        self,
        tracked_water_resids: list[np.ndarray],
        per_frame_dipoles: list[np.ndarray],
        per_frame_energies: dict[str, list[np.ndarray]],
        active_site_center_A: np.ndarray,
        active_site_radius_A: float,
        shell_indices: np.ndarray | None,
    ) -> None:
        if not self.checkpoint_file or not self.checkpoint_file.exists():
            return
        if len(per_frame_dipoles) >= len(tracked_water_resids):
            return

        ckpt = np.load(self.checkpoint_file)
        positions_A = np.asarray(ckpt["positions_nm"], dtype=np.float64) * 10.0
        box_A = None
        if "box_vectors_nm" in ckpt:
            box_vectors_nm = np.asarray(ckpt["box_vectors_nm"], dtype=np.float64)
            box_A = np.array(
                [box_vectors_nm[0, 0], box_vectors_nm[1, 1], box_vectors_nm[2, 2]],
                dtype=np.float64,
            ) * 10.0

        if shell_indices is None:
            shell_indices = self._select_non_water_shell(
                positions_A, active_site_center_A, active_site_radius_A, box_A
            )

        for fi in range(len(per_frame_dipoles), len(tracked_water_resids)):
            frame_resids = np.asarray(tracked_water_resids[fi], dtype=int)
            per_frame_dipoles.append(
                self._frame_dipoles(frame_resids, positions_A, box_A)
            )
            pw, ww = self._frame_energies(
                frame_resids,
                positions_A,
                box_A,
                shell_indices,
                active_site_center_A,
                active_site_radius_A,
            )
            per_frame_energies["pw"].append(pw)
            per_frame_energies["ww"].append(ww)

    def _select_non_water_shell(
        self,
        positions_A: np.ndarray,
        active_site_center_A: np.ndarray,
        active_site_radius_A: float,
        box_A: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(self.non_water_atom_indices) == 0:
            return self.non_water_atom_indices

        shell_radius_A = active_site_radius_A + self.cutoff_A + self.shell_margin_A
        delta = self._minimum_image_delta_A(
            positions_A[self.non_water_atom_indices] - active_site_center_A,
            box_A,
        )
        dists = np.linalg.norm(delta, axis=1)
        shell = self.non_water_atom_indices[dists <= shell_radius_A]
        return shell if len(shell) > 0 else self.non_water_atom_indices

    def _frame_dipoles(
        self,
        frame_resids: np.ndarray,
        positions_A: np.ndarray,
        box_A: np.ndarray | None,
    ) -> np.ndarray:
        dipoles = [
            self._reduced_dipole_vector(int(resid), positions_A, box_A)
            for resid in frame_resids
        ]
        if not dipoles:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(dipoles, dtype=np.float64)

    def _frame_energies(
        self,
        frame_resids: np.ndarray,
        positions_A: np.ndarray,
        box_A: np.ndarray | None,
        shell_indices: np.ndarray,
        active_site_center_A: np.ndarray,
        active_site_radius_A: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(frame_resids) == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty

        positions_nm = positions_A * 0.1
        box_nm = box_A * 0.1 if box_A is not None else None

        shell_radius_A = active_site_radius_A + self.cutoff_A + self.shell_margin_A
        water_oxygen_pos_A = positions_A[self.water_oxygen_indices]
        water_shell_mask = np.linalg.norm(
            self._minimum_image_delta_A(
                water_oxygen_pos_A - active_site_center_A,
                box_A,
            ),
            axis=1,
        ) <= shell_radius_A
        shell_water_resids = self.water_resids[water_shell_mask]

        pw_values = []
        ww_values = []
        for resid in frame_resids:
            record = self.water_records.get(int(resid))
            if record is None:
                pw_values.append(0.0)
                ww_values.append(0.0)
                continue

            pw_values.append(
                self._interaction_energy_kcal(
                    record.atom_indices,
                    shell_indices,
                    positions_nm,
                    box_nm,
                )
            )

            neighbour_atoms = [
                atom_idx
                for other_resid in shell_water_resids
                if int(other_resid) != int(resid)
                for atom_idx in self.water_records[int(other_resid)].atom_indices
            ]
            ww_values.append(
                self._interaction_energy_kcal(
                    record.atom_indices,
                    np.asarray(neighbour_atoms, dtype=int),
                    positions_nm,
                    box_nm,
                )
            )

        return np.asarray(pw_values, dtype=np.float64), np.asarray(
            ww_values, dtype=np.float64
        )

    def _reduced_dipole_vector(
        self,
        resid: int,
        positions_A: np.ndarray,
        box_A: np.ndarray | None,
    ) -> np.ndarray:
        record = self.water_records.get(resid)
        if record is None or len(record.hydrogen_indices) < 2:
            return np.zeros(3, dtype=np.float64)

        oxygen = positions_A[record.oxygen_index]
        oh1 = self._minimum_image_delta_A(
            positions_A[record.hydrogen_indices[0]] - oxygen, box_A
        )
        oh2 = self._minimum_image_delta_A(
            positions_A[record.hydrogen_indices[1]] - oxygen, box_A
        )
        dipole = oh1 + oh2
        norm_sq = float(np.inner(dipole, dipole))
        if norm_sq <= 1e-12:
            return np.zeros(3, dtype=np.float64)
        return dipole / norm_sq

    def _interaction_energy_kcal(
        self,
        atom_indices_a,
        atom_indices_b,
        positions_nm: np.ndarray,
        box_nm: np.ndarray | None,
    ) -> float:
        atom_indices_b = np.asarray(atom_indices_b, dtype=int)
        if len(atom_indices_b) == 0:
            return 0.0

        total_kj = 0.0
        for atom_i in atom_indices_a:
            deltas = positions_nm[atom_indices_b] - positions_nm[atom_i]
            if box_nm is not None:
                deltas -= box_nm * np.round(deltas / box_nm)
            distances = np.linalg.norm(deltas, axis=1)
            valid = (distances > 1e-12) & (distances < self.cutoff_nm)
            if not np.any(valid):
                continue

            partner_indices = atom_indices_b[valid]
            r = distances[valid]
            sigma = 0.5 * (self.sigmas_nm[atom_i] + self.sigmas_nm[partner_indices])
            epsilon = np.sqrt(
                self.epsilons_kj[atom_i] * self.epsilons_kj[partner_indices]
            )
            chargeprod = self.charges[atom_i] * self.charges[partner_indices]

            for idx, atom_j in enumerate(partner_indices):
                ex = self.exceptions.get((min(atom_i, int(atom_j)), max(atom_i, int(atom_j))))
                if ex is not None:
                    chargeprod[idx], sigma[idx], epsilon[idx] = ex

            coulomb_kj = ONE_4PI_EPS0 * chargeprod / r

            lj_kj = np.zeros_like(r)
            nonzero = (sigma > 0.0) & (epsilon > 0.0)
            if np.any(nonzero):
                sr6 = np.power(sigma[nonzero] / r[nonzero], 6)
                lj_kj[nonzero] = 4.0 * epsilon[nonzero] * (sr6 * (sr6 - 1.0))
                if (
                    self.use_switch
                    and self.switch_nm is not None
                    and self.cutoff_nm > self.switch_nm
                ):
                    switch_mask = r[nonzero] > self.switch_nm
                    if np.any(switch_mask):
                        t = (
                            (r[nonzero][switch_mask] - self.switch_nm)
                            / (self.cutoff_nm - self.switch_nm)
                        )
                        switch = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
                        lj_part = lj_kj[nonzero]
                        lj_part[switch_mask] *= switch
                        lj_kj[nonzero] = lj_part

            total_kj += float(np.sum(coulomb_kj + lj_kj))

        return total_kj * KJ_TO_KCAL

    @staticmethod
    def _minimum_image_delta_A(
        delta_A: np.ndarray,
        box_A: np.ndarray | None,
    ) -> np.ndarray:
        if box_A is None:
            return delta_A
        wrapped = np.asarray(delta_A, dtype=np.float64).copy()
        wrapped -= box_A * np.round(wrapped / box_A)
        return wrapped
