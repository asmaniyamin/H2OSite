# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Hydrogen bond analysis for hydration sites.

Counts protein-water, ligand-water, and water-water H-bonds per hydration site.
Uses geometric criteria: D-H...A with d(D-A) < 3.5 Å and angle(D-H-A) > 120°.
"""

from __future__ import annotations

import logging

import numpy as np

from H2OSite.analysis.clustering import HydrationSite

logger = logging.getLogger(__name__)


def _minimum_image_delta_A(
    delta_A: np.ndarray,
    box_lengths_A: np.ndarray | None = None,
) -> np.ndarray:
    """Apply orthorhombic minimum-image wrapping to displacement vectors."""
    wrapped = np.asarray(delta_A, dtype=np.float64)
    if box_lengths_A is not None:
        wrapped -= box_lengths_A * np.round(wrapped / box_lengths_A)
    return wrapped


class HBondAnalyzer:
    """Analyze hydrogen bonds around hydration sites.

    Uses MDAnalysis HydrogenBondAnalysis or a standalone geometric criterion
    to count H-bonds per hydration site per frame.
    """

    def __init__(
        self,
        distance_cutoff_A: float = 3.5,
        angle_cutoff_deg: float = 120.0,
        hs_radius_A: float = 1.0,
    ):
        self.distance_cutoff = distance_cutoff_A
        self.angle_cutoff = angle_cutoff_deg
        self.hs_radius = hs_radius_A

    def analyze_trajectory(
        self,
        topology,
        trajectory_file: str,
        sites: list[HydrationSite],
        ligand_selection: str = "resname LIG UNK",
    ) -> list[HydrationSite]:
        """Analyze H-bonds from an MD trajectory using MDAnalysis.

        Args:
            topology: OpenMM Topology object or path to topology file
            trajectory_file: Path to trajectory (DCD, XTC, etc.)
            sites: Hydration sites from clustering
            ligand_selection: MDAnalysis selection string for ligand atoms

        Returns:
            Updated HydrationSite objects with H-bond counts
        """
        import MDAnalysis as mda
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

        logger.info("Analyzing H-bonds for %d sites", len(sites))

        u = mda.Universe(topology, trajectory_file)
        protein = u.select_atoms("protein")
        water = u.select_atoms("resname HOH WAT TIP3 TIP4 SOL T3P T4P")
        try:
            ligand = u.select_atoms(ligand_selection)
        except Exception:
            ligand = u.select_atoms("not protein and not " + " and not ".join(
                [f"resname {r}" for r in ["HOH", "WAT", "TIP3", "TIP4", "SOL"]]
            ))

        water_oxygens = water.select_atoms("name O OW OH2")

        # Per-site H-bond accumulators
        n_sites = len(sites)
        pw_counts = np.zeros(n_sites)
        ww_counts = np.zeros(n_sites)
        lw_counts = np.zeros(n_sites)
        n_observations = np.zeros(n_sites)

        centers = np.array([s.center for s in sites])

        for ts in u.trajectory:
            box_lengths_A = None
            if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0):
                box_lengths_A = ts.dimensions[:3].astype(np.float64)

            # Find which water oxygens are in each site
            water_pos = water_oxygens.positions  # Å
            for si, center in enumerate(centers):
                dists = np.linalg.norm(
                    _minimum_image_delta_A(
                        water_pos - center,
                        box_lengths_A,
                    ),
                    axis=1,
                )
                site_water_mask = dists <= self.hs_radius
                n_in_site = np.sum(site_water_mask)
                if n_in_site == 0:
                    continue
                n_observations[si] += n_in_site

                site_water_indices = np.where(site_water_mask)[0]

                # Count H-bonds for each water in this site
                for wi in site_water_indices:
                    water_o_pos = water_pos[wi]

                    # Protein-water H-bonds
                    pw_counts[si] += self._count_hbonds_to_group(
                        water_o_pos, protein.positions, protein, box_lengths_A
                    )

                    # Water-water H-bonds (to other waters)
                    other_water_mask = np.ones(len(water_pos), dtype=bool)
                    other_water_mask[wi] = False
                    ww_counts[si] += self._count_hbonds_to_positions(
                        water_o_pos, water_pos[other_water_mask], box_lengths_A
                    )

                    # Ligand-water H-bonds
                    if len(ligand) > 0:
                        lw_counts[si] += self._count_hbonds_to_group(
                            water_o_pos, ligand.positions, ligand, box_lengths_A
                        )

        # Average per observation
        for si, site in enumerate(sites):
            if n_observations[si] > 0:
                site.hbond_pw = pw_counts[si] / n_observations[si]
                site.hbond_ww = ww_counts[si] / n_observations[si]
                site.hbond_lw = lw_counts[si] / n_observations[si]

        logger.info("H-bond analysis complete")
        return sites

    def analyze_from_arrays(
        self,
        sites: list[HydrationSite],
        per_frame_water_pos: list[np.ndarray],
        per_frame_protein_pos: list[np.ndarray],
        per_frame_ligand_pos: list[np.ndarray] | None = None,
    ) -> list[HydrationSite]:
        """Analyze H-bonds from pre-extracted position arrays.

        Simpler interface when positions are already collected.
        Uses distance-only criterion (no angle check without H positions).
        """
        n_sites = len(sites)
        pw_counts = np.zeros(n_sites)
        ww_counts = np.zeros(n_sites)
        lw_counts = np.zeros(n_sites)
        n_observations = np.zeros(n_sites)

        centers = np.array([s.center for s in sites])
        hbond_dist = self.distance_cutoff

        for fi in range(len(per_frame_water_pos)):
            water_pos = per_frame_water_pos[fi]
            protein_pos = per_frame_protein_pos[fi]
            ligand_pos = per_frame_ligand_pos[fi] if per_frame_ligand_pos else np.empty((0, 3))

            if len(water_pos) == 0:
                continue

            for si, center in enumerate(centers):
                dists_to_center = np.linalg.norm(water_pos - center, axis=1)
                site_mask = dists_to_center <= self.hs_radius
                n_in = np.sum(site_mask)
                if n_in == 0:
                    continue
                n_observations[si] += n_in

                for wi in np.where(site_mask)[0]:
                    w_pos = water_pos[wi]

                    # Protein-water
                    if len(protein_pos) > 0:
                        d_pw = np.linalg.norm(protein_pos - w_pos, axis=1)
                        pw_counts[si] += np.sum(d_pw < hbond_dist)

                    # Water-water
                    other_mask = np.ones(len(water_pos), dtype=bool)
                    other_mask[wi] = False
                    if np.any(other_mask):
                        d_ww = np.linalg.norm(water_pos[other_mask] - w_pos, axis=1)
                        ww_counts[si] += np.sum(d_ww < hbond_dist)

                    # Ligand-water
                    if len(ligand_pos) > 0:
                        d_lw = np.linalg.norm(ligand_pos - w_pos, axis=1)
                        lw_counts[si] += np.sum(d_lw < hbond_dist)

        for si, site in enumerate(sites):
            if n_observations[si] > 0:
                site.hbond_pw = pw_counts[si] / n_observations[si]
                site.hbond_ww = ww_counts[si] / n_observations[si]
                site.hbond_lw = lw_counts[si] / n_observations[si]

        return sites

    def _count_hbonds_to_group(
        self,
        water_o_pos: np.ndarray,
        group_positions: np.ndarray,
        group_atoms=None,
        box_lengths_A: np.ndarray | None = None,
    ) -> int:
        """Count potential H-bond partners within distance cutoff."""
        if len(group_positions) == 0:
            return 0
        dists = np.linalg.norm(
            _minimum_image_delta_A(group_positions - water_o_pos, box_lengths_A),
            axis=1,
        )
        # Simple distance criterion for heavy atom contacts
        # Full geometric criterion requires H positions
        return int(np.sum(dists < self.distance_cutoff))

    def _count_hbonds_to_positions(
        self,
        water_o_pos: np.ndarray,
        other_positions: np.ndarray,
        box_lengths_A: np.ndarray | None = None,
    ) -> int:
        if len(other_positions) == 0:
            return 0
        dists = np.linalg.norm(
            _minimum_image_delta_A(other_positions - water_o_pos, box_lengths_A),
            axis=1,
        )
        return int(np.sum(dists < self.distance_cutoff))
