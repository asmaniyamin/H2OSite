# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Ligand displacement scoring against H2OSite hydration sites.

For each hydration site, compute the fraction displaced by ligand atoms,
then weight by site thermodynamic properties to produce dG, dH, and -TdS scores.

Reference: Abel et al., JACS 2008; Young et al., PNAS 2007.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from H2OSite.analysis.clustering import HydrationSite

logger = logging.getLogger(__name__)

# Van der Waals radii (Å) for common elements
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}


@dataclass
class DisplacementResult:
    """Per-hydration-site displacement scores."""

    site_index: int
    overlap: float  # fraction of site displaced by ligand (0-1)
    dH: float  # enthalpy contribution (kcal/mol)
    minus_TdS: float  # entropy contribution (kcal/mol)
    dG: float  # free energy contribution (kcal/mol)


class DisplacementScorer:
    """Score ligand poses by hydration site displacement.

    For each hydration site, the overlap with the ligand determines
    what fraction of the site's thermodynamic penalty is "recovered"
    by the ligand displacing the water.
    """

    def __init__(self, scale_factor: float = 0.8):
        self.scale_factor = scale_factor

    def score(
        self,
        sites: list[HydrationSite],
        ligand_positions: np.ndarray,
        ligand_elements: list[str],
        hs_radius_A: float = 1.0,
    ) -> list[DisplacementResult]:
        """Score ligand against hydration sites.

        Args:
            sites: Hydration sites with computed thermodynamic properties
            ligand_positions: (N, 3) Å ligand atom positions
            ligand_elements: Element symbols for ligand atoms
            hs_radius_A: Hydration site radius

        Returns:
            Per-site DisplacementResult
        """
        logger.info(
            "Scoring %d sites against ligand with %d atoms",
            len(sites), len(ligand_positions),
        )

        results = []
        ligand_tree = cKDTree(ligand_positions)
        ligand_radii = np.array([
            VDW_RADII.get(e, 1.70) for e in ligand_elements
        ])

        for site in sites:
            overlap = self._compute_overlap(
                site.center, hs_radius_A,
                ligand_positions, ligand_radii, ligand_tree,
            )

            # Thermodynamic contributions scaled by overlap
            dH = site.relative_enthalpy * overlap
            minus_TdS = site.entropy * overlap
            dG = dH + minus_TdS

            results.append(DisplacementResult(
                site_index=site.index,
                overlap=overlap,
                dH=dH,
                minus_TdS=minus_TdS,
                dG=dG,
            ))

        total_dG = sum(r.dG for r in results)
        logger.info("Total displacement score: dG = %.2f kcal/mol", total_dG)

        return results

    def _compute_overlap(
        self,
        site_center: np.ndarray,
        site_radius: float,
        ligand_positions: np.ndarray,
        ligand_radii: np.ndarray,
        ligand_tree: cKDTree,
    ) -> float:
        """Compute fractional overlap between a hydration site sphere and ligand atoms.

        Uses Monte Carlo sampling of the hydration site sphere to estimate
        the fraction of volume that overlaps with ligand VdW spheres.
        Scaled by self.scale_factor.
        """
        # Find ligand atoms near the site
        max_radius = ligand_radii.max() + site_radius
        nearby_indices = ligand_tree.query_ball_point(site_center, max_radius)

        if not nearby_indices:
            return 0.0

        nearby_pos = ligand_positions[nearby_indices]
        nearby_radii = ligand_radii[nearby_indices] * self.scale_factor

        # Analytical approximation: fraction of site sphere overlapping
        # with nearest ligand atom sphere
        dists = np.linalg.norm(nearby_pos - site_center, axis=1)

        total_overlap = 0.0
        for d, r_lig in zip(dists, nearby_radii):
            overlap = self._sphere_overlap_fraction(site_radius, r_lig, d)
            total_overlap = max(total_overlap, overlap)  # Take max, not sum

        return min(1.0, total_overlap)

    @staticmethod
    def _sphere_overlap_fraction(r1: float, r2: float, d: float) -> float:
        """Fraction of sphere 1 overlapping with sphere 2 at distance d.

        Based on the volume of the lens-shaped intersection of two spheres.
        """
        if d >= r1 + r2:
            return 0.0
        if d + r1 <= r2:
            return 1.0  # Sphere 1 entirely inside sphere 2
        if d + r2 <= r1:
            # Sphere 2 entirely inside sphere 1
            return (r2 / r1) ** 3

        # Lens volume formula
        v1 = r1**3
        overlap_volume = (
            np.pi * (r1 + r2 - d) ** 2
            * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2)
            / (12 * d)
        )

        sphere1_volume = (4.0 / 3.0) * np.pi * v1
        if sphere1_volume <= 0:
            return 0.0

        return min(1.0, overlap_volume / sphere1_volume)

    def score_multiple_ligands(
        self,
        sites: list[HydrationSite],
        ligand_list: list[tuple[np.ndarray, list[str]]],
        hs_radius_A: float = 1.0,
    ) -> list[list[DisplacementResult]]:
        """Score multiple ligand poses against the same hydration sites."""
        return [
            self.score(sites, pos, elem, hs_radius_A)
            for pos, elem in ligand_list
        ]
