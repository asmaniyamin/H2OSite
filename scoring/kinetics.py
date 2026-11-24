# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
Kinetics classification of displaced hydration sites.

Classifies hydration sites displaced by a ligand into kinetic categories
per Pearlstein et al. (2010, 2013).

Categories:
  - Unfavorable (dH > 0): correlates with koff (dissociation rate)
  - Enthalpically favorable I (dH < 0, dG > 0): correlates with kon
  - Enthalpically favorable II (dH < 0, dG < 0): correlates with kon

Polar atom correction: if a favorable site is displaced by a polar atom,
its enthalpy contribution is set to 0 (assumed full H-bond replacement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from H2OSite.analysis.clustering import HydrationSite
from H2OSite.scoring.displacement import DisplacementResult, VDW_RADII

logger = logging.getLogger(__name__)

POLAR_ELEMENTS = {"N", "O", "S", "F", "Cl", "Br"}


@dataclass
class KineticsResult:
    """Kinetics classification for a ligand."""

    # Summed properties over displaced sites
    dG_dissoc: float  # dG_UF + (-TdS_F) - dissociation proxy
    dH_unfavorable: float  # sum of dH for unfavorable sites
    dG_unfavorable: float  # sum of dG for unfavorable sites

    # Per-category breakdown
    unfavorable_sites: list[int]  # site indices
    favorable_I_sites: list[int]  # dH<0, dG>0
    favorable_II_sites: list[int]  # dH<0, dG<0

    # Per-site details
    per_site: dict  # site_index -> {category, dH, dG, TdS, polar_corrected}


class KineticsClassifier:
    """Classify displaced hydration sites for kinetics prediction."""

    def __init__(self, overlap_threshold: float = 0.5):
        self.overlap_threshold = overlap_threshold

    def classify(
        self,
        sites: list[HydrationSite],
        displacement_results: list[DisplacementResult],
        ligand_positions: np.ndarray,
        ligand_elements: list[str],
    ) -> KineticsResult:
        """Classify displaced sites into kinetic categories.

        Args:
            sites: Hydration sites with thermodynamic data
            displacement_results: From DisplacementScorer.score()
            ligand_positions: Ligand atom positions (Å)
            ligand_elements: Ligand element symbols

        Returns:
            KineticsResult with classification and summed properties
        """
        # Identify displaced sites (overlap > threshold)
        displaced = [
            (r, sites[r.site_index])
            for r in displacement_results
            if r.overlap >= self.overlap_threshold
        ]

        logger.info("%d sites displaced (overlap >= %.2f)",
                     len(displaced), self.overlap_threshold)

        unfavorable = []
        favorable_I = []
        favorable_II = []
        per_site = {}

        ligand_tree = cKDTree(ligand_positions)

        dH_uf_sum = 0.0
        dG_uf_sum = 0.0
        TdS_f_sum = 0.0

        for result, site in displaced:
            dH = site.relative_enthalpy
            minus_TdS = site.entropy
            dG = dH + minus_TdS

            # Check if displaced by a polar atom
            polar_corrected = False
            dist, nearest_idx = ligand_tree.query(site.center)
            if nearest_idx < len(ligand_elements):
                nearest_element = ligand_elements[nearest_idx]
                is_polar = nearest_element in POLAR_ELEMENTS
            else:
                is_polar = False

            if dH > 0:
                # Unfavorable
                unfavorable.append(site.index)
                dH_uf_sum += dH
                dG_uf_sum += dG
                per_site[site.index] = {
                    "category": "unfavorable",
                    "dH": dH, "dG": dG, "TdS": -minus_TdS,
                    "polar_corrected": False,
                }
            elif dH < 0 and dG > 0:
                # Favorable I
                favorable_I.append(site.index)
                effective_dH = 0.0 if is_polar else dH
                polar_corrected = is_polar
                TdS_f_sum += -minus_TdS
                per_site[site.index] = {
                    "category": "favorable_I",
                    "dH": effective_dH, "dG": dG, "TdS": -minus_TdS,
                    "polar_corrected": polar_corrected,
                }
            else:
                # Favorable II (dH < 0, dG < 0)
                favorable_II.append(site.index)
                effective_dH = 0.0 if is_polar else dH
                polar_corrected = is_polar
                TdS_f_sum += -minus_TdS
                per_site[site.index] = {
                    "category": "favorable_II",
                    "dH": effective_dH, "dG": dG, "TdS": -minus_TdS,
                    "polar_corrected": polar_corrected,
                }

        dG_dissoc = dG_uf_sum + TdS_f_sum

        logger.info(
            "Kinetics: %d unfavorable, %d fav-I, %d fav-II, dG_dissoc=%.2f",
            len(unfavorable), len(favorable_I), len(favorable_II), dG_dissoc,
        )

        return KineticsResult(
            dG_dissoc=dG_dissoc,
            dH_unfavorable=dH_uf_sum,
            dG_unfavorable=dG_uf_sum,
            unfavorable_sites=unfavorable,
            favorable_I_sites=favorable_I,
            favorable_II_sites=favorable_II,
            per_site=per_site,
        )
