# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Thermodynamic calculations for hydration sites using IST.

Computes translational entropy from water positions, rotational entropy from
orientations, and enthalpy from protein-water and water-water interaction energies.
Entropies are stored in cal/(mol·K); HydrationSite.entropy converts to -TΔS in
kcal/mol using T/1000 = 0.300 at 300 K.

Reference: Young et al., PNAS 2007; Abel et al., JACS 2008.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from H2OSite.analysis.clustering import HydrationSite
from H2OSite.core.config import (
    BOLTZMANN_KCAL,
    KB_CAL_MOL_K,
    get_water_model_params,
)

logger = logging.getLogger(__name__)


class ThermodynamicCalculator:
    """Compute thermodynamic properties of hydration sites using IST.

    The key quantities for each hydration site:
      - Translational entropy: computed from the local water density
        distribution relative to bulk
      - Rotational entropy: computed from the orientation (dipole)
        distribution of water molecules
      - Enthalpy: computed from non-bonded interaction energies
        (protein-water and water-water) relative to bulk
    """

    def __init__(
        self,
        temperature_K: float = 300.0,
        hs_radius_A: float = 1.0,
        n_orientation_bins: int = 30,
        water_model: str = "tip4p",
    ):
        self.temperature = temperature_K
        self.hs_radius = hs_radius_A
        self.kT = BOLTZMANN_KCAL * temperature_K
        self.n_orient_bins = n_orientation_bins
        # Use water-model-specific bulk density.

        wm_params = get_water_model_params(water_model)
        self.bulk_density = wm_params["bulk_density"]
        self.water_model = water_model
        self.bulk_energy = wm_params["bulk_energy_kcal"]

    def compute_all(
        self,
        sites: list[HydrationSite],
        per_frame_positions: list[np.ndarray],
        per_frame_dipoles: list[np.ndarray] | None = None,
        per_frame_energies: dict | None = None,
    ) -> list[HydrationSite]:
        """Compute all thermodynamic properties for hydration sites.

        Args:
            sites: Hydration sites from clustering
            per_frame_positions: Per-frame water O positions in Å
            per_frame_dipoles: Per-frame water dipole vectors (optional)
            per_frame_energies: Dict with 'pw' and 'ww' per-frame energy arrays (optional)

        Returns:
            Updated list of HydrationSite objects
        """
        n_frames = len(per_frame_positions)
        logger.info(
            "Computing thermodynamics for %d sites over %d frames",
            len(sites), n_frames,
        )

        for site in sites:
            site.water_model = self.water_model
            site.bulk_water_energy_kcal = self.bulk_energy
            site.temperature_K = self.temperature
            site.properties["entropy_components_available"] = True
            # Translational entropy
            site.translational_entropy = self._translational_entropy(
                site, per_frame_positions, n_frames
            )

            # Rotational entropy
            if per_frame_dipoles is not None:
                site.rotational_entropy = self._rotational_entropy(
                    site, per_frame_positions, per_frame_dipoles, n_frames
                )

            # Enthalpy from energies
            if per_frame_energies is not None:
                self._compute_enthalpy(
                    site, per_frame_positions, per_frame_energies, n_frames
                )

            # Mean dipole
            if per_frame_dipoles is not None:
                site.dipole = self._mean_dipole(
                    site, per_frame_positions, per_frame_dipoles
                )

        return sites

    def _translational_entropy(
        self,
        site: HydrationSite,
        per_frame_positions: list[np.ndarray],
        n_frames: int,
    ) -> float:
        """Compute translational entropy using the density-based IST formula.

        Returns S_trans in cal/(mol·K). HydrationSite.entropy converts to
        kcal/mol using T/1000 at 300 K.

        S_trans = -kB * ln(ρ_local / ρ_bulk), where ρ_local is the average
        water density inside the site sphere and ρ_bulk is bulk density.
        """
        # Count waters in site per frame
        n_occupied = 0
        total_waters = 0
        for frame_pos in per_frame_positions:
            if len(frame_pos) == 0:
                continue
            dists = np.linalg.norm(frame_pos - site.center, axis=1)
            n_in = np.sum(dists <= self.hs_radius)
            if n_in > 0:
                n_occupied += 1
            total_waters += n_in

        if n_frames == 0 or total_waters == 0:
            return 0.0

        # Local density
        site_volume = (4.0 / 3.0) * np.pi * self.hs_radius**3  # ų
        local_density = total_waters / (n_frames * site_volume)

        # Density ratio - uses water-model-specific bulk density

        rho_ratio = local_density / self.bulk_density
        if rho_ratio <= 0:
            return 0.0

        # S_trans in cal/(mol·K).
        # kB in cal/(mol·K) = BOLTZMANN_KCAL * 1000 = 1.987204
        s_trans = -KB_CAL_MOL_K * np.log(rho_ratio)

        return s_trans

    def _rotational_entropy(
        self,
        site: HydrationSite,
        per_frame_positions: list[np.ndarray],
        per_frame_dipoles: list[np.ndarray],
        n_frames: int,
    ) -> float:
        """Compute rotational entropy from water dipole orientations.

        Returns S_rot in cal/(mol·K). HydrationSite.entropy converts to
        kcal/mol using T/1000 at 300 K.

        Uses the KL divergence from the uniform distribution on the sphere:
        S_rot = -kB * Σ_θ [p(θ) * ln(p(θ)/p_0(θ))] * Δcos(θ), where p_0 = 0.5.
        """
        # Collect dipoles of waters in this site
        dipoles = []
        for frame_pos, frame_dip in zip(per_frame_positions, per_frame_dipoles):
            if len(frame_pos) == 0:
                continue
            dists = np.linalg.norm(frame_pos - site.center, axis=1)
            mask = dists <= self.hs_radius
            if np.any(mask):
                dipoles.append(frame_dip[mask])

        if not dipoles:
            return 0.0

        all_dipoles = np.vstack(dipoles)
        n_samples = len(all_dipoles)
        if n_samples < 10:
            return 0.0

        # Normalize dipoles
        norms = np.linalg.norm(all_dipoles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_dipoles = all_dipoles / norms

        # Compute cos(θ) relative to mean dipole
        mean_dip = unit_dipoles.mean(axis=0)
        mean_norm = np.linalg.norm(mean_dip)
        if mean_norm < 1e-10:
            return 0.0  # Isotropic - no orientational preference

        mean_dip /= mean_norm
        cos_theta = unit_dipoles @ mean_dip
        cos_theta = np.clip(cos_theta, -1, 1)

        # Histogram of cos(θ) - uniform distribution gives flat histogram
        bins = np.linspace(-1, 1, self.n_orient_bins + 1)
        hist, _ = np.histogram(cos_theta, bins=bins, density=True)

        # Uniform distribution: p_0 = 1/2 for cos(θ) in [-1, 1]
        p_uniform = 0.5
        dcos = 2.0 / self.n_orient_bins

        # Dimensionless KL divergence (in kB units, negative for ordered)
        s_rot_kl = 0.0
        for p in hist:
            if p > 0:
                s_rot_kl -= p * np.log(p / p_uniform) * dcos

        # Return in cal/(mol·K) - matching r_wmap_r_ent convention.
        # kB [cal/(mol·K)] = 1.987204
        return KB_CAL_MOL_K * s_rot_kl

    def _compute_enthalpy(
        self,
        site: HydrationSite,
        per_frame_positions: list[np.ndarray],
        per_frame_energies: dict,
        n_frames: int,
    ) -> None:
        """Compute average protein-water and water-water interaction energies.

        Energy arrays in per_frame_energies should be keyed 'pw' and 'ww',
        each being a list of arrays matching per_frame_positions.
        """
        pw_energies = per_frame_energies.get("pw", [])
        ww_energies = per_frame_energies.get("ww", [])

        pw_sum, ww_sum = 0.0, 0.0
        n_obs = 0

        for fi, frame_pos in enumerate(per_frame_positions):
            if len(frame_pos) == 0:
                continue
            dists = np.linalg.norm(frame_pos - site.center, axis=1)
            mask = dists <= self.hs_radius
            n_in = np.sum(mask)
            if n_in == 0:
                continue

            if fi < len(pw_energies) and len(pw_energies[fi]) > 0:
                pw_sum += pw_energies[fi][mask].sum()
            if fi < len(ww_energies) and len(ww_energies[fi]) > 0:
                ww_sum += ww_energies[fi][mask].sum()
            n_obs += n_in

        if n_obs > 0:
            site.enthalpy_pw = pw_sum / n_obs
            site.enthalpy_ww = ww_sum / n_obs

    def _mean_dipole(
        self,
        site: HydrationSite,
        per_frame_positions: list[np.ndarray],
        per_frame_dipoles: list[np.ndarray],
    ) -> np.ndarray:
        """Compute the mean water dipole vector for a hydration site."""
        dipoles = []
        for frame_pos, frame_dip in zip(per_frame_positions, per_frame_dipoles):
            if len(frame_pos) == 0:
                continue
            dists = np.linalg.norm(frame_pos - site.center, axis=1)
            mask = dists <= self.hs_radius
            if np.any(mask):
                dipoles.append(frame_dip[mask])

        if not dipoles:
            return np.zeros(3)

        all_dipoles = np.vstack(dipoles)
        return all_dipoles.mean(axis=0)

    @staticmethod
    def compute_free_energy(
        site: HydrationSite,
        water_model: str | None = None,
    ) -> float:
        """Compute free energy relative to bulk: dG = (PE - PE_bulk) + (-TdS).

        Uses water-model-specific bulk energy.

        """
        if water_model is None:
            return site.free_energy
        wm_params = get_water_model_params(water_model)
        dH = site.potential_energy - wm_params["bulk_energy_kcal"]
        minus_TdS = site.entropy
        return dH + minus_TdS

    @staticmethod
    def compute_twobody_entropy(
        sites: list[HydrationSite],
        per_frame_positions: list[np.ndarray],
        hs_radius_A: float = 1.0,
        temperature_K: float = 300.0,
    ) -> np.ndarray:
        """Compute pairwise two-body entropy corrections between sites.

        Two-body entropy accounts for correlations between water occupancy
        at different hydration sites. This is expensive but improves accuracy.

        Returns (n_sites, n_sites) array of two-body entropy corrections.
        """
        n_sites = len(sites)
        kT = BOLTZMANN_KCAL * temperature_K
        centers = np.array([s.center for s in sites])

        # Pairwise occupancy correlations
        occupancy_matrix = np.zeros((n_sites, n_sites))
        single_occupancy = np.zeros(n_sites)

        for frame_pos in per_frame_positions:
            if len(frame_pos) == 0:
                continue
            tree = cKDTree(frame_pos)
            occupied = np.zeros(n_sites, dtype=bool)
            for i, center in enumerate(centers):
                nearby = tree.query_ball_point(center, hs_radius_A)
                occupied[i] = len(nearby) > 0
                if occupied[i]:
                    single_occupancy[i] += 1

            # Pairwise
            for i in range(n_sites):
                for j in range(i + 1, n_sites):
                    if occupied[i] and occupied[j]:
                        occupancy_matrix[i, j] += 1
                        occupancy_matrix[j, i] += 1

        n_frames = len(per_frame_positions)
        if n_frames == 0:
            return np.zeros((n_sites, n_sites))

        single_occupancy /= n_frames
        occupancy_matrix /= n_frames

        # Two-body entropy: S_2(i,j) = -kT * [p(i,j)*ln(p(i,j)/(p(i)*p(j)))]
        twobody = np.zeros((n_sites, n_sites))
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                p_ij = occupancy_matrix[i, j]
                p_i = single_occupancy[i]
                p_j = single_occupancy[j]
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    ratio = p_ij / (p_i * p_j)
                    twobody[i, j] = -kT * p_ij * np.log(ratio)
                    twobody[j, i] = twobody[i, j]

        return twobody
