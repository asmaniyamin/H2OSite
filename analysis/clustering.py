# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
Hydration site identification via density-based clustering.

Reimplements wmap_sphere_cluster: identifies locations where water molecules
cluster during MD, producing hydration sites with occupancy statistics.

Algorithm:
  1. Bin water oxygen positions onto a 3D grid
  2. Identify high-density grid voxels (> 2x bulk density)
  3. Cluster nearby high-density voxels using DBSCAN or sphere-based merging
  4. Refine hydration site centers to density-weighted centroids
  5. Assign per-frame waters to hydration sites
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from H2OSite.core.config import get_water_model_params

logger = logging.getLogger(__name__)


@dataclass
class HydrationSite:
    """A single hydration site identified from MD simulation."""

    index: int
    center: np.ndarray  # (3,) Å
    occupancy: float  # fraction of frames site is occupied
    n_water_frames: int  # total water observations
    waters_per_frame: np.ndarray  # per-frame count
    translational_entropy: float = 0.0  # cal/(mol*K)
    rotational_entropy: float = 0.0  # cal/(mol*K)
    enthalpy_pw: float = 0.0  # protein-water PE, kcal/mol
    enthalpy_ww: float = 0.0  # water-water PE, kcal/mol
    hbond_pw: float = 0.0  # avg protein-water H-bonds
    hbond_ww: float = 0.0  # avg water-water H-bonds
    hbond_lw: float = 0.0  # avg ligand-water H-bonds
    dipole: np.ndarray = field(default_factory=lambda: np.zeros(3))
    properties: dict = field(default_factory=dict)
    water_model: str | None = "tip4p"
    bulk_water_energy_kcal: float | None = None
    temperature_K: float = 300.0

    @property
    def entropy(self) -> float:
        """Total -TdS in kcal/mol at the site's analysis temperature.

        Translational and rotational entropies are stored in cal/(mol·K).
        The factor T/1000 converts cal/(mol*K) to kcal/mol.
        """
        return -(self.translational_entropy + self.rotational_entropy) * self.temperature_K / 1000.0

    @property
    def potential_energy(self) -> float:
        """Total potential energy in kcal/mol."""
        return self.enthalpy_pw + self.enthalpy_ww

    @property
    def free_energy(self) -> float:
        """dG relative to the site's recorded bulk-water reference."""
        return self.relative_enthalpy + self.entropy

    @property
    def bulk_reference_energy(self) -> float:
        """Bulk interaction energy in kcal/mol, including legacy CSV references."""
        if self.bulk_water_energy_kcal is not None:
            return self.bulk_water_energy_kcal
        if self.water_model is None:
            raise ValueError("Hydration site has no bulk-water reference")
        return get_water_model_params(self.water_model)["bulk_energy_kcal"]

    @property
    def relative_enthalpy(self) -> float:
        """Protein-water plus water-water energy relative to bulk, in kcal/mol."""
        return self.potential_energy - self.bulk_reference_energy


class HydrationSiteClustering:
    """Identify hydration sites from water oxygen positions.

    Uses a smoothed 3D density grid, voxel selection above twice the bulk
    density, DBSCAN clustering, and density-weighted centroid refinement.
    ThermodynamicCalculator computes site entropies separately.
    """

    def __init__(
        self,
        hs_radius_A: float = 1.0,
        dcutoff_A: float = 2.0,
        grid_spacing_A: float = 0.5,
        density_threshold_factor: float = 2.0,
        min_occupancy: float = 0.1,
        water_model: str = "tip4p",
    ):
        self.hs_radius = hs_radius_A
        self.dcutoff = dcutoff_A
        self.grid_spacing = grid_spacing_A
        self.density_threshold_factor = density_threshold_factor
        self.min_occupancy = min_occupancy
        self.water_model = water_model
        self.bulk_density = get_water_model_params(water_model)["bulk_density"]

    def cluster(
        self,
        water_positions_A: np.ndarray,
        n_frames: int,
        per_frame_positions: list[np.ndarray] | None = None,
    ) -> list[HydrationSite]:
        """Identify hydration sites from water oxygen positions.

        Args:
            water_positions_A: (N, 3) array of all water oxygen positions in Å
            n_frames: Total number of MD frames
            per_frame_positions: Optional list of per-frame position arrays

        Returns:
            List of HydrationSite objects
        """
        if len(water_positions_A) == 0:
            logger.warning("No water positions to cluster")
            return []

        logger.info(
            "Clustering %d water positions from %d frames",
            len(water_positions_A), n_frames,
        )

        # Step 1: Build 3D density grid
        grid, origin, shape = self._build_density_grid(water_positions_A, n_frames)

        # Step 2: Find high-density voxels
        threshold = self.bulk_density * self.density_threshold_factor
        high_density = grid > threshold
        n_high = np.sum(high_density)
        logger.info(
            "Found %d high-density voxels (threshold=%.4f molecules/Å³)",
            n_high, threshold,
        )

        if n_high == 0:
            logger.warning("No high-density regions found")
            return []

        # Step 3: Cluster high-density voxels
        high_coords = np.argwhere(high_density)
        high_positions = origin + high_coords * self.grid_spacing
        high_densities = grid[high_density]

        # Use DBSCAN with dcutoff as eps
        clustering = DBSCAN(
            eps=self.dcutoff,
            min_samples=3,
            metric="euclidean",
        ).fit(high_positions)

        labels = clustering.labels_
        unique_labels = set(labels) - {-1}
        logger.info("DBSCAN found %d clusters", len(unique_labels))

        # Step 4: Compute density-weighted centroids
        sites = []
        for label in sorted(unique_labels):
            mask = labels == label
            cluster_pos = high_positions[mask]
            cluster_dens = high_densities[mask]

            # Density-weighted centroid
            weights = cluster_dens / cluster_dens.sum()
            center = np.average(cluster_pos, axis=0, weights=weights)

            sites.append(center)

        # Step 5: Assign per-frame waters to sites and compute occupancy
        hydration_sites = self._assign_waters_to_sites(
            sites, water_positions_A, n_frames, per_frame_positions
        )

        # Filter by minimum occupancy
        hydration_sites = [
            hs for hs in hydration_sites if hs.occupancy >= self.min_occupancy
        ]

        # Re-index
        for i, hs in enumerate(hydration_sites):
            hs.index = i

        logger.info("Final hydration sites: %d (occupancy >= %.2f)",
                     len(hydration_sites), self.min_occupancy)

        return hydration_sites

    def _build_density_grid(
        self,
        positions: np.ndarray,
        n_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        """Construct a 3D density grid from water positions.

        Returns (grid, origin, shape) where grid values are in molecules/Å³.
        """
        margin = 2.0  # Å padding
        mins = positions.min(axis=0) - margin
        maxs = positions.max(axis=0) + margin

        shape = tuple(
            int(np.ceil((maxs[i] - mins[i]) / self.grid_spacing)) + 1
            for i in range(3)
        )

        grid = np.zeros(shape, dtype=np.float64)
        voxel_volume = self.grid_spacing**3

        # Bin positions
        indices = ((positions - mins) / self.grid_spacing).astype(int)
        indices = np.clip(indices, 0, np.array(shape) - 1)

        for idx in indices:
            grid[idx[0], idx[1], idx[2]] += 1

        # Normalize: count / (n_frames * voxel_volume) = density in molecules/Å³
        grid /= n_frames * voxel_volume

        # Gaussian smoothing (sigma ~ half grid spacing)
        sigma = 0.5 / self.grid_spacing
        grid = ndimage.gaussian_filter(grid, sigma=sigma)

        logger.info(
            "Density grid: shape=%s, origin=%s, max_density=%.4f molecules/Å³",
            shape, mins, grid.max(),
        )

        return grid, mins, shape

    def _assign_waters_to_sites(
        self,
        site_centers: list[np.ndarray],
        all_positions: np.ndarray,
        n_frames: int,
        per_frame_positions: list[np.ndarray] | None,
    ) -> list[HydrationSite]:
        """Assign water observations to hydration sites using KDTree."""
        if not site_centers:
            return []

        centers = np.array(site_centers)
        tree = cKDTree(centers)

        hydration_sites = []
        for i, center in enumerate(centers):
            hydration_sites.append(
                HydrationSite(
                    index=i,
                    center=center,
                    occupancy=0.0,
                    n_water_frames=0,
                    waters_per_frame=np.zeros(n_frames, dtype=int),
                    water_model=self.water_model,
                )
            )

        if per_frame_positions is not None:
            # Per-frame assignment for accurate occupancy
            for frame_idx, frame_pos in enumerate(per_frame_positions):
                if len(frame_pos) == 0:
                    continue
                dists, site_indices = tree.query(frame_pos, distance_upper_bound=self.hs_radius)
                for d, si in zip(dists, site_indices):
                    if d <= self.hs_radius and si < len(hydration_sites):
                        hydration_sites[si].waters_per_frame[frame_idx] += 1
                        hydration_sites[si].n_water_frames += 1
        else:
            # Bulk assignment from concatenated positions
            dists, site_indices = tree.query(all_positions, distance_upper_bound=self.hs_radius)
            for d, si in zip(dists, site_indices):
                if d <= self.hs_radius and si < len(hydration_sites):
                    hydration_sites[si].n_water_frames += 1

        # Compute occupancy
        for hs in hydration_sites:
            if per_frame_positions is not None:
                hs.occupancy = np.mean(hs.waters_per_frame > 0)
            else:
                hs.occupancy = hs.n_water_frames / n_frames

        return hydration_sites
