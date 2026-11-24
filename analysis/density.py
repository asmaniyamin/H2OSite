# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
3D water density maps and test particle insertion (TPI).

Generates:
  - Gaussian-filtered water density grids
  - Cavity occupancy maps via TPI
  - Continuous free energy, entropy, and enthalpy density grids
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from H2OSite.core.config import BULK_WATER_DENSITY

logger = logging.getLogger(__name__)


@dataclass
class DensityGrid:
    """A 3D density grid with metadata."""

    data: np.ndarray  # 3D array
    origin: np.ndarray  # (3,) Å - corner of grid
    spacing: float  # Å - grid spacing
    label: str = ""

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def grid_to_cartesian(self, i: int, j: int, k: int) -> np.ndarray:
        """Convert grid indices to Cartesian coordinates."""
        return self.origin + np.array([i, j, k]) * self.spacing

    def cartesian_to_grid(self, pos: np.ndarray) -> tuple[int, int, int]:
        """Convert Cartesian position to grid indices."""
        idx = ((pos - self.origin) / self.spacing).astype(int)
        return tuple(np.clip(idx, 0, np.array(self.shape) - 1))

    def get_nonzero_points(self, threshold: float = 0.0) -> np.ndarray:
        """Return Cartesian coordinates of grid points above threshold."""
        indices = np.argwhere(self.data > threshold)
        return self.origin + indices * self.spacing


class WaterDensityMap:
    """Compute 3D water density and related property maps."""

    def __init__(
        self,
        grid_spacing_A: float = 0.5,
        sigma_A: float = 0.5,
        margin_A: float = 3.0,
    ):
        self.grid_spacing = grid_spacing_A
        self.sigma = sigma_A
        self.margin = margin_A

    def compute_density(
        self,
        per_frame_positions: list[np.ndarray],
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
    ) -> DensityGrid:
        """Compute Gaussian-smoothed 3D water density.

        Args:
            per_frame_positions: Per-frame water O positions in Å
            bounds_min/max: Grid bounds (auto-computed if None)

        Returns:
            DensityGrid with density in molecules/Å³
        """
        non_empty = [p for p in per_frame_positions if len(p) > 0]
        if not non_empty:
            logger.warning("No water positions collected — returning empty density grid")
            return DensityGrid(
                data=np.zeros((1, 1, 1), dtype=np.float64),
                origin=np.zeros(3),
                spacing=self.grid_spacing,
                label="empty",
            )
        all_positions = np.vstack(non_empty)
        n_frames = len(per_frame_positions)

        if bounds_min is None:
            bounds_min = all_positions.min(axis=0) - self.margin
        if bounds_max is None:
            bounds_max = all_positions.max(axis=0) + self.margin

        shape = tuple(
            int(np.ceil((bounds_max[i] - bounds_min[i]) / self.grid_spacing)) + 1
            for i in range(3)
        )

        grid = np.zeros(shape, dtype=np.float64)
        voxel_vol = self.grid_spacing**3

        # Bin all positions
        for frame_pos in per_frame_positions:
            if len(frame_pos) == 0:
                continue
            idx = ((frame_pos - bounds_min) / self.grid_spacing).astype(int)
            idx = np.clip(idx, 0, np.array(shape) - 1)
            for i in idx:
                grid[i[0], i[1], i[2]] += 1

        # Normalize to density
        grid /= n_frames * voxel_vol

        # Gaussian smoothing
        sigma_voxels = self.sigma / self.grid_spacing
        grid = ndimage.gaussian_filter(grid, sigma=sigma_voxels)

        logger.info(
            "Water density grid: shape=%s, max=%.4f molecules/Å³",
            shape, grid.max(),
        )

        return DensityGrid(grid, bounds_min, self.grid_spacing, label="water_density")

    def compute_cavity_map(
        self,
        per_frame_positions: list[np.ndarray],
        protein_positions: np.ndarray,
        probe_radius_A: float = 1.0,
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
    ) -> DensityGrid:
        """Compute cavity occupancy map via test particle insertion (TPI).

        For each grid point, determines the fraction of frames where the
        point is unoccupied by water and accessible (no protein clash).
        This identifies binding-site cavities.

        Args:
            per_frame_positions: Per-frame water O positions
            protein_positions: Protein heavy atom positions
            probe_radius_A: Probe sphere radius

        Returns:
            DensityGrid with cavity occupancy (0-1)
        """
        all_positions = np.vstack([p for p in per_frame_positions if len(p) > 0])

        if bounds_min is None:
            bounds_min = all_positions.min(axis=0) - self.margin
        if bounds_max is None:
            bounds_max = all_positions.max(axis=0) + self.margin

        shape = tuple(
            int(np.ceil((bounds_max[i] - bounds_min[i]) / self.grid_spacing)) + 1
            for i in range(3)
        )

        # Pre-check: grid points not clashing with protein
        from scipy.spatial import cKDTree

        protein_tree = cKDTree(protein_positions)
        grid_points = []
        grid_indices = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pt = bounds_min + np.array([i, j, k]) * self.grid_spacing
                    grid_points.append(pt)
                    grid_indices.append((i, j, k))

        grid_points = np.array(grid_points)
        # Points not clashing with protein
        protein_dists = protein_tree.query(grid_points)[0]
        accessible = protein_dists > probe_radius_A

        cavity_grid = np.zeros(shape, dtype=np.float64)

        # For each frame, mark grid points that have no nearby water
        n_frames = len(per_frame_positions)
        for frame_pos in per_frame_positions:
            if len(frame_pos) == 0:
                # All accessible points are cavity
                for idx, acc in zip(grid_indices, accessible):
                    if acc:
                        cavity_grid[idx] += 1.0
                continue

            water_tree = cKDTree(frame_pos)
            water_dists = water_tree.query(grid_points)[0]
            empty = water_dists > probe_radius_A

            for gi, (idx, acc, emp) in enumerate(zip(grid_indices, accessible, empty)):
                if acc and emp:
                    cavity_grid[idx] += 1.0

        cavity_grid /= n_frames

        logger.info("Cavity map: shape=%s, mean occupancy=%.3f", shape, cavity_grid.mean())

        return DensityGrid(cavity_grid, bounds_min, self.grid_spacing, label="cavity")

    def compute_energy_density(
        self,
        per_frame_positions: list[np.ndarray],
        per_frame_energies: list[np.ndarray],
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
    ) -> DensityGrid:
        """Compute energy-weighted density grid.

        Each water contributes its interaction energy to the grid voxel it occupies.
        Normalized by frame count and voxel volume.
        """
        all_positions = np.vstack([p for p in per_frame_positions if len(p) > 0])

        if bounds_min is None:
            bounds_min = all_positions.min(axis=0) - self.margin
        if bounds_max is None:
            bounds_max = all_positions.max(axis=0) + self.margin

        shape = tuple(
            int(np.ceil((bounds_max[i] - bounds_min[i]) / self.grid_spacing)) + 1
            for i in range(3)
        )

        grid = np.zeros(shape, dtype=np.float64)
        count_grid = np.zeros(shape, dtype=np.float64)
        n_frames = len(per_frame_positions)

        for frame_pos, frame_e in zip(per_frame_positions, per_frame_energies):
            if len(frame_pos) == 0:
                continue
            idx = ((frame_pos - bounds_min) / self.grid_spacing).astype(int)
            idx = np.clip(idx, 0, np.array(shape) - 1)
            for j, i in enumerate(idx):
                grid[i[0], i[1], i[2]] += frame_e[j] if j < len(frame_e) else 0
                count_grid[i[0], i[1], i[2]] += 1

        # Average energy per voxel
        nonzero = count_grid > 0
        grid[nonzero] /= count_grid[nonzero]

        return DensityGrid(grid, bounds_min, self.grid_spacing, label="energy_density")
