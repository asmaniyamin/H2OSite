# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for analysis/clustering.py - hydration site identification."""

import numpy as np
import pytest

from H2OSite.analysis.clustering import HydrationSite, HydrationSiteClustering
from H2OSite.core.config import BULK_WATER_ENERGY


# ---- Fast: HydrationSite dataclass ----

class TestHydrationSite:
    def test_entropy_formula(self):
        hs = HydrationSite(
            index=0, center=np.zeros(3), occupancy=1.0,
            n_water_frames=100, waters_per_frame=np.ones(100),
        )
        hs.translational_entropy = 2.0
        hs.rotational_entropy = 1.0
        # entropy = -(trans + rot) * 0.300
        assert abs(hs.entropy - (-(2.0 + 1.0) * 0.300)) < 1e-10

    def test_potential_energy(self):
        hs = HydrationSite(
            index=0, center=np.zeros(3), occupancy=1.0,
            n_water_frames=100, waters_per_frame=np.ones(100),
        )
        hs.enthalpy_pw = -12.0
        hs.enthalpy_ww = -7.0
        assert hs.potential_energy == -19.0

    def test_free_energy(self):
        hs = HydrationSite(
            index=0, center=np.zeros(3), occupancy=1.0,
            n_water_frames=100, waters_per_frame=np.ones(100),
        )
        hs.enthalpy_pw = -12.0
        hs.enthalpy_ww = -7.0
        hs.translational_entropy = 2.0
        hs.rotational_entropy = 1.0
        dH = hs.potential_energy - BULK_WATER_ENERGY
        dG = dH + hs.entropy
        assert abs(hs.free_energy - dG) < 1e-10


# ---- Fast: Clustering with synthetic data ----

class TestClusteringSynthetic:
    def test_empty_input(self):
        c = HydrationSiteClustering()
        sites = c.cluster(np.empty((0, 3)), 10)
        assert sites == []

    def test_single_cluster(self, rng):
        center = np.array([5.0, 5.0, 5.0])
        positions = center + rng.normal(0, 0.2, size=(500, 3))
        sites = HydrationSiteClustering(
            hs_radius_A=1.0, dcutoff_A=2.0, min_occupancy=0.0,
        ).cluster(positions, n_frames=100)
        assert len(sites) >= 1
        # Center should be near (5, 5, 5)
        dists = [np.linalg.norm(s.center - center) for s in sites]
        assert min(dists) < 1.5

    def test_three_clusters(self, rng):
        """Well-separated clusters should be found individually."""
        centers = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0]], dtype=float)
        frames = []
        for _ in range(80):
            pts = [c + rng.normal(0, 0.2, (10, 3)) for c in centers]
            frames.append(np.vstack(pts))
        all_pos = np.vstack(frames)
        sites = HydrationSiteClustering(
            hs_radius_A=1.0, dcutoff_A=1.5, min_occupancy=0.05,
        ).cluster(all_pos, n_frames=len(frames), per_frame_positions=frames)
        assert 2 <= len(sites) <= 6

    def test_occupancy_computed(self, clustered_water_positions):
        all_pos = np.vstack(clustered_water_positions)
        sites = HydrationSiteClustering(
            hs_radius_A=1.5, dcutoff_A=2.0, min_occupancy=0.0,
        ).cluster(all_pos, n_frames=len(clustered_water_positions),
                  per_frame_positions=clustered_water_positions)
        for s in sites:
            assert 0.0 <= s.occupancy <= 1.0

    def test_min_occupancy_filter(self, rng):
        # Very sparse data - sites should be filtered
        positions = rng.uniform(-20, 20, size=(50, 3))
        sites = HydrationSiteClustering(
            min_occupancy=0.5
        ).cluster(positions, n_frames=1000)
        for s in sites:
            assert s.occupancy >= 0.5

    def test_sites_reindexed(self, clustered_water_positions):
        all_pos = np.vstack(clustered_water_positions)
        sites = HydrationSiteClustering(min_occupancy=0.0).cluster(
            all_pos, n_frames=50, per_frame_positions=clustered_water_positions,
        )
        for i, s in enumerate(sites):
            assert s.index == i


# ---- Medium: density grid construction ----

class TestDensityGrid:
    def test_grid_shape(self, rng):
        positions = rng.uniform(-5, 5, size=(1000, 3))
        c = HydrationSiteClustering(grid_spacing_A=0.5)
        grid, origin, shape = c._build_density_grid(positions, n_frames=10)
        assert grid.ndim == 3
        assert all(s > 10 for s in shape)  # At least 10 Å / 0.5 = 20 voxels
        assert grid.max() > 0

    def test_high_density_near_cluster(self, rng):
        center = np.zeros(3)
        positions = center + rng.normal(0, 0.3, size=(5000, 3))
        c = HydrationSiteClustering(grid_spacing_A=0.5)
        grid, origin, _ = c._build_density_grid(positions, n_frames=100)
        # Find voxel nearest to center
        center_idx = tuple(((center - origin) / 0.5).astype(int))
        # Density near center should be high
        assert grid[center_idx] > 0.01
