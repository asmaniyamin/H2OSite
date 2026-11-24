# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for analysis/density.py - 3D density maps and TPI."""

import numpy as np
import pytest

from H2OSite.analysis.density import DensityGrid, WaterDensityMap


# ---- Fast: DensityGrid ----

class TestDensityGrid:
    def test_properties(self):
        g = DensityGrid(
            data=np.zeros((10, 10, 10)),
            origin=np.array([0.0, 0.0, 0.0]),
            spacing=0.5,
            label="test",
        )
        assert g.shape == (10, 10, 10)
        np.testing.assert_allclose(g.grid_to_cartesian(2, 3, 4), [1.0, 1.5, 2.0])

    def test_cartesian_to_grid(self):
        g = DensityGrid(np.zeros((20, 20, 20)), np.array([-5, -5, -5]), 0.5)
        idx = g.cartesian_to_grid(np.array([0.0, 0.0, 0.0]))
        assert idx == (10, 10, 10)

    def test_nonzero_points(self):
        data = np.zeros((5, 5, 5))
        data[2, 2, 2] = 1.0
        data[3, 3, 3] = 0.5
        g = DensityGrid(data, np.zeros(3), 1.0)
        pts = g.get_nonzero_points(threshold=0.0)
        assert len(pts) == 2


# ---- Fast: Water density map computation ----

class TestWaterDensity:
    def test_basic_density(self, rng):
        frames = [rng.normal(0, 1, (30, 3)) for _ in range(20)]
        dm = WaterDensityMap(grid_spacing_A=0.5, sigma_A=0.5)
        grid = dm.compute_density(frames)
        assert grid.data.ndim == 3
        assert grid.data.max() > 0
        assert grid.spacing == 0.5

    def test_density_peaks_at_cluster(self, rng):
        center = np.array([5.0, 5.0, 5.0])
        frames = [center + rng.normal(0, 0.3, (50, 3)) for _ in range(100)]
        dm = WaterDensityMap(grid_spacing_A=0.5)
        grid = dm.compute_density(frames)
        # Find max density location
        max_idx = np.unravel_index(grid.data.argmax(), grid.data.shape)
        max_pos = grid.grid_to_cartesian(*max_idx)
        assert np.linalg.norm(max_pos - center) < 2.0

    def test_empty_frames(self):
        dm = WaterDensityMap()
        # Single frame with some data
        frames = [np.array([[0.0, 0.0, 0.0]])]
        grid = dm.compute_density(frames)
        assert grid.data.ndim == 3

    def test_custom_bounds(self, rng):
        frames = [rng.uniform(-5, 5, (20, 3)) for _ in range(10)]
        dm = WaterDensityMap(grid_spacing_A=1.0)
        grid = dm.compute_density(
            frames,
            bounds_min=np.array([-10, -10, -10]),
            bounds_max=np.array([10, 10, 10]),
        )
        assert all(s >= 20 for s in grid.shape)


class TestEnergyDensity:
    def test_basic(self, rng):
        frames_pos = [rng.normal(0, 1, (10, 3)) for _ in range(10)]
        frames_e = [rng.normal(-20, 5, 10) for _ in range(10)]
        dm = WaterDensityMap(grid_spacing_A=1.0)
        grid = dm.compute_energy_density(frames_pos, frames_e)
        assert grid.data.ndim == 3
        assert grid.label == "energy_density"
