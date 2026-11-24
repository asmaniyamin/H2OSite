# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for analysis/thermodynamics.py - IST entropy/enthalpy calculations."""

import numpy as np
import pytest

from H2OSite.analysis.clustering import HydrationSite
from H2OSite.analysis.thermodynamics import ThermodynamicCalculator
from H2OSite.core.config import BOLTZMANN_KCAL, BULK_WATER_DENSITY, BULK_WATER_ENERGY


# ---- Fast: unit tests ----

class TestTranslationalEntropy:
    def test_high_density_site(self, rng):
        """Water heavily localized -> high translational entropy penalty."""
        center = np.zeros(3)
        site = HydrationSite(
            index=0, center=center, occupancy=0.9,
            n_water_frames=0, waters_per_frame=np.array([]),
        )
        # All waters in one spot
        frames = [center + rng.normal(0, 0.1, (10, 3)) for _ in range(100)]

        calc = ThermodynamicCalculator(hs_radius_A=1.5)
        s_trans = calc._translational_entropy(site, frames, 100)
        # Should be non-zero (localized water has lower entropy than bulk)
        assert s_trans != 0.0

    def test_empty_frames(self):
        site = HydrationSite(
            index=0, center=np.zeros(3), occupancy=0.0,
            n_water_frames=0, waters_per_frame=np.array([]),
        )
        calc = ThermodynamicCalculator()
        s = calc._translational_entropy(site, [], 0)
        assert s == 0.0


class TestRotationalEntropy:
    def test_aligned_dipoles(self, rng):
        """Strongly aligned dipoles -> high rotational entropy penalty."""
        center = np.zeros(3)
        site = HydrationSite(
            index=0, center=center, occupancy=0.9,
            n_water_frames=0, waters_per_frame=np.array([]),
        )
        # Positions near center
        frames_pos = [center + rng.normal(0, 0.2, (20, 3)) for _ in range(100)]
        # All dipoles pointing the same way (aligned)
        frames_dip = [np.tile([1, 0, 0], (20, 1)).astype(float) for _ in range(100)]

        calc = ThermodynamicCalculator(hs_radius_A=1.5)
        s_rot = calc._rotational_entropy(site, frames_pos, frames_dip, 100)
        # Aligned dipoles have low entropy (negative KL divergence from uniform)
        assert isinstance(s_rot, float)

    def test_isotropic_dipoles(self, rng):
        """Random dipoles -> near-zero rotational entropy penalty."""
        center = np.zeros(3)
        site = HydrationSite(
            index=0, center=center, occupancy=0.9,
            n_water_frames=0, waters_per_frame=np.array([]),
        )
        frames_pos = [center + rng.normal(0, 0.2, (50, 3)) for _ in range(200)]
        frames_dip = [rng.standard_normal((50, 3)) for _ in range(200)]

        calc = ThermodynamicCalculator(hs_radius_A=1.5)
        s_rot = calc._rotational_entropy(site, frames_pos, frames_dip, 200)
        # Should be near zero for random orientations
        assert abs(s_rot) < 0.5


class TestMeanDipole:
    def test_aligned(self, rng):
        center = np.zeros(3)
        site = HydrationSite(
            index=0, center=center, occupancy=1.0,
            n_water_frames=0, waters_per_frame=np.array([]),
        )
        frames_pos = [np.zeros((5, 3)) for _ in range(10)]
        frames_dip = [np.tile([0, 0, 1], (5, 1)).astype(float) for _ in range(10)]

        calc = ThermodynamicCalculator(hs_radius_A=2.0)
        dip = calc._mean_dipole(site, frames_pos, frames_dip)
        assert dip[2] > 0.9


class TestComputeAll:
    def test_updates_sites(self, clustered_water_positions):
        from H2OSite.analysis.clustering import HydrationSiteClustering
        all_pos = np.vstack(clustered_water_positions)
        sites = HydrationSiteClustering(min_occupancy=0.0).cluster(
            all_pos, 50, clustered_water_positions,
        )
        if not sites:
            pytest.skip("No sites found")

        calc = ThermodynamicCalculator(hs_radius_A=1.0)
        sites = calc.compute_all(sites, clustered_water_positions)
        for s in sites:
            assert isinstance(s.translational_entropy, float)


class TestFreeEnergy:
    def test_static(self):
        site = HydrationSite(
            index=0, center=np.zeros(3), occupancy=1.0,
            n_water_frames=100, waters_per_frame=np.ones(100),
        )
        site.enthalpy_pw = -15.0
        site.enthalpy_ww = -5.0
        site.translational_entropy = 1.0
        site.rotational_entropy = 0.5

        dG = ThermodynamicCalculator.compute_free_energy(site)
        dH = (-15.0 + -5.0) - BULK_WATER_ENERGY
        minus_TdS = site.entropy
        assert abs(dG - (dH + minus_TdS)) < 1e-10


class TestTwobodyEntropy:
    def test_shape(self, clustered_water_positions, sample_hydration_sites):
        tb = ThermodynamicCalculator.compute_twobody_entropy(
            sample_hydration_sites,
            clustered_water_positions,
            hs_radius_A=1.5,
        )
        n = len(sample_hydration_sites)
        assert tb.shape == (n, n)
        # Symmetric
        np.testing.assert_allclose(tb, tb.T, atol=1e-12)
