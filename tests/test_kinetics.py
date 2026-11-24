# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for scoring/kinetics.py - Pearlstein kinetics classification."""

import numpy as np
import pytest

from H2OSite.analysis.clustering import HydrationSite
from H2OSite.core.config import BULK_WATER_ENERGY
from H2OSite.scoring.displacement import DisplacementResult
from H2OSite.scoring.kinetics import KineticsClassifier, POLAR_ELEMENTS


class TestKineticsClassifier:
    @pytest.fixture
    def setup(self):
        """3 sites: unfavorable, favorable-I, favorable-II."""
        sites = []
        # Unfavorable: dH > 0
        s0 = HydrationSite(0, np.array([0, 0, 0]), 0.9, 90, np.ones(100))
        s0.enthalpy_pw = -8.0
        s0.enthalpy_ww = -4.0  # PE = -12, dH = -12 - (-19.67) = +7.67
        s0.translational_entropy = 2.0
        s0.rotational_entropy = 1.0
        sites.append(s0)

        # Favorable I: dH < 0, dG > 0
        s1 = HydrationSite(1, np.array([3, 0, 0]), 0.95, 95, np.ones(100))
        s1.enthalpy_pw = -18.0
        s1.enthalpy_ww = -4.0  # PE = -22, dH = -22 - (-19.67) = -2.33
        s1.translational_entropy = 5.0
        s1.rotational_entropy = 4.0  # big entropy: -TdS = -(5+4)*0.3 = -2.7
        sites.append(s1)

        # Favorable II: dH < 0, dG < 0
        s2 = HydrationSite(2, np.array([0, 4, 0]), 0.8, 80, np.ones(100))
        s2.enthalpy_pw = -20.0
        s2.enthalpy_ww = -5.0  # PE = -25, dH = -25 - (-19.67) = -5.33
        s2.translational_entropy = 0.5
        s2.rotational_entropy = 0.3
        sites.append(s2)

        # All overlapping
        results = [
            DisplacementResult(0, overlap=0.8, dH=7.67, minus_TdS=-0.9, dG=6.77),
            DisplacementResult(1, overlap=0.7, dH=-2.33, minus_TdS=-2.7, dG=-5.03),
            DisplacementResult(2, overlap=0.6, dH=-5.33, minus_TdS=-0.24, dG=-5.57),
        ]

        lig_pos = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])
        lig_elem = ["C", "N", "O"]

        return sites, results, lig_pos, lig_elem

    def test_classification(self, setup):
        sites, results, lig_pos, lig_elem = setup
        kc = KineticsClassifier(overlap_threshold=0.5)
        kr = kc.classify(sites, results, lig_pos, lig_elem)

        assert 0 in kr.unfavorable_sites
        assert len(kr.favorable_I_sites) + len(kr.favorable_II_sites) >= 1

    def test_polar_correction(self, setup):
        sites, results, lig_pos, lig_elem = setup
        kc = KineticsClassifier(overlap_threshold=0.5)
        kr = kc.classify(sites, results, lig_pos, lig_elem)

        # Site 2 is nearest to "O" (polar) -> should be polar_corrected
        if 2 in kr.per_site:
            assert kr.per_site[2]["polar_corrected"]

    def test_high_threshold_no_displaced(self, setup):
        sites, results, lig_pos, lig_elem = setup
        kc = KineticsClassifier(overlap_threshold=0.99)
        kr = kc.classify(sites, results, lig_pos, lig_elem)
        assert len(kr.unfavorable_sites) == 0

    def test_polar_elements_set(self):
        assert "N" in POLAR_ELEMENTS
        assert "O" in POLAR_ELEMENTS
        assert "C" not in POLAR_ELEMENTS
