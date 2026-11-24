# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for analysis/hbond.py - hydrogen bond counting."""

import numpy as np
import pytest

from H2OSite.analysis.clustering import HydrationSite
from H2OSite.analysis.hbond import HBondAnalyzer


# ---- Fast: array-based analysis ----

class TestHBondFromArrays:
    @pytest.fixture
    def analyzer(self):
        return HBondAnalyzer(distance_cutoff_A=3.5, hs_radius_A=1.5)

    @pytest.fixture
    def sites_at_origin(self):
        return [HydrationSite(
            index=0, center=np.zeros(3), occupancy=1.0,
            n_water_frames=100, waters_per_frame=np.ones(10),
        )]

    def test_no_protein_no_hbonds(self, analyzer, sites_at_origin):
        water_frames = [np.array([[0.0, 0.0, 0.0]])] * 5
        protein_frames = [np.empty((0, 3))] * 5

        sites = analyzer.analyze_from_arrays(sites_at_origin, water_frames, protein_frames)
        assert sites[0].hbond_pw == 0.0

    def test_nearby_protein_gives_hbonds(self, analyzer, sites_at_origin):
        water_frames = [np.array([[0.0, 0.0, 0.0]])] * 10
        protein_frames = [np.array([[2.0, 0.0, 0.0]])] * 10  # within 3.5 Å

        sites = analyzer.analyze_from_arrays(sites_at_origin, water_frames, protein_frames)
        assert sites[0].hbond_pw > 0

    def test_distant_protein_no_hbonds(self, analyzer, sites_at_origin):
        water_frames = [np.array([[0.0, 0.0, 0.0]])] * 10
        protein_frames = [np.array([[10.0, 10.0, 10.0]])] * 10

        sites = analyzer.analyze_from_arrays(sites_at_origin, water_frames, protein_frames)
        assert sites[0].hbond_pw == 0.0

    def test_water_water_hbonds(self, analyzer, sites_at_origin):
        # Two waters close together, one in site
        water_frames = [np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])] * 10
        protein_frames = [np.empty((0, 3))] * 10

        sites = analyzer.analyze_from_arrays(sites_at_origin, water_frames, protein_frames)
        assert sites[0].hbond_ww > 0

    def test_ligand_water_hbonds(self, analyzer, sites_at_origin):
        water_frames = [np.array([[0.0, 0.0, 0.0]])] * 10
        protein_frames = [np.empty((0, 3))] * 10
        ligand_frames = [np.array([[1.5, 0.0, 0.0]])] * 10

        sites = analyzer.analyze_from_arrays(
            sites_at_origin, water_frames, protein_frames, ligand_frames,
        )
        assert sites[0].hbond_lw > 0

    def test_multiple_sites(self, analyzer):
        sites = [
            HydrationSite(0, np.array([0, 0, 0]), 1.0, 100, np.ones(10)),
            HydrationSite(1, np.array([10, 0, 0]), 1.0, 100, np.ones(10)),
        ]
        water_frames = [np.array([[0, 0, 0], [10, 0, 0]])] * 10
        protein_frames = [np.array([[1.5, 0, 0]])] * 10

        sites = analyzer.analyze_from_arrays(sites, water_frames, protein_frames)
        # Site 0 is near protein, site 1 is not
        assert sites[0].hbond_pw > sites[1].hbond_pw
