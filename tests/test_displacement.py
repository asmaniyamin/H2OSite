# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for scoring/displacement.py - ligand displacement scoring."""

import numpy as np
import pytest

from H2OSite.scoring.displacement import DisplacementScorer, VDW_RADII


class TestVdwRadii:
    def test_common_elements(self):
        assert VDW_RADII["C"] == 1.70
        assert VDW_RADII["N"] == 1.55
        assert VDW_RADII["O"] == 1.52


class TestSphereOverlap:
    def test_no_overlap(self):
        f = DisplacementScorer._sphere_overlap_fraction
        assert f(1.0, 1.0, 5.0) == 0.0

    def test_full_containment(self):
        f = DisplacementScorer._sphere_overlap_fraction
        assert f(0.5, 2.0, 0.0) == 1.0  # small inside big

    def test_touching(self):
        f = DisplacementScorer._sphere_overlap_fraction
        assert f(1.0, 1.0, 2.0) == 0.0  # exactly touching

    def test_partial_overlap(self):
        f = DisplacementScorer._sphere_overlap_fraction
        o = f(1.0, 1.0, 1.0)
        assert 0.0 < o < 1.0

    def test_coincident(self):
        f = DisplacementScorer._sphere_overlap_fraction
        assert f(1.0, 1.0, 0.0) == 1.0


class TestDisplacementScorer:
    def test_score_overlapping_ligand(self, sample_hydration_sites, sample_ligand_atoms):
        pos, elem = sample_ligand_atoms
        scorer = DisplacementScorer(scale_factor=0.8)
        results = scorer.score(sample_hydration_sites, pos, elem, hs_radius_A=1.0)

        assert len(results) == 3
        # Site at origin should have high overlap (ligand atom right there)
        assert results[0].overlap > 0.0

    def test_distant_ligand_no_overlap(self, sample_hydration_sites):
        pos = np.array([[100.0, 100.0, 100.0]])
        elem = ["C"]
        scorer = DisplacementScorer()
        results = scorer.score(sample_hydration_sites, pos, elem)
        for r in results:
            assert r.overlap == 0.0
            assert r.dG == 0.0

    def test_scale_factor_effect(self, sample_hydration_sites, sample_ligand_atoms):
        pos, elem = sample_ligand_atoms
        s1 = DisplacementScorer(scale_factor=0.5).score(sample_hydration_sites, pos, elem)
        s2 = DisplacementScorer(scale_factor=1.0).score(sample_hydration_sites, pos, elem)
        # Larger scale = larger effective VdW = more overlap
        overlaps_1 = sum(r.overlap for r in s1)
        overlaps_2 = sum(r.overlap for r in s2)
        assert overlaps_2 >= overlaps_1

    def test_score_multiple(self, sample_hydration_sites, sample_ligand_atoms):
        pos, elem = sample_ligand_atoms
        scorer = DisplacementScorer()
        results = scorer.score_multiple_ligands(
            sample_hydration_sites,
            [(pos, elem), (pos + 50, elem)],
        )
        assert len(results) == 2
        assert len(results[0]) == 3
