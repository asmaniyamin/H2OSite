# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for prep/system_builder.py - solvation and system construction."""

import pytest

from H2OSite.tests.conftest import requires_openmm


@requires_openmm
class TestSystemBuilderSmall:
    """Fast tests for SystemBuilder logic."""

    def test_water_ff_mapping(self):
        from H2OSite.prep.system_builder import SystemBuilder
        sb = SystemBuilder("dummy.pdb", water_model="tip4pew")
        assert "tip4pew" in sb._water_ff_file()

    def test_tip3p_mapping(self):
        from H2OSite.prep.system_builder import SystemBuilder
        sb = SystemBuilder("dummy.pdb", water_model="tip3p")
        assert "tip3p" in sb._water_ff_file()

    def test_invalid_water_model(self):
        from H2OSite.prep.system_builder import SystemBuilder
        sb = SystemBuilder("dummy.pdb", water_model="fake_water")
        with pytest.raises(ValueError, match="Unknown water model"):
            sb._water_ff_file()

    def test_openmmtools_system_builds(self):
        """Verify openmmtools AlanineDipeptideExplicit creates valid system."""
        from openmmtools.testsystems import AlanineDipeptideExplicit
        ala = AlanineDipeptideExplicit()
        assert ala.topology.getNumAtoms() > 22  # dipeptide + water
        assert ala.system.getNumParticles() == ala.topology.getNumAtoms()
