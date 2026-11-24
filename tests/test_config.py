# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for core/config.py - constants, H2OSiteConfig, YAML round-trip."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from H2OSite.core.config import (
    BOLTZMANN_KCAL,
    BULK_WATER_DENSITY,
    BULK_WATER_ENERGY,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_EQUILIBRATION_STAGES,
    DEFAULT_PRODUCTION_PARAMS,
    GCMC_CHEMICAL_POTENTIALS,
    TIP4P_EXCESS_CHEMICAL_POTENTIAL,
    WaterMapConfig,
)


# ---- Fast unit tests ----

class TestPhysicalConstants:
    def test_boltzmann(self):
        assert abs(BOLTZMANN_KCAL - 0.001987204) < 1e-8

    def test_bulk_water_energy(self):
        assert BULK_WATER_ENERGY == -19.67

    def test_bulk_water_density(self):
        assert 0.033 < BULK_WATER_DENSITY < 0.034

    def test_tip4p_mu_excess(self):
        assert TIP4P_EXCESS_CHEMICAL_POTENTIAL == -6.058

    def test_gcmc_potentials_tip4p(self):
        entry = GCMC_CHEMICAL_POTENTIALS[("TIP4P", 2.0)]
        assert entry["mu_excess"] == -6.058
        assert abs(entry["solvent_density"] - 0.033148) < 1e-6

    def test_gcmc_potentials_tip3p(self):
        assert ("TIP3P", 2.0) in GCMC_CHEMICAL_POTENTIALS


class TestDefaultParams:
    def test_equilibration_stages_count(self):
        assert len(DEFAULT_EQUILIBRATION_STAGES) == 4

    def test_brownian_first_stage(self):
        s = DEFAULT_EQUILIBRATION_STAGES[0]
        assert s["integrator"] == "Brownian"
        assert s["temperature_K"] == 10.0
        assert s["restraint_k"] == 50.0

    def test_production_defaults(self):
        assert DEFAULT_PRODUCTION_PARAMS["time_ps"] == 2000.0
        assert DEFAULT_PRODUCTION_PARAMS["temperature_K"] == 300.0

    def test_clustering_defaults(self):
        assert DEFAULT_CLUSTERING_PARAMS["hs_radius_A"] == 1.0
        assert DEFAULT_CLUSTERING_PARAMS["dcutoff_A"] == 2.0


class TestWaterMapConfig:
    def test_defaults(self):
        cfg = WaterMapConfig()
        # Default tip3p: matches the loch-backed GCMC calibration.
        assert cfg.water_model == "tip3p"
        assert cfg.simulation_time_ns == 5.0
        assert cfg.production_time_ps == 5000.0
        assert cfg.hs_radius_A == 1.0
        assert cfg.scale_factor == 0.8
        assert not cfg.skip_gcmc
        assert not cfg.twobody_entropy
        assert cfg.gcmc_flexible_water
        assert cfg.gcmc_timestep_fs == 1.0
        assert cfg.gcmc_minimize_after_customization
        assert not cfg.gcmc_redraw_velocities_after_initialise
        assert cfg.gcmc_ghost_min_distance_A == 1.5
        assert cfg.run_post_gcmc_equilibration
        assert not cfg.run_post_gcmc_without_gcmc

    def test_custom(self):
        cfg = WaterMapConfig(
            protein_path="/tmp/p.pdb",
            ligand_path="/tmp/l.sdf",
            simulation_time_ns=5.0,
            skip_gcmc=True,
        )
        assert cfg.production_time_ps == 5000.0
        assert cfg.skip_gcmc
        assert cfg.protein_path == Path("/tmp/p.pdb")

    def test_equilibration_populated(self):
        cfg = WaterMapConfig()
        assert len(cfg.equilibration_stages) == 4

    def test_yaml_round_trip(self):
        cfg = WaterMapConfig(
            protein_path="/tmp/p.pdb",
            ligand_path="/tmp/l.sdf",
            jobname="test_job",
            simulation_time_ns=3.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            cfg.to_yaml(f.name)
            loaded = WaterMapConfig.from_yaml(f.name)

        assert loaded.jobname == "test_job"
        assert loaded.simulation_time_ns == 3.0
        assert str(loaded.protein_path) == "/tmp/p.pdb"
