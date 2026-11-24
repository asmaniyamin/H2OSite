# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Shared synthetic fixtures and optional dependency checks for H2OSite tests."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------
def _check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


requires_openmm = pytest.mark.skipif(
    not _check_import("openmm"), reason="OpenMM not installed"
)
requires_mdanalysis = pytest.mark.skipif(
    not _check_import("MDAnalysis"), reason="MDAnalysis not installed"
)
requires_grand = pytest.mark.skipif(
    not _check_import("grand"), reason="grand package not installed"
)
requires_rdkit = pytest.mark.skipif(
    not _check_import("rdkit"), reason="RDKit not installed"
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data (fast, no dependencies)
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def bulk_water_positions(rng):
    """10 frames of ~50 random water O positions in a 20 Å cube."""
    frames = []
    for _ in range(10):
        n = rng.integers(40, 60)
        frames.append(rng.uniform(-10, 10, size=(n, 3)))
    return frames


@pytest.fixture
def clustered_water_positions(rng):
    """50 frames with 3 clear clusters + noise."""
    centers = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    frames = []
    for _ in range(50):
        pts = []
        for c in centers:
            n = rng.integers(3, 8)
            pts.append(c + rng.normal(0, 0.3, size=(n, 3)))
        pts.append(rng.uniform(-8, 8, size=(rng.integers(2, 5), 3)))
        frames.append(np.vstack(pts))
    return frames


@pytest.fixture
def sample_hydration_sites():
    """Pre-built hydration sites for scoring/kinetics tests."""
    from H2OSite.analysis.clustering import HydrationSite

    sites = []
    s0 = HydrationSite(
        index=0, center=np.array([0.0, 0.0, 0.0]),
        occupancy=0.9, n_water_frames=90,
        waters_per_frame=np.ones(100, dtype=int),
    )
    s0.enthalpy_pw = -10.0
    s0.enthalpy_ww = -5.0
    s0.translational_entropy = 3.0
    s0.rotational_entropy = 1.5
    sites.append(s0)

    s1 = HydrationSite(
        index=1, center=np.array([3.0, 0.0, 0.0]),
        occupancy=0.95, n_water_frames=95,
        waters_per_frame=np.ones(100, dtype=int),
    )
    s1.enthalpy_pw = -18.0
    s1.enthalpy_ww = -8.0
    s1.translational_entropy = 0.5
    s1.rotational_entropy = 0.3
    sites.append(s1)

    s2 = HydrationSite(
        index=2, center=np.array([0.0, 4.0, 0.0]),
        occupancy=0.6, n_water_frames=60,
        waters_per_frame=np.ones(100, dtype=int),
    )
    s2.enthalpy_pw = -14.0
    s2.enthalpy_ww = -6.0
    s2.translational_entropy = 2.0
    s2.rotational_entropy = 1.0
    sites.append(s2)

    return sites


@pytest.fixture
def sample_ligand_atoms():
    """Small synthetic ligand near origin."""
    positions = np.array([
        [0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0],
        [-0.5, -0.8, 0.3], [0.0, 0.5, 1.5],
    ])
    elements = ["C", "C", "N", "O", "C"]
    return positions, elements
