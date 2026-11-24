# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""H2OSite provides thermodynamic profiling of water in protein binding sites.

Uses OpenMM for molecular dynamics, MDAnalysis/MDTraj for trajectory analysis,
SciPy/scikit-learn for clustering, and NumPy for IST/GIST calculations.
"""

__version__ = "0.1.0"

from H2OSite.pipeline import WaterMapPipeline  # noqa: F401
