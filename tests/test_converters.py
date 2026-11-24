# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for io/converters.py - PDB, PyMOL, CSV export."""

import numpy as np
import pytest

from H2OSite.analysis.clustering import HydrationSite
from H2OSite.io.converters import watermap_to_csv, watermap_to_pdb, watermap_to_pymol


@pytest.fixture
def two_sites():
    s0 = HydrationSite(0, np.array([1.0, 2.0, 3.0]), 0.9, 90, np.ones(100))
    s0.enthalpy_pw = -15.0
    s0.enthalpy_ww = -5.0
    s0.translational_entropy = 1.0
    s0.rotational_entropy = 0.5
    s1 = HydrationSite(1, np.array([4.0, 5.0, 6.0]), 0.5, 50, np.ones(100))
    s1.enthalpy_pw = -10.0
    s1.enthalpy_ww = -3.0
    s1.translational_entropy = 2.0
    s1.rotational_entropy = 1.5
    return [s0, s1]


class TestPDBExport:
    def test_writes_hetatm(self, two_sites, tmp_path):
        out = tmp_path / "wm.pdb"
        watermap_to_pdb(two_sites, out)
        text = out.read_text()
        assert text.count("HETATM") == 2
        assert "WMP" in text
        assert "END" in text

    def test_coordinates_present(self, two_sites, tmp_path):
        out = tmp_path / "wm.pdb"
        watermap_to_pdb(two_sites, out)
        text = out.read_text()
        assert "1.000" in text
        assert "2.000" in text


class TestCSVExport:
    def test_header_and_rows(self, two_sites, tmp_path):
        out = tmp_path / "wm.csv"
        watermap_to_csv(two_sites, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "site_index" in lines[0]
        assert "dG" in lines[0]

    def test_values_parseable(self, two_sites, tmp_path):
        out = tmp_path / "wm.csv"
        watermap_to_csv(two_sites, out)
        lines = out.read_text().strip().split("\n")
        parts = lines[1].split(",")
        assert int(parts[0]) == 0
        float(parts[4])  # occupancy
        float(parts[9])  # dG


class TestPyMOLExport:
    def test_writes_script(self, two_sites, tmp_path):
        out = tmp_path / "wm.pml"
        watermap_to_pymol(two_sites, out)
        text = out.read_text()
        assert "load" in text
        assert "sphere" in text.lower()
        assert "set_color" in text

    def test_companion_pdb(self, two_sites, tmp_path):
        out = tmp_path / "wm.pml"
        watermap_to_pymol(two_sites, out)
        pdb = tmp_path / "wm.pdb"
        assert pdb.exists()
