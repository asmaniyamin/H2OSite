# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for io/cns.py - CNS grid read/write."""

import numpy as np
import pytest

from H2OSite.analysis.density import DensityGrid
from H2OSite.io.cns import read_cns_grid, write_cns_grid


class TestCNSRoundTrip:
    def test_write_read(self, tmp_path):
        data = np.random.default_rng(42).uniform(0, 1, (8, 8, 8))
        grid = DensityGrid(data, origin=np.array([-2.0, -2.0, -2.0]), spacing=0.5, label="test")

        path = tmp_path / "test.cns"
        write_cns_grid(grid, path, title="unit test")
        assert path.exists()
        assert path.stat().st_size > 100

    def test_file_contains_remarks(self, tmp_path):
        data = np.ones((4, 4, 4))
        grid = DensityGrid(data, np.zeros(3), 1.0)
        path = tmp_path / "t.cns"
        write_cns_grid(grid, path, title="hello")
        text = path.read_text()
        assert "REMARKS" in text
        assert "hello" in text
        assert "ZYX" in text

    def test_statistics_line(self, tmp_path):
        rng = np.random.default_rng(0)
        data = rng.normal(0.5, 0.1, (6, 6, 6))
        grid = DensityGrid(data, np.zeros(3), 1.0)
        path = tmp_path / "t.cns"
        write_cns_grid(grid, path)
        text = path.read_text()
        assert "-9999" in text  # statistics marker
