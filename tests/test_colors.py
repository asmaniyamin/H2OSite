# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Tests for vis/colors.py - color mapping functions."""

import numpy as np
import pytest

from H2OSite.vis.colors import (
    colormap_to_rgb_array,
    get_enthalpy_entropy_color,
    get_free_energy_color,
    get_occupancy_color,
    get_simple_color,
)


class TestFreeEnergyColor:
    def test_favorable_is_green(self):
        r, g, b = get_free_energy_color(-2.5)
        assert g > r

    def test_unfavorable_is_red(self):
        r, g, b = get_free_energy_color(7.5)
        assert r > g

    def test_midpoint(self):
        r, g, b = get_free_energy_color(2.5)
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0

    def test_clamp_below(self):
        r, g, b = get_free_energy_color(-100)
        assert r == 0.0 and g == 1.0

    def test_clamp_above(self):
        r, g, b = get_free_energy_color(100)
        assert r == 1.0 and g == 0.0


class TestEnthalpyEntropyColor:
    def test_enthalpically_unfavorable(self):
        r, g, b = get_enthalpy_entropy_color(2.0, -0.5)  # dH>0, TdS<0
        assert b > 0  # cyan component

    def test_entropically_unfavorable(self):
        r, g, b = get_enthalpy_entropy_color(-1.0, 3.0)  # dH<0, TdS>0
        assert r > 0  # yellow component

    def test_both_favorable(self):
        r, g, b = get_enthalpy_entropy_color(-1.0, -0.5)
        assert g == 1.0  # green

    def test_both_unfavorable(self):
        r, g, b = get_enthalpy_entropy_color(2.0, 3.0)
        assert r > 0 and g == 0.0  # red


class TestSimpleColor:
    def test_favorable(self):
        assert get_simple_color(-1.0) == (0.0, 0.8, 0.0)

    def test_unfavorable(self):
        assert get_simple_color(1.0) == (0.8, 0.0, 0.0)


class TestOccupancyColor:
    def test_full_occupancy(self):
        assert get_occupancy_color(1.0) == (0.0, 0.0, 1.0)

    def test_zero_occupancy(self):
        assert get_occupancy_color(0.0) == (1.0, 1.0, 1.0)


class TestColormapArray:
    def test_shape(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        rgb = colormap_to_rgb_array(vals)
        assert rgb.shape == (4, 3)

    def test_range(self):
        vals = np.linspace(-5, 5, 20)
        rgb = colormap_to_rgb_array(vals)
        assert np.all(rgb >= 0) and np.all(rgb <= 1)
