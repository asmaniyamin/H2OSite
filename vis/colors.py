# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Color mapping for hydration-site visualization.

Free energy: green (favorable) to red (unfavorable).
Enthalpy/entropy decomposition: cyan (enthalpy) to yellow (entropy).

Display thresholds in kcal/mol:
  dG: [-2.5, +7.5]
  dH: [-2.5, +2.5]
  TdS: [0, 5]
"""

from __future__ import annotations

import numpy as np


def get_free_energy_color(
    dG: float,
    dG_min: float = -2.5,
    dG_max: float = 7.5,
) -> tuple[float, float, float]:
    """Map free energy to RGB color.

    Green (0, 1, 0) for favorable (low dG)
    Red (1, 0, 0) for unfavorable (high dG)
    """
    ratio = _clamp_ratio(dG, dG_min, dG_max)
    r = ratio
    g = 1.0 - ratio
    b = 0.0
    return (r, g, b)


def get_enthalpy_entropy_color(
    dH: float,
    minus_TdS: float,
    dH_range: tuple[float, float] = (-2.5, 2.5),
    TdS_range: tuple[float, float] = (0.0, 5.0),
) -> tuple[float, float, float]:
    """Map enthalpy/entropy decomposition to RGB.

    Color scheme:
      - Cyan (0, 1, 1): enthalpically unfavorable (dH > 0)
      - Yellow (1, 1, 0): entropically unfavorable (-TdS > 0)
      - Green (0, 1, 0): both favorable
      - Red (1, 0, 0): both unfavorable
    """
    h_ratio = _clamp_ratio(dH, dH_range[0], dH_range[1])  # 0=favorable, 1=unfavorable
    s_ratio = _clamp_ratio(minus_TdS, TdS_range[0], TdS_range[1])

    if dH > 0 and minus_TdS <= 0:
        # Enthalpically unfavorable - cyan
        r = 0.0
        g = 1.0 - 0.5 * h_ratio
        b = h_ratio
    elif dH <= 0 and minus_TdS > 0:
        # Entropically unfavorable - yellow
        r = s_ratio
        g = 1.0 - 0.5 * s_ratio
        b = 0.0
    elif dH > 0 and minus_TdS > 0:
        # Both unfavorable - red
        intensity = max(h_ratio, s_ratio)
        r = intensity
        g = 0.0
        b = 0.0
    else:
        # Both favorable - green
        r = 0.0
        g = 1.0
        b = 0.0

    return (r, g, b)


def get_simple_color(dG: float, threshold: float = 0.0) -> tuple[float, float, float]:
    """Simple binary color: green if favorable, red if unfavorable."""
    if dG < threshold:
        return (0.0, 0.8, 0.0)
    return (0.8, 0.0, 0.0)


def get_occupancy_color(
    occupancy: float,
) -> tuple[float, float, float]:
    """Map occupancy (0-1) to blue intensity."""
    o = max(0.0, min(1.0, occupancy))
    return (1.0 - o, 1.0 - o, 1.0)


def colormap_to_rgb_array(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "RdYlGn_r",
) -> np.ndarray:
    """Convert values to RGB array using matplotlib colormap.

    Returns (N, 3) float array with RGB values in [0, 1].
    """
    try:
        from matplotlib import cm
        from matplotlib.colors import Normalize

        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        rgba = cmap(norm(values))
        return rgba[:, :3]
    except ImportError:
        # Fallback without matplotlib
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        ratios = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgb = np.zeros((len(values), 3))
        rgb[:, 0] = ratios  # R
        rgb[:, 1] = 1 - ratios  # G
        return rgb


def _clamp_ratio(value: float, vmin: float, vmax: float) -> float:
    """Linearly map value to [0, 1] clamped to [vmin, vmax]."""
    if vmax <= vmin:
        return 0.5
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
