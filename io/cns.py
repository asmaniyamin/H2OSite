# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
CNS/XPLOR grid file I/O.

Reads and writes crystallographic density maps in CNS format,
used for water density, cavity, energy, and entropy grids.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from H2OSite.analysis.density import DensityGrid

logger = logging.getLogger(__name__)


def write_cns_grid(grid: DensityGrid, path: str | Path, title: str = "") -> None:
    """Write a DensityGrid to CNS/XPLOR format.

    CNS format specification:
      - Header with title and grid dimensions
      - Section markers for each X-plane
      - Values in 6E12.5 format
    """
    path = Path(path)
    data = grid.data
    nx, ny, nz = data.shape

    # Compute cell parameters from grid
    a = nx * grid.spacing
    b = ny * grid.spacing
    c = nz * grid.spacing

    with open(path, "w") as f:
        # Header
        f.write("\n")
        f.write(f"       2 !NTITLE\n")
        f.write(f" REMARKS {title or grid.label}\n")
        f.write(f" REMARKS H2OSite density grid\n")

        # Grid dimensions
        f.write(
            f"{nx:8d}{0:8d}{nx - 1:8d}"
            f"{ny:8d}{0:8d}{ny - 1:8d}"
            f"{nz:8d}{0:8d}{nz - 1:8d}\n"
        )

        # Cell parameters (Å, degrees)
        f.write(
            f"{a:12.5E}{b:12.5E}{c:12.5E}"
            f"{'90.00000':>12s}{'90.00000':>12s}{'90.00000':>12s}\n"
        )

        # Axis order
        f.write("ZYX\n")

        # Data sections (Z-Y-X order)
        for iz in range(nz):
            f.write(f"{iz:8d}\n")
            values = data[:, :, iz].flatten()
            for i in range(0, len(values), 6):
                chunk = values[i:i + 6]
                f.write("".join(f"{v:12.5E}" for v in chunk) + "\n")

        # Statistics
        f.write(f"{-9999:8d}\n")
        mean = data.mean()
        std = data.std()
        f.write(f"{mean:12.4E}{std:12.4E}\n")

    logger.info("Wrote CNS grid to %s (%dx%dx%d)", path, nx, ny, nz)


def read_cns_grid(path: str | Path) -> DensityGrid:
    """Read a CNS/XPLOR format density grid.

    Returns a DensityGrid object.
    """
    path = Path(path)

    with open(path) as f:
        lines = f.readlines()

    # Parse header
    idx = 0
    # Skip blank lines
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    # Read NTITLE
    ntitle_line = lines[idx].strip()
    ntitle = int(ntitle_line.split()[0])
    idx += 1

    # Skip title lines
    idx += ntitle

    # Grid dimensions
    dim_line = lines[idx].strip()
    vals = [int(dim_line[i:i + 8]) for i in range(0, len(dim_line), 8)]
    nx = vals[0]
    x_start = vals[1]
    x_end = vals[2]
    ny = vals[3]
    y_start = vals[4]
    y_end = vals[5]
    nz = vals[6]
    z_start = vals[7]
    z_end = vals[8]
    idx += 1

    # Cell parameters
    cell_line = lines[idx].strip()
    cell_vals = []
    for i in range(0, len(cell_line), 12):
        try:
            cell_vals.append(float(cell_line[i:i + 12]))
        except ValueError:
            cell_vals.append(float(cell_line[i:i + 12].strip()))
    a, b, c = cell_vals[0], cell_vals[1], cell_vals[2]
    idx += 1

    # Axis order
    idx += 1  # Skip ZYX line

    # Read data
    grid_nx = x_end - x_start + 1
    grid_ny = y_end - y_start + 1
    grid_nz = z_end - z_start + 1

    data = np.zeros((grid_nx, grid_ny, grid_nz))

    for iz in range(grid_nz):
        idx += 1  # Skip section header
        n_values = grid_nx * grid_ny
        values = []
        while len(values) < n_values:
            line = lines[idx]
            # Parse values (12 characters each)
            for i in range(0, len(line.rstrip()), 12):
                try:
                    values.append(float(line[i:i + 12]))
                except ValueError:
                    pass
            idx += 1

        plane = np.array(values[:n_values]).reshape(grid_nx, grid_ny)
        data[:, :, iz] = plane

    spacing = a / grid_nx if grid_nx > 0 else 0.5
    origin = np.array([x_start * spacing, y_start * spacing, z_start * spacing])

    logger.info("Read CNS grid from %s: shape=(%d,%d,%d)", path, grid_nx, grid_ny, grid_nz)

    return DensityGrid(data, origin, spacing)
