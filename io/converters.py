# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""
Format converters for H2OSite output.

Provides structure conversion and PyMOL visualization export.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from H2OSite.analysis.clustering import HydrationSite

logger = logging.getLogger(__name__)


def watermap_to_pdb(
    sites: list[HydrationSite],
    output_path: str | Path,
    protein_pdb: str | Path | None = None,
    ligand_pdb: str | Path | None = None,
    bfactor_property: str = "dG",
    occupancy_property: str = "dH",
) -> None:
    """Write hydration sites to PDB format with thermodynamic properties.

    B-factor column: selected property (dG, dH, or -TdS)
    Occupancy column: selected property

    Args:
        sites: Hydration sites with computed properties
        output_path: Path for output PDB file
        protein_pdb: Optional protein PDB to prepend
        ligand_pdb: Optional ligand PDB to prepend
        bfactor_property: Property for B-factor column
        occupancy_property: Property for occupancy column
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write(f"REMARK H2OSite hydration sites\n")
        f.write(f"REMARK B-factor: {bfactor_property}\n")
        f.write(f"REMARK Occupancy: {occupancy_property}\n")

        atom_idx = 1

        # Write protein if provided
        if protein_pdb and Path(protein_pdb).exists():
            atom_idx = _append_pdb(f, Path(protein_pdb), atom_idx)
            f.write("TER\n")

        # Write ligand if provided
        if ligand_pdb and Path(ligand_pdb).exists():
            atom_idx = _append_pdb(f, Path(ligand_pdb), atom_idx)
            f.write("TER\n")

        # Write hydration sites as HETATM O atoms
        for site in sites:
            bfac = _get_property(site, bfactor_property)
            occ = _get_property(site, occupancy_property)
            x, y, z = site.center

            f.write(
                f"HETATM{atom_idx:5d}  O   WMP W{site.index + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occ:6.2f}{bfac:6.2f}           O\n"
            )
            atom_idx += 1

        f.write("END\n")

    logger.info("Wrote %d hydration sites to %s", len(sites), output_path)


def watermap_to_cif(
    sites: list[HydrationSite],
    output_path: str | Path,
) -> None:
    """Write hydration sites to an mmCIF/PDBx file."""
    output_path = Path(output_path)
    data_block = output_path.stem.replace("-", "_").replace(".", "_") or "watermap_sites"
    columns = [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_formal_charge",
        "auth_seq_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    ]

    with open(output_path, "w") as f:
        f.write(f"data_{data_block}\n#\n")
        f.write("loop_\n")
        for column in columns:
            f.write(f"_atom_site.{column}\n")

        for atom_idx, site in enumerate(sites, start=1):
            bfac = _get_property(site, "dG")
            occ = _get_property(site, "dH")
            x, y, z = site.center
            values = [
                "HETATM",
                str(atom_idx),
                "O",
                "O",
                ".",
                "WMP",
                "W",
                "1",
                str(site.index + 1),
                "?",
                f"{x:.3f}",
                f"{y:.3f}",
                f"{z:.3f}",
                f"{occ:.2f}",
                f"{bfac:.2f}",
                "?",
                str(site.index + 1),
                "WMP",
                "W",
                "O",
                "1",
            ]
            f.write(" ".join(values) + "\n")
        f.write("#\n")

    logger.info("Wrote %d hydration sites to %s", len(sites), output_path)


def watermap_to_pymol(
    sites: list[HydrationSite],
    output_path: str | Path,
    protein_pdb: str | Path | None = None,
    ligand_pdb: str | Path | None = None,
    color_by: str = "dG",
) -> None:
    """Generate a PyMOL script for visualizing H2OSite results.

    Creates spheres colored by thermodynamic properties:
      - Red: unfavorable (high dG)
      - Green: favorable (low dG)
      - Cyan: enthalpically unfavorable
      - Yellow: entropically unfavorable
    """
    output_path = Path(output_path)

    # Write PDB first
    pdb_path = output_path.with_suffix(".pdb")
    cif_path = output_path.with_suffix(".cif")
    watermap_to_pdb(sites, pdb_path, protein_pdb, ligand_pdb)
    watermap_to_cif(sites, cif_path)

    with open(output_path, "w") as f:
        f.write("# H2OSite PyMOL visualization\n")
        f.write(f"load {cif_path.name}, watermap\n")

        if protein_pdb:
            f.write(f"load {Path(protein_pdb).name}, protein\n")
            f.write("show cartoon, protein\n")
            f.write("color gray80, protein\n")

        if ligand_pdb:
            f.write(f"load {Path(ligand_pdb).name}, ligand\n")
            f.write("show sticks, ligand\n")
            f.write("color yellow, ligand\n")

        f.write("select hs, resn WMP\n")
        f.write("show spheres, hs\n")
        f.write("set sphere_scale, 0.5, hs\n\n")

        # Color by property
        for site in sites:
            sel = f"resn WMP and resi {site.index + 1}"
            r, g, b = _get_color(site, color_by)
            color_name = f"wm_color_{site.index}"
            f.write(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n")
            f.write(f"color {color_name}, {sel}\n")

        f.write("\nzoom hs, 5\n")
        f.write("set transparency, 0.3, hs\n")

    logger.info("Wrote PyMOL script to %s", output_path)


def watermap_to_csv(
    sites: list[HydrationSite],
    output_path: str | Path,
) -> None:
    """Export hydration site data to CSV."""
    output_path = Path(output_path)

    columns = [
        "site_index", "x", "y", "z", "occupancy", "dH_pw", "dH_ww",
        "dH_total", "minus_TdS", "dG", "hbond_pw", "hbond_ww", "hbond_lw",
        "dipole_x", "dipole_y", "dipole_z", "water_model",
        "bulk_water_energy_kcal", "temperature_K",
        "translational_entropy_cal_mol_K", "rotational_entropy_cal_mol_K",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for site in sites:
            components_known = site.properties.get("entropy_components_available", True)
            writer.writerow([
                site.index, *site.center, site.occupancy,
                site.enthalpy_pw, site.enthalpy_ww, site.relative_enthalpy,
                site.entropy, site.free_energy, site.hbond_pw, site.hbond_ww,
                site.hbond_lw, *site.dipole, site.water_model or "",
                site.bulk_reference_energy, site.temperature_K,
                site.translational_entropy if components_known else "",
                site.rotational_entropy if components_known else "",
            ])

    logger.info("Wrote %d sites to CSV: %s", len(sites), output_path)


def load_sites_csv(path: str | Path) -> list[HydrationSite]:
    """Load sites with entropy and bulk-water references preserved.

    New CSVs record the water model, reference energy, temperature, and entropy
    components. Older CSVs record total -TdS and relative enthalpy only; retain
    those totals without guessing an entropy decomposition or water-model name.
    """
    required = {"site_index", "x", "y", "z", "occupancy", "dH_pw", "dH_ww"}
    sites = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Site CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if not any(value for value in row.values()):
                continue
            try:
                def number(key: str, default=None) -> float:
                    value = row.get(key)
                    if value in (None, ""):
                        if default is None:
                            raise ValueError(f"Missing {key}")
                        return default
                    result = float(value)
                    if not np.isfinite(result):
                        raise ValueError(f"Non-finite {key}")
                    return result

                temperature = number("temperature_K", 300.0)
                if temperature <= 0:
                    raise ValueError("temperature_K must be positive")
                model = (row.get("water_model") or "").strip() or None
                site = HydrationSite(
                    index=int(row["site_index"]),
                    center=np.array([number(axis) for axis in ("x", "y", "z")]),
                    occupancy=number("occupancy"), n_water_frames=0,
                    waters_per_frame=np.array([], dtype=int),
                    enthalpy_pw=number("dH_pw"), enthalpy_ww=number("dH_ww"),
                    water_model=model, temperature_K=temperature,
                )
                if row.get("bulk_water_energy_kcal") not in (None, ""):
                    site.bulk_water_energy_kcal = number("bulk_water_energy_kcal")
                elif model is None:
                    # Recover the saved reference instead of assuming a model.
                    site.bulk_water_energy_kcal = site.potential_energy - number("dH_total")

                component_keys = ("translational_entropy_cal_mol_K", "rotational_entropy_cal_mol_K")
                components_known = all(row.get(key) not in (None, "") for key in component_keys)
                site.properties["entropy_components_available"] = components_known
                if components_known:
                    site.translational_entropy = number(component_keys[0])
                    site.rotational_entropy = number(component_keys[1])
                else:
                    total = number("minus_TdS")
                    site.translational_entropy = -total * 1000.0 / temperature
                    site.rotational_entropy = 0.0
                for key in ("hbond_pw", "hbond_ww", "hbond_lw"):
                    setattr(site, key, number(key, 0.0))
                site.dipole = np.array([number(f"dipole_{axis}", 0.0) for axis in "xyz"])
                sites.append(site)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid site CSV at line {reader.line_num}: {exc}") from exc
    return sites


def _get_property(site: HydrationSite, prop: str) -> float:
    dH = site.relative_enthalpy
    mapping = {
        "dG": dH + site.entropy,
        "dH": dH,
        "-TdS": site.entropy,
        "TdS": -site.entropy,
        "occupancy": site.occupancy,
    }
    return mapping.get(prop, 0.0)


def _get_color(site: HydrationSite, color_by: str) -> tuple[float, float, float]:
    """Get RGB color for a hydration site based on a property."""
    from H2OSite.vis.colors import (
        get_enthalpy_entropy_color,
        get_free_energy_color,
    )

    dH = site.relative_enthalpy
    dG = dH + site.entropy

    if color_by == "dG":
        return get_free_energy_color(dG)
    elif color_by == "dH_TdS":
        return get_enthalpy_entropy_color(dH, site.entropy)
    else:
        return get_free_energy_color(dG)


def _append_pdb(f, pdb_path: Path, start_idx: int) -> int:
    """Append ATOM/HETATM records from a PDB file, renumbering atoms."""
    idx = start_idx
    with open(pdb_path) as pdb:
        for line in pdb:
            if line.startswith(("ATOM", "HETATM")):
                f.write(f"{line[:6]}{idx:5d}{line[11:]}")
                idx += 1
    return idx
