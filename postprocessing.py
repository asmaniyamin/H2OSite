# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Generic postproduction display artifacts for H2OSite runs.

Creates viewer-friendly outputs directly from a completed pipeline run:
  - centered production trajectory
  - centered solvated PDB matching frame 0 of that trajectory
  - compatibility ``*_fixed`` aliases for corrected H2OSite outputs
  - centered/original-frame hydration-site PDBs
  - PyMOL helper scripts and a ``fixed/`` bundle

Run ``python -m H2OSite.postprocessing OUTPUT_DIR --reanalyze`` to
recompute analysis from a saved trajectory before generating these exports.
Without ``--reanalyze``, exports use the existing analysis results.

This module does not rely on system-specific residue IDs.  It derives 
the centered active-site marker from the actual tracking center used 
during production.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

from H2OSite.io.structure_io import (
    load_openmm_structure_file,
    structure_to_openmm,
    write_openmm_structure_file,
)

logger = logging.getLogger(__name__)

WATER_OH_BOND_A = 0.9572
WATER_HOH_ANGLE_DEG = 104.52

DEFAULT_CHECKPOINT_NAMES = (
    "post_gcmc_equil.npz", "gcmc.npz", "equilibration.npz",
)


def choose_checkpoint(output_dir: Path) -> Path:
    """Find the latest completed pipeline stage in a saved run."""
    for name in DEFAULT_CHECKPOINT_NAMES:
        path = output_dir / "checkpoints" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No pipeline checkpoint found in {output_dir / 'checkpoints'}")


def infer_jobname(output_dir: Path) -> str:
    """Infer the job stem from analysis outputs or prepared structures."""
    for path in sorted(output_dir.glob("*_wm.csv")):
        return path.name.removesuffix("_wm.csv")
    for pattern in ("*_protein.cif", "*_protein.pdb"):
        for path in sorted((output_dir / "prep").glob(pattern)):
            return path.stem.removesuffix("_protein")
    return output_dir.name


def find_production_trajectory(output_dir: Path) -> Path:
    """Resolve the saved trajectory, including the older filename."""
    for name in ("production.dcd", "trajectory.dcd"):
        path = output_dir / "production" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No raw production trajectory found in {output_dir / 'production'}")


def run_corrected_reanalysis(
    output_dir: Path,
    jobname: str,
    tracking_center_A: np.ndarray,
    site_radius_A: float = 10.0,
    hs_radius_A: float = 1.0,
    cutoff_A: float = 2.0,
    *,
    checkpoint_file: str | Path | None = None,
    trajectory_file: str | Path | None = None,
    water_model: str = "tip3p",
) -> dict[str, Path]:
    """Recompute analysis from raw MD into fixed outputs, preserving raw results.

    Uses an isolated temporary directory so failed reanalysis does not remove
    existing run directories or publish partially computed analysis results.
    """
    from H2OSite.core.config import WaterMapConfig
    from H2OSite.pipeline import WaterMapPipeline

    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint_file) if checkpoint_file else choose_checkpoint(output_dir)
    trajectory = Path(trajectory_file) if trajectory_file else find_production_trajectory(output_dir)
    for path in (checkpoint, trajectory):
        if not path.is_file():
            raise FileNotFoundError(path)
    production_checkpoint = trajectory.with_name("checkpoint.npz")

    with tempfile.TemporaryDirectory(prefix=".watermap-reanalysis-", dir=output_dir) as temp:
        temp_output = Path(temp)
        prep_dir = output_dir / "prep"
        if prep_dir.exists():
            shutil.copytree(prep_dir, temp_output / "prep")
        temp_jobname = f"{jobname}_reanalysis_tmp"
        config = WaterMapConfig(
            output_dir=temp_output,
            jobname=temp_jobname,
            ligand_distance_A=site_radius_A,
            hs_radius_A=hs_radius_A,
            clustering_cutoff_A=cutoff_A,
            generate_display_outputs=False,
            water_model=water_model,
        )
        pipeline = WaterMapPipeline(config)
        pipeline.run_analysis_from_trajectory(
            checkpoint_file=checkpoint,
            trajectory_file=trajectory,
            production_checkpoint_file=(
                production_checkpoint if production_checkpoint.is_file() else None
            ),
            active_site_center_A=np.asarray(tracking_center_A, dtype=float),
            active_site_radius_A=site_radius_A,
        )
        sources = {
            "fixed_csv": (temp_output / f"{temp_jobname}_wm.csv",
                          output_dir / f"{jobname}_wm_fixed.csv"),
            "fixed_water_cns": (temp_output / f"{temp_jobname}-water.cns",
                                output_dir / f"{jobname}-water_fixed.cns"),
            "fixed_positions": (temp_output / "retracked_water_positions.npy",
                                output_dir / "production" / "water_positions_fixed.npy"),
        }
        for source, _ in sources.values():
            if not source.is_file():
                raise FileNotFoundError(f"Reanalysis did not produce {source.name}")
        cavity_source = temp_output / f"{temp_jobname}-continuous_cavity.cns"
        cavity_target = output_dir / f"{jobname}-continuous_cavity_fixed.cns"
        if cavity_source.is_file():
            sources["fixed_cavity_cns"] = (cavity_source, cavity_target)
        artifacts = {}
        for key, (source, target) in sources.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            artifacts[key] = target
        if not cavity_source.is_file():
            # A previous fixed map must not masquerade as this reanalysis result.
            cavity_target.unlink(missing_ok=True)
            (output_dir / "fixed" / cavity_target.name).unlink(missing_ok=True)

    for fmt, writer in (("pdb", write_sites_pdb_from_csv), ("cif", write_sites_cif_from_csv)):
        target = output_dir / f"{jobname}_wm_fixed.{fmt}"
        writer(artifacts["fixed_csv"], target)
        artifacts[f"fixed_{fmt}"] = target
    return artifacts


def ensure_fixed_dir(output_dir: Path) -> Path:
    """Create and return the postprocessed fixed-output directory."""
    fixed_dir = output_dir / "fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    return fixed_dir


def find_prepared_protein(output_dir: Path) -> Path | None:
    """Return the prepared protein structure if one is present."""
    prep_dir = output_dir / "prep"
    matches = sorted(prep_dir.glob("*_protein.cif")) + sorted(
        prep_dir.glob("*_protein.pdb")
    )
    return matches[0] if matches else None


def find_prepared_ligand(output_dir: Path) -> Path | None:
    """Return the prepared ligand structure if one is present."""
    prep_dir = output_dir / "prep"
    matches = sorted(prep_dir.glob("*_ligand.cif")) + sorted(
        prep_dir.glob("*_ligand.pdb")
    )
    return matches[0] if matches else None


def find_prepared_ligand_sdf(output_dir: Path) -> Path | None:
    """Return the prepared ligand SDF if one is present."""
    prep_dir = output_dir / "prep"
    matches = sorted(prep_dir.glob("*_ligand.sdf"))
    return matches[0] if matches else None


def find_original_input_pdb(output_dir: Path) -> Path | None:
    """Return the original input-style protein structure when available."""
    excluded_names = {"solvated.pdb", "solvated.cif"}
    excluded_fragments = ("_wm", "_ligand", "ligand", "solvated")

    candidates = []
    for pattern in ("*.cif", "*.pdb"):
        for path in sorted(output_dir.glob(pattern)):
            lower_name = path.name.lower()
            if lower_name in excluded_names:
                continue
            if any(fragment in lower_name for fragment in excluded_fragments):
                continue
            candidates.append(path)

    if candidates:
        preferred = [
            path for path in candidates
            if path.name.lower().endswith("_his_fixed.pdb")
            or path.name.lower().endswith("_his_fixed.cif")
            or "his_fixed" in path.name.lower()
        ]
        if preferred:
            return max(preferred, key=lambda path: path.stat().st_size)
        return max(candidates, key=lambda path: path.stat().st_size)

    return find_prepared_protein(output_dir)


def load_translation_vector(output_dir: Path, jobname: str) -> np.ndarray | None:
    """Load the ligand-centering translation vector from prep metadata."""
    meta_path = output_dir / "prep" / f"{jobname}_meta.npz"
    if not meta_path.exists():
        matches = sorted((output_dir / "prep").glob("*_meta.npz"))
        if not matches:
            return None
        meta_path = matches[0]

    data = np.load(meta_path)
    if "translation_vector" not in data:
        return None
    return np.asarray(data["translation_vector"], dtype=float)


def load_drift_report(output_dir: Path, jobname: str) -> list[dict[str, str]]:
    """Load the optional drifted-water report written by the pipeline."""
    report_path = output_dir / f"{jobname}_drifted_waters.csv"
    if not report_path.exists():
        return []
    with open(report_path, newline="") as handle:
        return list(csv.DictReader(handle))


def _unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return a normalized vector or a normalized fallback if needed."""
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        vec = np.asarray(fallback, dtype=float)
        norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def _wrap_point_into_box(point: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap one Cartesian point into the primary periodic box."""
    wrapped = np.asarray(point, dtype=float).copy()
    box_lengths = np.asarray(box, dtype=float).reshape(-1)
    valid = box_lengths > 1e-8
    wrapped[valid] = np.mod(wrapped[valid], box_lengths[valid])
    return wrapped


def _structure_resnames(structure_path: Path | None) -> list[str]:
    """Return unique residue names present in a structure file."""
    if structure_path is None or not structure_path.exists():
        return []

    topology, _positions = load_openmm_structure_file(structure_path)
    resnames: list[str] = []
    seen: set[str] = set()
    for residue in topology.residues():
        resname = residue.name.strip()
        if not resname or resname in seen:
            continue
        seen.add(resname)
        resnames.append(resname)
    return resnames


def _water_hydrogen_positions(
    oxygen_xyz: np.ndarray,
    dipole_xyz: np.ndarray,
    bond_length_A: float = WATER_OH_BOND_A,
    angle_deg: float = WATER_HOH_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Build approximate H positions from the site's mean dipole direction."""
    oxygen = np.asarray(oxygen_xyz, dtype=float)
    bisector = _unit_vector(dipole_xyz, fallback=np.array([0.0, 0.0, 1.0]))

    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(bisector, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    perpendicular = np.cross(bisector, reference)
    perpendicular = _unit_vector(perpendicular, fallback=np.array([0.0, 1.0, 0.0]))

    half_angle = np.deg2rad(angle_deg / 2.0)
    h1 = oxygen + bond_length_A * (
        np.cos(half_angle) * bisector + np.sin(half_angle) * perpendicular
    )
    h2 = oxygen + bond_length_A * (
        np.cos(half_angle) * bisector - np.sin(half_angle) * perpendicular
    )
    return h1, h2


def _format_pdb_atom(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resid: int,
    xyz: np.ndarray,
    occupancy: float,
    bfactor: float,
    element: str,
    record: str = "HETATM",
) -> str:
    """Format one PDB ATOM/HETATM line."""
    x, y, z = [float(coord) for coord in xyz]
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s} {resname:>3s} {chain_id:1s}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}"
    )


def _unwrap_chain(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Image atoms in one chain so they stay contiguous across PBC."""
    pos = positions.copy()
    ref = pos[0].copy()
    for dim in range(3):
        diff = pos[:, dim] - ref[dim]
        pos[:, dim] -= np.round(diff / box[dim]) * box[dim]
    return pos


def _get_chain_indices(protein) -> dict[str, np.ndarray]:
    """Return per-chain atom indices for the protein selection."""
    chain_ids: list[str] = []
    for attr in ("chainIDs", "segids"):
        try:
            raw = getattr(protein, attr)
        except Exception:
            continue
        values = [cid for cid in sorted(set(raw)) if str(cid).strip()]
        if values:
            chain_ids = [str(cid) for cid in values]
            break

    if not chain_ids:
        return {"protein": protein.indices}

    chain_indices = {}
    for cid in chain_ids:
        try:
            sel = protein.select_atoms(f"chainID {cid}")
            if sel.n_atoms == 0:
                sel = protein.select_atoms(f"segid {cid}")
        except Exception:
            continue
        if sel.n_atoms:
            chain_indices[cid] = sel.indices
    return chain_indices or {"protein": protein.indices}


def _get_protein_component_indices(
    topology,
    protein_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return covalently connected protein components from an OpenMM topology."""
    protein_set = set(int(idx) for idx in np.asarray(protein_indices, dtype=int))
    if not protein_set:
        return {}

    adjacency: dict[int, set[int]] = {idx: set() for idx in protein_set}
    for atom1, atom2 in topology.bonds():
        idx1 = int(atom1.index)
        idx2 = int(atom2.index)
        if idx1 in protein_set and idx2 in protein_set:
            adjacency[idx1].add(idx2)
            adjacency[idx2].add(idx1)

    visited: set[int] = set()
    components: list[np.ndarray] = []
    for atom_index in sorted(protein_set):
        if atom_index in visited:
            continue
        stack = [atom_index]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited, reverse=True))
        components.append(np.array(sorted(component), dtype=int))

    components.sort(key=lambda arr: (-arr.size, int(arr[0]) if arr.size else 0))
    return {
        f"protein_component_{index + 1}": component
        for index, component in enumerate(components)
        if component.size > 0
    }


def fix_and_center_frame(
    positions: np.ndarray,
    box: np.ndarray,
    chain_indices: dict[str, np.ndarray],
    protein_indices: np.ndarray,
    solvent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap protein chains, center the protein, and wrap solvent back in."""
    box_center = box / 2.0
    pos = positions.copy()

    for idx in chain_indices.values():
        pos[idx] = _unwrap_chain(pos[idx], box)

    chain_names = list(chain_indices.keys())
    if len(chain_names) > 1:
        ref_center = pos[chain_indices[chain_names[0]]].mean(axis=0)
        for chain_name in chain_names[1:]:
            idx = chain_indices[chain_name]
            chain_center = pos[idx].mean(axis=0)
            for dim in range(3):
                shift = (
                    np.round((ref_center[dim] - chain_center[dim]) / box[dim])
                    * box[dim]
                )
                pos[idx, dim] += shift

    protein_center = pos[protein_indices].mean(axis=0)
    shift_A = box_center - protein_center
    pos += shift_A
    pos[solvent_indices] = pos[solvent_indices] % box
    return pos, shift_A


def write_centered_trajectory(
    topology,
    trajectory_file: str | Path,
    centered_dcd: Path,
    solvated_pdb: Path,
    solvated_cif: Path | None,
    solvated_final_pdb: Path | None,
    solvated_final_cif: Path | None,
    tracking_center_A: np.ndarray,
) -> np.ndarray:
    """Write a centered DCD plus frame-0 / final-frame solvated structures."""
    import MDAnalysis as mda
    import openmm.unit as unit

    raw_dcd = Path(trajectory_file)
    centered_dcd.parent.mkdir(parents=True, exist_ok=True)

    universe = mda.Universe(topology, str(raw_dcd))
    protein = universe.select_atoms("protein")
    non_protein = universe.select_atoms("not protein")
    if protein.n_atoms == 0:
        raise RuntimeError("No protein atoms found; cannot center trajectory")

    chain_indices = _get_protein_component_indices(topology, protein.indices)
    if not chain_indices:
        chain_indices = _get_chain_indices(protein)
    centered_tracking_center_A = np.asarray(tracking_center_A, dtype=float)

    with mda.Writer(str(centered_dcd), n_atoms=universe.atoms.n_atoms) as writer:
        for frame_index, ts in enumerate(universe.trajectory, start=1):
            if ts.dimensions is None or np.any(ts.dimensions[:3] <= 0):
                box = universe.atoms.positions.ptp(axis=0) + 10.0
            else:
                box = ts.dimensions[:3].astype(float)
            centered_pos, shift_A = fix_and_center_frame(
                ts.positions,
                box,
                chain_indices,
                protein.indices,
                non_protein.indices,
            )
            ts.positions = centered_pos
            writer.write(universe.atoms)
            if frame_index == 1:
                centered_tracking_center_A = _wrap_point_into_box(
                    np.asarray(tracking_center_A, dtype=float) + shift_A,
                    box,
                )

    centered_universe = mda.Universe(topology, str(centered_dcd))
    centered_universe.trajectory[0]
    centered_universe.atoms.write(str(solvated_pdb))
    if solvated_cif is not None:
        frame0_positions = unit.Quantity(
            np.asarray(centered_universe.atoms.positions, dtype=float),
            unit.angstrom,
        )
        write_openmm_structure_file(topology, frame0_positions, solvated_cif)
    if solvated_final_pdb is not None:
        centered_universe.trajectory[-1]
        centered_universe.atoms.write(str(solvated_final_pdb))
        if solvated_final_cif is not None:
            final_positions = unit.Quantity(
                np.asarray(centered_universe.atoms.positions, dtype=float),
                unit.angstrom,
            )
            write_openmm_structure_file(topology, final_positions, solvated_final_cif)
    return centered_tracking_center_A


def write_drifted_waters_final_pdb(
    *,
    topology,
    centered_trajectory: Path,
    drift_rows: list[dict[str, str]],
    output_pdb: Path,
    output_cif: Path | None = None,
) -> Path | None:
    """Write a final-frame structure containing waters reported as drifted."""
    import MDAnalysis as mda
    import openmm.unit as unit

    topology_resids: list[int] = []
    for row in drift_rows:
        try:
            topology_resids.append(int(row["topology_resid"]))
        except Exception:
            continue
    unique_resids = sorted(set(topology_resids))
    if not unique_resids:
        return None

    universe = mda.Universe(topology, str(centered_trajectory))
    universe.trajectory[-1]

    atom_indices: list[int] = []
    for residx in unique_resids:
        if 0 <= residx < len(universe.residues):
            atom_indices.extend(universe.residues[residx].atoms.indices.tolist())
    if not atom_indices:
        return None

    selection = universe.atoms[np.array(sorted(set(atom_indices)), dtype=int)]
    selection.write(str(output_pdb))
    if output_cif is not None:
        selection_topology, _selection_positions = load_openmm_structure_file(output_pdb)
        write_openmm_structure_file(
            selection_topology,
            unit.Quantity(np.asarray(selection.positions, dtype=float), unit.angstrom),
            output_cif,
        )
    return output_pdb


def write_final_ligand_sdf(
    *,
    topology,
    centered_trajectory: Path,
    prepared_ligand_sdf: Path | None,
    prepared_ligand_structure: Path | None,
    output_sdf: Path,
) -> Path | None:
    """Write a final-frame ligand SDF with bond orders preserved."""
    if (
        prepared_ligand_sdf is None
        or prepared_ligand_structure is None
        or not prepared_ligand_sdf.exists()
        or not prepared_ligand_structure.exists()
    ):
        return None

    from rdkit import Chem
    from rdkit.Geometry import Point3D
    import MDAnalysis as mda

    supplier = Chem.SDMolSupplier(str(prepared_ligand_sdf), removeHs=False)
    mol = next((candidate for candidate in supplier if candidate is not None), None)
    if mol is None:
        return None

    ligand_resnames = _structure_resnames(prepared_ligand_structure)
    if not ligand_resnames:
        return None

    universe = mda.Universe(topology, str(centered_trajectory))
    universe.trajectory[-1]
    selection_terms = " or ".join(f"resname {resname}" for resname in ligand_resnames)
    ligand_atoms = universe.select_atoms(selection_terms)
    if ligand_atoms.n_atoms != mol.GetNumAtoms():
        logger.warning(
            "Could not write final ligand SDF: ligand atom-count mismatch "
            "(trajectory=%d, template=%d)",
            ligand_atoms.n_atoms,
            mol.GetNumAtoms(),
        )
        return None

    conf = mol.GetConformer()
    for index, (x, y, z) in enumerate(np.asarray(ligand_atoms.positions, dtype=float)):
        conf.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()
    return output_sdf


def write_sites_pdb_from_csv(
    csv_path: Path,
    pdb_path: Path,
    shift_A: np.ndarray | None = None,
) -> Path:
    """Write H2OSite sites as a PDB, optionally translated by ``shift_A``."""
    shift = np.zeros(3, dtype=float) if shift_A is None else np.asarray(
        shift_A, dtype=float
    )

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    with open(pdb_path, "w") as handle:
        handle.write("REMARK H2OSite hydration sites\n")
        handle.write("REMARK B-factor: dG\n")
        handle.write("REMARK Occupancy: dH_total\n")
        if np.any(shift):
            handle.write(
                "REMARK Coordinates shifted by "
                f"[{shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f}] Angstrom\n"
            )

        for atom_idx, row in enumerate(rows, start=1):
            site_number = int(row["site_index"]) + 1
            xyz = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=float,
            ) + shift
            occupancy = float(row["dH_total"])
            bfactor = float(row["dG"])
            handle.write(
                _format_pdb_atom(
                    serial=atom_idx,
                    atom_name="O",
                    resname="WMP",
                    chain_id="W",
                    resid=site_number,
                    xyz=xyz,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element="O",
                )
                + "\n"
            )

        handle.write("END\n")

    return pdb_path


def write_sites_cif_from_csv(
    csv_path: Path,
    cif_path: Path,
    shift_A: np.ndarray | None = None,
) -> Path:
    """Write H2OSite sites as an mmCIF/PDBx file."""
    shift = np.zeros(3, dtype=float) if shift_A is None else np.asarray(
        shift_A, dtype=float
    )

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    cif_path.parent.mkdir(parents=True, exist_ok=True)
    data_block = cif_path.stem.replace("-", "_").replace(".", "_") or "watermap_sites"
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

    with open(cif_path, "w") as handle:
        handle.write(f"data_{data_block}\n#\n")
        handle.write("loop_\n")
        for column in columns:
            handle.write(f"_atom_site.{column}\n")

        for atom_idx, row in enumerate(rows, start=1):
            site_number = int(row["site_index"]) + 1
            xyz = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=float,
            ) + shift
            occupancy = float(row["dH_total"])
            bfactor = float(row["dG"])
            values = [
                "HETATM",
                str(atom_idx),
                "O",
                "O",
                ".",
                "WMP",
                "W",
                "1",
                str(site_number),
                "?",
                f"{xyz[0]:.3f}",
                f"{xyz[1]:.3f}",
                f"{xyz[2]:.3f}",
                f"{occupancy:.2f}",
                f"{bfactor:.2f}",
                "?",
                str(site_number),
                "WMP",
                "W",
                "O",
                "1",
            ]
            handle.write(" ".join(values) + "\n")
        handle.write("#\n")

    return cif_path


def write_fixed_site_pymol_script(
    output_dir: Path,
    jobname: str,
    prepared_protein: Path | None,
    prepared_ligand: Path | None,
    final_ligand_sdf: Path | None,
    site_radius_A: float,
) -> Path:
    """Write a PyMOL loader for corrected H2OSite sites in a holo-aware scene."""
    lines = ["# Corrected H2OSite sites and final centered frame"]
    ligand_resnames = _structure_resnames(prepared_ligand)
    ligand_selection = ""
    if ligand_resnames:
        ligand_selection = "+".join(ligand_resnames)

    solvated_final_cif = output_dir / "solvated_final.cif"
    centered_site_cif = output_dir / f"{jobname}_wm_fixed_centered.cif"
    centered_site_pdb = output_dir / f"{jobname}_wm_fixed_centered.pdb"
    has_centered_scene = solvated_final_cif.exists() and centered_site_cif.exists()

    if has_centered_scene:
        lines.extend(
            [
                "load solvated_final.cif, final_system",
                f"load {jobname}_wm_fixed_centered.cif, wm_sites_fixed",
                "hide everything, all",
                "show cartoon, final_system and polymer",
                "color gray80, final_system and polymer",
            ]
        )
        if ligand_selection:
            if final_ligand_sdf is not None and final_ligand_sdf.exists():
                try:
                    ligand_ref = final_ligand_sdf.relative_to(output_dir).as_posix()
                except ValueError:
                    ligand_ref = str(final_ligand_sdf)
                lines.extend(
                    [
                        f"hide everything, final_system and resn {ligand_selection}",
                        f"load {ligand_ref}, ligand_final",
                        "show sticks, ligand_final",
                        "util.cnc ligand_final",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"select ligand_sel, final_system and resn {ligand_selection}",
                        "show sticks, ligand_sel",
                        "util.cnc ligand_sel",
                    ]
                )
        lines.extend(
            [
                (
                    "select final_pocket_waters, "
                    "(final_system and resn HOH+WAT+TIP3+TIP4+SOL+T3P+T4P) "
                    f"within {site_radius_A:.1f} of wm_sites_fixed"
                ),
                "show spheres, final_pocket_waters",
                "set sphere_scale, 0.28, final_pocket_waters",
                "color red, final_pocket_waters and elem O",
                "color white, final_pocket_waters and elem H",
            ]
        )
    else:
        if prepared_protein is not None:
            try:
                prepared_ref = prepared_protein.relative_to(output_dir).as_posix()
            except ValueError:
                prepared_ref = str(prepared_protein)
            lines.extend(
                [
                    f"load {prepared_ref}, protein",
                    "show cartoon, protein",
                    "color gray80, protein",
                ]
            )
        if prepared_ligand is not None:
            try:
                ligand_ref = prepared_ligand.relative_to(output_dir).as_posix()
            except ValueError:
                ligand_ref = str(prepared_ligand)
            lines.extend(
                [
                    f"load {ligand_ref}, ligand",
                    "show sticks, ligand",
                    "util.cnc ligand",
                ]
            )
        if (output_dir / f"{jobname}_wm_fixed.cif").exists():
            lines.append(f"load {jobname}_wm_fixed.cif, wm_sites_fixed")
        else:
            lines.append(f"load {jobname}_wm_fixed.pdb, wm_sites_fixed")

    lines.extend(
        [
            "show spheres, wm_sites_fixed",
            "set sphere_scale, 0.55, wm_sites_fixed",
            "set sphere_transparency, 0.20, wm_sites_fixed",
            "spectrum b, green_yellow_red, wm_sites_fixed",
            'label wm_sites_fixed and name O, "WMP %s" % resi',
            "set label_size, 18, wm_sites_fixed",
            "set label_color, black, wm_sites_fixed",
            "zoom wm_sites_fixed, 8",
            "bg_color white",
            "set depth_cue, 0",
            "set fog, 0",
        ]
    )

    pml_path = output_dir / f"{jobname}_wm_fixed.pml"
    pml_path.write_text("\n".join(lines) + "\n")
    return pml_path


def write_centered_fixed_site_pdb(
    output_dir: Path,
    jobname: str,
    centered_tracking_center_A: np.ndarray,
    tracking_center_A: np.ndarray,
) -> Path:
    """Translate fixed sites into the centered trajectory frame."""
    fixed_csv = output_dir / f"{jobname}_wm_fixed.csv"
    centered_pdb = output_dir / f"{jobname}_wm_fixed_centered.pdb"
    shift = np.asarray(centered_tracking_center_A, dtype=float) - np.asarray(
        tracking_center_A,
        dtype=float,
    )
    write_sites_pdb_from_csv(fixed_csv, centered_pdb, shift_A=shift)
    return centered_pdb


def write_centered_fixed_site_cif(
    output_dir: Path,
    jobname: str,
    centered_tracking_center_A: np.ndarray,
    tracking_center_A: np.ndarray,
) -> Path:
    """Translate fixed sites into the centered trajectory frame as mmCIF."""
    fixed_csv = output_dir / f"{jobname}_wm_fixed.csv"
    centered_cif = output_dir / f"{jobname}_wm_fixed_centered.cif"
    shift = np.asarray(centered_tracking_center_A, dtype=float) - np.asarray(
        tracking_center_A,
        dtype=float,
    )
    write_sites_cif_from_csv(fixed_csv, centered_cif, shift_A=shift)
    return centered_cif


def write_original_frame_site_pdb(
    output_dir: Path,
    jobname: str,
    translation_vector_A: np.ndarray,
) -> Path:
    """Translate fixed sites back into the original input frame."""
    fixed_csv = output_dir / f"{jobname}_wm_fixed.csv"
    original_frame_pdb = output_dir / f"{jobname}_wm_fixed_input_frame.pdb"
    shift = -np.asarray(translation_vector_A, dtype=float)
    write_sites_pdb_from_csv(fixed_csv, original_frame_pdb, shift_A=shift)
    return original_frame_pdb


def write_original_frame_site_cif(
    output_dir: Path,
    jobname: str,
    translation_vector_A: np.ndarray,
) -> Path:
    """Translate fixed sites back into the original input frame as mmCIF."""
    fixed_csv = output_dir / f"{jobname}_wm_fixed.csv"
    original_frame_cif = output_dir / f"{jobname}_wm_fixed_input_frame.cif"
    shift = -np.asarray(translation_vector_A, dtype=float)
    write_sites_cif_from_csv(fixed_csv, original_frame_cif, shift_A=shift)
    return original_frame_cif


def write_original_protein_with_sites(
    original_protein_pdb: Path,
    site_csv: Path,
    output_pdb: Path,
    translation_vector_A: np.ndarray,
) -> Path:
    """Append corrected WMP sites to the original protein PDB frame."""
    shift = -np.asarray(translation_vector_A, dtype=float)
    site_rows = []
    with open(site_csv, newline="") as handle:
        site_rows = list(csv.DictReader(handle))

    lines = original_protein_pdb.read_text().splitlines()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    max_serial = 0
    last_structural_record = None
    kept_lines = []
    trailing_lines = []
    for line in lines:
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            try:
                max_serial = max(max_serial, int(line[6:11]))
            except ValueError:
                pass
            last_structural_record = record
        elif record == "TER":
            last_structural_record = record

        if record == "END":
            continue
        if record in {"CONECT", "MASTER"}:
            trailing_lines.append(line)
        else:
            kept_lines.append(line)

    with open(output_pdb, "w") as handle:
        for line in kept_lines:
            handle.write(f"{line}\n")

        if last_structural_record in {"ATOM", "HETATM"}:
            handle.write("TER\n")

        serial = max_serial + 1
        wmp_conect_lines = []
        for row in site_rows:
            site_number = int(row["site_index"]) + 1
            xyz = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=float,
            ) + shift
            occupancy = float(row["occupancy"])
            bfactor = float(row["dG"])
            dipole = np.array(
                [
                    float(row.get("dipole_x", 0.0)),
                    float(row.get("dipole_y", 0.0)),
                    float(row.get("dipole_z", 0.0)),
                ],
                dtype=float,
            )
            h1_xyz, h2_xyz = _water_hydrogen_positions(xyz, dipole)

            oxygen_serial = serial
            h1_serial = serial + 1
            h2_serial = serial + 2

            handle.write(
                _format_pdb_atom(
                    serial=oxygen_serial,
                    atom_name="O",
                    resname="WMP",
                    chain_id="W",
                    resid=site_number,
                    xyz=xyz,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element="O",
                )
                + "\n"
            )
            handle.write(
                _format_pdb_atom(
                    serial=h1_serial,
                    atom_name="H1",
                    resname="WMP",
                    chain_id="W",
                    resid=site_number,
                    xyz=h1_xyz,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element="H",
                )
                + "\n"
            )
            handle.write(
                _format_pdb_atom(
                    serial=h2_serial,
                    atom_name="H2",
                    resname="WMP",
                    chain_id="W",
                    resid=site_number,
                    xyz=h2_xyz,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element="H",
                )
                + "\n"
            )
            wmp_conect_lines.extend(
                [
                    f"CONECT{oxygen_serial:5d}{h1_serial:5d}{h2_serial:5d}",
                    f"CONECT{h1_serial:5d}{oxygen_serial:5d}",
                    f"CONECT{h2_serial:5d}{oxygen_serial:5d}",
                ]
            )
            serial += 3

        for line in wmp_conect_lines:
            handle.write(f"{line}\n")
        for line in trailing_lines:
            handle.write(f"{line}\n")
        handle.write("END\n")

    return output_pdb


def export_fixed_bundle(
    output_dir: Path,
    jobname: str,
    original_protein_pdb: Path | None,
    translation_vector_A: np.ndarray | None,
    prepared_ligand_pdb: Path | None,
) -> dict[str, Path]:
    """Copy fixed outputs into ``fixed/`` and add original-frame exports."""
    fixed_dir = ensure_fixed_dir(output_dir)
    exports: dict[str, Path] = {"fixed_dir": fixed_dir}

    root_fixed_files = [
        output_dir / f"{jobname}_wm_fixed.csv",
        output_dir / f"{jobname}_wm_fixed.pdb",
        output_dir / f"{jobname}_wm_fixed.cif",
        output_dir / f"{jobname}_wm_fixed_centered.pdb",
        output_dir / f"{jobname}_wm_fixed_centered.cif",
        output_dir / f"{jobname}_wm_fixed_input_frame.pdb",
        output_dir / f"{jobname}_wm_fixed_input_frame.cif",
        output_dir / f"{jobname}-water_fixed.cns",
        output_dir / f"{jobname}-continuous_cavity_fixed.cns",
        output_dir / f"{jobname}_wm_fixed.pml",
        output_dir / "view_watermap_traj.pml",
        output_dir / "view_watermap_traj_fixed.pml",
        output_dir / "solvated.pdb",
        output_dir / "solvated.cif",
        output_dir / "solvated_final.pdb",
        output_dir / "solvated_final.cif",
        output_dir / f"{jobname}_solvated_final_with_wm_fixed.pdb",
        output_dir / f"{jobname}_solvated_final_with_wm_fixed.cif",
        output_dir / f"{jobname}_drifted_waters.csv",
        output_dir / f"{jobname}_drifted_waters_final.pdb",
        output_dir / f"{jobname}_drifted_waters_final.cif",
        output_dir / f"{jobname}_ligand_final.sdf",
    ]
    for src in root_fixed_files:
        if src.exists():
            dst = fixed_dir / src.name
            shutil.copy2(src, dst)
            exports[src.name] = dst

    positions_src = output_dir / "production" / "water_positions_fixed.npy"
    if positions_src.exists():
        positions_dst = fixed_dir / positions_src.name
        shutil.copy2(positions_src, positions_dst)
        exports[positions_src.name] = positions_dst

    centered_traj = output_dir / "production" / "trajectory_centered.dcd"
    if centered_traj.exists():
        centered_dst = fixed_dir / centered_traj.name
        shutil.copy2(centered_traj, centered_dst)
        exports[centered_traj.name] = centered_dst

    if original_protein_pdb is not None:
        original_copy = fixed_dir / original_protein_pdb.name
        shutil.copy2(original_protein_pdb, original_copy)
        exports["original_protein"] = original_copy

        if translation_vector_A is not None:
            original_site_cif = output_dir / f"{jobname}_wm_fixed_input_frame.cif"
            if original_protein_pdb.suffix.lower() == ".pdb":
                merged_pdb = fixed_dir / f"{original_protein_pdb.stem}_with_wm_fixed.pdb"
                write_original_protein_with_sites(
                    original_protein_pdb=original_protein_pdb,
                    site_csv=output_dir / f"{jobname}_wm_fixed.csv",
                    output_pdb=merged_pdb,
                    translation_vector_A=translation_vector_A,
                )
                exports["merged_original_protein"] = merged_pdb
            if original_site_cif.exists():
                merged_cif = fixed_dir / f"{original_protein_pdb.stem}_with_wm_fixed.cif"
                write_original_protein_with_sites_cif(
                    original_protein=original_protein_pdb,
                    site_structure=original_site_cif,
                    output_cif=merged_cif,
                )
                exports["merged_original_protein_cif"] = merged_cif

    if prepared_ligand_pdb is not None and prepared_ligand_pdb.exists():
        ligand_copy = fixed_dir / prepared_ligand_pdb.name
        shutil.copy2(prepared_ligand_pdb, ligand_copy)
        exports["prepared_ligand"] = ligand_copy

    return exports


def write_centered_final_with_sites(
    *,
    output_dir: Path,
    jobname: str,
) -> Path | None:
    """Append centered H2OSite sites to the final centered solvated PDB."""
    final_pdb = output_dir / "solvated_final.pdb"
    centered_sites = output_dir / f"{jobname}_wm_fixed_centered.pdb"
    output_pdb = output_dir / f"{jobname}_solvated_final_with_wm_fixed.pdb"

    if not final_pdb.exists() or not centered_sites.exists():
        return None

    lines = final_pdb.read_text().splitlines()
    site_lines = [
        line.rstrip()
        for line in centered_sites.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    if not site_lines:
        return None

    max_serial = 0
    kept_lines: list[str] = []
    trailing_lines: list[str] = []
    for line in lines:
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            try:
                max_serial = max(max_serial, int(line[6:11]))
            except ValueError:
                pass
        if record == "END":
            continue
        if record in {"CONECT", "MASTER"}:
            trailing_lines.append(line)
        else:
            kept_lines.append(line)

    with open(output_pdb, "w") as handle:
        for line in kept_lines:
            handle.write(f"{line}\n")
        handle.write("TER\n")

        serial = max_serial + 1
        for line in site_lines:
            handle.write(f"{line[:6]}{serial:5d}{line[11:]}\n")
            serial += 1

        for line in trailing_lines:
            handle.write(f"{line}\n")
        handle.write("END\n")

    return output_pdb


def write_original_protein_with_sites_cif(
    *,
    original_protein: Path,
    site_structure: Path,
    output_cif: Path,
) -> Path | None:
    """Merge the original-frame protein structure with H2OSite sites as mmCIF."""
    import openmm.app as app

    if not original_protein.exists() or not site_structure.exists():
        return None

    protein_topology, protein_positions = load_openmm_structure_file(original_protein)
    site_topology, site_positions = load_openmm_structure_file(site_structure)
    modeller = app.Modeller(protein_topology, protein_positions)
    modeller.add(site_topology, site_positions)
    write_openmm_structure_file(modeller.topology, modeller.positions, output_cif)
    return output_cif


def write_centered_final_with_sites_cif(
    *,
    output_dir: Path,
    jobname: str,
    topology,
    centered_trajectory: Path,
) -> Path | None:
    """Append centered H2OSite sites to the final centered solvated mmCIF."""
    import MDAnalysis as mda
    import openmm.app as app
    import openmm.unit as unit

    centered_site_structure = output_dir / f"{jobname}_wm_fixed_centered.cif"
    output_cif = output_dir / f"{jobname}_solvated_final_with_wm_fixed.cif"

    if not centered_trajectory.exists() or not centered_site_structure.exists():
        return None

    universe = mda.Universe(topology, str(centered_trajectory))
    universe.trajectory[-1]
    final_positions = unit.Quantity(
        np.asarray(universe.atoms.positions, dtype=float),
        unit.angstrom,
    )
    site_topology, site_positions = load_openmm_structure_file(centered_site_structure)
    modeller = app.Modeller(topology, final_positions)
    modeller.add(site_topology, site_positions)
    write_openmm_structure_file(modeller.topology, modeller.positions, output_cif)
    return output_cif


def build_view_pymol_script(
    output_dir: Path,
    jobname: str,
    active_site_centered_A: np.ndarray,
    site_radius_A: float,
    nearby_radius_A: float,
    prepared_ligand: Path | None,
    final_ligand_sdf: Path | None,
    has_drifted_waters: bool,
) -> str:
    """Return the full PyMOL scene used for centered trajectory inspection."""
    x, y, z = [float(val) for val in active_site_centered_A]
    centered_site_cif = f"{jobname}_wm_fixed_centered.cif"
    ligand_resnames = _structure_resnames(prepared_ligand)
    ligand_selection = "+".join(ligand_resnames)

    lines = [
        "# H2OSite trajectory visualization - centered trajectory + corrected sites",
        "# Generated by H2OSite.postprocessing",
        "",
    ]

    lines.extend(
        [
            "# Centered solvated system + trajectory for dynamic waters",
            "load solvated.cif, system",
            "load_traj production/trajectory_centered.dcd, system",
            "frame 1",
            "",
        ]
    )
    protein_object = "system and polymer"
    lines.extend(
        [
            "# Active-site marker in centered coordinates",
            f"pseudoatom active_center, pos=[{x:.6f}, {y:.6f}, {z:.6f}]",
            "",
            "# Corrected H2OSite hydration sites",
            f"load {centered_site_cif}, wm_sites_fixed",
            "",
            "# Selections",
            f"select protein_sel, {protein_object}",
            (
                "select active_waters, "
                "(system and resn HOH+WAT+TIP3+TIP4+SOL+T3P+T4P) within "
                f"{site_radius_A:.1f} of active_center"
            ),
            (
                f"select nearby_residues, byres ({protein_object} within "
                f"{nearby_radius_A:.1f} of wm_sites_fixed)"
            ),
            "",
            "# Scene",
            "hide everything, all",
            "show cartoon, protein_sel",
            "color gray80, protein_sel",
            "",
            "# Nearby residues as sticks with clean labels",
            "show sticks, nearby_residues",
            "util.cnc nearby_residues",
            'label nearby_residues and name CA, "%s%s/%s" % (resn, resi, chain)',
            "set label_size, 16, nearby_residues",
            "set label_color, black, nearby_residues",
            "",
            "# Dynamic waters from the centered trajectory",
            "show spheres, active_waters",
            "set sphere_scale, 0.3, active_waters",
            "color red, active_waters and elem O",
            "color white, active_waters and elem H",
            "",
        ]
    )

    if ligand_selection:
        if final_ligand_sdf is not None and final_ligand_sdf.exists():
            try:
                ligand_ref = final_ligand_sdf.relative_to(output_dir).as_posix()
            except ValueError:
                ligand_ref = str(final_ligand_sdf)
            lines.extend(
                [
                    "# Ligand retained in holo runs",
                    f"hide everything, system and resn {ligand_selection}",
                    f"load {ligand_ref}, ligand_final",
                    "show sticks, ligand_final",
                    "util.cnc ligand_final",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "# Ligand retained in holo runs",
                    f"select ligand_sel, system and resn {ligand_selection}",
                    "show sticks, ligand_sel",
                    "util.cnc ligand_sel",
                    "",
                ]
            )

    if has_drifted_waters:
        lines.extend(
            [
                "# Waters that drifted out of the analysis sphere",
                f"load {jobname}_drifted_waters_final.cif, drifted_waters_final",
                "show spheres, drifted_waters_final",
                "set sphere_scale, 0.35, drifted_waters_final",
                "color cyan, drifted_waters_final",
                "",
            ]
        )

    lines.extend(
        [
            "# Active-site marker",
            "show spheres, active_center",
            "set sphere_scale, 0.5, active_center",
            "color yellow, active_center",
            "",
            "# Corrected H2OSite site overlay colored by dG (stored in B-factor)",
            "show spheres, wm_sites_fixed",
            "set sphere_scale, 0.55, wm_sites_fixed",
            "set sphere_transparency, 0.20, wm_sites_fixed",
            "spectrum b, green_yellow_red, wm_sites_fixed",
            'label wm_sites_fixed and name O, "WMP %s" % resi',
            "set label_size, 18, wm_sites_fixed",
            "set label_color, black, wm_sites_fixed",
            "",
            "# Camera and display",
            "zoom wm_sites_fixed, 15",
            "bg_color white",
            "set ray_shadow, 0",
            "set depth_cue, 0",
            "set fog, 0",
            "",
            "# Trajectory playback",
            "set movie_fps, 15",
            "smooth",
        ]
    )

    return "\n".join(lines) + "\n"


def write_pymol_scripts(
    output_dir: Path,
    jobname: str,
    active_site_centered_A: np.ndarray,
    site_radius_A: float,
    nearby_radius_A: float,
    prepared_ligand: Path | None,
    final_ligand_sdf: Path | None,
    has_drifted_waters: bool,
) -> tuple[Path, Path]:
    """Write the centered trajectory PyMOL scene scripts."""
    pml_text = build_view_pymol_script(
        output_dir=output_dir,
        jobname=jobname,
        active_site_centered_A=active_site_centered_A,
        site_radius_A=site_radius_A,
        nearby_radius_A=nearby_radius_A,
        prepared_ligand=prepared_ligand,
        final_ligand_sdf=final_ligand_sdf,
        has_drifted_waters=has_drifted_waters,
    )

    main_pml = output_dir / "view_watermap_traj.pml"
    fixed_pml = output_dir / "view_watermap_traj_fixed.pml"
    main_pml.write_text(pml_text)
    fixed_pml.write_text(pml_text)
    return main_pml, fixed_pml


def generate_display_artifacts(
    *,
    output_dir: str | Path,
    jobname: str,
    topology,
    trajectory_file: str | Path,
    tracking_center_A: np.ndarray,
    site_radius_A: float,
    nearby_radius_A: float = 5.0,
    reanalyze: bool = False,
    hs_radius_A: float = 1.0,
    cutoff_A: float = 2.0,
    checkpoint_file: str | Path | None = None,
    centered_active_site_A: np.ndarray | None = None,
    original_protein_path: str | Path | None = None,
    water_model: str = "tip3p",
) -> dict[str, Path]:
    """Generate display artifacts, optionally recomputing analysis from raw MD.

    By default, existing analysis is copied into the fixed bundle. With
    ``reanalyze=True``, fixed outputs come from new clustering, thermodynamics,
    and H-bond analysis instead. Original analysis files remain unchanged.
    """
    output_dir = Path(output_dir)
    trajectory_file = Path(trajectory_file)

    raw_csv = output_dir / f"{jobname}_wm.csv"
    if not reanalyze and not raw_csv.exists():
        raise FileNotFoundError(f"Missing WaterMap CSV: {raw_csv}")
    original_protein = (
        Path(original_protein_path).expanduser().resolve()
        if original_protein_path is not None else find_original_input_pdb(output_dir)
    )
    if original_protein is not None and not original_protein.is_file():
        raise FileNotFoundError(original_protein)

    if reanalyze:
        run_corrected_reanalysis(
            output_dir, jobname, tracking_center_A,
            site_radius_A=site_radius_A, hs_radius_A=hs_radius_A, cutoff_A=cutoff_A,
            checkpoint_file=checkpoint_file, trajectory_file=trajectory_file,
            water_model=water_model,
        )

    production_dir = output_dir / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    centered_dcd = production_dir / "trajectory_centered.dcd"
    solvated_pdb = output_dir / "solvated.pdb"
    solvated_cif = output_dir / "solvated.cif"
    solvated_final_pdb = output_dir / "solvated_final.pdb"
    solvated_final_cif = output_dir / "solvated_final.cif"
    centered_tracking_center_A = write_centered_trajectory(
        topology=topology,
        trajectory_file=trajectory_file,
        centered_dcd=centered_dcd,
        solvated_pdb=solvated_pdb,
        solvated_cif=solvated_cif,
        solvated_final_pdb=solvated_final_pdb,
        solvated_final_cif=solvated_final_cif,
        tracking_center_A=np.asarray(tracking_center_A, dtype=float),
    )
    if centered_active_site_A is not None:
        centered_tracking_center_A = np.asarray(centered_active_site_A, dtype=float)

    fixed_csv = output_dir / f"{jobname}_wm_fixed.csv"

    raw_water_cns = output_dir / f"{jobname}-water.cns"
    fixed_water_cns = output_dir / f"{jobname}-water_fixed.cns"

    raw_cavity_cns = output_dir / f"{jobname}-continuous_cavity.cns"
    fixed_cavity_cns = output_dir / f"{jobname}-continuous_cavity_fixed.cns"

    water_positions = production_dir / "water_positions.npy"
    fixed_positions = production_dir / "water_positions_fixed.npy"
    if not reanalyze:
        shutil.copy2(raw_csv, fixed_csv)
        for source, target in (
            (raw_water_cns, fixed_water_cns),
            (raw_cavity_cns, fixed_cavity_cns),
            (water_positions, fixed_positions),
        ):
            if source.exists():
                shutil.copy2(source, target)

    fixed_pdb = output_dir / f"{jobname}_wm_fixed.pdb"
    fixed_cif = output_dir / f"{jobname}_wm_fixed.cif"
    write_sites_pdb_from_csv(fixed_csv, fixed_pdb)
    write_sites_cif_from_csv(fixed_csv, fixed_cif)

    centered_fixed_pdb = write_centered_fixed_site_pdb(
        output_dir=output_dir,
        jobname=jobname,
        centered_tracking_center_A=centered_tracking_center_A,
        tracking_center_A=np.asarray(tracking_center_A, dtype=float),
    )
    centered_fixed_cif = write_centered_fixed_site_cif(
        output_dir=output_dir,
        jobname=jobname,
        centered_tracking_center_A=centered_tracking_center_A,
        tracking_center_A=np.asarray(tracking_center_A, dtype=float),
    )

    prepared_protein = find_prepared_protein(output_dir)
    prepared_ligand = find_prepared_ligand(output_dir)
    prepared_ligand_sdf = find_prepared_ligand_sdf(output_dir)
    translation_vector_A = load_translation_vector(output_dir, jobname)

    original_frame_pdb = None
    original_frame_cif = None
    if translation_vector_A is not None:
        original_frame_pdb = write_original_frame_site_pdb(
            output_dir=output_dir,
            jobname=jobname,
            translation_vector_A=translation_vector_A,
        )
        original_frame_cif = write_original_frame_site_cif(
            output_dir=output_dir,
            jobname=jobname,
            translation_vector_A=translation_vector_A,
        )

    drift_rows = load_drift_report(output_dir, jobname)
    drifted_waters_final_pdb = None
    drifted_waters_final_cif = None
    if drift_rows:
        drifted_waters_final_pdb = write_drifted_waters_final_pdb(
            topology=topology,
            centered_trajectory=centered_dcd,
            drift_rows=drift_rows,
            output_pdb=output_dir / f"{jobname}_drifted_waters_final.pdb",
            output_cif=output_dir / f"{jobname}_drifted_waters_final.cif",
        )
        drifted_waters_final_cif = output_dir / f"{jobname}_drifted_waters_final.cif"

    final_ligand_sdf = write_final_ligand_sdf(
        topology=topology,
        centered_trajectory=centered_dcd,
        prepared_ligand_sdf=prepared_ligand_sdf,
        prepared_ligand_structure=prepared_ligand,
        output_sdf=output_dir / f"{jobname}_ligand_final.sdf",
    )

    fixed_pml = write_fixed_site_pymol_script(
        output_dir=output_dir,
        jobname=jobname,
        prepared_protein=prepared_protein,
        prepared_ligand=prepared_ligand,
        final_ligand_sdf=final_ligand_sdf,
        site_radius_A=site_radius_A,
    )
    view_pml, view_fixed_pml = write_pymol_scripts(
        output_dir=output_dir,
        jobname=jobname,
        active_site_centered_A=centered_tracking_center_A,
        site_radius_A=site_radius_A,
        nearby_radius_A=nearby_radius_A,
        prepared_ligand=prepared_ligand,
        final_ligand_sdf=final_ligand_sdf,
        has_drifted_waters=drifted_waters_final_pdb is not None,
    )
    centered_final_with_sites = write_centered_final_with_sites(
        output_dir=output_dir,
        jobname=jobname,
    )
    centered_final_with_sites_cif = write_centered_final_with_sites_cif(
        output_dir=output_dir,
        jobname=jobname,
        topology=topology,
        centered_trajectory=centered_dcd,
    )

    bundle = export_fixed_bundle(
        output_dir=output_dir,
        jobname=jobname,
        original_protein_pdb=original_protein,
        translation_vector_A=translation_vector_A,
        prepared_ligand_pdb=prepared_ligand,
    )

    artifacts: dict[str, Path] = {
        "solvated_pdb": solvated_pdb,
        "solvated_cif": solvated_cif,
        "solvated_final_pdb": solvated_final_pdb,
        "solvated_final_cif": solvated_final_cif,
        "centered_dcd": centered_dcd,
        "fixed_csv": fixed_csv,
        "fixed_pdb": fixed_pdb,
        "fixed_cif": fixed_cif,
        "centered_fixed_pdb": centered_fixed_pdb,
        "centered_fixed_cif": centered_fixed_cif,
        "fixed_site_pml": fixed_pml,
        "view_pml": view_pml,
        "view_fixed_pml": view_fixed_pml,
    }
    if fixed_water_cns.exists():
        artifacts["fixed_water_cns"] = fixed_water_cns
    if fixed_cavity_cns.exists():
        artifacts["fixed_cavity_cns"] = fixed_cavity_cns
    if fixed_positions.exists():
        artifacts["fixed_positions"] = fixed_positions
    if original_frame_pdb is not None:
        artifacts["original_frame_pdb"] = original_frame_pdb
    if original_frame_cif is not None:
        artifacts["original_frame_cif"] = original_frame_cif
    if drifted_waters_final_pdb is not None:
        artifacts["drifted_waters_final_pdb"] = drifted_waters_final_pdb
    if drifted_waters_final_cif is not None and drifted_waters_final_cif.exists():
        artifacts["drifted_waters_final_cif"] = drifted_waters_final_cif
    if centered_final_with_sites is not None:
        artifacts["solvated_final_with_wm_fixed_pdb"] = centered_final_with_sites
    if centered_final_with_sites_cif is not None:
        artifacts["solvated_final_with_wm_fixed_cif"] = centered_final_with_sites_cif
    if final_ligand_sdf is not None:
        artifacts["final_ligand_sdf"] = final_ligand_sdf
    artifacts.update(bundle)

    logger.info(
        "Generated centered display artifacts for %s at %s",
        jobname,
        output_dir,
    )
    return artifacts


def main(argv: list[str] | None = None) -> None:
    """Postprocess a saved run, optionally repeating the analysis stages."""
    parser = argparse.ArgumentParser(
        description="Generate WaterMap display exports, optionally reanalyzing raw MD."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reanalyze", action="store_true",
                        help="Recompute clustering, thermodynamics and H-bonds into fixed outputs.")
    parser.add_argument("--jobname", help="Job stem (inferred from existing results/prep by default).")
    parser.add_argument("--checkpoint", type=Path, help="Pipeline stage checkpoint (.npz).")
    parser.add_argument("--trajectory", type=Path, help="Raw production trajectory.")
    parser.add_argument("--tracking-center", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"), help="Raw-frame tracking center in Angstrom.")
    parser.add_argument("--active-site", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Override the centered display/site-overlay coordinates in Angstrom.")
    parser.add_argument("--site-radius", type=float, default=10.0)
    parser.add_argument("--nearby-radius", type=float, default=5.0)
    parser.add_argument("--hs-radius", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=2.0)
    parser.add_argument("--water-model", default="tip3p",
                        help="Water model used during production (for reanalysis).")
    parser.add_argument("--original-protein-pdb", type=Path,
                        help="Original input protein for the fixed bundle (PDB or mmCIF).")
    args = parser.parse_args(argv)
    for name in ("site_radius", "nearby_radius", "hs_radius", "cutoff"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not np.all(np.isfinite(args.tracking_center)) or (
        args.active_site is not None and not np.all(np.isfinite(args.active_site))
    ):
        parser.error("Center coordinates must be finite")

    from H2OSite.pipeline import WaterMapPipeline

    output_dir = args.output_dir.expanduser().resolve()
    try:
        checkpoint = args.checkpoint or choose_checkpoint(output_dir)
        trajectory = args.trajectory or find_production_trajectory(output_dir)
        for path in (checkpoint, trajectory):
            if not path.is_file():
                raise FileNotFoundError(path)
        payload = WaterMapPipeline._deserialize_checkpoint(checkpoint)
        artifacts = generate_display_artifacts(
            output_dir=output_dir,
            jobname=args.jobname or infer_jobname(output_dir),
            topology=payload["topology"],
            trajectory_file=trajectory,
            tracking_center_A=np.asarray(args.tracking_center, dtype=float),
            site_radius_A=args.site_radius,
            nearby_radius_A=args.nearby_radius,
            reanalyze=args.reanalyze,
            hs_radius_A=args.hs_radius,
            cutoff_A=args.cutoff,
            checkpoint_file=checkpoint,
            centered_active_site_A=args.active_site,
            original_protein_path=args.original_protein_pdb,
            water_model=args.water_model,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(artifacts)} artifacts in {output_dir}")
    print(f"Fixed bundle: {artifacts['fixed_dir']}")

if __name__ == "__main__":
    main()

