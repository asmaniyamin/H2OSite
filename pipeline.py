# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""H2OSite pipeline orchestrator.

Coordinates input preparation, system building, equilibration, GCMC,
production MD, clustering, thermodynamics, H-bonds, and density/cavity outputs.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np

from H2OSite.core.config import (
    WaterMapConfig,
    get_gcmc_params,
)

logger = logging.getLogger(__name__)


class WaterMapPipeline:
    """End-to-end H2OSite pipeline."""

    def __init__(self, config: WaterMapConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sites = []
        self._timings: dict[str, float] = {}
        self._last_gcmc_stats: dict | None = None
        self._checkpoint_dir = self.output_dir / "checkpoints"
        self._water_drift_reports: list[dict[str, object]] = []
        self._last_stage7_pocket_snapshot: dict[str, object] | None = None

    # ------------------------------------------------------------------
    # Stage checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, stage: str, **data) -> None:
        """Save stage outputs so the pipeline can resume after a crash."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._checkpoint_dir / f"{stage}.npz"
        self._serialize_checkpoint(ckpt_path, **data)
        logger.info("Checkpoint saved: %s", stage)


    def _load_checkpoint(self, stage: str) -> dict | None:
        """Load a stage checkpoint. Returns None if not found."""
        ckpt_path = self._checkpoint_dir / f"{stage}.npz"
        if not ckpt_path.exists():
            return None

        result = self._deserialize_checkpoint(ckpt_path)
        logger.info("Checkpoint loaded: %s", stage)
        return result


    def run(self) -> list:
        """Execute the full pipeline. Returns list of HydrationSite."""
        from H2OSite.simulation.protocol import SimulationProtocol

        t0 = time.time()
        logger.info("=== H2OSite Pipeline ===")
        logger.info("Job: %s", self.config.jobname)

        # --- Try to resume from the latest checkpoint ---
        ckpt = self._load_checkpoint("post_gcmc_equil")
        if ckpt is not None:
            logger.info("Resuming after post-GCMC equilibration")
            topology = ckpt["topology"]
            system = ckpt["system"]
            eq_pos = ckpt["eq_pos"]
            eq_vel = ckpt["eq_vel"]
            eq_box = ckpt.get("eq_box")
            if eq_box is None and hasattr(topology, "getPeriodicBoxVectors"):
                eq_box = topology.getPeriodicBoxVectors()
            heavy_indices = list(ckpt["heavy_indices"])
            self._last_gcmc_stats = ckpt.get("gcmc_stats")
        else:
            ckpt = self._load_checkpoint("gcmc")
            if ckpt is not None:
                logger.info("Resuming after GCMC")
                topology = ckpt["topology"]
                system = ckpt["system"]
                eq_pos = ckpt["eq_pos"]
                ff = ckpt["ff"]
                self._last_gcmc_stats = ckpt.get("gcmc_stats")
                heavy_indices, protein_indices = self._classify_atoms(topology)

                # Stage 8: Post-GCMC re-equilibration
                logger.info("--- Stage 8: Post-GCMC re-equilibration ---")
                if self.config.run_post_gcmc_equilibration:
                    post_protocol = SimulationProtocol(
                        topology, eq_pos, system,
                        output_dir=self.output_dir / "post_gcmc_equil",
                        gpu_index=self.config.gpu_index,
                    )
                    eq_pos, eq_vel, eq_box = post_protocol.run_equilibration(
                        self.config.post_gcmc_equil_stages,
                        protein_indices=protein_indices,
                        heavy_atom_indices=heavy_indices,
                        use_respa=self.config.use_respa,
                    )
                    if eq_box is not None and hasattr(topology, "setPeriodicBoxVectors"):
                        topology.setPeriodicBoxVectors(eq_box)
                else:
                    eq_vel = None
                    eq_box = (
                        topology.getPeriodicBoxVectors()
                        if hasattr(topology, "getPeriodicBoxVectors")
                        else None
                    )

                self._save_checkpoint(
                    "post_gcmc_equil",
                    topology=topology, system=system,
                    eq_pos=eq_pos, eq_vel=eq_vel,
                    eq_box=eq_box,
                    heavy_indices=heavy_indices,
                    gcmc_stats=self._last_gcmc_stats,
                )
            else:
                ckpt = self._load_checkpoint("equilibration")
                if ckpt is not None:
                    logger.info("Resuming after equilibration")
                    topology = ckpt["topology"]
                    system = ckpt["system"]
                    ff = ckpt["ff"]
                    eq_pos = ckpt["eq_pos"]
                    eq_vel = ckpt["eq_vel"]
                    eq_box = ckpt["eq_box"]
                else:
                    # Stage 1-2: Prepare structures and build system
                    logger.info("--- Stage 1-2: Prepare structures & build system ---")
                    topology, positions, system, ff = self._prepare_and_build()

                    # Identify atom groups for restraints
                    heavy_indices, protein_indices = self._classify_atoms(topology)

                    # Stage 3-6: Equilibration
                    logger.info("--- Stage 3-6: Equilibration ---")
                    protocol = SimulationProtocol(
                        topology, positions, system,
                        output_dir=self.output_dir / "equilibration",
                        gpu_index=self.config.gpu_index,
                    )
                    eq_pos, eq_vel, eq_box = protocol.run_equilibration(
                        self.config.equilibration_stages,
                        protein_indices=protein_indices,
                        heavy_atom_indices=heavy_indices,
                        use_respa=self.config.use_respa,
                    )
                    if eq_box is not None and hasattr(topology, "setPeriodicBoxVectors"):
                        topology.setPeriodicBoxVectors(eq_box)
                    self._timings["equilibration"] = time.time() - t0

                    self._save_checkpoint(
                        "equilibration",
                        topology=topology, system=system, ff=ff,
                        eq_pos=eq_pos, eq_vel=eq_vel, eq_box=eq_box,
                    )

                # Identify atom groups (needed if loaded from equil checkpoint)
                heavy_indices, protein_indices = self._classify_atoms(topology)

                # Stage 7: GCMC (optional)
                gcmc_ran = False
                if not self.config.skip_gcmc:
                    logger.info("--- Stage 7: GCMC water insertion ---")
                    topology, system, eq_pos = self._run_gcmc(
                        topology, system, eq_pos, ff, eq_box
                    )
                    heavy_indices, protein_indices = self._classify_atoms(topology)
                    gcmc_ran = True

                    self._save_checkpoint(
                        "gcmc",
                        topology=topology, system=system, ff=ff,
                        eq_pos=eq_pos,
                        gcmc_stats=self._last_gcmc_stats,
                    )

                # Stage 8: Post-GCMC re-equilibration
                logger.info("--- Stage 8: Post-GCMC re-equilibration ---")
                if self.config.run_post_gcmc_equilibration and (
                    gcmc_ran or self.config.run_post_gcmc_without_gcmc
                ):
                    post_protocol = SimulationProtocol(
                        topology, eq_pos, system,
                        output_dir=self.output_dir / "post_gcmc_equil",
                        gpu_index=self.config.gpu_index,
                    )
                    eq_pos, eq_vel, eq_box = post_protocol.run_equilibration(
                        self.config.post_gcmc_equil_stages,
                        protein_indices=protein_indices,
                        heavy_atom_indices=heavy_indices,
                        use_respa=self.config.use_respa,
                    )
                    if eq_box is not None and hasattr(topology, "setPeriodicBoxVectors"):
                        topology.setPeriodicBoxVectors(eq_box)
                    if gcmc_ran and self._last_stage7_pocket_snapshot is not None:
                        stage8_snapshot = self._collect_pocket_water_snapshot(
                            topology,
                            eq_pos,
                            label="Stage 8 post-GCMC equilibration",
                            box_vectors=eq_box,
                        )
                        self._record_pocket_water_drift(
                            "Stage 8 post-GCMC equilibration",
                            self._last_stage7_pocket_snapshot,
                            stage8_snapshot,
                        )
                        self._last_stage7_pocket_snapshot = stage8_snapshot
                else:
                    eq_box = (
                        topology.getPeriodicBoxVectors()
                        if hasattr(topology, "getPeriodicBoxVectors")
                        else None
                    )

                self._save_checkpoint(
                    "post_gcmc_equil",
                    topology=topology, system=system,
                    eq_pos=eq_pos, eq_vel=eq_vel,
                    eq_box=eq_box,
                    heavy_indices=heavy_indices,
                    gcmc_stats=self._last_gcmc_stats,
                )

        # Stage 9: Production MD
        logger.info("--- Stage 9: Production MD (%.1f ns) ---",
                     self.config.simulation_time_ns)
        tracking_center_nm, tracking_center_source = (
            self._resolve_tracking_center_nm(eq_pos, box_vectors=eq_box)
        )
        tracker = self._run_production(
            topology, system, eq_pos, eq_vel,
            restrain_indices=heavy_indices,
            active_site_center_nm=tracking_center_nm,
            active_site_center_source=tracking_center_source,
            box_vectors=eq_box,
        )
        if getattr(tracker, "drift_events", None):
            self._water_drift_reports.extend(list(tracker.drift_events))
        self._timings["production"] = time.time() - t0

        # Stage 10: Clustering
        logger.info("--- Stage 10: Clustering ---")
        from H2OSite.analysis.clustering import HydrationSiteClustering

        clusterer = HydrationSiteClustering(
            hs_radius_A=self.config.hs_radius_A,
            dcutoff_A=self.config.clustering_cutoff_A,
            grid_spacing_A=self.config.grid_spacing_A,
            water_model=self.config.water_model,
        )
        per_frame = tracker.get_per_frame_positions()
        self.sites = clusterer.cluster(
            tracker.get_all_positions(),
            tracker.n_frames,
            per_frame_positions=per_frame,
        )
        self._timings["clustering"] = time.time() - t0

        # Stage 11b: thermodynamics + H-bonds from the production trajectory.
        logger.info("--- Stage 11b: Thermodynamics & H-bonds ---")

        # Compute dipoles, interaction energies, and H-bonds from sampled waters.
        per_frame_dipoles, per_frame_energies = self._extract_stage11b_data(
            topology, system, tracker
        )
        self._compute_thermodynamics(
            per_frame,
            per_frame_dipoles=per_frame_dipoles,
            per_frame_energies=per_frame_energies,
        )
        self._compute_hbonds(topology, tracker)

        # Stage 12: Generate output
        logger.info("--- Stage 12: Generate output ---")
        self._generate_output(per_frame)
        self._write_water_drift_report()
        if self.config.generate_display_outputs:
            self._generate_display_outputs(topology, tracker)
        self._timings["total"] = time.time() - t0

        logger.info("Pipeline complete in %.1f s — %d hydration sites",
                     self._timings["total"], len(self.sites))
        return self.sites

    def run_analysis_only(
        self,
        water_positions_npy: str | Path,
        n_frames: int | None = None,
        topology_file: str | None = None,
        trajectory_file: str | None = None,
    ) -> list:
        """Run only the analysis stages on pre-computed water positions.

        Useful when MD was run externally, such as with AMBER.
        """
        logger.info("=== Analysis-only mode ===")
        positions = np.load(water_positions_npy)

        # If per-frame data is stored as (n_frames, n_waters, 3)
        if positions.ndim == 3:
            per_frame = [positions[i] for i in range(positions.shape[0])]
            all_pos = positions.reshape(-1, 3)
            if n_frames is None:
                n_frames = positions.shape[0]
        else:
            per_frame = None
            all_pos = positions
            if n_frames is None:
                n_frames = 1

        from H2OSite.analysis.clustering import HydrationSiteClustering

        clusterer = HydrationSiteClustering(
            hs_radius_A=self.config.hs_radius_A,
            dcutoff_A=self.config.clustering_cutoff_A,
            grid_spacing_A=self.config.grid_spacing_A,
            water_model=self.config.water_model,
        )
        self.sites = clusterer.cluster(all_pos, n_frames, per_frame)

        if per_frame:
            self._compute_thermodynamics(per_frame)

        if topology_file and trajectory_file:
            from H2OSite.analysis.hbond import HBondAnalyzer
            hba = HBondAnalyzer(hs_radius_A=self.config.hs_radius_A)
            self.sites = hba.analyze_trajectory(
                topology_file, trajectory_file, self.sites
            )

        self._generate_output(per_frame or [all_pos])
        return self.sites

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_and_build(self):
        from H2OSite.prep.input_preparation import WaterMapInput
        from H2OSite.prep.system_builder import SystemBuilder

        wm_input = WaterMapInput(self.config)
        wm_input.prepare()
        prot_pdb, lig_pdb = wm_input.write_prepared(self.output_dir / "prep")
        protein_structure_path = (
            wm_input.prepared_protein_cif_path
            if getattr(wm_input, "prepared_protein_cif_path", None)
            and wm_input.prepared_protein_cif_path.exists()
            else prot_pdb
        )
        # Use the RDKit-produced PDB (not CIF) for the ligand when it will be
        # retained in the simulation. app.PDBxFile does not write CONECT-style
        # bonds for unknown residues such as UNK, so the ligand topology loaded
        # from a CIF has no bonds - SMIRNOFFTemplateGenerator then cannot match
        # it. The ligand PDB written by Chem.MolToPDBFile includes CONECT
        # records and loads with full bond topology.
        if self.config.retain_ligand and lig_pdb.exists():
            ligand_structure_path = lig_pdb
        elif getattr(wm_input, "prepared_ligand_cif_path", None) and wm_input.prepared_ligand_cif_path.exists():
            ligand_structure_path = wm_input.prepared_ligand_cif_path
        else:
            ligand_structure_path = lig_pdb

        builder = SystemBuilder(
            protein_pdb=protein_structure_path,
            # When retain_ligand=False (apo H2OSite), the ligand is used for
            # centering / truncation during prep but excluded from the
            # simulation system.
            ligand_pdb=ligand_structure_path if self.config.retain_ligand else None,
            ligand_sdf=(
                wm_input.prepared_ligand_sdf_path
                if self.config.retain_ligand
                else None
            ),
            water_model=self.config.water_model,
            forcefield=self.config.forcefield,
            ligand_forcefield=self.config.ligand_forcefield,
            buffer_A=self.config.ligand_distance_A,

            neutralize=self.config.neutralize_system,
            terminal_treatment=self.config.terminal_treatment,
            ligand_parameter_cache=(
                self.config.ligand_parameter_cache
                or (self._checkpoint_dir / "openff_ligand_cache.json")
            ),
        )
        return builder.build()

    @staticmethod
    def _positions_to_nm(positions) -> np.ndarray:
        import openmm.unit as unit

        if hasattr(positions, "value_in_unit"):
            return np.asarray(
                positions.value_in_unit(unit.nanometer),
                dtype=float,
            )
        return np.asarray(positions, dtype=float)

    @staticmethod
    def _box_lengths_nm(box_vectors) -> np.ndarray | None:
        import openmm.unit as unit

        if box_vectors is None:
            return None

        if hasattr(box_vectors, "value_in_unit"):
            box = np.asarray(
                box_vectors.value_in_unit(unit.nanometer),
                dtype=float,
            )
        else:
            try:
                box = np.asarray(box_vectors, dtype=float)
            except Exception:
                return None

        if box.shape == (3, 3):
            lengths = np.array([box[0, 0], box[1, 1], box[2, 2]], dtype=float)
        elif box.shape == (3,):
            lengths = box.astype(float, copy=False)
        else:
            return None

        if np.any(lengths <= 0):
            return None
        return lengths

    @staticmethod
    def _canonicalize_periodic_point_nm(
        point_nm: np.ndarray,
        box_lengths_nm: np.ndarray | None,
    ) -> np.ndarray:
        point_nm = np.asarray(point_nm, dtype=float)
        if box_lengths_nm is None:
            return point_nm
        return point_nm - box_lengths_nm * np.round(point_nm / box_lengths_nm)

    @staticmethod
    def _unwrap_periodic_points_nm(
        points_nm: np.ndarray,
        box_lengths_nm: np.ndarray | None,
    ) -> np.ndarray:
        points_nm = np.asarray(points_nm, dtype=float)
        if box_lengths_nm is None or len(points_nm) <= 1:
            return points_nm

        unwrapped = points_nm.copy()
        anchor = unwrapped[0].copy()
        for idx in range(1, len(unwrapped)):
            delta = unwrapped[idx] - anchor
            delta -= box_lengths_nm * np.round(delta / box_lengths_nm)
            unwrapped[idx] = anchor + delta
        return unwrapped

    def _prepared_ligand_centroid_nm(self) -> np.ndarray | None:
        import openmm.unit as unit
        from H2OSite.io.structure_io import load_openmm_structure_file

        prep_dir = self.output_dir / "prep"
        ligands = sorted(prep_dir.glob("*_ligand.cif")) + sorted(
            prep_dir.glob("*_ligand.pdb")
        )
        if not ligands:
            return None

        _topology, positions = load_openmm_structure_file(ligands[0])
        if len(positions) == 0:
            return None

        positions_A = np.asarray(
            positions.value_in_unit(unit.angstrom),
            dtype=float,
        )
        return 0.1 * np.mean(positions_A, axis=0)

    def _resolve_tracking_center_nm(
        self,
        positions,
        box_vectors=None,
    ) -> tuple[np.ndarray, str]:
        positions_nm = self._positions_to_nm(positions)
        box_lengths_nm = self._box_lengths_nm(box_vectors)
        stats = self._last_gcmc_stats if isinstance(self._last_gcmc_stats, dict) else {}
        reference_selection = stats.get("reference_selection")
        reference_indices = stats.get("tracking_reference_indices")
        reference_source = "gcmc_tracking_reference_indices"

        if reference_indices is None and reference_selection:
            matches = re.findall(
                r"atomnum\s+(\d+)",
                str(reference_selection),
                flags=re.IGNORECASE,
            )
            if matches:
                reference_indices = [max(0, int(atomnum) - 1) for atomnum in matches]
                reference_source = "gcmc_reference_selection"

        if reference_indices is None:
            reference_indices = stats.get("reference_indices")
            reference_source = "gcmc_reference_indices"

        if reference_indices is not None:
            indices = np.asarray(reference_indices, dtype=int).reshape(-1)
            valid = indices[(indices >= 0) & (indices < positions_nm.shape[0])]
            if valid.size == indices.size and valid.size > 0:
                ref_positions_nm = self._unwrap_periodic_points_nm(
                    positions_nm[valid],
                    box_lengths_nm,
                )
                raw_center_nm = np.mean(ref_positions_nm, axis=0)
                center_nm = self._canonicalize_periodic_point_nm(
                    raw_center_nm,
                    box_lengths_nm,
                )
                logger.info(
                    "Stage 9 tracking center from GCMC reference atoms: "
                    "source=%s raw_center=[%.4f, %.4f, %.4f] nm "
                    "raw_offset_from_origin=%.3f A canonical_center=[%.4f, %.4f, %.4f] nm "
                    "canonical_offset_from_origin=%.3f A n_ref=%d selection=%s",
                    reference_source,
                    raw_center_nm[0],
                    raw_center_nm[1],
                    raw_center_nm[2],
                    10.0 * float(np.linalg.norm(raw_center_nm)),
                    center_nm[0],
                    center_nm[1],
                    center_nm[2],
                    10.0 * float(np.linalg.norm(center_nm)),
                    valid.size,
                    reference_selection,
                )
                return center_nm, reference_source

            logger.warning(
                "Stage 9 GCMC reference indices are invalid for the current "
                "positions array (n_indices=%d, n_atoms=%d); falling back",
                indices.size,
                positions_nm.shape[0],
            )

        ligand_center_nm = self._prepared_ligand_centroid_nm()
        if ligand_center_nm is not None:
            ligand_center_nm = self._canonicalize_periodic_point_nm(
                ligand_center_nm,
                box_lengths_nm,
            )
            logger.info(
                "Stage 9 tracking center from prepared ligand centroid: "
                "source=prepared_ligand center=[%.4f, %.4f, %.4f] nm "
                "offset_from_origin=%.3f A",
                ligand_center_nm[0],
                ligand_center_nm[1],
                ligand_center_nm[2],
                10.0 * float(np.linalg.norm(ligand_center_nm)),
            )
            return ligand_center_nm, "prepared_ligand"

        logger.info(
            "Stage 9 tracking center fallback: source=origin "
            "center=[0.0000, 0.0000, 0.0000] nm offset_from_origin=0.000 A"
        )
        return np.zeros(3, dtype=float), "origin"

    def _collect_pocket_water_snapshot(
        self,
        topology,
        positions,
        *,
        label: str,
        box_vectors=None,
    ) -> dict[str, object]:
        from H2OSite.simulation.production import (
            _find_water_residue_records,
            _minimum_image_delta,
            _water_record_label,
        )

        center_nm, center_source = self._resolve_tracking_center_nm(
            positions,
            box_vectors=box_vectors,
        )
        positions_nm = self._positions_to_nm(positions)
        box_lengths_nm = self._box_lengths_nm(box_vectors)
        radius_nm = 0.1 * float(self.config.ligand_distance_A)
        if not hasattr(topology, "residues"):
            return {
                "label": label,
                "center_nm": center_nm,
                "radius_nm": radius_nm,
                "inside": {},
                "all": {},
            }
        water_records = _find_water_residue_records(topology)
        all_waters: dict[str, dict[str, object]] = {}

        if water_records:
            oxygen_indices = [record["oxygen_index"] for record in water_records]
            water_positions_nm = positions_nm[oxygen_indices]
            delta = _minimum_image_delta(
                water_positions_nm,
                center_nm,
                box_lengths_nm,
            )
            dists_nm = np.linalg.norm(delta, axis=1)
            for record, disp, dist_nm in zip(water_records, delta, dists_nm):
                label_text = str(record.get("label", _water_record_label(record)))
                all_waters[label_text] = {
                    "label": label_text,
                    "topology_resid": record.get("topology_resid"),
                    "residue_id": record.get("residue_id"),
                    "chain_id": record.get("chain_id", ""),
                    "residue_name": record.get("residue_name", "HOH"),
                    "position_nm": center_nm + disp,
                    "distance_nm": float(dist_nm),
                }

        inside = {
            key: value
            for key, value in all_waters.items()
            if float(value["distance_nm"]) <= radius_nm
        }
        nearest = sorted(
            all_waters.values(),
            key=lambda item: float(item["distance_nm"]),
        )[:5]
        logger.info(
            "%s pocket-water snapshot: %d within %.1f A (nearest tracked water %.3f A) source=%s",
            label,
            len(inside),
            self.config.ligand_distance_A,
            10.0 * float(nearest[0]["distance_nm"]) if nearest else float("nan"),
            center_source,
        )
        for item in nearest:
            pos_A = 10.0 * np.asarray(item["position_nm"], dtype=float)
            logger.info(
                "  %s nearest water %s at [%.3f, %.3f, %.3f] A dist=%.3f A",
                label,
                item["label"],
                pos_A[0],
                pos_A[1],
                pos_A[2],
                10.0 * float(item["distance_nm"]),
            )

        return {
            "label": label,
            "center_nm": center_nm,
            "radius_nm": radius_nm,
            "inside": inside,
            "all": all_waters,
        }

    def _record_pocket_water_drift(
        self,
        transition_label: str,
        before: dict[str, object] | None,
        after: dict[str, object] | None,
    ) -> None:
        if not before or not after:
            return

        before_inside = before.get("inside", {})
        after_inside = after.get("inside", {})
        after_all = after.get("all", {})
        drifted_labels = sorted(set(before_inside) - set(after_inside))
        if not drifted_labels:
            logger.info("%s: no waters drifted out of the pocket sphere", transition_label)
            return

        logger.warning(
            "%s: %d water(s) drifted out of the %.1f A sphere. "
            "Inspect the centered trajectory and drift-water PDB outputs for details.",
            transition_label,
            len(drifted_labels),
            self.config.ligand_distance_A,
        )
        for label in drifted_labels:
            previous = before_inside[label]
            current = after_all.get(label, {})
            pos_nm = current.get("position_nm")
            pos_A = (
                10.0 * np.asarray(pos_nm, dtype=float)
                if pos_nm is not None
                else np.asarray([np.nan, np.nan, np.nan], dtype=float)
            )
            event = {
                "transition": transition_label,
                "event": "left_stage_sphere",
                "label": label,
                "topology_resid": current.get(
                    "topology_resid",
                    previous.get("topology_resid"),
                ),
                "residue_id": current.get("residue_id", previous.get("residue_id")),
                "chain_id": current.get("chain_id", previous.get("chain_id", "")),
                "residue_name": current.get(
                    "residue_name",
                    previous.get("residue_name", "HOH"),
                ),
                "position_nm": pos_nm,
                "distance_nm": current.get("distance_nm"),
                "last_inside_position_nm": previous.get("position_nm"),
                "last_inside_distance_nm": previous.get("distance_nm"),
            }
            self._water_drift_reports.append(event)
            logger.warning(
                "  Drifted water %s -> [%.3f, %.3f, %.3f] A dist=%.3f A (was %.3f A)",
                label,
                pos_A[0],
                pos_A[1],
                pos_A[2],
                10.0 * float(current.get("distance_nm", float("nan"))),
                10.0 * float(previous.get("distance_nm", float("nan"))),
            )

    def _write_water_drift_report(self) -> Path | None:
        import csv

        rows = list(self._water_drift_reports)
        if not rows:
            return None

        report_path = self.output_dir / f"{self.config.jobname}_drifted_waters.csv"
        with open(report_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "transition",
                    "event",
                    "label",
                    "topology_resid",
                    "residue_id",
                    "chain_id",
                    "residue_name",
                    "position_x_A",
                    "position_y_A",
                    "position_z_A",
                    "distance_A",
                    "last_inside_x_A",
                    "last_inside_y_A",
                    "last_inside_z_A",
                    "last_inside_distance_A",
                ]
            )
            for row in rows:
                position_nm = row.get("position_nm")
                position_A = (
                    10.0 * np.asarray(position_nm, dtype=float)
                    if position_nm is not None
                    else np.asarray([np.nan, np.nan, np.nan], dtype=float)
                )
                inside_nm = row.get("last_inside_position_nm")
                inside_A = (
                    10.0 * np.asarray(inside_nm, dtype=float)
                    if inside_nm is not None
                    else np.asarray([np.nan, np.nan, np.nan], dtype=float)
                )
                writer.writerow(
                    [
                        row.get("transition", ""),
                        row.get("event", ""),
                        row.get("label", ""),
                        row.get("topology_resid", ""),
                        row.get("residue_id", ""),
                        row.get("chain_id", ""),
                        row.get("residue_name", ""),
                        position_A[0],
                        position_A[1],
                        position_A[2],
                        10.0 * float(row.get("distance_nm", float("nan"))),
                        inside_A[0],
                        inside_A[1],
                        inside_A[2],
                        10.0 * float(row.get("last_inside_distance_nm", float("nan"))),
                    ]
                )
        logger.warning(
            "Wrote drifted-water report to %s. "
            "Review it alongside production/trajectory_centered.dcd and "
            "view_watermap_traj_fixed.pml.",
            report_path,
        )
        return report_path

    def _run_gcmc(self, topology, system, positions, ff, box_vectors=None):
        from H2OSite.simulation.gcmc import (
            GCMCSolvation,
            extract_nonbonded_forcefield_kwargs,
        )
        from H2OSite.simulation.protocol import SimulationProtocol
        stage_started_at = time.perf_counter()

        # Look up water-model-specific GCMC chemical potential and density.


        from H2OSite.core.config import DEFAULT_GCMC_PARAMS

        gcmc_cal = get_gcmc_params(
            self.config.water_model,
            DEFAULT_GCMC_PARAMS["timestep_fs"],
        )

        center_nm = np.zeros(3)  # Ligand centroid is at origin after prep
        restrained_atom_indices = self._non_water_heavy_atom_indices(topology)
        gcmc = GCMCSolvation(
            topology, system, positions,
            center_nm=center_nm,
            radius_nm=self.config.ligand_distance_A * 0.1,

            mu_excess_kcal=gcmc_cal["mu_excess"],
            temperature_K=DEFAULT_GCMC_PARAMS["temperature_K"],
            forcefield=ff,
            n_ghosts=self.config.gcmc_n_ghosts,
            box_vectors=box_vectors,
            water_model=self.config.water_model,
            flexible_water=self.config.gcmc_flexible_water,
            timestep_fs=self.config.gcmc_timestep_fs,
            minimize_after_customization=(
                self.config.gcmc_minimize_after_customization
            ),
            redraw_velocities_after_initialise=(
                self.config.gcmc_redraw_velocities_after_initialise
            ),
            ghost_min_distance_A=self.config.gcmc_ghost_min_distance_A,
            reference_selection=self.config.gcmc_reference_selection,
            batch_size=self.config.gcmc_batch_size,
            bulk_sampling_probability=self.config.gcmc_bulk_sampling_probability,
            standard_volume_A3=self.config.gcmc_standard_volume_A3,
            platform=self.config.gcmc_platform,
            device=int(str(self.config.gpu_index).split(",")[0].strip()),
            restrained_atom_indices=restrained_atom_indices,
            restraint_k_kcal=self.config.gcmc_restraint_k_kcal,
            cache_dir=(
                self.config.gcmc_cache_dir
                or (self._checkpoint_dir / "loch_sampler_cache")
            ),
            output_dir=self.output_dir / "gcmc",
        )
        new_topology, final_pos, stats = gcmc.run(
            n_gcmc_moves=self.config.gcmc_n_moves,
            n_md_steps_per_cycle=self.config.gcmc_md_steps_per_cycle,
            n_cycles=self.config.gcmc_n_cycles,
        )
        self._timings["gcmc"] = time.perf_counter() - stage_started_at
        self._last_gcmc_stats = stats
        timing = stats.get("timing", {}) if isinstance(stats, dict) else {}
        if timing:
            logger.info(
                "Stage 7 Loch summary: prod_rate=%.1f attempts/s "
                "overall_rate=%.1f attempts/s total_time=%.3fs",
                timing.get("production_attempts_per_s", float("nan")),
                timing.get("overall_attempts_per_s", float("nan")),
                timing.get("total_time_s", float("nan")),
            )

        # Rebuild system from the (potentially modified) topology so
        # post-GCMC stages use the correct particle count.
        import openmm.app as app
        rebuild_kwargs = extract_nonbonded_forcefield_kwargs(system)
        rebuild_kwargs["constraints"] = app.HBonds
        rebuild_kwargs["rigidWater"] = True
        new_system = ff.createSystem(new_topology, **rebuild_kwargs)

        active_box_vectors = (
            new_topology.getPeriodicBoxVectors()
            if hasattr(new_topology, "getPeriodicBoxVectors")
            else None
        )
        pre_relax_snapshot = self._collect_pocket_water_snapshot(
            new_topology,
            final_pos,
            label="Stage 7 post-loch",
            box_vectors=active_box_vectors,
        )

        if self.config.run_gcmc_handoff_relaxation:
            logger.info(
                "Stage 7 handoff relaxation: %d short restrained stage(s) "
                "before saving the GCMC checkpoint",
                len(self.config.gcmc_handoff_relaxation_stages),
            )
            heavy_indices, protein_indices = self._classify_atoms(new_topology)
            relax_protocol = SimulationProtocol(
                new_topology,
                final_pos,
                new_system,
                output_dir=self.output_dir / "gcmc_handoff_relax",
                gpu_index=self.config.gpu_index,
            )
            final_pos, _, relax_box = relax_protocol.run_equilibration(
                self.config.gcmc_handoff_relaxation_stages,
                protein_indices=protein_indices,
                heavy_atom_indices=heavy_indices,
                use_respa=False,
            )
            if relax_box is not None:
                new_topology.setPeriodicBoxVectors(relax_box)
                active_box_vectors = relax_box
            post_relax_snapshot = self._collect_pocket_water_snapshot(
                new_topology,
                final_pos,
                label="Stage 7 after handoff relax",
                box_vectors=active_box_vectors,
            )
            self._record_pocket_water_drift(
                "Stage 7 handoff relaxation",
                pre_relax_snapshot,
                post_relax_snapshot,
            )
            self._last_stage7_pocket_snapshot = post_relax_snapshot
            logger.info("Stage 7 handoff relaxation complete")
        else:
            self._last_stage7_pocket_snapshot = pre_relax_snapshot

        return new_topology, new_system, final_pos

    def _run_production(
        self, topology, system, positions, velocities,
        restrain_indices=None,
        active_site_center_nm=None,
        active_site_center_source: str = "origin",
        box_vectors=None,
    ):
        from H2OSite.simulation.production import ProductionMD

        prod = ProductionMD(
            topology, system, positions, velocities,
            box_vectors=box_vectors,
            output_dir=self.output_dir / "production",
            gpu_index=self.config.gpu_index,
        )
        if active_site_center_nm is None:
            active_site_center_nm = np.zeros(3, dtype=float)
        active_site_center_nm = np.asarray(active_site_center_nm, dtype=float)
        logger.info(
            "Stage 9 production tracking configuration: source=%s "
            "center=[%.4f, %.4f, %.4f] nm offset_from_origin=%.3f A radius=%.3f A",
            active_site_center_source,
            active_site_center_nm[0],
            active_site_center_nm[1],
            active_site_center_nm[2],
            10.0 * float(np.linalg.norm(active_site_center_nm)),
            self.config.ligand_distance_A,
        )
        return prod.run(
            temperature_K=300.0,
            time_ps=self.config.production_time_ps,
            active_site_center_nm=active_site_center_nm,
            active_site_radius_nm=self.config.ligand_distance_A * 0.1,

            # Select the configured production thermostat.
            thermostat=self.config.production_thermostat,
            use_respa=self.config.use_respa,

            # Collect per-frame data during production when requested.
            collect_energies=(
                self.config.watermap_data_collection == "during"
            ),

            restrain_indices=restrain_indices,
            restraint_k_kcal=self.config.production_restraint_k_kcal,
        )

    def _non_water_heavy_atom_indices(self, topology) -> list[int]:
        water_res = {"HOH", "WAT", "TIP4", "SOL", "T4P", "T3P", "TIP3", "TIP3P", "TIP4P"}
        indices: list[int] = []
        for atom in topology.atoms():
            is_h = (
                (atom.element is not None and atom.element.symbol == "H")
                or atom.name.startswith("H")
            )
            if is_h:
                continue
            if atom.residue.name in water_res:
                continue
            indices.append(atom.index)
        return indices

    def _classify_atoms(self, topology) -> tuple[list[int], list[int]]:
        heavy, protein = [], []
        for atom in topology.atoms():
            is_h = (atom.element is not None and atom.element.symbol == "H") or atom.name.startswith("H")
            if not is_h:
                heavy.append(atom.index)
            resname = atom.residue.name
            protein_res = {
                "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
                "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
                "TYR","VAL","HIE","HID","HIP","CYX","ACE","NME",
            }
            if resname in protein_res:
                protein.append(atom.index)
        return heavy, protein

    def _extract_stage11b_data(self, topology, system, tracker):
        if (
            tracker.trajectory_path is None
            or not tracker.frame_resids
            or not tracker.trajectory_path.exists()
        ):
            return None, None


        # Replay the production trajectory to compute per-water properties.
        # Derive PW/WW energies and reduced dipole vectors from trajectory data.
        try:
            from H2OSite.analysis.trajectory_properties import (
                ProductionTrajectoryAnalyzer,
            )

            analyzer = ProductionTrajectoryAnalyzer(
                topology=topology,
                system=system,
                trajectory_file=tracker.trajectory_path,
                checkpoint_file=tracker.checkpoint_path,
            )
            tracking_center_A = 10.0 * np.asarray(tracker.center, dtype=float)
            return analyzer.compute_tracked_water_properties(
                tracker.frame_resids,
                active_site_center_A=tracking_center_A,
                active_site_radius_A=self.config.ligand_distance_A,
            )
        except Exception as exc:
            logger.warning("Could not replay Stage 11b trajectory properties: %s", exc)
            return None, None

    def _compute_thermodynamics(
        self,
        per_frame: list[np.ndarray],
        per_frame_dipoles: list[np.ndarray] | None = None,
        per_frame_energies: dict | None = None,
    ):
        from H2OSite.analysis.thermodynamics import (
            ThermodynamicCalculator,
        )

        # Pass water-model-specific bulk energy for correct enthalpy.

        calc = ThermodynamicCalculator(
            temperature_K=300.0,
            hs_radius_A=self.config.hs_radius_A,
            water_model=self.config.water_model,
        )
        self.sites = calc.compute_all(
            self.sites,
            per_frame,
            per_frame_dipoles=per_frame_dipoles,
            per_frame_energies=per_frame_energies,
        )

    def _compute_hbonds(self, topology, tracker) -> None:
        if (
            not self.sites
            or tracker.trajectory_path is None
            or not tracker.trajectory_path.exists()
        ):
            return


        # Replay production frames for hydrogen-bond analysis.
        # Average PW/LW/WW H-bond counts per hydration-site observation.
        try:
            from H2OSite.analysis.hbond import HBondAnalyzer

            hba = HBondAnalyzer(hs_radius_A=self.config.hs_radius_A)
            self.sites = hba.analyze_trajectory(
                topology,
                str(tracker.trajectory_path),
                self.sites,
            )
        except Exception as exc:
            logger.warning("Could not compute Stage 11b H-bonds: %s", exc)

    def _generate_output(self, per_frame: list[np.ndarray]):
        from H2OSite.io.converters import (
            watermap_to_cif,
            watermap_to_csv,
            watermap_to_pdb,
            watermap_to_pymol,
        )
        from H2OSite.io.cns import write_cns_grid
        from H2OSite.analysis.density import WaterDensityMap

        out = self.output_dir
        jn = self.config.jobname

        watermap_to_pdb(self.sites, out / f"{jn}_wm.pdb")
        watermap_to_cif(self.sites, out / f"{jn}_wm.cif")
        watermap_to_csv(self.sites, out / f"{jn}_wm.csv")
        watermap_to_pymol(self.sites, out / f"{jn}_wm.pml")

        if per_frame and len(per_frame) > 0:
            density_calc = WaterDensityMap(grid_spacing_A=self.config.grid_spacing_A)

            # Water density grid
            density_grid = density_calc.compute_density(per_frame)
            write_cns_grid(density_grid, out / f"{jn}-water.cns", title="Water density")

            # TPI / cavity map from the production trajectory.


            try:
                # Need protein positions for cavity clash check
                protein_pos = self._get_protein_positions()
                if protein_pos is not None:
                    cavity_grid = density_calc.compute_cavity_map(
                        per_frame,
                        protein_pos,
                        # probe_radius = 1.0
                        probe_radius_A=1.0,
                    )
                    write_cns_grid(
                        cavity_grid,
                        out / f"{jn}-continuous_cavity.cns",
                        title="Cavity occupancy (TPI)",
                    )
                    logger.info("Wrote cavity/TPI map (vrun equivalent)")
            except Exception as e:
                logger.warning("Could not compute cavity map: %s", e)

        logger.info("Output written to %s", out)

    def _generate_display_outputs(self, topology, tracker) -> None:
        if (
            tracker.trajectory_path is None
            or not tracker.trajectory_path.exists()
        ):
            return

        try:
            from H2OSite.postprocessing import (
                generate_display_artifacts,
            )

            artifacts = generate_display_artifacts(
                output_dir=self.output_dir,
                jobname=self.config.jobname,
                topology=topology,
                trajectory_file=tracker.trajectory_path,
                tracking_center_A=10.0 * np.asarray(tracker.center, dtype=float),
                site_radius_A=self.config.ligand_distance_A,
                nearby_radius_A=self.config.display_nearby_radius_A,
                water_model=self.config.water_model,
            )
            if artifacts:
                logger.info(
                    "Display artifacts written: %s",
                    ", ".join(sorted(artifacts)),
                )
        except Exception as exc:
            logger.warning(
                "Could not generate centered display outputs: %s",
                exc,
            )

    def _get_protein_positions(self) -> np.ndarray | None:
        """Load protein heavy-atom positions from the prepared PDB."""
        prep_dir = self.output_dir / "prep"
        import glob
        from H2OSite.io.structure_io import load_openmm_structure_file
        structure_files = sorted(glob.glob(str(prep_dir / "*protein*.cif"))) + sorted(
            glob.glob(str(prep_dir / "*protein*.pdb"))
        )
        if not structure_files:
            return None
        try:
            import openmm.unit as unit
            _topology, positions = load_openmm_structure_file(structure_files[0])
            pos = np.array(
                positions.value_in_unit(unit.angstroms)
            )
            return pos
        except Exception:
            return None

    @staticmethod
    def _serialize_checkpoint(ckpt_path: Path, **data) -> None:
        """Serialize checkpoint payload to an ``.npz`` file."""
        import openmm

        serialized: dict[str, bytes | np.ndarray] = {}
        for key, val in data.items():
            if hasattr(val, 'getNumParticles'):  # OpenMM System
                serialized[key] = np.void(
                    openmm.XmlSerializer.serialize(val).encode()
                )
            elif hasattr(val, 'chains'):  # OpenMM Topology
                serialized[key] = np.void(pickle.dumps(val))
            elif hasattr(val, 'getDefaultPeriodicBoxVectors'):  # ForceField
                serialized[key] = np.void(pickle.dumps(val))
            elif isinstance(val, np.ndarray):
                serialized[key] = val
            elif isinstance(val, list):
                serialized[key] = np.array(val)
            else:
                serialized[key] = np.void(pickle.dumps(val))

        np.savez(ckpt_path, **serialized)


    @staticmethod
    def _deserialize_checkpoint(ckpt_path: Path) -> dict:
        """Deserialize an ``.npz`` checkpoint file."""
        import openmm

        raw = np.load(ckpt_path, allow_pickle=True)
        result = {}
        for key in raw.files:
            val = raw[key]
            if val.dtype == np.void or val.ndim == 0:
                blob = bytes(val)
                # Try OpenMM XML first
                try:
                    result[key] = openmm.XmlSerializer.deserialize(
                        blob.decode()
                    )
                    continue
                except Exception:
                    pass
                # Fall back to pickle
                result[key] = pickle.loads(blob)
            else:
                result[key] = val

        return result


    def run_analysis_from_trajectory(
        self,
        checkpoint_file: str | Path,
        trajectory_file: str | Path,
        production_checkpoint_file: str | Path | None = None,
        active_site_center_A: np.ndarray | None = None,
        active_site_radius_A: float | None = None,
    ) -> list:
        """Re-run stages 10-12 from an existing production trajectory."""
        logger.info("=== Trajectory reanalysis mode ===")

        checkpoint_path = Path(checkpoint_file)
        trajectory_path = Path(trajectory_file)
        ckpt = self._deserialize_checkpoint(checkpoint_path)
        topology = ckpt["topology"]
        system = ckpt["system"]

        if active_site_center_A is None:
            active_site_center_A = np.zeros(3, dtype=np.float64)
        else:
            active_site_center_A = np.asarray(active_site_center_A, dtype=np.float64)
        if active_site_radius_A is None:
            active_site_radius_A = self.config.ligand_distance_A

        if production_checkpoint_file is None:
            candidate = trajectory_path.with_name("checkpoint.npz")
            if candidate.exists():
                production_checkpoint_file = candidate

        from H2OSite.analysis.clustering import HydrationSiteClustering
        from H2OSite.analysis.trajectory_properties import (
            ProductionTrajectoryAnalyzer,
        )
        from H2OSite.analysis.hbond import HBondAnalyzer
        from H2OSite.simulation.production import replay_water_tracking

        tracker = replay_water_tracking(
            topology=topology,
            trajectory_file=trajectory_path,
            center_nm=active_site_center_A * 0.1,
            radius_nm=active_site_radius_A * 0.1,
            checkpoint_file=production_checkpoint_file,
        )

        per_frame = tracker.get_per_frame_positions()
        clusterer = HydrationSiteClustering(
            hs_radius_A=self.config.hs_radius_A,
            dcutoff_A=self.config.clustering_cutoff_A,
            grid_spacing_A=self.config.grid_spacing_A,
            water_model=self.config.water_model,
        )
        self.sites = clusterer.cluster(
            tracker.get_all_positions(),
            tracker.n_frames,
            per_frame_positions=per_frame,
        )

        analyzer = ProductionTrajectoryAnalyzer(
            topology=topology,
            system=system,
            trajectory_file=trajectory_path,
            checkpoint_file=production_checkpoint_file,
        )
        per_frame_dipoles, per_frame_energies = analyzer.compute_tracked_water_properties(
            tracker.frame_resids,
            active_site_center_A=active_site_center_A,
            active_site_radius_A=active_site_radius_A,
        )
        self._compute_thermodynamics(
            per_frame,
            per_frame_dipoles=per_frame_dipoles,
            per_frame_energies=per_frame_energies,
        )

        hba = HBondAnalyzer(hs_radius_A=self.config.hs_radius_A)
        self.sites = hba.analyze_trajectory(
            topology,
            str(trajectory_path),
            self.sites,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            self.output_dir / "retracked_water_positions.npy",
            tracker.get_all_positions(),
        )
        self._generate_output(per_frame)
        return self.sites


