# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Production MD with active-site water tracking.

Uses NPT dynamics with MonteCarloBarostat. Thermostat choices are langevin
(friction 1.0/ps) or a weak-coupling mode (friction 0.01/ps).
Weak coupling produces slower temperature equilibration.

Water positions are collected during production. Interaction energies can be
computed post-hoc from the saved trajectory (collect_energies=False, default)
or during each sampled MD frame (collect_energies=True).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _water_record_label(record: dict[str, int | str]) -> str:
    """Build a readable, stable label for one tracked water residue."""
    resname = str(record.get("residue_name", "HOH"))
    chain_id = str(record.get("chain_id", "")).strip()
    residue_id = str(record.get("residue_id", record.get("topology_resid", "?")))
    if chain_id:
        return f"{resname}:{chain_id}:{residue_id}"
    return f"{resname}:{residue_id}"


def _box_lengths_nm(box_vectors_nm) -> np.ndarray | None:
    """Extract orthorhombic box lengths from OpenMM box vectors."""
    if box_vectors_nm is None:
        return None
    try:
        import openmm.unit as unit
    except Exception:
        unit = None

    if unit is not None and hasattr(box_vectors_nm, "value_in_unit"):
        box = np.asarray(
            box_vectors_nm.value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
    else:
        try:
            box = np.asarray(box_vectors_nm, dtype=np.float64)
        except Exception:
            rows = []
            try:
                for row in box_vectors_nm:
                    vals = []
                    for val in row:
                        if unit is not None and hasattr(val, "value_in_unit"):
                            vals.append(val.value_in_unit(unit.nanometer))
                        else:
                            vals.append(float(val))
                    rows.append(vals)
            except Exception:
                return None
            box = np.asarray(rows, dtype=np.float64)
    if box.shape == (3, 3):
        lengths = np.array([box[0, 0], box[1, 1], box[2, 2]], dtype=np.float64)
    elif box.shape == (3,):
        lengths = box.astype(np.float64, copy=False)
    else:
        return None
    if np.any(lengths <= 0):
        return None
    return lengths


def _minimum_image_delta(
    positions_nm: np.ndarray,
    center_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Return displacement vectors to the site center under minimum image."""
    delta = np.asarray(positions_nm, dtype=np.float64) - np.asarray(
        center_nm, dtype=np.float64
    )
    if box_lengths_nm is not None:
        delta -= box_lengths_nm * np.round(delta / box_lengths_nm)
    return delta


def _canonicalize_center_nm(
    center_nm: np.ndarray,
    box_lengths_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Map a periodic center onto its nearest image around the origin."""
    center_nm = np.asarray(center_nm, dtype=np.float64)
    if box_lengths_nm is not None:
        center_nm = center_nm - box_lengths_nm * np.round(center_nm / box_lengths_nm)
    return center_nm


def _find_water_residue_records(topology) -> list[dict[str, int]]:
    """Find tracked water residues and their oxygen indices."""
    records: list[dict[str, int]] = []
    for resid, residue in enumerate(topology.residues()):
        if residue.name not in ("HOH", "WAT", "TIP4", "SOL", "T4P", "T3P"):
            continue
        oxygen_index = None
        for atom in residue.atoms():
            if atom.element is not None and atom.element.symbol == "O":
                oxygen_index = atom.index
                break
            if atom.name in ("O", "OW", "OH2"):
                oxygen_index = atom.index
                break
        if oxygen_index is None:
            continue
        chain_id = ""
        try:
            chain_id = str(residue.chain.id)
        except Exception:
            chain_id = ""
        record = {
            "resid": resid,
            "topology_resid": resid,
            "residue_id": str(getattr(residue, "id", resid)),
            "chain_id": chain_id,
            "residue_name": residue.name,
            "oxygen_index": oxygen_index,
        }
        record["label"] = _water_record_label(record)
        records.append(record)
    return records


class WaterTracker:
    """Collect water oxygen positions within the active site during MD.

    Records water positions and optionally per-frame potential energies.
    """

    def __init__(self, center: np.ndarray, radius: float = 10.0):
        self.center = np.asarray(center)  # nm
        self.radius = radius  # nm
        self.frames: list[np.ndarray] = []  # list of (N_water, 3) arrays
        self.frame_resids: list[np.ndarray] = []
        self.energies: list[np.ndarray] = []
        # Per-frame total PE, populated when collect_energies=True
        # Optional per-frame energy collection during production.
        self.frame_energies: list[float] = []
        self.trajectory_path: Path | None = None
        self.checkpoint_path: Path | None = None
        self.n_frames = 0
        self.drift_events: list[dict[str, object]] = []
        self._previous_selected_labels: set[str] = set()
        self._ever_selected_labels: set[str] = set()
        self._water_metadata: dict[str, dict[str, object]] = {}

    def collect(
        self,
        positions_nm: np.ndarray,
        water_oxygen_indices: list[int],
        water_residue_ids: list[int] | None = None,
        box_lengths_nm: np.ndarray | None = None,
        water_records: list[dict[str, object]] | None = None,
    ) -> np.ndarray:
        """Record water O positions within active site for this frame."""
        water_pos = positions_nm[water_oxygen_indices]
        delta = _minimum_image_delta(water_pos, self.center, box_lengths_nm)
        dists = np.linalg.norm(delta, axis=1)
        mask = dists <= self.radius
        selected = np.where(mask)[0]
        self.frames.append((self.center + delta[mask]).copy())
        if water_residue_ids is None:
            self.frame_resids.append(np.empty(0, dtype=int))
        else:
            resid_array = np.asarray(water_residue_ids, dtype=int)
            self.frame_resids.append(resid_array[selected].copy())

        if water_records is not None:
            current_selected_labels: set[str] = set()
            current_positions_nm: dict[str, np.ndarray] = {}
            current_distances_nm: dict[str, float] = {}
            for idx, record in enumerate(water_records):
                label = str(record.get("label", _water_record_label(record)))
                current_positions_nm[label] = (self.center + delta[idx]).copy()
                current_distances_nm[label] = float(dists[idx])
                if mask[idx]:
                    current_selected_labels.add(label)
                    self._ever_selected_labels.add(label)
                    meta = dict(record)
                    meta["label"] = label
                    meta["last_inside_frame"] = self.n_frames
                    meta["last_inside_position_nm"] = current_positions_nm[label].copy()
                    meta["last_inside_distance_nm"] = current_distances_nm[label]
                    self._water_metadata[label] = meta

            exited_labels = sorted(
                self._previous_selected_labels - current_selected_labels
            )
            for label in exited_labels:
                meta = dict(self._water_metadata.get(label, {}))
                self.drift_events.append(
                    {
                        "frame_index": self.n_frames,
                        "label": label,
                        "topology_resid": meta.get("topology_resid"),
                        "residue_id": meta.get("residue_id"),
                        "chain_id": meta.get("chain_id", ""),
                        "residue_name": meta.get("residue_name", "HOH"),
                        "position_nm": current_positions_nm.get(label),
                        "distance_nm": current_distances_nm.get(label),
                        "last_inside_position_nm": meta.get("last_inside_position_nm"),
                        "last_inside_distance_nm": meta.get("last_inside_distance_nm"),
                        "event": "left_production_sphere",
                    }
                )

            self._previous_selected_labels = current_selected_labels
        self.n_frames += 1
        return selected

    def get_all_positions(self) -> np.ndarray:
        """Return concatenated water positions from all frames, in Å."""
        if not self.frames:
            return np.empty((0, 3))
        return np.vstack(self.frames) * 10.0  # nm -> Å

    def get_per_frame_positions(self) -> list[np.ndarray]:
        """Return list of per-frame water positions in Å."""
        return [f * 10.0 for f in self.frames]


def replay_water_tracking(
    topology,
    trajectory_file: str | Path,
    center_nm: np.ndarray,
    radius_nm: float,
    checkpoint_file: str | Path | None = None,
) -> WaterTracker:
    """Rebuild active-site water tracking from an existing trajectory."""
    import MDAnalysis as mda

    trajectory_file = Path(trajectory_file)
    tracker = WaterTracker(center_nm, radius_nm)
    tracker.trajectory_path = trajectory_file
    tracker.checkpoint_path = (
        Path(checkpoint_file) if checkpoint_file is not None else None
    )

    water_records = _find_water_residue_records(topology)
    water_oxygen_indices = [record["oxygen_index"] for record in water_records]
    water_residue_ids = [record["resid"] for record in water_records]

    universe = mda.Universe(topology, str(trajectory_file))
    last_positions_nm = None
    last_box_lengths_nm = None
    for ts in universe.trajectory:
        positions_nm = universe.atoms.positions.astype(np.float64, copy=False) * 0.1
        box_lengths_nm = None
        if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0):
            box_lengths_nm = ts.dimensions[:3].astype(np.float64) * 0.1
        tracker.collect(
            positions_nm,
            water_oxygen_indices,
            water_residue_ids=water_residue_ids,
            box_lengths_nm=box_lengths_nm,
        )
        last_positions_nm = positions_nm.copy()
        last_box_lengths_nm = (
            None if box_lengths_nm is None else box_lengths_nm.copy()
        )

    if checkpoint_file is not None:
        ckpt = np.load(checkpoint_file)
        positions_nm = np.asarray(ckpt["positions_nm"], dtype=np.float64)
        box_lengths_nm = None
        if "box_vectors_nm" in ckpt:
            box_lengths_nm = _box_lengths_nm(ckpt["box_vectors_nm"])
        same_positions = (
            last_positions_nm is not None
            and last_positions_nm.shape == positions_nm.shape
            and np.allclose(last_positions_nm, positions_nm)
        )
        same_box = (
            (last_box_lengths_nm is None and box_lengths_nm is None)
            or (
                last_box_lengths_nm is not None
                and box_lengths_nm is not None
                and np.allclose(last_box_lengths_nm, box_lengths_nm)
            )
        )
        if not (same_positions and same_box):
            tracker.collect(
                positions_nm,
                water_oxygen_indices,
                water_residue_ids=water_residue_ids,
                box_lengths_nm=box_lengths_nm,
            )

    logger.info(
        "Replayed water tracking from %s: %d frames, %d total water observations",
        trajectory_file,
        tracker.n_frames,
        sum(len(f) for f in tracker.frames),
    )
    return tracker


class ProductionMD:
    """Run NPT production MD and collect active-site water data."""

    def __init__(
        self,
        topology,
        system,
        positions,
        velocities=None,
        box_vectors=None,
        output_dir: Path = Path("production"),
        gpu_index: str = "0",
    ):
        self.topology = topology
        self.system = system
        self.positions = positions
        self.velocities = velocities
        self.box_vectors = box_vectors
        self.output_dir = Path(output_dir)
        self.gpu_index = gpu_index
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        temperature_K: float = 300.0,
        time_ps: float = 2000.0,
        trajectory_interval_ps: float = 1.5,
        active_site_center_nm: np.ndarray | None = None,
        active_site_radius_nm: float = 1.0,
        save_trajectory: bool = True,
        thermostat: str = "langevin",
        use_respa: bool = True,
        collect_energies: bool = False,
        restrain_indices: list[int] | None = None,
        restraint_k_kcal: float = 5.0,
    ) -> WaterTracker:
        """Run production MD with water tracking.

        Args:
            temperature_K: Production temperature
            time_ps: Simulation time
            trajectory_interval_ps: How often to save frames
            active_site_center_nm: Center of binding site (nm)
            active_site_radius_nm: Radius to track waters (nm)
            save_trajectory: Whether to write DCD trajectory
            thermostat: Standard Langevin (default) or low-friction coupling.

            use_respa: Use MTSLangevinIntegrator with 2-level RESPA.

            collect_energies: If True, compute per-water interaction energies
                each frame. If False, only positions are collected.

        Returns:
            WaterTracker with collected water positions
        """
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        import time as _time

        logger.info("Starting production MD: %.1f ps at %.1f K", time_ps, temperature_K)

        T = temperature_K * unit.kelvin

        # --- System copy with barostat ---
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(self.system)
        )

        # --- RESPA force-group assignment ---
        is_respa = use_respa
        if is_respa:
            try:
                for force in system.getForces():
                    if isinstance(force, openmm.NonbondedForce):
                        force.setForceGroup(1)
                        force.setReciprocalSpaceForceGroup(0)
                    else:
                        force.setForceGroup(1)
            except Exception:
                is_respa = False
                logger.warning("Could not assign RESPA force groups; "
                               "falling back to single timestep")

        # Barostat (in slow group for RESPA)
        baro = openmm.MonteCarloBarostat(1.0 * unit.atmospheres, T, 25)
        if is_respa:
            baro.setForceGroup(0)
        system.addForce(baro)

        # --- Positional restraints on solute heavy atoms ---
        # Apply positional restraints to non-water heavy atoms
        # throughout production so water structure is measured against a fixed
        # protein conformation.
        if restrain_indices:
            _KCAL_A2_TO_KJ_NM2 = 4.184 * 100.0  # kcal/mol/Å² -> kJ/mol/nm²
            k_kj_nm2 = restraint_k_kcal * _KCAL_A2_TO_KJ_NM2
            restraint = openmm.CustomExternalForce(
                "k_restr * periodicdistance(x, y, z, x0, y0, z0)^2"
            )
            restraint.addGlobalParameter("k_restr", k_kj_nm2)
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")
            # Fast force group for RESPA (restraints change rapidly)
            restraint.setForceGroup(1 if is_respa else 0)

            if hasattr(self.positions, "value_in_unit"):
                pos_nm = np.asarray(
                    self.positions.value_in_unit(unit.nanometers)
                )
            else:
                pos_nm = np.asarray(self.positions)

            for idx in restrain_indices:
                x, y, z = float(pos_nm[idx][0]), float(pos_nm[idx][1]), float(pos_nm[idx][2])
                restraint.addParticle(int(idx), [x, y, z])

            system.addForce(restraint)
            logger.info(
                "  Restraints: %.1f kcal/mol/A^2 on %d solute heavy atoms",
                restraint_k_kcal, len(restrain_indices),
            )

        # --- Integrator selection ---
        # Use Langevin dynamics with MonteCarloBarostat pressure coupling.
        if thermostat == "desmond_like":
            # Weak Langevin friction reduces stochastic coupling.
            friction = 0.01 / unit.picoseconds
            logger.info("  Thermostat: desmond_like (friction=0.01/ps)")
        else:
            # Standard Langevin - robust, well-tested default.
            friction = 1.0 / unit.picoseconds
            logger.info("  Thermostat: langevin (friction=1.0/ps)")

        if is_respa:
            # Production RESPA: [bonded=0.002, near=0.002, far=0.006] ps
            outer_dt = 0.006 * unit.picoseconds
            inner_ratio = 3  # 0.006 / 0.002
            try:
                integrator = openmm.MTSLangevinIntegrator(
                    T, friction, outer_dt,
                    [(0, 1), (1, inner_ratio)],
                )
                dt = outer_dt
                logger.info("  Using MTSLangevinIntegrator: outer=0.006 ps, "
                            "inner=0.002 ps (ratio 3)")
            except AttributeError:
                is_respa = False
                dt = 0.002 * unit.picoseconds
                integrator = openmm.LangevinMiddleIntegrator(T, friction, dt)
                logger.warning("  MTSLangevinIntegrator unavailable; "
                               "single-timestep fallback (0.002 ps)")
        else:
            dt = 0.002 * unit.picoseconds
            integrator = openmm.LangevinMiddleIntegrator(T, friction, dt)

        simulation = self._create_simulation(system, integrator, app)
        if self.box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.box_vectors)
        simulation.context.setPositions(self.positions)

        if self.velocities is not None:
            simulation.context.setVelocities(self.velocities)
        else:
            simulation.context.setVelocitiesToTemperature(T)

        # Identify water residues and oxygen indices
        water_residue_records = self._find_water_residue_records()
        water_oxygen_indices = [record["oxygen_index"] for record in water_residue_records]
        water_residue_ids = [record["resid"] for record in water_residue_records]
        logger.info("Tracking %d water oxygens", len(water_oxygen_indices))

        # Set up water tracker
        if active_site_center_nm is None:
            active_site_center_nm = np.zeros(3)
        initial_box_lengths_nm = _box_lengths_nm(
            self.box_vectors
            if self.box_vectors is not None
            else self.topology.getPeriodicBoxVectors()
        )
        canonical_center_nm = _canonicalize_center_nm(
            active_site_center_nm,
            initial_box_lengths_nm,
        )
        if not np.allclose(canonical_center_nm, active_site_center_nm):
            logger.info(
                "  Canonicalized active-site center under PBC: "
                "input=[%.4f, %.4f, %.4f] nm canonical=[%.4f, %.4f, %.4f] nm "
                "canonical_offset_from_origin=%.3f A",
                active_site_center_nm[0],
                active_site_center_nm[1],
                active_site_center_nm[2],
                canonical_center_nm[0],
                canonical_center_nm[1],
                canonical_center_nm[2],
                10.0 * float(np.linalg.norm(canonical_center_nm)),
            )
        active_site_center_nm = canonical_center_nm
        tracker = WaterTracker(active_site_center_nm, active_site_radius_nm)

        # Diagnostic: how far are waters from the tracking center at start?
        if water_oxygen_indices:
            init_pos = np.array(
                [self.positions[i].value_in_unit(unit.nanometers)
                 if hasattr(self.positions[i], 'value_in_unit')
                 else self.positions[i]
                 for i in water_oxygen_indices]
            )
            init_box_lengths = initial_box_lengths_nm
            dists = np.linalg.norm(
                _minimum_image_delta(
                    init_pos,
                    active_site_center_nm,
                    init_box_lengths,
                ),
                axis=1,
            )
            n_in = int(np.sum(dists <= active_site_radius_nm))
            logger.info(
                "  Active site sphere: center=%s nm, radius=%.3f nm",
                active_site_center_nm, active_site_radius_nm,
            )
            logger.info(
                "  Initial water O distances: min=%.3f nm, mean=%.3f nm, "
                "%d/%d within radius",
                dists.min(), dists.mean(), n_in, len(dists),
            )
        tracker.checkpoint_path = self.output_dir / "checkpoint.npz"

        # Reporters
        dt_ps = dt.value_in_unit(unit.picoseconds)
        n_steps = int(time_ps / dt_ps)
        traj_interval = max(1, int(trajectory_interval_ps / dt_ps))

        # State reporter
        simulation.reporters.append(
            app.StateDataReporter(
                str(self.output_dir / "production.csv"),
                traj_interval,
                step=True,
                time=True,
                temperature=True,
                potentialEnergy=True,
                kineticEnergy=True,
                density=True,
            )
        )

        # Trajectory
        if save_trajectory:
            tracker.trajectory_path = self.output_dir / "production.dcd"
            simulation.reporters.append(
                app.DCDReporter(
                    str(tracker.trajectory_path),
                    traj_interval,
                )
            )

        # Run with periodic water collection
        # collects water positions within active_site_distance each frame.
        # collects per-frame energy data during production.
        steps_per_collection = traj_interval
        n_collections = n_steps // steps_per_collection

        logger.info(
            "Starting production: %d collections over %.1f ps "
            "(%.1f ps per collection)",
            n_collections, time_ps, steps_per_collection * dt_ps,
        )
        _t0 = _time.time()

        for i in range(n_collections):
            simulation.step(steps_per_collection)

            state = simulation.context.getState(
                getPositions=True,
                getEnergy=collect_energies,
            )
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            box_lengths_nm = _box_lengths_nm(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometers
                )
            )
            selected = tracker.collect(
                pos,
                water_oxygen_indices,
                water_residue_ids=water_residue_ids,
                box_lengths_nm=box_lengths_nm,
                water_records=water_residue_records,
            )

            if i < 3:
                water_pos = pos[water_oxygen_indices]
                dists = np.linalg.norm(
                    _minimum_image_delta(
                        water_pos,
                        active_site_center_nm,
                        box_lengths_nm,
                    ),
                    axis=1,
                )
                logger.info(
                    "  Frame %d: %d waters within radius, "
                    "closest water=%.3f nm",
                    i, len(selected), dists.min(),
                )

            # During-production energy mode: store per-frame PE for later
            # per-water energy decomposition.
            if collect_energies:
                pe_val = state.getPotentialEnergy().value_in_unit(
                    unit.kilocalories_per_mole
                )
                tracker.frame_energies.append(pe_val)

            if (i + 1) % (n_collections // 10 or 1) == 0:
                pct = 100.0 * (i + 1) / n_collections
                sim_ps = (i + 1) * steps_per_collection * dt_ps
                elapsed = _time.time() - _t0
                total_waters = sum(len(f) for f in tracker.frames)
                pe = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                elapsed_m, elapsed_s = divmod(elapsed, 60)
                logger.info(
                    "  Production: %.0f%% | %.1f/%.1f ps | "
                    "%d water obs | PE=%s | wall %d:%02d",
                    pct, sim_ps, time_ps,
                    total_waters, pe,
                    int(elapsed_m), int(elapsed_s),
                )

        # Handle remaining steps
        remaining = n_steps - n_collections * steps_per_collection
        if remaining > 0:
            simulation.step(remaining)
            state = simulation.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            box_lengths_nm = _box_lengths_nm(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometers
                )
            )
            tracker.collect(
                pos,
                water_oxygen_indices,
                water_residue_ids=water_residue_ids,
                box_lengths_nm=box_lengths_nm,
                water_records=water_residue_records,
            )

        logger.info(
            "Production complete: %d frames, %d total water observations",
            tracker.n_frames,
            sum(len(f) for f in tracker.frames),
        )
        if tracker.drift_events:
            logger.warning(
                "Detected %d water drift-out event(s) during production. "
                "Inspect production/trajectory_centered.dcd with "
                "view_watermap_traj_fixed.pml to see where these waters moved.",
                len(tracker.drift_events),
            )
            for event in tracker.drift_events[:10]:
                position_nm = event.get("position_nm")
                if position_nm is None:
                    position_text = "<unknown>"
                    distance_A = float("nan")
                else:
                    pos_A = 10.0 * np.asarray(position_nm, dtype=float)
                    position_text = (
                        f"[{pos_A[0]:.3f}, {pos_A[1]:.3f}, {pos_A[2]:.3f}] A"
                    )
                    distance_A = 10.0 * float(event.get("distance_nm", float("nan")))
                logger.warning(
                    "  Drifted water %s at frame %s -> %s dist=%.3f A",
                    event.get("label", "<unknown>"),
                    event.get("frame_index", "?"),
                    position_text,
                    distance_A,
                )
            if len(tracker.drift_events) > 10:
                logger.warning(
                    "  ... %d additional production drift event(s) not shown",
                    len(tracker.drift_events) - 10,
                )

        # Save final state
        state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        self._save_checkpoint(state)

        # Save water positions
        np.save(
            self.output_dir / "water_positions.npy",
            tracker.get_all_positions(),
        )

        return tracker

    def _find_water_residue_records(self) -> list[dict[str, int]]:
        """Find tracked water residues and their oxygen indices."""
        return _find_water_residue_records(self.topology)

    def _save_checkpoint(self, state) -> None:
        """Save simulation state for restart."""
        import openmm.unit as unit

        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        vel = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometers / unit.picoseconds
        )
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometers)
        np.savez(
            self.output_dir / "checkpoint.npz",
            positions_nm=pos,
            velocities_nm_ps=vel,
            box_vectors_nm=box,
        )

    def _create_simulation(self, system, integrator, app):
        import openmm

        platform_specs = [
            ("CUDA", {"DeviceIndex": self.gpu_index, "Precision": "mixed"}),
            ("OpenCL", {}),
            ("CPU", {}),
        ]
        last_error = None

        for platform_name, properties in platform_specs:
            try:
                platform = openmm.Platform.getPlatformByName(platform_name)
                if properties:
                    simulation = app.Simulation(
                        self.topology, system, integrator, platform, properties
                    )
                else:
                    simulation = app.Simulation(
                        self.topology, system, integrator, platform
                    )
                logger.info("Using %s platform for production", platform_name)
                return simulation
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to initialize %s platform for production: %s",
                    platform_name,
                    exc,
                )

        raise RuntimeError(
            "Unable to initialize any OpenMM platform for production"
        ) from last_error
