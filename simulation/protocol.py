# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Multi-stage equilibration using OpenMM.

Stages include restrained Brownian NVT, cold Langevin NVT/NPT, warm NPT,
and post-GCMC equilibration. With use_respa=True, MTSLangevinIntegrator
separates slow reciprocal-space PME from fast bonded/direct-space interactions.
Brownian stages use a single-timestep CustomIntegrator. NPT stages use
MonteCarloBarostat for stochastic volume rescaling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Conversion: kcal/mol/Å² -> kJ/mol/nm²
_KCAL_A2_TO_KJ_NM2 = 4.184 * 100.0


class SimulationProtocol:
    """Execute the multi-stage H2OSite equilibration and production protocol."""

    def __init__(
        self,
        topology,
        positions,
        system,
        output_dir: Path,
        gpu_index: str = "0",
    ):
        self.topology = topology
        self.positions = positions
        self.system = system
        self.output_dir = Path(output_dir)
        self.gpu_index = gpu_index
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _log_step_start(label: str) -> float:
        logger.info("  [Equil startup] %s...", label)
        return time.perf_counter()

    @staticmethod
    def _log_step_done(label: str, started_at: float) -> None:
        elapsed_s = time.perf_counter() - started_at
        logger.info("  [Equil startup] %s done in %.3f s", label, elapsed_s)

    @staticmethod
    def _clone_system(system):
        import copy
        import openmm

        if hasattr(openmm.XmlSerializer, "clone"):
            return openmm.XmlSerializer.clone(system)
        return copy.deepcopy(system)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_equilibration(
        self,
        stages: list[dict],
        protein_indices: list[int] | None = None,
        heavy_atom_indices: list[int] | None = None,
        use_respa: bool = True,
    ) -> tuple:
        """Run multi-stage equilibration.

        Args:
            use_respa: When True and a stage specifies a RESPA timestep tuple
                       [bonded, near, far], use MTSLangevinIntegrator to split
                       reciprocal-space PME from bonded+direct forces.


        Returns final (positions, velocities, box_vectors).
        """
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        current_positions = self.positions
        current_velocities = None
        logger.info(
            "Equilibration system: %d particles, %d forces",
            self.system.getNumParticles(),
            self.system.getNumForces(),
        )
        if self.system.getNumParticles() >= 500000:
            logger.warning(
                "Large OpenMM system detected (%d particles). "
                "Stage startup may still take minutes even with fast cloning.",
                self.system.getNumParticles(),
            )

        for i, stage in enumerate(stages):
            name = stage.get("name", f"stage_{i}")
            logger.info("=== Equilibration stage: %s ===", name)

            clone_started_at = self._log_step_start(
                "Cloning base OpenMM system for stage setup"
            )
            system = self._clone_system(self.system)
            self._log_step_done(
                "Cloning base OpenMM system for stage setup",
                clone_started_at,
            )

            # --- RESPA force-group assignment ---
            # Assign force groups BEFORE adding restraints/barostat so those
            # new forces also end up in the fast group.
            is_respa = use_respa and self._is_respa_stage(stage)
            if is_respa:
                force_group_started_at = self._log_step_start(
                    "Assigning RESPA force groups"
                )
                self._assign_respa_force_groups(system)
                self._log_step_done(
                    "Assigning RESPA force groups",
                    force_group_started_at,
                )

            # --- Restraints via tunable k_restr parameter ---
            restraint_k_kcal = stage.get("restraint_k", 0.0)
            restrain_type = stage.get("restrain", "solute_heavy")
            indices = self._get_restrain_indices(
                restrain_type, protein_indices, heavy_atom_indices
            )
            restraint_started_at = self._log_step_start(
                "Adding positional restraints"
            )
            self._add_restraint_force(
                system, indices, restraint_k_kcal, current_positions,
                force_group=1 if is_respa else 0,
            )
            self._log_step_done(
                "Adding positional restraints",
                restraint_started_at,
            )

            # --- Barostat for NPT ---
            # Use OpenMM MonteCarloBarostat for NPT volume sampling.

            if stage.get("ensemble") == "NPT":
                barostat_started_at = self._log_step_start("Adding barostat")
                T_baro = stage["temperature_K"] * unit.kelvin
                baro = openmm.MonteCarloBarostat(
                    1.0 * unit.atmospheres, T_baro, 25
                )
                if is_respa:
                    baro.setForceGroup(0)  # slow group
                system.addForce(baro)
                self._log_step_done("Adding barostat", barostat_started_at)

            # --- Integrator ---
            integrator_started_at = self._log_step_start("Creating integrator")
            integrator = self._create_integrator(stage, use_respa=is_respa)
            self._log_step_done("Creating integrator", integrator_started_at)

            # --- Simulation ---
            simulation_started_at = self._log_step_start(
                "Creating OpenMM Simulation / Context"
            )
            sim = self._create_simulation(system, integrator)
            self._log_step_done(
                "Creating OpenMM Simulation / Context",
                simulation_started_at,
            )

            state_started_at = self._log_step_start(
                "Loading positions and velocities into the Context"
            )
            sim.context.setPositions(current_positions)
            if current_velocities is not None:
                sim.context.setVelocities(current_velocities)
            else:
                sim.context.setVelocitiesToTemperature(
                    stage["temperature_K"] * unit.kelvin
                )
            self._log_step_done(
                "Loading positions and velocities into the Context",
                state_started_at,
            )

            # --- Energy minimisation (always on first stage) ---
            if i == 0:
                logger.info("  Energy minimisation …")
                tol = 10 * unit.kilojoules_per_mole / unit.nanometer
                pe_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
                logger.info("    pre-minimization PE = %s", pe_before)
                for mini_round in range(10):
                    sim.minimizeEnergy(maxIterations=200, tolerance=tol)
                    pe = sim.context.getState(getEnergy=True).getPotentialEnergy()
                    logger.info("    iter %d/10: PE = %s", mini_round + 1, pe)
                logger.info("  Minimisation done: %s → %s", pe_before, pe)
                sim.context.setVelocitiesToTemperature(
                    stage["temperature_K"] * unit.kelvin
                )
                logger.info(
                    "  Velocities reset to %.1f K after minimisation",
                    stage["temperature_K"],
                )

            # --- Step ---
            time_ps = stage["time_ps"]
            dt_ps = self._get_effective_timestep(stage, is_respa)
            n_steps = max(1, int(round(time_ps / dt_ps)))
            logger.info(
                "  Running %s: %.3f ps (%d steps, dt=%.4f ps%s)",
                name, time_ps, n_steps, dt_ps,
                ", RESPA" if is_respa else "",
            )

            report_interval = max(1, min(n_steps // 10, 1000))
            sim.reporters.append(
                app.StateDataReporter(
                    str(self.output_dir / f"equil_{name}.csv"),
                    report_interval,
                    step=True,
                    temperature=True,
                    potentialEnergy=True,
                    density=True,
                )
            )

            sim.step(n_steps)

            # --- Extract state ---
            state = sim.context.getState(
                getPositions=True, getVelocities=True
            )
            current_positions = state.getPositions()
            current_velocities = state.getVelocities()

            pe = sim.context.getState(getEnergy=True).getPotentialEnergy()
            logger.info("  Final PE: %s", pe)

        return current_positions, current_velocities, state.getPeriodicBoxVectors()

    # ------------------------------------------------------------------
    # Integrator factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_integrator(stage: dict, use_respa: bool = False):
        """Create the OpenMM integrator for a stage.

        Brownian: CustomIntegrator with Maxwell-Boltzmann
                  velocity resampling, Verlet position update, and per-
                  component velocity clipping (delta_max).

        Langevin / MTK: LangevinMiddleIntegrator (or MTSLangevinIntegrator
                        when RESPA is enabled) with per-stage tau.


        """
        import openmm
        import openmm.unit as unit

        T = stage["temperature_K"] * unit.kelvin
        integrator_type = stage.get("integrator", "Langevin")

        if integrator_type == "Brownian":
            # Brownian always uses single timestep (RESPA not applicable).
            # but Brownian ignores RESPA; uses smallest (bonded) timestep.
            dt = SimulationProtocol._get_timestep(stage) * unit.picoseconds
            tau = stage.get("thermostat_tau_ps", 0.1)
            friction = (1.0 / tau) / unit.picoseconds
            delta_max_A = stage.get("delta_max", 0.1)  # Å
            delta_max = delta_max_A * 0.1 * unit.nanometers  # convert Å->nm
            return SimulationProtocol._make_brownian_integrator(
                T, friction, dt, delta_max=delta_max,
            )

        if integrator_type in ("Langevin", "MTK"):
            tau = stage.get("thermostat_tau_ps", 1.0) * unit.picoseconds
            friction = 1.0 / tau

            # --- RESPA multi-timestep ---
            # When bonded == near and far > bonded, use 2-level RESPA:
            #   Group 0 (slow):  PME reciprocal space  -> evaluated at far dt
            #   Group 1 (fast):  bonded + direct NB    -> evaluated at bonded dt
            if use_respa:
                ts = stage.get("timestep_ps")
                if isinstance(ts, (list, tuple)) and len(ts) >= 3:
                    bonded_dt, _near_dt, far_dt = ts[0], ts[1], ts[2]
                    ratio = max(1, int(round(far_dt / bonded_dt)))
                    if ratio > 1:
                        outer_dt = far_dt * unit.picoseconds
                        try:
                            integrator = openmm.MTSLangevinIntegrator(
                                T, friction, outer_dt,
                                # (forceGroup, subdivisions):
                                # group 0 = reciprocal PME at outer dt
                                # group 1 = bonded+direct at outer_dt/ratio
                                [(0, 1), (1, ratio)],
                            )
                            logger.info(
                                "  Using MTSLangevinIntegrator: "
                                "outer=%.4f ps, inner=%.4f ps (ratio %d)",
                                far_dt, bonded_dt, ratio,
                            )
                            return integrator
                        except AttributeError:
                            logger.warning(
                                "  MTSLangevinIntegrator not available "
                                "(requires OpenMM ≥ 7.6); falling back to "
                                "single-timestep LangevinMiddleIntegrator"
                            )

            # Single-timestep fallback
            dt = SimulationProtocol._get_timestep(stage) * unit.picoseconds
            return openmm.LangevinMiddleIntegrator(T, friction, dt)

        raise ValueError(f"Unknown integrator: {integrator_type}")

    @staticmethod
    def _make_brownian_integrator(temperature, friction_coeff, dt,
                                  delta_max=None):
        """Brownian integrator via CustomIntegrator.

        Steps per timestep:
          1. Maxwell-Boltzmann velocity resampling  (v = sqrt(kT/m) * gaussian)
          2. Velocity Verlet kick                   (v += dt * f/m)
          3. Per-component velocity clipping         (|v_i| ≤ dmax/dt)
          4. Position update                         (x += v * dt)
          5. Constraint projection
        """
        import openmm
        import openmm.unit as unit

        if delta_max is None:
            delta_max = 0.01 * unit.nanometers

        kT_value = (
            unit.BOLTZMANN_CONSTANT_kB * temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilojoules_per_mole)
        dmax_value = delta_max.value_in_unit(unit.nanometers)

        integrator = openmm.CustomIntegrator(dt)
        integrator.addGlobalVariable("kT", kT_value)
        integrator.addGlobalVariable("dmax", dmax_value)
        integrator.addPerDofVariable("x0", 0)

        # MB velocity resampling
        integrator.addComputePerDof("v", "sqrt(kT/m)*gaussian")
        # Verlet kick
        integrator.addComputePerDof("v", "v + dt*f/m")
        # Per-component velocity clipping
        integrator.addComputePerDof("v", "min(dmax/dt, max(-dmax/dt, v))")
        # Save old positions for constraint velocity correction
        integrator.addComputePerDof("x0", "x")
        # Position update
        integrator.addComputePerDof("x", "x + v*dt")
        # Constraint projection
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "(x - x0)/dt")
        integrator.addConstrainVelocities()

        return integrator

    # ------------------------------------------------------------------
    # Restraints - tunable k_restr
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # RESPA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_respa_stage(stage: dict) -> bool:
        """Return True if this stage has a RESPA timestep tuple with ratio > 1.

        Brownian stages always return False (RESPA not applicable).
        """
        if stage.get("integrator") == "Brownian":
            return False
        ts = stage.get("timestep_ps")
        if not isinstance(ts, (list, tuple)) or len(ts) < 3:
            return False
        bonded, _near, far = ts[0], ts[1], ts[2]
        return far > bonded

    @staticmethod
    def _assign_respa_force_groups(system) -> None:
        """Assign force groups for RESPA multi-timestep integration.

        Splits forces into:
          Group 0 (slow) - PME reciprocal space
          Group 1 (fast) - bonded + direct-space nonbonded + everything else

        The bonded and near-field timesteps are equal in the default stages,
        so a two-level split suffices.
        """
        import openmm

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                # Direct space in fast group, reciprocal in slow group
                force.setForceGroup(1)
                force.setReciprocalSpaceForceGroup(0)
            else:
                # Bonded forces, CMMotionRemover, etc. -> fast group
                force.setForceGroup(1)

    @staticmethod
    def _get_effective_timestep(stage: dict, is_respa: bool) -> float:
        """Return the effective outer timestep in ps for step counting.

        For RESPA stages, this is the far (outer) timestep.
        For single-timestep stages, this is the bonded (smallest) timestep.
        """
        ts = stage.get("timestep_ps")
        if is_respa and isinstance(ts, (list, tuple)) and len(ts) >= 3:
            return ts[2]  # far timestep = outer dt
        return SimulationProtocol._get_timestep(stage)

    # ------------------------------------------------------------------
    # Restraints - tunable k_restr
    # ------------------------------------------------------------------

    @staticmethod
    def _add_restraint_force(
        system,
        indices: list[int],
        k_kcal: float,
        positions,
        force_group: int = 0,
    ) -> None:
        """Add a positional-restraint CustomExternalForce with a tunable
        global parameter ``k_restr`` (kJ/mol/nm²).

        Matching the IFD-MD reference pattern, this allows per-stage
        restraint strength changes via::

            sim.context.setParameter("k_restr", new_value)

        without rebuilding the system.
        """
        import openmm
        import openmm.unit as unit

        k_kj_nm2 = k_kcal * _KCAL_A2_TO_KJ_NM2

        restraint = openmm.CustomExternalForce(
            "k_restr * periodicdistance(x, y, z, x0, y0, z0)^2"
        )
        restraint.addGlobalParameter("k_restr", k_kj_nm2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        # Assign to fast force group for RESPA (restraints change rapidly)
        restraint.setForceGroup(force_group)

        # Convert positions -> numpy nm
        if hasattr(positions, "value_in_unit"):
            pos_nm = np.asarray(positions.value_in_unit(unit.nanometers))
        else:
            pos_nm = np.asarray(positions)

        for idx in indices:
            x, y, z = (
                float(pos_nm[idx][0]),
                float(pos_nm[idx][1]),
                float(pos_nm[idx][2]),
            )
            restraint.addParticle(int(idx), [x, y, z])

        system.addForce(restraint)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_timestep(stage: dict) -> float:
        """Return timestep in ps.  Handles [bonded, near, far] tuples."""
        ts = stage.get("timestep_ps")
        if ts is not None:
            return ts[0] if isinstance(ts, (list, tuple)) else ts
        return 0.002  # 2 fs default

    def _get_restrain_indices(
        self,
        restrain_type: str,
        protein_indices: list[int] | None,
        heavy_atom_indices: list[int] | None,
    ) -> list[int]:
        if restrain_type in ("solute_heavy", "non_water_heavy"):
            return heavy_atom_indices or []
        return []

    def _create_simulation(self, system, integrator):
        """Create a Simulation, falling back if a platform cannot init."""
        import openmm
        import openmm.app as app

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
                    sim = app.Simulation(
                        self.topology, system, integrator, platform, properties
                    )
                else:
                    sim = app.Simulation(
                        self.topology, system, integrator, platform
                    )
                logger.info("Using %s platform", platform_name)
                return sim
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to initialize %s platform: %s",
                    platform_name,
                    exc,
                )

        raise RuntimeError("Unable to initialize any OpenMM platform") from last_error
