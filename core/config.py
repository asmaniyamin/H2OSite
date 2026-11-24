# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""H2OSite configuration and physical constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BOLTZMANN_KCAL = 0.001987204  # kcal/(mol·K)
# Boltzmann constant in cal/(mol·K)
# Entropies are stored in cal/(mol·K)
# and converted to kcal/mol via ``* 0.300`` (= T/1000 at 300 K).

KB_CAL_MOL_K = BOLTZMANN_KCAL * 1000.0  # 1.987204 cal/(mol·K)

# ---------------------------------------------------------------------------
# Water-model-specific constants
# ---------------------------------------------------------------------------
# Each water model has its own bulk energy, bulk density, and GCMC chemical
# potential, calibrated per water model and timestep.
#
# ``bulk_energy_kcal``: total interaction energy of one water molecule with
# all neighbours in bulk liquid; used as reference for enthalpy.
#
# ``bulk_density``: molecular number density in molecules/Å³.

WATER_MODEL_PARAMS: dict[str, dict[str, float]] = {
    "tip4p": {
        "bulk_energy_kcal": -19.67,
        "bulk_density": 0.033148,
    },
    "tip4pew": {
        "bulk_energy_kcal": -19.53,    # estimated from TIP4P-Ew ΔH_vap ratio
        "bulk_density": 0.03328,       # calibrated for TIP4P-Ew
    },
    "tip3p": {
        "bulk_energy_kcal": -19.30,    # estimated from TIP3P ΔH_vap
        "bulk_density": 0.03332,       # calibrated for TIP3P
    },
}

# Legacy single-value constants default to TIP4P values.

BULK_WATER_ENERGY = WATER_MODEL_PARAMS["tip4p"]["bulk_energy_kcal"]
BULK_WATER_DENSITY = WATER_MODEL_PARAMS["tip4p"]["bulk_density"]
TIP4P_EXCESS_CHEMICAL_POTENTIAL = -6.058  # kcal/mol (calibrated for TIP4P, 2 fs timestep)

# GCMC chemical potentials keyed by (water_model, timestep_fs).


GCMC_CHEMICAL_POTENTIALS: dict[tuple[str, float], dict[str, float]] = {
    ("TIP4P", 2.0): {
        "mu_excess": -6.058,
        "solvent_density": 0.033148,
    },
    ("TIP3P", 2.0): {
        "mu_excess": -6.324,
        "solvent_density": 0.03332,
    },
    ("TIP4PEW", 2.0): {
        "mu_excess": -6.20,
        "solvent_density": 0.03328,
    },
}


def get_water_model_params(water_model: str) -> dict[str, float]:
    """Return bulk energy and density for the given water model.

    Falls back to TIP4P values when the model is not in the calibration table.
    """
    return WATER_MODEL_PARAMS.get(water_model.lower(),
                                  WATER_MODEL_PARAMS["tip4p"])


def get_gcmc_params(water_model: str, timestep_fs: float = 2.0) -> dict[str, float]:
    """Look up GCMC chemical potential and solvent density.

    Calibration values are keyed by water model and timestep in femtoseconds.
    """
    key = (water_model.upper().replace("-", "").replace("_", ""), timestep_fs)
    if key not in GCMC_CHEMICAL_POTENTIALS:
        raise ValueError(
            f"Water model {water_model!r} not calibrated for GCMC at "
            f"{timestep_fs} fs timestep.  Calibrated combinations: "
            f"{list(GCMC_CHEMICAL_POTENTIALS.keys())}"
        )
    return GCMC_CHEMICAL_POTENTIALS[key]

# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------
DEFAULT_EQUILIBRATION_STAGES = [
    {

        "name": "brownian_nvt",
        "ensemble": "NVT",
        "integrator": "Brownian",
        "temperature_K": 10.0,
        "time_ps": 100.0,
        # RESPA timesteps [bonded, near, far] in ps
        "timestep_ps": (0.001, 0.001, 0.003),
        "restraint_k": 50.0,  # kcal/mol/Å²
        "restrain": "solute_heavy",
        "delta_max": 0.1,
    },
    {

        "name": "langevin_nvt",
        "ensemble": "NVT",
        "integrator": "Langevin",
        "temperature_K": 10.0,
        "time_ps": 12.0,
        # RESPA timesteps [0.001 0.001 0.003]
        "timestep_ps": (0.001, 0.001, 0.003),
        "thermostat_tau_ps": 0.1,
        "restraint_k": 5.0,
        "restrain": "non_water_heavy",
        "randomize_velocity_interval_ps": 1.0,
    },
    {

        "name": "langevin_npt_cold",
        "ensemble": "NPT",
        "integrator": "Langevin",
        "temperature_K": 10.0,
        "time_ps": 12.0,
        "thermostat_tau_ps": 0.1,
        "barostat_tau_ps": 50.0,
        "restraint_k": 5.0,
        "restrain": "non_water_heavy",
    },
    {

        "name": "mtk_npt",
        "ensemble": "NPT",
        "integrator": "MTK",
        "temperature_K": 300.0,
        "time_ps": 60.0,
        "thermostat_tau_ps": 0.3,
        "barostat_tau_ps": 2.0,
        "restraint_k": 5.0,
        "restrain": "non_water_heavy",
    },
]

# GCMC defaults - shared parameters that don't depend on water model.
# Water-model-specific values (mu_excess, solvent_density) are looked up
# dynamically via ``get_gcmc_params(water_model)`` at runtime.

DEFAULT_GCMC_PARAMS = {
    "temperature_K": 300.0,
    "time_ps": 1000.0,
    "timestep_fs": 2.0,
    "seed": 1234,
    "reservoir_size": 4000,
    "interval_ps": 1.0,
    "moves_per_cycle": 34000,
    "exclusion_radius_A": 1.5,
    # mu_excess and solvent_density intentionally omitted here - they MUST
    # be looked up per water model via get_gcmc_params() to avoid the bug
    # where TIP4P values were hardcoded with a TIP4PEW water model.

}


DEFAULT_POST_GCMC_EQUIL = {
    "name": "langevin_npt_post_gcmc",
    "ensemble": "NPT",
    "integrator": "Langevin",
    "temperature_K": 300.0,
    "time_ps": 120.0,
    "thermostat_tau_ps": 0.1,
    "barostat_tau_ps": 50.0,
    "restraint_k": 5.0,
    "restrain": "non_water_heavy",
}

# A short restrained cleanup immediately after loch Stage 7 so the saved
# ghost-free GCMC checkpoint is already well-behaved before Stage 8 starts.
# The 5 kcal/mol/A^2 restraint strength matches the downstream post-GCMC
# re-equilibration convention.
DEFAULT_GCMC_HANDOFF_RELAX = {
    "name": "gcmc_handoff_relax",
    "ensemble": "NVT",
    "integrator": "Brownian",
    "temperature_K": 10.0,
    "time_ps": 0.05,
    "timestep_ps": 0.001,
    "restraint_k": 5.0,
    "restrain": "non_water_heavy",
    "delta_max": 0.05,
}


DEFAULT_PRODUCTION_PARAMS = {
    "ensemble": "NPT",
    "temperature_K": 300.0,
    "time_ps": 2000.0,
    "trajectory_interval_ps": 1.5,
    "pressure_atm": 1.0,
}

# ---------------------------------------------------------------------------
# Clustering defaults
# ---------------------------------------------------------------------------
DEFAULT_CLUSTERING_PARAMS = {
    "hs_radius_A": 1.0,
    "dcutoff_A": 2.0,
    "grid_spacing_A": 0.5,
    "enable_continuous": True,
    "twobody_entropy": False,
    "active_site_distance_A": 10.0,
}

# ---------------------------------------------------------------------------
# Scoring defaults
# ---------------------------------------------------------------------------
DEFAULT_SCORING_PARAMS = {
    "scale_factor": 0.8,
    "energy_bulk_kcal": BULK_WATER_ENERGY,
}


# ---------------------------------------------------------------------------
# Main config dataclass
# ---------------------------------------------------------------------------
@dataclass
class WaterMapConfig:
    """Full configuration for an H2OSite run.

    Defaults to AMBER14 and TIP3P for the loch-backed GCMC workflow.
    TIP4P-Ew is available for non-GCMC workflows and future calibrated GCMC
    integrations. The tip4p option selects TIP4P thermodynamic reference
    constants while using OpenMM's amber14/tip4pew.xml internally.
    """

    # Input
    protein_path: str | Path = ""
    ligand_path: str | Path = ""
    output_dir: str | Path = "watermap_output"
    jobname: str = "watermap"

    # Structure preparation

    retain_ligand: bool = False
    ligand_resnames: list[str] = field(default_factory=list)
    truncate_protein: bool = True
    truncate_distance_A: float = 20.0
    ligand_distance_A: float = 10.0
    treat_solvent: Literal["as_solvent", "as_solute", "delete"] = "as_solvent"
    normalize_schrodinger_residues: bool = False
    manual_histidine_states: dict[str, str] = field(default_factory=dict)

    # Force field


    # Default: tip3p - keeps the main GCMC path aligned with loch's default
    # calibration. tip4pew remains available as a backup option.
    # Set to "tip4p" to use TIP4P thermodynamic reference constants.
    water_model: str = "tip3p"
    forcefield: str = "amber14-all.xml"  # OpenMM FF XML (OPLS3e unavailable)
    water_forcefield: str = ""           # auto-resolved from water_model
    ligand_forcefield: str = "openff-2.1.0.offxml"
    ligand_parameter_cache: str | Path | None = None

    # System building

    neutralize_system: bool = False

    # Terminal treatment for protein chains
    # "caps"                  - add ACE/NME neutral caps (default; preferred
    #                           for truncated proteins where termini are
    #                           artifacts of distance-based truncation). No
    #                           addHydrogens is called.
    # "add_hydrogens"         - fill missing H on natural termini (charged
    #                           NH3+/COO-). After adding H, only the newly
    #                           added atoms are minimized (the rest of the
    #                           structure is restrained).
    # "preserve_existing_caps" - keep terminal atoms/caps exactly as provided
    #                           in the input structure. This is the right mode
    #                           for already capped inputs such as ACE/NME or
    #                           ACE/NMA-capped receptors; the prep
    #                           layer will normalize NMA -> NME for OpenMM.
    terminal_treatment: Literal[
        "caps",
        "add_hydrogens",
        "preserve_existing_caps",
    ] = "caps"
    deduplicate_atoms: bool = False

    # Simulation
    simulation_time_ns: float = 5.0
    do_not_return_trajectory: bool = False
    skip_gcmc: bool = False
    twobody_entropy: bool = False
    num_cpus: int = 1
    gpu_index: str = "0"
    random_seed: int = 42

    # GCMC runtime parameters
    # gcmc_n_moves = total trial attempts, divided evenly across n_cycles.
    gcmc_n_moves: int = 68000
    gcmc_n_cycles: int = 200
    gcmc_md_steps_per_cycle: int = 500
    gcmc_n_ghosts: int = 300
    # loch/Sire GCMC-MD controls.
    gcmc_flexible_water: bool = True
    gcmc_timestep_fs: float = 1.0
    gcmc_minimize_after_customization: bool = True
    gcmc_redraw_velocities_after_initialise: bool = False
    gcmc_ghost_min_distance_A: float = 1.5
    gcmc_reference_selection: str | None = None
    gcmc_batch_size: int | None = None
    gcmc_bulk_sampling_probability: float = 0.1
    gcmc_standard_volume_A3: float = 30.543
    gcmc_platform: Literal["auto", "cuda", "opencl"] = "auto"
    gcmc_cache_dir: str | Path | None = None
    gcmc_restraint_k_kcal: float = 5.0
    run_gcmc_handoff_relaxation: bool = True
    gcmc_handoff_relaxation_stages: list[dict] = field(default_factory=list)
    generate_display_outputs: bool = True
    display_nearby_radius_A: float = 5.0

    # Post-GCMC re-equilibration stages
    # If empty, populated from DEFAULT_POST_GCMC_EQUIL in __post_init__.
    run_post_gcmc_equilibration: bool = True
    run_post_gcmc_without_gcmc: bool = False
    post_gcmc_equil_stages: list[dict] = field(default_factory=list)

    # RESPA multi-timestep integration
    # When enabled, OpenMM's
    # MTSLangevinIntegrator splits reciprocal-space PME (slow) from
    # bonded + direct nonbonded (fast).  See protocol.py for force-group
    # assignment details.
    use_respa: bool = True

    # Production thermostat

    # Use MonteCarloBarostat for pressure coupling.
    # "langevin"     - LangevinMiddleIntegrator, friction=1.0/ps (default)
    # Weak-coupling mode uses LangevinMiddleIntegrator, friction=0.01/ps.
    production_thermostat: Literal["langevin", "desmond_like"] = "langevin"
    production_restraint_k_kcal: float = 5.0

    # Energy collection mode
    # Energy collection can run during production or during postprocessing.
    # "post"   - collect from saved trajectory after production (default)
    # "during" - collect per-frame energy data during production MD
    #            (per-frame energy sampling during MD)
    watermap_data_collection: Literal["post", "during"] = "post"

    # Clustering
    hs_radius_A: float = 1.0
    clustering_cutoff_A: float = 2.0
    grid_spacing_A: float = 0.5

    # Scoring
    scale_factor: float = 0.8

    # Equilibration override (if empty, uses defaults)
    equilibration_stages: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.protein_path = Path(self.protein_path) if self.protein_path else Path()
        self.ligand_path = Path(self.ligand_path) if self.ligand_path else Path()
        self.output_dir = Path(self.output_dir)
        if self.gcmc_cache_dir:
            self.gcmc_cache_dir = Path(self.gcmc_cache_dir)
        if self.ligand_parameter_cache:
            self.ligand_parameter_cache = Path(self.ligand_parameter_cache)
        if not self.equilibration_stages:
            self.equilibration_stages = list(DEFAULT_EQUILIBRATION_STAGES)
        if not self.gcmc_handoff_relaxation_stages:
            self.gcmc_handoff_relaxation_stages = [
                dict(DEFAULT_GCMC_HANDOFF_RELAX)
            ]
        if not self.post_gcmc_equil_stages:
            self.post_gcmc_equil_stages = [dict(DEFAULT_POST_GCMC_EQUIL)]
        # Auto-resolve water force-field XML if not explicitly set
        if not self.water_forcefield:
            _wff_map = {
                "tip3p": "amber14/tip3p.xml",
                "tip4p": "amber14/tip4pew.xml",  # OpenMM ships TIP4P-Ew XML
                "tip4pew": "amber14/tip4pew.xml",
                "tip5p": "amber14/tip5p.xml",
                "tip3pfb": "amber14/tip3pfb.xml",
                "tip4pfb": "amber14/tip4pfb.xml",
            }
            self.water_forcefield = _wff_map.get(
                self.water_model.lower(), "amber14/tip4pew.xml"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> WaterMapConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        import dataclasses
        import json

        d = dataclasses.asdict(self)
        d["protein_path"] = str(d["protein_path"])
        d["ligand_path"] = str(d["ligand_path"])
        d["output_dir"] = str(d["output_dir"])
        # Round-trip through JSON to convert tuples -> lists (YAML-safe)
        d = json.loads(json.dumps(d))
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @property
    def production_time_ps(self) -> float:
        return self.simulation_time_ns * 1000.0
