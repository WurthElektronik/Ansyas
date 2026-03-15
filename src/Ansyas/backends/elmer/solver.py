"""
Elmer solver backend implementation.

Uses ElmerSolver via pyelmer for FEM solving.
"""

import os
import math
import numpy as np
from typing import Any, Dict, List, Optional, Union

try:
    from pyelmer import elmer, execute
    from pyelmer.post import scan_logfile
    HAS_PYELMER = True
except ImportError:
    HAS_PYELMER = False

try:
    from ... import MAS_models as MAS
except ImportError:
    try:
        import MAS_models as MAS
    except ImportError:
        MAS = None

from ..base import (
    SolverBackend,
    SolverSetup,
    GeometryObject,
)


class ElmerSolverBackend(SolverBackend):
    """
    Solver backend using ElmerSolver via pyelmer.
    
    Supports:
    - EddyCurrent / AC Magnetic (MagnetoDynamics with harmonic solver)
    - Magnetostatic (WhitneyAVSolver or MagnetoStatics)
    - Thermal / SteadyState (HeatSolver)
    - Electrostatic (StatElecSolver)
    - Transient (MagnetoDynamics with time stepping)
    
    Workflow:
    1. Create simulation and configure solvers
    2. Add materials, bodies, boundaries from other backends
    3. Write SIF file
    4. Execute ElmerSolver
    5. Parse results
    """
    
    # Mapping from Ansyas solver types to Elmer solver procedures
    SOLVER_MAP = {
        "EddyCurrent": {
            "solver": "MagnetoDynamics",
            "procedure": '"MagnetoDynamics" "WhitneyAVHarmonicSolver"',
            "calc_fields": True,
        },
        "AC Magnetic": {
            "solver": "MagnetoDynamics",
            "procedure": '"MagnetoDynamics" "WhitneyAVHarmonicSolver"',
            "calc_fields": True,
        },
        "Magnetostatic": {
            "solver": "MagnetoStatics",
            "procedure": '"MagnetoStatics" "WhitneyAVSolver"',
            "calc_fields": True,
        },
        "Transient": {
            "solver": "MagnetoDynamics",
            "procedure": '"MagnetoDynamics" "WhitneyAVSolver"',
            "calc_fields": True,
            "transient": True,
        },
        "TransientAPhiFormulation": {
            "solver": "MagnetoDynamics",
            "procedure": '"MagnetoDynamics" "WhitneyAVSolver"',
            "calc_fields": True,
            "transient": True,
        },
        "SteadyState": {
            "solver": "HeatSolver",
            "procedure": '"HeatSolve" "HeatSolver"',
            "calc_fields": False,
        },
        "Thermal": {
            "solver": "HeatSolver",
            "procedure": '"HeatSolve" "HeatSolver"',
            "calc_fields": False,
        },
        "Electrostatic": {
            "solver": "StatElecSolver",
            "procedure": '"StatElecSolve" "StatElecSolver"',
            "calc_fields": True,
        },
    }
    
    def __init__(self):
        self._simulation: Optional["elmer.Simulation"] = None
        self._solver_type: Optional[str] = None
        self._sim_dir: Optional[str] = None
        self._setup: Optional[SolverSetup] = None
        self._solvers: Dict[str, "elmer.Solver"] = {}
        self._equations: Dict[str, "elmer.Equation"] = {}
        self._bodies: Dict[str, "elmer.Body"] = {}
        self._frequency: Optional[float] = None
        self._frequencies: List[float] = []
        self._results: Dict[str, Any] = {}
    
    def initialize(self, solver_type: str = None, sim_dir: str = None, **kwargs) -> None:
        """
        Initialize the solver backend.
        
        Args:
            solver_type: Type of simulation ("EddyCurrent", "Thermal", etc.)
            sim_dir: Directory for simulation files
        """
        if not HAS_PYELMER:
            raise ImportError(
                "pyelmer is required for ElmerSolverBackend. "
                "Install with: pip install pyelmer"
            )
        
        self._solver_type = solver_type or "EddyCurrent"
        self._sim_dir = sim_dir or os.getcwd()
        
        os.makedirs(self._sim_dir, exist_ok=True)
        
        # Create pyelmer simulation
        self._simulation = elmer.Simulation()
        
        # Configure basic simulation settings
        self._configure_simulation_type()
    
    def _configure_simulation_type(self) -> None:
        """Configure simulation based on solver type."""
        solver_config = self.SOLVER_MAP.get(self._solver_type, self.SOLVER_MAP["EddyCurrent"])
        
        if solver_config.get("transient", False):
            self._simulation.data = {
                "Simulation Type": "Transient",
                "Coordinate System": "Cartesian 3D",
                "Timestepping Method": "BDF",
                "BDF Order": 2,
            }
        else:
            self._simulation.data = {
                "Simulation Type": "Steady State",
                "Coordinate System": "Cartesian 3D",
                "Steady State Max Iterations": 1,
            }
    
    def get_simulation(self) -> "elmer.Simulation":
        """Get the pyelmer simulation object."""
        return self._simulation
    
    def create_setup(self, setup: SolverSetup) -> Any:
        """
        Create a solver setup with the given configuration.
        
        Args:
            setup: SolverSetup configuration
            
        Returns:
            Main solver object
        """
        self._setup = setup
        self._frequency = setup.frequency
        
        solver_config = self.SOLVER_MAP.get(setup.solver_type, self.SOLVER_MAP["EddyCurrent"])
        
        # Create main solver
        main_solver = elmer.Solver(self._simulation, solver_config["solver"])
        main_solver.data = self._build_solver_data(setup, solver_config)
        self._solvers["main"] = main_solver
        
        # Create CalcFields solver for magnetic simulations
        if solver_config.get("calc_fields", False):
            calc_solver = self._create_calc_fields_solver(setup)
            self._solvers["calc_fields"] = calc_solver
        
        # Create result output solver
        output_solver = self._create_output_solver()
        self._solvers["output"] = output_solver
        
        # Create equation combining all solvers
        equation = elmer.Equation(
            self._simulation, 
            "MainEquation",
            list(self._solvers.values())
        )
        self._equations["main"] = equation
        
        return main_solver
    
    def _build_solver_data(self, setup: SolverSetup, solver_config: Dict) -> Dict:
        """Build solver configuration dictionary.
        
        Note: Use direct solver (UMFPACK) for robustness. Iterative solvers
        like BiCGStab tend to diverge on edge-element magnetostatic problems.
        """
        # Default to direct solver for robustness
        # Iterative solvers diverge on many magnetostatic problems
        use_direct = True
        
        if use_direct:
            data = {
                "Equation": solver_config["solver"],
                "Procedure": solver_config["procedure"],
                "Variable": self._get_solver_variable(solver_config["solver"]),
                "Linear System Solver": "Direct",
                "Linear System Direct Method": "UMFPACK",
                "Steady State Convergence Tolerance": setup.max_error_percent / 100.0,
            }
        else:
            data = {
                "Equation": solver_config["solver"],
                "Procedure": solver_config["procedure"],
                "Variable": self._get_solver_variable(solver_config["solver"]),
                "Linear System Solver": "Iterative",
                "Linear System Iterative Method": "BiCGStabl",
                "Linear System Max Iterations": setup.max_passes * 100,
                "Linear System Convergence Tolerance": setup.max_error_percent / 100.0,
                "Linear System Preconditioning": "ILU1",
                "BicgstabL polynomial degree": 4,
                "Linear System Residual Output": 10,
                "Optimize Bandwidth": True,
            }
        
        # Add frequency for harmonic solver
        if setup.frequency and self._solver_type in ["EddyCurrent", "AC Magnetic"]:
            angular_freq = 2 * math.pi * setup.frequency
            data["Angular Frequency"] = angular_freq
        
        # Add transient settings
        if solver_config.get("transient", False):
            if setup.time_step:
                self._simulation.data["Timestep Sizes"] = setup.time_step
            if setup.stop_time:
                self._simulation.data["Timestep Intervals"] = int(setup.stop_time / (setup.time_step or 1e-6))
        
        return data
    
    def _get_solver_variable(self, solver_name: str) -> str:
        """Get the primary variable for a solver.
        
        Note: Use simple "AV" for magnetostatic/magnetodynamic problems.
        Complex variable definitions like "AV[AV re:1 AV im:1]" are only
        needed for harmonic analysis.
        """
        variables = {
            "MagnetoDynamics": "AV",  # Simple variable - works for both static and dynamic
            "MagnetoStatics": "AV",
            "HeatSolver": "Temperature",
            "StatElecSolver": "Potential",
        }
        return variables.get(solver_name, "Field")
    
    def _create_calc_fields_solver(self, setup: SolverSetup) -> "elmer.Solver":
        """Create MagnetoDynamicsCalcFields solver for field post-processing.
        
        Note: Use direct solver (UMFPACK) to match main solver for stability.
        """
        solver = elmer.Solver(self._simulation, "MgDynCalcFields")
        solver.data = {
            "Equation": "MgDynCalcFields",
            "Procedure": '"MagnetoDynamics" "MagnetoDynamicsCalcFields"',
            "Potential Variable": '"AV"',  # Must match main solver variable
            "Calculate Magnetic Field Strength": True,
            "Calculate Magnetic Flux Density": True,
            "Calculate Joule Heating": True,
            "Calculate Current Density": True,
            "Calculate Electric Field": True,
            "Calculate Nodal Forces": True,
            "Calculate Maxwell Stress": False,
            "Linear System Solver": "Direct",
            "Linear System Direct Method": "UMFPACK",
            "Steady State Convergence Tolerance": 1.0e-6,
        }
        
        if setup.frequency:
            solver.data["Angular Frequency"] = 2 * math.pi * setup.frequency
        
        return solver
    
    def _create_output_solver(self) -> "elmer.Solver":
        """Create result output solver."""
        solver = elmer.Solver(self._simulation, "ResultOutput")
        solver.data = {
            "Equation": "ResultOutput",
            "Procedure": '"ResultOutputSolve" "ResultOutputSolver"',
            "Output File Name": "results",
            "Output Format": "vtu",
            "Vtu Format": True,
            "Binary Output": True,
            "Single Precision": True,
            "Save Geometry IDs": True,
        }
        return solver
    
    def add_frequency_sweep(
        self,
        setup: Any,
        start_freq: float,
        stop_freq: float,
        step_size: float,
        sweep_type: str = "LinearStep"
    ) -> bool:
        """
        Add a frequency sweep to the setup.
        
        Note: Elmer handles frequency sweeps differently than Ansys.
        We store the frequencies and run multiple steady-state solutions.
        """
        try:
            if sweep_type == "LinearStep":
                self._frequencies = list(np.arange(start_freq, stop_freq + step_size, step_size))
            else:
                # Logarithmic sweep
                self._frequencies = list(np.logspace(
                    np.log10(start_freq), 
                    np.log10(stop_freq), 
                    int((np.log10(stop_freq) - np.log10(start_freq)) / np.log10(1 + step_size/start_freq))
                ))
            return True
        except Exception:
            return False
    
    def add_default_frequency_sweeps(
        self, 
        setup: Any = None, 
        single_frequency: float = None
    ) -> bool:
        """Add default frequency sweeps for magnetic simulations."""
        if single_frequency is not None:
            self._frequencies = [single_frequency]
        else:
            # Default sweep similar to Ansys
            self._frequencies = [1, 10, 100, 1000, 10000, 100000]
        return True
    
    def create_body(
        self,
        name: str,
        physical_groups: List[int],
        material: "elmer.Material",
        equation: "elmer.Equation" = None,
        body_force: "elmer.BodyForce" = None,
        initial_condition: "elmer.InitialCondition" = None
    ) -> "elmer.Body":
        """Create an Elmer body with assigned properties."""
        body = elmer.Body(self._simulation, name, physical_groups)
        body.material = material
        body.equation = equation or self._equations.get("main")
        
        if body_force:
            body.body_force = body_force
        if initial_condition:
            body.initial_condition = initial_condition
        
        self._bodies[name] = body
        return body
    
    def analyze(self) -> bool:
        """Run the simulation."""
        try:
            # Write simulation files
            self._simulation.write_startinfo(self._sim_dir)
            self._simulation.write_sif(self._sim_dir)
            
            # Run solver
            if self._frequencies and len(self._frequencies) > 1:
                # Frequency sweep - run multiple times
                return self._run_frequency_sweep()
            else:
                # Single frequency or steady-state
                return self._run_single()
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            return False
    
    def _run_single(self) -> bool:
        """Run a single simulation."""
        try:
            execute.run_elmer_solver(self._sim_dir)
            
            # Check for errors
            errors, warnings, stats = scan_logfile(self._sim_dir)
            
            if errors:
                print(f"Elmer errors: {errors}")
                return False
            
            if warnings:
                print(f"Elmer warnings: {warnings}")
            
            self._results["stats"] = stats
            return True
            
        except Exception as e:
            print(f"ElmerSolver execution failed: {e}")
            return False
    
    def _run_frequency_sweep(self) -> bool:
        """Run frequency sweep by executing multiple simulations."""
        all_results = []
        
        for freq in self._frequencies:
            # Update frequency in solver
            if "main" in self._solvers:
                self._solvers["main"].data["Angular Frequency"] = 2 * math.pi * freq
            if "calc_fields" in self._solvers:
                self._solvers["calc_fields"].data["Angular Frequency"] = 2 * math.pi * freq
            
            # Rewrite SIF with new frequency
            self._simulation.write_sif(self._sim_dir)
            
            # Run simulation
            if not self._run_single():
                print(f"Failed at frequency {freq} Hz")
                continue
            
            # Store results for this frequency
            result = self._extract_results_for_frequency(freq)
            all_results.append(result)
        
        self._results["frequency_sweep"] = all_results
        return len(all_results) > 0
    
    def _extract_results_for_frequency(self, frequency: float) -> Dict:
        """Extract results for a specific frequency."""
        return {
            "frequency": frequency,
            # Results would be extracted from VTU files
        }
    
    def get_solution_data(
        self,
        expressions: List[str],
        context: Optional[Dict] = None
    ) -> Any:
        """Get solution data for post-processing."""
        # Parse VTU result files
        return self._results
    
    def get_impedance_matrix(self) -> List[Dict]:
        """
        Get impedance matrix results.
        
        Calculates Z = R + jωL from inductance and resistance.
        """
        l_matrix = self.get_inductance_matrix()
        r_matrix = self.get_resistance_matrix()
        
        z_matrix = []
        for l_data, r_data in zip(l_matrix, r_matrix):
            freq = l_data["frequency"]
            omega = 2 * math.pi * freq
            
            # Z = R + jωL
            n = len(l_data["magnitude"])
            z_mag = []
            z_phase = []
            
            for i in range(n):
                z_mag_row = []
                z_phase_row = []
                for j in range(n):
                    l_val = l_data["magnitude"][i][j]["nominal"]
                    r_val = r_data["magnitude"][i][j]["nominal"]
                    
                    # Complex impedance
                    z_real = r_val
                    z_imag = omega * l_val
                    z_abs = math.sqrt(z_real**2 + z_imag**2)
                    z_angle = math.degrees(math.atan2(z_imag, z_real))
                    
                    z_mag_row.append({"nominal": z_abs})
                    z_phase_row.append({"nominal": z_angle})
                
                z_mag.append(z_mag_row)
                z_phase.append(z_phase_row)
            
            z_matrix.append({
                "frequency": freq,
                "magnitude": z_mag,
                "phase": z_phase,
            })
        
        return z_matrix
    
    def get_inductance_matrix(self) -> List[Dict]:
        """
        Get inductance matrix results.
        
        For Elmer, inductance is calculated from magnetic energy:
        L = 2 * Wm / I²
        
        For multi-winding: Lij from flux linkage method
        """
        # This would parse the Elmer output files
        # Placeholder structure matching Ansys output format
        inductance_data = []
        
        for freq in self._frequencies or [self._frequency or 100000]:
            # Placeholder - actual implementation would read from result files
            inductance_data.append({
                "frequency": freq,
                "magnitude": [[{"nominal": 0.0}]],  # Would be populated from results
            })
        
        return inductance_data
    
    def get_resistance_matrix(self) -> List[Dict]:
        """
        Get resistance matrix results.
        
        Calculated from Joule losses:
        R = P / I²
        """
        resistance_data = []
        
        for freq in self._frequencies or [self._frequency or 100000]:
            resistance_data.append({
                "frequency": freq,
                "magnitude": [[{"nominal": 0.0}]],
            })
        
        return resistance_data
    
    def get_results(self) -> 'MAS.ImpedanceOutput':
        """Get full impedance output results in MAS format."""
        if MAS is None:
            raise ImportError("MAS_models not available")
        
        impedance_dict = {
            "methodUsed": "Elmer FEM",
            "origin": MAS.ResultOrigin.simulation,
            "inductanceMatrix": self.get_inductance_matrix(),
            "resistanceMatrix": self.get_resistance_matrix(),
            "impedanceMatrix": self.get_impedance_matrix(),
        }
        return MAS.ImpedanceOutput.from_dict(impedance_dict)
    
    def get_field_data(
        self,
        field_type: str,
        objects: Optional[List[GeometryObject]] = None
    ) -> Any:
        """
        Get field data for visualization or export.
        
        Reads from VTU result files.
        """
        # Field type mapping
        field_map = {
            "B": "magnetic flux density",
            "H": "magnetic field strength", 
            "J": "current density",
            "E": "electric field",
            "T": "temperature",
            "V": "potential",
        }
        
        field_name = field_map.get(field_type, field_type)
        
        # Would read from VTU files using meshio or pyvista
        return {"field": field_name, "data": None}
    
    def export_results(self, file_path: str, format: str = "csv") -> bool:
        """Export simulation results to file."""
        try:
            results = {
                "frequencies": self._frequencies,
                "inductance": self.get_inductance_matrix(),
                "resistance": self.get_resistance_matrix(),
                "impedance": self.get_impedance_matrix(),
            }
            
            if format.lower() == "csv":
                self._export_csv(file_path, results)
            elif format.lower() == "json":
                import json
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def _export_csv(self, file_path: str, results: Dict) -> None:
        """Export results to CSV format."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency (Hz)", "L (H)", "R (Ohm)", "|Z| (Ohm)", "Phase (deg)"])
            
            for l_data, r_data, z_data in zip(
                results["inductance"], 
                results["resistance"],
                results["impedance"]
            ):
                freq = l_data["frequency"]
                l_val = l_data["magnitude"][0][0]["nominal"]
                r_val = r_data["magnitude"][0][0]["nominal"]
                z_mag = z_data["magnitude"][0][0]["nominal"]
                z_phase = z_data["phase"][0][0]["nominal"]
                
                writer.writerow([freq, l_val, r_val, z_mag, z_phase])
    
    def save_project(self, file_path: Optional[str] = None) -> bool:
        """Save the project/simulation files."""
        try:
            save_dir = file_path or self._sim_dir
            self._simulation.write_startinfo(save_dir)
            self._simulation.write_sif(save_dir)
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """Close the solver and release resources."""
        self._simulation = None
        self._solvers.clear()
        self._equations.clear()
        self._bodies.clear()
    
    def change_design_settings(self, settings: Dict) -> bool:
        """Change design settings (compatibility with Ansys API)."""
        # Map Ansys settings to Elmer equivalents where possible
        return True
    
    def create_output_variable(self, name: str, expression: str) -> bool:
        """Create an output variable for post-processing."""
        # Would be implemented via MATC expressions in Elmer
        return True
