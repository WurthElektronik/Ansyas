"""
Ansys solver backend implementation.

Uses PyAEDT Maxwell3d/Icepak for FEM solving.
"""

import os
import math
from typing import Any, Dict, List, Optional

from ..base import (
    SolverBackend,
    SolverSetup,
    GeometryObject,
)
import MAS_models as MAS


class AnsysSolverBackend(SolverBackend):
    """
    Solver backend using Ansys AEDT.
    
    This implementation wraps PyAEDT's solve functionality for Maxwell3d
    and Icepak simulations.
    """
    
    def __init__(self):
        self._project = None
        self._setup = None
        self._solver_type = None
        self._frequency = None
    
    def initialize(self, solver_type: str = None, project=None, **kwargs) -> None:
        """
        Initialize with an existing PyAEDT project.
        
        Args:
            solver_type: Type of simulation (used for reference)
            project: A PyAEDT Maxwell3d, Icepak, or similar project instance.
        """
        if project is None:
            raise ValueError("AnsysSolverBackend requires a PyAEDT project instance")
        self._project = project
        self._solver_type = solver_type or project.solution_type
    
    def create_setup(self, setup: SolverSetup) -> Any:
        """Create a solver setup with the given configuration."""
        self._setup = self._project.create_setup("Setup")
        self._frequency = setup.frequency
        
        if self._solver_type in ["Transient", "TransientAPhiFormulation"]:
            if setup.stop_time:
                self._setup.props["StopTime"] = f"{setup.stop_time}s"
            if setup.time_step:
                self._setup.props["TimeStep"] = f"{setup.time_step}s"
        
        elif self._solver_type in ["EddyCurrent", "AC Magnetic"]:
            if setup.frequency:
                self._setup.props["Frequency"] = f"{setup.frequency * 3}"
            self._setup.props["PercentRefinement"] = setup.refinement_percent
            self._setup.props["MaximumPasses"] = setup.max_passes
            self._setup.props["PercentError"] = setup.max_error_percent
            self._setup.props["HasSweepSetup"] = True
        
        elif self._solver_type == "Electrostatic":
            if setup.frequency:
                self._setup.props["Frequency"] = f"{setup.frequency * 3}"
            self._setup.props["PercentRefinement"] = setup.refinement_percent
            self._setup.props["MaximumPasses"] = setup.max_passes
            self._setup.props["PercentError"] = setup.max_error_percent
        
        return self._setup
    
    def add_frequency_sweep(
        self,
        setup: Any,
        start_freq: float,
        stop_freq: float,
        step_size: float,
        sweep_type: str = "LinearStep"
    ) -> bool:
        """Add a frequency sweep to the setup."""
        try:
            setup.add_eddy_current_sweep(
                sweep_type=sweep_type,
                start_frequency=start_freq,
                stop_frequency=stop_freq,
                step_size=step_size,
                units="Hz",
                clear=False,
                save_all_fields=True
            )
            return True
        except Exception:
            return False
    
    def add_default_frequency_sweeps(self, setup: Any = None) -> bool:
        """Add default frequency sweeps for magnetic simulations."""
        if setup is None:
            setup = self._setup
        
        try:
            setup.add_eddy_current_sweep(
                sweep_type="LinearStep",
                start_frequency=1,
                stop_frequency=100000,
                step_size=10000,
                units="Hz",
                clear=False,
                save_all_fields=True
            )
            setup.add_eddy_current_sweep(
                sweep_type="LinearStep",
                start_frequency=100000,
                stop_frequency=1000000,
                step_size=100000,
                units="Hz",
                clear=False,
                save_all_fields=True
            )
            return True
        except Exception:
            return False
    
    def analyze(self) -> bool:
        """Run the simulation."""
        try:
            self._project.analyze()
            return True
        except Exception:
            return False
    
    def get_solution_data(
        self,
        expressions: List[str],
        context: Optional[Dict] = None
    ) -> Any:
        """Get solution data for post-processing."""
        return self._project.post.get_solution_data(
            expressions=expressions,
            context=context
        )
    
    def get_impedance_matrix(self) -> Dict:
        """Get impedance matrix results."""
        return self._get_matrix_data("Z", include_phase=True)
    
    def get_inductance_matrix(self) -> Dict:
        """Get inductance matrix results."""
        return self._get_matrix_data("L", include_phase=False)
    
    def get_resistance_matrix(self) -> Dict:
        """Get resistance matrix results."""
        return self._get_matrix_data("R", include_phase=False)
    
    def _get_matrix_data(self, category: str, include_phase: bool = False) -> List[Dict]:
        """Get matrix data for a specific category (L, R, or Z)."""
        category_data = []
        context = {"solution_matrix": "windings"}
        
        available_quantities = self._project.post.available_report_quantities(
            context=context,
            quantities_category=category
        )
        data = self._project.post.get_solution_data(
            expressions=available_quantities,
            context=context
        )
        
        number_windings = int(math.sqrt(len(available_quantities)))
        
        # Determine frequency multiplier
        frequency_multiplier = 1e9
        if data.units_sweeps == "GHz":
            frequency_multiplier = 1e9
        elif data.units_sweeps == "MHz":
            frequency_multiplier = 1e6
        elif data.units_sweeps == "kHz":
            frequency_multiplier = 1e3
        elif data.units_sweeps == "Hz":
            frequency_multiplier = 1
        
        for frequency in [x * frequency_multiplier for x in data.primary_sweep_values]:
            matrix_per_frequency = {"frequency": frequency, "magnitude": []}
            for _ in range(number_windings):
                matrix_per_frequency["magnitude"].append([None] * number_windings)
            if include_phase:
                matrix_per_frequency["phase"] = []
                for _ in range(number_windings):
                    matrix_per_frequency["phase"].append([None] * number_windings)
            category_data.append(matrix_per_frequency)
        
        for expression_index, expression in enumerate(available_quantities):
            horizontal_winding_index = int(math.floor(expression_index / number_windings))
            vertical_winding_index = expression_index % number_windings
            
            _, data_per_frequency = data.get_expression_data(
                expression=expression,
                formula="mag",
                convert_to_SI=True
            )
            for frequency_index, datum in enumerate(data_per_frequency):
                category_data[frequency_index]["magnitude"][horizontal_winding_index][vertical_winding_index] = {"nominal": datum}
            
            if include_phase:
                _, data_per_frequency = data.get_expression_data(
                    expression=expression,
                    formula="phasedeg"
                )
                for frequency_index, datum in enumerate(data_per_frequency):
                    category_data[frequency_index]["phase"][horizontal_winding_index][vertical_winding_index] = {"nominal": datum}
        
        return category_data
    
    def get_results(self) -> 'MAS.ImpedanceOutput':
        """Get full impedance output results."""
        impedance_dict = {
            "methodUsed": "Ansys Maxwell",
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
        """Get field data for visualization or export."""
        # Field data access would require more complex implementation
        # depending on the specific field type and requirements
        raise NotImplementedError("Field data extraction not yet implemented")
    
    def export_results(self, file_path: str, format: str = "csv") -> bool:
        """Export simulation results to file."""
        try:
            folder = os.path.dirname(file_path)
            self._project.export_results(export_folder=folder)
            return True
        except Exception:
            return False
    
    def save_project(self, file_path: Optional[str] = None) -> bool:
        """Save the project/simulation."""
        try:
            self._project.oeditor.CleanUpModel()
            if file_path:
                self._project.save_project(file_name=file_path)
            else:
                self._project.save_project()
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """Close the solver and release resources."""
        try:
            self._project.release_desktop(close_projects=True, close_desktop=True)
        except Exception:
            pass
    
    def change_design_settings(self, settings: Dict) -> bool:
        """Change design settings."""
        try:
            self._project.change_design_settings(settings)
            return True
        except Exception:
            return False
    
    def create_output_variable(self, name: str, expression: str) -> bool:
        """Create an output variable for post-processing."""
        try:
            self._project.create_output_variable(name, expression)
            return True
        except Exception:
            return False
    
    def create_report(self, name: str, expressions: List[str] = None) -> Any:
        """Create a report for visualization."""
        report = self._project.post.create_report(plot_name=name)
        if expressions:
            report.add_trace_to_report(expressions)
        return report
