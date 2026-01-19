"""
Ansys excitation backend implementation.

Uses PyAEDT for winding, coil, and boundary condition setup.
"""

from typing import Any, Dict, List, Optional, Union

from ..base import (
    ExcitationBackend,
    GeometryObject,
)


class AnsysExcitationBackend(ExcitationBackend):
    """
    Excitation backend using Ansys AEDT.
    
    This implementation wraps PyAEDT's excitation and boundary functionality.
    """
    
    def __init__(self):
        self._project = None
        self._windings = {}
        self._matrices = {}
    
    def initialize(self, project=None, **kwargs) -> None:
        """
        Initialize with an existing PyAEDT project.
        
        Args:
            project: A PyAEDT Maxwell3d, Icepak, or similar project instance.
        """
        if project is None:
            raise ValueError("AnsysExcitationBackend requires a PyAEDT project instance")
        self._project = project
    
    def create_winding(
        self,
        name: str,
        winding_type: str,
        amplitude: Union[float, str],
        is_solid: bool = True,
        resistance: float = 0.0,
        inductance: float = 0.0,
        phase: float = 0.0
    ) -> Any:
        """Create a winding excitation."""
        winding = self._project.assign_winding(
            winding_type=winding_type.title(),
            is_solid=is_solid,
            current=amplitude if winding_type.lower() == "current" else 0,
            resistance=resistance,
            inductance=inductance,
            voltage=amplitude if winding_type.lower() == "voltage" else 0,
            parallel_branches=1,
            phase=phase,
            name=name,
        )
        self._windings[name] = winding
        return winding
    
    def assign_coil(
        self,
        face: Any,
        conductors_number: int = 1,
        polarity: str = "Positive",
        name: str = "coil"
    ) -> str:
        """Assign a coil terminal to a face."""
        terminal = self._project.assign_coil(
            assignment=face,
            conductors_number=conductors_number,
            polarity=polarity,
            name=name
        )
        return terminal.name if hasattr(terminal, 'name') else name
    
    def add_coils_to_winding(
        self,
        winding_name: str,
        coil_names: List[str]
    ) -> bool:
        """Add coil terminals to a winding."""
        try:
            for coil_name in coil_names:
                self._project.add_winding_coils(
                    assignment=winding_name,
                    coils=coil_name
                )
            return True
        except Exception:
            return False
    
    def create_matrix(
        self,
        excitations: List[str],
        name: str
    ) -> Any:
        """Create an excitation matrix for multi-winding setups."""
        matrix = self._project.assign_matrix(
            assignment=excitations,
            matrix_name=name
        )
        self._matrices[name] = matrix
        return matrix
    
    def join_parallel(
        self,
        matrix: Any,
        sources: List[str],
        matrix_name: str,
        join_name: str
    ) -> bool:
        """Join parallel windings in a matrix."""
        try:
            matrix.join_parallel(
                sources=sources,
                matrix_name=matrix_name,
                join_name=join_name
            )
            return True
        except Exception:
            return False
    
    def create_waveform_dataset(
        self,
        name: str,
        time: List[float],
        data: List[float]
    ) -> Any:
        """Create a waveform dataset for transient excitation."""
        return self._project.create_dataset(
            name=name,
            x=time,
            y=data,
            is_project_dataset=False
        )
    
    def assign_floating(
        self,
        objects: List[GeometryObject],
        charge: float = 0.0,
        name: str = "floating"
    ) -> bool:
        """Assign floating boundary condition (for electrostatics)."""
        native_objs = [
            obj.native_object.name if isinstance(obj, GeometryObject) else obj.name
            for obj in objects
        ]
        try:
            self._project.assign_floating(
                assignment=native_objs,
                charge_value=charge,
                name=name
            )
            return True
        except Exception:
            return False
    
    def assign_heat_source(
        self,
        objects: List[GeometryObject],
        power: float,
        name: str = "heat_source"
    ) -> bool:
        """Assign heat source (for thermal simulations)."""
        native_objs = [
            obj.native_object.name if isinstance(obj, GeometryObject) else obj.name
            for obj in objects
        ]
        try:
            self._project.assign_source(
                assignment=native_objs,
                thermal_condition="Total Power",
                assignment_value=f"{power}W",
                boundary_name=name
            )
            return True
        except Exception:
            return False
    
    def assign_temperature_source(
        self,
        objects: List[GeometryObject],
        temperature: float,
        name: str = "temperature_source"
    ) -> bool:
        """Assign fixed temperature boundary condition."""
        native_objs = [
            obj.native_object.name if isinstance(obj, GeometryObject) else obj.name
            for obj in objects
        ]
        try:
            self._project.assign_source(
                assignment=native_objs,
                thermal_condition="Temperature",
                assignment_value=f"{temperature}cel",
                boundary_name=name
            )
            return True
        except Exception:
            return False
    
    def set_core_losses(
        self,
        objects: List[GeometryObject],
        on_field: bool = True
    ) -> bool:
        """Enable core loss calculation for magnetic cores."""
        native_objs = [
            obj.native_object.name if isinstance(obj, GeometryObject) else obj.name
            for obj in objects
        ]
        try:
            self._project.set_core_losses(
                assignment=native_objs,
                core_loss_on_field=on_field
            )
            return True
        except Exception:
            return False
    
    def assign_pressure_free_opening(
        self,
        faces: List[Any],
        name: str = "opening"
    ) -> bool:
        """Assign pressure-free opening boundary (for thermal/CFD)."""
        try:
            self._project.assign_pressure_free_opening(
                boundary_name=name,
                assignment=[int(str(f)) for f in faces]
            )
            return True
        except Exception:
            return False
    
    def assign_free_opening(
        self,
        faces: Any,
        flow_type: str = "Pressure",
        velocity: List[str] = None,
        temperature: str = "AmbientTemp",
        name: str = "opening"
    ) -> bool:
        """Assign free opening with velocity (for forced convection)."""
        try:
            self._project.assign_free_opening(
                boundary_name=name,
                assignment=faces,
                flow_type=flow_type,
                velocity=velocity or ["0m_per_sec", "0m_per_sec", "0m_per_sec"],
                temperature=temperature
            )
            return True
        except Exception:
            return False
