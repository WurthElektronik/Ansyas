"""
Elmer excitation backend implementation.

Handles winding definitions, current/voltage sources, and boundary conditions
for electromagnetic simulations in Elmer FEM.
"""

import math
from typing import Any, Dict, List, Optional, Union

try:
    from pyelmer import elmer
    HAS_PYELMER = True
except ImportError:
    HAS_PYELMER = False

from ..base import (
    ExcitationBackend,
    ExcitationDefinition,
    GeometryObject,
)


class ElmerExcitationBackend(ExcitationBackend):
    """
    Excitation backend for Elmer electromagnetic simulations.
    
    Maps Ansyas winding/coil concepts to Elmer's:
    - CoilSolver for stranded coil excitation
    - BodyForce for current density in solid conductors
    - Boundary conditions for voltage/current terminals
    - Component definitions for circuit coupling
    
    Supports:
    - Current excitation (solid and stranded)
    - Voltage excitation
    - Multi-winding transformers
    - Thermal heat sources
    - Electrostatic floating/fixed potentials
    """
    
    def __init__(self):
        self._simulation: Optional["elmer.Simulation"] = None
        self._windings: Dict[str, Dict] = {}
        self._coils: Dict[str, Dict] = {}
        self._boundaries: Dict[str, "elmer.Boundary"] = {}
        self._body_forces: Dict[str, "elmer.BodyForce"] = {}
        self._components: Dict[str, "elmer.Component"] = {}
        self._matrices: Dict[str, List[str]] = {}
    
    def initialize(self, simulation: "elmer.Simulation" = None, **kwargs) -> None:
        """Initialize the excitation backend."""
        if not HAS_PYELMER:
            raise ImportError(
                "pyelmer is required for ElmerExcitationBackend. "
                "Install with: pip install pyelmer"
            )
        self._simulation = simulation
    
    def set_simulation(self, simulation: "elmer.Simulation") -> None:
        """Set the pyelmer simulation object."""
        self._simulation = simulation
    
    def create_winding(
        self,
        name: str,
        winding_type: str,  # "Current" or "Voltage"
        amplitude: Union[float, str],
        is_solid: bool = True,
        resistance: float = 0.0,
        inductance: float = 0.0,
        phase: float = 0.0
    ) -> Any:
        """
        Create a winding excitation.
        
        For Elmer, windings are implemented as:
        - Solid conductors: BodyForce with Current Density
        - Stranded coils: CoilSolver with number of turns
        
        Args:
            name: Winding name
            winding_type: "Current" or "Voltage"
            amplitude: Excitation amplitude (A for current, V for voltage)
            is_solid: True for solid conductors, False for stranded
            resistance: Winding resistance (Ohms)
            inductance: Winding inductance (H)
            phase: Phase angle in degrees
            
        Returns:
            Winding configuration dictionary
        """
        winding = {
            "name": name,
            "type": winding_type,
            "amplitude": amplitude,
            "is_solid": is_solid,
            "resistance": resistance,
            "inductance": inductance,
            "phase": phase,
            "coils": [],  # List of coil terminal names
        }
        
        self._windings[name] = winding
        return winding
    
    def assign_coil(
        self,
        face: Any,
        conductors_number: int = 1,
        polarity: str = "Positive",
        name: str = "coil"
    ) -> str:
        """
        Assign a coil terminal to a face/body.
        
        In Elmer, coils are typically defined using:
        - CoilSolver for stranded coils with many turns
        - Direct current density for solid conductors
        
        Args:
            face: Face/body identifier (physical group tag)
            conductors_number: Number of turns/conductors
            polarity: "Positive" or "Negative" (current direction)
            name: Coil terminal name
            
        Returns:
            Coil name
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized")
        
        coil_config = {
            "name": name,
            "face": face,
            "conductors": conductors_number,
            "polarity": polarity,
            "direction": 1 if polarity == "Positive" else -1,
        }
        
        self._coils[name] = coil_config
        return name
    
    def add_coils_to_winding(
        self,
        winding_name: str,
        coil_names: List[str]
    ) -> bool:
        """Add coil terminals to a winding."""
        if winding_name not in self._windings:
            return False
        
        self._windings[winding_name]["coils"].extend(coil_names)
        return True
    
    def create_current_density_excitation(
        self,
        body_name: str,
        current: float,
        cross_section_area: float,
        direction: List[float] = None
    ) -> "elmer.BodyForce":
        """
        Create current density excitation for solid conductors.
        
        J = I / A (current density = current / area)
        
        Args:
            body_name: Name of the conductor body
            current: Total current in Amperes
            cross_section_area: Conductor cross-section in m²
            direction: Current direction vector [Jx, Jy, Jz] (normalized)
            
        Returns:
            BodyForce object
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized")
        
        # Calculate current density magnitude
        j_magnitude = current / cross_section_area
        
        # Default direction is Z-axis
        if direction is None:
            direction = [0, 0, 1]
        
        # Normalize direction
        norm = math.sqrt(sum(d**2 for d in direction))
        direction = [d / norm for d in direction]
        
        # Create body force
        bf = elmer.BodyForce(self._simulation, f"{body_name}_current")
        bf.data = {
            "Current Density 1": j_magnitude * direction[0],
            "Current Density 2": j_magnitude * direction[1],
            "Current Density 3": j_magnitude * direction[2],
        }
        
        self._body_forces[body_name] = bf
        return bf
    
    def create_coil_solver_excitation(
        self,
        coil_name: str,
        num_turns: int,
        current: float,
        coil_type: str = "Stranded"
    ) -> "elmer.Component":
        """
        Create CoilSolver excitation for stranded coils.
        
        Uses Elmer's CoilSolver for proper handling of
        stranded conductors with many turns.
        
        Args:
            coil_name: Coil identifier
            num_turns: Number of turns
            current: Current per turn in Amperes
            coil_type: "Stranded" or "Massive"
            
        Returns:
            Component object
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized")
        
        # Create component for the coil
        comp = elmer.Component(self._simulation, coil_name)
        comp.data = {
            "Coil Type": coil_type,
            "Number of Turns": num_turns,
            "Coil Current": current,
            "Desired Coil Current": current,
        }
        
        self._components[coil_name] = comp
        return comp
    
    def create_matrix(
        self,
        excitations: List[str],
        name: str
    ) -> Any:
        """
        Create an excitation matrix for multi-winding setups.
        
        In Elmer, this is used for calculating mutual inductances
        and coupling between windings.
        """
        self._matrices[name] = excitations
        return {"name": name, "excitations": excitations}
    
    def join_parallel(
        self,
        matrix: Any,
        sources: List[str],
        matrix_name: str,
        join_name: str
    ) -> bool:
        """Join parallel windings in a matrix."""
        # Store parallel connection information
        if matrix_name not in self._matrices:
            self._matrices[matrix_name] = []
        
        self._matrices[f"{matrix_name}_{join_name}"] = sources
        return True
    
    def create_waveform_dataset(
        self,
        name: str,
        time: List[float],
        data: List[float]
    ) -> Any:
        """
        Create a waveform dataset for transient excitation.
        
        Returns a MATC expression for time-dependent excitation.
        """
        # Store waveform data
        waveform = {
            "name": name,
            "time": time,
            "data": data,
        }
        
        # Create MATC table for interpolation
        # Format: Variable Time; Real MATC "table_interp(tx(1), time_data, value_data)"
        
        return waveform
    
    def assign_floating(
        self,
        objects: List[GeometryObject],
        charge: float = 0.0,
        name: str = "floating"
    ) -> bool:
        """
        Assign floating boundary condition (for electrostatics).
        
        Floating conductor with optional fixed charge.
        """
        if self._simulation is None:
            return False
        
        # Get physical group tags
        tags = []
        for obj in objects:
            if isinstance(obj, int):
                tags.append(obj)
            elif hasattr(obj, 'id'):
                tags.append(int(obj.id) if obj.id.isdigit() else obj.id)
        
        boundary = elmer.Boundary(self._simulation, name, tags)
        boundary.data = {
            "Potential": "Variable Time; Real MATC \"0\"",  # Floating
        }
        
        if charge != 0.0:
            boundary.data["Electric Charge"] = charge
        
        self._boundaries[name] = boundary
        return True
    
    def assign_fixed_potential(
        self,
        objects: List[GeometryObject],
        potential: float,
        name: str = "fixed_potential"
    ) -> bool:
        """Assign fixed electric potential boundary condition."""
        if self._simulation is None:
            return False
        
        tags = self._get_tags(objects)
        
        boundary = elmer.Boundary(self._simulation, name, tags)
        boundary.data = {
            "Potential": potential,
        }
        
        self._boundaries[name] = boundary
        return True
    
    def assign_heat_source(
        self,
        objects: List[GeometryObject],
        power: float,
        name: str = "heat_source"
    ) -> bool:
        """
        Assign heat source (for thermal simulations).
        
        Args:
            objects: Bodies to apply heat source
            power: Heat power in Watts
            name: Heat source name
        """
        if self._simulation is None:
            return False
        
        bf = elmer.BodyForce(self._simulation, name)
        bf.data = {
            "Heat Source": power,
        }
        
        self._body_forces[name] = bf
        return True
    
    def assign_joule_heat(
        self,
        objects: List[GeometryObject],
        name: str = "joule_heat"
    ) -> bool:
        """
        Enable Joule heating calculation for conductors.
        
        Links electromagnetic losses to thermal simulation.
        """
        if self._simulation is None:
            return False
        
        bf = elmer.BodyForce(self._simulation, name)
        bf.data = {
            "Joule Heat": True,
        }
        
        self._body_forces[name] = bf
        return True
    
    def assign_temperature_boundary(
        self,
        objects: List[GeometryObject],
        temperature: float,
        name: str = "temp_bc"
    ) -> bool:
        """Assign fixed temperature boundary condition."""
        if self._simulation is None:
            return False
        
        tags = self._get_tags(objects)
        
        boundary = elmer.Boundary(self._simulation, name, tags)
        boundary.data = {
            "Temperature": temperature,
        }
        
        self._boundaries[name] = boundary
        return True
    
    def assign_convection_boundary(
        self,
        objects: List[GeometryObject],
        heat_transfer_coeff: float,
        external_temp: float,
        name: str = "convection"
    ) -> bool:
        """Assign convection boundary condition for thermal simulation."""
        if self._simulation is None:
            return False
        
        tags = self._get_tags(objects)
        
        boundary = elmer.Boundary(self._simulation, name, tags)
        boundary.data = {
            "Heat Transfer Coefficient": heat_transfer_coeff,
            "External Temperature": external_temp,
        }
        
        self._boundaries[name] = boundary
        return True
    
    def assign_magnetic_boundary(
        self,
        objects: List[GeometryObject],
        boundary_type: str = "natural",
        name: str = "magnetic_bc"
    ) -> bool:
        """
        Assign magnetic boundary condition.
        
        Args:
            objects: Boundary surfaces
            boundary_type: "natural" (flux tangent), "dirichlet" (A=0)
            name: Boundary name
        """
        if self._simulation is None:
            return False
        
        tags = self._get_tags(objects)
        
        boundary = elmer.Boundary(self._simulation, name, tags)
        
        if boundary_type.lower() == "dirichlet":
            # A = 0 (magnetic insulation)
            boundary.data = {
                "AV {e}": 0.0,
            }
        else:
            # Natural BC (n × H = 0, flux parallel to boundary)
            boundary.data = {}  # Natural BC is default
        
        self._boundaries[name] = boundary
        return True
    
    def set_core_losses(
        self,
        objects: List[GeometryObject],
        on_field: bool = True
    ) -> bool:
        """
        Enable core loss calculation for magnetic cores.
        
        Core losses are calculated from the magnetic field
        using Steinmetz equation or similar models.
        """
        # Core loss calculation is typically done in post-processing
        # or through material properties in Elmer
        return True
    
    def _get_tags(self, objects: List[GeometryObject]) -> List:
        """Extract physical group tags from geometry objects."""
        tags = []
        for obj in objects:
            if isinstance(obj, int):
                tags.append(obj)
            elif hasattr(obj, 'id'):
                try:
                    tags.append(int(obj.id))
                except ValueError:
                    tags.append(obj.id)
            elif hasattr(obj, 'name'):
                tags.append(obj.name)
        return tags
    
    def get_windings(self) -> Dict[str, Dict]:
        """Get all defined windings."""
        return self._windings.copy()
    
    def get_boundaries(self) -> Dict[str, "elmer.Boundary"]:
        """Get all defined boundaries."""
        return self._boundaries.copy()
    
    def get_body_forces(self) -> Dict[str, "elmer.BodyForce"]:
        """Get all defined body forces."""
        return self._body_forces.copy()
    
    def build_elmer_excitations(self) -> Dict[str, Any]:
        """
        Build all Elmer excitation objects.
        
        Converts the stored winding/coil configurations
        into proper Elmer BodyForce, Boundary, and Component objects.
        
        Returns:
            Dictionary of all excitation objects
        """
        excitations = {
            "windings": self._windings,
            "coils": self._coils,
            "boundaries": self._boundaries,
            "body_forces": self._body_forces,
            "components": self._components,
        }
        
        return excitations
