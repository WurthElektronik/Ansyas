"""
Elmer material backend implementation.

Uses pyelmer for material definitions in Elmer FEM simulations.
"""

import math
from typing import Any, Dict, List, Optional, Union

try:
    from pyelmer import elmer
    HAS_PYELMER = True
except ImportError:
    HAS_PYELMER = False

from ..base import (
    MaterialBackend,
    MaterialDefinition,
    GeometryObject,
)


class ElmerMaterialBackend(MaterialBackend):
    """
    Material backend for Elmer using pyelmer.
    
    Supports:
    - Basic electromagnetic materials (permeability, conductivity)
    - Thermal materials (conductivity, heat capacity)
    - Complex permeability for core losses
    - Frequency-dependent properties via datasets
    - Litz wire homogenization
    """
    
    # Standard material library
    STANDARD_MATERIALS = {
        "air": {
            "Relative Permeability": 1.0,
            "Relative Permittivity": 1.0,
            "Electric Conductivity": 0.0,
            "Density": 1.205,
            "Heat Capacity": 1005.0,
            "Heat Conductivity": 0.0257,
        },
        "copper": {
            "Relative Permeability": 0.999994,
            "Relative Permittivity": 1.0,
            "Electric Conductivity": 5.96e7,
            "Density": 8960.0,
            "Heat Capacity": 385.0,
            "Heat Conductivity": 401.0,
        },
        "aluminum": {
            "Relative Permeability": 1.000022,
            "Relative Permittivity": 1.0,
            "Electric Conductivity": 3.77e7,
            "Density": 2700.0,
            "Heat Capacity": 897.0,
            "Heat Conductivity": 237.0,
        },
        "ferrite": {
            "Relative Permeability": 2000.0,
            "Relative Permittivity": 12.0,
            "Electric Conductivity": 0.01,
            "Density": 4800.0,
            "Heat Capacity": 750.0,
            "Heat Conductivity": 4.0,
        },
        "iron": {
            "Relative Permeability": 4000.0,
            "Relative Permittivity": 1.0,
            "Electric Conductivity": 1.03e7,
            "Density": 7874.0,
            "Heat Capacity": 449.0,
            "Heat Conductivity": 80.2,
        },
    }
    
    def __init__(self):
        self._simulation: Optional["elmer.Simulation"] = None
        self._materials: Dict[str, "elmer.Material"] = {}
        self._datasets: Dict[str, Dict] = {}
        self._body_material_map: Dict[str, str] = {}  # body_name -> material_name
    
    def initialize(self, simulation: "elmer.Simulation" = None, **kwargs) -> None:
        """Initialize the material backend."""
        if not HAS_PYELMER:
            raise ImportError(
                "pyelmer is required for ElmerMaterialBackend. "
                "Install with: pip install pyelmer"
            )
        
        self._simulation = simulation
    
    def set_simulation(self, simulation: "elmer.Simulation") -> None:
        """Set the pyelmer simulation object."""
        self._simulation = simulation
    
    def add_material(self, definition: MaterialDefinition) -> str:
        """
        Add a new material and return its name.
        
        Args:
            definition: MaterialDefinition with physical properties
            
        Returns:
            Material name/ID
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        mat = elmer.Material(self._simulation, definition.name)
        
        # Basic electromagnetic properties
        mat.data = {
            "Density": definition.density,
            "Heat Capacity": definition.specific_heat,
            "Heat Conductivity": definition.thermal_conductivity,
        }
        
        # Handle permeability (can be scalar, complex, or frequency-dependent)
        if isinstance(definition.permeability, dict):
            # Frequency-dependent permeability
            if "real" in definition.permeability and "imaginary" in definition.permeability:
                mat.data["Relative Permeability"] = definition.permeability["real"]
                mat.data["Relative Permeability im"] = definition.permeability["imaginary"]
            else:
                mat.data["Relative Permeability"] = definition.permeability.get("value", 1.0)
        else:
            mat.data["Relative Permeability"] = definition.permeability
        
        # Handle conductivity
        if isinstance(definition.conductivity, dict):
            mat.data["Electric Conductivity"] = definition.conductivity.get("value", 0.0)
        else:
            mat.data["Electric Conductivity"] = definition.conductivity
        
        # Permittivity
        mat.data["Relative Permittivity"] = definition.permittivity
        
        # Handle loss tangent for core losses
        if definition.loss_tangent is not None:
            if isinstance(definition.loss_tangent, dict):
                mat.data["Relative Permeability im"] = definition.loss_tangent.get("value", 0.0)
            else:
                # Convert loss tangent to imaginary permeability
                # μ'' = μ' * tan(δ)
                mu_real = mat.data.get("Relative Permeability", 1.0)
                mat.data["Relative Permeability im"] = mu_real * definition.loss_tangent
        
        # Handle Steinmetz coefficients for core loss calculation
        if definition.steinmetz_coefficients:
            # Store for post-processing core loss calculation
            self._datasets[f"{definition.name}_steinmetz"] = definition.steinmetz_coefficients
        
        # Handle litz wire
        if definition.is_litz and definition.strand_count and definition.strand_diameter:
            # Homogenized conductivity for litz wire
            # Effective conductivity considering proximity and skin effects
            self._apply_litz_homogenization(mat, definition)
        
        self._materials[definition.name] = mat
        return definition.name
    
    def _apply_litz_homogenization(
        self, 
        mat: "elmer.Material", 
        definition: MaterialDefinition
    ) -> None:
        """
        Apply litz wire homogenization model.
        
        For litz wire, we need to account for:
        - Reduced effective conductivity due to strand packing
        - Proximity effect losses
        """
        # Packing factor (typical for round strands)
        packing_factor = 0.9  # Approximate
        
        # Strand cross-sectional area
        strand_area = math.pi * (definition.strand_diameter / 2) ** 2
        total_copper_area = strand_area * definition.strand_count
        
        # Effective conductivity (reduced by packing)
        base_conductivity = mat.data.get("Electric Conductivity", 5.96e7)
        mat.data["Electric Conductivity"] = base_conductivity * packing_factor
        
        # Store litz parameters for proximity loss calculation
        self._datasets[f"{definition.name}_litz"] = {
            "strand_count": definition.strand_count,
            "strand_diameter": definition.strand_diameter,
            "packing_factor": packing_factor,
        }
    
    def add_standard_material(self, name: str) -> str:
        """Add a material from the standard library."""
        if name.lower() not in self.STANDARD_MATERIALS:
            raise ValueError(
                f"Unknown standard material: {name}. "
                f"Available: {list(self.STANDARD_MATERIALS.keys())}"
            )
        
        props = self.STANDARD_MATERIALS[name.lower()]
        
        definition = MaterialDefinition(
            name=name,
            permeability=props.get("Relative Permeability", 1.0),
            conductivity=props.get("Electric Conductivity", 0.0),
            permittivity=props.get("Relative Permittivity", 1.0),
            density=props.get("Density", 1000.0),
            thermal_conductivity=props.get("Heat Conductivity", 1.0),
            specific_heat=props.get("Heat Capacity", 1000.0),
        )
        
        return self.add_material(definition)
    
    def assign_material(self, obj: GeometryObject, material_name: str) -> bool:
        """
        Assign a material to a geometry object.
        
        Note: In Elmer, materials are assigned to Bodies, not directly
        to geometry objects. This stores the mapping for later use.
        """
        if material_name not in self._materials:
            return False
        
        self._body_material_map[obj.name] = material_name
        return True
    
    def get_material_for_body(self, body_name: str) -> Optional["elmer.Material"]:
        """Get the material assigned to a body."""
        material_name = self._body_material_map.get(body_name)
        if material_name:
            return self._materials.get(material_name)
        return None
    
    def create_dataset(
        self,
        name: str,
        x_values: List[float],
        y_values: List[float],
        x_unit: str = "",
        y_unit: str = ""
    ) -> Any:
        """
        Create a dataset for frequency/temperature dependent properties.
        
        In Elmer, this is handled through MATC expressions or tables.
        """
        dataset = {
            "x": x_values,
            "y": y_values,
            "x_unit": x_unit,
            "y_unit": y_unit,
        }
        self._datasets[name] = dataset
        
        # Create MATC interpolation expression
        # This would be used in the material definition
        return dataset
    
    def create_frequency_dependent_permeability(
        self,
        name: str,
        frequencies: List[float],
        mu_real: List[float],
        mu_imag: List[float]
    ) -> str:
        """
        Create frequency-dependent permeability material.
        
        Uses Elmer's capability for complex, frequency-dependent materials.
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized")
        
        mat = elmer.Material(self._simulation, name)
        
        # For frequency-dependent properties, Elmer uses MATC or tables
        # Here we create a simplified version using the first frequency point
        # Full implementation would use Variable Frequency with MATC
        
        mat.data = {
            "Relative Permeability": mu_real[0],
            "Relative Permeability im": mu_imag[0],
        }
        
        # Store full frequency data for reference
        self._datasets[f"{name}_permeability_freq"] = {
            "frequencies": frequencies,
            "mu_real": mu_real,
            "mu_imag": mu_imag,
        }
        
        self._materials[name] = mat
        return name
    
    def create_bh_curve_material(
        self,
        name: str,
        h_values: List[float],
        b_values: List[float]
    ) -> str:
        """
        Create a nonlinear magnetic material with B-H curve.
        
        Elmer supports nonlinear materials through cubic spline interpolation.
        """
        if self._simulation is None:
            raise RuntimeError("Simulation not initialized")
        
        mat = elmer.Material(self._simulation, name)
        
        # Elmer uses H-B curve (inverse of B-H)
        # Format: "H-B Curve" = Variable
        # We need to create a MATC expression or table file
        
        mat.data = {
            "H-B Curve": self._create_hb_table(h_values, b_values),
        }
        
        self._materials[name] = mat
        return name
    
    def _create_hb_table(self, h_values: List[float], b_values: List[float]) -> str:
        """Create H-B curve table for Elmer."""
        # Format for Elmer cubic spline
        # Variable h
        # Real MATC "..."
        
        # For simplicity, return the data as a structured string
        # Full implementation would write to a table file
        table_data = list(zip(h_values, b_values))
        return f"Cubic Monotone {len(table_data)}\n" + "\n".join(
            f"{h} {b}" for h, b in table_data
        )
    
    def get_material_property(self, material_name: str, property_name: str) -> Any:
        """Get a material property value."""
        if material_name not in self._materials:
            return None
        
        mat = self._materials[material_name]
        return mat.data.get(property_name)
    
    def get_all_materials(self) -> Dict[str, "elmer.Material"]:
        """Get all defined materials."""
        return self._materials.copy()
    
    def get_body_material_mapping(self) -> Dict[str, str]:
        """Get the body-to-material mapping."""
        return self._body_material_map.copy()
