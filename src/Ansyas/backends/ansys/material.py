"""
Ansys material backend implementation.

Uses PyAEDT materials for material definition and assignment.
"""

from typing import Any, Dict, List, Optional, Union
import math

from ..base import (
    MaterialBackend,
    MaterialDefinition,
    GeometryObject,
)


class AnsysMaterialBackend(MaterialBackend):
    """
    Material backend using Ansys AEDT materials.
    
    This implementation wraps PyAEDT's material functionality.
    """
    
    def __init__(self):
        self._project = None
        self._materials = None
        self._datasets = {}
    
    def initialize(self, project=None, **kwargs) -> None:
        """
        Initialize with an existing PyAEDT project.
        
        Args:
            project: A PyAEDT Maxwell3d, Icepak, or similar project instance.
        """
        if project is None:
            raise ValueError("AnsysMaterialBackend requires a PyAEDT project instance")
        self._project = project
        self._materials = project.materials
    
    def add_material(self, definition: MaterialDefinition) -> str:
        """Add a new material and return its ID/name."""
        aedt_material = self._materials.add_material(definition.name)
        
        # Set permeability
        if isinstance(definition.permeability, dict):
            # Complex permeability with frequency dependence
            if "real_dataset" in definition.permeability:
                aedt_material.permeability = definition.permeability["formula"]
            else:
                aedt_material.permeability = definition.permeability.get("value", 1.0)
        else:
            aedt_material.permeability = definition.permeability
        
        # Set conductivity
        if isinstance(definition.conductivity, dict):
            if "dataset" in definition.conductivity:
                aedt_material.conductivity.add_thermal_modifier_dataset(
                    definition.conductivity["dataset"]
                )
            else:
                aedt_material.conductivity = definition.conductivity.get("value", 0.0)
        else:
            aedt_material.conductivity = definition.conductivity
        
        # Set permittivity
        aedt_material.permittivity = definition.permittivity
        
        # Set density
        aedt_material.mass_density = definition.density
        
        # Set magnetic loss tangent if provided
        if definition.loss_tangent is not None:
            if isinstance(definition.loss_tangent, dict):
                aedt_material.magnetic_loss_tangent = definition.loss_tangent.get("formula", 0)
            else:
                aedt_material.magnetic_loss_tangent = definition.loss_tangent
        
        # Set Steinmetz coefficients if provided
        if definition.steinmetz_coefficients is not None:
            aedt_material.set_power_ferrite_coreloss(
                cm=definition.steinmetz_coefficients["k"],
                x=definition.steinmetz_coefficients["alpha"],
                y=definition.steinmetz_coefficients["beta"]
            )
        
        # Handle litz wire
        if definition.is_litz:
            aedt_material.stacking_type = "Litz Wire"
            aedt_material.wire_type = "Round"
            if definition.strand_count:
                aedt_material.strand_number = definition.strand_count
            if definition.strand_diameter:
                aedt_material.wire_diameter = f"{definition.strand_diameter}meter"
        
        return definition.name
    
    def add_litz_wire_material(
        self,
        name: str,
        strand_count: int,
        strand_diameter: float,
        base_material: str = "copper"
    ) -> str:
        """
        Add a litz wire material by duplicating a base material.
        
        Args:
            name: Name for the new material
            strand_count: Number of strands
            strand_diameter: Diameter of each strand in meters
            base_material: Base material to duplicate (default: copper)
        
        Returns:
            Name of the created material
        """
        wire_material = self._materials.duplicate_material(
            material=base_material,
            name=name
        )
        wire_material.stacking_type = "Litz Wire"
        wire_material.wire_type = "Round"
        wire_material.strand_number = strand_count
        wire_material.wire_diameter = f"{strand_diameter}meter"
        return name
    
    def assign_material(self, obj: GeometryObject, material_name: str) -> bool:
        """Assign a material to a geometry object."""
        native_obj = obj.native_object if isinstance(obj, GeometryObject) else obj
        try:
            self._project.assign_material(
                assignment=native_obj,
                material=material_name
            )
            return True
        except Exception:
            return False
    
    def create_dataset(
        self,
        name: str,
        x_values: List[float],
        y_values: List[float],
        x_unit: str = "",
        y_unit: str = ""
    ) -> Any:
        """Create a dataset for frequency/temperature dependent properties."""
        dataset = self._project.create_dataset(
            name=name,
            x=x_values,
            y=y_values,
            x_unit=x_unit if x_unit else None,
            y_unit=y_unit if y_unit else None,
        )
        self._datasets[name] = dataset
        return dataset
    
    def get_material_property(self, material_name: str, property_name: str) -> Any:
        """Get a material property value."""
        material = self._materials[material_name]
        return getattr(material, property_name, None)
    
    def assign_surface_material(
        self,
        objects: List[GeometryObject],
        material_name: str
    ) -> bool:
        """Assign a surface material (for thermal simulations)."""
        native_objs = [
            obj.native_object.name if isinstance(obj, GeometryObject) else obj.name
            for obj in objects
        ]
        try:
            self._project.assign_surface_material(
                obj=native_objs,
                mat=material_name
            )
            return True
        except Exception:
            return False
