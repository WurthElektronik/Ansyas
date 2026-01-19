"""
Ansys geometry backend implementation.

Uses PyAEDT modeler for 3D geometry creation.
"""

from typing import Any, Dict, List, Optional, Union
from ansys.aedt.core import constants as pyaedt_constants

from ..base import (
    GeometryBackend,
    GeometryObject,
    Axis,
    Plane,
)


class AnsysGeometryBackend(GeometryBackend):
    """
    Geometry backend using Ansys AEDT modeler.
    
    This implementation wraps PyAEDT's modeler functionality to create
    and manipulate 3D geometry.
    """
    
    # Mapping from abstract types to PyAEDT constants
    # Use the new Axis/Plane classes instead of deprecated AXIS/PLANE
    _AXIS_MAP = {
        Axis.X: pyaedt_constants.Axis.X,
        Axis.Y: pyaedt_constants.Axis.Y,
        Axis.Z: pyaedt_constants.Axis.Z,
    }
    
    _PLANE_MAP = {
        Plane.XY: pyaedt_constants.Plane.XY,
        Plane.YZ: pyaedt_constants.Plane.YZ,
        Plane.ZX: pyaedt_constants.Plane.ZX,
    }
    
    def __init__(self):
        self._project = None
        self._modeler = None
        self._units = "meter"
        self._object_cache: Dict[str, GeometryObject] = {}
    
    def initialize(self, project=None, **kwargs) -> None:
        """
        Initialize with an existing PyAEDT project.
        
        Args:
            project: A PyAEDT Maxwell3d, Icepak, or similar project instance.
        """
        if project is None:
            raise ValueError("AnsysGeometryBackend requires a PyAEDT project instance")
        self._project = project
        self._modeler = project.modeler
    
    def set_units(self, units: str) -> None:
        """Set the model units."""
        self._units = units
        self._modeler.model_units = units
    
    def _wrap_object(self, native_obj, name: Optional[str] = None) -> GeometryObject:
        """Wrap a PyAEDT object in a GeometryObject."""
        if native_obj is None:
            return None
        
        obj_name = name or getattr(native_obj, 'name', str(native_obj))
        obj = GeometryObject(
            id=obj_name,
            name=obj_name,
            native_object=native_obj,
            volume=getattr(native_obj, 'volume', None),
        )
        self._object_cache[obj_name] = obj
        return obj
    
    def _get_native(self, obj: Union[GeometryObject, Any]) -> Any:
        """Get the native PyAEDT object from a GeometryObject or return as-is."""
        if isinstance(obj, GeometryObject):
            return obj.native_object
        return obj
    
    def create_box(
        self,
        origin: List[float],
        sizes: List[float],
        name: str,
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a box primitive."""
        native_obj = self._modeler.create_box(
            origin=origin,
            sizes=sizes,
            name=name,
            material=material
        )
        return self._wrap_object(native_obj, name)
    
    def create_cylinder(
        self,
        axis: Axis,
        origin: List[float],
        radius: float,
        height: float,
        num_sides: int = 0,
        name: str = "cylinder",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a cylinder primitive."""
        native_obj = self._modeler.create_cylinder(
            orientation=self._AXIS_MAP[axis],
            origin=origin,
            radius=radius,
            height=height,
            num_sides=num_sides,
            name=name,
            material=material
        )
        return self._wrap_object(native_obj, name)
    
    def create_circle(
        self,
        plane: Plane,
        origin: List[float],
        radius: float,
        num_sides: int = 12,
        name: str = "circle",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a 2D circle (for sweeping)."""
        native_obj = self._modeler.create_circle(
            orientation=self._PLANE_MAP[plane],
            origin=origin,
            radius=radius,
            num_sides=num_sides,
            is_covered=True,
            name=name,
            material=material,
            non_model=False
        )
        return self._wrap_object(native_obj, name)
    
    def create_rectangle(
        self,
        plane: Plane,
        origin: List[float],
        sizes: List[float],
        name: str = "rectangle",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a 2D rectangle (for sweeping)."""
        native_obj = self._modeler.create_rectangle(
            orientation=self._PLANE_MAP[plane],
            origin=origin,
            sizes=sizes,
            is_covered=True,
            name=name,
            material=material,
            non_model=False
        )
        return self._wrap_object(native_obj, name)
    
    def move(self, obj: GeometryObject, vector: List[float]) -> bool:
        """Move an object by a vector."""
        native_obj = self._get_native(obj)
        return native_obj.move(vector)
    
    def rotate(self, obj: GeometryObject, axis: Axis, angle: float) -> bool:
        """Rotate an object around an axis."""
        native_obj = self._get_native(obj)
        return native_obj.rotate(axis=self._AXIS_MAP[axis], angle=angle)
    
    def clone(self, obj: GeometryObject) -> Optional[GeometryObject]:
        """Clone an object."""
        native_obj = self._get_native(obj)
        result, objects = self._modeler.clone(native_obj)
        if result and objects:
            new_native = self._modeler.get_object_from_name(objects[0])
            return self._wrap_object(new_native)
        return None
    
    def subtract(self, obj: GeometryObject, tool: GeometryObject, keep_tool: bool = True) -> bool:
        """Subtract tool from obj."""
        native_obj = self._get_native(obj)
        native_tool = self._get_native(tool)
        return native_obj.subtract(native_tool, keep_tool)
    
    def unite(self, objects: List[GeometryObject]) -> GeometryObject:
        """Unite multiple objects into one."""
        native_objs = [self._get_native(obj) for obj in objects]
        result_name = self._modeler.unite(native_objs)
        result_obj = self._modeler.get_object_from_name(result_name)
        return self._wrap_object(result_obj, result_name)
    
    def mirror(
        self,
        obj: GeometryObject,
        origin: List[float],
        normal: List[float]
    ) -> GeometryObject:
        """Mirror an object across a plane."""
        native_obj = self._get_native(obj)
        result = self._modeler.duplicate_and_mirror(
            assignment=native_obj,
            origin=origin,
            vector=normal
        )
        if result:
            new_native = self._modeler.get_object_from_name(result[0])
            return self._wrap_object(new_native)
        return None
    
    def sweep_along_vector(
        self,
        profile: GeometryObject,
        vector: List[float]
    ) -> GeometryObject:
        """Sweep a 2D profile along a vector."""
        native_profile = self._get_native(profile)
        native_obj = self._modeler.sweep_along_vector(
            assignment=native_profile,
            sweep_vector=vector
        )
        # Update the wrapped object
        profile.native_object = native_obj
        return profile
    
    def sweep_around_axis(
        self,
        profile: GeometryObject,
        axis: Axis,
        angle: float,
        num_segments: int = 12
    ) -> GeometryObject:
        """Sweep a 2D profile around an axis."""
        native_profile = self._get_native(profile)
        native_obj = self._modeler.sweep_around_axis(
            assignment=native_profile,
            axis=self._AXIS_MAP[axis],
            sweep_angle=angle,
            draft_angle=0,
            number_of_segments=num_segments
        )
        profile.native_object = native_obj
        return profile
    
    def import_step(self, file_path: str, healing: bool = True) -> List[GeometryObject]:
        """Import geometry from a STEP file."""
        import os
        self._modeler.import_3d_cad(
            input_file=file_path.replace("/", os.sep),
            healing=healing
        )
        # Return all imported objects
        # PyAEDT doesn't directly return imported object names, so we search
        return []  # TODO: Implement proper tracking of imported objects
    
    def export_step(self, objects: List[GeometryObject], file_path: str) -> bool:
        """Export geometry to a STEP file."""
        native_objs = [self._get_native(obj).name for obj in objects]
        return self._modeler.export_3d_model(
            file_name=file_path,
            object_list=native_objs
        )
    
    def get_objects_by_name(self, pattern: str) -> List[GeometryObject]:
        """Get objects matching a name pattern."""
        names = self._modeler.get_objects_w_string(pattern)
        return [
            self._wrap_object(self._modeler.get_object_from_name(name), name)
            for name in names
        ]
    
    def get_object_volume(self, obj: GeometryObject) -> float:
        """Get the volume of an object."""
        native_obj = self._get_native(obj)
        return native_obj.volume
    
    def get_object_faces(self, obj: GeometryObject) -> List[Any]:
        """Get the faces of an object."""
        native_obj = self._get_native(obj)
        return native_obj.faces
    
    def section(
        self,
        obj: GeometryObject,
        plane: Plane,
        create_new: bool = True
    ) -> Optional[GeometryObject]:
        """Create a cross-section of an object."""
        native_obj = self._get_native(obj)
        native_obj.section(
            plane=self._PLANE_MAP[plane],
            create_new=create_new
        )
        # Find the section object
        section_names = self._modeler.get_objects_w_string(f"{obj.name}_Section")
        if section_names:
            section_obj = self._modeler.get_object_from_name(section_names[0])
            return self._wrap_object(section_obj, section_names[0])
        return None
    
    def create_region(
        self,
        padding_percent: List[float],
        is_percentage: bool = True
    ) -> GeometryObject:
        """Create an air/boundary region around the model."""
        region = self._modeler.create_region(
            pad_percent=padding_percent,
            is_percentage=is_percentage
        )
        if region is None or region is False:
            # Try to get existing region
            region_names = self._modeler.get_objects_w_string("Region")
            if region_names:
                region = self._modeler.get_object_from_name(region_names[0])
        return self._wrap_object(region, "Region")
    
    def create_air_region(
        self,
        padding: Dict[str, float]
    ) -> GeometryObject:
        """Create an air region with specified padding percentages."""
        region = self._modeler.create_air_region(
            x_pos=padding.get("x_pos", 50),
            y_pos=padding.get("y_pos", 50),
            z_pos=padding.get("z_pos", 50),
            x_neg=padding.get("x_neg", 50),
            y_neg=padding.get("y_neg", 50),
            z_neg=padding.get("z_neg", 50)
        )
        if region is None or region is False:
            region_names = self._modeler.get_objects_w_string("Region")
            if region_names:
                region = self._modeler.get_object_from_name(region_names[0])
        return self._wrap_object(region, "Region")
    
    def fit_all(self) -> None:
        """Fit the view to show all objects."""
        self._modeler.fit_all()
    
    def separate_bodies(self, obj: GeometryObject) -> List[GeometryObject]:
        """Separate disconnected bodies in an object."""
        native_obj = self._get_native(obj)
        result = self._modeler.separate_bodies(native_obj)
        return [self._wrap_object(r) for r in result]
    
    def delete(self, obj: GeometryObject) -> bool:
        """Delete an object."""
        native_obj = self._get_native(obj)
        return self._modeler.delete(native_obj)
    
    def get_closest_face(self, obj: GeometryObject, position: List[float]) -> Any:
        """Get the face closest to a position."""
        native_obj = self._get_native(obj)
        # This is a utility function - need to implement face distance calculation
        # For now, return faces and let caller handle
        return native_obj.faces
    
    def set_color(self, obj: GeometryObject, color: tuple) -> None:
        """Set the color of an object."""
        native_obj = self._get_native(obj)
        native_obj.color = color
    
    def set_name(self, obj: GeometryObject, name: str) -> None:
        """Set the name of an object."""
        native_obj = self._get_native(obj)
        native_obj.name = name
        obj.name = name
        obj.id = name
