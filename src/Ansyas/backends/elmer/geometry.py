"""
Elmer geometry backend implementation.

Uses CadQuery for 3D geometry creation and gmsh for mesh preparation.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

from ..base import (
    GeometryBackend,
    GeometryObject,
    Axis,
    Plane,
)


class ElmerGeometryBackend(GeometryBackend):
    """
    Geometry backend using CadQuery for CAD and gmsh for mesh preparation.
    
    Workflow:
    1. Create geometry with CadQuery
    2. Export to STEP
    3. Import into gmsh for meshing
    
    This backend provides an open-source alternative to PyAEDT modeler.
    """
    
    # Mapping from abstract types to CadQuery equivalents
    _AXIS_TO_DIRECTION = {
        Axis.X: (1, 0, 0),
        Axis.Y: (0, 1, 0),
        Axis.Z: (0, 0, 1),
    }
    
    _PLANE_TO_CQ = {
        Plane.XY: "XY",
        Plane.YZ: "YZ",
        Plane.ZX: "XZ",
    }
    
    def __init__(self):
        self._objects: Dict[str, "cq.Workplane"] = {}
        self._object_cache: Dict[str, GeometryObject] = {}
        self._gmsh_initialized = False
        self._units = "meter"
        self._scale_factor = 1.0  # For unit conversion
        self._temp_dir = None
        self._step_files: List[str] = []
        self._gmsh_entities: Dict[str, List[int]] = {}  # name -> gmsh tag list
    
    def initialize(self, **kwargs) -> None:
        """Initialize the geometry backend."""
        if not HAS_CADQUERY:
            raise ImportError(
                "CadQuery is required for ElmerGeometryBackend. "
                "Install with: pip install cadquery"
            )
        if not HAS_GMSH:
            raise ImportError(
                "gmsh is required for ElmerGeometryBackend. "
                "Install with: pip install gmsh"
            )
        
        self._temp_dir = kwargs.get("temp_dir", tempfile.mkdtemp(prefix="ansyas_elmer_"))
        
        # Initialize gmsh
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.model.add("ansyas_model")
        self._gmsh_initialized = True
    
    def set_units(self, units: str) -> None:
        """Set the model units (e.g., 'meter', 'mm')."""
        self._units = units
        # CadQuery works in mm by default, so we need to track scale
        unit_scales = {
            "meter": 1000.0,  # CadQuery mm to meter
            "m": 1000.0,
            "mm": 1.0,
            "millimeter": 1.0,
            "cm": 10.0,
            "centimeter": 10.0,
            "in": 25.4,
            "inch": 25.4,
        }
        self._scale_factor = unit_scales.get(units.lower(), 1.0)
    
    def _wrap_object(self, cq_obj: "cq.Workplane", name: str) -> GeometryObject:
        """Wrap a CadQuery object in a GeometryObject."""
        if cq_obj is None:
            return None
        
        obj = GeometryObject(
            id=name,
            name=name,
            native_object=cq_obj,
            volume=None,  # Calculate if needed
        )
        self._object_cache[name] = obj
        self._objects[name] = cq_obj
        return obj
    
    def _get_cq_object(self, obj: Union[GeometryObject, "cq.Workplane", str]) -> "cq.Workplane":
        """Get the CadQuery workplane from various input types."""
        if isinstance(obj, GeometryObject):
            return obj.native_object
        elif isinstance(obj, str):
            return self._objects.get(obj)
        return obj
    
    def _scale(self, value: float) -> float:
        """Scale a value from user units to CadQuery mm."""
        return value * self._scale_factor
    
    def _scale_list(self, values: List[float]) -> List[float]:
        """Scale a list of values from user units to CadQuery mm."""
        return [v * self._scale_factor for v in values]
    
    # Primitive creation
    def create_box(
        self,
        origin: List[float],
        sizes: List[float],
        name: str,
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a box primitive."""
        scaled_origin = self._scale_list(origin)
        scaled_sizes = self._scale_list(sizes)
        
        # CadQuery creates boxes centered, so we need to adjust
        box = (
            cq.Workplane("XY")
            .box(scaled_sizes[0], scaled_sizes[1], scaled_sizes[2], centered=False)
            .translate(scaled_origin)
        )
        
        return self._wrap_object(box, name)
    
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
        scaled_origin = self._scale_list(origin)
        scaled_radius = self._scale(radius)
        scaled_height = self._scale(height)
        
        # Determine workplane based on axis
        plane_map = {
            Axis.X: "YZ",
            Axis.Y: "XZ",
            Axis.Z: "XY",
        }
        
        if num_sides > 0:
            # Create polygon approximation
            cylinder = (
                cq.Workplane(plane_map[axis])
                .polygon(num_sides, scaled_radius * 2)
                .extrude(scaled_height)
                .translate(scaled_origin)
            )
        else:
            # Create true cylinder
            cylinder = (
                cq.Workplane(plane_map[axis])
                .circle(scaled_radius)
                .extrude(scaled_height)
                .translate(scaled_origin)
            )
        
        return self._wrap_object(cylinder, name)
    
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
        scaled_origin = self._scale_list(origin)
        scaled_radius = self._scale(radius)
        
        cq_plane = self._PLANE_TO_CQ[plane]
        
        if num_sides > 0:
            circle = (
                cq.Workplane(cq_plane)
                .polygon(num_sides, scaled_radius * 2)
                .translate(scaled_origin)
            )
        else:
            circle = (
                cq.Workplane(cq_plane)
                .circle(scaled_radius)
                .translate(scaled_origin)
            )
        
        return self._wrap_object(circle, name)
    
    def create_rectangle(
        self,
        plane: Plane,
        origin: List[float],
        sizes: List[float],
        name: str = "rectangle",
        material: Optional[str] = None
    ) -> GeometryObject:
        """Create a 2D rectangle (for sweeping)."""
        scaled_origin = self._scale_list(origin)
        scaled_sizes = self._scale_list(sizes)
        
        cq_plane = self._PLANE_TO_CQ[plane]
        
        rect = (
            cq.Workplane(cq_plane)
            .rect(scaled_sizes[0], scaled_sizes[1], centered=False)
            .translate(scaled_origin)
        )
        
        return self._wrap_object(rect, name)
    
    # Operations
    def move(self, obj: GeometryObject, vector: List[float]) -> bool:
        """Move an object by a vector."""
        try:
            cq_obj = self._get_cq_object(obj)
            scaled_vector = self._scale_list(vector)
            moved = cq_obj.translate(tuple(scaled_vector))
            self._objects[obj.name] = moved
            obj.native_object = moved
            return True
        except Exception:
            return False
    
    def rotate(self, obj: GeometryObject, axis: Axis, angle: float) -> bool:
        """Rotate an object around an axis (angle in degrees)."""
        try:
            cq_obj = self._get_cq_object(obj)
            axis_vec = self._AXIS_TO_DIRECTION[axis]
            rotated = cq_obj.rotate((0, 0, 0), axis_vec, angle)
            self._objects[obj.name] = rotated
            obj.native_object = rotated
            return True
        except Exception:
            return False
    
    def clone(self, obj: GeometryObject) -> Optional[GeometryObject]:
        """Clone an object."""
        try:
            cq_obj = self._get_cq_object(obj)
            # CadQuery objects are immutable, so we can reference directly
            new_name = f"{obj.name}_clone"
            return self._wrap_object(cq_obj, new_name)
        except Exception:
            return None
    
    def subtract(self, obj: GeometryObject, tool: GeometryObject, keep_tool: bool = True) -> bool:
        """Subtract tool from obj."""
        try:
            cq_obj = self._get_cq_object(obj)
            tool_obj = self._get_cq_object(tool)
            
            result = cq_obj.cut(tool_obj)
            self._objects[obj.name] = result
            obj.native_object = result
            
            if not keep_tool:
                del self._objects[tool.name]
                del self._object_cache[tool.name]
            
            return True
        except Exception:
            return False
    
    def unite(self, objects: List[GeometryObject]) -> GeometryObject:
        """Unite multiple objects into one."""
        if not objects:
            return None
        
        result = self._get_cq_object(objects[0])
        for obj in objects[1:]:
            cq_obj = self._get_cq_object(obj)
            result = result.union(cq_obj)
        
        new_name = f"{objects[0].name}_united"
        return self._wrap_object(result, new_name)
    
    def mirror(
        self,
        obj: GeometryObject,
        origin: List[float],
        normal: List[float]
    ) -> GeometryObject:
        """Mirror an object across a plane."""
        cq_obj = self._get_cq_object(obj)
        scaled_origin = self._scale_list(origin)
        
        # CadQuery mirror requires plane definition
        mirrored = cq_obj.mirror(
            mirrorPlane=cq.Plane(
                origin=cq.Vector(*scaled_origin),
                normal=cq.Vector(*normal)
            )
        )
        
        new_name = f"{obj.name}_mirrored"
        return self._wrap_object(mirrored, new_name)
    
    def sweep_along_vector(
        self,
        profile: GeometryObject,
        vector: List[float]
    ) -> GeometryObject:
        """Sweep a 2D profile along a vector."""
        cq_profile = self._get_cq_object(profile)
        scaled_vector = self._scale_list(vector)
        
        swept = cq_profile.extrude(cq.Vector(*scaled_vector))
        new_name = f"{profile.name}_swept"
        return self._wrap_object(swept, new_name)
    
    def sweep_around_axis(
        self,
        profile: GeometryObject,
        axis: Axis,
        angle: float,
        num_segments: int = 12
    ) -> GeometryObject:
        """Sweep a 2D profile around an axis."""
        cq_profile = self._get_cq_object(profile)
        axis_vec = self._AXIS_TO_DIRECTION[axis]
        
        # CadQuery revolve
        swept = cq_profile.revolve(angle, axisStart=(0, 0, 0), axisEnd=axis_vec)
        new_name = f"{profile.name}_revolved"
        return self._wrap_object(swept, new_name)
    
    # Import/Export
    def import_step(self, file_path: str, healing: bool = True) -> List[GeometryObject]:
        """Import geometry from a STEP file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"STEP file not found: {file_path}")
        
        # Import into CadQuery
        result = cq.importers.importStep(file_path)
        
        # Also import into gmsh for meshing
        gmsh.model.occ.importShapes(file_path)
        gmsh.model.occ.synchronize()
        
        # Get the imported entities
        volumes = gmsh.model.getEntities(3)
        
        name = os.path.splitext(os.path.basename(file_path))[0]
        obj = self._wrap_object(result, name)
        
        # Track gmsh entities
        self._gmsh_entities[name] = [v[1] for v in volumes]
        self._step_files.append(file_path)
        
        return [obj]
    
    def export_step(self, objects: List[GeometryObject], file_path: str) -> bool:
        """Export geometry to a STEP file."""
        try:
            if not objects:
                return False
            
            # Combine all objects
            combined = self._get_cq_object(objects[0])
            for obj in objects[1:]:
                combined = combined.union(self._get_cq_object(obj))
            
            # Export
            cq.exporters.export(combined, file_path, exportType="STEP")
            return True
        except Exception:
            return False
    
    def export_all_to_gmsh(self, output_path: Optional[str] = None) -> str:
        """
        Export all CadQuery geometry to gmsh for meshing.
        
        Returns the path to the combined STEP file.
        """
        if output_path is None:
            output_path = os.path.join(self._temp_dir, "geometry.step")
        
        # Combine all objects
        all_objects = list(self._object_cache.values())
        if all_objects:
            self.export_step(all_objects, output_path)
            
            # Import into gmsh
            gmsh.model.occ.importShapes(output_path)
            gmsh.model.occ.synchronize()
        
        return output_path
    
    # Object queries
    def get_objects_by_name(self, pattern: str) -> List[GeometryObject]:
        """Get objects matching a name pattern."""
        import fnmatch
        return [
            obj for name, obj in self._object_cache.items()
            if fnmatch.fnmatch(name, pattern)
        ]
    
    def get_object_volume(self, obj: GeometryObject) -> float:
        """Get the volume of an object."""
        cq_obj = self._get_cq_object(obj)
        try:
            # CadQuery volume calculation
            solid = cq_obj.val()
            if hasattr(solid, "Volume"):
                # Convert from mm³ to user units
                return solid.Volume() / (self._scale_factor ** 3)
        except Exception:
            pass
        return 0.0
    
    def get_object_faces(self, obj: GeometryObject) -> List[Any]:
        """Get the faces of an object."""
        cq_obj = self._get_cq_object(obj)
        try:
            return cq_obj.faces().vals()
        except Exception:
            return []
    
    def section(
        self,
        obj: GeometryObject,
        plane: Plane,
        create_new: bool = True
    ) -> Optional[GeometryObject]:
        """Create a cross-section of an object."""
        cq_obj = self._get_cq_object(obj)
        cq_plane = self._PLANE_TO_CQ[plane]
        
        try:
            section = cq_obj.section(cq.Plane.named(cq_plane))
            if create_new:
                new_name = f"{obj.name}_section"
                return self._wrap_object(section, new_name)
            return None
        except Exception:
            return None
    
    def create_region(
        self,
        padding_percent: List[float],
        is_percentage: bool = True
    ) -> GeometryObject:
        """Create an air/boundary region around the model."""
        # Calculate bounding box of all objects
        min_pt = [float('inf')] * 3
        max_pt = [float('-inf')] * 3
        
        for cq_obj in self._objects.values():
            try:
                bb = cq_obj.val().BoundingBox()
                min_pt = [min(min_pt[0], bb.xmin), min(min_pt[1], bb.ymin), min(min_pt[2], bb.zmin)]
                max_pt = [max(max_pt[0], bb.xmax), max(max_pt[1], bb.ymax), max(max_pt[2], bb.zmax)]
            except Exception:
                continue
        
        if min_pt[0] == float('inf'):
            # No objects, create default region
            min_pt = [-1, -1, -1]
            max_pt = [1, 1, 1]
        
        # Calculate padding
        sizes = [max_pt[i] - min_pt[i] for i in range(3)]
        if is_percentage:
            padding = [sizes[i] * padding_percent[i] / 100 for i in range(3)]
        else:
            padding = self._scale_list(padding_percent)
        
        # Create region box
        region_min = [min_pt[i] - padding[i] for i in range(3)]
        region_sizes = [sizes[i] + 2 * padding[i] for i in range(3)]
        
        region = (
            cq.Workplane("XY")
            .box(region_sizes[0], region_sizes[1], region_sizes[2], centered=False)
            .translate(region_min)
        )
        
        # Subtract all existing objects to create hollow region
        for cq_obj in self._objects.values():
            try:
                region = region.cut(cq_obj)
            except Exception:
                continue
        
        return self._wrap_object(region, "Region")
    
    def create_air_region(self, padding: Dict[str, float]) -> GeometryObject:
        """Create an air region with specified padding percentages."""
        # For compatibility with Ansys backend API
        padding_list = [
            padding.get("x_pos", 50),
            padding.get("y_pos", 50),
            padding.get("z_pos", 50),
        ]
        return self.create_region(padding_list, is_percentage=True)
    
    def fit_all(self) -> None:
        """Fit the view to show all objects (no-op for non-GUI backend)."""
        pass  # No GUI in Elmer backend
    
    def finalize(self) -> None:
        """Finalize geometry and prepare for meshing."""
        if self._gmsh_initialized:
            gmsh.model.occ.synchronize()
    
    def cleanup(self) -> None:
        """Clean up temporary files and gmsh."""
        if self._gmsh_initialized:
            gmsh.finalize()
            self._gmsh_initialized = False
        
        # Clean up temp files
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
