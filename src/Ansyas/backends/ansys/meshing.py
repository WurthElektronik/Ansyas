"""
Ansys meshing backend implementation.

Uses PyAEDT mesh for mesh generation and refinement.
"""

from typing import Any, Dict, List, Optional

from ..base import (
    MeshingBackend,
    MeshSettings,
    GeometryObject,
)


class AnsysMeshingBackend(MeshingBackend):
    """
    Meshing backend using Ansys AEDT mesh.
    
    This implementation wraps PyAEDT's mesh functionality.
    """
    
    def __init__(self):
        self._project = None
        self._mesh = None
        self._settings: Optional[MeshSettings] = None
    
    def initialize(self, project=None, **kwargs) -> None:
        """
        Initialize with an existing PyAEDT project.
        
        Args:
            project: A PyAEDT Maxwell3d, Icepak, or similar project instance.
        """
        if project is None:
            raise ValueError("AnsysMeshingBackend requires a PyAEDT project instance")
        self._project = project
        self._mesh = project.mesh if hasattr(project, 'mesh') else None
    
    def set_global_settings(self, settings: MeshSettings) -> None:
        """Set global mesh settings."""
        self._settings = settings
        
        if self._mesh is not None:
            # Apply initial mesh slider (1-5 scale in PyAEDT)
            # Map settings to slider value
            slider_value = 3  # Default medium
            if settings.max_element_size:
                # Smaller max size = finer mesh = higher slider
                if settings.max_element_size < 0.001:
                    slider_value = 5
                elif settings.max_element_size < 0.005:
                    slider_value = 4
                elif settings.max_element_size < 0.01:
                    slider_value = 3
                else:
                    slider_value = 2
            
            self._mesh.assign_initial_mesh_from_slider(
                slider_value,
                curvilinear=settings.curvature_refinement
            )
    
    def assign_mesh_size(
        self,
        objects: List[GeometryObject],
        max_size: float,
        min_size: Optional[float] = None
    ) -> bool:
        """Assign mesh size to specific objects."""
        if self._mesh is None:
            return False
        
        native_objs = [
            obj.native_object if isinstance(obj, GeometryObject) else obj
            for obj in objects
        ]
        
        try:
            self._mesh.assign_length_mesh(
                assignment=native_objs,
                maximum_length=max_size
            )
            return True
        except Exception:
            return False
    
    def assign_skin_depth(
        self,
        faces: List[Any],
        skin_depth: float,
        num_layers: int = 2
    ) -> bool:
        """Assign skin depth mesh refinement for eddy current simulations."""
        if self._mesh is None:
            return False
        
        try:
            self._mesh.assign_skin_depth(
                faces,
                skin_depth=f"{skin_depth * 1000}mm",  # Convert to mm
                maximum_elements=None,
                layers_number=num_layers
            )
            return True
        except Exception:
            return False
    
    def assign_curvature_refinement(
        self,
        objects: List[GeometryObject],
        num_elements_per_curvature: int = 6
    ) -> bool:
        """Assign curvature-based mesh refinement."""
        if self._mesh is None:
            return False
        
        native_objs = [
            obj.native_object if isinstance(obj, GeometryObject) else obj
            for obj in objects
        ]
        
        try:
            self._mesh.assign_curvature_extraction(
                assignment=native_objs,
                num_per_curvature=num_elements_per_curvature
            )
            return True
        except Exception:
            return False
    
    def generate_mesh(self) -> bool:
        """Generate the mesh."""
        # In Ansys, mesh is generated during solve
        # This is a placeholder for explicit mesh generation if needed
        return True
    
    def export_mesh(self, file_path: str, format: str = "msh") -> bool:
        """Export mesh to file (for use with external solvers)."""
        # PyAEDT doesn't have direct mesh export capability
        # This would need to be implemented via scripting or other means
        raise NotImplementedError(
            "Ansys mesh export is not directly supported. "
            "Consider using STEP export and external meshing."
        )
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics (element count, quality, etc.)."""
        if self._mesh is None:
            return {}
        
        # PyAEDT doesn't expose mesh statistics directly before solve
        # After solve, statistics are in the solution data
        return {
            "available_after_solve": True,
            "mesh_generated": self._mesh is not None
        }
    
    def assign_initial_mesh_from_slider(
        self,
        level: int,
        curvilinear: bool = True
    ) -> bool:
        """
        Set initial mesh quality using slider (1-5 scale).
        
        Args:
            level: Mesh quality level (1=coarse, 5=fine)
            curvilinear: Whether to use curvilinear elements
        """
        if self._mesh is None:
            return False
        
        try:
            self._mesh.assign_initial_mesh_from_slider(level, curvilinear=curvilinear)
            return True
        except Exception:
            return False
    
    def set_global_mesh_settings_icepak(self, meshtype: int = 1) -> bool:
        """
        Set global mesh settings for Icepak thermal simulations.
        
        Args:
            meshtype: Mesh type (1=auto, 2=manual)
        """
        try:
            self._project.globalMeshSettings(meshtype=meshtype)
            return True
        except Exception:
            return False
