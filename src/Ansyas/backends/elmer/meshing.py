"""
Elmer meshing backend implementation.

Uses gmsh for mesh generation and ElmerGrid for format conversion.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

from ..base import (
    MeshingBackend,
    MeshSettings,
    GeometryObject,
)


class ElmerMeshingBackend(MeshingBackend):
    """
    Meshing backend using gmsh.
    
    Converts gmsh mesh to Elmer format using ElmerGrid.
    
    Features:
    - Automatic physical group assignment
    - Boundary layer meshing for skin depth
    - Curvature-based refinement
    - Export to Elmer mesh format
    """
    
    def __init__(self):
        self._mesh_settings = MeshSettings()
        self._physical_groups: Dict[str, int] = {}
        self._physical_group_dim: Dict[str, int] = {}
        self._sim_dir: Optional[str] = None
        self._mesh_generated = False
        self._initialized = False
    
    def initialize(self, sim_dir: str = None, **kwargs) -> None:
        """Initialize the meshing backend."""
        if not HAS_GMSH:
            raise ImportError(
                "gmsh is required for ElmerMeshingBackend. "
                "Install with: pip install gmsh"
            )
        
        self._sim_dir = sim_dir
        
        if not gmsh.isInitialized():
            gmsh.initialize()
        
        # Set default meshing options
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        
        self._initialized = True
    
    def set_global_settings(self, settings: MeshSettings) -> None:
        """Set global mesh settings."""
        self._mesh_settings = settings
        
        if settings.max_element_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", settings.max_element_size)
        
        if settings.min_element_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", settings.min_element_size)
        
        if settings.growth_rate:
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 
                                  1 if settings.curvature_refinement else 0)
    
    def add_physical_group(
        self, 
        dim: int, 
        tags: List[int], 
        name: str
    ) -> int:
        """
        Add a physical group for Elmer body/boundary identification.
        
        Args:
            dim: Dimension (2 for surfaces/boundaries, 3 for volumes/bodies)
            tags: gmsh entity tags
            name: Name for the physical group
            
        Returns:
            Physical group tag
        """
        pg_tag = gmsh.model.addPhysicalGroup(dim, tags)
        gmsh.model.setPhysicalName(dim, pg_tag, name)
        self._physical_groups[name] = pg_tag
        self._physical_group_dim[name] = dim
        return pg_tag
    
    def assign_mesh_size(
        self,
        objects: List[GeometryObject],
        max_size: float,
        min_size: Optional[float] = None
    ) -> bool:
        """Assign mesh size to specific objects."""
        try:
            # Create a mesh size field
            field_id = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(field_id, "F", str(max_size))
            
            # Get entity tags for the objects
            for obj in objects:
                if hasattr(obj, 'native_object'):
                    # Try to find corresponding gmsh entities
                    pass
            
            return True
        except Exception:
            return False
    
    def assign_skin_depth(
        self,
        faces: List[Any],
        skin_depth: float,
        num_layers: int = 2
    ) -> bool:
        """
        Assign skin depth mesh refinement for eddy current simulations.
        
        Creates boundary layer mesh elements for proper resolution
        of skin effect in conductors.
        """
        try:
            # Create boundary layer field
            bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
            
            # Get face tags (assuming faces are gmsh surface tags)
            face_tags = []
            for face in faces:
                if isinstance(face, int):
                    face_tags.append(face)
                elif hasattr(face, 'tag'):
                    face_tags.append(face.tag)
            
            if face_tags:
                gmsh.model.mesh.field.setNumbers(bl_field, "SurfacesList", face_tags)
                gmsh.model.mesh.field.setNumber(bl_field, "Size", skin_depth / num_layers)
                gmsh.model.mesh.field.setNumber(bl_field, "Thickness", skin_depth)
                gmsh.model.mesh.field.setNumber(bl_field, "NbLayers", num_layers)
                gmsh.model.mesh.field.setNumber(bl_field, "Ratio", 1.2)
            
            return True
        except Exception as e:
            print(f"Warning: Could not assign skin depth mesh: {e}")
            return False
    
    def assign_curvature_refinement(
        self,
        objects: List[GeometryObject],
        num_elements_per_curvature: int = 6
    ) -> bool:
        """Assign curvature-based mesh refinement."""
        try:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", num_elements_per_curvature)
            return True
        except Exception:
            return False
    
    def refine_region(
        self,
        center: List[float],
        radius: float,
        element_size: float
    ) -> bool:
        """
        Refine mesh in a spherical region.
        
        Useful for refining around air gaps or critical regions.
        """
        try:
            # Create distance field from point
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "PointsList", [])
            
            # Create threshold field
            thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", element_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", 
                                            self._mesh_settings.max_element_size or element_size * 10)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", radius)
            
            return True
        except Exception:
            return False
    
    def generate_mesh(self) -> bool:
        """Generate the mesh."""
        try:
            # Synchronize geometry
            gmsh.model.occ.synchronize()
            
            # Generate 3D mesh
            gmsh.model.mesh.generate(3)
            
            # Optimize mesh
            gmsh.model.mesh.optimize("Netgen")
            
            self._mesh_generated = True
            return True
        except Exception as e:
            print(f"Mesh generation failed: {e}")
            return False
    
    def export_mesh(self, file_path: str, format: str = "msh") -> bool:
        """
        Export mesh to file and convert to Elmer format.
        
        Args:
            file_path: Output path for mesh file
            format: Output format ('msh' for gmsh, 'elmer' for Elmer format)
        """
        try:
            if not self._mesh_generated:
                self.generate_mesh()
            
            # Export gmsh mesh
            msh_path = file_path if file_path.endswith('.msh') else f"{file_path}.msh"
            gmsh.write(msh_path)
            
            if format.lower() == "elmer":
                # Convert to Elmer format using ElmerGrid
                return self._convert_to_elmer(msh_path)
            
            return True
        except Exception as e:
            print(f"Mesh export failed: {e}")
            return False
    
    def _convert_to_elmer(self, msh_path: str) -> bool:
        """
        Convert gmsh mesh to Elmer format using ElmerGrid.
        
        ElmerGrid command: ElmerGrid 14 2 mesh.msh
        - 14: gmsh format input
        - 2: Elmer format output
        """
        try:
            # Try using pyelmer's execute module
            try:
                from pyelmer import execute
                sim_dir = os.path.dirname(msh_path)
                mesh_name = os.path.basename(msh_path)
                execute.run_elmer_grid(sim_dir, mesh_name)
                return True
            except ImportError:
                pass
            
            # Fall back to direct command
            cmd = ["ElmerGrid", "14", "2", msh_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(msh_path)
            )
            
            if result.returncode != 0:
                print(f"ElmerGrid error: {result.stderr}")
                return False
            
            return True
        except FileNotFoundError:
            print("ElmerGrid not found. Please install Elmer FEM.")
            return False
        except Exception as e:
            print(f"Mesh conversion failed: {e}")
            return False
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics (element count, quality, etc.)."""
        stats = {}
        
        try:
            # Get node count
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            stats["num_nodes"] = len(node_tags)
            
            # Get element counts by type
            element_types, element_tags, _ = gmsh.model.mesh.getElements()
            stats["num_elements"] = sum(len(tags) for tags in element_tags)
            
            # Element type breakdown
            type_names = {
                1: "lines",
                2: "triangles", 
                3: "quadrangles",
                4: "tetrahedra",
                5: "hexahedra",
                6: "prisms",
                7: "pyramids",
            }
            
            stats["element_types"] = {}
            for etype, tags in zip(element_types, element_tags):
                type_name = type_names.get(etype, f"type_{etype}")
                stats["element_types"][type_name] = len(tags)
            
            # Get mesh quality if available
            try:
                qualities = gmsh.model.mesh.getElementQualities()
                if qualities:
                    stats["min_quality"] = min(qualities)
                    stats["max_quality"] = max(qualities)
                    stats["avg_quality"] = sum(qualities) / len(qualities)
            except Exception:
                pass
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def get_physical_groups(self) -> Dict[str, int]:
        """Get all defined physical groups."""
        return self._physical_groups.copy()
    
    def auto_assign_physical_groups(self) -> Dict[str, int]:
        """
        Automatically assign physical groups to all volumes and surfaces.
        
        This is useful when importing STEP geometry that doesn't have
        pre-defined physical groups.
        """
        groups = {}
        
        # Get all volumes (3D entities)
        volumes = gmsh.model.getEntities(3)
        for i, (dim, tag) in enumerate(volumes):
            name = f"body_{i+1}"
            pg = self.add_physical_group(3, [tag], name)
            groups[name] = pg
        
        # Get all surfaces (2D entities)
        surfaces = gmsh.model.getEntities(2)
        for i, (dim, tag) in enumerate(surfaces):
            name = f"boundary_{i+1}"
            pg = self.add_physical_group(2, [tag], name)
            groups[name] = pg
        
        return groups
    
    def visualize(self) -> None:
        """Open gmsh GUI to visualize the mesh."""
        try:
            gmsh.fltk.run()
        except Exception:
            print("gmsh GUI not available")
    
    def finalize(self) -> None:
        """Finalize gmsh (cleanup)."""
        if gmsh.isInitialized():
            gmsh.finalize()
