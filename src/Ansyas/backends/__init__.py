"""
Backend interfaces for decoupled FEM simulation.

This module provides abstract interfaces for:
- Geometry creation (3D model building)
- Meshing
- Solving (FEM simulation)
- Excitation setup
- Material management

Implemented backends:
- Ansys (PyAEDT) - Commercial solution (requires license)
- Elmer (pyelmer + gmsh) - Open-source alternative
"""

from .base import (
    GeometryBackend,
    MeshingBackend,
    SolverBackend,
    ExcitationBackend,
    MaterialBackend,
    BackendRegistry,
    GeometryObject,
    MaterialDefinition,
    MeshSettings,
    SolverSetup,
    Axis,
    Plane,
)

# Track which backends are available
HAS_ANSYS = False
HAS_ELMER = False

# Try to import Ansys backends (optional)
try:
    from .ansys import (
        AnsysGeometryBackend,
        AnsysMaterialBackend,
        AnsysMeshingBackend,
        AnsysExcitationBackend,
        AnsysSolverBackend,
    )
    
    # Register Ansys backends
    BackendRegistry.register_geometry("ansys", AnsysGeometryBackend)
    BackendRegistry.register_material("ansys", AnsysMaterialBackend)
    BackendRegistry.register_meshing("ansys", AnsysMeshingBackend)
    BackendRegistry.register_excitation("ansys", AnsysExcitationBackend)
    BackendRegistry.register_solver("ansys", AnsysSolverBackend)
    
    HAS_ANSYS = True
except ImportError:
    # Ansys not available - create placeholder classes
    AnsysGeometryBackend = None
    AnsysMaterialBackend = None
    AnsysMeshingBackend = None
    AnsysExcitationBackend = None
    AnsysSolverBackend = None

# Try to import Elmer backends (optional)
try:
    from .elmer import (
        ElmerGeometryBackend,
        ElmerMeshingBackend,
        ElmerMaterialBackend,
        ElmerExcitationBackend,
        ElmerSolverBackend,
        ElmerPostprocessor,
    )
    
    # Register Elmer backends
    BackendRegistry.register_geometry("elmer", ElmerGeometryBackend)
    BackendRegistry.register_material("elmer", ElmerMaterialBackend)
    BackendRegistry.register_meshing("elmer", ElmerMeshingBackend)
    BackendRegistry.register_excitation("elmer", ElmerExcitationBackend)
    BackendRegistry.register_solver("elmer", ElmerSolverBackend)
    
    HAS_ELMER = True
except ImportError:
    # Elmer not available - create placeholder classes
    ElmerGeometryBackend = None
    ElmerMeshingBackend = None
    ElmerMaterialBackend = None
    ElmerExcitationBackend = None
    ElmerSolverBackend = None
    ElmerPostprocessor = None


__all__ = [
    # Abstract interfaces
    "GeometryBackend",
    "MeshingBackend",
    "SolverBackend",
    "ExcitationBackend",
    "MaterialBackend",
    "BackendRegistry",
    # Data classes
    "GeometryObject",
    "MaterialDefinition",
    "MeshSettings",
    "SolverSetup",
    "Axis",
    "Plane",
    # Availability flags
    "HAS_ANSYS",
    "HAS_ELMER",
    # Ansys implementations (may be None)
    "AnsysGeometryBackend",
    "AnsysMaterialBackend",
    "AnsysMeshingBackend",
    "AnsysExcitationBackend",
    "AnsysSolverBackend",
    # Elmer implementations (may be None)
    "ElmerGeometryBackend",
    "ElmerMeshingBackend",
    "ElmerMaterialBackend",
    "ElmerExcitationBackend",
    "ElmerSolverBackend",
    "ElmerPostprocessor",
]
