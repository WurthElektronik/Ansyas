"""
Backend interfaces for decoupled FEM simulation.

This module provides abstract interfaces for:
- Geometry creation (3D model building)
- Meshing
- Solving (FEM simulation)
- Excitation setup
- Material management

Currently implemented backend:
- Ansys (PyAEDT) - Complete commercial solution
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

# Import concrete implementations for registration
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
    # Ansys implementations
    "AnsysGeometryBackend",
    "AnsysMaterialBackend",
    "AnsysMeshingBackend",
    "AnsysExcitationBackend",
    "AnsysSolverBackend",
]
