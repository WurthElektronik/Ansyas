"""
Elmer FEM backend implementations for Ansyas.

This module provides open-source alternatives to the Ansys backends,
using ElmerFEM for solving, gmsh for meshing, and CadQuery for geometry.

Components:
- ElmerGeometryBackend: CadQuery + gmsh for 3D geometry
- ElmerMeshingBackend: gmsh for mesh generation
- ElmerMaterialBackend: pyelmer Material definitions
- ElmerExcitationBackend: Boundary conditions and body forces
- ElmerSolverBackend: ElmerSolver via pyelmer

Dependencies:
- pyelmer: Python interface to Elmer
- gmsh: Open-source mesh generator
- cadquery: Parametric 3D CAD

Supported simulation types:
- EddyCurrent / AC Magnetic (MagnetoDynamics)
- Magnetostatic (WhitneyAVSolver)
- Thermal / SteadyState (HeatSolver)
- Electrostatic (StatElecSolver)
- Transient (MagnetoDynamics with time stepping)
"""

from .geometry import ElmerGeometryBackend
from .meshing import ElmerMeshingBackend
from .material import ElmerMaterialBackend
from .excitation import ElmerExcitationBackend
from .solver import ElmerSolverBackend
from .postprocess import ElmerPostprocessor

# MAS processing utilities (optional - requires PyMKF and MVB)
try:
    from .mas_processor import (
        process_mas_magnetic,
        create_elmer_mesh,
        convert_mesh_to_elmer,
        setup_magnetostatic_simulation,
        run_elmer_simulation,
        check_dependencies,
        ElmerMagneticGeometry,
    )
    HAS_MAS_PROCESSOR = True
except ImportError:
    HAS_MAS_PROCESSOR = False

__all__ = [
    "ElmerGeometryBackend",
    "ElmerMeshingBackend",
    "ElmerMaterialBackend",
    "ElmerExcitationBackend",
    "ElmerSolverBackend",
    "ElmerPostprocessor",
    # MAS processor exports (when available)
    "process_mas_magnetic",
    "create_elmer_mesh",
    "convert_mesh_to_elmer",
    "setup_magnetostatic_simulation",
    "run_elmer_simulation",
    "check_dependencies",
    "ElmerMagneticGeometry",
    "HAS_MAS_PROCESSOR",
]
