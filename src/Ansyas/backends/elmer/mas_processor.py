"""
MAS data processor for Elmer FEM simulations.

This module provides functions to:
1. Build 3D geometry using OpenMagneticsVirtualBuilder
2. Import geometry into gmsh for meshing with physical groups
3. Convert mesh to Elmer format with ElmerGrid
4. Setup and run Elmer simulation
5. Extract results (inductance, losses, etc.)

The workflow bridges the MAS magnetic component format with Elmer's
mesh-based FEM requirements.
"""

import os
import sys
import json
import math
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add MVB path if not already in system
MVB_PATH = os.path.expanduser("~/OpenMagnetics/MVB/src")
if MVB_PATH not in sys.path:
    sys.path.insert(0, MVB_PATH)

try:
    import PyMKF
    HAS_PYMKF = True
except ImportError:
    PyMKF = None
    HAS_PYMKF = False

try:
    from OpenMagneticsVirtualBuilder.builder import Builder
    HAS_MVB = True
except ImportError:
    HAS_MVB = False
    Builder = None

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    gmsh = None


@dataclass
class ElmerSimulationResult:
    """Container for Elmer simulation results."""
    
    success: bool
    vtu_path: Optional[str] = None
    electromagnetic_energy: float = 0.0
    eddy_current_power: float = 0.0
    inductance: float = 0.0
    max_b_field: float = 0.0
    max_h_field: float = 0.0
    error_message: str = ""


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    # Check for ElmerSolver
    elmer_path = os.path.expanduser("~/elmer/install/bin/ElmerSolver")
    has_elmer = os.path.exists(elmer_path)
    
    return {
        "PyMKF": HAS_PYMKF,
        "MVB": HAS_MVB,
        "gmsh": HAS_GMSH,
        "ElmerSolver": has_elmer,
    }


def build_geometry_from_mas(
    magnetic_data: Dict,
    output_path: str,
    project_name: str = "magnetic",
    max_turns_per_winding: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Build 3D geometry from MAS magnetic data using MVB.
    
    Parameters
    ----------
    magnetic_data : dict
        MAS magnetic section with 'core' and 'coil' keys.
    output_path : str
        Directory to save output files.
    project_name : str
        Name prefix for output files.
    max_turns_per_winding : int, optional
        Limit turns per winding for faster meshing (useful for testing).
        
    Returns
    -------
    Tuple[str, str]
        Paths to (step_file, stl_file).
    """
    if not HAS_MVB:
        raise ImportError("OpenMagneticsVirtualBuilder is required")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Process core with PyMKF if geometricalDescription is missing
    core = magnetic_data.get('core', {})
    if core.get('geometricalDescription') is None and HAS_PYMKF:
        core = PyMKF.calculate_core_data(core, True)
        magnetic_data = {**magnetic_data, 'core': core}
    
    # Optionally limit turns for faster processing
    if max_turns_per_winding is not None:
        coil = magnetic_data.get('coil', {}).copy()
        turns_desc = coil.get('turnsDescription', [])
        
        if turns_desc:
            # Group turns by winding
            windings = {}
            for turn in turns_desc:
                winding_name = turn.get('name', '').split(' ')[0].lower()
                if winding_name not in windings:
                    windings[winding_name] = []
                windings[winding_name].append(turn)
            
            # Limit turns per winding
            limited_turns = []
            for winding_name, winding_turns in windings.items():
                limited_turns.extend(winding_turns[:max_turns_per_winding])
            
            coil['turnsDescription'] = limited_turns
            magnetic_data = {**magnetic_data, 'coil': coil}
    
    # Build geometry with MVB
    builder = Builder()
    result = builder.get_magnetic(
        magnetic_data,
        project_name=project_name,
        output_path=output_path,
        export_files=True
    )
    
    if isinstance(result, tuple):
        return result  # (step_path, stl_path)
    else:
        step_path = os.path.join(output_path, f"{project_name}.step")
        stl_path = os.path.join(output_path, f"{project_name}.stl")
        return step_path, stl_path


def create_elmer_mesh(
    step_file: str,
    output_path: str,
    mesh_name: str = "mesh",
    air_padding: float = 10.0,
    max_element_size: float = 4.0,
    min_element_size: float = 1.0,
    scale_to_meters: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    Create Elmer-compatible mesh from STEP geometry.
    
    Parameters
    ----------
    step_file : str
        Path to STEP geometry file.
    output_path : str
        Directory for output mesh.
    mesh_name : str
        Name for mesh files.
    air_padding : float
        Padding around geometry for air region (mm).
    max_element_size : float
        Maximum mesh element size (mm).
    min_element_size : float
        Minimum mesh element size (mm).
    scale_to_meters : bool
        Scale from mm to meters using ElmerGrid.
        
    Returns
    -------
    Tuple[str, Dict[str, int]]
        Path to Elmer mesh directory and dict of body numbers.
    """
    if not HAS_GMSH:
        raise ImportError("gmsh is required for mesh generation")
    
    os.makedirs(output_path, exist_ok=True)
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("magnetic")
    
    try:
        # Import STEP geometry
        entities = gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        
        # Get volumes and classify
        volumes = gmsh.model.getEntities(3)
        core_tags = []
        coil_tags = []
        
        for dim, tag in volumes:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            width = max(bbox[3]-bbox[0], bbox[4]-bbox[1])
            height = bbox[5] - bbox[2]
            
            # Core pieces are larger (>25mm width or >15mm height)
            if width > 25 or height > 15:
                core_tags.append(tag)
            else:
                coil_tags.append(tag)
        
        # Create air box
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        air_box = gmsh.model.occ.addBox(
            xmin - air_padding, ymin - air_padding, zmin - air_padding,
            (xmax - xmin) + 2*air_padding,
            (ymax - ymin) + 2*air_padding,
            (zmax - zmin) + 2*air_padding
        )
        
        # Fragment to create conformal mesh
        all_volumes = [(3, v[1]) for v in volumes]
        gmsh.model.occ.fragment([(3, air_box)], all_volumes)
        gmsh.model.occ.synchronize()
        
        # Re-identify volumes after fragmentation
        new_volumes = gmsh.model.getEntities(3)
        new_core = []
        new_coil = []
        new_air = []
        
        for dim, tag in new_volumes:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            vol = (bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2])
            width = max(bbox[3]-bbox[0], bbox[4]-bbox[1])
            
            if vol > 50000:  # Air (largest)
                new_air.append(tag)
            elif width > 25:  # Core
                new_core.append(tag)
            else:  # Coil
                new_coil.append(tag)
        
        # Create physical groups with specific tags
        body_numbers = {}
        
        if new_core:
            gmsh.model.addPhysicalGroup(3, new_core, tag=1, name="core")
            body_numbers["core"] = 1
        
        if new_coil:
            gmsh.model.addPhysicalGroup(3, new_coil, tag=2, name="coil")
            body_numbers["coil"] = 2
        
        if new_air:
            gmsh.model.addPhysicalGroup(3, new_air, tag=3, name="air")
            body_numbers["air"] = 3
        
        # Outer boundary
        outer_bbox = [xmin - air_padding, ymin - air_padding, zmin - air_padding,
                      xmax + air_padding, ymax + air_padding, zmax + air_padding]
        surfaces = gmsh.model.getEntities(2)
        outer_surfs = []
        tol = 0.5
        
        for dim, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            on_outer = (abs(bbox[0] - outer_bbox[0]) < tol or 
                        abs(bbox[3] - outer_bbox[3]) < tol or
                        abs(bbox[1] - outer_bbox[1]) < tol or 
                        abs(bbox[4] - outer_bbox[4]) < tol or
                        abs(bbox[2] - outer_bbox[2]) < tol or 
                        abs(bbox[5] - outer_bbox[5]) < tol)
            if on_outer:
                outer_surfs.append(tag)
        
        if outer_surfs:
            gmsh.model.addPhysicalGroup(2, outer_surfs, tag=4, name="outer_boundary")
            body_numbers["outer_boundary"] = 1  # After renumbering
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_element_size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal3D
        gmsh.option.setNumber("Mesh.Optimize", 1)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        
        # Save mesh in gmsh 2.2 format for ElmerGrid
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        msh_path = os.path.join(output_path, f"{mesh_name}.msh")
        gmsh.write(msh_path)
        
    finally:
        gmsh.finalize()
    
    # Convert to Elmer format with ElmerGrid
    elmer_grid = os.path.expanduser("~/elmer/install/bin/ElmerGrid")
    mesh_dir = os.path.join(output_path, mesh_name)
    
    # Build command - use basename since we're running in output_path
    msh_basename = os.path.basename(msh_path)
    cmd = [elmer_grid, "14", "2", msh_basename, "-autoclean"]
    if scale_to_meters:
        cmd.extend(["-scale", "0.001", "0.001", "0.001"])
    
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/elmer/install/bin") + ":" + env.get("PATH", "")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_path, env=env)
    
    # Check if mesh directory was created
    if not os.path.exists(mesh_dir):
        raise RuntimeError(f"ElmerGrid failed to create mesh directory: {result.stderr}")
    
    return mesh_dir, body_numbers


def generate_sif_file(
    output_path: str,
    mesh_dir: str,
    body_numbers: Dict[str, int],
    core_permeability: float = 2000.0,
    current_density: float = 1.0e6,
    simulation_type: str = "magnetostatic",
) -> str:
    """
    Generate Elmer SIF file for magnetostatic simulation.
    
    Parameters
    ----------
    output_path : str
        Directory to save SIF file.
    mesh_dir : str
        Path to Elmer mesh directory.
    body_numbers : dict
        Dictionary mapping body names to body numbers.
    core_permeability : float
        Relative permeability of core material.
    current_density : float
        Current density in coil (A/m^2).
    simulation_type : str
        Type of simulation ('magnetostatic', 'eddy_current').
        
    Returns
    -------
    str
        Path to generated SIF file.
    """
    mesh_rel = os.path.basename(mesh_dir)
    
    sif_content = f'''! Elmer {simulation_type} simulation
! Generated by Ansyas Elmer backend

Check Keywords "Warn"

Header
  Mesh DB "." "{mesh_rel}"
  Results Directory "."
End

Simulation
  Coordinate System = Cartesian 3D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Max Output Level = 5
End

Constants
  Permittivity Of Vacuum = 8.854e-12
  Permeability Of Vacuum = 1.2566e-6
End

! Materials
Material 1
  Name = "Ferrite"
  Relative Permeability = {core_permeability}
  Electric Conductivity = 0.0
End

Material 2
  Name = "Copper"
  Relative Permeability = 1.0
  Electric Conductivity = 5.96e7
End

Material 3
  Name = "Air"
  Relative Permeability = 1.0
  Electric Conductivity = 0.0
End

! Bodies
Body 1
  Name = "Core"
  Equation = 1
  Material = 1
End

Body 2
  Name = "Coil"
  Equation = 1
  Material = 2
  Body Force = 1
End

Body 3
  Name = "Air"
  Equation = 1
  Material = 3
End

! Current density body force
Body Force 1
  Name = "CurrentDensity"
  Current Density 3 = Real {current_density}
End

! Equation
Equation 1
  Name = "MagnetoDynamics"
  Active Solvers(3) = 1 2 3
End

! Solvers
Solver 1
  Equation = MGDynamics
  Procedure = "MagnetoDynamics" "WhitneyAVSolver"
  Variable = AV
  
  Linear System Solver = Direct
  Linear System Direct Method = UMFPACK
  
  Steady State Convergence Tolerance = 1.0e-8
End

Solver 2
  Equation = MGDynamicsCalc
  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"
  
  Potential Variable = "AV"
  Calculate Magnetic Field Strength = True
  Calculate Magnetic Flux Density = True
  Calculate Current Density = True
  Calculate Electric Field = False
  
  Linear System Solver = Iterative
  Linear System Iterative Method = CG
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Preconditioning = ILU0
  
  Steady State Convergence Tolerance = 1.0e-6
End

Solver 3
  Equation = ResultOutput
  Exec Solver = After Timestep
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  
  Output File Name = "results"
  Vtu Format = True
  Save Geometry Ids = True
End

! Boundary condition
Boundary Condition 1
  Name = "OuterBoundary"
  AV {{e}} = 0.0
End
'''
    
    sif_path = os.path.join(output_path, "case.sif")
    with open(sif_path, 'w') as f:
        f.write(sif_content)
    
    # Create STARTINFO file
    startinfo_path = os.path.join(output_path, "ELMERSOLVER_STARTINFO")
    with open(startinfo_path, 'w') as f:
        f.write("case.sif\n")
    
    return sif_path


def run_elmer_simulation(sim_dir: str, timeout: int = 300) -> ElmerSimulationResult:
    """
    Run ElmerSolver on a prepared simulation.
    
    Parameters
    ----------
    sim_dir : str
        Simulation directory containing case.sif.
    timeout : int
        Timeout in seconds.
        
    Returns
    -------
    ElmerSimulationResult
        Simulation results.
    """
    elmer_solver = os.path.expanduser("~/elmer/install/bin/ElmerSolver")
    
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/elmer/install/bin") + ":" + env.get("PATH", "")
    
    try:
        result = subprocess.run(
            [elmer_solver],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        output = result.stdout + result.stderr
        
        # Parse results from output
        em_energy = 0.0
        eddy_power = 0.0
        
        import re
        for line in output.split('\n'):
            if 'ElectroMagnetic Field Energy:' in line:
                try:
                    # Extract the number after "Energy:"
                    match = re.search(r'Energy:\s+([\d.E+-]+)', line)
                    if match:
                        em_energy = float(match.group(1))
                except:
                    pass
            elif 'Eddy current power:' in line:
                try:
                    # Extract the number after "power:"
                    match = re.search(r'power:\s+([\d.E+-]+)', line)
                    if match:
                        eddy_power = float(match.group(1))
                except:
                    pass
        
        # Find VTU file
        vtu_path = None
        mesh_dir = os.path.join(sim_dir, "mesh")
        if os.path.exists(mesh_dir):
            for f in os.listdir(mesh_dir):
                if f.endswith('.vtu'):
                    vtu_path = os.path.join(mesh_dir, f)
                    break
        
        # Check if simulation succeeded
        if result.returncode == 0 and 'ALL DONE' in output:
            return ElmerSimulationResult(
                success=True,
                vtu_path=vtu_path,
                electromagnetic_energy=em_energy,
                eddy_current_power=eddy_power,
            )
        else:
            return ElmerSimulationResult(
                success=False,
                error_message=output[-1000:] if len(output) > 1000 else output
            )
            
    except subprocess.TimeoutExpired:
        return ElmerSimulationResult(
            success=False,
            error_message=f"Simulation timed out after {timeout} seconds"
        )
    except Exception as e:
        return ElmerSimulationResult(
            success=False,
            error_message=str(e)
        )


def extract_results(vtu_path: str, current: float = 1.0) -> Dict[str, Any]:
    """
    Extract simulation results from VTU file.
    
    Parameters
    ----------
    vtu_path : str
        Path to VTU results file.
    current : float
        Applied current (A) for inductance calculation.
        
    Returns
    -------
    dict
        Dictionary with extracted results.
    """
    try:
        import pyvista as pv
        import numpy as np
    except ImportError:
        return {"error": "pyvista not installed"}
    
    mesh = pv.read(vtu_path)
    
    results = {
        "num_points": mesh.n_points,
        "num_cells": mesh.n_cells,
    }
    
    if "magnetic flux density" in mesh.array_names:
        B = mesh["magnetic flux density"]
        B_mag = np.linalg.norm(B, axis=1)
        results["B_max"] = float(np.max(B_mag))
        results["B_mean"] = float(np.mean(B_mag))
    
    if "magnetic field strength" in mesh.array_names:
        H = mesh["magnetic field strength"]
        H_mag = np.linalg.norm(H, axis=1)
        results["H_max"] = float(np.max(H_mag))
        results["H_mean"] = float(np.mean(H_mag))
    
    if "current density" in mesh.array_names:
        J = mesh["current density"]
        J_mag = np.linalg.norm(J, axis=1)
        results["J_max"] = float(np.max(J_mag))
        results["J_mean"] = float(np.mean(J_mag))
    
    return results


def run_mas_simulation(
    mas_file: str,
    output_path: str,
    max_turns_per_winding: Optional[int] = None,
    core_permeability: float = 2000.0,
    current_density: float = 1.0e6,
) -> Dict[str, Any]:
    """
    Run complete Elmer simulation from MAS file.
    
    This is the main entry point for MAS-to-Elmer workflow.
    
    Parameters
    ----------
    mas_file : str
        Path to MAS JSON file.
    output_path : str
        Directory for output files.
    max_turns_per_winding : int, optional
        Limit turns per winding for faster meshing.
    core_permeability : float
        Relative permeability of core material.
    current_density : float
        Current density in coil (A/m^2).
        
    Returns
    -------
    dict
        Simulation results and metadata.
    """
    # Check dependencies
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v]
    if missing:
        return {"error": f"Missing dependencies: {missing}"}
    
    os.makedirs(output_path, exist_ok=True)
    
    # Load MAS file
    with open(mas_file) as f:
        data = json.load(f)
    
    magnetic_data = data.get('magnetic', {})
    
    # Step 1: Build geometry
    print("Building geometry with MVB...")
    step_file, stl_file = build_geometry_from_mas(
        magnetic_data,
        output_path,
        project_name="magnetic",
        max_turns_per_winding=max_turns_per_winding
    )
    
    # Step 2: Create mesh
    print("Creating mesh with gmsh...")
    mesh_dir, body_numbers = create_elmer_mesh(
        step_file,
        output_path,
        mesh_name="mesh"
    )
    
    # Step 3: Generate SIF file
    print("Generating Elmer SIF file...")
    sif_path = generate_sif_file(
        output_path,
        mesh_dir,
        body_numbers,
        core_permeability=core_permeability,
        current_density=current_density
    )
    
    # Step 4: Run simulation
    print("Running Elmer simulation...")
    result = run_elmer_simulation(output_path)
    
    # Step 5: Extract results
    results = {
        "success": result.success,
        "electromagnetic_energy": result.electromagnetic_energy,
        "eddy_current_power": result.eddy_current_power,
        "geometry_file": step_file,
        "mesh_dir": mesh_dir,
        "sif_file": sif_path,
    }
    
    if result.success and result.vtu_path:
        print("Extracting results...")
        vtu_results = extract_results(result.vtu_path)
        results.update(vtu_results)
        results["vtu_file"] = result.vtu_path
    else:
        results["error"] = result.error_message
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python mas_processor.py <mas_file.json> <output_dir>")
        sys.exit(1)
    
    mas_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    results = run_mas_simulation(mas_file, output_dir, max_turns_per_winding=2)
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
