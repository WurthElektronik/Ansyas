"""
MAS data processor for Elmer FEM simulations.

This module provides the main entry point for running Elmer FEM simulations
on MAS magnetic component descriptions. It handles:
1. Loading and normalizing MAS data (both simple and complete formats)
2. Building 3D geometry using OpenMagneticsVirtualBuilder
3. Meshing with gmsh (subprocess-safe) + Netgen fallback
4. Running magnetostatic/harmonic Elmer simulations
5. Extracting results into MAS Outputs format

The proven simulation logic lives in tests/validate_elmer_inductance.py.
This module wraps it with proper MAS I/O handling.
"""

import os
import sys
import json
import math
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Add paths
MVB_PATH = os.path.expanduser("~/OpenMagnetics/MVB/src")
ANSYAS_PATH = os.path.expanduser("~/wuerth/Ansyas/src")
TESTS_PATH = os.path.expanduser("~/wuerth/Ansyas/tests")
for path in [MVB_PATH, ANSYAS_PATH, TESTS_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from PyOpenMagnetics import PyOpenMagnetics as PyOM
    HAS_PYOM = True
except ImportError:
    PyOM = None
    HAS_PYOM = False

try:
    from OpenMagneticsVirtualBuilder.builder import Builder
    HAS_MVB = True
except ImportError:
    HAS_MVB = False
    Builder = None

# Import proven simulation functions from validation script
from validate_elmer_inductance import (
    TurnInfo,
    load_mas_file,
    get_core_data,
    extract_turns_info,
    get_material_permeability,
    calculate_analytical_inductance,
    build_geometry,
    create_mesh_with_turns,
    create_mesh_with_winding_regions,
    create_mesh_with_netgen,
    generate_sif_with_coil_solver,
    generate_sif_with_tangential_current,
    run_elmer,
    calculate_inductance_from_energy,
)

# Import MAS models
try:
    import MAS_models as MAS
    HAS_MAS = True
except ImportError:
    MAS = None
    HAS_MAS = False


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    max_turns: Optional[int] = None
    total_current: float = 1.0
    method: str = "coilsolver"
    include_bobbin: bool = True
    timeout: int = 600


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    return {
        "PyOpenMagnetics": HAS_PYOM,
        "MVB": HAS_MVB,
        "ElmerSolver": shutil.which("ElmerSolver") is not None,
        "ElmerGrid": shutil.which("ElmerGrid") is not None,
        "MAS_models": HAS_MAS,
    }


def normalize_mas_data(data: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Normalize MAS data from either simple or complete format.

    Simple format: {"magnetic": {...}, "operatingPoints": [...]}
    Complete format: {"inputs": {"operatingPoints": [...]}, "magnetic": {...}, "outputs": [...]}

    Returns:
        (magnetic_data, operating_points)
    """
    magnetic = data.get('magnetic', data)

    # Operating points can be at top level or under inputs
    operating_points = data.get('operatingPoints',
                        magnetic.get('operatingPoints',
                        data.get('inputs', {}).get('operatingPoints', [])))

    return magnetic, operating_points


def autocomplete_magnetic(magnetic: Dict) -> Dict:
    """Run PyOpenMagnetics autocomplete to fill in missing coil/core data."""
    if not HAS_PYOM:
        raise ImportError("PyOpenMagnetics required for autocomplete")

    result = PyOM.magnetic_autocomplete(magnetic, {})
    if isinstance(result, str):
        if result.startswith('Exception:'):
            raise ValueError(f"Autocomplete failed: {result}")
        return json.loads(result)
    return result


def detect_core_type(magnetic_data: Dict) -> str:
    """Detect whether core is concentric or toroidal from MAS data."""
    core = magnetic_data.get('core', {})
    func_desc = core.get('functionalDescription', {})
    shape = func_desc.get('shape', {})
    shape_name = shape.get('name', '') if isinstance(shape, dict) else str(shape)

    # Toroidal shapes start with 'T' (e.g., T 40/24/15)
    if shape_name.strip().startswith('T ') or shape_name.strip().startswith('T_'):
        return 'toroidal'
    # Also check family
    family = shape.get('family', '') if isinstance(shape, dict) else ''
    if 'toroid' in family.lower():
        return 'toroidal'
    return 'concentric'


def get_winding_names(magnetic_data: Dict) -> List[str]:
    """Get list of winding names from MAS coil functionalDescription."""
    coil = magnetic_data.get('coil', {})
    func_desc = coil.get('functionalDescription', [])
    return [w.get('name', f'Winding {i}') for i, w in enumerate(func_desc)]


def run_magnetostatic_inductance(
    mas_file_or_data,
    output_path: str,
    config: Optional[SimulationConfig] = None,
) -> Dict[str, Any]:
    """
    Run magnetostatic simulation for inductance extraction.

    This is the main entry point. It handles:
    - Loading and normalizing MAS data
    - Autocompleting missing coil/core geometry
    - Building 3D geometry, meshing, running Elmer
    - Returning results in a structured dict

    Parameters
    ----------
    mas_file_or_data : str or dict
        Path to MAS JSON file, or pre-loaded MAS data dict.
    output_path : str
        Directory for output files.
    config : SimulationConfig, optional
        Simulation configuration. Defaults to sensible values.

    Returns
    -------
    dict with keys:
        success, magnetic_data, core_type, num_turns, core_permeability,
        analytical_inductance_H, elmer_inductance_H, error_percent,
        electromagnetic_energy_J, winding_names, ...
    """
    if config is None:
        config = SimulationConfig()

    os.makedirs(output_path, exist_ok=True)
    results = {'success': False}

    # Load data
    if isinstance(mas_file_or_data, str):
        with open(mas_file_or_data) as f:
            raw_data = json.load(f)
        results['mas_file'] = mas_file_or_data
    else:
        raw_data = mas_file_or_data

    magnetic_data, operating_points = normalize_mas_data(raw_data)
    results['operating_points_count'] = len(operating_points)

    # Autocomplete
    print("Auto-completing magnetic data...")
    magnetic_data = autocomplete_magnetic(magnetic_data)

    core_type = detect_core_type(magnetic_data)
    results['core_type'] = core_type
    winding_names = get_winding_names(magnetic_data)
    results['winding_names'] = winding_names
    print(f"Core type: {core_type}, Windings: {winding_names}")

    # Extract turns info
    turns_info = extract_turns_info(magnetic_data, core_type)
    num_turns = len(turns_info)
    results['total_turns'] = num_turns

    # Apply turn limit if set
    if config.max_turns and config.max_turns < num_turns:
        primary_turns = [t for t in turns_info if 'primary' in t.winding.lower() or
                         'winding 1' in t.winding.lower() or
                         t.winding == winding_names[0]][:config.max_turns]
        num_sim_turns = len(primary_turns)
    else:
        primary_turns = [t for t in turns_info if 'primary' in t.winding.lower() or
                         'winding 1' in t.winding.lower() or
                         t.winding == winding_names[0]]
        num_sim_turns = len(primary_turns)

    results['simulated_turns'] = num_sim_turns
    print(f"Total turns: {num_turns}, Simulating: {num_sim_turns} primary turns")

    if num_sim_turns == 0:
        results['error'] = "No primary turns found"
        return results

    # Get core permeability
    core_data = get_core_data(magnetic_data)
    func_desc = core_data.get('functionalDescription', {})
    mat = func_desc.get('material', 'N87')
    if isinstance(mat, dict):
        mat = mat.get('name', 'N87')
    core_permeability = get_material_permeability(mat)
    results['core_permeability'] = core_permeability
    results['material_name'] = mat

    # Analytical inductance
    L_analytical = calculate_analytical_inductance(core_data, num_sim_turns)
    if L_analytical:
        results['analytical_inductance_H'] = L_analytical
        print(f"Analytical inductance: {L_analytical*1e6:.2f} uH")

    # Build geometry
    print("Building geometry...")
    try:
        step_file, core_type_detected = build_geometry(
            magnetic_data, output_path,
            max_turns=config.max_turns,
            include_bobbin=config.include_bobbin,
        )
        results['step_file'] = step_file
        print(f"Geometry: {step_file}")
    except Exception as e:
        results['error'] = f"Geometry build failed: {e}"
        print(f"ERROR: {e}")
        return results

    # Mesh — gmsh in subprocess, Netgen fallback
    print("Creating mesh...")
    import multiprocessing as _mp
    import pickle as _pickle

    mesh_dir = None
    body_numbers = None
    turn_bodies = None

    # Get bobbin params for mesh classification
    bobbin_params = None
    coil = magnetic_data.get('coil', {})
    bobbin = coil.get('bobbin', {})
    if bobbin:
        bd = bobbin.get('processedDescription', bobbin.get('functionalDescription', {}))
        if isinstance(bd, list) and bd:
            bd = bd[0]
        if isinstance(bd, dict):
            cw = bd.get('columnWidth') or bd.get('wallThickness', 0)
            cd = bd.get('columnDepth') or cw
            cs = bd.get('columnShape', 'round')
            if cw:
                bobbin_params = {
                    'column_width': cw * 1000,
                    'column_depth': cd * 1000,
                    'column_shape': cs,
                }

    # For many turns (>20), use winding-region approach (annular cylinders)
    # instead of individual pipe-sweep turns. This avoids gmsh fragment failures
    # and Netgen CalcFields crashes with many discontinuous bodies.
    use_winding_regions = num_sim_turns > 20 and core_type == "concentric"
    if use_winding_regions:
        print(f"  Using winding-region approach ({num_sim_turns} turns > 20)")
        config.method = "tangential"  # CoilSolver needs per-turn bodies

    # Mesh in subprocess (segfault-safe for both gmsh and Netgen)
    def _mesh_worker(step, outdir, turns_pkl, bobbin_pkl, ctype, use_regions, result_file):
        turns = _pickle.loads(turns_pkl)
        bobbin = _pickle.loads(bobbin_pkl) if bobbin_pkl else None
        try:
            if use_regions:
                r = create_mesh_with_winding_regions(step, outdir, turns, core_type=ctype)
            else:
                r = create_mesh_with_turns(step, outdir, turns,
                                           bobbin_params=bobbin, core_type=ctype)
            with open(result_file, 'wb') as f:
                _pickle.dump(r, f)
        except Exception as e:
            with open(result_file, 'wb') as f:
                _pickle.dump(e, f)

    try:
        result_file = os.path.join(output_path, "_mesh_result.pkl")
        p = _mp.Process(target=_mesh_worker, args=(
            step_file, output_path,
            _pickle.dumps(primary_turns),
            _pickle.dumps(bobbin_params) if bobbin_params else None,
            core_type, use_winding_regions, result_file,
        ))
        p.start()
        p.join(timeout=config.timeout)
        if p.is_alive():
            p.kill()
            p.join()
            raise RuntimeError("meshing timed out")
        if p.exitcode != 0:
            raise RuntimeError(f"meshing crashed (exit {p.exitcode})")
        with open(result_file, 'rb') as f:
            mesh_result = _pickle.load(f)
        os.remove(result_file)
        if isinstance(mesh_result, Exception):
            raise mesh_result
        mesh_dir, body_numbers, turn_bodies = mesh_result
        print(f"Mesh: {mesh_dir}, {len(turn_bodies)} turn/region bodies")
    except Exception as e:
        print(f"Primary meshing failed: {e}, trying Netgen...")

        # Netgen in subprocess (OCC state isolation)
        def _netgen_worker(step, outdir, turns_pkl, ctype, result_file):
            turns = _pickle.loads(turns_pkl)
            try:
                r = create_mesh_with_netgen(step, outdir, turns, core_type=ctype)
                with open(result_file, 'wb') as f:
                    _pickle.dump(r, f)
            except Exception as ex:
                with open(result_file, 'wb') as f:
                    _pickle.dump(ex, f)

        try:
            result_file = os.path.join(output_path, "_mesh_result.pkl")
            p = _mp.Process(target=_netgen_worker, args=(
                step_file, output_path,
                _pickle.dumps(primary_turns),
                core_type, result_file,
            ))
            p.start()
            p.join(timeout=config.timeout)
            if p.is_alive():
                p.kill()
                p.join()
                raise RuntimeError("Netgen timed out")
            if p.exitcode != 0:
                raise RuntimeError(f"Netgen crashed (exit {p.exitcode})")
            with open(result_file, 'rb') as f:
                mesh_result = _pickle.load(f)
            os.remove(result_file)
            if isinstance(mesh_result, Exception):
                raise mesh_result
            mesh_dir, body_numbers, turn_bodies = mesh_result
            print(f"Mesh (Netgen): {mesh_dir}, {len(turn_bodies)} turn bodies")
        except Exception as e2:
            results['error'] = f"Meshing failed: {e2}"
            print(f"ERROR: {e2}")
            return results

    results['mesh_dir'] = mesh_dir
    results['body_numbers'] = body_numbers

    # Generate SIF
    print("Generating SIF...")
    try:
        if config.method == "coilsolver":
            sif_path = generate_sif_with_coil_solver(
                output_path, body_numbers, turn_bodies,
                core_permeability=core_permeability,
                total_current=config.total_current,
                num_turns=num_sim_turns,
                core_type=core_type,
            )
        else:
            sif_path = generate_sif_with_tangential_current(
                output_path, body_numbers, turn_bodies,
                core_permeability=core_permeability,
                total_current=config.total_current,
                core_type=core_type,
            )
        results['sif_path'] = sif_path
    except Exception as e:
        results['error'] = f"SIF generation failed: {e}"
        return results

    # Patch SIF: ensure all body IDs 1..max are defined
    elements_path = os.path.join(mesh_dir, "mesh.elements")
    if os.path.exists(elements_path):
        mesh_bodies = set()
        with open(elements_path) as ef:
            for line in ef:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mesh_bodies.add(int(parts[1]))
        max_body = max(mesh_bodies) if mesh_bodies else 0
        with open(sif_path) as sf:
            sif_content = sf.read()
        missing = [b for b in range(1, max_body + 1) if f"Body {b}\n" not in sif_content]
        if missing:
            with open(sif_path, 'a') as sf:
                for b in missing:
                    sf.write(f"\nBody {b}\n  Name = \"Unclassified_{b}\"\n  Equation = 1\n  Material = 3\nEnd\n")
            print(f"Added {len(missing)} unclassified bodies as air")

    # Run Elmer
    print("Running Elmer...")
    success, energy, output = run_elmer(output_path, timeout=config.timeout)
    results['elmer_success'] = success
    results['electromagnetic_energy_J'] = energy

    # CoilSolver fallback to tangential
    if not success and config.method == "coilsolver":
        print("CoilSolver failed, retrying with tangential...")
        try:
            sif_path = generate_sif_with_tangential_current(
                output_path, body_numbers, turn_bodies,
                core_permeability=core_permeability,
                total_current=config.total_current,
                core_type=core_type,
            )
            # Re-patch missing bodies
            with open(sif_path) as sf:
                sif_content = sf.read()
            missing = [b for b in range(1, max_body + 1) if f"Body {b}\n" not in sif_content]
            if missing:
                with open(sif_path, 'a') as sf:
                    for b in missing:
                        sf.write(f"\nBody {b}\n  Name = \"Unclassified_{b}\"\n  Equation = 1\n  Material = 3\nEnd\n")
            success, energy, output = run_elmer(output_path, timeout=config.timeout)
            results['elmer_success'] = success
            results['electromagnetic_energy_J'] = energy
        except Exception:
            pass

    if not success:
        results['error'] = "Elmer simulation failed"
        return results

    # Calculate inductance
    L_elmer = calculate_inductance_from_energy(energy, config.total_current)
    results['elmer_inductance_H'] = L_elmer
    results['elmer_inductance_uH'] = L_elmer * 1e6

    if L_analytical:
        error_pct = abs(L_elmer - L_analytical) / L_analytical * 100
        results['error_percent'] = error_pct
        results['success'] = error_pct < 25.0
        print(f"Inductance: analytical={L_analytical*1e6:.2f} uH, "
              f"FEM={L_elmer*1e6:.2f} uH, error={error_pct:.1f}%")
    else:
        results['success'] = True
        print(f"Inductance: FEM={L_elmer*1e6:.2f} uH (no analytical reference)")

    return results


def build_mas_outputs(results: Dict, operating_points: List[Dict] = None) -> Optional[Any]:
    """
    Build MAS.Outputs object from simulation results.

    Returns MAS.Outputs or None if MAS_models not available.
    """
    if not HAS_MAS:
        return None

    L_H = results.get('elmer_inductance_H', 0)

    # Magnetizing inductance
    mag_ind = MAS.MagnetizingInductanceOutput(
        magnetizingInductance=MAS.DimensionWithTolerance(nominal=L_H),
        methodUsed="Elmer FEM (CoilSolver, magnetostatic)",
        origin=MAS.ResultOrigin.simulation,
    )

    outputs = MAS.Outputs(
        magnetizingInductance=mag_ind,
    )

    return outputs


def run_mas_simulation(
    mas_file: str,
    output_path: str,
    max_turns_per_winding: Optional[int] = None,
    total_current: float = 1.0,
    method: str = "coilsolver",
) -> Dict[str, Any]:
    """
    Run complete Elmer simulation from MAS file.

    This is the main entry point for the MAS-to-Elmer workflow.
    Handles both simple format (examples/) and complete format (examples/complete/).

    Parameters
    ----------
    mas_file : str
        Path to MAS JSON file.
    output_path : str
        Directory for output files.
    max_turns_per_winding : int, optional
        Limit turns per winding. None = use all turns.
    total_current : float
        Test current in Amperes.
    method : str
        "coilsolver" or "tangential".

    Returns
    -------
    dict
        Simulation results including inductance, energy, MAS outputs.
    """
    config = SimulationConfig(
        max_turns=max_turns_per_winding,
        total_current=total_current,
        method=method,
    )

    results = run_magnetostatic_inductance(mas_file, output_path, config)

    # Build MAS outputs
    with open(mas_file) as f:
        raw_data = json.load(f)
    _, operating_points = normalize_mas_data(raw_data)

    mas_outputs = build_mas_outputs(results, operating_points)
    if mas_outputs:
        results['mas_outputs'] = mas_outputs.to_dict()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Elmer FEM simulation from MAS file")
    parser.add_argument("mas_file", help="Path to MAS JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("-t", "--turns", type=int, default=None, help="Max turns (None=all)")
    parser.add_argument("-I", "--current", type=float, default=1.0, help="Test current (A)")
    parser.add_argument("-m", "--method", choices=["tangential", "coilsolver"],
                        default="coilsolver", help="Current application method")

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.mas_file), "../output/elmer")

    results = run_mas_simulation(
        args.mas_file, args.output,
        max_turns_per_winding=args.turns,
        total_current=args.current,
        method=args.method,
    )

    print("\n" + "=" * 60)
    print(json.dumps(results, indent=2, default=str))
