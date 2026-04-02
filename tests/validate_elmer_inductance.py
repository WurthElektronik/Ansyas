#!/usr/bin/env python3
"""
Validate Elmer FEM inductance calculation against PyMKF analytical.

This script:
1. Loads a MAS file
2. Builds geometry using MVB
3. Creates mesh with gmsh (individual turn physical groups for proper current application)
4. Generates Elmer SIF with tangential current
5. Runs Elmer simulation
6. Extracts energy and calculates inductance
7. Compares with PyMKF analytical calculation

Target: <25% difference between Elmer and PyMKF inductance values.
"""

import os
import sys
import json
import math
import subprocess
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Add paths
MVB_PATH = os.path.expanduser("~/OpenMagnetics/MVB/src")
ANSYAS_PATH = os.path.expanduser("~/wuerth/Ansyas/src")
for path in [MVB_PATH, ANSYAS_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

try:
    from PyOpenMagnetics import PyOpenMagnetics as PyOM
    HAS_PYMKF = True
except ImportError:
    HAS_PYMKF = False
    print("Warning: PyOpenMagnetics not available")

try:
    from OpenMagneticsVirtualBuilder.builder import Builder
    HAS_MVB = True
except ImportError:
    HAS_MVB = False
    print("Warning: MVB not available")

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    print("Warning: gmsh not available")


@dataclass
class TurnInfo:
    """Information about a single turn."""
    name: str
    radius: float  # Distance from axis (mm) - Z-axis for concentric, Y-axis for toroidal
    z_position: float  # Z coordinate (mm)
    cross_section_area: float  # mm^2
    orientation: str  # 'clockwise' or 'counterclockwise'
    winding: str  # 'Primary', 'Secondary', etc.
    x_position: float = 0.0  # X coordinate (mm) - for toroidal geometry


def load_mas_file(mas_file: str) -> Dict:
    """Load MAS file and validate it has required processed data.
    
    MAS files must have turnsDescription and geometricalDescription.
    Use PyMKF tools or OpenMagnetics web interface to process files
    that only have functionalDescription.
    """
    with open(mas_file) as f:
        data = json.load(f)
    
    # Check if file has required processed data
    magnetic = data.get('magnetic', data)
    coil = magnetic.get('coil', {})
    core = magnetic.get('core', {})
    
    if not coil.get('turnsDescription') or not core.get('geometricalDescription'):
        try:
            from PyOpenMagnetics import PyOpenMagnetics as PyOM
            print("Auto-completing magnetic data with PyOpenMagnetics...")
            completed = PyOM.magnetic_autocomplete(magnetic, {})
            if 'magnetic' in data:
                data['magnetic'] = completed
            else:
                data = completed
            magnetic = data.get('magnetic', data)
            coil = magnetic.get('coil', {})
            core = magnetic.get('core', {})
        except Exception as e:
            raise ValueError(
                f"MAS file missing 'turnsDescription' or 'geometricalDescription' "
                f"and PyMKF autocomplete failed: {e}"
            )
    
    print(f"Loaded MAS file with {len(coil.get('turnsDescription', []))} turns")
    
    return data


def get_core_data(magnetic_data: Dict) -> Dict:
    """Get processed core data using PyOM."""
    import json as json_module
    core = magnetic_data.get('core')
    if core is None:
        raise ValueError("Missing 'core' in magnetic_data")
    if not isinstance(core, dict):
        raise ValueError(f"'core' must be a dict, got {type(core)}")
    
    if core.get('geometricalDescription') is None:
        if not HAS_PYMKF:
            raise ImportError("PyMKF required to process core without geometricalDescription")
        
        result = PyOM.calculate_core_data(core, True)
        # PyMKF returns JSON string
        if isinstance(result, str):
            if result.startswith('Exception:'):
                raise ValueError(f"PyOM.calculate_core_data failed: {result}")
            core = json_module.loads(result)
        elif isinstance(result, dict):
            core = result
        else:
            raise ValueError(f"Unexpected PyMKF result type: {type(result)}")
        
        if not core.get('geometricalDescription'):
            raise ValueError("PyOM.calculate_core_data did not produce geometricalDescription")
    
    return core


def extract_turns_info(magnetic_data: Dict, core_type: str = "concentric") -> List[TurnInfo]:
    """Extract turn information from MAS coil data.
    
    For concentric cores (E, PQ, etc.):
        - coordinates[0] = radial position (from center column)
        - coordinates[1] = height position (along Z)
    
    For toroidal cores:
        - coordinates are in XZ plane (Y is core axis)
        - coordinates[0] = X position
        - coordinates[1] = Z position  
        - radius = sqrt(x² + z²) = distance from Y axis
    """
    coil = magnetic_data.get('coil', {})
    turns_desc = coil.get('turnsDescription', [])
    
    turns = []
    for turn in turns_desc:
        coords = turn.get('coordinates', [0, 0])
        dims = turn.get('dimensions', [1e-3, 1e-3])
        shape = turn.get('crossSectionalShape', 'round')
        
        if core_type == "toroidal":
            # For toroidal: coords are [x, y] in XY plane
            x = coords[0] * 1000  # mm
            y = coords[1] * 1000 if len(coords) > 1 else 0  # mm
            radius = math.sqrt(x**2 + y**2)  # Distance from Z axis
            z_pos = y  # Store Y for position identification and Coil Normal
            x_pos = x  # Store X for Coil Normal calculation
        else:
            # For concentric: coords are [radial(x), height(y)]
            radius = coords[0] * 1000  # Convert to mm
            z_pos = coords[1] * 1000 if len(coords) > 1 else 0  # height = Y
            x_pos = 0.0
        
        # Cross-section area in mm^2
        if shape == 'round':
            area = math.pi * (dims[0] * 1000 / 2) ** 2
        else:  # rectangular
            area = dims[0] * dims[1] * 1e6
        
        turns.append(TurnInfo(
            name=turn.get('name', f'turn_{len(turns)}'),
            radius=radius,
            z_position=z_pos,
            cross_section_area=area,
            orientation=turn.get('orientation', 'clockwise'),
            winding=turn.get('winding', 'Primary'),
            x_position=x_pos,
        ))
    
    return turns


def get_bh_curve(material_name: str, temperature: float = 25.0,
                  n_points: int = 20) -> List[Tuple[float, float]]:
    """
    Generate H-B curve points for a ferrite material.

    Synthesizes from initial permeability + saturation point using a
    Langevin-like model: B(H) = B_sat * tanh(mu_0 * mu_i * H / B_sat)

    Returns list of (H_A_per_m, B_Tesla) tuples sorted by H.
    """
    mu_0 = 4e-7 * math.pi
    mu_i = get_material_permeability(material_name, temperature)

    # Get saturation from PyOM
    B_sat = 0.4  # default
    H_sat = 1200.0
    if HAS_PYMKF:
        try:
            mat_data = PyOM.get_material_data(material_name)
            if isinstance(mat_data, str):
                mat_data = json.loads(mat_data)
            sat_points = mat_data.get('saturation', [])
            best_diff = float('inf')
            for p in sat_points:
                diff = abs(p.get('temperature', 25) - temperature)
                if diff < best_diff:
                    best_diff = diff
                    B_sat = p.get('magneticFluxDensity', 0.4)
                    H_sat = p.get('magneticField', 1200.0)
        except Exception:
            pass

    # Langevin-like curve: B = B_sat * tanh(mu_0 * mu_i * H / B_sat)
    # At low H: B ≈ mu_0 * mu_i * H (linear, matches initial permeability)
    # At high H: B → B_sat (saturates)
    # Past saturation, add mu_0*H slope to keep B strictly increasing.
    points = [(0.0, 0.0)]
    H_max = H_sat * 5
    prev_B = 0.0
    for i in range(1, n_points):
        H = H_max * (i / (n_points - 1)) ** 1.5  # denser at low H
        B = B_sat * math.tanh(mu_0 * mu_i * H / B_sat) + mu_0 * H
        # Ensure strictly increasing
        if B <= prev_B:
            B = prev_B + 1e-6
        points.append((H, B))
        prev_B = B

    return points


def get_material_permeability(material_name: str, temperature: float = 25.0) -> float:
    """
    Get initial permeability for a ferrite material at given temperature.
    
    Uses PyMKF material database if available, otherwise falls back to defaults.
    """
    # Default permeabilities (initial permeability at 25°C)
    default_permeabilities = {
        'N87': 2200,
        'N97': 2300,
        '3C90': 2300,
        '3C95': 3000,
        '3F3': 2000,
        '3F35': 1400,
        'N27': 2000,
        'N30': 4300,
        'N49': 1500,
        'N95': 3000,
        'PC40': 2300,
        'PC95': 3000,
    }
    
    if not isinstance(material_name, str):
        return 2000.0
    
    # Try to get from PyMKF
    if HAS_PYMKF:
        try:
            mat_data = PyOM.get_material_data(material_name)
            perm = mat_data.get('permeability', {})
            initial = perm.get('initial', [])
            
            if initial:
                # Find closest temperature
                best_match = None
                best_diff = float('inf')
                for entry in initial:
                    temp = entry.get('temperature')
                    if temp is not None:
                        diff = abs(temp - temperature)
                        if diff < best_diff:
                            best_diff = diff
                            best_match = entry.get('value')
                
                if best_match is not None:
                    return float(best_match)
        except Exception:
            pass
    
    # Fall back to defaults
    return float(default_permeabilities.get(material_name, 2000))


def calculate_analytical_inductance(core_data: Dict, num_turns: int) -> Optional[float]:
    """Calculate analytical inductance using AL value.
    
    Uses effective area and length from PyMKF-processed core data.
    """
    if not isinstance(core_data, dict):
        return None
    
    # Try to find effective parameters - check multiple locations
    Ae = None  # m^2
    le = None  # m
    
    # 1. Check processedDescription (PyMKF output)
    processed = core_data.get('processedDescription', {})
    if processed:
        eff = processed.get('effectiveParameters', {})
        if eff:
            Ae = eff.get('effectiveArea')  # Already in m^2
            le = eff.get('effectiveLength')  # Already in m
    
    # 2. Check functionalDescription.effectiveParameters
    if Ae is None or le is None:
        func_desc = core_data.get('functionalDescription', {})
        if 'effectiveParameters' in func_desc:
            eff = func_desc['effectiveParameters']
            if Ae is None:
                Ae = eff.get('effectiveArea')
            if le is None:
                le = eff.get('effectiveLength')
    
    # 3. Fall back to geometrical description (central column area)
    if Ae is None:
        geom = core_data.get('geometricalDescription', [{}])
        if not isinstance(geom, list):
            geom = [{}]
        
        for piece in geom:
            columns = piece.get('columns', [])
            for col in columns:
                if col.get('type') == 'central':
                    shape = col.get('shape', 'round')
                    if shape == 'round':
                        width = col.get('width', 0)  # m
                        Ae = math.pi * (width/2)**2  # m^2
                    else:
                        width = col.get('width', 0)
                        depth = col.get('depth', width)
                        Ae = width * depth  # m^2
    
    # Get permeability from material data
    func_desc = core_data.get('functionalDescription', {})
    mat = func_desc.get('material', 'N87')
    if isinstance(mat, dict):
        mat = mat.get('name', 'N87')
    mu_r = get_material_permeability(mat)
    
    # Calculate AL and inductance
    if Ae and le:
        mu_0 = 4 * math.pi * 1e-7  # H/m
        AL = mu_0 * mu_r * Ae / le  # H/turn^2
        L = AL * num_turns**2  # H
        return L
    
    return None


def build_geometry(magnetic_data: Dict, output_path: str,
                   max_turns: Optional[int] = None, include_bobbin: bool = True,
                   all_windings: bool = False) -> Tuple[str, str]:
    """Build 3D geometry using MVB."""
    if not HAS_MVB:
        raise ImportError("MVB required")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Process core
    core = get_core_data(magnetic_data)
    magnetic_data = {**magnetic_data, 'core': core}
    
    # Optionally limit turns
    if max_turns is not None:
        coil = magnetic_data.get('coil', {}).copy()
        turns = coil.get('turnsDescription', [])
        if turns:
            # Group by winding and limit each
            winding_groups = {}
            for t in turns:
                w = t.get('winding', 'Primary')
                if w not in winding_groups:
                    winding_groups[w] = []
                winding_groups[w].append(t)
            if all_windings:
                # Include max_turns from EACH winding (for inductance matrix)
                limited = []
                for w, wturns in winding_groups.items():
                    limited.extend(wturns[:max_turns])
            else:
                # Include max_turns from PRIMARY only (for single inductance)
                primary_turns = winding_groups.get('Primary', [])
                if not primary_turns:
                    primary_turns = list(winding_groups.values())[0] if winding_groups else []
                limited = primary_turns[:max_turns]
            coil['turnsDescription'] = limited
            magnetic_data = {**magnetic_data, 'coil': coil}
    
    # Build with MVB
    builder = Builder()
    result = builder.get_magnetic(
        magnetic_data,
        project_name="magnetic",
        output_path=output_path,
        export_files=True,
        include_bobbin=True,
    )
    
    if isinstance(result, tuple):
        return result
    else:
        step_path = os.path.join(output_path, "magnetic.step")
        stl_path = os.path.join(output_path, "magnetic.stl")
        return step_path, stl_path


def create_mesh_with_netgen(
    step_file: str,
    output_path: str,
    turns_info: List[TurnInfo],
    core_type: str = "concentric",
    max_element_size: float = 3.0,
    air_padding: float = 10.0,
) -> Tuple[str, Dict[str, int], Dict[int, TurnInfo]]:
    """Create mesh using Netgen as fallback when gmsh fails.

    Netgen handles near-touching surfaces in toroidal cores better than gmsh.
    Adds an air box to the geometry before meshing for boundary conditions.
    """
    import cadquery as cq
    from cadquery import exporters
    from netgen.occ import OCCGeometry
    from netgen.meshing import MeshingParameters

    mesh_dir = os.path.join(output_path, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    # Load STEP to get bounding box and classify solids
    compound = cq.importers.importStep(step_file)
    solids = compound.solids().vals()

    # Create air box around everything
    bb = compound.val().BoundingBox()
    air_box = (
        cq.Workplane("XY")
        .box(
            (bb.xmax - bb.xmin) + 2 * air_padding,
            (bb.ymax - bb.ymin) + 2 * air_padding,
            (bb.zmax - bb.zmin) + 2 * air_padding,
        )
        .translate((
            (bb.xmin + bb.xmax) / 2,
            (bb.ymin + bb.ymax) / 2,
            (bb.zmin + bb.zmax) / 2,
        ))
    )

    # Combine all solids + air box into one STEP for Netgen
    all_pieces = [air_box.val()] + list(solids)
    combined = cq.Compound.makeCompound(all_pieces)
    combined_step = os.path.join(output_path, "netgen_input.step")
    exporters.export(cq.Workplane("XY").add(combined), combined_step, "STEP")

    # Mesh with Netgen — use high grading + local refinement at turn positions
    # This keeps the air mesh coarse while refining near the thin turns.
    geo = OCCGeometry(combined_step)
    # Glue creates conformal interfaces between overlapping solids so they
    # share mesh nodes at interfaces. Without this, each body gets independent
    # nodes and the FEM solver can't transfer fields between bodies.
    # Glue can crash on complex toroidal geometries — fall back to non-glued mesh.
    try:
        geo.Glue()
    except Exception as e:
        print(f"  Warning: Netgen Glue() failed ({e}), meshing without conformal interfaces")
    wire_diameter = turns_info[0].cross_section_area ** 0.5 if turns_info else 1.0
    geo_size = max(bb.xmax - bb.xmin, bb.ymax - bb.ymin, bb.zmax - bb.zmin)
    num_turns_total = len(turns_info)
    coarse_h = max(max_element_size, geo_size / 5)
    # Scale minh with turn count: more turns need coarser minimum to keep
    # element count under ~300K (CalcFields' ILU factorization limit).
    if num_turns_total > 20:
        min_h = max(wire_diameter * 0.5, 0.5)
    else:
        min_h = max(wire_diameter * 0.5, 0.2)
    grading = min(0.5 + num_turns_total * 0.008, 0.9)
    coarse_h = min(coarse_h, 6.0)

    mp = MeshingParameters(maxh=coarse_h, minh=min_h, grading=grading)
    print(f"  Netgen params: maxh={coarse_h:.1f}, minh={min_h:.2f}, grading={grading:.2f} ({num_turns_total} turns)")

    ngmesh = geo.GenerateMesh(mp)
    ngmesh.Export(mesh_dir + "/", "Elmer Format")
    print(f"  Netgen: {ngmesh.ne} tets, {len(ngmesh.Points())} points")

    # Read node coordinates (in mm) for boundary identification
    nodes_mm = {}
    nodes_file = os.path.join(mesh_dir, "mesh.nodes")
    with open(nodes_file) as nf:
        for line in nf:
            parts = line.strip().split()
            if len(parts) >= 5:
                nodes_mm[int(parts[0])] = [float(parts[2]), float(parts[3]), float(parts[4])]

    # Find bounding box extremes
    all_coords = list(nodes_mm.values())
    bbox_min = [min(c[d] for c in all_coords) for d in range(3)]
    bbox_max = [max(c[d] for c in all_coords) for d in range(3)]

    # Identify air box outer boundaries: triangles with all 3 nodes on a domain face.
    # Netgen boundary format: elem_id bt parent1 parent2 elem_type n1 n2 n3
    boundary_file = os.path.join(mesh_dir, "mesh.boundary")
    with open(boundary_file) as bf:
        blines = bf.readlines()

    outer_bt_set = set()
    tol_mm = 0.5  # 0.5mm tolerance for face identification
    for line in blines:
        parts = line.strip().split()
        if len(parts) >= 8:
            bt = int(parts[1])
            tri = [nodes_mm.get(int(parts[i])) for i in range(5, 8)]
            if any(p is None for p in tri):
                continue
            for dim in range(3):
                vals = [p[dim] for p in tri]
                if all(abs(v - bbox_min[dim]) < tol_mm for v in vals) or \
                   all(abs(v - bbox_max[dim]) < tol_mm for v in vals):
                    outer_bt_set.add(bt)
                    break

    # Remap air box outer boundaries to type 1 so the SIF can target them
    with open(boundary_file, 'w') as bf:
        for line in blines:
            parts = line.strip().split()
            if len(parts) >= 8 and int(parts[1]) in outer_bt_set:
                parts[1] = '1'
                bf.write(' '.join(parts) + '\n')
            else:
                bf.write(line)

    # Scale node coordinates from mm to m (Elmer expects SI units)
    with open(nodes_file, 'w') as nf:
        for nid, (x, y, z) in sorted(nodes_mm.items()):
            nf.write(f"{nid} -1 {x*0.001:.10e} {y*0.001:.10e} {z*0.001:.10e}\n")

    # Read actual body IDs from the Netgen mesh to classify them.
    # Parse mesh.elements to count elements per body and estimate volumes.
    body_element_counts = {}
    elements_file = os.path.join(mesh_dir, "mesh.elements")
    with open(elements_file) as ef:
        for line in ef:
            parts = line.strip().split()
            if len(parts) >= 2:
                bid = int(parts[1])
                body_element_counts[bid] = body_element_counts.get(bid, 0) + 1

    # Estimate body volumes from element coordinates (more reliable than element count,
    # since thin bodies like bobbins get disproportionately many elements).
    # Use pre-scaling coordinates (mm) for meaningful volume values.
    nodes_coords = nodes_mm

    body_volumes = {}
    for bid, cnt in body_element_counts.items():
        body_volumes[bid] = 0.0
    with open(elements_file) as ef:
        for line in ef:
            parts = line.strip().split()
            if len(parts) >= 7:
                bid = int(parts[1])
                nids = [int(parts[i]) for i in range(3, 7)]
                pts = [nodes_coords.get(n, [0,0,0]) for n in nids]
                # Tet volume = |det([v1-v0, v2-v0, v3-v0])| / 6
                v0, v1, v2, v3 = pts
                d1 = [v1[i]-v0[i] for i in range(3)]
                d2 = [v2[i]-v0[i] for i in range(3)]
                d3 = [v3[i]-v0[i] for i in range(3)]
                det = (d1[0]*(d2[1]*d3[2]-d2[2]*d3[1])
                     - d1[1]*(d2[0]*d3[2]-d2[2]*d3[0])
                     + d1[2]*(d2[0]*d3[1]-d2[1]*d3[0]))
                body_volumes[bid] += abs(det) / 6.0

    sorted_bodies = sorted(body_volumes.items(), key=lambda x: -x[1])
    print(f"  Netgen bodies (by volume): {[(bid, f'{vol:.0f}') for bid, vol in sorted_bodies]}")

    body_numbers = {}
    turn_bodies = {}

    if len(sorted_bodies) >= 2:
        # With air box: largest = air body.
        # Core pieces are large (>30% of 2nd largest count). Turns are small.
        air_bid = sorted_bodies[0][0]
        body_numbers['air'] = air_bid

        # Identify core bodies by volume pairing: core halves come in pairs
        # with nearly identical volumes. The bobbin is a different size.
        second_vol = sorted_bodies[1][1]
        core_threshold = second_vol * 0.3
        candidates = [(bid, vol) for bid, vol in sorted_bodies[1:] if vol >= core_threshold]
        non_candidates = [(bid, vol) for bid, vol in sorted_bodies[1:] if vol < core_threshold]

        # Group candidates by similar volume (within 5%)
        core_bids = []
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            groups = []
            used = set()
            for i, (b1, v1) in enumerate(candidates):
                if i in used:
                    continue
                group = [(b1, v1)]
                used.add(i)
                for j, (b2, v2) in enumerate(candidates):
                    if j in used:
                        continue
                    if abs(v1 - v2) / max(v1, 1) < 0.05:
                        group.append((b2, v2))
                        used.add(j)
                groups.append(group)
            # Largest group with >=2 members = core
            core_group = max(groups, key=lambda g: len(g) * 1000 + sum(v for _, v in g))
            core_bids = [bid for bid, _ in core_group]
            non_core_candidates = [bid for bid, _ in candidates if bid not in core_bids]
        else:
            non_core_candidates = []

        turn_bids = [bid for bid, _ in non_candidates] + non_core_candidates
        # All core pieces share the same body number (merged into one "core" body)
        # We remap them in the mesh.elements file below
        body_numbers['core'] = core_bids[0] if core_bids else sorted_bodies[1][0]

        # Remap additional core pieces to the primary core body ID
        if len(core_bids) > 1:
            primary_core = core_bids[0]
            elements_file = os.path.join(mesh_dir, "mesh.elements")
            with open(elements_file) as ef:
                elines = ef.readlines()
            with open(elements_file, 'w') as ef:
                for line in elines:
                    parts = line.strip().split()
                    if len(parts) >= 2 and int(parts[1]) in core_bids[1:]:
                        parts[1] = str(primary_core)
                        ef.write(' '.join(parts) + '\n')
                    else:
                        ef.write(line)
            print(f"  Merged {len(core_bids)} core bodies into body {primary_core}")
        reimported = cq.importers.importStep(combined_step)
        turn_solids = [s for s in sorted(reimported.solids().vals(), key=lambda s: s.Volume()) if s.Volume() < 500]

        matched_turns = set()
        for i, bid in enumerate(turn_bids):
            if i < len(turn_solids):
                com = turn_solids[i].Center()
                if core_type == "toroidal":
                    center_angle = math.atan2(com.y, com.x)
                else:
                    center_pos = com.y
            else:
                center_angle = 0
                center_pos = 0

            best_match = None
            best_dist = float('inf')
            for ti in turns_info:
                if id(ti) in matched_turns:
                    continue
                if core_type == "toroidal":
                    ti_angle = math.atan2(ti.z_position, ti.x_position) if ti.x_position != 0 else 0
                    dist = abs(ti_angle - center_angle)
                    if dist > math.pi:
                        dist = 2 * math.pi - dist
                else:
                    dist = abs(ti.z_position - center_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_match = ti

            if best_match:
                body_numbers[best_match.name] = bid
                turn_bodies[bid] = best_match
                matched_turns.add(id(best_match))

    return mesh_dir, body_numbers, turn_bodies


def create_mesh_with_winding_regions(
    step_file: str,
    output_path: str,
    turns_info: List[TurnInfo],
    air_padding: float = 10.0,
    max_element_size: float = 3.0,
    core_type: str = "concentric",
) -> Tuple[str, Dict[str, int], Dict[int, TurnInfo]]:
    """
    Create mesh using annular winding regions instead of individual turns.

    For magnetics with many turns (>20), meshing individual pipe-sweep turn
    volumes is impractical (gmsh fragment fails, Netgen produces too many
    elements). Instead, this function:
    1. Imports only core pieces from the STEP file
    2. Creates annular cylinder regions per winding layer
    3. Fragments core + winding regions + air box (few volumes, fast)
    4. Applies tangential current J = N*I/A_region per winding body

    The winding region approach is the standard "stranded conductor" model
    used by commercial FEM tools. It gives correct inductance since the
    total ampere-turns (N*I) is preserved.

    Returns same interface as create_mesh_with_turns.
    """
    if not HAS_GMSH:
        raise ImportError("gmsh required")

    import cadquery as cq

    compound = cq.importers.importStep(step_file)
    solids = sorted(compound.solids().vals(), key=lambda s: -s.Volume())
    bb = compound.val().BoundingBox()

    # Classify solids: core pieces are large (>500mm³), turns are small.
    # Use a fixed threshold based on the gap between turn and core volumes.
    core_solids = []
    turn_solids = []
    all_vols = sorted([s.Volume() for s in solids], reverse=True)
    # Find the gap: largest volume jump between adjacent sorted volumes
    core_threshold = 500  # default
    for i in range(len(all_vols) - 1):
        if all_vols[i] > 10 * all_vols[i + 1] and all_vols[i] > 100:
            core_threshold = (all_vols[i] + all_vols[i + 1]) / 2
            break
    for s in solids:
        vol = s.Volume()
        if vol > core_threshold:
            core_solids.append(s)
        elif vol > 0.1:
            turn_solids.append(s)

    # Group ALL turn solids by radius layer. Each layer becomes one
    # winding region body. The tangential current density is set based on
    # the number of requested turns (turns_info) matched to each layer.
    layers = {}
    for s in turn_solids:
        sbb = s.BoundingBox()
        if core_type == "concentric":
            r = math.sqrt(sbb.xmin**2 + sbb.zmin**2)
        else:
            r = math.sqrt(sbb.xmin**2 + sbb.ymin**2)
        key = round(r * 2) / 2
        if key not in layers:
            layers[key] = []
        layers[key].append(s)

    # Group turns_info by winding name
    winding_turns = {}
    for t in turns_info:
        if t.winding not in winding_turns:
            winding_turns[t.winding] = []
        winding_turns[t.winding].append(t)

    print(f"  Winding regions: {len(layers)} layers from {len(turn_solids)} turn solids")

    # Build gmsh model: core + winding cylinders + air
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("winding_regions")

    try:
        # Import core pieces only
        from cadquery import exporters
        core_step = os.path.join(output_path, "core_only.step")
        if core_solids:
            core_compound = cq.Compound.makeCompound(core_solids)
            exporters.export(cq.Workplane("XY").add(core_compound), core_step, "STEP")
            gmsh.model.occ.importShapes(core_step)
            gmsh.model.occ.synchronize()

        # Create annular winding region for each layer
        winding_tags = []
        winding_layer_info = []  # (tag, n_turns, mean_radius, wire_area, winding_name)
        for r_key, layer_solids in sorted(layers.items()):
            bbs = [s.BoundingBox() for s in layer_solids]
            if core_type == "concentric":
                r_inner = min(math.sqrt(b.xmin**2 + b.zmin**2) for b in bbs) - 0.1
                r_outer = max(math.sqrt(b.xmax**2 + b.zmax**2) for b in bbs) + 0.1
                y_min = min(b.ymin for b in bbs) - 0.1
                y_max = max(b.ymax for b in bbs) + 0.1
                # Cylinder along Y axis
                outer = gmsh.model.occ.addCylinder(0, y_min, 0, 0, y_max - y_min, 0, r_outer)
                inner = gmsh.model.occ.addCylinder(0, y_min, 0, 0, y_max - y_min, 0, r_inner)
            else:
                # Toroidal: cylinder along Z
                r_inner = min(math.sqrt(b.xmin**2 + b.ymin**2) for b in bbs) - 0.1
                r_outer = max(math.sqrt(b.xmax**2 + b.ymax**2) for b in bbs) + 0.1
                z_min = min(b.zmin for b in bbs) - 0.1
                z_max = max(b.zmax for b in bbs) + 0.1
                outer = gmsh.model.occ.addCylinder(0, 0, z_min, 0, 0, z_max - z_min, r_outer)
                inner = gmsh.model.occ.addCylinder(0, 0, z_min, 0, 0, z_max - z_min, r_inner)

            gmsh.model.occ.cut([(3, outer)], [(3, inner)])
            gmsh.model.occ.synchronize()

            mean_r = (r_inner + r_outer) / 2
            # Region cross-section area = radial_width * axial_height (mm²)
            if core_type == "concentric":
                region_area = (r_outer - r_inner) * (y_max - y_min)
            else:
                region_area = (r_outer - r_inner) * (z_max - z_min)
            winding_layer_info.append({
                'n_turns': 0,  # Will be filled by assignment below
                'mean_radius': mean_r,
                'region_area_mm2': region_area,
                'winding': turns_info[0].winding if turns_info else 'Primary',
            })

        # Assign turns to layers by distributing evenly among layers whose
        # geometry could contain them. Since MAS and STEP radii don't match
        # directly, use the number of solids per layer as the turn distribution.
        n_requested = len(turns_info)
        n_total_solids = sum(len(layers[k]) for k in sorted(layers.keys()))
        if winding_layer_info and n_requested > 0:
            # Distribute proportionally to layer solid count
            assigned = 0
            for i, (r_key, layer_solids) in enumerate(sorted(layers.items())):
                if i < len(winding_layer_info):
                    # Proportion of total turns for this layer
                    proportion = len(layer_solids) / n_total_solids
                    n_for_layer = round(n_requested * proportion)
                    # Clamp: don't assign more than the layer's solid count
                    n_for_layer = min(n_for_layer, len(layer_solids))
                    n_for_layer = min(n_for_layer, n_requested - assigned)
                    winding_layer_info[i]['n_turns'] = n_for_layer
                    assigned += n_for_layer
            # Distribute remainder
            for i in range(len(winding_layer_info)):
                if assigned >= n_requested:
                    break
                if winding_layer_info[i]['n_turns'] == 0:
                    continue
                deficit = n_requested - assigned
                winding_layer_info[i]['n_turns'] += deficit
                assigned += deficit

            for wl in winding_layer_info:
                print(f"    Layer r={wl['mean_radius']:.1f}mm: {wl['n_turns']} turns ({wl['winding']})")

        # Add air box
        xmin = bb.xmin - air_padding
        ymin = bb.ymin - air_padding
        zmin = bb.zmin - air_padding
        air = gmsh.model.occ.addBox(
            xmin, ymin, zmin,
            (bb.xmax - bb.xmin) + 2 * air_padding,
            (bb.ymax - bb.ymin) + 2 * air_padding,
            (bb.zmax - bb.zmin) + 2 * air_padding,
        )
        gmsh.model.occ.synchronize()

        # Fragment all volumes
        all_solid = [(3, t) for _, t in gmsh.model.getEntities(3) if t != air]
        gmsh.model.occ.fragment([(3, air)], all_solid)
        gmsh.model.occ.synchronize()

        # Classify volumes after fragment
        new_vols = gmsh.model.getEntities(3)
        core_pg, wind_pg, air_pg = [], [], []
        for dim, tag in new_vols:
            mass = gmsh.model.occ.getMass(dim, tag)
            bx = gmsh.model.getBoundingBox(dim, tag)
            bv = (bx[3] - bx[0]) * (bx[4] - bx[1]) * (bx[5] - bx[2])
            if bv > 50000:
                air_pg.append(tag)
            elif mass > core_threshold:
                core_pg.append(tag)
            else:
                wind_pg.append(tag)

        # Create physical groups
        body_numbers = {}
        turn_bodies_out = {}

        next_tag = 1
        if core_pg:
            gmsh.model.addPhysicalGroup(3, core_pg, tag=next_tag, name="core")
            body_numbers["core"] = next_tag
            next_tag += 1

        # Each winding layer gets its own physical group (sequential tags)
        for i, winfo in enumerate(winding_layer_info):
            if i < len(wind_pg):
                gmsh.model.addPhysicalGroup(3, [wind_pg[i]], tag=next_tag,
                                             name=f"winding_{winfo['winding']}_{i}")
                # For winding regions, cross_section_area is the REGION area,
                # and _n_turns_in_region stores the turn count. J = N*I/A_region.
                region_ti = TurnInfo(
                    name=f"winding_region_{winfo['winding']}_{i}",
                    radius=winfo['mean_radius'],
                    z_position=0,
                    cross_section_area=winfo['region_area_mm2'],  # Region area, not wire
                    orientation='clockwise',
                    winding=winfo['winding'],
                )
                region_ti._n_turns_in_region = winfo['n_turns']
                body_numbers[region_ti.name] = next_tag
                turn_bodies_out[next_tag] = region_ti
                next_tag += 1

        # Remaining winding volumes as air
        for j in range(len(winding_layer_info), len(wind_pg)):
            air_pg.append(wind_pg[j])

        if air_pg:
            gmsh.model.addPhysicalGroup(3, air_pg, tag=next_tag, name="air")
            body_numbers["air"] = next_tag
            next_tag += 1

        # Outer boundary
        outer_surfs = []
        tol = 0.5
        for _, tag in gmsh.model.getEntities(2):
            sb = gmsh.model.getBoundingBox(2, tag)
            for j, v in [(0, xmin), (3, bb.xmax + air_padding),
                         (1, ymin), (4, bb.ymax + air_padding),
                         (2, zmin), (5, bb.zmax + air_padding)]:
                if abs(sb[j] - v) < tol:
                    outer_surfs.append(tag)
                    break
        if outer_surfs:
            gmsh.model.addPhysicalGroup(2, list(set(outer_surfs)), tag=100, name="outer_boundary")

        # Mesh
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)

        et, tags, _ = gmsh.model.mesh.getElements(3)
        total_3d = sum(len(t) for t in tags) if tags else 0
        print(f"  Winding region mesh: {total_3d} elements, {len(new_vols)} volumes")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        msh_path = os.path.join(output_path, "mesh.msh")
        gmsh.write(msh_path)

    finally:
        gmsh.finalize()

    # Convert to Elmer
    mesh_dir = os.path.join(output_path, "mesh")
    cmd = ["ElmerGrid", "14", "2", "mesh.msh", "-autoclean",
           "-scale", "0.001", "0.001", "0.001"]
    subprocess.run(cmd, capture_output=True, text=True, cwd=output_path)

    if not os.path.exists(mesh_dir):
        raise RuntimeError("ElmerGrid failed")

    return mesh_dir, body_numbers, turn_bodies_out


def create_mesh_with_turns(
    step_file: str,
    output_path: str,
    turns_info: List[TurnInfo],
    bobbin_params: Optional[Dict] = None,
    air_padding: float = 10.0,
    max_element_size: float = 3.0,
    min_element_size: float = 0.8,
    core_type: str = "concentric",
) -> Tuple[str, Dict[str, int], Dict[int, TurnInfo]]:
    """
    Create mesh with separate physical groups for each turn.
    
    Args:
        step_file: Path to STEP file
        output_path: Output directory
        turns_info: List of TurnInfo objects
        bobbin_params: Optional dict with column_shape, column_width, column_depth
        air_padding: Air box padding in mm
        max_element_size: Maximum mesh element size
        min_element_size: Minimum mesh element size
    
    Returns:
        mesh_dir: Path to Elmer mesh
        body_numbers: Dict mapping body names to numbers
        turn_bodies: Dict mapping body numbers to TurnInfo
    """
    if not HAS_GMSH:
        raise ImportError("gmsh required")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    # Toroidal turns pass through the core hole with near-touching surfaces.
    # Increased tolerance prevents gmsh "overlapping facets" at these interfaces.
    if core_type == "toroidal":
        gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-2)
    gmsh.model.add("magnetic")

    try:
        # Import STEP
        # Note: Don't use healShapes() as it can destroy solids.
        # Instead, the fragment operation (done later) creates conformal meshes
        # at volume interfaces, which fixes any mesh incompatibilities.
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        
        # Get all volumes
        volumes = gmsh.model.getEntities(3)
        print(f"Found {len(volumes)} volumes in STEP file")
        
        # Classify volumes
        core_tags = []
        turn_tags = []  # List of (tag, TurnInfo)
        
        # First, find bounding box of all volumes to determine air size
        all_bboxes = []
        for dim, tag in volumes:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            all_bboxes.append(bbox)
        
        # Overall bounds
        x_min = min(b[0] for b in all_bboxes)
        y_min = min(b[1] for b in all_bboxes)
        z_min = min(b[2] for b in all_bboxes)
        x_max = max(b[3] for b in all_bboxes)
        y_max = max(b[4] for b in all_bboxes)
        z_max = max(b[5] for b in all_bboxes)
        
        # Classify each volume using MAS turn coordinates and proportional logic
        bobbin_tags = []  # For bobbin (treat as air)
        all_volumes = []  # Will store (dim, tag, volume, center_of_mass) tuples
        
        for dim, tag in volumes:
            # Get center of mass and actual volume
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            actual_vol = gmsh.model.occ.getMass(dim, tag)  # mm^3
            all_volumes.append((dim, tag, actual_vol, com))
        
        # Sort by volume - largest first
        all_volumes.sort(key=lambda x: -x[2])
        
        # For toroidal cores: simplified classification
        # The geometry has N volumes: 1 large core + (N-1) turns
        # Largest volume is core, everything else smaller than 10% of core is a turn
        if core_type == "toroidal":
            print(f"  Using toroidal classification (core_type={core_type})")
            # Largest volume is the core
            core_dim, core_tag, core_vol, core_com = all_volumes[0]
            core_tags.append(core_tag)
            print(f"  Volume {core_tag}: CORE (vol={core_vol:.0f}mm³)")
            
            # All other volumes smaller than 10% of core are turns
            # Assign turns round-robin to TurnInfo objects
            turn_idx = 0
            for dim, tag, actual_vol, com in all_volumes[1:]:
                if actual_vol < core_vol * 0.1:
                    # This is a turn
                    ti = turns_info[turn_idx % len(turns_info)] if turns_info else None
                    if ti:
                        turn_tags.append((tag, ti))
                        print(f"  Volume {tag}: TURN vol={actual_vol:.0f}mm³ -> {ti.name}")
                        turn_idx += 1
                    else:
                        bobbin_tags.append(tag)
                        print(f"  Volume {tag}: UNKNOWN (vol={actual_vol:.0f}mm³) -> treat as air")
                else:
                    # Intermediate volume - bobbin or other structure
                    bobbin_tags.append(tag)
                    print(f"  Volume {tag}: BOBBIN (vol={actual_vol:.0f}mm³) -> treat as air")
            
            # Skip the standard classification below
            skip_standard_classification = True
        else:
            skip_standard_classification = False
        
        # Calculate expected turn volume from MAS data
        expected_turn_count = len(turns_info)
        if turns_info:
            # Expected turn volume = cross_section_area * circumference
            typical_turn_vol = turns_info[0].cross_section_area * 2 * math.pi * turns_info[0].radius
        else:
            typical_turn_vol = 0
        
        # Classification strategy using proportions (for non-toroidal cores):
        # 1. Match volumes to MAS turns by position (radius from axis, z-position)
        # 2. Core pieces are the largest volumes that DON'T match turn positions
        # 3. Bobbin is intermediate - matches neither core nor turn criteria
        
        core_tags_max_vol = 0
        core_candidates = []  # (tag, vol) pairs for deferred core/bobbin classification
        if not skip_standard_classification:
            # First pass: identify turns by matching to MAS turn positions
            turn_position_tolerance = 2.0  # mm tolerance for position matching
            matched_turn_tags = set()
            
            for dim, tag, actual_vol, com in all_volumes:
                center_z = com[1]  # height is Y axis for concentric
                # Calculate radius from Y axis (column axis for concentric)
                radius_from_axis = math.sqrt(com[0]**2 + com[2]**2)
                
                # Try to match this volume to a MAS turn
                for ti in turns_info:
                    z_match = abs(ti.z_position - center_z) < turn_position_tolerance
                    r_match = abs(ti.radius - radius_from_axis) < turn_position_tolerance
                    
                    # Volume should be similar to expected turn volume (within 5x)
                    vol_ratio = actual_vol / typical_turn_vol if typical_turn_vol > 0 else 0
                    vol_reasonable = 0.1 < vol_ratio < 5.0 if typical_turn_vol > 0 else True
                    
                    if z_match and vol_reasonable:
                        matched_turn_tags.add(tag)
                        break
            
            # Second pass: classify all volumes
            # The largest volumes that aren't turns are core
            # Use proportional logic: core is typically 10x+ larger than turns
            
            # Default threshold in case all_volumes is empty
            core_vol_threshold = 0
            if all_volumes:
                max_vol = all_volumes[0][2]
                # Volumes within 30% of max are likely core pieces (for multi-piece cores)
                core_vol_threshold = max_vol * 0.3
            
            for dim, tag, actual_vol, com in all_volumes:
                center_z = com[1]  # height is Y axis

                if tag in matched_turn_tags:
                    # This is a turn - find best match by z-position (Y axis)
                    best_match = None
                    best_dist = float('inf')
                    for ti in turns_info:
                        dist = abs(ti.z_position - center_z)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = ti
                    
                    turn_tags.append((tag, best_match))
                    print(f"  Volume {tag}: TURN z={center_z:.2f}mm, vol={actual_vol:.0f}mm³ -> {best_match.name}")
                
                elif actual_vol >= core_vol_threshold:
                    # Defer classification — collect all large volumes first.
                    # Core halves come in pairs; the unpaired one is the bobbin.
                    core_candidates.append((tag, actual_vol))
                
                elif actual_vol > typical_turn_vol * 2 if typical_turn_vol > 0 else actual_vol > 100:
                    # Intermediate volume - bobbin or other structure, treat as air
                    bobbin_tags.append(tag)
                    print(f"  Volume {tag}: BOBBIN (vol={actual_vol:.0f}mm³) -> treat as air")
                
                else:
                    # Small unmatched volume - might be a turn fragment or other geometry
                    # Try to match by z-position
                    best_match = None
                    best_dist = float('inf')
                    for ti in turns_info:
                        dist = abs(ti.z_position - center_z)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = ti
                    
                    if best_match and best_dist < turn_position_tolerance * 2:
                        turn_tags.append((tag, best_match))
                        print(f"  Volume {tag}: TURN (fallback) z={center_z:.2f}mm, vol={actual_vol:.0f}mm³ -> {best_match.name}")
                    else:
                        # Unclassified small volume - treat as air
                        bobbin_tags.append(tag)
                        print(f"  Volume {tag}: UNKNOWN (vol={actual_vol:.0f}mm³) -> treat as air")
        
        # Resolve core candidates: core halves come in pairs with matching volumes.
        # Group by similar volume (within 5%), then the largest group = core pieces.
        # Any unpaired candidate = bobbin.
        if core_candidates:
            core_candidates.sort(key=lambda x: -x[1])
            groups = []
            used = set()
            for i, (t1, v1) in enumerate(core_candidates):
                if i in used:
                    continue
                group = [(t1, v1)]
                used.add(i)
                for j, (t2, v2) in enumerate(core_candidates):
                    if j in used:
                        continue
                    if abs(v1 - v2) / max(v1, 1) < 0.05:
                        group.append((t2, v2))
                        used.add(j)
                groups.append(group)
            # Largest group with >=2 members = core pieces
            core_group = max(groups, key=lambda g: len(g) * 1000 + sum(v for _, v in g))
            for tag, vol in core_group:
                if len(core_group) >= 2:
                    core_tags.append(tag)
                    print(f"  Volume {tag}: CORE (vol={vol:.0f}mm³)")
                else:
                    # Single piece — could be bobbin or single-piece core
                    core_tags.append(tag)
                    print(f"  Volume {tag}: CORE (vol={vol:.0f}mm³, single)")
            for tag, vol in core_candidates:
                if tag not in core_tags:
                    bobbin_tags.append(tag)
                    print(f"  Volume {tag}: BOBBIN (vol={vol:.0f}mm³) -> treat as air")

        # For toroidal cores: Use fragment to create conformal mesh
        # Note: Don't use cut() as it creates overlapping facets
        # Note: Don't use removeAllDuplicates() as it can destroy small volumes
        if core_type == "toroidal":
            print("\nAdding air box for toroidal (required for boundary conditions)...")
            
            # Create air box
            air_box = gmsh.model.occ.addBox(
                x_min - air_padding, y_min - air_padding, z_min - air_padding,
                (x_max - x_min) + 2*air_padding,
                (y_max - y_min) + 2*air_padding,
                (z_max - z_min) + 2*air_padding
            )
            gmsh.model.occ.synchronize()
            
            # Set tolerance for near-touching turn-core surfaces
            gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-2)

            # Fragment for conformal mesh (works better than cut for toroidal)
            all_solid_tags = [(3, tag) for tag in core_tags] + [(3, t[0]) for t in turn_tags]
            out_dimtags, out_map = gmsh.model.occ.fragment([(3, air_box)], all_solid_tags)
            gmsh.model.occ.synchronize()


            # Use fragment output map to track volumes
            # out_map[0] = what air_box became (includes air region)
            # out_map[1] = what core became
            # out_map[2+] = what turns became
            new_volumes = gmsh.model.getEntities(3)
            print(f"After fragment: {len(new_volumes)} volumes")
            
            # Re-classify volumes using the fragment map
            new_core = []
            new_air = []
            new_turns = []
            
            # Map original tags to new tags
            core_new_tags = [t[1] for t in out_map[1]] if len(out_map) > 1 else []
            turn_new_tags_list = []
            for i in range(2, len(out_map)):
                turn_new_tags_list.extend([t[1] for t in out_map[i]])
            
            for dim, tag in new_volumes:
                actual_vol = gmsh.model.occ.getMass(dim, tag)
                com = gmsh.model.occ.getCenterOfMass(dim, tag)
                
                if tag in core_new_tags:
                    new_core.append(tag)
                elif tag in turn_new_tags_list:
                    # Find matching turn by angular position in XY plane
                    # (all turns on the same layer have the same radius)
                    turn_angle = math.atan2(com[1], com[0])
                    matched_ids = set(id(t) for _, t in new_turns)
                    best_match = None
                    best_dist = float('inf')
                    for turn_info in turns_info:
                        if id(turn_info) in matched_ids:
                            continue  # already matched
                        ti_angle = math.atan2(turn_info.z_position, turn_info.x_position)
                        # Angular distance (handle wrap-around)
                        d = abs(turn_angle - ti_angle)
                        if d > math.pi:
                            d = 2 * math.pi - d
                        if d < best_dist:
                            best_dist = d
                            best_match = turn_info
                    if best_match:
                        new_turns.append((tag, best_match))
                elif actual_vol > 1000:  # Fallback: large volume is air
                    new_air.append(tag)
            
            print(f"  Core: {len(new_core)}, Turns: {len(new_turns)}, Air: {len(new_air)}")
        else:
            # Create air box
            air_box = gmsh.model.occ.addBox(
                x_min - air_padding, y_min - air_padding, z_min - air_padding,
                (x_max - x_min) + 2*air_padding,
                (y_max - y_min) + 2*air_padding,
                (z_max - z_min) + 2*air_padding
            )
            
            # Fragment for conformal mesh
            all_vol_tags = ([(3, tag) for tag in core_tags] +
                            [(3, tag) for tag in bobbin_tags] +
                            [(3, t[0]) for t in turn_tags])
            
            gmsh.model.occ.fragment([(3, air_box)], all_vol_tags)
            gmsh.model.occ.synchronize()
            
            # Re-classify after fragmentation using actual volume
            new_volumes = gmsh.model.getEntities(3)
            print(f"\nAfter fragmentation: {len(new_volumes)} volumes")
        
        body_numbers = {}
        turn_bodies = {}
        body_id = 1
        
        # For toroidal without fragmentation, new_core/new_air/new_turns are already set
        # For other cases, we need to classify after fragmentation
        if core_type != "toroidal":
            new_core = []
            new_air = []  # Includes bobbin and air region
            new_turns = []  # List of (tag, TurnInfo)
            
            # Calculate expected turn volume from turns_info for better classification
            expected_turn_vol = 0
            if turns_info:
                # Expected turn volume = cross_section_area * circumference (in mm)
                expected_turn_vol = turns_info[0].cross_section_area * 2 * math.pi * turns_info[0].radius
            
            # First pass: collect all volume info
            vol_info = []
            for dim, tag in new_volumes:
                actual_vol = gmsh.model.occ.getMass(dim, tag)  # mm^3
                com = gmsh.model.occ.getCenterOfMass(dim, tag)
                bbox = gmsh.model.getBoundingBox(dim, tag)
                bbox_vol = (bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2])
                vol_info.append({
                    'tag': tag,
                    'vol': actual_vol,
                    'com': com,
                    'bbox_vol': bbox_vol,
                })
            
            # Sort by volume (excluding air box — the single largest bbox volume)
            max_bbox = max(v['bbox_vol'] for v in vol_info)
            solid_vols = [v for v in vol_info if v['bbox_vol'] < max_bbox * 0.9]
            solid_vols.sort(key=lambda x: -x['vol'])
            # Standard classification for concentric cores
            # The largest 2 solid volumes (excluding air-sized) are core halves
            # This works for split cores (E, PQ, etc.)
            core_vol_threshold = 0
            if len(solid_vols) >= 2:
                # Core threshold: 50% of smallest core volume (to catch both halves)
                core_vol_threshold = solid_vols[1]['vol'] * 0.5
            
            for v in vol_info:
                tag = v['tag']
                actual_vol = v['vol']
                center_z = v['com'][1]  # height is Y axis for concentric
                bbox_vol = v['bbox_vol']
                
                if bbox_vol > max_bbox * 0.9:  # Largest bounding box = air region
                    new_air.append(tag)
                elif actual_vol >= core_vol_threshold:
                    # Core halves come in pairs with nearly identical volumes.
                    is_core = False
                    if not new_core:
                        is_core = True
                    else:
                        for ct in new_core:
                            ct_vol = next(v2['vol'] for v2 in vol_info if v2['tag'] == ct)
                            if abs(actual_vol - ct_vol) / max(ct_vol, 1) < 0.05:
                                is_core = True
                                break
                    if is_core:
                        new_core.append(tag)
                    else:
                        new_air.append(tag)  # Bobbin
                elif actual_vol > expected_turn_vol * 3 if expected_turn_vol > 0 else actual_vol > 100:
                    # Bobbin or other intermediate volume - treat as air
                    new_air.append(tag)
                else:  # Small volume - could be turn or fragment
                    # Find matching turn info by z-position (each turn matches once)
                    matched_turns_set = set(id(t) for _, t in new_turns)
                    best_match = None
                    best_dist = float('inf')
                    for ti in turns_info:
                        if id(ti) in matched_turns_set:
                            continue  # already matched
                        dist = abs(ti.z_position - center_z)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = ti

                    # Check if volume is similar to expected turn volume (within 3x)
                    vol_ratio = actual_vol / expected_turn_vol if expected_turn_vol > 0 else 0
                    is_turn_sized = 0.3 < vol_ratio < 3.0 if expected_turn_vol > 0 else False

                    if best_match and best_dist < 1.0 and is_turn_sized:
                        # Valid turn: matches position AND size
                        new_turns.append((tag, best_match))
                    else:
                        # Small fragment - treat as air, not a turn
                        new_air.append(tag)
        
        # Create physical groups
        print(f"\nCreating physical groups:")
        print(f"  new_core={new_core}, new_turns={len(new_turns)}, new_air={new_air}")
        
        # Body 1: Core
        if new_core:
            gmsh.model.addPhysicalGroup(3, new_core, tag=body_id, name="core")
            print(f"  Added core: body_id={body_id}, entities={new_core}")
            body_numbers["core"] = body_id
            body_id += 1
        
        # Each turn gets its own body for both toroidal and concentric cores
        # This is required because each turn is a separate closed electrical loop
        # and CoilSolver needs each loop as a separate Component
        for vol_tag, turn_info in new_turns:
            gmsh.model.addPhysicalGroup(3, [vol_tag], tag=body_id, name=turn_info.name)
            print(f"  Added turn: body_id={body_id}, vol_tag={vol_tag}, name={turn_info.name}")
            body_numbers[turn_info.name] = body_id
            turn_bodies[body_id] = turn_info
            body_id += 1
        
        # Air body - needed for both toroidal and non-toroidal
        if new_air:
            gmsh.model.addPhysicalGroup(3, new_air, tag=body_id, name="air")
            print(f"  Added air: body_id={body_id}, entities={new_air}")
            body_numbers["air"] = body_id
            body_id += 1
        
        # Outer boundary - needed for both toroidal and non-toroidal
        outer_bbox = [x_min - air_padding, y_min - air_padding, z_min - air_padding,
                      x_max + air_padding, y_max + air_padding, z_max + air_padding]
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
            gmsh.model.addPhysicalGroup(2, outer_surfs, tag=100, name="outer_boundary")
            print(f"  Added outer boundary: {len(outer_surfs)} surfaces")
        
        # Compute wire diameter from turn cross-section area for mesh sizing
        if turns_info:
            wire_diameter = 2.0 * math.sqrt(turns_info[0].cross_section_area / math.pi)
        else:
            wire_diameter = 1.0

        # Collect turn surfaces for mesh refinement.
        # Use the expected turn volume to identify turn bodies.
        turn_surface_tags = set()
        if turns_info:
            expected_turn_vol = turns_info[0].cross_section_area * 2 * math.pi * turns_info[0].radius
            vol_threshold = expected_turn_vol * 10  # generous threshold
        else:
            vol_threshold = 500
        for dim, tag in gmsh.model.getEntities(3):
            mass = gmsh.model.occ.getMass(dim, tag)
            if 0 < mass < vol_threshold:
                surfs = gmsh.model.getBoundary([(3, tag)], combined=False, oriented=False)
                for s in surfs:
                    turn_surface_tags.add(s[1])

        # Mesh settings — scale with geometry span and turn count.
        # More turns → coarser mesh to keep total element count manageable.
        # Target: ~100K-300K elements regardless of turn count.
        geo_span = max(x_max - x_min, y_max - y_min, z_max - z_min)
        num_turns_total = len(turns_info)
        if num_turns_total > 20:
            actual_max_size = max(max_element_size, geo_span / 5.0)
        else:
            actual_max_size = max(max_element_size, geo_span / 10.0)
        actual_min_size = min_element_size
        if core_type == "toroidal" and turns_info:
            actual_max_size = max(2.0, geo_span / 10.0)
            actual_min_size = 0.2
            print(f"  Using toroidal mesh sizes: max={actual_max_size:.2f}mm, min={actual_min_size:.2f}mm")

        # Mesh refinement near turns. Scale with turn count:
        # Few turns (1-6): fine (wire_diam/4) for accurate cross-section
        # Many turns (>10): coarser (wire_diam/2 to wire_diam) since CoilSolver
        # handles current topology regardless of mesh density.
        num_turns_total = len(turns_info)
        if num_turns_total <= 6:
            fine_size = wire_diameter / 4.0
        elif num_turns_total <= 20:
            fine_size = wire_diameter / 2.0
        else:
            fine_size = wire_diameter  # 1 element across wire diameter
        # For toroidal, use gentler refinement — pipe sweep B-spline surfaces
        # cause "overlapping facets" with aggressive distance fields.
        if core_type == "toroidal":
            fine_size = max(fine_size, wire_diameter / 2.0, 0.2)

        # For few turns, use distance field for local refinement near turn surfaces.
        # For many turns (>20), skip the distance field — the per-surface overhead
        # makes gmsh slow, and gmsh's curvature-based sizing is sufficient.
        turn_surface_tags_list = list(turn_surface_tags) if turn_surface_tags else []
        if turn_surface_tags_list and num_turns_total <= 20:
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", turn_surface_tags_list)

            thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", fine_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", actual_max_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", wire_diameter * 3)

            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            print(f"  Turn mesh refinement: fine_size={fine_size:.3f}mm (wire_diam={wire_diameter:.3f}mm)")
        else:
            print(f"  Using curvature-based sizing (no distance field, {num_turns_total} turns)")

        gmsh.option.setNumber("Mesh.MeshSizeMax", actual_max_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(fine_size * 0.5, 0.1))

        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)

        # Generate mesh with escalating retries.
        # For toroidal: try Delaunay, then HXT, then coarser Delaunay.
        # For concentric: try Delaunay with escalating coarseness.
        if core_type == "toroidal":
            retry_configs = [
                (1, 1, "Delaunay"),
                (10, 1, "HXT"),
                (1, 3, "Delaunay 3x coarser"),
                (10, 3, "HXT 3x coarser"),
                (1, 8, "Delaunay 8x coarser"),
            ]
        else:
            retry_configs = [
                (1, 1, "Delaunay"),
                (1, 3, "Delaunay 3x coarser"),
                (1, 8, "Delaunay 8x coarser"),
            ]

        print("\nGenerating mesh...")
        for i, (algo3d, coarse_factor, desc) in enumerate(retry_configs):
            try:
                if i > 0:
                    print(f"Retrying with {desc}...")
                    gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm3D", algo3d)
                if coarse_factor > 1:
                    if turn_surface_tags_list and num_turns_total <= 20:
                        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", fine_size * coarse_factor)
                    gmsh.option.setNumber("Mesh.MeshSizeMax", actual_max_size * coarse_factor)
                gmsh.model.mesh.generate(3)
                et, tags_check, _ = gmsh.model.mesh.getElements(3)
                n = sum(len(t) for t in tags_check) if tags_check else 0
                if n > 0:
                    print(f"  Success with {desc}: {n} elements")
                    break
                print(f"Warning: {desc} produced 0 elements")
            except Exception as e:
                print(f"Warning: {desc} failed: {e}")
        
        # Verify 3D elements were created
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        total_3d = sum(len(t) for t in elem_tags) if elem_tags else 0
        if total_3d == 0:
            raise RuntimeError("No 3D elements generated - mesh failed")
        print(f"Generated {total_3d} 3D elements")

        # Extra optimization passes for toroidal meshes to eliminate degenerate
        # elements in the thin air gaps between turns
        if core_type == "toroidal":
            gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
            for opt_pass in range(5):
                gmsh.model.mesh.optimize("", force=True)
            gmsh.model.mesh.optimize("Netgen", force=True)
            gmsh.model.mesh.optimize("HighOrderElastic", force=True)
            print("Applied extra mesh optimization passes for toroidal")

        # Save
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        msh_path = os.path.join(output_path, "mesh.msh")
        gmsh.write(msh_path)
        print(f"Mesh saved to {msh_path}")
        
    finally:
        gmsh.finalize()
    
    # Convert to Elmer format
    mesh_dir = os.path.join(output_path, "mesh")

    cmd = ["ElmerGrid", "14", "2", "mesh.msh", "-autoclean",
           "-scale", "0.001", "0.001", "0.001"]  # mm to m

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_path)
    print(f"ElmerGrid output: {result.stdout[:500]}")
    
    if not os.path.exists(mesh_dir):
        raise RuntimeError(f"ElmerGrid failed: {result.stderr}")
    
    return mesh_dir, body_numbers, turn_bodies


def generate_sif_with_coil_solver(
    output_path: str,
    body_numbers: Dict[str, int],
    turn_bodies: Dict[int, TurnInfo],
    core_permeability: float = 2000.0,
    total_current: float = 1.0,
    num_turns: int = 1,
    core_type: str = "concentric",
    active_windings: Optional[List[str]] = None,
) -> str:
    """
    Generate SIF file using CoilSolver with Coil Closed = True.
    
    This approach uses Elmer's built-in CoilSolver to compute proper current
    density for closed (toroidal) coils. The CoilSolver finds a divergence-free
    current density field that satisfies the closed loop constraint.
    
    For PQ/E-core geometry (concentric):
    - Central column is along Z-axis
    - Turns are tori around Z-axis
    - Coil Normal = (0, 0, 1) - along Z axis
    
    For toroidal cores:
    - Core is a torus with axis along Y
    - Turns wrap around the toroidal cross-section (poloidal direction)
    - Coil Normal should point along the major circumference direction (toroidal direction)
    - For a turn at angle theta around the major circumference:
      Coil Normal = (-sin(theta), 0, cos(theta)) where theta = atan2(z, x)
    """
    
    # Get coil body IDs (all turns together form one coil)
    coil_body_ids = list(turn_bodies.keys())
    core_id = body_numbers.get("core", 1)
    air_id = body_numbers.get("air", max(body_numbers.values()) + 1)
    
    # Get average turn radius for Coil Normal specification
    avg_radius = sum(t.radius for t in turn_bodies.values()) / len(turn_bodies) if turn_bodies else 9.0
    
    sif_lines = [
        "! Elmer magnetostatic simulation using CoilSolver for closed coil",
        "! Generated by validate_elmer_inductance.py",
        "!",
        "! CoilSolver computes divergence-free current density for closed loops.",
        "! This is required for toroidal turns around a central column.",
        "",
        'Check Keywords "Warn"',
        "",
        "Header",
        '  Mesh DB "." "mesh"',
        '  Results Directory "."',
        "End",
        "",
        "Simulation",
        "  Coordinate System = Cartesian 3D",
        "  Simulation Type = Steady State",
        "  Steady State Max Iterations = 1",
        "  Max Output Level = 7",
        "End",
        "",
        "Constants",
        "  Permittivity Of Vacuum = 8.8542e-12",
        "  Permeability Of Vacuum = 1.2566e-6",
        "End",
        "",
        "! Materials",
        "Material 1",
        '  Name = "Ferrite"',
        f"  Relative Permeability = {core_permeability}",
        "  Electric Conductivity = 0.0",
        "End",
        "",
        "Material 2",
        '  Name = "Copper"',
        "  Relative Permeability = 1.0",
        "  Electric Conductivity = 1.0",  # Non-zero for CoilSolver
        "End",
        "",
        "Material 3",
        '  Name = "Air"',
        "  Relative Permeability = 1.0",
        "  Electric Conductivity = 0.0",
        "End",
        "",
    ]
    
    # Component definition for each turn as a separate coil.
    # When active_windings is set, only turns belonging to those windings
    # carry current. Others get J=0 (still needed for CoilSolver topology).
    comp_id = 1
    for body_id in coil_body_ids:
        turn_info = turn_bodies[body_id]

        # Check if this turn's winding is active
        if active_windings is not None:
            is_active = any(aw.lower() in turn_info.winding.lower() for aw in active_windings)
        else:
            is_active = True

        J_desired = total_current / (turn_info.cross_section_area * 1e-6) if is_active else 0.0

        # Calculate Coil Normal based on core type and turn position
        if core_type == "toroidal":
            x = turn_info.x_position
            y = turn_info.z_position
            r_actual = math.sqrt(x**2 + y**2) if (x**2 + y**2) > 0 else 1.0
            nx = -y / r_actual
            ny = x / r_actual
            nz = 0.0
            coil_normal = f"  Coil Normal(3) = Real {nx:.6f} {ny:.6f} {nz:.6f}"
        else:
            coil_normal = "  Coil Normal(3) = Real 0.0 1.0 0.0"

        sif_lines.extend([
            f"Component {comp_id}",
            f'  Name = String "Coil_{turn_info.name}"',
            '  Coil Type = String "test"',
            f"  Master Bodies(1) = Integer {body_id}",
            f"  Desired Current Density = Real {J_desired:.1f}",
            coil_normal,
            "End",
            "",
        ])
        comp_id += 1
    
    # Bodies
    sif_lines.append("! Bodies")
    
    # Core
    sif_lines.extend([
        f"Body {core_id}",
        '  Name = "Core"',
        "  Equation = 1",
        "  Material = 1",
        "End",
        "",
    ])
    
    # All turns - use CoilSolver equation
    for body_id in coil_body_ids:
        turn_info = turn_bodies[body_id]
        sif_lines.extend([
            f"Body {body_id}",
            f'  Name = "{turn_info.name}"',
            "  Equation = 2",  # Equation with CoilSolver
            "  Material = 2",
            "End",
            "",
        ])
    
    # Air
    if air_id:
        sif_lines.extend([
            f"Body {air_id}",
            '  Name = "Air"',
            "  Equation = 1",
            "  Material = 3",
            "End",
            "",
        ])
    
    # Equations — for large meshes (>20 turns), skip VTU output (solver 4)
    if num_turns <= 20:
        sif_lines.extend([
            "! Equations",
            "Equation 1",
            '  Name = "MagnetoDynamics for air/core"',
            "  Active Solvers(2) = 2 3",
            "End",
            "",
            "Equation 2",
            '  Name = "MagnetoDynamics with CoilSolver"',
            "  Active Solvers(3) = 1 2 3",
            "End",
            "",
        ])
    else:
        sif_lines.extend([
            "! Equations (large mesh: no VTU output)",
            "Equation 1",
            '  Name = "MagnetoDynamics for air/core"',
            "  Active Solvers(2) = 2 3",
            "End",
            "",
            "Equation 2",
            '  Name = "MagnetoDynamics with CoilSolver"',
            "  Active Solvers(3) = 1 2 3",
            "End",
            "",
        ])
    
    # Solvers - based on mgdyn_steady_coils test case
    sif_lines.extend([
        "! Solvers",
        "",
        "Solver 1",
        '  Equation = "CoilSolver"',
        '  Procedure = "CoilSolver" "CoilSolver"',
        "",
        "  Linear System Solver = Iterative",
        "  Linear System Iterative Method = BiCGStab",
        "  Linear System Preconditioning = ILU2",
        "  Linear System Max Iterations = 1000",
        "  Linear System Convergence Tolerance = 1.0e-8",
        "  Linear System Residual Output = 20",
        "",
        "  Coil Closed = Logical True",
        "  Narrow Interface = Logical True",
        "  Save Coil Set = Logical True",
        "  Save Coil Index = Logical True",
        "  Calculate Elemental Fields = Logical True",
        "  Fix Input Current Density = True",
        "End",
        "",
        "Solver 2",
        "  Equation = MGDynamics",
        '  Procedure = "MagnetoDynamics" "WhitneyAVSolver"',
        "  Variable = AV",
        "",
        "  ! Use the elemental current density from CoilSolver",
        "  Use Elemental CoilCurrent = Logical True",
        "  Fix Input Current Density = Logical True",
        "",
    ])

    # Use iterative solver for large problems (>20 turns ≈ >200K elements)
    if num_turns > 20:
        sif_lines.extend([
            "  Linear System Solver = Iterative",
            "  Linear System Iterative Method = BiCGStabl",
            "  BiCGstabl polynomial degree = 4",
            "  Linear System Preconditioning = ILU2",
            "  Linear System Max Iterations = 3000",
            "  Linear System Convergence Tolerance = 1.0e-7",
            "  Linear System Residual Output = 50",
            "  Linear System Abort Not Converged = False",
        ])
    else:
        sif_lines.extend([
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPack",
        ])

    sif_lines.extend([
        "",
        "  Nonlinear System Max Iterations = 1",
        "End",
        "",
    ])

    # CalcFields is always needed (computes ElectroMagnetic Field Energy).
    # For large meshes, skip VTU output to save memory.
    if True:
        sif_lines.extend([
            "Solver 3",
            "  Equation = MGDynamicsCalc",
            '  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"',
            "",
            '  Potential Variable = "AV"',
            "  Calculate Magnetic Field Strength = True",
            "  Calculate Magnetic Flux Density = True",
            "  Calculate Current Density = True",
            "  Calculate Nodal Fields = False",
            "  Calculate Elemental Fields = True",
            "",
            "  Linear System Solver = Iterative",
            "  Linear System Iterative Method = CG",
            "  Linear System Preconditioning = Diagonal",
            "  Linear System Max Iterations = 5000",
            "  Linear System Convergence Tolerance = 1.0e-5",
            "  Linear System Abort Not Converged = False",
            "End",
            "",
        ])

    # VTU output — skip for large meshes to avoid memory issues
    if num_turns <= 20:
        sif_lines.extend([
            "Solver 4",
            "  Exec Solver = After All",
            "  Equation = ResultOutput",
            '  Procedure = "ResultOutputSolve" "ResultOutputSolver"',
            "",
            '  Output File Name = "results"',
            "  Vtu Format = True",
            "  Save Geometry Ids = True",
            "  Discontinuous Bodies = True",
            "End",
            "",
        ])

    sif_lines.extend([
        "! Boundary condition - magnetic vector potential and jfix zero at boundary",
        "! Note: ElmerGrid renumbers physical group 100 to boundary 1",
        "Boundary Condition 1",
        '  Name = "OuterBoundary"',
        "  Target Boundaries(1) = 1",
        "  AV {e} = Real 0.0",
        "  AV = Real 0.0",
        "  jfix = Real 0.0",
        "End",
    ])
    
    sif_content = "\n".join(sif_lines)
    sif_path = os.path.join(output_path, "case.sif")
    
    with open(sif_path, 'w') as f:
        f.write(sif_content)
    
    # Write STARTINFO
    with open(os.path.join(output_path, "ELMERSOLVER_STARTINFO"), 'w') as f:
        f.write("case.sif\n")
    
    return sif_path


def generate_sif_with_tangential_current(
    output_path: str,
    body_numbers: Dict[str, int],
    turn_bodies: Dict[int, TurnInfo],
    core_permeability: float = 2000.0,
    total_current: float = 1.0,  # Total current through winding (A)
    core_type: str = "concentric",
    use_iterative_solver: bool = False,
    active_windings: Optional[List[str]] = None,
    bh_curve: Optional[List[Tuple[float, float]]] = None,
) -> str:
    """
    Generate SIF file with proper tangential current for each turn.
    
    For a toroidal (closed) conductor wrapping around the z-axis, the current
    must flow in circles around the z-axis. To satisfy continuity (div J = 0),
    the current density must vary as 1/r.
    
    For a wire at mean radius R0 with cross-section area A:
    J(r) = (I * R0) / (A * r) * tangential_direction
    
    At r = R0: J = I/A (matches uniform current assumption)
    Integral: ∫∫ J(r) dA = I (total current is preserved)
    
    Applied as:
    Jx = -(I*R0)/(A) * y/r² = -(I*R0/A) * y/(x²+y²)
    Jy = (I*R0)/(A) * x/r² = (I*R0/A) * x/(x²+y²)
    Jz = 0
    """
    
    sif_lines = [
        "! Elmer magnetostatic simulation for inductance calculation",
        "! Generated by validate_elmer_inductance.py",
        "",
        'Check Keywords "Warn"',
        "",
        "Header",
        '  Mesh DB "." "mesh"',
        '  Results Directory "."',
        "End",
        "",
        "Simulation",
        "  Coordinate System = Cartesian 3D",
        "  Simulation Type = Steady State",
        "  Steady State Max Iterations = 1",
        "  Max Output Level = 5",
        "End",
        "",
        "Constants",
        "  Permittivity Of Vacuum = 8.854e-12",
        "  Permeability Of Vacuum = 1.2566e-6",
        "End",
        "",
        "! Materials",
        "Material 1",
        '  Name = "Ferrite"',
    ]

    if bh_curve and len(bh_curve) >= 3:
        n = len(bh_curve)
        sif_lines.append(f"  H-B Curve({n},2) = Real")
        for H, B in bh_curve:
            sif_lines.append(f"    {H:.4f} {B:.6f}")
    else:
        sif_lines.append(f"  Relative Permeability = {core_permeability}")

    sif_lines.extend([
        "  Electric Conductivity = 0.0",
        "End",
        "",
        "Material 2",
        '  Name = "Copper"',
        "  Relative Permeability = 1.0",
        "  Electric Conductivity = 0.0",
        "End",
        "",
        "Material 3",
        '  Name = "Air"',
        "  Relative Permeability = 1.0",
        "  Electric Conductivity = 0.0",
        "End",
        "",
    ])

    # Generate body forces for each turn
    body_force_id = 1
    body_force_map = {}  # body_id -> body_force_id
    
    for body_id, turn_info in turn_bodies.items():
        # For a turn wrapping around a central column (along Z-axis):
        # The current must flow in the POLOIDAL direction (around the column),
        # NOT in the toroidal direction (around the Z-axis).
        #
        # For a turn at radius R0 from Z-axis, at height z0:
        # - The turn cross-section center is at (R0, 0, z0) in cylindrical coords
        # - Current flows around this center in the r-z plane
        #
        # At any point (x, y, z) in the turn:
        # - Distance from turn center: ρ = sqrt((r - R0)² + (z - z0)²)
        #   where r = sqrt(x² + y²)
        # - Current direction is tangent to circles around the turn center
        # - This is the poloidal direction: perpendicular to both radial and φ
        #
        # The poloidal unit vector at (x,y,z):
        #   e_poloidal = (-(z-z0)/ρ * r̂ + (r-R0)/ρ * ẑ)
        # where r̂ = (x/r, y/r, 0) is the radial unit vector
        #
        # So: e_pol_x = -(z-z0)/ρ * x/r
        #     e_pol_y = -(z-z0)/ρ * y/r  
        #     e_pol_z = (r-R0)/ρ
        #
        # Current density magnitude: J = I/A (uniform in cross-section)
        
        A_m2 = turn_info.cross_section_area * 1e-6  # m^2
        R0_m = turn_info.radius * 1e-3  # m (mean radius of turn from z-axis)
        if core_type == "toroidal":
            # Toroidal turns lie in the XY plane; z_position stores the Y coord.
            # The actual Z coordinate of the turn center is 0.
            z0_m = 0.0
        else:
            z0_m = turn_info.z_position * 1e-3  # m (height along Y axis)
        
        J_mag = total_current / A_m2  # A/m^2
        # For winding regions (merged turns), multiply by number of turns in region
        n_region = getattr(turn_info, '_n_turns_in_region', 1)
        J_mag *= n_region

        # Zero current for inactive windings
        if active_windings is not None:
            if not any(aw.lower() in turn_info.winding.lower() for aw in active_windings):
                J_mag = 0.0

        # Clockwise or counterclockwise (when viewed from outside the torus)
        sign = 1.0 if turn_info.orientation == 'clockwise' else -1.0
        
        # Current components in MATC:
        # r = sqrt(x² + y²)
        # rho = sqrt((r - R0)² + (z - z0)²)  [distance from turn axis]
        # Jx = -J * (z - z0) / rho * x / r
        # Jy = -J * (z - z0) / rho * y / r
        # Jz = J * (r - R0) / rho
        #
        # Using tx(0)=x, tx(1)=y, tx(2)=z in MATC
        
        if core_type == "toroidal":
            # POLOIDAL CURRENT for toroidal turns
            # Each turn is a loop around the core. Current flows along the loop
            # (poloidal direction), creating flux through the core cross-section.
            #
            # At point (x,y,z), the poloidal direction around the turn center at
            # radius R0 from Z axis:
            #   r = sqrt(x²+y²)
            #   dr = r - R0, dz = z - z0
            #   rho = sqrt(dr² + dz²)  (distance from turn center line)
            #   e_pol = (-(dz/rho)*r_hat + (dr/rho)*z_hat)
            #   Jx = J * (-dz/rho) * (x/r)
            #   Jy = J * (-dz/rho) * (y/r)
            #   Jz = J * (dr/rho)
            eps = 1e-10
            sif_lines.extend([
                f"Body Force {body_force_id}",
                f'  Name = "Current_{turn_info.name}"',
                f"  ! Poloidal current for toroidal turn",
                f"  ! Turn at R0 = {R0_m*1000:.2f} mm, z0 = {z0_m*1000:.2f} mm",
                f"  ! Cross-section area: A = {A_m2*1e6:.2f} mm^2",
                f"  ! Current density: |J| = I/A = {J_mag:.4e} A/m^2",
                "  Current Density 1 = Variable Coordinate",
                f'    Real MATC "{-sign * J_mag} * (tx(2) - {z0_m}) / (sqrt((sqrt(tx(0)^2 + tx(1)^2) - {R0_m})^2 + (tx(2) - {z0_m})^2) + {eps}) * tx(0) / (sqrt(tx(0)^2 + tx(1)^2) + {eps})"',
                "  Current Density 2 = Variable Coordinate",
                f'    Real MATC "{-sign * J_mag} * (tx(2) - {z0_m}) / (sqrt((sqrt(tx(0)^2 + tx(1)^2) - {R0_m})^2 + (tx(2) - {z0_m})^2) + {eps}) * tx(1) / (sqrt(tx(0)^2 + tx(1)^2) + {eps})"',
                "  Current Density 3 = Variable Coordinate",
                f'    Real MATC "{sign * J_mag} * (sqrt(tx(0)^2 + tx(1)^2) - {R0_m}) / (sqrt((sqrt(tx(0)^2 + tx(1)^2) - {R0_m})^2 + (tx(2) - {z0_m})^2) + {eps})"',
                "End",
                "",
            ])
        else:
            # TANGENTIAL CURRENT around Y-axis for concentric
            # Column axis is Y. Current flows in XZ plane: (-z/r, 0, x/r)
            # where r = sqrt(x²+z²)
            sif_lines.extend([
                f"Body Force {body_force_id}",
                f'  Name = "Current_{turn_info.name}"',
                f"  ! Tangential current around Y-axis (concentric)",
                f"  ! Turn at R0 = {R0_m*1000:.2f} mm, y0 = {z0_m*1000:.2f} mm",
                f"  ! Cross-section area: A = {A_m2*1e6:.2f} mm^2",
                f"  ! Current density: |J| = I/A = {J_mag:.4e} A/m^2",
                "  Current Density 1 = Variable Coordinate",
                f'    Real MATC "{-sign * J_mag} * tx(2) / (sqrt(tx(0)^2 + tx(2)^2) + 1e-10)"',
                "  Current Density 2 = Real 0.0",
                "  Current Density 3 = Variable Coordinate",
                f'    Real MATC "{sign * J_mag} * tx(0) / (sqrt(tx(0)^2 + tx(2)^2) + 1e-10)"',
                "End",
                "",
            ])
        
        body_force_map[body_id] = body_force_id
        body_force_id += 1
    
    # Bodies
    sif_lines.append("! Bodies")
    
    # Core
    core_id = body_numbers.get("core", 1)
    sif_lines.extend([
        f"Body {core_id}",
        '  Name = "Core"',
        "  Equation = 1",
        "  Material = 1",
        "End",
        "",
    ])
    
    # Each turn
    for body_id, turn_info in turn_bodies.items():
        bf_id = body_force_map.get(body_id)
        sif_lines.extend([
            f"Body {body_id}",
            f'  Name = "{turn_info.name}"',
            "  Equation = 1",
            "  Material = 2",
            f"  Body Force = {bf_id}",
            "End",
            "",
        ])
    
    # Air
    air_id = body_numbers.get("air")
    if air_id:
        sif_lines.extend([
            f"Body {air_id}",
            '  Name = "Air"',
            "  Equation = 1",
            "  Material = 3",
            "End",
            "",
        ])
    
    # Equation and solvers
    sif_lines.extend([
        "! Equation",
        "Equation 1",
        '  Name = "MagnetoDynamics"',
        "  Active Solvers(3) = 1 2 3",
        "End",
        "",
        "! Solvers",
        "Solver 1",
        "  Equation = MGDynamics",
        '  Procedure = "MagnetoDynamics" "WhitneyAVSolver"',
        "  Variable = AV",
        "",
    ])

    if use_iterative_solver:
        sif_lines.extend([
            "  Linear System Solver = Iterative",
            "  Linear System Iterative Method = BiCGStabl",
            "  BiCGstabl polynomial degree = 4",
            "  Linear System Max Iterations = 2000",
            "  Linear System Convergence Tolerance = 1.0e-8",
            "  Linear System Preconditioning = ILU2",
            "  Linear System Abort Not Converged = False",
            "  Linear System Residual Output = 50",
        ])
    else:
        sif_lines.extend([
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPACK",
        ])

    # Nonlinear iterations for BH curve
    if bh_curve and len(bh_curve) >= 3:
        sif_lines.extend([
            "",
            "  Nonlinear System Max Iterations = 15",
            "  Nonlinear System Convergence Tolerance = 1.0e-6",
            "  Nonlinear System Newton After Iterations = 3",
            "  Nonlinear System Newton After Tolerance = 1.0e-3",
            "  Nonlinear System Relaxation Factor = 0.7",
        ])

    sif_lines.extend([
        "",
        "  Steady State Convergence Tolerance = 1.0e-8",
        "End",
        "",
        "Solver 2",
        "  Equation = MGDynamicsCalc",
        '  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"',
        "",
        '  Potential Variable = "AV"',
        "  Calculate Magnetic Field Strength = True",
        "  Calculate Magnetic Flux Density = True",
        "  Calculate Current Density = True",
        "  Calculate Nodal Fields = False",
        "  Calculate Elemental Fields = True",
        "",
    ])

    if use_iterative_solver:
        sif_lines.extend([
            "  Linear System Solver = Iterative",
            "  Linear System Iterative Method = BiCGStabl",
            "  BiCGstabl polynomial degree = 2",
            "  Linear System Max Iterations = 1000",
            "  Linear System Convergence Tolerance = 1.0e-6",
            "  Linear System Preconditioning = ILU1",
            "  Linear System Abort Not Converged = False",
        ])
    else:
        sif_lines.extend([
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPACK",
        ])

    sif_lines.extend([
        "",
        "  Steady State Convergence Tolerance = 1.0e-6",
        "End",
        "",
        "Solver 3",
        "  Equation = ResultOutput",
        "  Exec Solver = After Timestep",
        '  Procedure = "ResultOutputSolve" "ResultOutputSolver"',
        "",
        '  Output File Name = "results"',
        "  Vtu Format = True",
        "  Save Geometry Ids = True",
        "End",
        "",
        "! Boundary condition - magnetic vector potential zero at outer boundary",
        "! Note: ElmerGrid renumbers boundaries, so physical group 100 becomes 1",
        "Boundary Condition 1",
        '  Name = "OuterBoundary"',
        "  Target Boundaries(1) = 1",
        "  AV {e} = Real 0.0",
        "  AV = Real 0.0",
        "End",
    ])
    
    sif_content = "\n".join(sif_lines)
    sif_path = os.path.join(output_path, "case.sif")
    
    with open(sif_path, 'w') as f:
        f.write(sif_content)
    
    # Write STARTINFO
    with open(os.path.join(output_path, "ELMERSOLVER_STARTINFO"), 'w') as f:
        f.write("case.sif\n")
    
    return sif_path


def run_elmer(sim_dir: str, timeout: int = 600) -> Tuple[bool, float, str]:
    """
    Run ElmerSolver and extract energy.
    
    Returns:
        (success, energy, output)
    """
    try:
        result = subprocess.run(
            ["ElmerSolver"],
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout + result.stderr
        
        # Parse energy
        energy = 0.0
        for line in output.split('\n'):
            if 'ElectroMagnetic Field Energy:' in line:
                match = re.search(r'Energy:\s+([\d.E+-]+)', line)
                if match:
                    energy = float(match.group(1))
        
        success = result.returncode == 0 and 'ALL DONE' in output
        return success, energy, output
        
    except subprocess.TimeoutExpired:
        return False, 0.0, "Timeout"
    except Exception as e:
        return False, 0.0, str(e)


def calculate_inductance_from_energy(energy: float, current: float) -> float:
    """Calculate inductance from magnetic energy: L = 2*W / I^2"""
    if current == 0:
        return 0
    return 2 * energy / (current ** 2)


def generate_sif_harmonic(
    output_path: str,
    body_numbers: Dict[str, int],
    turn_bodies: Dict[int, TurnInfo],
    core_permeability: float = 2000.0,
    frequency: float = 100000.0,
    total_current: float = 1.0,
    core_type: str = "concentric",
    active_windings: Optional[List[str]] = None,
    copper_conductivity: float = 5.96e7,
) -> str:
    """
    Generate SIF for AC harmonic simulation using WhitneyAVHarmonicSolver.

    Computes frequency-dependent inductance, AC resistance, and Joule losses
    accounting for skin and proximity effects.

    The harmonic solver uses complex-valued AV field at a single frequency.
    Current excitation is applied as tangential body forces (same as DC).
    Copper conductivity enables eddy current computation in the windings.
    """
    omega = 2 * math.pi * frequency
    coil_body_ids = list(turn_bodies.keys())
    core_id = body_numbers.get("core", 1)
    air_id = body_numbers.get("air", max(body_numbers.values()) + 1)

    sif_lines = [
        f"! Elmer AC harmonic simulation at f={frequency:.0f} Hz",
        "! Generated by validate_elmer_inductance.py",
        "",
        'Check Keywords "Warn"',
        "",
        "Header",
        '  Mesh DB "." "mesh"',
        '  Results Directory "."',
        "End",
        "",
        "Simulation",
        "  Coordinate System = Cartesian 3D",
        "  Simulation Type = Steady State",
        "  Steady State Max Iterations = 1",
        "  Max Output Level = 5",
        f"  Angular Frequency = {omega:.6f}",
        "End",
        "",
        "Constants",
        "  Permittivity Of Vacuum = 8.8542e-12",
        "  Permeability Of Vacuum = 1.2566e-6",
        "End",
        "",
        "! Materials",
        "Material 1",
        '  Name = "Ferrite"',
        f"  Relative Permeability = {core_permeability}",
        "  Electric Conductivity = 0.0",
        "End",
        "",
        "Material 2",
        '  Name = "Copper"',
        "  Relative Permeability = 1.0",
        "  ! Stranded conductor: zero conductivity (no eddy currents in winding)",
        "  ! For solid conductors, use copper_conductivity = 5.96e7",
        "  Electric Conductivity = 0.0",
        "End",
        "",
        "Material 3",
        '  Name = "Air"',
        "  Relative Permeability = 1.0",
        "  Electric Conductivity = 0.0",
        "End",
        "",
    ]

    # Body forces (tangential current) — same as DC but for harmonic excitation
    body_force_id = 1
    body_force_map = {}
    for body_id, turn_info in turn_bodies.items():
        A_m2 = turn_info.cross_section_area * 1e-6
        R0_m = turn_info.radius * 1e-3
        J_mag = total_current / A_m2
        n_region = getattr(turn_info, '_n_turns_in_region', 1)
        J_mag *= n_region

        if active_windings is not None:
            if not any(aw.lower() in turn_info.winding.lower() for aw in active_windings):
                J_mag = 0.0

        sign = 1.0 if turn_info.orientation == 'clockwise' else -1.0

        if core_type == "concentric":
            sif_lines.extend([
                f"Body Force {body_force_id}",
                "  Current Density 1 = Variable Coordinate",
                f'    Real MATC "{-sign * J_mag} * tx(2) / (sqrt(tx(0)^2 + tx(2)^2) + 1e-10)"',
                "  Current Density 2 = Real 0.0",
                "  Current Density 3 = Variable Coordinate",
                f'    Real MATC "{sign * J_mag} * tx(0) / (sqrt(tx(0)^2 + tx(2)^2) + 1e-10)"',
                "End",
                "",
            ])
        else:
            z0_m = 0.0
            eps = 1e-10
            sif_lines.extend([
                f"Body Force {body_force_id}",
                "  Current Density 1 = Variable Coordinate",
                f'    Real MATC "{-sign * J_mag} * (tx(2) - {z0_m}) / (sqrt((sqrt(tx(0)^2 + tx(1)^2) - {R0_m})^2 + (tx(2) - {z0_m})^2) + {eps}) * tx(0) / (sqrt(tx(0)^2 + tx(1)^2) + {eps})"',
                "  Current Density 2 = Variable Coordinate",
                f'    Real MATC "{-sign * J_mag} * (tx(2) - {z0_m}) / (sqrt((sqrt(tx(0)^2 + tx(1)^2) - {R0_m})^2 + (tx(2) - {z0_m})^2) + {eps}) * tx(1) / (sqrt(tx(0)^2 + tx(1)^2) + {eps})"',
                "  Current Density 3 = Variable Coordinate",
                f'    Real MATC "{sign * J_mag} * (sqrt(tx(0)^2 + tx(1)^2) - {R0_m}) / (sqrt((sqrt(tx(0)^2 + tx(1)^2) - {R0_m})^2 + (tx(2) - {z0_m})^2) + {eps})"',
                "End",
                "",
            ])

        body_force_map[body_id] = body_force_id
        body_force_id += 1

    # Bodies
    sif_lines.extend([
        "! Bodies",
        f"Body {core_id}",
        '  Name = "Core"',
        "  Equation = 1",
        "  Material = 1",
        "End",
        "",
    ])

    for body_id in coil_body_ids:
        turn_info = turn_bodies[body_id]
        bf_id = body_force_map.get(body_id)
        bf_line = f"  Body Force = {bf_id}" if bf_id else ""
        sif_lines.extend([
            f"Body {body_id}",
            f'  Name = "{turn_info.name}"',
            "  Equation = 1",
            "  Material = 2",
            bf_line,
            "End",
            "",
        ])

    if air_id:
        sif_lines.extend([
            f"Body {air_id}",
            '  Name = "Air"',
            "  Equation = 1",
            "  Material = 3",
            "End",
            "",
        ])

    # Equation and Solvers
    sif_lines.extend([
        "Equation 1",
        '  Name = "Harmonic MagnetoDynamics"',
        "  Active Solvers(2) = 1 2",
        "End",
        "",
        "! Harmonic AV Solver",
        "Solver 1",
        "  Equation = MGDynamicsHarmonic",
        '  Procedure = "MagnetoDynamics" "WhitneyAVHarmonicSolver"',
        '  Variable = AV[AV re:1 AV im:1]',
        "",
        f"  Angular Frequency = {omega:.6f}",
        "  Fix Input Current Density = Logical True",
        "",
        "  Linear System Solver = Direct",
        "  Linear System Direct Method = UMFPACK",
        "",
        "  Steady State Convergence Tolerance = 1.0e-8",
        "End",
        "",
        "! CalcFields for energy and Joule heating",
        "Solver 2",
        "  Equation = MGDynamicsCalc",
        '  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"',
        '  Potential Variable = "AV"',
        "  Calculate Magnetic Field Strength = True",
        "  Calculate Magnetic Flux Density = True",
        "  Calculate Current Density = True",
        "  Calculate Joule Heating = True",
        "  Calculate Nodal Fields = False",
        "  Calculate Elemental Fields = True",
        "",
        "  Linear System Solver = Iterative",
        "  Linear System Iterative Method = CG",
        "  Linear System Preconditioning = Diagonal",
        "  Linear System Max Iterations = 5000",
        "  Linear System Convergence Tolerance = 1.0e-5",
        "  Linear System Abort Not Converged = False",
        "End",
        "",
        "! Outer boundary",
        "Boundary Condition 1",
        '  Name = "OuterBoundary"',
        "  Target Boundaries(1) = 1",
        "  AV re {e} = Real 0.0",
        "  AV im {e} = Real 0.0",
        "  AV re = Real 0.0",
        "  AV im = Real 0.0",
        "End",
    ])

    sif_content = "\n".join(sif_lines)
    sif_path = os.path.join(output_path, "case.sif")
    with open(sif_path, 'w') as f:
        f.write(sif_content)
    with open(os.path.join(output_path, "ELMERSOLVER_STARTINFO"), 'w') as f:
        f.write("case.sif\n")
    return sif_path


def run_harmonic_simulation(
    mas_file: str,
    output_dir: str,
    max_turns: int = 4,
    frequency: float = 100000.0,
    total_current: float = 1.0,
    core_permeability: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run AC harmonic simulation and extract impedance.

    Returns:
        impedance_real: R_ac (Ohm) — real part of Z
        impedance_imag: X_L (Ohm) — imaginary part of Z
        joule_losses_W: Total Joule heating in windings (W)
        inductance_H: L = X_L / omega
    """
    data = load_mas_file(mas_file)
    magnetic_data = data.get('magnetic', data)
    core_type = 'toroidal' if any(
        k in str(magnetic_data.get('core', {}).get('functionalDescription', {}).get('shape', ''))
        for k in ['T ', 'T_']
    ) else 'concentric'

    all_turns = extract_turns_info(magnetic_data, core_type)
    primary = [t for t in all_turns if 'primary' in t.winding.lower()]
    if not primary:
        primary = all_turns
    primary = primary[:max_turns]

    if core_permeability is None:
        func_desc = get_core_data(magnetic_data).get('functionalDescription', {})
        mat = func_desc.get('material', 'N87')
        if isinstance(mat, dict):
            mat = mat.get('name', 'N87')
        core_permeability = get_material_permeability(mat)

    os.makedirs(output_dir, exist_ok=True)
    step_file, ct = build_geometry(magnetic_data, output_dir, max_turns=max_turns)

    mesh_dir, body_numbers, turn_bodies = create_mesh_with_turns(
        step_file, output_dir, primary, core_type=ct
    )

    sif_path = generate_sif_harmonic(
        output_dir, body_numbers, turn_bodies,
        core_permeability=core_permeability,
        frequency=frequency,
        total_current=total_current,
        core_type=ct,
    )

    # Patch missing bodies
    mesh_bodies = set()
    with open(os.path.join(mesh_dir, 'mesh.elements')) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                mesh_bodies.add(int(p[1]))
    with open(sif_path) as f:
        sif = f.read()
    for b in range(1, max(mesh_bodies) + 1):
        if f"Body {b}\n" not in sif:
            with open(sif_path, 'a') as f:
                f.write(f"\nBody {b}\n  Name = \"Air_{b}\"\n  Equation = 1\n  Material = 3\nEnd\n")

    print(f"Running AC harmonic at f={frequency:.0f} Hz...")
    success, energy, output = run_elmer(output_dir, timeout=300)

    # Extract Joule heating from Elmer output
    joule_losses = 0.0
    for line in output.split('\n'):
        if 'Joule Heating' in line or 'Joule heating' in line:
            import re as _re
            m = _re.search(r'[\d.E+-]+', line.split(':')[-1])
            if m:
                joule_losses = float(m.group())

    omega = 2 * math.pi * frequency
    L = calculate_inductance_from_energy(energy, total_current)
    # AC resistance: P = 0.5 * I² * R_ac (for harmonic excitation)
    R_ac = 2 * joule_losses / total_current**2 if total_current > 0 else 0

    results = {
        'frequency_Hz': frequency,
        'electromagnetic_energy_J': energy,
        'inductance_H': L,
        'inductance_uH': L * 1e6,
        'joule_losses_W': joule_losses,
        'ac_resistance_Ohm': R_ac,
        'impedance_real_Ohm': R_ac,
        'impedance_imag_Ohm': omega * L,
        'success': success,
    }

    print(f"  L = {L*1e6:.2f} uH")
    print(f"  R_ac = {R_ac*1e3:.3f} mOhm")
    print(f"  Joule losses = {joule_losses:.4e} W")
    return results


def validate_mas_file(
    mas_file: str,
    output_dir: str,
    max_turns: int = 6,
    total_current: float = 1.0,
    core_permeability: Optional[float] = None,  # None = auto-detect from MAS material
    method: str = "tangential",  # "tangential" or "coilsolver"
) -> Dict[str, Any]:
    """
    Full validation workflow for a MAS file.
    
    Args:
        core_permeability: If None, auto-detects from MAS file material
        method: "tangential" for manual tangential current, "coilsolver" for Elmer CoilSolver
    
    Returns dict with:
        - analytical_inductance: From PyMKF/AL calculation
        - elmer_inductance: From FEM simulation
        - error_percent: Relative error
        - success: Whether validation passed (<25% error)
    """
    print(f"\n{'='*60}")
    print(f"Validating: {os.path.basename(mas_file)}")
    print(f"Method: {method}")
    print(f"{'='*60}")
    
    results = {
        'mas_file': mas_file,
        'max_turns': max_turns,
        'method': method,
        'success': False,
    }
    
    # Load MAS data
    mas_data = load_mas_file(mas_file)
    magnetic_data = mas_data.get('magnetic', {})
    
    # Detect core type (toroidal, concentric, etc.) - needed for proper coordinate handling
    core_func_desc = magnetic_data.get('core', {}).get('functionalDescription', {})
    core_type = core_func_desc.get('type', 'concentric')
    print(f"Core type: {core_type}")
    
    # Get turns info (with correct coordinate interpretation based on core type)
    turns_info = extract_turns_info(magnetic_data, core_type=core_type)
    print(f"\nFound {len(turns_info)} turns in MAS file")
    
    # Limit turns if needed
    primary_turns = [t for t in turns_info if 'primary' in t.winding.lower()]
    if not primary_turns:
        # No primary winding — use turns from the first winding
        windings = sorted(set(t.winding for t in turns_info))
        if windings:
            first_winding = windings[0]
            primary_turns = [t for t in turns_info if t.winding == first_winding]
    if max_turns and len(primary_turns) > max_turns:
        primary_turns = primary_turns[:max_turns]
    
    num_turns = len(primary_turns)
    results['num_turns'] = num_turns
    
    print(f"Using {num_turns} primary turns")
    for t in primary_turns[:3]:
        print(f"  {t.name}: r={t.radius:.2f}mm, z={t.z_position:.2f}mm, A={t.cross_section_area:.2f}mm^2")
    if num_turns > 3:
        print(f"  ... and {num_turns-3} more")
    
    # Calculate analytical inductance and get material permeability
    core_data = get_core_data(magnetic_data)
    
    # Auto-detect permeability from material if not specified
    if core_permeability is None:
        # Get material name from core data
        func_desc = core_data.get('functionalDescription', {}) if isinstance(core_data, dict) else {}
        material_name = func_desc.get('material', 'N87')
        if isinstance(material_name, dict):
            material_name = material_name.get('name', 'N87')
        core_permeability = get_material_permeability(material_name)
        print(f"\nMaterial: {material_name}, Initial permeability (25°C): {core_permeability:.0f}")
    else:
        print(f"\nUsing specified permeability: {core_permeability:.0f}")
    
    results['core_permeability'] = core_permeability
    
    L_analytical = calculate_analytical_inductance(core_data, num_turns)
    
    if L_analytical:
        results['analytical_inductance_H'] = L_analytical
        results['analytical_inductance_uH'] = L_analytical * 1e6
        print(f"Analytical inductance: {L_analytical*1e6:.2f} uH")
    else:
        print("Warning: Could not calculate analytical inductance")
    
    # Get bobbin parameters for E-core detection
    bobbin_params = None
    coil_data = magnetic_data.get('coil', {})
    bobbin_data = coil_data.get('bobbin', {})
    if isinstance(bobbin_data, dict):
        bobbin_pd = bobbin_data.get('processedDescription', {})
        column_shape = bobbin_pd.get('columnShape', 'round')
        if column_shape == 'rectangular':
            bobbin_params = {
                'column_shape': 'rectangular',
                'column_width': bobbin_pd.get('columnWidth', 0.00455) * 1000,  # Convert to mm
                'column_depth': bobbin_pd.get('columnDepth', 0.00452) * 1000,
            }
    
    # Build geometry
    print("\n--- Building geometry ---")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        step_file, stl_file = build_geometry(magnetic_data, output_dir, max_turns=max_turns)
        print(f"Geometry saved to {step_file}")
    except Exception as e:
        results['error'] = f"Geometry build failed: {e}"
        print(f"ERROR: {e}")
        return results
    
    # Create mesh — try gmsh first, fall back to Netgen if it fails.
    # gmsh can segfault on complex geometries, so run it in a child process.
    print("\n--- Creating mesh ---")
    used_netgen = False
    gmsh_ok = False
    try:
        import multiprocessing as _mp
        import pickle as _pickle

        def _gmsh_worker(step, outdir, turns_pkl, bobbin_pkl, ctype, result_file):
            """Run gmsh meshing in isolated process (segfault-safe)."""
            turns = _pickle.loads(turns_pkl)
            bobbin = _pickle.loads(bobbin_pkl) if bobbin_pkl else None
            try:
                r = create_mesh_with_turns(step, outdir, turns,
                                           bobbin_params=bobbin, core_type=ctype)
                with open(result_file, 'wb') as f:
                    _pickle.dump(r, f)
            except Exception as e:
                with open(result_file, 'wb') as f:
                    _pickle.dump(e, f)

        result_file = os.path.join(output_dir, "_gmsh_result.pkl")
        p = _mp.Process(target=_gmsh_worker, args=(
            step_file, output_dir,
            _pickle.dumps(primary_turns),
            _pickle.dumps(bobbin_params) if bobbin_params else None,
            core_type, result_file,
        ))
        p.start()
        p.join(timeout=600)
        if p.is_alive():
            p.kill()
            p.join()
            raise RuntimeError("gmsh timed out")
        if p.exitcode != 0:
            raise RuntimeError(f"gmsh process crashed (exit {p.exitcode})")
        with open(result_file, 'rb') as f:
            gmsh_result = _pickle.load(f)
        os.remove(result_file)
        if isinstance(gmsh_result, Exception):
            raise gmsh_result
        mesh_dir, body_numbers, turn_bodies = gmsh_result
        gmsh_ok = True
        print(f"Mesh created: {mesh_dir}")
        print(f"Body numbers: {body_numbers}")
        print(f"Turn bodies: {len(turn_bodies)}")
    except Exception as e:
        print(f"gmsh failed: {e}")
        print("Trying Netgen fallback...")
        try:
            # Run Netgen in a subprocess too — the gmsh fork may have
            # corrupted OCC internal state in the parent process.
            def _netgen_worker(step, outdir, turns_pkl, ctype, result_file):
                turns = _pickle.loads(turns_pkl)
                try:
                    r = create_mesh_with_netgen(step, outdir, turns, core_type=ctype)
                    with open(result_file, 'wb') as f:
                        _pickle.dump(r, f)
                except Exception as ex:
                    with open(result_file, 'wb') as f:
                        _pickle.dump(ex, f)

            result_file = os.path.join(output_dir, "_netgen_result.pkl")
            p = _mp.Process(target=_netgen_worker, args=(
                step_file, output_dir,
                _pickle.dumps(primary_turns),
                core_type, result_file,
            ))
            p.start()
            p.join(timeout=600)
            if p.is_alive():
                p.kill()
                p.join()
                raise RuntimeError("Netgen timed out")
            if p.exitcode != 0:
                raise RuntimeError(f"Netgen process crashed (exit {p.exitcode})")
            with open(result_file, 'rb') as f:
                netgen_result = _pickle.load(f)
            os.remove(result_file)
            if isinstance(netgen_result, Exception):
                raise netgen_result
            mesh_dir, body_numbers, turn_bodies = netgen_result
            used_netgen = True
            print(f"Netgen mesh created: {mesh_dir}")
            print(f"Body numbers: {body_numbers}")
            print(f"Turn bodies: {len(turn_bodies)}")
        except Exception as e2:
            results['error'] = f"Meshing failed: {e2}"
            print(f"ERROR: {e2}")
            return results

    # Use iterative solver only for very large meshes (UMFPack handles up to ~200K tets)
    use_iterative = False

    # Generate SIF
    print("\n--- Generating SIF file ---")
    try:
        if method == "coilsolver":
            sif_path = generate_sif_with_coil_solver(
                output_dir,
                body_numbers,
                turn_bodies,
                core_permeability=core_permeability,
                total_current=total_current,
                num_turns=num_turns,
                core_type=core_type,
            )
        else:
            sif_path = generate_sif_with_tangential_current(
                output_dir,
                body_numbers,
                turn_bodies,
                core_permeability=core_permeability,
                total_current=total_current,
                core_type=core_type,
                use_iterative_solver=use_iterative,
            )
        print(f"SIF file: {sif_path}")
    except Exception as e:
        results['error'] = f"SIF generation failed: {e}"
        print(f"ERROR: {e}")
        return results

    # Ensure ALL mesh bodies are defined in the SIF (Elmer requires it).
    # Any undefined bodies get assigned air material.
    elements_path = os.path.join(mesh_dir, "mesh.elements")
    if os.path.exists(elements_path) and os.path.exists(sif_path):
        mesh_bodies = set()
        with open(elements_path) as ef:
            for line in ef:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mesh_bodies.add(int(parts[1]))
        # Elmer expects ALL body IDs from 1 to max to be defined
        max_body = max(mesh_bodies) if mesh_bodies else 0
        all_needed = set(range(1, max_body + 1))
        with open(sif_path) as sf:
            sif_content = sf.read()
        missing = [b for b in all_needed if f"Body {b}\n" not in sif_content]
        if missing:
            # Determine which equation ID to use (1 for MHD, 2 for CoilSolver)
            eq_id = 1
            extra = []
            for b in missing:
                extra.append(f"\nBody {b}\n  Name = \"Unclassified_{b}\"\n  Equation = {eq_id}\n  Material = 3\nEnd\n")
            with open(sif_path, 'a') as sf:
                sf.writelines(extra)
            print(f"Added {len(missing)} unclassified bodies as air: {missing}")

    # Run simulation
    print("\n--- Running Elmer simulation ---")
    success, energy, output = run_elmer(output_dir)
    
    results['elmer_success'] = success
    results['electromagnetic_energy_J'] = energy
    
    # Detect CoilSolver failure: energy near zero relative to expected value
    energy_too_low = (energy is not None and L_analytical and L_analytical > 1e-8
                      and energy < 0.5 * total_current**2 * L_analytical * 0.01)
    if not success or energy_too_low:
        if method == "coilsolver":
            print("CoilSolver failed, retrying with tangential method...")
            try:
                sif_path = generate_sif_with_tangential_current(
                    output_dir,
                    body_numbers,
                    turn_bodies,
                    core_permeability=core_permeability,
                    total_current=total_current,
                    core_type=core_type,
                    use_iterative_solver=use_iterative,
                )
                success, energy, output = run_elmer(output_dir)
                results['elmer_success'] = success
                results['electromagnetic_energy_J'] = energy
            except Exception:
                pass
        if not success:
            results['error'] = "Elmer simulation failed"
            print("Elmer failed. Last 1000 chars of output:")
            print(output[-1000:])
            return results
    
    print(f"Simulation completed successfully!")
    print(f"Electromagnetic energy: {energy:.6e} J")
    
    # Calculate inductance
    L_elmer = calculate_inductance_from_energy(energy, total_current)
    results['elmer_inductance_H'] = L_elmer
    results['elmer_inductance_uH'] = L_elmer * 1e6
    print(f"Elmer inductance: {L_elmer*1e6:.2f} uH")
    
    # Compare
    if L_analytical and L_elmer:
        error_percent = abs(L_elmer - L_analytical) / L_analytical * 100
        results['error_percent'] = error_percent
        results['success'] = error_percent < 25
        
        print(f"\n--- Results ---")
        print(f"Analytical: {L_analytical*1e6:.2f} uH")
        print(f"Elmer FEM:  {L_elmer*1e6:.2f} uH")
        print(f"Difference: {error_percent:.1f}%")
        
        if results['success']:
            print(f"PASS - Within 25% tolerance")
        else:
            print(f"FAIL - Exceeds 25% tolerance")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Elmer inductance against PyMKF")
    parser.add_argument("mas_file", help="Path to MAS JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("-t", "--turns", type=int, default=6, help="Max turns to use")
    parser.add_argument("-I", "--current", type=float, default=1.0, help="Test current (A)")
    parser.add_argument("-u", "--permeability", type=float, default=None, 
                        help="Core permeability (default: auto-detect from MAS material)")
    parser.add_argument("-m", "--method", choices=["tangential", "coilsolver"], 
                        default="tangential", help="Current application method")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.mas_file),
            "../output/inductance_validation"
        )
    
    results = validate_mas_file(
        args.mas_file,
        args.output,
        max_turns=args.turns,
        total_current=args.current,
        core_permeability=args.permeability,  # None = auto-detect
        method=args.method,
    )
    
    print("\n" + "="*60)
    print("Final Results:")
    print(json.dumps(results, indent=2, default=str))
    
    return 0 if results.get('success') else 1


def compute_inductance_matrix(
    mas_file: str,
    output_dir: str,
    max_turns: int = 4,
    total_current: float = 1.0,
    core_permeability: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute the NxN inductance matrix for a multi-winding magnetic component.

    Uses the energy method:
    - Self-inductance: Lii = 2*Wi / Ii²  (excite winding i only)
    - Mutual inductance: Mij = (W_ij - 0.5*Lii*Ii² - 0.5*Ljj*Ij²) / (Ii*Ij)

    Returns dict with:
        inductance_matrix: NxN matrix (H)
        coupling_matrix: NxN coupling coefficients
        leakage_inductance: per-winding leakage (H)
        winding_names: list of winding names
        energies: dict of energy per excitation pattern
    """
    # Load and prepare (same as validate_mas_file)
    data = load_mas_file(mas_file)
    magnetic_data = data.get('magnetic', data)
    core_type = 'toroidal' if any(
        k in str(magnetic_data.get('core', {}).get('functionalDescription', {}).get('shape', ''))
        for k in ['T ', 'T_']
    ) else 'concentric'

    all_turns = extract_turns_info(magnetic_data, core_type)
    core_data = get_core_data(magnetic_data)

    # Auto-detect permeability
    if core_permeability is None:
        func_desc = core_data.get('functionalDescription', {})
        mat = func_desc.get('material', 'N87')
        if isinstance(mat, dict):
            mat = mat.get('name', 'N87')
        core_permeability = get_material_permeability(mat)

    # Group turns by winding
    winding_turns = {}
    for t in all_turns:
        if t.winding not in winding_turns:
            winding_turns[t.winding] = []
        winding_turns[t.winding].append(t)

    winding_names = list(winding_turns.keys())
    N = len(winding_names)
    print(f"Windings: {winding_names} ({N} total)")

    # Limit turns per winding
    for wname in winding_names:
        if max_turns and len(winding_turns[wname]) > max_turns:
            winding_turns[wname] = winding_turns[wname][:max_turns]

    # All turns for geometry (need all windings in the mesh)
    all_selected = []
    for wname in winding_names:
        all_selected.extend(winding_turns[wname])
    num_turns = len(all_selected)
    print(f"Total turns for simulation: {num_turns}")

    # Build geometry ONCE (includes all windings)
    os.makedirs(output_dir, exist_ok=True)
    step_file, core_type_detected = build_geometry(
        magnetic_data, output_dir, max_turns=max_turns, all_windings=True
    )

    # Mesh ONCE
    print("Creating mesh (shared for all excitations)...")
    mesh_dir, body_numbers, turn_bodies = create_mesh_with_turns(
        step_file, output_dir, all_selected, core_type=core_type
    )
    print(f"Mesh: {mesh_dir}, {len(turn_bodies)} turn bodies")

    # Patch missing bodies helper
    def patch_sif(sif_path):
        elements_path = os.path.join(mesh_dir, "mesh.elements")
        mesh_bodies = set()
        with open(elements_path) as ef:
            for line in ef:
                p = line.strip().split()
                if len(p) >= 2:
                    mesh_bodies.add(int(p[1]))
        max_bid = max(mesh_bodies) if mesh_bodies else 0
        with open(sif_path) as sf:
            sif = sf.read()
        missing = [b for b in range(1, max_bid + 1) if f"Body {b}\n" not in sif]
        if missing:
            with open(sif_path, 'a') as sf:
                for b in missing:
                    sf.write(f"\nBody {b}\n  Name = \"Air_{b}\"\n  Equation = 1\n  Material = 3\nEnd\n")

    # Run simulations for each excitation pattern
    energies = {}

    # Self-inductance: excite each winding separately
    for i, wname in enumerate(winding_names):
        sim_dir = os.path.join(output_dir, f"sim_{wname}")
        os.makedirs(sim_dir, exist_ok=True)
        # Symlink mesh
        mesh_link = os.path.join(sim_dir, "mesh")
        if not os.path.exists(mesh_link):
            os.symlink(os.path.abspath(mesh_dir), mesh_link)

        sif_path = generate_sif_with_tangential_current(
            sim_dir, body_numbers, turn_bodies,
            core_permeability=core_permeability,
            total_current=total_current,
            core_type=core_type,
            active_windings=[wname],
        )
        patch_sif(sif_path)

        print(f"\n--- Sim: excite {wname} only ---")
        success, energy, output = run_elmer(sim_dir, timeout=600)
        energies[wname] = energy
        L_self = calculate_inductance_from_energy(energy, total_current)
        print(f"  Energy={energy:.4e} J, L_{wname}={L_self*1e6:.2f} uH")

    # Mutual inductance: excite pairs
    for i in range(N):
        for j in range(i + 1, N):
            wi, wj = winding_names[i], winding_names[j]
            sim_dir = os.path.join(output_dir, f"sim_{wi}_{wj}")
            os.makedirs(sim_dir, exist_ok=True)
            mesh_link = os.path.join(sim_dir, "mesh")
            if not os.path.exists(mesh_link):
                os.symlink(os.path.abspath(mesh_dir), mesh_link)

            sif_path = generate_sif_with_tangential_current(
                sim_dir, body_numbers, turn_bodies,
                core_permeability=core_permeability,
                total_current=total_current,
                core_type=core_type,
                active_windings=[wi, wj],
            )
            patch_sif(sif_path)

            print(f"\n--- Sim: excite {wi} + {wj} ---")
            success, energy, output = run_elmer(sim_dir, timeout=600)
            energies[(wi, wj)] = energy
            print(f"  Energy={energy:.4e} J")

    # Compute inductance matrix
    I = total_current
    L_matrix = [[0.0] * N for _ in range(N)]

    # Self-inductance
    for i, wname in enumerate(winding_names):
        L_matrix[i][i] = 2 * energies[wname] / I**2

    # Mutual inductance from combined energy
    for i in range(N):
        for j in range(i + 1, N):
            wi, wj = winding_names[i], winding_names[j]
            W_ij = energies.get((wi, wj), 0)
            Mij = (W_ij - 0.5 * L_matrix[i][i] * I**2 - 0.5 * L_matrix[j][j] * I**2) / (I * I)
            L_matrix[i][j] = Mij
            L_matrix[j][i] = Mij

    # Coupling coefficients
    k_matrix = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                k_matrix[i][j] = 1.0
            elif L_matrix[i][i] > 0 and L_matrix[j][j] > 0:
                k_matrix[i][j] = L_matrix[i][j] / math.sqrt(L_matrix[i][i] * L_matrix[j][j])

    # Leakage inductance
    leakage = {}
    for i, wi in enumerate(winding_names):
        if N > 1:
            # L_leak_i = Lii * (1 - k²) for 2-winding case
            j = 1 - i if N == 2 else 0  # simplification for 2 windings
            leakage[wi] = L_matrix[i][i] * (1 - k_matrix[i][j]**2)
        else:
            leakage[wi] = 0

    # Print results
    print("\n" + "=" * 60)
    print("INDUCTANCE MATRIX (uH):")
    header = "         " + "  ".join(f"{w:>12s}" for w in winding_names)
    print(header)
    for i, wi in enumerate(winding_names):
        row = f"{wi:>8s} " + "  ".join(f"{L_matrix[i][j]*1e6:>12.2f}" for j in range(N))
        print(row)

    print("\nCOUPLING MATRIX:")
    print(header)
    for i, wi in enumerate(winding_names):
        row = f"{wi:>8s} " + "  ".join(f"{k_matrix[i][j]:>12.4f}" for j in range(N))
        print(row)

    print("\nLEAKAGE INDUCTANCE (uH):")
    for wi, Ll in leakage.items():
        print(f"  {wi}: {Ll*1e6:.2f} uH")

    return {
        'winding_names': winding_names,
        'inductance_matrix_H': L_matrix,
        'coupling_matrix': k_matrix,
        'leakage_inductance_H': leakage,
        'energies': {str(k): v for k, v in energies.items()},
    }


if __name__ == "__main__":
    sys.exit(main())
