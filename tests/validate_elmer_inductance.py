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
    import PyMKF
    HAS_PYMKF = True
except ImportError:
    HAS_PYMKF = False
    print("Warning: PyMKF not available")

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
    
    if not coil.get('turnsDescription'):
        raise ValueError(
            f"MAS file missing 'turnsDescription' in coil. "
            f"File has functionalDescription with {len(coil.get('functionalDescription', []))} windings. "
            f"Use PyMKF or OpenMagnetics web tools to wind the coil first."
        )
    
    if not core.get('geometricalDescription'):
        raise ValueError(
            f"MAS file missing 'geometricalDescription' in core. "
            f"Use PyMKF or OpenMagnetics web tools to process the core first."
        )
    
    print(f"Loaded MAS file with {len(coil.get('turnsDescription', []))} turns")
    
    return data


def get_core_data(magnetic_data: Dict) -> Dict:
    """Get processed core data using PyMKF."""
    import json as json_module
    core = magnetic_data.get('core')
    if core is None:
        raise ValueError("Missing 'core' in magnetic_data")
    if not isinstance(core, dict):
        raise ValueError(f"'core' must be a dict, got {type(core)}")
    
    if core.get('geometricalDescription') is None:
        if not HAS_PYMKF:
            raise ImportError("PyMKF required to process core without geometricalDescription")
        
        result = PyMKF.calculate_core_data(core, True)
        # PyMKF returns JSON string
        if isinstance(result, str):
            if result.startswith('Exception:'):
                raise ValueError(f"PyMKF.calculate_core_data failed: {result}")
            core = json_module.loads(result)
        elif isinstance(result, dict):
            core = result
        else:
            raise ValueError(f"Unexpected PyMKF result type: {type(result)}")
        
        if not core.get('geometricalDescription'):
            raise ValueError("PyMKF.calculate_core_data did not produce geometricalDescription")
    
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
            # For toroidal: coords are [x, z] in XZ plane
            # Radial distance from Y axis (core axis)
            x = coords[0] * 1000  # mm
            z = coords[1] * 1000 if len(coords) > 1 else 0  # mm
            radius = math.sqrt(x**2 + z**2)  # Distance from Y axis
            z_pos = z  # Use Z for position identification
            x_pos = x  # Store X for Coil Normal calculation
        else:
            # For concentric: coords are [radial, z]
            radius = coords[0] * 1000  # Convert to mm
            z_pos = coords[1] * 1000 if len(coords) > 1 else 0
            x_pos = 0.0  # Not used for concentric
        
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
            mat_data = PyMKF.get_material_data(material_name)
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
                   max_turns: Optional[int] = None) -> Tuple[str, str]:
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
            # Keep only first max_turns from primary winding
            primary_turns = [t for t in turns if 'primary' in t.get('winding', '').lower()]
            coil['turnsDescription'] = primary_turns[:max_turns]
            magnetic_data = {**magnetic_data, 'coil': coil}
    
    # Build with MVB
    builder = Builder()
    result = builder.get_magnetic(
        magnetic_data,
        project_name="magnetic",
        output_path=output_path,
        export_files=True
    )
    
    if isinstance(result, tuple):
        return result
    else:
        step_path = os.path.join(output_path, "magnetic.step")
        stl_path = os.path.join(output_path, "magnetic.stl")
        return step_path, stl_path


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
        
        if not skip_standard_classification:
            # First pass: identify turns by matching to MAS turn positions
            turn_position_tolerance = 2.0  # mm tolerance for position matching
            matched_turn_tags = set()
            
            for dim, tag, actual_vol, com in all_volumes:
                center_z = com[2]
                # Calculate radius from Z axis (for concentric/toroidal cores)
                radius_from_axis = math.sqrt(com[0]**2 + com[1]**2)
                
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
                center_z = com[2]
                
                if tag in matched_turn_tags:
                    # This is a turn - find best match by z-position
                    best_match = None
                    best_dist = float('inf')
                    for ti in turns_info:
                        dist = abs(ti.z_position - center_z)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = ti
                    
                    turn_tags.append((tag, best_match))
                    print(f"  Volume {tag}: TURN z={center_z:.2f}mm, vol={actual_vol:.0f}mm³ -> {best_match.name}")
                
                elif actual_vol >= core_vol_threshold and len(core_tags) < 2:
                    # Large volume, likely core (allow up to 2 pieces for half-set cores)
                    core_tags.append(tag)
                    print(f"  Volume {tag}: CORE (vol={actual_vol:.0f}mm³)")
                
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
                    # Find matching turn by position
                    center_radial = math.sqrt(com[0]**2 + com[2]**2)
                    best_match = None
                    best_dist = float('inf')
                    for turn_info in turns_info:
                        dist = abs(turn_info.radius - center_radial)
                        if dist < best_dist:
                            best_dist = dist
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
            # Include all volumes (core, bobbin, turns) in fragmentation for proper conformal mesh
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
            
            # Sort by volume (excluding air - identified by large bbox)
            solid_vols = [v for v in vol_info if v['bbox_vol'] < 50000]
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
                center_z = v['com'][2]
                bbox_vol = v['bbox_vol']
                
                if bbox_vol > 50000:  # Large bounding box = air region
                    new_air.append(tag)
                elif actual_vol >= core_vol_threshold and len(new_core) < 2:
                    # Core piece (up to 2 for split cores)
                    new_core.append(tag)
                elif actual_vol > expected_turn_vol * 3 if expected_turn_vol > 0 else actual_vol > 100:
                    # Bobbin or other intermediate volume - treat as air
                    new_air.append(tag)
                else:  # Small volume - could be turn or fragment
                    # Find matching turn info by z-position
                    best_match = None
                    best_dist = float('inf')
                    for ti in turns_info:
                        dist = abs(ti.z_position - center_z)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = ti
                    
                    # Check if volume is similar to expected turn volume (within 3x)
                    vol_ratio = actual_vol / expected_turn_vol if expected_turn_vol > 0 else 0
                    is_turn_sized = 0.3 < vol_ratio < 3.0 if expected_turn_vol > 0 else False
                    
                    if best_match and best_dist < 2.0 and is_turn_sized:
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
        
        # Mesh settings - use finer mesh for toroidal cores with small turns
        actual_max_size = max_element_size
        actual_min_size = min_element_size
        if core_type == "toroidal" and turns_info:
            # For toroidal turns, use fixed fine mesh sizes that work reliably
            actual_max_size = 2.0  # Fixed value that works
            actual_min_size = 0.2  # Fixed value that works
            print(f"  Using toroidal mesh sizes: max={actual_max_size:.2f}mm, min={actual_min_size:.2f}mm")
        
        gmsh.option.setNumber("Mesh.MeshSizeMax", actual_max_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", actual_min_size)
        
        # Use HXT algorithm for toroidal (handles thin volumes better)
        # Use Delaunay for concentric cores
        if core_type == "toroidal":
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        else:
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        
        # Generate mesh
        # Note: Don't call removeDuplicateNodes/Elements for toroidal - it can destroy volumes
        print("\nGenerating mesh...")
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:
            print(f"Warning: Initial mesh generation failed: {e}")
            print("Retrying with coarser settings...")
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_element_size * 2)
            gmsh.model.mesh.generate(3)
        
        # Verify 3D elements were created
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        total_3d = sum(len(t) for t in elem_tags) if elem_tags else 0
        if total_3d == 0:
            raise RuntimeError("No 3D elements generated - mesh failed")
        print(f"Generated {total_3d} 3D elements")
        
        # Save
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        msh_path = os.path.join(output_path, "mesh.msh")
        gmsh.write(msh_path)
        print(f"Mesh saved to {msh_path}")
        
    finally:
        gmsh.finalize()
    
    # Convert to Elmer format
    elmer_grid = os.path.expanduser("~/elmer/install/bin/ElmerGrid")
    mesh_dir = os.path.join(output_path, "mesh")
    
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/elmer/install/bin") + ":" + env.get("PATH", "")
    
    cmd = [elmer_grid, "14", "2", "mesh.msh", "-autoclean", 
           "-scale", "0.001", "0.001", "0.001"]  # mm to m
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_path, env=env)
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
    
    # Component definition for each turn as a separate coil
    # This allows CoilSolver to handle each closed loop correctly
    comp_id = 1
    for body_id in coil_body_ids:
        turn_info = turn_bodies[body_id]
        
        # Calculate Coil Normal based on core type and turn position
        if core_type == "toroidal":
            # For toroidal: turn position is in XZ plane, Y is core axis
            # The turn wraps around the toroidal cross-section (poloidal)
            # Coil Normal should point in toroidal direction (along major circumference)
            x = turn_info.x_position  # x in mm (stored from MAS coordinates)
            z = turn_info.z_position  # z in mm
            
            # Normalize to get unit vector in XZ plane
            r_actual = math.sqrt(x**2 + z**2) if (x**2 + z**2) > 0 else 1.0
            
            # Toroidal direction is tangent to major circumference
            # At point (x, 0, z), tangent is (-z/r, 0, x/r)
            nx = -z / r_actual
            ny = 0.0
            nz = x / r_actual
            coil_normal = f"  Coil Normal(3) = Real {nx:.6f} {ny:.6f} {nz:.6f}"
        else:
            # For concentric (PQ, E-core): turns are around Z axis
            coil_normal = "  Coil Normal(3) = Real 0.0 0.0 1.0"
        
        sif_lines.extend([
            f"Component {comp_id}",
            f'  Name = String "Coil_{turn_info.name}"',
            '  Coil Type = String "test"',
            f"  Master Bodies(1) = Integer {body_id}",
            f"  Desired Current Density = Real {total_current / (turn_info.cross_section_area * 1e-6):.1f}",
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
    
    # Equations
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
        "  ! Use direct solver for robustness with high permeability contrast",
        "  Linear System Solver = Direct",
        "  Linear System Direct Method = UMFPack",
        "",
        "  Nonlinear System Max Iterations = 1",
        "End",
        "",
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
        "  Linear System Preconditioning = ILU0",
        "  Linear System Max Iterations = 5000",
        "  Linear System Convergence Tolerance = 1.0e-8",
        "",
        "  Nonlinear System Consistent Norm = True",
        "  Discontinuous Bodies = True",
        "End",
        "",
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
        f"  Relative Permeability = {core_permeability}",
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
    ]
    
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
        z0_m = turn_info.z_position * 1e-3  # m (z position of turn center)
        
        J_mag = total_current / A_m2  # A/m^2
        
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
        
        # TANGENTIAL CURRENT around Z-axis (original approach)
        # 
        # For a toroidal turn centered at R0 from Z-axis:
        # Current flows tangentially around Z (φ direction)
        # J = (I/A) * tangential_direction
        # 
        # Tangential direction at (x,y): (-y/r, x/r, 0) where r = sqrt(x²+y²)
        # 
        # This creates H-field circling around Z-axis (like a solenoid's field)
        # which is what we want for flux through the core legs.
        
        sif_lines.extend([
            f"Body Force {body_force_id}",
            f'  Name = "Current_{turn_info.name}"',
            f"  ! Tangential current around Z-axis",
            f"  ! Turn at R0 = {R0_m*1000:.2f} mm, z0 = {z0_m*1000:.2f} mm",
            f"  ! Cross-section area: A = {A_m2*1e6:.2f} mm^2",
            f"  ! Current density: |J| = I/A = {J_mag:.4e} A/m^2",
            f"  ! orientation = {turn_info.orientation}",
            "  ! Tangential: Jx = -J*y/r, Jy = J*x/r, Jz = 0",
            "  Current Density 1 = Variable Coordinate",
            f'    Real MATC "{-sign * J_mag} * tx(1) / (sqrt(tx(0)^2 + tx(1)^2) + 1e-10)"',
            "  Current Density 2 = Variable Coordinate",
            f'    Real MATC "{sign * J_mag} * tx(0) / (sqrt(tx(0)^2 + tx(1)^2) + 1e-10)"',
            "  Current Density 3 = Real 0.0",
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
        "  Linear System Solver = Direct",
        "  Linear System Direct Method = UMFPACK",
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
        "  Linear System Solver = Direct",
        "  Linear System Direct Method = UMFPACK",
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
    
    # Create mesh
    print("\n--- Creating mesh ---")
    try:
        mesh_dir, body_numbers, turn_bodies = create_mesh_with_turns(
            step_file, output_dir, primary_turns, bobbin_params=bobbin_params,
            core_type=core_type
        )
        print(f"Mesh created: {mesh_dir}")
        print(f"Body numbers: {body_numbers}")
        print(f"Turn bodies: {len(turn_bodies)}")
    except Exception as e:
        results['error'] = f"Meshing failed: {e}"
        print(f"ERROR: {e}")
        return results
    
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
            )
        print(f"SIF file: {sif_path}")
    except Exception as e:
        results['error'] = f"SIF generation failed: {e}"
        print(f"ERROR: {e}")
        return results
    
    # Run simulation
    print("\n--- Running Elmer simulation ---")
    success, energy, output = run_elmer(output_dir)
    
    results['elmer_success'] = success
    results['electromagnetic_energy_J'] = energy
    
    if not success:
        results['error'] = "Elmer simulation failed"
        # Print last part of output
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


if __name__ == "__main__":
    sys.exit(main())
