"""
HDI Planar Transformer - Single Secondary Winding
A U-shaped wire around a single magnetic pole.
"""

import cadquery as cq
import os
import math

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# Pole dimensions
POLE_DIAMETER = 4.0        # mm - diameter of the magnetic pole
POLE_HEIGHT = 5.0          # mm - height of pole

# Wire dimensions
WIRE_THICKNESS = 0.5       # mm - thickness of wire (radial)
WIRE_HEIGHT = 0.5          # mm - height of wire
WIRE_CLEARANCE = 0.5       # mm - gap between pole surface and wire inner surface

# Secondary winding parameters
SECONDARY_STRAIGHT_LENGTH = 5.0  # mm - length of straight wire extensions
SECONDARY_Z_OFFSET = 2.0         # mm - offset between parallel secondaries

# Cube parameters
WIRE_OUTSIDE_CUBE = 1.0          # mm - length of straight wire sticking out of cube
CUBE_MARGIN = 1.0                # mm - extra margin around wires in all directions

# =============================================================================
# CALCULATED VALUES
# =============================================================================

POLE_RADIUS = POLE_DIAMETER / 2
WIRE_INNER_RADIUS = POLE_RADIUS + WIRE_CLEARANCE
WIRE_OUTER_RADIUS = WIRE_INNER_RADIUS + WIRE_THICKNESS
WIRE_CENTER_RADIUS = (WIRE_INNER_RADIUS + WIRE_OUTER_RADIUS) / 2

# Pole center at origin
pole_center = (0, 0)

# =============================================================================
# CREATE SECONDARY WINDING - U-shape
# =============================================================================

# Z position for the arc (at top of pole)
arc_z = POLE_HEIGHT - WIRE_HEIGHT

# Top 180° arc - half ring at the top
# Create full ring then cut to get top half (0° to 180°, the +Y side)
secondary_arc_full = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_center[0], pole_center[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Cut away bottom half (the -Y side)
arc_cut_box = (
    cq.Workplane("XY")
    .workplane(offset=arc_z - 0.1)
    .center(pole_center[0], pole_center[1] - WIRE_OUTER_RADIUS)
    .rect(WIRE_OUTER_RADIUS * 3, WIRE_OUTER_RADIUS * 2)
    .extrude(WIRE_HEIGHT + 0.2)
)

secondary_arc = secondary_arc_full.cut(arc_cut_box)

# Straight connections extending from arc ends along -Y direction
# Arc ends are at 0° (right side, +X) and 180° (left side, -X)

# Left end of arc (180°)
arc_left_x = pole_center[0] - WIRE_CENTER_RADIUS
arc_left_y = pole_center[1]

# Right end of arc (0°)
arc_right_x = pole_center[0] + WIRE_CENTER_RADIUS
arc_right_y = pole_center[1]

# Create straight wires going in -Y direction
secondary_left_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(arc_left_x, arc_left_y - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_right_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(arc_right_x, arc_right_y - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

# Combine into single U-shape secondary (first layer)
secondary_1 = (
    secondary_arc
    .union(secondary_left_wire)
    .union(secondary_right_wire)
)

print(f"Secondary 1: U-shape")
print(f"  Arc Z: {arc_z:.3f} mm")
print(f"  Straight wires: {SECONDARY_STRAIGHT_LENGTH} mm along -Y")

# =============================================================================
# CREATE SECOND SECONDARY WINDING - parallel in Z axis
# =============================================================================

arc_z_2 = arc_z - SECONDARY_Z_OFFSET

# Top 180° arc for second layer
secondary_arc_full_2 = (
    cq.Workplane("XY")
    .workplane(offset=arc_z_2)
    .center(pole_center[0], pole_center[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

arc_cut_box_2 = (
    cq.Workplane("XY")
    .workplane(offset=arc_z_2 - 0.1)
    .center(pole_center[0], pole_center[1] - WIRE_OUTER_RADIUS)
    .rect(WIRE_OUTER_RADIUS * 3, WIRE_OUTER_RADIUS * 2)
    .extrude(WIRE_HEIGHT + 0.2)
)

secondary_arc_2 = secondary_arc_full_2.cut(arc_cut_box_2)

# Straight wires for second layer
secondary_left_wire_2 = (
    cq.Workplane("XY")
    .workplane(offset=arc_z_2)
    .center(arc_left_x, arc_left_y - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_right_wire_2 = (
    cq.Workplane("XY")
    .workplane(offset=arc_z_2)
    .center(arc_right_x, arc_right_y - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

# Combine into second U-shape secondary
secondary_2 = (
    secondary_arc_2
    .union(secondary_left_wire_2)
    .union(secondary_right_wire_2)
)

print(f"Secondary 2: U-shape (parallel)")
print(f"  Arc Z: {arc_z_2:.3f} mm")

# =============================================================================
# COMBINE BOTH SECONDARIES
# =============================================================================

secondaries = secondary_1.union(secondary_2)

# =============================================================================
# CREATE CUBE COVERING THE U-SHAPES
# =============================================================================
# The cube should cover the arc and most of the straight wires,
# with only WIRE_OUTSIDE_CUBE length sticking out

# Calculate cube dimensions
# Width: needs to cover the full arc diameter + margin
cube_width = WIRE_OUTER_RADIUS * 2 + WIRE_THICKNESS + 2 * CUBE_MARGIN

# Height (Y): covers from arc center to where wires exit + margin
# Wires go from arc_left_y (=0) to arc_left_y - SECONDARY_STRAIGHT_LENGTH
# We want WIRE_OUTSIDE_CUBE to stick out, so cube covers up to:
#   arc_left_y - (SECONDARY_STRAIGHT_LENGTH - WIRE_OUTSIDE_CUBE)
wire_inside_cube = SECONDARY_STRAIGHT_LENGTH - WIRE_OUTSIDE_CUBE
cube_height = WIRE_OUTER_RADIUS + wire_inside_cube + 2 * CUBE_MARGIN

# Depth (Z): covers both secondary layers + margin
cube_z_bottom = arc_z_2 - CUBE_MARGIN
cube_z_top = arc_z + WIRE_HEIGHT + CUBE_MARGIN
cube_depth = cube_z_top - cube_z_bottom

# Cube center position
cube_center_x = pole_center[0]
cube_center_y = pole_center[1] + WIRE_OUTER_RADIUS/2 - wire_inside_cube/2 + CUBE_MARGIN/2

cube = (
    cq.Workplane("XY")
    .workplane(offset=cube_z_bottom)
    .center(cube_center_x, cube_center_y)
    .rect(cube_width, cube_height)
    .extrude(cube_depth)
)

print(f"\nCube dimensions: {cube_width:.2f} x {cube_height:.2f} x {cube_depth:.2f} mm")
print(f"  Wire outside cube: {WIRE_OUTSIDE_CUBE} mm")

# =============================================================================
# CREATE FERRITE (CUBE WITH WIRE CHANNELS)
# =============================================================================

# Subtract windings from cube to create channels in the ferrite
ferrite = cube.cut(secondaries)

print("Ferrite: cube with wire channels cut out")

# =============================================================================
# EXPORT
# =============================================================================

os.makedirs("output", exist_ok=True)

# Export secondaries only
cq.exporters.export(secondaries, "output/hdi_secondary_single.step")
print("\nExported: output/hdi_secondary_single.step (windings only)")

# Export ferrite (cube with channels)
cq.exporters.export(ferrite, "output/hdi_secondary_ferrite.step")
print("Exported: output/hdi_secondary_ferrite.step (ferrite with wire channels)")

# Export assembly (ferrite + windings as separate bodies)
assembly = cq.Assembly()
assembly.add(ferrite, name="ferrite", color=cq.Color(0.3, 0.3, 0.3, 1))  # dark gray
assembly.add(secondaries, name="windings", color=cq.Color(0.8, 0.5, 0.2, 1))  # copper
assembly.save("output/hdi_secondary_assembly.step", "STEP")
print("Exported: output/hdi_secondary_assembly.step (ferrite + windings)")

print(f"\nWire clearance: {WIRE_CLEARANCE} mm")
print(f"Wire thickness: {WIRE_THICKNESS} mm")
print(f"Wire height: {WIRE_HEIGHT} mm")
print(f"Z offset between layers: {SECONDARY_Z_OFFSET} mm")
