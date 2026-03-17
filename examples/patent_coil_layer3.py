"""
3D Model of Patent US5525941 - Figure 2a Layer 3 (Dashed Line)
Step 6: Add internal tangent wire between top-left and top-right rings
"""

import cadquery as cq
import os
import math

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# Pole dimensions
POLE_DIAMETER = 4.0  # mm - diameter of each magnetic pole
POLE_HEIGHT = 5.0  # mm - height of poles

# Pole spacing
HORIZONTAL_SPACING = 10.0  # mm - horizontal distance between pole centers
VERTICAL_SPACING = 10.0  # mm - vertical distance between pole centers

# Plate dimensions (square plates closing the magnetic path)
PLATE_THICKNESS = 1.0  # mm - thickness of top and bottom plates
PLATE_MARGIN = 2.0  # mm - extra margin beyond the poles

# Wire dimensions
WIRE_THICKNESS = 0.5  # mm - thickness of wire (radial)
WIRE_HEIGHT = 0.5  # mm - height of wire
WIRE_CLEARANCE = 0.5  # mm - gap between pole surface and wire inner surface
LAYER_CLEARANCE = 0.1  # mm - vertical gap between adjacent layers

# Calculate Z offset between layer bottoms (wire height + clearance)
WIRE_Z_OFFSET = WIRE_HEIGHT + LAYER_CLEARANCE  # mm - z offset between adjacent layers

# Center all 3 winding layers vertically within poles
# Layers from bottom to top:
#   1. Primary -Z layer (bottom)
#   2. Primary +Z layer (middle)
#   3. Secondary layer (top)
#
# Total stack height = 3 * WIRE_HEIGHT + 2 * LAYER_CLEARANCE
# Center of stack should align with pole center

POLE_CENTER_Z = POLE_HEIGHT / 2
TOTAL_STACK_HEIGHT = 3 * WIRE_HEIGHT + 2 * LAYER_CLEARANCE
STACK_BOTTOM_Z = POLE_CENTER_Z - TOTAL_STACK_HEIGHT / 2

# Primary -Z layer is at bottom of stack
# Primary +Z layer is one step up
# Secondary layer is two steps up
WIRE_Z = STACK_BOTTOM_Z  # Z position for wire bottom (before layer offset)

# Secondary layer Z offset (above the primary +Z layer) - same clearance as between primaries
# Primary layers are at -WIRE_Z_OFFSET and +WIRE_Z_OFFSET, so gap between them is 2*WIRE_Z_OFFSET
# Secondary should have the same gap from primary +Z
SECONDARY_Z_OFFSET = 2 * WIRE_Z_OFFSET  # Same spacing as between primary layers

# Straight wire length
STRAIGHT_WIRE_LENGTH = 5.0  # mm - length of the straight wire segment

# Layer assignments
POSITIVE_Z_SEGMENTS = {8, 5, 9, 3, 10, 2}  # Segments on upper layer (+Z)
NEGATIVE_Z_SEGMENTS = {1, 4, 6, 7, 11, 12}  # Segments on lower layer (-Z)

# =============================================================================
# CALCULATED VALUES
# =============================================================================

POLE_RADIUS = POLE_DIAMETER / 2
WIRE_INNER_RADIUS = POLE_RADIUS + WIRE_CLEARANCE
WIRE_OUTER_RADIUS = WIRE_INNER_RADIUS + WIRE_THICKNESS
WIRE_CENTER_RADIUS = (WIRE_INNER_RADIUS + WIRE_OUTER_RADIUS) / 2

# Pole centers
pole_tl = (-HORIZONTAL_SPACING / 2, +VERTICAL_SPACING / 2)  # Top Left
pole_tr = (+HORIZONTAL_SPACING / 2, +VERTICAL_SPACING / 2)  # Top Right
pole_bl = (-HORIZONTAL_SPACING / 2, -VERTICAL_SPACING / 2)  # Bottom Left
pole_br = (+HORIZONTAL_SPACING / 2, -VERTICAL_SPACING / 2)  # Bottom Right

# =============================================================================
# CALCULATE EXTERNAL TANGENT POINTS (BL to TR)
# =============================================================================

# Vector from BL to TR
dx_ext = pole_tr[0] - pole_bl[0]
dy_ext = pole_tr[1] - pole_bl[1]
dist_ext = math.sqrt(dx_ext * dx_ext + dy_ext * dy_ext)

# Unit vector along the line
ux_ext = dx_ext / dist_ext
uy_ext = dy_ext / dist_ext

# Perpendicular unit vector (to the LEFT when going from BL to TR)
px_ext = uy_ext
py_ext = -ux_ext

# Tangent points
tangent_bl_x = pole_bl[0] + px_ext * WIRE_CENTER_RADIUS
tangent_bl_y = pole_bl[1] + py_ext * WIRE_CENTER_RADIUS

tangent_tr_x = pole_tr[0] + px_ext * WIRE_CENTER_RADIUS
tangent_tr_y = pole_tr[1] + py_ext * WIRE_CENTER_RADIUS

print(f"External tangent BL: ({tangent_bl_x:.3f}, {tangent_bl_y:.3f})")
print(f"External tangent TR: ({tangent_tr_x:.3f}, {tangent_tr_y:.3f})")

# =============================================================================
# CALCULATE INTERNAL TANGENT POINTS (TL to TR)
# =============================================================================
# Internal tangent: the tangent line CROSSES BETWEEN the two circles
#
# Setup: Two circles with equal radius r, centers C1 (TL) and C2 (TR), distance d apart.
# For horizontally aligned circles: C1 = (-d/2, 0), C2 = (d/2, 0) in local coords.
#
# The internal tangent line passes through the midpoint M = (0, 0).
# Let the tangent line have angle phi from horizontal.
#
# The tangent line equation: y = x * tan(phi), or: -x*sin(phi) + y*cos(phi) = 0
#
# Distance from C1 = (-d/2, 0) to this line:
#   |-(-d/2)*sin(phi) + 0*cos(phi)| / 1 = (d/2)*|sin(phi)|
#
# This must equal r:
#   (d/2)*sin(phi) = r   (taking positive phi for upward-going tangent)
#   sin(phi) = 2r/d
#   phi = arcsin(2r/d)
#
# Tangent points: radius is perpendicular to tangent line
# - On C1: tangent point is at angle (phi - pi/2) from center, BUT we need to check the sign
#   The tangent line is ABOVE C1 for positive phi, so the radius points UP toward the line.
#   Radius direction from C1 to tangent point = direction perpendicular to line, pointing toward line
#   Line direction: (cos(phi), sin(phi))
#   Perpendicular pointing "up" (toward positive y from origin): (-sin(phi), cos(phi))
#   Wait no - perpendicular to line pointing from C1 toward line...
#   C1 is at (-d/2, 0). Line passes through origin. Perpendicular from C1 to line points toward origin.
#   Unit vector from C1 toward origin: (1, 0) -- no this is wrong too.
#
# Let me just compute it directly:
# The tangent point T1 on circle C1 satisfies:
#   1. T1 is on circle: |T1 - C1| = r
#   2. T1 is on tangent line: T1 is on line through M with angle phi
#   3. C1-T1 is perpendicular to the tangent line
#
# From (3): T1 - C1 = r * (normal to tangent line)
# Normal to tangent line (unit vector): (-sin(phi), cos(phi)) or (sin(phi), -cos(phi))
#
# For C1 on the LEFT and tangent line going up-right through M:
# C1 is BELOW the tangent line (since line has positive slope through origin, and C1 is at (-d/2, 0))
# So the normal from C1 to the line points in (+sin(phi), +cos(phi)) direction? No wait...
#
# Tangent line: y = x*tan(phi). At x = -d/2: y_line = -d/2 * tan(phi) < 0 for positive phi.
# So the line is BELOW C1 at x = -d/2! That means C1 is ABOVE the tangent line.
# Normal from C1 DOWN to line: (sin(phi), -cos(phi))
# T1 = C1 + r * (sin(phi), -cos(phi))
#
# For C2 on the RIGHT:
# At x = d/2: y_line = d/2 * tan(phi) > 0. So C2 = (d/2, 0) is BELOW the line.
# Normal from C2 UP to line: (-sin(phi), cos(phi))
# T2 = C2 + r * (-sin(phi), cos(phi))

# Distance between TL and TR centers
dx_int = pole_tr[0] - pole_tl[0]
dy_int = pole_tr[1] - pole_tl[1]
dist_int = math.sqrt(dx_int * dx_int + dy_int * dy_int)

# Angle of center-line from TL to TR
centerline_angle = math.atan2(dy_int, dx_int)

# Angle phi of tangent line (in local coordinates where centerline is horizontal)
# sin(phi) = 2r/d
sin_phi = 2 * WIRE_CENTER_RADIUS / dist_int
phi = math.asin(sin_phi)

print(f"Distance TL-TR: {dist_int:.3f} mm")
print(f"Phi angle (local): {math.degrees(phi):.1f} degrees")

# Convert to global coordinates
# Local tangent line angle phi becomes (centerline_angle + phi) in global coords
tangent_line_angle = centerline_angle + phi

print(f"Tangent line angle (global): {math.degrees(tangent_line_angle):.1f} degrees")

# Tangent point on TL (C1, the left circle):
# In local coords: T1 = C1 + r * (sin(phi), -cos(phi))
# In global coords, rotate by centerline_angle:
# The offset vector (sin(phi), -cos(phi)) rotated by centerline_angle:
#   x' = sin(phi)*cos(ctr) - (-cos(phi))*sin(ctr) = sin(phi)*cos(ctr) + cos(phi)*sin(ctr) = sin(phi + ctr)
#   y' = sin(phi)*sin(ctr) + (-cos(phi))*cos(ctr) = sin(phi)*sin(ctr) - cos(phi)*cos(ctr) = -cos(phi + ctr)
# So offset = (sin(centerline_angle + phi), -cos(centerline_angle + phi))

offset_tl_x = math.sin(centerline_angle + phi)
offset_tl_y = -math.cos(centerline_angle + phi)
int_tangent_tl_x = pole_tl[0] + WIRE_CENTER_RADIUS * offset_tl_x
int_tangent_tl_y = pole_tl[1] + WIRE_CENTER_RADIUS * offset_tl_y

# Tangent point on TR (C2, the right circle):
# In local coords: T2 = C2 + r * (-sin(phi), cos(phi))
# Rotated by centerline_angle:
#   x' = -sin(phi)*cos(ctr) - cos(phi)*sin(ctr) = -sin(phi + ctr)
#   y' = -sin(phi)*sin(ctr) + cos(phi)*cos(ctr) = cos(phi + ctr)
# So offset = (-sin(centerline_angle + phi), cos(centerline_angle + phi))

offset_tr_x = -math.sin(centerline_angle + phi)
offset_tr_y = math.cos(centerline_angle + phi)
int_tangent_tr_x = pole_tr[0] + WIRE_CENTER_RADIUS * offset_tr_x
int_tangent_tr_y = pole_tr[1] + WIRE_CENTER_RADIUS * offset_tr_y

print(f"Internal tangent TL: ({int_tangent_tl_x:.3f}, {int_tangent_tl_y:.3f})")
print(f"Internal tangent TR: ({int_tangent_tr_x:.3f}, {int_tangent_tr_y:.3f})")

# Verify: the angle from tangent point TL to tangent point TR should equal tangent_line_angle
verify_angle = math.atan2(
    int_tangent_tr_y - int_tangent_tl_y, int_tangent_tr_x - int_tangent_tl_x
)
print(
    f"Verification - angle between tangent points: {math.degrees(verify_angle):.1f} degrees (should be {math.degrees(tangent_line_angle):.1f})"
)

# =============================================================================
# CALCULATE EXTERNAL TANGENT POINTS (TL to BR) - top side
# =============================================================================
# External tangent: both tangent points are on the SAME side of each circle
# For equal radius circles, the external tangent is parallel to the centerline,
# offset perpendicular by the wire center radius.
#
# We want the tangent on the TOP side (when going from TL to BR, offset to the left)

# Vector from TL to BR
dx_ext2 = pole_br[0] - pole_tl[0]
dy_ext2 = pole_br[1] - pole_tl[1]
dist_ext2 = math.sqrt(dx_ext2 * dx_ext2 + dy_ext2 * dy_ext2)

# Unit vector along the centerline (TL to BR)
ux_ext2 = dx_ext2 / dist_ext2
uy_ext2 = dy_ext2 / dist_ext2

# Perpendicular unit vector (to the LEFT when going from TL to BR)
# Going from TL (-5, 5) to BR (5, -5) is down-right, so LEFT is up-right
# Perpendicular to (ux, uy) rotated 90° CCW: (-uy, ux)
px_ext2 = -uy_ext2
py_ext2 = ux_ext2

print(f"\nExternal tangent TL-BR:")
print(f"  Centerline direction: ({ux_ext2:.3f}, {uy_ext2:.3f})")
print(f"  Perpendicular (left): ({px_ext2:.3f}, {py_ext2:.3f})")

# Tangent points - offset from pole centers by perpendicular * radius
ext2_tangent_tl_x = pole_tl[0] + px_ext2 * WIRE_CENTER_RADIUS
ext2_tangent_tl_y = pole_tl[1] + py_ext2 * WIRE_CENTER_RADIUS

ext2_tangent_br_x = pole_br[0] + px_ext2 * WIRE_CENTER_RADIUS
ext2_tangent_br_y = pole_br[1] + py_ext2 * WIRE_CENTER_RADIUS

print(f"  TL tangent point: ({ext2_tangent_tl_x:.3f}, {ext2_tangent_tl_y:.3f})")
print(f"  BR tangent point: ({ext2_tangent_br_x:.3f}, {ext2_tangent_br_y:.3f})")

# Verify angle matches centerline
ext2_verify_angle = math.atan2(
    ext2_tangent_br_y - ext2_tangent_tl_y, ext2_tangent_br_x - ext2_tangent_tl_x
)
ext2_centerline_angle = math.atan2(dy_ext2, dx_ext2)
print(
    f"  Verification - tangent angle: {math.degrees(ext2_verify_angle):.1f}° (should be {math.degrees(ext2_centerline_angle):.1f}°)"
)

# =============================================================================
# CALCULATE INTERNAL TANGENT POINTS (BL to BR) - top of BL to bottom of BR
# =============================================================================
# Internal tangent: the tangent line CROSSES BETWEEN the two circles
# Using the same math as TL-TR internal tangent.
#
# BL is on the left, BR is on the right (same horizontal level)
# We want: top of BL to bottom of BR (the other internal tangent)

# Distance between BL and BR centers
dx_int2 = pole_br[0] - pole_bl[0]  # from BL to BR (going right)
dy_int2 = pole_br[1] - pole_bl[1]
dist_int2 = math.sqrt(dx_int2 * dx_int2 + dy_int2 * dy_int2)

# Angle of center-line from BL to BR
centerline_angle2 = math.atan2(dy_int2, dx_int2)

# Angle phi of tangent line (in local coordinates where centerline is horizontal)
# sin(phi) = 2r/d
sin_phi2 = 2 * WIRE_CENTER_RADIUS / dist_int2
phi2 = math.asin(sin_phi2)

print(f"\nInternal tangent BL-BR:")
print(f"  Distance BL-BR: {dist_int2:.3f} mm")
print(f"  Phi angle (local): {math.degrees(phi2):.1f} degrees")

# Convert to global coordinates
# Use -phi2 to get the OTHER internal tangent (top of BL to bottom of BR)
tangent_line_angle2 = centerline_angle2 - phi2

print(f"  Tangent line angle (global): {math.degrees(tangent_line_angle2):.1f} degrees")

# For the -phi internal tangent:
# The tangent line has angle (centerline - phi) from horizontal.
# Tangent points are perpendicular to the tangent line.
#
# For BL (left circle): tangent point is on the upper-right, toward the tangent line
#   radius direction = tangent_line_angle2 + pi/2 (perpendicular, pointing "up" from line)
# For BR (right circle): tangent point is on the lower-left, toward the tangent line
#   radius direction = tangent_line_angle2 - pi/2 (perpendicular, pointing "down" from line)

# Tangent point on BL (top side - above the tangent line):
radius_angle_bl = tangent_line_angle2 + math.pi / 2
int2_tangent_bl_x = pole_bl[0] + WIRE_CENTER_RADIUS * math.cos(radius_angle_bl)
int2_tangent_bl_y = pole_bl[1] + WIRE_CENTER_RADIUS * math.sin(radius_angle_bl)

# Tangent point on BR (bottom side - below the tangent line):
radius_angle_br = tangent_line_angle2 - math.pi / 2
int2_tangent_br_x = pole_br[0] + WIRE_CENTER_RADIUS * math.cos(radius_angle_br)
int2_tangent_br_y = pole_br[1] + WIRE_CENTER_RADIUS * math.sin(radius_angle_br)

print(f"  BL tangent point: ({int2_tangent_bl_x:.3f}, {int2_tangent_bl_y:.3f})")
print(f"  BR tangent point: ({int2_tangent_br_x:.3f}, {int2_tangent_br_y:.3f})")

# Verify
verify_angle2 = math.atan2(
    int2_tangent_br_y - int2_tangent_bl_y, int2_tangent_br_x - int2_tangent_bl_x
)
print(
    f"  Verification - angle between tangent points: {math.degrees(verify_angle2):.1f}° (should be {math.degrees(tangent_line_angle2):.1f}°)"
)

# =============================================================================
# CREATE POLES
# =============================================================================

pole_positions = [
    pole_tl,  # Pole 1 - Top Left
    pole_tr,  # Pole 2 - Top Right
    pole_bl,  # Pole 3 - Bottom Left
    pole_br,  # Pole 4 - Bottom Right
]

poles = (
    cq.Workplane("XY")
    .pushPoints(pole_positions)
    .circle(POLE_RADIUS)
    .extrude(POLE_HEIGHT)
)

# =============================================================================
# CREATE TOP AND BOTTOM PLATES (closing the magnetic path)
# =============================================================================

# Calculate plate size to cover all poles with margin
plate_width = HORIZONTAL_SPACING + POLE_DIAMETER + 2 * PLATE_MARGIN
plate_height = VERTICAL_SPACING + POLE_DIAMETER + 2 * PLATE_MARGIN

# Bottom plate (at Z = -PLATE_THICKNESS to Z = 0)
bottom_plate = (
    cq.Workplane("XY")
    .workplane(offset=-PLATE_THICKNESS)
    .rect(plate_width, plate_height)
    .extrude(PLATE_THICKNESS)
)

# Top plate (at Z = POLE_HEIGHT to Z = POLE_HEIGHT + PLATE_THICKNESS)
top_plate = (
    cq.Workplane("XY")
    .workplane(offset=POLE_HEIGHT)
    .rect(plate_width, plate_height)
    .extrude(PLATE_THICKNESS)
)

# Combine plates
plates = bottom_plate.union(top_plate)

print(f"\nPlates: {plate_width:.1f} x {plate_height:.1f} x {PLATE_THICKNESS:.1f} mm")

# =============================================================================
# CREATE PARTIAL ARC for bottom-right pole (remove left arc between tangent wires)
# =============================================================================
# The two tangent points on BR ring:
# - External tangent from TL: (ext2_tangent_br_x, ext2_tangent_br_y) - upper right side
# - Internal tangent from BL: (int2_tangent_br_x, int2_tangent_br_y) - lower left side
#
# We need to cut the arc on the LEFT side between these points.

# Calculate angles of tangent points relative to BR center
angle_ext2_br = math.atan2(
    ext2_tangent_br_y - pole_br[1], ext2_tangent_br_x - pole_br[0]
)
angle_int2_br = math.atan2(
    int2_tangent_br_y - pole_br[1], int2_tangent_br_x - pole_br[0]
)

print(f"\nBR ring tangent point angles:")
print(f"  External tangent (from TL): {math.degrees(angle_ext2_br):.1f}°")
print(f"  Internal tangent (from BL): {math.degrees(angle_int2_br):.1f}°")

# Create full BR ring first
ring_br_full = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(pole_br[0], pole_br[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Create a wedge to cut out the left arc (between the two tangent points)
# The left arc goes from angle_ext2_br (45°, upper right) CCW through 180° to angle_int2_br (-123.4°, lower left)
wedge_radius_br = WIRE_OUTER_RADIUS * 3
angle_mid_br = math.pi  # 180 degrees (left side)

wedge_br_p1_x = pole_br[0] + wedge_radius_br * math.cos(angle_ext2_br)
wedge_br_p1_y = pole_br[1] + wedge_radius_br * math.sin(angle_ext2_br)
wedge_br_mid_x = pole_br[0] + wedge_radius_br * math.cos(angle_mid_br)
wedge_br_mid_y = pole_br[1] + wedge_radius_br * math.sin(angle_mid_br)
wedge_br_p2_x = pole_br[0] + wedge_radius_br * math.cos(angle_int2_br)
wedge_br_p2_y = pole_br[1] + wedge_radius_br * math.sin(angle_int2_br)

cut_wedge_br = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_br[0], pole_br[1])
    .lineTo(wedge_br_p1_x, wedge_br_p1_y)
    .lineTo(wedge_br_mid_x, wedge_br_mid_y)
    .lineTo(wedge_br_p2_x, wedge_br_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

ring_br_partial = ring_br_full.cut(cut_wedge_br)

# =============================================================================
# CREATE PARTIAL ARC for top-right pole (remove section between tangent wires)
# =============================================================================
# The two tangent points on TR ring:
# - Internal tangent from TL: (int_tangent_tr_x, int_tangent_tr_y) - upper left side
# - External tangent from BL: (tangent_tr_x, tangent_tr_y) - lower left side
#
# We need to cut the arc BETWEEN these two points on the LEFT side of the ring.

# Calculate angles of tangent points relative to TR center
angle_int_tr = math.atan2(int_tangent_tr_y - pole_tr[1], int_tangent_tr_x - pole_tr[0])
angle_ext_tr = math.atan2(tangent_tr_y - pole_tr[1], tangent_tr_x - pole_tr[0])

print(f"\nTR ring tangent point angles:")
print(f"  Internal tangent (from TL): {math.degrees(angle_int_tr):.1f}°")
print(f"  External tangent (from BL): {math.degrees(angle_ext_tr):.1f}°")

# Create full TR ring first
ring_tr_full = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(pole_tr[0], pole_tr[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Create a wedge to cut out the bottom-left arc (between the two tangent points)
# The bottom-left arc goes from angle_ext_tr (-45°) CCW through 180° to angle_int_tr (123.4°)
# We need to cut this arc, keeping the top-right arc.
#
# To cut the bottom-left arc, we create a wedge that covers that region.
# The wedge should go from angle_ext_tr, through -180°/180°, to angle_int_tr
#
# Simple approach: create a wedge from center through both tangent points,
# but order the points so the wedge covers the bottom-left side (through 180°)

wedge_radius_tr = WIRE_OUTER_RADIUS * 3

# To cut the bottom-left arc, we go from angle_ext_tr (-45°) to angle_int_tr (123.4°)
# but we need to go the "long way" through 180° (the left/bottom side)
# We can do this by adding an intermediate point at 180° (or close to it)
angle_mid_tr = math.pi  # 180 degrees, pointing left

wedge_tr_p1_x = pole_tr[0] + wedge_radius_tr * math.cos(angle_ext_tr)
wedge_tr_p1_y = pole_tr[1] + wedge_radius_tr * math.sin(angle_ext_tr)
wedge_tr_mid_x = pole_tr[0] + wedge_radius_tr * math.cos(angle_mid_tr)
wedge_tr_mid_y = pole_tr[1] + wedge_radius_tr * math.sin(angle_mid_tr)
wedge_tr_p2_x = pole_tr[0] + wedge_radius_tr * math.cos(angle_int_tr)
wedge_tr_p2_y = pole_tr[1] + wedge_radius_tr * math.sin(angle_int_tr)

cut_wedge_tr = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_tr[0], pole_tr[1])
    .lineTo(wedge_tr_p1_x, wedge_tr_p1_y)
    .lineTo(wedge_tr_mid_x, wedge_tr_mid_y)
    .lineTo(wedge_tr_p2_x, wedge_tr_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

ring_tr_partial = ring_tr_full.cut(cut_wedge_tr)

# =============================================================================
# CREATE PARTIAL ARC for top-left pole (remove section between tangent wires)
# =============================================================================
# The two tangent points on TL ring:
# - Internal tangent to TR: (int_tangent_tl_x, int_tangent_tl_y) - lower right side
# - External tangent to BR: (ext2_tangent_tl_x, ext2_tangent_tl_y) - upper right side
#
# We need to cut the arc BETWEEN these two points on the RIGHT side of the ring.

# Calculate angles of tangent points relative to TL center
angle_int_tl = math.atan2(int_tangent_tl_y - pole_tl[1], int_tangent_tl_x - pole_tl[0])
angle_ext2_tl = math.atan2(
    ext2_tangent_tl_y - pole_tl[1], ext2_tangent_tl_x - pole_tl[0]
)

print(f"\nTL ring tangent point angles:")
print(f"  Internal tangent (to TR): {math.degrees(angle_int_tl):.1f}°")
print(f"  External tangent (to BR): {math.degrees(angle_ext2_tl):.1f}°")

# Create full TL ring first
ring_tl_full = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(pole_tl[0], pole_tl[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Create wedges to cut out the right arc AND split the left arc at 180°
# The right arc goes from angle_ext2_tl (45°) to angle_int_tl (-56.6°)
# The left arc goes from angle_ext2_tl (45°) through 180° to angle_int_tl (-56.6°)
# We split it at 180° into:
#   - ring_tl_upper: from 45° to 180° (upper left)
#   - ring_tl_lower: from 180° to -56.6° (lower left)

wedge_radius = WIRE_OUTER_RADIUS * 3  # Make it big enough to cut through the ring
angle_left_tl = math.pi  # 180 degrees (leftmost point)

# Cut wedge for right arc (from 45° through 0° to -56.6°)
wedge_tl_right_p1_x = pole_tl[0] + wedge_radius * math.cos(angle_ext2_tl)
wedge_tl_right_p1_y = pole_tl[1] + wedge_radius * math.sin(angle_ext2_tl)
wedge_tl_right_p2_x = pole_tl[0] + wedge_radius * math.cos(angle_int_tl)
wedge_tl_right_p2_y = pole_tl[1] + wedge_radius * math.sin(angle_int_tl)

cut_wedge_tl_right = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_tl[0], pole_tl[1])
    .lineTo(wedge_tl_right_p1_x, wedge_tl_right_p1_y)
    .lineTo(wedge_tl_right_p2_x, wedge_tl_right_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Cut wedge for lower arc (from 180° through -90° to -56.6°) - to get upper arc only
wedge_tl_lower_p1_x = pole_tl[0] + wedge_radius * math.cos(angle_left_tl)
wedge_tl_lower_p1_y = pole_tl[1] + wedge_radius * math.sin(angle_left_tl)
wedge_tl_lower_mid_x = pole_tl[0] + wedge_radius * math.cos(-math.pi / 2)  # -90°
wedge_tl_lower_mid_y = pole_tl[1] + wedge_radius * math.sin(-math.pi / 2)
wedge_tl_lower_p2_x = pole_tl[0] + wedge_radius * math.cos(angle_int_tl)
wedge_tl_lower_p2_y = pole_tl[1] + wedge_radius * math.sin(angle_int_tl)

cut_wedge_tl_lower = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_tl[0], pole_tl[1])
    .lineTo(wedge_tl_lower_p1_x, wedge_tl_lower_p1_y)
    .lineTo(wedge_tl_lower_mid_x, wedge_tl_lower_mid_y)
    .lineTo(wedge_tl_lower_p2_x, wedge_tl_lower_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Cut wedge for upper arc (from 45° through 90° to 180°) - to get lower arc only
wedge_tl_upper_p1_x = pole_tl[0] + wedge_radius * math.cos(angle_ext2_tl)
wedge_tl_upper_p1_y = pole_tl[1] + wedge_radius * math.sin(angle_ext2_tl)
wedge_tl_upper_mid_x = pole_tl[0] + wedge_radius * math.cos(math.pi / 2)  # 90°
wedge_tl_upper_mid_y = pole_tl[1] + wedge_radius * math.sin(math.pi / 2)
wedge_tl_upper_p2_x = pole_tl[0] + wedge_radius * math.cos(angle_left_tl)
wedge_tl_upper_p2_y = pole_tl[1] + wedge_radius * math.sin(angle_left_tl)

cut_wedge_tl_upper = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_tl[0], pole_tl[1])
    .lineTo(wedge_tl_upper_p1_x, wedge_tl_upper_p1_y)
    .lineTo(wedge_tl_upper_mid_x, wedge_tl_upper_mid_y)
    .lineTo(wedge_tl_upper_p2_x, wedge_tl_upper_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Create the two TL arcs
ring_tl_upper = ring_tl_full.cut(cut_wedge_tl_right).cut(
    cut_wedge_tl_lower
)  # Upper left arc (45° to 180°)
ring_tl_lower = ring_tl_full.cut(cut_wedge_tl_right).cut(
    cut_wedge_tl_upper
)  # Lower left arc (180° to -56.6°)

# Vertical connection point at 180° (leftmost point of TL ring)
via_tl_x = pole_tl[0] + WIRE_CENTER_RADIUS * math.cos(angle_left_tl)
via_tl_y = pole_tl[1] + WIRE_CENTER_RADIUS * math.sin(angle_left_tl)

print(f"  TL split at: 180° (leftmost point)")
print(f"  Via TL position: ({via_tl_x:.3f}, {via_tl_y:.3f})")

# =============================================================================
# CREATE PARTIAL ARC for bottom-left pole (remove small arc on right side between tangents)
# =============================================================================
# The two tangent points on BL ring:
# - External tangent to TR: (tangent_bl_x, tangent_bl_y)
# - Internal tangent to BR: (int2_tangent_bl_x, int2_tangent_bl_y)
#
# We need to cut the small arc between these points on the RIGHT side.

# Calculate angles of tangent points relative to BL center
angle_ext_bl = math.atan2(tangent_bl_y - pole_bl[1], tangent_bl_x - pole_bl[0])
angle_int2_bl = math.atan2(
    int2_tangent_bl_y - pole_bl[1], int2_tangent_bl_x - pole_bl[0]
)

print(f"\nBL ring tangent point angles:")
print(f"  External tangent (to TR): {math.degrees(angle_ext_bl):.1f}°")
print(f"  Internal tangent (to BR): {math.degrees(angle_int2_bl):.1f}°")

ring_bl_full = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(pole_bl[0], pole_bl[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Create a wedge to cut out the RIGHT arc (the small one)
# Tangent points: angle_ext_bl = -45°, angle_int2_bl = 56.6°
# Right arc goes from -45° CW (through 0°) to 56.6° - this is the short arc
wedge_radius_bl = WIRE_OUTER_RADIUS * 3

wedge_bl_p1_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_int2_bl)
wedge_bl_p1_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_int2_bl)
wedge_bl_p2_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_ext_bl)
wedge_bl_p2_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_ext_bl)

# Cut wedge for right arc (from 56.6° through 0° to -45°)
cut_wedge_bl_right = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_bl[0], pole_bl[1])
    .lineTo(wedge_bl_p1_x, wedge_bl_p1_y)
    .lineTo(wedge_bl_p2_x, wedge_bl_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Define angles for horizontal wire tangent points
angle_top_bl = math.pi / 2  # 90 degrees (top)
angle_bottom_bl = -math.pi / 2  # -90 degrees (bottom)
angle_left_bl = math.pi  # 180 degrees (left)

# Cut wedge for left arc (from 90° through 180° to -90°)
wedge_bl_left_p1_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_top_bl)
wedge_bl_left_p1_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_top_bl)
wedge_bl_left_mid_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_left_bl)
wedge_bl_left_mid_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_left_bl)
wedge_bl_left_p2_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_bottom_bl)
wedge_bl_left_p2_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_bottom_bl)

cut_wedge_bl_left = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_bl[0], pole_bl[1])
    .lineTo(wedge_bl_left_p1_x, wedge_bl_left_p1_y)
    .lineTo(wedge_bl_left_mid_x, wedge_bl_left_mid_y)
    .lineTo(wedge_bl_left_p2_x, wedge_bl_left_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Cut wedge for top arc (from 56.6° to 90°) - to isolate bottom arc
wedge_bl_top_p1_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_int2_bl)
wedge_bl_top_p1_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_int2_bl)
wedge_bl_top_p2_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_top_bl)
wedge_bl_top_p2_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_top_bl)

cut_wedge_bl_top = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_bl[0], pole_bl[1])
    .lineTo(wedge_bl_top_p1_x, wedge_bl_top_p1_y)
    .lineTo(wedge_bl_top_p2_x, wedge_bl_top_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Cut wedge for bottom arc (from -90° to -45°) - to isolate top arc
wedge_bl_bottom_p1_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_bottom_bl)
wedge_bl_bottom_p1_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_bottom_bl)
wedge_bl_bottom_p2_x = pole_bl[0] + wedge_radius_bl * math.cos(angle_ext_bl)
wedge_bl_bottom_p2_y = pole_bl[1] + wedge_radius_bl * math.sin(angle_ext_bl)

cut_wedge_bl_bottom = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z - 0.1)
    .moveTo(pole_bl[0], pole_bl[1])
    .lineTo(wedge_bl_bottom_p1_x, wedge_bl_bottom_p1_y)
    .lineTo(wedge_bl_bottom_p2_x, wedge_bl_bottom_p2_y)
    .close()
    .extrude(WIRE_HEIGHT + 0.2)
)

# Create the two BL arcs:
# - ring_bl_upper: from 56.6° (internal tangent to BR) to 90° (top, horizontal wire)
# - ring_bl_lower: from -90° (bottom, horizontal wire) to -45° (external tangent to TR)
ring_bl_upper = (
    ring_bl_full.cut(cut_wedge_bl_right).cut(cut_wedge_bl_left).cut(cut_wedge_bl_bottom)
)
ring_bl_lower = (
    ring_bl_full.cut(cut_wedge_bl_right).cut(cut_wedge_bl_left).cut(cut_wedge_bl_top)
)

print(
    f"  BL upper arc: from {math.degrees(angle_int2_bl):.1f}° to {math.degrees(angle_top_bl):.1f}°"
)
print(
    f"  BL lower arc: from {math.degrees(angle_bottom_bl):.1f}° to {math.degrees(angle_ext_bl):.1f}°"
)

# =============================================================================
# CREATE STRAIGHT WIRES - from bottom-left ring, both going left
# =============================================================================

bottom_point_x = pole_bl[0]
bottom_point_y = pole_bl[1] - WIRE_CENTER_RADIUS

top_point_x = pole_bl[0]
top_point_y = pole_bl[1] + WIRE_CENTER_RADIUS

straight_wire_bottom = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(bottom_point_x - STRAIGHT_WIRE_LENGTH / 2, bottom_point_y)
    .rect(STRAIGHT_WIRE_LENGTH, WIRE_THICKNESS)
    .extrude(WIRE_HEIGHT)
)

straight_wire_top = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(top_point_x - STRAIGHT_WIRE_LENGTH / 2, top_point_y)
    .rect(STRAIGHT_WIRE_LENGTH, WIRE_THICKNESS)
    .extrude(WIRE_HEIGHT)
)

# =============================================================================
# CREATE DIAGONAL TANGENT WIRE - from BL to TR (external, left side)
# =============================================================================

tangent_length = math.sqrt(
    (tangent_tr_x - tangent_bl_x) ** 2 + (tangent_tr_y - tangent_bl_y) ** 2
)
tangent_center_x = (tangent_bl_x + tangent_tr_x) / 2
tangent_center_y = (tangent_bl_y + tangent_tr_y) / 2
tangent_angle = math.atan2(tangent_tr_y - tangent_bl_y, tangent_tr_x - tangent_bl_x)
tangent_angle_deg = math.degrees(tangent_angle)

diagonal_wire = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(tangent_center_x, tangent_center_y)
    .rect(tangent_length, WIRE_THICKNESS)
    .extrude(WIRE_HEIGHT)
    .rotate(
        (tangent_center_x, tangent_center_y, 0),
        (tangent_center_x, tangent_center_y, 1),
        tangent_angle_deg,
    )
)

# =============================================================================
# CREATE INTERNAL TANGENT WIRE - from TL bottom to TR top (diagonal, crosses between poles)
# =============================================================================

int_tangent_length = math.sqrt(
    (int_tangent_tr_x - int_tangent_tl_x) ** 2
    + (int_tangent_tr_y - int_tangent_tl_y) ** 2
)
int_tangent_center_x = (int_tangent_tl_x + int_tangent_tr_x) / 2
int_tangent_center_y = (int_tangent_tl_y + int_tangent_tr_y) / 2
# Use the calculated tangent line angle directly
int_tangent_angle_deg = math.degrees(tangent_line_angle)

print(f"Internal tangent angle: {int_tangent_angle_deg:.1f} degrees")

# This is a diagonal wire crossing between TL and TR poles
internal_tangent_wire = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(int_tangent_center_x, int_tangent_center_y)
    .rect(int_tangent_length, WIRE_THICKNESS)
    .extrude(WIRE_HEIGHT)
    .rotate(
        (int_tangent_center_x, int_tangent_center_y, 0),
        (int_tangent_center_x, int_tangent_center_y, 1),
        int_tangent_angle_deg,
    )
)

# =============================================================================
# CREATE EXTERNAL TANGENT WIRE - from TL to BR (top side)
# =============================================================================

ext2_tangent_length = math.sqrt(
    (ext2_tangent_br_x - ext2_tangent_tl_x) ** 2
    + (ext2_tangent_br_y - ext2_tangent_tl_y) ** 2
)
ext2_tangent_center_x = (ext2_tangent_tl_x + ext2_tangent_br_x) / 2
ext2_tangent_center_y = (ext2_tangent_tl_y + ext2_tangent_br_y) / 2
ext2_tangent_angle_deg = math.degrees(ext2_centerline_angle)

print(f"External tangent TL-BR angle: {ext2_tangent_angle_deg:.1f} degrees")

external_tangent_tl_br = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(ext2_tangent_center_x, ext2_tangent_center_y)
    .rect(ext2_tangent_length, WIRE_THICKNESS)
    .extrude(WIRE_HEIGHT)
    .rotate(
        (ext2_tangent_center_x, ext2_tangent_center_y, 0),
        (ext2_tangent_center_x, ext2_tangent_center_y, 1),
        ext2_tangent_angle_deg,
    )
)

# =============================================================================
# CREATE INTERNAL TANGENT WIRE - from BR bottom to BL top (diagonal, crosses between poles)
# =============================================================================

int2_tangent_length = math.sqrt(
    (int2_tangent_bl_x - int2_tangent_br_x) ** 2
    + (int2_tangent_bl_y - int2_tangent_br_y) ** 2
)
int2_tangent_center_x = (int2_tangent_br_x + int2_tangent_bl_x) / 2
int2_tangent_center_y = (int2_tangent_br_y + int2_tangent_bl_y) / 2
int2_tangent_angle_deg = math.degrees(tangent_line_angle2)

print(f"Internal tangent BR-BL angle: {int2_tangent_angle_deg:.1f} degrees")

internal_tangent_br_bl = (
    cq.Workplane("XY")
    .workplane(offset=WIRE_Z)
    .center(int2_tangent_center_x, int2_tangent_center_y)
    .rect(int2_tangent_length, WIRE_THICKNESS)
    .extrude(WIRE_HEIGHT)
    .rotate(
        (int2_tangent_center_x, int2_tangent_center_y, 0),
        (int2_tangent_center_x, int2_tangent_center_y, 1),
        int2_tangent_angle_deg,
    )
)

# =============================================================================
# CREATE VERTICAL VIA - connecting segments 1 and 2 at TL 180° point
# =============================================================================
# The via connects the upper layer (+Z) to the lower layer (-Z)
# It's a vertical rectangular prism at the junction point

via_height = (
    2 * WIRE_Z_OFFSET + WIRE_HEIGHT
)  # spans from -Z layer bottom to +Z layer top
via_z_bottom = WIRE_Z - WIRE_Z_OFFSET  # bottom of via (at lower layer bottom)

via_tl = (
    cq.Workplane("XY")
    .workplane(offset=via_z_bottom)  # start at lower layer bottom
    .center(via_tl_x, via_tl_y)
    .rect(WIRE_THICKNESS, WIRE_THICKNESS)
    .extrude(via_height)
)

print(f"Via TL height: {via_height:.3f} mm")
print(f"Via TL Z range: {via_z_bottom:.3f} to {via_z_bottom + via_height:.3f} mm")

# =============================================================================
# NAMED SEGMENTS (numbered for layer assignment)
# =============================================================================

segments_base = {
    # Ring arcs
    1: ("ring_tl_upper", ring_tl_upper),  # TL upper arc (45° to 180°)
    2: ("ring_tl_lower", ring_tl_lower),  # TL lower arc (180° to -56.6°)
    3: ("ring_tr_partial", ring_tr_partial),  # TR arc (top-right side)
    4: ("ring_bl_upper", ring_bl_upper),  # BL upper arc (56.6° to 90°)
    5: ("ring_bl_lower", ring_bl_lower),  # BL lower arc (-90° to -45°)
    6: ("ring_br_partial", ring_br_partial),  # BR arc (right side)
    # Straight wires
    7: ("straight_wire_top", straight_wire_top),  # Horizontal wire from top of BL
    8: (
        "straight_wire_bottom",
        straight_wire_bottom,
    ),  # Horizontal wire from bottom of BL
    # Diagonal tangent wires
    9: ("diagonal_wire_bl_tr", diagonal_wire),  # External tangent BL→TR (45°)
    10: (
        "internal_tangent_tl_tr",
        internal_tangent_wire,
    ),  # Internal tangent TL→TR (33.4°)
    11: (
        "external_tangent_tl_br",
        external_tangent_tl_br,
    ),  # External tangent TL→BR (-45°)
    12: (
        "internal_tangent_bl_br",
        internal_tangent_br_bl,
    ),  # Internal tangent BL→BR (-33.4°)
    # Vertical vias
    13: ("via_tl", via_tl),  # Via connecting segments 1 and 2 at TL 180°
}

# Vias span both layers, no offset needed
VIA_SEGMENTS = {13}

# Apply Z offsets to segments
segments = {}
for num, (name, segment) in segments_base.items():
    if num in VIA_SEGMENTS:
        # Vias already span both layers
        segments[num] = (name, segment, "VIA")
    elif num in POSITIVE_Z_SEGMENTS:
        z_offset = WIRE_Z_OFFSET
        layer = "+Z"
        segment_moved = segment.translate((0, 0, z_offset))
        segments[num] = (name, segment_moved, layer)
    else:
        z_offset = -WIRE_Z_OFFSET
        layer = "-Z"
        segment_moved = segment.translate((0, 0, z_offset))
        segments[num] = (name, segment_moved, layer)

print("\n" + "=" * 60)
print("SEGMENT LIST:")
print("=" * 60)
print(f"  {'#':>2}  {'Name':<25} {'Layer':<5}")
print("-" * 60)
for num, (name, _, layer) in segments.items():
    print(f"  {num:2d}: {name:<25} {layer}")
print("=" * 60 + "\n")

# =============================================================================
# CREATE WINDING 0 - all wire segments combined into one object
# =============================================================================

# Start with the first segment and union all others
segment_list = list(segments.values())
winding_0 = segment_list[0][1]  # Get the geometry from first segment
for name, segment, layer in segment_list[1:]:
    winding_0 = winding_0.union(segment)

print("Winding 0: all segments combined into single object")

# =============================================================================
# CREATE SECONDARY WINDINGS - U-shaped wires around each pole
# =============================================================================

# Secondary winding parameters
SECONDARY_STRAIGHT_LENGTH = 5.0  # mm - length of straight wire extensions

# Secondary winding around BL pole
# Start with just the top 180° arc

# Z position for the secondary arc (one layer above primary +Z)
# Primary +Z is at WIRE_Z + WIRE_Z_OFFSET, secondary is WIRE_Z_OFFSET above that
arc_z = WIRE_Z + WIRE_Z_OFFSET + SECONDARY_Z_OFFSET

# Top 180° arc - half ring at the top
# Create full ring then cut to get top half (180° arc from 0° to 180°)
secondary_bl_arc_full = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_bl[0], pole_bl[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Cut away bottom half (from 180° to 360°, i.e., the -Y side)
arc_cut_box = (
    cq.Workplane("XY")
    .workplane(offset=arc_z - 0.1)
    .center(pole_bl[0], pole_bl[1] - WIRE_OUTER_RADIUS)
    .rect(WIRE_OUTER_RADIUS * 3, WIRE_OUTER_RADIUS * 2)
    .extrude(WIRE_HEIGHT + 0.2)
)

secondary_bl_arc = secondary_bl_arc_full.cut(arc_cut_box)

# Straight connections extending from arc ends along -Y direction
# Arc ends are at 0° (right side, +X) and 180° (left side, -X)
# Both at Y = pole_bl[1] (center Y of pole)

# Left end of arc (180°)
arc_left_x = pole_bl[0] - WIRE_CENTER_RADIUS
arc_left_y = pole_bl[1]

# Right end of arc (0°)
arc_right_x = pole_bl[0] + WIRE_CENTER_RADIUS
arc_right_y = pole_bl[1]

# Create straight wires going in -Y direction
secondary_bl_left_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(arc_left_x, arc_left_y - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_bl_right_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(arc_right_x, arc_right_y - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

# Combine arc and straight wires
secondary_bl = secondary_bl_arc.union(secondary_bl_left_wire).union(
    secondary_bl_right_wire
)

print(f"Secondary BL: U-shape around bottom-left pole (pointing -Y)")

# -----------------------------------------------------------------------------
# Secondary winding around BR pole (pointing -Y, same as BL)
# -----------------------------------------------------------------------------

secondary_br_arc_full = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_br[0], pole_br[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

arc_cut_box_br = (
    cq.Workplane("XY")
    .workplane(offset=arc_z - 0.1)
    .center(pole_br[0], pole_br[1] - WIRE_OUTER_RADIUS)
    .rect(WIRE_OUTER_RADIUS * 3, WIRE_OUTER_RADIUS * 2)
    .extrude(WIRE_HEIGHT + 0.2)
)

secondary_br_arc = secondary_br_arc_full.cut(arc_cut_box_br)

secondary_br_left_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_br[0] - WIRE_CENTER_RADIUS, pole_br[1] - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_br_right_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_br[0] + WIRE_CENTER_RADIUS, pole_br[1] - SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_br = secondary_br_arc.union(secondary_br_left_wire).union(
    secondary_br_right_wire
)

print(f"Secondary BR: U-shape around bottom-right pole (pointing -Y)")

# -----------------------------------------------------------------------------
# Secondary winding around TL pole (pointing +Y, rotated 180°)
# -----------------------------------------------------------------------------

secondary_tl_arc_full = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_tl[0], pole_tl[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Cut away top half (the +Y side) to keep bottom half
arc_cut_box_tl = (
    cq.Workplane("XY")
    .workplane(offset=arc_z - 0.1)
    .center(pole_tl[0], pole_tl[1] + WIRE_OUTER_RADIUS)
    .rect(WIRE_OUTER_RADIUS * 3, WIRE_OUTER_RADIUS * 2)
    .extrude(WIRE_HEIGHT + 0.2)
)

secondary_tl_arc = secondary_tl_arc_full.cut(arc_cut_box_tl)

# Straight wires going in +Y direction
secondary_tl_left_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_tl[0] - WIRE_CENTER_RADIUS, pole_tl[1] + SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_tl_right_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_tl[0] + WIRE_CENTER_RADIUS, pole_tl[1] + SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_tl = secondary_tl_arc.union(secondary_tl_left_wire).union(
    secondary_tl_right_wire
)

print(f"Secondary TL: U-shape around top-left pole (pointing +Y)")

# -----------------------------------------------------------------------------
# Secondary winding around TR pole (pointing +Y, rotated 180°)
# -----------------------------------------------------------------------------

secondary_tr_arc_full = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_tr[0], pole_tr[1])
    .circle(WIRE_OUTER_RADIUS)
    .circle(WIRE_INNER_RADIUS)
    .extrude(WIRE_HEIGHT)
)

# Cut away top half (the +Y side) to keep bottom half
arc_cut_box_tr = (
    cq.Workplane("XY")
    .workplane(offset=arc_z - 0.1)
    .center(pole_tr[0], pole_tr[1] + WIRE_OUTER_RADIUS)
    .rect(WIRE_OUTER_RADIUS * 3, WIRE_OUTER_RADIUS * 2)
    .extrude(WIRE_HEIGHT + 0.2)
)

secondary_tr_arc = secondary_tr_arc_full.cut(arc_cut_box_tr)

# Straight wires going in +Y direction
secondary_tr_left_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_tr[0] - WIRE_CENTER_RADIUS, pole_tr[1] + SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_tr_right_wire = (
    cq.Workplane("XY")
    .workplane(offset=arc_z)
    .center(pole_tr[0] + WIRE_CENTER_RADIUS, pole_tr[1] + SECONDARY_STRAIGHT_LENGTH / 2)
    .rect(WIRE_THICKNESS, SECONDARY_STRAIGHT_LENGTH)
    .extrude(WIRE_HEIGHT)
)

secondary_tr = secondary_tr_arc.union(secondary_tr_left_wire).union(
    secondary_tr_right_wire
)

print(f"Secondary TR: U-shape around top-right pole (pointing +Y)")

# -----------------------------------------------------------------------------
# Combine all secondaries into one part
# -----------------------------------------------------------------------------

secondaries = secondary_bl.union(secondary_br).union(secondary_tl).union(secondary_tr)

print(f"\nSecondaries: all 4 U-shapes combined into single object")
print(f"  Arc Z: {arc_z:.3f} mm")
print(f"  Straight wires: {SECONDARY_STRAIGHT_LENGTH} mm")

# =============================================================================
# COMBINE CORE (poles + plates)
# =============================================================================

core = poles.union(plates)

# =============================================================================
# COMBINE ALL (core + winding + secondaries)
# =============================================================================

result = core.union(winding_0).union(secondaries)

# =============================================================================
# EXPORT
# =============================================================================

os.makedirs("output", exist_ok=True)

# Export complete assembly
cq.exporters.export(result, "output/layer3_coil.step")
print("\nExported: output/layer3_coil.step (complete assembly)")

# Export winding 0 separately
cq.exporters.export(winding_0, "output/winding_0.step")
print("Exported: output/winding_0.step (winding only)")

# Export secondaries separately
cq.exporters.export(secondaries, "output/secondaries.step")
print("Exported: output/secondaries.step (all 4 secondaries)")

# Export core (poles + plates) separately
cq.exporters.export(core, "output/core.step")
print("Exported: output/core.step (poles + plates)")

# Export poles separately
cq.exporters.export(poles, "output/poles.step")
print("Exported: output/poles.step (poles only)")

# Export plates separately
cq.exporters.export(plates, "output/plates.step")
print("Exported: output/plates.step (top + bottom plates)")

print(f"\nPoles: {len(pole_positions)}")
print(f"Pole diameter: {POLE_DIAMETER} mm")
print(f"Spacing: {HORIZONTAL_SPACING} x {VERTICAL_SPACING} mm")
print(f"Wire clearance: {WIRE_CLEARANCE} mm")
print(f"Wire thickness: {WIRE_THICKNESS} mm")
print(f"Wire height: {WIRE_HEIGHT} mm")
