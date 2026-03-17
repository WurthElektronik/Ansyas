"""
HDI Planar Transformer - Multiple Secondary Windings
Multiple rectangular U-shaped wires stacked in parallel.
With separate ferrite regions: inner cores (inside U-turns) and outer ferrite.
"""

import cadquery as cq
import os
import math

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# Number of U-turns in parallel
NUM_TURNS = 32

# U-turn wire dimensions
WIRE_HEIGHT = 2.64  # mm - height of wire (Z dimension of the U-turn)
WIRE_WIDTH = 0.2  # mm - width/thickness of wire
U_TURN_INNER_SEPARATION = (
    5.6  # mm - separation between internal faces of U-turn legs (doubled)
)

# Gap between adjacent U-turns in Z direction
TURN_GAP = 2.0  # mm - gap between adjacent turns

# Core dimensions
CORE_X = 12.4  # mm - core dimension in X (doubled margin: 3.2mm each side)
CORE_Y = 1.0  # mm - core dimension in Y
# CORE_Z will be calculated based on number of turns

# Wire outside core (0 = flush with core/region boundary)
WIRE_OUTSIDE_CORE = 0.0  # mm - length of wire sticking out of core in -Y

# =============================================================================
# CALCULATED VALUES
# =============================================================================

# The rectangular U-turn consists of:
# - Two vertical legs (straight wires) going in -Y direction
# - A horizontal bar connecting them at the top (+Y side)

# Distance between leg centers
LEG_CENTER_SEPARATION = U_TURN_INNER_SEPARATION + WIRE_WIDTH

# Leg positions (X coordinates)
LEFT_LEG_X = -LEG_CENTER_SEPARATION / 2
RIGHT_LEG_X = LEG_CENTER_SEPARATION / 2

# Length of straight wire legs (Y direction)
STRAIGHT_LENGTH = CORE_Y  # Wire legs match core Y dimension

# Top bar dimensions
TOP_BAR_LENGTH = (
    LEG_CENTER_SEPARATION + WIRE_WIDTH
)  # Full width including wire thickness
TOP_BAR_WIDTH = WIRE_WIDTH  # Same as wire width

# Inner core dimensions (inside each U-turn)
INNER_CORE_X = U_TURN_INNER_SEPARATION  # Width between the U legs
INNER_CORE_Y = CORE_Y - WIRE_WIDTH  # Depth minus top bar width

# Calculate total Z height needed for all turns
TOTAL_TURNS_HEIGHT = NUM_TURNS * WIRE_HEIGHT + (NUM_TURNS - 1) * TURN_GAP
CORE_Z = TOTAL_TURNS_HEIGHT + 0.5  # Add margin at top and bottom (doubled)

print(f"Configuration:")
print(f"  Number of U-turns: {NUM_TURNS}")
print(f"  Wire height (Z): {WIRE_HEIGHT} mm")
print(f"  Wire width: {WIRE_WIDTH} mm")
print(f"  Inner separation: {U_TURN_INNER_SEPARATION} mm")
print(f"  Gap between turns: {TURN_GAP} mm")
print(f"  Total turns height: {TOTAL_TURNS_HEIGHT:.3f} mm")
print(f"  Core Z (calculated): {CORE_Z:.3f} mm")

# =============================================================================
# CREATE ALL U-TURNS AND INNER CORES
# =============================================================================

print(f"\nCreating {NUM_TURNS} U-turns...")

all_secondaries = []
all_inner_cores = []

# Starting Z position (small margin from bottom)
start_z = (CORE_Z - TOTAL_TURNS_HEIGHT) / 2

for turn_idx in range(NUM_TURNS):
    # Z position for this turn
    turn_z = start_z + turn_idx * (WIRE_HEIGHT + TURN_GAP)

    # Create U-turn
    # Left leg
    left_leg = (
        cq.Workplane("XY")
        .workplane(offset=turn_z)
        .center(LEFT_LEG_X, -STRAIGHT_LENGTH / 2)
        .rect(WIRE_WIDTH, STRAIGHT_LENGTH)
        .extrude(WIRE_HEIGHT)
    )

    # Right leg
    right_leg = (
        cq.Workplane("XY")
        .workplane(offset=turn_z)
        .center(RIGHT_LEG_X, -STRAIGHT_LENGTH / 2)
        .rect(WIRE_WIDTH, STRAIGHT_LENGTH)
        .extrude(WIRE_HEIGHT)
    )

    # Top bar
    top_bar = (
        cq.Workplane("XY")
        .workplane(offset=turn_z)
        .center(0, -WIRE_WIDTH / 2)
        .rect(TOP_BAR_LENGTH, WIRE_WIDTH)
        .extrude(WIRE_HEIGHT)
    )

    # Combine into U-shape
    u_turn = left_leg.union(right_leg).union(top_bar)
    all_secondaries.append(u_turn)

    # Create inner core for this U-turn
    inner_core_y_min = -STRAIGHT_LENGTH
    inner_core_y_max = -WIRE_WIDTH
    inner_core_y_center = (inner_core_y_min + inner_core_y_max) / 2
    inner_core_y_size = inner_core_y_max - inner_core_y_min

    inner_core = (
        cq.Workplane("XY")
        .workplane(offset=turn_z)
        .center(0, inner_core_y_center)
        .rect(INNER_CORE_X, inner_core_y_size)
        .extrude(WIRE_HEIGHT)
    )
    all_inner_cores.append(inner_core)

    if turn_idx < 3 or turn_idx >= NUM_TURNS - 2:
        print(
            f"  Turn {turn_idx + 1}: z = {turn_z:.3f} to {turn_z + WIRE_HEIGHT:.3f} mm"
        )
    elif turn_idx == 3:
        print(f"  ... ({NUM_TURNS - 4} more turns) ...")

# Combine all U-turns into one object
print(f"\nCombining {NUM_TURNS} U-turns...")
secondaries = all_secondaries[0]
for u_turn in all_secondaries[1:]:
    secondaries = secondaries.union(u_turn)

# Combine all inner cores into one object
print(f"Combining {NUM_TURNS} inner cores...")
inner_cores = all_inner_cores[0]
for core in all_inner_cores[1:]:
    inner_cores = inner_cores.union(core)

# =============================================================================
# CREATE OUTER FERRITE (surrounding)
# =============================================================================

print("\nCreating outer ferrite...")

core_y_min = -STRAIGHT_LENGTH
core_y_max = core_y_min + CORE_Y
core_y_center = (core_y_min + core_y_max) / 2

core_x_center = 0
core_z_min = 0

print(f"Core dimensions:")
print(f"  X: {-CORE_X / 2:.3f} to {CORE_X / 2:.3f} mm (width: {CORE_X} mm)")
print(f"  Y: {core_y_min:.3f} to {core_y_max:.3f} mm (depth: {CORE_Y} mm)")
print(f"  Z: {core_z_min:.3f} to {CORE_Z:.3f} mm (height: {CORE_Z:.3f} mm)")

# Create the full core block
core_block = (
    cq.Workplane("XY")
    .workplane(offset=core_z_min)
    .center(core_x_center, core_y_center)
    .rect(CORE_X, CORE_Y)
    .extrude(CORE_Z)
)

# Subtract windings and inner cores from outer ferrite
outer_ferrite = core_block.cut(secondaries).cut(inner_cores)

print("Outer ferrite: core block with wire channels and inner cores cut out")

# =============================================================================
# EXPORT
# =============================================================================

os.makedirs("output", exist_ok=True)

# Export secondaries only
cq.exporters.export(secondaries, "output/hdi_secondary_single.step")
print(f"\nExported: output/hdi_secondary_single.step ({NUM_TURNS} windings)")

# Export inner cores (low permeability)
cq.exporters.export(inner_cores, "output/hdi_inner_cores.step")
print(f"Exported: output/hdi_inner_cores.step ({NUM_TURNS} inner cores, mu_r=60)")

# Export outer ferrite (high permeability)
cq.exporters.export(outer_ferrite, "output/hdi_outer_ferrite.step")
print("Exported: output/hdi_outer_ferrite.step (outer ferrite, mu_r=2000)")

# Export combined ferrite (for backwards compatibility - inner + outer)
combined_ferrite = inner_cores.union(outer_ferrite)
cq.exporters.export(combined_ferrite, "output/hdi_secondary_ferrite.step")
print("Exported: output/hdi_secondary_ferrite.step (combined ferrite)")

# Export assembly
assembly = cq.Assembly()
assembly.add(
    outer_ferrite, name="outer_ferrite", color=cq.Color(0.3, 0.3, 0.3, 1)
)  # dark gray
assembly.add(
    inner_cores, name="inner_cores", color=cq.Color(0.5, 0.5, 0.5, 1)
)  # lighter gray
assembly.add(secondaries, name="windings", color=cq.Color(0.8, 0.5, 0.2, 1))  # copper
assembly.save("output/hdi_secondary_assembly.step", "STEP")
print("Exported: output/hdi_secondary_assembly.step (full assembly)")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n" + "=" * 60)
print("GEOMETRY SUMMARY")
print("=" * 60)
print(f"Number of U-turns: {NUM_TURNS}")
print(f"\nRectangular U-turn dimensions:")
print(f"  Height (Z): {WIRE_HEIGHT} mm")
print(f"  Width: {WIRE_WIDTH} mm")
print(f"  Inner separation: {U_TURN_INNER_SEPARATION} mm")
print(f"  Leg length (Y): {STRAIGHT_LENGTH} mm")
print(f"  Top bar length (X): {TOP_BAR_LENGTH} mm")
print(f"  Gap between turns: {TURN_GAP} mm")
print(f"\nCore dimensions:")
print(f"  X: {CORE_X} mm")
print(f"  Y: {CORE_Y} mm")
print(f"  Z: {CORE_Z:.3f} mm (calculated for {NUM_TURNS} turns)")
inner_core_y_size = CORE_Y - WIRE_WIDTH
print(f"\nInner cores ({NUM_TURNS} total):")
print(f"  X: {INNER_CORE_X} mm")
print(f"  Y: {inner_core_y_size:.3f} mm")
print(f"  Z: {WIRE_HEIGHT} mm (per core)")
print(f"  Material: Ferrite (mu_r = 60)")
print(f"\nOuter ferrite (surrounding):")
print(f"  Material: Ferrite (mu_r = 2000)")
print(f"\nWire outside core: {WIRE_OUTSIDE_CORE} mm (flush)")
print("=" * 60)
