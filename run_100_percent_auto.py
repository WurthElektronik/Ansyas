#!/usr/bin/env python3
"""
100% FULLY AUTOMATED TEMPERATURE EXTRACTION
Uses _odesign.ExportSolutionOverview which works with Student version!
"""

import os
import sys
import json
import time
import subprocess
import re
from datetime import datetime

sys.path.insert(0, r'C:\Users\Alfonso\wuerth\Ansyas\src')
sys.path.insert(0, r'C:\Users\Alfonso\wuerth\Ansyas\src\Ansyas')

import ansys.aedt.core as pyaedt
from Ansyas import Ansyas
import mas_autocomplete

OUTPUT_DIR = r'C:\Users\Alfonso\wuerth\Ansyas\output'
MKF_TEST_FILE = r'\wsl.localhost\Ubuntu-22.04\home\alf\OpenMagnetics\MKF\tests\TestTemperature.cpp'

# Get MAS file from command line argument or use default
if len(sys.argv) > 1:
    MAS_FILE = sys.argv[1]
else:
    MAS_FILE = r'C:\Users\Alfonso\wuerth\Ansyas\examples\concentric_flyback_rectangular_column.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print(f"100% AUTOMATED - PyAEDT {pyaedt.__version__}")
print("Using _odesign.ExportSolutionOverview (bypasses gRPC)")
print("="*80)

# Cleanup
print("\n1. Cleaning up...")
subprocess.run(["wmic", "process", "where", "name like '%ansys%'", "delete"], capture_output=True)
subprocess.run(["wmic", "process", "where", "name like '%aedt%'", "delete"], capture_output=True)
time.sleep(3)

# Create simulation
print("\n2. Creating Icepak simulation...")
with open(MAS_FILE, 'r') as f:
    mas_dict = json.load(f)
mas = mas_autocomplete.autocomplete(mas_dict)

ansyas = Ansyas(
    number_segments_arcs=12,
    initial_mesh_configuration=2,
    maximum_error_percent=5,
    refinement_percent=30,
    maximum_passes=15,
    scale=1
)

# Use unique project name to avoid conflicts with existing files
project_name = f"auto_temp_{int(time.time())}"
project = ansyas.create_project(
    outputs_folder=OUTPUT_DIR,
    project_name=project_name,
    non_graphical=True,
    solution_type="SteadyState",
    new_desktop_session=True
)

ansyas.set_units("meter")
ansyas.create_magnetic_simulation(mas=mas, simulate=False, operating_point_index=0)

# Enable gravity
setup = project.setups[0]
setup.props['Include Gravity'] = True
setup.update()

print("\n3. Running thermal simulation...")
project.analyze_setup(setup.name)
project.save_project()
print("   [OK] Simulation completed")

# Export using _odesign.ExportSolutionOverview (WORKS with Student version!)
print("\n4. Exporting temperature data...")

export_file = os.path.join(OUTPUT_DIR, "solution_overview.txt").replace('\\', '/')

try:
    odesign = project._odesign
    
    # This works with Student version!
    odesign.ExportSolutionOverview(
        [
            "SetupName:=", "Setup",
            "DesignVariationKey:=", "",
            "ExportFilePath:=", export_file,
            "TimeStep:=", -1,
            "Overwrite:=", True
        ]
    )
    
    print(f"   [OK] Exported to: {export_file}")
    
except Exception as e:
    print(f"   [ERROR] Export failed: {e}")
    sys.exit(1)

# Parse temperatures from the exported file
print("\n5. Extracting temperatures from export...")

temperatures = {}

if os.path.exists(export_file):
    with open(export_file, 'r') as f:
        content = f.read()
    
    # Find "Maximum Temperatures For Thermal BCs" section
    # Pattern: Object name followed by temperature value
    temp_section = re.search(
        r'# Maximum Temperatures For Thermal BCs:\s*\n'
        r'Object\s+Temperature \[C\]\s*\n'
        r'(.*?)(?=\n#|\Z)',
        content,
        re.DOTALL
    )
    
    if temp_section:
        temp_lines = temp_section.group(1).strip().split('\n')
        
        for line in temp_lines:
            if line.strip():
                # Parse: ObjectName    Temperature
                # Format: core_0_losses     80.11
                parts = line.strip().split()
                if len(parts) >= 2:
                    obj_name = parts[0]
                    try:
                        temp = float(parts[-1])  # Last part is temperature
                        
                        # Map loss boundary names to object names
                        # e.g., "core_0_losses" -> "core_0"
                        clean_name = obj_name.replace('_losses', '')
                        
                        if any(kw in clean_name.lower() for kw in ['core', 'bobbin', 'copper', 'turn', 'primary', 'secondary']):
                            temperatures[clean_name] = temp
                            print(f"     {clean_name}: {temp:.2f}C")
                    except ValueError:
                        pass

print(f"\n   [OK] Extracted {len(temperatures)} temperatures")

if not temperatures:
    print("   [ERROR] No temperatures found!")
    print("   File content preview:")
    print(content[:1000])
    sys.exit(1)

# Generate C++ test
print("\n6. Generating C++ test case...")

mas_basename = os.path.splitext(os.path.basename(MAS_FILE))[0]
test_name = mas_basename

core_temps = [t for name, t in temperatures.items() if 'core' in name.lower()]
bobbin_temps = [t for name, t in temperatures.items() if 'bobbin' in name.lower()]
turn_temps = [(name, t) for name, t in temperatures.items() 
              if any(kw in name.lower() for kw in ['turn', 'copper', 'primary', 'secondary'])]

avg_core = sum(core_temps) / len(core_temps) if core_temps else 0
max_core = max(core_temps) if core_temps else 0
min_core = min(core_temps) if core_temps else 0
avg_bobbin = sum(bobbin_temps) / len(bobbin_temps) if bobbin_temps else 0

cpp_code = f"""// =============================================================================
// Auto-generated Icepak Temperature Test - PyAEDT {pyaedt.__version__}
// Source MAS: {mas_basename}
// Generated: {datetime.now().isoformat()}
// Method: Icepak thermal simulation with natural convection + gravity
// =============================================================================

TEST_CASE("Temperature: {test_name}", "[temperature]") {{
    auto jsonPath = OpenMagneticsTesting::get_test_data_path(std::source_location::current(), "{mas_basename}.json");
    auto mas = OpenMagneticsTesting::mas_loader(jsonPath);
    
    auto magnetic = OpenMagnetics::magnetic_autocomplete(mas.get_magnetic());
    auto inputs = OpenMagnetics::inputs_autocomplete(mas.get_inputs(), magnetic);
    
    // Run magnetic simulation to get actual losses
    auto losses = getLossesFromSimulation(magnetic, inputs);
    
    TemperatureConfig config;
    config.ambientTemperature = losses.ambientTemperature;
    config.coreLosses = losses.coreLosses;
    if (losses.windingLossesOutput.has_value()) {{
        config.windingLosses = losses.windingLosses;
        config.windingLossesOutput = losses.windingLossesOutput.value();
    }} else {{
        applySimulatedLosses(config, magnetic);
    }}
    config.plotSchematic = true;
    
    Temperature temp(magnetic, config);
    auto result = temp.calculateTemperatures();
    
    // Reference values from Icepak simulation (Student Version)
    REQUIRE(result.converged);
    REQUIRE(result.maximumTemperature > config.ambientTemperature);
    
    // Export temperature field and thermal circuit schematic for visualization
    exportTemperatureFieldSvg("{test_name}", magnetic, result.nodeTemperatures, config.ambientTemperature);
    exportThermalCircuitSchematic("{test_name}", temp);
    
    // Get temperatures by component type
    auto tempsByType = temp.getTemperaturesByComponentType();
    auto tempsPerTurn = temp.getTemperaturePerTurn();
    
    SECTION("Core temperature validation against Icepak") {{
        // Icepak core temperatures from solution overview
"""

# Add core temperature checks
for name, temp_val in temperatures.items():
    if 'core' in name.lower():
        cpp_code += f"        // {name}: {temp_val:.2f}°C (from Icepak)\n"

if core_temps:
    max_core_temp = max(core_temps)
    cpp_code += f"        REQUIRE(tempsByType.at(\"core\") <= {max_core_temp * 1.25:.2f}); // Max Icepak: {max_core_temp:.2f}°C + 25% tolerance\n"
    cpp_code += f"        REQUIRE(tempsByType.at(\"core\") >= {max_core_temp * 0.75:.2f}); // Min Icepak: {max_core_temp:.2f}°C - 25% tolerance\n"

cpp_code += """    }
    
    SECTION("Bobbin temperature validation against Icepak") {
        if (tempsByType.find("bobbin") != tempsByType.end()) {
"""

if bobbin_temps:
    max_bobbin = max(bobbin_temps)
    cpp_code += f"            REQUIRE_THAT(tempsByType.at(\"bobbin\"), Catch::Matchers::WithinRel({max_bobbin:.2f}, 0.25)); // Icepak: {max_bobbin:.2f}°C + 25% tolerance\n"
else:
    cpp_code += "            // No bobbin temperature data from Icepak\n"

cpp_code += """        }
    }
    
    SECTION("Individual turn temperatures from Icepak export") {
        // Validate specific turn temperatures exported from Icepak
"""

# Add individual turn checks
for name, temp_val in turn_temps:
    # Create a turn identifier that matches MKF naming
    # e.g., Secondary_Parallel_0_Turn_4_copper -> W0_T4 or similar
    short_name = name.replace('Primary_Parallel_', 'P').replace('Secondary_Parallel_', 'S').replace('_copper', '')
    # Try to map to MKF turn naming convention
    turn_id = short_name.replace('P', 'W').replace('S', 'W')
    cpp_code += f"        // {name}: {temp_val:.2f}°C (Icepak)\n"
    cpp_code += f"        // Check if turn {turn_id} exists in results\n"
    cpp_code += f"        if (tempsPerTurn.find(\"Turn_{turn_id}\") != tempsPerTurn.end()) {{\n"
    cpp_code += f"            REQUIRE_THAT(tempsPerTurn.at(\"Turn_{turn_id}\"), Catch::Matchers::WithinRel({temp_val:.2f}, 0.25)); // 25% tolerance\n"
    cpp_code += "        }\n"

cpp_code += """    }
    
    SECTION("Winding temperature by index") {
        // Check winding temperatures using getTemperaturesByComponentType
"""

# Count windings and add checks
winding_count = 0
for name, _ in turn_temps:
    if 'Primary_Parallel_0' in name or 'Secondary_Parallel_0' in name:
        winding_count = max(winding_count, 1)
    elif 'Primary_Parallel_1' in name:
        winding_count = max(winding_count, 2)

for i in range(winding_count):
    cpp_code += f"        if (tempsByType.find(\"winding {i}\") != tempsByType.end()) {{\n"
    cpp_code += f"            REQUIRE(tempsByType.at(\"winding {i}\") > config.ambientTemperature);\n"
    cpp_code += "        }\n"

cpp_code += """    }
}
"""

# Save
test_file = os.path.join(OUTPUT_DIR, f"test_{mas_basename}.cpp")
with open(test_file, 'w') as f:
    f.write(cpp_code)

print(f"   [OK] Test saved: {test_file}")

# Append to MKF
try:
    if os.path.exists(MKF_TEST_FILE):
        with open(MKF_TEST_FILE, 'r') as f:
            existing = f.read()
        
        if test_name not in existing:
            import shutil
            shutil.copy2(MKF_TEST_FILE, f"{MKF_TEST_FILE}.backup_{int(time.time())}")
            
            with open(MKF_TEST_FILE, 'a') as f:
                if not existing.endswith('\n'):
                    f.write('\n')
                f.write(cpp_code)
            
            print(f"   [OK] Test appended to: {MKF_TEST_FILE}")
        else:
            print(f"   [INFO] Test already exists")
except Exception as e:
    print(f"   [WARN] Could not append: {e}")

# Cleanup
ansyas.solver_backend.close()

print("\n" + "="*80)
print("SUCCESS! 100% AUTOMATED")
print("="*80)
print(f"Temperatures extracted: {len(temperatures)}")
print(f"Test case: {test_name}")
print(f"Output: {OUTPUT_DIR}")
print("\nNo manual GUI interaction required!")
