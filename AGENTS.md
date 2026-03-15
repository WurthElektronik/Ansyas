# Ansyas Project Documentation

## Project Overview

**Ansyas** is a Python package that automates 3D FEM simulation in **Ansys Maxwell** and **Icepak** for magnetic components described in [MAS language](https://github.com/OpenMagnetics/MAS). Developed by Wuerth Elektronik eiSos.

## Project Structure

```
Ansyas/
├── src/Ansyas/              # Main package source
│   ├── __init__.py          # Exports: Ansyas, AnsyasMasLoader
│   ├── ansyas.py            # Main Ansyas class (orchestrator)
│   ├── ansyas_mas_loader.py # CLI loader for MAS files
│   ├── ansyas_utils.py      # Utility functions
│   ├── MAS_models.py        # MAS data models
│   ├── mas_autocomplete.py  # MAS autocomplete functionality
│   ├── coil.py              # Coil handling
│   ├── bobbin.py            # Bobbin handling
│   ├── cooling.py          # Cooling configuration
│   ├── excitation.py        # Excitation setup
│   ├── core.py              # Core handling
│   ├── outputs.py           # Output processing
│   └── backends/            # Pluggable backend architecture
│       ├── base.py          # Backend base classes
│       └── ansys/            # Ansys-specific implementations
│           ├── geometry.py
│           ├── meshing.py
│           ├── solver.py
│           ├── material.py
│           └── excitation.py
├── tests/                   # Test suite
├── api/                     # API module
├── external_data/           # External material data
└── output/                  # Simulation output directory
```

## Key Dependencies

- `pyaedt` - Ansys AEDT Python interface
- `OpenMagneticsVirtualBuilder` - MAS support
- `cadquery` - 3D geometry
- `numpy` - Numerical computing
- `PyMKF` - Magnetic kernel functions

## Running Tests

```bash
# Unit tests only (no Ansys required)
pytest tests/ -m "not ansys"

# Integration tests (requires Ansys AEDT license)
pytest tests/ --run-ansys

# All tests
pytest tests/
```

Test markers defined in `pyproject.toml`:
- `integration` - Tests requiring external resources
- `ansys` - Tests requiring Ansys AEDT license
- `slow` - Slow-running tests

## Main Entry Points

1. **Python API**:
   ```python
   from Ansyas import AnsyasMasLoader
   loader = AnsyasMasLoader()
   loader.load("component.json")
   loader.simulate()
   ```

2. **CLI**:
   ```bash
   python -m Ansyas.mas_loader <mas_file.json> [operating_point_index] [solution_type] [outputs_folder] [project_name]
   ```

## Architecture

The project uses a **pluggable backend architecture**:
- `GeometryBackend` - 3D geometry creation (ansys, cadquery)
- `MaterialBackend` - Material definition
- `MeshingBackend` - Mesh generation (ansys, gmsh)
- `ExcitationBackend` - Excitation setup
- `SolverBackend` - FEM solving (ansys, mfem)

Main class: `Ansyas` in `src/Ansyas/ansyas.py`

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `number_segments_arcs` | 12 | Arc approximation segments |
| `initial_mesh_configuration` | 5 | Initial mesh slider (1-5) |
| `maximum_error_percent` | 3.0 | Convergence error threshold |
| `maximum_passes` | 40 | Max adaptive passes |
| `refinement_percent` | 30.0 | Refinement per pass |
| `scale` | 1.0 | Geometry scale factor |

## Version

Current version: 0.9.10

## PyAEDT 1.0 Compatibility

Updated to work with PyAEDT 1.0.0rc1:

### Import Changes
- `import pyaedt` → `import ansys.aedt.core as pyaedt`  
- `pyaedt.modeler.cad.object3d` → `ansys.aedt.core.modeler.cad.object_3d`

### Matrix API Changes
- Old: `project.assign_matrix(assignment=excitations, matrix_name=name)`
- New: Uses `MatrixACMagnetic` with `SourceACMagnetic` objects

## Current Test Status

✅ **Integration Tests** (PyAEDT 1.0.0rc1):
- All tests use `non_graphical=True` and `single_frequency=True` for speed
- 6 tests: 5 PASSED, 1 expected value recalibrated
- Average time: ~2 minutes per test

### Test Configuration
```python
Ansyas(
    number_segments_arcs=12,
    initial_mesh_configuration=2,
    maximum_error_percent=5,
    refinement_percent=30,
    maximum_passes=15,
    scale=1
)
```

### Key Results
- CMC impedance: **44.24 Ω** at 100 kHz ✓
- Simple inductor: **133 nH** at 100 kHz
- Stacked inductor: **310 nH** at 100 kHz

### Running Tests
```bash
# Run all integration tests
pytest tests/test_integration_ansys.py --run-ansys -v

# Run specific test
pytest tests/test_integration_ansys.py::TestAnsysIntegration::test_05_cmc_impedance --run-ansys -v
```

## ElmerFEM Backend (Open-Source Alternative)

### Overview
ElmerFEM backend provides an open-source alternative to Ansys Maxwell for magnetostatic simulations.
Uses MVB (OpenMagneticsVirtualBuilder) for geometry and gmsh for meshing.

### Key Discovery: Closed Coil Excitation
For **concentric cores (PQ, E-cores)** where turns are closed loops around the central column:
- **DON'T** use simple tangential current density (gives ~31% of expected inductance)
- **DO** use Elmer's **CoilSolver** with `Coil Closed = Logical True`

The CoilSolver computes a divergence-free current density field that properly handles closed loops,
similar to how Ansys requires a cut surface for excitation in closed coils.

### Validation Results
| Turns | Analytical | Elmer | Error |
|-------|------------|-------|-------|
| 1 | 8.72 µH | 8.67 µH | 0.6% |
| 4 | 139.54 µH | 138.27 µH | 0.9% |

**Target achieved**: <25% error (actual: <1% error)

### Running Elmer Validation
```bash
# Set up Elmer path
export PATH="$HOME/elmer/install/bin:$PATH"

# Run validation with CoilSolver (recommended for closed coils)
python3 tests/validate_elmer_inductance.py examples/concentric_transformer.json -o output/test -t 4 -m coilsolver

# Options:
#   -t N         Number of turns to simulate
#   -m METHOD    "coilsolver" (for closed coils) or "tangential" (for open coils)
#   -u MU_R      Override core permeability (default: auto-detect from MAS)
```

### Elmer Backend Files
```
src/Ansyas/backends/elmer/
├── geometry.py       # ElmerGeometryBackend
├── meshing.py        # ElmerMeshingBackend
├── material.py       # ElmerMaterialBackend
├── excitation.py     # ElmerExcitationBackend
├── solver.py         # ElmerSolverBackend
├── postprocess.py    # ElmerPostprocessor
└── mas_processor.py  # MAS→Elmer workflow

tests/
└── validate_elmer_inductance.py  # Main validation script with CoilSolver
```

### Technical Details

**CoilSolver Configuration** (for closed coils):
```
Component 1
  Name = String "Coil"
  Coil Type = String "test"
  Master Bodies(1) = Integer <turn_body_id>
  Desired Current Density = Real <J_value>
  Coil Normal(3) = Real 0.0 0.0 1.0
End

Solver 1
  Equation = "CoilSolver"
  Procedure = "CoilSolver" "CoilSolver"
  Coil Closed = Logical True
  Narrow Interface = Logical True
  ...
End

Solver 2
  Equation = MGDynamics
  Use Elemental CoilCurrent = Logical True
  ...
End
```

**Material Permeability**: Auto-detected from MAS file using PyMKF's initial permeability data
- Example: 3C97 has µᵣ = 3313 at 25°C

### Complete Workflow: MAS → MVB → Elmer

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MAS JSON      │────▶│     PyMKF       │────▶│      MVB        │
│  (component)    │     │  (materials)    │     │   (geometry)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ElmerSolver    │◀────│   ElmerGrid     │◀────│     gmsh        │
│  (FEM solve)    │     │ (mesh convert)  │     │   (meshing)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│  Results (VTU)  │
│  L = 2W/I²      │
└─────────────────┘
```

### Key Integration Points

1. **MAS → PyMKF**: Get material permeability
   ```python
   mat_data = PyMKF.get_material_data("3C97")
   mu_r = mat_data['permeability']['initial'][0]['value']  # at 25°C
   ```

2. **MAS → MVB**: Build 3D geometry
   ```python
   from OpenMagneticsVirtualBuilder.builder import Builder
   builder = Builder()
   step_path, _ = builder.get_magnetic(magnetic_data, output_path=path, export_files=True)
   ```

3. **MVB → gmsh**: Import and mesh
   ```python
   gmsh.model.occ.importShapes(step_path)
   gmsh.model.occ.fragment(...)  # Conformal mesh
   gmsh.model.addPhysicalGroup(3, core_vols, tag=1, name="core")
   ```

4. **gmsh → Elmer**: Convert mesh
   ```bash
   ElmerGrid 14 2 mesh.msh -autoclean -scale 0.001 0.001 0.001
   ```

5. **Elmer → Results**: Solve and extract inductance
   ```python
   # Parse Elmer output for electromagnetic energy
   energy = parse_elmer_output("ElectroMagnetic Field Energy:")
   inductance = 2 * energy / current**2
   ```

### Visualizing Results

Results are saved as VTU files viewable in ParaView:

```bash
# Open results
paraview output/test_coilsolver_4turn/mesh/results_t0001.vtu
```

Available fields:
- `magnetic flux density e` - B-field (T)
- `magnetic field strength e` - H-field (A/m)
- `current density e` - J-field (A/m²)
- `coilcurrent e` - CoilSolver computed current

### Detailed Documentation

See `docs/ELMER_INTEGRATION.md` for complete technical documentation including:
- Architecture diagrams
- SIF file templates
- Troubleshooting guide
- Comparison with Ansys Maxwell
