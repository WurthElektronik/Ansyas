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
