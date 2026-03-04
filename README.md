# Ansyas

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Ansyas** automates 3D FEM simulation in **Ansys Maxwell** and **Icepak** for any magnetic component described in [MAS language](https://github.com/OpenMagnetics/MAS). Developed by [Wuerth Elektronik eiSos](https://www.we-online.com).

## Installation

```bash
pip install Ansyas
```

## Quick Start

```python
from Ansyas import AnsyasMasLoader
loader = AnsyasMasLoader()
loader.load("component.json")
loader.simulate()
```

## Testing

```bash
pytest tests/ -m "not ansys"   # unit tests
pytest tests/ --run-ansys       # integration tests
```

## Configuration

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `number_segments_arcs` | 12 | >= 3 |
| `initial_mesh_configuration` | 5 | 1-5 |
| `maximum_error_percent` | 3.0 | > 0 |
| `maximum_passes` | 40 | >= 1 |
| `refinement_percent` | 30.0 | (0, 100] |
| `scale` | 1.0 | > 0 |

## License

MIT. Contact: Alfonso.Martinez@we-online.com
