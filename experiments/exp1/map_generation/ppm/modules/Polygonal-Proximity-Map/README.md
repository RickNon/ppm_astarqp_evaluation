# Polygonal-Proximity-Map

Workspace for building Polygonal Proximity Map (PPM).  
The root manages the execution environment with uv, while the implementation lives in the `ppm_builder` package.

## Structure
- Root: dependency and environment management via uv
- `ppm_builder/`: library implementation (a distributable Python package)

## Entry Point
This repository does not provide a CLI.  
**The entry point is importing the `ppm_builder` package** in your Python code.

Example:
```python
from ppm_builder import PPMBuilder2D, PPMBuilder3D, SensorPose, ProximityPointSet
```

See `ppm_builder/README.md` for detailed usage examples.

## Run with uv
Prerequisite: Python 3.11+

1) Sync dependencies
```bash
uv sync
```

2) Quick run (example)
```bash
uv run ppm_builder/test/quick_test.py
```
