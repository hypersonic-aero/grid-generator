# grid-generator

Structured grid generation utilities for Python.

[![PyPI](https://img.shields.io/pypi/v/grid-generator)](https://pypi.org/project/grid-generator/)
[![Docs](https://img.shields.io/badge/docs-zensical-blue)](https://uahypersonics.github.io/grid-generator/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](https://www.python.org/downloads/)

`grid-generator` builds structured computational grids and writes them through
the canonical `cfd-io` data model.

## Install

```bash
pip install grid-generator
```

## Quick Start

```bash
grid-generator cartesian init
grid-generator cartesian run
```

The generated TOML file defines required `grid.x` and optional `grid.y` and
`grid.z` point distributions. Omitting optional axes creates singleton
dimensions, so Cartesian grids always use the structured `(ni, nj, nk)` array
contract.

The same workflow is available from Python:

```python
from grid_generator import generate_cartesian

dataset = generate_cartesian("cartesian.toml")
print(dataset.grid.shape)
```

Body-fitted 2D grids around blunt bodies (blunted plate, wedge, sphere-cone, tangent ogive)
use the geometry definitions from
[canonical-geometries](https://github.com/uahypersonics/canonical-geometries):

```bash
grid-generator plate init
grid-generator plate run
```

`run` reads the file `init` writes (`plate.toml`); pass another one with `-c`.

For a shock-line grid the bow shock is a grid line: the same index `j` on every
wall-normal line, so a solution's captured shock stays in one row of cells.
`init --shock` writes that layout; it reads the shock curve from a CSV file
(e.g. `flow-theory shock-shape run` with a `.csv` output, or a shock taken from a
CFD solution). Grid line `i = 0` is the symmetry line on the axis `y = 0`, so the
curve starts there; a curve that starts a little above it (a shock from cell-centre
data starts half a cell away) is continued to the axis, with a warning:

```bash
grid-generator cone init --shock
grid-generator cone run
```

## Features

- One-, two-, and three-dimensional Cartesian tensor-product grids
- Body-fitted 2D grids with a polar nose block, SU2 boundary markers, and a grid
  quality report: constant height, or shock-line grids whose bow shock is a grid line
- Extrusion of plate and wedge grids in z into 3D grids
- Equidistant, geometric, tanh, two-sided (Vinokur), and segmented point distributions
- `distribution info` reference of the distribution types and their keys, and
  `distribution init/run` for 1D point files (.dat, .csv, .txt)
- Composable cubic and quintic transitions between equidistant regions
- Focused `cartesian`, `plate`, `wedge`, `cone` and `tangent-ogive` CLI workflows, each with `init/run` (and `examples`)
- HDF5 and SU2 output through `cfd-io`
- Typed Python configuration and generation API

## Documentation

Documentation site: https://uahypersonics.github.io/grid-generator/

## Releasing

This project publishes to PyPI from Git tags that match `v*`.

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

The GitHub Actions publish workflow will build the package, run tests, and then
publish to PyPI using Trusted Publishing.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for the
complete license terms.
