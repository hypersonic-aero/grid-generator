# Installation

## Requirements

- Python 3.11 or later (uses `tomllib` from the standard library)

All other dependencies are installed automatically when installing the package

## From PyPI

```bash
pip install grid-generator
```

## From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/hypersonic-aero/grid-generator.git
cd grid-generator
pip install -e .
```

## Development Installation

For development, install with the `dev` extras:

```bash
pip install -e ".[dev]"
```

This includes:

- [pytest](https://docs.pytest.org/) and [pytest-cov](https://pytest-cov.readthedocs.io/) for testing
- [ruff](https://docs.astral.sh/ruff/) for linting

To build the documentation, also install the `docs` extras:

```bash
pip install -e ".[dev,docs]"
```

This adds [Zensical](https://zensical.org/) for the documentation site.

## Verify Installation

```python
import grid_generator
print(grid_generator.__version__)
```

Or from the command line:

```bash
grid-generator --version
```
