# grid-generator

`grid-generator` creates structured computational grids

## Quick Start

Install the [PyPI package](https://pypi.org/project/grid-generator/) (see
[Installation](installation.md) for detailed instructions).

```bash
pip install grid-generator
```

Create a starter configuration, edit it, and generate the grid:

```bash
# initialize grid generator: writes grid.toml (the body in its [geometry] table)
grid-generator plate init
# run grid generator (reads grid.toml, the file init writes)
grid-generator plate run
```

## License

GNU General Public License v3.0 or later. See
[LICENSE](https://github.com/hypersonic-aero/grid-generator/blob/main/LICENSE) for the
complete license terms.
