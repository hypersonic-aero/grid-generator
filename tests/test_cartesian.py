"""Tests for Cartesian grid generation, its configuration, and the cartesian CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cfd_io import read_file
from typer.testing import CliRunner

from grid_generator import (
    CartesianConfig,
    EquidistantSpec,
    GeometricSpec,
    generate_cartesian,
    load_cartesian_config,
)
from grid_generator.cli import app

runner = CliRunner()

# axes used across the tests
X_SPEC = EquidistantSpec(start=0.0, end=1.0, n_points=11)
Y_SPEC = GeometricSpec(start=0.0, end=0.05, n_points=21, first_spacing=1.0e-4)
Z_SPEC = EquidistantSpec(start=-0.1, end=0.1, n_points=5)

# the default boundary names per grid side ([markers] left at its defaults)
X_MARKERS = {"imin": "x_min", "imax": "x_max"}
Y_MARKERS = {"jmin": "y_min", "jmax": "y_max"}
Z_MARKERS = {"kmin": "z_min", "kmax": "z_max"}


# --------------------------------------------------
# generation
# --------------------------------------------------
@pytest.mark.parametrize(
    ("y_spec", "z_spec", "shape", "dimensionality", "markers"),
    [
        (None, None, (11, 1, 1), 1, X_MARKERS),
        (Y_SPEC, None, (11, 21, 1), 2, X_MARKERS | Y_MARKERS),
        (Y_SPEC, Z_SPEC, (11, 21, 5), 3, X_MARKERS | Y_MARKERS | Z_MARKERS),
        (None, Z_SPEC, (11, 1, 5), 2, X_MARKERS | Z_MARKERS),
    ],
)
def test_grid_shape_and_attrs(
    y_spec, z_spec, shape: tuple, dimensionality: int, markers: dict
) -> None:
    """Omitted axes become singleton dimensions; attrs describe the grid and mark its ends."""

    # build
    config = CartesianConfig(x=X_SPEC, y=y_spec, z=z_spec, output=Path("grid.h5"))
    dataset = generate_cartesian(config)

    # check
    assert dataset.grid.shape == shape
    assert dataset.attrs["dimensionality"] == dimensionality
    assert dataset.attrs["grid_type"] == "cartesian"
    assert (dataset.attrs["ni"], dataset.attrs["nj"], dataset.attrs["nk"]) == shape
    assert dataset.attrs["markers"]["structured"] == markers


def test_coordinates_follow_the_axes() -> None:
    """x varies with i only, y with j only, z with k only (cfd-io's (ni, nj, nk) layout)."""

    # build
    config = CartesianConfig(x=X_SPEC, y=Y_SPEC, z=Z_SPEC, output=Path("grid.h5"))
    grid = generate_cartesian(config).grid

    # build the expected axes
    x_axis = np.linspace(0.0, 1.0, 11)
    y_axis = grid.y[0, :, 0]
    z_axis = np.linspace(-0.1, 0.1, 5)

    # check each coordinate against its axis, broadcast over the other two indices
    np.testing.assert_allclose(grid.x, np.broadcast_to(x_axis[:, None, None], grid.shape))
    np.testing.assert_allclose(grid.y, np.broadcast_to(y_axis[None, :, None], grid.shape))
    np.testing.assert_allclose(grid.z, np.broadcast_to(z_axis[None, None, :], grid.shape))
    assert y_axis[1] - y_axis[0] == pytest.approx(1.0e-4, rel=1e-8)


# --------------------------------------------------
# configuration
# --------------------------------------------------
VALID_CONFIG = """\
[grid]
type = "cartesian"

[grid.x]
type = "equidistant"
start = 0.0
end = 1.0
n_points = 11

[grid.y]
type = "geometric"
start = 0.0
end = 0.05
n_points = 21
first_spacing = 1.0e-4

[output]
filename = "out/grid.h5"
dtype = "f8"
"""


def test_load_resolves_output_next_to_the_config(tmp_path: Path) -> None:
    """A relative output path is resolved against the config file's folder."""

    # write and load
    config_path = tmp_path / "cartesian.toml"
    config_path.write_text(VALID_CONFIG)
    config = load_cartesian_config(config_path)

    # check
    assert config.output == tmp_path / "out" / "grid.h5"
    assert isinstance(config.y, GeometricSpec)
    assert config.z is None


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('type = "cartesian"', 'type = "polar"', "must be 'cartesian'"),
        ("n_points = 11", "n_points = 11\nnpoints = 3", "unknown"),
        ('dtype = "f8"', 'dtype = "f2"', "dtype"),
        ('filename = "out/grid.h5"', 'filename = "out/grid.su2"', ".h5"),
        ("[output]", "[extra]\nkey = 1\n\n[output]", "unknown configuration sections"),
        ("[output]", '[markers]\nx_mni = "inflow"\n\n[output]', "x_mni"),
        ("[output]", '[markers]\nx_min = "in flow"\n\n[output]', "one word"),
        ("[output]", '[markers]\nx_min = "fluid"\n\n[output]', "reserved"),
        ("[output]", '[markers]\nz_max = "top"\n\n[output]', r"only apply with a \[grid.z\]"),
    ],
)
def test_load_rejects_bad_config(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Wrong grid type, unknown keys or sections, bad dtype or output suffix, bad markers raise."""

    # break one line of a valid config
    config_text = VALID_CONFIG.replace(old, new, 1)
    config_path = tmp_path / "cartesian.toml"
    config_path.write_text(config_text)

    with pytest.raises(ValueError, match=message):
        load_cartesian_config(config_path)


def test_markers_rename_the_sides_in_the_hdf5_file(tmp_path: Path) -> None:
    """[markers] names reach the dataset and the HDF5 file; sides may share a name."""

    # rename three sides, two of them to the same name
    markers_table = '[markers]\nx_min = "inflow"\nx_max = "outflow"\ny_max = "outflow"\n\n'
    config_text = VALID_CONFIG.replace("[output]", markers_table + "[output]", 1)
    config_path = tmp_path / "cartesian.toml"
    config_path.write_text(config_text)

    # generate and write
    from cfd_io import write_file

    dataset = generate_cartesian(config_path)
    write_file(tmp_path / "grid.h5", dataset)

    # check: the renamed sides, the default for the rest, restored from the file
    expected = {"imin": "inflow", "imax": "outflow", "jmin": "y_min", "jmax": "outflow"}
    assert dataset.attrs["markers"]["structured"] == expected
    written = read_file(tmp_path / "grid.h5", orient=False)
    assert written.attrs["markers"] == {"structured": expected}


# --------------------------------------------------
# CLI end to end
# --------------------------------------------------
def test_cli_init_then_run_writes_a_readable_grid(tmp_path: Path) -> None:
    """cartesian init writes a valid template; run writes HDF5 that cfd-io reads back."""

    # init and run
    config_path = tmp_path / "cartesian.toml"
    result_init = runner.invoke(app, ["cartesian", "init", "--output", str(config_path)])
    result_run = runner.invoke(app, ["cartesian", "run", "-c", str(config_path)])

    # check the commands
    assert result_init.exit_code == 0, result_init.output
    assert result_run.exit_code == 0, result_run.output

    # read back without reorienting and compare with a direct generation
    expected = generate_cartesian(config_path).grid
    written = read_file(tmp_path / "grid.hdf5", orient=False).grid
    assert written.shape == expected.shape
    np.testing.assert_allclose(written.x, expected.x)
    np.testing.assert_allclose(written.y, expected.y)


@pytest.mark.parametrize(("dims", "dimensionality"), [(1, 1), (2, 2), (3, 3)])
def test_cli_init_dims_sets_the_axes(tmp_path: Path, dims: int, dimensionality: int) -> None:
    """cartesian init --dims writes x, then y, then z; the grid has that many dimensions."""

    # init and run
    config_path = tmp_path / "cartesian.toml"
    result_init = runner.invoke(
        app, ["cartesian", "init", "--output", str(config_path), "--dims", str(dims)]
    )
    result_run = runner.invoke(app, ["cartesian", "run", "-c", str(config_path)])

    # check
    assert result_init.exit_code == 0, result_init.output
    assert result_run.exit_code == 0, result_run.output
    assert generate_cartesian(config_path).attrs["dimensionality"] == dimensionality


def test_cli_init_rejects_bad_dims(tmp_path: Path) -> None:
    """--dims outside 1-3 fails before any file is written."""

    # run
    config_path = tmp_path / "cartesian.toml"
    result = runner.invoke(app, ["cartesian", "init", "--output", str(config_path), "--dims", "4"])

    # check
    assert result.exit_code == 1
    assert "--dims must be 1, 2 or 3" in result.output
    assert not config_path.exists()


def test_cli_init_refuses_to_overwrite(tmp_path: Path) -> None:
    """init leaves an existing file alone unless --force is given."""

    # existing user file
    config_path = tmp_path / "cartesian.toml"
    config_path.write_text("# mine\n")

    # run without --force
    result = runner.invoke(app, ["cartesian", "init", "--output", str(config_path)])

    # check
    assert result.exit_code == 1
    assert config_path.read_text() == "# mine\n"
