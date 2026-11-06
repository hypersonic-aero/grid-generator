"""Tests for swept (3D) grids: [sweep] revolve (cone, tangent ogive) and extrude (plate, wedge)."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from cfd_io import read_file
from typer.testing import CliRunner

from grid_generator import generate_body_fitted, load_body_fitted_config
from grid_generator.cli import app
from grid_generator.config import body_fitted_template
from grid_generator.quality import sweep_quality
from grid_generator.sweep import sweep_transform

runner = CliRunner()

# the templates' [sweep] point count, and the count these tests cut it to
N_SWEEP_TEMPLATE = 21
N_SWEEP = 5

# the template's local grids: points along the wall and away from it
NI = 201
NJ = 101

# the template's sweep sizes
ANGLE_DEG = 10.0
WIDTH = 2.0e-3

# the template's x_end, for the side-face tolerance
X_END = 0.01


def write_sweep_config(
    tmp_path: Path, geometry_type: str, replacements: tuple[tuple[str, str], ...] = ()
) -> Path:
    """Write the template with [sweep] (5 planes) for a geometry, with optional replacements."""

    # template with the sweep, cut to 5 planes
    config_text = body_fitted_template(geometry_type, sweep=True)
    full_count = f"n_points = {N_SWEEP_TEMPLATE}\n"
    assert config_text.count(full_count) == 1
    config_text = config_text.replace(full_count, f"n_points = {N_SWEEP}\n")

    # optional changes
    for old, new in replacements:
        assert old in config_text, old
        config_text = config_text.replace(old, new, 1)

    # write
    config_path = tmp_path / f"{geometry_type}.toml"
    config_path.write_text(config_text)

    return config_path


def plane_of(tmp_path: Path, geometry_type: str):
    """The same geometry's 2D grid, without the sweep."""

    # template without the sweep
    plane_dir = tmp_path / "plane"
    plane_dir.mkdir()
    config_path = plane_dir / f"{geometry_type}.toml"
    config_path.write_text(body_fitted_template(geometry_type))
    grid = generate_body_fitted(config_path).grid

    return grid


# --------------------------------------------------
# geometry
# --------------------------------------------------
@pytest.mark.parametrize("geometry_type", ["cone", "tangent-ogive"])
def test_revolve_keeps_the_plane_and_turns_it(tmp_path: Path, geometry_type: str) -> None:
    """k = 0 is the 2D grid exactly; every plane is it turned about the x-axis by k * step."""

    # build the sector and the 2D grid
    dataset = generate_body_fitted(write_sweep_config(tmp_path, geometry_type))
    grid = dataset.grid
    plane = plane_of(tmp_path, geometry_type)

    # check: the plane k = 0 bit for bit
    assert grid.shape == (NI, NJ, N_SWEEP)
    np.testing.assert_array_equal(grid.x[:, :, 0], plane.x[:, :, 0])
    np.testing.assert_array_equal(grid.y[:, :, 0], plane.y[:, :, 0])
    np.testing.assert_array_equal(grid.z[:, :, 0], 0.0)

    # check: each plane is the meridian plane at angle k * step (same x and radius)
    step = math.radians(ANGLE_DEG) / (N_SWEEP - 1)
    radius = plane.y[:, :, 0]
    off_axis = radius > 0.0
    for k in range(N_SWEEP):
        np.testing.assert_array_equal(grid.x[:, :, k], plane.x[:, :, 0])
        np.testing.assert_allclose(np.hypot(grid.y[:, :, k], grid.z[:, :, k]), radius, rtol=1e-15)
        angle = np.arctan2(grid.z[:, :, k], grid.y[:, :, k])
        np.testing.assert_allclose(angle[off_axis], k * step, rtol=1e-13)

    # check: the symmetry line is on the axis on every plane
    np.testing.assert_array_equal(grid.y[0], 0.0)
    np.testing.assert_array_equal(grid.z[0], 0.0)

    # check the description: 3D, not to be run axisymmetric, six sides named
    attrs = dataset.attrs
    assert attrs["dimensionality"] == 3
    assert attrs["axisymmetric"] is False
    assert attrs["markers"]["structured"]["kmin"] == "sweep_low"
    assert attrs["markers"]["structured"]["kmax"] == "sweep_high"
    assert attrs["sweep"]["rotation"]["angle_deg"] == ANGLE_DEG


@pytest.mark.parametrize("geometry_type", ["plate", "wedge"])
def test_extrude_keeps_the_plane_and_repeats_it(tmp_path: Path, geometry_type: str) -> None:
    """k = 0 is the 2D grid exactly; the same plane at equally spaced z up to the width."""

    # build the slab and the 2D grid
    dataset = generate_body_fitted(write_sweep_config(tmp_path, geometry_type))
    grid = dataset.grid
    plane = plane_of(tmp_path, geometry_type)

    # check
    assert grid.shape == (NI, NJ, N_SWEEP)
    z_expected = np.linspace(0.0, WIDTH, N_SWEEP)
    for k in range(N_SWEEP):
        np.testing.assert_array_equal(grid.x[:, :, k], plane.x[:, :, 0])
        np.testing.assert_array_equal(grid.y[:, :, k], plane.y[:, :, 0])
        np.testing.assert_array_equal(grid.z[:, :, k], z_expected[k])
    assert grid.z[0, 0, -1] == WIDTH
    assert dataset.attrs["sweep"]["translation"] == [0.0, 0.0, WIDTH]


@pytest.mark.parametrize("geometry_type", ["cone", "tangent-ogive", "plate", "wedge"])
def test_side_faces_match_and_cells_have_volume(tmp_path: Path, geometry_type: str) -> None:
    """The high side face is the low one transformed; all cells positive; axis cells counted."""

    # build and measure
    config = load_body_fitted_config(write_sweep_config(tmp_path, geometry_type))
    dataset = generate_body_fitted(config)
    transform = sweep_transform(config.sweep)
    report = sweep_quality(dataset.grid, transform, "sweep_low", "sweep_high")

    # check: matched faces, right-handed cells
    assert report.side_face_mismatch < 1.0e-12 * X_END
    assert report.min_cell_volume > 0.0

    # check: the axis layer of a revolved grid, none for an extruded one
    if transform.sweep_type == "revolve":
        assert report.axis_cells == (NJ - 1) * (N_SWEEP - 1)
        assert report.min_axis_cell_volume > 0.0
    else:
        assert report.axis_cells == 0


def test_shock_line_grid_revolves(tmp_path: Path) -> None:
    """A shock-line cone grid revolves like any other: same plane, the shock line kept."""

    # template with the shock line and the sweep; a straight shock curve next to it
    config_text = body_fitted_template("cone", sweep=True, shock=True)
    config_text = config_text.replace(f"n_points = {N_SWEEP_TEMPLATE}\n", f"n_points = {N_SWEEP}\n")
    config_path = tmp_path / "cone.toml"
    config_path.write_text(config_text)
    config = load_body_fitted_config(config_path)
    shock_x = np.linspace(-1.0e-3, 0.02, 400)
    shock_y = 0.5e-3 + 2.0 * np.sqrt(1.0e-3 * (shock_x + 1.0e-3)) + 0.2 * (shock_x + 1.0e-3)
    shock_y[0] = 0.0
    with config.shock.curve.open("w") as stream:
        stream.write("x,y\n")
        for x_value, y_value in zip(shock_x, shock_y, strict=True):
            stream.write(f"{float(x_value)!r},{float(y_value)!r}\n")

    # build
    dataset = generate_body_fitted(config)

    # check: 3D, the shock line named, the axis exact
    assert dataset.grid.shape[2] == N_SWEEP
    assert dataset.attrs["shock_line_j"] == config.points.shock_layer.n_points - 1
    np.testing.assert_array_equal(dataset.grid.y[0], 0.0)


# --------------------------------------------------
# configuration
# --------------------------------------------------
@pytest.mark.parametrize(
    ("geometry_type", "replacements", "message"),
    [
        ("cone", (('type = "revolve"', 'type = "extrude"'),), "width"),
        (
            "cone",
            (
                (
                    'type = "revolve"\n# sector angle in [deg]\nangle_deg = 10.0',
                    'type = "extrude"\nwidth = 0.01',
                ),
            ),
            "revolved",
        ),
        (
            "plate",
            (
                (
                    'type = "extrude"\n# span in z in [m]\nwidth = 2.0e-3',
                    'type = "revolve"\nangle_deg = 10.0',
                ),
            ),
            "extruded",
        ),
        ("cone", (("angle_deg = 10.0", "angle_deg = 360.0"),), "less than 360"),
        ("cone", (("angle_deg = 10.0", "angle_deg = 10.0\nwidth = 0.01"),), "width"),
        ("plate", (("n_points = 5", "n_points = 1"),), "n_points"),
        ("cone", (('filename = "cone_local.su2"', 'filename = "cone_local.p2dfmt"'),), ".p3dfmt"),
        (
            "cone",
            (
                ('filename = "cone_local.su2"', 'filename = "cone_local.msh"'),
                ("order = 1", "order = 2"),
            ),
            r"\[sweep\] 5 points|odd",
        ),
        ("plate", (("scale = 1.0", "scale = 0.0"),), "scale"),
    ],
)
def test_load_rejects_bad_sweep(
    tmp_path: Path, geometry_type: str, replacements: tuple[tuple[str, str], ...], message: str
) -> None:
    """Wrong sweep for the body, bad sizes, unknown keys, 2D formats, even counts at order 2."""

    # an even sweep count for the order-2 case (5 is odd)
    if ("order = 1", "order = 2") in replacements:
        replacements = (*replacements, ("n_points = 5", "n_points = 4"))

    # write a broken config
    config_path = write_sweep_config(tmp_path, geometry_type, replacements)

    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


def test_markers_name_the_side_faces(tmp_path: Path) -> None:
    """sweep_low and sweep_high name the planes k = 0 and k = n_points - 1 (here as PyFR pairs)."""

    # rename the side faces
    replacements = (
        ('sweep_low = "sweep_low"', 'sweep_low = "periodic_0_l"'),
        ('sweep_high = "sweep_high"', 'sweep_high = "periodic_0_r"'),
    )
    config_path = write_sweep_config(tmp_path, "cone", replacements)

    # check
    dataset = generate_body_fitted(config_path)
    markers = dataset.attrs["markers"]["structured"]
    assert (markers["kmin"], markers["kmax"]) == ("periodic_0_l", "periodic_0_r")
    assert dataset.attrs["sweep"]["faces"] == {"low": "periodic_0_l", "high": "periodic_0_r"}


# --------------------------------------------------
# solver lines
# --------------------------------------------------
def test_solver_lines_rotate_the_low_face_onto_the_high_one(tmp_path: Path) -> None:
    """SU2 turns the first marker onto the second by +angle about x; Eilmer's bottom gets -angle."""

    # build the transform
    config = load_body_fitted_config(write_sweep_config(tmp_path, "cone"))
    transform = sweep_transform(config.sweep)

    # check SU2: centre 0, rotation about x, no translation
    line = transform.su2_marker_periodic("sweep_low", "sweep_high")
    assert line == (
        "MARKER_PERIODIC= ( sweep_low, sweep_high, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0 )"
    )

    # check Eilmer: the bottom face's vectors come from the top, turned back by the angle
    matrices = transform.eilmer_rmatrices()
    bottom = np.array(matrices["bottom"]).reshape(3, 3)
    top = np.array(matrices["top"]).reshape(3, 3)
    high_normal = np.array(
        [0.0, -math.sin(math.radians(ANGLE_DEG)), math.cos(math.radians(ANGLE_DEG))]
    )
    np.testing.assert_allclose(bottom @ high_normal, [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(top @ bottom, np.eye(3), atol=1e-15)


# --------------------------------------------------
# command line: init, run, writers
# --------------------------------------------------
@pytest.mark.parametrize("geometry_type", ["plate", "wedge", "cone", "tangent-ogive"])
def test_cli_init_sweep_writes_a_3d_config(tmp_path: Path, geometry_type: str) -> None:
    """init --sweep adds the body's [sweep]; the config runs into a 3D grid and its JSON."""

    # init with --sweep, then run
    config_path = tmp_path / f"{geometry_type}.toml"
    result_init = runner.invoke(app, [geometry_type, "init", "-o", str(config_path), "--sweep"])
    result = runner.invoke(app, [geometry_type, "run", "-c", str(config_path)])

    # check
    assert result_init.exit_code == 0, result_init.output
    assert result.exit_code == 0, result.output
    assert f"x {N_SWEEP_TEMPLATE} points" in result.output
    assert "MARKER_PERIODIC=" in result.output
    assert "--sweep" in config_path.read_text().splitlines()[0]
    metadata = json.loads((tmp_path / f"{geometry_type}_local.grid.json").read_text())
    assert metadata["sweep"]["n_planes"] == N_SWEEP_TEMPLATE
    assert metadata["nk"] == N_SWEEP_TEMPLATE


def test_cli_revolve_writes_su2_prisms_around_the_axis(tmp_path: Path) -> None:
    """A cone sector in SU2: prisms around the axis, no symmetry marker, Eilmer lines printed."""

    # run
    config_path = write_sweep_config(tmp_path, "cone")
    result = runner.invoke(app, ["cone", "run", "-c", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "cells around the axis" in result.output
    assert "Eilmer: bottom Rmatrix=" in result.output

    # read the element types and markers
    su2_lines = (tmp_path / "cone_local.su2").read_text().splitlines()
    elem_start = next(n for n, line in enumerate(su2_lines) if line.startswith("NELEM="))
    n_elements = int(su2_lines[elem_start].split("=")[1])
    types = [
        int(line.split()[0]) for line in su2_lines[elem_start + 1 : elem_start + 1 + n_elements]
    ]
    tags = [line.split("=")[1].strip() for line in su2_lines if line.startswith("MARKER_TAG")]

    # check: hexes and prisms (SU2 types 12, 13); the axis has no marker
    cells_k = N_SWEEP - 1
    assert types.count(13) == (NJ - 1) * cells_k
    assert types.count(12) == (NI - 2) * (NJ - 1) * cells_k
    assert tags == ["outflow", "wall", "farfield", "sweep_low", "sweep_high"]


@pytest.mark.parametrize("order", [1, 2])
def test_cli_revolve_writes_gmsh(tmp_path: Path, order: int) -> None:
    """A cone sector in Gmsh, first or second order; gmsh reads it with positive Jacobians."""

    # run
    replacements = (
        ('filename = "cone_local.su2"', 'filename = "cone_local.msh"'),
        ("order = 1", f"order = {order}"),
    )
    config_path = write_sweep_config(tmp_path, "cone", replacements)
    result = runner.invoke(app, ["cone", "run", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    # gmsh is optional; its wheel can also fail to load without system graphics libraries
    try:
        import gmsh
    except (ImportError, OSError) as error:
        pytest.skip(f"gmsh not available: {error}")

    # read
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "cone_local.msh"))
        groups = {
            gmsh.model.getPhysicalName(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups()
        }
        element_types, element_tags, _ = gmsh.model.mesh.getElements(dim=3)
        min_jacobian = min(
            gmsh.model.mesh.getElementQualities(tags, "minSJ").min() for tags in element_tags
        )
    finally:
        gmsh.finalize()

    # check: hexahedra and prisms, no group for the axis
    expected_types = {1: {5, 6}, 2: {12, 13}}[order]
    assert set(int(t) for t in element_types) == expected_types
    assert groups == {"outflow", "wall", "farfield", "sweep_low", "sweep_high", "fluid"}
    assert min_jacobian > 0.0


@pytest.mark.parametrize("suffix", [".h5", ".p3dfmt"])
def test_cli_structured_files_read_back(tmp_path: Path, suffix: str) -> None:
    """HDF5 and Plot3D keep the sector as it is, the collapsed axis face included."""

    # run
    output_name = f"cone_local{suffix}"
    replacements = (('filename = "cone_local.su2"', f'filename = "{output_name}"'),)
    config_path = write_sweep_config(tmp_path, "cone", replacements)
    result = runner.invoke(app, ["cone", "run", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    # read back and compare
    expected = generate_body_fitted(config_path).grid
    grid = read_file(tmp_path / output_name).grid
    assert grid.shape == expected.shape
    for name in ("x", "y", "z"):
        np.testing.assert_allclose(getattr(grid, name), getattr(expected, name), rtol=0, atol=1e-15)


def test_cli_scale_multiplies_the_coordinates_in_the_file(tmp_path: Path) -> None:
    """[output] scale = 1000 writes millimetres; the JSON's translation is in the file's units."""

    # run a plate slab at scale 1000 into HDF5
    replacements = (
        ('filename = "plate_local.su2"', 'filename = "plate_local.h5"'),
        ("scale = 1.0", "scale = 1000.0"),
    )
    config_path = write_sweep_config(tmp_path, "plate", replacements)
    result = runner.invoke(app, ["plate", "run", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    # check: coordinates and the transform scaled, the report in metres
    expected = generate_body_fitted(config_path).grid
    written = read_file(tmp_path / "plate_local.h5")
    np.testing.assert_allclose(written.grid.z, 1000.0 * expected.z, rtol=1e-15)
    np.testing.assert_allclose(written.grid.x, 1000.0 * expected.x, rtol=1e-15)
    metadata = json.loads((tmp_path / "plate_local.grid.json").read_text())
    assert metadata["coordinate_scale"] == 1000.0
    sweep_data = metadata["sweep"]
    assert sweep_data["translation"] == [0.0, 0.0, 1000.0 * WIDTH]
    assert "0.0, 0.0, 2.0 )" in sweep_data["su2"]
    assert f"extruded {WIDTH:.4g} m along z" in result.output
