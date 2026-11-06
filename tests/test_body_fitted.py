"""Tests for constant-height body-fitted grids, their configuration, and the CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from canonical_geometries import BluntedTangentOgive
from cfd_io import read_file
from typer.testing import CliRunner

from grid_generator import (
    BodyFittedConfig,
    generate_body_fitted,
    grid_quality,
    load_body_fitted_config,
)
from grid_generator.cli import app
from grid_generator.config import (
    BODY_BLOCKS,
    BODY_FITTED_GEOMETRY_BLOCKS,
    NORMAL_BLOCKS,
    body_fitted_template,
)
from grid_generator.config.points import arc_length_at_x
from grid_generator.distributions import build_distribution
from grid_generator.quality import folded_cells

runner = CliRunner()

# the template's local plate case
NOSE_RADIUS = 100.0e-6
X_END = 0.01
HEIGHT = 5.0e-3
N_NOSE = 81
N_BODY = 121
N_NORMAL = 101
FIRST_NORMAL_SPACING = 1.0e-6

# the default boundary names per grid side ([markers] left at its defaults)
BOUNDARY_MARKERS = {"imin": "symmetry", "imax": "outflow", "jmin": "wall", "jmax": "farfield"}

# a [sweep] table that extrudes (planar bodies only; see test_sweep.py for sweeps)
EXTRUDE_TABLE = '[sweep]\ntype = "extrude"\nwidth = 2.0e-3\nn_points = 5'


def write_config(tmp_path: Path, old: str = "", new: str = "") -> Path:
    """Write the starter template, optionally with one replacement, and return its path."""

    # build the config text
    config_text = body_fitted_template("plate")
    if old:
        assert old in config_text
        config_text = config_text.replace(old, new, 1)

    # write
    config_path = tmp_path / "plate.toml"
    config_path.write_text(config_text)

    return config_path


@pytest.fixture
def plate_config(tmp_path: Path) -> BodyFittedConfig:
    """The template's 10 mm local plate grid."""
    return load_body_fitted_config(write_config(tmp_path))


@pytest.fixture
def cone_config(tmp_path: Path) -> BodyFittedConfig:
    """A 7 deg cone on the same grid settings."""
    config_path = write_config(tmp_path, 'type = "plate"', 'type = "cone"\nhalf_angle = 7.0')
    return load_body_fitted_config(config_path)


def wall_distances(config: BodyFittedConfig, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance of each grid point from the wall along the matching wall normal."""

    # wall points and normals, one per i
    profile = config.build_profile()
    x_wall = x[:, 0]
    y_wall = y[:, 0]
    s_wall = np.zeros_like(x_wall)

    # recover s from the wall points (nose: polar angle, body: distance from the junction)
    s_junction = profile.junction_arc_length
    x_j, y_j = profile.junction_point
    on_nose = np.arange(x_wall.size) < N_NOSE
    s_wall[on_nose] = NOSE_RADIUS * np.arctan2(y_wall[on_nose], NOSE_RADIUS - x_wall[on_nose])
    s_wall[~on_nose] = s_junction + np.hypot(x_wall[~on_nose] - x_j, y_wall[~on_nose] - y_j)
    nx, ny = profile.normal(s_wall)

    # projection of each point's offset from its wall point onto the normal
    distance = (x - x_wall[:, None]) * nx[:, None] + (y - y_wall[:, None]) * ny[:, None]

    return distance


# --------------------------------------------------
# grid layout
# --------------------------------------------------
def test_shape_and_markers(plate_config: BodyFittedConfig) -> None:
    """201 x 101 points (200 x 100 cells), 2D, with the four SU2 boundary markers."""

    # build
    dataset = generate_body_fitted(plate_config)

    # check
    assert dataset.grid.shape == (N_NOSE + N_BODY - 1, N_NORMAL, 1)
    assert dataset.attrs["markers"]["structured"] == BOUNDARY_MARKERS
    assert dataset.attrs["geometry_type"] == "plate"
    assert dataset.attrs["axisymmetric"] is False
    np.testing.assert_array_equal(dataset.grid.z, 0.0)


@pytest.mark.parametrize("config_name", ["plate_config", "cone_config"])
def test_wall_points_lie_on_the_profile(config_name: str, request) -> None:
    """j = 0 starts at the stagnation point, ends at x_end, and stays on the exact profile."""

    # build
    config = request.getfixturevalue(config_name)
    grid = generate_body_fitted(config).grid
    x_wall = grid.x[:, 0, 0]
    y_wall = grid.y[:, 0, 0]

    # check the ends
    assert (x_wall[0], y_wall[0]) == (0.0, 0.0)
    assert x_wall[-1] == pytest.approx(X_END, rel=1e-14)

    # check the nose points sit on the circle and the body points on the body line
    profile = config.build_profile()
    radius = np.hypot(x_wall[:N_NOSE] - NOSE_RADIUS, y_wall[:N_NOSE])
    np.testing.assert_allclose(radius, NOSE_RADIUS, rtol=1e-12)
    theta = math.radians(profile.half_angle)
    x_j, y_j = profile.junction_point
    off_line = -(x_wall[N_NOSE:] - x_j) * math.sin(theta) + (y_wall[N_NOSE:] - y_j) * math.cos(
        theta
    )
    np.testing.assert_allclose(off_line, 0.0, atol=1e-15)


@pytest.mark.parametrize("config_name", ["plate_config", "cone_config"])
def test_j_lines_are_wall_normals_of_constant_height(config_name: str, request) -> None:
    """Every j-line runs along its exact wall normal, and the outer boundary is at height H."""

    # build
    config = request.getfixturevalue(config_name)
    grid = generate_body_fitted(config).grid
    x = grid.x[:, :, 0]
    y = grid.y[:, :, 0]

    # distance along the normal, and the full offset from the wall point
    normal_distance = wall_distances(config, x, y)
    offset = np.hypot(x - x[:, :1], y - y[:, :1])

    # check: offsets are purely along the normal, and j = jmax is at H everywhere
    np.testing.assert_allclose(normal_distance, offset, atol=1e-15)
    np.testing.assert_allclose(normal_distance[:, -1], HEIGHT, rtol=1e-12)
    assert offset[0, 1] == pytest.approx(FIRST_NORMAL_SPACING, rel=1e-8)


def test_nose_is_a_polar_block(plate_config: BodyFittedConfig) -> None:
    """Nose wall points are equally spaced in angle from 0 to 90 deg; nose j-lines are radial."""

    # build
    grid = generate_body_fitted(plate_config).grid
    x_nose = grid.x[:N_NOSE, :, 0]
    y_nose = grid.y[:N_NOSE, :, 0]

    # polar angle of every nose point about the nose centre
    phi = np.arctan2(y_nose, NOSE_RADIUS - x_nose)

    # check: equal angle steps along i, constant angle along each j-line
    expected = np.linspace(0.0, 0.5 * math.pi, N_NOSE)
    np.testing.assert_allclose(phi[:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(phi, np.broadcast_to(phi[:, :1], phi.shape), atol=1e-12)


def test_wall_spacing_is_continuous_at_the_junction(plate_config: BodyFittedConfig) -> None:
    """The first body spacing equals the nose spacing, and the body spacing then grows."""

    # wall spacings in arc length (chord lengths on the nose, exact on the body)
    grid = generate_body_fitted(plate_config).grid
    wall_spacing = np.hypot(np.diff(grid.x[:, 0, 0]), np.diff(grid.y[:, 0, 0]))

    # nose arc spacing (chord of one angle step)
    angle_step = 0.5 * math.pi / (N_NOSE - 1)
    nose_chord = 2.0 * NOSE_RADIUS * math.sin(0.5 * angle_step)
    nose_arc = NOSE_RADIUS * angle_step

    # check
    np.testing.assert_allclose(wall_spacing[: N_NOSE - 1], nose_chord, rtol=1e-10)
    assert wall_spacing[N_NOSE - 1] == pytest.approx(nose_arc, rel=1e-8)
    assert np.all(np.diff(wall_spacing[N_NOSE - 1 :]) > 0.0)


@pytest.mark.parametrize("config_name", ["plate_config", "cone_config"])
def test_all_cells_valid_and_counter_clockwise(config_name: str, request) -> None:
    """Every cell has positive area with corners counter-clockwise in (i, j)."""

    # build and measure
    config = request.getfixturevalue(config_name)
    quality = grid_quality(generate_body_fitted(config).grid)

    # check
    assert quality.min_cell_area > 0.0
    assert quality.max_spacing_ratio_j < 1.1


# corners (i, j) of one unit cell: i along x, j along y (counter-clockwise)
UNIT_CELL_X = [[0.0, 0.0], [1.0, 1.0]]
UNIT_CELL_Y = [[0.0, 1.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    ("cell_x", "cell_y", "expected"),
    [
        (UNIT_CELL_X, UNIT_CELL_Y, False),
        # twisted: the corners at i + 1 swapped (a bow-tie)
        (UNIT_CELL_X, [[0.0, 1.0], [1.0, 0.0]], True),
        # collapsed: both i-lines at x = 0
        ([[0.0, 0.0], [0.0, 0.0]], UNIT_CELL_Y, True),
        # mirrored: clockwise in (i, j)
        ([[1.0, 1.0], [0.0, 0.0]], UNIT_CELL_Y, True),
        # a corner that isn't a number
        ([[0.0, 0.0], [1.0, float("nan")]], UNIT_CELL_Y, True),
    ],
)
def test_folded_cells_marks_twisted_collapsed_and_nan_cells(
    cell_x: list, cell_y: list, expected: bool
) -> None:
    """Only a cell that turns counter-clockwise at all four corners counts as valid."""

    # check
    folded = folded_cells(np.array(cell_x), np.array(cell_y))
    assert folded.shape == (1, 1)
    assert bool(folded[0, 0]) is expected


def test_cli_run_refuses_a_folded_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A grid with crossing lines is an error (exit 1), and no file is written."""

    # grid lines with the second wall point pushed past the third: two i-lines cross
    import grid_generator.body_fitted as body_fitted

    build_lines = body_fitted.build_grid_lines

    def crossing_lines(*arguments):
        x_plane, y_plane = build_lines(*arguments)
        x_plane[1, :] = x_plane[2, :] + 1.0e-3
        return x_plane, y_plane

    monkeypatch.setattr(body_fitted, "build_grid_lines", crossing_lines)

    # init and run
    config_path = tmp_path / "plate.toml"
    runner.invoke(app, ["plate", "init", "--output", str(config_path)])
    result = runner.invoke(app, ["plate", "run", "-c", str(config_path)])

    # check: one error line, nothing written
    assert result.exit_code == 1
    assert "error: " in result.output
    assert "folded or collapsed cells between i = 1 and 2" in result.output
    assert not (tmp_path / "plate_local.su2").exists()


def test_load_rejects_p3dfmt_without_sweep(tmp_path: Path) -> None:
    """A 2D grid can't go to a .p3dfmt file (the extension names the dimensionality)."""

    # write a 2D config with the 3D extension
    config_path = write_config(tmp_path, "plate_local.su2", "plate_local.p3dfmt")

    with pytest.raises(ValueError, match="grid is 2D"):
        load_body_fitted_config(config_path)


# --------------------------------------------------
# point distributions
# --------------------------------------------------
# the template's geometric body table, and a segmented body in x: the nose spacing up to
# 0.2 mm, a quintic transition up to 2 mm, then 50 micron spacing to x_end
GEOMETRIC_BODY = (
    '[points.body]\ntype = "geometric"\n'
    "# points on the body; the growth ratio follows from the count\nn_points = 121\n"
)
SEGMENTED_BODY = """[points.body]
type = "segmented"
[[points.body.segments]]
type = "equidistant"
end = 0.2e-3
[[points.body.segments]]
type = "quintic"
end = 2.0e-3
n_points = 81
[[points.body.segments]]
type = "equidistant"
spacing = 50.0e-6
"""
FINE_SPACING_END = 0.2e-3
TRANSITION_END = 2.0e-3
COARSE_SPACING = 50.0e-6


def write_template(
    tmp_path: Path,
    geometry_type: str = "plate",
    replacements: tuple[tuple[str, str], ...] = (),
    body: str = "geometric",
) -> Path:
    """Write a starter template with any number of text replacements, and return its path."""

    # build the config text
    config_text = body_fitted_template(geometry_type, body=body)
    for old, new in replacements:
        assert old in config_text
        config_text = config_text.replace(old, new, 1)

    # write
    config_path = tmp_path / f"{geometry_type}.toml"
    config_path.write_text(config_text)

    return config_path


@pytest.mark.parametrize("geometry_type", ["plate", "wedge", "tangent-ogive"])
def test_segmented_body_is_placed_in_x(tmp_path: Path, geometry_type: str) -> None:
    """Segment ends are in x, spacings along the wall; the body starts at the nose spacing."""

    # build
    config_path = write_template(tmp_path, geometry_type, body="segmented")
    config = load_body_fitted_config(config_path)
    grid = generate_body_fitted(config).grid
    x_wall = grid.x[:, 0, 0]
    y_wall = grid.y[:, 0, 0]
    wall_spacing = np.hypot(np.diff(x_wall), np.diff(y_wall))

    # nose spacing in arc length, and the first body spacing after the junction
    profile = config.build_profile()
    nose_spacing = profile.junction_arc_length / (N_NOSE - 1)
    assert wall_spacing[N_NOSE - 1] == pytest.approx(nose_spacing, rel=1e-8)

    # check the regions in x: fine end within one nose spacing of 0.2 mm (snapped to a
    # whole number of spacings), transition end exactly at 2 mm
    assert np.min(np.abs(x_wall - FINE_SPACING_END)) < nose_spacing
    assert np.min(np.abs(x_wall - TRANSITION_END)) < 1e-12
    assert x_wall[-1] == pytest.approx(X_END, rel=1e-14)

    # check the coarse region: equal steps along the wall, 50 micron rounded to whole
    # cells over the 8 mm (in x) from 2 mm to x_end, so within 1 %
    coarse = wall_spacing[-10:]
    np.testing.assert_allclose(coarse, coarse[0], rtol=1e-9)
    assert coarse[0] == pytest.approx(COARSE_SPACING, rel=1e-2)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            'type = "geometric"\n# points in the wall-normal',
            'type = "tanh"\nstrength = 3.0\ncluster = "both"\n# points in the wall-normal',
        ),
        (
            "to the height\n# max_spacing = 1.0e-4",
            "to the height\nmax_spacing = 1.0e-4",
        ),
    ],
)
def test_normal_takes_other_distributions(tmp_path: Path, old: str, new: str) -> None:
    """The normal direction takes any distribution; it always ends at the height."""

    # tanh needs no first_spacing; drop it there
    replacements = [(old, new)]
    if "tanh" in new:
        replacements.append(("first_spacing = 1.0e-6\n", ""))

    # build
    config = load_body_fitted_config(write_template(tmp_path, replacements=tuple(replacements)))
    grid = generate_body_fitted(config).grid
    outer = np.hypot(grid.x[:, -1, 0] - grid.x[:, 0, 0], grid.y[:, -1, 0] - grid.y[:, 0, 0])
    normal_spacing = np.hypot(np.diff(grid.x[-1, :, 0]), np.diff(grid.y[-1, :, 0]))

    # check: height reached everywhere; capped growth ends at the cap
    np.testing.assert_allclose(outer, HEIGHT, rtol=1e-12)
    if "max_spacing" in new:
        assert normal_spacing[-1] == pytest.approx(1.0e-4, rel=1e-9)
    else:
        assert normal_spacing[0] == pytest.approx(normal_spacing[-1], rel=1e-9)


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        (
            (('type = "geometric"\n# points on the body', 'type = "tanh"\n# points on the body'),),
            "can't start from the nose spacing",
        ),
        (
            (("n_points = 121\n", "n_points = 121\nfirst_spacing = 1.0e-6\n"),),
            r"\[points.body\] don't set first_spacing",
        ),
        (
            (("first_spacing = 1.0e-6\n", "first_spacing = 1.0e-6\nstart = 0.0\n"),),
            r"\[points.normal\] don't set start",
        ),
        # a first spacing too large for the height: the cells would shrink
        (
            (("first_spacing = 1.0e-6\n", "first_spacing = 1.0e-4\n"),),
            "cells would shrink away from the wall",
        ),
        (
            (
                (
                    GEOMETRIC_BODY,
                    SEGMENTED_BODY.replace("spacing = 50.0e-6", "end = 0.01\nspacing = 50.0e-6"),
                ),
            ),
            r"the last segment ends at \[domain\] x_end; remove its end",
        ),
        (
            (
                (
                    GEOMETRIC_BODY,
                    '[points.body]\ntype = "segmented"\n[[points.body.segments]]\ntype = "equidistant"\n',
                ),
            ),
            "at least two segments",
        ),
        (
            (
                (
                    GEOMETRIC_BODY,
                    SEGMENTED_BODY.replace("end = 0.2e-3", "end = 0.2e-3\nn_points = 5"),
                ),
            ),
            "its spacing is the nose spacing",
        ),
        (
            ((GEOMETRIC_BODY, SEGMENTED_BODY.replace("n_points = 81", "n_points = 121")),),
            r"\[points.body\] distribution coordinates must be strictly increasing",
        ),
    ],
)
def test_load_rejects_bad_points(
    tmp_path: Path, replacements: tuple[tuple[str, str], ...], message: str
) -> None:
    """Body types that can't start at the nose spacing, fixed keys and bad segments raise."""

    # write a broken config
    config_path = write_template(tmp_path, replacements=replacements)

    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


# --------------------------------------------------
# tangent ogive (curved body, optional cylinder)
# --------------------------------------------------
# a short ogive (base at about x = 20 mm, R = 4 mm), so x_end can lie past its base
SHORT_OGIVE = (
    ("length = 0.5", "length = 0.02"),
    ("base_radius = 0.05", "base_radius = 4.0e-3"),
)


def test_arc_length_at_x_inverts_the_profile() -> None:
    """arc_length_at_x finds the wall arc length of an x on a curved body to round-off."""

    # ogive profile and arc lengths along its body
    profile = BluntedTangentOgive(nose_radius=NOSE_RADIUS, length=0.5, base_radius=0.05)
    s_junction = profile.junction_arc_length
    s_body = np.linspace(s_junction, profile.total_length, 50)
    x_body, _ = profile.point(s_body)

    # check: back from x to s
    s_found = arc_length_at_x(profile, x_body, s_junction)
    np.testing.assert_allclose(s_found, s_body, rtol=0.0, atol=1e-15)


def test_ogive_grid_ends_at_x_end_on_the_ogive(tmp_path: Path) -> None:
    """With x_end on the ogive, the wall runs around the nose and along the arc to x_end."""

    # build the template's ogive grid (x_end = 10 mm, base at about 0.5 m)
    config = load_body_fitted_config(write_template(tmp_path, "tangent-ogive"))
    grid = generate_body_fitted(config).grid
    x_wall = grid.x[:, 0, 0]
    y_wall = grid.y[:, 0, 0]

    # nose circle around (r_n, 0) and ogive arc around (x_base, R - rho)
    profile = config.build_profile()
    x_base, base_radius = profile.base_point
    rho = profile.ogive_radius
    on_nose = np.arange(x_wall.size) < N_NOSE
    nose_distance = np.hypot(x_wall[on_nose] - NOSE_RADIUS, y_wall[on_nose])
    ogive_distance = np.hypot(x_wall[~on_nose] - x_base, y_wall[~on_nose] - (base_radius - rho))

    # check: shape, end at x_end, points on the exact pieces, outer boundary at H
    assert grid.shape == (N_NOSE + N_BODY - 1, N_NORMAL, 1)
    assert x_wall[-1] == pytest.approx(X_END, rel=1e-14)
    np.testing.assert_allclose(nose_distance, NOSE_RADIUS, rtol=1e-12)
    np.testing.assert_allclose(ogive_distance, rho, rtol=1e-14)
    outer = np.hypot(grid.x[:, -1, 0] - x_wall, grid.y[:, -1, 0] - y_wall)
    np.testing.assert_allclose(outer, HEIGHT, rtol=1e-12)
    assert grid_quality(grid).min_cell_area > 0.0


def test_ogive_grid_continues_on_a_cylinder_past_the_base(tmp_path: Path) -> None:
    """With x_end past the base, a cylinder of the base radius carries the wall to x_end."""

    # short ogive, x_end 10 mm past its base
    replacements = (*SHORT_OGIVE, ("x_end = 0.01", "x_end = 0.03"))
    config = load_body_fitted_config(write_template(tmp_path, "tangent-ogive", replacements))
    grid = generate_body_fitted(config).grid
    x_wall = grid.x[:, 0, 0]
    y_wall = grid.y[:, 0, 0]

    # check: two junctions (nose/ogive, ogive/cylinder), last point (x_end, R), wall on the
    # cylinder at y = R, valid cells
    profile = config.build_profile()
    x_base, base_radius = profile.base_point
    assert profile.junctions.size == 2
    assert x_wall[-1] == pytest.approx(0.03, rel=1e-14)
    assert y_wall[-1] == pytest.approx(base_radius, rel=1e-14)
    np.testing.assert_allclose(y_wall[x_wall > x_base + 1e-12], base_radius, rtol=1e-14)
    assert grid_quality(grid).min_cell_area > 0.0


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        (
            (("base_radius = 0.05", "base_radius = 0.05\ncylinder_length = 0.1"),),
            "cylinder_length isn't used",
        ),
        ((("length = 0.5\n", ""),), "length"),
        ((("x_end = 0.01", "x_end = 50.0e-6"),), "must reach past the nose"),
        ((("scale = 1.0\n", "scale = 1.0\n\n" + EXTRUDE_TABLE + "\n"),), "axisymmetric"),
    ],
)
def test_load_rejects_bad_ogive_config(
    tmp_path: Path, replacements: tuple[tuple[str, str], ...], message: str
) -> None:
    """cylinder_length comes from x_end, length is required, x_end must pass the nose, no extrusion."""

    # write a broken config
    config_path = write_template(tmp_path, "tangent-ogive", replacements)

    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


# --------------------------------------------------
# init options: every template combination runs
# --------------------------------------------------
@pytest.mark.parametrize("geometry_type", list(BODY_FITTED_GEOMETRY_BLOCKS))
@pytest.mark.parametrize("body", list(BODY_BLOCKS))
@pytest.mark.parametrize("normal", list(NORMAL_BLOCKS))
def test_cli_init_options_write_runnable_configs(
    tmp_path: Path, geometry_type: str, body: str, normal: str
) -> None:
    """init --body/--normal writes a config that runs, and names the options in its header."""

    # init with the options, then run
    config_path = tmp_path / f"{geometry_type}.toml"
    options = ["--body", body, "--normal", normal]
    result_init = runner.invoke(app, [geometry_type, "init", "-o", str(config_path), *options])
    result_run = runner.invoke(app, [geometry_type, "run", "-c", str(config_path)])

    # check
    assert result_init.exit_code == 0, result_init.output
    assert result_run.exit_code == 0, result_run.output
    config_text = config_path.read_text()
    assert f'[points.body]\ntype = "{body}"' in config_text
    assert f'[points.normal]\ntype = "{normal}"' in config_text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["plate", "init", "--body", "tanh"], "--body must be one of"),
        (["plate", "init", "--normal", "cubic"], "--normal must be one of"),
    ],
)
def test_cli_init_rejects_bad_options(tmp_path: Path, arguments: list[str], message: str) -> None:
    """Bad option values fail before any file is written."""

    # run
    config_path = tmp_path / "config.toml"
    result = runner.invoke(app, [*arguments, "-o", str(config_path)])

    # check
    assert result.exit_code == 1
    assert message in result.output
    assert not config_path.exists()


def test_normal_segmented_ends_at_the_height(tmp_path: Path) -> None:
    """A segmented normal table leaves out the last end; the last segment ends at the height."""

    # equidistant near the wall, a quintic transition, then 0.1 mm spacing to the height
    normal_block = NORMAL_BLOCKS["geometric"]
    segmented_normal = """\
[points.normal]
type = "segmented"
[[points.normal.segments]]
type = "equidistant"
end = 1.0e-4
n_points = 11
[[points.normal.segments]]
type = "quintic"
end = 1.0e-3
n_points = 21
[[points.normal.segments]]
type = "equidistant"
spacing = 1.0e-4
"""
    config = load_body_fitted_config(
        write_template(tmp_path, replacements=((normal_block, segmented_normal),))
    )
    normal_distances = build_distribution(config.normal_distribution())

    # check: ends at the height, uniform 0.1 mm cells at the top
    assert normal_distances[-1] == HEIGHT
    np.testing.assert_allclose(np.diff(normal_distances)[-5:], 1.0e-4, rtol=1e-9)


# --------------------------------------------------
# boundary names ([markers])
# --------------------------------------------------
def test_markers_rename_the_sides_in_the_su2_file(tmp_path: Path) -> None:
    """[markers] names reach the dataset and the SU2 file; sides may share a name."""

    # rename the wall, and merge outflow into the outer boundary
    replacements = (
        ('wall = "wall"', 'wall = "plate_wall"'),
        ('outflow = "outflow"', 'outflow = "farfield"'),
    )
    config_path = write_template(tmp_path, replacements=replacements)
    dataset = generate_body_fitted(config_path)
    result = runner.invoke(app, ["plate", "run", "-c", str(config_path)])

    # check the dataset
    assert result.exit_code == 0, result.output
    assert dataset.attrs["markers"]["structured"] == {
        "imin": "symmetry",
        "imax": "farfield",
        "jmin": "plate_wall",
        "jmax": "farfield",
    }

    # check the SU2 file: three markers, the shared one holding both sides' faces
    su2_lines = (tmp_path / "plate_local.su2").read_text().splitlines()
    tags = [line.split("=")[1].strip() for line in su2_lines if line.startswith("MARKER_TAG")]
    counts = [
        int(su2_lines[n + 1].split("=")[1])
        for n, line in enumerate(su2_lines)
        if line.startswith("MARKER_TAG")
    ]
    marker_faces = dict(zip(tags, counts, strict=True))
    assert sorted(marker_faces) == ["farfield", "plate_wall", "symmetry"]
    assert marker_faces["farfield"] == (N_NOSE + N_BODY - 2) + (N_NORMAL - 1)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('wall = "wall"', 'wall = "no slip"', "must be one word"),
        ('wall = "wall"', 'wall = ""', "must be one word"),
        ('wall = "wall"', 'wall = "Fluid"', "reserved"),
        ('wall = "wall"', 'wal = "wall"', "wal"),
        ('symmetry = "symmetry"', 'symmetry = "symmetry"\nsweep_low = "a"', "only apply with a"),
    ],
)
def test_load_rejects_bad_markers(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Names with spaces, empty or reserved names, unknown keys, side face names without a sweep."""

    # write a broken config
    config_path = write_config(tmp_path, old, new)

    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


# --------------------------------------------------
# configuration
# --------------------------------------------------
@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("nose_radius = 100.0e-6", "nose_radius = 100.0e-6\nlength = 0.5", "length isn't used"),
        ("x_end = 0.01", "x_end = 50.0e-6", "must reach past the nose"),
        ("first_spacing = 1.0e-6", "first_spacing = 1.0e-2", "first_spacing must be between"),
        ("n_points = 81", "n_points = 1", "n_points"),
        ("height = 5.0e-3", "hieght = 5.0e-3", "hieght"),
        ('type = "plate"', 'type = "cone"', "half_angle"),
    ],
)
def test_load_rejects_bad_config(tmp_path: Path, old: str, new: str, message: str) -> None:
    """x_end outside the body, a too-large first spacing, bad counts and keys raise."""

    # write a broken config
    config_path = write_config(tmp_path, old, new)

    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


def write_order_2_config(tmp_path: Path, old: str = "", new: str = "") -> Path:
    """Write the template as a second-order .msh config, optionally with one more replacement."""

    # Gmsh output, second order
    config_text = body_fitted_template("plate")
    config_text = config_text.replace(
        'filename = "plate_local.su2"', 'filename = "plate_local.msh"'
    )
    config_text = config_text.replace("order = 1", "order = 2")

    # optional extra change
    if old:
        assert old in config_text
        config_text = config_text.replace(old, new, 1)

    # write
    config_path = tmp_path / "plate.toml"
    config_path.write_text(config_text)

    return config_path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('filename = "plate_local.msh"', 'filename = "plate_local.su2"', "needs a .msh filename"),
        ("n_points = 81", "n_points = 80", r"\[points.nose\] 80 points"),
        ("n_points = 121", "n_points = 120", r"\[points.body\] 120 points"),
        ("n_points = 101", "n_points = 100", r"\[points.normal\] 100 points"),
        ("order = 2", "order = 3", "order"),
    ],
)
def test_load_rejects_bad_order(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Order 2 needs .msh output and odd point counts (junction on an element corner)."""

    # write a broken config
    config_path = write_order_2_config(tmp_path, old, new)

    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


def test_load_keeps_the_order(tmp_path: Path) -> None:
    """Resolving the output path keeps the rest of [output]."""

    # check
    config = load_body_fitted_config(write_order_2_config(tmp_path))
    assert config.output.order == 2
    assert config.output.filename == tmp_path / "plate_local.msh"


def test_load_resolves_output_next_to_the_config(
    plate_config: BodyFittedConfig, tmp_path: Path
) -> None:
    """A relative output filename is resolved against the config file's folder."""

    # check
    assert plate_config.output.filename == tmp_path / "plate_local.su2"


# --------------------------------------------------
# CLI end to end
# --------------------------------------------------
def test_cli_init_then_run_writes_su2(tmp_path: Path) -> None:
    """plate init + run writes an SU2 grid with the right size and four markers."""

    # init and run
    config_path = tmp_path / "plate.toml"
    result_init = runner.invoke(app, ["plate", "init", "--output", str(config_path)])
    result_run = runner.invoke(app, ["plate", "run", "-c", str(config_path)])

    # check the commands and the report
    assert result_init.exit_code == 0, result_init.output
    assert result_run.exit_code == 0, result_run.output
    assert "200 x 100 cells" in result_run.output
    assert "smallest cell area" in result_run.output

    # check the file: sizes and marker names in the SU2 header blocks
    su2_text = (tmp_path / "plate_local.su2").read_text()
    assert "NDIME= 2" in su2_text
    assert "NELEM= 20000" in su2_text
    assert "NPOIN= 20301" in su2_text
    for marker in BOUNDARY_MARKERS.values():
        assert f"MARKER_TAG= {marker}" in su2_text


@pytest.mark.parametrize("geometry_type", ["wedge", "cone", "tangent-ogive"])
def test_cli_init_then_run_other_geometries(tmp_path: Path, geometry_type: str) -> None:
    """wedge and cone init + run write their own template and grid with the default names."""

    # init and run with the default config name, inside tmp_path
    config_path = tmp_path / f"{geometry_type}.toml"
    result_init = runner.invoke(app, [geometry_type, "init", "--output", str(config_path)])
    result_run = runner.invoke(app, [geometry_type, "run", "-c", str(config_path)])

    # check the commands, the template's type and the output file
    assert result_init.exit_code == 0, result_init.output
    assert result_run.exit_code == 0, result_run.output
    assert f'type = "{geometry_type}"' in config_path.read_text()
    assert f"Generated {geometry_type} grid" in result_run.output
    assert (tmp_path / f"{geometry_type}_local.su2").exists()


def test_cli_run_rejects_a_config_for_another_geometry(tmp_path: Path) -> None:
    """A plate config given to `cone run` fails and names the right command."""

    # plate config, cone command
    config_path = write_config(tmp_path)
    result = runner.invoke(app, ["cone", "run", "-c", str(config_path)])

    # check: error points at the plate command, nothing written
    assert result.exit_code == 1
    assert "grid-generator plate run" in result.output
    assert not (tmp_path / "plate_local.su2").exists()


def test_cli_su2_reads_back_with_the_same_points(tmp_path: Path) -> None:
    """cfd-io reads the SU2 file back with the generated wall and outer points."""

    # init and run
    config_path = tmp_path / "plate.toml"
    runner.invoke(app, ["plate", "init", "--output", str(config_path)])
    runner.invoke(app, ["plate", "run", "-c", str(config_path)])

    # read back and compare point clouds (SU2 is unstructured, so compare sorted coordinates)
    expected = generate_body_fitted(config_path).grid
    written = read_file(tmp_path / "plate_local.su2", orient=False)
    expected_points = np.column_stack([expected.x.ravel(), expected.y.ravel()])
    written_points = np.column_stack([np.ravel(written.grid.x), np.ravel(written.grid.y)])
    expected_sorted = expected_points[np.lexsort(expected_points.T)]
    written_sorted = written_points[np.lexsort(written_points.T)]
    np.testing.assert_allclose(written_sorted, expected_sorted, rtol=1e-12, atol=1e-18)


def test_cli_hdf5_keeps_points_and_markers(tmp_path: Path) -> None:
    """HDF5 output reads back with the same structured points and the boundary markers."""

    # config writing HDF5 instead of SU2
    config_path = write_config(tmp_path, "plate_local.su2", "plate_local.h5")
    result = runner.invoke(app, ["plate", "run", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    # read back
    expected = generate_body_fitted(config_path).grid
    written = read_file(tmp_path / "plate_local.h5", orient=False)

    # check: same (ni, nj, 1) points in double precision, markers restored as a dict
    np.testing.assert_array_equal(written.grid.x, expected.x)
    np.testing.assert_array_equal(written.grid.y, expected.y)
    assert written.attrs["markers"] == {"structured": BOUNDARY_MARKERS}


@pytest.mark.parametrize(
    ("order", "line_type", "quad_type", "n_quads"), [(1, 1, 3, 20000), (2, 8, 10, 5000)]
)
def test_cli_msh_has_the_element_order(
    tmp_path: Path, order: int, line_type: int, quad_type: int, n_quads: int
) -> None:
    """.msh output uses the configured order: 4-node quads per cell, or 9-node quads per 2 x 2 cells."""

    # config writing Gmsh at the given order
    config_path = write_order_2_config(tmp_path, "order = 2", f"order = {order}")
    result = runner.invoke(app, ["plate", "run", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    # element block headers: "dimension entity type count"
    msh_lines = (tmp_path / "plate_local.msh").read_text().splitlines()
    elements_start = msh_lines.index("$Elements")
    n_blocks = int(msh_lines[elements_start + 1].split()[0])
    block_headers = []
    cursor = elements_start + 2
    for _ in range(n_blocks):
        dim, _, element_type, count = (int(v) for v in msh_lines[cursor].split())
        block_headers.append((dim, element_type, count))
        cursor += 1 + count

    # check: four boundary-line blocks, then the quads
    assert [header[1] for header in block_headers[:4]] == [line_type] * 4
    assert block_headers[4] == (2, quad_type, n_quads)


def test_cli_run_reports_errors_cleanly(tmp_path: Path) -> None:
    """A bad config exits with an error message, not a traceback."""

    # config with x_end inside the nose
    config_path = write_config(tmp_path, "x_end = 0.01", "x_end = 50.0e-6")
    result = runner.invoke(app, ["plate", "run", "-c", str(config_path)])

    # check
    assert result.exit_code == 1
    assert "must reach past the nose" in result.output
