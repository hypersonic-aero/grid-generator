"""Tests for shock-aligned grids ([shock]): the shock is the grid line j = j_s on every line."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
from bodies import write_geometry
from cfd_io import StructuredGrid
from typer.testing import CliRunner

from grid_generator import generate_body_fitted, load_body_fitted_config
from grid_generator.cli import app
from grid_generator.config.points import build_shock_layer_spec
from grid_generator.distributions import build_distribution
from grid_generator.quality import folded_cells, shock_line_quality
from grid_generator.shock import start_on_axis

runner = CliRunner()

# a 7 deg sphere-cone, 2.5 mm nose, to x = 0.1 m
NOSE_RADIUS = 2.5e-3
HALF_ANGLE = 7.0
X_END = 0.1

# points from the wall to the shock, and from the shock to the outer boundary
N_SHOCK_LAYER = 61
N_FREESTREAM = 11
FIRST_SPACING = 1.0e-6
END_SPACING_FRACTION = 0.02

# the cone's geometry file: `geometry cone init`'s file, with this cone
CONE_GEOMETRY = (
    ("nose_radius = 100.0e-6", f"nose_radius = {NOSE_RADIUS}"),
    ("length = 0.5", f"length = {X_END}"),
    ("half_angle = 7.0", f"half_angle = {HALF_ANGLE}"),
)

# a shock-aligned grid config; the tables in braces are filled per test
CONFIG_TEMPLATE = """
[geometry]
file = "geometry.toml"

[domain]
x_end = {x_end}
{domain_extra}

{shock}

[points.nose]
n_points = 41

[points.body]
type = "geometric"
n_points = 81

{normal_points}

[output]
filename = "grid.h5"
"""

SHOCK_TABLE = """
[shock]
curve = "shock.csv"
margin_factor = {margin_factor}
min_offset = {min_offset}
"""

SHOCK_LINE_POINTS = f"""
[points.shock_layer]
type = "vinokur"
n_points = {N_SHOCK_LAYER}
first_spacing = {FIRST_SPACING}
end_spacing_fraction = {END_SPACING_FRACTION}

[points.freestream]
type = "geometric"
n_points = {N_FREESTREAM}
"""

CONSTANT_HEIGHT_POINTS = """
[points.normal]
type = "geometric"
n_points = 61
first_spacing = 1.0e-6
"""


# --------------------------------------------------
# helpers
# --------------------------------------------------
def write_shock(path: Path, x_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Billig-type hyperbola: vertex 0.4 mm ahead of the nose, asymptote at 11 deg."""

    # x = -delta + R_c cot^2(beta) [sqrt(1 + y^2 tan^2(beta) / R_c^2) - 1]
    delta, radius, beta = 0.4e-3, 3.0e-3, math.radians(11.0)
    y = np.linspace(0.0, 0.05, 2001)
    x = -delta + radius / math.tan(beta) ** 2 * (
        np.sqrt(1.0 + (y * math.tan(beta) / radius) ** 2) - 1.0
    )
    keep = x <= x_end
    np.savetxt(path, np.column_stack([x[keep], y[keep]]), delimiter=",", header="x,y", comments="")

    return x[keep], y[keep]


def write_config(
    folder: Path,
    shock: bool = True,
    height: float | None = None,
    margin_factor: float = 1.2,
    min_offset: float = 1.0e-3,
    normal_points: str = SHOCK_LINE_POINTS,
) -> Path:
    """Write a cone config with a [shock] table and/or a [domain] height."""

    # assemble the optional parts
    shock_text = ""
    if shock:
        shock_text = SHOCK_TABLE.format(margin_factor=margin_factor, min_offset=min_offset)
    domain_extra = f"height = {height}" if height is not None else ""

    # write the grid config, and the cone it points at
    write_geometry(folder, "cone", CONE_GEOMETRY)
    path = folder / "grid.toml"
    path.write_text(
        CONFIG_TEMPLATE.format(
            x_end=X_END,
            domain_extra=domain_extra,
            shock=shock_text,
            normal_points=normal_points,
        ),
        encoding="utf-8",
    )

    return path


def shock_line_grid(folder: Path, **options) -> tuple[np.ndarray, np.ndarray, dict]:
    """Write a shock well past the outflow and a config, then generate: x, y, attrs."""

    # write the shock and the config, then generate
    write_shock(folder / "shock.csv", 2.0 * X_END)
    dataset = generate_body_fitted(load_body_fitted_config(write_config(folder, **options)))
    x = dataset.grid.x[:, :, 0]
    y = dataset.grid.y[:, :, 0]

    return x, y, dataset.attrs


def distance_to_polyline(
    point_x: np.ndarray, point_y: np.ndarray, line_x: np.ndarray, line_y: np.ndarray
) -> np.ndarray:
    """Shortest distance from each point to a polyline."""

    # nearest point on every segment, for every point
    edge_x = np.diff(line_x)
    edge_y = np.diff(line_y)
    along = (
        (point_x[:, None] - line_x[None, :-1]) * edge_x
        + (point_y[:, None] - line_y[None, :-1]) * edge_y
    ) / (edge_x**2 + edge_y**2)
    along = np.clip(along, 0.0, 1.0)
    gap_x = line_x[None, :-1] + along * edge_x - point_x[:, None]
    gap_y = line_y[None, :-1] + along * edge_y - point_y[:, None]
    distances = np.hypot(gap_x, gap_y).min(axis=1)

    return distances


# --------------------------------------------------
# the shock line
# --------------------------------------------------
def test_the_shock_curve_is_the_grid_line_j_s(tmp_path: Path) -> None:
    """Every line's point j_s = [points.shock_layer] n_points - 1 lies on the shock curve."""

    # generate
    shock_x, shock_y = write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    x, y, attrs = shock_line_grid(tmp_path)
    j_s = N_SHOCK_LAYER - 1

    # check: the grid names its shock line, and has both segments' points
    assert attrs["shock_line_j"] == j_s
    assert x.shape == (41 + 81 - 1, N_SHOCK_LAYER + N_FREESTREAM - 1)

    # check: j_s on the shock curve, to round-off
    distance = distance_to_polyline(x[:, j_s], y[:, j_s], shock_x, shock_y)
    assert distance.max() < 1.0e-12


def test_the_spacing_is_continuous_across_the_shock_line(tmp_path: Path) -> None:
    """The cells either side of j_s have the same size, and the line has no kink there."""

    # generate
    x, y, attrs = shock_line_grid(tmp_path)
    j_s = attrs["shock_line_j"]

    # cells below and above the shock line
    below_x = x[:, j_s] - x[:, j_s - 1]
    below_y = y[:, j_s] - y[:, j_s - 1]
    above_x = x[:, j_s + 1] - x[:, j_s]
    above_y = y[:, j_s + 1] - y[:, j_s]
    below = np.hypot(below_x, below_y)
    above = np.hypot(above_x, above_y)

    # turn of the grid line across j_s
    cross = below_x * above_y - below_y * above_x
    dot = below_x * above_x + below_y * above_y
    turn_deg = np.degrees(np.abs(np.arctan2(cross, dot)))

    # check
    np.testing.assert_allclose(above / below, 1.0, atol=1.0e-4)
    assert turn_deg.max() < 0.5


def test_the_spacing_at_the_shock_is_a_fraction_of_the_shock_distance(tmp_path: Path) -> None:
    """The last cell below j_s is end_spacing_fraction x the wall-to-shock distance."""

    # generate
    x, y, attrs = shock_line_grid(tmp_path)
    j_s = attrs["shock_line_j"]

    # wall-to-shock distance and the cell just below the shock
    shock_distance = np.hypot(x[:, j_s] - x[:, 0], y[:, j_s] - y[:, 0])
    below = np.hypot(x[:, j_s] - x[:, j_s - 1], y[:, j_s] - y[:, j_s - 1])

    # check: the fraction, up to the slight bend of the line
    np.testing.assert_allclose(below / shock_distance, END_SPACING_FRACTION, rtol=0.01)


def test_shock_line_grid_keeps_the_first_spacing_and_has_no_folded_cells(tmp_path: Path) -> None:
    """The wall spacing is the configured one on every line (none too short); all cells are valid."""

    # generate
    x, y, _ = shock_line_grid(tmp_path)

    # check: first spacing on every line, and no folded cells
    first = np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])
    np.testing.assert_allclose(first, FIRST_SPACING, rtol=1.0e-8)
    assert not np.any(folded_cells(x, y))


def test_the_shock_layer_spec_keeps_the_wall_spacing_on_short_lines() -> None:
    """On a line too short for both end spacings the shock end gives way, not the wall.

    201 points, 1 um at the wall, 2 % at the shock: a 10 mm line holds both; a 0.5 mm
    line (h = 2.5 um) can't (sqrt(1 um x 10 um) > h), so its shock spacing becomes
    h^2 / 1 um = 6.25 um; a 0.195 mm line (h = 0.975 um) can't hold 1 um cells at all,
    so its cells are uniform.
    """

    # long line: both spacings as configured
    long_spec = build_shock_layer_spec(201, 1.0e-6, 0.02, 10.0e-3)
    assert long_spec.first_spacing == 1.0e-6
    assert long_spec.end_spacing == pytest.approx(0.02 * 10.0e-3, rel=1.0e-15)

    # short line: the wall keeps its spacing, the shock spacing gives way
    short_spec = build_shock_layer_spec(201, 1.0e-6, 0.02, 0.5e-3)
    assert short_spec.first_spacing == 1.0e-6
    assert short_spec.end_spacing == pytest.approx((0.5e-3 / 200) ** 2 / 1.0e-6, rel=1.0e-12)

    # shorter than 200 x 1 um: uniform cells, the largest wall spacing that fits
    uniform_spec = build_shock_layer_spec(201, 1.0e-6, 0.02, 0.195e-3)
    uniform = 0.195e-3 / 200
    assert uniform_spec.first_spacing == pytest.approx(uniform, rel=1.0e-12)
    assert uniform_spec.end_spacing == pytest.approx(uniform, rel=1.0e-12)
    np.testing.assert_allclose(np.diff(build_distribution(uniform_spec)), uniform, rtol=1.0e-9)


def test_on_short_lines_the_wall_keeps_its_spacing_and_the_shock_end_gives_way(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Three kinds of line, from 201 points with 3 um at the wall and 2 % at the shock.

    Lines shorter than 200 x 3 um = 0.6 mm (the nose, from 0.4 mm) get uniform cells;
    lines up to 200^2 x 3 um x 2 % = 2.4 mm keep the 3 um at the wall and get a
    smaller spacing at the shock; longer lines get both spacings as configured. On
    every line the cells grow from both ends toward the middle, or monotonically
    from one end to the other: they never shrink toward the middle.
    """

    # generate with 201 points and a 3 um wall spacing
    n_points = 201
    wall_spacing = 3.0e-6
    points = SHOCK_LINE_POINTS.replace(
        f"n_points = {N_SHOCK_LAYER}", f"n_points = {n_points}"
    ).replace(f"first_spacing = {FIRST_SPACING}", f"first_spacing = {wall_spacing}")
    with caplog.at_level(logging.INFO, logger="grid_generator"):
        x, y, attrs = shock_line_grid(tmp_path, normal_points=points)
    j_s = attrs["shock_line_j"]
    assert j_s == n_points - 1

    # every line's cells from the wall to the shock, measured along its wall normal (the
    # direction [points.shock_layer] spaces the points in; the line bends across it)
    first_x = x[:, 1] - x[:, 0]
    first_y = y[:, 1] - y[:, 0]
    first = np.hypot(first_x, first_y)
    normal_x = (first_x / first)[:, None]
    normal_y = (first_y / first)[:, None]
    cells = (
        np.diff(x[:, : j_s + 1], axis=1) * normal_x + np.diff(y[:, : j_s + 1], axis=1) * normal_y
    )
    height = cells.sum(axis=1)
    uniform = height / (n_points - 1)

    # the three kinds of line
    is_uniform = height < (n_points - 1) * wall_spacing
    is_lowered = ~is_uniform & (height < (n_points - 1) ** 2 * wall_spacing * END_SPACING_FRACTION)
    is_long = ~is_uniform & ~is_lowered
    assert is_uniform.any() and is_lowered.any() and is_long.any()

    # check: uniform cells on the shortest lines
    np.testing.assert_allclose(cells[is_uniform] / uniform[is_uniform, None], 1.0, rtol=1.0e-5)

    # check: the wall spacing kept on every other line
    np.testing.assert_allclose(first[~is_uniform], wall_spacing, rtol=1.0e-8)

    # check: the spacing at the shock lowered to h^2 / wall spacing, or as configured
    shock_fraction = cells[:, -1] / height
    lowered_fraction = uniform**2 / wall_spacing / height
    np.testing.assert_allclose(
        shock_fraction[is_lowered], lowered_fraction[is_lowered], rtol=1.0e-5
    )
    np.testing.assert_allclose(shock_fraction[is_long], END_SPACING_FRACTION, rtol=1.0e-5)

    # check: no line has a cell smaller than a cell on each side of it
    largest_before = np.maximum.accumulate(cells, axis=1)[:, :-2]
    largest_after = np.maximum.accumulate(cells[:, ::-1], axis=1)[:, ::-1][:, 2:]
    bound = np.minimum(largest_before, largest_after)
    assert np.all(cells[:, 1:-1] >= bound * (1.0 - 1.0e-9))
    assert not np.any(folded_cells(x, y))

    # check: the quality report counts the lines the generator lowered (and logged)
    grid = StructuredGrid(x=x[:, :, None], y=y[:, :, None], z=np.zeros_like(x)[:, :, None])
    report = shock_line_quality(grid, j_s, wall_spacing, END_SPACING_FRACTION)
    assert report.lowered_wall_spacing == int(is_uniform.sum())
    assert report.lowered_shock_spacing == int((is_uniform | is_lowered).sum())
    assert f"first_spacing lowered on {report.lowered_wall_spacing} lines" in caplog.text
    assert f"spacing at the shock lowered on {report.lowered_shock_spacing} lines" in caplog.text
    assert report.min_wall_spacing == pytest.approx(uniform.min(), rel=1.0e-5)


# --------------------------------------------------
# the outer boundary
# --------------------------------------------------
def test_outer_boundary_stays_a_margin_beyond_the_shock(tmp_path: Path) -> None:
    """Every outer point is at least min_offset from the shock, on its far side."""

    # generate
    shock_x, shock_y = write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    x, y, _ = shock_line_grid(tmp_path, margin_factor=1.3, min_offset=5.0e-4)
    outer_x, outer_y = x[:, -1], y[:, -1]

    # nearest distance from each outer point to the (densely sampled) shock
    nearest = distance_to_polyline(outer_x, outer_y, shock_x, shock_y)

    # check: at least min_offset away, and outside (above the shock at the same x)
    assert np.all(nearest >= 0.99 * 5.0e-4)
    beside = outer_x > shock_x[0]
    assert np.all(outer_y[beside] > np.interp(outer_x[beside], shock_x, shock_y))


@pytest.mark.parametrize("j_line", ["shock", "outer"])
def test_shock_line_and_outer_boundary_are_smooth_through_the_nose_junction(
    tmp_path: Path, j_line: str
) -> None:
    """No corner and no spacing jump where the nose meets the cone (straight normals had both)."""

    # generate
    x, y, attrs = shock_line_grid(tmp_path)
    j = attrs["shock_line_j"] if j_line == "shock" else -1
    line_x, line_y = x[:, j], y[:, j]

    # direction change between neighbouring segments, and their length ratio
    heading = np.degrees(np.unwrap(np.arctan2(np.diff(line_y), np.diff(line_x))))
    turning = np.abs(np.diff(heading))
    segment = np.hypot(np.diff(line_x), np.diff(line_y))
    jump = np.maximum(segment[1:] / segment[:-1], segment[:-1] / segment[1:])

    # check around the junction (wall point 40 of 41 nose points): smooth turning, even spacing
    junction = 40
    around = slice(junction - 4, junction + 3)
    assert turning[around].max() < 1.5 * np.median(turning[junction - 10 : junction - 4])
    assert jump.max() < 1.2


def test_stagnation_line_meets_the_shock_vertex_and_ends_ahead_of_it(tmp_path: Path) -> None:
    """The line ahead of the nose (on the axis) crosses the shock at its vertex."""

    # generate
    x, y, attrs = shock_line_grid(tmp_path)
    j_s = attrs["shock_line_j"]

    # check: vertex at x = -0.4 mm, the outer boundary at -(1.2 * 0.4 mm + 1 mm)
    assert x[0, j_s] == pytest.approx(-0.4e-3, rel=1.0e-9)
    assert x[0, -1] == pytest.approx(-(1.2 * 0.4e-3 + 1.0e-3), rel=1.0e-6)
    np.testing.assert_allclose(y[0, :], 0.0, atol=1.0e-12)


# --------------------------------------------------
# shock curves taken from a CFD solution
# --------------------------------------------------
def test_noise_in_the_shock_curve_does_not_fold_the_grid(tmp_path: Path) -> None:
    """A sub-micron dent near the vertex, like grid-scale noise in a CFD shock, is harmless.

    Close to the vertex the shock is nearly vertical, so a 0.5 um dent makes x run
    backwards for a few points; pushed out along the raw normals by about 1.5 mm,
    the curve folded the grid there.
    """

    # the clean shock and its grid
    clean_folder = tmp_path / "clean"
    clean_folder.mkdir()
    shock_x, shock_y = write_shock(clean_folder / "shock.csv", 2.0 * X_END)
    clean = generate_body_fitted(load_body_fitted_config(write_config(clean_folder))).grid

    # the same shock, sampled every micron near the vertex and dented toward the body
    fine_y = np.arange(0.0, 1.0e-3, 1.0e-6)
    fine_x = np.interp(fine_y, shock_y, shock_x)
    fine_x = fine_x + 5.0e-7 * np.exp(-(((fine_y - 3.0e-5) / 5.0e-6) ** 2))
    coarse = shock_y >= 1.0e-3
    noisy_x = np.concatenate([fine_x, shock_x[coarse]])
    noisy_y = np.concatenate([fine_y, shock_y[coarse]])
    assert np.any(np.diff(noisy_x) < 0.0)

    # repeat a few points, as a CFD extraction can
    noisy_x = np.repeat(noisy_x, 2)[:-1]
    noisy_y = np.repeat(noisy_y, 2)[:-1]

    # generate from the noisy shock (a folded grid would raise here)
    noisy_folder = tmp_path / "noisy"
    noisy_folder.mkdir()
    np.savetxt(
        noisy_folder / "shock.csv",
        np.column_stack([noisy_x, noisy_y]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    noisy = generate_body_fitted(load_body_fitted_config(write_config(noisy_folder))).grid

    # check: the outer boundary barely moves, the shock line by no more than the dent
    j_s = N_SHOCK_LAYER - 1
    outer_shift = np.hypot(
        noisy.x[:, -1, 0] - clean.x[:, -1, 0], noisy.y[:, -1, 0] - clean.y[:, -1, 0]
    )
    shock_shift = np.hypot(
        noisy.x[:, j_s, 0] - clean.x[:, j_s, 0], noisy.y[:, j_s, 0] - clean.y[:, j_s, 0]
    )
    assert outer_shift.max() < 1.0e-6
    assert shock_shift.max() < 1.0e-6


def test_a_shock_curve_with_missing_values_is_rejected(tmp_path: Path) -> None:
    """nan or inf in the shock curve fails with a clear message."""

    # a shock with one value missing
    shock_x, shock_y = write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    shock_x[10] = np.nan
    np.savetxt(
        tmp_path / "shock.csv",
        np.column_stack([shock_x, shock_y]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    config = load_body_fitted_config(write_config(tmp_path))

    # check
    with pytest.raises(ValueError, match="aren't finite"):
        generate_body_fitted(config)


def test_a_shock_curve_for_another_body_is_refused(tmp_path: Path) -> None:
    """A curve made for another nose radius stands too far off this nose: refused, not built.

    The test shock stands 0.4 mm ahead of the nose, right for the 2.5 mm nose; on a
    25 um nose that is 16 nose radii, where no bow shock stands.
    """

    # the shock and config, then a body with a 100 times smaller nose
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    config_path = write_config(tmp_path)
    small_nose = (
        ("nose_radius = 100.0e-6", "nose_radius = 25.0e-6"),
        ("length = 0.5", f"length = {X_END}"),
    )
    write_geometry(tmp_path, "cone", small_nose)
    config = load_body_fitted_config(config_path)

    # check: the message names the curve's file and the distance in nose radii
    with pytest.raises(ValueError, match=r"shock\.csv\) crosses the axis .* 16 nose radii"):
        generate_body_fitted(config)


def test_a_shock_that_ends_too_early_is_reported(tmp_path: Path) -> None:
    """A shock curve that stops before the outflow fails with a clear message.

    The message names the curve's file, so it says which curve the grid read (e.g. an
    example's curve left in the folder instead of a new one).
    """

    # a shock that ends halfway along the body
    write_shock(tmp_path / "shock.csv", 0.5 * X_END)
    config = load_body_fitted_config(write_config(tmp_path))

    # check
    with pytest.raises(ValueError, match="too far before the outflow") as raised:
        generate_body_fitted(config)
    assert "shock.csv" in str(raised.value)


@pytest.mark.parametrize("shock_end", [X_END, X_END - 2.0e-3])
def test_a_shock_that_ends_at_the_outflow_is_continued(tmp_path: Path, shock_end: float) -> None:
    """A shock that stops at the outflow (or 2 mm before), like one taken from a CFD
    solution, is continued straight, and gives the grid of a longer shock."""

    # reference: the shock computed well past the outflow
    long_folder = tmp_path / "long"
    long_folder.mkdir()
    write_shock(long_folder / "shock.csv", 2.0 * X_END)
    reference = generate_body_fitted(load_body_fitted_config(write_config(long_folder))).grid

    # the same shock, cut at the outflow
    short_folder = tmp_path / "short"
    short_folder.mkdir()
    write_shock(short_folder / "shock.csv", shock_end)
    grid = generate_body_fitted(load_body_fitted_config(write_config(short_folder))).grid

    # check: the shock line and the outer boundary barely move (the shock is nearly
    # straight there)
    shift = np.hypot(grid.x[:, :, 0] - reference.x[:, :, 0], grid.y[:, :, 0] - reference.y[:, :, 0])
    assert shift.max() < 1.0e-5


def test_the_shock_path_is_relative_to_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Like [output], a relative shock path is resolved against the config's folder."""

    # config and shock in a subfolder, run from elsewhere
    case = tmp_path / "case"
    case.mkdir()
    write_shock(case / "shock.csv", 2.0 * X_END)
    config_path = write_config(case)
    monkeypatch.chdir(tmp_path)

    # check
    config = load_body_fitted_config(config_path)
    assert config.shock.curve == case / "shock.csv"
    assert generate_body_fitted(config).grid.shape[0] == 41 + 81 - 1


# --------------------------------------------------
# configuration errors
# --------------------------------------------------
@pytest.mark.parametrize(
    ("shock", "height", "normal_points", "message"),
    [
        (True, 0.05, SHOCK_LINE_POINTS, "not both"),
        (False, None, CONSTANT_HEIGHT_POINTS, "needs an outer boundary"),
        (True, None, CONSTANT_HEIGHT_POINTS, r"\[points.normal\] is for constant-height"),
        (
            True,
            None,
            SHOCK_LINE_POINTS.split("[points.freestream]")[0],
            r"needs \[points.freestream\]",
        ),
        (False, 0.05, SHOCK_LINE_POINTS, r"are for shock-aligned grids"),
        (False, 0.05, "", r"needs \[points.normal\]"),
    ],
)
def test_each_layout_takes_its_own_tables(
    tmp_path: Path, shock: bool, height: float | None, normal_points: str, message: str
) -> None:
    """Constant height takes [points.normal]; a shock line takes the two shock-aligned tables."""

    # write
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    config_path = write_config(tmp_path, shock=shock, height=height, normal_points=normal_points)

    # check
    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("margin_factor = 1.2\nmin_offset = 0.001", "margin_factor = 1.0", "on the shock"),
        ("end_spacing_fraction = 0.02", "end_spacing_fraction = 0.6", "end_spacing_fraction"),
        ('type = "vinokur"', 'type = "tanh"', "vinokur"),
        ("end_spacing_fraction = 0.02", "end_spacing = 0.02", "end_spacing"),
        (f"n_points = {N_FREESTREAM}", "n_points = 2", "n_points"),
        ('curve = "shock.csv"', 'shock = "shock.csv"', "curve"),
    ],
)
def test_load_rejects_bad_shock_tables(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Out-of-range or misspelled [shock] and shock-aligned [points] keys are rejected."""

    # write, then break one key
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    config_path = write_config(tmp_path)
    config_text = config_path.read_text(encoding="utf-8")
    assert old in config_text
    config_path.write_text(config_text.replace(old, new), encoding="utf-8")

    # check
    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(config_path)


def test_too_many_freestream_points_are_reported(tmp_path: Path) -> None:
    """Cells that would shrink toward the outer boundary fail, naming the line."""

    # no extra offset: 0.2 d beyond the shock, but 21 cells of 2 % of d need 0.42 d
    more_points = SHOCK_LINE_POINTS.replace(f"n_points = {N_FREESTREAM}", "n_points = 22")
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    config = load_body_fitted_config(
        write_config(tmp_path, min_offset=0.0, normal_points=more_points)
    )

    # check
    with pytest.raises(ValueError, match=r"wall x = 0 m .*\[points.freestream\] .* shrink"):
        generate_body_fitted(config)


def test_order_2_needs_odd_shock_line_counts(tmp_path: Path) -> None:
    """Second-order elements need odd counts in both shock-aligned tables."""

    # write, then ask for 9-node quads with an even shock-layer count
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    config_path = write_config(tmp_path)
    config_text = config_path.read_text(encoding="utf-8")
    config_text = config_text.replace('filename = "grid.h5"', 'filename = "grid.msh"\norder = 2')
    config_text = config_text.replace(f"n_points = {N_SHOCK_LAYER}", "n_points = 60")
    config_path.write_text(config_text, encoding="utf-8")

    # check
    with pytest.raises(ValueError, match=r"\[points.shock_layer\] 60 points"):
        load_body_fitted_config(config_path)


# --------------------------------------------------
# CLI
# --------------------------------------------------
def test_cli_run_reports_the_shock_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run prints the shock line's index, distances and spacings, and writes j_s to the file."""

    # write and run
    monkeypatch.chdir(tmp_path)
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    write_config(tmp_path)
    result = runner.invoke(app, ["cone", "run"])

    # check
    assert result.exit_code == 0, result.output
    assert f"shock line              j = {N_SHOCK_LAYER - 1}" in result.output
    assert "wall spacing lowered    0 lines" in result.output
    assert "shock spacing lowered   0 lines" in result.output
    assert (tmp_path / "grid.h5").exists()

    # check: the JSON next to the grid names the shock line, for formats without attributes
    metadata = json.loads((tmp_path / "grid.json").read_text(encoding="utf-8"))
    assert metadata["shock_line_j"] == N_SHOCK_LAYER - 1
    assert metadata["grid_file"] == "grid.h5"
    assert metadata["markers"]["structured"]["jmin"] == "wall"


def test_the_grid_record_holds_what_made_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """grid.json records versions, settings as applied, the body and the shock curve itself.

    The same inputs give the same record, byte for byte (no timestamp), so a diff of two
    records shows whether anything that made the grid changed.
    """

    # write and run twice
    monkeypatch.chdir(tmp_path)
    shock_x, shock_y = write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    write_config(tmp_path)
    first_run = runner.invoke(app, ["cone", "run"])
    first_record = (tmp_path / "grid.json").read_text(encoding="utf-8")
    second_run = runner.invoke(app, ["cone", "run"])
    second_record = (tmp_path / "grid.json").read_text(encoding="utf-8")
    record = json.loads(first_record)

    # check: the runs, and the same record twice
    assert first_run.exit_code == 0, first_run.output
    assert second_run.exit_code == 0, second_run.output
    assert first_record == second_record

    # check: versions, the config, and the body with its source
    assert set(record["versions"]) == {"grid-generator", "canonical-geometries", "cfd-io"}
    assert record["config_file"] == "grid.toml"
    assert record["body"]["source"] == "geometry.toml"
    assert record["body"]["nose_radius"] == NOSE_RADIUS

    # check: settings as applied, paths as the config gives them
    assert record["settings"]["domain"]["x_end"] == X_END
    assert record["settings"]["points"]["shock_layer"]["n_points"] == N_SHOCK_LAYER
    assert record["settings"]["shock"]["curve"] == "shock.csv"
    assert record["settings"]["output"]["filename"] == "grid.h5"

    # check: the curve itself, exactly as written
    assert record["shock_curve"]["file"] == "shock.csv"
    np.testing.assert_array_equal(record["shock_curve"]["x"], shock_x)
    np.testing.assert_array_equal(record["shock_curve"]["y"], shock_y)


@pytest.mark.parametrize("geometry_type", ["plate", "wedge", "cone", "tangent-ogive"])
def test_cli_init_shock_writes_a_runnable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, geometry_type: str
) -> None:
    """init --shock-aligned writes a shock-aligned config; with a shock curve next to it, it runs."""

    # the starter body, then init
    monkeypatch.chdir(tmp_path)
    write_geometry(tmp_path, geometry_type)
    init_result = runner.invoke(app, [geometry_type, "init", "--shock-aligned"])
    assert init_result.exit_code == 0, init_result.output
    config_text = Path("grid.toml").read_text(encoding="utf-8")
    assert "[shock]" in config_text
    assert "[points.normal]" not in config_text

    # a shock for the template's 0.1 mm nose: vertex 40 um ahead, 30 deg asymptote
    delta, radius, beta = 40.0e-6, 0.15e-3, math.radians(30.0)
    y = np.linspace(0.0, 0.02, 2001)
    x = -delta + radius / math.tan(beta) ** 2 * (
        np.sqrt(1.0 + (y * math.tan(beta) / radius) ** 2) - 1.0
    )
    np.savetxt("shock_shape.csv", np.column_stack([x, y]), delimiter=",", header="x,y", comments="")

    # run
    run_result = runner.invoke(app, [geometry_type, "run"])
    assert run_result.exit_code == 0, run_result.output
    assert "shock line" in run_result.output


def test_cli_init_rejects_shock_with_a_normal_choice(tmp_path: Path) -> None:
    """--normal picks [points.normal], which a shock-aligned grid doesn't have."""

    # check
    result = runner.invoke(
        app, ["cone", "init", "--shock-aligned", "--normal", "tanh", "-o", str(tmp_path / "c.toml")]
    )
    assert result.exit_code == 1
    assert "--normal" in result.output


# --------------------------------------------------
# the symmetry axis (i = 0)
# --------------------------------------------------
def write_off_axis_shock(path: Path, gap: float) -> None:
    """The test shock, starting a small gap above the axis (like cell-centre data)."""

    # the full curve, then only its part above the gap, from an interpolated first point
    x, y = write_shock(path, 2.0 * X_END)
    keep = y > gap
    first_x = float(np.interp(gap, y, x))
    off_x = np.concatenate([[first_x], x[keep]])
    off_y = np.concatenate([[gap], y[keep]])
    np.savetxt(path, np.column_stack([off_x, off_y]), delimiter=",", header="x,y", comments="")


def test_start_on_axis_continues_a_curve_that_starts_off_the_axis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A parabola x = a + b y^2 starting off the axis gets (a, 0) in front, with a warning."""

    # a symmetric curve near the axis, starting 10 um above it
    y = np.linspace(1.0e-5, 1.0e-3, 50)
    x = -4.0e-4 + 120.0 * y**2

    with caplog.at_level(logging.WARNING, logger="grid_generator"):
        axis_x, axis_y = start_on_axis(x, y, NOSE_RADIUS, "shock curve")

    assert axis_y[0] == 0.0
    assert axis_x[0] == pytest.approx(-4.0e-4, abs=1.0e-15)
    np.testing.assert_array_equal(axis_x[1:], x)
    np.testing.assert_array_equal(axis_y[1:], y)
    assert "off the axis" in caplog.text


def test_start_on_axis_leaves_a_curve_on_the_axis_alone() -> None:
    """A curve that starts on the axis (e.g. from vertex data) is returned as it is."""
    y = np.linspace(0.0, 1.0e-3, 50)
    x = -4.0e-4 + 120.0 * y**2

    axis_x, axis_y = start_on_axis(x, y, NOSE_RADIUS, "shock curve")

    assert axis_x is x
    assert axis_y is y


def test_start_on_axis_refuses_a_curve_that_misses_the_axis() -> None:
    """Far more than half a cell off the axis is not a gap to fill but a wrong curve."""
    y = np.linspace(0.5 * NOSE_RADIUS, 0.05, 50)
    x = -4.0e-4 + 0.2 * y

    with pytest.raises(ValueError, match="must start on the axis"):
        start_on_axis(x, y, NOSE_RADIUS, "shock curve")


def test_start_on_axis_refuses_a_curve_with_points_missing_near_the_axis() -> None:
    """Five point spacings off the axis (but within a tenth of the nose radius) is refused.

    Cell-centre data starts half a spacing above the axis; a curve that starts several
    spacings above it has lost points there, and the continuation would be a guess.
    """

    # points every 20 um, the first at 100 um (0.04 nose radii)
    y = np.linspace(1.0e-4, 1.0e-3, 46)
    x = -4.0e-4 + 120.0 * y**2

    with pytest.raises(ValueError, match="points are missing near the axis"):
        start_on_axis(x, y, NOSE_RADIUS, "shock curve")


def test_a_shock_curve_off_the_axis_keeps_the_symmetry_line_on_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A curve starting half a cell off the axis: line i = 0 still lies exactly on y = 0.

    Without the continuation, the line bent toward the curve's first point, and a solver
    compressed the freestream against that bent symmetry line ahead of the shock. With
    it, the grid is the one the full curve (starting on the axis) gives.
    """

    # the full curve's grid
    on_folder = tmp_path / "on"
    on_folder.mkdir()
    x_on, y_on, attrs = shock_line_grid(on_folder)

    # the same curve starting 10 um above the axis
    off_folder = tmp_path / "off"
    off_folder.mkdir()
    write_off_axis_shock(off_folder / "shock.csv", 1.0e-5)
    config = load_body_fitted_config(write_config(off_folder))
    with caplog.at_level(logging.WARNING, logger="grid_generator"):
        dataset = generate_body_fitted(config)
    x_off = dataset.grid.x[:, :, 0]
    y_off = dataset.grid.y[:, :, 0]

    # check: the symmetry line on the axis, the user told, the grid as with the full curve
    assert np.all(y_off[0, :] == 0.0)
    assert "off the axis" in caplog.text
    nose_cell = np.hypot(np.diff(x_on[:2, 0]), np.diff(y_on[:2, 0]))[0]
    assert np.hypot(x_off - x_on, y_off - y_on).max() < 1.0e-3 * nose_cell

    # check: the quality report says so
    report = shock_line_quality(
        dataset.grid, attrs["shock_line_j"], FIRST_SPACING, END_SPACING_FRACTION
    )
    assert report.axis_offset == 0.0
    assert "symmetry line off axis  0.000e+00 m" in report.summary()


def test_a_symmetry_line_off_the_axis_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check behind the fix: without the continuation, line i = 0 leaves the axis."""

    # the old behaviour: the curve used as it is
    monkeypatch.setattr(
        "grid_generator.body_fitted.start_on_axis",
        lambda shock_x, shock_y, radius, curve_name: (shock_x, shock_y),
    )
    write_off_axis_shock(tmp_path / "shock.csv", 1.0e-5)
    config = load_body_fitted_config(write_config(tmp_path))

    with pytest.raises(ValueError, match="symmetry line i = 0 leaves the axis"):
        generate_body_fitted(config)
