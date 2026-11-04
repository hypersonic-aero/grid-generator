"""Tests for the shock-aligned outer boundary ([outer])."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from grid_generator import generate_body_fitted, load_body_fitted_config

# a 7 deg sphere-cone, 2.5 mm nose, to x = 0.1 m
NOSE_RADIUS = 2.5e-3
HALF_ANGLE = 7.0
X_END = 0.1

# a shock-aligned grid config; {outer} and {domain_extra} are filled per test
CONFIG_TEMPLATE = """
[geometry]
type = "cone"
nose_radius = {nose_radius}
half_angle = {half_angle}

[domain]
x_end = {x_end}
{domain_extra}

{outer}

[points.nose]
n_points = 41

[points.body]
type = "geometric"
n_points = 81

[points.normal]
type = "geometric"
n_points = 61
first_spacing = 1.0e-6

[output]
filename = "grid.h5"
"""

OUTER_TABLE = """
[outer]
shock = "shock.csv"
margin_factor = {margin_factor}
min_offset = {min_offset}
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
    outer: bool = True,
    height: float | None = None,
    margin_factor: float = 1.2,
    min_offset: float = 1.0e-3,
) -> Path:
    """Write a cone config with an [outer] table and/or a [domain] height."""

    # assemble the optional parts
    outer_text = (
        OUTER_TABLE.format(margin_factor=margin_factor, min_offset=min_offset) if outer else ""
    )
    domain_extra = f"height = {height}" if height is not None else ""

    # write
    path = folder / "cone.toml"
    path.write_text(
        CONFIG_TEMPLATE.format(
            nose_radius=NOSE_RADIUS,
            half_angle=HALF_ANGLE,
            x_end=X_END,
            domain_extra=domain_extra,
            outer=outer_text,
        ),
        encoding="utf-8",
    )

    return path


# --------------------------------------------------
# shock-aligned grids
# --------------------------------------------------
def outer_boundary(tmp_path: Path, **margins) -> tuple[np.ndarray, np.ndarray]:
    """Outer boundary (j = jmax) of a shock-aligned test grid."""

    # write the shock and the config, then generate
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    grid = generate_body_fitted(load_body_fitted_config(write_config(tmp_path, **margins))).grid

    return grid.x[:, -1, 0], grid.y[:, -1, 0]


def test_outer_boundary_stays_a_margin_beyond_the_shock(tmp_path: Path) -> None:
    """Every outer point is at least min_offset from the shock, on its far side."""

    # generate
    shock_x, shock_y = write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    outer_x, outer_y = outer_boundary(tmp_path, margin_factor=1.3, min_offset=5.0e-4)

    # nearest distance from each outer point to the (densely sampled) shock
    gap = np.hypot(outer_x[:, None] - shock_x[None, :], outer_y[:, None] - shock_y[None, :])
    nearest = gap.min(axis=1)

    # check: at least min_offset away, and outside (above the shock at the same x)
    assert np.all(nearest >= 0.99 * 5.0e-4)
    beside = outer_x > shock_x[0]
    assert np.all(outer_y[beside] > np.interp(outer_x[beside], shock_x, shock_y))


def test_outer_boundary_is_smooth_through_the_nose_junction(tmp_path: Path) -> None:
    """No corner and no spacing jump where the nose meets the cone (straight normals had both)."""

    # generate
    outer_x, outer_y = outer_boundary(tmp_path)

    # direction change between neighbouring outer segments, and their length ratio
    heading = np.degrees(np.unwrap(np.arctan2(np.diff(outer_y), np.diff(outer_x))))
    turning = np.abs(np.diff(heading))
    segment = np.hypot(np.diff(outer_x), np.diff(outer_y))
    jump = np.maximum(segment[1:] / segment[:-1], segment[:-1] / segment[1:])

    # check around the junction (wall point 40 of 41 nose points): smooth turning, even spacing
    junction = 40
    around = slice(junction - 4, junction + 3)
    assert turning[around].max() < 1.5 * np.median(turning[junction - 10 : junction - 4])
    assert jump.max() < 1.2


def test_shock_aligned_grid_keeps_the_first_spacing_and_has_no_folded_cells(tmp_path: Path) -> None:
    """The first wall-normal spacing is the configured one on every line; all cells are valid."""

    # generate
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    grid = generate_body_fitted(load_body_fitted_config(write_config(tmp_path))).grid
    x, y = grid.x[:, :, 0], grid.y[:, :, 0]

    # check: first spacing on every line
    first = np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])
    np.testing.assert_allclose(first, 1.0e-6, rtol=1.0e-9)

    # check: positive signed cell areas
    ax, ay = x[1:, :-1] - x[:-1, :-1], y[1:, :-1] - y[:-1, :-1]
    bx, by = x[:-1, 1:] - x[:-1, :-1], y[:-1, 1:] - y[:-1, :-1]
    assert np.all(ax * by - ay * bx > 0.0)


def test_stagnation_line_ends_ahead_of_the_shock_vertex(tmp_path: Path) -> None:
    """The line ahead of the nose (on the axis) crosses the shock at its vertex."""

    # generate
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    grid = generate_body_fitted(load_body_fitted_config(write_config(tmp_path))).grid

    # check: vertex at x = -0.4 mm, so the boundary sits at -(1.2 * 0.4 mm + 1 mm)
    assert grid.x[0, -1, 0] == pytest.approx(-(1.2 * 0.4e-3 + 1.0e-3), rel=1.0e-6)
    assert grid.y[0, -1, 0] == pytest.approx(0.0, abs=1.0e-12)


def test_noise_in_the_shock_curve_does_not_fold_the_grid(tmp_path: Path) -> None:
    """A sub-micron dent near the vertex, like grid-scale noise in a CFD shock, is harmless.

    Close to the vertex the shock is nearly vertical, so a 0.5 um dent makes x run
    backwards for a few points; pushed out along the raw normals by about 1.5 mm,
    the curve folded the grid there.
    """

    # the clean shock and its outer boundary
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

    # check: the outer boundary barely moves
    shift = np.hypot(noisy.x[:, -1, 0] - clean.x[:, -1, 0], noisy.y[:, -1, 0] - clean.y[:, -1, 0])
    assert shift.max() < 1.0e-6


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


# --------------------------------------------------
# configuration errors
# --------------------------------------------------
@pytest.mark.parametrize(
    ("outer", "height", "message"),
    [(True, 0.05, "not both"), (False, None, "needs an outer boundary")],
)
def test_exactly_one_outer_boundary(
    tmp_path: Path, outer: bool, height: float | None, message: str
) -> None:
    """[domain] height and [outer] exclude each other; one of them is required."""

    # check
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    with pytest.raises(ValueError, match=message):
        load_body_fitted_config(write_config(tmp_path, outer=outer, height=height))


def test_a_shock_that_ends_too_early_is_reported(tmp_path: Path) -> None:
    """A shock curve that stops before the outflow fails with a clear message."""

    # a shock that ends halfway along the body
    write_shock(tmp_path / "shock.csv", 0.5 * X_END)
    config = load_body_fitted_config(write_config(tmp_path))

    # check
    with pytest.raises(ValueError, match="doesn't reach the outer"):
        generate_body_fitted(config)


def test_lines_too_short_for_the_first_spacing_are_reported(tmp_path: Path) -> None:
    """A line shorter than (n_points - 1) x first_spacing fails, naming the line."""

    # no extra offset: the stagnation line is 1.2 x 0.4 mm = 0.48 mm long, but 61
    # points from a 10 um first spacing need more than 0.6 mm
    write_shock(tmp_path / "shock.csv", 2.0 * X_END)
    config_path = write_config(tmp_path, min_offset=0.0)
    config_text = config_path.read_text().replace(
        "first_spacing = 1.0e-6", "first_spacing = 1.0e-5"
    )
    config_path.write_text(config_text)
    config = load_body_fitted_config(config_path)

    # check
    with pytest.raises(ValueError, match=r"wall x = 0 m .* cells would shrink away from the wall"):
        generate_body_fitted(config)


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
    assert config.outer.shock == case / "shock.csv"
    assert generate_body_fitted(config).grid.shape[0] == 41 + 81 - 1
