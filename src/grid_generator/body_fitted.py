"""Constant-height body-fitted 2D grid generation.

Wall points (i) run from the stagnation point, around the nose, along the body to
x_end. Each j-line runs straight out along the exact wall normal to the same
height H:

    x(i, j) = wall(s_i) + n_j * normal(s_i)

The nose and the body get separate wall distributions that meet at the junction:
equal steps in arc length on the nose (equal steps in angle, so the nose block is
polar about the nose centre), then the configured body distribution starting from
the nose spacing, so the spacing is continuous across the junction. The body may be
straight (plate, wedge, cone) or curved (tangent ogive, with an optional cylinder).

H is either constant ([domain] height, picked large enough to clear the bow shock),
or the grid is a shock-line grid ([shock]): the bow shock is the grid line j = j_s
on every line, and the outer boundary is a smooth curve a margin beyond it (see
grid_generator.shock). Each line is then built in two segments that meet on the
shock. The first leaves the wall along the normal and bends toward the line's
shock point,

    x(i, j) = wall(s_i) + n_j * normal(s_i) + (n_j / h_i)^2 * shift_i,

with h_i the shock point's distance along the normal and shift_i its offset across
it; its points are [points.shock_layer], a two-sided stretching from the wall
spacing to a spacing at the shock. The second leaves the shock point in the
direction the first arrives in (so the line has no kink there) and bends the same
way toward the outer point; its points are [points.freestream], growing from the
spacing at the shock. Every line has the same point counts, so j_s is the same
everywhere.

With a [span] table (plate and wedge), the 2D plane is extruded in z: the same
(x, y) plane at every z of the span distribution, giving a 3D grid.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from canonical_geometries import Profile2D
from cfd_io import Dataset, StructuredGrid
from numpy.typing import NDArray

from grid_generator.config.body_fitted import (
    AXISYMMETRIC_TYPES,
    BodyFittedConfig,
    load_body_fitted_config,
)
from grid_generator.distributions import DistributionSpec, EquidistantSpec, build_distribution
from grid_generator.quality import folded_cells
from grid_generator.shock import extend_shock, offset_shock, points_on_curve, read_shock_curve

logger = logging.getLogger(__name__)

# --------------------------------------------------
# constants
# --------------------------------------------------

# grid side of each [markers] key (the names themselves come from the config): imin is
# the stagnation line ahead of the nose, imax the outflow at x_end, jmin the wall,
# jmax the outer boundary
MARKER_SIDES = {
    "symmetry": "imin",
    "outflow": "imax",
    "wall": "jmin",
    "outer": "jmax",
}

# the two extra sides of an extruded (3D) grid: the first and last z-planes; the
# solver decides what they are (periodic, symmetry, ...)
SPAN_MARKER_SIDES = {
    "span_min": "kmin",
    "span_max": "kmax",
}


# --------------------------------------------------
# public API
# --------------------------------------------------
def generate_body_fitted(source: BodyFittedConfig | str | Path) -> Dataset:
    """Generate a body-fitted grid from a config object or TOML file.

    The grid is 2D, or 3D when the config has a [span] to extrude it in z.

    Args:
        source: Validated configuration or configuration file path.

    Returns:
        Grid-only CFD dataset with shape (ni, nj, 1), or (ni, nj, nk) when extruded, in
        cfd-io's convention, with the boundary markers SU2 output needs. A shock-line
        grid also names its shock line: attrs["shock_line_j"] = j_s.

    Raises:
        ValueError: If the grid lines can't be built, or cross each other (folded or
            collapsed cells).
    """

    # load configuration when the caller supplied a file path
    if isinstance(source, BodyFittedConfig):
        config = source
    else:
        config = load_body_fitted_config(source)

    # wall profile, truncated at x_end
    profile = config.build_profile()

    # arc lengths of the wall points: nose, then body
    s_wall = wall_arc_lengths(profile, config.points.nose.n_points, config.body_distribution())

    # exact wall points and normals
    x_wall, y_wall = profile.point(s_wall)
    nx, ny = profile.normal(s_wall)

    # j-lines from the wall out to the outer boundary: (ni, nj)
    x_plane, y_plane = build_grid_lines(config, x_wall, y_wall, nx, ny)

    # check: no folded, collapsed or NaN cells, so a broken grid is never written
    folded = folded_cells(x_plane, y_plane)
    if np.any(folded):
        bad_i = np.nonzero(np.any(folded, axis=1))[0]
        first_i = int(bad_i[0])
        last_i = int(bad_i[-1]) + 1
        raise ValueError(
            f"{int(folded.sum())} folded or collapsed cells between i = {first_i} and "
            f"{last_i} (wall x = {x_wall[first_i]:.4g} to {x_wall[last_i]:.4g} m); "
            "the grid lines cross there: check the shock curve or the height"
        )

    # span: the z-planes to extrude to, or the single plane z = 0 for a 2D grid
    if config.span is not None:
        z_axis = build_distribution(config.span)
    else:
        z_axis = np.array([0.0])
    nk = z_axis.size

    # cfd-io's (ni, nj, nk) layout: the same (x, y) plane at every z
    x = np.repeat(x_plane[:, :, None], nk, axis=2)
    y = np.repeat(y_plane[:, :, None], nk, axis=2)
    z = np.broadcast_to(z_axis[None, None, :], x.shape).copy()
    grid = StructuredGrid(x=x, y=y, z=z)

    # markers: four sides in 2D, plus the two z-planes in 3D, named by [markers]
    markers = {}
    for key, side in MARKER_SIDES.items():
        markers[side] = getattr(config.markers, key)
    if nk > 1:
        for key, side in SPAN_MARKER_SIDES.items():
            markers[side] = getattr(config.markers, key)
    dimensionality = 3 if nk > 1 else 2

    # describe the grid; markers are what cfd-io's SU2 writer needs
    geometry_type = config.geometry.type
    attrs = {
        "dataset_type": "grid",
        "grid_type": "body-fitted",
        "coordinate_system": "cartesian",
        "dimensionality": dimensionality,
        "producer": "grid-generator",
        "geometry_type": geometry_type,
        "axisymmetric": geometry_type in AXISYMMETRIC_TYPES,
        "ni": grid.shape[0],
        "nj": grid.shape[1],
        "nk": grid.shape[2],
        "markers": {"structured": markers},
    }

    # a shock-line grid names the grid line on the shock (for solvers and post-processing)
    if config.shock is not None:
        attrs["shock_line_j"] = config.points.shock_layer.n_points - 1

    dataset = Dataset(grid=grid, attrs=attrs)
    return dataset


def build_grid_lines(
    config: BodyFittedConfig,
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The j-lines: from each wall point out to the outer boundary.

    Args:
        config: Validated configuration.
        x_wall: Wall points x [m].
        y_wall: Wall points y [m].
        nx: Unit wall normals, x component.
        ny: Unit wall normals, y component.

    Returns:
        Grid x and y [m], shape (ni, nj).

    Raises:
        ValueError: If a wall normal misses the shock or the outer boundary, or a line
            can't hold its points.
    """

    # constant height: straight lines along the wall normals, all with the same distribution
    if config.shock is None:
        n_line = build_distribution(config.normal_distribution())
        x_plane = x_wall[:, None] + n_line[None, :] * nx[:, None]
        y_plane = y_wall[:, None] + n_line[None, :] * ny[:, None]
        return x_plane, y_plane

    # shock line: two segments per line, meeting on the shock
    x_plane, y_plane = _shock_line_grid(config, x_wall, y_wall, nx, ny)

    return x_plane, y_plane


def wall_arc_lengths(
    profile: Profile2D, n_nose: int, body_spec: DistributionSpec
) -> NDArray[np.float64]:
    """Arc lengths of the wall points: equidistant on the nose, then the body distribution.

    The nose ends exactly at the profile's junction. The body distribution starts there,
    at the nose spacing (BodyFittedConfig.body_distribution builds it that way), so the
    spacing is continuous across the junction.

    Args:
        profile: Wall profile; its first junction is the end of the nose (later junctions,
            e.g. an ogive base, lie inside the body distribution).
        n_nose: Points on the nose, including both ends.
        body_spec: Body distribution in wall arc length, from the junction to the end.

    Returns:
        Arc lengths of the wall points: n_nose + body points - 1 (the junction appears once).

    Raises:
        ValueError: If the profile has no junction (no nose).
    """

    # validate inputs
    junctions = np.asarray(profile.junctions, dtype=float)
    if junctions.size == 0:
        raise ValueError("expected a profile with a nose junction, got none")
    s_junction = float(junctions[0])

    # nose: equal arc-length steps, i.e. equal angle steps on the circular arc
    nose_spec = EquidistantSpec(start=0.0, end=s_junction, n_points=n_nose)
    s_nose = build_distribution(nose_spec)

    # body: the configured distribution, starting at the junction
    s_body = build_distribution(body_spec)

    # join, keeping the shared junction point once
    s_wall = np.concatenate([s_nose, s_body[1:]])

    return s_wall


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _shock_line_grid(
    config: BodyFittedConfig,
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The j-lines of a shock-line grid: wall to shock point (j = j_s) to outer point.

    Args:
        config: Validated configuration with a [shock] table.
        x_wall: Wall points x [m].
        y_wall: Wall points y [m].
        nx: Unit wall normals, x component.
        ny: Unit wall normals, y component.

    Returns:
        Grid x and y [m], shape (ni, nj).

    Raises:
        ValueError: If a wall normal misses the shock or the outer boundary, or a line
            can't hold its points.
    """

    # the shock curve, continued straight past the outflow if it stops short of it
    shock = config.shock
    shock_x, shock_y = read_shock_curve(shock.curve)
    shock_x, shock_y = extend_shock(
        shock_x, shock_y, x_wall, y_wall, shock.margin_factor, shock.min_offset
    )

    # shock line: where each wall normal crosses the shock, evenly spaced along it
    shock_point_x, shock_point_y = points_on_curve(
        x_wall, y_wall, nx, ny, shock_x, shock_y, "shock curve"
    )

    # outer boundary: the shock pushed out by the margin, its points found the same way
    curve_x, curve_y = offset_shock(
        shock_x, shock_y, x_wall, y_wall, shock.margin_factor, shock.min_offset
    )
    outer_x, outer_y = points_on_curve(x_wall, y_wall, nx, ny, curve_x, curve_y, "outer boundary")

    # each line: two segments meeting on the shock
    configured_wall_spacing = config.points.shock_layer.first_spacing
    lowered = []
    lines_x = []
    lines_y = []
    for i in range(x_wall.size):
        # segment 1: its length along the wall normal, up to the shock point
        to_shock_x = shock_point_x[i] - x_wall[i]
        to_shock_y = shock_point_y[i] - y_wall[i]
        shock_height = to_shock_x * nx[i] + to_shock_y * ny[i]
        if shock_height <= 0.0:
            raise ValueError(
                f"the shock curve lies on or inside the body at wall x = {x_wall[i]:.4g} m"
            )

        # segment 1: wall to shock, two-sided stretching (wall spacing lowered if needed)
        try:
            inner_spec = config.shock_layer_distribution(float(shock_height))
            inner_distance = build_distribution(inner_spec)
        except ValueError as error:
            raise ValueError(
                f"on the grid line at wall x = {x_wall[i]:.4g} m (wall to shock "
                f"{shock_height:.4g} m): {error}"
            ) from None
        if inner_spec.first_spacing < configured_wall_spacing:
            lowered.append(i)

        # segment 1: leaves the wall along its normal, ends on the shock point
        inner_x, inner_y, arrival_x, arrival_y = _bent_segment(
            x_wall[i],
            y_wall[i],
            nx[i],
            ny[i],
            shock_point_x[i],
            shock_point_y[i],
            inner_distance,
        )

        # segment 2: its length in the direction segment 1 arrives in, up to the outer point
        to_outer_x = outer_x[i] - shock_point_x[i]
        to_outer_y = outer_y[i] - shock_point_y[i]
        outer_length = to_outer_x * arrival_x + to_outer_y * arrival_y
        if outer_length <= 0.0:
            raise ValueError(
                f"the outer boundary doesn't lie beyond the shock at wall x = {x_wall[i]:.4g} m"
            )

        # segment 2: shock to outer boundary, growing from segment 1's last cell
        shock_spacing = np.hypot(inner_x[-1] - inner_x[-2], inner_y[-1] - inner_y[-2])
        try:
            outer_spec = config.freestream_distribution(float(outer_length), float(shock_spacing))
            outer_distance = build_distribution(outer_spec)
        except ValueError as error:
            raise ValueError(
                f"on the grid line at wall x = {x_wall[i]:.4g} m (shock to outer boundary "
                f"{outer_length:.4g} m): {error}"
            ) from None

        # segment 2: leaves the shock point the way segment 1 arrives, ends on the outer point
        outer_line_x, outer_line_y, _, _ = _bent_segment(
            shock_point_x[i],
            shock_point_y[i],
            arrival_x,
            arrival_y,
            outer_x[i],
            outer_y[i],
            outer_distance,
        )

        # join, keeping the shock point once
        lines_x.append(np.concatenate([inner_x, outer_line_x[1:]]))
        lines_y.append(np.concatenate([inner_y, outer_line_y[1:]]))

    # report the lines whose wall spacing was lowered (visible with --verbose)
    if lowered:
        logger.info(
            "[points.shock_layer] first_spacing lowered on %d lines too short for it "
            "(wall x = %.4g to %.4g m)",
            len(lowered),
            x_wall[lowered[0]],
            x_wall[lowered[-1]],
        )

    x_plane = np.vstack(lines_x)
    y_plane = np.vstack(lines_y)

    return x_plane, y_plane


def _bent_segment(
    start_x: float,
    start_y: float,
    direction_x: float,
    direction_y: float,
    end_x: float,
    end_y: float,
    distance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Points on a segment that leaves start along a direction and bends to end.

        x(d) = start + d * direction + (d / length)^2 * shift,

    with length the reach to end along the direction and shift the reach across it,
    so the segment leaves start exactly along the direction and ends exactly at end.

    Args:
        start_x: Start point x [m].
        start_y: Start point y [m].
        direction_x: Unit direction at the start, x component.
        direction_y: Unit direction at the start, y component.
        end_x: End point x [m].
        end_y: End point y [m].
        distance: Distances along the direction, from 0 to the length [m].

    Returns:
        Points x and y [m], and the unit direction (x, y components) the segment
        arrives at end in.
    """

    # split the reach into a part along the direction and a shift across it
    reach_x = end_x - start_x
    reach_y = end_y - start_y
    length = reach_x * direction_x + reach_y * direction_y
    shift_x = reach_x - length * direction_x
    shift_y = reach_y - length * direction_y

    # the shift grows quadratically, so the segment leaves start along the direction
    fraction = distance / length
    x = start_x + distance * direction_x + fraction**2 * shift_x
    y = start_y + distance * direction_y + fraction**2 * shift_y

    # the last point exactly on end (no round-off)
    x[-1] = end_x
    y[-1] = end_y

    # arrival direction: the derivative at d = length, direction + 2 shift / length
    arrival_x = direction_x + 2.0 * shift_x / length
    arrival_y = direction_y + 2.0 * shift_y / length
    arrival_norm = float(np.hypot(arrival_x, arrival_y))
    arrival_x = float(arrival_x / arrival_norm)
    arrival_y = float(arrival_y / arrival_norm)

    return x, y, arrival_x, arrival_y
