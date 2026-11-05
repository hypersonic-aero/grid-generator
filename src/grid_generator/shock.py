"""Shock-line grids: the shock curve, the grid's points on it, and the outer boundary beyond it.

The [shock] table names a shock curve (x, y columns in a CSV file, e.g. written by
`flow-theory shock-shape run` with a .csv output, or a shock taken from a CFD
solution). A curve that stops at the outflow or a little before it (a shock taken
from a CFD solution ends at its domain end) is first continued straight past it.
Two curves then carry one point per grid line:

1. Shock line (j = j_s): each wall normal's crossing with the shock curve.
2. Outer curve: every shock point is pushed out along the shock's own outward
   normal by (margin_factor - 1) * d + min_offset, with d its distance from the
   wall. Offsetting a convex curve outward keeps it as smooth as the shock, so
   the outer boundary has no corners (e.g. where the nose meets the body).
   The normal is averaged along the shock over that same distance, so wiggles
   smaller than the margin (e.g. noise in a shock taken from a CFD solution)
   can't fold the outer curve. Its points are each wall normal's crossing too.

On both curves the spacing between neighbouring crossings is then smoothed (in log
scale, over a few points), so it has no jumps. The first point (on the axis) and the
last (the outflow line) stay where they are. The grid lines (built in body_fitted)
leave the wall along its normal, bend toward their shock point, and continue from
there to their outer point.

The shock is assumed symmetric about the axis (or symmetry plane) y = 0; both curves
are mirrored there, so the stagnation line ahead of the nose crosses them cleanly.
That needs the curve to start on the axis. A curve that starts a little off it (a shock
taken from cell-centre data starts half a cell away) is first continued to the axis,
crossing it at right angles as a symmetric shock does (start_on_axis); otherwise the
symmetry line i = 0 would bend off y = 0 toward the curve's first point.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# --------------------------------------------------
# constants
# --------------------------------------------------

# tolerance on the segment parameter when a ray meets a segment at its end point
_SEGMENT_TOLERANCE = 1.0e-9

# width (standard deviation, in points) of the smoothing of the outer spacing
_SPACING_SMOOTHING_WIDTH = 3.0

# a point within this fraction of the nose radius of y = 0 lies on the axis
AXIS_TOLERANCE = 1.0e-12

# a curve starting farther from the axis than this fraction of the nose radius is not
# just half a cell off it; continuing it there would be a guess, so it is refused
_AXIS_GAP_LIMIT = 0.1

# points the axis continuation is fitted through (x = a + b y^2)
_AXIS_FIT_POINTS = 5


# --------------------------------------------------
# public API
# --------------------------------------------------
def read_shock_curve(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a shock curve from a CSV file with x and y columns (one header line).

    Args:
        path: CSV file, e.g. the .csv output of `flow-theory shock-shape run`.

    Returns:
        Shock x and y [m], ordered from the vertex (smallest y) outward.

    Raises:
        OSError: If the file can't be read.
        ValueError: If it doesn't hold at least two distinct (x, y) points, or holds a
            value that isn't finite.
    """

    # convert to Path object
    path = Path(path)

    # read: two columns after the header line
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] < 2 or values.shape[0] < 2:
        raise ValueError(f"{path}: expected x, y columns with at least two points")

    # check: every x and y is a number
    if not np.all(np.isfinite(values[:, :2])):
        raise ValueError(f"{path}: the shock curve holds values that aren't finite (nan, inf)")

    # order from the vertex outward
    order = np.argsort(values[:, 1], kind="stable")
    shock_x = values[order, 0]
    shock_y = values[order, 1]

    # drop repeated points (a curve taken from a CFD solution can repeat one)
    repeated = (np.diff(shock_x) == 0.0) & (np.diff(shock_y) == 0.0)
    keep = np.concatenate([[True], ~repeated])
    shock_x = shock_x[keep]
    shock_y = shock_y[keep]
    if shock_x.size < 2:
        raise ValueError(f"{path}: expected at least two distinct shock points")

    return shock_x, shock_y


def start_on_axis(
    shock_x: NDArray[np.float64],
    shock_y: NDArray[np.float64],
    nose_radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make the shock curve start on the axis y = 0, where the symmetry line i = 0 meets it.

    A symmetric shock crosses the axis at right angles (dx/dy = 0 there), so near it
    the curve is x = a + b y^2. A curve whose first point lies off the axis is fitted
    that way through its first few points, and the point (a, 0) is put in front of it;
    a warning names the first point's distance from the axis.

    Args:
        shock_x: Shock x [m], from its smallest y outward.
        shock_y: Shock y [m].
        nose_radius: Nose radius of the body [m], the scale of "close to the axis".

    Returns:
        Shock x and y [m], starting on the axis (unchanged if it already did).

    Raises:
        ValueError: If the curve starts farther from the axis than a tenth of the nose
            radius (more than the half cell of cell-centre data), or below it.
    """

    # on the axis already: nothing to do
    first_y = float(shock_y[0])
    if abs(first_y) <= AXIS_TOLERANCE * nose_radius:
        return shock_x, shock_y

    # check: a small gap above the axis, not a curve that misses the nose
    if first_y < 0.0 or first_y > _AXIS_GAP_LIMIT * nose_radius:
        raise ValueError(
            f"the shock curve starts at y = {first_y:.4g} m; it must start on the axis "
            f"y = 0, or within {_AXIS_GAP_LIMIT:g} nose radii above it (e.g. half a cell "
            "off, from cell-centre data), to be continued there"
        )

    # fit x = a + b y^2 through the first points: a is where the curve meets the axis
    n_fit = min(_AXIS_FIT_POINTS, shock_x.size)
    y_fit = shock_y[:n_fit]
    x_fit = shock_x[:n_fit]
    matrix = np.column_stack([np.ones(n_fit), y_fit**2])
    coefficients, *_ = np.linalg.lstsq(matrix, x_fit, rcond=None)
    axis_x = float(coefficients[0])

    # the curve, continued to the axis
    axis_shock_x = np.concatenate([[axis_x], shock_x])
    axis_shock_y = np.concatenate([[0.0], shock_y])

    # tell the user: the grid differs from the curve's points there
    logger.warning(
        "the shock curve starts at y = %.4g m, off the axis (a curve from cell-centre "
        "data starts half a cell away); continued it to the axis at x = %.6g m",
        first_y,
        axis_x,
    )

    return axis_shock_x, axis_shock_y


def points_on_curve(
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    curve_x: NDArray[np.float64],
    curve_y: NDArray[np.float64],
    curve_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One point per grid line on a curve (the shock or the outer curve), smoothly spaced.

    Args:
        x_wall: Wall points x [m], from the stagnation point to the outflow.
        y_wall: Wall points y [m].
        nx: Unit wall normals, x component.
        ny: Unit wall normals, y component.
        curve_x: Curve x [m], from the axis outward.
        curve_y: Curve y [m], from the axis outward.
        curve_name: Name of the curve for error messages, e.g. "shock curve".

    Returns:
        Points x and y [m] on the curve, one per wall point.

    Raises:
        ValueError: If a wall normal doesn't reach the curve.
    """

    # where each wall normal crosses it (arc length along the curve)
    arc = normal_crossings(x_wall, y_wall, nx, ny, curve_x, curve_y, curve_name)

    # even out the spacing between neighbouring crossings, keeping both ends
    arc_smooth = smooth_spacing(arc)

    # the points at those arc lengths
    curve_arc = _cumulative_arc_length(curve_x, curve_y)
    point_x = np.interp(arc_smooth, curve_arc, curve_x)
    point_y = np.interp(arc_smooth, curve_arc, curve_y)

    return point_x, point_y


def extend_shock(
    shock_x: NDArray[np.float64],
    shock_y: NDArray[np.float64],
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    margin_factor: float,
    min_offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Continue a shock curve straight past the outflow when it stops short of it.

    A shock taken from a CFD solution stops at that solution's domain end, about
    where the new grid ends too. Its last stretch is continued along the chord over
    one shock-layer height (so noise at the end doesn't tilt it), until the curve
    passes the outflow by twice the outer boundary's offset there.

    Args:
        shock_x: Shock x [m], from the vertex outward.
        shock_y: Shock y [m], from the vertex outward.
        x_wall: Wall points x [m], from the stagnation point to the outflow.
        y_wall: Wall points y [m].
        margin_factor: Factor on the shock's distance from the wall (>= 1).
        min_offset: Distance added beyond the scaled shock distance [m].

    Returns:
        Shock x and y [m], continued past the outflow where needed.

    Raises:
        ValueError: If the curve stops more than one shock-layer height (its
            distance from the wall) before the outflow, or its end doesn't run
            downstream.
    """

    # distance of the curve's end from the wall, and the outer boundary's offset there
    end_distance = float(_distance_to_polyline(shock_x[-1:], shock_y[-1:], x_wall, y_wall)[0])
    end_offset = (margin_factor - 1.0) * end_distance + min_offset

    # nothing to do when the curve already passes the outflow far enough
    x_outflow = float(x_wall[-1])
    x_reach = x_outflow + 2.0 * end_offset
    if shock_x[-1] >= x_reach:
        return shock_x, shock_y

    # check: a straight continuation is only trusted over about one shock-layer height
    if shock_x[-1] < x_outflow - end_distance:
        raise ValueError(
            f"the shock curve ends at x = {shock_x[-1]:.4g} m, too far before the outflow "
            f"at x = {x_outflow:.4g} m to continue it straight (more than its distance from "
            f"the wall, {end_distance:.3g} m); give a longer curve (flow-theory: raise x_e)"
        )

    # direction of the last stretch: the chord over one shock-layer height
    arc = _cumulative_arc_length(shock_x, shock_y)
    start = int(np.searchsorted(arc, arc[-1] - end_distance))
    start = min(start, shock_x.size - 2)
    chord_x = shock_x[-1] - shock_x[start]
    chord_y = shock_y[-1] - shock_y[start]
    chord_length = float(np.hypot(chord_x, chord_y))
    direction_x = chord_x / chord_length
    direction_y = chord_y / chord_length

    # check: the end must run downstream to ever reach the outflow
    if direction_x <= 0.0:
        raise ValueError(
            f"the shock curve's end (x = {shock_x[-1]:.4g} m) doesn't run downstream, "
            "so it can't be continued past the outflow"
        )

    # straight continuation in ten segments, until x passes the reach
    length = (x_reach - shock_x[-1]) / direction_x
    steps = np.linspace(0.0, length, 11)[1:]
    extended_x = np.concatenate([shock_x, shock_x[-1] + steps * direction_x])
    extended_y = np.concatenate([shock_y, shock_y[-1] + steps * direction_y])

    # report the continuation (visible with --verbose)
    logger.info(
        "shock curve continued straight by %.3g m, from x = %.4g m to x = %.4g m",
        length,
        shock_x[-1],
        extended_x[-1],
    )

    return extended_x, extended_y


def offset_shock(
    shock_x: NDArray[np.float64],
    shock_y: NDArray[np.float64],
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    margin_factor: float,
    min_offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Push every shock point out along the shock's outward normal by the margin.

    Args:
        shock_x: Shock x [m], from the vertex outward.
        shock_y: Shock y [m], from the vertex outward.
        x_wall: Wall points x [m] (a polyline along the body).
        y_wall: Wall points y [m].
        margin_factor: Factor on the shock's distance from the wall (>= 1).
        min_offset: Distance added beyond the scaled shock distance [m].

    Returns:
        Outer curve x and y [m], from the axis outward.
    """

    # distance of each shock point from the wall
    distance = _distance_to_polyline(shock_x, shock_y, x_wall, y_wall)

    # how far each point is pushed out: the margin beyond the shock
    offset = (margin_factor - 1.0) * distance + min_offset

    # outward normals, averaged along the shock over the offset distance: a wiggle
    # smaller than the margin (e.g. grid-scale noise in a shock taken from a CFD
    # solution) would otherwise tilt the normal and fold the pushed-out curve
    normal_x, normal_y = _smoothed_normals(shock_x, shock_y, offset)

    # push out by the margin beyond the shock
    curve_x = shock_x + offset * normal_x
    curve_y = shock_y + offset * normal_y

    return curve_x, curve_y


def normal_crossings(
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    curve_x: NDArray[np.float64],
    curve_y: NDArray[np.float64],
    curve_name: str,
) -> NDArray[np.float64]:
    """Arc length along a curve where each wall normal crosses it.

    The curve starts on the axis (y = 0; see start_on_axis) and is mirrored there, so
    the stagnation line ahead of the nose crosses it at its first point.

    Args:
        x_wall: Wall points x [m].
        y_wall: Wall points y [m].
        nx: Unit wall normals, x component.
        ny: Unit wall normals, y component.
        curve_x: Curve x [m], from the axis outward.
        curve_y: Curve y [m], from the axis outward.
        curve_name: Name of the curve for error messages, e.g. "shock curve".

    Returns:
        Arc length [m] of each crossing, measured along the curve from the axis.

    Raises:
        ValueError: If a wall normal doesn't reach the curve.
    """

    # mirror the curve about y = 0; arc lengths count from its point on the axis
    mirrored_x = np.concatenate([curve_x[:0:-1], curve_x])
    mirrored_y = np.concatenate([-curve_y[:0:-1], curve_y])
    mirrored_arc = _cumulative_arc_length(mirrored_x, mirrored_y)
    mirrored_arc = mirrored_arc - mirrored_arc[curve_x.size - 1]

    # segments of the mirrored curve: start points, direction vectors, arc lengths
    start_x = mirrored_x[:-1]
    start_y = mirrored_y[:-1]
    edge_x = np.diff(mirrored_x)
    edge_y = np.diff(mirrored_y)
    edge_length = np.diff(mirrored_arc)

    # intersect each wall normal with every segment: p + t n = a + u e
    arc = np.empty(x_wall.size)
    for i in range(x_wall.size):
        # 2D cross products of the 2x2 system
        denominator = nx[i] * edge_y - ny[i] * edge_x
        offset_x = start_x - x_wall[i]
        offset_y = start_y - y_wall[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset_x * edge_y - offset_y * edge_x) / denominator
            u = (offset_x * ny[i] - offset_y * nx[i]) / denominator

        # crossings ahead of the wall, inside a segment
        inside = (u >= -_SEGMENT_TOLERANCE) & (u <= 1.0 + _SEGMENT_TOLERANCE)
        ahead = np.isfinite(t) & (t > 0.0) & inside
        if not np.any(ahead):
            raise ValueError(
                f"the wall normal at x = {x_wall[i]:.6g} m doesn't reach the {curve_name}; "
                "the shock curve must extend past the outflow (raise its x_e)"
            )

        # nearest crossing, as arc length along the curve (not below the axis)
        candidates = np.nonzero(ahead)[0]
        nearest = candidates[np.argmin(t[candidates])]
        crossing = mirrored_arc[nearest] + np.clip(u[nearest], 0.0, 1.0) * edge_length[nearest]
        arc[i] = max(float(crossing), 0.0)

    return arc


def smooth_spacing(arc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth the spacing of increasing points, keeping the first and last point.

    The spacing is smoothed in log scale (a Gaussian over a few points), so a jump
    in spacing becomes a gradual change, then rescaled to the original total.

    Args:
        arc: Increasing positions (arc lengths) [m].

    Returns:
        Positions with the same ends and a smoothly varying spacing.
    """

    # spacing between neighbours, in log scale
    spacing = np.diff(arc)
    log_spacing = np.log(np.maximum(spacing, np.finfo(float).tiny))

    # Gaussian kernel over a few points
    half_width = int(np.ceil(3.0 * _SPACING_SMOOTHING_WIDTH))
    offsets = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (offsets / _SPACING_SMOOTHING_WIDTH) ** 2)
    kernel = kernel / kernel.sum()

    # smooth, mirroring the ends so they aren't pulled toward zero; the odd reflection
    # continues a trend past the ends (geometric growth is a straight line in log
    # spacing), so growing cells aren't cut short there, which the rescale below would
    # otherwise make up by shifting every point along the curve
    padded = np.pad(log_spacing, half_width, mode="reflect", reflect_type="odd")
    smoothed = np.exp(np.convolve(padded, kernel, mode="valid"))

    # rescale to the original total, so both ends stay fixed
    smoothed = smoothed * (arc[-1] - arc[0]) / smoothed.sum()
    arc_smooth = arc[0] + np.concatenate([[0.0], np.cumsum(smoothed)])

    return arc_smooth


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _cumulative_arc_length(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Arc length along a polyline, from its first point."""

    # segment lengths, summed
    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    return arc


def _smoothed_normals(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    width: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit outward normals of a curve, averaged along it with Gaussian weights.

    The curve runs from the axis outward and is mirrored about y = 0, so the
    average at the vertex is symmetric and points straight upstream. Each segment
    counts with its length, so clustered points don't weigh more.

    Args:
        x: Curve x [m], from the axis outward.
        y: Curve y [m].
        width: Width (standard deviation) of the average at each point [m]; never
            narrower than the point spacing there.

    Returns:
        Normals x and y components, one per point.
    """

    # segments: length and arc length of their midpoints (repeated points carry no weight)
    edge_x = np.diff(x)
    edge_y = np.diff(y)
    edge_length = np.hypot(edge_x, edge_y)
    keep = edge_length > 0.0
    arc = _cumulative_arc_length(x, y)
    edge_arc = 0.5 * (arc[:-1] + arc[1:])

    # outward normal of each segment: its direction turned away from the body
    edge_normal_x = -edge_y[keep] / edge_length[keep]
    edge_normal_y = edge_x[keep] / edge_length[keep]

    # mirror about y = 0: the mirrored segments sit at negative arc length, their
    # normals flipped in y
    all_arc = np.concatenate([-edge_arc[keep], edge_arc[keep]])
    all_length = np.concatenate([edge_length[keep], edge_length[keep]])
    all_normal_x = np.concatenate([edge_normal_x, edge_normal_x])
    all_normal_y = np.concatenate([-edge_normal_y, edge_normal_y])

    # floor on the width: the mean length of the segments next to each point
    spacing = np.empty(x.size)
    spacing[0] = edge_length[0]
    spacing[-1] = edge_length[-1]
    spacing[1:-1] = 0.5 * (edge_length[:-1] + edge_length[1:])
    width = np.maximum(width, spacing)

    # average at each point: Gaussian weight in arc length, times segment length
    normal_x = np.empty(x.size)
    normal_y = np.empty(x.size)
    for k in range(x.size):
        weight = all_length * np.exp(-0.5 * ((all_arc - arc[k]) / width[k]) ** 2)
        sum_x = np.sum(weight * all_normal_x)
        sum_y = np.sum(weight * all_normal_y)
        norm = np.hypot(sum_x, sum_y)
        normal_x[k] = sum_x / norm
        normal_y[k] = sum_y / norm

    return normal_x, normal_y


def _distance_to_polyline(
    point_x: NDArray[np.float64],
    point_y: NDArray[np.float64],
    line_x: NDArray[np.float64],
    line_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Shortest distance from each point to a polyline."""

    # segments of the polyline
    start_x = line_x[:-1]
    start_y = line_y[:-1]
    edge_x = np.diff(line_x)
    edge_y = np.diff(line_y)
    edge_squared = edge_x**2 + edge_y**2

    # nearest point on every segment, for every point
    distances = np.empty(point_x.size)
    for k in range(point_x.size):
        along = ((point_x[k] - start_x) * edge_x + (point_y[k] - start_y) * edge_y) / edge_squared
        along = np.clip(along, 0.0, 1.0)
        gap_x = start_x + along * edge_x - point_x[k]
        gap_y = start_y + along * edge_y - point_y[k]
        distances[k] = float(np.min(np.hypot(gap_x, gap_y)))

    return distances
