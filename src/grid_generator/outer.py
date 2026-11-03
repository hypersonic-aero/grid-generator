"""Shock-aligned outer boundary: a smooth curve beyond the shock, and the grid's outer points on it.

The [outer] table names a shock curve (x, y columns in a CSV file, e.g. written by
`flow-theory shock-shape run` with a .csv output). The outer boundary is built in
three steps:

1. Outer curve: every shock point is pushed out along the shock's own outward
   normal by (margin_factor - 1) * d + min_offset, with d its distance from the
   wall. Offsetting a convex curve outward keeps it as smooth as the shock, so
   the outer boundary has no corners (e.g. where the nose meets the body).
2. Outer points: each wall normal's crossing with the outer curve, then the
   spacing between neighbouring crossings is smoothed (in log scale, over a few
   points), so it has no jumps either. The first (on the axis) and the last (the
   outflow line) stay where they are.
3. The grid lines (built in body_fitted) leave the wall along its normal and bend
   toward these points.

The shock is assumed symmetric about the axis (or symmetry plane) y = 0; the outer
curve is mirrored there, so the stagnation line ahead of the nose crosses it cleanly.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------
# constants
# --------------------------------------------------

# tolerance on the segment parameter when a ray meets a segment at its end point
_SEGMENT_TOLERANCE = 1.0e-9

# width (standard deviation, in points) of the smoothing of the outer spacing
_SPACING_SMOOTHING_WIDTH = 3.0


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
        ValueError: If it doesn't hold at least two (x, y) points.
    """

    # convert to Path object
    path = Path(path)

    # read: two columns after the header line
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] < 2 or values.shape[0] < 2:
        raise ValueError(f"{path}: expected x, y columns with at least two points")

    # order from the vertex outward
    order = np.argsort(values[:, 1])
    shock_x = values[order, 0]
    shock_y = values[order, 1]

    return shock_x, shock_y


def outer_points(
    x_wall: NDArray[np.float64],
    y_wall: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    shock_x: NDArray[np.float64],
    shock_y: NDArray[np.float64],
    margin_factor: float,
    min_offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Outer end of every grid line: on a smooth curve beyond the shock, smoothly spaced.

    Args:
        x_wall: Wall points x [m], from the stagnation point to the outflow.
        y_wall: Wall points y [m].
        nx: Unit wall normals, x component.
        ny: Unit wall normals, y component.
        shock_x: Shock x [m], from the vertex outward.
        shock_y: Shock y [m], from the vertex outward.
        margin_factor: Factor on the shock's distance from the wall (>= 1).
        min_offset: Distance added beyond the scaled shock distance [m].

    Returns:
        Outer points x and y [m], one per wall point.

    Raises:
        ValueError: If a wall normal doesn't reach the outer curve (e.g. the shock
            curve ends before the outflow).
    """

    # outer curve: the shock pushed out by the margin
    curve_x, curve_y = offset_shock(shock_x, shock_y, x_wall, y_wall, margin_factor, min_offset)

    # where each wall normal crosses it (arc length along the curve)
    arc = normal_crossings(x_wall, y_wall, nx, ny, curve_x, curve_y)

    # even out the spacing between neighbouring crossings, keeping both ends
    arc_smooth = smooth_spacing(arc)

    # the outer points at those arc lengths
    curve_arc = _cumulative_arc_length(curve_x, curve_y)
    point_x = np.interp(arc_smooth, curve_arc, curve_x)
    point_y = np.interp(arc_smooth, curve_arc, curve_y)

    return point_x, point_y


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

    # unit tangent of the shock, from the vertex outward
    tangent_x = np.gradient(shock_x)
    tangent_y = np.gradient(shock_y)
    length = np.hypot(tangent_x, tangent_y)
    tangent_x = tangent_x / length
    tangent_y = tangent_y / length

    # outward normal: the tangent turned to the side away from the body (upstream at the vertex)
    normal_x = -tangent_y
    normal_y = tangent_x

    # the vertex lies on the axis: its normal points straight upstream
    normal_x[0] = -1.0
    normal_y[0] = 0.0

    # distance of each shock point from the wall
    distance = _distance_to_polyline(shock_x, shock_y, x_wall, y_wall)

    # push out by the margin beyond the shock
    offset = (margin_factor - 1.0) * distance + min_offset
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
) -> NDArray[np.float64]:
    """Arc length along a curve where each wall normal crosses it.

    The curve starts on the axis (y = 0) and is mirrored there, so the stagnation
    line ahead of the nose crosses it at its first point.

    Args:
        x_wall: Wall points x [m].
        y_wall: Wall points y [m].
        nx: Unit wall normals, x component.
        ny: Unit wall normals, y component.
        curve_x: Curve x [m], from the axis outward.
        curve_y: Curve y [m], from the axis outward.

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
                f"the wall normal at x = {x_wall[i]:.6g} m doesn't reach the outer "
                "boundary; the shock curve must extend past the outflow (raise its x_e)"
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

    # smooth, mirroring the ends so they aren't pulled toward zero
    padded = np.pad(log_spacing, half_width, mode="reflect")
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
