"""Quality measures for 2D structured grids (nk = 1)."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cfd_io import StructuredGrid
from numpy.typing import NDArray


# --------------------------------------------------
# report
# --------------------------------------------------
@dataclass(frozen=True, slots=True)
class GridQuality:
    """Summary of a 2D structured grid's quality.

    Attributes:
        min_cell_area: Smallest signed cell area [m^2]; > 0 means every cell is
            valid and counter-clockwise in (i, j).
        max_spacing_ratio_i: Largest ratio of neighbouring cell sizes along i.
        max_spacing_ratio_j: Largest ratio of neighbouring cell sizes along j.
        max_wall_skew_deg: Largest deviation from 90 deg between the wall (j = 0)
            and the first j-line, in degrees.
    """

    min_cell_area: float
    max_spacing_ratio_i: float
    max_spacing_ratio_j: float
    max_wall_skew_deg: float

    def summary(self) -> str:
        """Return the report as aligned text lines, for the CLI."""

        lines = [
            f"  smallest cell area      {self.min_cell_area:.3e} m^2",
            f"  max spacing ratio, i    {self.max_spacing_ratio_i:.4f}",
            f"  max spacing ratio, j    {self.max_spacing_ratio_j:.4f}",
            f"  max wall skew           {self.max_wall_skew_deg:.3e} deg",
        ]
        text = "\n".join(lines)
        return text


# --------------------------------------------------
# public API
# --------------------------------------------------
def grid_quality(grid: StructuredGrid) -> GridQuality:
    """Measure cell areas, neighbour spacing ratios and wall skew of a 2D grid.

    Args:
        grid: Structured grid with shape (ni, nj, 1).

    Returns:
        Quality report.

    Raises:
        ValueError: If the grid is not 2D with nk = 1.
    """

    # validate inputs
    ni, nj, nk = grid.shape
    if nk != 1 or ni < 3 or nj < 3:
        raise ValueError(
            f"grid_quality needs a 2D grid with ni, nj >= 3 and nk = 1, got {grid.shape}"
        )

    # work on the (ni, nj) plane
    x = grid.x[:, :, 0]
    y = grid.y[:, :, 0]

    # signed cell areas (shoelace over the corners (i,j), (i+1,j), (i+1,j+1), (i,j+1))
    areas = _cell_areas(x, y)

    # neighbouring cell sizes along each index direction
    lengths_i = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    lengths_j = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    ratio_i = _max_neighbour_ratio(lengths_i, axis=0)
    ratio_j = _max_neighbour_ratio(lengths_j, axis=1)

    # wall skew: angle between the wall direction (along i at j = 0) and the first j-line;
    # second-order differences at the ends too, so a curved wall doesn't show a
    # spurious skew of half the angle step at i = 0
    wall_dx = np.gradient(x[:, 0], edge_order=2)
    wall_dy = np.gradient(y[:, 0], edge_order=2)
    line_dx = x[:, 1] - x[:, 0]
    line_dy = y[:, 1] - y[:, 0]
    cos_angle = (wall_dx * line_dx + wall_dy * line_dy) / (
        np.hypot(wall_dx, wall_dy) * np.hypot(line_dx, line_dy)
    )
    skew_deg = np.abs(90.0 - np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    report = GridQuality(
        min_cell_area=float(areas.min()),
        max_spacing_ratio_i=ratio_i,
        max_spacing_ratio_j=ratio_j,
        max_wall_skew_deg=float(skew_deg.max()),
    )
    return report


def folded_cells(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark the cells of a 2D plane that are folded, collapsed or not finite.

    A valid cell turns counter-clockwise in (i, j) at all four corners: the cross
    product of the two edges meeting at each corner is positive. A positive area
    alone isn't enough, since a twisted (bow-tie) cell can still have one.

    Args:
        x: Grid x [m], shape (ni, nj).
        y: Grid y [m], shape (ni, nj).

    Returns:
        True for every bad cell, shape (ni - 1, nj - 1).
    """

    # corners of each cell, counter-clockwise: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    corners_x = [x[:-1, :-1], x[1:, :-1], x[1:, 1:], x[:-1, 1:]]
    corners_y = [y[:-1, :-1], y[1:, :-1], y[1:, 1:], y[:-1, 1:]]

    # at each corner: cross product of the edge to the next corner and the edge
    # to the previous one; NaN compares as not positive, so it's caught too
    valid = np.ones(x[:-1, :-1].shape, dtype=bool)
    for corner in range(4):
        following = (corner + 1) % 4
        preceding = (corner - 1) % 4
        next_x = corners_x[following] - corners_x[corner]
        next_y = corners_y[following] - corners_y[corner]
        prev_x = corners_x[preceding] - corners_x[corner]
        prev_y = corners_y[preceding] - corners_y[corner]
        cross = next_x * prev_y - next_y * prev_x
        valid = valid & (cross > 0.0)

    folded = ~valid

    return folded


# --------------------------------------------------
# private helpers
# --------------------------------------------------
def _cell_areas(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed area of every cell, positive when its corners run counter-clockwise."""

    # corners of each cell in (i, j) order around the cell
    x0, y0 = x[:-1, :-1], y[:-1, :-1]
    x1, y1 = x[1:, :-1], y[1:, :-1]
    x2, y2 = x[1:, 1:], y[1:, 1:]
    x3, y3 = x[:-1, 1:], y[:-1, 1:]

    # shoelace formula for a quadrilateral
    twice_area = (
        (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3)
    )
    areas = 0.5 * twice_area

    return areas


def _max_neighbour_ratio(lengths: NDArray[np.float64], axis: int) -> float:
    """Largest ratio between neighbouring cell sizes along one axis (always >= 1)."""

    # neighbouring sizes along the axis
    if axis == 0:
        first = lengths[:-1, :]
        second = lengths[1:, :]
    else:
        first = lengths[:, :-1]
        second = lengths[:, 1:]

    # larger over smaller, so growth and shrinkage count the same
    ratio = np.maximum(first / second, second / first)

    return float(ratio.max())
