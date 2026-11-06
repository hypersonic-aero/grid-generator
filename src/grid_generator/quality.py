"""Quality measures for structured grids: the 2D plane (nk = 1), and the swept 3D grid."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cfd_io import StructuredGrid
from numpy.typing import NDArray

from grid_generator.sweep import SweepTransform


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


@dataclass(frozen=True, slots=True)
class ShockLineQuality:
    """Summary of the shock line (j = j_s) of a shock-aligned grid.

    Attributes:
        shock_line_j: Index j_s of the grid line on the shock.
        shock_distance: Smallest and largest wall-to-shock distance over the lines [m].
        spacing_below: Smallest and largest cell size just inside the shock
            (j_s - 1 to j_s) [m].
        spacing_above: Smallest and largest cell size just outside the shock
            (j_s to j_s + 1) [m].
        lowered_wall_spacing: Lines whose wall spacing is below the configured
            first spacing (lines too short for it, near a small nose).
        axis_offset: Largest distance of the symmetry line (i = 0) from the axis
            y = 0 [m]; 0 on a sound grid.
        min_wall_spacing: Smallest wall spacing (j = 0 to 1) over the lines [m].
        min_wall_spacing_x: Wall x of the line with the smallest wall spacing [m].
        lowered_shock_spacing: Lines whose spacing at the shock is below the
            configured fraction of their wall-to-shock distance (lines too short for
            both end spacings).
        min_shock_spacing_fraction: Smallest spacing at the shock (j_s - 1 to j_s,
            along the wall normal) as a fraction of the wall-to-shock distance along it.
    """

    shock_line_j: int
    shock_distance: tuple[float, float]
    spacing_below: tuple[float, float]
    spacing_above: tuple[float, float]
    lowered_wall_spacing: int
    axis_offset: float
    min_wall_spacing: float
    min_wall_spacing_x: float
    lowered_shock_spacing: int
    min_shock_spacing_fraction: float

    def summary(self) -> str:
        """Return the report as aligned text lines, for the CLI."""

        # the spacings at both ends of the shock layer, and how many lines gave way (where
        # none did, the smallest wall spacing is the configured one up to round-off)
        wall_text = f"{self.min_wall_spacing:.3e} m on every line"
        if self.lowered_wall_spacing > 0:
            wall_text = (
                f"{self.min_wall_spacing:.3e} m smallest (wall x = {self.min_wall_spacing_x:.3e} m)"
            )
        shock_percent = 100.0 * self.min_shock_spacing_fraction
        shock_text = (
            f"{self.lowered_shock_spacing} lines (smallest {shock_percent:.3g} % of the "
            "wall-to-shock distance)"
        )

        lines = [
            f"  shock line              j = {self.shock_line_j}",
            f"  wall-to-shock distance  {_range_text(self.shock_distance)} m",
            f"  spacing below the shock {_range_text(self.spacing_below)} m",
            f"  spacing above the shock {_range_text(self.spacing_above)} m",
            f"  wall spacing            {wall_text}",
            f"  wall spacing lowered    {self.lowered_wall_spacing} lines",
            f"  shock spacing lowered   {shock_text}",
            f"  symmetry line off axis  {self.axis_offset:.3e} m (max |y| on i = 0)",
        ]
        text = "\n".join(lines)
        return text


@dataclass(frozen=True, slots=True)
class SweepQuality:
    """Summary of a swept (3D) grid: its cells, the axis, and the side faces.

    Attributes:
        transform: How the high side face follows from the low one.
        low_name: Marker of the side face k = 0.
        high_name: Marker of the side face k = nk - 1.
        min_cell_volume: Smallest cell volume [m^3]; > 0 means every cell is
            right-handed in (i, j, k).
        axis_cells: Cells touching the axis (revolve): prisms in .su2 and .msh.
        min_axis_cell_volume: Smallest volume of those cells [m^3] (NaN without them).
        side_face_mismatch: Largest distance between the high side face and the low
            one transformed onto it [m]; round-off on a sound grid.
        scale: Factor the coordinates are written with ([output] scale).
    """

    transform: SweepTransform
    low_name: str
    high_name: str
    min_cell_volume: float
    axis_cells: int
    min_axis_cell_volume: float
    side_face_mismatch: float
    scale: float

    def summary(self) -> str:
        """Return the report as aligned text lines, for the CLI, with the solver lines."""

        # how the grid was swept
        transform = self.transform
        if transform.sweep_type == "revolve":
            sweep_text = f"revolved {transform.angle_deg:g} deg about the x-axis"
        else:
            sweep_text = f"extruded {transform.translation[2]:.4g} m along z"
        lines = [
            f"  sweep                   {sweep_text}, {transform.n_planes} planes",
            f"  smallest cell volume    {self.min_cell_volume:.3e} m^3",
        ]

        # the axis: prisms in the unstructured formats
        if self.axis_cells > 0:
            lines.append(
                f"  cells around the axis   {self.axis_cells} (prisms in .su2/.msh, the axis no "
                f"boundary there), smallest "
                f"{self.min_axis_cell_volume:.3e} m^3"
            )

        # the side faces, and the solver lines for a periodic pair
        lines.append(
            f"  side faces              {self.low_name} (k = 0), {self.high_name} "
            f"(k = {transform.n_planes - 1}); mismatch {self.side_face_mismatch:.3e} m"
        )
        if self.scale != 1.0:
            lines.append(f"  coordinates in file     x {self.scale:g} (solver lines in file units)")
        su2_line = transform.su2_marker_periodic(self.low_name, self.high_name, self.scale)
        lines.append(f"  SU2:    {su2_line}")
        rmatrices = transform.eilmer_rmatrices()
        for face, matrix in rmatrices.items():
            numbers = ", ".join(f"{value:.17g}" for value in matrix)
            lines.append(f"  Eilmer: {face} Rmatrix={{{numbers}}}")
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


def shock_line_quality(
    grid: StructuredGrid, shock_line_j: int, first_spacing: float, end_spacing_fraction: float
) -> ShockLineQuality:
    """Measure the shock line of a 2D shock-aligned grid: distances, spacings, lowered lines.

    The spacing at the shock is measured along the wall normal (the direction of the
    first cell), as [points.shock_layer] sets it: a line bends toward its shock point
    across the normal, which would make the plain cell size larger.

    Args:
        grid: Structured grid with shape (ni, nj, 1).
        shock_line_j: Index j_s of the grid line on the shock.
        first_spacing: Configured wall spacing ([points.shock_layer] first_spacing) [m].
        end_spacing_fraction: Configured spacing at the shock, as a fraction of the
            wall-to-shock distance ([points.shock_layer] end_spacing_fraction).

    Returns:
        Report on the shock line.

    Raises:
        ValueError: If the grid is not 2D, or j_s isn't an inner grid line.
    """

    # validate inputs
    ni, nj, nk = grid.shape
    if nk != 1 or not 0 < shock_line_j < nj - 1:
        raise ValueError(
            f"shock_line_quality needs a 2D grid with 0 < j_s < nj - 1, got shape "
            f"{grid.shape} and j_s = {shock_line_j}"
        )

    # work on the (ni, nj) plane
    x = grid.x[:, :, 0]
    y = grid.y[:, :, 0]
    j_s = shock_line_j

    # wall-to-shock distance, and the cells either side of the shock line
    shock_distance = np.hypot(x[:, j_s] - x[:, 0], y[:, j_s] - y[:, 0])
    below = np.hypot(x[:, j_s] - x[:, j_s - 1], y[:, j_s] - y[:, j_s - 1])
    above = np.hypot(x[:, j_s + 1] - x[:, j_s], y[:, j_s + 1] - y[:, j_s])

    # lines whose wall spacing was lowered (allowance for the round-off of the stretching)
    wall_spacing = np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])
    lowered = int(np.sum(wall_spacing < first_spacing * (1.0 - 1.0e-6)))

    # the smallest wall spacing, and where it is
    i_smallest = int(np.argmin(wall_spacing))
    min_wall_spacing = float(wall_spacing[i_smallest])
    min_wall_spacing_x = float(x[i_smallest, 0])

    # wall normal of each line: the direction of its first cell
    normal_x = (x[:, 1] - x[:, 0]) / wall_spacing
    normal_y = (y[:, 1] - y[:, 0]) / wall_spacing

    # along the normal: the wall-to-shock distance and the last cell below the shock
    normal_height = (x[:, j_s] - x[:, 0]) * normal_x + (y[:, j_s] - y[:, 0]) * normal_y
    normal_below = (x[:, j_s] - x[:, j_s - 1]) * normal_x + (y[:, j_s] - y[:, j_s - 1]) * normal_y
    shock_fraction = normal_below / normal_height

    # lines whose shock spacing was lowered (allowance for the first cell's slight tilt
    # off the normal toward the bend, which measures the fraction up to ~1e-4 larger)
    shock_lowered = int(np.sum(shock_fraction < end_spacing_fraction * (1.0 - 1.0e-3)))

    # the symmetry line's distance from the axis
    axis_offset = float(np.max(np.abs(y[0, :])))

    report = ShockLineQuality(
        shock_line_j=j_s,
        shock_distance=(float(shock_distance.min()), float(shock_distance.max())),
        spacing_below=(float(below.min()), float(below.max())),
        spacing_above=(float(above.min()), float(above.max())),
        lowered_wall_spacing=lowered,
        axis_offset=axis_offset,
        min_wall_spacing=min_wall_spacing,
        min_wall_spacing_x=min_wall_spacing_x,
        lowered_shock_spacing=shock_lowered,
        min_shock_spacing_fraction=float(shock_fraction.min()),
    )
    return report


def sweep_quality(
    grid: StructuredGrid,
    transform: SweepTransform,
    low_name: str,
    high_name: str,
    scale: float = 1.0,
) -> SweepQuality:
    """Measure a swept grid: cell volumes, the cells around the axis, the side-face match.

    Args:
        grid: Structured grid with shape (ni, nj, nk), nk >= 2, in metres.
        transform: How the high side face follows from the low one.
        low_name: Marker of the side face k = 0.
        high_name: Marker of the side face k = nk - 1.
        scale: Factor the coordinates are written with ([output] scale).

    Returns:
        Sweep report.

    Raises:
        ValueError: If the grid is not 3D.
    """

    # validate inputs
    ni, nj, nk = grid.shape
    if nk < 2:
        raise ValueError(f"sweep_quality needs a 3D grid (nk >= 2), got {grid.shape}")

    # cell volumes
    xyz = np.stack([grid.x, grid.y, grid.z], axis=-1)
    volumes = cell_volumes(xyz)

    # cells around the axis: the first i-layer of a revolved grid
    axis_cells = 0
    min_axis_volume = float("nan")
    if transform.sweep_type == "revolve":
        axis_cells = (nj - 1) * (nk - 1)
        min_axis_volume = float(volumes[0].min())

    # side faces: the low face transformed onto the high one
    low_face = xyz[:, :, 0]
    high_face = xyz[:, :, -1]
    mismatch = float(np.max(np.linalg.norm(transform.apply(low_face) - high_face, axis=-1)))

    report = SweepQuality(
        transform=transform,
        low_name=low_name,
        high_name=high_name,
        min_cell_volume=float(volumes.min()),
        axis_cells=axis_cells,
        min_axis_cell_volume=min_axis_volume,
        side_face_mismatch=mismatch,
        scale=scale,
    )
    return report


def cell_volumes(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed volume of every cell of a 3D grid, from six tetrahedra around its diagonal.

    The tetrahedra share the diagonal from corner (i, j, k) to (i + 1, j + 1, k + 1)
    and fill a cell with planar faces exactly (a swept grid's faces are planar). A
    cell touching an axis, its two corners on the axis each merged with another, gets
    its prism volume: the tetrahedra with merged corners have none.

    Args:
        xyz: Grid points [m], shape (ni, nj, nk, 3).

    Returns:
        Volumes [m^3], shape (ni - 1, nj - 1, nk - 1); positive for cells
        right-handed in (i, j, k).
    """

    # cell corners, named by their (di, dj, dk) offset
    ni, nj, nk, _ = xyz.shape

    def corner(di: int, dj: int, dk: int) -> NDArray[np.float64]:
        return xyz[di : ni - 1 + di, dj : nj - 1 + dj, dk : nk - 1 + dk]

    # the diagonal, and the six corners around it, in turn
    start = corner(0, 0, 0)
    diagonal = corner(1, 1, 1) - start
    ring = [
        corner(1, 0, 0),
        corner(1, 1, 0),
        corner(0, 1, 0),
        corner(0, 1, 1),
        corner(0, 0, 1),
        corner(1, 0, 1),
    ]

    # six tetrahedra (start, ring[n], ring[n + 1], end)
    volumes = np.zeros(start.shape[:3])
    for n in range(6):
        edge_1 = ring[n] - start
        edge_2 = ring[(n + 1) % 6] - start
        triple = np.einsum("...i,...i->...", np.cross(edge_1, edge_2), diagonal)
        volumes = volumes + triple / 6.0

    return volumes


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


def _range_text(bounds: tuple[float, float]) -> str:
    """Smallest and largest value as "a to b"."""

    # format
    text = f"{bounds[0]:.3e} to {bounds[1]:.3e}"

    return text


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
