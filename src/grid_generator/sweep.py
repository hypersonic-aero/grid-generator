"""3D grids from the 2D plane: revolved about the x-axis, or extruded along z ([sweep]).

The plane k = 0 of the 3D grid is the 2D grid exactly; sweeping only adds the third
index, in n_points equal steps:

- revolve (cone, tangent ogive): the meridian plane (x, r = y) turns about the x-axis
  through angle_deg, from the plane z = 0 toward +z: y = r cos(theta), z = r sin(theta).
  The symmetry line i = 0 lies on the axis, so its points are the same on every plane;
  cfd-io's unstructured writers (.su2, .msh) merge them into one node per point and
  make prisms of the cells around the axis, which then is an edge, not a boundary.
- extrude (plate, wedge): the plane is repeated at z from 0 to width.

(i, j, k) is right-handed either way: i along the body, j away from it, k toward +z.

The two side faces, k = 0 (sweep_low) and k = n_points - 1 (sweep_high), match point
for point: the high face is the low face rotated about the x-axis (or translated along
z). Whether a solver treats them as periodic or as symmetry planes is its setting;
SweepTransform gives the lines SU2 and Eilmer need for the periodic case.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from grid_generator.config.body_fitted import ExtrudeConfig, RevolveConfig


# --------------------------------------------------
# the transform between the side faces
# --------------------------------------------------
@dataclass(frozen=True, slots=True)
class SweepTransform:
    """How the high side face (k = n_planes - 1) follows from the low one (k = 0).

    Attributes:
        sweep_type: "revolve" or "extrude".
        n_planes: Points in the sweep direction (planes k = 0 .. n_planes - 1).
        angle_deg: Rotation about the +x axis through the origin [deg] (revolve; 0
            for extrude). Positive turns +y toward +z.
        translation: Shift from the low face to the high face [m] (extrude; zeros
            for revolve).
    """

    sweep_type: str
    n_planes: int
    angle_deg: float
    translation: tuple[float, float, float]

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map points of the low face onto the high face.

        Args:
            points: Points [m], shape (..., 3).

        Returns:
            The transformed points, same shape.
        """

        # rotation about +x, then the translation
        rotation = self.rotation_matrix()
        moved = points @ rotation.T + np.asarray(self.translation)

        return moved

    def rotation_matrix(self) -> NDArray[np.float64]:
        """Rotation from the low face to the high face, as a 3 x 3 matrix (identity for extrude)."""

        # rotation about the x-axis by the sector angle
        angle = math.radians(self.angle_deg)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_angle, -sin_angle],
                [0.0, sin_angle, cos_angle],
            ]
        )

        return rotation

    def su2_marker_periodic(self, low_name: str, high_name: str, scale: float = 1.0) -> str:
        """SU2's MARKER_PERIODIC line for the side faces as a periodic pair.

        SU2 maps the first marker onto the second: it rotates by the angles (about
        x, then y, then z, in degrees) about the centre, then translates.

        Args:
            low_name: Marker of the low face (k = 0).
            high_name: Marker of the high face.
            scale: Factor the coordinates are written with ([output] scale).

        Returns:
            The config line.
        """

        # centre, rotation angles, translation (in the file's units)
        tx, ty, tz = (scale * value for value in self.translation)
        values = [0.0, 0.0, 0.0, self.angle_deg, 0.0, 0.0, tx, ty, tz]
        numbers = ", ".join(_number_text(value) for value in values)
        line = f"MARKER_PERIODIC= ( {low_name}, {high_name}, {numbers} )"

        return line

    def eilmer_rmatrices(self) -> dict[str, list[float]]:
        """Eilmer's Rmatrix for ExchangeBC_FullFace on each side face (revolve only).

        Each face's ghost cells take the flow from the other face; Rmatrix turns its
        vectors into this face's frame (as in Eilmer's 3D taylor-couette example). The
        k = 0 face is Eilmer's bottom, the last k-plane its top.

        Returns:
            {"bottom": 9 numbers, row by row (from the top face: a rotation by
            -angle), "top": 9 numbers (from the bottom face: +angle)}; empty for
            extrude, which needs no reorientation.
        """

        # extruded faces are parallel: vectors need no turning
        if self.sweep_type != "revolve":
            return {}

        # bottom receives from the top: rotate back by the angle; top: forward
        to_high = self.rotation_matrix()
        to_low = to_high.T
        matrices = {
            "bottom": [float(value) for value in to_low.ravel()],
            "top": [float(value) for value in to_high.ravel()],
        }

        return matrices

    def as_dict(self, low_name: str, high_name: str, scale: float = 1.0) -> dict:
        """The transform as plain data, for the JSON file written next to the grid.

        Args:
            low_name: Marker of the low face (k = 0).
            high_name: Marker of the high face.
            scale: Factor the coordinates are written with ([output] scale).

        Returns:
            Type, side faces, transform (in the file's units) and the solver lines.
        """

        # the transform, in the file's units
        data = {
            "type": self.sweep_type,
            "n_planes": self.n_planes,
            "coordinate_scale": scale,
            "faces": {"low": low_name, "high": high_name},
        }
        if self.sweep_type == "revolve":
            data["rotation"] = {
                "axis": [1.0, 0.0, 0.0],
                "centre": [0.0, 0.0, 0.0],
                "angle_deg": self.angle_deg,
            }
        else:
            data["translation"] = [scale * value for value in self.translation]

        # ready-to-paste solver settings for periodic side faces
        data["su2"] = self.su2_marker_periodic(low_name, high_name, scale)
        rmatrices = self.eilmer_rmatrices()
        if rmatrices:
            data["eilmer_rmatrix"] = rmatrices

        return data


# --------------------------------------------------
# public API
# --------------------------------------------------
def sweep_plane(
    x_plane: NDArray[np.float64],
    y_plane: NDArray[np.float64],
    sweep: RevolveConfig | ExtrudeConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Revolve or extrude the (ni, nj) plane into an (ni, nj, n_points) grid.

    Args:
        x_plane: Plane x [m], shape (ni, nj).
        y_plane: Plane y [m] (the radius, for revolve: >= 0), shape (ni, nj).
        sweep: The validated [sweep] table.

    Returns:
        Grid x, y, z [m], each shape (ni, nj, n_points); the plane k = 0 is the
        input plane exactly (z = 0).
    """

    # extract
    n_planes = sweep.n_points
    x = np.repeat(x_plane[:, :, None], n_planes, axis=2)

    # revolve: equal angle steps about the x-axis (cos 0 = 1, sin 0 = 0 keep k = 0 exact)
    if sweep.type == "revolve":
        theta = np.linspace(0.0, math.radians(sweep.angle_deg), n_planes)
        y = y_plane[:, :, None] * np.cos(theta)[None, None, :]
        z = y_plane[:, :, None] * np.sin(theta)[None, None, :]
        return x, y, z

    # extrude: the same plane at equally spaced z
    z_planes = np.linspace(0.0, sweep.width, n_planes)
    y = np.repeat(y_plane[:, :, None], n_planes, axis=2)
    z = np.broadcast_to(z_planes[None, None, :], x.shape).copy()

    return x, y, z


def sweep_transform(sweep: RevolveConfig | ExtrudeConfig) -> SweepTransform:
    """The transform from the low side face to the high one, for a [sweep] table.

    Args:
        sweep: The validated [sweep] table.

    Returns:
        The rotation (revolve) or translation (extrude).
    """

    # revolve: rotation about +x through the sector angle
    if sweep.type == "revolve":
        transform = SweepTransform(
            sweep_type="revolve",
            n_planes=sweep.n_points,
            angle_deg=sweep.angle_deg,
            translation=(0.0, 0.0, 0.0),
        )
        return transform

    # extrude: translation along z by the width
    transform = SweepTransform(
        sweep_type="extrude",
        n_planes=sweep.n_points,
        angle_deg=0.0,
        translation=(0.0, 0.0, sweep.width),
    )

    return transform


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _number_text(value: float) -> str:
    """A number for a solver config: plain decimals, full precision, -0 as 0."""

    # avoid "-0.0" and exponent noise in the pasted line
    if value == 0.0:
        return "0.0"
    text = repr(float(value))

    return text
