"""The [markers] tables: boundary names written into the grid file.

The names are labels (SU2 marker tags, Gmsh physical names, HDF5 attributes); the
solver input assigns the boundary conditions to them. Each grid type names its sides
in its own terms: body-fitted grids by what the side is (wall, outer, ...), Cartesian
grids by axis and end (x_min, x_max, ...). Both share the rule for what a name may be.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------
# constants
# --------------------------------------------------
# boundary names: one word each, and not the Gmsh volume's name
MARKER_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
RESERVED_MARKER_NAME = "fluid"


# --------------------------------------------------
# name rule shared by every [markers] table
# --------------------------------------------------
class MarkerNames(BaseModel):
    """Base of the [markers] tables: unknown keys are errors, every value is a valid name."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    @field_validator("*")
    @classmethod
    def _check_name(cls, name: str) -> str:
        """Names must be single words that every grid format can carry."""

        # check: letters, digits, '_', '-' and '.' only (no spaces: SU2 tags are words)
        if not MARKER_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"[markers] name {name!r} must be one word of letters, digits, '_', '-' or '.'"
            )

        # check: the volume of a Gmsh file is called "fluid"
        if name.lower() == RESERVED_MARKER_NAME:
            raise ValueError(f"[markers] name {name!r} is reserved for the volume in Gmsh output")

        return name


# --------------------------------------------------
# body-fitted grids
# --------------------------------------------------
class MarkersConfig(MarkerNames):
    """Boundary names written into the grid file, one per physical side of the grid.

    The names are labels (SU2 marker tags, Gmsh physical names, HDF5 attributes); the
    solver input assigns the boundary conditions to them. Two sides may share a name.
    """

    symmetry: str = Field(
        default="symmetry",
        description="Stagnation line ahead of the nose (on the axis for axisymmetric bodies)",
    )
    outflow: str = Field(default="outflow", description="Outflow plane at x_end")
    wall: str = Field(default="wall", description="The body")
    outer: str = Field(default="farfield", description="Outer boundary")
    sweep_low: str = Field(
        default="sweep_low", description="Side face k = 0 (the 2D plane), with [sweep]"
    )
    sweep_high: str = Field(
        default="sweep_high", description="Side face k = n_points - 1, with [sweep]"
    )


# --------------------------------------------------
# Cartesian grids
# --------------------------------------------------
class CartesianMarkersConfig(MarkerNames):
    """Boundary names of a Cartesian grid, one per end of each axis.

    Only the ends of the axes the grid has may be named ([grid.y] for y_min/y_max,
    [grid.z] for z_min/z_max). Two sides may share a name.
    """

    x_min: str = Field(default="x_min", description="First x-plane")
    x_max: str = Field(default="x_max", description="Last x-plane")
    y_min: str = Field(default="y_min", description="First y-plane, with [grid.y]")
    y_max: str = Field(default="y_max", description="Last y-plane, with [grid.y]")
    z_min: str = Field(default="z_min", description="First z-plane, with [grid.z]")
    z_max: str = Field(default="z_max", description="Last z-plane, with [grid.z]")
