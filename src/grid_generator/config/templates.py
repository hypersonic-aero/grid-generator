"""Starter configurations written by `grid-generator <command> init`.

The templates are put together from TOML blocks: per geometry (description,
[geometry], [domain]), and shared blocks for the points, span, markers and output. The body
and wall-normal distributions each have a choice of block (init --body, --normal),
the span block is optional (init --span), a shock-line grid replaces the constant
height and the wall-normal block (init --shock), and Cartesian grids pick their axes
(init --dims). 1D distributions take their example from the distribution reference
(init --type). Every combination is a complete, valid file.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from grid_generator.distributions.catalog import DISTRIBUTIONS

# --------------------------------------------------
# Cartesian grids: one block per axis
# --------------------------------------------------
CARTESIAN_GRID_BLOCK = """\
[grid]
type = "cartesian"
"""

# axes in order; --dims 1, 2, 3 takes the first 1, 2, 3 of them
CARTESIAN_AXIS_BLOCKS: dict[str, str] = {
    "x": """\
[grid.x]
type = "equidistant"
start = 0.1
end = 1.0
n_points = 10
""",
    "y": """\
[grid.y]
type = "geometric"
start = 0.0
end = 0.05
n_points = 100
first_spacing = 1.0e-6
""",
    "z": """\
[grid.z]
type = "equidistant"
start = -0.1
end = 0.1
n_points = 20
""",
}

CARTESIAN_SEGMENTED_EXAMPLE = """\
# Segmented axes compose equidistant and polynomial regions. For example:
# [grid.x]
# type = "segmented"
# start = 0.0
#
# [[grid.x.segments]]
# type = "equidistant"
# end = 0.1
# spacing = 0.005
#
# [[grid.x.segments]]
# type = "quintic"
# end = 0.4
# n_points = 31
#
# [[grid.x.segments]]
# type = "equidistant"
# end = 1.0
# spacing = 0.01
"""

CARTESIAN_MARKERS_BLOCK = """\
# Boundary names in the grid file (one per end of each axis); the solver input assigns
# the boundary conditions to them. Two sides may share a name.
[markers]
"""

# [markers] lines of each axis, added with the axis
CARTESIAN_MARKER_LINES: dict[str, str] = {
    "x": 'x_min = "x_min"\nx_max = "x_max"\n',
    "y": 'y_min = "y_min"\ny_max = "y_max"\n',
    "z": 'z_min = "z_min"\nz_max = "z_max"\n',
}

CARTESIAN_OUTPUT_BLOCK = """\
[output]
filename = "grid.hdf5"
dtype = "f8"
"""

# default number of dimensions (x and y)
CARTESIAN_DEFAULT_DIMS = 2


# --------------------------------------------------
# body-fitted grids: description, [geometry] and [domain], one block per geometry
# --------------------------------------------------
BODY_FITTED_GEOMETRY_BLOCKS: dict[str, str] = {
    "plate": """\
# Flat plate with a circular leading edge (upper half), same contract as
# canonical-geometries (geometry plate init) without length: the plate ends at
# [domain] x_end.
# Stagnation point at the origin, x downstream, y up.
[geometry]
# type marker
type = "plate"
# nose radius in [m]
nose_radius = 100.0e-6

# Extent of the grid.
[domain]
# end of the grid (and of the body) along x, measured from nose tip in [m]
x_end = 0.01
""",
    "wedge": """\
# Planar blunted wedge (upper half), same contract as canonical-geometries
# (geometry wedge init) without length: the wedge ends at [domain] x_end.
# The nose arc runs to 90 deg - half_angle, tangent to the body.
# Stagnation point at the origin, x downstream, y up.
[geometry]
# type marker
type = "wedge"
# nose radius in [m]
nose_radius = 100.0e-6
# half-angle of the body in [deg]
half_angle = 7.0

# Extent of the grid.
[domain]
# end of the grid (and of the body) along x, measured from nose tip in [m]
x_end = 0.01
""",
    "cone": """\
# Sphere-cone, axisymmetric about the x axis, same contract as canonical-geometries
# (geometry cone init) without length: the cone ends at [domain] x_end.
# The grid is the meridian plane: run the solver axisymmetric.
# The nose arc runs to 90 deg - half_angle, tangent to the cone.
# Stagnation point at the origin, x along the axis, y radial.
[geometry]
# type marker
type = "cone"
# nose radius in [m]
nose_radius = 100.0e-6
# cone half-angle in [deg]
half_angle = 7.0

# Extent of the grid.
[domain]
# end of the grid (and of the body) along x, measured from nose tip in [m]
x_end = 0.01
""",
    "tangent-ogive": """\
# Spherically blunted tangent ogive, axisymmetric about the x axis, same contract as
# canonical-geometries (geometry tangent-ogive init) without cylinder_length: when
# [domain] x_end lies past the ogive base, a cylinder carries the body on to x_end.
# The grid is the meridian plane: run the solver axisymmetric.
# Stagnation point at the origin, x along the axis, y radial.
[geometry]
# type marker
type = "tangent-ogive"
# nose radius in [m] (smaller than base_radius)
nose_radius = 100.0e-6
# length of the sharp ogive, tip to base, in [m]; blunting makes the body slightly
# shorter (the sharp tip would lie upstream of the stagnation point)
length = 0.5
# base radius in [m] (smaller than length)
base_radius = 0.05

# Extent of the grid.
[domain]
# end of the grid along x, measured from nose tip in [m]: on the ogive, or past its
# base (then on a cylinder of the base radius)
x_end = 0.01
""",
}


# --------------------------------------------------
# body-fitted grids: the outer boundary, a constant height or a shock line
# --------------------------------------------------
# a constant height ends the geometry's [domain] block
CONSTANT_HEIGHT_LINES = """\
# constant wall-normal height of the grid in [m]; must clear the bow shock at x_end
height = 5.0e-3
"""

HEIGHT_LINES: dict[str, str] = {
    "plate": """\
# constant wall-normal height of the grid in [m]; must clear the bow shock at x_end
# (for this Mach 6 plate case the shock is at about 2.8 mm at x = 10 mm)
height = 5.0e-3
""",
    "wedge": CONSTANT_HEIGHT_LINES,
    "cone": CONSTANT_HEIGHT_LINES,
    "tangent-ogive": CONSTANT_HEIGHT_LINES,
}

# a shock line follows [domain] instead (init --shock)
SHOCK_BLOCK = """\
# Shock line: the bow shock is a grid line, the same one (j = [points.shock_layer]
# n_points - 1) on every grid line, so a solution's captured shock stays in one row
# of cells. The lines run from the wall to the shock, then on to the outer boundary.
[shock]
# shock curve for this body: CSV with x, y columns and a header line, e.g. from
# flow-theory shock-shape run with a .csv output, or a shock taken from a CFD
# solution; the path is relative to this file
curve = "shock.csv"
# outer boundary: margin_factor x the wall-to-shock distance, plus min_offset
margin_factor = 1.5
# distance added beyond the scaled shock distance in [m]
min_offset = 0.0
"""


# --------------------------------------------------
# body-fitted grids: points along the wall (nose, then a choice of body block)
# --------------------------------------------------
BODY_FITTED_NOSE_BLOCK = """\
# Point distributions: the same types and keys as the Cartesian axes, without start
# and end (the geometry and [domain] fix them). Wall points: nose + body - 1 (nose
# and body share the junction).

# along the wall, nose arc: equal angle steps (a polar block around the nose)
[points.nose]
# points on the nose arc
n_points = 81
"""

BODY_BLOCKS: dict[str, str] = {
    "geometric": """\
# along the wall, body: from the junction to x_end. The first spacing is the nose
# spacing (automatic), so the wall spacing is continuous across the junction.
[points.body]
type = "geometric"
# points on the body; the growth ratio follows from the count
n_points = 121
# optional: stop growing at this spacing along the wall in [m], uniform from there
# max_spacing = 1.0e-4
""",
    "segmented": """\
# along the wall, body: from the junction to x_end, in regions. Segment ends in x [m];
# spacings along the wall [m], rounded to whole cells. The first region keeps the
# nose spacing (its end moves to a whole number of nose spacings), so the wall spacing
# is continuous across the junction; the last region ends at x_end.
[points.body]
type = "segmented"

# fine: the nose spacing up to x = 0.2 mm
[[points.body.segments]]
type = "equidistant"
end = 0.2e-3

# transition: the spacing changes smoothly (quintic) up to x = 2 mm
[[points.body.segments]]
type = "quintic"
end = 2.0e-3
n_points = 81

# coarse: 50 micron spacing up to x_end
[[points.body.segments]]
type = "equidistant"
spacing = 50.0e-6
""",
}


# --------------------------------------------------
# body-fitted grids: points away from the wall (a choice of normal block)
# --------------------------------------------------
NORMAL_BLOCKS: dict[str, str] = {
    "geometric": """\
# away from the wall: from the wall to [domain] height, geometric growth
[points.normal]
type = "geometric"
# points in the wall-normal direction
n_points = 101
# first wall-normal spacing in [m]
first_spacing = 1.0e-6
# optional: stop growing at this spacing in [m], uniform from there to the height
# max_spacing = 1.0e-4
""",
    "tanh": """\
# away from the wall: from the wall to [domain] height, tanh clustering
[points.normal]
type = "tanh"
# points in the wall-normal direction
n_points = 101
# clustering strength: larger means smaller cells where clustered (3.0 gives a first
# spacing of about 1.5 micron for this height and n_points)
strength = 3.0
# cluster at the wall ("start"), at the outer boundary ("end"), or at both ("both")
cluster = "start"
""",
}


# a shock-line grid's points away from the wall, in its two segments (init --shock)
SHOCK_LINE_POINTS_BLOCK = """\
# away from the wall, in the shock layer: from the wall to the shock, two-sided
# stretching (fine at the wall for the boundary layer, and again at the shock)
[points.shock_layer]
# distribution type: vinokur (a given spacing at both ends)
type = "vinokur"
# points from the wall to the shock, both included
n_points = 161
# wall spacing in [m]; lowered on lines too short for it (near a small nose)
first_spacing = 1.0e-6
# spacing at the shock, as a fraction of each line's wall-to-shock distance
end_spacing_fraction = 0.01

# beyond the shock: from the shock to the outer boundary, growing from the spacing at
# the shock (automatic), so the spacing is continuous across the shock
[points.freestream]
# distribution type: geometric
type = "geometric"
# points from the shock to the outer boundary, both included
n_points = 41
"""


# --------------------------------------------------
# body-fitted grids: optional span, and output
# --------------------------------------------------
SPAN_BLOCK = """\
# Extrusion in z (spanwise) into a 3D grid: the same 2D plane at every z. Same
# distribution types as the Cartesian axes. The two new sides are marked span_min and
# span_max (e.g. periodic or symmetry in the solver). 3D output: .su2, .h5 or .p3dfmt
# (not .msh or .p2dfmt).
[span]
type = "equidistant"
start = 0.0
end = 2.0e-3
n_points = 21
"""

MARKERS_BLOCK = """\
# Boundary names in the grid file; the solver input assigns the boundary conditions to
# them. Two sides may share a name.
[markers]
# the body
wall = "wall"
# outer boundary
outer = "farfield"
# outflow plane at x_end
outflow = "outflow"
# stagnation line ahead of the nose (on the axis for axisymmetric bodies)
symmetry = "symmetry"
"""

# extra [markers] lines for an extruded grid (init --span)
SPAN_MARKERS_LINES = """\
# first and last z-plane of the extruded grid
span_min = "span_min"
span_max = "span_max"
"""

BODY_FITTED_OUTPUT_BLOCK = """\
# The path is relative to this file.
[output]
# grid file; the extension picks the format via cfd-io (.su2, .h5, .p2dfmt, .msh, ...)
filename = "{geometry_type}_local.su2"
# element order, for .msh only: 1 (straight-edged quads) or 2 (curved 9-node quads
# for high-order solvers; needs odd point counts in every [points] table)
order = 1
"""

# default choices for init
DEFAULT_BODY = "geometric"
DEFAULT_NORMAL = "geometric"


# --------------------------------------------------
# 1D distributions: the reference example of the chosen type, and output
# --------------------------------------------------
DISTRIBUTION_OUTPUT_BLOCK = """\
# The path is relative to this file.
[output]
# point file: .dat (Tecplot ASCII) or .csv with the columns i, x, dx, ratio;
# .txt with x only (one value per line, no header)
filename = "distribution.dat"
"""

# default type for init
DEFAULT_DISTRIBUTION = "geometric"


# --------------------------------------------------
# public API
# --------------------------------------------------
def cartesian_template(dims: int = CARTESIAN_DEFAULT_DIMS) -> str:
    """Starter configuration for a Cartesian grid with 1, 2 or 3 axes (x, then y, then z).

    Args:
        dims: Number of axes.

    Returns:
        TOML text, as `grid-generator cartesian init` writes it.

    Raises:
        ValueError: If dims isn't 1, 2 or 3.
    """

    # validate inputs
    axis_names = list(CARTESIAN_AXIS_BLOCKS)
    if dims not in range(1, len(axis_names) + 1):
        raise ValueError(f"dims must be 1, 2 or 3, got {dims}")

    # header, with the options that made this file
    options = f" --dims {dims}" if dims != CARTESIAN_DEFAULT_DIMS else ""
    header = (
        f"# Generated by: grid-generator cartesian init{options}\n"
        "# Edit this file, then run: grid-generator cartesian run -c cartesian.toml\n"
    )

    # join the blocks
    blocks = [header, CARTESIAN_GRID_BLOCK]
    for name in axis_names[:dims]:
        blocks.append(CARTESIAN_AXIS_BLOCKS[name])
    blocks.append(CARTESIAN_SEGMENTED_EXAMPLE)
    marker_lines = [CARTESIAN_MARKER_LINES[name] for name in axis_names[:dims]]
    blocks.append(CARTESIAN_MARKERS_BLOCK + "".join(marker_lines))
    blocks.append(CARTESIAN_OUTPUT_BLOCK)
    template = "\n".join(blocks)

    return template


def body_fitted_template(
    geometry_type: str,
    body: str = DEFAULT_BODY,
    normal: str | None = None,
    span: bool = False,
    shock: bool = False,
) -> str:
    """Starter configuration for a body-fitted grid.

    Args:
        geometry_type: Key in BODY_FITTED_GEOMETRY_BLOCKS, e.g. "plate".
        body: Key in BODY_BLOCKS: the body distribution along the wall.
        normal: Key in NORMAL_BLOCKS: the wall-normal distribution of a
            constant-height grid (default: DEFAULT_NORMAL).
        span: Add a [span] table (extrusion in z; planar bodies only).
        shock: Write a shock-line grid ([shock], [points.shock_layer],
            [points.freestream]) instead of a constant-height one.

    Returns:
        TOML text, as `grid-generator <geometry_type> init` writes it.

    Raises:
        ValueError: If a key isn't in its registry, or a normal distribution is
            picked for a shock-line grid.
    """

    # validate inputs: a shock-line grid has no [points.normal] to pick
    if shock and normal is not None:
        raise ValueError(
            "a normal distribution sets [points.normal], which a shock-line grid doesn't have"
        )
    if normal is None:
        normal = DEFAULT_NORMAL

    # validate inputs
    choices = [
        (geometry_type, BODY_FITTED_GEOMETRY_BLOCKS, "geometry type"),
        (body, BODY_BLOCKS, "body distribution"),
        (normal, NORMAL_BLOCKS, "normal distribution"),
    ]
    for key, registry, name in choices:
        if key not in registry:
            raise ValueError(f"unknown {name} {key!r}; choose from: {', '.join(registry)}")

    # header, with the options that made this file
    options = ""
    if body != DEFAULT_BODY:
        options += f" --body {body}"
    if normal != DEFAULT_NORMAL:
        options += f" --normal {normal}"
    if span:
        options += " --span"
    if shock:
        options += " --shock"
    header = (
        f"# Generated by: grid-generator {geometry_type} init{options}\n"
        f"# Edit this file, then run: grid-generator {geometry_type} run -c {geometry_type}.toml\n"
    )

    # geometry and [domain], ended by the constant height, or followed by [shock]
    geometry_block = BODY_FITTED_GEOMETRY_BLOCKS[geometry_type]
    if shock:
        outer_blocks = [geometry_block, SHOCK_BLOCK]
        normal_block = SHOCK_LINE_POINTS_BLOCK
    else:
        outer_blocks = [geometry_block + HEIGHT_LINES[geometry_type]]
        normal_block = NORMAL_BLOCKS[normal]

    # join the blocks
    blocks = [header, *outer_blocks, BODY_FITTED_NOSE_BLOCK, BODY_BLOCKS[body], normal_block]
    if span:
        blocks.append(SPAN_BLOCK)
        blocks.append(MARKERS_BLOCK + SPAN_MARKERS_LINES)
    else:
        blocks.append(MARKERS_BLOCK)
    blocks.append(BODY_FITTED_OUTPUT_BLOCK.format(geometry_type=geometry_type))
    template = "\n".join(blocks)

    return template


def distribution_template(kind: str = DEFAULT_DISTRIBUTION) -> str:
    """Starter configuration for a 1D point distribution of the given type.

    Args:
        kind: Key in the distribution reference, e.g. "geometric".

    Returns:
        TOML text, as `grid-generator distribution init` writes it.

    Raises:
        ValueError: If kind isn't a distribution type.
    """

    # validate inputs
    if kind not in DISTRIBUTIONS:
        raise ValueError(
            f"unknown distribution type {kind!r}; choose from: {', '.join(DISTRIBUTIONS)}"
        )

    # header, with the options that made this file
    options = f" --type {kind}" if kind != DEFAULT_DISTRIBUTION else ""
    header = (
        f"# Generated by: grid-generator distribution init{options}\n"
        "# Edit this file, then run: grid-generator distribution run -c distribution.toml\n"
    )

    # the reference example, as the [distribution] table
    info = DISTRIBUTIONS[kind]
    distribution_block = (
        f"# {kind}: {info.summary}\n"
        f"# keys and their meaning: grid-generator distribution info {kind}\n"
        "[distribution]\n" + info.example.format(table="distribution")
    )

    # join the blocks
    template = "\n".join([header, distribution_block, DISTRIBUTION_OUTPUT_BLOCK])

    return template
