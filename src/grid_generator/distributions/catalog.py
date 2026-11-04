"""Reference for the distribution types: what each one does, its keys, and an example.

`grid-generator distribution info` prints it, and `grid-generator distribution init`
writes its example toml files.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------
# data carriers
# --------------------------------------------------
@dataclass(frozen=True, slots=True)
class KeyInfo:
    """One key of a distribution table, described with JSON Schema's vocabulary.

    Attributes:
        name: Key name in the TOML table.
        description: What the key sets.
        required: The key must be given.
        default: Value used when the key is left out (as TOML text), if any.
        one_of: Group of keys of which exactly one must be given (this key included).
        requires: Keys that must also be given when this key is.
    """

    name: str
    description: str
    required: bool = False
    default: str | None = None
    one_of: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DistributionInfo:
    """Reference entry for one distribution (or segment) type.

    Attributes:
        summary: One line on what the type does.
        keys: Its keys (besides the "type" marker), in the order to list them.
        example: TOML lines for the table, without the table header; "{table}"
            stands for the table name (segmented tables use it).
        notes: Extra rules, one line each.
    """

    summary: str
    keys: tuple[KeyInfo, ...]
    example: str = ""
    notes: tuple[str, ...] = ()


# --------------------------------------------------
# shared keys
# --------------------------------------------------
START = KeyInfo(name="start", description="first point [m]", required=True)
END = KeyInfo(name="end", description="last point [m]", required=True)
N_POINTS = KeyInfo(
    name="n_points", description="number of points, both ends included", required=True
)

# the two ways to set a geometric distribution's growth
SPACING_OR_RATIO = ("first_spacing", "ratio")

# the two ways to size an equidistant segment
POINTS_OR_SPACING = ("n_points", "spacing")


# --------------------------------------------------
# distribution types
# --------------------------------------------------
DISTRIBUTIONS: dict[str, DistributionInfo] = {
    "equidistant": DistributionInfo(
        summary="equally spaced points",
        keys=(START, END, N_POINTS),
        example="""\
type = "equidistant"
start = 0.0
end = 1.0
n_points = 11
""",
    ),
    "geometric": DistributionInfo(
        summary="the spacing changes by a constant ratio from one cell to the next",
        keys=(
            START,
            END,
            N_POINTS,
            KeyInfo(
                name="first_spacing",
                description="first spacing [m]; sets the ratio",
                one_of=SPACING_OR_RATIO,
            ),
            KeyInfo(
                name="ratio",
                description="ratio of neighbouring spacings",
                one_of=SPACING_OR_RATIO,
            ),
            KeyInfo(
                name="max_spacing",
                description="spacing cap [m]; uniform beyond it",
                requires=("first_spacing",),
            ),
        ),
        example="""\
type = "geometric"
start = 0.0
end = 5.0e-3
n_points = 101
first_spacing = 1.0e-6
""",
    ),
    "tanh": DistributionInfo(
        summary="hyperbolic-tangent clustering at one or both ends",
        keys=(
            START,
            END,
            N_POINTS,
            KeyInfo(
                name="strength",
                description="clustering strength; larger is finer where clustered",
                required=True,
            ),
            KeyInfo(
                name="cluster",
                description='where to cluster: "start", "end" or "both"',
                default='"start"',
            ),
        ),
        example="""\
type = "tanh"
start = 0.0
end = 5.0e-3
n_points = 101
strength = 3.0
cluster = "start"
""",
    ),
    "vinokur": DistributionInfo(
        summary="two-sided stretching: a given spacing at each end (Vinokur)",
        keys=(
            START,
            END,
            N_POINTS,
            KeyInfo(name="first_spacing", description="spacing at start [m]", required=True),
            KeyInfo(name="end_spacing", description="spacing at end [m]", required=True),
        ),
        example="""\
type = "vinokur"
start = 0.0
end = 5.0e-3
n_points = 101
first_spacing = 1.0e-6
end_spacing = 2.0e-5
""",
        notes=(
            "the cells are finest at the ends: sqrt(first_spacing x end_spacing) may not "
            "exceed the uniform spacing (end - start) / (n_points - 1)",
        ),
    ),
    "segmented": DistributionInfo(
        summary="equidistant regions joined by smooth polynomial transitions",
        keys=(
            START,
            KeyInfo(
                name="segments",
                description="the regions in order, each a [[<table>.segments]] table",
                required=True,
            ),
        ),
        example="""\
type = "segmented"
start = 0.0

[[{table}.segments]]
type = "equidistant"
end = 0.1
spacing = 0.005

[[{table}.segments]]
type = "quintic"
end = 0.4
n_points = 31

[[{table}.segments]]
type = "equidistant"
end = 1.0
spacing = 0.01
""",
        notes=(
            "regions start with an equidistant one and alternate equidistant / polynomial",
            "the distribution ends where the last region ends",
            "an equidistant spacing must divide its region exactly",
        ),
    ),
}


# --------------------------------------------------
# segment types (regions of a segmented distribution)
# --------------------------------------------------
SEGMENT_END = KeyInfo(name="end", description="end of the region [m]", required=True)

SEGMENTS: dict[str, DistributionInfo] = {
    "equidistant": DistributionInfo(
        summary="uniform region",
        keys=(
            SEGMENT_END,
            KeyInfo(
                name="n_points",
                description="points in the region, both ends included",
                one_of=POINTS_OR_SPACING,
            ),
            KeyInfo(
                name="spacing",
                description="spacing in the region [m]",
                one_of=POINTS_OR_SPACING,
            ),
        ),
    ),
    "quintic": DistributionInfo(
        summary="smooth transition between two equidistant regions",
        keys=(
            SEGMENT_END,
            KeyInfo(
                name="n_points",
                description="points in the region, both ends included",
                required=True,
            ),
        ),
    ),
    "cubic": DistributionInfo(
        summary="smooth transition from the previous region's spacing (last region only)",
        keys=(
            SEGMENT_END,
            KeyInfo(
                name="n_points",
                description="points in the region, both ends included",
                required=True,
            ),
        ),
    ),
}


# --------------------------------------------------
# public API
# --------------------------------------------------
def requirement_text(key: KeyInfo) -> str:
    """Whether a key must be given, as one phrase: "required", "optional; default ...", ...

    Args:
        key: Key to describe.

    Returns:
        "required", "exactly one of: a, b", or "optional", followed by
        "; default ..." and "; requires ..." where they apply.
    """

    # first: required, part of an exactly-one-of group, or optional
    if key.one_of:
        parts = [f"exactly one of: {', '.join(key.one_of)}"]
    elif key.required:
        parts = ["required"]
    else:
        parts = ["optional"]

    # then the default and the keys this one depends on
    if key.default is not None:
        parts.append(f"default {key.default}")
    if key.requires:
        parts.append(f"requires {', '.join(key.requires)}")
    text = "; ".join(parts)

    return text


# --------------------------------------------------
# where the distributions are used, and the rules of body-fitted grids
# --------------------------------------------------
USED_IN = "[grid.x/y/z], [points.body], [points.normal], [span], [distribution]"

BODY_FITTED_RULES: tuple[str, ...] = (
    "[points.body], [points.normal]: no start or end (the geometry and [domain] set them)",
    "segmented there: the last segment has no end (it ends at x_end or height)",
    "[points.body]: geometric or segmented only; it starts at the nose spacing",
    "[points.body] geometric: no first_spacing or ratio (the nose spacing is the first)",
    "[points.body] segmented: the first segment gives only its end (at the nose spacing)",
    "[points.body] segments: ends in x, spacings along the wall (rounded to whole cells)",
    "[points.shock_layer] (shock-line grids, instead of [points.normal]): vinokur from "
    "the wall to the shock, with end_spacing_fraction (of each line's wall-to-shock "
    "distance) instead of end_spacing",
    "[points.freestream] (shock-line grids): geometric from the shock to the outer "
    "boundary, n_points only (it starts at the spacing at the shock)",
)
