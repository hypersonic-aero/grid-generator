"""[points] and [span] tables of a body-fitted grid config, turned into distribution specs.

The tables use the same distribution types and keys as the Cartesian axes, without
start and end: those come from the geometry and [domain]. The body starts at the
nose spacing, so the wall spacing is continuous across the nose/body junction.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
from canonical_geometries import Profile2D
from numpy.typing import ArrayLike, NDArray

from grid_generator.distributions.models import DistributionSpec, GeometricSpec
from grid_generator.distributions.parse import parse_distribution


# --------------------------------------------------
# public API
# --------------------------------------------------
def parse_span(value: object) -> object:
    """Turn a raw [span] table into a distribution spec (same rules as a Cartesian axis).

    Args:
        value: Raw [span] table, or anything else (None, an existing spec), which
            passes through unchanged.

    Returns:
        Distribution spec for a table, otherwise value unchanged.

    Raises:
        ValueError: If the table is invalid.
    """

    # tables come from TOML; anything else (None, an existing spec) passes through
    span = value
    if isinstance(value, dict):
        span = parse_table(value, "span")

    return span


def parse_table(table: dict[str, Any], label: str) -> DistributionSpec:
    """Parse one distribution table, with the table name on every error.

    Args:
        table: Raw TOML table, with start/end already filled in where needed.
        label: Table name for error messages, e.g. "points.normal".

    Returns:
        Validated distribution spec.

    Raises:
        ValueError: If the table is invalid (also for shape errors, so pydantic
            reports them as validation errors).
    """

    try:
        spec = parse_distribution(table, label)
    except (TypeError, ValueError) as error:
        # parser messages name the table already; spec checks don't
        message = str(error)
        if not message.startswith("["):
            message = f"[{label}] {message}"
        raise ValueError(message) from None

    return spec


def build_body_spec(
    table: dict[str, Any], profile: Profile2D, n_nose: int, x_end: float
) -> DistributionSpec:
    """Body distribution in wall arc length, from the nose junction to x_end.

    It starts at the nose spacing. Positions in the table (segment ends) are given in
    x; spacings (segment spacing, max_spacing) are measured along the wall. The grid
    may end before the end of the profile (e.g. partway along an ogive).

    Args:
        table: Raw [points.body] table.
        profile: Wall profile; its first junction is the end of the nose.
        n_nose: Points on the nose arc (sets the nose spacing).
        x_end: End of the grid along x [m].

    Returns:
        Geometric or segmented distribution spec in wall arc length [m].

    Raises:
        ValueError: If the table is invalid, sets a key the grid fixes, or uses a type
            that can't start from a given spacing.
    """

    # extract the nose junction, the nose spacing and the grid's end on the body
    label = "points.body"
    s_junction = float(profile.junctions[0])
    s_end = float(arc_length_at_x(profile, x_end, s_junction))
    nose_spacing = s_junction / (n_nose - 1)

    # geometric: growth from the nose spacing to x_end
    kind = table.get("type", "equidistant")
    if kind == "geometric":
        _reject_keys(
            table,
            {"start", "end", "first_spacing", "ratio"},
            label,
            "the body runs from the junction to x_end, starts at the nose spacing, "
            "and n_points sets the growth",
        )
        raw = dict(table)
        raw["start"] = s_junction
        raw["end"] = s_end
        raw["first_spacing"] = nose_spacing
        spec = parse_table(raw, label)
        return spec

    # segmented: regions from the junction to x_end, the first at the nose spacing
    if kind == "segmented":
        _reject_keys(table, {"start"}, label, "the body starts at the nose/body junction")
        segments = _segments_ending_at(table, x_end, label, "[domain] x_end")
        if len(segments) < 2:
            raise ValueError(
                f"[{label}] needs at least two segments: the first keeps the nose spacing, "
                "and a later one ends at x_end"
            )
        raw = dict(table)
        raw["start"] = s_junction
        raw["segments"] = _body_segments_in_arc_length(
            segments, profile, s_junction, s_end, nose_spacing
        )
        spec = parse_table(raw, label)
        return spec

    # other types can't start from a given spacing
    raise ValueError(
        f"[{label}] type = {kind!r} can't start from the nose spacing; "
        "use 'geometric' or 'segmented'"
    )


def arc_length_at_x(profile: Profile2D, x: ArrayLike, s_start: float) -> NDArray[np.float64]:
    """Arc length where the profile reaches the given x, searched from s_start onwards.

    x must grow along the profile from s_start (true on the body of every geometry
    here: the wall never turns back upstream). Bisection, to round-off.

    Args:
        profile: Wall profile.
        x: Axial position(s) [m], between x(s_start) and the end of the profile.
        s_start: Arc length to search from (e.g. the nose junction).

    Returns:
        Arc length(s) [m] with the shape of x.

    Raises:
        ValueError: If an x lies outside the searched part of the profile.
    """

    # validate inputs
    x_arr = np.asarray(x, dtype=float)
    s_total = profile.total_length
    x_first = float(profile.point(s_start)[0])
    x_last = float(profile.point(s_total)[0])
    tol = 1e-12 * max(abs(x_last), 1.0)
    if np.any(x_arr < x_first - tol) or np.any(x_arr > x_last + tol):
        raise ValueError(
            f"x must lie between {x_first} and {x_last} on the profile, "
            f"got values in [{x_arr.min()}, {x_arr.max()}]"
        )

    # bisection on [s_start, s_total] for every x at once
    s_lo = np.full_like(x_arr, s_start)
    s_hi = np.full_like(x_arr, s_total)
    for _ in range(200):
        s_mid = 0.5 * (s_lo + s_hi)
        x_mid, _ = profile.point(s_mid)
        below = x_mid < x_arr
        s_lo = np.where(below, s_mid, s_lo)
        s_hi = np.where(below, s_hi, s_mid)
    s_found = 0.5 * (s_lo + s_hi)

    # the ends exactly (no round-off from the search)
    s_found = np.where(x_arr >= x_last, s_total, s_found)
    s_found = np.where(x_arr <= x_first, s_start, s_found)

    return s_found


def build_normal_spec(table: dict[str, Any], height: float) -> DistributionSpec:
    """Wall-normal distribution from the wall (0) to height; any distribution type.

    Args:
        table: Raw [points.normal] table.
        height: Wall-normal height of the grid [m].

    Returns:
        Distribution spec in wall distance [m].

    Raises:
        ValueError: If the table is invalid or sets start or end, or a geometric
            first_spacing doesn't fit the height (the cells would shrink).
    """

    # validate inputs: start and end come from [domain]
    label = "points.normal"
    _reject_keys(
        table, {"start", "end"}, label, "the normal direction runs from 0 to [domain] height"
    )

    # fill in start and end (a segmented table's last segment ends at the height)
    raw = dict(table)
    raw["start"] = 0.0
    kind = raw.get("type", "equidistant")
    if kind == "segmented":
        raw["segments"] = _segments_ending_at(table, height, label, "[domain] height")
    else:
        raw["end"] = height

    # parse
    spec = parse_table(raw, label)

    # check: from a fixed first spacing the cells must grow away from the wall; on a
    # line shorter than (n_points - 1) x first_spacing a geometric distribution would
    # shrink them instead, down to vanishing cells at the outer end
    if isinstance(spec, GeometricSpec) and spec.first_spacing is not None:
        needed = (spec.n_points - 1) * spec.first_spacing
        if needed > height:
            raise ValueError(
                f"[{label}] {spec.n_points} points from first_spacing = "
                f"{spec.first_spacing:.3g} m need a line longer than (n_points - 1) x "
                f"first_spacing = {needed:.3g} m, but this one is {height:.3g} m, so the "
                "cells would shrink away from the wall: lower n_points or first_spacing, "
                "or lengthen the line ([domain] height, or [outer] min_offset)"
            )

    return spec


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _reject_keys(table: dict[str, Any], keys: set[str], label: str, reason: str) -> None:
    """Raise if the table sets any key that the grid fixes on its own."""

    # check
    given = sorted(keys & set(table))
    if given:
        raise ValueError(f"[{label}] don't set {', '.join(given)}: {reason}")


def _body_segments_in_arc_length(
    segments: list[dict[str, Any]],
    profile: Profile2D,
    s_junction: float,
    s_end: float,
    nose_spacing: float,
) -> list[dict[str, Any]]:
    """Convert body segments to wall arc length; the first gets the nose spacing.

    Segment ends are given in x and found along the profile. The first segment is
    equidistant at the nose spacing, so its end moves to the nearest whole number of
    nose spacings (by less than one spacing). Later equidistant segments give their
    spacing along the wall; it is rounded to a whole number of cells in the segment.
    """

    # validate inputs: the first segment's spacing is the nose spacing
    first = segments[0]
    if not isinstance(first, dict) or first.get("type") != "equidistant":
        raise ValueError(
            "[points.body.segments[0]] must be equidistant: it continues the nose spacing"
        )
    _reject_keys(
        first,
        {"spacing", "n_points"},
        "points.body.segments[0]",
        "its spacing is the nose spacing; give only end",
    )

    # validate inputs: every end lies on the body, past the nose
    x_junction = float(profile.point(s_junction)[0])
    for index, segment in enumerate(segments):
        if isinstance(segment, dict) and "end" in segment and float(segment["end"]) <= x_junction:
            raise ValueError(
                f"[points.body.segments[{index}]] end ({segment['end']}) must lie past the "
                f"nose (x = {x_junction})"
            )

    # convert every segment's end from x to arc length
    converted = []
    for segment in segments:
        new_segment = dict(segment)
        if "end" in new_segment:
            new_segment["end"] = float(
                arc_length_at_x(profile, float(new_segment["end"]), s_junction)
            )
        converted.append(new_segment)

    # the last segment ends exactly at the end of the grid (no round-off from the search)
    converted[-1]["end"] = s_end

    # first segment: nose spacing, end moved to a whole number of spacings
    n_intervals = max(1, round((converted[0]["end"] - s_junction) / nose_spacing))
    converted[0]["end"] = s_junction + n_intervals * nose_spacing
    converted[0]["spacing"] = nose_spacing

    # later equidistant segments: spacing along the wall, rounded to whole cells
    segment_start = converted[0]["end"]
    for segment in converted[1:]:
        if segment.get("type") == "equidistant" and "spacing" in segment and "end" in segment:
            length = segment["end"] - segment_start
            n_cells = max(1, round(length / float(segment["spacing"])))
            del segment["spacing"]
            segment["n_points"] = n_cells + 1
        segment_start = segment.get("end", segment_start)

    return converted


def _segments_ending_at(
    table: dict[str, Any], end: float, label: str, end_name: str
) -> list[dict[str, Any]]:
    """Copy of a segmented table's segments, with the last one ending at the given end.

    The last segment always ends where the grid does (x_end or height), so it doesn't
    give an end of its own; giving one is an error.
    """

    # check the segment list shape
    segments = table.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"[[{label}.segments]] must define a list of tables")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError(f"[[{label}.segments]] must define a list of tables")

    # check: the last segment's end is implied
    if "end" in segments[-1]:
        raise ValueError(f"[{label}] the last segment ends at {end_name}; remove its end")

    # copy, filling in the last end
    filled = [dict(segment) for segment in segments]
    filled[-1]["end"] = end

    return filled
