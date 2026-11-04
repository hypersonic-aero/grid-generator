"""Parse distribution tables from TOML into typed distribution specs.

Shared by the Cartesian axes ([grid.x], [grid.y], [grid.z]), the body-fitted [points]
and [span] tables, and 1D distribution files ([distribution]), so all of them accept
the same distribution types and keys.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from typing import Any

from grid_generator.distributions.models import (
    CubicSegment,
    DistributionSpec,
    EquidistantSegment,
    EquidistantSpec,
    GeometricSpec,
    QuinticSegment,
    SegmentedSpec,
    SegmentSpec,
    TanhSpec,
    VinokurSpec,
)

# --------------------------------------------------
# accepted keys (the reference in distributions.catalog lists the same keys)
# --------------------------------------------------
# keys per distribution type, including the "type" marker
DISTRIBUTION_KEYS: dict[str, set[str]] = {
    "equidistant": {"type", "start", "end", "n_points"},
    "geometric": {"type", "start", "end", "n_points", "ratio", "first_spacing", "max_spacing"},
    "tanh": {"type", "start", "end", "n_points", "strength", "cluster"},
    "vinokur": {"type", "start", "end", "n_points", "first_spacing", "end_spacing"},
    "segmented": {"type", "start", "segments"},
}

# keys per segment type in a segmented table
SEGMENT_KEYS: dict[str, set[str]] = {
    "equidistant": {"type", "end", "n_points", "spacing"},
    "cubic": {"type", "end", "n_points"},
    "quintic": {"type", "end", "n_points"},
}


# --------------------------------------------------
# public API
# --------------------------------------------------
def parse_distribution(value: Any, label: str) -> DistributionSpec:
    """Parse and validate one distribution table (a Cartesian axis or a span).

    Args:
        value: Raw TOML table.
        label: Table name used in error messages, e.g. "grid.x" or "span".

    Returns:
        Validated distribution.

    Raises:
        TypeError: If the table or its segment list has the wrong shape.
        ValueError: If a field is unknown, missing or out of range.
    """

    if not isinstance(value, dict):
        raise TypeError(f"[{label}] must be a table")

    kind = str(value.get("type", "equidistant"))
    if kind == "segmented":
        distribution = _parse_segmented(value, label)
    else:
        distribution = _parse_complete_distribution(value, label, kind)
    return distribution


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _parse_complete_distribution(
    value: dict[str, Any],
    label: str,
    kind: str,
) -> DistributionSpec:
    """Parse an equidistant, geometric, tanh, or vinokur axis."""

    if kind not in DISTRIBUTION_KEYS:
        valid = ", ".join(DISTRIBUTION_KEYS)
        raise ValueError(f"[{label}].type must be one of: {valid}")

    allowed_keys = DISTRIBUTION_KEYS[kind]
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        unknown = ", ".join(unknown_keys)
        raise ValueError(f"unknown [{label}] fields: {unknown}")

    required_keys = {"start", "end", "n_points"}
    missing_keys = sorted(required_keys - set(value))
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"[{label}] is missing fields: {missing}")

    common = {
        "start": float(value["start"]),
        "end": float(value["end"]),
        "n_points": int(value["n_points"]),
    }
    if kind == "equidistant":
        distribution: DistributionSpec = EquidistantSpec(**common)
    elif kind == "geometric":
        ratio = value.get("ratio")
        first_spacing = value.get("first_spacing")
        max_spacing = value.get("max_spacing")
        distribution = GeometricSpec(
            **common,
            ratio=float(ratio) if ratio is not None else None,
            first_spacing=float(first_spacing) if first_spacing is not None else None,
            max_spacing=float(max_spacing) if max_spacing is not None else None,
        )
    elif kind == "vinokur":
        spacing_keys = {"first_spacing", "end_spacing"}
        missing_spacings = sorted(spacing_keys - set(value))
        if missing_spacings:
            raise ValueError(f"[{label}] is missing fields: {', '.join(missing_spacings)}")
        distribution = VinokurSpec(
            **common,
            first_spacing=float(value["first_spacing"]),
            end_spacing=float(value["end_spacing"]),
        )
    else:
        if "strength" not in value:
            raise ValueError(f"[{label}] is missing fields: strength")
        distribution = TanhSpec(
            **common,
            strength=float(value["strength"]),
            cluster=str(value.get("cluster", "start")),
        )
    return distribution


def _parse_segmented(value: dict[str, Any], label: str) -> SegmentedSpec:
    """Parse an ordered segmented distribution."""

    allowed_keys = DISTRIBUTION_KEYS["segmented"]
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        unknown = ", ".join(unknown_keys)
        raise ValueError(f"unknown [{label}] fields: {unknown}")
    if "start" not in value:
        raise ValueError(f"[{label}] is missing fields: start")

    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise TypeError(f"[[{label}.segments]] must define a list of tables")

    segments = tuple(
        _parse_segment(segment, label, index) for index, segment in enumerate(raw_segments)
    )
    distribution = SegmentedSpec(start=float(value["start"]), segments=segments)
    return distribution


def _parse_segment(value: Any, parent_label: str, index: int) -> SegmentSpec:
    """Parse one linear or polynomial segment table."""

    label = f"{parent_label}.segments[{index}]"
    if not isinstance(value, dict):
        raise TypeError(f"[{label}] must be a table")

    kind = value.get("type")
    if kind not in SEGMENT_KEYS:
        raise ValueError(f"[{label}].type must be 'equidistant', 'cubic', or 'quintic'")
    allowed_keys = SEGMENT_KEYS[kind]

    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        unknown = ", ".join(unknown_keys)
        raise ValueError(f"unknown [{label}] fields: {unknown}")
    if "end" not in value:
        raise ValueError(f"[{label}] is missing fields: end")

    end = float(value["end"])
    n_points = value.get("n_points")
    if kind == "equidistant":
        spacing = value.get("spacing")
        segment: SegmentSpec = EquidistantSegment(
            end=end,
            n_points=int(n_points) if n_points is not None else None,
            spacing=float(spacing) if spacing is not None else None,
        )
    elif kind == "cubic":
        if n_points is None:
            raise ValueError(f"[{label}] is missing fields: n_points")
        segment = CubicSegment(end=end, n_points=int(n_points))
    else:
        if n_points is None:
            raise ValueError(f"[{label}] is missing fields: n_points")
        segment = QuinticSegment(end=end, n_points=int(n_points))
    return segment
