"""Tests for the one-dimensional point distributions."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest

from grid_generator.distributions import (
    CubicSegment,
    EquidistantSegment,
    EquidistantSpec,
    GeometricSpec,
    QuinticSegment,
    SegmentedSpec,
    TanhSpec,
    VinokurSpec,
    build_distribution,
)


def check_endpoints_and_count(points: np.ndarray, start: float, end: float, n_points: int) -> None:
    """Check exact end points, the point count, and strictly increasing coordinates."""

    # check
    assert points.shape == (n_points,)
    assert points[0] == start
    assert points[-1] == end
    assert np.all(np.diff(points) > 0.0)


# --------------------------------------------------
# equidistant
# --------------------------------------------------
def test_equidistant_spacing_is_uniform() -> None:
    """Equidistant points have exact end points and one constant spacing."""

    # build
    points = build_distribution(EquidistantSpec(start=-1.0, end=3.0, n_points=9))

    # check
    check_endpoints_and_count(points, -1.0, 3.0, 9)
    np.testing.assert_allclose(np.diff(points), 0.5, rtol=1e-14)


# --------------------------------------------------
# geometric
# --------------------------------------------------
@pytest.mark.parametrize("ratio", [1.05, 0.9])
def test_geometric_ratio_is_constant(ratio: float) -> None:
    """Consecutive spacings grow (or shrink) by exactly the ratio."""

    # build
    points = build_distribution(GeometricSpec(start=0.0, end=2.0, n_points=31, ratio=ratio))
    spacings = np.diff(points)

    # check
    check_endpoints_and_count(points, 0.0, 2.0, 31)
    np.testing.assert_allclose(spacings[1:] / spacings[:-1], ratio, rtol=1e-10)


@pytest.mark.parametrize(
    ("first_spacing", "n_points"),
    [
        # growing: first spacing far below the uniform 0.01
        (1.0e-6, 101),
        # shrinking: first spacing above the uniform 0.05 (moderate, so the last
        # spacings stay large enough to difference accurately)
        (0.08, 21),
    ],
)
def test_geometric_first_spacing_is_honoured(first_spacing: float, n_points: int) -> None:
    """The first spacing matches the request, for growing and shrinking spacing."""

    # build
    points = build_distribution(
        GeometricSpec(start=0.0, end=1.0, n_points=n_points, first_spacing=first_spacing)
    )
    spacings = np.diff(points)

    # check: first spacing, and a constant ratio
    check_endpoints_and_count(points, 0.0, 1.0, n_points)
    assert spacings[0] == pytest.approx(first_spacing, rel=1e-8)
    ratios = spacings[1:] / spacings[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)


def test_geometric_uniform_first_spacing_gives_equidistant() -> None:
    """A first spacing equal to the uniform spacing gives equidistant points."""

    # build
    points = build_distribution(GeometricSpec(start=0.0, end=1.0, n_points=11, first_spacing=0.1))

    # check
    np.testing.assert_allclose(points, np.linspace(0.0, 1.0, 11), rtol=1e-14)


def test_geometric_max_spacing_caps_the_growth() -> None:
    """With max_spacing, spacing grows by one ratio from first_spacing, then stays at the cap."""

    # build: 1 micron at the wall, capped at 0.1 mm, over 5 mm
    spec = GeometricSpec(
        start=0.0, end=5.0e-3, n_points=101, first_spacing=1.0e-6, max_spacing=1.0e-4
    )
    points = build_distribution(spec)
    spacings = np.diff(points)

    # split into the growing part and the capped part
    capped = np.isclose(spacings, 1.0e-4, rtol=1e-9)
    n_growing = int(np.argmax(capped))

    # check: exact ends, first spacing, constant ratio while growing, then a uniform tail
    check_endpoints_and_count(points, 0.0, 5.0e-3, 101)
    assert spacings[0] == pytest.approx(1.0e-6, rel=1e-9)
    ratios = spacings[1:n_growing] / spacings[: n_growing - 1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
    assert np.all(capped[n_growing:])
    assert np.all(spacings <= 1.0e-4 * (1.0 + 1e-9))


# --------------------------------------------------
# tanh
# --------------------------------------------------
@pytest.mark.parametrize("cluster", ["start", "end", "both"])
def test_tanh_clusters_where_requested(cluster: str) -> None:
    """Spacing is smallest at the clustered end(s) and largest away from them."""

    # build
    points = build_distribution(
        TanhSpec(start=0.0, end=1.0, n_points=51, strength=3.0, cluster=cluster)
    )
    spacings = np.diff(points)
    middle = spacings[len(spacings) // 2]

    # check end points, then where the smallest spacing sits
    check_endpoints_and_count(points, 0.0, 1.0, 51)
    if cluster == "start":
        assert np.all(np.diff(spacings) > 0.0)
    elif cluster == "end":
        assert np.all(np.diff(spacings) < 0.0)
    else:
        assert spacings[0] < middle
        assert spacings[-1] < middle
        np.testing.assert_allclose(spacings, spacings[::-1], rtol=1e-10)


# --------------------------------------------------
# vinokur
# --------------------------------------------------
@pytest.mark.parametrize(
    ("start", "end", "n_points", "first_spacing", "end_spacing"),
    [
        # boundary layer at the wall, finer cells again at the end (a shock layer)
        (0.0, 0.1, 161, 1.0e-6, 1.0e-3),
        # the same, mirrored: fine at the end
        (0.0, 0.1, 161, 1.0e-3, 1.0e-6),
        # mild stretching, away from the origin
        (2.0, 3.0, 41, 0.01, 0.02),
        # equal spacings
        (0.0, 1.0, 101, 1.0e-4, 1.0e-4),
        # few points
        (0.0, 1.0, 4, 0.1, 0.3),
        # at the limit: sqrt(first_spacing x end_spacing) is the uniform spacing
        (0.0, 1.0, 51, 0.02**2 / 0.05, 0.05),
    ],
)
def test_vinokur_matches_both_end_spacings(
    start: float, end: float, n_points: int, first_spacing: float, end_spacing: float
) -> None:
    """The first and last cell have the requested sizes, and the cells are finest at the ends."""

    # build
    spec = VinokurSpec(
        start=start,
        end=end,
        n_points=n_points,
        first_spacing=first_spacing,
        end_spacing=end_spacing,
    )
    points = build_distribution(spec)
    spacings = np.diff(points)

    # check: end points and both end spacings
    check_endpoints_and_count(points, start, end, n_points)
    assert spacings[0] == pytest.approx(first_spacing, rel=1e-10)
    assert spacings[-1] == pytest.approx(end_spacing, rel=1e-10)

    # check: no cell inside is smaller than the smaller end cell
    assert spacings.min() >= min(spacings[0], spacings[-1]) * (1.0 - 1e-9)


def test_vinokur_with_equal_spacings_is_tanh_clustered_at_both_ends() -> None:
    """With the same spacing at both ends, Vinokur's function is the symmetric tanh."""

    # tanh clustered at both ends, and vinokur with its end spacing
    tanh_points = build_distribution(
        TanhSpec(start=0.0, end=1.0, n_points=81, strength=2.5, cluster="both")
    )
    spacing = float(tanh_points[1] - tanh_points[0])
    vinokur_points = build_distribution(
        VinokurSpec(start=0.0, end=1.0, n_points=81, first_spacing=spacing, end_spacing=spacing)
    )

    # check
    np.testing.assert_allclose(vinokur_points, tanh_points, rtol=0.0, atol=1e-12)


# --------------------------------------------------
# segmented
# --------------------------------------------------
def test_segmented_linear_quintic_linear() -> None:
    """Linear - quintic - linear joins without duplicate points and with matching spacing."""

    # fine linear region, quintic transition, coarse linear region
    spec = SegmentedSpec(
        start=0.0,
        segments=(
            EquidistantSegment(end=0.1, spacing=0.005),
            QuinticSegment(end=0.4, n_points=31),
            EquidistantSegment(end=1.0, spacing=0.02),
        ),
    )
    points = build_distribution(spec)
    spacings = np.diff(points)

    # check: 21 + 31 + 31 points, joined points counted once
    check_endpoints_and_count(points, 0.0, 1.0, 21 + 31 + 31 - 2)
    np.testing.assert_allclose(spacings[:20], 0.005, rtol=1e-10)
    np.testing.assert_allclose(spacings[-30:], 0.02, rtol=1e-10)

    # spacing changes smoothly into and out of the transition (within a few percent)
    assert spacings[20] == pytest.approx(0.005, rel=0.05)
    assert spacings[-31] == pytest.approx(0.02, rel=0.05)


def test_segmented_terminal_cubic() -> None:
    """A final cubic segment starts with the incoming linear spacing and ends exactly."""

    # linear region followed by a growing cubic
    spec = SegmentedSpec(
        start=0.0,
        segments=(
            EquidistantSegment(end=0.1, n_points=11),
            CubicSegment(end=1.0, n_points=21),
        ),
    )
    points = build_distribution(spec)
    spacings = np.diff(points)

    # check
    check_endpoints_and_count(points, 0.0, 1.0, 11 + 21 - 1)
    assert spacings[10] == pytest.approx(0.01, rel=0.05)
    assert spacings[-1] > spacings[10]


def test_segment_spacing_must_divide_the_length() -> None:
    """An equidistant segment spacing that doesn't divide its length raises."""

    # spacing 0.03 does not divide 0.1
    spec = SegmentedSpec(start=0.0, segments=(EquidistantSegment(end=0.1, spacing=0.03),))

    with pytest.raises(ValueError, match="divisible"):
        build_distribution(spec)


def test_non_monotonic_result_rejected() -> None:
    """A cubic that would fold back (start spacing too large) is rejected by the check."""

    # a coarse linear start can't shrink into a short cubic monotonically
    spec = SegmentedSpec(
        start=0.0,
        segments=(
            EquidistantSegment(end=0.5, n_points=3),
            CubicSegment(end=0.6, n_points=11),
        ),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        build_distribution(spec)


# --------------------------------------------------
# spec validation
# --------------------------------------------------
@pytest.mark.parametrize(
    ("build_spec", "message"),
    [
        (lambda: EquidistantSpec(start=0.0, end=1.0, n_points=1), "n_points"),
        (lambda: EquidistantSpec(start=1.0, end=1.0, n_points=5), "end must be greater"),
        (lambda: GeometricSpec(start=0.0, end=1.0, n_points=5), "exactly one"),
        (
            lambda: GeometricSpec(start=0.0, end=1.0, n_points=5, ratio=1.1, first_spacing=0.1),
            "exactly one",
        ),
        (lambda: GeometricSpec(start=0.0, end=1.0, n_points=5, ratio=0.0), "ratio"),
        (lambda: GeometricSpec(start=0.0, end=1.0, n_points=5, first_spacing=1.0), "first_spacing"),
        (
            lambda: GeometricSpec(start=0.0, end=1.0, n_points=5, ratio=1.1, max_spacing=0.5),
            "needs first_spacing",
        ),
        (
            lambda: GeometricSpec(
                start=0.0, end=1.0, n_points=5, first_spacing=0.1, max_spacing=0.1
            ),
            "larger than first_spacing",
        ),
        (
            lambda: GeometricSpec(
                start=0.0, end=1.0, n_points=11, first_spacing=0.1, max_spacing=0.5
            ),
            "nothing to grow",
        ),
        (
            lambda: GeometricSpec(
                start=0.0, end=1.0, n_points=5, first_spacing=0.01, max_spacing=0.1
            ),
            "too small to reach",
        ),
        (lambda: TanhSpec(start=0.0, end=1.0, n_points=5, strength=0.0), "strength"),
        (
            lambda: TanhSpec(start=0.0, end=1.0, n_points=5, strength=1.0, cluster="middle"),
            "cluster",
        ),
        (
            lambda: VinokurSpec(start=0.0, end=1.0, n_points=3, first_spacing=0.1, end_spacing=0.1),
            "at least 4",
        ),
        (
            lambda: VinokurSpec(
                start=0.0, end=1.0, n_points=11, first_spacing=0.0, end_spacing=0.1
            ),
            "greater than zero",
        ),
        (
            lambda: VinokurSpec(
                start=0.0, end=1.0, n_points=11, first_spacing=0.6, end_spacing=0.5
            ),
            "smaller than the length",
        ),
        # both spacings larger than the uniform spacing 0.1
        (
            lambda: VinokurSpec(
                start=0.0, end=1.0, n_points=11, first_spacing=0.12, end_spacing=0.15
            ),
            "shrink toward the middle",
        ),
        # one finer, one coarser, but their geometric mean above 0.1
        (
            lambda: VinokurSpec(
                start=0.0, end=1.0, n_points=11, first_spacing=0.05, end_spacing=0.3
            ),
            "shrink toward the middle",
        ),
        (lambda: EquidistantSegment(end=1.0), "exactly one"),
        (lambda: EquidistantSegment(end=1.0, n_points=1), "at least 2"),
        (lambda: EquidistantSegment(end=1.0, spacing=-0.1), "spacing"),
        (lambda: SegmentedSpec(start=0.0, segments=()), "at least one"),
        (
            lambda: SegmentedSpec(start=0.0, segments=(CubicSegment(end=1.0, n_points=5),)),
            "begin with an equidistant",
        ),
        (
            lambda: SegmentedSpec(
                start=0.0,
                segments=(
                    EquidistantSegment(end=0.5, n_points=5),
                    CubicSegment(end=0.8, n_points=5),
                    EquidistantSegment(end=1.0, n_points=5),
                ),
            ),
            "final segment",
        ),
        (
            lambda: SegmentedSpec(
                start=0.0,
                segments=(
                    EquidistantSegment(end=0.5, n_points=5),
                    QuinticSegment(end=1.0, n_points=5),
                ),
            ),
            "adjacent equidistant",
        ),
        (
            lambda: SegmentedSpec(
                start=0.0,
                segments=(
                    EquidistantSegment(end=0.5, n_points=5),
                    EquidistantSegment(end=1.0, n_points=5),
                ),
            ),
            "alternate",
        ),
        (
            lambda: SegmentedSpec(
                start=0.0,
                segments=(
                    EquidistantSegment(end=0.5, n_points=5),
                    QuinticSegment(end=0.4, n_points=5),
                    EquidistantSegment(end=1.0, n_points=5),
                ),
            ),
            "strictly increasing",
        ),
    ],
)
def test_invalid_specs_rejected(build_spec, message: str) -> None:
    """Each validation rule in the spec models raises with a message naming the problem."""

    with pytest.raises(ValueError, match=message):
        build_spec()
