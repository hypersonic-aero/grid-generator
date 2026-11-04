"""Two-sided (Vinokur) one-dimensional point distribution.

Vinokur's stretching function puts given spacings at both ends of an interval, with
smoothly varying cells between them.

Reference:
    Vinokur, M., "On one-dimensional stretching functions for finite-difference
    calculations", Journal of Computational Physics 50 (1983), 215-234.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from grid_generator.distributions.models import VinokurSpec

# --------------------------------------------------
# constants
# --------------------------------------------------

# Newton iteration on the two inputs: tolerance on the log of the end spacings,
# iteration limit, and finite-difference step for the Jacobian
_TOLERANCE = 1.0e-13
_MAX_ITERATIONS = 100
_JACOBIAN_STEP = 1.0e-6

# bisection steps for the stretching parameter (enough for double precision)
_BISECTION_STEPS = 100

# B this close to 1 is the unstretched (linear) limit
_LINEAR_LIMIT = 1.0e-12


# --------------------------------------------------
# public API
# --------------------------------------------------
def build_vinokur(spec: VinokurSpec) -> NDArray[np.float64]:
    """Build a two-sided stretched coordinate array with the given end spacings.

    Vinokur's function matches the slopes at both ends, which sets the first and last
    spacing only to first order. Its two inputs are therefore corrected by Newton's
    method until the first and last cells have exactly the requested sizes.

    Args:
        spec: Validated distribution.

    Returns:
        Coordinates from start to end.

    Raises:
        ValueError: If the Newton iteration doesn't converge.
    """

    # extract: the requested end spacings in units of the length, in log scale
    length = spec.end - spec.start
    n_intervals = spec.n_points - 1
    target = np.log([spec.first_spacing / length, spec.end_spacing / length])

    # Newton on the log of the two inputs, starting from the requested spacings
    inputs = target.copy()
    converged = False
    for _ in range(_MAX_ITERATIONS):
        residual = _log_end_spacings(inputs, n_intervals) - target
        if np.max(np.abs(residual)) < _TOLERANCE:
            converged = True
            break

        # Jacobian by finite differences, one input at a time
        jacobian = np.empty((2, 2))
        for k in range(2):
            stepped = inputs.copy()
            stepped[k] += _JACOBIAN_STEP
            stepped_residual = _log_end_spacings(stepped, n_intervals) - target
            jacobian[:, k] = (stepped_residual - residual) / _JACOBIAN_STEP

        # damped step: the inputs change by at most a factor e per iteration
        step = np.linalg.solve(jacobian, -residual)
        largest = float(np.max(np.abs(step)))
        if largest > 1.0:
            step = step / largest
        inputs = inputs + step

    # check
    if not converged:
        raise ValueError(
            f"vinokur: the end spacings didn't converge (first_spacing = "
            f"{spec.first_spacing:.6g}, end_spacing = {spec.end_spacing:.6g}, "
            f"n_points = {spec.n_points})"
        )

    # the stretching s(xi) on [0, 1], and 1 - s
    s, one_minus_s = _stretching(math.exp(inputs[0]), math.exp(inputs[1]), n_intervals)

    # coordinates: the first half measured from start, the second from end, so small
    # cells at either end keep their digits
    half = n_intervals // 2
    coordinates = np.empty(spec.n_points)
    coordinates[: half + 1] = spec.start + length * s[: half + 1]
    coordinates[half + 1 :] = spec.end - length * one_minus_s[half + 1 :]
    coordinates[0] = spec.start
    coordinates[-1] = spec.end

    return coordinates


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _log_end_spacings(inputs: NDArray[np.float64], n_intervals: int) -> NDArray[np.float64]:
    """Log of the first and last cell of the stretching, for log inputs."""

    # build the stretching and measure its end cells
    s, one_minus_s = _stretching(math.exp(inputs[0]), math.exp(inputs[1]), n_intervals)
    first_cell = s[1]
    last_cell = one_minus_s[-2]

    return np.log([first_cell, last_cell])


def _stretching(
    first: float, last: float, n_intervals: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vinokur's two-sided stretching s(xi) and 1 - s(xi), at xi = i / n_intervals.

    first and last are the end slopes ds/dxi divided by n_intervals, i.e. the end
    spacings to first order (in units of the length). The function is built from a
    symmetric stretching u(xi), clustered equally at both ends, and an asymmetric
    map that tilts it toward the end with the smaller spacing:

        A = sqrt(last / first),   B = 1 / (n_intervals sqrt(first last))
        u = 1/2 [1 + tanh(dy (xi - 1/2)) / tanh(dy / 2)],   sinh(dy) / dy = B
        s = u / (A + (1 - A) u)

    B > 1 clusters at the ends (sinh); B < 1 is finer in the middle and uses
    sin(dy) / dy = B with tan for tanh (only reached while Newton corrects the
    inputs); B = 1 is the linear limit u = xi.
    """

    # asymmetry A, and the stretching parameter B of the symmetric part
    xi = np.arange(n_intervals + 1) / n_intervals
    asymmetry = math.sqrt(last / first)
    stretch = 1.0 / (n_intervals * math.sqrt(first * last))

    # symmetric part u(xi), in the form 1/2 sinh(dy xi) / (cosh(dy (xi - 1/2)) sinh(dy / 2)),
    # which equals the tanh form but keeps its digits near xi = 0
    if stretch > 1.0 + _LINEAR_LIMIT:
        dy = _solve_sinh_ratio(stretch)
        u = 0.5 * np.sinh(dy * xi) / (np.cosh(dy * (xi - 0.5)) * math.sinh(0.5 * dy))
    elif stretch < 1.0 - _LINEAR_LIMIT:
        dy = _solve_sin_ratio(stretch)
        u = 0.5 * np.sin(dy * xi) / (np.cos(dy * (xi - 0.5)) * math.sin(0.5 * dy))
    else:
        u = xi.copy()

    # u is symmetric about xi = 1/2, so 1 - u(xi) = u(1 - xi): the reversed array,
    # accurate near xi = 1 too
    v = u[::-1]

    # asymmetric map: s = u / (u + A v), 1 - s = A v / (u + A v)
    denominator = u + asymmetry * v
    s = u / denominator
    one_minus_s = asymmetry * v / denominator

    return s, one_minus_s


def _solve_sinh_ratio(ratio: float) -> float:
    """Solve sinh(x) / x = ratio for x > 0 (ratio > 1), by bisection."""

    # bracket: double the upper bound until it is past the root (sinh overflows past 710)
    lower = 0.0
    upper = 1.0
    while upper < 700.0 and math.sinh(upper) / upper < ratio:
        upper *= 2.0

    # bisect: sinh(x) / x grows with x
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        if math.sinh(middle) / middle < ratio:
            lower = middle
        else:
            upper = middle
    root = 0.5 * (lower + upper)

    return root


def _solve_sin_ratio(ratio: float) -> float:
    """Solve sin(x) / x = ratio for 0 < x < pi (0 < ratio < 1), by bisection."""

    # bisect on (0, pi): sin(x) / x falls from 1 to 0
    lower = 0.0
    upper = math.pi
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        if math.sin(middle) / middle > ratio:
            lower = middle
        else:
            upper = middle
    root = 0.5 * (lower + upper)

    return root
