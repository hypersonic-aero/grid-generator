"""Dispatch and common validation for one-dimensional distributions."""

import numpy as np
from numpy.typing import NDArray

from grid_generator.distributions.equidistant import build_equidistant
from grid_generator.distributions.geometric import build_geometric
from grid_generator.distributions.models import (
    DistributionSpec,
    EquidistantSpec,
    GeometricSpec,
    SegmentedSpec,
    TanhSpec,
    VinokurSpec,
)
from grid_generator.distributions.segmented import build_segmented
from grid_generator.distributions.tanh import build_tanh
from grid_generator.distributions.vinokur import build_vinokur


def build_distribution(spec: DistributionSpec) -> NDArray[np.float64]:
    """Build and validate one configured point distribution."""

    if isinstance(spec, EquidistantSpec):
        coordinates = build_equidistant(spec)
    elif isinstance(spec, GeometricSpec):
        coordinates = build_geometric(spec)
    elif isinstance(spec, TanhSpec):
        coordinates = build_tanh(spec)
    elif isinstance(spec, VinokurSpec):
        coordinates = build_vinokur(spec)
    elif isinstance(spec, SegmentedSpec):
        coordinates = build_segmented(spec)
    else:
        raise TypeError(f"unsupported distribution spec: {type(spec).__name__}")

    if coordinates.ndim != 1:
        raise ValueError("distribution coordinates must be one-dimensional")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("distribution coordinates must contain only finite values")
    if not np.all(np.diff(coordinates) > 0.0):
        raise ValueError("distribution coordinates must be strictly increasing")
    return coordinates
