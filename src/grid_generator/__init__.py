"""Public package interface for grid_generator."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from importlib.metadata import (
    PackageNotFoundError,
    version,
)

from grid_generator.body_fitted import generate_body_fitted
from grid_generator.cartesian import generate_cartesian
from grid_generator.config import (
    BodyFittedConfig,
    CartesianConfig,
    load_body_fitted_config,
    load_cartesian_config,
)
from grid_generator.distributions import (
    CubicSegment,
    DistributionSpec,
    EquidistantSegment,
    EquidistantSpec,
    GeometricSpec,
    QuinticSegment,
    SegmentedSpec,
    TanhSpec,
    VinokurSpec,
    build_distribution,
)
from grid_generator.quality import GridQuality, ShockLineQuality, grid_quality, shock_line_quality

# --------------------------------------------------
# resolve package version
# --------------------------------------------------
try:
    __version__ = version("grid-generator")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = [
    "BodyFittedConfig",
    "CubicSegment",
    "CartesianConfig",
    "DistributionSpec",
    "EquidistantSegment",
    "EquidistantSpec",
    "GeometricSpec",
    "GridQuality",
    "QuinticSegment",
    "SegmentedSpec",
    "ShockLineQuality",
    "TanhSpec",
    "VinokurSpec",
    "__version__",
    "build_distribution",
    "generate_body_fitted",
    "generate_cartesian",
    "grid_quality",
    "load_body_fitted_config",
    "load_cartesian_config",
    "shock_line_quality",
]
