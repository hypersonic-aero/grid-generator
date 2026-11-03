"""Configuration contract for constant-height body-fitted 2D grids.

The [geometry] table is canonical-geometries' contract, without the keys that only
say how far the body extends: the grid's body ends at [domain] x_end. For plate,
wedge and cone the loader fills length in from x_end; for the tangent ogive (whose
length is a shape parameter) a cylinder carries the body on when x_end lies past the
base. The config is parsed and validated with pydantic; the generator works from the
validated object.

The [points] tables use the same distribution types and keys as the Cartesian axes,
without start and end: those come from the geometry and [domain]. The body starts at
the nose spacing, so the wall spacing is continuous across the junction.

An optional [span] table extrudes the 2D grid in z into a 3D grid (plate and wedge
only). It takes the same distribution types and keys as the Cartesian axes.

The grid reaches either a constant height from the wall ([domain] height) or a
shock-aligned outer boundary ([outer]: a shock curve and a margin; see
grid_generator.outer).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from canonical_geometries import Profile2D
from canonical_geometries.config import GeometryConfig
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from grid_generator.config.markers import MarkersConfig
from grid_generator.config.points import build_body_spec, build_normal_spec, parse_span
from grid_generator.distributions.generate import build_distribution
from grid_generator.distributions.models import DistributionSpec

# --------------------------------------------------
# constants
# --------------------------------------------------
# Plot3D ASCII extensions name the dimensionality, so they must match the grid
PLOT3D_2D_SUFFIX = ".p2dfmt"
PLOT3D_3D_SUFFIX = ".p3dfmt"

# [geometry] key per type that only says how far the body extends; [domain] x_end
# sets it instead (the tangent ogive's length is a shape parameter and stays)
EXTENT_KEYS = {
    "plate": "length",
    "wedge": "length",
    "cone": "length",
    "tangent-ogive": "cylinder_length",
}

# [markers] keys that only apply to an extruded grid
SPAN_MARKER_KEYS = ("span_min", "span_max")

# axisymmetric bodies: the grid is the meridian plane, and [span] doesn't apply
AXISYMMETRIC_TYPES = {"cone", "tangent-ogive"}


# --------------------------------------------------
# tables
# --------------------------------------------------
class DomainConfig(BaseModel):
    """Extent of the grid: where it ends along x, and how far it reaches from the wall."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    x_end: float = Field(gt=0.0, description="End of the grid along x, from the nose tip [m]")
    height: float | None = Field(
        default=None,
        gt=0.0,
        description="Constant wall-normal height of the grid [m] (or a shock-aligned [outer])",
    )


class OuterConfig(BaseModel):
    """Shock-aligned outer boundary: a shock curve, and how far beyond it the grid reaches.

    Each wall-normal grid line ends at margin_factor * d + min_offset, with d its
    distance to the shock. A relative shock path is resolved against the config
    file's folder.
    """

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    shock: Path = Field(description="Shock curve: CSV file with x, y columns and a header line")
    margin_factor: float = Field(
        default=1.2, ge=1.0, description="Factor on the wall-to-shock distance"
    )
    min_offset: float = Field(
        default=0.0, ge=0.0, description="Distance added beyond the scaled shock distance [m]"
    )


class NosePointsConfig(BaseModel):
    """Points on the nose arc, equally spaced in angle (a polar block around the nose)."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    n_points: int = Field(ge=2, description="Points on the nose arc, including both ends")


class PointsConfig(BaseModel):
    """Point distributions along the wall (nose, body) and away from it.

    body and normal stay raw distribution tables here: their start, end and (for the
    body) first spacing come from the geometry and [domain], so BodyFittedConfig turns
    them into distribution specs (body_distribution, normal_distribution).
    """

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    nose: NosePointsConfig
    body: dict[str, Any]
    normal: dict[str, Any]


class OutputConfig(BaseModel):
    """Grid file; a relative path is resolved against the config file's folder."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    filename: Path = Field(description="Grid file; the extension picks the format via cfd-io")
    order: Literal[1, 2] = Field(
        default=1,
        description="Element order for .msh: 1 (4-node quads) or 2 (curved 9-node quads)",
    )

    @model_validator(mode="after")
    def _check_order_fits_the_format(self) -> OutputConfig:
        """Only Gmsh output has second-order elements."""

        # order 2 means 9-node quads, which only the .msh writer produces
        suffix = self.filename.suffix.lower()
        if self.order == 2 and suffix != ".msh":
            raise ValueError(f"[output] order = 2 needs a .msh filename (Gmsh), got {suffix!r}")

        return self


# --------------------------------------------------
# whole file
# --------------------------------------------------
class BodyFittedConfig(BaseModel):
    """Contents of a body-fitted grid TOML file."""

    # reject unknown sections
    model_config = ConfigDict(extra="forbid")

    geometry: GeometryConfig
    domain: DomainConfig
    points: PointsConfig
    output: OutputConfig
    span: Annotated[DistributionSpec | None, BeforeValidator(parse_span)] = None
    markers: MarkersConfig = Field(default_factory=MarkersConfig)
    outer: OuterConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _end_the_body_at_x_end(cls, data: Any) -> Any:
        """Reject the [geometry] extent key; fill in length from [domain] x_end where it applies.

        The grid's body ends at x_end, so a separate body extent would only be a
        second number to keep in sync. For plate, wedge and cone, length is filled in
        from x_end; the tangent ogive's cylinder is added in build_profile.
        """

        # only raw tables need filling in; missing tables are reported by the fields
        if not isinstance(data, dict):
            return data
        geometry = data.get("geometry")
        domain = data.get("domain")
        if not isinstance(geometry, dict) or not isinstance(domain, dict):
            return data

        # check: no extent of its own (unknown types are reported by [geometry])
        extent_key = EXTENT_KEYS.get(geometry.get("type"))
        if extent_key is not None and extent_key in geometry:
            raise ValueError(
                f"[geometry] {extent_key} isn't used: the grid's body ends at [domain] x_end; "
                f"remove {extent_key}"
            )

        # fill in length where it is only the extent (a missing x_end is reported by [domain])
        filled = data
        if extent_key == "length" and "x_end" in domain:
            filled = dict(data)
            filled["geometry"] = {**geometry, "length": domain["x_end"]}

        return filled

    @model_validator(mode="after")
    def _check_one_outer_boundary(self) -> BodyFittedConfig:
        """The grid lines end either at [domain] height or at the shock-aligned [outer]."""

        # check: exactly one of the two
        has_height = self.domain.height is not None
        has_outer = self.outer is not None
        if has_height and has_outer:
            raise ValueError(
                "give either [domain] height (constant height) or an [outer] table "
                "(shock-aligned), not both"
            )
        if not has_height and not has_outer:
            raise ValueError(
                "the grid needs an outer boundary: [domain] height (constant height) or an "
                "[outer] table (shock-aligned)"
            )

        return self

    @model_validator(mode="after")
    def _check_domain_fits_the_geometry(self) -> BodyFittedConfig:
        """Check x_end lies on the body, and the point distributions fit the geometry."""

        # x_end must reach past the nose (building a plate, wedge or cone profile checks this)
        x_end = self.domain.x_end
        try:
            profile = self.build_profile()
        except ValueError as error:
            raise ValueError(
                f"[domain] x_end ({x_end}) must reach past the nose: {error}"
            ) from None

        # same check for every body, against the end of the nose
        s_junction = float(profile.junctions[0])
        x_junction = float(profile.point(s_junction)[0])
        if x_end <= x_junction:
            raise ValueError(
                f"[domain] x_end ({x_end}) must reach past the nose (x = {x_junction})"
            )

        # build the body and normal distributions (raises labelled errors for bad tables)
        point_counts = {"points.nose": self.points.nose.n_points}
        # (shock-aligned grids get their heights from the shock; check the table at x_end)
        nominal_height = self.domain.height if self.domain.height is not None else x_end
        specs = {
            "points.body": self.body_distribution(),
            "points.normal": self.normal_distribution(nominal_height),
        }
        for label, spec in specs.items():
            try:
                coordinates = build_distribution(spec)
            except ValueError as error:
                raise ValueError(f"[{label}] {error}") from None
            point_counts[label] = coordinates.size

        # second-order elements span 2 x 2 grid cells: odd point counts, and an odd
        # n_nose puts the nose/body junction on an element corner, not inside an element
        if self.output.order == 2:
            even_counts = []
            for label, count in point_counts.items():
                if count % 2 == 0:
                    even_counts.append(f"[{label}] {count} points")
            if even_counts:
                raise ValueError(
                    f"[output] order = 2 needs odd point counts; even: {', '.join(even_counts)}"
                )

        return self

    @model_validator(mode="after")
    def _check_span_fits_the_geometry_and_format(self) -> BodyFittedConfig:
        """Check [span] is only used on planar bodies and with a format that holds 3D grids."""

        # check: axisymmetric bodies get their 3D grid from revolving, not extruding
        geometry_type = self.geometry.type
        if self.span is not None and geometry_type in AXISYMMETRIC_TYPES:
            raise ValueError(
                f"[span] extrudes planar bodies (plate, wedge) in z; a {geometry_type} is "
                "axisymmetric and its 3D grid would be revolved, which isn't supported yet"
            )

        # check: span marker names only with a span to name
        span_names_given = [key for key in SPAN_MARKER_KEYS if key in self.markers.model_fields_set]
        if self.span is None and span_names_given:
            raise ValueError(
                f"[markers] {', '.join(span_names_given)} only apply with a [span] table"
            )

        # check: the Gmsh writer is 2D only
        suffix = self.output.filename.suffix.lower()
        if self.span is not None and suffix == ".msh":
            raise ValueError(
                "[span] makes a 3D grid, but .msh (Gmsh) output is 2D only; "
                "use .su2, .h5 or .p3dfmt"
            )

        # check: the Plot3D ASCII extension matches the grid's dimensionality
        if self.span is not None and suffix == PLOT3D_2D_SUFFIX:
            raise ValueError(
                f"[span] makes a 3D grid; use {PLOT3D_3D_SUFFIX} instead of {PLOT3D_2D_SUFFIX}"
            )
        if self.span is None and suffix == PLOT3D_3D_SUFFIX:
            raise ValueError(
                f"without [span] the grid is 2D; use {PLOT3D_2D_SUFFIX} instead of "
                f"{PLOT3D_3D_SUFFIX}"
            )

        return self

    def build_profile(self) -> Profile2D:
        """Build the wall profile, from the stagnation point to (at least) x_end.

        Plate, wedge and cone end at x_end (their length is x_end). The tangent ogive
        ends at its base, or, when x_end lies past the base, at x_end on a cylinder;
        when x_end lies on the ogive, the grid stops there (body_distribution).
        """

        # build the body as given
        profile = self.geometry.build()

        # tangent ogive: a cylinder carries the body on to an x_end past the base
        if self.geometry.type == "tangent-ogive":
            x_base, _ = profile.base_point
            if self.domain.x_end > x_base:
                cylinder_length = self.domain.x_end - x_base
                extended = self.geometry.model_copy(update={"cylinder_length": cylinder_length})
                profile = extended.build()

        return profile

    def body_distribution(self) -> DistributionSpec:
        """Body points in wall arc length, from the junction to x_end, starting at the nose spacing.

        Returns:
            Geometric or segmented distribution spec in wall arc length [m].

        Raises:
            ValueError: If the [points.body] table is invalid or its type can't start
                from a given spacing.
        """

        # extract
        profile = self.build_profile()
        n_nose = self.points.nose.n_points
        body_spec = build_body_spec(self.points.body, profile, n_nose, self.domain.x_end)

        return body_spec

    def normal_distribution(self, height: float | None = None) -> DistributionSpec:
        """Wall-normal points, from the wall (0) to a height.

        Args:
            height: Height of this grid line [m]; [domain] height when omitted
                (shock-aligned grids pass each line's own height).

        Returns:
            Distribution spec in wall distance [m].

        Raises:
            ValueError: If the [points.normal] table is invalid, or no height is known.
        """

        # default: the constant [domain] height
        if height is None:
            height = self.domain.height
        if height is None:
            raise ValueError("a shock-aligned grid gives each line its own height")

        # extract
        normal_spec = build_normal_spec(self.points.normal, height)

        return normal_spec


# --------------------------------------------------
# public API
# --------------------------------------------------
def load_body_fitted_config(path: str | Path, expected_type: str | None = None) -> BodyFittedConfig:
    """Read and validate a body-fitted grid TOML file.

    A relative output filename is resolved against the folder holding the config file.

    Args:
        path: Configuration file.
        expected_type: If given, the [geometry] type the file must describe
            (e.g. "cone" when called from `grid-generator cone run`).

    Returns:
        Validated configuration.

    Raises:
        OSError: If the file can't be read.
        tomllib.TOMLDecodeError: If the file isn't valid TOML.
        pydantic.ValidationError: If a value is missing, misspelled or out of range.
        ValueError: If the file describes a different geometry type than expected.
    """

    # convert to Path object
    path = Path(path)

    # read
    with path.open("rb") as stream:
        raw_dict = tomllib.load(stream)

    # validate
    config = BodyFittedConfig.model_validate(raw_dict)

    # check the file matches the command it was given to
    geometry_type = config.geometry.type
    if expected_type is not None and geometry_type != expected_type:
        raise ValueError(
            f"{path} describes a '{geometry_type}' geometry, not '{expected_type}'; "
            f"run it with: grid-generator {geometry_type} run {path}"
        )

    # resolve the output path relative to the config file
    output_file = path.parent / config.output.filename
    resolved_output = config.output.model_copy(update={"filename": output_file})
    resolved_config = config.model_copy(update={"output": resolved_output})

    # resolve the shock curve path the same way
    if resolved_config.outer is not None:
        shock_file = path.parent / resolved_config.outer.shock
        resolved_outer = resolved_config.outer.model_copy(update={"shock": shock_file})
        resolved_config = resolved_config.model_copy(update={"outer": resolved_outer})

    return resolved_config
