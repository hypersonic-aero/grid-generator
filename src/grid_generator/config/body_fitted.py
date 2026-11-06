"""Configuration contract for body-fitted grids: constant height or shock-aligned, 2D or swept to 3D.

The [geometry] table is canonical-geometries' contract, without the keys that only
say how far the body extends: the grid's body ends at [domain] x_end. For plate,
wedge and cone the loader fills length in from x_end; for the tangent ogive (whose
length is a shape parameter) a cylinder carries the body on when x_end lies past the
base. The config is parsed and validated with pydantic; the generator works from the
validated object.

The [points] tables use the same distribution types and keys as the Cartesian axes,
without start and end: those come from the geometry and [domain]. The body starts at
the nose spacing, so the wall spacing is continuous across the junction.

An optional [sweep] table makes a 3D grid from the 2D one: revolved about the x-axis
into a sector (cone, tangent ogive), or extruded along z (plate, wedge); see
grid_generator.sweep.

The grid reaches either a constant height from the wall ([domain] height, with the
wall-normal points in [points.normal]), or it is a shock-aligned grid ([shock]: a shock
curve that becomes a grid line, and a margin beyond it). A shock-aligned grid's lines
run from the wall to the shock ([points.shock_layer]) and on to the outer boundary
([points.freestream]); see grid_generator.shock.
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
    ConfigDict,
    Field,
    model_validator,
)

from grid_generator.config.markers import MarkersConfig
from grid_generator.config.points import (
    build_body_spec,
    build_freestream_spec,
    build_normal_spec,
    build_shock_layer_spec,
)
from grid_generator.distributions.generate import build_distribution
from grid_generator.distributions.models import DistributionSpec, GeometricSpec, VinokurSpec

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

# [markers] keys that only apply to a swept (3D) grid
SWEEP_MARKER_KEYS = ("sweep_low", "sweep_high")

# axisymmetric bodies: the grid is the meridian plane; [sweep] revolves it (the
# others, planar, are extruded)
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
        description="Constant wall-normal height of the grid [m] (or a shock line: [shock])",
    )


class ShockConfig(BaseModel):
    """Shock-aligned grid: a shock curve that becomes the grid line j = j_s, and a margin beyond it.

    The outer boundary lies margin_factor * d + min_offset from the wall, with d the
    wall-to-shock distance. A relative curve path is resolved against the config
    file's folder.
    """

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    curve: Path = Field(description="Shock curve: CSV file with x, y columns and a header line")
    margin_factor: float = Field(
        default=1.5, ge=1.0, description="Factor on the wall-to-shock distance"
    )
    min_offset: float = Field(
        default=0.0, ge=0.0, description="Distance added beyond the scaled shock distance [m]"
    )

    @model_validator(mode="after")
    def _check_outer_boundary_lies_beyond_the_shock(self) -> ShockConfig:
        """The lines need room beyond the shock line to reach the outer boundary."""

        # check: margin_factor 1 without an offset puts the outer boundary on the shock
        if self.margin_factor == 1.0 and self.min_offset == 0.0:
            raise ValueError(
                "[shock] margin_factor = 1 with min_offset = 0 puts the outer boundary on "
                "the shock; raise one of them"
            )

        return self


class RevolveConfig(BaseModel):
    """[sweep] type = "revolve": the meridian plane turned about the x-axis into a sector.

    The sector runs from the plane z = 0 (the 2D grid) toward +z, in equal angle steps.
    """

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    type: Literal["revolve"] = Field(description="Revolve about the x-axis (cone, tangent ogive)")
    angle_deg: float = Field(gt=0.0, lt=360.0, description="Sector angle [deg]")
    n_points: int = Field(ge=2, description="Points in the angle, both side faces included")


class ExtrudeConfig(BaseModel):
    """[sweep] type = "extrude": the plane repeated along z, from 0 to width, equally spaced."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    type: Literal["extrude"] = Field(description="Extrude along z (plate, wedge)")
    width: float = Field(gt=0.0, description="Span in z [m]")
    n_points: int = Field(ge=2, description="Points in z, both side faces included")


class NosePointsConfig(BaseModel):
    """Points on the nose arc, equally spaced in angle (a polar block around the nose)."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    n_points: int = Field(ge=2, description="Points on the nose arc, including both ends")


class ShockLayerPointsConfig(BaseModel):
    """Points from the wall to the shock line (shock-aligned grids): two-sided stretching.

    The wall spacing is a length (the boundary layer needs a physical size); the
    spacing at the shock is a fraction of each line's wall-to-shock distance (the
    shock layer's size). On lines too short for both (near a small nose) the spacing
    at the shock is lowered, so the wall keeps its spacing; only a line shorter than
    (n_points - 1) x first_spacing gets uniform cells, finer than first_spacing.
    """

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    type: Literal["vinokur"] = Field(
        default="vinokur", description="Distribution type: a given spacing at both ends"
    )
    n_points: int = Field(ge=4, description="Points from the wall to the shock, both included")
    first_spacing: float = Field(gt=0.0, description="Spacing at the wall [m]")
    end_spacing_fraction: float = Field(
        gt=0.0,
        lt=0.5,
        description="Spacing at the shock, as a fraction of the wall-to-shock distance",
    )


class FreestreamPointsConfig(BaseModel):
    """Points from the shock line to the outer boundary (shock-aligned grids): geometric growth.

    The first spacing is the shock layer's last one (automatic), so the spacing is
    continuous across the shock line.
    """

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    type: Literal["geometric"] = Field(
        default="geometric", description="Distribution type: growth from the spacing at the shock"
    )
    n_points: int = Field(
        ge=3, description="Points from the shock to the outer boundary, both included"
    )


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
    normal: dict[str, Any] | None = None
    shock_layer: ShockLayerPointsConfig | None = None
    freestream: FreestreamPointsConfig | None = None


class OutputConfig(BaseModel):
    """Grid file; a relative path is resolved against the config file's folder."""

    # reject misspelled keys
    model_config = ConfigDict(extra="forbid")

    filename: Path = Field(description="Grid file; the extension picks the format via cfd-io")
    order: Literal[1, 2] = Field(
        default=1,
        description=(
            "Element order for .msh: 1 (straight-edged elements) or 2 (curved 9-node quads, "
            "27-node hexahedra)"
        ),
    )
    scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Factor on the coordinates in the file (1000: millimetres)",
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
    sweep: Annotated[RevolveConfig | ExtrudeConfig, Field(discriminator="type")] | None = None
    markers: MarkersConfig = Field(default_factory=MarkersConfig)
    shock: ShockConfig | None = None

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
    def _check_one_layout(self) -> BodyFittedConfig:
        """Constant height ([domain] height, [points.normal]) or a shock line ([shock], ...).

        A shock-aligned grid takes [points.shock_layer] (wall to shock) and
        [points.freestream] (shock to outer boundary) instead of [points.normal].
        """

        # check: exactly one outer boundary
        has_height = self.domain.height is not None
        has_shock = self.shock is not None
        if has_height and has_shock:
            raise ValueError(
                "give either [domain] height (constant height) or a [shock] table "
                "(shock line), not both"
            )
        if not has_height and not has_shock:
            raise ValueError(
                "the grid needs an outer boundary: [domain] height (constant height) or a "
                "[shock] table (shock line)"
            )

        # check: the wall-normal tables of that layout, and only those
        points = self.points
        if has_shock:
            if points.normal is not None:
                raise ValueError(
                    "[points.normal] is for constant-height grids; a shock-aligned grid "
                    "([shock]) takes [points.shock_layer] (wall to shock) and "
                    "[points.freestream] (shock to outer boundary)"
                )
            missing = []
            if points.shock_layer is None:
                missing.append("[points.shock_layer] (wall to shock)")
            if points.freestream is None:
                missing.append("[points.freestream] (shock to outer boundary)")
            if missing:
                raise ValueError(f"a shock-aligned grid ([shock]) needs {' and '.join(missing)}")
        else:
            shock_tables = []
            if points.shock_layer is not None:
                shock_tables.append("[points.shock_layer]")
            if points.freestream is not None:
                shock_tables.append("[points.freestream]")
            if shock_tables:
                raise ValueError(
                    f"{' and '.join(shock_tables)} are for shock-aligned grids ([shock]); a "
                    "constant-height grid ([domain] height) takes [points.normal]"
                )
            if points.normal is None:
                raise ValueError("a constant-height grid ([domain] height) needs [points.normal]")

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
        specs = {"points.body": self.body_distribution()}
        if self.shock is None:
            specs["points.normal"] = self.normal_distribution()
        else:
            # (shock-aligned grids get their line lengths from the shock; the counts are fixed)
            point_counts["points.shock_layer"] = self.points.shock_layer.n_points
            point_counts["points.freestream"] = self.points.freestream.n_points
        if self.sweep is not None:
            point_counts["sweep"] = self.sweep.n_points
        for label, spec in specs.items():
            try:
                coordinates = build_distribution(spec)
            except ValueError as error:
                raise ValueError(f"[{label}] {error}") from None
            point_counts[label] = coordinates.size

        # second-order elements span 2 x 2 grid cells: odd point counts, and an odd
        # n_nose puts the nose/body junction on an element corner, not inside an element
        # (an odd [points.shock_layer] count does the same for the shock line)
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
    def _check_sweep_fits_the_geometry_and_format(self) -> BodyFittedConfig:
        """Check [sweep] suits the body (revolve or extrude) and the file holds the grid."""

        # check: axisymmetric bodies are revolved, planar ones extruded
        geometry_type = self.geometry.type
        sweep = self.sweep
        axisymmetric = geometry_type in AXISYMMETRIC_TYPES
        if sweep is not None and axisymmetric and sweep.type != "revolve":
            raise ValueError(
                f"a {geometry_type} is axisymmetric: its 3D grid is revolved, [sweep] "
                f'type = "revolve"'
            )
        if sweep is not None and not axisymmetric and sweep.type != "extrude":
            raise ValueError(
                f'a {geometry_type} is planar: its 3D grid is extruded, [sweep] type = "extrude"'
            )

        # check: side face names only with a sweep to name
        sweep_names_given = [
            key for key in SWEEP_MARKER_KEYS if key in self.markers.model_fields_set
        ]
        if sweep is None and sweep_names_given:
            raise ValueError(
                f"[markers] {', '.join(sweep_names_given)} only apply with a [sweep] table"
            )

        # check: the Plot3D ASCII extension matches the grid's dimensionality
        suffix = self.output.filename.suffix.lower()
        if sweep is not None and suffix == PLOT3D_2D_SUFFIX:
            raise ValueError(
                f"[sweep] makes a 3D grid; use {PLOT3D_3D_SUFFIX} instead of {PLOT3D_2D_SUFFIX}"
            )
        if sweep is None and suffix == PLOT3D_3D_SUFFIX:
            raise ValueError(
                f"without [sweep] the grid is 2D; use {PLOT3D_2D_SUFFIX} instead of "
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

    def normal_distribution(self) -> DistributionSpec:
        """Wall-normal points of a constant-height grid, from the wall (0) to [domain] height.

        Returns:
            Distribution spec in wall distance [m].

        Raises:
            ValueError: If the [points.normal] table is invalid, or the grid isn't a
                constant-height one.
        """

        # validate: only constant-height grids have [points.normal]
        height = self.domain.height
        if height is None or self.points.normal is None:
            raise ValueError("[points.normal] belongs to constant-height grids ([domain] height)")

        # extract
        normal_spec = build_normal_spec(self.points.normal, height)

        return normal_spec

    def shock_layer_distribution(self, height: float) -> VinokurSpec:
        """Points of one shock-aligned grid line from the wall (0) to the shock (height).

        Args:
            height: Wall-to-shock distance of this line [m].

        Returns:
            Two-sided distribution in wall distance [m]. On lines too short for both
            end spacings its end_spacing is below end_spacing_fraction x height; its
            first_spacing is below [points.shock_layer] first_spacing only on lines
            shorter than (n_points - 1) x first_spacing.

        Raises:
            ValueError: If the grid isn't a shock-aligned one, or the line can't hold the
                two end cells.
        """

        # validate: only shock-aligned grids have [points.shock_layer]
        table = self.points.shock_layer
        if table is None:
            raise ValueError("[points.shock_layer] belongs to shock-aligned grids ([shock])")

        # extract
        shock_layer_spec = build_shock_layer_spec(
            table.n_points, table.first_spacing, table.end_spacing_fraction, height
        )

        return shock_layer_spec

    def freestream_distribution(self, length: float, first_spacing: float) -> GeometricSpec:
        """Points of one shock-aligned grid line from the shock (0) to the outer boundary (length).

        Args:
            length: Distance from the shock to the outer boundary along the line [m].
            first_spacing: Spacing at the shock (the shock layer's last cell) [m].

        Returns:
            Geometric distribution in distance from the shock [m].

        Raises:
            ValueError: If the grid isn't a shock-aligned one, or the cells would shrink
                toward the outer boundary.
        """

        # validate: only shock-aligned grids have [points.freestream]
        table = self.points.freestream
        if table is None:
            raise ValueError("[points.freestream] belongs to shock-aligned grids ([shock])")

        # extract
        freestream_spec = build_freestream_spec(table.n_points, length, first_spacing)

        return freestream_spec


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
            f"run it with: grid-generator {geometry_type} run -c {path}"
        )

    # resolve the output path relative to the config file
    output_file = path.parent / config.output.filename
    resolved_output = config.output.model_copy(update={"filename": output_file})
    resolved_config = config.model_copy(update={"output": resolved_output})

    # resolve the shock curve path the same way
    if resolved_config.shock is not None:
        shock_file = path.parent / resolved_config.shock.curve
        resolved_shock = resolved_config.shock.model_copy(update={"curve": shock_file})
        resolved_config = resolved_config.model_copy(update={"shock": resolved_shock})

    return resolved_config
