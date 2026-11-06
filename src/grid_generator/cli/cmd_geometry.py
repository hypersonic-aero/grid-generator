"""Per-geometry command groups: `grid-generator <type> init` and `grid-generator <type> run`."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Annotated

import typer

from grid_generator import __version__
from grid_generator.body_fitted import generate_body_fitted
from grid_generator.cli.examples_command import build_examples_command, list_examples
from grid_generator.cli.help_text import format_examples
from grid_generator.config import (
    BODY_BLOCKS,
    NORMAL_BLOCKS,
    body_fitted_template,
    load_body_fitted_config,
)
from grid_generator.config.body_fitted import AXISYMMETRIC_TYPES
from grid_generator.config.templates import DEFAULT_BODY, DEFAULT_NORMAL
from grid_generator.quality import grid_quality, shock_line_quality, sweep_quality
from grid_generator.sweep import sweep_transform

# --------------------------------------------------
# registry: one command group per geometry type
# --------------------------------------------------
# keys must match BODY_FITTED_GEOMETRY_BLOCKS (and the [geometry] type values)
GEOMETRY_HELP: dict[str, str] = {
    "plate": "Body-fitted grid around a flat plate with a circular leading edge.",
    "wedge": "Body-fitted grid around a planar blunted wedge.",
    "cone": "Body-fitted grid of a sphere-cone (axisymmetric).",
    "tangent-ogive": "Body-fitted grid of a spherically blunted tangent ogive (axisymmetric).",
}


# --------------------------------------------------
# command group factory
# --------------------------------------------------
def build_geometry_app(geometry_type: str) -> typer.Typer:
    """Build the `init` / `run` command group for one geometry type.

    Every geometry gets the same two commands; only the template that `init` writes
    and the type `run` accepts differ. The commands below close over geometry_type.

    Args:
        geometry_type: Key in GEOMETRY_HELP, e.g. "plate".

    Returns:
        Typer app to register as `grid-generator <geometry_type>`.
    """

    # examples shown under the group's help (--sweep revolves or extrudes, by body)
    planar = geometry_type not in AXISYMMETRIC_TYPES
    sweep_text = "extruded along z (3D)" if planar else "revolved into a sector (3D)"
    command = f"grid-generator {geometry_type}"
    examples = [
        (f"{command} init", "defaults"),
        (f"{command} init --body segmented", "segmented body"),
        (f"{command} init --normal tanh", "tanh wall-normal"),
        (f"{command} init --shock", "shock-line grid (needs a shock curve)"),
        (f"{command} init --sweep", sweep_text),
        (f"{command} run -c {geometry_type}.toml", "generate the grid"),
    ]

    # command group
    geometry_app = typer.Typer(
        name=geometry_type,
        help=GEOMETRY_HELP[geometry_type],
        epilog=format_examples(examples),
        no_args_is_help=True,
    )
    default_config = Path(f"{geometry_type}.toml")

    # init summary names its options, so they show in the command list
    init_help = "Create a starter configuration (--body, --normal, --shock, --sweep)."

    # init: write the starter configuration for this geometry
    @geometry_app.command("init", help=init_help)
    def cmd_init(
        output: Annotated[
            Path,
            typer.Option("--output", "-o", help="Configuration file to create."),
        ] = default_config,
        force: Annotated[
            bool,
            typer.Option("--force", "-f", help="Overwrite an existing configuration."),
        ] = False,
        body: Annotated[
            str,
            typer.Option(help=f"Body distribution along the wall: {', '.join(BODY_BLOCKS)}."),
        ] = DEFAULT_BODY,
        normal: Annotated[
            str | None,
            typer.Option(
                help=(
                    f"Wall-normal distribution: {', '.join(NORMAL_BLOCKS)} "
                    f"(default {DEFAULT_NORMAL}; constant-height grids)."
                )
            ),
        ] = None,
        sweep: Annotated[
            bool,
            typer.Option(
                "--sweep",
                # the backslash keeps the help formatter from reading [sweep] as markup
                help=(
                    "Add a \\[sweep] table: 3D grid, revolved into a sector (cone, "
                    "tangent-ogive) or extruded along z (plate, wedge)."
                ),
            ),
        ] = False,
        shock: Annotated[
            bool,
            typer.Option(
                "--shock",
                help="Shock-line grid: the shock (a curve file) is a grid line.",
            ),
        ] = False,
    ) -> None:
        """Create a starter configuration."""

        # check the options before touching any file
        if body not in BODY_BLOCKS:
            typer.echo(f"error: --body must be one of: {', '.join(BODY_BLOCKS)}", err=True)
            raise typer.Exit(code=1)
        if normal is not None and normal not in NORMAL_BLOCKS:
            typer.echo(f"error: --normal must be one of: {', '.join(NORMAL_BLOCKS)}", err=True)
            raise typer.Exit(code=1)
        if shock and normal is not None:
            typer.echo(
                "error: --normal picks [points.normal], which a shock-line grid (--shock) "
                "doesn't have",
                err=True,
            )
            raise typer.Exit(code=1)

        # protect existing user configuration unless overwrite was requested
        if output.exists() and not force:
            typer.echo(f"error: {output} already exists. Use --force to overwrite.", err=True)
            raise typer.Exit(code=1)

        # write and validate the starter configuration
        template = body_fitted_template(
            geometry_type, body=body, normal=normal, sweep=sweep, shock=shock
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(template, encoding="utf-8")
        load_body_fitted_config(output, expected_type=geometry_type)

        typer.echo(f"Created: {output}")
        if shock:
            typer.echo("Put the shock curve next to it (see [shock] curve).")
        typer.echo(f"Edit the file, then run: grid-generator {geometry_type} run -c {output}")

    # run: generate the grid from a configuration, report its quality and write it
    @geometry_app.command("run")
    def cmd_run(
        config: Annotated[
            Path,
            typer.Option("--config", "-c", help="Configuration file."),
        ] = default_config,
    ) -> None:
        """Generate the grid from a configuration."""

        try:
            from cfd_io import Dataset, StructuredGrid, write_file
            from cfd_io.writers.gmsh import write_gmsh

            # read and validate (pydantic's ValidationError is a ValueError)
            body_fitted_config = load_body_fitted_config(config, expected_type=geometry_type)

            # generate
            dataset = generate_body_fitted(body_fitted_config)

            # measure the plane k = 0 (a swept grid's first plane is the 2D grid)
            grid = dataset.grid
            plane = StructuredGrid(x=grid.x[:, :, :1], y=grid.y[:, :, :1], z=grid.z[:, :, :1])
            quality = grid_quality(plane)

            # shock-line grids: where the shock line sits, and the spacing around it
            shock_line_j = dataset.attrs.get("shock_line_j")
            shock_quality = None
            if shock_line_j is not None:
                shock_layer = body_fitted_config.points.shock_layer
                shock_quality = shock_line_quality(
                    plane,
                    shock_line_j,
                    shock_layer.first_spacing,
                    shock_layer.end_spacing_fraction,
                )

            # swept grids: the cells in 3D, the axis, and the side faces
            scale = body_fitted_config.output.scale
            sweep_report = None
            transform = None
            markers = dataset.attrs["markers"]["structured"]
            if body_fitted_config.sweep is not None:
                transform = sweep_transform(body_fitted_config.sweep)
                sweep_report = sweep_quality(
                    grid, transform, markers["kmin"], markers["kmax"], scale
                )

            # coordinates in the file's units ([output] scale; the report stays in metres)
            written = dataset
            if scale != 1.0:
                scaled_grid = StructuredGrid(x=scale * grid.x, y=scale * grid.y, z=scale * grid.z)
                scaled_attrs = dict(dataset.attrs, coordinate_scale=scale)
                if transform is not None:
                    scaled_attrs["sweep"] = transform.as_dict(
                        markers["kmin"], markers["kmax"], scale
                    )
                written = Dataset(grid=scaled_grid, attrs=scaled_attrs)

            # write: Gmsh gets the element order, which the file extension can't carry;
            # everything else by extension, in double precision (cfd-io's HDF5 default is float32)
            output_file = body_fitted_config.output.filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.suffix.lower() == ".msh":
                output_path = write_gmsh(
                    output_file, written, order=body_fitted_config.output.order
                )
            else:
                output_path = write_file(output_file, written, dtype="f8")

            # every grid: its description as JSON next to it, for formats that can't carry
            # it (shock line j_s, side-face transform and solver lines, markers, scale)
            metadata = {
                "grid_file": Path(output_path).name,
                "grid_generator_version": __version__,
            }
            metadata.update(written.attrs)
            metadata["coordinate_scale"] = scale
            metadata_path = output_file.with_suffix(".grid.json")
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        except (ImportError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(code=1) from None

        # report: sizes, then the plane's quality
        ni, nj, nk = dataset.grid.shape
        if nk > 1:
            points = f"{ni} x {nj} x {nk} points"
            cells = f"{ni - 1} x {nj - 1} x {nk - 1} cells"
        else:
            points = f"{ni} x {nj} points"
            cells = f"{ni - 1} x {nj - 1} cells"
        typer.echo(f"Generated {geometry_type} grid: {points} ({cells})")
        if nk > 1:
            typer.echo("  quality of the plane k = 0 (the 2D grid):")
        typer.echo(quality.summary())
        if shock_quality is not None:
            typer.echo(shock_quality.summary())
        if sweep_report is not None:
            typer.echo(sweep_report.summary())
        typer.echo(f"Wrote: {output_path}")
        typer.echo(f"Wrote: {metadata_path}")

    # examples: bundled cases, copied into ./<name>/ to run (only types that ship some)
    if list_examples(geometry_type):
        geometry_app.command("examples")(build_examples_command(geometry_type))

    return geometry_app
