"""Per-geometry command groups: `grid-generator <type> init` and `grid-generator <type> run`."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

import typer

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
from grid_generator.quality import grid_quality

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

    # examples shown under the group's help (--span only for planar bodies)
    planar = geometry_type not in AXISYMMETRIC_TYPES
    command = f"grid-generator {geometry_type}"
    examples = [
        (f"{command} init", "defaults"),
        (f"{command} init --body segmented", "segmented body"),
        (f"{command} init --normal tanh", "tanh wall-normal"),
    ]
    if planar:
        examples.append((f"{command} init --span", "extruded in z (3D)"))
    examples.append((f"{command} run {geometry_type}.toml", "generate the grid"))

    # command group
    geometry_app = typer.Typer(
        name=geometry_type,
        help=GEOMETRY_HELP[geometry_type],
        epilog=format_examples(examples),
        no_args_is_help=True,
    )
    default_config = Path(f"{geometry_type}.toml")

    # init summary names its options, so they show in the command list
    init_options = "--body, --normal, --span" if planar else "--body, --normal"
    init_help = f"Create a starter configuration ({init_options})."

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
            str,
            typer.Option(help=f"Wall-normal distribution: {', '.join(NORMAL_BLOCKS)}."),
        ] = DEFAULT_NORMAL,
        span: Annotated[
            bool,
            typer.Option(
                "--span",
                # the backslash keeps the help formatter from reading [span] as markup
                help="Add a \\[span] table: extrude the grid in z (3D).",
            ),
        ] = False,
    ) -> None:
        """Create a starter configuration."""

        # check the options before touching any file
        if body not in BODY_BLOCKS:
            typer.echo(f"Error: --body must be one of: {', '.join(BODY_BLOCKS)}", err=True)
            raise typer.Exit(code=1)
        if normal not in NORMAL_BLOCKS:
            typer.echo(f"Error: --normal must be one of: {', '.join(NORMAL_BLOCKS)}", err=True)
            raise typer.Exit(code=1)
        if span and geometry_type in AXISYMMETRIC_TYPES:
            typer.echo(
                f"Error: --span extrudes planar bodies; a {geometry_type} is axisymmetric",
                err=True,
            )
            raise typer.Exit(code=1)

        # protect existing user configuration unless overwrite was requested
        if output.exists() and not force:
            typer.echo(f"Error: {output} already exists. Use --force to overwrite.", err=True)
            raise typer.Exit(code=1)

        # write and validate the starter configuration
        template = body_fitted_template(geometry_type, body=body, normal=normal, span=span)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(template, encoding="utf-8")
        load_body_fitted_config(output, expected_type=geometry_type)

        typer.echo(f"Created: {output}")
        typer.echo(f"Edit the file, then run: grid-generator {geometry_type} run {output}")

    # run: generate the grid from a configuration, report its quality and write it
    @geometry_app.command("run")
    def cmd_run(
        config: Annotated[
            Path,
            typer.Argument(help="Configuration file."),
        ] = default_config,
    ) -> None:
        """Generate the grid from a configuration."""

        try:
            from cfd_io import StructuredGrid, write_file
            from cfd_io.writers.gmsh import write_gmsh

            # read and validate (pydantic's ValidationError is a ValueError)
            body_fitted_config = load_body_fitted_config(config, expected_type=geometry_type)

            # generate
            dataset = generate_body_fitted(body_fitted_config)

            # measure the first z-plane (an extruded grid has the same plane at every z)
            grid = dataset.grid
            plane = StructuredGrid(x=grid.x[:, :, :1], y=grid.y[:, :, :1], z=grid.z[:, :, :1])
            quality = grid_quality(plane)

            # write: Gmsh gets the element order, which the file extension can't carry;
            # everything else by extension, in double precision (cfd-io's HDF5 default is float32)
            output_file = body_fitted_config.output.filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.suffix.lower() == ".msh":
                output_path = write_gmsh(
                    output_file, dataset, order=body_fitted_config.output.order
                )
            else:
                output_path = write_file(output_file, dataset, dtype="f8")
        except (ImportError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
            typer.echo(f"Error: {error}", err=True)
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
            typer.echo("  quality of the (x, y) plane, the same at every z:")
        typer.echo(quality.summary())
        typer.echo(f"Wrote: {output_path}")

    # examples: bundled cases, copied into ./<name>/ to run (only types that ship some)
    if list_examples(geometry_type):
        geometry_app.command("examples")(build_examples_command(geometry_type))

    return geometry_app
