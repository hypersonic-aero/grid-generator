"""Root application for the grid-generator CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from typing import Annotated

import typer

from grid_generator.cli.callbacks import verbose_callback, version_callback
from grid_generator.cli.cmd_cartesian import cartesian_app
from grid_generator.cli.cmd_distribution import distribution_app
from grid_generator.cli.cmd_geometry import GEOMETRY_HELP, build_geometry_app

# --------------------------------------------------
# root application
# --------------------------------------------------
app = typer.Typer(
    name="grid-generator",
    help="Generate structured computational grids.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------
# register callbacks
# --------------------------------------------------
@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            callback=verbose_callback,
            help="Enable verbose (debug) output.",
        ),
    ] = False,
) -> None:
    """Generate structured computational grids."""

    del version, verbose


# --------------------------------------------------
# register subcommands (note: order of registration is order of appearance in the CLI help)
# --------------------------------------------------

# 1D point distributions: reference (info) and 1D point files (init / run)
app.add_typer(distribution_app, name="distribution")

# Cartesian grids: structured grids aligned with the coordinate axes
app.add_typer(cartesian_app, name="cartesian")

# body-fitted grids, one command group per geometry: plate, wedge, cone, ogive, ... (add more as needed)
for geometry_type in GEOMETRY_HELP:
    app.add_typer(build_geometry_app(geometry_type), name=geometry_type)
