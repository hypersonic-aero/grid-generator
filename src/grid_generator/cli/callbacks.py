"""Shared callbacks for the grid-generator CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
import logging

import typer

from grid_generator import __version__


# --------------------------------------------------
# callbacks
# --------------------------------------------------
def version_callback(value: bool) -> None:
    """Print the installed package version and exit."""

    if value:
        typer.echo(f"grid-generator {__version__}")
        raise typer.Exit()


# --------------------------------------------------
# verbose callback
# --------------------------------------------------
def verbose_callback(value: bool) -> None:
    """Configure console logging; --verbose adds debug output from this tool.

    By default only warnings are shown. With --verbose, this tool logs at
    debug level, while the packages it calls (canonical-geometries, cfd-io,
    h5py, ...) stay quiet.
    """

    # default console logging for every logger: warnings only
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

    # debug output from this tool only; the packages it calls keep their levels
    if value:
        logging.getLogger("grid_generator").setLevel(logging.DEBUG)
