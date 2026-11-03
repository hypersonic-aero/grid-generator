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

# loggers that --verbose switches to debug: this tool and the uahypersonics
# packages it calls
DEBUG_LOGGERS = ("grid_generator", "canonical_geometries", "cfd_io")


def verbose_callback(value: bool) -> None:
    """Configure console logging; --verbose adds debug output from our own packages.

    By default only warnings are shown. With --verbose, this tool and the
    packages it calls (e.g. cfd-io writing a grid) log at debug level, while
    third-party libraries (h5py, ...) stay quiet.
    """

    # default console logging for every logger: warnings only
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

    # debug output from this tool and the uahypersonics packages it calls
    if value:
        for logger_name in DEBUG_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
