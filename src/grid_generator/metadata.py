"""The grid's record, written next to it as <grid>.json: what the grid is and how it was made.

Two kinds of entries:

1. What the grid is, for scripts and solvers (formats like Tecplot ASCII can't carry
   these): point counts, the boundary name of every side, the shock line j_s of a
   shock-aligned grid, the side-face transform and solver lines of a swept grid, and
   the coordinate scale of the file.
2. How it was made, for provenance: the package versions, the settings as applied
   (defaults included, so a later change of a default can't hide), the body and where
   it came from, and the shock curve itself (one point per grid line). With these, the
   grid can be made again from the record alone, even after its input files changed.

There is no timestamp: the same inputs give the same file, byte for byte, so a diff of
two records shows whether anything that made the grid changed.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from grid_generator.config.body_fitted import BodyFittedConfig
from grid_generator.shock import read_shock_curve

# --------------------------------------------------
# constants
# --------------------------------------------------
# the packages whose code made the grid (the body's profile comes from
# canonical-geometries, the file from cfd-io)
RECORDED_PACKAGES = ("grid-generator", "canonical-geometries", "cfd-io")


# --------------------------------------------------
# public API
# --------------------------------------------------
def grid_record(
    config: BodyFittedConfig,
    config_path: Path,
    grid_file: Path,
    grid_attrs: dict[str, Any],
    coordinate_scale: float,
) -> dict[str, Any]:
    """The record of one generated grid, as a JSON-ready dict.

    Args:
        config: The validated configuration the grid was made from (x_end filled in).
        config_path: Configuration file, as given to run.
        grid_file: The written grid file.
        grid_attrs: The written dataset's attrs (point counts, markers, shock line,
            sweep), as the file has them.
        coordinate_scale: Factor on the coordinates in the file ([output] scale).

    Returns:
        The record: grid_file, versions, config_file, the grid's attrs and
        coordinate_scale, then body, settings and (shock-aligned grids) shock_curve.
    """

    # paths are written relative to the config file's folder, as the config gives them,
    # so the record doesn't depend on where the case folder lives
    config_folder = config_path.parent

    # what the grid is: the file and the attrs the file can't carry
    record: dict[str, Any] = {
        "grid_file": grid_file.name,
        "versions": {package: version(package) for package in RECORDED_PACKAGES},
        "config_file": config_path.name,
    }
    record.update(grid_attrs)
    record["coordinate_scale"] = coordinate_scale

    # how it was made: the body and where it came from
    body_source = "[geometry]"
    if config.geometry_file is not None:
        body_source = _relative(config.geometry_file, config_folder)
    body = {"source": body_source}
    body.update(config.geometry.model_dump(mode="json"))
    record["body"] = body

    # how it was made: every setting as applied (the body is in body)
    settings = config.model_dump(mode="json", exclude={"geometry", "geometry_file"})
    settings["output"]["filename"] = _relative(config.output.filename, config_folder)
    if config.shock is not None:
        settings["shock"]["curve"] = _relative(config.shock.curve, config_folder)
    record["settings"] = settings

    # how it was made: the shock curve itself, as read (before the grid continues it to
    # the axis or past the outflow)
    if config.shock is not None:
        shock_x, shock_y = read_shock_curve(config.shock.curve)
        record["shock_curve"] = {
            "file": _relative(config.shock.curve, config_folder),
            "x": shock_x.tolist(),
            "y": shock_y.tolist(),
        }

    return record


# --------------------------------------------------
# private helpers
# --------------------------------------------------
def _relative(path: Path, folder: Path) -> str:
    """A path as written relative to folder (forward slashes, as in the TOML files).

    Args:
        path: Resolved path (folder / the path the config gives).
        folder: The config file's folder.

    Returns:
        The path relative to folder.
    """

    # relative to the config's folder (paths outside it come out as ../...)
    relative_path = os.path.relpath(path, folder)
    relative_text = Path(relative_path).as_posix()

    return relative_text
