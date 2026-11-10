"""The grid's record, written next to it as <grid>.json: what the grid is and how it was made.

Two kinds of entries:

1. What the grid is, for scripts and solvers (formats like Tecplot ASCII can't carry
   these): point counts, the boundary name of every side, the shock line j_s of a
   shock-aligned grid, the side-face transform and solver lines of a swept grid, and
   the coordinate scale of the file.
2. How it was made, for provenance: the package versions (for an editable install,
   the git state of its source folder, since the installed version is the one of the
   last `pip install -e`), the settings as applied
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

import json
import os
import subprocess
from importlib.metadata import Distribution, distribution
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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
        "versions": {package: package_version(package) for package in RECORDED_PACKAGES},
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


def package_version(package: str) -> str:
    """The version of an installed package, or for an editable install its git state.

    An editable install (pip install -e) runs the code straight from its source folder,
    but its installed version is the one of the last `pip install -e`: commits since
    then don't change it. For such an install the source folder's git state is
    recorded instead, as `git describe` gives it: the last tag, the commits since,
    the commit, and -dirty for uncommitted changes (e.g. "git v0.6.0-2-g7d86704-dirty").

    Args:
        package: Distribution name, e.g. "grid-generator".

    Returns:
        The installed version, "git <describe>" for an editable install, or the
        installed version marked "(editable, git state unknown)" when git can't tell.
    """

    # the installed version
    installed = distribution(package)
    installed_version = installed.version

    # an ordinary install: its version is exact
    source_folder = _editable_source_folder(installed)
    if source_folder is None:
        return installed_version

    # an editable install: what the source folder holds now
    git_state = _git_describe(source_folder)
    if git_state is None:
        return f"{installed_version} (editable, git state unknown)"
    package_state = f"git {git_state}"

    return package_state


# --------------------------------------------------
# private helpers
# --------------------------------------------------
def _editable_source_folder(installed: Distribution) -> Path | None:
    """The source folder of an editable install, or None for an ordinary install.

    pip records how a package was installed in direct_url.json (PEP 610): an editable
    install has dir_info.editable = true and the folder as a file:// url.

    Args:
        installed: The installed distribution.

    Returns:
        The source folder, or None if the install isn't editable.
    """

    # read pip's record (installs from an index have none)
    direct_url_text = installed.read_text("direct_url.json")
    if direct_url_text is None:
        return None
    direct_url = json.loads(direct_url_text)

    # check: an editable install from a local folder
    editable = direct_url.get("dir_info", {}).get("editable", False)
    url = urlparse(direct_url.get("url", ""))
    if not editable or url.scheme != "file":
        return None

    # the folder
    source_folder = Path(unquote(url.path))

    return source_folder


def _git_describe(folder: Path) -> str | None:
    """git describe of a folder: last tag, commits since, commit, -dirty; None without git.

    Args:
        folder: A folder inside a git repository.

    Returns:
        e.g. "v0.6.0-2-g7d86704-dirty" (only the commit when there is no tag), or None
        if git isn't installed or the folder isn't in a repository.
    """

    # ask git (--long: always the commit count and commit, also right on a tag)
    command = ["git", "-C", str(folder), "describe", "--tags", "--long", "--dirty", "--always"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None

    # check: git answered
    if result.returncode != 0:
        return None
    git_state = result.stdout.strip()

    return git_state


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
