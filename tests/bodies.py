"""Geometry files for the tests: a body given as [geometry] file = "geometry.toml".

A grid config gives its body in [geometry]: the body's keys right there (as the plain
templates do), or a canonical-geometries geometry file. For the file form, the tests
write the file `geometry <type> init` writes, so the grid templates are tested together
with the geometry templates they are meant to be used with; a test that needs another
body changes that file's text.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

from canonical_geometries.config import GEOMETRY_TEMPLATES

# --------------------------------------------------
# constants
# --------------------------------------------------
# the name the grid templates point at
GEOMETRY_FILE_NAME = "geometry.toml"


# --------------------------------------------------
# public API
# --------------------------------------------------
def write_geometry(
    folder: Path,
    geometry_type: str,
    replacements: tuple[tuple[str, str], ...] = (),
) -> Path:
    """Write geometry.toml into folder: the starter file of `geometry <type> init`, edited.

    Args:
        folder: Folder of the grid config (its geometry key is relative to it).
        geometry_type: Geometry type, e.g. "cone".
        replacements: (old, new) text replacements, each applied once and each required
            to match (so a test can't silently stop changing the body).

    Returns:
        Path of the written geometry file.
    """

    # build the file text
    geometry_text = GEOMETRY_TEMPLATES[geometry_type]
    for old, new in replacements:
        assert old in geometry_text, old
        geometry_text = geometry_text.replace(old, new, 1)

    # write
    geometry_path = folder / GEOMETRY_FILE_NAME
    geometry_path.write_text(geometry_text)

    return geometry_path


def use_geometry_file(config_text: str, file_name: str = GEOMETRY_FILE_NAME) -> str:
    """Replace a config's [geometry] table (the body's keys) by file = file_name.

    Args:
        config_text: Grid config text whose [geometry] table holds the body's keys.
        file_name: Geometry file the config should point at instead.

    Returns:
        The config text with [geometry] file = file_name.
    """

    # the [geometry] table runs from its header to the first blank line
    start = config_text.index("[geometry]\n")
    end = config_text.index("\n\n", start) + 1

    # replace it by the file form
    file_table = f'[geometry]\nfile = "{file_name}"\n'
    file_text = config_text[:start] + file_table + config_text[end:]

    return file_text
