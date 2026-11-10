"""Tests for the grid record's package versions: exact for ordinary installs, git for editable ones."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from grid_generator.metadata import _editable_source_folder, _git_describe, package_version

# the repository these tests run in
REPOSITORY = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# helpers
# --------------------------------------------------
class FakeDistribution:
    """A distribution whose direct_url.json (pip's record of the install) is given."""

    def __init__(self, direct_url: dict | None) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        """direct_url.json as pip writes it, or None for an install from an index."""
        if filename != "direct_url.json" or self.direct_url is None:
            return None
        return json.dumps(self.direct_url)


# --------------------------------------------------
# tests
# --------------------------------------------------
def test_an_ordinary_install_records_its_version() -> None:
    """A package installed from an index (numpy here) is recorded by its exact version."""
    assert package_version("numpy") == version("numpy")


def test_only_an_editable_install_has_a_source_folder() -> None:
    """pip's direct_url.json: editable from a folder gives the folder; anything else None."""

    # editable from a local folder, an ordinary local install, an install from an index
    editable = FakeDistribution(
        {"dir_info": {"editable": True}, "url": "file:///home/student/dev/grid-generator"}
    )
    local = FakeDistribution({"dir_info": {}, "url": "file:///home/student/dev/grid-generator"})
    from_index = FakeDistribution(None)

    # check
    assert _editable_source_folder(editable) == Path("/home/student/dev/grid-generator")
    assert _editable_source_folder(local) is None
    assert _editable_source_folder(from_index) is None


def test_git_describe_names_the_current_commit(tmp_path: Path) -> None:
    """In a repository git describe ends with the commit (and -dirty); outside one, None."""

    # the commit this repository is at
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # check: the commit is named here, and a plain folder has no git state
    assert head in _git_describe(REPOSITORY)
    assert _git_describe(tmp_path) is None
