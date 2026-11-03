"""Tests for the bundled examples and the `grid-generator <type> examples` command."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grid_generator.cli import app
from grid_generator.cli.cmd_geometry import GEOMETRY_HELP
from grid_generator.cli.examples_command import list_examples

runner = CliRunner()

# every bundled example as (geometry type, name)
ALL_EXAMPLES = [
    (geometry_type, name)
    for geometry_type in GEOMETRY_HELP
    for name in list_examples(geometry_type)
]


# --------------------------------------------------
# every example runs
# --------------------------------------------------
def test_examples_exist() -> None:
    """At least one geometry ships examples (guards against a packaging slip)."""
    assert ALL_EXAMPLES


@pytest.mark.parametrize(("geometry_type", "name"), ALL_EXAMPLES)
def test_every_example_copies_and_runs(
    geometry_type: str, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each example lands in ./<name>/ and runs there with run's default config name."""

    # list: the example appears with its description
    monkeypatch.chdir(tmp_path)
    list_result = runner.invoke(app, [geometry_type, "examples"])
    assert list_result.exit_code == 0
    assert name in list_result.output
    assert list_examples(geometry_type)[name] in list_result.output

    # copy, then run from inside the copy
    copy_result = runner.invoke(app, [geometry_type, "examples", name])
    assert copy_result.exit_code == 0
    monkeypatch.chdir(tmp_path / name)
    run_result = runner.invoke(app, [geometry_type, "run"])

    # check: the run succeeded and wrote the grid its config names
    assert run_result.exit_code == 0, run_result.output
    config = tomllib.loads(Path(f"{geometry_type}.toml").read_text(encoding="utf-8"))
    assert Path(config["output"]["filename"]).exists()


# --------------------------------------------------
# command behaviour
# --------------------------------------------------
def test_examples_refuse_an_existing_folder_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second copy fails without --force and overwrites with it."""
    monkeypatch.chdir(tmp_path)
    geometry_type, name = ALL_EXAMPLES[0]
    runner.invoke(app, [geometry_type, "examples", name])

    refused = runner.invoke(app, [geometry_type, "examples", name])
    forced = runner.invoke(app, [geometry_type, "examples", name, "--force"])

    assert refused.exit_code == 1
    assert "already exists" in refused.output
    assert forced.exit_code == 0


def test_examples_reject_an_unknown_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown name fails, lists the available examples, and writes nothing."""
    monkeypatch.chdir(tmp_path)
    geometry_type, name = ALL_EXAMPLES[0]

    result = runner.invoke(app, [geometry_type, "examples", "nope"])

    assert result.exit_code == 1
    assert "unknown example" in result.output
    assert name in result.output
    assert not (tmp_path / "nope").exists()


@pytest.mark.parametrize(
    "geometry_type", [geometry for geometry in GEOMETRY_HELP if not list_examples(geometry)]
)
def test_types_without_examples_have_no_examples_command(geometry_type: str) -> None:
    """The command only appears for geometry types that ship examples."""
    result = runner.invoke(app, [geometry_type, "examples"])
    assert result.exit_code != 0
