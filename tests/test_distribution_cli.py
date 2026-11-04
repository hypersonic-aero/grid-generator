"""Tests for 1D point distributions: the reference (info), the config, the writers, and the CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from grid_generator.cli import app
from grid_generator.config import distribution_template, load_distribution_config
from grid_generator.distributions import GeometricSpec, build_distribution
from grid_generator.distributions.catalog import (
    BODY_FITTED_RULES,
    DISTRIBUTIONS,
    SEGMENTS,
    KeyInfo,
    requirement_text,
)
from grid_generator.distributions.parse import DISTRIBUTION_KEYS, SEGMENT_KEYS, parse_distribution
from grid_generator.distributions.writers import distribution_columns, write_distribution

runner = CliRunner()


# --------------------------------------------------
# the reference matches the parser
# --------------------------------------------------
@pytest.mark.parametrize("kind", list(DISTRIBUTION_KEYS))
def test_reference_lists_the_keys_the_parser_accepts(kind: str) -> None:
    """Every distribution type the parser knows is in the reference, with the same keys."""

    # keys in the reference, plus the type marker
    listed = {key.name for key in DISTRIBUTIONS[kind].keys} | {"type"}

    # check
    assert listed == DISTRIBUTION_KEYS[kind]


@pytest.mark.parametrize("kind", list(SEGMENT_KEYS))
def test_reference_lists_the_segment_keys_the_parser_accepts(kind: str) -> None:
    """Every segment type the parser knows is in the reference, with the same keys."""

    # keys in the reference, plus the type marker
    listed = {key.name for key in SEGMENTS[kind].keys} | {"type"}

    # check
    assert listed == SEGMENT_KEYS[kind]


def test_reference_has_no_extra_types() -> None:
    """The reference lists no type the parser doesn't know."""

    # check
    assert set(DISTRIBUTIONS) == set(DISTRIBUTION_KEYS)
    assert set(SEGMENTS) == set(SEGMENT_KEYS)


def example_table(kind: str) -> dict:
    """The reference example of a distribution type, as a parsed TOML table."""

    # parse the example under a [distribution] header
    example = DISTRIBUTIONS[kind].example.format(table="distribution")
    table = tomllib.loads("[distribution]\n" + example)["distribution"]

    return table


@pytest.mark.parametrize(
    ("kind", "key"),
    [(kind, key.name) for kind, info in DISTRIBUTIONS.items() for key in info.keys if key.required],
)
def test_required_keys_are_required(kind: str, key: str) -> None:
    """Leaving out a key the reference marks as required fails to parse."""

    # example without the key
    table = example_table(kind)
    del table[key]

    with pytest.raises((TypeError, ValueError)):
        parse_distribution(table, "distribution")


@pytest.mark.parametrize(
    ("segment_index", "key"),
    [
        (index, key.name)
        for index, segment in enumerate(example_table("segmented")["segments"])
        for key in SEGMENTS[segment["type"]].keys
        if key.required
    ],
)
def test_required_segment_keys_are_required(segment_index: int, key: str) -> None:
    """Leaving out a segment key the reference marks as required fails to parse."""

    # segmented example with the key removed from one segment
    table = example_table("segmented")
    del table["segments"][segment_index][key]

    with pytest.raises((TypeError, ValueError)):
        parse_distribution(table, "distribution")


@pytest.mark.parametrize(
    ("kind", "key"),
    [
        (kind, key)
        for kind, info in DISTRIBUTIONS.items()
        for key in info.keys
        if key.default is not None
    ],
)
def test_defaults_are_the_parsed_values(kind: str, key: KeyInfo) -> None:
    """Leaving out a key with a default gives the default the reference lists."""

    # example without the key
    table = example_table(kind)
    table.pop(key.name, None)
    spec = parse_distribution(table, "distribution")

    # check: the default, read as TOML, is what the spec holds
    default_value = tomllib.loads(f"value = {key.default}")["value"]
    assert getattr(spec, key.name) == default_value


@pytest.mark.parametrize(
    ("kind", "group"),
    [
        (kind, key.one_of)
        for kind, info in DISTRIBUTIONS.items()
        for key in info.keys
        if key.one_of and key.name == key.one_of[0]
    ],
)
def test_one_of_groups_take_exactly_one(kind: str, group: tuple[str, ...]) -> None:
    """The example gives exactly one key of each group; giving a second one fails."""

    # check the example
    table = example_table(kind)
    given = [name for name in group if name in table]
    assert len(given) == 1

    # add a second key of the group
    other = next(name for name in group if name not in table)
    table[other] = 2

    with pytest.raises(ValueError, match="exactly one"):
        parse_distribution(table, "distribution")


def test_max_spacing_requires_first_spacing() -> None:
    """max_spacing with a ratio instead of first_spacing fails, as the reference says."""

    # check the reference
    keys = {key.name: key for key in DISTRIBUTIONS["geometric"].keys}
    assert keys["max_spacing"].requires == ("first_spacing",)

    # geometric example with ratio and a cap
    table = example_table("geometric")
    del table["first_spacing"]
    table["ratio"] = 1.05
    table["max_spacing"] = 1.0e-4

    with pytest.raises(ValueError, match="first_spacing"):
        parse_distribution(table, "distribution")


def test_requirement_text() -> None:
    """The requirement phrase follows the structured fields."""

    # check one key of each kind
    assert requirement_text(KeyInfo(name="a", description="", required=True)) == "required"
    assert requirement_text(KeyInfo(name="a", description="", default='"x"')) == (
        'optional; default "x"'
    )
    assert requirement_text(KeyInfo(name="a", description="", one_of=("a", "b"))) == (
        "exactly one of: a, b"
    )
    assert requirement_text(KeyInfo(name="a", description="", requires=("b",))) == (
        "optional; requires b"
    )


# --------------------------------------------------
# info
# --------------------------------------------------
def test_info_lists_every_type_and_the_body_fitted_rules() -> None:
    """info without a type lists every type and the body-fitted rules."""

    # run
    result = runner.invoke(app, ["distribution", "info"])

    # check
    assert result.exit_code == 0, result.output
    for kind in DISTRIBUTIONS:
        assert kind in result.output
    for rule in BODY_FITTED_RULES:
        assert rule in result.output


@pytest.mark.parametrize("kind", list(DISTRIBUTIONS))
def test_info_shows_one_type_in_detail(kind: str) -> None:
    """info <type> shows each key and an example table."""

    # run
    result = runner.invoke(app, ["distribution", "info", kind])

    # check
    assert result.exit_code == 0, result.output
    for key in DISTRIBUTIONS[kind].keys:
        assert key.name in result.output
    assert f'type = "{kind}"' in result.output


def test_info_rejects_an_unknown_type() -> None:
    """info with an unknown type fails and lists the types."""

    # run
    result = runner.invoke(app, ["distribution", "info", "cubic"])

    # check
    assert result.exit_code == 1
    assert "choose from" in result.output


# --------------------------------------------------
# init and run
# --------------------------------------------------
@pytest.mark.parametrize("kind", list(DISTRIBUTIONS))
def test_cli_init_then_run_every_type(tmp_path: Path, kind: str) -> None:
    """Every type's starter file runs and writes the points the config describes."""

    # init and run
    config_path = tmp_path / "distribution.toml"
    result_init = runner.invoke(
        app, ["distribution", "init", "-o", str(config_path), "--type", kind]
    )
    result_run = runner.invoke(app, ["distribution", "run", "-c", str(config_path)])

    # check the commands
    assert result_init.exit_code == 0, result_init.output
    assert result_run.exit_code == 0, result_run.output

    # check the file: one row per point, the x column matches the distribution
    expected = build_distribution(load_distribution_config(config_path).distribution)
    table = np.loadtxt(tmp_path / "distribution.dat", skiprows=3)
    assert table.shape == (expected.size, 4)
    np.testing.assert_array_equal(table[:, 1], expected)


def test_cli_init_rejects_an_unknown_type(tmp_path: Path) -> None:
    """--type outside the reference fails before any file is written."""

    # run
    config_path = tmp_path / "distribution.toml"
    result = runner.invoke(app, ["distribution", "init", "-o", str(config_path), "--type", "x"])

    # check
    assert result.exit_code == 1
    assert "--type must be one of" in result.output
    assert not config_path.exists()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'filename = "distribution.dat"',
            'filename = "distribution.xyz"',
            "no distribution writer",
        ),
        ("first_spacing = 1.0e-6", "first_spcing = 1.0e-6", "first_spcing"),
        ("[output]", "[extra]\nkey = 1\n\n[output]", "extra"),
    ],
)
def test_cli_run_rejects_bad_config(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Unknown extensions, misspelled keys and unknown tables fail with a clean message."""

    # write a broken config
    config_text = distribution_template("geometric")
    assert old in config_text
    config_path = tmp_path / "distribution.toml"
    config_path.write_text(config_text.replace(old, new, 1))

    # run
    result = runner.invoke(app, ["distribution", "run", "-c", str(config_path)])

    # check
    assert result.exit_code == 1
    assert message in result.output


# --------------------------------------------------
# columns and writers
# --------------------------------------------------
def test_columns_spacing_and_ratio() -> None:
    """dx is the spacing to the next point, ratio its growth; both 0 where undefined."""

    # geometric points with a known ratio
    coordinates = build_distribution(GeometricSpec(start=0.0, end=1.0, n_points=6, ratio=1.5))
    columns = distribution_columns(coordinates)

    # check
    np.testing.assert_array_equal(columns["i"], np.arange(6))
    np.testing.assert_allclose(columns["dx"][:-1], np.diff(coordinates), rtol=1e-15)
    assert columns["dx"][-1] == 0.0
    np.testing.assert_allclose(columns["ratio"][1:-1], 1.5, rtol=1e-12)
    assert columns["ratio"][0] == 0.0
    assert columns["ratio"][-1] == 0.0


def test_writers_csv_and_bare_column(tmp_path: Path) -> None:
    """CSV has a header and four columns; .txt holds only x, one value per line."""

    # write both
    coordinates = np.linspace(0.0, 1.0, 5)
    write_distribution(coordinates, tmp_path / "points.csv")
    write_distribution(coordinates, tmp_path / "points.txt")

    # check the CSV
    csv_lines = (tmp_path / "points.csv").read_text().splitlines()
    assert csv_lines[0] == "i,x,dx,ratio"
    csv_table = np.loadtxt(tmp_path / "points.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(csv_table[:, 1], coordinates)

    # check the bare column
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "points.txt"), coordinates)


def test_writer_rejects_an_unknown_extension(tmp_path: Path) -> None:
    """write_distribution names the supported extensions."""

    with pytest.raises(ValueError, match=r"\.dat, \.csv, \.txt"):
        write_distribution(np.linspace(0.0, 1.0, 5), tmp_path / "points.xyz")
