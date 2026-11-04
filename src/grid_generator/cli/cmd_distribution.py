"""1D point distributions: `grid-generator distribution info`, `init` and `run`."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from grid_generator.cli.help_text import format_examples
from grid_generator.config import distribution_template, load_distribution_config
from grid_generator.config.templates import DEFAULT_DISTRIBUTION
from grid_generator.distributions import build_distribution
from grid_generator.distributions.catalog import (
    BODY_FITTED_RULES,
    DISTRIBUTIONS,
    SEGMENTS,
    USED_IN,
    DistributionInfo,
    requirement_text,
)
from grid_generator.distributions.writers import distribution_columns, write_distribution

# --------------------------------------------------
# distribution command group
# --------------------------------------------------
distribution_app = typer.Typer(
    name="distribution",
    help="1D point distributions: the reference of all types, and 1D point files.",
    epilog=format_examples(
        [
            ("grid-generator distribution info", "list the types"),
            ("grid-generator distribution info geometric", "keys and an example"),
            ("grid-generator distribution init --type tanh", "starter file"),
            ("grid-generator distribution run -c distribution.toml", "write the points"),
        ]
    ),
    no_args_is_help=True,
)


@distribution_app.command(
    "info", help="List the distribution types, or show one type's keys and an example."
)
def cmd_info(
    name: Annotated[
        str | None,
        typer.Argument(help=f"Distribution type: {', '.join(DISTRIBUTIONS)}."),
    ] = None,
) -> None:
    """Print the distribution reference: all types, or one type in detail."""

    # no type given: list them all, then the body-fitted rules
    if name is None:
        typer.echo(_overview_text())
        return

    # one type in detail
    if name not in DISTRIBUTIONS:
        typer.echo(
            f"error: unknown type {name!r}; choose from: {', '.join(DISTRIBUTIONS)}", err=True
        )
        raise typer.Exit(code=1)
    typer.echo(_type_text(name, DISTRIBUTIONS[name]))


@distribution_app.command("init", help="Create a starter configuration (--type).")
def cmd_init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Configuration file to create."),
    ] = Path("distribution.toml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite an existing configuration."),
    ] = False,
    kind: Annotated[
        str,
        typer.Option("--type", help=f"Distribution type: {', '.join(DISTRIBUTIONS)}."),
    ] = DEFAULT_DISTRIBUTION,
) -> None:
    """Create a 1D distribution configuration."""

    # check the options before touching any file
    if kind not in DISTRIBUTIONS:
        typer.echo(f"error: --type must be one of: {', '.join(DISTRIBUTIONS)}", err=True)
        raise typer.Exit(code=1)

    # protect existing user configuration unless overwrite was requested
    if output.exists() and not force:
        typer.echo(f"error: {output} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    # write and validate the starter configuration
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(distribution_template(kind), encoding="utf-8")
    load_distribution_config(output)

    typer.echo(f"Created: {output}")
    typer.echo(f"Edit the file, then run: grid-generator distribution run -c {output}")


@distribution_app.command("run", help="Write the points of a distribution configuration.")
def cmd_run(
    config: Annotated[
        Path,
        typer.Option(
            "--config", "-c", help="Distribution TOML configuration. Defaults to distribution.toml."
        ),
    ] = Path("distribution.toml"),
) -> None:
    """Generate the points of a 1D distribution and write them."""

    try:
        # read and validate (pydantic's ValidationError is a ValueError)
        distribution_config = load_distribution_config(config)

        # generate
        coordinates = build_distribution(distribution_config.distribution)

        # write
        output_file = distribution_config.output.filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_path = write_distribution(coordinates, output_file)
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None

    # report: size, range, spacings
    columns = distribution_columns(coordinates)
    spacings = np.diff(coordinates)
    typer.echo(
        f"Generated distribution: {coordinates.size} points "
        f"from {coordinates[0]:.6g} to {coordinates[-1]:.6g}"
    )
    typer.echo(f"  first spacing      {spacings[0]:.4e}")
    typer.echo(f"  last spacing       {spacings[-1]:.4e}")
    typer.echo(f"  smallest spacing   {spacings.min():.4e}")
    typer.echo(f"  largest spacing    {spacings.max():.4e}")
    typer.echo(f"  max spacing ratio  {_max_ratio(columns['ratio']):.4f}")
    typer.echo(f"Wrote: {output_path}")


# --------------------------------------------------
# helpers: reference text
# --------------------------------------------------
def _overview_text() -> str:
    """All distribution types, one line each, then the body-fitted rules."""

    # types
    width = max(len(name) for name in DISTRIBUTIONS)
    lines = ["Distribution types", f"(used in {USED_IN})", ""]
    for name, info in DISTRIBUTIONS.items():
        lines.append(f"  {name.ljust(width)}  {info.summary}")

    # rules
    lines += ["", "Body-fitted grids:"]
    for rule in BODY_FITTED_RULES:
        lines.append(f"  - {rule}")

    # pointer to the details
    lines += ["", "Keys and an example: grid-generator distribution info <type>"]
    text = "\n".join(lines)

    return text


def _type_text(name: str, info: DistributionInfo) -> str:
    """One type in detail: summary, keys, notes, segment types (segmented), example."""

    # summary and keys
    lines = [f"{name}: {info.summary}", "", "keys:"]
    lines += _key_lines(info)

    # segment types (only the segmented type has them)
    if name == "segmented":
        lines += ["", "segment types ([[<table>.segments]], in order):"]
        for segment_name, segment_info in SEGMENTS.items():
            lines.append(f"  {segment_name}: {segment_info.summary}")
            lines += ["  " + line for line in _key_lines(segment_info)]

    # extra rules
    if info.notes:
        lines += ["", "rules:"]
        for note in info.notes:
            lines.append(f"  - {note}")

    # example, in a [distribution] table
    lines += ["", "example:", "  [distribution]"]
    for line in info.example.format(table="distribution").splitlines():
        lines.append(f"  {line}".rstrip())
    text = "\n".join(lines)

    return text


def _key_lines(info: DistributionInfo) -> list[str]:
    """Aligned lines: key name, description, and whether the key must be given."""

    # align the names and descriptions
    name_width = max(len(key.name) for key in info.keys)
    description_width = max(len(key.description) for key in info.keys)

    # build
    lines = []
    for key in info.keys:
        name = key.name.ljust(name_width)
        description = key.description.ljust(description_width)
        lines.append(f"  {name}  {description}  {requirement_text(key)}")

    return lines


def _max_ratio(ratios: np.ndarray) -> float:
    """Largest growth between neighbouring spacings, counting shrinking as growth too."""

    # the ratio column is 0 where undefined (first and last point)
    defined = ratios[ratios > 0.0]
    if defined.size == 0:
        return 1.0

    # larger over smaller, so growth and shrinkage count the same
    largest = float(np.max(np.maximum(defined, 1.0 / defined)))
    return largest
