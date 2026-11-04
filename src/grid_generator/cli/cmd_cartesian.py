"""Cartesian-grid CLI commands."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from grid_generator.cartesian import generate_cartesian
from grid_generator.cli.help_text import format_examples
from grid_generator.config import cartesian_template, load_cartesian_config
from grid_generator.config.templates import CARTESIAN_DEFAULT_DIMS

# --------------------------------------------------
# Cartesian command group
# --------------------------------------------------
cartesian_app = typer.Typer(
    name="cartesian",
    help="Generate Cartesian grids (2D and 3D).",
    epilog=format_examples(
        [
            ("grid-generator cartesian init", "x and y (default)"),
            ("grid-generator cartesian init --dims 3", "x, y and z"),
            ("grid-generator cartesian run -c cartesian.toml", "generate the grid"),
        ]
    ),
    no_args_is_help=True,
)


@cartesian_app.command("init", help="Create a starter configuration (--dims).")
def cmd_init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Configuration file to create."),
    ] = Path("cartesian.toml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite an existing configuration."),
    ] = False,
    dims: Annotated[
        int,
        typer.Option(help="Number of dimensions: 1 (x), 2 (x, y) or 3 (x, y, z)."),
    ] = CARTESIAN_DEFAULT_DIMS,
) -> None:
    """Create a Cartesian-grid configuration."""

    # check the options before touching any file
    if dims not in (1, 2, 3):
        typer.echo("error: --dims must be 1, 2 or 3", err=True)
        raise typer.Exit(code=1)

    # protect existing user configuration unless overwrite was requested
    if output.exists() and not force:
        typer.echo(f"error: {output} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    # write and validate the starter configuration
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(cartesian_template(dims), encoding="utf-8")
    load_cartesian_config(output)

    typer.echo(f"Created: {output}")
    typer.echo(f"Edit the file, then run: grid-generator cartesian run -c {output}")


@cartesian_app.command("run")
def cmd_run(
    config: Annotated[
        Path,
        typer.Option(
            "--config", "-c", help="Cartesian-grid TOML configuration. Defaults to cartesian.toml."
        ),
    ] = Path("cartesian.toml"),
) -> None:
    """Generate a Cartesian grid from a configuration."""

    try:
        from cfd_io import write_file

        cartesian_config = load_cartesian_config(config)
        dataset = generate_cartesian(cartesian_config)
        cartesian_config.output.parent.mkdir(parents=True, exist_ok=True)
        output_path = write_file(
            cartesian_config.output,
            dataset,
            dtype=cartesian_config.dtype,
        )
    except (ImportError, OSError, TypeError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Generated Cartesian grid: {dataset.grid.shape}")
    typer.echo(f"Wrote: {output_path}")
