"""Bundled example cases and the `grid-generator <type> examples` command.

Each example is a folder under ``grid_generator/examples/<type>/`` holding the
files of one case, with the config named like run's default (e.g. ``cone.toml``).
The folder name is the example's name and the first text line of its config is
the description, so the folders are the registry: adding an example means
adding a folder.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated

import typer


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _type_root(geometry_type: str) -> Traversable:
    """Folder of one geometry type's examples inside the installed package."""
    return files("grid_generator") / "examples" / geometry_type


def _description(config_text: str) -> str:
    """First text line of a config's header comment (dividers skipped)."""

    # walk the leading comment block
    for line in config_text.splitlines():
        stripped = line.strip()

        # the header ends at the first line that isn't a comment
        if stripped and not stripped.startswith("#"):
            break

        # skip blank comment lines and dividers like "# -----"
        text = stripped.lstrip("#").strip()
        if text and set(text) - set("-=#"):
            return text

    return ""


# --------------------------------------------------
# public API
# --------------------------------------------------
def list_examples(geometry_type: str) -> dict[str, str]:
    """Return a name -> description mapping of one geometry type's examples.

    Args:
        geometry_type: Geometry type, e.g. "cone".

    Returns:
        Dict of {name: description}, sorted by name (empty without examples).
    """

    # a type without examples has no folder
    root = _type_root(geometry_type)
    if not root.is_dir():
        return {}

    # every folder with the type's config is an example
    examples: dict[str, str] = {}
    config_name = f"{geometry_type}.toml"
    for folder in sorted(root.iterdir(), key=lambda item: item.name):
        config = folder / config_name
        if folder.is_dir() and config.is_file():
            examples[folder.name] = _description(config.read_text(encoding="utf-8"))

    return examples


def copy_example(
    geometry_type: str, name: str, destination: str | Path, force: bool = False
) -> list[Path]:
    """Copy one example's files into a folder.

    Args:
        geometry_type: Geometry type, e.g. "cone".
        name: Example name (see list_examples).
        destination: Folder to create and copy the files into.
        force: Overwrite files that already exist in the folder.

    Returns:
        The paths of the written files.

    Raises:
        KeyError: If the type has no example of that name.
        FileExistsError: If the folder exists and force is False.
    """

    # check: a known example
    if name not in list_examples(geometry_type):
        raise KeyError(name)

    # check: don't overwrite an earlier copy unless asked to
    destination = Path(destination)
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists (use --force to overwrite)")

    # copy every file of the example folder
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    source = _type_root(geometry_type) / name
    for item in sorted(source.iterdir(), key=lambda entry: entry.name):
        if item.is_file():
            target = destination / item.name
            target.write_bytes(item.read_bytes())
            written.append(target)

    return written


def build_examples_command(geometry_type: str) -> Callable[..., None]:
    """Build the `examples` subcommand of one geometry type.

    Args:
        geometry_type: Geometry type, e.g. "cone" (also the examples folder).

    Returns:
        The command function to register as ``<geometry_app>.command("examples")``.
    """

    def cmd_examples(
        name: Annotated[
            str | None,
            typer.Argument(help="Example to copy into ./<name>/. Omit to list the examples."),
        ] = None,
        force: Annotated[
            bool,
            typer.Option("--force", "-f", help="Overwrite an existing ./<name>/ folder."),
        ] = False,
    ) -> None:
        """List the bundled examples, or copy one into a new folder to run."""

        # list: name and description of every example
        examples = list_examples(geometry_type)
        if name is None:
            width = max(len(example_name) for example_name in [*examples, "<name>"])
            typer.echo("")
            typer.echo(f"  {'<name>':<{width}}  description")
            typer.echo(f"  {'-' * width}  {'-' * 11}")
            for example_name, description in examples.items():
                typer.echo(f"  {example_name:<{width}}  {description}")
            typer.echo("")
            typer.echo("command sequence:")
            typer.echo("  - check out the example")
            typer.echo(f"     grid-generator {geometry_type} examples <name>")
            typer.echo("  - change into its folder")
            typer.echo("     cd <name>")
            typer.echo("  - run the example")
            typer.echo(f"     grid-generator {geometry_type} run")
            typer.echo("")
            return

        # copy the example into ./<name>/
        try:
            written = copy_example(geometry_type, name, Path(name), force=force)
        except KeyError:
            known = ", ".join(examples)
            typer.echo(f"error: unknown example {name!r}; available: {known}", err=True)
            raise typer.Exit(1) from None
        except FileExistsError as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(1) from None

        # report the files and the next step
        for path in written:
            typer.echo(f"wrote {path}")
        typer.echo(f"then: cd {name} && grid-generator {geometry_type} run")

    return cmd_examples
