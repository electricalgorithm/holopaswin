"""Command-line interface of holopaswin.

holopaswin sweep SWEEP_ROOT --model NAME=CHECKPOINT [OPTIONS]
    score models on test sets recorded at a wrong distance
"""

import pathlib
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]
    import torch

app = typer.Typer(
    help="Reconstruct in-line holograms with HoloPASWIN and evaluate the reconstructions.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Reconstruct in-line holograms with HoloPASWIN and evaluate the reconstructions."""
    # The callback keeps the subcommand form (holopaswin sweep ...) while sweep is
    # the only command.


def select_device(name: str | None) -> "torch.device":
    """Choose the device to run the models on.

    Args:
        name (str | None): A torch device name such as "cpu" or "cuda", or None
            for CUDA, then MPS, then the CPU.

    Returns:
        torch.device: The device.
    """
    import torch  # noqa: PLC0415  (imported here so that --help stays fast)

    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.command()
def sweep(  # noqa: PLR0913 (one argument per command-line option)
    sweep_root: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Folder with one hologen dataset per distance offset, e.g. made with hologen generate --z-offset.",
            exists=True,
            file_okay=False,
        ),
    ],
    model: Annotated[
        list[str],
        typer.Option(
            "--model",
            "-m",
            help="Model to score as NAME=CHECKPOINT; repeat for several. The first two are compared at every offset.",
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option("--output", "-o", help="Folder for per_sample.csv, summary.json and z_sweep.png."),
    ] = pathlib.Path("z_sweep"),
    batch_size: Annotated[int, typer.Option(help="Holograms per forward pass.", min=1)] = 4,
    train_error: Annotated[
        float,
        typer.Option(help="Half-width in um of the distance range the models were trained on, shaded in the figure."),
    ] = 6.0,
    limit: Annotated[
        int | None,
        typer.Option(help="Holograms to use from each set, for a quick check. (default: all)", min=1),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device, e.g. cpu or cuda. (default: CUDA, then MPS, then CPU)"),
    ] = None,
) -> None:
    """Score models on test sets recorded at a wrong distance.

    Every model is told only the nominal distance, as a user would. Two references
    need no network: back-propagation at the nominal distance and, as an oracle,
    at the true one. Every hologram is scored with the metrics of the paper's
    tables, and the summary gives, per metric, object family and offset, the
    median with a bootstrap interval and a paired Wilcoxon test between the first
    two models.

    \f
    Args:
        sweep_root (pathlib.Path): Folder with one hologen dataset per offset.
        model (list[str]): NAME=CHECKPOINT specifications.
        output (pathlib.Path): Folder for the results.
        batch_size (int): Holograms per forward pass.
        train_error (float): Half-width of the training distance range in um.
        limit (int | None): Holograms to use from each set, or None for all.
        device (str | None): Torch device name, or None to choose one.

    Raises:
        typer.Exit: With code 2 if a model specification or the sweep sets are invalid.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    # Imported here so that `holopaswin --help` stays fast.
    from holopaswin.evaluation.sweep import parse_model_specs, run_sweep, write_results  # noqa: PLC0415

    try:
        checkpoints: dict[str, pathlib.Path] = parse_model_specs(model)
        table: pd.DataFrame = run_sweep(sweep_root, checkpoints, select_device(device), batch_size, limit)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    metrics: list[str] = write_results(table, list(checkpoints), train_error, output)
    typer.echo(f"Scored {len(table)} reconstructions with {', '.join(metrics)}; results in {output}")
