"""Command-line interface of holopaswin.

holopaswin config [--output FILE]
    write the default training configuration as TOML
holopaswin train DATA_DIR [--config FILE] [OPTIONS]
    train a network on a hologen dataset
holopaswin evaluate TEST_DIR [--model NAME=RUN] [--baseline NAME] [OPTIONS]
    score networks and classical baselines on a test set
holopaswin figure TEST_DIR --samples 0,5 [--model NAME=RUN] [--baseline NAME] [OPTIONS]
    draw the reconstructions of some test holograms side by side
holopaswin benchmark (--model RUN | --architecture NAME --data DIR) [OPTIONS]
    time the network alone and the whole path from a stored hologram
holopaswin sweep SWEEP_ROOT --model NAME=RUN [OPTIONS]
    score models on test sets recorded at a wrong distance
"""

import pathlib
from typing import TYPE_CHECKING, Annotated, Any

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
    if torch.cuda.is_available():  # pragma: no cover  (no CUDA on the test machine)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.command("config")
def config_command(
    output: Annotated[
        pathlib.Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of printing."),
    ] = None,
) -> None:
    """Write the default training configuration, with the source of every value.

    \f
    Args:
        output (pathlib.Path | None): File to write to; printed if None.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    from holopaswin.training.config import config_to_toml, default_config  # noqa: PLC0415

    text: str = config_to_toml(default_config())
    if output is None:
        typer.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    typer.echo(f"Configuration written to {output}")


@app.command()
def train(  # noqa: PLR0913 (one argument per command-line option)
    data_dir: Annotated[
        pathlib.Path,
        typer.Argument(
            help="hologen dataset to train on; its recipe.toml gives the geometry.", exists=True, file_okay=False
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option(
            "--output", "-o", help="Run folder for model.pth, last.pth, config.toml, recipe.toml and history.csv."
        ),
    ] = pathlib.Path("run"),
    config: Annotated[
        pathlib.Path | None,
        typer.Option(
            help="Training configuration (see holopaswin config). The options below override it. "
            "(default: the defaults, or the run's own config with --resume)",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    resume: Annotated[bool, typer.Option(help="Continue the run in --output after its last finished epoch.")] = False,
    architecture: Annotated[str | None, typer.Option(help="Network: holopaswin, unet, resnet_unet or hrnet.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Total number of epochs.", min=1)] = None,
    batch_size: Annotated[int | None, typer.Option(help="Holograms per step.", min=1)] = None,
    seed: Annotated[int | None, typer.Option(help="Seed of the training and validation split.")] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device, e.g. cpu or cuda. (default: CUDA, then MPS, then CPU)"),
    ] = None,
) -> None:
    """Train a network on a hologen dataset.

    Every architecture is trained with the same loop and the physics loss. The run
    folder keeps the best model by validation loss, the state for resuming, the
    configuration, the dataset's geometry and the loss per epoch.

    \f
    Args:
        data_dir (pathlib.Path): Training set folder.
        output (pathlib.Path): Run folder.
        config (pathlib.Path | None): Configuration file, or None.
        resume (bool): Whether to continue the run in the folder.
        architecture (str | None): Network override.
        epochs (int | None): Epochs override.
        batch_size (int | None): Batch size override.
        seed (int | None): Split seed override.
        device (str | None): Torch device name, or None to choose one.

    Raises:
        typer.Exit: With code 2 if the configuration or the run folder is invalid,
            and with code 130 if training is interrupted.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    from holopaswin.training.config import (  # noqa: PLC0415
        TrainingConfig,
        default_config,
        load_config,
        with_overrides,
    )
    from holopaswin.training.trainer import train as run_training  # noqa: PLC0415

    try:
        base: TrainingConfig
        if config is not None:
            base = load_config(config)
        elif resume and (output / "config.toml").is_file():
            base = load_config(output / "config.toml")
        else:
            base = default_config()
        chosen: TrainingConfig = with_overrides(base, architecture, epochs, batch_size, seed)
        run_training(data_dir, chosen, output, select_device(device), resume=resume, report=typer.echo)
    except (ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    except KeyboardInterrupt as interrupt:
        raise typer.Exit(code=130) from interrupt
    typer.echo(f"Run saved to {output}")


@app.command()
def evaluate(  # noqa: PLR0913 (one argument per command-line option)
    test_dir: Annotated[
        pathlib.Path,
        typer.Argument(help="hologen test set; its recipe.toml gives the geometry.", exists=True, file_okay=False),
    ],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to score as NAME=RUN, where RUN is a run folder of holopaswin train or a HoloPASWIN "
            "checkpoint; repeat for several.",
        ),
    ] = None,
    baseline: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline",
            "-b",
            help="Classical reconstruction to score: asm (back-propagation), gs50 or gs100 (Gerchberg-Saxton), "
            "lf50 or lf100 (Latychevskaia-Fink); repeat for several.",
        ),
    ] = None,
    output: Annotated[
        pathlib.Path, typer.Option("--output", "-o", help="Folder for per_sample.csv and summary.json.")
    ] = pathlib.Path("evaluation"),
    batch_size: Annotated[int, typer.Option(help="Holograms per forward pass.", min=1)] = 4,
    limit: Annotated[
        int | None, typer.Option(help="Holograms to use, for a quick check. (default: all)", min=1)
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device, e.g. cpu or cuda. (default: CUDA, then MPS, then CPU)"),
    ] = None,
) -> None:
    """Score networks and classical baselines on a test set.

    Every hologram is scored with the metrics of the paper's tables. The summary
    gives, per metric and method, the mean and standard deviation and the median
    with a bootstrap interval, overall, per object family and per noise
    configuration. Paired Wilcoxon tests compare the first two methods, counting
    the models in the order given and then the baselines, with a Holm correction
    over the groups of each metric.

    \f
    Args:
        test_dir (pathlib.Path): Test set folder.
        model (list[str] | None): NAME=RUN specifications.
        baseline (list[str] | None): Baseline names.
        output (pathlib.Path): Folder for the results.
        batch_size (int): Holograms per forward pass.
        limit (int | None): Holograms to use, or None for all.
        device (str | None): Torch device name, or None to choose one.

    Raises:
        typer.Exit: With code 2 if a model, a baseline or the test set is invalid.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    from holopaswin.evaluation.report import run_evaluation, write_evaluation  # noqa: PLC0415
    from holopaswin.evaluation.scoring import parse_model_specs  # noqa: PLC0415

    baselines: list[str] = baseline or []
    try:
        runs: dict[str, pathlib.Path] = parse_model_specs(model or [])
        table: pd.DataFrame = run_evaluation(test_dir, runs, baselines, select_device(device), batch_size, limit)
    except (ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    metrics: list[str] = write_evaluation(table, [*runs, *baselines], output)
    typer.echo(f"Scored {len(table)} reconstructions with {', '.join(metrics)}; results in {output}")


def parse_integers(text: str, option: str, count: int | None = None) -> list[int]:
    """Read a comma-separated list of integers given on the command line.

    Args:
        text (str): The option value, e.g. "0,5,9".
        option (str): Name of the option, for the error message.
        count (int | None): Number of integers required, or None for any number.

    Returns:
        list[int]: The integers.

    Raises:
        ValueError: If the value is not a list of integers of the required length.
    """
    try:
        values: list[int] = [int(part) for part in text.split(",")]
    except ValueError as error:
        msg: str = f"{option} expects comma-separated integers, got {text!r}."
        raise ValueError(msg) from error
    if count is not None and len(values) != count:
        msg = f"{option} expects {count} comma-separated integers, got {text!r}."
        raise ValueError(msg)
    return values


@app.command()
def figure(  # noqa: PLR0913 (one argument per command-line option)
    test_dir: Annotated[
        pathlib.Path,
        typer.Argument(help="hologen test set; its recipe.toml gives the geometry.", exists=True, file_okay=False),
    ],
    samples: Annotated[str, typer.Option(help="Indices of the holograms to draw, e.g. 0,5,9.")],
    model: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Model to draw as NAME=RUN (run folder or checkpoint); repeat for several."),
    ] = None,
    baseline: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline", "-b", help="Classical reconstruction to draw: asm, gs50, gs100, lf50 or lf100; repeatable."
        ),
    ] = None,
    output: Annotated[pathlib.Path, typer.Option("--output", "-o", help="PNG file to write.")] = pathlib.Path(
        "reconstructions.png"
    ),
    zoom: Annotated[
        str | None, typer.Option(help="Region to enlarge in every panel as COLUMN,ROW,SIZE in pixels.")
    ] = None,
    error: Annotated[bool, typer.Option(help="Add a row with the absolute phase error of every method.")] = False,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device, e.g. cpu or cuda. (default: CUDA, then MPS, then CPU)"),
    ] = None,
) -> None:
    """Draw the reconstructions of some test holograms side by side.

    Each sample gets a row with the hologram, the phase of every method and the
    true phase on one colour scale.

    \f
    Args:
        test_dir (pathlib.Path): Test set folder.
        samples (str): Comma-separated sample indices.
        model (list[str] | None): NAME=RUN specifications.
        baseline (list[str] | None): Baseline names.
        output (pathlib.Path): PNG file.
        zoom (str | None): COLUMN,ROW,SIZE of the region to enlarge, or None.
        error (bool): Whether to add error rows.
        device (str | None): Torch device name, or None to choose one.

    Raises:
        typer.Exit: With code 2 if a model, a baseline, a sample or the zoom is invalid.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    from holopaswin.evaluation.figures import check_zoom, draw_figure, reconstruct_samples  # noqa: PLC0415
    from holopaswin.evaluation.scoring import parse_model_specs  # noqa: PLC0415
    from holopaswin.models import read_geometry  # noqa: PLC0415

    try:
        indices: list[int] = parse_integers(samples, "--samples")
        region: list[int] | None = None if zoom is None else parse_integers(zoom, "--zoom", count=3)
        check_zoom(None if region is None else (region[0], region[1], region[2]), read_geometry(test_dir).resolution)
        holograms, targets, fields = reconstruct_samples(
            test_dir, parse_model_specs(model or []), baseline or [], indices, select_device(device)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        draw_figure(
            holograms,
            targets,
            fields,
            output,
            zoom=None if region is None else (region[0], region[1], region[2]),
            error=error,
        )
    except (ValueError, FileNotFoundError) as problem:
        typer.echo(f"Error: {problem}", err=True)
        raise typer.Exit(code=2) from problem
    typer.echo(f"Figure saved to {output}")


@app.command()
def benchmark(  # noqa: PLR0913 (one argument per command-line option)
    model: Annotated[
        pathlib.Path | None,
        typer.Option("--model", "-m", help="Run folder of holopaswin train, or a HoloPASWIN checkpoint with --data."),
    ] = None,
    architecture: Annotated[
        str | None,
        typer.Option(help="Time an untrained network instead: holopaswin, unet, resnet_unet or hrnet. Needs --data."),
    ] = None,
    data: Annotated[
        pathlib.Path | None,
        typer.Option(
            help="hologen dataset; gives the geometry and adds the end-to-end timing from stored holograms.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    batch_size: Annotated[
        list[int] | None, typer.Option("--batch-size", help="Batch size to time; repeat for several. (default: 1)")
    ] = None,
    iterations: Annotated[int, typer.Option(help="Timed passes per batch size.", min=1)] = 100,
    warmup: Annotated[int, typer.Option(help="Untimed passes before timing.", min=0)] = 10,
    output: Annotated[pathlib.Path, typer.Option("--output", "-o", help="JSON file for the timings.")] = pathlib.Path(
        "benchmark.json"
    ),
    device: Annotated[
        str | None,
        typer.Option(help="Torch device, e.g. cpu or cuda. (default: CUDA, then MPS, then CPU)"),
    ] = None,
) -> None:
    """Time the network alone and the whole path from a stored hologram.

    The network is timed per batch size with the device synchronised, after a
    warm-up. With --data, reading, normalising and reconstructing stored
    holograms one by one is timed too.

    \f
    Args:
        model (pathlib.Path | None): Run folder or checkpoint, or None.
        architecture (str | None): Untrained network to time, or None.
        data (pathlib.Path | None): Dataset for the geometry and end-to-end timing.
        batch_size (list[int] | None): Batch sizes, or None for 1.
        iterations (int): Timed passes per batch size.
        warmup (int): Untimed passes.
        output (pathlib.Path): JSON file.
        device (str | None): Torch device name, or None to choose one.

    Raises:
        typer.Exit: With code 2 if the model or its geometry cannot be determined.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    import json  # noqa: PLC0415

    from holopaswin.evaluation.benchmark import run_benchmark  # noqa: PLC0415

    try:
        report: dict[str, Any] = run_benchmark(
            model, architecture, data, batch_size or [1], iterations, warmup, select_device(device)
        )
    except (ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    row: dict[str, float]
    for row in report["model"]:
        typer.echo(
            f"batch {int(row['batch_size'])}: latency {row['latency_ms']:.1f} ms, "
            f"{row['holograms_per_second']:.1f} holograms/s"
        )
    if "end_to_end" in report:
        typer.echo(f"end to end: {report['end_to_end']['total_ms']:.1f} ms per hologram")
    typer.echo(f"Timings saved to {output}")


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
            help="Model to score as NAME=RUN, where RUN is a run folder of holopaswin train or a HoloPASWIN "
            "checkpoint; repeat for several. The first two are compared at every offset.",
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
        model (list[str]): NAME=RUN specifications.
        output (pathlib.Path): Folder for the results.
        batch_size (int): Holograms per forward pass.
        train_error (float): Half-width of the training distance range in um.
        limit (int | None): Holograms to use from each set, or None for all.
        device (str | None): Torch device name, or None to choose one.

    Raises:
        typer.Exit: With code 2 if a model specification or the sweep sets are invalid.
    """  # noqa: D301 (typer hides the text after the form feed from --help)
    # Imported here so that `holopaswin --help` stays fast.
    from holopaswin.evaluation.scoring import parse_model_specs  # noqa: PLC0415
    from holopaswin.evaluation.sweep import run_sweep, write_results  # noqa: PLC0415

    try:
        runs: dict[str, pathlib.Path] = parse_model_specs(model)
        table: pd.DataFrame = run_sweep(sweep_root, runs, select_device(device), batch_size, limit)
    except (ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    metrics: list[str] = write_results(table, list(runs), train_error, output)
    typer.echo(f"Scored {len(table)} reconstructions with {', '.join(metrics)}; results in {output}")
