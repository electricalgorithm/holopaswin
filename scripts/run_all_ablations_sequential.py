"""Sequential execution of all ablation experiments.

This script runs all experiments one after another to avoid memory issues.
Designed to run unattended for ~20 hours.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def log(message: str, log_file: Path) -> None:
    """Log message to both console and file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with log_file.open("a") as f:
        f.write(log_msg + "\n")


def run_experiment(
    script_name: str,
    args: list[str],
    log_file: Path,
    main_log: Path,
) -> tuple[bool, float]:
    """Run an experiment and return success status and duration."""
    log(f"Starting: {script_name}", main_log)
    log(f"  Args: {' '.join(args)}", main_log)
    log(f"  Log file: {log_file}", main_log)

    cmd = [sys.executable, f"scripts/{script_name}", *args]

    start_time = time.time()

    with log_file.open("w") as f:
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            duration = time.time() - start_time

            if result.returncode == 0:
                log(f"✓ {script_name} completed successfully ({duration / 60:.1f} min)", main_log)
                return True, duration

            log(f"✗ {script_name} failed with code {result.returncode} ({duration / 60:.1f} min)", main_log)
            return False, duration  # noqa: TRY300

        except (subprocess.SubprocessError, OSError) as e:
            duration = time.time() - start_time
            log(f"✗ {script_name} crashed: {e} ({duration / 60:.1f} min)", main_log)
            return False, duration


def main() -> None:  # noqa: PLR0915
    """Run all ablation experiments sequentially."""
    parser = argparse.ArgumentParser(description="Run all ablation experiments sequentially")
    parser.add_argument("--output-dir", type=str, default="results/ablation_sequential", help="Output directory")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness tests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_log = output_dir / "main.log"

    log("=" * 80, main_log)
    log("SEQUENTIAL ABLATION STUDY EXECUTION", main_log)
    log("=" * 80, main_log)
    log(f"Output directory: {output_dir}", main_log)
    log(f"Start time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", main_log)
    log("", main_log)

    overall_start = time.time()
    results = []

    # Experiment 1: Loss component ablation (5 configs x 5 epochs)
    log("", main_log)
    log("=" * 80, main_log)
    log("EXPERIMENT 1: Loss Component Ablation", main_log)
    log("Expected duration: ~10 hours", main_log)
    log("=" * 80, main_log)

    success, duration = run_experiment(
        "ablation_loss_components.py",
        ["--output-dir", str(output_dir / "loss_components")],
        output_dir / "loss_components.log",
        main_log,
    )
    results.append(("Loss Components", success, duration))

    # Experiment 2: Loss weight sensitivity
    log("", main_log)
    log("=" * 80, main_log)
    log("EXPERIMENT 2: Loss Weight Sensitivity", main_log)
    log("Expected duration: ~3-4 hours", main_log)
    log("=" * 80, main_log)

    success, duration = run_experiment(
        "ablation_loss_weights_sensitivity.py",
        ["--output-dir", str(output_dir / "loss_weights")],
        output_dir / "loss_weights.log",
        main_log,
    )
    results.append(("Loss Weights", success, duration))

    # Experiment 3: Architecture ablation (4 models x 5 epochs)
    log("", main_log)
    log("=" * 80, main_log)
    log("EXPERIMENT 3: Architecture Ablation", main_log)
    log("Expected duration: ~8 hours", main_log)
    log("=" * 80, main_log)

    success, duration = run_experiment(
        "ablation_architecture.py",
        ["--output-dir", str(output_dir / "architecture")],
        output_dir / "architecture.log",
        main_log,
    )
    results.append(("Architecture", success, duration))

    # Experiment 4: Robustness tests (requires trained model)
    if not args.skip_robustness:
        log("", main_log)
        log("=" * 80, main_log)
        log("EXPERIMENT 4: Robustness Tests", main_log)
        log("Expected duration: ~1 hour", main_log)
        log("=" * 80, main_log)

        model_path = output_dir / "loss_components" / "full_model_best.pth"
        if model_path.exists():
            success, duration = run_experiment(
                "ablation_robustness.py",
                [
                    "--model-path",
                    str(model_path),
                    "--output-dir",
                    str(output_dir / "robustness"),
                ],
                output_dir / "robustness.log",
                main_log,
            )
            results.append(("Robustness", success, duration))
        else:
            log(f"⚠ Skipping robustness tests: model not found at {model_path}", main_log)
            results.append(("Robustness", False, 0))

    # Summary
    total_time = time.time() - overall_start

    log("", main_log)
    log("=" * 80, main_log)
    log("EXECUTION SUMMARY", main_log)
    log("=" * 80, main_log)
    log(f"Total time: {total_time / 3600:.2f} hours ({total_time / 60:.1f} minutes)", main_log)
    log("", main_log)

    for name, success, duration in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        log(f"{status:12} {name:25} {duration / 60:6.1f} min", main_log)

    log("", main_log)
    log("Results saved to:", main_log)
    log(f"  - Loss components: {output_dir / 'loss_components'}", main_log)
    log(f"  - Loss weights: {output_dir / 'loss_weights'}", main_log)
    log(f"  - Architecture: {output_dir / 'architecture'}", main_log)
    if not args.skip_robustness:
        log(f"  - Robustness: {output_dir / 'robustness'}", main_log)

    log("", main_log)
    log("Next steps:", main_log)
    log("1. Review CSV files in each results directory", main_log)
    log("2. Copy LaTeX tables to article/src/tables/", main_log)
    log("3. Copy z_mismatch_plot.png to article/src/figs/", main_log)
    log("4. Update article tables with actual values", main_log)
    log("5. Compile article with pdflatex", main_log)

    log("", main_log)
    log(f"End time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", main_log)
    log("=" * 80, main_log)

    # Exit with error if any experiment failed
    if not all(success for _, success, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
