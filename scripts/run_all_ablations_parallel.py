"""Parallel execution wrapper for all ablation experiments.

This script runs all ablation experiments in parallel to save time.
"""

import argparse
import subprocess
import time
from pathlib import Path


def run_experiment(script_name: str, args: list[str], log_file: Path) -> subprocess.Popen:
    """Run an experiment script in the background."""
    cmd = ["python", f"scripts/{script_name}", *args]
    print(f"Starting: {script_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with log_file.open("w") as f:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return process


def main() -> None:  # noqa: PLR0915
    """Run all ablation experiments in parallel."""
    parser = argparse.ArgumentParser(description="Run all ablation experiments in parallel")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode for testing")
    parser.add_argument("--skip-architecture", action="store_true", help="Skip architecture ablation (longest)")
    args = parser.parse_args()

    output_dir = Path("results/ablation_parallel")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PARALLEL ABLATION STUDY EXECUTION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Dry-run mode: {args.dry_run}")
    print()

    processes = []

    # Experiment 1: Loss component ablation
    exp1_args = ["--output-dir", str(output_dir / "loss_components")]
    if args.dry_run:
        exp1_args.append("--dry-run")

    p1 = run_experiment(
        "ablation_loss_components.py",
        exp1_args,
        output_dir / "loss_components.log",
    )
    processes.append(("Loss Components", p1))

    # Experiment 2: Loss weight sensitivity
    exp2_args = ["--output-dir", str(output_dir / "loss_weights")]

    p2 = run_experiment(
        "ablation_loss_weights_sensitivity.py",
        exp2_args,
        output_dir / "loss_weights.log",
    )
    processes.append(("Loss Weights", p2))

    # Experiment 3: Architecture ablation (optional, longest)
    if not args.skip_architecture:
        exp3_args = ["--output-dir", str(output_dir / "architecture")]
        if args.dry_run:
            exp3_args.append("--dry-run")

        p3 = run_experiment(
            "ablation_architecture.py",
            exp3_args,
            output_dir / "architecture.log",
        )
        processes.append(("Architecture", p3))

    print()
    print("=" * 80)
    print("All experiments started. Monitoring progress...")
    print("=" * 80)
    print()

    # Monitor processes
    start_time = time.time()

    while processes:
        time.sleep(30)  # Check every 30 seconds

        still_running = []
        for name, process in processes:
            if process.poll() is None:
                still_running.append((name, process))
            else:
                elapsed = time.time() - start_time
                if process.returncode == 0:
                    print(f"✓ {name} completed successfully (elapsed: {elapsed / 60:.1f} min)")
                else:
                    print(f"✗ {name} failed with code {process.returncode} (elapsed: {elapsed / 60:.1f} min)")

        processes = still_running

        if processes:
            print(f"  Still running: {', '.join([name for name, _ in processes])}")

    total_time = time.time() - start_time
    print()
    print("=" * 80)
    print(f"All experiments completed in {total_time / 60:.1f} minutes")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Check logs in:", output_dir)
    print("2. Review results CSV files")
    print("3. Run robustness tests with trained model:")
    print(
        "   python scripts/ablation_robustness.py --model-path results/ablation_parallel/loss_components/full_model_best.pth"
    )


if __name__ == "__main__":
    main()
