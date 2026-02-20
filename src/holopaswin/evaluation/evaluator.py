"""Unified evaluator for comparing baseline methods.

Provides a consistent interface for evaluating different reconstruction methods
(neural networks, iterative algorithms) on the same test data.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from holopaswin.evaluation.metrics import compute_all_metrics


class BaselineEvaluator:
    """Evaluator for comparing holographic reconstruction methods.

    Handles both neural network models and iterative methods with a unified API.
    Computes per-sample metrics and aggregates results to CSV/JSON.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        output_dir: str | Path = "results/baseline_comparison",
    ) -> None:
        """Initialize the evaluator.

        Args:
            device: PyTorch device to use. If None, auto-detects.
            output_dir: Directory to save results.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: dict[str, list[dict[str, Any]]] = {}

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,  # type: ignore[type-arg]
        model_name: str,
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Evaluate a single model on the test dataset.

        Args:
            model: PyTorch model (or GS algorithm wrapped as Module).
            dataloader: DataLoader yielding (hologram, gt_object) pairs.
            model_name: Name for this model in results.
            max_samples: Optional limit on samples to evaluate.

        Returns:
            Dictionary with aggregated metrics.
        """
        model = model.to(self.device)
        model.eval()

        per_sample_metrics: list[dict[str, Any]] = []
        inference_times: list[float] = []

        sample_count = 0
        with torch.no_grad():
            for hologram, gt_object in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                if max_samples and sample_count >= max_samples:
                    break

                hologram = hologram.to(self.device)

                # Measure inference time
                start_time = time.time()
                pred_clean, _ = model(hologram)
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                inference_time = (time.time() - start_time) * 1000  # ms

                inference_times.append(inference_time)

                # Convert to numpy
                pred_2ch = pred_clean.squeeze().cpu().numpy()  # (2, H, W)
                gt_2ch = gt_object.squeeze().numpy()           # (2, H, W)

                # Compute amplitude and phase
                pred_complex = pred_2ch[0] + 1j * pred_2ch[1]
                gt_complex = gt_2ch[0] + 1j * gt_2ch[1]

                pred_amp = np.abs(pred_complex)
                pred_phase = np.angle(pred_complex)
                gt_amp = np.abs(gt_complex)
                gt_phase = np.angle(gt_complex)

                # Compute metrics
                metrics = compute_all_metrics(
                    pred_amp, pred_phase, gt_amp, gt_phase,
                    pred_2ch[0], pred_2ch[1], gt_2ch[0], gt_2ch[1],
                )
                metrics["sample_idx"] = sample_count
                metrics["inference_time_ms"] = inference_time
                per_sample_metrics.append(metrics)

                sample_count += 1

        # Store per-sample results
        self.results[model_name] = per_sample_metrics

        # Aggregate metrics
        aggregated = self._aggregate_metrics(per_sample_metrics, inference_times)
        aggregated["model"] = model_name
        aggregated["num_samples"] = sample_count

        return aggregated

    def _aggregate_metrics(
        self,
        per_sample: list[dict[str, Any]],
        inference_times: list[float],
    ) -> dict[str, float]:
        """Aggregate per-sample metrics to mean values."""
        if not per_sample:
            return {}

        # Extract metric names (exclude non-numeric fields)
        metric_names = [k for k in per_sample[0].keys()
                        if k not in ("sample_idx", "inference_time_ms")]

        aggregated: dict[str, float] = {}
        for name in metric_names:
            values = [m[name] for m in per_sample]
            aggregated[f"{name}_mean"] = float(np.mean(values))
            aggregated[f"{name}_std"] = float(np.std(values))

        # Inference time stats
        aggregated["inference_time_ms_mean"] = float(np.mean(inference_times))
        aggregated["inference_time_ms_std"] = float(np.std(inference_times))

        return aggregated

    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_results(
        self,
        aggregated_results: list[dict[str, Any]],
        filename_prefix: str = "baseline",
    ) -> None:
        """Save results to CSV and JSON files.

        Args:
            aggregated_results: List of aggregated results per model.
            filename_prefix: Prefix for output filenames.
        """
        # Save aggregated results as CSV
        df = pd.DataFrame(aggregated_results)
        csv_path = self.output_dir / f"{filename_prefix}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved aggregated results to: {csv_path}")

        # Save per-sample results as JSON
        json_path = self.output_dir / f"{filename_prefix}_per_sample.json"
        with json_path.open("w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved per-sample results to: {json_path}")

    def generate_latex_table(
        self,
        aggregated_results: list[dict[str, Any]],
        filename: str = "baseline_table.tex",
    ) -> str:
        """Generate LaTeX table from aggregated results.

        Args:
            aggregated_results: List of aggregated results per model.
            filename: Output filename for LaTeX table.

        Returns:
            LaTeX table as string.
        """
        latex = r"""\begin{table}[h]
\centering
\caption{Baseline comparison on test set. Best values in \textbf{bold}, second-best \underline{underlined}. B/S$\downarrow$ indicates lower is better.}
\label{tab:baseline_comparison}
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Method} & \textbf{Phase PSNR} & \textbf{Phase SSIM} & \textbf{Amp. SSIM} & \textbf{B/S$\downarrow$} & \textbf{Time (ms)} & \textbf{Params (M)} \\
\midrule
"""
        for result in aggregated_results:
            model = result.get("model", "Unknown")
            phase_psnr = result.get("phase_psnr_mean", 0)
            phase_ssim = result.get("phase_ssim_mean", 0)
            amp_ssim = result.get("amp_ssim_mean", 0)
            bs_ratio = result.get("bs_ratio_mean", 0)
            inf_time = result.get("inference_time_ms_mean", 0)
            params = result.get("num_parameters", 0) / 1e6

            # Format params: "-" if 0 (for iterative methods)
            params_str = f"{params:.2f}" if params > 0 else "—"
            inf_str = f"{inf_time:.1f}" if inf_time > 0 else "—"

            latex += f"{model} & {phase_psnr:.2f} & {phase_ssim:.4f} & {amp_ssim:.4f} & {bs_ratio:.4f} & {inf_str} & {params_str} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        # Save to file
        tex_path = self.output_dir / filename
        with tex_path.open("w") as f:
            f.write(latex)
        print(f"Saved LaTeX table to: {tex_path}")

        return latex
