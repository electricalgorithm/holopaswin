"""Evaluation script for baseline comparison.

Evaluates all methods on the test dataset and generates comparison tables:
- ASM-only (dirty baseline)
- Gerchberg-Saxton (50 and 100 iterations)
- U-Net baseline
- ResNet-U-Net
- HoloPASWIN (ours)

Usage:
    python scripts/eval_baselines.py \
        --test-data ../hologen/test-dataset-224 \
        --output-dir results/baseline_comparison \
        --holopaswin-ckpt results/experiment9/holopaswin_exp9.pth
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from holopaswin.baselines.gerchberg_saxton import GerchbergSaxton
from holopaswin.baselines.hrnet import HRNet
from holopaswin.baselines.unet_baseline import UNetBaseline
from holopaswin.dataset import HoloDataset
from holopaswin.evaluation.evaluator import BaselineEvaluator
from holopaswin.model import HoloPASWIN
from holopaswin.propagator import AngularSpectrumPropagator
from holopaswin.resnet_unet import ResNetUNet

# Optics configuration (same as training)
IMG_SIZE = 224
WAVELENGTH = 532e-9  # 532 nm
PIXEL_SIZE = 4.65e-6  # 4.65 µm
Z_DIST = 0.02  # 20 mm


class ASMBaseline(torch.nn.Module):
    """ASM-only baseline (dirty reconstruction, no learning)."""

    def __init__(self, img_size: int, wavelength: float, pixel_size: float, z_distance: float) -> None:  # noqa: D107
        super().__init__()
        self.propagator = AngularSpectrumPropagator((img_size, img_size), wavelength, pixel_size, z_distance)

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Just ASM back-propagation, no refinement."""
        amplitude = torch.sqrt(hologram + 1e-8)
        complex_holo = torch.complex(amplitude, torch.zeros_like(amplitude))
        dirty_complex = self.propagator(complex_holo, backward=True)
        dirty_2ch = torch.cat([dirty_complex.real, dirty_complex.imag], dim=1)
        return dirty_2ch, dirty_2ch


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:  # noqa: PLR0915, PLR0912
    """Run baseline comparison evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate baseline methods")
    parser.add_argument(
        "--test-data",
        type=str,
        default="../hologen/test-dataset-224",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--holopaswin-ckpt",
        type=str,
        default="results/experiment9/holopaswin_exp9.pth",
        help="Path to HoloPASWIN checkpoint",
    )
    parser.add_argument(
        "--unet-ckpt",
        type=str,
        default=None,
        help="Path to U-Net checkpoint (optional, will skip if not provided)",
    )
    parser.add_argument(
        "--resnet-unet-ckpt",
        type=str,
        default=None,
        help="Path to ResNet-U-Net checkpoint (optional, will skip if not provided)",
    )
    parser.add_argument(
        "--hrnet-ckpt",
        type=str,
        default=None,
        help="Path to HRNet checkpoint (optional, will skip if not provided)",
    )
    parser.add_argument(
        "--gs-iterations",
        type=int,
        nargs="+",
        default=[50, 100],
        help="GS iteration counts to test (default: 50 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test samples (for quick testing)",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load test dataset
    print(f"Loading test dataset from {args.test_data}...")
    try:
        test_dataset = HoloDataset(data_dir=args.test_data, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading dataset: {e}")
        return

    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize evaluator
    evaluator = BaselineEvaluator(device=device, output_dir=output_dir)

    # Results accumulator
    all_results: list[dict] = []

    # 1. ASM-only baseline (dirty reconstruction)
    print("\n" + "=" * 60)
    print("Evaluating: ASM (dirty baseline)")
    print("=" * 60)
    asm_model = ASMBaseline(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST)
    asm_results = evaluator.evaluate_model(asm_model, test_loader, "ASM (dirty)", args.max_samples)
    asm_results["num_parameters"] = 0
    all_results.append(asm_results)

    # 2. Gerchberg-Saxton
    for n_iter in args.gs_iterations:
        print("\n" + "=" * 60)
        print(f"Evaluating: GS ({n_iter} iterations)")
        print("=" * 60)
        gs_model = GerchbergSaxton(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, iterations=n_iter)
        gs_results = evaluator.evaluate_model(gs_model, test_loader, f"GS ({n_iter} iter)", args.max_samples)
        gs_results["num_parameters"] = 0
        all_results.append(gs_results)

    # 3. U-Net baseline
    if args.unet_ckpt and Path(args.unet_ckpt).exists():
        print("\n" + "=" * 60)
        print("Evaluating: U-Net baseline")
        print("=" * 60)
        unet_model = UNetBaseline(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, residual_mode=True)
        unet_model.load_state_dict(torch.load(args.unet_ckpt, map_location=device, weights_only=True))
        unet_results = evaluator.evaluate_model(unet_model, test_loader, "U-Net", args.max_samples)
        unet_results["num_parameters"] = count_parameters(unet_model)
        all_results.append(unet_results)
    else:
        print(f"\nSkipping U-Net (checkpoint not found: {args.unet_ckpt})")

    # 4. ResNet-U-Net
    if args.resnet_unet_ckpt and Path(args.resnet_unet_ckpt).exists():
        print("\n" + "=" * 60)
        print("Evaluating: ResNet-U-Net")
        print("=" * 60)
        resnet_model = ResNetUNet(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, residual_mode=True)
        resnet_model.load_state_dict(torch.load(args.resnet_unet_ckpt, map_location=device, weights_only=True))
        resnet_results = evaluator.evaluate_model(resnet_model, test_loader, "ResNet-U-Net", args.max_samples)
        resnet_results["num_parameters"] = count_parameters(resnet_model)
        all_results.append(resnet_results)
    else:
        print(f"\nSkipping ResNet-U-Net (checkpoint not found: {args.resnet_unet_ckpt})")

    # 4.5. HRNet
    if args.hrnet_ckpt and Path(args.hrnet_ckpt).exists():
        print("\n" + "=" * 60)
        print("Evaluating: HRNet")
        print("=" * 60)
        hrnet_model = HRNet(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, residual_mode=True)
        hrnet_model.load_state_dict(torch.load(args.hrnet_ckpt, map_location=device, weights_only=True))
        hrnet_results = evaluator.evaluate_model(hrnet_model, test_loader, "HRNet", args.max_samples)
        hrnet_results["num_parameters"] = count_parameters(hrnet_model)
        all_results.append(hrnet_results)
    else:
        print(f"\nSkipping HRNet (checkpoint not found: {args.hrnet_ckpt})")

    # 5. HoloPASWIN (ours)
    if Path(args.holopaswin_ckpt).exists():
        print("\n" + "=" * 60)
        print("Evaluating: HoloPASWIN (ours)")
        print("=" * 60)
        holopaswin = HoloPASWIN(
            IMG_SIZE,
            WAVELENGTH,
            PIXEL_SIZE,
            Z_DIST,
            use_pretrained=True,
            residual_mode=True,  # experiment9 uses residual mode
        )
        holopaswin.load_state_dict(torch.load(args.holopaswin_ckpt, map_location=device, weights_only=True))
        holopaswin_results = evaluator.evaluate_model(holopaswin, test_loader, "HoloPASWIN (ours)", args.max_samples)
        holopaswin_results["num_parameters"] = count_parameters(holopaswin)
        all_results.append(holopaswin_results)
    else:
        print(f"\nWARNING: HoloPASWIN checkpoint not found: {args.holopaswin_ckpt}")

    # Save results
    evaluator.save_results(all_results, "baseline")

    # Generate LaTeX table
    evaluator.generate_latex_table(all_results)

    # Print summary
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 100)

    header = f"{'Method':<20} | {'Phase PSNR':>12} | {'Phase SSIM':>12} | {'Amp SSIM':>12} | {'B/S Ratio':>10} | {'Time (ms)':>10} | {'Params (M)':>10}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        model = r.get("model", "?")
        phase_psnr = r.get("phase_psnr_mean", 0)
        phase_ssim = r.get("phase_ssim_mean", 0)
        amp_ssim = r.get("amp_ssim_mean", 0)
        bs_ratio = r.get("bs_ratio_mean", 0)
        inf_time = r.get("inference_time_ms_mean", 0)
        params = r.get("num_parameters", 0) / 1e6

        params_str = f"{params:.2f}" if params > 0 else "—"
        print(
            f"{model:<20} | {phase_psnr:>12.2f} | {phase_ssim:>12.4f} | {amp_ssim:>12.4f} | {bs_ratio:>10.4f} | {inf_time:>10.1f} | {params_str:>10}"
        )

    print("=" * 100)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
