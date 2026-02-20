#!/bin/bash
# run_baseline_tests.sh
set -e

# Change directory to the root of the holopaswin project
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

# Activate virtual environment
source .venv/bin/activate

echo "==================================="
echo "  Starting Baseline Tests"
echo "==================================="
echo $(date)

# Train U-Net
echo ""
echo ">>> Training U-Net baseline..."
python scripts/train_baselines.py --model unet --epochs 5 --output-dir results/baseline_comparison

# Train ResNet-U-Net
echo ""
echo ">>> Training ResNet-U-Net baseline..."
python scripts/train_baselines.py --model resnet_unet --epochs 5 --output-dir results/baseline_comparison

# Evaluate all baselines against HoloPASWIN Experiment 12
echo ""
echo ">>> Evaluating all models..."
python scripts/eval_baselines.py \
    --test-data ../hologen/test-dataset-224 \
    --output-dir results/baseline_comparison \
    --holopaswin-ckpt results/experiment12/holopaswin_exp12.pth \
    --unet-ckpt results/baseline_comparison/unet_best.pth \
    --resnet-unet-ckpt results/baseline_comparison/resnet_unet_best.pth

echo ""
echo "==================================="
echo "  All Tasks Completed Successfully!"
echo "==================================="
echo $(date)
