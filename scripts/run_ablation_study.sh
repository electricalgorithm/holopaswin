#!/bin/bash
# Sequential ablation study execution script
# Run this and leave - it will complete in ~20 hours

set -e  # Exit on error

# Create log file with timestamp
LOG_FILE="ablation_study_$(date +%Y%m%d_%H%M%S).log"

echo "Starting ablation study at $(date)" | tee "$LOG_FILE"
echo "This will take approximately 20 hours to complete." | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate virtual environment
source ../.venv/bin/activate

# Run sequential ablation script (tee outputs to both console and file)
python run_all_ablations_sequential.py \
    --output-dir ../results/ablation_final 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Ablation study completed at $(date)" | tee -a "$LOG_FILE"
echo "Results are in: ../results/ablation_final/" | tee -a "$LOG_FILE"
echo "Shell log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
