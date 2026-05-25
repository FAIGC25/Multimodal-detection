#!/bin/bash
#SBATCH --job-name=compare_depth_models
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/compare_models_%j.log
#SBATCH --error=logs/compare_models_%j.err

# Compare SIDA Depth vs EfficientNet-B0 Student models

echo "Starting model comparison on $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================"

# Create logs directory
mkdir -p logs

# Navigate to project directory
cd /mnt/tank/scratch/dstoronkin/MultimodalDetection || exit

# Make sure we're on dev branch
git fetch origin
git checkout dev
git pull origin dev

echo "Running on branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"
echo "================================"

# Run comparison script
# Test on first 50 samples for quick validation
python scripts/compare_depth_models.py \
    --dataset /mnt/tank/scratch/dstoronkin/depth \
    --output results/comparison_results_$(date +%Y%m%d_%H%M%S).json \
    --num-samples 50

echo "================================"
echo "Model comparison completed on $(date)"
