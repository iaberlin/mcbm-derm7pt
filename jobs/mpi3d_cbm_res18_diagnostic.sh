#!/bin/bash
#SBATCH --job-name=mpi3d-cbm-diagnostic-r18-v3
#SBATCH --partition=gpu_a100_il
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=25G
#SBATCH --output=logs/cbm/%x-%j.out
#SBATCH --error=logs/cbm/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
set -x

SEED="${1:-42}"
SPLIT_SEED=42
HOME="/home/ma/ma_ma/ma_iadam"
PROJECT="$HOME/projects/mcbm-derm7pt"
PYTHON="$HOME/.conda/envs/minimal_cbm/bin/python"

cd "$PROJECT"

echo "split seed: $SPLIT_SEED"
echo "train seed: $SEED"

"$PYTHON" bin/train.py mpi3d-cbm-res18-diagnostic --seed "$SEED" --split_seed "$SPLIT_SEED"
"$PYTHON" bin/test.py mpi3d-cbm-res18-diagnostic -s "$SEED" --split_seed "$SPLIT_SEED"
