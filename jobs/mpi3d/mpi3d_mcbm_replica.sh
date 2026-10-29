#!/bin/bash
#SBATCH --job-name=mpi3d-mcbm-replica
#SBATCH --partition=gpu_a100_il
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
set -x

SEED="${1:-42}"
HOME="/home/ma/ma_ma/ma_iadam"
PROJECT="$HOME/projects/mcbm-derm7pt"
PYTHON="$HOME/.conda/envs/minimal_cbm/bin/python"

cd "$PROJECT"

"$PYTHON" bin/train.py mpi3d-mcbm-replica -s "$SEED"
"$PYTHON" bin/test.py mpi3d-mcbm-replica -s "$SEED"
