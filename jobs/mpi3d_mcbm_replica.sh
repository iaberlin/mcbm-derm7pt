#!/bin/bash
#SBATCH --job-name=mpi3d-mcbm-replica
#SBATCH --partition=gpu_a100_short
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
set -x

HOME="/home/ma/ma_ma/ma_iadam"
PROJECT="$HOME/projects/mcbm-derm7pt"
PYTHON="$HOME/.conda/envs/minimal_cbm/bin/python"

cd "$PROJECT"

# echo "Python:"
# "$PYTHON" --version

# echo "Python path:"
# "$PYTHON" -c "import sys; print(sys.executable)"

# echo "Torch:"
# "$PYTHON" -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# echo "Starting MCBM smoke test"
# "$PYTHON" bin/train.py mpi3d-mcbm-smoke -s 42
# """

"$PYTHON" bin/train.py mpi3d-mcbm-replica -s 42