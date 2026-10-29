#!/bin/bash
set -euo pipefail

CONFIG="${1:?Bitte Config angeben}"
SEED="${2:-42}"
SPLIT_SEED="${3:-42}"

PROJECT="/home/ma/ma_ma/ma_iadam/projects/mcbm-derm7pt"
PYTHON="/home/ma/ma_ma/ma_iadam/.conda/envs/minimal_cbm/bin/python"

cd "$PROJECT"

# Metadaten aus der Konfiguration lesen
META=$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

config = sys.argv[1]
files = list(Path("configs").rglob(f"{config}.yaml"))

if len(files) != 1:
    raise ValueError(f"Config nicht eindeutig gefunden: {config}")

with open(files[0]) as f:
    cfg = yaml.safe_load(f)

print(
    cfg["data"]["dataset"],
    cfg["model"]["model_type"],
    cfg["model"]["encoder"]["arch"],
    sep="\t"
)
PY
)

IFS=$'\t' read -r DATASET MODEL BACKBONE <<< "$META"

JOB_NAME="${DATASET}-${MODEL}-${BACKBONE}-${SEED}"

LOG_DIR="$PROJECT/logs/$DATASET/$MODEL/$BACKBONE/seed_$SEED"
mkdir -p "$LOG_DIR"

# Hier den tatsächlichen Namen deines SLURM-Skripts einsetzen
SLURM_SCRIPT="jobs/mpi3d/mpi3d_eval.sh"

sbatch \
    --job-name="$JOB_NAME" \
    --output="$LOG_DIR/${JOB_NAME}-%j.out" \
    --error="$LOG_DIR/${JOB_NAME}-%j.err" \
    "$SLURM_SCRIPT" "$CONFIG" "$SEED" "$SPLIT_SEED"
