#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu_a100_il
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --export=NONE

set -euo pipefail

CONFIG="${1:?Bitte Konfiguration angeben}"
SEED="${2:-42}"
SPLIT_SEED="${3:-42}"

HOME="/home/ma/ma_ma/ma_iadam"
PROJECT="$HOME/projects/mcbm-derm7pt"
PYTHON="$HOME/.conda/envs/minimal_cbm/bin/python"

cd "$PROJECT"

# YAML-Datei finden
CONFIG_PATH=$(find configs -type f -name "${CONFIG}.yaml" -print -quit)
if [ -z "$CONFIG_PATH" ]; then
    echo "Konfiguration nicht gefunden: $CONFIG"
    exit 1
fi

# Metadaten aus YAML auslesen
META=$("$PYTHON" - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

print(
    cfg["data"]["dataset"],
    cfg["model"]["model_type"],
    cfg["model"]["encoder"]["arch"]
)
PY
)

read -r DATASET MODEL BACKBONE <<< "$META"

# Dynamischen Jobnamen erzeugen
JOB_NAME="${DATASET}-${MODEL}-${BACKBONE}-${SEED}-${SLURM_JOB_ID}"


# Jobnamen in SLURM aktualisieren
scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" \
    || echo "Warnung: SLURM-Jobname konnte nicht geändert werden."

set -x

echo "Job: $JOB_NAME"
echo "Config: $CONFIG"
echo "Train seed: $SEED"
echo "Split seed: $SPLIT_SEED"

"$PYTHON" bin/train.py "$CONFIG" -s "$SEED" --split_seed "$SPLIT_SEED"
"$PYTHON" bin/test.py "$CONFIG" -s "$SEED" --split_seed "$SPLIT_SEED"
