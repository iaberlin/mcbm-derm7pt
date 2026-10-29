#!/bin/bash
set -euo pipefail

# Konfigurationen (ohne .yaml)
CONFIGS=(
    "mpi3d-vanilla-res18-diagnostic"
    "mpi3d-vanilla-res20-diagnostic"
    "mpi3d-cbm-res18-diagnostic"
    "mpi3d-cbm-res20-diagnostic"
    "mpi3d-mcbm-res18-diagnostic"
    "mpi3d-mcbm-res20-diagnostic"
)

# Train-Seeds
SEEDS=(5 16 42 67 102)

# Gleicher Datensplit für alle Runs
SPLIT_SEED=42

# Ins Projektverzeichnis wechseln
PROJECT="/home/ma/ma_ma/ma_iadam/projects/mcbm-derm7pt"
cd "$PROJECT"

# Alle Kombinationen einreichen
for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        echo "Submitting: $CONFIG | Train-Seed: $SEED | Split-Seed: $SPLIT_SEED"

        bash jobs/submit.sh "$CONFIG" "$SEED" "$SPLIT_SEED"

    done
done

echo "Alle 30 Jobs wurden eingereicht."
