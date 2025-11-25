#!/bin/bash
#SBATCH --job-name=SASxlongnormal
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# === Configuration ===
DATASET="xlong"
MAXLEN_LIST=(100 300 500)
BASE_DIR="/home/zik/SASRec/xlong2/normal"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  # Get current timestamp
LOG_DIR="${BASE_DIR}/${DATASET}_${TIMESTAMP}"  # Unique directory per run

# === Create log directory ===
mkdir -p "$LOG_DIR"                      # Create timestamped directory

# === Loop through maxlen values ===
for MAXLEN in "${MAXLEN_LIST[@]}"; do
    LOG_FILE="${LOG_DIR}/${DATASET}_${MAXLEN}.log"
    ERR_FILE="${LOG_DIR}/${DATASET}_${MAXLEN}.err"

    echo "Running maxlen=${MAXLEN} — logs: $LOG_FILE"

    srun --output="$LOG_FILE" --error="$ERR_FILE" \
    python main.py \
        --dataset=$DATASET \
        --train_dir=default \
        --maxlen=$MAXLEN \
        --eval_mode=sample \
        --lr=0.0001 \
        --sample_mode=normal \
        --num_epochs=100 \
        --dropout_rate=0.2 \
        --device=cuda
done