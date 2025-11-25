#!/bin/bash
#SBATCH --job-name=SASfoursquarerandom_chrono
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# === Configuration ===
DATASET="foursquare"
MAXLEN_LIST=(20 45 100)
BASE_DIR="/home/zik/SASRec/foursquare_final_v3/random_chrono"
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
        --lr=1e-5 \
        --sample_mode=random_chrono \
        --num_epochs=200 \
        --dropout_rate=0.2 \
        --device=cuda
done