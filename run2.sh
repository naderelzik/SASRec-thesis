#!/bin/bash
#SBATCH --job-name=ml-1m200normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#cd /home/zik/SASRec

srun python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --eval_mode=sample --sample_mode=normal --num_epochs=200 --dropout_rate=0.2 --device=cuda