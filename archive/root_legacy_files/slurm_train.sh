#!/bin/bash
#SBATCH --job-name=fr-gloss-train
#SBATCH --partition=interactive10
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activer l'environnement virtuel
source ~/fr_gloss_env/bin/activate

# Lancer l'entraînement avec GPU
cd ~/fr_gloss_project
python train_model.py --dataset dataset_5000.csv --epochs 30 --batch-size 32

echo "=== Job finished at $(date) ==="
