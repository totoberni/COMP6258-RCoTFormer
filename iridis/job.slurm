#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=iridis/slurm_%j.out
#SBATCH --error=iridis/slurm_%j.err

# ── Load modules ─────────────────────────────────────────────
module purge
module load python/3.11.3
module load cuda/12.1
module load cudnn/8.9.7-cuda12

# ── Virtual environment ─────────────────────────────────────
VENV_DIR="$HOME/venvs/rcotformer"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    source "$VENV_DIR/bin/activate"
fi

# ── Run ──────────────────────────────────────────────────────
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
nvidia-smi

python iridis/test_gpu.py
