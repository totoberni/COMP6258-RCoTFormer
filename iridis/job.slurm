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

# ── Load Apptainer ───────────────────────────────────────────
module load apptainer

# ── Run inside container ─────────────────────────────────────
# --nv: exposes host NVIDIA drivers + GPUs to the container
# --unsquash: required on compute nodes (no fusermount)
# --bind $PWD: makes project files visible inside the container
echo "Node: $(hostname)"
nvidia-smi

apptainer exec --nv --unsquash --bind $PWD rcotformer.sif \
    python3 iridis/test_gpu.py
