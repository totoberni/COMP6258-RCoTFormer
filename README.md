# COMP 6258 Differentiable Programming and Deep Learning

## Table of Contents

- [Reference Papers](#reference-papers)
- [Repo Structure](#repo-structure)
- [QuickStart](#quickstart)
  - [Iridis X (A100 GPU)](#iridis-x-a100-gpu)
- [Use Guide](#use-guide)
- [Further Contributions](#further-contributions)
- [Acknowledgements](#acknowledgements)

## Reference Papers

## Repo Structure

```
.
├── README.md
└── iridis/
    ├── test_gpu.py          ← PyTorch + CUDA smoke test
    └── job.slurm            ← Slurm submission script (A100 partition)
```

## QuickStart

### Iridis X (A100 GPU)

Iridis X is the University of Southampton's HPC cluster. The steps below verify that you can submit a GPU job and run PyTorch on an A100 node.

**Prerequisites:** SSH access to Iridis X. Add this to `~/.ssh/config`:

```
Host iridis-x
    HostName loginX002.iridis.soton.ac.uk
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

**1. Upload the test files** (from the repo root):

```bash
scp iridis/test_gpu.py iridis/job.slurm iridis-x:~/rcotformer/iridis/
```

**2. Submit the job:**

```bash
ssh iridis-x
cd ~/rcotformer
sbatch iridis/job.slurm
```

On first run the job creates a virtual environment at `~/venvs/rcotformer` and installs PyTorch with CUDA 12.1 support. Subsequent runs reuse the existing environment.

**3. Monitor:**

```bash
squeue -u $(whoami)                   # Job status
cat iridis/slurm_<job_id>.out         # Output after completion
scancel <job_id>                      # Cancel if needed
```

A successful run prints the GPU name, memory, and a matrix-multiply result confirming CUDA works end-to-end.

> **Tip:** Use `sinfo -p gpu` to check available GPU nodes and `seff <job_id>` for post-run efficiency stats.

## Use Guide

## Further Contributions

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster, which provides the A100 GPU resources used for training and evaluation in this project.
