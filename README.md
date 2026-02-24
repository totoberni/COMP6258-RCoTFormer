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
├── rcotformer.sif               ← Apptainer container image (gitignored)
└── iridis/
    ├── rcotformer.def           ← Container recipe (PyTorch + CUDA 12.1)
    ├── test_gpu.py              ← PyTorch + CUDA smoke test
    └── job.slurm                ← Slurm submission script (A100 partition)
```

## QuickStart

### Iridis X (A100 GPU)

Iridis X is the University of Southampton's HPC cluster. All dependencies (PyTorch, CUDA) are packaged in an Apptainer container so we don't depend on cluster-installed modules.

**Prerequisites:**

- WSL2 with `apptainer` installed (`sudo apt install apptainer`)
- SSH access to Iridis X — add this to `~/.ssh/config`:

```
Host iridis-x
    HostName loginX002.iridis.soton.ac.uk
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

**1. Build the container** (once, locally in WSL):

```bash
cd iridis
sudo apptainer build ../rcotformer.sif rcotformer.def
```

Validate locally (GPU check will be skipped without a GPU):

```bash
apptainer exec rcotformer.sif python3 -c "import torch; print(torch.__version__)"
```

**2. Upload to Iridis** (first time — includes the large `.sif`):

```bash
scp rcotformer.sif iridis-x:~/rcotformer/
scp iridis/test_gpu.py iridis/job.slurm iridis-x:~/rcotformer/iridis/
```

Subsequent uploads (scripts only):

```bash
scp iridis/test_gpu.py iridis/job.slurm iridis-x:~/rcotformer/iridis/
```

**3. Submit the job:**

```bash
ssh iridis-x
cd ~/rcotformer
sbatch iridis/job.slurm
```

**4. Monitor:**

```bash
squeue -u $(whoami)                   # Job status
cat iridis/slurm_<job_id>.out         # Output after completion
scancel <job_id>                      # Cancel if needed
```

A successful run prints the GPU name, memory, and a matrix-multiply result confirming CUDA works end-to-end.

> **Tip:** Use `sinfo -p gpu` to check available GPU nodes and `seff <job_id>` for post-run efficiency stats. Compute nodes lack `fusermount`, which is why the slurm script uses `--unsquash`.

## Use Guide

## Further Contributions

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster, which provides the A100 GPU resources used for training and evaluation in this project.
