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

Validate locally in WSL (requires an NVIDIA GPU on the host machine):

```bash
apptainer exec --nv \
    --bind /usr/lib/wsl:/usr/lib/wsl \
    --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    rcotformer.sif python3 iridis/test_gpu.py
```

> **WSL2 note:** The `--bind /usr/lib/wsl:/usr/lib/wsl` and `LD_LIBRARY_PATH` overrides are required because WSL2 uses a thin `libcuda` shim that communicates with the Windows GPU driver via `/dev/dxg`. Apptainer's `--nv` flag alone does not inject the D3D12/dxcore libraries the shim depends on. This is only needed on WSL — Iridis compute nodes have native NVIDIA drivers and `--nv` works as-is.

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
