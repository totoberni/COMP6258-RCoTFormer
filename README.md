# COMP 6258 Differentiable Programming and Deep Learning

## Table of Contents

- [Reference Papers](#reference-papers)
- [Repo Structure](#repo-structure)
- [QuickStart](#quickstart)
  - [Iridis X (A100 GPU)](#iridis-x-a100-gpu)
  - [Adding a New Package](#adding-a-new-package)
- [Use Guide](#use-guide)
- [Further Contributions](#further-contributions)
- [Acknowledgements](#acknowledgements)

## Reference Papers

## Repo Structure

```
.
├── README.md
├── .gitignore
├── rcotformer.def               ← Container recipe (PyTorch + CUDA 12.1)
├── rcotformer.sif               ← Container image (gitignored)
├── job.slurm.example            ← Slurm template for new packages
├── rcotformer.def.example       ← Annotated container recipe template
│
└── iridis_gpu_test/             ← Package: GPU smoke test
    ├── job.slurm                ← Submit: sbatch iridis_gpu_test/job.slurm
    └── test_gpu.py
```

Each package is a self-contained directory with its own `job.slurm` and scripts. The `.sif` container image is shared at the project root. All jobs are submitted from the project root (`~/dpdl/` on Iridis).

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
sudo apptainer build rcotformer.sif rcotformer.def
```

Validate locally in WSL (requires an NVIDIA GPU on the host machine):

```bash
apptainer exec --nv \
    --bind /usr/lib/wsl:/usr/lib/wsl \
    --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    rcotformer.sif python3 iridis_gpu_test/test_gpu.py
```

> **WSL2 note:** The `--bind /usr/lib/wsl:/usr/lib/wsl` and `LD_LIBRARY_PATH` overrides are required because WSL2 uses a thin `libcuda` shim that communicates with the Windows GPU driver via `/dev/dxg`. Apptainer's `--nv` flag alone does not inject the D3D12/dxcore libraries the shim depends on. This is only needed on WSL — Iridis compute nodes have native NVIDIA drivers and `--nv` works as-is.

**2. Upload to Iridis** (first time — includes the large `.sif`):

```bash
scp rcotformer.sif iridis-x:~/dpdl/
scp iridis_gpu_test/test_gpu.py iridis_gpu_test/job.slurm iridis-x:~/dpdl/iridis_gpu_test/
```

Subsequent uploads (scripts only):

```bash
scp iridis_gpu_test/test_gpu.py iridis_gpu_test/job.slurm iridis-x:~/dpdl/iridis_gpu_test/
```

**3. Submit the job:**

```bash
ssh iridis-x
cd ~/dpdl
sbatch iridis_gpu_test/job.slurm
```

**4. Monitor:**

```bash
squeue -u $(whoami)                          # Job status
cat iridis_gpu_test/slurm_<job_id>.out       # Output after completion
scancel <job_id>                             # Cancel if needed
seff <job_id>                                # Post-run efficiency stats
```

A successful run prints the GPU name, memory, and a matrix-multiply result confirming CUDA works end-to-end.

> **Tip:** Use `sinfo -p gpu` to check available GPU nodes. Compute nodes lack `fusermount`, which is why the slurm script uses `--unsquash`.

### Adding a New Package

1. Create the package directory and copy the templates:
   ```bash
   mkdir -p <your_package>
   cp job.slurm.example <your_package>/job.slurm
   ```

2. Edit `<your_package>/job.slurm` — replace every `<PKG_NAME>` with your directory name and adjust resources (GPU count, memory, wall time).

3. Add your Python scripts to `<your_package>/`.

4. If you need extra Python packages, edit `rcotformer.def`, rebuild the `.sif`, and re-upload it. See `rcotformer.def.example` for guidance.

5. Upload and submit:
   ```bash
   scp <your_package>/*.py <your_package>/job.slurm iridis-x:~/dpdl/<your_package>/
   ssh iridis-x "cd ~/dpdl && sbatch <your_package>/job.slurm"
   ```

## Use Guide

## Further Contributions

## Acknowledgements

We gratefully acknowledge the University of Southampton's ECS faculty for granting access to the Iridis X high-performance computing cluster, which provides the A100 GPU resources used for training and evaluation in this project.
