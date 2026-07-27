# Conda Inside Singularity Containers (Slurm Jobs)

When Slurm jobs run inside Singularity containers, `/home` is mounted **read-only**
and the container's internal conda package cache (`/opt/miniforge3/pkgs`) is also
read-only. Scripts that use conda MUST handle this.

## Required Environment Setup

Add these at the TOP of every script that uses conda inside a Slurm container:

```bash
export CONDA_NO_PLUGINS=true              # Disables ToS plugin (needs ~/.conda/ which is read-only)
export CONDA_PKGS_DIRS=/tmp/conda_pkgs_$$ # Redirect package cache to writable location
mkdir -p "$CONDA_PKGS_DIRS"
```

## Why Each Variable Is Needed

| Variable | Purpose |
|----------|---------|
| `CONDA_NO_PLUGINS=true` | The `conda-anaconda-tos` plugin requires ToS acceptance stored in `~/.conda/tos/`, but `/home` is read-only |
| `CONDA_PKGS_DIRS` | Default package cache is inside the read-only container image; redirect to `/tmp` or work_dir |

## Additional Recommendations

- Use `--solver=classic` flag for conda create/install (libmamba solver may not be available)
- Use `--override-channels -c conda-forge -c bioconda` (this EXACT order — conda-forge first, bioconda second; reversing causes solver to pick ancient packages)
- Set `export HOME=/tmp/home_$$; mkdir -p $HOME` if conda still complains about home directory
- Create conda env with `--prefix <work_dir>/conda_env` (inside writable work_dir)

## Conda Activation Pattern

```bash
#!/bin/bash
# Initialize conda shell hooks
eval "$(conda shell.bash hook)"
# OR explicit path (more reliable in containers):
source /opt/miniforge3/etc/profile.d/conda.sh

# Activate by PREFIX PATH (most reliable)
conda activate /data1/user/work_dir/conda_envs/myenv

# NEVER use this — virtualenv syntax, FAILS for conda prefix envs:
# source /path/to/env/bin/activate  ← WRONG
```

## Full Template: Conda Install in Slurm Container

```bash
#!/bin/bash
#SBATCH --job-name=conda_setup
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# === CONTAINER WORKAROUNDS ===
export CONDA_NO_PLUGINS=true
export CONDA_PKGS_DIRS=/tmp/conda_pkgs_$$
export HOME=/tmp/home_$$
mkdir -p "$CONDA_PKGS_DIRS" "$HOME"

# === INIT CONDA ===
eval "$(conda shell.bash hook)"

# === CREATE ENVIRONMENT ===
CONDA_PREFIX="${WORK_DIR}/conda_envs/myenv"
conda create --prefix "$CONDA_PREFIX" \
    --solver=classic \
    --override-channels -c conda-forge -c bioconda \
    python=3.11 numpy pandas -y

# === ACTIVATE AND VERIFY ===
conda activate "$CONDA_PREFIX"
python -c "import numpy; print(f'numpy {numpy.__version__} OK')"
```

## Nested Singularity (Building Containers Inside IrisAI)

IrisAI runs inside a Singularity container. When a Slurm job runs another
`singularity build` or `singularity exec`, the outer container's bind mounts leak
into the inner context and cause failures.

**Fix:** At the TOP of any Slurm job that invokes singularity:
```bash
export SINGULARITY_BIND=''
export APPTAINER_BIND=''
```

This clears inherited bind mounts. The inner singularity command uses only its own `-B` flags.

## Fakeroot Builds (Container Construction)

Under fakeroot, `/root` is NOT writable. Conda/mamba fails immediately unless:
```bash
# Set at TOP of %post in .def file (before ANY conda/mamba call):
export HOME=/opt/mambaforge_home
export CONDA_PKGS_DIRS=/opt/mambaforge/pkgs
export CONDA_ENVS_PATH=/opt/mambaforge/envs
export XDG_CACHE_HOME=/opt/mambaforge_home/.cache
mkdir -p /opt/mambaforge_home /opt/mambaforge_home/.cache /opt/mambaforge/pkgs /opt/mambaforge/envs
```

Why XDG_CACHE_HOME: libmamba v2 ignores CONDA_PKGS_DIRS for its internal shard cache
and follows XDG Base Directory spec → `~/.cache/conda/pkgs/cache/shards`.
