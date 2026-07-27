---
name: container-building
description: Build Singularity/Apptainer containers — .def files, fakeroot builds,
  nested singularity, Docker Hub pulls, conda inside containers, custom environments
allowed_tools:
  - submit_slurm_job
  - slurm_monitor_job
  - write_text_file
  - edit_file
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - execute_dynamic_task
  - batch
  - register_software
  - query_software
model: null
max_iterations: 30
guardrails:
- BEFORE building: ALWAYS call query_software to check if a suitable container already exists
- AFTER successful build: ALWAYS call register_software with source="container" — no exceptions
- ALWAYS set SINGULARITY_BIND='' and APPTAINER_BIND='' in Slurm job scripts (nested singularity fix)
- ALWAYS set fakeroot env vars (HOME, CONDA_PKGS_DIRS, etc.) at TOP of %post before any conda/mamba
- NEVER build on login node — always submit_slurm_job
- Use --fakeroot (user cannot use sudo/root)
- Default build resources — 16 CPUs, 64G, 2 hours
---

# Container Building

Build custom Singularity/Apptainer containers on the HPC cluster. Handles
.def file creation, fakeroot builds (no root needed), conda/mamba inside
containers, Docker Hub base images, and the nested-singularity challenge
of building from within IrisAI's own container.

## When to Use This Skill

**Triggers:**
- "Build a container" / "Create a Singularity image"
- "Write a .def file" / "Container definition"
- "Package my environment as a container"
- "Pull from Docker Hub" / "Convert Docker to Singularity"
- "I need a custom container with..."
- "Fakeroot build"

**NOT for (route elsewhere):**
- "Install packages" (into existing env, not container) → software-management
- "Run inside a container" (existing .sif) → code-execution or hpc-submit-job
- "What containers exist?" → get_environment_info

## Complete Workflow

### Step 0: Check Registry (ALWAYS do this FIRST)

Before writing a .def file or pulling a container, check if one already exists:

```
query_software(search="{container_name_or_purpose}")
query_software(search="container")
```

If a matching container is found:
- Tell the user it's already available at the registered path
- Provide the .sif path and `singularity exec` instructions
- SKIP the build entirely

If not found, proceed to Step 1.

### Step 1: Context & Discovery

1. **Check existing containers:**
   → get_environment_info('category:containers')
   → find_files(pattern="*.sif", directory=work_dir)
   → A suitable container may already exist

2. **Understand requirements:**
   → What software needs to be inside?
   → What base image? (ubuntu, nvidia/cuda, continuumio/miniconda3)
   → Does it need GPU support? (nvidia/cuda base required)
   → Does it need conda/mamba? (special handling for fakeroot)

### Step 2: Write the .def File

```
write_text_file(
    path="{work_dir}/container.def",
    content="..."
)
```

Template:
```
Bootstrap: docker
From: ubuntu:22.04

%labels
    Author {username}
    Description {what this container is for}

%post
    # === FAKEROOT WORKAROUNDS (MANDATORY) ===
    export HOME=/opt/container_home
    export CONDA_PKGS_DIRS=/opt/conda/pkgs
    export CONDA_ENVS_PATH=/opt/conda/envs
    export XDG_CACHE_HOME=/opt/container_home/.cache
    mkdir -p /opt/container_home /opt/container_home/.cache /opt/conda/pkgs /opt/conda/envs

    # System packages
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget curl git ca-certificates \
        && rm -rf /var/lib/apt/lists/*

    # Install Miniforge (conda/mamba)
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3
    rm Miniforge3-Linux-x86_64.sh

    # Install packages
    /opt/miniforge3/bin/mamba install -y -c conda-forge -c bioconda \
        python=3.11 numpy pandas {packages}

    # Cleanup
    /opt/miniforge3/bin/mamba clean -afy

%environment
    export PATH=/opt/miniforge3/bin:$PATH

%runscript
    exec "$@"
```

### Step 3: Write Build Job Script

**CRITICAL:** Must clear inherited bind mounts from IrisAI's outer container.

```bash
#!/bin/bash
set -euo pipefail

# === CLEAR NESTED SINGULARITY BINDS (MANDATORY) ===
export SINGULARITY_BIND=''
export APPTAINER_BIND=''

# === BUILD CACHE DIRS ===
export APPTAINER_TMPDIR={work_dir}/.cache/singularity_tmp
export APPTAINER_CACHEDIR={work_dir}/.cache/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# === BUILD ===
SINGULARITY="${SINGULARITY:-singularity}"
$SINGULARITY build --fakeroot --no-cleanup \
    {work_dir}/{output_name}.sif \
    {work_dir}/container.def

echo "BUILD COMPLETE: {work_dir}/{output_name}.sif"
ls -lh {work_dir}/{output_name}.sif
```

### Step 4: Submit Build Job

```
submit_slurm_job(
    job_name="build_{container_name}",
    script_content="{build_script}",
    work_dir="{work_dir}",
    time_limit="02:00:00",
    memory="64G",
    cpus=16,
    partition="cpu"
)
```

### Step 5: Verify

After build completes:
1. Check .sif file exists and has reasonable size
2. Test basic functionality:
   ```
   execute_dynamic_task(
       commands=["$SINGULARITY exec {sif_path} python3 --version"]
   )
   ```
3. Report to user: container path, size, key contents

### Step 6: Register Container (MANDATORY)

After successful build and verification, register the container:

```
register_software(
    name="{container_name}",
    version="{version_or_date}",
    prefix="{directory_containing_sif}",
    source="container",
    purpose="{what_it_contains — be specific}",
    categories=["container", "{domain_tag}", "{tool_tags}"],
    project="{project_name_if_scoped}",
    notes="Built from {base_image}. Contains: {key_packages}. File: {sif_filename}"
)
```

For Docker Hub pulls:
```
register_software(
    name="{tool_name}",
    version="{tag_version}",
    prefix="{directory_containing_sif}",
    source="container",
    purpose="{tool_name} in Singularity container",
    categories=["container", "{domain}"],
    notes="Pulled from docker://{registry}/{image}:{tag}. File: {sif_filename}"
)
```

This ensures future sessions can find containers via `query_software` without rebuilding.

## Key Recipes

### Recipe: Python + Scientific Packages

```
%post
    export HOME=/opt/container_home
    mkdir -p $HOME
    
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3
    /opt/miniforge3/bin/mamba install -y -c conda-forge \
        python=3.11 numpy pandas scipy scikit-learn matplotlib seaborn
```

### Recipe: GPU Container (CUDA + PyTorch)

```
Bootstrap: docker
From: nvidia/cuda:12.1.1-devel-ubuntu22.04

%post
    export HOME=/opt/container_home
    export CONDA_PKGS_DIRS=/opt/conda/pkgs
    mkdir -p $HOME $CONDA_PKGS_DIRS

    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3
    /opt/miniforge3/bin/mamba install -y -c conda-forge -c pytorch \
        python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1
```

### Recipe: R + Bioconductor

```
%post
    export HOME=/opt/container_home
    mkdir -p $HOME

    apt-get update && apt-get install -y r-base libcurl4-openssl-dev libssl-dev libxml2-dev
    
    Rscript -e 'install.packages("BiocManager", repos="https://cran.r-project.org")'
    Rscript -e 'BiocManager::install(c("DESeq2", "edgeR", "Seurat"))'
```

### Recipe: Pull from Docker Hub (No Build Needed)

```bash
export SINGULARITY_BIND=''
export APPTAINER_BIND=''
$SINGULARITY pull \
    {work_dir}/{name}.sif \
    docker://biocontainers/samtools:1.17--h6899075_1
```

## Critical: Fakeroot Environment Variables

Under fakeroot, /root is NOT writable. Set these at TOP of %post:

| Variable | Value | Why |
|----------|-------|-----|
| HOME | /opt/container_home | Redirects ~ from /root (not writable) |
| CONDA_PKGS_DIRS | /opt/conda/pkgs | Package cache redirect |
| CONDA_ENVS_PATH | /opt/conda/envs | Environment directory redirect |
| XDG_CACHE_HOME | /opt/container_home/.cache | libmamba v2 shard cache |

All four + mkdir BEFORE any conda/mamba call.

## Critical: Nested Singularity

IrisAI runs inside a container. Building another container from within requires:
```bash
export SINGULARITY_BIND=''
export APPTAINER_BIND=''
```
at the TOP of the Slurm job script (before the singularity build command).
Without this, the inherited bind mounts cause immediate build failure.

## Best Practices & Pitfalls

### Always:
- Check query_software before building — a usable container may already exist
- Register the container with register_software after build completes
- Build via submit_slurm_job (never on login node)
- Clear SINGULARITY_BIND/APPTAINER_BIND in job script
- Set all fakeroot env vars before conda/mamba
- Clean package caches in .def (mamba clean -afy) to reduce image size
- Test the container after building

### Never:
- Don't skip registration — without it, the next session will rebuild from scratch
- Don't build interactively (fakeroot build can't be interrupted safely)
- Don't skip the nested singularity fix (will fail immediately)
- Don't use /root as HOME in %post (not writable under fakeroot)
- Don't forget --fakeroot flag (user has no sudo)

## Tools

- `submit_slurm_job` — Submit build job (primary)
- `slurm_monitor_job` — Wait for build
- `write_text_file` — Write .def file and build script
- `edit_file` — Modify .def file
- `read_text_file` — Read build logs
- `execute_dynamic_task` — Quick verification after build
- `find_files` — Locate .sif files
- `query_software` — Check if container already exists before building (ALWAYS call first)
- `register_software` — Register built container so future sessions find it (ALWAYS call after build)

## References

- `references/fakeroot-constraints.md` — Load WHEN build fails with permission errors
- `references/def-file-best-practices.md` — Load WHEN writing complex .def files
- `../shared/references/conda-in-containers.md` — Load WHEN adding conda to containers
