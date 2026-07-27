---
name: software-management
description: Install packages, create conda environments, manage software on HPC —
  conda create/install, pip install, module load, verify installations, resolve
  dependency conflicts, check what software is installed, register/track software
  paths and versions in central registry, remember where software is located
allowed_tools:
  - submit_slurm_job
  - slurm_monitor_job
  - execute_dynamic_task
  - batch
  - write_text_file
  - edit_file
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - register_software
  - query_software
  - update_software_entry
  - remove_software_entry
model: null
max_iterations: 45
guardrails:
- ALWAYS use submit_slurm_job for conda/pip installs — they ALWAYS exceed the 5min execute_dynamic_task timeout
- NEVER install system-wide (no sudo, no --system)
- NEVER use pip install without --user or a virtual/conda env prefix
- After install completes, ALWAYS verify with an import test AND register_software
- Use --prefix for conda envs (inside work_dir/software/envs/), never --name (writes to read-only container paths)
- Channel order MUST be conda-forge first, bioconda second — reversing breaks solver
- NEVER force-install into an existing env when dependencies conflict — create a NEW env
- After EVERY successful install, call register_software — no exceptions
- When user tells you they have software at a path → ALWAYS call register_software (don't just save to memory)
- For ANY software path discovery or lookup → call query_software FIRST before checking memory
---

# Software Management

Install and manage software packages on the HPC cluster. Handles conda environment
creation, pip installations, module loads, dependency resolution, and verification.

## When to Use This Skill

**Triggers:**
- "Install X" / "Set up environment" / "Create conda env"
- "I need package X" / "Add library Y"
- "pip install" / "conda install" / "mamba install"
- "Module load" / "What modules are available?"
- "My import fails" / "Package not found"
- "Update package X" / "Upgrade to version Y"
- "Set up Python/R environment for my analysis"

**NOT for (route elsewhere):**
- "Run my script" (already has env) → code-execution
- "Build a Singularity container" → container-building
- "Submit a job" (not installing) → hpc-submit-job

## Defaults & Conventions

- Default env location: {work_dir}/software/envs/{env_name}/ (prefix-based)
- Default Python: 3.11 (or match user's existing project)
- Default channels: --override-channels -c conda-forge -c bioconda
- Default solver: --solver=classic (libmamba not reliably available in containers)
- Verification: always run import test after install
- Memory: always save successful env paths to project memory

## Complete Workflow

### Step 0: Check Registry (YOUR VERY FIRST TOOL CALL)

Your FIRST tool call must be query_software. Before ANY other action — before
get_environment_info, before execute_dynamic_task, before read_memory — call:

```
query_software(search="{package_name}")
```

If found:
- Tell the user it's already available at the registered path
- Provide activation instructions
- SKIP installation entirely — do NOT reinstall what's already registered

If not found:
- Do NOT search the filesystem with find/ls/grep for binaries — query_software is authoritative
- Proceed directly to Step 1

### Step 1: Context & Discovery

1. **Check project memory:**
   → read_memory(project)
   → Look for: existing conda env paths, previously installed packages, known issues
   → If an env already exists with the needed package, just use it (skip install!)

2. **Discover existing software:**
   → get_environment_info('packages') — check if package is already in shared registry
   → get_environment_info('package:<name>') — read knowledge for available packages
   → If the package is already available, tell the user and skip installation

3. **Check for existing conda envs in work_dir:**
   → find_files(pattern="conda_env*", directory=work_dir)
   → list_directory(path="{work_dir}/conda_envs") if exists
   → An existing env might already have what's needed

4. **Understand what the user needs:**
   → Which packages? (name, version constraints)
   → Python or R?
   → Any GPU requirements? (pytorch, tensorflow, jax with CUDA)
   → Any bioinformatics channels needed? (bioconda)

### Step 2: Plan the Installation

Choose installation method:

```
Package needed
│
├─ Already in shared registry (get_environment_info)?
│  └─ DONE — tell user it's available, provide activation path
│
├─ Already in an existing conda env in work_dir?
│  └─ DONE — tell user which env has it
│
├─ Simple pip package (pure Python, no compilation)?
│  ├─ User has existing env? → pip install into it
│  └─ No env? → Create new conda env first, THEN pip install
│
├─ Conda package (or needs C dependencies)?
│  └─ Create new conda env with package included
│
├─ GPU package (pytorch, tensorflow, jax)?
│  └─ Create conda env with CUDA toolkit from conda-forge
│
└─ R package?
   └─ Create conda env with r-base + r-<package>
```

### Step 3: Write Installation Script

**CRITICAL: All installs go through submit_slurm_job.** The 5-minute timeout on
execute_dynamic_task is too short for ANY non-trivial installation.

Template for conda env creation:
```bash
#!/bin/bash
set -euo pipefail

# === CONTAINER WORKAROUNDS (MANDATORY) ===
export CONDA_NO_PLUGINS=true
export CONDA_PKGS_DIRS=/tmp/conda_pkgs_$$
export HOME=/tmp/home_$$
mkdir -p "$CONDA_PKGS_DIRS" "$HOME"

# === INIT CONDA ===
eval "$(conda shell.bash hook)"

# === CREATE ENVIRONMENT ===
ENV_PREFIX="{work_dir}/software/envs/{env_name}"

conda create --prefix "$ENV_PREFIX" \
    --solver=classic \
    --override-channels -c conda-forge -c bioconda \
    python=3.11 {packages} -y

# === ACTIVATE AND VERIFY ===
conda activate "$ENV_PREFIX"
{verification_commands}

echo "SUCCESS: Environment created at $ENV_PREFIX"
echo "Packages installed:"
conda list --prefix "$ENV_PREFIX" | grep -E "{package_patterns}"
```

### Step 4: Submit Installation Job

```
submit_slurm_job(
    job_name="install_{env_name}",
    script_content="{script from step 3}",
    work_dir="{work_dir}",
    time_limit="01:00:00",
    memory="16G",
    cpus=4
)
```

Then ALWAYS wait:
```
slurm_monitor_job(job_id="{job_id}", wait=true)
```

### Step 5: Verify Installation

After job completes:
1. Read the job output: `read_text_file(path=slurm_output)`
2. Check for "SUCCESS" message
3. If failed, read error messages and diagnose

Quick verification (OK to use execute_dynamic_task for this — it's fast):
```
execute_dynamic_task(
    task_description="Verify conda env",
    commands=[
        "source /opt/miniforge3/etc/profile.d/conda.sh && conda activate {env_prefix} && python -c 'import {package}; print(f\"{package} {package.__version__} OK\")'"
    ]
)
```

### Step 6: Register & Persist

**ALWAYS register after successful install (MANDATORY — no exceptions):**
```
register_software(
    name="{package_or_env_name}",
    version="{version}",
    prefix="{absolute_install_path}",
    source="conda",  # or "pip", "manual", "spack", "system"
    purpose="{what it's for — be specific}",
    categories=["{tag1}", "{tag2}"],
    project="{project_name_if_scoped}",  # omit for shared
    notes="{any compatibility notes}"
)
```

Also save to memory for project context:
```
update_memory(
    project="{project_name}",
    key="conda_env_{env_name}",
    value="Path: {env_prefix}, Packages: {package_list}, Created: {date}"
)
```

This ensures future sessions find the software via `query_software` without re-installing.

### User Provides Software Location

When user says "I have X at /path/to/X":
1. Verify it exists: `execute_dynamic_task` with `{path}/bin/{name} --version` or similar
2. Register it:
   ```
   register_software(name="{name}", version="{detected_version}",
       prefix="{user_provided_path}", source="manual",
       purpose="{what user says it's for}", categories=[...])
   ```
3. Done — now discoverable via `query_software` in all future sessions.

### Dependency Conflict Policy

When installation fails due to dependency conflicts:
1. **NEVER** force-install or `--force-reinstall` into the existing env
2. **CREATE** a new separate environment
3. **REGISTER** the new env with `notes` explaining the conflict:
   ```
   register_software(name="jax-env", version="0.4",
       prefix="{work_dir}/software/envs/jax-cuda11",
       source="conda", purpose="JAX with CUDA 11",
       notes="Separate from ml-cuda12 env — JAX needs cuda11, incompatible with pytorch-cuda12")
   ```
4. Two working envs > one broken env

## Key Recipes

### Recipe: Create New Conda Environment

**Trigger:** "Create a Python environment with X, Y, Z"

**Workflow:**
1. Check if packages already exist (get_environment_info, read_memory)
2. Write install script with container workarounds
3. submit_slurm_job → time_limit="01:00:00", memory="16G"
4. slurm_monitor_job → wait for completion
5. Verify: activate env, import packages
6. Save env path to memory

### Recipe: Add Package to Existing Environment

**Trigger:** "Install X into my existing env"

**Workflow:**
1. read_memory → find existing env path
2. Write script:
   ```bash
   export CONDA_NO_PLUGINS=true
   export CONDA_PKGS_DIRS=/tmp/conda_pkgs_$$
   mkdir -p "$CONDA_PKGS_DIRS"
   eval "$(conda shell.bash hook)"
   conda activate {existing_env_prefix}
   conda install --solver=classic -c conda-forge {package} -y
   # OR for pip-only packages:
   pip install {package}
   python -c "import {package}; print('OK')"
   ```
3. submit_slurm_job + slurm_monitor_job
4. Verify import
5. Update memory with new package list

### Recipe: GPU Environment (PyTorch/TensorFlow)

**Trigger:** "Set up PyTorch with GPU" / "TensorFlow with CUDA"

**Workflow:**
1. Check GPU availability: get_environment_info('category:ml')
2. Write script with CUDA-aware packages:
   ```bash
   # PyTorch with CUDA from conda-forge
   conda create --prefix "$ENV_PREFIX" \
       --solver=classic \
       --override-channels -c conda-forge -c pytorch \
       python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -y
   
   # Verify GPU access
   conda activate "$ENV_PREFIX"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
3. Submit with GPU partition (for verification step):
   submit_slurm_job with partition hint for GPU access
4. Verify CUDA detection

### Recipe: R Environment

**Trigger:** "Install R with Seurat" / "Set up R for analysis"

**Workflow:**
1. Write script:
   ```bash
   conda create --prefix "$ENV_PREFIX" \
       --solver=classic \
       --override-channels -c conda-forge -c bioconda \
       r-base=4.3 r-seurat r-ggplot2 r-dplyr -y
   
   conda activate "$ENV_PREFIX"
   Rscript -e 'library(Seurat); cat("Seurat", packageVersion("Seurat"), "OK\n")'
   ```
2. Submit, verify, save to memory

### Recipe: Bioinformatics Tools (bioconda)

**Trigger:** "Install STAR/BWA/samtools/scanpy"

**Workflow:**
1. Check if tool is in shared registry (get_environment_info)
2. Channel order is CRITICAL: conda-forge FIRST, bioconda SECOND
   ```bash
   conda create --prefix "$ENV_PREFIX" \
       --solver=classic \
       --override-channels -c conda-forge -c bioconda \
       python=3.11 star samtools picard -y
   ```
3. Submit, verify with `--version` check, save to memory

### Recipe: Fix Broken Installation

**Trigger:** "Install failed" / "Can't import after install"

**Diagnosis workflow:**
1. Read the Slurm job output: look for error messages
2. Common issues:
   - "Solving environment: failed" → channel conflict or impossible version
   - "CondaError: PKGS_DIRS" → container workaround missing
   - "Permission denied" → writing to read-only path
   - "No matching distribution" → package not in channel
3. Fixes:
   - Channel conflict: try removing version pin, or use pip instead of conda
   - PKGS_DIRS: ensure CONDA_PKGS_DIRS is set to /tmp
   - Permission: use --prefix to work_dir, not --name
   - Not in channel: check PyPI (pip install) or different channel

### Recipe: Module Load (System Software)

**Trigger:** "Load module" / "Use system CUDA" / "What modules exist?"

**Workflow:**
1. Check available modules:
   ```
   execute_dynamic_task(commands=["module avail 2>&1 | head -50"])
   ```
2. Load and verify:
   ```
   execute_dynamic_task(commands=["module load cuda/12.1 && nvcc --version"])
   ```
3. Note: modules are session-scoped — for persistent use, add to script headers

## Container Workaround Reference

Every conda script inside a Slurm container needs these at the TOP:

```bash
# MANDATORY — without these, conda fails in read-only container environments
export CONDA_NO_PLUGINS=true              # Bypass ToS plugin (needs writable ~/.conda/)
export CONDA_PKGS_DIRS=/tmp/conda_pkgs_$$ # Redirect package cache from read-only container
export HOME=/tmp/home_$$                  # Redirect home from read-only /home mount
mkdir -p "$CONDA_PKGS_DIRS" "$HOME"

# Initialize conda
eval "$(conda shell.bash hook)"
```

**Why each is needed:**
| Variable | Without it... |
|----------|--------------|
| CONDA_NO_PLUGINS=true | ToS plugin blocks non-interactive installs (needs ~/.conda/tos/) |
| CONDA_PKGS_DIRS | Package cache writes to read-only /opt/miniforge3/pkgs/ |
| HOME | Conda config writes to read-only /home/user/.condarc |

## Best Practices & Pitfalls

### Always:
- Check existing software first (get_environment_info) — avoid redundant installs
- Use --prefix (writable work_dir), never --name (default path is read-only)
- Use submit_slurm_job for ALL installs (even "quick" ones — pip compiles C extensions)
- Verify with import test after installation completes
- Save env path to memory (prevents re-doing work next session)
- Set container workaround env vars before any conda command
- Use --solver=classic (libmamba may not be available)

### Never:
- Don't use execute_dynamic_task for installs (5min timeout is too short)
- Don't install system-wide (no sudo, no conda install --name base)
- Don't use `source activate` (use `conda activate` with shell hook init)
- Don't skip verification — a "successful" install can still have broken imports
- Don't assume channels are correct — always specify --override-channels explicitly
- Don't use --name without checking where CONDA_ENVS_PATH points (usually read-only)
- Don't reverse channel order — conda-forge MUST come before bioconda

### Common Pitfalls:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| "Solving environment: failed" | Impossible version combo or missing channel | Relax version pins, check channel order |
| "CondaError: cannot create /opt/..." | Writing to read-only container path | Set CONDA_PKGS_DIRS to /tmp |
| "conda-anaconda-tos" prompt | CONDA_NO_PLUGINS not set | Add export CONDA_NO_PLUGINS=true |
| Install "succeeds" but import fails | Different Python on PATH | Use explicit {env}/bin/python |
| "No space left on device" | /tmp full from package cache | Use CONDA_PKGS_DIRS={work_dir}/.conda_cache |
| Bioconda packages ancient | Channel order reversed | conda-forge FIRST, bioconda SECOND |
| "activate: No such file or directory" | Using source activate (virtualenv syntax) | Use conda activate after shell hook init |
| LibMamba error | Solver not available in container | Use --solver=classic |

## Tools

- `submit_slurm_job` — Submit installation as Slurm job (PRIMARY tool for all installs)
- `slurm_monitor_job` — Wait for installation job to complete
- `execute_dynamic_task` — Quick checks only (module avail, verify imports, list envs)
- `register_software` — Register software in central registry (MANDATORY after every install)
- `query_software` — Check what's already installed before attempting install
- `update_software_entry` — Fix version, update notes/purpose/categories
- `remove_software_entry` — Deregister software (does NOT delete files)
- `write_text_file` — Write complex installation scripts
- `edit_file` — Modify existing scripts
- `read_text_file` — Read job output, check environment files
- `grep_file` — Search for packages in conda list output
- `find_files` — Locate existing conda envs
- `list_directory` — Browse env contents

## References

- `../shared/references/conda-in-containers.md` — Load WHEN conda install fails with permission/path errors
- `references/dependency-resolution.md` — Load WHEN solver fails or packages conflict
