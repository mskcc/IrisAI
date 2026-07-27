# User Workspace Directory Structure

IrisAI manages a structured workspace directory (`work_dir`) for each user. This document defines the directory layout, file placement rules, and conventions.

---

## Directory Layout

```
<work_dir>/
├── reports/          # Analysis reports, summaries (.md, .pdf, .html, .pptx)
├── scripts/          # Standalone scripts (.py, .sh)
├── docs/             # Documentation, plans, worksheets, design docs
├── configs/          # Config files (.yml, .yaml, .json, .toml, .ini)
│   └── software_registry.json  # Persistent software/package registry
├── slurm_jobs/       # Slurm jobs organized by date
│   └── YYYY-MM-DD/   # Date-based subdirectories
│       └── jobname_timestamp/
│           ├── submit.sh    # Job submission script
│           ├── task.sh      # Actual task script
│           ├── stdout.log   # Job standard output
│           └── stderr.log   # Job standard error
├── projects/         # Per-project multi-file work (subdirs by project name)
├── images/           # Saved visualizations (auto-managed by save_image)
├── uploads/          # User file uploads (auto-managed)
├── dynamic_tasks/    # Task execution scratch space (auto-created)
│   └── task_name/
│       └── task_id/
├── conda_envs/       # Conda environments
├── venvs/            # Python virtual environments
├── .cache/           # Temporary/scratch data
│   ├── conda_pkgs/   # Conda package cache
│   ├── pip/           # Pip cache
│   ├── xdg/           # XDG cache (container compat)
│   └── fake_home/     # Container HOME override
└── alphafold3/       # AlphaFold3 weights and outputs (if applicable)
```

---

## File Placement Rules

These rules are enforced by the agent — files are NEVER placed at the workspace root.

| File Type | Target Directory | Examples |
|-----------|-----------------|----------|
| Reports & analyses | `reports/` | `analysis.md`, `summary.pdf` |
| Scripts | `scripts/` | `process_data.py`, `setup.sh` |
| Documentation | `docs/` | `plan.md`, `design.md` |
| Configurations | `configs/` | `params.yaml`, `settings.json` |
| Slurm jobs | `slurm_jobs/YYYY-MM-DD/` | Auto-organized by date |
| Project work | `projects/<name>/` | Multi-file project outputs |
| Images/figures | `images/` | Auto-managed by `save_image` |
| Uploads | `uploads/` | Auto-managed by file upload |
| Task scratch | `dynamic_tasks/` | Auto-created, never cleaned |

---

## Path-Sensitive Directories

### `slurm_jobs/`
- Organized by date: `slurm_jobs/2026-07-17/jobname_1234567890/`
- Contains: `submit.sh` (SBATCH script), `task.sh` (user script), `stdout.log`, `stderr.log`
- Auto-created by `submit_slurm_job` tool

### `dynamic_tasks/`
- Organized by task name and ID: `dynamic_tasks/my_task/abc12345/`
- Used by `execute_dynamic_task` for temporary script execution
- NOT cleaned automatically — persists for debugging

### `.cache/`
- Container compatibility directory
- Used when containers can't write to `/home` (read-only in Singularity)
- Environment variables point here: `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `HOME`

---

## Software Registry

The persistent software registry lives at `configs/software_registry.json`:

```json
{
  "packages": [
    {
      "name": "samtools",
      "version": "1.19",
      "prefix": "/path/to/install",
      "source": "conda",
      "purpose": "BAM/SAM file manipulation",
      "categories": ["bioinformatics", "genomics"],
      "default": true
    }
  ]
}
```

### Registry Rules
- `register_software` is the ONLY way to add entries (not `update_memory`)
- `query_software` searches the registry before any installation
- Registry persists across sessions — software is discovered once, remembered forever
- Entries include: name, version, install path, source, purpose, categories

---

## Conventions

1. **No root-level files** — Everything goes in a subdirectory
2. **Date-organized jobs** — Slurm jobs auto-organized by execution date
3. **Project isolation** — Multi-file work goes in `projects/<name>/`
4. **Immutable tasks** — `dynamic_tasks/` entries are never overwritten or deleted
5. **Cache separation** — Container caches isolated from host caches in `.cache/`
6. **Registry authority** — Software paths ONLY valid if in `software_registry.json`
