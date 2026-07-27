# IrisAI Containers

IrisAI runs entirely inside Singularity/Apptainer containers on the HPC cluster. Two containers are launched per OOD session:

1. **Chainlit container** — Runs `app.py` + Chainlit UI (port 8000)
2. **MCP container** — Runs all 4 MCP tool servers (ports 8001–8005)

## Container Inventory

| File | Definition | Purpose | GPU |
|------|-----------|---------|-----|
| `mcp_servers_v2.sif` | `mcp_servers_v3.def` | **Default** — MCP tools, Python, core libs. Used for ALL Slurm jobs. | ✅ |
| `chainlit.sif` | `chainlit.def` | Chainlit UI server | ❌ |

> **Note:** `mcp_servers_v3.def` is the active definition for building `mcp_servers_v2.sif`.
> Earlier versions (`mcp_servers_v1.def`, `mcp_servers_v2.def`, `mcp_servers.def`) are kept for reference.

## Building Containers

### Prerequisites

- Apptainer/Singularity ≥1.5 with `--fakeroot` support
- Sufficient disk space (~20 GB build cache)
- Recommended: submit as a Slurm job (16 CPUs, 64 GB RAM, 2h walltime)

### Required Environment Variables

Set these before building to avoid permission errors:

```bash
export SINGULARITY_CACHEDIR=/path/to/writable/.cache/singularity
export SINGULARITY_TMPDIR=/path/to/writable/.cache/singularity_tmp
export APPTAINER_CACHEDIR=/path/to/writable/.cache/apptainer
export APPTAINER_TMPDIR=/path/to/writable/.cache/apptainer_tmp
```

### Build Commands

```bash
export SINGULARITY=/path/to/singularity  # or apptainer binary

# Build MCP servers container (used for all Slurm jobs)
$SINGULARITY build --fakeroot --no-cleanup mcp_servers_v2.sif containers/mcp_servers_v3.def

# Build Chainlit UI container
$SINGULARITY build --fakeroot --no-cleanup chainlit.sif containers/chainlit.def
```

## Conda Environment YAMLs

- `hpcagent02.yml` — Legacy agent environment (deprecated)
- `mcptool001.yml` — Legacy MCP tools environment (deprecated)

The active conda environment spec used during container builds is embedded in the `.def` files.
