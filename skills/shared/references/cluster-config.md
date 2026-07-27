# Cluster Configuration Reference

## Scheduler

- **Slurm** with backfill scheduling (180s cycle, 10080-min lookahead, bf_max_job_test=8000)
- **Resource enforcement:** CGroups v2 (memory hard-enforced, CPU pinning, GPU isolation)
- **GPU detection:** AutoDetect=nvml in gres.conf
- **Job routing:** job_submit.lua plugin (auto-routes partitions, sets account/QOS)

## Node Families

| Family | CPUs | GPUs | Memory | Notes |
|--------|------|------|--------|-------|
| isca | 56 | 0 | 256GB | Standard CPU |
| iscb | 56 | 4×V100 (16GB) | 384GB | Legacy GPU |
| iscc | 56 | 0 | 1.5TB | High-memory |
| iscd | 56 | 4×A100 (40GB) | 512GB | GPU compute |
| isce | 96 | 0 | 1.5TB | High-memory, high-CPU |
| iscf | 104 | 4×A100 (80GB) | 1TB | GPU + high-memory |
| iscg | 128 | 8×A100 (80GB) | 2TB | Large GPU (8-GPU) |
| isch | 104 | 4×L40S (48GB) | 512GB | GPU compute |
| isci | 128 | 4×H100 (80GB) | 1.5TB | Latest GPU |
| iscj | 192 | 4×H100-NVL (80GB) | 1.5TB | NVLink GPU |
| isck | 192 | 0 | 1.5TB | High-CPU |
| iscl | 128 | 4×A100 (80GB) | 1TB | GPU compute |
| iscm | 128 | 0 | 2TB | Ultra high-memory |
| iscn | 192 | 8×H100-NVL (80GB) | 1.5TB | Large NVLink GPU (8-GPU) |
| isco | 128 | 4×H200 (141GB) | 1.5TB | Latest GPU |
| iscp | 192 | 8×H200 (141GB) | 3TB | Largest GPU node (8-GPU) |

## Partition Tiers

### General-Access
| Partition | Purpose | Key Limits |
|-----------|---------|------------|
| cpu | Standard CPU work | MaxCPUsPerNode=52 |
| cpushort | CPU jobs ≤2h (auto-routed) | MaxTime=2:00:00 |
| gpu | GPU work | All GPU nodes |
| gpushort | GPU jobs ≤2h (auto-routed) | MaxTime=2:00:00 |
| cpu_highmem | High-memory CPU jobs | High-RAM nodes |
| interactive | Interactive sessions | max 8 CPU, 1 GPU, 64GB, 2 concurrent |
| preemptable | Fault-tolerant jobs | PriorityTier=1, can be killed |
| datatransfer | Data movement only | 1 CPU max |

### PI/Group-Owned (Examples)
Named after PIs or groups (morrisq, lareauc_gpu, componc_cpu, etc.).
Require specific account — always check access first.

### Institutional
COMPONC, CMOBIC, BIC — department-level shared partitions.

## Dual-Partition Design

- GPU nodes appear in BOTH cpu and gpu partitions
- MaxCPUsPerNode=52 cap in cpu partition (reserves CPUs for GPU jobs)
- CPU jobs can use idle GPU node CPUs (via cpu partition)
- GPU jobs ONLY run through gpu partition

## Job Priority

Multifactor: `SITE×1 + AGE×1000 + ASSOC×5000 + FAIRSHARE×10000 + QOS×5000`
- **FAIRSHARE dominates** (weight 10000, 7-day half-life)
- Heavy recent usage → lower priority → longer queue
- AGE slowly boosts pending jobs (weight 1000)
- "priority" QOS adds +1000 boost

## QOS Definitions

| QOS | Key Limits |
|-----|-----------|
| normal | Default, no special limits |
| gen_inter | 8 CPU, 1 GPU, 64GB max (interactive partition) |
| preemptable | PriorityTier=1, PreemptMode=REQUEUE |
| datatransfer | 1 CPU max |
| priority | +1000 priority boost |

## Container Architecture

- ALL jobs run inside Singularity containers
- submit_slurm_job auto-wraps in container
- GPU passthrough: `--nv` flag (auto-handled)
- Standard bind mounts: /data1, /home (read-only), /scratch, /usersoftware

## Job Routing Plugin (job_submit.lua)

| Condition | Action |
|-----------|--------|
| GPU job on cpu/cpushort | → gpu |
| CPU job on gpu/gpushort | → cpu |
| Job ≤2h on cpu | → cpushort |
| Job >2h on cpushort | → cpu |
| GPU job ≤2h on gpu | → gpushort |
| GPU job >2h on gpushort | → gpu |
| Preemptable | → auto-sets QOS=preemptable, account=preemptable |
| PI partition | → auto-sets account based on partition name |
