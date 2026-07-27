---
name: hpc-submit-job
description: Submit Slurm jobs — choose partition/resources, validate access, handle
  containers, GPU selection, job dependencies, pre-submission checks
allowed_tools:
  - submit_slurm_job
  - slurm_monitor_job
  - check_user_slurm_access
  - query_slurm_cluster
  - execute_dynamic_task
  - batch
  - write_text_file
  - edit_file
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - analyze_files
  - run_pipeline_script
  - query_software
model: null
max_iterations: 30
guardrails:
- NEVER submit to a PI partition without checking access first via check_user_slurm_access
- ALWAYS set memory explicitly — default is ~2GB/CPU which is too low for scientific work
- ALWAYS set time_limit explicitly — do NOT rely on partition defaults
- Do NOT request more GPUs than exist on any single node (max 4 typically, 8 on large nodes)
- Do NOT set --account or QOS by default — job_submit.lua plugin handles routing automatically
- NEVER manually pick cpushort/gpushort — submit to cpu/gpu and let Slurm route by --time
- For multi-step workflows use dependency parameter, NEVER poll in a loop
- For ANY software path discovery or lookup → call query_software FIRST before checking memory
---

# HPC Job Submission

Submit computational jobs to the Slurm scheduler with correct partition, resources,
and container configuration. Handles pre-submission validation, access checking,
GPU selection, and job dependency chains.

## When to Use This Skill

**Triggers:**
- "Submit a job" / "Run this on the cluster"
- "I need GPUs for this" / "Run on A100/H100"
- "Submit with 32GB memory and 8 CPUs"
- "Chain these jobs" / "Run B after A finishes"
- "What partition should I use?"
- "Can I access the gpu partition?"
- Complex resource requirements beyond execute_dynamic_task's capabilities

**NOT for (route elsewhere):**
- "Check my job status" / "Why is it pending?" → hpc-monitor
- "How many free GPUs?" / "Cluster status" → hpc-query
- "Install packages" (even via Slurm) → software-management
- "Run this quick command" (<2 min, no GPU) → code-execution

## Defaults & Conventions

- Default partition: cpu (Slurm auto-routes GPU jobs to gpu partition)
- Default memory: 8G (override based on workload — see Resource Guidelines)
- Default CPUs: 4
- Default time: Set explicitly based on workload (never rely on defaults)
- Container: Always (submit_slurm_job wraps in Singularity automatically)
- Account/QOS: NEVER set manually (job_submit.lua handles this)

## Complete Workflow

### Step 1: Context & Discovery

1. **Check project memory:**
   → read_memory(project)
   → Look for: previously successful submission parameters, known partition access,
     conda env paths that scripts need

2. **Understand the workload:**
   → What is being run? (Python script, R analysis, compiled binary)
   → How long will it take? (minutes, hours, days)
   → Does it need GPU? How much VRAM?
   → How much memory? (scRNA-seq: 100-500GB, ML training: GPU-bound, genomics: 8-32GB)
   → How many CPUs? (I/O-bound → more CPUs for data loading)

3. **Check software availability:**
   → get_environment_info('packages') — does the script's dependencies exist?
   → If not: escalate to software-management first, THEN come back to submit

### Step 2: Pre-Submission Validation

**MANDATORY before any non-trivial submission:**

1. **Check partition access:**
   ```
   check_user_slurm_access(username="{user}", partition="{target_partition}")
   ```
   If access denied → suggest alternatives or check other partitions.

2. **Verify resources fit partition limits:**
   - CPUs ≤ MaxCPUsPerNode (52 for cpu partition, varies for PI partitions)
   - Memory ≤ node physical RAM minus ~20GB reserved
   - GPUs ≤ max per node (4 typical, 8 on iscg/iscn/iscp nodes)
   - Time ≤ partition MaxTime

3. **GPU type availability** (if requesting specific GPU):
   ```
   query_slurm_cluster(
       query="What GPU types are available in the gpu partition and how many are idle?",
       commands=["sinfo -p gpu -N -o '%N|%G|%T' -h | grep idle"]
   )
   ```

### Step 3: Choose Resources

Use this decision tree:

```
What kind of job?
│
├─ Quick script (< 30 min, no GPU, < 8GB RAM)
│  └─ partition: cpu, time: 00:30:00, mem: 8G, cpus: 4
│
├─ Standard analysis (< 4 hours, no GPU)
│  └─ partition: cpu, time: 04:00:00, mem: 16-64G, cpus: 4-16
│
├─ Memory-intensive (> 100GB RAM needed)
│  └─ partition: cpu, time: varies, mem: 128-500G, cpus: 8-32
│     (targets iscc/isce/isck nodes with 1.5-2TB)
│
├─ Single-GPU job
│  └─ partition: gpu, time: varies, gres: "gpu:1" or "gpu:a100:1"
│     mem: 32-64G, cpus: 8-16 (for data loading)
│
├─ Multi-GPU job (training)
│  └─ partition: gpu, time: varies, gres: "gpu:h100:4"
│     mem: 128-512G, cpus: 32-64
│
├─ Long-running (> 24 hours)
│  └─ partition: cpu/gpu, time: up to 7d
│     Consider: preemptable (if checkpointing possible)
│
└─ Fault-tolerant (can restart from checkpoint)
   └─ partition: preemptable
      Preempt mode: REQUEUE — job may be killed and restarted
```

### Step 4: Submit

```
submit_slurm_job(
    job_name="{descriptive_name}",
    script_content="{complete_bash_script}",
    work_dir="{work_dir}",
    time_limit="{HH:MM:SS}",
    memory="{N}G",
    cpus={N},
    partition="{partition}",      # Optional — defaults to cpu
    gres="gpu:{type}:{count}",   # Only if GPU needed
    dependency="afterok:{job_id}" # Only if chaining
)
```

**Script content must be self-contained:**
- Initialize conda if needed (eval "$(conda shell.bash hook)")
- Set all environment variables
- Include error handling (set -euo pipefail)
- Print clear output about progress

### Step 5: Report & Monitor

After submission:
1. Report job_id to user immediately
2. For short jobs (< 2 min expected):
   ```
   slurm_monitor_job(job_id="{id}", wait=true)
   ```
3. For long jobs: report job_id and tell user to check back
4. For chained jobs: report all job_ids and dependency structure

## Automatic Job Routing (job_submit.lua)

The cluster has a routing plugin that handles partition/QOS automatically:

| What you submit | What happens |
|----------------|--------------|
| GPU job to cpu/cpushort | Auto-routed to gpu |
| CPU job to gpu/gpushort | Auto-routed to cpu |
| Any job ≤2h to cpu | Auto-routed to cpushort |
| Any job >2h to cpushort | Auto-routed to cpu |
| GPU job ≤2h to gpu | Auto-routed to gpushort |
| GPU job >2h to gpushort | Auto-routed to gpu |
| Job to preemptable | Auto-sets QOS=preemptable, account=preemptable |
| Job to PI partition | Auto-sets account based on partition |

**Implication:** Just submit to `gpu` or `cpu`. Don't overthink partition selection.
Slurm handles short/long routing based on your --time value.

## Key Recipes

### Recipe: Simple Script Submission

**Trigger:** "Run my_script.py on the cluster with 16GB RAM"

**Workflow:**
1. check_user_slurm_access (for target partition)
2. submit_slurm_job:
   ```bash
   #!/bin/bash
   set -euo pipefail
   eval "$(conda shell.bash hook)"
   conda activate {env_path}
   python3 {work_dir}/my_script.py
   echo "DONE"
   ```
3. slurm_monitor_job (if short) or report job_id (if long)

### Recipe: GPU Training Job

**Trigger:** "Train my model on 4 A100s"

**Workflow:**
1. check_user_slurm_access(partition="gpu")
2. query free A100s: query_slurm_cluster
3. submit_slurm_job:
   - gres="gpu:a100:4"
   - memory="256G" (enough for data loading)
   - cpus=32 (8 per GPU for data loading)
   - time_limit based on expected training time
4. Report job_id — training is long, don't wait

### Recipe: Job Dependency Chain

**Trigger:** "Run A first, then B after A succeeds"

**Workflow:**
1. Submit job A:
   ```
   submit_slurm_job(job_name="step_A", script_content=..., ...)
   → returns job_id_A
   ```
2. Submit job B with dependency:
   ```
   submit_slurm_job(job_name="step_B", script_content=...,
                    dependency="afterok:{job_id_A}", ...)
   → returns job_id_B
   ```
3. Report both IDs: "Job A ({id_A}) will run first. Job B ({id_B}) starts after A succeeds."

### Recipe: Array Job (Same Script, Multiple Inputs)

**Trigger:** "Run this for all 20 samples" / "Process files in parallel"

**Workflow:**
1. Write script using $SLURM_ARRAY_TASK_ID:
   ```bash
   #!/bin/bash
   SAMPLES=(sample1 sample2 ... sample20)
   SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
   python3 process.py --input "$SAMPLE"
   ```
2. submit_slurm_job with array parameter (if supported by tool)
   OR submit individual jobs with dependency chain

### Recipe: Preemptable Job (Long, Fault-Tolerant)

**Trigger:** "Run for a week" / "It can restart" / "I don't mind if it's interrupted"

**Workflow:**
1. Confirm user has checkpointing in their script
2. submit_slurm_job with partition="preemptable"
   - Plugin auto-sets account/QOS
   - Warn: "Your job may be killed and requeued at any time"
3. Recommend: save checkpoint every N hours

## GPU Reference

| GPU Type | VRAM | Nodes | Partition |
|----------|------|-------|-----------|
| V100 | 16 GB | iscb | gpu |
| A100 (40GB) | 40 GB | iscd | gpu |
| A100 (80GB) | 80 GB | iscf, iscg, iscl | gpu |
| L40S | 48 GB | isch | gpu |
| H100 | 80 GB | isci | gpu |
| H100-NVL | 80 GB | iscj, iscn | gpu |
| H200 | 141 GB | isco | gpu |
| H200 (8-GPU) | 141 GB | iscp | gpu |

**GPU GRES format:** `gpu:<type>:<count>` (e.g., `gpu:a100:2`, `gpu:h100:4`)

**When requested GPU is unavailable,** suggest alternatives with equal or more VRAM:
- V100 (16GB) → L40S (48GB) or A100 (40/80GB)
- A100 (80GB) → H100 (80GB) or H200 (141GB)
- H100 (80GB) → H200 (141GB)

## Resource Guidelines

| Workload | Memory | CPUs | GPUs | Time |
|----------|--------|------|------|------|
| Quick script | 4-8G | 2-4 | 0 | 00:30:00 |
| scRNA-seq analysis | 100-500G | 8-32 | 0 | 02:00:00 - 12:00:00 |
| Alignment (STAR) | 40G | 8-16 | 0 | 01:00:00 - 04:00:00 |
| Deep learning (training) | 64-256G | 8-32/GPU | 1-8 | 04:00:00 - 7d |
| Deep learning (inference) | 32-64G | 8 | 1 | 00:30:00 - 02:00:00 |
| Variant calling | 16-64G | 4-16 | 0 | 02:00:00 - 24:00:00 |
| Container build | 64G | 16 | 0 | 02:00:00 |
| Conda install | 16G | 4 | 0 | 01:00:00 |

## Access Determination (When check_user_slurm_access Not Available)

If you need to manually check access:

**Step 1:** Get user's accounts:
```bash
sacctmgr show assoc user=<username> format=Account,Partition,QOS -P --noheader
```

**Step 2:** Get partition restrictions:
```bash
scontrol show partition=<partition_name>
```
Look for DenyAccounts= and AllowAccounts= fields.

**Step 3:** Cross-reference:
- AllowAccounts=ALL → allowed (unless in DenyAccounts)
- AllowAccounts=<list> → must have account in list
- DenyAccounts=<list> → blocked if ALL accounts are denied
- At least one account NOT denied → access granted

## Best Practices & Pitfalls

### Always:
- Check access before submitting to non-default partitions
- Set memory explicitly (defaults are too low for science)
- Set time explicitly (better scheduling, clear expectations)
- Use job dependencies for multi-step workflows (not polling loops)
- Report job_id immediately after submission
- Submit to cpu or gpu (let Slurm handle short/long routing)

### Never:
- Don't set --account or --qos manually (plugin handles it)
- Don't pick cpushort/gpushort explicitly (auto-routed by time)
- Don't request more GPUs than a single node has
- Don't submit without checking access to PI partitions
- Don't poll for job completion in a loop (use slurm_monitor_job or dependency)
- Don't guess partition names — query first

### Common Pitfalls:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| "Invalid account" error | Plugin can't determine account | Run sacctmgr, retry with --account |
| Job immediately FAILED | Script error (check .out file) | Read slurm-{id}.out for traceback |
| OOM killed | Requested too little memory | Increase --mem, check actual usage |
| Timeout (CANCELLED) | Time limit too short | Increase --time or split job |
| PENDING (Resources) | Not enough idle resources | Wait, or try different GPU type/partition |
| PENDING (Priority) | Low fairshare from heavy usage | Wait, or use preemptable |
| "Partition denied" | User lacks access | check_user_slurm_access, try general partition |

## Tools

- `submit_slurm_job` — Submit a batch job to Slurm (primary tool)
- `slurm_monitor_job` — Check/wait for job completion
- `check_user_slurm_access` — Verify user can access a partition
- `query_slurm_cluster` — Run Slurm queries (sinfo, squeue, sacctmgr)
- `execute_dynamic_task` — Run shell pipelines for precise aggregation (counts, rankings)
- `write_text_file` — Write job scripts
- `edit_file` — Modify existing job scripts
- `read_text_file` — Read job output files
- `grep_file` — Search job logs for errors
- `find_files` — Locate output files
- `list_directory` — Browse job output directories
- `analyze_files` — Analyze multiple job outputs at once
- `run_pipeline_script` — Multi-command Slurm queries

## References

- `../shared/references/cluster-config.md` — Load WHEN need full partition details or node families
- `../shared/references/conda-in-containers.md` — Load WHEN script needs conda activation
- `references/resource-guidelines.md` — Load WHEN unsure about resource allocation for specific workload
