---
name: pipeline-orchestration
description: Multi-step workflow orchestration — Nextflow on Slurm, Snakemake profiles,
  array jobs, job dependency chains, multi-sample processing, parallel execution
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
  - check_user_slurm_access
  - query_slurm_cluster
model: null
max_iterations: 40
guardrails:
- For array jobs, ALWAYS verify sample list before submission (missing samples = wasted allocation)
- Use job dependencies (afterok/afterany) for sequential steps — NEVER poll in a loop
- Set --array throttle to avoid overwhelming the scheduler (e.g. --array=0-19%5 for 5 concurrent)
- ALWAYS include error handling per array task (one failure shouldn't block others)
- For Nextflow/Snakemake, verify the executor/profile config before first run
---

# Pipeline Orchestration

Orchestrate multi-step computational workflows on Slurm. Handles Nextflow
pipelines, Snakemake workflows, Slurm array jobs, job dependency chains,
and multi-sample parallel processing.

## When to Use This Skill

**Triggers:**
- "Run Nextflow pipeline" / "nf-core"
- "Snakemake on the cluster"
- "Process all 20 samples in parallel"
- "Array job" / "Job array"
- "Run step B after step A" / "Chain jobs"
- "Parallelize this across samples"
- "Run my pipeline on all files in this directory"

**NOT for (route elsewhere):**
- "Submit one job" (single step) → hpc-submit-job
- "Check job status" → hpc-monitor
- "Run a quick command" → code-execution
- "Install Nextflow" → software-management

## Complete Workflow

### Step 1: Context & Discovery

1. **Understand the pipeline:**
   → How many samples/inputs?
   → How many steps? Are they sequential or parallelizable?
   → What tools does each step need?
   → What are the dependencies between steps?

2. **Check available orchestrators:**
   → get_environment_info('package:nextflow')
   → get_environment_info('package:snakemake')
   → If neither: use native Slurm arrays + dependencies

3. **Check resources:**
   → check_user_slurm_access for target partition
   → Estimate per-task resources

### Step 2: Choose Orchestration Method

| Scenario | Method |
|----------|--------|
| nf-core pipeline exists for task | Nextflow |
| Complex DAG with many rules | Snakemake |
| Same script, many inputs | Slurm array job |
| 2-5 sequential steps | Slurm dependency chain |
| Mix of parallel + sequential | Array jobs + dependencies |

### Step 3: Implement

#### Slurm Array Jobs

For running the same script on multiple inputs:

```bash
#!/bin/bash
#SBATCH --job-name=process_samples
#SBATCH --array=0-19%5    # 20 tasks, max 5 concurrent
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Sample list
SAMPLES=(sample_01 sample_02 ... sample_20)
SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}

echo "Processing $SAMPLE (task $SLURM_ARRAY_TASK_ID)"

# Your processing here
python3 process.py --input "${DATA_DIR}/${SAMPLE}" --output "${OUT_DIR}/${SAMPLE}"

echo "DONE: $SAMPLE"
```

#### Job Dependency Chains

```
# Step 1: Align all samples (parallel)
JOB_ALIGN=$(sbatch --parsable align_all.sh)

# Step 2: Merge (depends on all alignments)
JOB_MERGE=$(sbatch --parsable --dependency=afterok:$JOB_ALIGN merge.sh)

# Step 3: Call variants (depends on merge)
JOB_CALL=$(sbatch --parsable --dependency=afterok:$JOB_MERGE call_variants.sh)
```

Using submit_slurm_job:
```
# Submit step 1
id_1 = submit_slurm_job(job_name="step_1", ...)

# Submit step 2 with dependency
id_2 = submit_slurm_job(job_name="step_2", dependency="afterok:{id_1}", ...)

# Submit step 3
id_3 = submit_slurm_job(job_name="step_3", dependency="afterok:{id_2}", ...)
```

#### Nextflow on Slurm

Config file for Slurm executor:
```groovy
// nextflow.config
process {
    executor = 'slurm'
    queue = 'cpu'
    time = '4h'
    memory = '16 GB'
    cpus = 4
    
    withLabel: 'gpu' {
        queue = 'gpu'
        clusterOptions = '--gres=gpu:1'
        memory = '64 GB'
    }
    withLabel: 'highmem' {
        memory = '128 GB'
        cpus = 16
    }
}

singularity {
    enabled = true
    autoMounts = true
}
```

Run:
```bash
nextflow run {pipeline} -profile slurm -c nextflow.config \
    --input {samplesheet} --outdir {results_dir}
```

#### Snakemake on Slurm

Profile config (`~/.config/snakemake/slurm/config.yaml`):
```yaml
executor: slurm
default-resources:
  mem_mb: 16000
  runtime: 240
  cpus_per_task: 4
  partition: cpu
```

Run:
```bash
snakemake --profile slurm -j 20 --use-singularity
```

## Key Recipes

### Recipe: Process N Samples with Array Job

**Trigger:** "Run this on all 20 samples"

1. Create sample list file
2. Write array job script using SLURM_ARRAY_TASK_ID
3. Submit with --array=0-{N-1}%{max_concurrent}
4. Monitor with slurm_monitor_job or squeue

### Recipe: Sequential Pipeline (align → sort → call)

**Trigger:** "Run alignment then variant calling"

1. Submit each step with dependency on previous
2. Report all job IDs with dependency structure
3. User can monitor overall progress

### Recipe: nf-core Pipeline

**Trigger:** "Run nf-core/rnaseq on my data"

1. Write nextflow.config for Slurm executor
2. Create samplesheet.csv
3. Submit Nextflow launcher as Slurm job:
   ```bash
   nextflow run nf-core/rnaseq -r 3.14.0 \
       --input samplesheet.csv \
       --genome GRCh38 \
       --outdir results \
       -profile singularity
   ```

## Dependency Types

| Type | Meaning | Use When |
|------|---------|----------|
| afterok:ID | Start after ID succeeds | Default — step B needs step A output |
| afterany:ID | Start after ID finishes (success or fail) | Cleanup/notification jobs |
| afternotok:ID | Start only if ID fails | Error handling jobs |
| after:ID | Start after ID starts | Loose coupling |

## Best Practices & Pitfalls

### Always:
- Verify sample list before array submission
- Throttle array jobs (%N) to avoid overwhelming scheduler
- Include error handling per task
- Use --parsable for programmatic job ID capture
- Save dependency structure to a file for debugging

### Never:
- Don't submit hundreds of jobs without throttling
- Don't poll for completion in a loop (use dependencies)
- Don't assume all array tasks succeed (check each)
- Don't hardcode sample lists in scripts (use external file)

## Tools

- `submit_slurm_job` — Submit pipeline/array jobs
- `slurm_monitor_job` — Check job completion
- `write_text_file` — Write pipeline configs and scripts
- `execute_dynamic_task` — Quick config verification
- `check_user_slurm_access` — Verify partition access
- `query_slurm_cluster` — Check cluster state before large submission

## References

- `references/nextflow-slurm-config.md` — Load WHEN setting up Nextflow executor
- `references/job-dependency-patterns.md` — Load WHEN complex dependency DAG needed
