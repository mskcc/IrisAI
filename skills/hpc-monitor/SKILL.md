---
name: hpc-monitor
description: Monitor Slurm jobs — check status, diagnose why pending, read output
  logs, diagnose failures, check runtime, cancel jobs
allowed_tools:
  - slurm_monitor_job
  - slurm_cancel_job
  - query_slurm_cluster
  - execute_dynamic_task
  - batch
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - analyze_files
  - summarize_command_output
  - run_pipeline_script
model: null
max_iterations: 30
guardrails:
- ALWAYS ask user confirmation before cancelling any job
- NEVER report job status without running the actual Slurm command first
- NEVER guess or estimate — if a command fails, say so
- When output is large (>50 lines), summarize with key information only
- For "why pending" questions, check BOTH squeue reason AND squeue --start for estimated start
- Read the ACTUAL error output file before diagnosing failures — do not speculate
---

# HPC Job Monitoring

Monitor, diagnose, and manage running/pending/failed Slurm jobs. Check status,
explain why jobs are pending, read and interpret output logs, diagnose failures,
and cancel jobs when needed.

## When to Use This Skill

**Triggers:**
- "Check my job" / "What's the status of job X?"
- "Why is my job pending?" / "When will it start?"
- "My job failed" / "What went wrong?"
- "Show me the output" / "Read the job log"
- "Cancel job X" / "Stop my job"
- "How long has it been running?"
- "Is my job still running?"

**NOT for (route elsewhere):**
- "Submit a new job" → hpc-submit-job
- "How many free GPUs?" / "Cluster status" → hpc-query
- "Fix my script and resubmit" → code-execution + hpc-submit-job

## Complete Workflow

### Step 1: Identify the Job

1. **If user gives job ID:** Use directly
2. **If user says "my job" / "my latest job":**
   ```
   query_slurm_cluster(
       query="Show user's recent jobs",
       commands=["squeue -u {username} -o '%i|%j|%P|%T|%M|%r' -h"]
   )
   ```
3. **If user gives job name:**
   ```
   query_slurm_cluster(
       query="Find job by name",
       commands=["squeue -u {username} -n '{job_name}' -o '%i|%T|%M' -h"]
   )
   ```

### Step 2: Check Status

```
slurm_monitor_job(job_id="{id}")
```

OR for more detail:
```
query_slurm_cluster(
    query="Detailed job status",
    commands=["scontrol show job {id}"]
)
```

### Step 3: Respond Based on State

```
Job state?
│
├─ RUNNING
│  └─ Report: runtime so far, resources used, node, output so far
│     Optional: tail the output file
│
├─ PENDING
│  └─ Diagnose WHY (see "Why Pending?" recipe below)
│
├─ COMPLETED
│  └─ Report: final status, runtime, read output file
│
├─ FAILED / CANCELLED / TIMEOUT / OUT_OF_MEMORY
│  └─ Diagnose failure (see "Diagnose Failure" recipe below)
│
└─ NOT FOUND (sacct might have it)
   └─ query_slurm_cluster with sacct -j {id}
```

## Key Recipes

### Recipe: Check Job Status

**Trigger:** "What's the status of job 12345?"

**Workflow:**
1. slurm_monitor_job(job_id="12345")
2. Report state, runtime, resources
3. If RUNNING: offer to read current output
4. If COMPLETED/FAILED: offer to read final output

### Recipe: Why Is My Job Pending?

**Trigger:** "Why is my job pending?" / "When will it start?"

**Workflow:**
1. Get the pending reason:
   ```
   query_slurm_cluster(
       query="Why is job pending and when will it start?",
       commands=[
           "squeue -j {id} -o '%i|%T|%r|%Q|%C|%m|%b|%P' -h",
           "squeue --start -j {id}"
       ]
   )
   ```

2. Interpret the reason:

| Reason | Meaning | Advice |
|--------|---------|--------|
| Priority | Others have higher fairshare | Wait, or try preemptable |
| Resources | Not enough free resources | Wait, or request fewer resources |
| QOSMaxJobsPerUserLimit | Hit QOS job limit | Wait for current jobs to finish |
| ReqNodeNotAvail | Nodes in maintenance/drain | Try different partition |
| PartitionNodeLimit | Too many nodes requested | Reduce node count |
| AssocGrpCPULimit | Account CPU limit reached | Wait or use different account |
| Dependency | Waiting on another job | Check dependent job status |
| BeginTime | Scheduled for future start | Wait for scheduled time |

3. Check estimated start time from `squeue --start`
4. If no start time or very long wait → suggest alternatives:
   - Different GPU type
   - Preemptable partition
   - Fewer resources
   - Different time of day

### Recipe: Diagnose Job Failure

**Trigger:** "My job failed" / "Job X has error"

**Workflow:**
1. Get failure details:
   ```
   query_slurm_cluster(
       query="Job failure details",
       commands=["sacct -j {id} --format=JobID,State,ExitCode,Reason,MaxRSS,Elapsed -P"]
   )
   ```

2. Read output/error files:
   ```
   find_files(pattern="slurm-{id}*", directory=work_dir)
   read_text_file(path="{output_file}")
   ```

3. Diagnose based on exit code and output:

| Exit Code | State | Likely Cause | Fix |
|-----------|-------|-------------|-----|
| 0:0 | COMPLETED | Success (rare to query) | — |
| 1:0 | FAILED | Script error (Python traceback, etc.) | Read output, fix script |
| 0:9 | OUT_OF_MEMORY | OOM killed by CGroups | Request more --mem |
| 0:15 | TIMEOUT | Hit time limit | Request more --time |
| 0:1 | CANCELLED | User or admin cancelled | Check if intentional |
| 137 | FAILED | SIGKILL (OOM without CGroups) | Request more memory |
| 139 | FAILED | SIGSEGV | Bug in code or library |

4. Report diagnosis clearly:
   - What happened (state + exit code)
   - Why (from output file)
   - How to fix (specific recommendation)

### Recipe: Read Job Output

**Trigger:** "Show me the output of job X"

**Workflow:**
1. Find output file:
   ```
   find_files(pattern="slurm-{id}*", directory=work_dir)
   ```
   If not found, check scontrol for StdOut path:
   ```
   query_slurm_cluster(commands=["scontrol show job {id} | grep StdOut"])
   ```

2. Read the file:
   - Small output: read_text_file(path=output_path)
   - Large output (>100 lines): read tail + summarize
     ```
     read_text_file(path=output_path, start_line=-50)
     ```

3. Present relevant portions to user

### Recipe: Cancel a Job

**Trigger:** "Cancel job X" / "Stop my job"

**Workflow:**
1. Confirm with user: "Are you sure you want to cancel job {id} ({job_name})?"
2. Only after confirmation:
   ```
   slurm_cancel_job(job_id="{id}")
   ```
3. Verify cancellation:
   ```
   slurm_monitor_job(job_id="{id}")
   ```
4. Report: "Job {id} has been cancelled."

### Recipe: Check All User Jobs

**Trigger:** "Show all my jobs" / "What's running?"

**Workflow:**
1. Query all user jobs:
   ```
   query_slurm_cluster(
       query="All jobs for user",
       commands=["squeue -u {username} -o '%i|%j|%P|%T|%M|%C|%m|%b|%r' -h"]
   )
   ```
2. Present as formatted table
3. Highlight any issues (PENDING with bad reason, long runtime)

## Failure Diagnosis Protocol

When a job fails:

1. **Read actual error FIRST** — don't guess
   - Check slurm-{id}.out in work_dir
   - Check sacct for exit code and state
   - Look for Python tracebacks, shell errors, OOM messages

2. **Classify the failure:**
   - **Script error** (exit code 1): bug in user's code → read traceback
   - **Resource error** (OOM/timeout): insufficient resources → suggest increase
   - **Environment error** (ModuleNotFoundError): missing package → escalate to software-management
   - **Permission error**: wrong path or partition → fix paths
   - **System error** (node failure): infrastructure issue → resubmit

3. **Recommend specific fix:**
   - Don't just say "increase memory" — say "request --mem=128G (was 64G, peaked at 98G)"
   - Don't just say "fix the script" — point to the exact error line

## Best Practices & Pitfalls

### Always:
- Run the actual Slurm command before reporting status
- Read the output file for failure diagnosis (never speculate)
- Ask confirmation before cancelling
- Report exact numbers from command output
- Use sacct for completed/failed jobs (squeue only shows active)

### Never:
- Don't guess why a job is pending — check the Reason field
- Don't report status without running squeue/sacct
- Don't cancel without explicit user confirmation
- Don't dump >50 lines of raw output — summarize key points
- Don't assume a job succeeded without checking exit code

## Tools

- `slurm_monitor_job` — Check/wait for job status (primary tool)
- `slurm_cancel_job` — Cancel a running/pending job (requires user confirmation)
- `query_slurm_cluster` — Run squeue/sacct/scontrol queries
- `execute_dynamic_task` — Run shell pipelines for precise aggregation (counts, rankings)
- `read_text_file` — Read job output/error files
- `grep_file` — Search output logs for errors
- `find_files` — Locate slurm output files
- `list_directory` — Browse output directory
- `analyze_files` — Analyze multiple job outputs
- `summarize_command_output` — Summarize verbose output
- `run_pipeline_script` — Multi-command diagnostic queries

## References

- `references/job-failure-diagnosis.md` — Load WHEN job has non-obvious failure mode
- `../shared/references/cluster-config.md` — Load WHEN need partition/QOS context for diagnosis
