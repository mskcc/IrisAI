---
name: hpc-query
description: Query cluster status — free GPUs, partition info, node health, queue
  length, cluster utilization, check user access
allowed_tools:
  - query_slurm_cluster
  - check_user_slurm_access
  - execute_dynamic_task
  - batch
  - run_pipeline_script
  - summarize_command_output
  - query_software
  - hpc_directory
model: null
max_iterations: 20
guardrails:
- NEVER report cluster numbers without running the actual Slurm command first
- NEVER estimate or guess — if a command fails, say so
- Always aggregate multi-row sinfo output before reporting (one partition may have multiple rows)
- When output is large (>50 rows) or you need counts/rankings, use execute_dynamic_task with shell pipelines (awk, sort, uniq -c) — do NOT count rows yourself
- NEVER report partition access without running BOTH sacctmgr AND scontrol
- For ANY software path discovery or lookup → call query_software FIRST before checking memory
---

# HPC Cluster Queries

Query the cluster for status information: free GPUs, partition availability,
node health, queue length, utilization, and user access verification.

## When to Use This Skill

**Triggers:**
- "How many free GPUs?" / "Are there any A100s available?"
- "Cluster status" / "How busy is the cluster?"
- "What partitions can I use?" / "Can I access gpu?"
- "Who's using the GPUs?" / "Queue length?"
- "How many nodes are down?" / "Node health"
- "What's the wait time for gpu partition?"

**NOT for (route elsewhere):**
- "Submit a job" → hpc-submit-job
- "Check my job status" → hpc-monitor
- "Run this on the cluster" → hpc-submit-job or code-execution

## Complete Workflow

### Step 1: Understand the Question

What does the user want to know?
- Resource availability (free GPUs, idle nodes, memory)
- Queue status (pending jobs, wait times)
- User access (can I use partition X?)
- Cluster health (down nodes, maintenance)
- Utilization (how busy, who's using what)

### Step 2: Run the Right Query

Choose the appropriate command(s) based on the question type.

### Step 3: Process & Report

- For numerical answers: use execute_dynamic_task with shell aggregation (awk/sort/uniq -c) for precision
- For multi-row output: aggregate with awk before presenting
- Always distinguish: allocated vs idle vs total
- Always show your source command

## Key Recipes

### Recipe: Free GPUs

**Trigger:** "How many free GPUs?" / "Available A100s?"

```
query_slurm_cluster(
    query="Free GPUs by type",
    commands=["sinfo -p gpu -N -o '%N|%G|%T' -h | grep -E 'idle|mix'"]
)
```
For precise counts, pipe through shell:
```
execute_dynamic_task(
    command="sinfo -p gpu -N -o '%N|%G|%T' -h | grep -E 'idle|mix' | awk -F'|' '{print $2}' | sort | uniq -c"
)
```

### Recipe: Cluster Status Overview

**Trigger:** "How's the cluster?" / "Cluster status"

```
query_slurm_cluster(
    query="Cluster overview - nodes, jobs, utilization",
    commands=[
        "sinfo -o '%P|%a|%D|%C|%G' -h | head -20",
        "squeue -t RUNNING -h | wc -l",
        "squeue -t PENDING -h | wc -l"
    ]
)
```

### Recipe: Check User Access

**Trigger:** "Can I use the gpu partition?" / "What partitions do I have access to?"

```
check_user_slurm_access(username="{user}", partition="{partition}")
```

Or for full access picture:
```
query_slurm_cluster(
    query="User's partition access",
    commands=[
        "sacctmgr show assoc user={username} format=Account,Partition,QOS -P --noheader"
    ]
)
```

### Recipe: Queue Wait Time

**Trigger:** "How long is the wait?" / "When will jobs start?"

```
query_slurm_cluster(
    query="Pending job queue depth and estimated starts",
    commands=[
        "squeue -p {partition} -t PENDING -o '%i|%u|%Q|%r|%C|%m|%b' -h | wc -l",
        "squeue -p {partition} -t PENDING --start -o '%i|%S|%r' -h | head -10"
    ]
)
```

### Recipe: Who's Using Resources

**Trigger:** "Who's using the GPUs?" / "Top users"

```
query_slurm_cluster(
    query="Resource usage by user",
    commands=["squeue -p gpu -t RUNNING -o '%u|%b|%P' -h | sort | uniq -c | sort -rn | head -10"]
)
```

### Recipe: Usage by Department / Program

**Trigger:** "Who's using the cluster by department?" / "Break down by program"

Slurm accounts are PI usernames — they don't contain department/program info natively.
Use `hpc_directory(username=...)` to resolve accounts to organizational metadata
(department, program) from the investigators API.

1. Get top accounts from Slurm:
```
execute_dynamic_task(
    command="squeue -t RUNNING -o '%a' -h | sort | uniq -c | sort -rn | head -20"
)
```

2. Enrich each account with department/program:
```
hpc_directory(username="{account_name}")
```

3. Group results by department/program and present to user.

### Recipe: Node Health

**Trigger:** "Any nodes down?" / "Maintenance?"

```
query_slurm_cluster(
    query="Down or draining nodes",
    commands=["sinfo -t down,drain,draining -o '%N|%T|%E' -h"]
)
```

## Accuracy Principles

1. **Always aggregate multi-row output.** sinfo returns multiple rows per partition
   (one per GRES type). Sum across rows for partition totals.

2. **Verify your numbers.** If you compute a total, cross-check with a different
   command/format.

3. **Show your work.** For numerical answers, state what command produced the number.

4. **Never report a single row as the whole partition.** A partition with 4 node
   types has 4 rows — the total is the SUM.

5. **Summarize large output in shell.** Use awk/sort/uniq to reduce >50 rows
   before presenting.

## Slurm Query Reference

**squeue format fields:**
%u user, %P partition, %T state, %C CPUs, %m memory, %b GRES, %i jobid,
%j name, %D nodes, %l timelimit, %M runtime, %N nodelist, %r reason,
%Q priority, %a account, %S start, %e end

**sinfo format fields:**
%P partition, %N nodes, %T state, %C CPUs(A/I/O/T), %m memory, %e free_mem,
%G GRES, %D node_count, %a avail, %F nodes(A/I/O/T), %l timelimit

**Tips:**
- Always use `-h` (suppress headers)
- Use `|` delimiter for reliable parsing
- `%C` in sinfo = "allocated/idle/other/total" (4 numbers / separated)
- Memory in squeue is MB by default
- GRES format: `gpu:type:count`

## Partition Classification (for Display)

| Category | Examples |
|----------|----------|
| General GPU | gpu, gpushort |
| General CPU | cpu, cpushort, cpu_highmem, batch |
| Preemptable | preemptable |
| Interactive | interactive |
| PI/Group | morrisq, lareauc_gpu, componc_cpu |
| Special | datatransfer, debug |

Focus on general-access by default. Mention PI partitions only if user has access.

## Tools

- `query_slurm_cluster` — Run sinfo/squeue/sacctmgr queries (primary)
- `execute_dynamic_task` — Run shell pipelines for precise aggregation (counts, rankings, sums)
- `check_user_slurm_access` — Verify partition access for a user
- `run_pipeline_script` — Multi-command queries with aggregation
- `summarize_command_output` — Summarize verbose output
- `hpc_directory` — Look up users (department, program, full name) or groups (members, PI). Use username= for Slurm accounts

## References

- `../shared/references/cluster-config.md` — Load WHEN need partition details or node families
- `../shared/references/slurm-query-reference.md` — Load WHEN complex query formatting needed
