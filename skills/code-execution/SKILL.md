---
name: code-execution
description: Run scripts, execute code, quick shell commands — Python, R, shell scripts
  in Singularity containers on HPC. NOT for package installation (use software-management)
  or publication figures (use visualization).
allowed_tools:
  - run_pipeline_script
  - execute_dynamic_task
  - batch
  - submit_slurm_job
  - slurm_monitor_job
  - read_text_file
  - write_text_file
  - edit_file
  - grep_file
  - find_files
  - list_directory
  - review_codebase_section
  - analyze_files
  - summarize_command_output
  - render_image_inline
  - query_software
  - save_image
model: null
max_iterations: 60
guardrails:
- Multi-step workflows (edit + test, or 2+ shell commands) → use `batch` (one call)
- Single quick command (<5 min) → execute_dynamic_task OR batch with one shell op
- Long-running scripts or anything needing GPU → submit_slurm_job
- NEVER install packages here — escalate to software-management skill
- HARD 5-MINUTE LIMIT on execute_dynamic_task — system kills at 300s with no recovery
- After submit_slurm_job, ALWAYS call slurm_monitor_job to wait, then read output
- NEVER attempt conda/pip installs via execute_dynamic_task — they WILL timeout
- For ANY software path discovery or lookup → call query_software FIRST before checking memory
---

# Code Execution

Run scripts and commands on the HPC cluster. This skill handles all forms of code
execution: Python scripts, R scripts, shell commands, data processing pipelines,
and quick computations.

## When to Use This Skill

**Triggers:**
- "Run this script"
- "Execute this Python/R code"
- "Process this file with..."
- "Calculate/compute..."
- "Parse this data"
- "Convert this file format"
- "Run my pipeline"
- Any request to execute code that produces data output (not figures)

**NOT for (route elsewhere):**
- "Install X" / "Create conda env" → software-management
- "Make a plot" / "Generate figure" / "Visualize" → visualization
- "Submit a job" (with complex Slurm requirements) → hpc-submit-job
- "Build a container" → container-building
- "Align reads" / "Run STAR" / "Cell Ranger" → sequence-processing

## Defaults & Conventions

- Default execution: execute_dynamic_task (for commands completing in <2 minutes)
- Default timeout: 300 seconds (hard kill, no extension)
- Default container: Agent's Singularity container (already active)
- Output location: User's work_dir (from system context)
- Script location: Write to work_dir before executing

## Complete Workflow

### Step 1: Context & Discovery (ALWAYS do this first)

Before writing or running ANY code:

1. **Check project memory:**
   → read_memory(project)
   → Look for: previously discovered conda env paths, working scripts, data locations
   → If a conda env was set up in prior sessions, use it directly

2. **Discover software environment** (MANDATORY before importing libraries):
   → get_environment_info('packages') — see the full software registry
   → get_environment_info('package:<name>') — read usage knowledge for specific packages
   → The knowledge file contains API patterns, pitfalls, and version-specific guidance
   → If the package isn't in the registry, check if a conda env already exists in work_dir

3. **Check user context:**
   → work_dir from system context = where to write scripts and output
   → project_dir from system context = the active project directory
   → NEVER construct paths from scratch — use what's provided

4. **Assess execution method:**
   → Will this take <2 minutes? → execute_dynamic_task
   → Will this take 2-5 minutes? → execute_dynamic_task (borderline, but OK)
   → Will this take >5 minutes? → submit_slurm_job
   → Does it need GPU? → submit_slurm_job with GPU partition
   → Does it need >8GB RAM? → submit_slurm_job with appropriate memory

### Step 2: Write the Script

Write ONE comprehensive script rather than multiple small tool calls:

```
write_text_file(
    path="{work_dir}/run_task.py",  # or .R, .sh
    content="..."
)
```

**Script requirements:**
- Self-contained: sets up its own paths, imports, error handling
- Clear output: prints results in a parseable format
- Error handling: catches exceptions, prints informative messages
- Exit codes: non-zero on failure so the wrapper detects it

**Script patterns by language:**

Python:
```python
#!/usr/bin/env python3
import sys
try:
    # ... actual work ...
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
```

R:
```r
#!/usr/bin/env Rscript
tryCatch({
    # ... actual work ...
    cat(sprintf("Result: %s\n", result))
}, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message), file = stderr())
    quit(status = 1)
})
```

Shell:
```bash
#!/bin/bash
set -euo pipefail
# ... actual work ...
echo "Result: ${result}"
```

### Step 3: Execute

**For quick commands (< 2 min):**

```
execute_dynamic_task(
    task_description="Run the processing script",
    commands=["python3 {work_dir}/run_task.py"],
    work_dir="{work_dir}"
)
```

**For longer tasks or GPU work:**

```
submit_slurm_job(
    job_name="process_data",
    script_content="#!/bin/bash\npython3 {work_dir}/run_task.py",
    work_dir="{work_dir}",
    time_limit="00:30:00",
    memory="8G",
    cpus=4
)
```

Then ALWAYS monitor:
```
slurm_monitor_job(job_id="{job_id}", wait=true)
```

Then read output:
```
read_text_file(path="{work_dir}/slurm-{job_id}.out")
```

### Step 4: Verify & Report

After execution completes:
1. Check exit code / job status
2. Read output file(s) to confirm expected results
3. If producing data files: verify they exist and have expected size
4. Report results clearly to the user

If an error occurred:
1. Read the FULL error output — don't just look at the last line
2. Diagnose: is it a missing package? → escalate to software-management
3. Diagnose: is it a path error? → check work_dir bindings
4. Diagnose: is it a timeout? → switch to submit_slurm_job
5. Fix and re-run (max 3 attempts before asking user for help)

### Step 5: Persist Knowledge

After completing work:
- If you discovered a working conda env path → update_memory(knowledge)
- If you found a non-obvious execution pattern → update_memory(knowledge)
- If a script will be reused → note its path in memory

## Tool Selection Decision Tree

```
User request arrives
│
├─ Is it a one-liner? (ls, cat, grep, wc, head, date)
│  └─ execute_dynamic_task — single command, no script needed
│
├─ Is it a short script? (<2 min runtime, no GPU, <8GB RAM)
│  └─ write_text_file + execute_dynamic_task
│
├─ Does it need GPU or >8GB or will run >5 min?
│  └─ write_text_file + submit_slurm_job + slurm_monitor_job
│
├─ Is it a multi-step pipeline? (3+ sequential operations)
│  └─ write ONE comprehensive script + submit_slurm_job
│
├─ Does it need package installation?
│  └─ STOP — escalate to software-management
│     request_additional_skill("software-management")
│
└─ Does it produce a publication figure?
   └─ STOP — escalate to visualization
      request_additional_skill("visualization")
```

## Key Recipes

### Recipe: Run a Python Script

**Trigger:** "Run this Python script" / "Execute my_script.py"

**Workflow:**
1. get_environment_info('packages') → check if needed libraries exist
2. read_text_file(script_path) → understand what the script does
3. Assess runtime: short → execute_dynamic_task, long → submit_slurm_job
4. Execute with appropriate tool
5. Read output, report to user

### Recipe: Process Data Files

**Trigger:** "Process all .csv files in this directory" / "Parse this JSON"

**Workflow:**
1. find_files(pattern="*.csv", directory=data_dir) → discover inputs
2. get_environment_info('packages') → check pandas/polars available
3. Write a processing script that handles all files
4. Execute (choose tool by estimated runtime)
5. Report: files processed, output location, any errors

### Recipe: Quick Shell Command

**Trigger:** "How many lines in this file?" / "What's in this directory?"

**Workflow:**
1. execute_dynamic_task(commands=["wc -l {file}"]) → direct execution
2. Report result

No script writing needed for one-liners.

### Recipe: Multi-Step Data Pipeline

**Trigger:** "First filter, then transform, then aggregate this data"

**Workflow:**
1. get_environment_info('packages') → confirm tools available
2. Write ONE comprehensive script that does all steps sequentially:
   - Step 1: Read input
   - Step 2: Filter
   - Step 3: Transform
   - Step 4: Aggregate
   - Step 5: Write output
   - Each step prints progress
3. submit_slurm_job (pipelines are safer as Slurm jobs — no timeout risk)
4. slurm_monitor_job → wait for completion
5. Read output, verify, report

### Recipe: Run with Specific Conda Environment

**Trigger:** "Run using my pytorch env" / "Execute with the scanpy environment"

**Workflow:**
1. read_memory(project) → check if env path is known
2. If not known: find_files(pattern="conda_env*", directory=work_dir)
3. Write script with conda activation:
   ```bash
   #!/bin/bash
   source /opt/miniforge3/etc/profile.d/conda.sh
   conda activate /path/to/env
   python3 script.py
   ```
4. Execute with appropriate tool
5. Save env path to memory for future: update_memory(knowledge)

### Recipe: Debug a Failing Script

**Trigger:** "This script is failing" / "Why does this error?"

**Workflow:**
1. read_text_file(script_path) → understand the code
2. Execute with verbose output / debug flags
3. Read error output carefully — full traceback, not just last line
4. Diagnose category:
   - ImportError → missing package → escalate to software-management
   - FileNotFoundError → wrong path → fix path using work_dir context
   - MemoryError → needs more RAM → re-submit with higher memory
   - TimeoutError → needs Slurm → switch to submit_slurm_job
5. Apply fix, re-run
6. Max 3 fix iterations before asking user

## Execution Environment Details

### Singularity Container Architecture

The IrisAI agent runs INSIDE a Singularity container on the HPC cluster:
- Container has its own filesystem overlay
- Host filesystems are bind-mounted: /data1, /home, /scratch
- GPU passthrough via `--nv` flag (handled automatically by Slurm submission)
- Conda environments on host are accessible via bind mounts
- System packages CANNOT be installed (no root inside container)
- /tmp is writable and local to the node

### Filesystem Visibility

| Path | Writable? | Notes |
|------|-----------|-------|
| work_dir (/data1/...) | YES | Primary workspace — write scripts and output here |
| /home/user | READ-ONLY | Home is mounted read-only in Slurm containers |
| /tmp | YES | Local scratch — fast but not persistent |
| /scratch | YES | Shared scratch — persistent but quota-limited |
| Container internals | NO | /opt, /usr, etc. are read-only overlay |

### Conda Activation in Scripts

When a script needs a conda environment, use this pattern:
```bash
#!/bin/bash
# Initialize conda (works whether in container or not)
eval "$(conda shell.bash hook)"
# OR the explicit path:
source /opt/miniforge3/etc/profile.d/conda.sh

# Activate by prefix path (most reliable)
conda activate /data1/user/work_dir/conda_envs/myenv

# Now run Python from that env
python3 script.py
```

NEVER use `source /path/to/env/bin/activate` — that is virtualenv syntax
and WILL FAIL for conda prefix environments.

### Resource Limits for execute_dynamic_task

- **Time:** 300 seconds HARD KILL — no warning, no cleanup
- **Memory:** Whatever the login node provides (~16GB typically)
- **CPU:** Single node, shared with other users
- **GPU:** NONE — use submit_slurm_job for GPU work
- **Network:** Available (can download small files)
- **Disk:** Write to work_dir only

### When to Use submit_slurm_job Instead

ALWAYS prefer submit_slurm_job when:
- Script might take > 2 minutes (even if you think it's fast — be safe)
- Script needs GPU access
- Script needs > 8GB RAM
- Script processes large files (>1GB input)
- Script runs a trained model (inference)
- Script involves any compilation
- You're unsure about runtime → default to Slurm (safer)

## Error Diagnosis Protocol

When a script produces errors, follow this escalation:

**Attempt 1: Obvious fix**
- Read full error output
- Fix the immediate cause (typo, wrong path, missing argument)
- Re-run

**Attempt 2: Deeper investigation**
- Layer-test: isolate which component fails
  - Can you import the library? (`python -c "import X"`)
  - Is the input file readable? (`head -1 file`)
  - Does a minimal version work? (strip to 5 lines)
- Fix based on layer test results

**Attempt 3: Different approach**
- If the same error persists, the hypothesis is WRONG
- Try a fundamentally different method (different library, different algorithm)
- Or escalate: "I've tried X and Y but both fail because Z. Shall I try W?"

**After 3 attempts: STOP and report**
- Do NOT keep cycling through variations
- Tell the user what you've tried, what the error is, what you think the root cause is
- Ask for guidance or suggest escalation to another skill

## Best Practices & Pitfalls

### Always:
- Write self-contained scripts (no dependency on prior tool calls in same session)
- Check exit codes — non-zero means failure
- Use work_dir for ALL file output (it's guaranteed writable)
- Read get_environment_info BEFORE importing unfamiliar libraries
- Print clear progress messages in scripts (so output is readable)

### Never:
- Install packages in this skill (escalate to software-management)
- Use execute_dynamic_task for anything that MIGHT exceed 5 minutes
- Assume a package is available without checking get_environment_info
- Write to /home (read-only in Slurm containers)
- Use `sudo` or attempt system-wide changes
- Run infinite loops or daemons via execute_dynamic_task

### Common Pitfalls:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| "ModuleNotFoundError" | Package not in base env | Check get_environment_info; use existing conda env or escalate to software-management |
| "Killed" (no error) | OOM or timeout | Switch to submit_slurm_job with more memory/time |
| "Permission denied" on /home | Slurm container mounts home read-only | Write to work_dir instead |
| Script hangs then dies | Hit 300s timeout | Switch to submit_slurm_job |
| "conda: not found" | Shell not initialized | Add `eval "$(conda shell.bash hook)"` to script |
| Wrong Python version | Multiple pythons on PATH | Use explicit path: /path/to/env/bin/python3 |
| "No space left on device" | /tmp full or quota hit | Use work_dir on /data1 instead of /tmp |

## Tools

- `execute_dynamic_task` — Run shell commands with <5min timeout (quick tasks only)
- `submit_slurm_job` — Submit a script as a Slurm batch job (anything >2min or needing GPU/memory)
- `slurm_monitor_job` — Wait for and check status of a submitted Slurm job
- `run_pipeline_script` — Execute a multi-step pipeline script
- `write_text_file` — Write a script to disk before execution
- `edit_file` — Modify an existing script
- `read_text_file` — Read script content or output files
- `grep_file` — Search for patterns in files
- `find_files` — Locate files by pattern
- `list_directory` — Browse directory contents
- `review_codebase_section` — Understand code structure
- `analyze_files` — Get file analysis
- `summarize_command_output` — Summarize verbose command output
- `render_image_inline` — Display generated images in chat
- `save_image` — Save images to disk

## References

- `references/timeout-limits.md` — Load WHEN user hits timeout or asks about execution limits
- `../shared/references/conda-in-containers.md` — Load WHEN script needs conda activation in a Slurm job
- `../shared/references/cluster-config.md` — Load WHEN choosing partition/resources for submit_slurm_job
