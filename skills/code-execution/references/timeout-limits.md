# Execution Timeout Limits

## execute_dynamic_task

| Property | Value |
|----------|-------|
| Hard timeout | 300 seconds (5 minutes) |
| Soft warning | None — kills instantly at limit |
| Recovery | None — process is SIGKILL'd, partial output lost |
| Retry | Must re-run from scratch |

The 300s timeout is enforced by the MCP server process manager. There is NO way
to extend it, request more time, or catch the signal. When the timer fires, the
process tree is killed immediately.

**Consequences of hitting timeout:**
- Partial output may be returned (whatever was flushed to stdout before kill)
- Any files being written may be corrupted (incomplete write)
- Conda installations that timeout leave broken env directories
- No cleanup hooks run

## submit_slurm_job

| Property | Value |
|----------|-------|
| Default time limit | Set by user (required parameter) |
| Maximum | Partition-dependent (up to 7 days on some partitions) |
| Warning | Slurm sends SIGTERM 60s before SIGKILL |
| Recovery | Script can trap SIGTERM for cleanup |

**Best practice:** Always set time_limit explicitly. Slurm default is partition max,
which wastes scheduler priority.

## When to Switch from execute_dynamic_task to submit_slurm_job

Switch when ANY of these apply:
- Task might take > 2 minutes (be conservative — switching is free, timeouts are not)
- Task involves network downloads (unpredictable latency)
- Task processes files > 500MB (I/O time varies)
- Task involves any compilation (even "quick" pip installs with C extensions)
- Task involves conda/mamba (always > 5 min for non-trivial envs)
- Task needs GPU (not available on login nodes)
- Task needs > 8GB RAM (login nodes are shared)
- You're uncertain — default to Slurm (safer, no downside except ~30s queue wait)

## run_pipeline_script

| Property | Value |
|----------|-------|
| Timeout | Same as execute_dynamic_task (300s) |
| Use case | Multi-step composed operations |
| Difference | Runs steps sequentially, reports per-step |

Same timeout rules apply — if any step might be slow, use submit_slurm_job instead.
