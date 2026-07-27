# Copyright 2026 Lohit Valleru and contributors at
# Memorial Sloan Kettering Cancer Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastmcp import FastMCP
from shared_auth import StaticBearerProvider
import shlex
import subprocess
import time
import os
from datetime import date
from pathlib import Path
from typing import Annotated
import pwd
import json
import re

from pydantic import Field

from slurm_parsers import (
    parse_sinfo_partitions,
    parse_sacctmgr_associations,
    parse_squeue_jobs,
    parse_sshare,
    parse_sacct_job_status,
    build_cluster_summary,
    parse_job_state_summary,
    parse_cluster_jobs,
    parse_node_efficiency,
    _parse_mem_to_gb,
    _extract_gpu_type,
    parse_node_available_resources,
    parse_pending_job_demands,
    check_job_fitment,
    parse_node_resources_multi_partition,
    aggregate_partition_availability,
    find_runnable_jobs,
    estimate_concurrent_capacity,
    _parse_timelimit_to_minutes,
    parse_scontrol_partition,
    determine_partition_access,
)

mcp = FastMCP("Slurm Job Management Server", auth=StaticBearerProvider())


def _get_work_dir() -> str:
    """Get authoritative work_dir. Checks env var first, then settings file.

    Env var takes precedence: set at container launch, updated by
    set_user_work_directory() in the file_ops MCP server. Settings file
    is the cross-session/cross-container fallback.
    """
    env_wd = os.environ.get("WORK_DIR", "")
    if env_wd:
        return env_wd
    try:
        username = pwd.getpwuid(os.getuid()).pw_name
        app_name = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")
        settings_path = Path(f"/home/{username}/{app_name}/usersettings.json")
        if settings_path.exists():
            with open(settings_path, "r") as f:
                settings = json.load(f)
            work_dir = settings.get("work_dir", "")
            if work_dir:
                return work_dir
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Helper: run a shell command and return stdout (or error string)
# ---------------------------------------------------------------------------
def _run_cmd(cmd: list, timeout: int = 30) -> dict:
    """Run a command, return {"ok": True, "out": stdout} or {"ok": False, "err": ...}."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return {"ok": True, "out": r.stdout.strip()}
        return {"ok": False, "err": r.stderr.strip() or r.stdout.strip()}
    except FileNotFoundError:
        return {"ok": False, "err": f"Command not found: {cmd[0]}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"ok": False, "err": str(e)}


# ---------------------------------------------------------------------------
# Helper: build Singularity exec command for container-wrapped jobs
# ---------------------------------------------------------------------------
def _build_singularity_cmd(container_image: str, work_dir: str,
                           use_gpu: bool = False) -> str:
    """Build a singularity exec command string with standard bind mounts.

    Args:
        container_image: Path to the .sif container image.
        work_dir: User's writable work directory.
        use_gpu: If True, add --nv flag for NVIDIA GPU visibility.

    Returns:
        A multi-line string with the singularity exec command prefix,
        ending with a backslash so the caller can append the actual command.
    """
    SINGULARITY_BIN = os.environ.get("SINGULARITY_BIN", "singularity")
    lines = [f"{SINGULARITY_BIN} exec \\"]

    # GPU flag — must come before bind mounts
    if use_gpu:
        lines.append("  --nv \\")

    # Standard bind mounts (matching OnDemand pattern)
    lines.append("  --no-mount tmp \\")
    lines.append("  -B /var/run/munge:/var/run/munge \\")
    # Bind /data1 subdirs individually as ro (whole-dir mount may not
    # enforce :ro on sub-mounted filesystems — matches template/script.sh.erb)
    for subdir in sorted(Path("/data1").iterdir()):
        if subdir.is_dir():
            lines.append(f"  -B {subdir}:{subdir}:ro \\")
    lines.append("  -B /admin:/admin:ro \\")
    lines.append("  -B /home:/home:ro \\")
    username = pwd.getpwuid(os.getuid()).pw_name
    lines.append(f"  -B /home/{username}:/home/{username}:ro \\")
    lines.append("  -B /scratch:/scratch:ro \\")
    lines.append("  # Add your cluster bind mounts via SINGULARITY_EXTRA_BINDS env var
    # e.g. export SINGULARITY_EXTRA_BINDS="/shared:/shared:ro"\")
    lines.append(f"  -B {work_dir}:{work_dir}:rw \\")
    # Redirect container /tmp to work_dir/.tmp (no host /tmp writes)
    tmp_dir = f"{work_dir}/.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    lines.append(f"  -B {tmp_dir}:/tmp:rw \\")

    # Mount user settings directory as read-write (for software path hints)
    irisai_app_name = os.environ.get("IRISAI_APP_NAME", "IrisAI")
    user_data_dir = f"/home/{username}/{irisai_app_name}"
    if os.path.isdir(user_data_dir):
        lines.append(f"  -B {user_data_dir}:{user_data_dir}:rw \\")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1: query_slurm_cluster
# ---------------------------------------------------------------------------
@mcp.tool
def query_slurm_cluster(
    include_node_details: Annotated[bool, Field(description="If true, return per-node state info from sinfo -N. Only set true when diagnosing why a specific job is not running.", default=False)],
) -> dict:
    """Query the IRIS HPC cluster to discover current partitions, GPU types, node availability, and resource limits. Call this BEFORE submitting a job to understand what resources are available. Also call when user asks about cluster status or available GPUs. DO NOT call repeatedly in the same conversation — call once and reuse results. Returns dict with partitions, gpu_summary, cluster_summary, cluster_job_summary."""
    result = {}
    errors = []

    # 1. Get partition info
    sinfo = _run_cmd([
        "sinfo", "-o", "%P|%l|%a|%F|%G|%C|%m|%D", "--noheader"
    ])
    if sinfo["ok"]:
        result["partitions"] = parse_sinfo_partitions(sinfo["out"])
    else:
        errors.append(f"sinfo failed: {sinfo['err']}")
        result["partitions"] = []

    # 2. Build GPU and cluster summaries
    summaries = build_cluster_summary(result["partitions"])
    result["gpu_summary"] = summaries["gpu_summary"]
    result["cluster_summary"] = summaries["cluster_summary"]

    # 3. Get real job counts from squeue (one lightweight call)
    squeue_states = _run_cmd([
        "squeue", "-a", "--noheader", "-o", "%T"
    ])
    if squeue_states["ok"]:
        result["cluster_job_summary"] = parse_job_state_summary(
            squeue_states["out"]
        )
    else:
        errors.append(f"squeue failed: {squeue_states.get('err', 'no output')}")
        result["cluster_job_summary"] = parse_job_state_summary("")

    # 4. Optional: per-node details for debugging (includes free memory)
    if include_node_details:
        # %e = free memory (MB), critical for diagnosing pending jobs
        node_info = _run_cmd([
            "sinfo", "-N", "-o", "%N|%P|%T|%C|%m|%e|%G", "--noheader"
        ])
        if node_info["ok"]:
            nodes = []
            all_lines = node_info["out"].splitlines()
            for line in all_lines[:50]:  # sample first 50 nodes to avoid context bloat
                parts = line.split("|")
                if len(parts) >= 7:
                    total_mem_mb = parts[4].strip()
                    free_mem_mb = parts[5].strip()
                    # Convert to GB for easier consumption
                    try:
                        total_gb = round(int(total_mem_mb) / 1024, 1)
                    except (ValueError, TypeError):
                        total_gb = 0.0
                    try:
                        free_gb = round(int(free_mem_mb) / 1024, 1)
                    except (ValueError, TypeError):
                        free_gb = 0.0
                    nodes.append({
                        "node": parts[0].strip(),
                        "partition": parts[1].strip(),
                        "state": parts[2].strip(),
                        "cpus": parts[3].strip(),
                        "total_memory_gb": total_gb,
                        "free_memory_gb": free_gb,
                        "memory_utilization_pct": (
                            round((1 - free_gb / total_gb) * 100, 1)
                            if total_gb > 0 else 0.0
                        ),
                        "gres": parts[6].strip(),
                    })
            result["node_details"] = nodes
            result["node_details_total"] = len(all_lines)
        else:
            errors.append(f"sinfo -N failed: {node_info['err']}")

    if errors:
        result["errors"] = errors

    # --- Build pre-formatted summary (Python arithmetic, no Haiku) ---
    lines = []
    lines.append("=== CLUSTER OVERVIEW ===")

    # Partition summary
    parts = result.get("partitions", [])
    lines.append(f"Partitions: {len(parts)} total")
    for p in parts:
        name = p.get("name", "?")
        state_str = p.get("nodes_alloc_idle_other_total", "")
        timelimit = p.get("timelimit", "?")
        gres = p.get("gres", "(none)")
        lines.append(f"  {name}: timelimit={timelimit}, nodes={state_str}, gres={gres}")

    # GPU summary
    gpu_sum = result.get("gpu_summary", {})
    if gpu_sum:
        lines.append("")
        lines.append("--- GPU Summary ---")
        for gtype, gdata in gpu_sum.items():
            if isinstance(gdata, dict):
                lines.append(f"  {gtype}: {gdata}")
            else:
                lines.append(f"  {gtype}: {gdata}")

    # Cluster summary
    cs = result.get("cluster_summary", {})
    if cs:
        lines.append("")
        lines.append("--- Cluster Summary ---")
        for k, v in cs.items():
            lines.append(f"  {k}: {v}")

    # Job summary
    js = result.get("cluster_job_summary", {})
    if js:
        lines.append("")
        lines.append("--- Job Summary ---")
        for k, v in js.items():
            lines.append(f"  {k}: {v}")

    result["summary"] = "\n".join(lines)

    return result


# ---------------------------------------------------------------------------
# Tool 2: check_user_slurm_access
# ---------------------------------------------------------------------------
@mcp.tool
def check_user_slurm_access(
    username: Annotated[str, Field(description="OS username to check. Leave empty to check the current user.", default="")],
) -> dict:
    """Check which Slurm partitions and accounts a user has access to, and their current resource usage vs limits. Call this BEFORE submitting a job to verify the user can use the target partition. DO NOT guess partition access — always verify first. Returns dict with accounts, accessible partitions, QOS limits, current usage."""
    if not username:
        username = os.environ.get("USER", "")
        if not username:
            whoami = _run_cmd(["whoami"])
            username = whoami["out"] if whoami["ok"] else "unknown"

    result = {"username": username}
    errors = []

    # ── Step 1: Get user's Slurm associations (accounts) ──────────────────
    # This tells us WHICH accounts the user belongs to.
    # Having an account does NOT guarantee partition access — see Step 2.
    assoc = _run_cmd([
        "sacctmgr", "show", "assoc", f"user={username}",
        "format=Account,Partition,QOS,MaxTRES,MaxJobs,MaxWall",
        "-P", "--noheader"
    ])
    if assoc["ok"] and assoc["out"]:
        parsed = parse_sacctmgr_associations(assoc["out"])
        result["accounts"] = parsed["accounts"]
        result["associations"] = parsed["associations"]
    else:
        errors.append(f"sacctmgr failed: {assoc.get('err', 'no output')}")
        result["accounts"] = []
        result["associations"] = []

    user_accounts = result["accounts"]

    # ── Step 2: Check actual partition access via DenyAccounts/AllowAccounts ──
    # For each partition the user might want to use, we call
    # `scontrol show partition=<name>` and cross-reference DenyAccounts.
    # This is the AUTHORITATIVE access check — sacctmgr alone is NOT enough.
    partitions_to_check = ["gpu", "gpushort", "cpu", "cpushort", "preemptable"]
    partition_access = {}
    for pname in partitions_to_check:
        sctrl = _run_cmd(["scontrol", "show", f"partition={pname}"])
        if sctrl["ok"] and sctrl["out"]:
            pconfig = parse_scontrol_partition(sctrl["out"])
            access = determine_partition_access(user_accounts, pconfig)
            partition_access[pname] = {
                "can_access": access["can_access"],
                "granted_via": access["granted_via"],
                "denied_accounts": access["denied_accounts"],
                "reason": access["reason"],
                "deny_accounts_list": pconfig["deny_accounts"],
                "allow_accounts_list": pconfig["allow_accounts"],
            }
        else:
            # Partition may not exist on this cluster — skip silently
            partition_access[pname] = {
                "can_access": None,
                "reason": f"Partition '{pname}' not found or scontrol failed.",
            }

    result["partition_access"] = partition_access
    # Convenience: list of partitions the user can actually submit to
    result["accessible_partitions"] = [
        p for p, v in partition_access.items() if v.get("can_access") is True
    ]

    # ── Step 3: Get user's current running/pending jobs (with memory) ─────
    jobs = _run_cmd([
        "squeue", "-u", username,
        "-o", "%i|%P|%j|%T|%M|%l|%C|%m|%b",
        "--noheader"
    ])
    if jobs["ok"]:
        parsed_jobs = parse_squeue_jobs(jobs["out"])
        result["current_jobs"] = parsed_jobs["jobs"]
        result["running_count"] = parsed_jobs["running_count"]
        result["pending_count"] = parsed_jobs["pending_count"]

        # Compute user's total memory usage from running jobs
        total_mem_gb = 0.0
        for j in parsed_jobs["jobs"]:
            if j["state"] == "RUNNING":
                total_mem_gb += _parse_mem_to_gb(j.get("memory", ""))
        result["current_memory_usage_gb"] = round(total_mem_gb, 1)
    else:
        errors.append(f"squeue failed: {jobs.get('err', 'no output')}")

    # ── Step 4: Get user's recent fairshare info ───────────────────────────
    # Use -A flag with user's accounts to limit output to relevant rows only
    # (sshare -u alone still returns all parent accounts)
    fairshare_accounts = ",".join(user_accounts) if user_accounts else ""
    fairshare_cmd = [
        "sshare", "-u", username, "-o",
        "Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare",
        "-P", "--noheader"
    ]
    if fairshare_accounts:
        fairshare_cmd.extend(["-A", fairshare_accounts])
    fairshare = _run_cmd(fairshare_cmd)
    if fairshare["ok"] and fairshare["out"]:
        all_shares = parse_sshare(fairshare["out"])
        # Filter to only rows for this user or their direct accounts
        result["fairshare"] = [
            s for s in all_shares
            if s["user"] == username or s["account"] in user_accounts
        ]

    # ── Step 5: Add cluster-wide context (one lightweight squeue call) ─────
    squeue_states = _run_cmd([
        "squeue", "-a", "--noheader", "-o", "%T"
    ])
    if squeue_states["ok"]:
        result["cluster_context"] = parse_job_state_summary(
            squeue_states["out"]
        )
    else:
        errors.append(
            f"squeue cluster context failed: "
            f"{squeue_states.get('err', 'no output')}"
        )

    if errors:
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Tool 3: query_cluster_jobs (NEW)
# ---------------------------------------------------------------------------
@mcp.tool
def query_cluster_jobs(
    top_n: Annotated[int, Field(description="Number of top users to include in rankings (default: 20). Set lower (e.g. 5) for quick summaries.", default=20, ge=1)],
) -> dict:
    """Cluster-wide job analysis — returns pre-aggregated totals, top users, per-partition breakdown, pending reasons, and zombie detection. This is for CLUSTER-WIDE overview only. WARNING: top_users are NOT filtered by partition. For partition-specific queries use execute_dynamic_task with 'squeue -p <partition>'. For per-job resource demands use query_pending_queue instead. Returns dict with total_running, total_pending, top_users, by_partition, pending_reasons, zombie_jobs."""
    errors = []

    # ONE squeue call with a rich format string
    squeue = _run_cmd([
        "squeue", "-a", "--noheader",
        "-o", "%i|%u|%P|%j|%T|%M|%l|%C|%m|%b|%r"
    ], timeout=60)

    if not squeue["ok"]:
        return {
            "error": f"squeue failed: {squeue['err']}",
            "total_running": 0,
            "total_pending": 0,
            "total_jobs": 0,
        }

    result = parse_cluster_jobs(squeue["out"], top_n=top_n)
    return result


# ---------------------------------------------------------------------------
# Tool 4: query_node_efficiency (NEW)
# ---------------------------------------------------------------------------
@mcp.tool
def query_node_efficiency(
    waste_threshold: Annotated[float, Field(description="Percentage of idle CPUs on a GPU-allocated node above which it's flagged as high waste (default: 50.0).", default=50.0, ge=0.0, le=100.0)],
) -> dict:
    """Analyze cluster resource efficiency — find GPU nodes with wasted CPUs. Returns a compact summary plus only the high-waste outlier nodes. Use when investigating resource waste or answering 'are GPUs being used efficiently?'. Returns dict with cluster_totals, high_waste_nodes, waste_summary."""
    errors = []

    # Call 1: node-level info
    sinfo = _run_cmd([
        "sinfo", "-N", "--noheader",
        "-o", "%N|%P|%T|%C|%m|%G"
    ])
    sinfo_raw = sinfo["out"] if sinfo["ok"] else ""
    if not sinfo["ok"]:
        errors.append(f"sinfo failed: {sinfo['err']}")

    # Call 2: running jobs with node assignments
    squeue = _run_cmd([
        "squeue", "-a", "--noheader", "-t", "RUNNING",
        "-o", "%N|%C|%b"
    ])
    squeue_raw = squeue["out"] if squeue["ok"] else ""
    if not squeue["ok"]:
        errors.append(f"squeue failed: {squeue['err']}")

    result = parse_node_efficiency(
        sinfo_raw, squeue_raw, waste_threshold=waste_threshold
    )

    if errors:
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Tool 5: submit_slurm_job
# ---------------------------------------------------------------------------
@mcp.tool
def submit_slurm_job(
    script_content: Annotated[str, Field(description="Shell commands to run (the job body, NOT #SBATCH lines — those are auto-generated). Must be non-empty.", min_length=1)],
    job_name: Annotated[str, Field(description="Short descriptive job name. Alphanumeric, underscores, hyphens only.", default="irisai_job", pattern=r"^[a-zA-Z0-9_-]+$")],
    partition: Annotated[str, Field(description="Slurm partition name. Empty = cluster default. GPU jobs auto-redirect to GPU partitions.", default="")],
    time_limit: Annotated[str, Field(description="Max wall time in HH:MM:SS or D-HH:MM:SS format. Empty = partition default (~2h).", default="")],
    cpus: Annotated[int, Field(description="CPUs per task (default: 1). Most CPU nodes have 52 cores.", default=1, ge=1)],
    memory: Annotated[str, Field(description="Memory per node, e.g. '4G', '64G', '500G'. Empty = 2GB per CPU. OOM kills if exceeded.", default="")],
    gpus: Annotated[int, Field(description="GPUs to request. 0=none. If >0, cluster auto-redirects to GPU partition.", default=0, ge=0)],
    gpu_type: Annotated[str, Field(description="GPU type: 'a100', 'a40', 'l40s', 'h100', or 'nvidia_h200_nvl'. Empty = any. Must match query_slurm_cluster GRES names.", default="")],
    nodes: Annotated[int, Field(description="Number of nodes (default: 1). Only increase for multi-node MPI jobs.", default=1, ge=1)],
    tasks_per_node: Annotated[int, Field(description="Tasks per node (default: 1). For MPI: set to ranks per node.", default=1, ge=1)],
    array: Annotated[str, Field(description="Job array spec, e.g. '1-100' or '1-100%10' (max 10 concurrent). Empty = no array.", default="")],
    work_dir: Annotated[str, Field(description="Working directory. Empty = user's current work directory.", default="")],
    additional_sbatch: Annotated[str, Field(description="Extra #SBATCH lines verbatim, one per line. For advanced options like '--exclusive'.", default="")],
    container_image: Annotated[str, Field(description="Singularity .sif container path. REQUIRED — bare-metal not allowed. GPUs auto-get --nv flag.", default=os.environ.get("DEFAULT_CONTAINER", "/path/to/containers/mcp_servers_v2.sif"))],
    dependency: Annotated[str, Field(description="Slurm dependency for job chaining. Use 'afterok:JOB_ID' to start this job only after JOB_ID succeeds. For multi-step workflows, chain jobs this way instead of polling. Example: 'afterok:812312'.", default="")],
) -> dict:
    """Submit a Slurm job — pass shell commands and resource requirements. Auto-generates all #SBATCH headers, wraps in a Singularity container, and auto-initializes conda/mamba if detected. Use for ANY task needing Slurm: conda installs, pipelines, GPU jobs, array jobs, or multi-node MPI. Returns dict with job_id, job_dir, script_path. For multi-step workflows: chain jobs with dependency='afterok:PREV_JOB_ID'. To wait for short jobs (<60s): call slurm_monitor_job(job_id, wait=True). NEVER poll in a loop."""

    # GUARD: Container is mandatory — no bare-metal execution allowed
    if not container_image:
        return {
            "error": "container_image is required. All jobs must run inside a container. "
                     "Default: value of DEFAULT_CONTAINER env var (configure in before.sh.erb)",
            "hint": "Do not pass an empty container_image. The default will be used if omitted."
        }

    if not script_content or not script_content.strip():
        return {"error": "script_content cannot be empty"}

    # Determine work directory from settings file (single source of truth)
    if not work_dir:
        work_dir = _get_work_dir()
        if not work_dir:
            work_dir = os.path.expanduser("~")

    # Constrain work_dir to MCP WORK_DIR — prevents model from making
    # arbitrary paths writable via Singularity bind mounts
    mcp_work_dir = os.environ.get("WORK_DIR", "")
    if mcp_work_dir:
        resolved_wd = str(Path(work_dir).resolve())
        resolved_mcp = str(Path(mcp_work_dir).resolve())
        if not resolved_wd.startswith(resolved_mcp + os.sep) and resolved_wd != resolved_mcp:
            work_dir = mcp_work_dir

    # Build the SBATCH script
    lines = ["#!/bin/bash"]

    # Core SBATCH directives
    lines.append(f"#SBATCH --job-name={job_name}")

    if partition:
        lines.append(f"#SBATCH --partition={partition}")

    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")

    lines.append(f"#SBATCH --nodes={nodes}")
    lines.append(f"#SBATCH --ntasks-per-node={tasks_per_node}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")

    if memory:
        lines.append(f"#SBATCH --mem={memory}")

    if gpus > 0:
        if gpu_type:
            lines.append(f"#SBATCH --gres=gpu:{gpu_type}:{gpus}")
        else:
            lines.append(f"#SBATCH --gres=gpu:{gpus}")

    if array:
        lines.append(f"#SBATCH --array={array}")

    if dependency:
        dep = dependency.strip()
        if re.match(r'^(after|afterok|afternotok|afterany|aftercorr)(:\d+)+$', dep):
            lines.append(f"#SBATCH --dependency={dep}")
        else:
            return {"error": f"Invalid dependency format: {dep!r}. Use 'afterok:JOB_ID' format."}

    # Job directory: slurm_jobs/YYYY-MM-DD/jobname_timestamp/
    today = date.today().isoformat()
    timestamp = int(time.time())
    job_dir_name = f"{job_name}_{timestamp}"
    job_dir = os.path.join(work_dir, "slurm_jobs", today, job_dir_name)
    os.makedirs(job_dir, exist_ok=True)
    lines.append(f"#SBATCH --output={job_dir}/stdout.log")
    lines.append(f"#SBATCH --error={job_dir}/stderr.log")

    # Additional SBATCH lines (validated to prevent injection)
    _SBATCH_OPTION_RE = re.compile(r'^#SBATCH\s+--[a-zA-Z][a-zA-Z0-9_-]*(=\S+)?$')
    if additional_sbatch:
        for extra in additional_sbatch.strip().splitlines():
            extra = extra.strip()
            if extra:
                if not extra.startswith("#SBATCH"):
                    extra = f"#SBATCH {extra}"
                # Validate: must be a well-formed #SBATCH directive
                if _SBATCH_OPTION_RE.match(extra):
                    lines.append(extra)
                else:
                    # Skip malformed/suspicious directives
                    pass

    # Blank line before script body
    lines.append("")

    # Wrap in Singularity container (always — container is mandatory)
    if container_image:
        use_gpu = gpus > 0
        lines.append("# --- Running inside Singularity container ---")
        lines.append(f"# Container: {container_image}")
        if use_gpu:
            lines.append("# GPU passthrough enabled (--nv)")
        lines.append("")

        # Detect if script needs conda/mamba shell initialization
        _script_lower = script_content.lower()
        _needs_env_init = any(
            kw in _script_lower
            for kw in ["conda", "mamba", "spack", "pip install"]
        )

        env_preamble = ""
        if _needs_env_init:
            env_preamble = """
# Initialize conda/mamba if available (path-agnostic)
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
if command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash 2>/dev/null)" || true
fi
"""

        # Write the user's script to the job dir
        lines.append("# Write task script for container execution")
        lines.append(f"cat > {job_dir}/task.sh << 'IRISAI_TASK_EOF'")
        lines.append("#!/bin/bash")
        lines.append("set -e")
        lines.append("")
        lines.append(f'echo "=========================================="')
        lines.append(f'echo "SLURM JOB: {job_name}"')
        lines.append(f'echo "=========================================="')
        lines.append(f'echo "Job ID: $SLURM_JOB_ID"')
        lines.append(f'echo "Node: $(hostname)"')
        lines.append(f'echo "Time: $(date)"')
        lines.append(f'echo ""')
        if env_preamble:
            lines.append(env_preamble.strip())
            lines.append("")
        lines.append(script_content.strip())
        lines.append("")
        lines.append('echo ""')
        lines.append('echo "=========================================="')
        lines.append('echo "JOB COMPLETED"')
        lines.append('echo "=========================================="')
        lines.append('echo "Completed: $(date)"')
        lines.append("IRISAI_TASK_EOF")
        lines.append(f"chmod +x {job_dir}/task.sh")
        lines.append("")

        # Clear inherited bind variables from the MCP container environment —
        # they contain container-internal paths (/external, /current_session,
        # /all_sessions) that don't exist on compute node host filesystems.
        lines.append("unset SINGULARITY_BIND APPTAINER_BIND")
        lines.append("")

        # Build the singularity exec command
        sing_cmd = _build_singularity_cmd(container_image, work_dir,
                                          use_gpu=use_gpu)
        lines.append(f"{sing_cmd}")
        lines.append(f"  {shlex.quote(container_image)} \\")
        lines.append(f"  bash {job_dir}/task.sh")
        lines.append("")
        lines.append("EXIT_CODE=$?")
        lines.append('echo "Container exited with code: ${EXIT_CODE}"')
        lines.append("exit ${EXIT_CODE}")
    else:
        lines.append("# --- Job script body (bare metal) ---")
        lines.append(script_content.strip())

    lines.append("")

    script_text = "\n".join(lines)

    # Write the script to disk
    script_path = os.path.join(job_dir, "submit.sh")

    try:
        with open(script_path, "w") as f:
            f.write(script_text)
        os.chmod(script_path, 0o755)
    except Exception as e:
        return {"error": f"Failed to write script: {e}"}

    # Submit via sbatch
    submit = _run_cmd(["sbatch", script_path], timeout=30)
    if not submit["ok"]:
        return {
            "error": f"sbatch failed: {submit['err']}",
            "script_path": script_path,
            "script_content": script_text,
        }

    # Parse job ID from "Submitted batch job 12345"
    job_id = ""
    match = re.search(r"Submitted batch job (\d+)", submit["out"])
    if match:
        job_id = match.group(1)

    return {
        "success": True,
        "job_id": job_id,
        "job_dir": job_dir,
        "script_path": script_path,
        "partition": partition if partition else "(cluster default)",
        "time_limit": time_limit if time_limit else "(partition default)",
        "cpus": cpus,
        "memory": memory if memory else "(default: 2G per CPU)",
        "gpus": gpus,
        "gpu_type": gpu_type if gpu_type else "(any)",
        "nodes": nodes,
        "container": container_image,
        "gpu_passthrough": gpus > 0,
        "output_log": f"{job_dir}/stdout.log",
        "error_log": f"{job_dir}/stderr.log",
    }


# ---------------------------------------------------------------------------
# Existing tools (kept as-is)
# ---------------------------------------------------------------------------


@mcp.tool
def slurm_monitor_job(
    job_id: Annotated[int, Field(description="Numeric Slurm job ID returned by submit_slurm_job.", ge=1)],
    wait: Annotated[bool, Field(description="If True, poll internally until job finishes or max_wait expires. Use for short jobs (<60s). Do NOT use for long-running jobs — use dependency chaining instead.", default=False)],
    max_wait: Annotated[int, Field(description="Max seconds to wait when wait=True. Hard cap: 120s.", default=60, ge=5, le=120)],
) -> dict:
    """Check status of a Slurm job. With wait=False (default): returns current status immediately.
    With wait=True: polls internally (every 5s for first 30s, then every 10s) until job reaches
    terminal state or max_wait expires. Returns timed_out=True if still running after max_wait."""
    job_id_str = str(job_id)
    max_wait = min(max_wait, 120)

    def _check_once():
        sacct = subprocess.run(
            ["sacct", "-j", job_id_str, "--format=State,Elapsed,JobName,Partition,AllocCPUS,MaxRSS,ExitCode,NodeList",
             "-P", "-n"],
            capture_output=True, text=True
        )
        return parse_sacct_job_status(sacct.stdout)

    try:
        result = _check_once()
        if not wait or result.get("finished"):
            result["timed_out"] = False
            result["waited_seconds"] = 0
            return result

        elapsed = 0
        while elapsed < max_wait:
            interval = 5 if elapsed < 30 else 10
            time.sleep(interval)
            elapsed += interval
            result = _check_once()
            if result.get("finished"):
                result["timed_out"] = False
                result["waited_seconds"] = elapsed
                return result

        result["timed_out"] = True
        result["waited_seconds"] = elapsed
        result["hint"] = "Job still running. Chain next step with dependency='afterok:{0}' or report status to user.".format(job_id_str)
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def slurm_cancel_job(
    job_id: Annotated[int, Field(description="Numeric Slurm job ID to cancel.", ge=1)],
) -> dict:
    """Cancel a running or pending Slurm job. Call ONLY when user explicitly requests cancellation. Returns dict with success status and message."""
    job_id = str(job_id)
    try:
        result = subprocess.run(
            ["scancel", job_id], capture_output=True, text=True, check=True
        )
        return {"success": True, "message": result.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}



# ---------------------------------------------------------------------------
# Tool 5: query_pending_queue (NEW — closes the tool gap)
# ---------------------------------------------------------------------------
@mcp.tool
def query_pending_queue(
    check_fitment: Annotated[bool, Field(description="If True, also check if a job with the specified resources can fit on any node now. Set False for just queue analysis.", default=True)],
    cpus: Annotated[int, Field(description="CPUs to check fitment for. 0 = skip fitment check.", default=0, ge=0)],
    memory_gb: Annotated[float, Field(description="Memory in GB to check fitment for. 0 = skip memory fitment.", default=0.0, ge=0.0)],
    gpus: Annotated[int, Field(description="GPUs to check fitment for. 0 = skip GPU fitment.", default=0, ge=0)],
    gpu_type: Annotated[str, Field(description="GPU type to check fitment for, e.g. 'a100'. Empty = any.", default="")],
    partition: Annotated[str, Field(description="Partition to check fitment for. Empty = all partitions.", default="")],
) -> dict:
    """Analyze pending job queue with per-job resource demands and optional fitment check. Use when asking 'why is my job pending?' or 'will my job fit?'. Use query_cluster_jobs for cluster-wide overview instead. Returns dict with pending_demand (totals, sample jobs, by-reason) and optionally fitment_result."""
    result = {}
    errors = []

    # 1. Get pending job details with resource requests
    squeue_pd = _run_cmd([
        "squeue", "-a", "--noheader", "--state=PD",
        "-o", "%i|%u|%P|%j|%C|%m|%b|%l|%r"
    ], timeout=60)

    if squeue_pd["ok"]:
        result["pending_demand"] = parse_pending_job_demands(squeue_pd["out"])
    else:
        errors.append(f"squeue pending failed: {squeue_pd['err']}")
        result["pending_demand"] = parse_pending_job_demands("")

    # 2. Optional: fitment check
    if check_fitment and (cpus > 0 or memory_gb > 0 or gpus > 0):
        # Get node availability (with free memory via %e)
        sinfo_nodes = _run_cmd([
            "sinfo", "-N", "--noheader",
            "-o", "%N|%P|%T|%C|%m|%e|%G"
        ])

        # Get running job GPU allocations
        squeue_running = _run_cmd([
            "squeue", "-a", "--noheader", "-t", "RUNNING",
            "-o", "%N|%C|%b"
        ])

        sinfo_raw = sinfo_nodes["out"] if sinfo_nodes["ok"] else ""
        squeue_raw = squeue_running["out"] if squeue_running["ok"] else ""

        if not sinfo_nodes["ok"]:
            errors.append(f"sinfo for fitment failed: {sinfo_nodes['err']}")
        if not squeue_running["ok"]:
            errors.append(
                f"squeue running for fitment failed: {squeue_running['err']}"
            )

        node_resources = parse_node_available_resources(sinfo_raw, squeue_raw)

        # Get partition time limits for validation
        partition_timelimits = {}
        sinfo_parts = _run_cmd([
            "sinfo", "-o", "%P|%l", "--noheader"
        ])
        if sinfo_parts["ok"]:
            for line in sinfo_parts["out"].strip().splitlines():
                p = line.split("|")
                if len(p) >= 2:
                    pname = p[0].strip().rstrip("*")
                    partition_timelimits[pname] = p[1].strip()

        result["fitment_result"] = check_job_fitment(
            cpus=cpus,
            memory_gb=memory_gb,
            gpus=gpus,
            gpu_type=gpu_type,
            partition=partition,
            node_resources=node_resources,
            partition_timelimits=partition_timelimits,
        )

        # Also include a summary of available resources by GPU type
        gpu_availability = {}
        for n in node_resources:
            if n["gpus_total"] > 0:
                gt = n["gpu_type"] if n["gpu_type"] else "(unknown)"
                if gt not in gpu_availability:
                    gpu_availability[gt] = {
                        "total_gpus": 0,
                        "free_gpus": 0,
                        "nodes_with_free": 0,
                    }
                gpu_availability[gt]["total_gpus"] += n["gpus_total"]
                gpu_availability[gt]["free_gpus"] += n["gpus_free"]
                if n["gpus_free"] > 0:
                    gpu_availability[gt]["nodes_with_free"] += 1

        result["gpu_availability"] = gpu_availability

    if errors:
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Tool: find_runnable_pending_jobs (NEW — bulk fitment analysis)
# ---------------------------------------------------------------------------
@mcp.tool
def find_runnable_pending_jobs(
    partition: Annotated[str, Field(description="Only check jobs in this partition. Empty = all partitions.", default="")],
    user: Annotated[str, Field(description="Only check jobs from this user. Empty = all users.", default="")],
    max_results: Annotated[int, Field(description="Max runnable jobs to return in detail (default: 50).", default=50, ge=1)],
) -> dict:
    """Deep fitment analysis — checks EVERY pending job against EVERY node to find which jobs could run right now. Heavier than query_pending_queue (full cross-join). Use for 'is the cluster actually full or is it priority?' or 'which jobs could run?'. Returns dict with summary, runnable_jobs, blocked_summary, partition_bottlenecks, concurrent_capacity."""
    errors = []

    # 1. Get all pending jobs with resource requests
    squeue_pd = _run_cmd([
        "squeue", "-a", "--noheader", "--state=PD",
        "-o", "%i|%u|%P|%j|%C|%m|%b|%l|%r"
    ], timeout=60)

    if not squeue_pd["ok"]:
        return {"error": f"squeue pending failed: {squeue_pd['err']}"}

    pending_data = parse_pending_job_demands(squeue_pd["out"])
    # We need the full job list for fitment, not the truncated sample.
    # Re-parse to get all jobs (parse_pending_job_demands truncates to 25).
    all_pending_jobs = []
    for line in squeue_pd["out"].strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue
        from slurm_parsers import _extract_gpu_count, _extract_gpu_type as _egt
        gres_str = parts[6].strip()
        mem_str = parts[5].strip()
        try:
            cpus = int(parts[4].strip())
        except (ValueError, TypeError):
            cpus = 0
        gpu_count = _extract_gpu_count(gres_str)
        gpu_type = _egt(gres_str)
        memory_gb = _parse_mem_to_gb(mem_str)
        all_pending_jobs.append({
            "job_id": parts[0].strip(),
            "user": parts[1].strip(),
            "partition": parts[2].strip(),
            "name": parts[3].strip(),
            "cpus": cpus,
            "memory_gb": round(memory_gb, 1),
            "gpus": gpu_count,
            "gpu_type": gpu_type,
            "timelimit": parts[7].strip(),
            "reason": parts[8].strip(),
        })

    # Apply optional filters
    if partition:
        all_pending_jobs = [j for j in all_pending_jobs
                           if j["partition"] == partition]
    if user:
        all_pending_jobs = [j for j in all_pending_jobs
                           if j["user"] == user]

    # 2. Get node resources with multi-partition tracking
    sinfo_nodes = _run_cmd([
        "sinfo", "-N", "--noheader",
        "-o", "%N|%P|%T|%C|%m|%e|%G"
    ])
    squeue_running = _run_cmd([
        "squeue", "-a", "--noheader", "-t", "RUNNING",
        "-o", "%N|%C|%b"
    ])

    sinfo_raw = sinfo_nodes["out"] if sinfo_nodes["ok"] else ""
    squeue_raw = squeue_running["out"] if squeue_running["ok"] else ""

    if not sinfo_nodes["ok"]:
        errors.append(f"sinfo failed: {sinfo_nodes['err']}")
    if not squeue_running["ok"]:
        errors.append(f"squeue running failed: {squeue_running['err']}")

    node_resources = parse_node_resources_multi_partition(sinfo_raw, squeue_raw)

    # 3. Get partition time limits
    partition_timelimits = {}
    sinfo_parts = _run_cmd(["sinfo", "-o", "%P|%l", "--noheader"])
    if sinfo_parts["ok"]:
        for line in sinfo_parts["out"].strip().splitlines():
            p = line.split("|")
            if len(p) >= 2:
                pname = p[0].strip().rstrip("*")
                partition_timelimits[pname] = p[1].strip()

    # 4. Run bulk fitment analysis
    result = find_runnable_jobs(
        pending_jobs=all_pending_jobs,
        node_resources=node_resources,
        partition_timelimits=partition_timelimits,
        max_runnable=max_results,
    )

    # 5. Run concurrent capacity estimate
    concurrent = estimate_concurrent_capacity(
        pending_jobs=all_pending_jobs,
        node_resources=node_resources,
        partition_timelimits=partition_timelimits,
    )
    result["concurrent_capacity"] = {
        "can_run_simultaneously": concurrent["can_run_simultaneously"],
        "total_pending_checked": concurrent["total_pending_checked"],
        "placed_jobs_total": concurrent.get("placed_jobs_total", len(concurrent.get("placed_jobs", []))),
        "remaining_capacity": concurrent["remaining_capacity"],
        "skipped_dependency": concurrent["skipped_dependency"],
        "note": "Greedy first-fit simulation. Actual Slurm scheduling may differ.",
    }

    # 6. Add the pending demand summary for context
    result["pending_demand_summary"] = pending_data.get("demand_summary", "")
    result["filters_applied"] = {
        "partition": partition if partition else "(all)",
        "user": user if user else "(all)",
    }

    if errors:
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Tool: get_partition_availability (NEW — lightweight resource summary)
# ---------------------------------------------------------------------------
@mcp.tool
def get_partition_availability(
    partition: Annotated[str, Field(description="Only show this partition. Empty = show all partitions.", default="")],
) -> dict:
    """LIGHTWEIGHT: answers 'what's free right now?' — aggregated free CPUs, GPUs, memory per partition. Much lighter than query_slurm_cluster. Use when asking 'is there a free A100?' or 'which partition has free GPUs?'. Returns dict with per-partition free_cpus, free_gpus, free_memory_gb, gpu_types, node counts."""
    errors = []

    # Get node resources with multi-partition tracking
    sinfo_nodes = _run_cmd([
        "sinfo", "-N", "--noheader",
        "-o", "%N|%P|%T|%C|%m|%e|%G"
    ])
    squeue_running = _run_cmd([
        "squeue", "-a", "--noheader", "-t", "RUNNING",
        "-o", "%N|%C|%b"
    ])

    sinfo_raw = sinfo_nodes["out"] if sinfo_nodes["ok"] else ""
    squeue_raw = squeue_running["out"] if squeue_running["ok"] else ""

    if not sinfo_nodes["ok"]:
        errors.append(f"sinfo failed: {sinfo_nodes['err']}")
    if not squeue_running["ok"]:
        errors.append(f"squeue running failed: {squeue_running['err']}")

    node_resources = parse_node_resources_multi_partition(sinfo_raw, squeue_raw)
    availability = aggregate_partition_availability(node_resources)

    # Filter to specific partition if requested
    if partition:
        if partition in availability:
            availability = {partition: availability[partition]}
        else:
            availability = {}
            errors.append(
                f"Partition '{partition}' not found or has no usable nodes. "
                f"Available partitions: {sorted(aggregate_partition_availability(node_resources).keys())}"
            )

    result = {
        "partitions": availability,
        "total_partitions": len(availability),
    }

    # --- Build pre-formatted summary (Python arithmetic, no Haiku) ---
    lines = []
    lines.append(f"=== PARTITION AVAILABILITY ({len(availability)} partitions) ===")
    for pname, pdata in sorted(availability.items()):
        gpu_str = ""
        if pdata.get("free_gpus", 0) > 0 or any(
            gd.get("total", 0) > 0 for gd in pdata.get("gpu_types", {}).values()
        ):
            gpu_str = f", GPUs free: {pdata.get('free_gpus', 0)}"
            for gt, gd in pdata.get("gpu_types", {}).items():
                gpu_str += f" [{gt}: {gd.get('free', 0)}/{gd.get('total', 0)}]"
        lines.append(
            f"  {pname}: {pdata.get('free_cpus', 0)} free CPUs, "
            f"{pdata.get('free_memory_gb', 0)}GB free mem"
            f"{gpu_str} | "
            f"{pdata.get('nodes_with_free_cpus', 0)}/{pdata.get('total_nodes', 0)} nodes w/free CPUs"
        )
    result["summary"] = "\n".join(lines)

    if errors:
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Tool: get_cluster_utilization (NEW — comprehensive utilization report)
# ---------------------------------------------------------------------------
@mcp.tool
def get_cluster_utilization(
    partition: Annotated[str, Field(description="Only show this partition. Empty = all partitions.", default="")],
) -> dict:
    """Comprehensive cluster utilization — CPU, memory, AND GPU usage per partition with bottleneck analysis. THE tool for 'how busy is the cluster?'. Returns cluster_wide utilization, per_partition breakdown, per-GPU-type utilization, busiest_partitions ranking, bottleneck_analysis."""
    from slurm_parsers import (
        parse_node_resources_multi_partition,
        compute_cluster_utilization,
    )
    errors = []

    # Get node resources with multi-partition tracking
    sinfo_nodes = _run_cmd([
        "sinfo", "-N", "--noheader",
        "-o", "%N|%P|%T|%C|%m|%e|%G"
    ])
    squeue_running = _run_cmd([
        "squeue", "-a", "--noheader", "-t", "RUNNING",
        "-o", "%N|%C|%b"
    ])

    sinfo_raw = sinfo_nodes["out"] if sinfo_nodes["ok"] else ""
    squeue_raw = squeue_running["out"] if squeue_running["ok"] else ""

    if not sinfo_nodes["ok"]:
        errors.append(f"sinfo failed: {sinfo_nodes['err']}")
    if not squeue_running["ok"]:
        errors.append(f"squeue running failed: {squeue_running['err']}")

    node_resources = parse_node_resources_multi_partition(sinfo_raw, squeue_raw)
    result = compute_cluster_utilization(node_resources)

    # --- Build pre-formatted summary (Python arithmetic, no Haiku) ---
    # This summary is <3000 chars and uses exact values from the computed
    # result. The compression pipeline in sub_agent.py will detect the
    # "summary" key and skip Haiku, preventing rounding/conflation.
    cw = result.get("cluster_wide", {})
    lines = []
    lines.append("=== CLUSTER UTILIZATION SUMMARY ===")
    lines.append(f"CPUs: {cw.get('used_cpus', 0)}/{cw.get('total_cpus', 0)} used ({cw.get('cpu_util_pct', 0)}%)")
    lines.append(f"Memory: {cw.get('used_memory_gb', 0)}/{cw.get('total_memory_gb', 0)} GB used ({cw.get('memory_util_pct', 0)}%)")
    lines.append(f"GPUs: {cw.get('used_gpus', 0)}/{cw.get('total_gpus', 0)} used ({cw.get('gpu_util_pct', 0)}%)")
    lines.append(f"Nodes: {cw.get('total_nodes', 0)} total, {cw.get('unavailable_nodes', 0)} unavailable")
    if cw.get("unavailable_cpus", 0) > 0:
        lines.append(f"Unavailable resources: {cw.get('unavailable_cpus', 0)} CPUs, {cw.get('unavailable_memory_gb', 0)} GB mem, {cw.get('unavailable_gpus', 0)} GPUs")

    # Per-partition breakdown (top 10 busiest)
    busiest = result.get("busiest_partitions", [])[:10]
    if busiest:
        lines.append("")
        lines.append("--- Busiest Partitions (by avg utilization) ---")
        for bp in busiest:
            pname = bp["partition"]
            pdata = result.get("per_partition", {}).get(pname, {})
            gpu_str = ""
            if pdata.get("total_gpus", 0) > 0:
                gpu_str = f", GPU: {pdata.get('used_gpus', 0)}/{pdata.get('total_gpus', 0)} ({pdata.get('gpu_util_pct', 0)}%)"
                # GPU type detail
                for gt, gd in pdata.get("gpu_types", {}).items():
                    gpu_str += f" [{gt}: {gd.get('used', 0)}/{gd.get('total', 0)}]"
            lines.append(
                f"  {pname}: CPU {pdata.get('used_cpus', 0)}/{pdata.get('total_cpus', 0)} ({pdata.get('cpu_util_pct', 0)}%), "
                f"Mem {pdata.get('used_memory_gb', 0)}/{pdata.get('total_memory_gb', 0)}GB ({pdata.get('memory_util_pct', 0)}%)"
                f"{gpu_str} | {pdata.get('total_nodes', 0)} nodes"
            )

    # Bottleneck analysis
    bottlenecks = result.get("bottleneck_analysis", {})
    if bottlenecks:
        lines.append("")
        lines.append("--- Bottlenecks ---")
        for pname, bdata in sorted(bottlenecks.items(), key=lambda x: x[1].get("bottleneck_util_pct", 0), reverse=True)[:10]:
            lines.append(f"  {pname}: {bdata.get('bottleneck', '?')} at {bdata.get('bottleneck_util_pct', 0)}%")

    result["summary"] = "\n".join(lines)

    # Filter to specific partition if requested
    if partition:
        if partition in result["per_partition"]:
            result["per_partition"] = {partition: result["per_partition"][partition]}
            result["bottleneck_analysis"] = {
                k: v for k, v in result["bottleneck_analysis"].items()
                if k == partition
            }
        else:
            available = sorted(result["per_partition"].keys())
            result["per_partition"] = {}
            result["bottleneck_analysis"] = {}
            errors.append(
                f"Partition '{partition}' not found. "
                f"Available: {available}"
            )

    if errors:
        result["errors"] = errors

    return result



if __name__ == "__main__":
    port = int(os.environ.get("MCP_SLURM_PORT", 8003))
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port, path="/")
