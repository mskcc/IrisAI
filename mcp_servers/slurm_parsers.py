"""Structured output parsers for Slurm command output.

These are pure-logic functions that convert raw Slurm command output
into clean, structured JSON-friendly dicts.
They contain NO subprocess calls and NO I/O — making them fully testable.

Used by mcp_servers/slurm_management.py to parse sinfo, sacctmgr,
squeue, sshare, and sacct output at the source before returning
results to the supervisor.
"""

import re
from collections import Counter
import copy


# ---------------------------------------------------------------------------
# Nodelist expansion utility
# ---------------------------------------------------------------------------

def expand_nodelist(nodelist: str) -> list:
    """Expand a Slurm compressed nodelist into individual node names.

    Handles all common Slurm nodelist formats:
        "node01"                    → ["node01"]
        "node01,node02"             → ["node01", "node02"]
        "node[01-04]"               → ["node01", "node02", "node03", "node04"]
        "node[01,03,05]"            → ["node01", "node03", "node05"]
        "node[01-03,05]"            → ["node01", "node02", "node03", "node05"]
        "gpu[01-02],cpu[01-03]"     → ["gpu01", "gpu02", "cpu01", "cpu02", "cpu03"]
        "rack1-node[1-3]"           → ["rack1-node1", "rack1-node2", "rack1-node3"]

    Pure Python — no subprocess calls. Safe for use in pure-logic parsers.

    Args:
        nodelist: Slurm compressed nodelist string.

    Returns:
        List of individual node name strings.
    """
    if not nodelist or not nodelist.strip():
        return []

    result = []
    # Split on commas that are NOT inside brackets
    # We do this by tracking bracket depth
    parts = []
    current = []
    depth = 0
    for ch in nodelist:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if this part has a bracket range
        bracket_match = re.match(r'^(.*?)\[(.+)\](.*)$', part)
        if bracket_match:
            prefix = bracket_match.group(1)
            range_spec = bracket_match.group(2)
            suffix = bracket_match.group(3)
            # Parse the range spec: "01-04" or "01,03,05" or "01-03,05"
            for segment in range_spec.split(','):
                segment = segment.strip()
                if '-' in segment:
                    start_str, end_str = segment.split('-', 1)
                    # Preserve zero-padding width
                    width = len(start_str)
                    try:
                        start_val = int(start_str)
                        end_val = int(end_str)
                        # Guard against excessively large ranges
                        if end_val - start_val > 10000:
                            result.append(f"{prefix}{segment}{suffix}")
                        else:
                            for i in range(start_val, end_val + 1):
                                result.append(f"{prefix}{str(i).zfill(width)}{suffix}")
                    except (ValueError, TypeError):
                        # Can't parse — keep as-is
                        result.append(f"{prefix}{segment}{suffix}")
                else:
                    result.append(f"{prefix}{segment}{suffix}")
        else:
            result.append(part)

    return result


# ---------------------------------------------------------------------------
# Memory parsing helper
# ---------------------------------------------------------------------------

def _parse_mem_to_gb(mem_str: str) -> float:
    """Convert a Slurm memory string to gigabytes.

    Handles all formats seen in squeue %m output:
        "16G"       → 16.0
        "512M"      → 0.5
        "4096M"     → 4.0
        "1T"        → 1024.0
        "256000"    → 0.244  (bare number = megabytes by Slurm default)
        "256000M"   → 250.0
        "0"         → 0.0
        ""          → 0.0
        "16Gn"      → 16.0  (per-node suffix)
        "4Gc"       → 4.0   (per-cpu suffix)

    Args:
        mem_str: Memory string from Slurm output.

    Returns:
        Memory in GB as a float. Returns 0.0 for unparseable input.
    """
    if not mem_str:
        return 0.0

    s = mem_str.strip()
    if not s or s in ("0", "0M", "0G"):
        return 0.0

    # Strip per-node (n) or per-cpu (c) suffix if present
    if s and s[-1] in ("n", "c"):
        s = s[:-1]

    # Try to parse with unit suffix
    s_upper = s.upper()
    try:
        if s_upper.endswith("T"):
            return float(s_upper[:-1]) * 1024.0
        elif s_upper.endswith("G"):
            return float(s_upper[:-1])
        elif s_upper.endswith("M"):
            return float(s_upper[:-1]) / 1024.0
        elif s_upper.endswith("K"):
            return float(s_upper[:-1]) / (1024.0 * 1024.0)
        else:
            # Bare number — Slurm default is megabytes
            return float(s) / 1024.0
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Time limit parsing helper
# ---------------------------------------------------------------------------

def _parse_timelimit_to_minutes(tl_str: str) -> float:
    """Convert a Slurm time limit string to minutes.

    Handles all formats seen in squeue %l and sinfo %l output:
        "1:00:00"       → 60.0       (HH:MM:SS)
        "2-00:00:00"    → 2880.0     (D-HH:MM:SS)
        "30:00"         → 30.0       (MM:SS)
        "UNLIMITED"     → float('inf')
        "INVALID"       → float('inf')  (treat as no limit)
        ""              → float('inf')  (treat as no limit)
        "NOT_SET"       → float('inf')

    Args:
        tl_str: Time limit string from Slurm output.

    Returns:
        Time limit in minutes as a float. Returns inf for unparseable/unlimited.
    """
    if not tl_str:
        return float('inf')

    s = tl_str.strip().upper()
    if not s or s in ('UNLIMITED', 'INVALID', 'NOT_SET', 'N/A'):
        return float('inf')

    try:
        # Format: D-HH:MM:SS
        if '-' in s:
            day_part, time_part = s.split('-', 1)
            days = int(day_part)
            parts = time_part.split(':')
            if len(parts) == 3:
                hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                hours, mins = int(parts[0]), int(parts[1])
                secs = 0
            else:
                return float('inf')
            return days * 1440 + hours * 60 + mins + secs / 60.0

        parts = s.split(':')
        if len(parts) == 3:
            # HH:MM:SS
            hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 60 + mins + secs / 60.0
        elif len(parts) == 2:
            # MM:SS
            mins, secs = int(parts[0]), int(parts[1])
            return mins + secs / 60.0
        elif len(parts) == 1:
            # Just minutes
            return float(parts[0])
        else:
            return float('inf')
    except (ValueError, TypeError):
        return float('inf')

# ---------------------------------------------------------------------------
# Pending reason interpretations — helps the AI avoid misdiagnosis
# ---------------------------------------------------------------------------

PENDING_REASON_INTERPRETATIONS = {
    "Priority": (
        "Job is next in queue by priority order. May be waiting for: "
        "(a) higher-priority job reservation to clear, (b) memory to free up "
        "on fitting nodes, or (c) backfill cycle to evaluate. Does NOT mean "
        "job is blocked by low priority score alone."
    ),
    "Resources": (
        "Waiting for sufficient resources (CPUs, memory, GPUs, or nodes) to "
        "become available. The specific scarce resource is not indicated — "
        "check node free memory and CPU availability to determine the bottleneck."
    ),
    "ReqNodeNotAvail": (
        "Requested node(s) are not available — may be down, drained, reserved, "
        "or in maintenance. Check node states with sinfo -R."
    ),
    "Dependency": (
        "Waiting for a dependent job to complete. If the dependency job failed "
        "or was cancelled, this job will wait forever (zombie). Check the "
        "dependency job's status."
    ),
    "DependencyNeverSatisfied": (
        "ZOMBIE: The dependency job failed, was cancelled, or never existed. "
        "This job will NEVER run. It should be cancelled."
    ),
    "JobHeldAdmin": (
        "ZOMBIE: Job is held by an administrator. It will not run until "
        "an admin releases it. Contact cluster support."
    ),
    "JobHeldUser": (
        "Job is held by the user. Release with: scontrol release <job_id>"
    ),
    "QOSMaxJobsPerUserLimit": (
        "User has reached the maximum number of running jobs allowed by their "
        "QOS. Other jobs must complete before this one can start."
    ),
    "QOSMaxGRESPerUser": (
        "User has reached the maximum GPU allocation allowed by their QOS."
    ),
    "AssocMaxJobsLimit": (
        "Account/association has reached its maximum running jobs limit."
    ),
    "PartitionNodeLimit": (
        "Job requests more nodes than the partition allows."
    ),
    "PartitionTimeLimit": (
        "Job's time limit exceeds the partition's maximum wall time."
    ),
    "BadConstraints": (
        "Job requests a feature or constraint that no node can satisfy."
    ),
}


# ---------------------------------------------------------------------------
# Partition classification
# ---------------------------------------------------------------------------

# Well-known general partitions (not PI-owned).
# Partition names are lowercased before lookup.
_GENERAL_PARTITIONS = frozenset({
    "cpu", "cpushort", "cpu_highmem", "batch",
    "gpu", "gpushort", "gpu_project", "gputest",
    "all", "interactive", "preemptable", "datatransfer",
    "debug",
})


def classify_partition(name: str, has_gpu: bool) -> str:
    """Classify a partition into a human-friendly type category.

    Classification is based on the partition **name** (the authoritative
    signal for intended purpose), NOT on whether the partition happens to
    share nodes that have GPUs.  For example, the 'cpu' partition may
    contain some GPU nodes (shared with the 'gpu' partition), but its
    intended purpose is CPU jobs, so it is classified as 'general_cpu'.

    Categories returned:
        general_gpu   — General-access GPU partition (gpu, gpushort, …)
        general_cpu   — General-access CPU partition (cpu, cpushort, batch, …)
        pi_gpu        — PI-owned partition intended for GPU jobs
        pi_cpu        — PI-owned partition intended for CPU jobs
        preempt       — Preemptable / scavenger partition
        interactive   — Interactive partition
        other         — Catch-all (all, test, datatransfer, …)

    Args:
        name: Partition name (e.g. 'gpu', 'componc_cpu', 'preemptable').
        has_gpu: Whether the partition has any GPU GRES (from sinfo).
                 Used as a fallback for PI partitions whose names don't
                 contain 'gpu' or 'cpu'.

    Returns:
        One of the category strings listed above.
    """
    low = name.lower()

    # --- Preempt (check first — some PI partitions have 'preem' in name) ---
    if "preempt" in low or "preem" in low:
        return "preempt"

    # --- Interactive ---
    if low == "interactive":
        return "interactive"

    # --- General partitions (well-known names) ---
    if low in _GENERAL_PARTITIONS:
        # Distinguish GPU vs CPU by name keywords
        if "gpu" in low:
            return "general_gpu"
        if "cpu" in low or low in ("batch", "datatransfer", "debug"):
            return "general_cpu"
        # 'all' and any other general partition → other
        return "other"

    # --- PI partitions (everything else) ---
    # Use name keywords first, then fall back to has_gpu flag
    if "gpu" in low:
        return "pi_gpu"
    if "cpu" in low or "pipeline" in low:
        return "pi_cpu"

    # No keyword in name — use has_gpu as fallback
    if has_gpu:
        return "pi_gpu"
    return "pi_cpu"


# ---------------------------------------------------------------------------
# Slurm parsers
# ---------------------------------------------------------------------------

def parse_sinfo_partitions(raw: str) -> list:
    """Parse sinfo -o '%P|%l|%a|%F|%G|%C|%m|%D' output into partition dicts.

    Deduplicates partitions that appear on multiple rows due to different
    GRES types (e.g. same partition listed once for gpu:a100 and once for
    gpu:l40s).  Node/CPU counts come from the FIRST row seen for each
    partition name; GPU types are merged across all rows.

    Args:
        raw: Raw stdout from sinfo with pipe-delimited fields.

    Returns:
        List of partition dicts with structured fields.
    """
    # First pass: collect all rows grouped by partition name
    partition_rows = {}  # name -> list of parsed row dicts
    row_order = []       # preserve first-seen order

    for line in raw.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 8:
            continue
        name = parts[0].rstrip("*")
        is_default = parts[0].endswith("*")
        timelimit = parts[1].strip()
        avail = parts[2].strip()
        node_states = parts[3].strip()
        gres = parts[4].strip()
        cpus_info = parts[5].strip()
        max_mem_mb = parts[6].strip()

        # Parse GPU info from GRES
        gpu_types = []
        gpu_total = 0
        if gres and gres != "(null)":
            for g in gres.split(","):
                g = g.strip()
                if g.startswith("gpu:"):
                    gpu_parts = g.split(":")
                    if len(gpu_parts) >= 3:
                        gpu_types.append(gpu_parts[1])
                        try:
                            gpu_total += int(gpu_parts[2])
                        except ValueError:
                            pass
                    elif len(gpu_parts) == 2:
                        try:
                            gpu_total += int(gpu_parts[1])
                        except ValueError:
                            gpu_types.append(gpu_parts[1])

        # Parse node state counts: A/I/O/T
        node_alloc, node_idle, node_other, node_total = 0, 0, 0, 0
        ns_parts = node_states.split("/")
        if len(ns_parts) == 4:
            try:
                node_alloc = int(ns_parts[0])
                node_idle = int(ns_parts[1])
                node_other = int(ns_parts[2])
                node_total = int(ns_parts[3])
            except ValueError:
                pass

        # Parse CPU counts: A/I/O/T
        cpu_alloc, cpu_idle, cpu_other, cpu_total = 0, 0, 0, 0
        cpu_parts = cpus_info.split("/")
        if len(cpu_parts) == 4:
            try:
                cpu_alloc = int(cpu_parts[0])
                cpu_idle = int(cpu_parts[1])
                cpu_other = int(cpu_parts[2])
                cpu_total = int(cpu_parts[3])
            except ValueError:
                pass

        row = {
            "name": name,
            "is_default": is_default,
            "timelimit": timelimit,
            "available": avail,
            "nodes_total": node_total,
            "nodes_idle": node_idle,
            "nodes_allocated": node_alloc,
            "nodes_other": node_other,
            "cpus_total": cpu_total,
            "cpus_idle": cpu_idle,
            "cpus_allocated": cpu_alloc,
            "max_mem_per_node_mb": max_mem_mb,
            "gpu_types": gpu_types,
            "gpu_total_per_node": gpu_total,
            "gres_raw": gres if gres != "(null)" else "",
        }

        if name not in partition_rows:
            partition_rows[name] = []
            row_order.append(name)
        partition_rows[name].append(row)

    # Second pass: deduplicate — merge GPU types, keep first row's counts
    partitions = []
    for name in row_order:
        rows = partition_rows[name]
        first = rows[0]

        # Merge GPU types from all rows for this partition
        all_gpu_types = set()
        all_gres_raw = []
        has_any_gpu = False
        for r in rows:
            all_gpu_types.update(r["gpu_types"])
            if r["gres_raw"]:
                all_gres_raw.append(r["gres_raw"])
            if r["gpu_types"] or r["gpu_total_per_node"] > 0:
                has_any_gpu = True

        partitions.append({
            "name": first["name"],
            "is_default": first["is_default"],
            "timelimit": first["timelimit"],
            "available": first["available"],
            "nodes_total": first["nodes_total"],
            "nodes_idle": first["nodes_idle"],
            "nodes_allocated": first["nodes_allocated"],
            "nodes_other": first["nodes_other"],
            "cpus_total": first["cpus_total"],
            "cpus_idle": first["cpus_idle"],
            "cpus_allocated": first["cpus_allocated"],
            "max_mem_per_node_mb": first["max_mem_per_node_mb"],
            "gpu_types": sorted(all_gpu_types) if all_gpu_types else [],
            "has_gpu": has_any_gpu,
            "gres_raw": ", ".join(all_gres_raw) if all_gres_raw else "",
            "partition_type": classify_partition(first["name"], has_any_gpu),
        })
    return partitions


def parse_sacctmgr_associations(raw: str) -> dict:
    """Parse sacctmgr show assoc output into structured user access data.

    Args:
        raw: Raw stdout from sacctmgr with pipe-delimited fields
             (Account|Partition|QOS|MaxTRES|MaxJobs|MaxWall).

    Returns:
        Dict with 'accounts', 'accessible_partitions', and 'associations'.
    """
    associations = []
    accounts = set()
    accessible_partitions = set()

    for line in raw.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        acct = parts[0].strip()
        part = parts[1].strip()
        qos = parts[2].strip()
        max_tres = parts[3].strip()
        max_jobs = parts[4].strip()
        max_wall = parts[5].strip()

        if acct:
            accounts.add(acct)
        if part:
            accessible_partitions.add(part)

        associations.append({
            "account": acct,
            "partition": part if part else "(all allowed)",
            "qos": qos,
            "max_tres": max_tres if max_tres else "no limit",
            "max_jobs": max_jobs if max_jobs else "no limit",
            "max_wall": max_wall if max_wall else "no limit",
        })

    return {
        "accounts": sorted(accounts),
        "accessible_partitions": sorted(accessible_partitions) if accessible_partitions else [
            "(determined by account \u2014 use sinfo to see available)"
        ],
        "associations": associations,
    }


def parse_squeue_jobs(raw: str) -> dict:
    """Parse squeue output into structured job list with summary counts.

    Args:
        raw: Raw stdout from squeue -o '%i|%P|%j|%T|%M|%l|%C|%m|%b'.

    Returns:
        Dict with 'jobs' list, 'running_count', and 'pending_count'.
    """
    jobs = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 9:
            jobs.append({
                "job_id": parts[0].strip(),
                "partition": parts[1].strip(),
                "name": parts[2].strip(),
                "state": parts[3].strip(),
                "elapsed": parts[4].strip(),
                "time_limit": parts[5].strip(),
                "cpus": parts[6].strip(),
                "memory": parts[7].strip(),
                "gres": parts[8].strip(),
            })

    # Pre-aggregate counts before returning
    running_count = sum(1 for j in jobs if j["state"] == "RUNNING")
    pending_count = sum(1 for j in jobs if j["state"] == "PENDING")

    # Return a sample of jobs to avoid context bloat on busy clusters.
    # The full list is computed but only a representative sample is returned.
    # Aggregated counts (running_count, pending_count, total_count) always
    # reflect the COMPLETE dataset — nothing is lost.
    MAX_JOBS_SAMPLE = 100

    return {
        "jobs": jobs[:MAX_JOBS_SAMPLE],
        "running_count": running_count,
        "pending_count": pending_count,
        "total_count": len(jobs),
        "jobs_truncated": len(jobs) > MAX_JOBS_SAMPLE,
    }


def parse_sshare(raw: str) -> list:
    """Parse sshare output into structured fairshare data.

    Args:
        raw: Raw stdout from sshare with pipe-delimited fields.

    Returns:
        List of fairshare dicts.
    """
    shares = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 7:
            shares.append({
                "account": parts[0].strip(),
                "user": parts[1].strip(),
                "raw_shares": parts[2].strip(),
                "norm_shares": parts[3].strip(),
                "raw_usage": parts[4].strip(),
                "effective_usage": parts[5].strip(),
                "fairshare": parts[6].strip(),
            })
    return shares


def parse_sacct_job_status(raw: str) -> dict:
    """Parse sacct output for a single job into structured status.

    Args:
        raw: Raw stdout from sacct -j <id> with pipe-delimited fields
             (State|Elapsed|JobName|Partition|AllocCPUS|MaxRSS|ExitCode|NodeList).

    Returns:
        Dict with job status fields, 'finished' boolean, and 'success' boolean.
    """
    if not raw or not raw.strip():
        return {
            "status": "UNKNOWN",
            "elapsed": "N/A",
            "finished": False,
            "success": False,
        }

    first_line = raw.strip().splitlines()[0]
    fields = first_line.split("|")

    state = fields[0].strip() if fields else "UNKNOWN"
    elapsed = fields[1].strip() if len(fields) > 1 else "N/A"
    job_name = fields[2].strip() if len(fields) > 2 else ""
    partition = fields[3].strip() if len(fields) > 3 else ""
    alloc_cpus = fields[4].strip() if len(fields) > 4 else ""
    max_rss = fields[5].strip() if len(fields) > 5 else ""
    exit_code = fields[6].strip() if len(fields) > 6 else ""
    nodelist = fields[7].strip() if len(fields) > 7 else ""

    finished_states = {
        "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
        "OUT_OF_MEMORY", "NODE_FAIL",
    }

    return {
        "status": state,
        "elapsed": elapsed,
        "finished": state in finished_states,
        "success": state == "COMPLETED",
        "job_name": job_name,
        "partition": partition,
        "alloc_cpus": alloc_cpus,
        "max_rss": max_rss,
        "exit_code": exit_code,
        "nodelist": nodelist,
    }


def build_cluster_summary(partitions: list) -> dict:
    """Build a cluster-wide summary from parsed partition data.

    Args:
        partitions: List of partition dicts from parse_sinfo_partitions.

    Returns:
        Dict with gpu_summary and cluster_summary.
    """
    # GPU summary across all partitions
    gpu_summary = {}
    for p in partitions:
        for gt in p.get("gpu_types", []):
            if gt not in gpu_summary:
                gpu_summary[gt] = {"partitions": [], "total_nodes": 0,
                                   "note": "total_nodes may include duplicates across partitions sharing nodes"}
            gpu_summary[gt]["partitions"].append(p["name"])
            gpu_summary[gt]["total_nodes"] += p["nodes_total"]

    gpu_partitions = [p["name"] for p in partitions if p["has_gpu"]]

    cluster_summary = {
        "note": "Node/CPU counts may be duplicated across partitions that share nodes. "
                "Use get_cluster_utilization for deduplicated totals.",
        "partition_count": len(partitions),
        "gpu_partition_count": len(gpu_partitions),
        "gpu_types_available": sorted(gpu_summary.keys()),
    }

    return {
        "gpu_summary": gpu_summary,
        "cluster_summary": cluster_summary,
    }


# ---------------------------------------------------------------------------
# NEW: Lightweight job state summary (for embedding in other tools)
# ---------------------------------------------------------------------------

def parse_job_state_summary(raw: str) -> dict:
    """Parse lightweight squeue state-only output into job count summary.

    Designed for a single squeue call: squeue -a --noheader -o '%T'
    Returns just the counts — no job details, minimal tokens.

    Args:
        raw: Raw stdout from squeue -a --noheader -o '%T'.
             Each line is a single job state like RUNNING, PENDING, etc.

    Returns:
        Dict with running, pending, completing, other, and total counts.
    """
    if not raw or not raw.strip():
        return {
            "running": 0,
            "pending": 0,
            "completing": 0,
            "other": 0,
            "total": 0,
        }

    counts = Counter()
    for line in raw.strip().splitlines():
        state = line.strip()
        if state:
            counts[state] += 1

    running = counts.get("RUNNING", 0)
    pending = counts.get("PENDING", 0)
    completing = counts.get("COMPLETING", 0)
    known = running + pending + completing
    total = sum(counts.values())

    return {
        "running": running,
        "pending": pending,
        "completing": completing,
        "other": total - known,
        "total": total,
    }


# ---------------------------------------------------------------------------
# NEW: Comprehensive cluster jobs parser (one squeue call, all analysis)
# ---------------------------------------------------------------------------

def parse_cluster_jobs(raw: str, top_n: int = 20) -> dict:
    """Parse full squeue -a output into comprehensive cluster job analysis.

    Designed for ONE squeue call with a rich format string:
        squeue -a --noheader -o '%i|%u|%P|%j|%T|%M|%l|%C|%m|%b|%r'

    All grouping, counting, top-N, zombie detection, GPU totals, and
    memory aggregation are computed in Python from this single dataset.
    No shell piping needed.

    Args:
        raw: Raw stdout from squeue with pipe-delimited fields:
             job_id|user|partition|name|state|elapsed|timelimit|cpus|mem|gres|reason
        top_n: Number of top users to include in rankings (default: 20).

    Returns:
        Dict with:
        - total_running, total_pending, total_jobs
        - total_cpus_allocated, total_gpus_allocated, total_memory_allocated_gb
        - top_users_running: [{user, count, cpus, gpus, memory_gb}, ...]
        - top_users_pending: [{user, count}, ...]
        - by_partition: [{partition, running, pending, total}, ...]
        - pending_reasons: {reason: {count, interpretation}, ...}
        - zombie_jobs: [{job_id, user, reason, elapsed}, ...]
    """
    if not raw or not raw.strip():
        return {
            "total_running": 0,
            "total_pending": 0,
            "total_jobs": 0,
            "total_cpus_allocated": 0,
            "total_gpus_allocated": 0,
            "total_memory_allocated_gb": 0.0,
            "top_users_running": [],
            "top_users_pending": [],
            "by_partition": [],
            "pending_reasons": {},
            "zombie_jobs": [],
        }

    # Parse all jobs
    jobs = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 11:
            continue
        jobs.append({
            "job_id": parts[0].strip(),
            "user": parts[1].strip(),
            "partition": parts[2].strip(),
            "name": parts[3].strip(),
            "state": parts[4].strip(),
            "elapsed": parts[5].strip(),
            "timelimit": parts[6].strip(),
            "cpus": parts[7].strip(),
            "mem": parts[8].strip(),
            "gres": parts[9].strip(),
            "reason": parts[10].strip(),
        })

    # --- Counts ---
    running_jobs = [j for j in jobs if j["state"] == "RUNNING"]
    pending_jobs = [j for j in jobs if j["state"] == "PENDING"]

    # --- CPU/GPU/Memory totals (running jobs only) ---
    total_cpus = 0
    total_gpus = 0
    total_memory_gb = 0.0
    for j in running_jobs:
        try:
            total_cpus += int(j["cpus"])
        except (ValueError, TypeError):
            pass
        total_gpus += _extract_gpu_count(j["gres"])
        total_memory_gb += _parse_mem_to_gb(j["mem"])

    # --- Top users by running jobs (with memory) ---
    user_running = Counter()
    user_running_cpus = Counter()
    user_running_gpus = Counter()
    user_running_mem = Counter()  # float counter for memory in GB
    for j in running_jobs:
        user_running[j["user"]] += 1
        try:
            user_running_cpus[j["user"]] += int(j["cpus"])
        except (ValueError, TypeError):
            pass
        user_running_gpus[j["user"]] += _extract_gpu_count(j["gres"])
        user_running_mem[j["user"]] += _parse_mem_to_gb(j["mem"])

    top_users_running = []
    for user, count in user_running.most_common(top_n):
        top_users_running.append({
            "user": user,
            "count": count,
            "cpus": user_running_cpus[user],
            "gpus": user_running_gpus[user],
            "memory_gb": round(user_running_mem[user], 1),
        })

    # --- Top users by pending jobs ---
    user_pending = Counter()
    for j in pending_jobs:
        user_pending[j["user"]] += 1

    top_users_pending = []
    for user, count in user_pending.most_common(top_n):
        top_users_pending.append({"user": user, "count": count})

    # --- By partition ---
    part_running = Counter()
    part_pending = Counter()
    part_total = Counter()
    for j in jobs:
        part_total[j["partition"]] += 1
        if j["state"] == "RUNNING":
            part_running[j["partition"]] += 1
        elif j["state"] == "PENDING":
            part_pending[j["partition"]] += 1

    by_partition = []
    for part in sorted(part_total.keys()):
        by_partition.append({
            "partition": part,
            "running": part_running[part],
            "pending": part_pending[part],
            "total": part_total[part],
        })

    # --- Pending reasons with interpretations ---
    reason_counts = Counter()
    for j in pending_jobs:
        reason_counts[j["reason"]] += 1

    pending_reasons = {}
    for reason, count in reason_counts.most_common():
        entry = {"count": count}
        if reason in PENDING_REASON_INTERPRETATIONS:
            entry["interpretation"] = PENDING_REASON_INTERPRETATIONS[reason]
        pending_reasons[reason] = entry

    # --- Zombie detection ---
    zombie_reasons = {
        "DependencyNeverSatisfied",
        "JobHeldAdmin",
        "JobHeldUser",
    }
    zombie_jobs = []
    for j in pending_jobs:
        if j["reason"] in zombie_reasons:
            zombie_jobs.append({
                "job_id": j["job_id"],
                "user": j["user"],
                "reason": j["reason"],
                "elapsed": j["elapsed"],
                "name": j["name"],
            })

    # Cap zombie_jobs to a representative sample to avoid context bloat.
    # zombie_jobs_total always reflects the true count.
    MAX_ZOMBIE_SAMPLE = 50

    return {
        "total_running": len(running_jobs),
        "total_pending": len(pending_jobs),
        "total_jobs": len(jobs),
        "total_cpus_allocated": total_cpus,
        "total_gpus_allocated": total_gpus,
        "total_memory_allocated_gb": round(total_memory_gb, 1),
        "top_users_running": top_users_running,
        "top_users_pending": top_users_pending,
        "by_partition": by_partition,
        "pending_reasons": pending_reasons,
        "zombie_jobs": zombie_jobs[:MAX_ZOMBIE_SAMPLE],
        "zombie_jobs_total": len(zombie_jobs),
    }


# ---------------------------------------------------------------------------
# GPU count extraction — handles ALL real-world Slurm GRES formats
# ---------------------------------------------------------------------------

# Regex to find a GPU count in various GRES formats.
# Matches the LAST integer before an optional (IDX:...) suffix in a gpu segment.
#
# Formats handled:
#   sinfo  Gres:     gpu:a100:4          gpu:4          gpu:h100:8
#   sinfo  GresUsed: gpu:a100:4(IDX:0-3) gpu:4(IDX:0-3) gpu:(null)
#   squeue %b:       gres/gpu:1          gres/gpu:h100:8 gres/gpu:a100:2
#   equals variant:  gpu=a100:2
_GPU_COUNT_RE = re.compile(
    r'(?:gres/)?gpu[=:]'           # prefix: "gpu:", "gpu=", or "gres/gpu:"
    r'(?:[a-zA-Z][a-zA-Z0-9_]*:)?' # optional type name like "a100:" or "h100:"
    r'(\d+)'                        # the count we want to capture
)


def _extract_gpu_count(gres: str) -> int:
    """Extract total GPU count from a GRES string.

    Handles ALL real-world Slurm GRES formats observed on production clusters:

    From sinfo (Gres field):
        gpu:a100:4          → 4
        gpu:4               → 4
        gpu:h100:8          → 8

    From sinfo (GresUsed field — has IDX suffix):
        gpu:a100:4(IDX:0-3) → 4
        gpu:h100:8(IDX:0-7) → 8
        gpu:l40s:2(IDX:0,2) → 2
        gpu:4(IDX:0-3)      → 4
        gpu:(null)           → 0

    From squeue (%b / GRES field — has gres/ prefix):
        gres/gpu:1           → 1
        gres/gpu:h100:8      → 8
        gres/gpu:a100:2      → 2
        gres/gpu:a40:1       → 1

    Other:
        gpu=a100:2           → 2  (equals separator variant)
        gpu:a100:2,gpu:l40s:1 → 3 (comma-separated multi-GPU)
        (null)               → 0
        N/A                  → 0
        ""                   → 0

    Args:
        gres: GRES string from sinfo or squeue output.

    Returns:
        Integer GPU count (0 if no GPUs or unparseable).
    """
    if not gres:
        return 0

    # Quick reject for known empty values
    stripped = gres.strip()
    if stripped in ("(null)", "N/A", ""):
        return 0

    # Handle gpu:(null) — sinfo GresUsed when no GPUs allocated
    if "gpu:(null)" in stripped or "gpu=(null)" in stripped:
        return 0

    total = 0
    # Split on comma for multi-GRES (e.g. "gpu:a100:2,gpu:l40s:1")
    for part in stripped.split(","):
        part = part.strip()
        match = _GPU_COUNT_RE.search(part)
        if match:
            total += int(match.group(1))
    return total


# ---------------------------------------------------------------------------
# NEW: Node efficiency parser (joins sinfo -N with squeue)
# ---------------------------------------------------------------------------

def parse_node_efficiency(sinfo_raw: str, squeue_raw: str,
                          waste_threshold: float = 50.0) -> dict:
    """Compute cluster resource efficiency by joining node and job data.

    Designed for two calls:
        sinfo -N --noheader -o '%N|%P|%T|%C|%m|%G'
        squeue -a --noheader -t RUNNING -o '%N|%C|%b'

    Returns a compact summary + only the high-waste outlier nodes.
    Does NOT return all 200+ nodes — the LLM only needs the summary
    and the problems.

    Args:
        sinfo_raw: Raw stdout from sinfo -N with pipe-delimited fields:
                   node|partition|state|cpus(A/I/O/T)|memory|gres
        squeue_raw: Raw stdout from squeue with pipe-delimited fields:
                    nodelist|cpus|gres
        waste_threshold: Percentage of idle CPUs on a GPU-allocated node
                        above which it's flagged as "high waste" (default: 50%).

    Returns:
        Dict with:
        - cluster_totals: {total_cpus, allocated_cpus, idle_cpus,
                          total_gpus, allocated_gpus, idle_gpus,
                          cpu_utilization_pct, gpu_utilization_pct}
        - high_waste_nodes: [{node, partition, cpus_total, cpus_idle,
                            gpus_total, gpus_allocated, idle_cpu_pct}, ...]
        - waste_summary: human-readable string
    """
    # --- Parse sinfo nodes ---
    # Include ALL nodes regardless of state for accurate totals.
    # Tag each node as usable or unavailable.
    usable_states = {"idle", "mixed", "alloc", "allocated", "completing"}
    nodes = {}  # node_name -> {partition, state, cpus_total, cpus_alloc, ...}
    for line in (sinfo_raw or "").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        node_name = parts[0].strip()
        partition = parts[1].strip()
        raw_state = parts[2].strip()
        state = raw_state.lower().rstrip("*-~#$@!")
        cpus_info = parts[3].strip()
        mem = parts[4].strip()
        gres = parts[5].strip()

        # Parse CPU A/I/O/T
        cpu_alloc, cpu_idle, cpu_total = 0, 0, 0
        cpu_parts = cpus_info.split("/")
        if len(cpu_parts) == 4:
            try:
                cpu_alloc = int(cpu_parts[0])
                cpu_idle = int(cpu_parts[1])
                cpu_total = int(cpu_parts[3])
            except ValueError:
                pass

        # Parse GPU count from GRES
        gpu_total = _extract_gpu_count(gres)
        is_usable = state in usable_states

        # Keep first occurrence per node (sinfo -N may list a node
        # multiple times for different partitions)
        if node_name not in nodes:
            nodes[node_name] = {
                "partition": partition,
                "state": state,
                "is_usable": is_usable,
                "cpus_total": cpu_total,
                "cpus_allocated": cpu_alloc,
                "cpus_idle": cpu_idle,
                "gpus_total": gpu_total,
                "gpus_allocated": 0,  # filled from squeue
            }

    # --- Parse squeue to get actual GPU allocations per node ---
    node_gpu_alloc = Counter()
    for line in (squeue_raw or "").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        nodelist_raw = parts[0].strip()
        gres = parts[2].strip()
        gpu_count = _extract_gpu_count(gres)
        if gpu_count > 0 and nodelist_raw:
            # Expand compressed nodelists like node[01-04]
            for node_name in expand_nodelist(nodelist_raw):
                node_name = node_name.strip()
                if node_name:
                    node_gpu_alloc[node_name] += gpu_count

    # Apply GPU allocations to nodes
    for node_name, gpu_count in node_gpu_alloc.items():
        if node_name in nodes:
            nodes[node_name]["gpus_allocated"] = gpu_count

    # --- Compute cluster totals ---
    # Count ALL nodes for totals (including drained/down)
    total_cpus = sum(n["cpus_total"] for n in nodes.values())
    total_gpus = sum(n["gpus_total"] for n in nodes.values())

    # Only count allocations from usable nodes
    alloc_cpus = sum(n["cpus_allocated"] for n in nodes.values() if n["is_usable"])
    idle_cpus = sum(n["cpus_idle"] for n in nodes.values() if n["is_usable"])
    alloc_gpus = sum(n["gpus_allocated"] for n in nodes.values() if n["is_usable"])

    # Track unavailable resources separately
    unavail_nodes = sum(1 for n in nodes.values() if not n["is_usable"])
    unavail_cpus = sum(n["cpus_total"] for n in nodes.values() if not n["is_usable"])
    unavail_gpus = sum(n["gpus_total"] for n in nodes.values() if not n["is_usable"])

    idle_gpus = total_gpus - alloc_gpus - unavail_gpus

    cpu_util = (alloc_cpus / total_cpus * 100) if total_cpus > 0 else 0.0
    gpu_util = (alloc_gpus / total_gpus * 100) if total_gpus > 0 else 0.0

    cluster_totals = {
        "total_cpus": total_cpus,
        "allocated_cpus": alloc_cpus,
        "idle_cpus": idle_cpus,
        "total_gpus": total_gpus,
        "allocated_gpus": alloc_gpus,
        "idle_gpus": max(0, idle_gpus),
        "cpu_utilization_pct": round(cpu_util, 1),
        "gpu_utilization_pct": round(gpu_util, 1),
        "total_nodes": len(nodes),
        "unavailable_nodes": unavail_nodes,
        "unavailable_cpus": unavail_cpus,
        "unavailable_gpus": unavail_gpus,
    }

    # --- Find high-waste nodes ---
    # A node is "high waste" if it has GPUs allocated but >threshold% idle CPUs
    # Only check usable nodes (drained nodes aren't wasting resources)
    high_waste = []
    for node_name, n in nodes.items():
        if not n["is_usable"]:
            continue
        if n["gpus_allocated"] > 0 and n["cpus_total"] > 0:
            idle_pct = (n["cpus_idle"] / n["cpus_total"]) * 100
            if idle_pct >= waste_threshold:
                high_waste.append({
                    "node": node_name,
                    "partition": n["partition"],
                    "cpus_total": n["cpus_total"],
                    "cpus_idle": n["cpus_idle"],
                    "gpus_total": n["gpus_total"],
                    "gpus_allocated": n["gpus_allocated"],
                    "idle_cpu_pct": round(idle_pct, 1),
                })

    # Sort by waste percentage descending
    high_waste.sort(key=lambda x: x["idle_cpu_pct"], reverse=True)

    waste_summary = (
        f"{len(high_waste)} GPU nodes have >{waste_threshold}% idle CPUs "
        f"while GPUs are allocated"
    )

    return {
        "cluster_totals": cluster_totals,
        "high_waste_nodes": high_waste,
        "waste_summary": waste_summary,
    }


# ---------------------------------------------------------------------------
# GPU type extraction — companion to _extract_gpu_count
# ---------------------------------------------------------------------------

# Regex to extract GPU type name from GRES strings.
# Captures the type name (e.g. "a100", "h100", "l40s", "nvidia_h200_nvl")
# from formats like:
#   gpu:a100:4          → "a100"
#   gres/gpu:h100:8     → "h100"
#   gres/gpu:a100:2     → "a100"
#   gpu=a100:2          → "a100"
#   gpu:4               → None (no type specified)
#   gres/gpu:1           → None (no type specified)
_GPU_TYPE_RE = re.compile(
    r'(?:gres/)?gpu[=:]'              # prefix: "gpu:", "gpu=", or "gres/gpu:"
    r'([a-zA-Z][a-zA-Z0-9_]*)'       # the type name we want to capture
    r':\d+'                            # followed by :count
)


def _extract_gpu_type(gres: str) -> str:
    """Extract GPU type name from a GRES string.

    Returns the first GPU type found, or empty string if no type specified.

    Handles all real-world Slurm GRES formats:
        gpu:a100:4          → "a100"
        gpu:h100:8          → "h100"
        gres/gpu:a100:2     → "a100"
        gres/gpu:l40s:1     → "l40s"
        gpu=a100:2          → "a100"
        gpu:a100:4(IDX:0-3) → "a100"
        gpu:4               → ""  (no type)
        gres/gpu:1          → ""  (no type)
        (null)              → ""
        ""                  → ""

    Args:
        gres: GRES string from sinfo or squeue output.

    Returns:
        GPU type name as lowercase string, or "" if no type specified.
    """
    if not gres:
        return ""

    stripped = gres.strip()
    if stripped in ("(null)", "N/A", ""):
        return ""

    if "gpu:(null)" in stripped or "gpu=(null)" in stripped:
        return ""

    # Search for type name in each comma-separated segment
    for part in stripped.split(","):
        part = part.strip()
        match = _GPU_TYPE_RE.search(part)
        if match:
            return match.group(1).lower()

    return ""


# ---------------------------------------------------------------------------
# NEW: Parse per-node available resources for fitment analysis
# ---------------------------------------------------------------------------

def parse_node_available_resources(sinfo_raw: str, squeue_raw: str) -> list:
    """Parse per-node available resources for job fitment analysis.

    Designed for two calls:
        sinfo -N --noheader -o '%N|%P|%T|%C|%m|%e|%G'
        squeue -a --noheader -t RUNNING -o '%N|%C|%b'

    Returns a list of nodes with their AVAILABLE (free) resources,
    suitable for checking whether a job request can fit on any node.

    Only includes nodes in usable states (idle, mixed, allocated).
    Excludes drained, down, reserved, etc.

    Args:
        sinfo_raw: Raw stdout from sinfo -N with pipe-delimited fields:
                   node|partition|state|cpus(A/I/O/T)|total_mem|free_mem|gres
        squeue_raw: Raw stdout from squeue with pipe-delimited fields:
                    nodelist|cpus|gres  (for GPU allocation tracking)

    Returns:
        List of node dicts, each with:
        - node: node name
        - partition: partition name
        - state: node state
        - cpus_total: total CPUs on node
        - cpus_free: idle CPUs available
        - memory_total_gb: total memory in GB
        - memory_free_gb: free memory in GB
        - gpus_total: total GPUs on node
        - gpus_free: available GPUs
        - gpu_type: GPU type name (e.g. "a100") or "" if no GPUs
    """
    # --- Parse sinfo nodes ---
    nodes = {}  # node_name -> dict
    usable_states = {"idle", "mixed", "alloc", "allocated", "completing"}

    for line in (sinfo_raw or "").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue

        node_name = parts[0].strip()
        partition = parts[1].strip()
        state = parts[2].strip().lower().rstrip("*-~#$@!")  # remove trailing Slurm state suffixes
        cpus_info = parts[3].strip()
        total_mem_mb = parts[4].strip()
        free_mem_mb = parts[5].strip()
        gres = parts[6].strip()

        # Only include usable nodes
        if state not in usable_states:
            continue

        # Parse CPU A/I/O/T
        cpu_idle, cpu_total = 0, 0
        cpu_parts = cpus_info.split("/")
        if len(cpu_parts) == 4:
            try:
                cpu_idle = int(cpu_parts[1])
                cpu_total = int(cpu_parts[3])
            except ValueError:
                pass

        # Parse memory
        try:
            mem_total_gb = round(int(total_mem_mb) / 1024, 1)
        except (ValueError, TypeError):
            mem_total_gb = 0.0
        try:
            mem_free_gb = round(int(free_mem_mb) / 1024, 1)
        except (ValueError, TypeError):
            mem_free_gb = 0.0

        # Parse GPU info
        gpu_total = _extract_gpu_count(gres)
        gpu_type = _extract_gpu_type(gres)

        # Keep first occurrence per node (sinfo -N may list a node
        # multiple times for different partitions)
        if node_name not in nodes:
            nodes[node_name] = {
                "node": node_name,
                "partition": partition,
                "state": state,
                "cpus_total": cpu_total,
                "cpus_free": cpu_idle,
                "memory_total_gb": mem_total_gb,
                "memory_free_gb": mem_free_gb,
                "gpus_total": gpu_total,
                "gpus_allocated": 0,  # filled from squeue
                "gpu_type": gpu_type,
            }

    # --- Parse squeue to get actual GPU allocations per node ---
    node_gpu_alloc = Counter()
    for line in (squeue_raw or "").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        nodelist_raw = parts[0].strip()
        gres = parts[2].strip()
        gpu_count = _extract_gpu_count(gres)
        if gpu_count > 0 and nodelist_raw:
            # Expand compressed nodelists like node[01-04]
            for nn in expand_nodelist(nodelist_raw):
                nn = nn.strip()
                if nn:
                    node_gpu_alloc[nn] += gpu_count

    # Apply GPU allocations and compute free GPUs
    result = []
    for node_name, n in nodes.items():
        gpus_allocated = node_gpu_alloc.get(node_name, 0)
        gpus_free = max(0, n["gpus_total"] - gpus_allocated)
        result.append({
            "node": n["node"],
            "partition": n["partition"],
            "state": n["state"],
            "cpus_total": n["cpus_total"],
            "cpus_free": n["cpus_free"],
            "memory_total_gb": n["memory_total_gb"],
            "memory_free_gb": n["memory_free_gb"],
            "gpus_total": n["gpus_total"],
            "gpus_free": gpus_free,
            "gpu_type": n["gpu_type"],
        })

    return result


# ---------------------------------------------------------------------------
# NEW: Parse pending job resource demands
# ---------------------------------------------------------------------------

def parse_pending_job_demands(raw: str) -> dict:
    """Parse pending jobs from squeue into structured resource demands.

    Designed for ONE squeue call:
        squeue -a --noheader --state=PD -o '%i|%u|%P|%j|%C|%m|%b|%l|%r'

    Extracts per-job resource requests (CPUs, memory, GPUs, GPU type)
    and aggregates totals for the entire pending queue.

    Args:
        raw: Raw stdout from squeue with pipe-delimited fields:
             job_id|user|partition|name|cpus|mem|gres|timelimit|reason

    Returns:
        Dict with:
        - total_pending: total pending job count
        - total_cpus_requested: sum of CPUs requested by all pending jobs
        - total_gpus_requested: sum of GPUs requested by all pending jobs
        - total_memory_requested_gb: sum of memory requested
        - gpu_type_demand: {gpu_type: {count: N, gpus: N}} — demand by GPU type
        - pending_jobs: list of job dicts with resource details
        - by_reason: {reason: {count, interpretation, cpus, gpus, memory_gb}}
        - demand_summary: human-readable summary string
    """
    if not raw or not raw.strip():
        return {
            "total_pending": 0,
            "total_cpus_requested": 0,
            "total_gpus_requested": 0,
            "total_memory_requested_gb": 0.0,
            "gpu_type_demand": {},
            "pending_jobs": [],
            "by_reason": {},
            "demand_summary": "No pending jobs.",
        }

    jobs = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue

        gres_str = parts[6].strip()
        mem_str = parts[5].strip()

        try:
            cpus = int(parts[4].strip())
        except (ValueError, TypeError):
            cpus = 0

        gpu_count = _extract_gpu_count(gres_str)
        gpu_type = _extract_gpu_type(gres_str)
        memory_gb = _parse_mem_to_gb(mem_str)

        jobs.append({
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

    # --- Aggregate totals ---
    total_cpus = sum(j["cpus"] for j in jobs)
    total_gpus = sum(j["gpus"] for j in jobs)
    total_memory = sum(j["memory_gb"] for j in jobs)

    # --- GPU type demand ---
    gpu_type_demand = {}
    for j in jobs:
        if j["gpus"] > 0:
            gt = j["gpu_type"] if j["gpu_type"] else "(any)"
            if gt not in gpu_type_demand:
                gpu_type_demand[gt] = {"count": 0, "gpus": 0}
            gpu_type_demand[gt]["count"] += 1
            gpu_type_demand[gt]["gpus"] += j["gpus"]

    # --- By reason (with resource aggregation) ---
    reason_data = {}  # reason -> {count, cpus, gpus, memory_gb}
    for j in jobs:
        r = j["reason"]
        if r not in reason_data:
            reason_data[r] = {"count": 0, "cpus": 0, "gpus": 0, "memory_gb": 0.0}
        reason_data[r]["count"] += 1
        reason_data[r]["cpus"] += j["cpus"]
        reason_data[r]["gpus"] += j["gpus"]
        reason_data[r]["memory_gb"] += j["memory_gb"]

    by_reason = {}
    for reason, data in sorted(reason_data.items(),
                                key=lambda x: x[1]["count"], reverse=True):
        entry = {
            "count": data["count"],
            "cpus": data["cpus"],
            "gpus": data["gpus"],
            "memory_gb": round(data["memory_gb"], 1),
        }
        if reason in PENDING_REASON_INTERPRETATIONS:
            entry["interpretation"] = PENDING_REASON_INTERPRETATIONS[reason]
        by_reason[reason] = entry

    # --- Summary ---
    gpu_demand_parts = []
    for gt, d in sorted(gpu_type_demand.items()):
        gpu_demand_parts.append(f"{d['gpus']}x {gt}")
    gpu_demand_str = ", ".join(gpu_demand_parts) if gpu_demand_parts else "none"

    demand_summary = (
        f"{len(jobs)} pending jobs requesting "
        f"{total_cpus} CPUs, {round(total_memory, 1)} GB memory, "
        f"GPUs: {gpu_demand_str}"
    )

    # --- Top resource requests (heaviest pending jobs) ---
    # Sort by GPUs descending, then memory descending, then CPUs descending.
    # This surfaces the most resource-hungry jobs that are most likely
    # to be blocked — the LLM can reason about these without seeing all jobs.
    top_resource_requests = sorted(
        jobs,
        key=lambda j: (j["gpus"], j["memory_gb"], j["cpus"]),
        reverse=True,
    )[:10]

    # Return a representative sample instead of the full list.
    # Aggregated totals (total_pending, by_reason, gpu_type_demand) always
    # reflect the COMPLETE dataset — nothing is lost.
    MAX_PENDING_SAMPLE = 25

    return {
        "total_pending": len(jobs),
        "total_cpus_requested": total_cpus,
        "total_gpus_requested": total_gpus,
        "total_memory_requested_gb": round(total_memory, 1),
        "gpu_type_demand": gpu_type_demand,
        "pending_jobs": jobs[:MAX_PENDING_SAMPLE],
        "pending_jobs_total": len(jobs),
        "pending_jobs_truncated": len(jobs) > MAX_PENDING_SAMPLE,
        "top_resource_requests": top_resource_requests,
        "by_reason": by_reason,
        "demand_summary": demand_summary,
    }


# ---------------------------------------------------------------------------
# NEW: Job fitment analysis — can a job request fit on any available node?
# ---------------------------------------------------------------------------

def check_job_fitment(cpus: int, memory_gb: float, gpus: int,
                      gpu_type: str, partition: str,
                      node_resources: list,
                      partition_timelimits: dict = None,
                      job_timelimit: str = "") -> dict:
    """Check if a job request can fit on any available node.

    Pure-logic function — takes pre-parsed node resources and a job request,
    returns whether the job can fit and why/why not.

    Args:
        cpus: Number of CPUs requested.
        memory_gb: Memory requested in GB.
        gpus: Number of GPUs requested.
        gpu_type: GPU type requested (e.g. "a100"). Empty = any type.
        partition: Target partition name. Empty = any partition.
        node_resources: List of node dicts from parse_node_available_resources.
        partition_timelimits: Optional dict of {partition_name: timelimit_str}
                            for time limit validation.
        job_timelimit: Optional job time limit string for validation.

    Returns:
        Dict with:
        - can_fit: bool — whether at least one node can satisfy the request
        - fitting_nodes: list of node names that can fit the job
        - fitting_count: number of nodes that can fit
        - bottleneck: string describing what's blocking (if can_fit is False)
        - analysis: detailed breakdown of why nodes don't fit
        - max_available: {cpus, memory_gb, gpus, gpu_types} — best available
        - suggestions: list of actionable suggestions
    """
    if not node_resources:
        return {
            "can_fit": False,
            "fitting_nodes": [],
            "fitting_count": 0,
            "bottleneck": "No node resource data available",
            "analysis": {},
            "max_available": {},
            "suggestions": ["Call query_slurm_cluster with include_node_details=True first"],
        }

    # Filter by partition if specified
    candidates = node_resources
    if partition:
        candidates = [n for n in candidates if n["partition"] == partition]
        if not candidates:
            return {
                "can_fit": False,
                "fitting_nodes": [],
                "fitting_count": 0,
                "bottleneck": f"No usable nodes in partition '{partition}'",
                "analysis": {"partition_filter": f"0 nodes in partition '{partition}'"},
                "max_available": {},
                "suggestions": [
                    f"Check if partition '{partition}' exists and has available nodes",
                    "Try without specifying a partition",
                ],
            }

    # --- Check each node ---
    fitting = []
    fail_reasons = Counter()  # reason -> count
    max_cpus_free = 0
    max_mem_free = 0.0
    max_gpus_free = 0
    available_gpu_types = set()

    for n in candidates:
        max_cpus_free = max(max_cpus_free, n["cpus_free"])
        max_mem_free = max(max_mem_free, n["memory_free_gb"])
        max_gpus_free = max(max_gpus_free, n["gpus_free"])
        if n["gpu_type"]:
            available_gpu_types.add(n["gpu_type"])

        # Check CPU fit
        if n["cpus_free"] < cpus:
            fail_reasons["insufficient_cpus"] += 1
            continue

        # Check memory fit
        if n["memory_free_gb"] < memory_gb:
            fail_reasons["insufficient_memory"] += 1
            continue

        # Check GPU fit
        if gpus > 0:
            if n["gpus_free"] < gpus:
                fail_reasons["insufficient_gpus"] += 1
                continue

            # Check GPU type match
            if gpu_type and n["gpu_type"] and n["gpu_type"] != gpu_type.lower():
                fail_reasons["wrong_gpu_type"] += 1
                continue

            # Node has no GPUs at all
            if n["gpus_total"] == 0:
                fail_reasons["no_gpus_on_node"] += 1
                continue

        # All checks passed — this node can fit the job
        fitting.append(n["node"])

    # --- Build bottleneck analysis ---
    bottleneck = ""
    if not fitting:
        # Determine the primary bottleneck
        if gpus > 0 and fail_reasons.get("wrong_gpu_type", 0) > 0:
            bottleneck = (
                f"No nodes with free {gpu_type} GPUs. "
                f"Available GPU types: {sorted(available_gpu_types) if available_gpu_types else 'none'}"
            )
        elif gpus > 0 and fail_reasons.get("insufficient_gpus", 0) > 0:
            bottleneck = (
                f"No node has {gpus} free GPUs (max available: {max_gpus_free})"
            )
        elif gpus > 0 and fail_reasons.get("no_gpus_on_node", 0) > 0:
            bottleneck = "No GPU nodes available in the target partition"
        elif fail_reasons.get("insufficient_memory", 0) > 0:
            bottleneck = (
                f"No node has {memory_gb} GB free memory "
                f"(max available: {max_mem_free} GB)"
            )
        elif fail_reasons.get("insufficient_cpus", 0) > 0:
            bottleneck = (
                f"No node has {cpus} free CPUs (max available: {max_cpus_free})"
            )
        else:
            bottleneck = "No nodes match the combined resource requirements"

    # --- Suggestions ---
    suggestions = []
    if not fitting:
        if gpus > 0 and gpu_type and fail_reasons.get("wrong_gpu_type", 0) > 0:
            other_types = sorted(available_gpu_types - {gpu_type.lower()})
            if other_types:
                suggestions.append(
                    f"Try a different GPU type: {other_types}"
                )
        if memory_gb > max_mem_free and max_mem_free > 0:
            suggestions.append(
                f"Reduce memory request from {memory_gb}G to <={max_mem_free}G"
            )
        if cpus > max_cpus_free and max_cpus_free > 0:
            suggestions.append(
                f"Reduce CPU request from {cpus} to <={max_cpus_free}"
            )
        if gpus > max_gpus_free and max_gpus_free > 0:
            suggestions.append(
                f"Reduce GPU request from {gpus} to <={max_gpus_free}"
            )
        if not suggestions:
            suggestions.append(
                "Wait for running jobs to complete and free resources"
            )

    # --- Time limit check ---
    time_warning = ""
    if partition_timelimits and job_timelimit and partition:
        part_limit = partition_timelimits.get(partition, "")
        if part_limit and part_limit != "infinite":
            time_warning = (
                f"Partition '{partition}' has max time limit: {part_limit}. "
                f"Ensure your job time ({job_timelimit}) does not exceed this."
            )
            if time_warning:
                suggestions.append(time_warning)

    return {
        "can_fit": len(fitting) > 0,
        "fitting_nodes": fitting[:20],  # cap at 20 to avoid token bloat
        "fitting_count": len(fitting),
        "bottleneck": bottleneck,
        "analysis": {
            "nodes_checked": len(candidates),
            "fail_reasons": dict(fail_reasons),
        },
        "max_available": {
            "cpus": max_cpus_free,
            "memory_gb": max_mem_free,
            "gpus": max_gpus_free,
            "gpu_types": sorted(available_gpu_types) if available_gpu_types else [],
        },
        "suggestions": suggestions,
    }



# ---------------------------------------------------------------------------
# NEW: Multi-partition node resource parser for bulk fitment
# ---------------------------------------------------------------------------

def parse_node_resources_multi_partition(sinfo_raw: str, squeue_raw: str) -> list:
    """Parse per-node available resources, tracking ALL partitions per node.

    Unlike parse_node_available_resources (which keeps only the first partition),
    this parser records every partition a node belongs to. This is critical for
    bulk fitment analysis where a pending job targets a specific partition and
    we need to know which nodes are accessible in that partition.

    Designed for two calls:
        sinfo -N --noheader -o '%N|%P|%T|%C|%m|%e|%G'
        squeue -a --noheader -t RUNNING -o '%N|%C|%b'

    Args:
        sinfo_raw: Raw stdout from sinfo -N with pipe-delimited fields:
                   node|partition|state|cpus(A/I/O/T)|total_mem|free_mem|gres
        squeue_raw: Raw stdout from squeue with pipe-delimited fields:
                    nodelist|cpus|gres  (for GPU allocation tracking)

    Returns:
        List of node dicts, each with:
        - node: node name
        - partitions: list of partition names this node belongs to
        - state: node state
        - cpus_total, cpus_free, memory_total_gb, memory_free_gb
        - gpus_total, gpus_free, gpu_type
    """
    # --- Parse sinfo nodes, collecting ALL partitions per node ---
    # Include ALL nodes regardless of state for accurate totals.
    # Tag each node as usable or unavailable. For unavailable nodes,
    # set free resources to 0 (can't schedule on them).
    nodes = {}  # node_name -> dict
    usable_states = {"idle", "mixed", "alloc", "allocated", "completing"}

    for line in (sinfo_raw or "").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue

        node_name = parts[0].strip()
        partition = parts[1].strip()
        state = parts[2].strip().lower().rstrip("*-~#$@!")
        cpus_info = parts[3].strip()
        total_mem_mb = parts[4].strip()
        free_mem_mb = parts[5].strip()
        gres = parts[6].strip()

        is_usable = state in usable_states

        if node_name in nodes:
            # Node already seen — just add the partition
            if partition not in nodes[node_name]["partitions"]:
                nodes[node_name]["partitions"].append(partition)
            continue

        # Parse CPU A/I/O/T
        cpu_idle, cpu_total = 0, 0
        cpu_parts = cpus_info.split("/")
        if len(cpu_parts) == 4:
            try:
                cpu_idle = int(cpu_parts[1])
                cpu_total = int(cpu_parts[3])
            except ValueError:
                pass

        # Parse memory
        try:
            mem_total_gb = round(int(total_mem_mb) / 1024, 1)
        except (ValueError, TypeError):
            mem_total_gb = 0.0
        try:
            mem_free_gb = round(int(free_mem_mb) / 1024, 1)
        except (ValueError, TypeError):
            mem_free_gb = 0.0

        # Parse GPU info
        gpu_total = _extract_gpu_count(gres)
        gpu_type = _extract_gpu_type(gres)

        # For unavailable nodes, set free resources to 0
        nodes[node_name] = {
            "node": node_name,
            "partitions": [partition],
            "state": state,
            "is_usable": is_usable,
            "cpus_total": cpu_total,
            "cpus_free": cpu_idle if is_usable else 0,
            "memory_total_gb": mem_total_gb,
            "memory_free_gb": mem_free_gb if is_usable else 0.0,
            "gpus_total": gpu_total,
            "gpus_allocated": 0,
            "gpu_type": gpu_type,
        }

    # --- Parse squeue to get actual GPU allocations per node ---
    node_gpu_alloc = Counter()
    for line in (squeue_raw or "").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        nodelist_raw = parts[0].strip()
        gres = parts[2].strip()
        gpu_count = _extract_gpu_count(gres)
        if gpu_count > 0 and nodelist_raw:
            # Expand compressed nodelists like node[01-04]
            for nn in expand_nodelist(nodelist_raw):
                nn = nn.strip()
                if nn:
                    node_gpu_alloc[nn] += gpu_count

    # Apply GPU allocations and compute free GPUs
    result = []
    for node_name, n in nodes.items():
        gpus_allocated = node_gpu_alloc.get(node_name, 0)
        # For unavailable nodes, gpus_free is always 0
        if n["is_usable"]:
            gpus_free = max(0, n["gpus_total"] - gpus_allocated)
        else:
            gpus_free = 0
        result.append({
            "node": n["node"],
            "partitions": n["partitions"],
            "state": n["state"],
            "is_usable": n["is_usable"],
            "cpus_total": n["cpus_total"],
            "cpus_free": n["cpus_free"],
            "memory_total_gb": n["memory_total_gb"],
            "memory_free_gb": n["memory_free_gb"],
            "gpus_total": n["gpus_total"],
            "gpus_free": gpus_free,
            "gpu_type": n["gpu_type"],
        })

    return result


# ---------------------------------------------------------------------------
# NEW: Aggregate partition availability from node data
# ---------------------------------------------------------------------------

def aggregate_partition_availability(node_resources: list) -> dict:
    """Aggregate free resources per partition from multi-partition node data.

    Takes the output of parse_node_resources_multi_partition and produces
    a per-partition summary of available resources. This answers the question
    "what's free in each partition right now?" without dumping node-level data.

    Args:
        node_resources: List of node dicts from parse_node_resources_multi_partition.
                       Each node has a 'partitions' list (not a single string).

    Returns:
        Dict keyed by partition name, each with:
        - total_nodes: nodes in this partition
        - nodes_with_free_cpus: nodes with at least 1 free CPU
        - nodes_with_free_gpus: nodes with at least 1 free GPU
        - free_cpus: total free CPUs across all nodes in partition
        - free_gpus: total free GPUs across all nodes in partition
        - free_memory_gb: total free memory across all nodes in partition
        - gpu_types: dict of {type: {free: N, total: N}} for each GPU type
    """
    partitions = {}

    for n in node_resources:
        for part in n.get("partitions", []):
            if part not in partitions:
                partitions[part] = {
                    "total_nodes": 0,
                    "nodes_with_free_cpus": 0,
                    "nodes_with_free_gpus": 0,
                    "free_cpus": 0,
                    "free_gpus": 0,
                    "free_memory_gb": 0.0,
                    "gpu_types": {},
                }
            p = partitions[part]
            p["total_nodes"] += 1
            p["free_cpus"] += n["cpus_free"]
            p["free_gpus"] += n["gpus_free"]
            p["free_memory_gb"] += n["memory_free_gb"]

            if n["cpus_free"] > 0:
                p["nodes_with_free_cpus"] += 1
            if n["gpus_free"] > 0:
                p["nodes_with_free_gpus"] += 1

            # Track GPU types
            if n["gpus_total"] > 0:
                gt = n["gpu_type"] if n["gpu_type"] else "(unknown)"
                if gt not in p["gpu_types"]:
                    p["gpu_types"][gt] = {"free": 0, "total": 0}
                p["gpu_types"][gt]["free"] += n["gpus_free"]
                p["gpu_types"][gt]["total"] += n["gpus_total"]

    # Round memory values
    for p in partitions.values():
        p["free_memory_gb"] = round(p["free_memory_gb"], 1)

    return partitions


# ---------------------------------------------------------------------------
# NEW: Bulk fitment — find which pending jobs could run right now
# ---------------------------------------------------------------------------

def find_runnable_jobs(pending_jobs: list, node_resources: list,
                       partition_timelimits: dict = None,
                       max_runnable: int = 50) -> dict:
    """Check ALL pending jobs against ALL available nodes to find runnable jobs.

    This is the bulk version of check_job_fitment. Instead of checking one job,
    it checks every pending job against every available node, considering:
    - CPUs requested vs free
    - Memory requested vs free
    - GPUs requested vs free
    - GPU type requested vs available
    - Partition requested vs node membership
    - Time limit requested vs partition max time

    IMPORTANT SEMANTICS — INDEPENDENT CHECK, NOT SCHEDULING SIMULATION:
    Each job is checked INDEPENDENTLY against the CURRENT snapshot of free
    resources. "Runnable now" means "at least one node has enough free
    resources for THIS specific job right now." It does NOT mean all
    runnable jobs can start simultaneously — if 500 jobs each need 1 GPU
    and only 1 GPU is free, all 500 are marked runnable because each one
    individually fits. For concurrent capacity estimation, use
    estimate_concurrent_capacity() which simulates greedy placement.

    Jobs with dependency or held reasons are SKIPPED from resource fitment
    entirely — they will never run regardless of available resources, so
    checking nodes for them is wasted work and produces misleading results.
    They are counted separately in the summary as blocked_by_dependency
    and blocked_by_held.

    Jobs that have resources available but are still pending are flagged as
    "likely blocked by priority/fairshare" — a very useful diagnostic.

    Args:
        pending_jobs: List of pending job dicts from parse_pending_job_demands.
                     Each must have: job_id, user, partition, cpus, memory_gb,
                     gpus, gpu_type, timelimit, reason.
        node_resources: List of node dicts from parse_node_resources_multi_partition.
                       Each must have: node, partitions (list), cpus_free,
                       memory_free_gb, gpus_free, gpu_type.
        partition_timelimits: Optional dict of {partition_name: timelimit_str}.
                             Used to check if a job's requested time exceeds
                             the partition's maximum allowed time.
        max_runnable: Maximum runnable jobs to return in detail (default: 50).

    Returns:
        Dict with:
        - summary: {total_checked, runnable_now, blocked_by_resources,
                   blocked_by_partition, blocked_by_gpu_type,
                   blocked_by_timelimit, blocked_by_dependency,
                   blocked_by_held, blocked_other}
        - runnable_jobs: list of jobs that COULD run (resources exist) with
                        fitting_nodes and likely_blocked_by
        - blocked_summary: {reason: count} — why non-runnable jobs are blocked
        - partition_bottlenecks: per-partition breakdown of blocked jobs
        - NOTE: "runnable_now" means each job individually has fitting
                resources. For how many can run simultaneously, see
                estimate_concurrent_capacity().
    """
    if not pending_jobs:
        return {
            "summary": {
                "total_checked": 0, "runnable_now": 0,
                "blocked_by_resources": 0, "blocked_by_partition": 0,
                "blocked_by_gpu_type": 0, "blocked_other": 0,
            },
            "runnable_jobs": [],
            "blocked_summary": {},
            "partition_bottlenecks": {},
        }

    if not node_resources:
        return {
            "summary": {
                "total_checked": len(pending_jobs), "runnable_now": 0,
                "blocked_by_resources": len(pending_jobs),
                "blocked_by_partition": 0, "blocked_by_gpu_type": 0,
                "blocked_other": 0,
            },
            "runnable_jobs": [],
            "blocked_summary": {"no_node_data": len(pending_jobs)},
            "partition_bottlenecks": {},
        }

    # Pre-index nodes by partition for fast lookup
    nodes_by_partition = {}  # partition -> [node_dicts]
    all_nodes = node_resources  # for jobs with no partition specified
    for n in node_resources:
        for part in n.get("partitions", []):
            if part not in nodes_by_partition:
                nodes_by_partition[part] = []
            nodes_by_partition[part].append(n)

    runnable = []
    blocked_reasons = Counter()
    partition_bottlenecks = Counter()

    for job in pending_jobs:
        j_cpus = job.get("cpus", 0)
        j_mem = job.get("memory_gb", 0.0)
        j_gpus = job.get("gpus", 0)
        j_gpu_type = job.get("gpu_type", "").lower()
        j_partition = job.get("partition", "")
        j_timelimit = job.get("timelimit", "")
        j_reason = job.get("reason", "")

        # --- Early exit: skip jobs that will NEVER run regardless of resources ---
        # Dependency jobs are waiting for another job, not for a node.
        # Zombie/held jobs will never run until manually released.
        # Checking nodes for these is wasted work and produces misleading results.
        _SKIP_REASONS = {
            "Dependency", "DependencyNeverSatisfied",
            "JobHeldAdmin", "JobHeldUser",
        }
        if j_reason in _SKIP_REASONS:
            blocked_reasons[j_reason] += 1
            continue

        # --- Time limit check ---
        # If the job requests more time than the partition allows, it will
        # never run in that partition regardless of available resources.
        if partition_timelimits and j_partition and j_timelimit:
            partition_max_str = partition_timelimits.get(j_partition, "")
            if partition_max_str:
                job_minutes = _parse_timelimit_to_minutes(j_timelimit)
                partition_max_minutes = _parse_timelimit_to_minutes(partition_max_str)
                if job_minutes > partition_max_minutes:
                    blocked_reasons["exceeds_partition_timelimit"] += 1
                    if j_partition:
                        partition_bottlenecks[j_partition] += 1
                    continue

        # Determine candidate nodes based on partition
        if j_partition:
            candidates = nodes_by_partition.get(j_partition, [])
            if not candidates:
                blocked_reasons["no_nodes_in_partition"] += 1
                partition_bottlenecks[j_partition] += 1
                continue
        else:
            candidates = all_nodes

        # Check each candidate node
        fitting_nodes = []
        fail_reason = ""

        for n in candidates:
            # CPU check
            if n["cpus_free"] < j_cpus:
                if not fail_reason:
                    fail_reason = "insufficient_cpus"
                continue

            # Memory check
            if n["memory_free_gb"] < j_mem:
                if not fail_reason:
                    fail_reason = "insufficient_memory"
                continue

            # GPU checks
            if j_gpus > 0:
                if n["gpus_total"] == 0:
                    if not fail_reason:
                        fail_reason = "no_gpus_on_node"
                    continue
                if n["gpus_free"] < j_gpus:
                    if not fail_reason:
                        fail_reason = "insufficient_gpus"
                    continue
                if j_gpu_type and n["gpu_type"] and n["gpu_type"] != j_gpu_type:
                    if not fail_reason:
                        fail_reason = "wrong_gpu_type"
                    continue

            # All checks passed
            fitting_nodes.append(n["node"])

        if fitting_nodes:
            # Resources exist but job is still pending → likely priority/fairshare
            likely_blocked_by = "priority_or_fairshare"
            if j_reason in ("Priority", "Resources"):
                likely_blocked_by = "priority_or_fairshare"
            elif j_reason == "Dependency":
                likely_blocked_by = "dependency"
            elif j_reason in ("QOSMaxJobsPerUserLimit", "QOSMaxGRESPerUser",
                              "AssocMaxJobsLimit"):
                likely_blocked_by = "qos_limits"
            elif j_reason in ("JobHeldAdmin", "JobHeldUser"):
                likely_blocked_by = "held"
            elif j_reason:
                likely_blocked_by = j_reason.lower()

            runnable.append({
                "job_id": job.get("job_id", ""),
                "user": job.get("user", ""),
                "partition": j_partition,
                "requested": {
                    "cpus": j_cpus,
                    "memory_gb": j_mem,
                    "gpus": j_gpus,
                    "gpu_type": j_gpu_type if j_gpu_type else "(any)",
                },
                "fitting_node_count": len(fitting_nodes),
                "fitting_nodes_sample": fitting_nodes[:5],
                "likely_blocked_by": likely_blocked_by,
                "pending_reason": j_reason,
            })
        else:
            # Job cannot fit — record why
            if fail_reason:
                blocked_reasons[fail_reason] += 1
            else:
                blocked_reasons["combined_requirements"] += 1
            if j_partition:
                partition_bottlenecks[j_partition] += 1

    # Build summary
    n_runnable = len(runnable)
    n_blocked_resources = sum(v for k, v in blocked_reasons.items()
                              if k in ("insufficient_cpus", "insufficient_memory",
                                       "insufficient_gpus", "combined_requirements"))
    n_blocked_partition = blocked_reasons.get("no_nodes_in_partition", 0)
    n_blocked_gpu_type = (blocked_reasons.get("wrong_gpu_type", 0) +
                          blocked_reasons.get("no_gpus_on_node", 0))
    n_blocked_timelimit = blocked_reasons.get("exceeds_partition_timelimit", 0)
    n_blocked_dependency = (blocked_reasons.get("Dependency", 0) +
                            blocked_reasons.get("DependencyNeverSatisfied", 0))
    n_blocked_held = (blocked_reasons.get("JobHeldAdmin", 0) +
                      blocked_reasons.get("JobHeldUser", 0))
    n_accounted = (n_runnable + n_blocked_resources + n_blocked_partition +
                   n_blocked_gpu_type + n_blocked_timelimit +
                   n_blocked_dependency + n_blocked_held)
    n_blocked_other = max(0, len(pending_jobs) - n_accounted)

    summary = {
        "total_checked": len(pending_jobs),
        "runnable_now": n_runnable,
        "blocked_by_resources": n_blocked_resources,
        "blocked_by_partition": n_blocked_partition,
        "blocked_by_gpu_type": n_blocked_gpu_type,
        "blocked_by_timelimit": n_blocked_timelimit,
        "blocked_by_dependency": n_blocked_dependency,
        "blocked_by_held": n_blocked_held,
        "blocked_other": max(0, n_blocked_other),
    }

    return {
        "summary": summary,
        "runnable_jobs": runnable[:max_runnable],
        "runnable_jobs_total": n_runnable,
        "blocked_summary": dict(blocked_reasons),
        "partition_bottlenecks": dict(partition_bottlenecks),
    }





# ---------------------------------------------------------------------------
# NEW: Concurrent capacity estimate — how many jobs can run simultaneously
# ---------------------------------------------------------------------------

def estimate_concurrent_capacity(pending_jobs: list, node_resources: list,
                                  partition_timelimits: dict = None) -> dict:
    """Estimate how many pending jobs could run SIMULTANEOUSLY.

    Unlike find_runnable_jobs (which checks each job independently against
    the current resource snapshot), this function simulates greedy placement:
    it "places" jobs one by one onto nodes, subtracting resources as it goes.
    This answers the question "how many of these pending jobs could actually
    start at the same time right now?"

    The simulation is greedy (first-fit) — it does NOT find the optimal
    packing (which is NP-hard bin packing). Real Slurm scheduling uses
    backfill and priority, so actual placement may differ. But this gives
    a realistic lower bound on concurrent capacity.

    Jobs are sorted by resource intensity (GPUs desc, then memory desc,
    then CPUs desc) to improve packing — placing large jobs first avoids
    fragmentation.

    Args:
        pending_jobs: List of pending job dicts (same format as find_runnable_jobs).
        node_resources: List of node dicts from parse_node_resources_multi_partition.
        partition_timelimits: Optional dict of {partition_name: timelimit_str}.

    Returns:
        Dict with:
        - can_run_simultaneously: number of jobs that fit concurrently
        - total_pending_checked: total jobs considered
        - placed_jobs: list of placed job summaries (job_id, node, partition)
        - remaining_capacity: per-partition summary of leftover resources
          after all placements
        - skipped_dependency: count of dependency/held jobs skipped
    """
    if not pending_jobs or not node_resources:
        return {
            "can_run_simultaneously": 0,
            "total_pending_checked": len(pending_jobs) if pending_jobs else 0,
            "placed_jobs": [],
            "remaining_capacity": {},
            "skipped_dependency": 0,
        }

    # Deep copy node resources so we can subtract without mutating originals
    sim_nodes = copy.deepcopy(node_resources)

    # Pre-index nodes by partition
    nodes_by_partition = {}
    for n in sim_nodes:
        for part in n.get("partitions", []):
            if part not in nodes_by_partition:
                nodes_by_partition[part] = []
            nodes_by_partition[part].append(n)

    _SKIP_REASONS = {
        "Dependency", "DependencyNeverSatisfied",
        "JobHeldAdmin", "JobHeldUser",
    }

    # Filter out dependency/held jobs
    eligible_jobs = []
    skipped_dep = 0
    for job in pending_jobs:
        if job.get("reason", "") in _SKIP_REASONS:
            skipped_dep += 1
            continue
        # Time limit check
        if partition_timelimits and job.get("partition") and job.get("timelimit"):
            partition_max_str = partition_timelimits.get(job["partition"], "")
            if partition_max_str:
                job_minutes = _parse_timelimit_to_minutes(job["timelimit"])
                partition_max_minutes = _parse_timelimit_to_minutes(partition_max_str)
                if job_minutes > partition_max_minutes:
                    continue
        eligible_jobs.append(job)

    # Sort by resource intensity: GPUs desc, memory desc, CPUs desc
    # This improves packing by placing large jobs first
    eligible_jobs.sort(key=lambda j: (
        -(j.get("gpus", 0)),
        -(j.get("memory_gb", 0)),
        -(j.get("cpus", 0)),
    ))

    placed = []

    for job in eligible_jobs:
        j_cpus = job.get("cpus", 0)
        j_mem = job.get("memory_gb", 0.0)
        j_gpus = job.get("gpus", 0)
        j_gpu_type = job.get("gpu_type", "").lower()
        j_partition = job.get("partition", "")

        # Get candidate nodes
        if j_partition:
            candidates = nodes_by_partition.get(j_partition, [])
        else:
            candidates = sim_nodes

        # Try to place on first fitting node
        placed_on = None
        for n in candidates:
            if n["cpus_free"] < j_cpus:
                continue
            if n["memory_free_gb"] < j_mem:
                continue
            if j_gpus > 0:
                if n.get("gpus_total", 0) == 0:
                    continue
                if n["gpus_free"] < j_gpus:
                    continue
                if j_gpu_type and n.get("gpu_type", "") and n["gpu_type"] != j_gpu_type:
                    continue
            placed_on = n
            break

        if placed_on:
            # Subtract resources from this node
            placed_on["cpus_free"] -= j_cpus
            placed_on["memory_free_gb"] -= j_mem
            if j_gpus > 0:
                placed_on["gpus_free"] -= j_gpus

            placed.append({
                "job_id": job.get("job_id", ""),
                "user": job.get("user", ""),
                "partition": j_partition,
                "node": placed_on["node"],
                "requested": {
                    "cpus": j_cpus,
                    "memory_gb": j_mem,
                    "gpus": j_gpus,
                    "gpu_type": j_gpu_type if j_gpu_type else "(any)",
                },
            })

    # Compute remaining capacity per partition after all placements
    remaining = {}
    for n in sim_nodes:
        for part in n.get("partitions", []):
            if part not in remaining:
                remaining[part] = {"free_cpus": 0, "free_memory_gb": 0.0, "free_gpus": 0}
            remaining[part]["free_cpus"] += max(0, n["cpus_free"])
            remaining[part]["free_memory_gb"] += max(0.0, n["memory_free_gb"])
            remaining[part]["free_gpus"] += max(0, n["gpus_free"])

    for p in remaining.values():
        p["free_memory_gb"] = round(p["free_memory_gb"], 1)

    return {
        "can_run_simultaneously": len(placed),
        "total_pending_checked": len(pending_jobs),
        "placed_jobs": placed[:50],  # cap detail output
        "placed_jobs_total": len(placed),
        "remaining_capacity": remaining,
        "skipped_dependency": skipped_dep,
    }

# ---------------------------------------------------------------------------
# NEW: Per-partition cluster utilization — answers "how busy is the cluster?"
# ---------------------------------------------------------------------------

def validate_cluster_data(cluster_wide: dict) -> dict:
    """Sanity-check cluster_wide utilization data for impossible/suspicious values.

    Returns a dict with:
      - 'valid': bool — False if any hard impossibility is detected
      - 'warnings': list of str — human-readable warning messages
      - 'suspicious': list of str — values that are unusual but not impossible
    """
    warnings = []
    suspicious = []

    total_cpus = cluster_wide.get("total_cpus", 0)
    used_cpus = cluster_wide.get("used_cpus", 0)
    total_mem = cluster_wide.get("total_memory_gb", 0.0)
    used_mem = cluster_wide.get("used_memory_gb", 0.0)
    total_gpus = cluster_wide.get("total_gpus", 0)
    used_gpus = cluster_wide.get("used_gpus", 0)
    total_nodes = cluster_wide.get("total_nodes", 0)
    alloc_nodes = cluster_wide.get("allocated_nodes", 0)

    # Hard impossibilities — these indicate a parsing bug
    if used_cpus > total_cpus and total_cpus > 0:
        warnings.append(
            f"IMPOSSIBLE: used_cpus ({used_cpus}) > total_cpus ({total_cpus})"
        )
    if used_mem > total_mem and total_mem > 0:
        warnings.append(
            f"IMPOSSIBLE: used_memory_gb ({used_mem:.1f}) > total_memory_gb ({total_mem:.1f})"
        )
    if used_gpus > total_gpus and total_gpus > 0:
        warnings.append(
            f"IMPOSSIBLE: used_gpus ({used_gpus}) > total_gpus ({total_gpus})"
        )
    if alloc_nodes > total_nodes and total_nodes > 0:
        warnings.append(
            f"IMPOSSIBLE: allocated_nodes ({alloc_nodes}) > total_nodes ({total_nodes})"
        )

    # Suspicious but possible values
    cpu_pct = cluster_wide.get("cpu_util_pct", 0.0)
    mem_pct = cluster_wide.get("memory_util_pct", 0.0)
    gpu_pct = cluster_wide.get("gpu_util_pct", 0.0)

    if cpu_pct > 100.0:
        warnings.append(f"IMPOSSIBLE: cpu_util_pct ({cpu_pct}) > 100%")
    if mem_pct > 100.0:
        warnings.append(f"IMPOSSIBLE: memory_util_pct ({mem_pct}) > 100%")
    if gpu_pct > 100.0:
        warnings.append(f"IMPOSSIBLE: gpu_util_pct ({gpu_pct}) > 100%")

    # Recompute expected percentages and check for large discrepancies
    if total_cpus > 0:
        expected_cpu_pct = round((used_cpus / total_cpus) * 100, 1)
        if abs(expected_cpu_pct - cpu_pct) > 1.0:
            suspicious.append(
                f"cpu_util_pct mismatch: stored={cpu_pct}% but "
                f"used/total={used_cpus}/{total_cpus} implies {expected_cpu_pct}%"
            )
    if total_mem > 0:
        expected_mem_pct = round((used_mem / total_mem) * 100, 1)
        if abs(expected_mem_pct - mem_pct) > 1.0:
            suspicious.append(
                f"memory_util_pct mismatch: stored={mem_pct}% but "
                f"used/total={used_mem:.1f}/{total_mem:.1f} implies {expected_mem_pct}%"
            )
    if total_gpus > 0:
        expected_gpu_pct = round((used_gpus / total_gpus) * 100, 1)
        if abs(expected_gpu_pct - gpu_pct) > 1.0:
            suspicious.append(
                f"gpu_util_pct mismatch: stored={gpu_pct}% but "
                f"used/total={used_gpus}/{total_gpus} implies {expected_gpu_pct}%"
            )

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "suspicious": suspicious,
    }


def compute_cluster_utilization(node_resources: list) -> dict:
    """Compute per-partition and cluster-wide utilization for CPUs, memory, and GPUs.

    Takes the output of parse_node_resources_multi_partition and produces
    a comprehensive utilization report. This answers the question
    "how busy is the cluster?" with per-partition breakdowns.

    Unlike query_node_efficiency (which only gives cluster-wide CPU/GPU %),
    this gives per-partition utilization for ALL three resource types:
    CPUs, memory, and GPUs — including GPU type breakdown.

    Args:
        node_resources: List of node dicts from parse_node_resources_multi_partition.
                       Each node has: partitions (list), cpus_total, cpus_free,
                       memory_total_gb, memory_free_gb, gpus_total, gpus_free, gpu_type.

    Returns:
        Dict with:
        - cluster_wide: {cpu_util_pct, memory_util_pct, gpu_util_pct,
                        total_cpus, used_cpus, total_memory_gb, used_memory_gb,
                        total_gpus, used_gpus}
        - per_partition: {partition_name: {cpu_util_pct, memory_util_pct,
                         gpu_util_pct, total_nodes, gpu_types: {type: {total, used, util_pct}}, ...}}
        - busiest_partitions: top 5 partitions by overall utilization
        - bottleneck_analysis: which resource type is the bottleneck per partition
    """
    if not node_resources:
        empty_cluster = {
            "cpu_util_pct": 0.0, "memory_util_pct": 0.0, "gpu_util_pct": 0.0,
            "total_cpus": 0, "used_cpus": 0,
            "total_memory_gb": 0.0, "used_memory_gb": 0.0,
            "total_gpus": 0, "used_gpus": 0,
        }
        return {
            "cluster_wide": empty_cluster,
            "per_partition": {},
            "busiest_partitions": [],
            "bottleneck_analysis": {},
            "sanity_check": validate_cluster_data(empty_cluster),
        }

    # --- Per-partition aggregation ---
    partitions = {}
    # Track unique nodes for cluster-wide totals (avoid double-counting
    # nodes that appear in multiple partitions)
    seen_nodes = set()
    cluster_cpus_total = 0
    cluster_cpus_free = 0
    cluster_mem_total = 0.0
    cluster_mem_free = 0.0
    cluster_gpus_total = 0
    cluster_gpus_free = 0
    unavail_nodes = 0
    unavail_cpus = 0
    unavail_mem = 0.0
    unavail_gpus = 0

    for n in node_resources:
        node_name = n.get("node", "")
        is_usable = n.get("is_usable", True)  # default True for backward compat

        # Cluster-wide: only count each physical node once
        if node_name not in seen_nodes:
            seen_nodes.add(node_name)
            cluster_cpus_total += n.get("cpus_total", 0)
            cluster_cpus_free += n.get("cpus_free", 0)
            cluster_mem_total += n.get("memory_total_gb", 0.0)
            cluster_mem_free += n.get("memory_free_gb", 0.0)
            cluster_gpus_total += n.get("gpus_total", 0)
            cluster_gpus_free += n.get("gpus_free", 0)
            if not is_usable:
                unavail_nodes += 1
                unavail_cpus += n.get("cpus_total", 0)
                unavail_mem += n.get("memory_total_gb", 0.0)
                unavail_gpus += n.get("gpus_total", 0)

        # Per-partition: count node in each partition it belongs to
        for part in n.get("partitions", []):
            if part not in partitions:
                partitions[part] = {
                    "total_nodes": 0,
                    "cpus_total": 0, "cpus_free": 0,
                    "memory_total_gb": 0.0, "memory_free_gb": 0.0,
                    "gpus_total": 0, "gpus_free": 0,
                    "gpu_types": {},
                }
            p = partitions[part]
            p["total_nodes"] += 1
            p["cpus_total"] += n.get("cpus_total", 0)
            p["cpus_free"] += n.get("cpus_free", 0)
            p["memory_total_gb"] += n.get("memory_total_gb", 0.0)
            p["memory_free_gb"] += n.get("memory_free_gb", 0.0)
            p["gpus_total"] += n.get("gpus_total", 0)
            p["gpus_free"] += n.get("gpus_free", 0)

            # GPU type breakdown
            if n.get("gpus_total", 0) > 0:
                gt = n.get("gpu_type", "") or "(unknown)"
                if gt not in p["gpu_types"]:
                    p["gpu_types"][gt] = {"total": 0, "used": 0}
                p["gpu_types"][gt]["total"] += n.get("gpus_total", 0)
                gpus_used = n.get("gpus_total", 0) - n.get("gpus_free", 0)
                p["gpu_types"][gt]["used"] += max(0, gpus_used)

    # --- Compute utilization percentages ---
    def _pct(used, total):
        return round((used / total) * 100, 1) if total > 0 else 0.0

    cluster_cpus_used = cluster_cpus_total - cluster_cpus_free
    cluster_mem_used = cluster_mem_total - cluster_mem_free
    cluster_gpus_used = cluster_gpus_total - cluster_gpus_free

    cluster_wide = {
        "cpu_util_pct": _pct(cluster_cpus_used, cluster_cpus_total),
        "memory_util_pct": _pct(cluster_mem_used, cluster_mem_total),
        "gpu_util_pct": _pct(cluster_gpus_used, cluster_gpus_total),
        "total_cpus": cluster_cpus_total,
        "used_cpus": cluster_cpus_used,
        "total_memory_gb": round(cluster_mem_total, 1),
        "used_memory_gb": round(cluster_mem_used, 1),
        "total_gpus": cluster_gpus_total,
        "used_gpus": max(0, cluster_gpus_used),
        "total_nodes": len(seen_nodes),
        "unavailable_nodes": unavail_nodes,
        "unavailable_cpus": unavail_cpus,
        "unavailable_memory_gb": round(unavail_mem, 1),
        "unavailable_gpus": unavail_gpus,
    }

    per_partition = {}
    bottleneck_analysis = {}

    for pname, p in partitions.items():
        cpus_used = p["cpus_total"] - p["cpus_free"]
        mem_used = p["memory_total_gb"] - p["memory_free_gb"]
        gpus_used = p["gpus_total"] - p["gpus_free"]

        cpu_pct = _pct(cpus_used, p["cpus_total"])
        mem_pct = _pct(mem_used, p["memory_total_gb"])
        gpu_pct = _pct(gpus_used, p["gpus_total"])

        # GPU type utilization
        gpu_type_util = {}
        for gt, gdata in p["gpu_types"].items():
            gpu_type_util[gt] = {
                "total": gdata["total"],
                "used": gdata["used"],
                "util_pct": _pct(gdata["used"], gdata["total"]),
            }

        per_partition[pname] = {
            "total_nodes": p["total_nodes"],
            "cpu_util_pct": cpu_pct,
            "memory_util_pct": mem_pct,
            "gpu_util_pct": gpu_pct,
            "total_cpus": p["cpus_total"],
            "used_cpus": cpus_used,
            "total_memory_gb": round(p["memory_total_gb"], 1),
            "used_memory_gb": round(mem_used, 1),
            "total_gpus": p["gpus_total"],
            "used_gpus": max(0, gpus_used),
            "gpu_types": gpu_type_util,
        }

        # Determine bottleneck for this partition
        utils = {"cpu": cpu_pct, "memory": mem_pct}
        if p["gpus_total"] > 0:
            utils["gpu"] = gpu_pct
        if utils:
            bottleneck = max(utils, key=utils.get)
            bottleneck_analysis[pname] = {
                "bottleneck": bottleneck,
                "bottleneck_util_pct": utils[bottleneck],
            }

    # --- Busiest partitions (by average of available resource utilizations) ---
    partition_scores = []
    for pname, pdata in per_partition.items():
        scores = [pdata["cpu_util_pct"], pdata["memory_util_pct"]]
        if pdata["total_gpus"] > 0:
            scores.append(pdata["gpu_util_pct"])
        avg_util = sum(scores) / len(scores) if scores else 0.0
        partition_scores.append({"partition": pname, "avg_util_pct": round(avg_util, 1)})

    partition_scores.sort(key=lambda x: x["avg_util_pct"], reverse=True)

    return {
        "cluster_wide": cluster_wide,
        "per_partition": per_partition,
        "busiest_partitions": partition_scores[:10],
        "bottleneck_analysis": bottleneck_analysis,
        "sanity_check": validate_cluster_data(cluster_wide),
    }


# ---------------------------------------------------------------------------
# Partition access determination (2-step verification)
# ---------------------------------------------------------------------------

def parse_scontrol_partition(scontrol_output: str) -> dict:
    """Parse the output of `scontrol show partition=<name>` into a dict.

    Extracts the fields most relevant to access control:
      - PartitionName
      - State
      - AllowAccounts  (comma-separated list, or "ALL")
      - DenyAccounts   (comma-separated list, or "")
      - AllowGroups    (comma-separated list, or "ALL")
      - DenyGroups     (comma-separated list, or "")
      - AllowQos       (comma-separated list, or "ALL")
      - DenyQos        (comma-separated list, or "")

    Args:
        scontrol_output: Raw stdout from `scontrol show partition=<name>`.

    Returns:
        dict with keys: partition_name, state, allow_accounts, deny_accounts,
        allow_groups, deny_groups, allow_qos, deny_qos.
        Lists are empty when the field is "ALL" or absent (meaning no restriction).
    """
    result = {
        "partition_name": "",
        "state": "",
        "allow_accounts": [],   # empty = ALL allowed
        "deny_accounts": [],    # empty = none denied
        "allow_groups": [],     # empty = ALL allowed
        "deny_groups": [],      # empty = none denied
        "allow_qos": [],        # empty = ALL allowed
        "deny_qos": [],         # empty = none denied
        "raw": scontrol_output,
    }

    def _extract(text: str, key: str) -> str:
        """Return the value for key=value in scontrol output, or ''."""
        import re as _re
        m = _re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", text)
        return m.group(1) if m else ""

    import re as _re

    def _to_list(val: str) -> list:
        """Convert 'ALL', '', or 'a,b,c' to a list. ALL/empty → []."""
        if not val or val.upper() == "ALL" or val == "(null)":
            return []
        return [v.strip() for v in val.split(",") if v.strip()]

    result["partition_name"] = _extract(scontrol_output, "PartitionName")
    result["state"] = _extract(scontrol_output, "State")
    result["allow_accounts"] = _to_list(_extract(scontrol_output, "AllowAccounts"))
    result["deny_accounts"] = _to_list(_extract(scontrol_output, "DenyAccounts"))
    result["allow_groups"] = _to_list(_extract(scontrol_output, "AllowGroups"))
    result["deny_groups"] = _to_list(_extract(scontrol_output, "DenyGroups"))
    result["allow_qos"] = _to_list(_extract(scontrol_output, "AllowQos"))
    result["deny_qos"] = _to_list(_extract(scontrol_output, "DenyQos"))

    return result


def determine_partition_access(user_accounts: list, partition_config: dict) -> dict:
    """Determine whether a user can access a partition given their accounts
    and the parsed partition configuration from parse_scontrol_partition().

    Implements the authoritative 2-step access check:
      1. If AllowAccounts is non-empty, user must have at least one account in it.
      2. If DenyAccounts is non-empty, ALL of the user's accounts must NOT be in it
         (i.e. access is granted if at least one account is NOT denied).

    Args:
        user_accounts: List of Slurm account names the user belongs to.
        partition_config: Dict returned by parse_scontrol_partition().

    Returns:
        dict with keys:
          - can_access (bool): True if user can submit to this partition
          - granted_via (list[str]): accounts that grant access
          - denied_accounts (list[str]): user's accounts that are explicitly denied
          - reason (str): human-readable explanation
    """
    allow_accounts = partition_config.get("allow_accounts", [])
    deny_accounts = partition_config.get("deny_accounts", [])
    partition_name = partition_config.get("partition_name", "unknown")

    denied = [a for a in user_accounts if a in deny_accounts]
    allowed_by_allowlist = []

    if allow_accounts:
        # Explicit allowlist — user must have at least one account in it
        allowed_by_allowlist = [a for a in user_accounts if a in allow_accounts]
        # Also remove any that are in deny list
        allowed_by_allowlist = [a for a in allowed_by_allowlist if a not in deny_accounts]
    else:
        # No allowlist (AllowAccounts=ALL) — all accounts are candidates
        allowed_by_allowlist = [a for a in user_accounts if a not in deny_accounts]

    can_access = len(allowed_by_allowlist) > 0

    if not user_accounts:
        reason = f"User has no Slurm accounts — cannot access partition '{partition_name}'."
    elif can_access:
        via = ", ".join(allowed_by_allowlist)
        if denied:
            blocked = ", ".join(denied)
            reason = (
                f"Access GRANTED to '{partition_name}' via account(s): {via}. "
                f"Note: account(s) {blocked} are denied but user has other valid accounts."
            )
        else:
            reason = f"Access GRANTED to '{partition_name}' via account(s): {via}."
    else:
        if denied:
            blocked = ", ".join(denied)
            reason = (
                f"Access DENIED to '{partition_name}': all user account(s) ({blocked}) "
                f"are in the partition's DenyAccounts list."
            )
        elif allow_accounts:
            needed = ", ".join(allow_accounts)
            have = ", ".join(user_accounts)
            reason = (
                f"Access DENIED to '{partition_name}': partition only allows accounts "
                f"[{needed}] but user only has [{have}]."
            )
        else:
            reason = f"Access DENIED to '{partition_name}': unknown reason."

    return {
        "can_access": can_access,
        "granted_via": allowed_by_allowlist,
        "denied_accounts": denied,
        "reason": reason,
    }
