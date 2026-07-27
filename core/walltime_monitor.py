"""Walltime monitor — tracks SLURM job remaining time and fires warnings."""

import asyncio
import os
from typing import Callable, Coroutine, Optional, Set

WALLTIME_THRESHOLDS = [
    (45, "info", "45 minutes remaining"),
    (20, "warning", "20 minutes remaining — consider saving your work"),
    (10, "error", "⚠️ 10 minutes remaining — session ending soon!"),
    (5, "error", "🚨 5 MINUTES LEFT — save immediately!"),
]


def parse_slurm_time_to_minutes(time_str: str) -> Optional[float]:
    """Parse SLURM time format to minutes. Handles D-H:MM:SS, H:MM:SS, M:SS."""
    if not time_str:
        return None
    try:
        days = 0
        if "-" in time_str:
            day_part, time_str = time_str.split("-", 1)
            days = int(day_part)

        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        else:
            return None

        total_minutes = days * 24 * 60 + hours * 60 + minutes + seconds / 60.0
        return total_minutes
    except (ValueError, IndexError):
        return None


async def get_remaining_minutes(job_id: str) -> Optional[float]:
    """Query squeue for remaining walltime of a SLURM job."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "squeue", "-j", job_id, "-h", "-o", "%L",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        time_str = stdout.decode().strip()

        if not time_str or proc.returncode != 0:
            return None

        return parse_slurm_time_to_minutes(time_str)
    except Exception:
        return None


async def walltime_monitor_loop(
    send_toast_fn: Callable[[str, str], Coroutine],
    poll_interval: int = 60,
):
    """Background loop that monitors SLURM walltime and fires toast warnings.

    Args:
        send_toast_fn: async callable(message, toast_type) to display warnings
        poll_interval: seconds between checks
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("[WALLTIME] No SLURM_JOB_ID — walltime monitor disabled")
        return

    fired: Set[int] = set()
    await asyncio.sleep(30)  # let session fully initialize

    while True:
        remaining = await get_remaining_minutes(job_id)

        if remaining is not None:
            for threshold_min, toast_type, message in WALLTIME_THRESHOLDS:
                if remaining <= threshold_min and threshold_min not in fired:
                    fired.add(threshold_min)
                    await send_toast_fn(f"⏱️ {message}", toast_type)
                    print(f"[WALLTIME] {message} (job {job_id}, {remaining:.1f}min left)")
        else:
            print(f"[WALLTIME] Could not query remaining time for job {job_id}")

        await asyncio.sleep(poll_interval)
