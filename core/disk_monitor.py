"""Disk Space Monitor — alerts users before disk-full conditions disrupt work.

Checks home directory and work directory free space on startup and
periodically during the session. Thresholds:
  WARNING:  < 1 GB free
  CRITICAL: < 500 MB free (protocol recording auto-pauses)

No Chainlit dependencies — returns status dicts for the caller to display.
"""

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class DiskStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


WARN_THRESHOLD_BYTES = 1_073_741_824  # 1 GB
CRITICAL_THRESHOLD_BYTES = 524_288_000  # 500 MB


@dataclass
class DiskCheckResult:
    path: str
    label: str  # "home" or "work_dir"
    free_bytes: int
    total_bytes: int
    status: DiskStatus
    message: str

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)

    @property
    def free_mb(self) -> float:
        return self.free_bytes / (1024**2)

    @property
    def usage_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return ((self.total_bytes - self.free_bytes) / self.total_bytes) * 100


def check_disk_space(path: str, label: str = "") -> DiskCheckResult:
    """Check free space on the filesystem containing the given path."""
    resolved = str(Path(path).resolve())
    try:
        usage = shutil.disk_usage(resolved)
        free = usage.free
        total = usage.total
    except (OSError, FileNotFoundError):
        return DiskCheckResult(
            path=resolved,
            label=label or "unknown",
            free_bytes=0,
            total_bytes=0,
            status=DiskStatus.CRITICAL,
            message=f"Cannot check disk space for {resolved}",
        )

    if free < CRITICAL_THRESHOLD_BYTES:
        status = DiskStatus.CRITICAL
        message = f"{label}: only {free / (1024**2):.0f} MB free — CRITICAL! Operations may fail."
    elif free < WARN_THRESHOLD_BYTES:
        status = DiskStatus.WARNING
        message = f"{label}: {free / (1024**3):.2f} GB free — running low."
    else:
        status = DiskStatus.OK
        message = f"{label}: {free / (1024**3):.1f} GB free"

    return DiskCheckResult(
        path=resolved,
        label=label or "unknown",
        free_bytes=free,
        total_bytes=total,
        status=status,
        message=message,
    )


def run_startup_check(home_dir: str, work_dir: Optional[str] = None) -> List[DiskCheckResult]:
    """Run disk checks on startup. Returns results sorted worst-first."""
    results = []
    results.append(check_disk_space(home_dir, "Home directory"))

    if work_dir and os.path.isdir(work_dir):
        # Only check work_dir if it's on a different filesystem
        home_dev = _get_device(home_dir)
        work_dev = _get_device(work_dir)
        if work_dev != home_dev:
            results.append(check_disk_space(work_dir, "Work directory"))

    # Sort: critical first, then warning, then ok
    priority = {DiskStatus.CRITICAL: 0, DiskStatus.WARNING: 1, DiskStatus.OK: 2}
    results.sort(key=lambda r: priority[r.status])
    return results


def should_pause_recording(results: List[DiskCheckResult]) -> bool:
    """Return True if any filesystem is at CRITICAL level."""
    return any(r.status == DiskStatus.CRITICAL for r in results)


def get_worst_status(results: List[DiskCheckResult]) -> DiskStatus:
    """Return the worst status across all checked paths."""
    if not results:
        return DiskStatus.OK
    if any(r.status == DiskStatus.CRITICAL for r in results):
        return DiskStatus.CRITICAL
    if any(r.status == DiskStatus.WARNING for r in results):
        return DiskStatus.WARNING
    return DiskStatus.OK


def format_startup_toast(results: List[DiskCheckResult]) -> Optional[Dict[str, str]]:
    """Format results into a toast message. Returns None if all OK."""
    worst = get_worst_status(results)
    if worst == DiskStatus.OK:
        return None

    messages = [r.message for r in results if r.status != DiskStatus.OK]
    toast_type = "error" if worst == DiskStatus.CRITICAL else "warning"
    return {
        "message": " | ".join(messages),
        "type": toast_type,
    }


def _get_device(path: str) -> int:
    """Get device ID for a path (to detect same-filesystem mounts)."""
    try:
        return os.stat(path).st_dev
    except (OSError, FileNotFoundError):
        return -1
