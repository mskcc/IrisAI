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
import os
import stat
import logging
from pathlib import Path
from fnmatch import fnmatch
from typing import Annotated
import mimetypes
import hashlib
import datetime

from pydantic import Field
import pwd
import grp
import json
import time
import re
from collections import deque
import shutil
import subprocess
import urllib.request
import ssl

logger = logging.getLogger(__name__)

from user_parsers import (
    structure_user_info,
    structure_group_info,
    structure_accessible_dirs,
    parse_adquery_user,
    parse_adquery_group,
    structure_hpc_user_lookup,
    structure_hpc_group_lookup,
    filter_hpc_users,
)

# ── User data directory helper ──────────────────────────────────────────
IRISAI_APP_NAME = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")

def get_user_data_dir(username: str) -> Path:
    """Get the user's IrisAI data directory."""
    return Path(f"/home/{username}/{IRISAI_APP_NAME}")

mcp = FastMCP("File Operations & Search Server",auth=StaticBearerProvider()) 

# ── Get current user settings ──────────────────────────────────────────────────



def _format_knowledge_base_summary(kb: dict) -> str:
    """Format a knowledge_base dict into human-readable text for agents.
    Inlined in file_ops.py to keep MCP server self-contained."""
    if not kb or not isinstance(kb, dict):
        return ""

    lines = ["KNOWLEDGE BASE (from user settings):"]

    # Environment section
    env = kb.get("environment", {})
    if env and isinstance(env, dict):
        for key, value in env.items():
            lines.append(f"  {key}: {value}")

    # Software section
    sw = kb.get("software", {})
    if sw and isinstance(sw, dict):
        for key, value in sw.items():
            lines.append(f"  {key}: {value}")

    # SLURM section
    slurm = kb.get("slurm", {})
    if slurm and isinstance(slurm, dict):
        for key, value in slurm.items():
            lines.append(f"  slurm_{key}: {value}")

    # Git section
    git = kb.get("git", {})
    if git and isinstance(git, dict):
        for key, value in git.items():
            lines.append(f"  git_{key}: {value}")

    # Projects section
    projects = kb.get("projects", {})
    if projects and isinstance(projects, dict):
        for key, value in projects.items():
            lines.append(f"  project_{key}: {value}")

    # Warnings
    warnings = kb.get("warnings", [])
    if warnings and isinstance(warnings, list):
        lines.append("WARNINGS:")
        for w in warnings:
            if isinstance(w, str) and w.strip():
                lines.append(f"  WARNING: {w}")

    # Learned facts
    learned = kb.get("learned", [])
    if learned and isinstance(learned, list):
        lines.append("LEARNED FACTS:")
        for fact in learned:
            if isinstance(fact, str) and fact.strip():
                lines.append(f"  - {fact}")

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


@mcp.tool
def get_user_settings() -> dict:
    """Load the current user's persistent settings from disk. Call this at the start of any workflow to get work_dir, weights_path, software_paths, and knowledge_base. Takes no parameters — uses the OS username automatically. Returns dict with settings including work_dir, weights_path (if set), software_paths (list of directory hints where user-installed software may be found), knowledge_base (accumulated facts about the environment), and knowledge_base_summary (human-readable text of key facts). NOTE: project_name is NOT in settings — use the 'Project name' and 'Project directory' from your USER ENVIRONMENT context instead. DO NOT call this repeatedly in the same conversation — call once and reuse the values. CHECK the knowledge_base section BEFORE searching for software or paths — it may already have the answer."""
    debug_log = []

    try:
        # Get current username
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        debug_log.append(f"Current username: {username}")

        # Get settings file path
        base_dir = get_user_data_dir(username)
        base_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(base_dir, stat.S_IRWXU)  # 700: user only
        file_path = base_dir / "usersettings.json"

        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    settings = json.load(f)
                debug_log.append("Settings loaded successfully")

                # Format knowledge_base into human-readable summary
                kb_summary = _format_knowledge_base_summary(
                    settings.get("knowledge_base", {})
                )

                return {
                    "success": True,
                    "username": username,
                    "settings": settings,
                    "knowledge_base_summary": kb_summary,
                    "debug_log": debug_log,
                    "message": "User settings loaded"
                }
            except Exception as e:
                debug_log.append(f"Failed to read settings file: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to read settings: {str(e)}",
                    "debug_log": debug_log
                }
        else:
            debug_log.append("No settings file found (first time)")
            return {
                "success": True,
                "username": username,
                "settings": {},
                "knowledge_base_summary": "",
                "debug_log": debug_log,
                "message": "No settings file found — first time user"
            }
    except Exception as e:
        debug_log.append(f"Top-level error: {str(e)}")
        return {"success": False, "error": str(e), "debug_log": debug_log}


# ── Basic listing & discovery ──────────────────────────────────────────────────

@mcp.tool
def list_directory(path: str, show_hidden: bool = False, limit: int = 200) -> dict:
    """List files and subdirectories in a directory. Call this to explore directory contents or verify files exist. DO NOT call this on very large directories with thousands of files — use find_files with a pattern instead. For 2+ directory listings, use batch with type='list' instead — one call handles all.

        Args:
            path: Absolute path to the directory. Must start with /.
            show_hidden: If true, include files starting with '.' (default: false).
            limit: Maximum number of entries to return (default: 200).
        Returns dict with list of files/dirs, their sizes, and modification times."""
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return {"error": f"{path} is not a directory"}

        items = []
        total_count = 0
        for item in p.iterdir():
            if not show_hidden and item.name.startswith('.'):
                continue
            total_count += 1
            if len(items) < limit:
                stat_info = item.stat()
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size_bytes": stat_info.st_size,
                })
        result = {"contents": items, "path": str(p), "total_items": total_count}
        if total_count > limit:
            result["truncated"] = True
            result["message"] = f"Showing {limit} of {total_count} items. Use find_files with a pattern to narrow results."
        return result
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def find_files(pattern: str = "*", start_path: str = ".", recursive: bool = True, max_depth: int = 5) -> dict:
    """Search for files matching a glob pattern, optionally recursive. Call this to locate specific files by name or extension. Prefer this over list_directory when you know the filename pattern. DO NOT use overly broad patterns like '*' in large directory trees — be specific (e.g. '*.fasta', '*.json'). For 2+ find operations, use batch with type='find' instead — one call handles all searches.

        Args:
            pattern: Glob pattern like '*.fasta', 'fold_input*.json'. Be specific.
            start_path: Absolute path to start searching from. Must start with /.
            recursive: Search subdirectories (default: true).
            max_depth: Maximum directory depth to search (default: 5). Keep low for performance.
        Returns dict with list of matching file paths."""
    try:

        start = Path(start_path).resolve()
        results = []
        count = 0
        
        def search(current: Path, depth: int):
            nonlocal count
            if depth > max_depth:
                return
            try:
                for item in current.iterdir():
                    if fnmatch(item.name, pattern):
                        results.append(str(item))
                        count += 1
                    if item.is_dir():
                        search(item, depth + 1)
            except Exception:
                pass  # skip inaccessible dirs

        search(start, 0)
        return {
            "pattern": pattern,
            "found": count,
            "files": results[:200] if len(results) > 200 else results,  # safety limit
            "truncated": len(results) > 200
        }
    except Exception as e:
        return {"error": str(e)}

# ── File reading & metadata ────────────────────────────────────────────────────

# Safety cap for read_text_file to prevent context window overflow.
# 50KB of text ≈ 17K tokens — safe for LLM context.
# Files larger than this MUST use large file tools (read_file_head, read_file_lines, etc.)
READ_TEXT_FILE_SAFE_LIMIT = 50 * 1024  # 50KB

@mcp.tool
def read_text_file(path: str, max_bytes: int = 1048576) -> dict:  # 1MB hard limit
    """Read the full content of a small text file (<50KB) and return it as a string. ALWAYS call get_file_info(path=...) first to check file size before calling this. Call this for config files, scripts, logs, JSON, markdown, or small text files. DO NOT call this for: (1) FASTA/sequence files — pass the file path directly to bio tools like prepare_af3_json_from_sequences or mutate_fasta instead, (2) files over 50KB — use get_file_overview, read_file_head, or grep_file instead, (3) binary files. Returns error if file exceeds 50KB. If you get a 'File too large' error, switch to large file tools immediately — do NOT retry read_text_file on the same file. For 2+ file reads, use batch with type='read' instead — one call handles all reads.

    Args:
        path: Absolute path to the text file. Must start with /. Do NOT pass FASTA file paths here.
        max_bytes: Hard size limit in bytes (default: 1MB). The 50KB soft limit applies first.
    Returns dict with file content string, size, and mime type."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file"}
        
        file_size = p.stat().st_size
        
        if file_size > max_bytes:
            return {"error": f"File too large (> {max_bytes/1024/1024} MB)"}
        
        # ── SAFETY CAP: Prevent context window overflow ──
        # If the file is larger than the safe limit, return a helpful error
        # directing the agent to use large file tools instead.
        if file_size > READ_TEXT_FILE_SAFE_LIMIT:
            return {
                "error": f"File too large for read_text_file ({_human_size(file_size)}). "
                         f"This would consume ~{file_size // 3:,} tokens and overflow the context window. "
                         f"Use these large file tools instead:\n"
                         f"  - get_file_overview(path=\"{path}\") — quick summary with samples\n"
                         f"  - read_file_head(path=\"{path}\", num_lines=100) — first 100 lines\n"
                         f"  - read_file_tail(path=\"{path}\", num_lines=100) — last 100 lines\n"
                         f"  - read_file_lines(path=\"{path}\", start_line=1, end_line=100) — specific range\n"
                         f"  - grep_file(path=\"{path}\", pattern=\"search_term\") — search for pattern",
                "file_size_bytes": file_size,
                "file_size_human": _human_size(file_size),
                "suggestion": "Use large file tools (read_file_head, read_file_lines, grep_file, get_file_overview) for files over 50KB"
            }
        
        content = p.read_text(encoding="utf-8", errors="replace")
        mime_type, _ = mimetypes.guess_type(str(p))
        return {
            "content": content,
            "size_bytes": file_size,
            "mime_type": mime_type or "text/plain",
            "last_modified": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def get_file_checksum(path: str, algorithm: str = "sha256") -> dict:
    """Calculate a cryptographic checksum of a file. Call this to verify file integrity or compare two files. Useful after file transfers or mutations to confirm content matches. DO NOT call this just to check if a file exists — use get_file_info or check_directory_exists instead.

        Args:
            path: Absolute path to the file. Must start with /.
            algorithm: Hash algorithm — one of: md5, sha1, sha256, sha512 (default: sha256).
        Returns dict with the hex checksum string."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file"}
        
        hash_obj = getattr(hashlib, algorithm)()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return {"algorithm": algorithm, "checksum": hash_obj.hexdigest()}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def get_file_info(path: str) -> dict:
    """Get detailed metadata about a file or directory without reading its content. Call this BEFORE read_text_file to check file size — if size_bytes > 50000, do NOT call read_text_file, use get_file_overview or read_file_head instead. Also useful for checking permissions, ownership, and modification times. DO NOT call this if you just need to check existence — use check_directory_exists or check_directory_has_files instead.

        Args:
            path: Absolute path to the file or directory. Must start with /.
        Returns dict with size_bytes, permissions, owner, group, modification time, and type."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return {"error": "Path does not exist"}
        
        stat = p.stat()
        return {
            "exists": True,
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
            "size_bytes": stat.st_size if p.is_file() else None,
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "owner_uid": stat.st_uid,
            "owner_gid": stat.st_gid
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def get_current_user_info() -> dict:
    """Get information about the current OS user running this tool. Call this once at session start if you need the username, home directory, or group memberships. DO NOT call this repeatedly — the user doesn't change during a session. Prefer get_user_settings if you only need the username and work_dir.

        Returns dict with username, uid, home_dir, primary_group, and all_groups."""
    try:
        uid = os.getuid()
        user_info = pwd.getpwuid(uid)
        primary_group = grp.getgrgid(user_info.pw_gid).gr_name
        groups = [g.gr_name for g in grp.getgrall() if user_info.pw_name in g.gr_mem] + [primary_group]
        
        return structure_user_info(
            username=user_info.pw_name,
            uid=uid,
            home_dir=user_info.pw_dir,
            primary_group=primary_group,
            all_groups=groups,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool
def get_user_groups(username: str = None) -> dict:
    """Get the groups and GIDs that a user belongs to. DO NOT call this for the current user — get_current_user_info already includes groups. Only use this to check another user's group memberships for shared directory access.

        Args:
            username: OS username to look up. If not provided, uses the current user.
        Returns dict with primary_group, all groups with GIDs and members."""
    try:
        if username is None:
            username = pwd.getpwuid(os.getuid()).pw_name
        
        user_info = pwd.getpwnam(username)
        primary_gid = user_info.pw_gid
        primary_group = grp.getgrgid(primary_gid).gr_name
        
        all_groups = []
        for group in grp.getgrall():
            if username in group.gr_mem or group.gr_gid == primary_gid:
                all_groups.append({
                    "group_name": group.gr_name,
                    "gid": group.gr_gid,
                    "members": group.gr_mem
                })
        
        return structure_group_info(
            username=username,
            primary_group=primary_group,
            primary_gid=primary_gid,
            groups=all_groups,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool
def list_group_accessible_dirs(
    username: str = None,
    level1_roots: list[str] = ["/home", "/data1", "/scratch", "/usersoftware", "/localscratch"]
) -> dict:
    """List directories in common root paths that the user can access via group permissions. Call this when the user needs to find their data directories on the HPC system. DO NOT call this for routine file operations — use list_directory or find_files instead.

        Args:
            username: OS username. If not provided, uses the current user.
            level1_roots: List of root paths to scan (default: /home, /data1, /scratch, /usersoftware, /localscratch).
        Returns dict with accessible directories grouped by root path."""
    debug_log = []
    debug_log.append("Tool started")

    try:
        # Resolve username and UID
        if username is None:
            uid = os.getuid()
            try:
                username = pwd.getpwuid(uid).pw_name
                debug_log.append(f"Resolved username from current UID {uid}: {username}")
            except Exception as e:
                debug_log.append(f"Failed to resolve username from UID {uid}: {str(e)}")
                username = "unknown"
                uid = -1
        else:
            try:
                uid = pwd.getpwnam(username).pw_uid
                debug_log.append(f"Resolved UID for provided username {username}: {uid}")
            except Exception as e:
                debug_log.append(f"Failed to resolve UID for {username}: {str(e)}")
                uid = -1

        # Get user groups
        user_groups = []
        try:
            if uid >= 0:
                primary_gid = pwd.getpwuid(uid).pw_gid
                primary_group = grp.getgrgid(primary_gid).gr_name
                user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem] + [primary_group]
            else:
                user_groups = []
        except Exception as e:
            debug_log.append(f"Failed to get groups: {str(e)}")

        accessible_dirs = []
        all_scanned_dirs = []

        debug_log.append(f"Roots to check: {level1_roots}")

        for root_path_str in level1_roots:
            debug_log.append(f"Checking root: {root_path_str}")
            try:
                root = Path(root_path_str).resolve()
                debug_log.append(f"Resolved root: {root}")

                if not root.exists():
                    debug_log.append("  - Does not exist")
                    continue
                if not root.is_dir():
                    debug_log.append("  - Not a directory")
                    continue

                children = []
                try:
                    children = list(root.iterdir())
                except Exception as e:
                    continue

                for item in children:
                    all_scanned_dirs.append(str(item))
                    if item.is_dir():
                        try:
                            # Direct stat() call — no tool
                            stat_info = item.stat()
                            owner_uid = stat_info.st_uid
                            group_gid = stat_info.st_gid
                            permissions = oct(stat_info.st_mode)[-3:]

                            group_name = None
                            try:
                                group_name = grp.getgrgid(group_gid).gr_name
                            except Exception:
                                debug_log.append(f"      Could not resolve group GID {group_gid}")

                            if group_name in user_groups or owner_uid == uid:
                                debug_log.append("      MATCH → adding")
                                accessible_dirs.append({
                                    "path": str(item),
                                    "owner_uid": owner_uid,
                                    "owner_name": pwd.getpwuid(owner_uid).pw_name if owner_uid else "unknown",
                                    "group_name": group_name,
                                    "group_gid": group_gid,
                                    "permissions": permissions
                                })
                            else:
                                debug_log.append("      No match")
                        except Exception as e:
                            debug_log.append(f"      stat() failed for {item}: {str(e)}")

                debug_log.append(f"  All dirs scanned in this root: {all_scanned_dirs[-len(children):]}")
            except Exception as e:
                debug_log.append(f"  Root-level failure for {root_path_str}: {str(e)}")

        debug_log.append("Tool finished")
        return structure_accessible_dirs(
            username=username,
            user_groups=user_groups,
            accessible_dirs=accessible_dirs,
            roots_checked=level1_roots,
        )

    except Exception as e:
        debug_log.append(f"Top-level failure: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "debug_log": debug_log
        }

@mcp.tool
def make_directory(path: str, exist_ok: bool = True) -> dict:
    """Create a directory, including parent directories if needed. Safe to call on existing directories when exist_ok is true. Note: write_text_file creates parent directories automatically, so you often do NOT need to call this before writing files.

        Args:
            path: Absolute path for the new directory. Must start with /.
            exist_ok: If true (default), don't error if directory already exists.
        Returns dict with success status."""
    try:
        p = Path(path).resolve()
        p.mkdir(parents=True, exist_ok=exist_ok)
        return {"success": True, "path": str(p), "message": "Directory created successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool
def check_directory_exists(path: str) -> dict:
    """Check if a directory exists at the given path. Call this for a quick existence check. Returns true/false — does NOT list contents. For checking if a directory has files, use check_directory_has_files instead.

        Args:
            path: Absolute path to check. Must start with /.
        Returns dict with exists boolean."""
    try:
        p = Path(path).resolve()
        exists = p.exists() and p.is_dir()
        return {"success": True, "exists": exists, "path": str(p)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool
def check_directory_has_files(path: str) -> dict:
    """Check if a directory exists AND contains any files or subdirectories. Call this to verify a directory is not empty — e.g. checking if weights are present, if output was generated. More useful than check_directory_exists alone.

        Args:
            path: Absolute path to the directory. Must start with /.
        Returns dict with exists boolean, has_files boolean, and file count."""
    try:
        p = Path(path).resolve()
        if not p.exists() or not p.is_dir():
            return {"success": False, "error": "Path does not exist or is not a directory"}
        has_files = any(p.iterdir())
        return {"success": True, "has_files": has_files, "path": str(p)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _get_user_settings_path(username: str) -> Path:
    base_dir = get_user_data_dir(username)
    base_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(base_dir, stat.S_IRWXU)  # 700: user only
    return base_dir / "usersettings.json"


def _get_work_dir() -> str:
    """Get authoritative work_dir. Checks env var first, then settings file.

    In production, both agree (bootstrap syncs env→file, set_user_work_directory
    updates file+env). Env var takes precedence because:
    - In containers, it's set explicitly at launch (intended runtime value)
    - In tests, it's monkeypatched to isolate from real settings
    - set_user_work_directory() updates both, so they stay in sync
    """
    env_wd = os.environ.get("WORK_DIR", "")
    if env_wd:
        return env_wd
    try:
        username = pwd.getpwuid(os.getuid()).pw_name
        file_path = _get_user_settings_path(username)
        if file_path.exists():
            with open(file_path, "r") as f:
                settings = json.load(f)
            work_dir = settings.get("work_dir", "")
            if work_dir:
                return work_dir
    except Exception:
        pass
    return ""

@mcp.tool
def set_user_work_directory(path: str) -> dict:
    """Set and persist the user's work directory. Call this ONCE when the user first specifies their work directory. Validates the path exists, is a directory, and is writable. Saves to persistent settings file. DO NOT call this if work_dir is already set in get_user_settings — only call when the user explicitly wants to change it.

        Args:
            path: Absolute path to an existing, writable directory. Must start with /.
        Returns dict with the validated work_dir path."""
    debug_log = []
    debug_log.append(f"Setting work directory to: {path}")

    try:
        # Inline: get current username
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        debug_log.append(f"Current username: {username}")

        work_dir = Path(path).resolve()
        debug_log.append(f"Resolved path: {work_dir}")

        if not work_dir.is_absolute():
            return {"success": False, "error": "Path must be absolute", "debug_log": debug_log}
        if ".." in str(work_dir):
            return {"success": False, "error": "Path cannot contain '..'", "debug_log": debug_log}
        if not work_dir.exists() or not work_dir.is_dir():
            return {"success": False, "error": "Path does not exist or is not a directory", "debug_log": debug_log}

        # Check writeable
        try:
            test_file = work_dir / f".test_write_{int(time.time())}"
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            debug_log.append("Path is writeable")
        except Exception as e:
            debug_log.append(f"Write test failed: {str(e)}")
            return {"success": False, "error": f"Path is not writeable: {str(e)}", "debug_log": debug_log}

        # Inline: load existing settings
        settings = {}
        file_path = _get_user_settings_path(username)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    settings = json.load(f)
                debug_log.append("Loaded existing settings")
            except Exception as e:
                debug_log.append(f"Failed to load settings: {str(e)}")

        settings["work_dir"] = str(work_dir)
        settings["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save to settings file (shared filesystem — source of truth)
        with open(file_path, "w") as f:
            json.dump(settings, f, indent=2)
        os.chmod(file_path, 0o600)

        # Update local process env so in-process code picks it up immediately
        os.environ["WORK_DIR"] = str(work_dir)

        debug_log.append(f"Saved to: {file_path}")

        return {
            "success": True,
            "work_dir": str(work_dir),
            "message": f"Work directory set to {work_dir}"
        }
    except Exception as e:
        debug_log.append(f"Error: {str(e)}")
        return {"success": False, "error": str(e), "debug_log": debug_log}



# ── Knowledge base management ──────────────────────────────────────────────────

_VALID_KB_CATEGORIES = frozenset([
    "environment", "software", "warnings", "learned",
    "slurm", "git", "projects",
])
_MAX_LIST_ITEMS = 50

_MAX_SKILL_KB_ENTRIES = 20

# ── Unified Memory System Tools ──────────────────────────────────────────────
_ALLOWED_MEMORY_FILES = {"status.md", "knowledge.md", "history.md"}

@mcp.tool
def update_memory(
    filename: Annotated[str, Field(description="Must be one of: 'status.md', 'knowledge.md', 'history.md'.")],
    content: Annotated[str, Field(description="New content. For status.md: full replacement. For knowledge.md/history.md: append new entries.", min_length=1)],
    project: Annotated[str, Field(description="Project name, or '_global' for cross-project knowledge.")] = "",
    reason: Annotated[str, Field(description="Optional reason for the update (logged for audit trail).")] = "",
) -> dict:
    """Update one of the 3 project memory files (status.md, knowledge.md, history.md) or global knowledge (_global). Call immediately when you learn something important or state changes. Returns dict with success status."""
    try:
        if not filename or not filename.strip():
            return {"success": False, "error": "filename must be non-empty"}
        filename = filename.strip()
        if filename not in _ALLOWED_MEMORY_FILES:
            return {
                "success": False,
                "error": f"Invalid filename '{filename}'. Must be one of: {sorted(_ALLOWED_MEMORY_FILES)}. "
                         f"Each project has exactly 3 memory files.",
            }
        if not content or not content.strip():
            return {"success": False, "error": "content must be non-empty"}
        if not project or not project.strip():
            return {"success": False, "error": "project is required for update_memory (use '_global' for cross-project knowledge)"}

        project = project.strip()
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        memory_root = get_user_data_dir(username) / "memory"

        if project == "_global":
            if filename != "knowledge.md":
                return {
                    "success": False,
                    "error": "Only 'knowledge.md' is allowed for _global. Status and history are per-project only.",
                }
            fpath = memory_root / "knowledge.md"
            memory_root.mkdir(parents=True, exist_ok=True)
        else:
            project_dir = memory_root / "projects" / project
            project_dir.mkdir(parents=True, exist_ok=True)
            fpath = project_dir / filename

        fpath.write_text(content.strip() + "\n", encoding="utf-8")

        if reason:
            logger.info(f"[MEMORY_UPDATE] {project}/{filename}: {reason[:100]}")

        result = {
            "success": True,
            "filename": filename,
            "project": project,
            "chars_written": len(content.strip()),
            "message": f"Updated {project}/{filename}",
        }

        # Detect software-path content and suggest register_software instead
        import re as _re
        _sw_path = _re.search(r'/\S+/(bin|lib|envs?|conda|spack|software|packages)/\S*', content)
        _has_version = bool(_re.search(r'\b\d+\.\d+(\.\d+)?\b', content))
        _sw_keywords = any(kw in content.lower() for kw in [
            'installed', 'compiled', 'version', 'binary', 'executable',
            'samtools', 'bwa', 'conda env', 'spack',
        ])
        if _sw_path and (_has_version or _sw_keywords):
            result["warning"] = (
                "This content appears to describe installed software. "
                "Software paths and versions MUST be registered using "
                "register_software() — the software registry is the canonical "
                "location for installed software discovery. Your write to "
                "knowledge.md succeeded, but please ALSO call register_software() "
                "to ensure the software is discoverable via query_software()."
            )

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# NEW MEMORY TOOLS — Strict 3-file model (status.md, knowledge.md, history.md)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool
def read_memory(project: str = "") -> dict:
    """Read all project memory in one call. Returns the 3 project files (status.md, knowledge.md, history.md), global knowledge, available projects, and last turn info. Call this FIRST when starting work, resuming a session, when the user asks 'what happened', 'resume', 'continue', or when you need to refresh your understanding of the project state. Also useful after the user says your memory is stale.

        Args:
            project: Project name. If empty, auto-detects from the last active project.
        Returns dict with status, knowledge, history, global_knowledge, available_projects, active_project, last_turn, and recent_attempts."""
    try:
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        memory_root = get_user_data_dir(username) / "memory"

        result = {
            "success": True,
            "status": None,
            "knowledge": None,
            "history": None,
            "global_knowledge": None,
            "available_projects": [],
            "active_project": None,
            "last_turn": None,
            "recent_attempts": None,
        }

        # Auto-detect project from last_turn.json if not provided
        last_turn_path = memory_root / "session" / "last_turn.json"
        if last_turn_path.exists():
            try:
                result["last_turn"] = json.loads(last_turn_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        effective_project = project.strip() if project else ""
        if not effective_project and result["last_turn"]:
            effective_project = result["last_turn"].get("project", "")
        result["active_project"] = effective_project or None

        # Read the 3 project files
        if effective_project:
            project_dir = memory_root / "projects" / effective_project
            if project_dir.is_dir():
                for fname in ("status.md", "knowledge.md", "history.md"):
                    fpath = project_dir / fname
                    if fpath.exists():
                        content = fpath.read_text(encoding="utf-8")
                        # Strip frontmatter if present
                        if content.startswith("---"):
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                content = parts[2].strip()
                        key = fname.replace(".md", "")
                        # Cap history to last 3000 chars (grows over time)
                        if key == "history" and len(content) > 3000:
                            content = "...(earlier entries omitted)\n" + content[-3000:]
                        # Cap knowledge at 8000 chars
                        elif key == "knowledge" and len(content) > 8000:
                            content = content[:8000] + "\n...(truncated)"
                        result[key] = content if content.strip() else None

                # Recent attempts (last 5)
                attempts_path = project_dir / "attempts_log.jsonl"
                if attempts_path.exists():
                    try:
                        entries = []
                        for line in attempts_path.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    entries.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        if entries:
                            result["recent_attempts"] = entries[-5:]
                    except Exception:
                        pass

        # Global knowledge (cross-project facts)
        global_knowledge_path = memory_root / "knowledge.md"
        if global_knowledge_path.exists():
            try:
                gk_content = global_knowledge_path.read_text(encoding="utf-8")
                if gk_content.startswith("---"):
                    parts = gk_content.split("---", 2)
                    if len(parts) >= 3:
                        gk_content = parts[2].strip()
                if gk_content.strip():
                    if len(gk_content) > 6000:
                        gk_content = gk_content[:6000] + "\n...(truncated)"
                    result["global_knowledge"] = gk_content
            except Exception:
                pass

        # Available projects
        projects_index_path = memory_root / "projects_index.json"
        if projects_index_path.exists():
            try:
                idx = json.loads(projects_index_path.read_text(encoding="utf-8"))
                result["available_projects"] = [
                    {"name": p.get("name", ""), "description": p.get("description", "")}
                    for p in idx.get("projects", []) if p.get("name")
                ]
            except Exception:
                pass
        # Also check filesystem for unregistered projects
        projects_dir = memory_root / "projects"
        if projects_dir.is_dir():
            known_names = {p["name"] for p in result["available_projects"]}
            for entry in projects_dir.iterdir():
                if entry.is_dir() and entry.name not in known_names and not entry.name.startswith("_"):
                    result["available_projects"].append({"name": entry.name, "description": ""})

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def list_projects() -> dict:
    """List all available projects with their descriptions and last activity date. Call this when the user asks 'what projects do I have', 'show my projects', or when you need to discover which projects exist before switching context.

        Returns dict with a list of projects (name, description, date_last_active)."""
    try:
        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        memory_root = get_user_data_dir(username) / "memory"

        projects = []
        projects_index_path = memory_root / "projects_index.json"
        if projects_index_path.exists():
            try:
                idx = json.loads(projects_index_path.read_text(encoding="utf-8"))
                for p in idx.get("projects", []):
                    if p.get("name"):
                        projects.append({
                            "name": p["name"],
                            "description": p.get("description", ""),
                            "date_last_active": p.get("date_last_active", ""),
                        })
            except Exception:
                pass

        # Check filesystem for unregistered projects
        projects_dir = memory_root / "projects"
        if projects_dir.is_dir():
            known_names = {p["name"] for p in projects}
            for entry in projects_dir.iterdir():
                if entry.is_dir() and entry.name not in known_names and not entry.name.startswith("_"):
                    projects.append({
                        "name": entry.name,
                        "description": "",
                        "date_last_active": "",
                    })

        return {
            "success": True,
            "projects": projects,
            "count": len(projects),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def add_project(name: str, description: str = "") -> dict:
    """Register a new project and create its memory structure. Call this when the user explicitly wants to start tracking a new project, or when you detect work on a project that doesn't exist yet.

        Args:
            name: Project name (short, no slashes or special chars). E.g. 'segger', 'alphafold3_braf'.
            description: One-line description of the project.
        Returns dict with success status and project directory path."""
    try:
        if not name or not name.strip():
            return {"success": False, "error": "name must be non-empty"}
        name = name.strip()
        if "/" in name or "\\" in name or ".." in name:
            return {"success": False, "error": "name cannot contain slashes or '..'"}

        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        memory_root = get_user_data_dir(username) / "memory"

        # Register in projects_index.json (idempotent)
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.memory_state import register_project
            register_project(name, description)
        except ImportError:
            # Fallback: manual registration
            index_path = memory_root / "projects_index.json"
            data = {"projects": []}
            if index_path.exists():
                try:
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            existing = [p for p in data.get("projects", []) if p.get("name") == name]
            if not existing:
                data.setdefault("projects", []).append({
                    "name": name,
                    "description": description,
                    "date_created": time.strftime("%Y-%m-%d"),
                    "date_last_active": time.strftime("%Y-%m-%d"),
                })
                index_path.parent.mkdir(parents=True, exist_ok=True)
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

        # Create the 3 initial files
        project_dir = memory_root / "projects" / name
        project_dir.mkdir(parents=True, exist_ok=True)

        initial_files = {
            "status.md": f"# {name} — Status\n\nNew project. No work done yet.\n",
            "knowledge.md": f"# {name} — Knowledge\n\n",
            "history.md": f"# {name} — History\n\n",
        }
        for fname, content in initial_files.items():
            fpath = project_dir / fname
            if not fpath.exists():
                fpath.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "project_name": name,
            "project_dir": str(project_dir),
            "message": f"Project '{name}' registered with memory structure created.",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def remove_project(name: str, archive: bool = True) -> dict:
    """Remove or archive a project's memory. Use archive=True (default) to preserve the data in an _archived/ directory. Use archive=False only when the user explicitly says to permanently delete.

        Args:
            name: Project name to remove.
            archive: If True, moves to _archived/{name}_{timestamp}/ (recoverable). If False, permanently deletes.
        Returns dict with success status and action taken."""
    try:
        if not name or not name.strip():
            return {"success": False, "error": "name must be non-empty"}
        name = name.strip()

        uid = os.getuid()
        username = pwd.getpwuid(uid).pw_name
        memory_root = get_user_data_dir(username) / "memory"

        project_dir = memory_root / "projects" / name
        if not project_dir.exists():
            return {"success": False, "error": f"Project '{name}' not found at {project_dir}"}

        if archive:
            archive_dir = memory_root / "_archived"
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dest = archive_dir / f"{name}_{timestamp}"
            shutil.move(str(project_dir), str(dest))
            action = f"Archived to {dest}"
        else:
            shutil.rmtree(str(project_dir))
            action = "Permanently deleted"

        # Remove from projects_index.json
        index_path = memory_root / "projects_index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                data["projects"] = [p for p in data.get("projects", []) if p.get("name") != name]
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        return {
            "success": True,
            "project_name": name,
            "action": action,
            "message": f"Project '{name}' removed. {action}.",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _load_package_registry(env: dict) -> tuple:
    """Load package registry YAML. Returns (packages_list, registry_dict) or ([], {}).
    Resolves ${WORK_DIR} in path and falls back to fallback_path if primary not found."""
    import yaml as _yaml
    pkg_reg_config = env.get("package_registry", {})
    reg_path = pkg_reg_config.get("path", "")

    # Resolve ${WORK_DIR} variable in path
    if reg_path and "${WORK_DIR}" in reg_path:
        work_dir = _get_work_dir()
        if work_dir:
            reg_path = reg_path.replace("${WORK_DIR}", work_dir)
        else:
            reg_path = ""

    # Fall back to fallback_path if primary doesn't exist
    if not reg_path or not Path(reg_path).exists():
        reg_path = pkg_reg_config.get("fallback_path", "")

    if not reg_path or not Path(reg_path).exists():
        return [], {}

    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = _yaml.safe_load(f) or {}
        return reg.get("packages", []), reg
    except Exception:
        return [], {}


def _get_package_registry(env: dict) -> dict:
    """Return full package registry with all packages and their purposes."""
    packages, reg = _load_package_registry(env)
    if not packages:
        return {"success": False, "error": "Package registry not found or empty"}

    pkg_reg_config = env.get("package_registry", {})
    result = {
        "success": True,
        "python_bin": reg.get("python_bin", pkg_reg_config.get("python_bin", "")),
        "knowledge_dir": reg.get("knowledge_dir", pkg_reg_config.get("knowledge_dir", "")),
        "packages": [],
    }
    for p in packages:
        if not isinstance(p, dict):
            continue
        entry = {
            "name": p.get("name", ""),
            "version": p.get("version", ""),
            "source": p.get("source", ""),
            "purpose": p.get("purpose", ""),
            "categories": p.get("categories", []),
            "status": p.get("status", "installed"),
        }
        if p.get("knowledge_file"):
            entry["knowledge_file"] = p["knowledge_file"]
        result["packages"].append(entry)
    return result


def _get_package_detail(env: dict, pkg_name: str) -> dict:
    """Return detailed info for a specific package, including knowledge file content."""
    packages, reg = _load_package_registry(env)
    if not packages:
        return {"success": False, "error": "Package registry not found"}

    pkg_name_lower = pkg_name.lower()
    pkg = None
    for p in packages:
        if isinstance(p, dict) and p.get("name", "").lower() == pkg_name_lower:
            pkg = p
            break

    if not pkg:
        available = [p["name"] for p in packages if isinstance(p, dict)]
        return {"success": False, "error": f"Package '{pkg_name}' not found. Available: {available}"}

    result = {"success": True, "package": dict(pkg)}

    # Load knowledge file if it exists
    knowledge_file = pkg.get("knowledge_file", "")
    if knowledge_file:
        pkg_reg_config = env.get("package_registry", {})
        # knowledge_file can be relative to registry dir or absolute
        knowledge_path = Path(knowledge_file)
        if not knowledge_path.is_absolute():
            reg_dir = Path(pkg_reg_config.get("path", "")).parent
            knowledge_path = reg_dir / knowledge_file
        if knowledge_path.exists():
            try:
                result["knowledge"] = knowledge_path.read_text(encoding="utf-8")
            except Exception:
                result["knowledge_error"] = f"Could not read {knowledge_path}"
        else:
            result["knowledge_error"] = f"Knowledge file not found: {knowledge_path}"

    return result


def _get_packages_by_category(env: dict, category: str) -> dict:
    """Return all packages matching a category."""
    packages, reg = _load_package_registry(env)
    if not packages:
        return {"success": False, "error": "Package registry not found"}

    category_lower = category.lower()
    matches = []
    for p in packages:
        if not isinstance(p, dict):
            continue
        cats = [c.lower() for c in p.get("categories", [])]
        if category_lower in cats:
            matches.append({
                "name": p.get("name", ""),
                "version": p.get("version", ""),
                "purpose": p.get("purpose", ""),
                "status": p.get("status", "installed"),
            })

    if not matches:
        all_cats = set()
        for p in packages:
            if isinstance(p, dict):
                all_cats.update(c.lower() for c in p.get("categories", []))
        return {"success": False, "error": f"No packages in category '{category}'. Available categories: {sorted(all_cats)}"}

    return {"success": True, "category": category, "packages": matches}


@mcp.tool
def get_environment_info(topic: str = "") -> dict:
    """Get detailed HPC environment information for a specific topic. Call this BEFORE submitting jobs, choosing containers, or using software. The system prompt only shows an index of available names — this tool returns full paths, specs, constraints, and notes.

        Args:
            topic: What to look up. One of:
                - "containers" — all containers with paths, software lists, GPU capability
                - "partitions" — all partitions with GPU types, time limits, availability
                - "software" — installed software with paths, versions, commands
                - "default_container" — the default container for Slurm jobs
                - "container_building" — best practices for building Singularity .def files (fakeroot env vars, bind mounts, cache dirs)
                - "packages" — full package registry (all scientific/visualization software with purposes)
                - "package:<name>" — detailed info + knowledge for a specific package (e.g., "package:drawsvg")
                - "category:<name>" — all packages in a category (e.g., "category:visualization")
                - A specific name (e.g., "alphafold3", "gpu_h200", "drawsvg") — returns that entry's full details
                - "" (empty) — returns everything (full environment registry)
        Returns dict with the requested environment details."""
    import yaml

    env_yaml_path = Path(__file__).parent.parent / "config" / "environment.yaml"
    if not env_yaml_path.exists():
        return {"success": False, "error": "environment.yaml not found"}

    try:
        with open(env_yaml_path, "r", encoding="utf-8") as f:
            env = yaml.safe_load(f) or {}
    except Exception as e:
        return {"success": False, "error": f"Failed to parse environment.yaml: {e}"}

    topic = topic.strip().lower()

    if not topic:
        return {"success": True, "environment": env}

    if topic == "containers":
        return {"success": True, "containers": env.get("containers", {}),
                "default_container": env.get("default_container", {})}

    if topic == "partitions":
        return {"success": True, "partitions": env.get("partitions", {})}

    if topic == "software":
        return {"success": True, "software": env.get("software", {})}

    if topic in ("default_container", "default"):
        return {"success": True, "default_container": env.get("default_container", {})}

    if topic in ("container_building", "building", "build_container", "singularity_build", "fakeroot"):
        return {"success": True, "container_building": env.get("container_building", {})}

    # Package registry lookups
    if topic in ("packages", "package_registry", "registry"):
        return _get_package_registry(env)

    if topic.startswith("package:"):
        pkg_name = topic[len("package:"):].strip()
        return _get_package_detail(env, pkg_name)

    # Category-based package search
    if topic.startswith("category:"):
        category = topic[len("category:"):].strip()
        return _get_packages_by_category(env, category)

    # Search for a specific name across all sections
    containers = env.get("containers", {})
    if topic in containers:
        return {"success": True, "container": {topic: containers[topic]}}

    partitions = env.get("partitions", {})
    if topic in partitions:
        return {"success": True, "partition": {topic: partitions[topic]}}

    software = env.get("software", {})
    if topic in software:
        return {"success": True, "software": {topic: software[topic]}}

    # Check package registry (search by name — after static sections above)
    pkg_result = _get_package_detail(env, topic)
    if pkg_result.get("success"):
        return pkg_result

    # Fuzzy match: check if topic is a substring of any key
    matches = {}
    for section_name, section in [("containers", containers), ("partitions", partitions), ("software", software)]:
        for key, val in section.items():
            if topic in key or topic in str(val).lower():
                matches[f"{section_name}/{key}"] = val

    if matches:
        return {"success": True, "matches": matches}

    return {"success": False, "error": f"No entry found for '{topic}'. "
            f"Available: containers={list(containers.keys())}, "
            f"partitions={list(partitions.keys())}, software={list(software.keys())}. "
            f"Also try: 'packages' (full registry), 'package:<name>' (specific package), "
            f"'category:<name>' (by category)"}


@mcp.tool
def write_text_file(path: str, content: str) -> dict:
    """Write text content to a file, creating parent directories if needed. Call this to save scripts, configs, notes, or generated content. Path MUST be within the user's work directory (WORK_DIR). DO NOT call this with empty content or placeholder values. DO NOT use this to write FASTA files — bio tools handle that internally. DO NOT write AlphaFold3 fold_input.json manually — use prepare_af3_json_from_sequences instead.

        Args:
            path: Absolute path for the file. Must start with / and be inside WORK_DIR.
            content: The actual text content to write. Must be non-empty. No placeholders.
        Returns dict with success status and the written path."""
    if not path or not isinstance(path, str) or not path.strip():
        return {"error": "Path must be a non-empty string"}
    if not content or not isinstance(content, str) or not content.strip():
        return {"error": "Content must be a non-empty string — do not send empty or placeholder values"}
    work_dir = _get_work_dir()
    if not work_dir:
        return {"error": "No work directory configured. Set one with set_user_work_directory()."}
    if not os.path.isabs(work_dir):
        return {"error": "WORK_DIR must be an absolute path"}
    work_dir_resolved = str(Path(work_dir).resolve())
    p = Path(path).resolve()
    if not str(p).startswith(work_dir_resolved + os.sep) and str(p) != work_dir_resolved:
        return {"error": "Path must be within work directory"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"success": True, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}



@mcp.tool
def edit_file(path: str, old_text: str, new_text: str) -> dict:
    """Make a precise text replacement in an existing file. For a SINGLE edit in one file. If you need 2+ edits across different files, use the 'batch' tool with type='edit' instead — one call handles all edits. Requires that old_text appears EXACTLY ONCE in the file. The old_text is replaced with new_text. ALWAYS read the file first to get the exact text to replace.

        Args:
            path: Absolute path to the file. Must start with / and be inside WORK_DIR.
            old_text: The exact text to find and replace. Must appear exactly once in the file. Include enough context (surrounding lines) to ensure uniqueness.
            new_text: The replacement text. Can be empty string to delete the old_text.
        Returns dict with success status, path, and a preview of the edited region with context."""
    # Validate inputs
    if not path or not isinstance(path, str) or not path.strip():
        return {"error": "Path must be a non-empty string"}
    if not isinstance(old_text, str) or not old_text:
        return {"error": "old_text must be a non-empty string — specify the exact text to replace"}
    if not isinstance(new_text, str):
        return {"error": "new_text must be a string (can be empty to delete text)"}

    # Validate path is within work directory
    work_dir = _get_work_dir()
    if not work_dir:
        return {"error": "No work directory configured. Set one with set_user_work_directory()."}
    if not os.path.isabs(work_dir):
        return {"error": "WORK_DIR must be an absolute path"}
    work_dir_resolved = str(Path(work_dir).resolve())
    p = Path(path).resolve()
    if not str(p).startswith(work_dir_resolved + os.sep) and str(p) != work_dir_resolved:
        return {"error": "Path must be within work directory"}
    if not p.is_file():
        return {"error": f"File not found: {path}"}

    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    # Count occurrences
    count = content.count(old_text)
    if count == 0:
        # Provide helpful context: show first 100 chars of old_text for debugging
        preview = old_text[:100].replace("\n", "\\n")
        return {"error": f"old_text not found in file. Make sure you copied the exact text including whitespace and newlines. Searched for: {preview!r}"}
    if count > 1:
        return {"error": f"old_text appears {count} times in the file. It must appear exactly once. Include more surrounding context lines in old_text to make it unique."}

    # Perform the replacement
    new_content = content.replace(old_text, new_text, 1)

    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return {"error": f"Failed to write file: {e}"}

    # Build a preview of the edited region with context
    new_lines = new_content.splitlines()
    # Find where the new_text starts in the result
    # Use the position of the replacement to find the right line range
    pre_replacement = content[:content.index(old_text)]
    start_line = pre_replacement.count("\n")
    replacement_line_count = new_text.count("\n") + 1 if new_text else 0
    end_line = start_line + replacement_line_count

    # Show 3 lines of context before and after
    context_before = max(0, start_line - 3)
    context_after = min(len(new_lines), end_line + 3)
    preview_lines = []
    for i in range(context_before, context_after):
        if i < len(new_lines):
            preview_lines.append(f"{i + 1}: {new_lines[i]}")

    return {
        "success": True,
        "path": str(p),
        "replacements": 1,
        "preview": "\n".join(preview_lines),
    }

@mcp.tool
def remove_file(path: str) -> dict:
    """Delete a file from disk. Path MUST be within the user's work directory (WORK_DIR). CRITICAL: The agent MUST ask the user for explicit confirmation BEFORE calling this tool. Never call this without user saying 'yes' or 'confirm'. This action is irreversible.

        Args:
            path: Absolute path to the file to delete. Must start with / and be inside WORK_DIR.
        Returns dict with success status."""
    work_dir = _get_work_dir()
    if not work_dir:
        return {"error": "No work directory configured. Set one with set_user_work_directory()."}
    if not os.path.isabs(work_dir):
        return {"error": "Work directory must be an absolute path"}
    work_dir_resolved = str(Path(work_dir).resolve())
    p = Path(path).resolve()
    if not str(p).startswith(work_dir_resolved + os.sep) and str(p) != work_dir_resolved:
        return {"error": "Path must be within work directory"}
    if not p.is_file():
        return {"error": "Not a file or does not exist"}
    try:
        p.unlink()
        return {"success": True, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}


# ── Recent uploads discovery ──────────────────────────────────────────────────

@mcp.tool
def list_recent_uploads(work_dir: str = None, project_dir: str = "", hours: int = 48, max_files: int = 20) -> dict:
    """List files recently uploaded via the upload tool. Checks both project-specific uploads (project_dir/uploads/) and global uploads (work_dir/uploads/). Returns file paths that can be passed directly to bio tools (e.g. prepare_af3_json_from_sequences). DO NOT read the content of uploaded FASTA files — pass their paths directly to the appropriate tool.

        Args:
            work_dir: User's work directory. If not provided, loads from user settings automatically.
            project_dir: Active project directory. If provided, also scans project_dir/uploads/.
            hours: How far back to look in hours (default: 48).
            max_files: Maximum files to return (default: 20).
        Returns dict with list of files including their absolute paths, sizes, and upload times."""
    try:
        # Resolve work_dir from settings file (single source of truth)
        if not work_dir:
            work_dir = _get_work_dir()
            if not work_dir:
                return {
                    "success": False,
                    "error": "No work_dir provided and none found in user settings.",
                    "files": []
                }

        cutoff_time = time.time() - (hours * 3600)
        recent_files = []
        seen_names = set()

        # Scan project-specific uploads first (higher priority)
        if project_dir:
            project_uploads = Path(project_dir) / "uploads"
            if project_uploads.exists() and project_uploads.is_dir():
                for item in project_uploads.iterdir():
                    if not item.is_file():
                        continue
                    try:
                        stat_info = item.stat()
                        if stat_info.st_mtime >= cutoff_time:
                            name = item.name
                            original_name = name
                            if re.match(r"^\d{8}_\d{6}_", name):
                                original_name = name[16:]
                            recent_files.append({
                                "path": str(item),
                                "name": original_name,
                                "full_name": name,
                                "size_bytes": stat_info.st_size,
                                "uploaded_at": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                                "age_hours": round((time.time() - stat_info.st_mtime) / 3600, 1),
                                "location": "project",
                            })
                            seen_names.add(name)
                    except Exception:
                        continue

        # Scan global uploads (skip duplicates already found in project dir)
        uploads_dir = Path(work_dir) / "uploads"
        if uploads_dir.exists() and uploads_dir.is_dir():
            for item in uploads_dir.iterdir():
                if not item.is_file():
                    continue
                if item.name in seen_names:
                    continue
                try:
                    stat_info = item.stat()
                    if stat_info.st_mtime >= cutoff_time:
                        name = item.name
                        original_name = name
                        if re.match(r"^\d{8}_\d{6}_", name):
                            original_name = name[16:]
                        recent_files.append({
                            "path": str(item),
                            "name": original_name,
                            "full_name": name,
                            "size_bytes": stat_info.st_size,
                            "uploaded_at": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                            "age_hours": round((time.time() - stat_info.st_mtime) / 3600, 1),
                            "location": "global",
                        })
                except Exception:
                    continue

        if not recent_files:
            return {
                "success": True,
                "files": [],
                "uploads_dir": str(uploads_dir),
                "message": "No files have been uploaded recently."
            }

        # Sort by upload time, newest first
        recent_files.sort(key=lambda f: f["uploaded_at"], reverse=True)
        recent_files = recent_files[:max_files]

        return {
            "success": True,
            "files": recent_files,
            "count": len(recent_files),
            "uploads_dir": str(uploads_dir),
            "project_uploads_dir": str(Path(project_dir) / "uploads") if project_dir else None,
            "hours_searched": hours,
            "message": f"Found {len(recent_files)} file(s) uploaded in the last {hours} hours."
        }

    except Exception as e:
        return {"success": False, "error": str(e), "files": []}


# ── Large file tools (bounded output, streaming reads) ─────────────────────────

# ── HARD CEILING for max_chars in all large file tools ──
# This prevents the agent from passing a huge max_chars value that would
# overflow the LLM context window. 10,000 chars ≈ 3,300 tokens — safe.
# The agent can call tools multiple times to paginate through large files.
MAX_CHARS_CEILING = 50_000


def _clamp_max_chars(max_chars: int) -> int:
    """Enforce hard ceiling on max_chars regardless of what the caller passes."""
    return min(max(max_chars, 100), MAX_CHARS_CEILING)


@mcp.tool
def read_file_lines(path: str, start_line: int = 1, end_line: int = 200, max_chars: int = 20000) -> dict:
    """Read a specific range of lines from any file, no matter how large. Use this INSTEAD of read_text_file for files over 50KB. Call this when you need a specific section of a large file (e.g. lines 100-200). Lines are 1-indexed. Memory-efficient — reads line-by-line. Output is capped to prevent context overflow.

        Args:
            path: Absolute path to the file. Must start with /.
            start_line: First line to read, 1-indexed (default: 1).
            end_line: Last line to read, inclusive (default: 200).
            max_chars: Maximum characters to return (default: 20000, hard ceiling: 50000).
        Returns dict with the line content, total lines in file, and truncation status."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file or does not exist"}
        if start_line < 1:
            start_line = 1
        if end_line < 1:
            end_line = 1
        if end_line < start_line:
            return {"error": f"end_line ({end_line}) must be >= start_line ({start_line})"}

        # Enforce hard ceiling on max_chars
        max_chars = _clamp_max_chars(max_chars)

        collected = []
        total_chars = 0
        truncated = False
        line_num = 0

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_num += 1
                if line_num < start_line:
                    continue
                if line_num > end_line:
                    break
                line_text = line.rstrip("\n")
                if total_chars + len(line_text) + 10 > max_chars:
                    truncated = True
                    break
                collected.append(f"{line_num}: {line_text}")
                total_chars += len(line_text) + 10  # account for line number prefix

        return {
            "success": True,
            "path": str(p),
            "start_line": start_line,
            "end_line": min(end_line, line_num),
            "lines_returned": len(collected),
            "truncated_by_max_chars": truncated,
            "max_chars_used": max_chars,
            "content": "\n".join(collected)
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def read_file_head(path: str, num_lines: int = 100, max_chars: int = 20000) -> dict:
    """Read the first N lines of any file, no matter how large. Use this INSTEAD of read_text_file for files over 50KB. Call this to check file format, headers, column names, or structure. Memory-efficient. Output is capped to prevent context overflow.

        Args:
            path: Absolute path to the file. Must start with /.
            num_lines: Number of lines from the start (default: 100).
            max_chars: Maximum characters to return (default: 20000, hard ceiling: 50000).
        Returns dict with the head content, total lines in file, and truncation status."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file or does not exist"}
        if num_lines < 1:
            num_lines = 1

        # Enforce hard ceiling on max_chars
        max_chars = _clamp_max_chars(max_chars)

        collected = []
        total_chars = 0
        truncated = False
        total_lines_in_file = 0

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines_in_file += 1
                if len(collected) >= num_lines:
                    # Keep counting total lines for metadata
                    continue
                line_text = line.rstrip("\n")
                if total_chars + len(line_text) + 10 > max_chars:
                    truncated = True
                    # Stop collecting but keep counting
                    continue
                collected.append(f"{total_lines_in_file}: {line_text}")
                total_chars += len(line_text) + 10

        file_size = p.stat().st_size

        return {
            "success": True,
            "path": str(p),
            "file_size_bytes": file_size,
            "file_size_human": _human_size(file_size),
            "total_lines_in_file": total_lines_in_file,
            "lines_returned": len(collected),
            "truncated_by_max_chars": truncated,
            "max_chars_used": max_chars,
            "content": "\n".join(collected)
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def read_file_tail(path: str, num_lines: int = 100, max_chars: int = 20000) -> dict:
    """Read the last N lines of any file, no matter how large. Use this INSTEAD of read_text_file for files over 50KB. Call this to check the end of log files, recent output, or job completion status. Memory-efficient — uses a sliding window. Output is capped to prevent context overflow.

        Args:
            path: Absolute path to the file. Must start with /.
            num_lines: Number of lines from the end (default: 100).
            max_chars: Maximum characters to return (default: 20000, hard ceiling: 50000).
        Returns dict with the tail content, total lines in file, and truncation status."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file or does not exist"}
        if num_lines < 1:
            num_lines = 1

        # Enforce hard ceiling on max_chars
        max_chars = _clamp_max_chars(max_chars)

        tail_buffer = deque(maxlen=num_lines)
        total_lines_in_file = 0

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines_in_file += 1
                tail_buffer.append((total_lines_in_file, line.rstrip("\n")))

        # Build output with char cap
        collected = []
        total_chars = 0
        truncated = False
        for line_num, line_text in tail_buffer:
            if total_chars + len(line_text) + 10 > max_chars:
                truncated = True
                break
            collected.append(f"{line_num}: {line_text}")
            total_chars += len(line_text) + 10

        file_size = p.stat().st_size

        return {
            "success": True,
            "path": str(p),
            "file_size_bytes": file_size,
            "file_size_human": _human_size(file_size),
            "total_lines_in_file": total_lines_in_file,
            "lines_returned": len(collected),
            "truncated_by_max_chars": truncated,
            "max_chars_used": max_chars,
            "content": "\n".join(collected)
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def grep_file(
    path: str,
    pattern: str,
    max_matches: int = 20,
    context_lines: int = 0,
    max_chars_per_match: int = 500,
    case_sensitive: bool = True
) -> dict:
    """Search for a regex pattern in any file, no matter how large. Use this INSTEAD of read_text_file when searching for specific content in files over 50KB. Call this to find specific content, error messages, or patterns. Returns matching lines with optional context. Memory-efficient — reads line-by-line. For 2+ searches, use batch with type='grep' instead — one call handles all greps.

        Args:
            path: Absolute path to the file. Must start with /.
            pattern: Python regex pattern to search for.
            max_matches: Maximum matches to return (default: 20, hard ceiling: 50).
            context_lines: Lines of context before and after each match (default: 0).
            max_chars_per_match: Max characters per match line (default: 500).
            case_sensitive: Whether search is case-sensitive (default: true).
        Returns dict with matching lines, their line numbers, total match count, and context."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file or does not exist"}

        flags = 0 if case_sensitive else re.IGNORECASE
        # Guard against regex DoS: reject overly complex patterns
        if len(pattern) > 1000:
            return {"error": "Regex pattern too long (max 1000 chars)"}
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {str(e)}"}

        # Enforce hard ceiling on max_chars_per_match
        max_chars_per_match = min(max_chars_per_match, MAX_CHARS_CEILING)
        # Enforce hard ceiling on max_matches to prevent huge output
        max_matches = min(max_matches, 50)

        matches = []
        total_match_count = 0
        # For context lines, keep a rolling buffer of previous lines
        before_buffer = deque(maxlen=max(context_lines, 0))
        # Track lines that need "after" context
        pending_after = 0
        after_lines = []
        current_match_entry = None

        line_num = 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_num += 1
                line_text = line.rstrip("\n")

                # Collect "after" context for previous match
                if pending_after > 0 and current_match_entry is not None:
                    current_match_entry["after_context"].append(
                        f"{line_num}: {line_text[:max_chars_per_match]}"
                    )
                    pending_after -= 1

                if compiled.search(line_text):
                    total_match_count += 1

                    if len(matches) < max_matches:
                        # Finalize previous match's after-context
                        current_match_entry = {
                            "line_num": line_num,
                            "text": line_text[:max_chars_per_match],
                            "before_context": [f"{bn}: {bt[:max_chars_per_match]}" for bn, bt in before_buffer],
                            "after_context": []
                        }
                        matches.append(current_match_entry)
                        pending_after = context_lines
                    else:
                        current_match_entry = None
                        pending_after = 0

                # Add to before-buffer for next potential match
                before_buffer.append((line_num, line_text))

        # Format output
        formatted_matches = []
        for m in matches:
            entry = {"line": m["line_num"], "text": m["text"]}
            if context_lines > 0:
                entry["before"] = m["before_context"]
                entry["after"] = m["after_context"]
            formatted_matches.append(entry)

        return {
            "success": True,
            "path": str(p),
            "pattern": pattern,
            "total_matches": total_match_count,
            "matches_returned": len(formatted_matches),
            "truncated": total_match_count > max_matches,
            "matches": formatted_matches
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def count_pattern(path: str, pattern: str, case_sensitive: bool = True) -> dict:
    """Count how many lines match a regex pattern in a file without returning any content. Use this INSTEAD of read_text_file when you only need to know how many matches exist. Call this to understand file composition before reading specific sections — e.g. count error lines, count FASTA headers. Returns ONLY the count, zero file content. Memory-efficient.

        Args:
            path: Absolute path to the file. Must start with /.
            pattern: Python regex pattern to count.
            case_sensitive: Whether search is case-sensitive (default: true).
        Returns dict with match_count, total_lines, and match_percentage."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file or does not exist"}

        flags = 0 if case_sensitive else re.IGNORECASE
        # Guard against regex DoS: reject overly complex patterns
        if len(pattern) > 1000:
            return {"error": "Regex pattern too long (max 1000 chars)"}
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {str(e)}"}

        match_count = 0
        total_lines = 0

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                if compiled.search(line):
                    match_count += 1

        return {
            "success": True,
            "path": str(p),
            "pattern": pattern,
            "match_count": match_count,
            "total_lines": total_lines,
            "match_percentage": round(100.0 * match_count / total_lines, 2) if total_lines > 0 else 0.0
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def get_file_overview(path: str, num_sample_lines: int = 10, max_chars: int = 3000) -> dict:
    """Get a quick structural overview of any file without reading the whole thing. Call this as the FIRST step when exploring an unknown or large file. Use this INSTEAD of read_text_file when file size is unknown or exceeds 50KB. Returns file size, line count, first 5 lines, last 5 lines, and sampled lines from the middle. Works on files of any size (1KB to 10GB+). Memory-efficient.

        Args:
            path: Absolute path to the file. Must start with /.
            num_sample_lines: Number of evenly-spaced sample lines from the middle (default: 10).
            max_chars: Maximum total characters to return (default: 3000).
        Returns dict with file metadata, line statistics, and sampled content."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"error": "Not a file or does not exist"}

        # Enforce hard ceiling on max_chars
        max_chars = _clamp_max_chars(max_chars)

        file_size = p.stat().st_size
        modified = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        mime_type, _ = mimetypes.guess_type(str(p))

        # First pass: count lines and collect first/last lines
        total_lines = 0
        first_lines = []  # first 5
        tail_buffer = deque(maxlen=5)  # last 5
        line_lengths = []  # track lengths for stats (sampled)
        total_length_sum = 0
        min_length = float("inf")
        max_length = 0

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                line_text = line.rstrip("\n")
                line_len = len(line_text)
                total_length_sum += line_len
                if line_len < min_length:
                    min_length = line_len
                if line_len > max_length:
                    max_length = line_len

                if total_lines <= 5:
                    first_lines.append(f"{total_lines}: {line_text}")
                tail_buffer.append((total_lines, line_text))

        if total_lines == 0:
            return {
                "success": True,
                "path": str(p),
                "file_size_bytes": file_size,
                "file_size_human": _human_size(file_size),
                "total_lines": 0,
                "message": "File is empty"
            }

        last_lines = [f"{ln}: {lt}" for ln, lt in tail_buffer]
        avg_length = round(total_length_sum / total_lines, 1) if total_lines > 0 else 0

        # Second pass: collect evenly-sampled lines from the middle
        sampled_lines = []
        if num_sample_lines > 0 and total_lines > 10:
            # Calculate which line numbers to sample (skip first 5 and last 5)
            sample_start = 6
            sample_end = total_lines - 5
            if sample_end > sample_start:
                step = max(1, (sample_end - sample_start) // num_sample_lines)
                target_lines = set(range(sample_start, sample_end, step))
                # Limit to num_sample_lines
                target_lines = set(list(target_lines)[:num_sample_lines])

                if target_lines:
                    line_num = 0
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line_num += 1
                            if line_num in target_lines:
                                line_text = line.rstrip("\n")
                                sampled_lines.append(f"{line_num}: {line_text}")
                                target_lines.discard(line_num)
                                if not target_lines:
                                    break

        # Truncate all sections to fit within max_chars
        def truncate_lines(lines, budget):
            result = []
            used = 0
            for line in lines:
                if used + len(line) + 1 > budget:
                    result.append("... (truncated)")
                    break
                result.append(line)
                used += len(line) + 1
            return result

        char_budget_per_section = max_chars // 3
        first_lines = truncate_lines(first_lines, char_budget_per_section)
        last_lines = truncate_lines(last_lines, char_budget_per_section)
        sampled_lines = truncate_lines(sampled_lines, char_budget_per_section)

        return {
            "success": True,
            "path": str(p),
            "file_size_bytes": file_size,
            "file_size_human": _human_size(file_size),
            "mime_type": mime_type or "unknown",
            "last_modified": modified,
            "total_lines": total_lines,
            "line_length_stats": {
                "avg": avg_length,
                "min": min_length if min_length != float("inf") else 0,
                "max": max_length
            },
            "first_lines": first_lines,
            "last_lines": last_lines,
            "sampled_lines": sampled_lines,
            "num_samples": len(sampled_lines),
            "max_chars_used": max_chars
        }
    except Exception as e:
        return {"error": str(e)}



# ── Image file management ──────────────────────────────────────────────────
# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.webp', '.tiff', '.tif'}

@mcp.tool
def save_image(source_path: str, context_name: str = '', work_dir: str = None) -> dict:
    """Copy an image file to an organized location under work_dir/images/. Call this after generating plots, structure renderings, or visualizations to save them persistently. DO NOT call this for non-image files — only image formats (.png, .jpg, .svg, .pdf, .tiff, .bmp, .gif, .webp) are accepted.

        Args:
            source_path: Absolute path to the image file to save. Must start with /.
            context_name: Optional topic name for organizing (e.g. 'protein_analysis'). Uses date if not provided.
            work_dir: User's work directory. If not provided, loads from user settings.
        Returns dict with the saved image path and metadata."""
    try:
        # Resolve work_dir
        if not work_dir:
            uid = os.getuid()
            username = pwd.getpwuid(uid).pw_name
            settings_path = _get_user_settings_path(username)
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                work_dir = settings.get('work_dir')
            if not work_dir:
                return {'success': False, 'error': 'No work_dir provided and none found in user settings.'}

        src = Path(source_path).resolve()
        if not src.exists() or not src.is_file():
            return {'success': False, 'error': f'Source file does not exist: {source_path}'}

        # Check it is an image
        ext = src.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            return {'success': False, 'error': f'Not a recognized image file ({ext}). Supported: {sorted(IMAGE_EXTENSIONS)}'}

        # Build destination directory
        if not context_name:
            context_name = datetime.datetime.now().strftime('%Y-%m-%d')
        # Sanitize context_name — only allow alphanumeric, dash, underscore, dot
        safe_context = re.sub(r'[^\w.\-]', '_', context_name)

        images_dir = Path(work_dir) / 'images' / safe_context
        images_dir.mkdir(parents=True, exist_ok=True)

        dest = images_dir / src.name
        # If file already exists, add timestamp suffix
        if dest.exists():
            stem = src.stem
            timestamp = datetime.datetime.now().strftime('%H%M%S')
            dest = images_dir / f'{stem}_{timestamp}{ext}'

        shutil.copy2(str(src), str(dest))

        return {
            'success': True,
            'saved_path': str(dest),
            'source_path': str(src),
            'context': safe_context,
            'size_bytes': dest.stat().st_size,
            'message': f'Image saved to {dest}',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@mcp.tool
def list_saved_images(work_dir: str = None, context_name: str = '') -> dict:
    """List previously saved images from work_dir/images/. Call this when the user asks to see their saved visualizations or plots.

        Args:
            work_dir: User's work directory. If not provided, loads from user settings.
            context_name: Optional — filter to a specific context subdirectory.
        Returns dict with list of image files, their paths, sizes, and contexts."""
    try:
        # Resolve work_dir
        if not work_dir:
            uid = os.getuid()
            username = pwd.getpwuid(uid).pw_name
            settings_path = _get_user_settings_path(username)
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                work_dir = settings.get('work_dir')
            if not work_dir:
                return {'success': False, 'error': 'No work_dir provided and none found in user settings.'}

        images_base = Path(work_dir) / 'images'
        if not images_base.exists():
            return {'success': True, 'images': [], 'message': 'No images directory found. No images have been saved yet.'}

        if context_name:
            safe_context = re.sub(r'[^\w.\-]', '_', context_name)
            search_dir = images_base / safe_context
            if not search_dir.exists():
                return {'success': True, 'images': [], 'message': f'No images found for context: {context_name}'}
            dirs_to_search = [search_dir]
        else:
            dirs_to_search = [d for d in images_base.iterdir() if d.is_dir()]
            if not dirs_to_search:
                return {'success': True, 'images': [], 'message': 'Images directory is empty.'}

        images = []
        for d in sorted(dirs_to_search):
            ctx = d.name
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append({
                        'path': str(f),
                        'name': f.name,
                        'context': ctx,
                        'size_bytes': f.stat().st_size,
                        'modified': datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })

        return {
            'success': True,
            'images': images,
            'total_count': len(images),
            'contexts': sorted(set(img['context'] for img in images)),
            'message': f'Found {len(images)} saved image(s).',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}



# ── HPC User/Group Directory API config ─────────────────────────────────
HPC_USER_API_URL = os.environ.get("HPC_USER_API_URL", "https://your-hpc-api-host/users")
HPC_GROUP_API_URL = os.environ.get("HPC_GROUP_API_URL", "https://your-hpc-api-host/groups")
HPC_DISPLAYNAMES_API_URL = os.environ.get("HPC_DISPLAYNAMES_API_URL", "https://your-hpc-api-host/displayNames")
HPC_INVESTIGATORS_API_URL = os.environ.get("HPC_INVESTIGATORS_API_URL", "https://your-hpc-api-host/investigators")
_HPC_API_TIMEOUT = 10  # seconds

# Module-level cache for investigators data (fetched once per session)
_investigators_cache = None


def _fetch_all_investigators() -> dict:
    """Fetch all investigators and return a dict keyed by username.

    Returns dict mapping username -> {"program": ..., "department": ...}.
    Uses module-level cache so the API is called at most once per session.
    Returns empty dict on failure (graceful degradation).
    """
    global _investigators_cache
    if _investigators_cache is not None:
        return _investigators_cache
    try:
        data = _fetch_hpc_api(HPC_INVESTIGATORS_API_URL)
        _investigators_cache = {
            entry["username"]: {
                "program": entry.get("program", ""),
                "department": entry.get("department", ""),
            }
            for entry in data
            if entry.get("username")
        }
    except Exception:
        _investigators_cache = {}
    return _investigators_cache


def _fetch_hpc_api(url: str) -> list:
    """Fetch JSON data from the HPC MongoDB API.

    Uses SSL context that doesn't verify certs (internal service).
    Returns parsed JSON list or raises on failure.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=_HPC_API_TIMEOUT, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_display_names(usernames: list) -> dict:
    """Batch-resolve usernames to display names via the displayNames API.

    Primary method for resolving username → full name. Accepts a list of
    usernames and returns a dict mapping each username to its display name.
    Falls back gracefully — callers should handle empty dict on failure.

    Args:
        usernames: List of username strings, e.g. ["jsmith", "jdoe"].

    Returns:
        Dict mapping username → display name string. Missing users map to "".
    """
    if not usernames:
        return {}
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    payload = json.dumps([{"username": u} for u in usernames]).encode("utf-8")
    req = urllib.request.Request(
        HPC_DISPLAYNAMES_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HPC_API_TIMEOUT, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # Handle both list-of-dicts and single-dict responses
    if isinstance(data, dict):
        data = [data]
    return {entry.get("username", ""): entry.get("displayName", "") for entry in data if entry.get("username")}


def _run_adquery(args: list) -> str:
    """Run an adquery command and return stdout.

    Args:
        args: Command arguments, e.g. ["user", "jsmith", "-A"].

    Returns:
        stdout string.

    Raises:
        RuntimeError: If the command fails.
    """
    result = subprocess.run(
        ["adquery"] + args,
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"adquery {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _lookup_single_user(uname: str) -> dict:
    """Look up a single user across all directory sources."""
    try:
        full_name = ""
        try:
            names = _fetch_display_names([uname])
            full_name = names.get(uname, "")
        except Exception:
            pass

        investigators = _fetch_all_investigators()
        investigator_info = investigators.get(uname, {})

        api_primary_group = ""
        try:
            users = _fetch_hpc_api(HPC_USER_API_URL)
            for u in users:
                if u.get("username") == uname:
                    api_primary_group = u.get("primary_group", "")
                    break
        except Exception:
            pass

        parsed = {"username": uname}
        try:
            raw = _run_adquery(["user", uname, "-A"])
            parsed = parse_adquery_user(raw) or {"username": uname}
        except Exception:
            pass

        if not parsed.get("username"):
            parsed["username"] = uname

        result = structure_hpc_user_lookup(parsed, api_primary_group, investigator_info)

        if full_name:
            result["full_name"] = full_name

        if not result.get("full_name") and not investigator_info and not api_primary_group and not parsed.get("uid"):
            return {"success": False, "error": f"User '{uname}' not found in any directory source."}

        return result

    except Exception as e:
        return {"success": False, "error": f"Failed to look up user '{uname}': {str(e)}"}


def _lookup_bulk_users(usernames_list: list) -> dict:
    """Bulk lookup: fetch department/program/name for multiple users in one call."""
    try:
        # Batch fetch display names (single API call for all)
        display_names = {}
        try:
            display_names = _fetch_display_names(usernames_list)
        except Exception:
            pass

        # Investigators (cached, single fetch)
        investigators = _fetch_all_investigators()

        # HPC users API (cached after first call in session)
        hpc_users_map = {}
        try:
            users = _fetch_hpc_api(HPC_USER_API_URL)
            hpc_users_map = {u["username"]: u.get("primary_group", "") for u in users}
        except Exception:
            pass

        results = []
        for uname in usernames_list:
            entry = {"username": uname}
            if display_names.get(uname):
                entry["full_name"] = display_names[uname]
            inv = investigators.get(uname, {})
            if inv.get("department"):
                entry["department"] = inv["department"]
            if inv.get("program"):
                entry["program"] = inv["program"]
            if hpc_users_map.get(uname):
                entry["primary_group"] = hpc_users_map[uname]
            results.append(entry)

        return {"success": True, "users": results, "total": len(results)}

    except Exception as e:
        return {"success": False, "error": f"Bulk user lookup failed: {str(e)}"}


def _lookup_single_group(gname: str) -> dict:
    """Look up a single group's members and details."""
    try:
        raw = _run_adquery(["group", gname, "-A"])
        parsed = parse_adquery_group(raw)

        if not parsed.get("group_name"):
            return {"success": False, "error": f"Group '{gname}' not found in Active Directory."}

        api_investigator = ""
        try:
            groups = _fetch_hpc_api(HPC_GROUP_API_URL)
            for g in groups:
                if g.get("group_name") == gname:
                    api_investigator = g.get("investigator", "")
                    break
        except Exception:
            pass

        return structure_hpc_group_lookup(parsed, api_investigator)

    except RuntimeError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to look up group '{gname}': {str(e)}"}


def _lookup_bulk_groups(groups_list: list) -> dict:
    """Bulk lookup: fetch details for multiple groups in one call."""
    try:
        # Get API data once
        api_groups_map = {}
        try:
            api_groups = _fetch_hpc_api(HPC_GROUP_API_URL)
            api_groups_map = {g["group_name"]: g.get("investigator", "") for g in api_groups}
        except Exception:
            pass

        results = []
        for gname in groups_list:
            entry = {"group_name": gname}
            try:
                raw = _run_adquery(["group", gname, "-A"])
                parsed = parse_adquery_group(raw)
                if parsed.get("group_name"):
                    entry["members"] = parsed.get("members", [])
                    entry["gid"] = parsed.get("gid", "")
                else:
                    entry["error"] = "Not found in Active Directory"
            except RuntimeError as e:
                entry["error"] = str(e)
            except Exception as e:
                entry["error"] = f"Lookup failed: {str(e)}"

            if api_groups_map.get(gname):
                entry["investigator"] = api_groups_map[gname]
            results.append(entry)

        return {"success": True, "groups": results, "total": len(results)}

    except Exception as e:
        return {"success": False, "error": f"Bulk group lookup failed: {str(e)}"}


@mcp.tool
def hpc_directory(username: str = "", query: str = "", group_name: str = "", list_groups: bool = False) -> dict:
    """Look up users, groups, or organizational info in the HPC directory. Use exactly one mode:
    - username: Get full details for one or more users/Slurm PI accounts. Comma-separated for bulk (e.g. 'mauraf,widmana,choderaj').
    - query: Search users by partial name (e.g. 'chen' matches cheny1, chenge1). Returns department/program for matches.
    - group_name: Get members/details for one or more groups. Comma-separated for bulk (e.g. 'grp_hpc_mauraf,grp_hpc_choderaj').
    - list_groups: Set True to list all HPC groups with their PIs.
    For Slurm accounts (PI names from squeue output), use username=.

    Args:
        username: One or more usernames/Slurm accounts, comma-separated for bulk lookup (e.g. 'mauraf,widmana,razavip').
        query: Substring to search for in usernames (case-insensitive). Example: 'jsmi' matches 'jsmith'.
        group_name: One or more group names, comma-separated for bulk lookup (e.g. 'grp_hpc_users,grp_hpc_admins').
        list_groups: Set True to list all HPC groups with their investigators.
    Returns dict with results depending on mode used."""
    modes = sum([bool(username), bool(query), bool(group_name), list_groups])
    if modes == 0:
        return {"success": False, "error": "Provide one of: username, query, group_name, or list_groups=True"}
    if modes > 1:
        return {"success": False, "error": "Use only one mode at a time: username OR query OR group_name OR list_groups"}

    # ── Mode: username (exact user/PI lookup — supports comma-separated bulk) ──
    if username:
        usernames_list = [u.strip() for u in username.split(",") if u.strip()]

        if len(usernames_list) == 1:
            return _lookup_single_user(usernames_list[0])
        else:
            return _lookup_bulk_users(usernames_list)

    # ── Mode: query (pattern search with investigators fallback) ──
    elif query:
        try:
            users = _fetch_hpc_api(HPC_USER_API_URL)
            result = filter_hpc_users(users, query=query)

            if result.get("users"):
                usernames = [u["username"] for u in result["users"]]
                try:
                    names = _fetch_display_names(usernames)
                    for u in result["users"]:
                        u["full_name"] = names.get(u["username"], "")
                except Exception:
                    pass

                investigators = _fetch_all_investigators()
                for u in result["users"]:
                    inv = investigators.get(u["username"], {})
                    if inv.get("program"):
                        u["program"] = inv["program"]
                    if inv.get("department"):
                        u["department"] = inv["department"]
            else:
                # Fallback: search investigators API for PI accounts not in HPC users API
                investigators = _fetch_all_investigators()
                query_lower = query.lower()
                matches = [
                    {"username": uname, "department": info.get("department", ""), "program": info.get("program", "")}
                    for uname, info in investigators.items()
                    if query_lower in uname.lower()
                ]
                if matches:
                    try:
                        names = _fetch_display_names([m["username"] for m in matches])
                        for m in matches:
                            m["full_name"] = names.get(m["username"], "")
                    except Exception:
                        pass
                    result = {"success": True, "users": matches, "total": len(matches), "query": query, "source": "investigators"}

            return result
        except Exception as e:
            return {"success": False, "error": f"Failed to search HPC directory: {str(e)}"}

    # ── Mode: group_name (group details + members — supports comma-separated bulk) ──
    elif group_name:
        groups_list = [g.strip() for g in group_name.split(",") if g.strip()]

        if len(groups_list) == 1:
            return _lookup_single_group(groups_list[0])
        else:
            return _lookup_bulk_groups(groups_list)

    # ── Mode: list_groups ──
    elif list_groups:
        try:
            groups = _fetch_hpc_api(HPC_GROUP_API_URL)
            return {
                "success": True,
                "groups": groups,
                "total": len(groups),
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to query HPC group API: {str(e)}"}


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# ── Session Log Tools ──────────────────────────────────────────────────
# Two sources: OnDemand output.log (debug) and JSONL conversation logs.

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.session_log import (
    list_session_logs as _list_conversation_logs,
    search_session_logs as _search_conversation_logs,
)

# These tools give the agent self-awareness of its own session logs.
# OnDemand stores session output in:
#   /home/<user>/ondemand/data/sys/dashboard/batch_connect/<app>/output/<session_id>/output.log
# The agent can read its own log to self-diagnose errors, and browse
# previous session logs for context recovery after killed sessions.

# Base path for OnDemand batch connect session data
_ONDEMAND_BC_BASE = "ondemand/data/sys/dashboard/batch_connect"


def _get_ondemand_base(username: str) -> Path:
    """Get the OnDemand batch_connect base directory for a user."""
    return Path(f"/home/{username}") / _ONDEMAND_BC_BASE


def _find_app_sessions(username: str, app_name: str = None) -> list:
    """Find OnDemand session directories for a specific app or all IrisAI apps.

    Returns list of dicts sorted by modification time (newest first).
    Each dict has: session_id, app_path, output_log, modified, size_bytes.
    """
    bc_base = _get_ondemand_base(username)
    if not bc_base.exists():
        return []

    # If app_name given, search that specific app path
    # Otherwise, search known IrisAI app paths
    if app_name:
        app_paths = [app_name]
    else:
        # Auto-detect IrisAI apps by looking for common app names
        app_paths = []
        for dev_or_sys in ["dev", "sys", "usr"]:
            dev_dir = bc_base / dev_or_sys
            if dev_dir.exists():
                try:
                    for entry in dev_dir.iterdir():
                        if entry.is_dir():
                            name_lower = entry.name.lower()
                            if "iris" in name_lower or "irisai" in name_lower:
                                app_paths.append(f"{dev_or_sys}/{entry.name}")
                except PermissionError:
                    pass

    sessions = []
    for app_path in app_paths:
        output_dir = bc_base / app_path / "output"
        if not output_dir.exists():
            continue
        try:
            for session_dir in output_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                log_file = session_dir / "output.log"
                if log_file.exists():
                    try:
                        st = log_file.stat()
                        sessions.append({
                            "session_id": session_dir.name,
                            "app_path": app_path,
                            "output_log": str(log_file),
                            "modified": datetime.datetime.fromtimestamp(
                                st.st_mtime
                            ).isoformat(),
                            "size_bytes": st.st_size,
                            "size_human": _human_size(st.st_size),
                        })
                    except (OSError, PermissionError):
                        pass
        except PermissionError:
            pass

    # Sort by modification time, newest first
    sessions.sort(key=lambda s: s["modified"], reverse=True)
    return sessions


@mcp.tool
def read_session_log(
    section: str = "tail",
    num_lines: int = 100,
    search_pattern: str = "",
    session_id: str = "",
    app_name: str = "",
    source: str = "debug",
) -> dict:
    """Read session logs — either infrastructure debug logs (output.log) or past conversation content (JSONL).

    Use read_memory() FIRST for project context. Only use this for raw session history not
    captured in project memory (e.g., conversations without a project, or specific past exchanges).

    Args:
        source: 'debug' (default) — reads OnDemand output.log (stack traces, MCP errors, lifecycle).
            'conversation' — reads/searches JSONL conversation logs (what user and assistant said).
        section: Which part to read — 'tail' (last N entries, default), 'head'
            (first N), or 'search' (grep for a pattern).
        num_lines: Number of entries to return for head/tail (default: 100).
        search_pattern: Pattern to search for when section='search'.
            For debug: regex (finds [AUTO-COMPRESS], [SKILL_SELECT], etc.).
            For conversation: plain text substring match.
        session_id: Specific session to read. For debug: OnDemand UUID from list_session_logs.
            For conversation: session ID from list_session_logs(source='conversation').
            Leave EMPTY to read the most recent session.
        app_name: OnDemand app path (only used for source='debug'). Do NOT pass Chainlit
            or conversation session IDs here — they are different.
    Returns dict with log content, session info, and metadata."""
    try:
        username = os.environ.get("USER", "")
        if not username:
            return {"success": False, "error": "Cannot determine username from USER env var"}

        # ── Conversation source: search/read JSONL session logs ──
        if source == "conversation":
            if section == "search":
                if not search_pattern:
                    return {"success": False, "error": "search_pattern required for conversation search"}
                matches = _search_conversation_logs(username, search_pattern, max_results=num_lines)
                return {
                    "success": True,
                    "source": "conversation",
                    "section": "search",
                    "messages": matches,
                    "total_matches": len(matches),
                    "hint": "Each result has: role, content, ts, session_id",
                }
            elif section == "tail":
                all_logs = _list_conversation_logs(username)
                if not all_logs:
                    return {"success": False, "error": "No conversation logs found."}
                if session_id:
                    target = next((l for l in all_logs if l["session_id"] == session_id), None)
                    if not target:
                        return {
                            "success": False,
                            "error": f"Session '{session_id}' not found",
                            "available": [l["session_id"] for l in all_logs[:10]],
                        }
                else:
                    target = all_logs[0]
                messages = []
                try:
                    with open(target["path"], "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if entry.get("type") == "message":
                                messages.append(entry)
                except OSError as e:
                    return {"success": False, "error": f"Cannot read session log: {e}"}
                tail_msgs = messages[-num_lines:]
                return {
                    "success": True,
                    "source": "conversation",
                    "section": "tail",
                    "session_id": target["session_id"],
                    "messages": tail_msgs,
                    "total_messages": len(messages),
                    "showing": len(tail_msgs),
                }
            else:
                return {"success": False, "error": f"section='{section}' not supported for conversation source. Use 'tail' or 'search'."}

        # ── Debug source (default): read OnDemand output.log ──
        # Check for mounted session directories (set by template/script.sh.erb)
        current_session_dir = os.environ.get("CURRENT_SESSION_DIR", "")
        all_sessions_dir = os.environ.get("ALL_SESSIONS_DIR", "")

        # Find the target session
        _original_session_id = session_id
        _fallback_available_sessions = []
        if session_id:
            # Look for specific session
            found = False
            log_path = None

            # First try ALL_SESSIONS_DIR mount (direct access to all sessions)
            if all_sessions_dir:
                candidate = Path(all_sessions_dir) / session_id / "output.log"
                if candidate.exists():
                    log_path = candidate
                    found = True
                    app_name = app_name or "mounted_sessions"

            # Fall back to OnDemand batch_connect directory search
            if not found:
                bc_base = _get_ondemand_base(username)
                # Search across app paths
                if app_name and app_name != "mounted_sessions":
                    log_path = bc_base / app_name / "output" / session_id / "output.log"
                    found = log_path.exists()
                else:
                    # Search all apps for this session ID
                    for dev_or_sys in ["dev", "sys", "usr"]:
                        dev_dir = bc_base / dev_or_sys
                        if not dev_dir.exists():
                            continue
                        try:
                            for app_dir in dev_dir.iterdir():
                                candidate = app_dir / "output" / session_id / "output.log"
                                if candidate.exists():
                                    log_path = candidate
                                    app_name = f"{dev_or_sys}/{app_dir.name}"
                                    found = True
                                    break
                        except PermissionError:
                            continue
                        if found:
                            break
            if not found or log_path is None:
                # Broader search: list all available sessions, try partial match or most recent
                _fallback_available_sessions = []
                for _dos in ["dev", "sys", "usr"]:
                    _ddir = bc_base / _dos
                    if not _ddir.exists():
                        continue
                    try:
                        for _adir in _ddir.iterdir():
                            _odir = _adir / "output"
                            if not _odir.exists():
                                continue
                            for _sdir in sorted(_odir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                                if _sdir.is_dir() and (_sdir / "output.log").exists():
                                    _fallback_available_sessions.append({
                                        "session_id": _sdir.name,
                                        "app": f"{_dos}/{_adir.name}",
                                        "path": str(_sdir / "output.log"),
                                    })
                    except (PermissionError, OSError):
                        continue

                # Try partial match (first 8 chars)
                if session_id and len(session_id) >= 8:
                    _prefix = session_id[:8].lower()
                    for _s in _fallback_available_sessions:
                        if _s["session_id"].lower().startswith(_prefix):
                            log_path = Path(_s["path"])
                            session_id = _s["session_id"]
                            found = True
                            app_name = _s["app"]
                            break

                # Fall back to most recent session
                if not found and _fallback_available_sessions:
                    _most_recent = _fallback_available_sessions[0]
                    log_path = Path(_most_recent["path"])
                    session_id = _most_recent["session_id"]
                    found = True
                    app_name = _most_recent["app"]

                if not found:
                    return {
                        "success": False,
                        "error": f"Session '{session_id}' not found. No OnDemand output directories found.",
                        "searched": str(bc_base),
                    }
        else:
            # Find most recent session
            # Primary: use CURRENT_SESSION_DIR mount (the running session's workspace)
            if current_session_dir:
                candidate_log = Path(current_session_dir) / "output.log"
                if candidate_log.exists():
                    log_path = candidate_log
                    # Extract session_id from the directory name
                    # CURRENT_SESSION_DIR points to /current_session which is
                    # mounted from the actual session dir
                    # We need to get the real session ID - check ALL_SESSIONS_DIR
                    session_id = "current"
                    app_name = app_name or "current_session"
                    # Try to resolve actual session_id from the real path
                    try:
                        real_path = candidate_log.resolve()
                        session_id = real_path.parent.name
                    except (OSError, RuntimeError):
                        pass
                else:
                    # CURRENT_SESSION_DIR set but no output.log yet, fall back
                    log_path = None

            if log_path is None:
                # Fall back to _find_app_sessions discovery
                sessions = _find_app_sessions(username, app_name or None)
                if not sessions:
                    return {
                        "success": False,
                        "error": "No IrisAI session logs found. Check that OnDemand "
                                 "session data exists at ~/ondemand/data/sys/dashboard/batch_connect/",
                    }
                latest = sessions[0]
                log_path = Path(latest["output_log"])
                session_id = latest["session_id"]
                app_name = latest["app_path"]

        # Read the log file
        log_path_str = str(log_path)
        file_size = log_path.stat().st_size

        if section == "tail":
            # Read last N lines efficiently
            lines = deque(maxlen=num_lines)
            with open(log_path_str, "r", errors="replace") as f:
                for line in f:
                    lines.append(line)
            content = "".join(lines)
            total_lines = sum(1 for _ in open(log_path_str, "r", errors="replace"))

        elif section == "head":
            # Read first N lines
            content_lines = []
            with open(log_path_str, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= num_lines:
                        break
                    content_lines.append(line)
            content = "".join(content_lines)
            total_lines = sum(1 for _ in open(log_path_str, "r", errors="replace"))

        elif section == "search":
            if not search_pattern:
                return {"success": False, "error": "search_pattern is required when section='search'"}
            try:
                pattern = re.compile(search_pattern, re.IGNORECASE)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex pattern: {e}"}

            matches = []
            total_lines = 0
            with open(log_path_str, "r", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    total_lines += 1
                    if pattern.search(line):
                        matches.append({"line_num": i, "text": line.rstrip()[:500]})
                        if len(matches) >= 50:
                            break
            content = "\n".join(f"L{m['line_num']}: {m['text']}" for m in matches)

        else:
            return {"success": False, "error": f"Invalid section: {section}. Use 'tail', 'head', or 'search'."}

        # Truncate content to prevent context overflow
        max_chars = 10000
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        result = {
            "success": True,
            "session_id": session_id,
            "app_name": app_name,
            "log_path": log_path_str,
            "section": section,
            "content": content,
            "total_lines": total_lines,
            "file_size": _human_size(file_size),
            "truncated": truncated,
        }

        # If we fell back to a different session via broader search, note it
        if _fallback_available_sessions and _original_session_id and session_id != _original_session_id:
            result["note"] = f"Exact session '{_original_session_id}' not found. Showing most recent session instead."
            result["available_sessions"] = [
                {"id": s["session_id"], "app": s["app"]}
                for s in _fallback_available_sessions[:10]
            ]

        return result

    except Exception as e:
        return {"success": False, "error": f"Failed to read session log: {str(e)}"}


@mcp.tool
def list_session_logs(
    app_name: str = "",
    max_sessions: int = 10,
    source: str = "debug",
) -> dict:
    """List available session logs. Use source='conversation' to find past conversations,
    or source='debug' (default) to find OnDemand container debug logs.

    Args:
        app_name: OnDemand app path like 'dev/irisaitest' (only for source='debug').
        max_sessions: Maximum number of sessions to return (default: 10).
        source: 'debug' (default) — lists OnDemand output.log sessions.
            'conversation' — lists JSONL conversation log sessions.
    Returns dict with list of sessions sorted by recency."""
    try:
        username = os.environ.get("USER", "")
        if not username:
            return {"success": False, "error": "Cannot determine username from USER env var"}

        if source == "conversation":
            all_logs = _list_conversation_logs(username)
            return {
                "success": True,
                "source": "conversation",
                "sessions": all_logs[:max_sessions],
                "total_sessions": len(all_logs),
                "hint": "Use read_session_log(session_id='...', source='conversation') to read a specific session.",
            }

        sessions = []

        # Primary: use ALL_SESSIONS_DIR mount (all sessions in the output/ dir)
        all_sessions_dir = os.environ.get("ALL_SESSIONS_DIR", "")
        if all_sessions_dir:
            all_sessions_path = Path(all_sessions_dir)
            if all_sessions_path.exists() and all_sessions_path.is_dir():
                try:
                    for session_dir in all_sessions_path.iterdir():
                        if not session_dir.is_dir():
                            continue
                        log_file = session_dir / "output.log"
                        if log_file.exists():
                            try:
                                st = log_file.stat()
                                sessions.append({
                                    "session_id": session_dir.name,
                                    "app_path": "mounted_sessions",
                                    "output_log": str(log_file),
                                    "modified": datetime.datetime.fromtimestamp(
                                        st.st_mtime
                                    ).isoformat(),
                                    "size_bytes": st.st_size,
                                    "size_human": _human_size(st.st_size),
                                })
                            except (OSError, PermissionError):
                                pass
                except PermissionError:
                    pass
                # Sort mounted sessions by modification time, newest first
                if sessions:
                    sessions.sort(key=lambda s: s["modified"], reverse=True)

        # Also get sessions from OnDemand batch_connect directory discovery
        # (may find sessions from other apps not in the mounted dir)
        bc_sessions = _find_app_sessions(username, app_name or None)

        # Merge: add bc_sessions that aren't already in mounted sessions
        mounted_ids = {s["session_id"] for s in sessions}
        for s in bc_sessions:
            if s["session_id"] not in mounted_ids:
                sessions.append(s)

        # Re-sort combined list by modification time
        sessions.sort(key=lambda s: s["modified"], reverse=True)

        # Group by app for summary
        app_counts = {}
        for s in sessions:
            app = s["app_path"]
            app_counts[app] = app_counts.get(app, 0) + 1

        return {
            "success": True,
            "sessions": sessions[:max_sessions],
            "total_sessions": len(sessions),
            "apps_found": app_counts,
            "hint": "Use read_session_log(session_id='...') to read a specific session's log. "
                    "The first session in the list is usually the current active session.",
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to list session logs: {str(e)}"}


# ── Software Registry Tools ──────────────────────────────────────────────────

_SPACK_BIN = "/opt/spack/bin/spack"

def _ensure_software_dir(work_dir: str) -> Path:
    """Create $WORK_DIR/software/ structure on first use. Idempotent."""
    sw_root = Path(work_dir) / "software"
    for subdir in ("spack_env", "envs", "packages"):
        (sw_root / subdir).mkdir(parents=True, exist_ok=True)

    registry_path = sw_root / "registry.yaml"
    if not registry_path.exists():
        import yaml as _yaml
        initial = {
            "software_root": str(sw_root),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "packages": [],
        }
        with open(registry_path, "w") as f:
            _yaml.dump(initial, f, default_flow_style=False, sort_keys=False)

    spack_env_dir = sw_root / "spack_env"
    spack_yaml = spack_env_dir / "spack.yaml"
    if not spack_yaml.exists() and Path(_SPACK_BIN).exists():
        try:
            subprocess.run(
                [_SPACK_BIN, "env", "create", "-d", str(spack_env_dir)],
                capture_output=True, timeout=30
            )
        except Exception:
            pass

    return sw_root


def _load_registry(sw_root: Path) -> dict:
    """Load registry.yaml, return parsed dict."""
    import yaml as _yaml
    registry_path = sw_root / "registry.yaml"
    if not registry_path.exists():
        return {"software_root": str(sw_root), "last_updated": "", "packages": []}
    with open(registry_path, "r") as f:
        return _yaml.safe_load(f) or {"software_root": str(sw_root), "packages": []}


def _save_registry(sw_root: Path, registry: dict):
    """Write registry.yaml atomically."""
    import yaml as _yaml
    registry["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    registry_path = sw_root / "registry.yaml"
    tmp_path = registry_path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        _yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    tmp_path.rename(registry_path)


def _register_spack_external(sw_root: Path, name: str, version: str, prefix: str):
    """Register package as spack external in the software env."""
    import yaml as _yaml
    spack_env_dir = sw_root / "spack_env"
    packages_yaml = spack_env_dir / "packages.yaml"

    if packages_yaml.exists():
        with open(packages_yaml, "r") as f:
            data = _yaml.safe_load(f) or {}
    else:
        data = {}

    pkgs = data.setdefault("packages", {})
    pkg_entry = pkgs.setdefault(name, {})
    externals = pkg_entry.setdefault("externals", [])

    spec_str = f"{name}@{version}"
    existing = [e for e in externals if e.get("spec") == spec_str]
    if existing:
        existing[0]["prefix"] = prefix
    else:
        externals.append({"spec": spec_str, "prefix": prefix})

    with open(packages_yaml, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _remove_spack_external(sw_root: Path, name: str, version: str):
    """Remove a spack external entry."""
    import yaml as _yaml
    packages_yaml = sw_root / "spack_env" / "packages.yaml"
    if not packages_yaml.exists():
        return

    with open(packages_yaml, "r") as f:
        data = _yaml.safe_load(f) or {}

    pkgs = data.get("packages", {})
    if name not in pkgs:
        return

    spec_str = f"{name}@{version}"
    externals = pkgs[name].get("externals", [])
    pkgs[name]["externals"] = [e for e in externals if e.get("spec") != spec_str]

    if not pkgs[name]["externals"]:
        del pkgs[name]

    with open(packages_yaml, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@mcp.tool
def register_software(
    name: Annotated[str, Field(description="Package/env name, e.g. 'samtools', 'ml-env'")],
    version: Annotated[str, Field(description="Version string, e.g. '1.19', '3.11'")],
    prefix: Annotated[str, Field(description="Absolute path to install root (directory with bin/, lib/, etc.)")],
    source: Annotated[str, Field(description="How it was installed: 'spack', 'conda', 'pip', 'manual', 'container', 'system'")],
    purpose: Annotated[str, Field(description="What this software is for (free text, be specific)")] = "",
    categories: Annotated[list, Field(description="Searchable tags, e.g. ['bioinformatics', 'genomics']")] = [],
    project: Annotated[str, Field(description="Project scope (empty = shared/available to all projects)")] = "",
    default: Annotated[bool, Field(description="Mark as the default version for this name")] = False,
    notes: Annotated[str, Field(description="Dependency notes, conflict warnings, or usage hints")] = "",
) -> dict:
    """Register software in the central registry so it can be discovered in future sessions. Call this AFTER every successful installation, or when a user tells you where existing software lives. The registry is the single source of truth for 'what software exists and where'.

        Args:
            name: Package or environment name (e.g. 'samtools', 'analysis-env')
            version: Version string (e.g. '1.19', '3.11')
            prefix: Absolute path to the install root directory
            source: Installation method used
            purpose: What the software is for — be specific so future queries find it
            categories: Searchable tags for discovery
            project: Optional project scope (empty = shared globally)
            default: Whether this is the preferred version when multiple exist
            notes: Dependency notes or conflict warnings
        Returns dict with success status and the registered entry."""
    work_dir = _get_work_dir()
    if not work_dir:
        return {"success": False, "error": "WORK_DIR not set. Set work directory first."}

    if not prefix or not prefix.startswith("/"):
        return {"success": False, "error": "prefix must be an absolute path"}

    if not Path(prefix).exists():
        return {"success": False, "error": f"prefix path does not exist: {prefix}"}

    valid_sources = ("spack", "conda", "pip", "manual", "container", "system")
    if source not in valid_sources:
        return {"success": False, "error": f"source must be one of: {valid_sources}"}

    sw_root = _ensure_software_dir(work_dir)
    registry = _load_registry(sw_root)

    # Make prefix relative to software_root if it's inside it
    sw_root_str = str(sw_root)
    if prefix.startswith(sw_root_str + "/"):
        rel_prefix = prefix[len(sw_root_str) + 1:]
    else:
        rel_prefix = prefix  # external path, store absolute

    entry = {
        "name": name,
        "version": version,
        "prefix": rel_prefix,
        "source": source,
        "purpose": purpose,
        "categories": categories if categories else [],
        "registered": time.strftime("%Y-%m-%d"),
    }
    if project:
        entry["project"] = project
    if default:
        entry["default"] = True
    if notes:
        entry["notes"] = notes

    # Upsert: replace if same name+version+project exists
    packages = registry.get("packages", [])
    replaced = False
    for i, pkg in enumerate(packages):
        if (pkg.get("name") == name and
            pkg.get("version") == version and
            pkg.get("project", "") == project):
            packages[i] = entry
            replaced = True
            break

    if not replaced:
        packages.append(entry)

    registry["packages"] = packages
    _save_registry(sw_root, registry)

    # Register in spack
    _register_spack_external(sw_root, name, version, prefix)

    return {
        "success": True,
        "action": "updated" if replaced else "registered",
        "entry": entry,
        "registry_path": str(sw_root / "registry.yaml"),
    }


@mcp.tool
def query_software(
    search: Annotated[str, Field(description="Search by name, category, or purpose keyword. Empty string returns all registered software.")] = "",
    project: Annotated[str, Field(description="Filter by project (empty = show all including shared)")] = "",
) -> dict:
    """Query the software registry to discover what's installed and where. Call this BEFORE attempting any installation to check if software already exists. Also use to answer 'what do I have?' questions.

        Args:
            search: Name, category tag, or keyword to search for. Empty returns everything.
            project: Optional project filter. Returns project-scoped entries + shared entries.
        Returns dict with matching packages and their full details."""
    work_dir = _get_work_dir()
    if not work_dir:
        return {"success": False, "error": "WORK_DIR not set. Set work directory first."}

    sw_root = _ensure_software_dir(work_dir)
    registry = _load_registry(sw_root)
    packages = registry.get("packages", [])

    if not packages:
        return {"success": True, "packages": [], "message": "No software registered yet."}

    sw_root_str = str(sw_root)

    # Filter by project
    if project:
        packages = [p for p in packages if p.get("project", "") in ("", project)]

    # Filter by search term
    if search:
        search_lower = search.lower()
        filtered = []
        for p in packages:
            if (search_lower in p.get("name", "").lower() or
                search_lower in p.get("purpose", "").lower() or
                search_lower in " ".join(p.get("categories", [])).lower() or
                search_lower in p.get("notes", "").lower()):
                filtered.append(p)
        packages = filtered

    # Resolve absolute paths for output
    results = []
    for p in packages:
        entry = dict(p)
        prefix = entry.get("prefix", "")
        if prefix and not prefix.startswith("/"):
            entry["absolute_path"] = f"{sw_root_str}/{prefix}"
        else:
            entry["absolute_path"] = prefix
        results.append(entry)

    return {
        "success": True,
        "software_root": sw_root_str,
        "packages": results,
        "total": len(results),
    }


@mcp.tool
def update_software_entry(
    name: Annotated[str, Field(description="Package name to update")],
    version: Annotated[str, Field(description="Version of the entry to update")],
    project: Annotated[str, Field(description="Project scope of the entry (empty = shared)")] = "",
    new_version: Annotated[str, Field(description="New version string (if correcting version)")] = "",
    purpose: Annotated[str, Field(description="Updated purpose description")] = "",
    default: Annotated[bool, Field(description="Set as default version")] = False,
    notes: Annotated[str, Field(description="Updated notes")] = "",
    categories: Annotated[list, Field(description="Updated category tags")] = [],
) -> dict:
    """Update an existing software registry entry. Use to correct version numbers, update purpose descriptions, add notes about conflicts, or mark a version as default.

        Args:
            name: Package name to find
            version: Current version of the entry to update
            project: Project scope (empty = shared entry)
            new_version: If provided, corrects the version string
            purpose: If provided, replaces the purpose
            default: If True, marks this as the default version
            notes: If provided, replaces the notes
            categories: If provided, replaces the categories
        Returns dict with the updated entry."""
    work_dir = _get_work_dir()
    if not work_dir:
        return {"success": False, "error": "WORK_DIR not set."}

    sw_root = _ensure_software_dir(work_dir)
    registry = _load_registry(sw_root)
    packages = registry.get("packages", [])

    target = None
    for i, pkg in enumerate(packages):
        if (pkg.get("name") == name and
            pkg.get("version") == version and
            pkg.get("project", "") == project):
            target = i
            break

    if target is None:
        return {"success": False, "error": f"No entry found for {name}@{version} (project='{project}')"}

    entry = packages[target]
    if new_version:
        # Update spack registration
        old_prefix = entry.get("prefix", "")
        abs_prefix = old_prefix if old_prefix.startswith("/") else str(sw_root / old_prefix)
        _remove_spack_external(sw_root, name, version)
        _register_spack_external(sw_root, name, new_version, abs_prefix)
        entry["version"] = new_version
    if purpose:
        entry["purpose"] = purpose
    if default:
        entry["default"] = True
    if notes:
        entry["notes"] = notes
    if categories:
        entry["categories"] = categories

    packages[target] = entry
    registry["packages"] = packages
    _save_registry(sw_root, registry)

    return {"success": True, "entry": entry}


@mcp.tool
def remove_software_entry(
    name: Annotated[str, Field(description="Package name to remove")],
    version: Annotated[str, Field(description="Version of the entry to remove")],
    project: Annotated[str, Field(description="Project scope (empty = shared entry)")] = "",
) -> dict:
    """Remove a software entry from the registry. This only removes the registry entry — it does NOT delete the actual files. Use when software has been uninstalled or an entry was created in error.

        Args:
            name: Package name to deregister
            version: Version to remove
            project: Project scope (empty = shared)
        Returns dict confirming removal."""
    work_dir = _get_work_dir()
    if not work_dir:
        return {"success": False, "error": "WORK_DIR not set."}

    sw_root = _ensure_software_dir(work_dir)
    registry = _load_registry(sw_root)
    packages = registry.get("packages", [])

    original_len = len(packages)
    packages = [
        p for p in packages
        if not (p.get("name") == name and
                p.get("version") == version and
                p.get("project", "") == project)
    ]

    if len(packages) == original_len:
        return {"success": False, "error": f"No entry found for {name}@{version} (project='{project}')"}

    registry["packages"] = packages
    _save_registry(sw_root, registry)
    _remove_spack_external(sw_root, name, version)

    return {"success": True, "removed": f"{name}@{version}", "remaining": len(packages)}


if __name__ == "__main__":
    port = int(os.environ.get("MCP_FILE_PORT", 8001))
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port, path="/")
