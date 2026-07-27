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

"""Unified memory system for IrisAI.

Architecture:
- Global knowledge: {user_data_dir}/memory/knowledge.md (cross-project facts)
- Project memories: {user_data_dir}/memory/projects/{project}/ (status.md, knowledge.md, history.md)

All storage is under /home/{user}/{app}/memory/ — never in work_dir.
"""
import os
from pathlib import Path
from typing import List, Optional

from core.persistence import get_user_data_dir


# ── Directory helpers ─────────────────────────────────────────────────────

def get_global_memory_dir(username: str) -> Path:
    """Get the global memory directory (persistent, independent of work_dir)."""
    return get_user_data_dir(username) / "memory"


def get_project_memory_dir(work_dir: str, project_name: str) -> Path:
    """Get a project-specific memory directory.

    The work_dir parameter is kept for API compatibility but storage is always
    under the unified memory root: /home/{user}/{app}/memory/projects/{project}/
    """
    username = os.environ.get("USER", "unknown")
    return get_user_data_dir(username) / "memory" / "projects" / project_name


def get_protocols_dir(username: str, project_name: Optional[str] = None) -> Path:
    """Get the protocols directory, scoped to project if one is active.

    Global:  /home/{user}/{app}/memory/protocols/
    Project: /home/{user}/{app}/memory/projects/{project}/protocols/
    """
    if project_name:
        return get_user_data_dir(username) / "memory" / "projects" / project_name / "protocols"
    return get_user_data_dir(username) / "memory" / "protocols"


# ── Context block building ────────────────────────────────────────────────


def build_memory_context_block(
    username: str,
    work_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    skill_names: Optional[List[str]] = None,
    task_keywords: Optional[List[str]] = None,
    max_auto_inject: int = 5,
    environment_index: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> str:
    """Build the unified context block for system prompt injection.

    Injects: user env info + HPC environment index + project status.md +
    project knowledge.md + global knowledge.md + recent attempts log.
    """

    lines = ["=== USER ENVIRONMENT ==="]
    has_content = False

    if username:
        lines.append(f"Username: {username}")
        has_content = True

    if work_dir:
        lines.append(f"Work directory: {work_dir}")
        has_content = True
        lines.append("")
        lines.append("WORKDIR FILE PLACEMENT RULES (always follow these):")
        lines.append("  reports/       → All reports, analyses, summaries (.md, .pdf, .html, .pptx)")
        lines.append("  scripts/       → Standalone scripts (.py, .sh)")
        lines.append("  docs/          → Documentation, plans, worksheets, design docs")
        lines.append("  configs/       → Config files (.yml, .yaml, .json, .toml, .ini)")
        lines.append("  slurm_jobs/    → Slurm jobs by date (YYYY-MM-DD/jobname_ts/submit.sh, stdout.log)")
        lines.append("  projects/      → Per-project multi-file work (subdirs by project name)")
        lines.append("  .cache/        → Temporary/scratch data")
        lines.append("  uploads/       → User uploads (auto-managed)")
        lines.append("  images/        → Saved visualizations (auto-managed)")
        lines.append("  dynamic_tasks/ → Task execution (auto-created, never cleaned)")
        lines.append("  NEVER place files at the workdir root — always use the appropriate subdirectory above.")
        lines.append("  ENVIRONMENT KNOWLEDGE (3 layers — project overrides user, user overrides system):")
        lines.append("    System (read-only): paths/containers/partitions shown below. Call get_environment_info(topic) for full details.")
        lines.append("    User (your memories): software you've discovered, weights, data paths. Use read_memory/update_memory.")
        lines.append("    Project (project memories): project-specific paths, containers, data sources.")
        lines.append("    NEVER use which/find/locate to search for software or paths.")

    if environment_index and environment_index.strip():
        lines.append("")
        lines.append(environment_index.strip())
        has_content = True

    if weights_path:
        lines.append(f"AlphaFold3 weights path: {weights_path}")
        has_content = True

    if project_name:
        lines.append(f"Project name: {project_name}")
        if project_name != "general" and work_dir:
            lines.append(f"Project directory: {Path(work_dir) / 'projects' / project_name}")
        has_content = True

    # Direct injection of project memory files (strict 3-file model)
    if project_name:
        proj_dir = get_project_memory_dir(work_dir or "", project_name)
        for fname, label, max_chars in [
            ("status.md", "PROJECT STATUS", 6000),
            ("knowledge.md", "PROJECT KNOWLEDGE", 8000),
        ]:
            fpath = proj_dir / fname
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    # Strip frontmatter if present
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            content = parts[2].strip()
                    # Detect scaffold-only content (newly created project)
                    _is_scaffold = not content.strip() or content.strip() in (
                        f"# {project_name} — Status\n\nNew project. No work done yet.",
                        f"# {project_name} — Knowledge",
                        f"# {project_name} — Status",
                    )
                    if content.strip() and not _is_scaffold:
                        if len(content) > max_chars:
                            content = content[:max_chars] + "\n...(truncated)"
                        lines.append(f"\n=== {label}: {project_name} ===")
                        lines.append(content)
                        lines.append(f"=== END {label} ===")
                        has_content = True
                    else:
                        lines.append(f"\n=== {label}: {project_name} ===")
                        lines.append(
                            "(No prior knowledge or history for this project. "
                            "This is the first run — do not search for previous results.)"
                        )
                        lines.append(f"=== END {label} ===")
                        has_content = True
                except Exception:
                    pass

    # Inject global knowledge (cross-project facts)
    if username:
        global_knowledge_path = get_global_memory_dir(username) / "knowledge.md"
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
                    lines.append("\n=== GLOBAL KNOWLEDGE (cross-project) ===")
                    lines.append(gk_content)
                    lines.append("=== END GLOBAL KNOWLEDGE ===")
                    has_content = True
            except Exception:
                pass

    # Inject recent attempts log (episodic context)
    if project_name:
        from core.memory_state import load_attempts_log
        recent_attempts = load_attempts_log(project_name, limit=10)
        if recent_attempts:
            lines.append("")
            lines.append("=== RECENT ATTEMPTS (episodic — conditions may have changed) ===")
            for attempt in recent_attempts:
                ts = attempt.get("timestamp", "")[:16]
                action = attempt.get("action", "")[:100]
                error = attempt.get("error", "")
                result = attempt.get("result", "")[:80]
                if error:
                    lines.append(f"  [{ts}] {action} → ERROR: {error[:80]}")
                else:
                    lines.append(f"  [{ts}] {action} → {result}")
            lines.append("=== END RECENT ATTEMPTS ===")
            has_content = True

    # Inject software registry summary so model knows what's installed
    if work_dir:
        registry_path = Path(work_dir) / "software" / "registry.yaml"
        if registry_path.exists():
            try:
                import yaml
                with open(registry_path, "r") as f:
                    reg_data = yaml.safe_load(f)
                packages = reg_data.get("packages", []) if reg_data else []
                if packages:
                    packages.sort(key=lambda p: p.get("registered", ""), reverse=True)
                    lines.append("")
                    lines.append("=== INSTALLED SOFTWARE (from registry) ===")
                    for pkg in packages[:10]:
                        name = pkg.get("name", "?")
                        version = pkg.get("version", "?")
                        prefix = pkg.get("prefix", "?")
                        if prefix and not prefix.startswith("/"):
                            prefix = f"{work_dir}/software/{prefix}"
                        purpose = pkg.get("purpose", "")
                        line = f"  {name} {version} → {prefix}"
                        if purpose:
                            line += f" ({purpose})"
                        lines.append(line)
                    if len(packages) > 10:
                        lines.append(f"  ... and {len(packages) - 10} more (call query_software for full list)")
                    lines.append("=== END INSTALLED SOFTWARE ===")
                    has_content = True
            except Exception:
                pass

    if not has_content:
        return ""

    lines.append("")
    lines.append("=== END USER ENVIRONMENT ===")
    return "\n".join(lines)
