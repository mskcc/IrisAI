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

"""Dynamic skill loader — discovers and indexes skill files from a directory.

Skills are Markdown files with YAML frontmatter (delimited by ---).
The loader scans a directory, parses each .md file, and builds a manifest
that can be injected into the agent's system prompt.

No external dependencies — uses a manual --- block parser instead of
the python-frontmatter package.

Phase 1 of the IrisAI architecture redesign.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("core.skill_loader")

# Old flat skills that are superseded by new folder-based skills.
# When both exist, the manifest hides the old one from the skill selector
# (but it remains loadable for backward compatibility with tests/escalation).
_SUPERSEDED_BY = {
    "code_execution": "code-execution",
    "bioinformatics": "bioinformatics-analysis",
    "hpc_cluster": "hpc-submit-job",  # Split into 3, selector sees the new ones
    "websearch": "web-research",
    "user_settings": "user-settings",
}


def parse_frontmatter(text: str) -> tuple:
    """Parse YAML frontmatter from a Markdown file.

    Expects the file to start with '---', followed by YAML content,
    followed by another '---', then the body content.

    Args:
        text: Full text content of the Markdown file.

    Returns:
        Tuple of (metadata_dict, body_string).
        If no valid frontmatter is found, returns ({}, full_text).
    """
    text = text.strip()
    if not text.startswith("---"):
        return {}, text

    # Find the closing ---
    # Start searching after the first ---
    second_marker = text.find("---", 3)
    if second_marker == -1:
        return {}, text

    yaml_block = text[3:second_marker].strip()
    body = text[second_marker + 3:].strip()

    try:
        metadata = yaml.safe_load(yaml_block)
        if not isinstance(metadata, dict):
            return {}, text
        return metadata, body
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        return {}, text


class SkillLoader:
    """Discovers, loads, and indexes skill files dynamically.

    Scans a directory for .md files with YAML frontmatter containing:
        - name: skill identifier (used in structured output)
        - description: what the skill does (used in manifest)
        - allowed_tools: list of tool names this skill can use
        - model: optional LLM model override (e.g. 'opus', 'sonnet')
        - max_iterations: optional AgentExecutor iteration limit
        - guardrails: optional list of hard rules

    The body (after ---) is the full system prompt content for the skill.

    Usage:
        loader = SkillLoader(Path("skills/"))
        manifest = loader.get_manifest()  # inject into system prompt
        content = loader.get_skill_content("hpc_cluster")
        tools = loader.get_allowed_tools("hpc_cluster")
    """

    def __init__(self, skills_dir: Path, extra_dirs: Optional[List[Path]] = None):
        self.skills_dir = Path(skills_dir)
        self._extra_dirs = extra_dirs or []
        self._skills: Dict[str, dict] = {}  # name → {meta, content}
        self.reload()

    def _scan_directory(self, directory: Path, is_user_extension: bool = False) -> None:
        """Scan a directory for .md skill files and add to index.

        Supports two formats:
        - Flat: skills/<name>.md (legacy)
        - Folder: skills/<name>/SKILL.md (new format, takes precedence)

        Args:
            directory: Path to scan for *.md files.
            is_user_extension: If True, skills from this dir cannot override core skills.
        """
        if not directory.exists() or not directory.is_dir():
            return

        # Collect all skill files: folder-based SKILL.md + flat .md files
        skill_files = []
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir() and subdir.name != "shared":
                skill_md = subdir / "SKILL.md"
                if skill_md.exists():
                    skill_files.append(skill_md)
        skill_files.extend(sorted(directory.glob("*.md")))

        for md_file in skill_files:
            try:
                text = md_file.read_text(encoding="utf-8")
                metadata, body = parse_frontmatter(text)

                if not metadata.get("name"):
                    logger.warning(
                        f"Skill file {md_file.name} missing 'name' in frontmatter, skipping"
                    )
                    continue

                name = metadata["name"]
                if name in self._skills:
                    if is_user_extension:
                        logger.warning(
                            f"User extension skill '{name}' in {md_file} conflicts "
                            f"with core skill from {self._skills[name].get('_source', 'unknown')}, skipping"
                        )
                        continue
                    existing_source = self._skills[name].get("_source", "")
                    if "SKILL.md" in existing_source and md_file.name != "SKILL.md":
                        logger.debug(
                            f"Flat skill '{name}' in {md_file.name} superseded by "
                            f"folder-based {existing_source}, skipping"
                        )
                        continue
                    logger.warning(
                        f"Duplicate skill name '{name}' in {md_file.name}, "
                        f"overwriting previous from {existing_source}"
                    )

                self._skills[name] = {
                    "meta": metadata,
                    "content": body,
                    "_source": str(md_file),
                    "_is_user_extension": is_user_extension,
                }
                logger.info(f"Loaded skill: {name} from {md_file}")

            except Exception as e:
                logger.error(f"Failed to load skill file {md_file}: {e}")

    def reload(self) -> None:
        """Scan skills/ directory and extra dirs, parse all *.md files, rebuild index."""
        self._skills.clear()

        if not self.skills_dir.exists() or not self.skills_dir.is_dir():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        self._scan_directory(self.skills_dir, is_user_extension=False)

        for extra_dir in self._extra_dirs:
            self._scan_directory(Path(extra_dir), is_user_extension=True)

    def get_manifest(self) -> str:
        """Auto-generated manifest text for injection into the system prompt.

        Returns a formatted string listing all available skills with their
        descriptions. This is what the LLM reads to decide which skill(s)
        to select.

        Returns:
            Formatted manifest string, e.g.:
            "Available skills:
            - hpc_cluster: Slurm job management, cluster status...
            - dev: Modify application code..."
        """
        if not self._skills:
            return "No skills available."

        lines = ["Available skills:"]
        for name in sorted(self._skills.keys()):
            # Skip superseded skills when their replacement is loaded
            replacement = _SUPERSEDED_BY.get(name)
            if replacement and replacement in self._skills:
                continue
            desc = self._skills[name]["meta"].get("description", "No description")
            lines.append(f"- {name}: {desc}")

        return "\n".join(lines)

    def list_skill_names(self) -> List[str]:
        """Return sorted list of all loaded skill names."""
        return sorted(self._skills.keys())

    def get_skill_content(self, name: str) -> str:
        """Return the full Markdown body (system prompt content) for a skill.

        Args:
            name: Skill name as defined in frontmatter.

        Returns:
            The body content string.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        return self._skills[name]["content"]

    def get_allowed_tools(self, name: str) -> List[str]:
        """Return the allowed_tools list for a skill.

        Args:
            name: Skill name.

        Returns:
            List of tool name strings. Empty list if not specified.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        tools = self._skills[name]["meta"].get("allowed_tools", [])
        return tools if isinstance(tools, list) else []

    def get_model(self, name: str) -> Optional[str]:
        """Return the model override for a skill, or None.

        Args:
            name: Skill name.

        Returns:
            Model string (e.g. 'opus', 'sonnet') or None if not specified.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        return self._skills[name]["meta"].get("model")

    def get_max_iterations(self, name: str) -> Optional[int]:
        """Return the max_iterations for a skill, or None.

        Args:
            name: Skill name.

        Returns:
            Integer max_iterations or None if not specified.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        val = self._skills[name]["meta"].get("max_iterations")
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                return None
        return None

    def get_description(self, name: str) -> str:
        """Return the description for a skill.

        Args:
            name: Skill name.

        Returns:
            Description string.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        return self._skills[name]["meta"].get("description", "")

    def get_guardrails(self, name: str) -> List[str]:
        """Return the guardrails list for a skill.

        Args:
            name: Skill name.

        Returns:
            List of guardrail strings. Empty list if not specified.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        rails = self._skills[name]["meta"].get("guardrails", [])
        return rails if isinstance(rails, list) else []

    def get_workflow_requirements(self, name: str) -> Optional[Dict]:
        """Return the workflow_required configuration for a skill, or None.

        The workflow_required field in frontmatter declares what steps must
        be completed after certain trigger tools are called. This enables
        skill-agnostic workflow enforcement in app.py.

        Expected frontmatter structure:
            workflow_required:
              trigger_tools: [edit_file, write_text_file]
              required_after_trigger:
                - step_name: test_execution
                  tool: execute_dynamic_task
                  must_contain_any: [pytest, test_unit]
                - step_name: git_commit
                  tool: execute_dynamic_task
                  must_contain_any: [git commit, git add]
              skip_allowed: true
              skip_requires: "SKIP REASON:"
              env_checks:
                - name: git_repo
                  check: has_dotgit
                - name: tests_exist
                  check: file_exists
                  path: tests/test_unit.py

        Args:
            name: Skill name.

        Returns:
            Dict with workflow requirements, or None if skill has no workflow.

        Raises:
            KeyError: If skill name not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_skill_names()}")
        wf = self._skills[name]["meta"].get("workflow_required")
        if wf and isinstance(wf, dict):
            return wf
        return None

    def get_merged_workflow_requirements(self, names: List[str]) -> List[Dict]:
        """Return workflow requirements for all active skills that have them.

        Collects workflow_required configs from all named skills, returning
        a list of dicts each tagged with the skill name. Used by app.py to
        enforce workflows across any combination of active skills.

        Args:
            names: List of active skill names.

        Returns:
            List of dicts, each with 'skill_name' added to the workflow config.
            Empty list if no active skills have workflow requirements.
        """
        results = []
        for name in names:
            try:
                wf = self.get_workflow_requirements(name)
                if wf:
                    # Tag with skill name so the enforcer knows which skill it belongs to
                    tagged = dict(wf)
                    tagged["skill_name"] = name
                    results.append(tagged)
            except KeyError:
                continue
        return results

    def get_merged_content(self, names: List[str]) -> str:
        """Return merged system prompt content for multiple skills.

        Used for multi-skill selection — concatenates the body content
        of all selected skills with clear separators.

        Args:
            names: List of skill names to merge.

        Returns:
            Merged content string with skill separators.

        Raises:
            KeyError: If any skill name not found.
        """
        sections = []
        for name in names:
            content = self.get_skill_content(name)
            guardrails = self.get_guardrails(name)
            section = content
            if guardrails:
                rails_text = "\n".join(f"- {r}" for r in guardrails)
                section += f"\n\nGUARDRAILS:\n{rails_text}"
            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def get_merged_tools(self, names: List[str]) -> List[str]:
        """Return union of allowed_tools for multiple skills.

        Args:
            names: List of skill names.

        Returns:
            Deduplicated list of tool names (order preserved).

        Raises:
            KeyError: If any skill name not found.
        """
        seen = set()
        result = []
        for name in names:
            for tool in self.get_allowed_tools(name):
                if tool not in seen:
                    seen.add(tool)
                    result.append(tool)
        return result

    def get_primary_max_iterations(self, names: List[str]) -> int:
        """Return the maximum max_iterations across all specified skills.

        Uses max() across all skills that specify max_iterations, so that
        combining a high-iteration skill (e.g. dev=50) with a lower one
        (e.g. file_search=30) always yields the higher value (50), not
        whichever skill happens to be listed first.

        Falls back to 50 if no skill specifies max_iterations.

        Args:
            names: List of skill names.

        Returns:
            Integer max_iterations value.
        """
        values = []
        for name in names:
            val = self.get_max_iterations(name)
            if val is not None:
                values.append(val)
        return max(values) if values else 50  # default raised from 60 to 50 (was too high)

    def get_primary_model(self, names: List[str]) -> Optional[str]:
        """Return the model from the first skill that specifies one.

        Args:
            names: List of skill names (primary skill first).

        Returns:
            Model string or None.
        """
        for name in names:
            val = self.get_model(name)
            if val is not None:
                return val
        return None

    def get_tool_registry(self) -> Dict[str, List[str]]:
        """Build a reverse mapping of tool_name → list of skill names that provide it.

        This is used to tell the agent which skill to request if it needs
        a tool that's not in its current toolset.

        Returns:
            Dict mapping tool name strings to lists of skill names.
            Example: {"query_slurm_cluster": ["hpc_cluster"],
                      "execute_dynamic_task": ["hpc_cluster", "code_execution", ...]}
        """
        registry: Dict[str, List[str]] = {}
        for name in sorted(self._skills.keys()):
            for tool in self.get_allowed_tools(name):
                if tool not in registry:
                    registry[tool] = []
                registry[tool].append(name)
        return registry

    def get_tool_registry_text(self, exclude_skills: Optional[List[str]] = None) -> str:
        """Generate a compact text representation of the tool registry.

        Shows which tools are available in which skills — so the agent
        knows where to find tools it doesn't currently have access to.
        Only shows tools NOT in the currently active skills (to avoid noise).

        Args:
            exclude_skills: List of currently active skill names. Tools
                that are ONLY in these skills are excluded from the output
                (the agent already has them).

        Returns:
            Formatted string for system prompt injection. Empty string if
            no additional tools exist outside the active skills.
        """
        registry = self.get_tool_registry()
        exclude_skills = exclude_skills or []
        exclude_set = set(exclude_skills)

        # Build list of tools that exist in OTHER skills (not currently active)
        other_tools: Dict[str, List[str]] = {}
        for tool, skills in registry.items():
            # Only include if at least one providing skill is NOT in the active set
            other_skills = [s for s in skills if s not in exclude_set]
            if other_skills:
                # Only show if the tool is NOT already available via active skills
                active_skills_with_tool = [s for s in skills if s in exclude_set]
                if not active_skills_with_tool:
                    other_tools[tool] = other_skills

        if not other_tools:
            return ""

        lines = [
            "=== TOOLS IN OTHER SKILLS (NOT directly callable) ===",
            "These tools are NOT in your current toolset. You CANNOT call them directly.",
            "To use any tool below: call request_additional_skill(skill_name) as your",
            "ONLY tool call (no other tool calls in the same response). The system will",
            "then restart you with those tools available.",
            "",
        ]
        # Group by skill for readability
        skill_tools: Dict[str, List[str]] = {}
        for tool, skills in sorted(other_tools.items()):
            for skill in skills:
                if skill not in skill_tools:
                    skill_tools[skill] = []
                skill_tools[skill].append(tool)

        for skill in sorted(skill_tools.keys()):
            tools_list = ", ".join(sorted(skill_tools[skill]))
            lines.append(f"  {skill}: {tools_list}")

        lines.append("=== END TOOLS IN OTHER SKILLS ===")
        return "\n".join(lines)

    def skill_count(self) -> int:
        """Return the number of loaded skills."""
        return len(self._skills)
