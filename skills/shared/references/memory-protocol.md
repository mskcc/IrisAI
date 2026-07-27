# Memory Protocol

## When to Check Memory

Before acting on any task, check project memory for relevant prior knowledge:
- Previously discovered paths (conda envs, reference genomes, data locations)
- User preferences and past decisions
- Known issues or workarounds for this project

```
read_memory(project)
```

## Evidence-Gated Trust Rules

Memory is a record of what was true AT A GIVEN POINT IN TIME. It may be stale.

1. **Verify before acting on memory claims:**
   - Memory says file exists → check it still exists (find_files)
   - Memory says env has package X → verify (python -c "import X")
   - Memory says partition Y is accessible → verify (check_user_slurm_access)

2. **Trust hierarchy:**
   - Current tool output > Memory > Assumptions
   - If memory contradicts current observation, trust current state

3. **When to update memory:**
   - After discovering new paths, environments, or configurations
   - After user expresses a preference
   - After completing work that produced reusable artifacts
   - NEVER save transient state (job IDs, temporary paths, session-specific info)

4. **What to save:**
   - Conda env paths that were successfully created
   - Reference genome/index locations
   - User preferences for output format, style, partition
   - Known issues and their solutions
   - Project structure decisions

5. **What NOT to save:**
   - Job IDs (they expire)
   - Temporary file paths (/tmp)
   - Information derivable from the filesystem or git
   - Entire command outputs

## Memory API

```
read_memory(project)           # Read project knowledge
update_memory(project, key, value)  # Save/update knowledge
list_projects()                # List all projects
```

## Pattern: Discovery → Action → Persist

Every workflow should follow this pattern:
1. **Check memory** for existing knowledge
2. **Discover** if memory is empty or stale (get_environment_info, find_files)
3. **Act** using discovered information
4. **Persist** new discoveries to memory for future sessions
