---
name: storage-management
description: Disk space management — check quota, find large files, clean scratch,
  archive projects, du/df analysis, identify space hogs
allowed_tools:
  - execute_dynamic_task
  - batch
  - find_files
  - list_directory
  - read_text_file
  - grep_file
  - summarize_command_output
model: null
max_iterations: 20
guardrails:
- NEVER delete files without explicit user confirmation
- ALWAYS show what will be deleted/moved BEFORE doing it
- When reporting disk usage, distinguish between user's files and shared/system files
- For large deletions (>1GB), list files and get explicit approval
---

# Storage Management

Check disk usage, quotas, find large files, clean up scratch space, and
manage storage across the HPC filesystem tiers.

## When to Use This Skill

**Triggers:**
- "How much disk space am I using?"
- "Check my quota" / "Am I over quota?"
- "Find large files" / "What's using all my space?"
- "Clean up scratch" / "Remove old files"
- "Archive this project"
- "Disk full" / "No space left"

**NOT for:**
- "Find a specific file" → file-operations
- "Download a dataset" → data-transfer
- "Read this file" → file-operations

## Key Operations

### Check Disk Usage
```
execute_dynamic_task(commands=["du -sh {dir} | sort -rh | head -20"])
```

### Check Quota
```
execute_dynamic_task(commands=["quota -s" or "lfs quota -u {user} {filesystem}"])
```

### Find Large Files
```
execute_dynamic_task(commands=["find {dir} -type f -size +1G -exec ls -lh {} \\; | sort -k5 -rh | head -20"])
```

### Storage Tiers
| Path | Purpose | Quota | Backed Up |
|------|---------|-------|-----------|
| /home | Config, small files | Limited | Yes |
| /data1 | Project data, results | Per-group | Snapshots |
| /scratch | Temporary compute | Large | No |

## Tools

- `execute_dynamic_task` — Run du, df, quota, find commands
- `find_files` — Locate files by pattern/size
- `list_directory` — Browse directories
- `summarize_command_output` — Summarize large du output
