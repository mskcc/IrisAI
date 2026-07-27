# Skills Reference

Complete catalog of IrisAI's skill system — how skills are defined, selected, and composed into agent capabilities.

---

## Skill System Architecture

### How Skills Work

A **skill** is a self-contained capability definition that includes:
- Instructions (system prompt content)
- Tool whitelist (which tools the agent can use)
- Guardrails (constraints injected into the prompt)
- Optional model override (force Opus for complex skills)
- Optional reference documents (loaded on demand)

When a user sends a message, the **skill selector** (Haiku) classifies the request and selects 1-3 relevant skills. The agent is then composed with those skills' combined tools and instructions.

### Skill File Formats

**Folder-based (current standard — 21 skills):**
```
skills/hpc-submit-job/
├── SKILL.md              # Main definition with YAML frontmatter
└── references/           # Optional reference docs
    ├── slurm-guide.md
    └── container-best-practices.md
```

**Flat-file (legacy — 12 skills):**
```
skills/hpc_cluster.md     # Single file with YAML frontmatter
```

### Superseded Skills

6 legacy flat-file skills are hidden from the selector because newer folder-based skills replace them:

| Legacy Skill | Superseded By |
|--------------|---------------|
| `hpc_cluster.md` | `hpc-submit-job/`, `hpc-monitor/`, `hpc-query/` |
| `code_execution.md` | `code-execution/` |
| `conversational.md` | `conversational/` |
| `file_search.md` | Removed — consolidated into `file-operations/` |
| `history.md` | `history/` |
| `user_settings.md` | `user-settings/` |

### Skill Selection Algorithm

1. User message arrives
2. `build_skill_selection_prompt()` creates a classifier prompt with all skill names + descriptions
3. **Haiku** returns structured `SkillSelection`:
   ```json
   {
     "skills": ["hpc-submit-job", "file-operations"],
     "complexity": "moderate",
     "needs_research": false,
     "needs_planning": true,
     "needs_slurm": true,
     "parallel_subtasks": []
   }
   ```
4. Selected skills' tools are unioned + always-available tools added
5. Skill markdown content concatenated into system prompt
6. Agent created with this composite configuration

### Cross-Skill Escalation

If the agent discovers it needs tools from a skill not initially selected:
```
Agent calls: request_additional_skill(skill_name="bioinformatics-analysis")
    → Turn interrupts
    → New skill added to selection
    → Agent rebuilt with expanded tools
    → Turn resumes (max 2 escalations)
```

---

## Active Skills Catalog

### HPC & Compute

#### `hpc-submit-job`
Submit and configure Slurm HPC jobs with container isolation.

**Key Tools:** `submit_slurm_job`, `slurm_monitor_job`, `check_user_slurm_access`, `query_slurm_cluster`  
**Guardrails:** All jobs MUST run inside containers; validate resource requests  

#### `hpc-monitor`
Monitor running/completed Slurm jobs, check status, troubleshoot failures.

**Key Tools:** `slurm_monitor_job`, `query_slurm_cluster`, `slurm_cancel_job`  
**Guardrails:** Never cancel jobs without user confirmation  

#### `hpc-query`
Query cluster state — partitions, GPU availability, queue status.

**Key Tools:** `query_slurm_cluster`, `check_user_slurm_access`, `hpc_directory`  
**Guardrails:** Report accurate counts; don't estimate  

#### `pipeline-orchestration`
Multi-step HPC workflows — chained Slurm jobs with dependencies.

**Key Tools:** `submit_slurm_job`, `slurm_monitor_job`, `check_user_slurm_access`, `query_slurm_cluster`  
**Guardrails:** Use `dependency='afterok:JOBID'` for chains; verify each step before next  

#### `ml-training`
Machine learning model training on GPU nodes.

**Key Tools:** `submit_slurm_job`, `slurm_monitor_job`, `check_user_slurm_access`, `query_slurm_cluster`  
**Guardrails:** Request appropriate GPU type; set reasonable walltimes; monitor for convergence  

---

### Bioinformatics & Science

#### `bioinformatics-analysis`
Analyze biological data — scRNA-seq (h5ad), VCF variants, gene expression.

**Key Tools:** `extract_h5ad_summary`, `inspect_vcf_summary`, `list_obs_columns`, `get_unique_values`, `summarize_cell_types`, `extract_coding_variants`  
**Guardrails:** Never load entire h5ad into memory; use summary tools  

#### `sequence-processing`
FASTA/FASTQ processing, sequence alignment, variant calling workflows.

**Key Tools:** `submit_slurm_job`, `check_user_slurm_access`  
**Guardrails:** Large sequence jobs go through Slurm, not dynamic task  

#### `pathway-analysis`
Biological pathway enrichment, gene set analysis, network visualization.

**Key Tools:** MCP bio_processing tools + visualization  
**Guardrails:** Use established pathway databases (KEGG, Reactome, GO)  

#### `clinical-biostatistics`
Survival analysis, clinical trial statistics, patient cohort comparisons.

**Key Tools:** `execute_dynamic_task`, `submit_slurm_job`  
**Guardrails:** Use appropriate statistical tests; report confidence intervals  

---

### Development & Infrastructure

#### `code-execution`
Run scripts, install packages, execute shell commands.

**Key Tools:** `execute_dynamic_task`, `submit_slurm_job`  
**Guardrails:** 5-minute timeout for dynamic tasks; GPU/long tasks go to Slurm  

#### `container-building`
Build Singularity/Apptainer container images (.sif/.def).

**Key Tools:** `submit_slurm_job`, `execute_dynamic_task`  
**Guardrails:** Use fakeroot environment variables; handle bind mount conflicts  

#### `software-management`
Install, discover, and manage software packages and conda environments.

**Key Tools:** `query_software`, `register_software`, `submit_slurm_job`  
**Guardrails:** Always check registry before installing; register after successful install  

---

### Data & File Operations

#### `file-operations`
Find, read, write, search, and manage files and directories.

**Key Tools:** `find_files`, `read_text_file`, `write_text_file`, `edit_file`, `grep_file`, `list_directory`, `get_file_info`, `make_directory`, `render_image_inline`  
**Guardrails:** Confirm before overwriting; never read binary files with text tools  

#### `data-transfer`
Move data between systems — S3, SCP, rsync, downloads.

**Key Tools:** `submit_slurm_job`, `check_user_slurm_access`  
**Guardrails:** Use datatransfer partition for large transfers  

#### `storage-management`
Disk usage analysis, quota checking, cleanup recommendations.

**Key Tools:** `execute_dynamic_task`, `hpc_directory`  
**Guardrails:** Never delete without user confirmation; report sizes accurately  

#### `visualization`
Create publication-quality figures, charts, and plots.

**Key Tools:** `execute_dynamic_task`, `render_image_inline`, `save_image`  
**Guardrails:** Always use `matplotlib.use('Agg')`; always render_image_inline for display; save PNG + PDF  

---

### Research & Communication

#### `web-research`
Search the web and fetch external documentation.

**Key Tools:** `web_search`, `fetch_url_content`, `fetch_web_image`  
**Guardrails:** Cite sources; verify claims with multiple sources  

#### `conversational`
General questions, explanations, help — no tools needed.

**Key Tools:** `web_search`, `fetch_url_content` (optional)  
**Guardrails:** Answer from knowledge when possible; use web search for current info  

#### `spend`
Check AI usage costs and budget status.

**Key Tools:** `get_daily_activity`, `get_user_budget`  
**Guardrails:** Report costs accurately; never modify budgets  

#### `history`
Browse and search past conversation sessions.

**Key Tools:** `list_session_logs`, `read_session_log`  
**Guardrails:** Don't expose raw session data; summarize relevantly  

#### `user-settings`
View and modify user preferences and configuration.

**Key Tools:** `get_user_settings`, `get_current_user_info`, `set_user_work_directory`, `hpc_directory`  
**Guardrails:** Confirm changes before applying  

---

## Legacy Skills (Superseded)

These flat-file skills exist for backward compatibility but are hidden from the skill selector:

| File | Status | Replaced By |
|------|--------|-------------|
| `alphafold.md` | Active (not superseded) | — |
| `bioinformatics.md` | Superseded | `bioinformatics-analysis/` |
| `code_execution.md` | Superseded | `code-execution/` |
| `conversational.md` | Superseded | `conversational/` |
| `dev.md` | Active (development agent) | — |
| `file_search.md` | Removed | Consolidated into `file-operations/` — skill file deleted |
| `history.md` | Superseded | `history/` |
| `hpc_cluster.md` | Superseded | `hpc-submit-job/` + `hpc-monitor/` + `hpc-query/` |
| `spend.md` | Superseded | `spend/` |
| `toolmaker.md` | Active (experimental) | — |
| `user_settings.md` | Superseded | `user-settings/` |
| `websearch.md` | Superseded | `web-research/` |

---

## Adding a New Skill

### 1. Create the skill directory

```bash
mkdir -p skills/my-new-skill/references/
```

### 2. Write SKILL.md

```markdown
---
name: my-new-skill
description: One-line description for the skill selector
model: sonnet  # or opus for complex reasoning tasks
max_iterations: 15
allowed_tools:
  - tool_name_1
  - tool_name_2
guardrails:
  - "Constraint 1 that MUST be followed"
  - "Constraint 2"
---

# My New Skill

## When to Use This Skill
- Trigger phrases / request patterns

## Workflow
- Step-by-step instructions for the agent

## Key Recipes
- Common task patterns with tool sequences

## Best Practices
- Do's and don'ts
```

### 3. Register in skill loader

The `SkillLoader` auto-discovers skills from the `skills/` directory. No manual registration needed — just place the file and restart.

### 4. Test

```bash
# Verify skill loads correctly
pytest tests/test_unit.py -k "skill_loader" -v
```

---

## Skill Composition Rules

1. **Max 3 skills per turn** — Keeps prompt size manageable
2. **Tool union** — All tools from selected skills are available
3. **Prompt concatenation** — Skill instructions appear in selection order
4. **Guardrail injection** — All guardrails from all selected skills are enforced
5. **Model override** — If ANY selected skill requires Opus, Opus is used
6. **Always-available** — Memory, escalation, and basic file tools are always present regardless of skill selection
