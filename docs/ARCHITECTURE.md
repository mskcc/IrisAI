# IrisAI Architecture

This document explains IrisAI's internal architecture — how components interact, why design decisions were made, and how to extend the system.

## High-Level Architecture

IrisAI is a **single-agent system with skill-based routing**. One LLM agent is dynamically composed per user turn with only the tools and instructions relevant to the task. This replaces multi-agent architectures (supervisor → specialized agents) which are slower, more expensive, and harder to debug.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser (Chainlit UI)                        │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ WebSocket
┌──────────────────────────────────▼──────────────────────────────────┐
│                          app.py (Orchestrator)                      │
│                                                                     │
│  ┌──────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│  │  Skill   │  │  NativeAgent     │  │  Policy Enforcement    │   │
│  │ Selector │→ │  Executor        │→ │  Layer (PEL)           │   │
│  │ (Haiku)  │  │  (Anthropic API) │  │  (config/policy.yaml)  │   │
│  └──────────┘  └────────┬─────────┘  └────────────────────────┘   │
│                          │ LLM calls (all models)                  │
│              ┌───────────▼───────────────────────┐                 │
│              │  LiteLLM Proxy → LLM Backend       │                 │
│              │  (AWS Bedrock, Azure, OpenAI, etc.) │                 │
│              └───────────────────────────────────┘                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Tool Calls (results returned to LLM)
         ┌──────────────────┼────────────────────────┐
         ▼                  ▼                        ▼
┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐
│  MCP Servers │  │  Local Tools    │  │   Sub-Agents       │
│  (4 servers) │  │  (batch, spend, │  │  (worker_agent,    │
│  ~65 tools   │  │   websearch)    │  │   analyze_files)   │
└──────────────┘  └─────────────────┘  └────────────────────┘
  file ops, bio,    read/write files,     delegated LLM calls
  slurm, code exec  cost tracking, web    via app.py → LiteLLM
```

## Core Execution Flow

Every user message follows this pipeline:

1. **User message** → Chainlit WebSocket → `app.py`
2. **Skill Selection**: Haiku structured output → `SkillSelection` (skills, complexity, phase flags)
3. **Phase determination**: `needs_research` → research first; `needs_planning` → plan first; else → execute
4. **Agent Assembly**: Merge skill prompts + phase prompt + user context → filter tools to phase-allowed set
5. **Execution**: `NativeAgentExecutor` (Anthropic SDK format via LiteLLM proxy, with prompt caching)
6. **Tool calls**: PEL checks → MCP/local tool execution → result compaction → back to LLM
7. **Phase transitions**: `write_findings`/`write_plan` → user approval gate → advance to next phase
8. **Escalation**: `request_additional_skill` → interrupt → reload with expanded skill set (max 2)
9. **Response delivery**: Validate → stream to UI → save session log → PEL counter reset

## Agent Architecture

### NativeAgentExecutor

The default execution engine uses the **Anthropic Messages API format** directly — not LangChain's AgentExecutor. All LLM calls route through the **LiteLLM proxy** (`LITELLM_URL`), which forwards to the actual model backend.

**Why not LangChain?** Direct API control enables:
- **Prompt caching** — system prompt cached across turns (significant cost reduction)
- **Extended thinking** — Opus 4 with configurable thinking budget
- **Structural tool gating** — tools absent from schema (not just blocked at runtime)

### LLM Provider Abstraction

Two provider implementations behind a common `LLMProvider` protocol:
- `AnthropicProvider` — Anthropic Messages API format
- `OpenAIProvider` — OpenAI-compatible API format

Both route through the LiteLLM proxy. Supported models: Claude Sonnet 4, Opus 4, Haiku 4.5 (via AWS Bedrock), plus OpenAI-compatible endpoints.

## Skill System

Skills are markdown files (`skills/*/SKILL.md`) with YAML frontmatter defining:
- `name`, `description` — for Haiku skill selector
- `allowed_tools` — which tools are available for this skill
- `system_prompt` — injected into the agent context for that turn

The `SkillLoader` auto-discovers all skills at startup. The Haiku skill selector returns a `SkillSelection` — the union of selected skills' tool sets becomes the agent's tool schema for that turn.

**30+ skills** cover: HPC (hpc-submit-job, hpc-monitor), bioinformatics (sequence-processing, pathway-analysis), AlphaFold, code execution, data transfer, visualization, file operations, and more.

## Policy Enforcement Layer (PEL)

Defined in `config/policy.yaml`, enforced by `core/policy_enforcement.py`. Rules include:

- **Per-tool call budgets** — e.g., `execute_dynamic_task` max 15 calls/turn
- **Global turn budget** — 120 tool calls/turn maximum
- **Consecutive limits** — prevent tool call loops
- **Blocked patterns** — regex-matched commands that require approval
- **Approval flows** — destructive operations require user confirmation

Every tool call is checked against PEL before execution. Results are logged to `logs/policy_audit.jsonl`.

## Phased Execution

Complex tasks flow through enforced phases:

```
Research Phase → Planning Phase → Execution Phase
```

Tools not in the current phase's allowed set are **absent from the API schema** — the LLM cannot call tools that aren't in its schema (structural prevention, not prompt-based).

Phase transitions require calling `write_findings` or `write_plan`, which triggers a user approval gate before advancing.

## Context Management

Three-layer token management:

1. **Per-tool compression** (`core/context_compactor.py`) — Haiku compresses large tool outputs before they enter conversation history
2. **Sliding window** (`core/history.py`) — Keeps last 6 turns raw; older turns compacted at 100,000 token threshold with 40,000 token budget
3. **Session-end curation** (`core/memory_state.py`) — Writes `status.md` (replace), `knowledge.md` (append), `history.md` (append) to persistent project memory

## MCP Server Architecture

Four MCP servers run as separate processes inside a Singularity container and communicate with the main app via HTTP + bearer token auth:

| Server | Default Port | Tools | Purpose |
|--------|-------------|-------|---------|
| `file_ops_server.py` | 8001 | ~38 | File listing, reading, writing, search |
| `bio_processing_server.py` | 8004 | ~12 | Bioinformatics analysis, AlphaFold3 |
| `slurm_management_server.py` | 8003 | ~11 | Slurm job submit, monitor, cancel |
| `code_execution_server.py` | 8005 | ~4 | Dynamic code execution |

## Deployment

IrisAI deploys as an Open OnDemand (OOD) interactive app:

1. User launches IrisAI from the OOD web portal
2. OOD runs `template/before.sh.erb` — sets env vars, generates LiteLLM virtual key, finds available ports
3. Slurm allocates a compute node
4. Two Singularity containers are launched:
   - **Chainlit container** — `app.py` + UI on port 8000
   - **MCP container** — all 4 MCP servers on ports 8001–8005
5. OOD proxies port 8000 to the user's browser

### LiteLLM Virtual Key Authentication

Each user session gets a per-user virtual key generated at startup:

```bash
# In template/before.sh.erb (runs with OOD privileges)
LITELLM_VIRTUAL_KEY=$(curl -s -X POST "${LITELLM_API_BASE}/key/generate" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"duration": "8h", "metadata": {"user": "'${USER}'"}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")
export LITELLM_VIRTUAL_KEY
```

This key is valid for the session duration and scoped to the user, enabling per-user rate limiting and spend tracking.
