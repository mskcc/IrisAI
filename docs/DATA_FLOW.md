# Data Flow — Request Lifecycle

This document traces a user message from browser to final response, showing every processing step, decision point, and data transformation.

---

## Overview

```
Browser → WebSocket → app.py → Skill Selection → Phase Check → Agent Creation
    → LLM Call → Tool Decision → PEL Check → Tool Execution → Result Compaction
    → LLM Continues or Responds → Stream to UI → Session Log → Done
```

---

## Phase 1: Message Ingestion

### 1.1 WebSocket Reception

```
Browser (Chainlit JS client)
    │ WebSocket message
    ▼
app.py @cl.on_message handler
    │
    ▼
Message object: { content, id, author, created_at }
```

### 1.2 Input Sanitization

- Token count estimation (tiktoken)
- Truncation if message exceeds `MAX_SINGLE_MESSAGE_TOKENS` (30,000)
- Unicode normalization
- Attachment handling (file uploads → MCP file_ops)

### 1.3 Session Context Loading

- Load conversation history from session state
- Load project memory (status.md, knowledge.md)
- Load user settings (work_dir, preferences)
- Check for active plan/findings from previous phases

---

## Phase 2: Skill Selection

### 2.1 Classification

The skill selector uses **Haiku** (fast, cheap) with structured output:

```
Input: User message + list of available skill descriptions
Output: SkillSelection {
    skills: ["hpc-submit-job", "file-operations"],
    complexity: "moderate",
    needs_research: false,
    needs_planning: true,
    needs_slurm: true,
    parallel_subtasks: []
}
```

### 2.2 Skill Resolution

1. Load skill definitions from `SkillLoader` registry
2. Filter out superseded skills (6 legacy skills hidden by new equivalents)
3. Union tool whitelists from all selected skills
4. Add always-available tools (memory, escalation, file basics)

### 2.3 Phase Determination

| Condition | Starting Phase |
|-----------|---------------|
| `needs_research: true` | Research |
| `needs_planning: true` | Plan |
| Active plan exists (from prior turn) | Execute |
| Otherwise | Execute |

---

## Phase 3: Agent Creation

### 3.1 System Prompt Composition

The system prompt is assembled from:
1. **Base prompt** — Core identity, rules, guardrails
2. **Phase prompt** — Phase-specific instructions and constraints
3. **Skill prompts** — Concatenated SKILL.md content for selected skills
4. **User context** — Environment info, project status, memory
5. **Guardrails** — Skill-specific guardrails injected at end

### 3.2 Tool Filtering

Tools are filtered by the current phase (structural gating):

| Phase | Available Tools |
|-------|----------------|
| Research | Read-only: `find_files`, `read_text_file`, `grep_file`, `list_directory`, `execute_shell_readonly`, `analyze_files`, `review_codebase_section`, `write_findings` |
| Plan | Same as research + `write_plan`, `edit_plan` |
| Execute | ALL tools from selected skills |

**Key**: Tools not in the phase set are **absent from the API schema** — the LLM cannot call them.

### 3.3 NativeAgentExecutor Configuration

```python
executor = NativeAgentExecutor(
    provider=AnthropicProvider(...),  # or OpenAIProvider
    tools=filtered_tool_list,
    system_prompt=composed_prompt,
    max_iterations=15,
    thinking_budget=5000,  # Sonnet; 10000 for Opus
    cache_system=True,
    cache_last=True,
)
```

---

## Phase 4: LLM Execution Loop

### 4.1 First LLM Call

```
Messages: [system_prompt, ...conversation_history, user_message]
    │
    ▼
Anthropic Messages API (with prompt caching)
    │
    ▼
Response: { content: [text_blocks, tool_use_blocks] }
```

### 4.2 Tool Call Processing

For each `tool_use` block in the response:

```
tool_use: { name: "submit_slurm_job", input: {...} }
    │
    ▼
[PEL Check] ─── Is this call allowed?
    │
    ├── BLOCK → Return error message to LLM
    ├── REQUIRES_APPROVAL → Show UI prompt, wait for user
    └── ALLOW → Continue
    │
    ▼
[Tool Execution] ─── MCP HTTP call or local function
    │
    ▼
[Result Compaction] ─── If output > 3× budget, Haiku compresses
    │
    ▼
tool_result: { content: "..." }  → Added to messages for next LLM call
```

### 4.3 Iteration Loop

The executor loops until:
- LLM responds with only text (no tool calls) → **done**
- Max iterations reached (15) → force stop with summary
- Intra-turn token ceiling hit (150K) → force stop
- Phase gate triggered (write_findings/write_plan) → phase transition

### 4.4 Parallel Tool Calls

When the LLM returns multiple `tool_use` blocks in one response:
- All are executed (subject to PEL checks)
- Results are batched and returned together
- Reduces round-trips for independent operations

---

## Phase 5: Phase Transitions

### 5.1 Research → Plan

```
Agent calls write_findings(content="...")
    │
    ▼
Findings saved to disk (project memory)
    │
    ▼
AskActionMessage displayed to user:
    "Research complete. Proceed to planning?"
    [Proceed] [Cancel]
    │
    ▼
User clicks [Proceed]
    │
    ▼
Phase advances to "plan"
Agent rebuilt with plan-phase tools
New turn begins
```

### 5.2 Plan → Execute

```
Agent calls write_plan(content="- [ ] Step 1: ...")
    │
    ▼
Plan saved to disk (project memory)
    │
    ▼
AskActionMessage displayed to user:
    "Plan ready. Execute?"
    [Execute] [Edit] [Cancel]
    │
    ▼
User clicks [Execute]
    │
    ▼
Phase advances to "execute"
Agent rebuilt with ALL tools + plan injected into context
```

---

## Phase 6: Escalation & Recovery

### 6.1 Skill Escalation

```
Agent determines it needs tools not in current skill set
    │
    ▼
Calls request_additional_skill(skill_name="bioinformatics-analysis")
    │
    ▼
IMMEDIATE INTERRUPT — current turn stops
    │
    ▼
New skill's tools added to whitelist
Agent rebuilt with expanded tool set
Turn resumes from scratch with escalated capabilities
    │
    ▼
(Max 2 escalations per turn — MAX_ESCALATION_ITERATIONS)
```

### 6.2 Stuck Detection

```
Agent makes same error 3+ turns in a row
    │
    ▼
System detects stuck pattern
    │
    ▼
Injects suggestion: "Consider using web_search for documented solutions"
```

### 6.3 Refinement Detection

```
User response indicates dissatisfaction
("that's wrong", "try again", "no, I meant...")
    │
    ▼
Auto-escalate to Opus for next turn
(Better reasoning for corrective action)
```

---

## Phase 7: Response Delivery

### 7.1 Streaming

- Text blocks are streamed token-by-token to the UI via Chainlit
- Tool call indicators shown as "thinking" animations
- Phase transitions shown as action buttons

### 7.2 Post-Response Processing

After the agent's final response:

1. **PEL reset** — Turn counters zeroed for next message
2. **Session log** — Full turn (messages + tool calls + results) appended to JSONL
3. **Cost tracking** — Token counts and estimated cost recorded via LiteLLM
4. **History management** — If total tokens > 80K, trigger compaction

### 7.3 Error Handling

| Error Type | Recovery |
|------------|----------|
| MCP connection lost | Retry with exponential backoff (3 attempts) |
| LLM API error (429) | Wait and retry with backoff |
| LLM API error (500) | Retry once, then surface to user |
| Tool execution error | Return error as tool result (agent self-corrects) |
| Token limit exceeded | Trigger compaction, retry with shorter context |
| Phase violation | Block tool call, return phase-appropriate error |

---

## Data Persistence

### What's Saved Per Turn

| Data | Format | Location |
|------|--------|----------|
| Full conversation | JSONL (incremental) | Session log directory |
| Project memory | Markdown | Memory files (status/knowledge/history) |
| Tool audit log | JSONL | `logs/policy_audit.jsonl` |
| Cost data | Via LiteLLM | LiteLLM proxy tracking |
| Plans/findings | Markdown | Project plans directory |

### What's NOT Persisted

- Raw LLM API responses (only parsed content kept)
- MCP bearer tokens (regenerated each session)
- Intermediate compaction results (only final summary kept)
- Tool execution temp files (cleaned up after response)

---

## Performance Characteristics

| Metric | Typical Value |
|--------|--------------|
| Skill selection | ~1s (Haiku) |
| Simple tool use turn | 3-8s |
| Complex multi-tool turn | 15-60s |
| Worker agent delegation | Up to 900s (15 min timeout) |
| Slurm job submission | 2-5s (job start time varies) |
| Context compaction | 3-5s (Haiku) |
| Phase transition (with UI gate) | User-dependent |
