# Policy Enforcement Layer (PEL) Reference

The Policy Enforcement Layer is a **deterministic safety system** that operates independently of LLM instructions. It intercepts every tool call and applies rule-based decisions before execution.

---

## Design Philosophy

1. **If it can be code, it must be code** — Safety is enforced programmatically, not via prompt instructions
2. **Fail loudly, fail early** — Structured errors before wasting resources
3. **Universal enforcement** — Write once, apply everywhere
4. **100% deterministic** — Pattern matching, not probabilistic LLM judgment
5. **Defense in depth** — Even if the LLM is jailbroken, dangerous operations are blocked

---

## Architecture

```
LLM returns tool_use(name, args)
    │
    ▼
PolicyEnforcementLayer.check(tool_name, tool_args)
    │
    ├── [1] Global turn cap check
    ├── [2] Per-tool budget check
    ├── [3] Consecutive call detection
    ├── [4] Blocked pattern matching
    ├── [5] Required parameter validation
    ├── [6] Precondition checks
    │
    ▼
Decision: ALLOW | BLOCK | REQUIRES_APPROVAL
    │
    ├── ALLOW → Tool executes normally
    ├── BLOCK → Error returned to LLM (with suggestion)
    └── REQUIRES_APPROVAL → UI prompt shown to user
```

### Implementation

- **Source:** `core/policy_enforcement.py`
- **Configuration:** `config/policy.yaml`
- **Audit log:** `logs/policy_audit.jsonl`

---

## Decision Types

| Decision | Behavior | User Impact |
|----------|----------|-------------|
| `ALLOW` | Tool call proceeds normally | None — transparent |
| `BLOCK` | Call rejected with structured error | LLM sees error, adapts approach |
| `REQUIRES_APPROVAL` | Chainlit `AskActionMessage` shown | User clicks Allow/Deny |

---

## Check Order

Checks are evaluated in priority order. First match determines the decision:

### 1. Global Turn Cap

```yaml
global_limits:
  max_total_per_turn: 120
```

After 120 tool calls in a single user turn, ALL subsequent calls are blocked. The counter resets when the turn ends (agent responds with final text).

### 2. Per-Tool Budgets

Each tool has individual limits:

```yaml
tool_budgets:
  execute_dynamic_task:
    max_per_turn: 8
    warn_at: 6
    on_exceed: block
  submit_slurm_job:
    max_per_turn: 4
    on_exceed: block
  slurm_monitor_job:
    max_per_turn: 3
    warn_at: 2
    on_exceed: block
  web_search:
    max_per_turn: 5
    on_exceed: block
  find_files:
    max_per_turn: 6
    on_exceed: block
  grep_file:
    max_per_turn: 8
    on_exceed: block
  read_text_file:
    max_per_turn: 15
    warn_at: 12
    on_exceed: block
  read_file_lines:
    max_per_turn: 15
    warn_at: 12
    on_exceed: block
  list_directory:
    max_per_turn: 12
    warn_at: 10
    on_exceed: block
  get_file_info:
    max_per_turn: 10
    on_exceed: block
  edit_file:
    max_per_turn: 8
    on_exceed: block
  _default:
    max_per_turn: 15
```

**`warn_at`**: When hit, a warning is injected into the LLM's context suggesting it finish soon.  
**`on_exceed`**: Action when `max_per_turn` is reached (always `block`).

### 3. Consecutive Call Detection

Prevents the LLM from calling the same tool repeatedly when a better tool exists:

```yaml
consecutive_limits:
  execute_dynamic_task:
    max_consecutive: 2
    redirect_to: "batch tool"
    suggestion: "Use 'batch' tool for multiple shell commands"
  grep_file:
    max_consecutive: 4
    redirect_to: "batch with type='grep'"
  read_file_lines:
    max_consecutive: 4
    redirect_to: "analyze_files"
  read_text_file:
    max_consecutive: 5
    redirect_to: "batch with type='read'"
  find_files:
    max_consecutive: 4
    redirect_to: "batch with type='find'"
  slurm_monitor_job:
    max_consecutive: 3
    redirect_to: "slurm_monitor_job with wait=True"
  web_search:
    max_consecutive: 3
    redirect_to: "fetch_url_content for specific URLs"
```

When consecutive limit is hit, the PEL blocks the call and returns a suggestion message pointing to the better tool.

### 4. Blocked Patterns

Hard blocks on dangerous operations — no override possible:

```yaml
blocked_patterns:
  - pattern: "rm -rf /"
    severity: critical
    message: "Recursive deletion of root filesystem blocked"
  - pattern: "sudo"
    severity: critical
    message: "Privilege escalation blocked"
  - pattern: "which|locate"
    in_args: "command"
    severity: medium
    message: "Use query_software or get_environment_info instead"
```

### 5. Required Parameters

Certain tools MUST have specific parameters:

```yaml
required_params:
  submit_slurm_job:
    - param: container_image
      message: "All Slurm jobs MUST run inside a container"
```

### 6. Precondition Checks

Runtime validations before tool execution:

```yaml
preconditions:
  remove_file:
    requires_user_confirmation: true
    message: "File deletion requires explicit user approval"
```

---

## Audit Logging

Every PEL decision is logged to `logs/policy_audit.jsonl`:

```json
{
  "timestamp": "2026-07-17T14:32:01.123Z",
  "tool_name": "execute_dynamic_task",
  "decision": "ALLOW",
  "turn_count": 3,
  "consecutive_count": 1,
  "reason": null
}
```

```json
{
  "timestamp": "2026-07-17T14:32:05.456Z",
  "tool_name": "execute_dynamic_task",
  "decision": "BLOCK",
  "turn_count": 4,
  "consecutive_count": 3,
  "reason": "Consecutive limit exceeded (max: 2). Suggestion: Use 'batch' tool"
}
```

### Audit Fields

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp |
| `tool_name` | Tool that was checked |
| `decision` | ALLOW, BLOCK, or REQUIRES_APPROVAL |
| `turn_count` | How many times this tool was called this turn |
| `consecutive_count` | Current consecutive call streak |
| `reason` | Why blocked/approved (null for ALLOW) |

---

## Counter Management

### Reset Points

- **Per-turn counters** reset when the agent produces a final text response (turn ends)
- **Consecutive counters** reset when a DIFFERENT tool is called
- **No cross-session persistence** — all counters start at 0 each turn

### Counter Flow Example

```
Turn starts (all counters = 0)
    │
    Agent calls read_text_file → turn_count[read_text_file] = 1, consecutive = 1
    Agent calls read_text_file → turn_count[read_text_file] = 2, consecutive = 2
    Agent calls grep_file      → turn_count[grep_file] = 1, consecutive[read_text_file] resets
    Agent calls grep_file      → turn_count[grep_file] = 2, consecutive = 2
    ...
    Agent produces final text  → ALL counters reset
```

---

## Integration with Phased Execution

The PEL works alongside (but is independent of) the phase system:

- **Phase system** removes tools from the API schema (structural prevention)
- **PEL** enforces runtime limits on tools that ARE available

Both systems must pass for a tool call to execute. They complement each other:
- Phase system = "CAN this tool exist in this context?"
- PEL = "SHOULD this specific call be allowed right now?"

---

## Configuration File

Full configuration lives in `config/policy.yaml`. Key sections:

```yaml
version: "1.0"

global_limits:
  max_total_per_turn: 120

consecutive_limits:
  # ... (see Section 3 above)

tool_budgets:
  # ... (see Section 2 above)

blocked_patterns:
  # ... (see Section 4 above)

required_params:
  # ... (see Section 5 above)

preconditions:
  # ... (see Section 6 above)

audit:
  enabled: true
  output: "logs/policy_audit.jsonl"

protocol:
  mode: "enabled"
  max_steps: 50
  hard_stop_on_deviation: true
```

---

## Extending the PEL

### Adding a New Blocked Pattern

Edit `config/policy.yaml`:
```yaml
blocked_patterns:
  - pattern: "your_dangerous_pattern"
    severity: critical  # or medium, low
    message: "Human-readable explanation of why blocked"
```

### Adding a New Tool Budget

```yaml
tool_budgets:
  my_new_tool:
    max_per_turn: 5
    warn_at: 3
    on_exceed: block
```

### Adding Consecutive Limit

```yaml
consecutive_limits:
  my_new_tool:
    max_consecutive: 3
    redirect_to: "better_alternative_tool"
    suggestion: "Use X instead of calling Y repeatedly"
```

No code changes required — the PEL reads `policy.yaml` at startup and applies rules dynamically.
