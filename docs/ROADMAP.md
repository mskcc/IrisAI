# IrisAI Roadmap

Feature completion status and future development direction.

---

## Completed ✅

### Core Architecture
- [x] Single-agent architecture (replaced multi-agent supervisor)
- [x] NativeAgentExecutor (direct Anthropic Messages API, replacing LangChain execution)
- [x] Multi-model support (6 models: 3 Anthropic + 3 OpenAI-compatible)
- [x] Extended thinking (Sonnet: 5K budget, Opus: 10K budget)
- [x] Prompt caching (cache_system + cache_last)
- [x] LiteLLM proxy integration (cost tracking, multi-provider routing)

### Skill System
- [x] Skill-based routing with dynamic tool composition
- [x] ~27 unique active skills (21 folder-based + legacy)
- [x] Skill selector (Haiku structured output classification)
- [x] Cross-skill escalation (`request_additional_skill`, max 2 iterations)
- [x] Superseded skill mapping (6 legacy skills hidden by new equivalents)
- [x] Folder-based skill format with `references/` subdirectory

### Phased Execution
- [x] Research phase (read-only tools + write_findings)
- [x] Plan phase (read-only + write_plan/edit_plan)
- [x] Execute phase (full tool access)
- [x] Structural tool gating (tools absent from schema, not runtime-blocked)
- [x] AskActionMessage blocking gates for phase transitions

### Tool Ecosystem
- [x] 4 MCP servers (file_ops, bio_processing, slurm_management, code_execution)
- [x] ~65 MCP tools across servers
- [x] ~20 local tools (batch, sub-agent, chainlit, spend, websearch)
- [x] Unified batch tool (shell, edit, read, grep in one call)
- [x] Worker agent (full sub-agent with 900s timeout, 20 iterations)
- [x] Context-isolated sub-agents (analyze_files, review_codebase, summarize)
- [x] Streamable HTTP transport for MCP (not stdio)

### Safety & Policy
- [x] Policy Enforcement Layer (PEL) — deterministic tool gating
- [x] Global turn cap (120 tool calls/turn)
- [x] Per-tool budgets with warnings and hard limits
- [x] Consecutive call detection with redirect suggestions
- [x] Blocked pattern matching (dangerous operations)
- [x] Approval flows for destructive operations
- [x] Full audit logging (policy_audit.jsonl)

### Context Management
- [x] 3-layer compaction (per-tool Haiku, sliding window, session-end curation)
- [x] Sliding window: 6 messages within 40K token budget; older messages compacted
- [x] Compaction threshold: 100K tokens
- [x] Anchored compaction: project-scoped context block preserved across cycles
- [x] Per-turn transcript writer (`dynamic_tasks/.turn_transcripts/`)
- [x] Intent-aware summarization (`core/intent_summarizer.py`)
- [x] Per-tool output compression (Haiku, 65/35 head/tail pre-trim)
- [x] Session-end memory curation (status/knowledge/history)

### Intelligence Features
- [x] Complexity-based model escalation (simple→Sonnet, complex→Opus)
- [x] Refinement detection (user dissatisfaction → auto-escalate to Opus)
- [x] Stuck detection (agent looping → suggest web search)
- [x] Walltime monitoring (background Slurm job tracking with toast warnings)

### Persistence & Recovery
- [x] Crash-safe JSONL session logging
- [x] Per-project memory (status.md, knowledge.md, history.md)
- [x] Software registry (persistent package/environment tracking)
- [x] Cost tracking per session via LiteLLM

### Deployment
- [x] Open OnDemand integration
- [x] Two-container architecture (Chainlit + MCP servers)
- [x] Singularity/Apptainer container isolation
- [x] Bearer token MCP authentication

---

## In Progress 🔄

- [ ] Documentation cleanup for external sharing (this effort)
- [ ] Performance optimization for long-running sessions
- [ ] Improved error recovery for MCP connection drops

---

## Near-Term Plans 📋

### Agent Quality
- [ ] Improved planning quality metrics
- [ ] Better handling of ambiguous user requests
- [ ] Structured output validation for tool arguments

### Tooling
- [ ] Additional MCP servers for specialized domains
- [ ] Tool usage analytics (most/least used, error rates)
- [ ] Automated tool testing framework

### Deployment
- [ ] CI/CD pipeline for automated testing
- [ ] Container image versioning and registry
- [ ] Blue/green deployment strategy

---

## Future Considerations 🔮

- [ ] Multi-user collaboration features
- [ ] Custom skill authoring by end users
- [ ] Integration with additional LLM providers
- [ ] Fine-tuned models for specialized domains
- [ ] Real-time resource monitoring dashboards

---

## Architecture Evolution

| Date | Change | Rationale |
|------|--------|-----------|
| Feb 2026 | Monolithic → modular `core/` | Testability, separation of concerns |
| Feb 2026 | No tests → comprehensive test suite | Reliability, CI readiness |
| Feb 2026 | Chat-only → HPC workflow agent | Domain specialization |
| Apr 2026 | Multi-agent → single agent + skills | Simplicity, cost, debuggability |
| May 2026 | No safety → PEL (Policy Enforcement Layer) | Defense in depth |
| Jun 2026 | Reactive → phased execution (research/plan/execute) | Quality of complex outputs |
| Jul 2026 | Anthropic-only → multi-provider (6 models) | Flexibility, cost optimization |
| Jul 2026 | ~9 skills → 27+ skills (folder-based format) | Richer capabilities, reference docs |
