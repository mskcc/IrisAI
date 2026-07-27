# Development Guide

IrisAI is designed to run as an **Open OnDemand (OOD) interactive app** on an HPC cluster. It requires:

- A **Slurm HPC cluster** with Singularity/Apptainer
- A running **LiteLLM proxy** instance with virtual key authentication
- **Open OnDemand** configured for interactive apps
- **AWS Bedrock** (or compatible) access for Claude models
- **Singularity containers** for MCP servers and the Chainlit UI

These are institutional infrastructure dependencies — there is no standalone local development mode. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design and [deployment.md](deployment.md) for setup instructions.

---

## Code Conventions

### Branch Strategy

```
feat/<feature-name>      # New features
fix/<bug-description>    # Bug fixes
docs/<doc-change>        # Documentation only
refactor/<what>          # Code restructuring (no behavior change)
test/<what>              # Test additions/improvements
chore/<what>             # Build, CI, dependency updates
```

### Commit Message Format

```
<type>(<scope>): <short description>

[optional body — explain WHY, not WHAT]
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

**Common scopes:** `pel`, `skills`, `mcp`, `executor`, `history`, `memory`, `compaction`

**Examples:**
```
feat(skills): add pathway-analysis skill with KEGG integration
fix(pel): consecutive counter not resetting after different tool call
docs(architecture): update context management section with correct constants
```

### Python Style

- Type hints on all function signatures
- Google-style docstrings on all public functions
- f-strings for string formatting
- `pathlib.Path` for file paths
- Constants in `UPPER_SNAKE_CASE` at module level
- Maximum line length: 120 characters

### Adding a New Skill

1. Create `skills/<skill-name>/SKILL.md` with YAML frontmatter
2. Add `references/` subdirectory if needed
3. Verify auto-discovery works

### Adding a New MCP Tool

1. Add tool function to appropriate server in `mcp_servers/`
2. Add tool to `config/tool_schemas.json`
3. Add tool to relevant skill's `allowed_tools` list
4. Add PEL budget if needed in `config/policy.yaml`

---

## Security

- **NEVER** commit API keys, tokens, or credentials
- **NEVER** commit internal IP addresses or hostnames — use environment variables
- Use placeholder values in documentation (e.g., `<your-litellm-proxy-url>`)
- Audit `git diff` before committing for accidental credential exposure
