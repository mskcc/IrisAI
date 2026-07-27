# Contributing to IrisAI

## Prerequisites

IrisAI requires institutional HPC infrastructure (Slurm, OOD, LiteLLM proxy, AWS Bedrock) to run. See [DEVELOPMENT.md](DEVELOPMENT.md) for the full infrastructure requirements.

## Branch Workflow

1. Fork or create a feature branch from `main`
2. Make focused, well-tested changes
3. Commit with [conventional commit format](DEVELOPMENT.md#commit-message-format)
4. Open a pull request

## Commit Format

```
<type>(<scope>): <short description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

## Code Style

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full Python style guide.

Key points:
- Type hints on all functions
- Google-style docstrings
- No magic numbers — use named constants
- Maximum line length: 120 characters

## Security Guidelines

- **NEVER** commit API keys, tokens, or credentials
- **NEVER** commit internal IP addresses or hostnames
- Audit `git diff` before committing for accidental credential exposure
