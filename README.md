# IrisAI — AI-powered HPC Research Assistant

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Sonnet%204%20%2F%20Opus%204%20%2F%20Haiku%204.5-purple)]()
[![Skills](https://img.shields.io/badge/skills-33-orange)]()
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-65%2B-brightgreen)]()
[![Models](https://img.shields.io/badge/models-6-blueviolet)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

AI-powered HPC research assistant — conversational interface for Slurm clusters, bioinformatics workflows, and scientific computing.

IrisAI enables researchers to interact with HPC resources through natural language: submit and monitor Slurm jobs, run AlphaFold3 structure predictions, analyze genomics data, manage files, and execute code — all from a Chainlit chat interface deployed via Open OnDemand.

## Architecture

IrisAI is a **single-agent system with skill-based routing**. One LLM agent is dynamically composed per user turn with only the tools and instructions relevant to the task.

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

**Key components:**
- **Skill Router** — Haiku classifies each request and selects relevant skill(s) + tool subset
- **NativeAgentExecutor** — Direct Anthropic Messages API (not LangChain); enables prompt caching and extended thinking
- **Policy Enforcement Layer (PEL)** — Per-tool call budgets, blocking rules, audit logging
- **MCP Servers** — 4 modular tool servers (~65 tools): file ops, bio processing, Slurm management, code execution
- **Phased Execution** — Research → Plan → Execute phases with structural tool gating
- **Context Compaction** — 3-layer token management (per-tool compression, sliding window, session-end curation)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## Infrastructure Requirements

IrisAI is designed for institutional HPC deployment. It requires:

| Component | Purpose |
|-----------|---------|
| **Slurm** | HPC workload manager |
| **Singularity/Apptainer ≥1.5** | Container runtime for MCP servers and code execution |
| **Open OnDemand** | Web portal for user authentication and interactive app launch |
| **LiteLLM Proxy** | Routes LLM calls to your provider (AWS Bedrock, Azure, etc.) |
| **PostgreSQL** | Required by LiteLLM for virtual key management |
| **AWS Bedrock** (or compatible) | LLM backend — Claude Sonnet 4, Opus 4, Haiku 4.5 |

There is no standalone local development mode. See [docs/deployment.md](docs/deployment.md) for setup instructions.

## Features

- **30+ skills** — HPC, bioinformatics, AlphaFold3, code execution, data analysis, file ops, and more
- **Slurm integration** — Submit, monitor, and cancel jobs via natural language
- **AlphaFold3** — Structure prediction with automated JSON preparation
- **Bioinformatics tools** — VCF analysis, H5AD/single-cell data, sequence processing
- **File system operations** — Read, write, search files on HPC storage
- **Multi-model support** — Claude Sonnet/Opus/Haiku + OpenAI-compatible alternatives
- **Policy Engine** — Per-tool rate limits, blocked patterns, approval flows
- **Session memory** — Persistent project memory across conversations
- **Containerized execution** — All compute runs inside Singularity containers

## Repository Structure

```
├── app.py                      # Main Chainlit orchestrator
├── core/                       # Pure Python logic (45+ modules)
│   ├── native_executor.py      # NativeAgentExecutor
│   ├── policy_enforcement.py   # Policy Enforcement Layer
│   ├── skill_loader.py         # Skill discovery & parsing
│   ├── history.py              # Token management & context compaction
│   └── ...
├── skills/                     # 30+ skill definitions (SKILL.md files)
├── mcp_servers/                # MCP tool servers
│   ├── file_ops_server.py      # File operations (~38 tools)
│   ├── bio_processing_server.py # Bioinformatics (~12 tools)
│   ├── slurm_management_server.py # Slurm (~11 tools)
│   └── code_execution_server.py  # Code execution (~4 tools)
├── config/
│   ├── policy.yaml             # PEL rules & per-tool limits
│   ├── mcp_servers.yaml        # MCP server registry
│   └── tool_schemas.json       # Tool schema definitions
├── containers/                 # Singularity container definitions
├── template/                   # Open OnDemand lifecycle scripts
└── docs/                       # Documentation
```

## Environment Variables

All environment variables are injected by Open OnDemand at session start via `template/before.sh.erb`. See [.env.example](.env.example) for the full list with descriptions.

Key variables:

| Variable | Purpose |
|----------|---------|
| `LITELLM_URL` | LiteLLM proxy URL |
| `LITELLM_API_BASE` | Same as LITELLM_URL (SDK compatibility alias) |
| `LITELLM_VIRTUAL_KEY` | Per-user auth token (generated at session start) |
| `MCP_SHARED_BEARER_TOKEN` | MCP server authentication token |

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for code conventions, testing, and contribution guidelines.

## Citation

If you use IrisAI in your research or publications, please cite:

```bibtex
@inproceedings{valleru2026irisai,
  title     = {Empowering Cancer Researchers: An Agentic {AI} System for
               Intuitive Interaction with High-Performance Computing},
  author    = {Valleru, Lohit and others},
  booktitle = {Proceedings of the Practice and Experience in Advanced
               Research Computing (PEARC '26)},
  year      = {2026},
  publisher = {ACM},
}
```

> Lohit Valleru et al. "Empowering Cancer Researchers: An Agentic AI System for
> Intuitive Interaction with High-Performance Computing." PEARC 2026.

## License

Copyright 2026 Lohit Valleru and contributors at Memorial Sloan Kettering Cancer Center

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for the full text.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
