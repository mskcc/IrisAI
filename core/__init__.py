"""IrisAI Core — testable modules extracted from app.py

Modules:
- agent_utils: Error classification, hallucination detection, retry logic
- skill_loader: Dynamic skill discovery and manifest generation
- single_agent: Single agent with dynamic skill selection (Phase 1 engine)
- spend_tools: Budget and usage tracking tools (LiteLLM proxy)
- websearch_tools: Web search and URL fetching tools (SearXNG + approval gate)
- chainlit_tools: Chainlit-dependent tools (upload, render image/PDB/CIF, weights)
- sub_agent: Sub-agent tools for isolated LLM calls (Phase 3 — context isolation)
- history: Token estimation, history trimming, text conversion
- serialization: JSON serialization utilities
- config: YAML config loading (MCP servers)
- persistence: Conversation history, user settings, task state I/O
- checkpointing: Tool call progress tracking for recovery
"""
