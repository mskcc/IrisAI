---
name: web-research
description: Search the web for documentation, best practices, troubleshooting, latest
  software versions, reference implementations, scientific literature
allowed_tools:
  - web_search
  - fetch_url_content
  - fetch_web_image
model: null
max_iterations: 20
guardrails:
- Both web_search and fetch_url_content have mandatory user approval gates — do NOT ask permission yourself
- NEVER say "I don't know" without searching first
- After 3-5 searches, STOP and summarize findings — do not endlessly refine queries
- If search returns "engines temporarily suspended" — wait, try different query, don't repeat same words
- Include key findings in concise summary format
---

# Web Research

Search the internet for documentation, best practices, troubleshooting solutions,
latest software versions, and scientific references. Uses SearXNG meta-search engine.

## When to Use This Skill

**Triggers:**
- "Search for..." / "Look up..."
- "What's the latest version of X?"
- "How do I do Y?" (when internal knowledge insufficient)
- "Find documentation for..."
- "Best practices for..."
- "Troubleshoot this error" (when local diagnostics insufficient)
- User explicitly requests web search

**NOT for:**
- Questions answerable from local files or cluster state → other skills
- "What software is installed?" → get_environment_info (core tool)
- Tasks that need execution, not research → appropriate execution skill

## Workflow

1. **Search:** web_search(query, num_results=5)
2. **Read promising results:** fetch_url_content(url, max_chars=5000)
3. **Summarize:** Key findings relevant to user's question
4. **Refine if needed:** Different query formulation (max 3-4 refinements)

## Search Strategy

- Start with specific, targeted queries
- If first results miss: try different terminology or add context
- For error messages: quote the exact error in search
- For software: include version numbers and platform
- For scientific: include key terms, organism, method name

## Rate Limit Handling

If "engines temporarily suspended" or "too many requests":
- Search service is temporarily rate-limited (NOT a bad query)
- Try a DIFFERENT query formulation
- If 3+ searches rate-limited: proceed with existing knowledge, note search was unavailable

## When to Stop Searching

After 3-5 searches without satisfying results:
1. STOP — more queries won't help
2. Report what you DID find (even partial)
3. Suggest alternative approaches (local diagnostics, ask user for more context)

## Tools

- `web_search` — Search the web (SearXNG meta-engine)
- `fetch_url_content` — Read full content of a URL
- `fetch_web_image` — Fetch images from URLs
