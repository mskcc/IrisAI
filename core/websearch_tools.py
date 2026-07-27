"""Web search tools — SearXNG-powered search with human-in-the-loop approval.

Standalone tool module migrated from agents/websearch.py (Phase 1 Step 4).
These tools provide:
- web_search: SearXNG HTTP search with mandatory approval gate
- fetch_url_content: URL content fetching with mandatory approval gate

Both tools use Chainlit's cl.AskActionMessage for approval dialogs.
An asyncio.Lock ensures only one approval dialog is active at a time.

Safety guarantee: No web request can happen without the user clicking Approve.
This is enforced at the TOOL level, not the prompt level.

These tools are ALWAYS registered in the tool pool. If web search is not
yet enabled (globe button OFF), the approval gate asks the user to enable
it before proceeding. This allows any agent (executor, worker, researcher)
to request web search at any point during execution.
"""
import asyncio
import os
import re

import aiohttp
try:
    import chainlit as cl
except ImportError:
    cl = None  # type: ignore[assignment]
from langchain.tools import tool


# ── SearXNG config ──────────────────────────────────────────────────────
SEARXNG_URL = os.environ.get(
    "SEARXNG_URL",
    f"http://localhost:{os.environ.get('SEARXNG_NGINX_PORT', '8080')}"
)
SEARXNG_TOKEN = os.environ.get("SEARXNG_TOKEN", "")


# ── WebSearchFailureTracker: detect unproductive searches within a turn ──
class WebSearchFailureTracker:
    """Track unproductive web searches and suggest diagnostic pivot after N failures."""

    MAX_UNPRODUCTIVE = 3

    def __init__(self):
        self._unproductive_count = 0

    def reset(self):
        self._unproductive_count = 0

    def record_search(self, result: str) -> str | None:
        """Record a search result. Returns pivot hint if threshold reached."""
        if "No results found" in result or "Search error" in result or len(result.strip()) < 100:
            self._unproductive_count += 1
        else:
            self._unproductive_count = 0
            return None
        if self._unproductive_count >= self.MAX_UNPRODUCTIVE:
            return (
                "\n\n[DIAGNOSTIC PIVOT] Web search is not finding solutions after "
                f"{self._unproductive_count} attempts. Stop reformulating queries. "
                "Switch to local diagnostic debugging: isolate minimal case, "
                "enable verbose output, test dependencies independently."
            )
        return None


def _get_search_tracker() -> WebSearchFailureTracker:
    """Get or create a per-session WebSearchFailureTracker."""
    tracker = cl.user_session.get("_websearch_failure_tracker")
    if tracker is None:
        tracker = WebSearchFailureTracker()
        cl.user_session.set("_websearch_failure_tracker", tracker)
    return tracker


# ── Helper: Get or create a per-session approval lock ───────────────────
def _get_approval_lock() -> asyncio.Lock:
    """Get or create a per-session asyncio.Lock for approval dialogs.

    Chainlit only supports one active cl.AskActionMessage at a time.
    This lock ensures that when the agent calls multiple tools in parallel
    (e.g. fetch_url_content for 2 URLs), the approval dialogs are shown
    sequentially — preventing the second from cancelling the first.
    """
    lock = cl.user_session.get("websearch_approval_lock")
    if lock is None:
        lock = asyncio.Lock()
        cl.user_session.set("websearch_approval_lock", lock)
    return lock


# ── Helper: Extract action value from cl.AskActionMessage response ──────
def _extract_action_value(res) -> str:
    """Extract the action name/value from a cl.AskActionMessage response.

    Handles multiple response formats across Chainlit versions.
    This is a pure function — no Chainlit dependency, fully testable.
    """
    if res is None:
        return "cancel"
    if isinstance(res, dict):
        payload = res.get("payload", res)
        val = payload.get("value") if isinstance(payload, dict) else None
        if val is None:
            val = res.get("name")
        return val or "cancel"
    if hasattr(res, "name"):
        return res.name
    if hasattr(res, "value"):
        return res.value
    return "cancel"


# ── Helper: Run the approval gate (must be called under lock) ───────────
async def _approval_gate(query: str) -> str:
    """Show approval dialog and return the approved query.

    Returns the query string if approved, or raises ValueError if cancelled.
    This is the ONLY path through which a search can be approved.

    If web search is not yet enabled (globe OFF), first asks the user to
    enable it. This allows any agent to trigger web search at any time
    without waiting for the next turn.

    IMPORTANT: This function MUST be called while holding the approval lock
    to prevent concurrent cl.AskActionMessage dialogs.
    """
    websearch_enabled = cl.user_session.get("websearch_enabled", False)

    if not websearch_enabled:
        res = await cl.AskActionMessage(
            content=(
                "🌐 **The agent needs web search to answer your question.**\n\n"
                f"Requested search: **{query}**\n\n"
                "Enable web search for this session?"
            ),
            actions=[
                cl.Action(name="approve", payload={"value": "approve"}, label="✅ Enable & Search"),
                cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
            ],
            timeout=120,
        ).send()

        action = _extract_action_value(res)
        if action != "approve":
            raise ValueError("Web search not enabled by user.")

        cl.user_session.set("websearch_enabled", True)
        print(f"[WEBSEARCH] User enabled web search mid-turn")
        return query

    res = await cl.AskActionMessage(
        content=f"🔍 The agent wants to search the web for:\n\n**{query}**",
        actions=[
            cl.Action(name="approve", payload={"value": "approve"}, label="✅ Search"),
            cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
        ],
        timeout=120,
    ).send()

    action = _extract_action_value(res)

    if action == "approve":
        return query

    raise ValueError("Search cancelled by user.")


# ── Tool: Web Search (with approval gate) ───────────────────────────────
@tool
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using SearXNG. The user will be asked to approve before the search executes.

    Use this when you need current information, latest software versions,
    documentation, or anything that requires up-to-date web data.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (default 5, max 10)
    """
    # ── Acquire approval lock — serializes the full approve→search cycle ─
    lock = _get_approval_lock()
    async with lock:
        # ── MANDATORY APPROVAL GATE ──────────────────────────────────
        try:
            approved_query = await _approval_gate(query)
        except ValueError as e:
            return str(e)

        # ── Only reaches here after explicit user approval ───────────
        print(f"[WEBSEARCH] Query: {approved_query}")
        await cl.Message(content=f"🔍 Searching for: **{approved_query}**...").send()

        # ── Search execution (inside lock — prevents race conditions) ─
        if not SEARXNG_TOKEN:
            return "Error: SearXNG auth token not configured (SEARXNG_TOKEN not set)"

        num_results = min(max(num_results, 1), 10)
        headers = {"Authorization": f"Bearer {SEARXNG_TOKEN}"}
        params = {
            "q": approved_query,
            "format": "json",
            "engines": "yahoo,brave,duckduckgo,pubmed,wikipedia,arxiv,startpage,core",
            "language": "en",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{SEARXNG_URL}/search",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        result_str = f"Search error: SearXNG returned status {resp.status}: {text[:500]}"
                        tracker = _get_search_tracker()
                        pivot_hint = tracker.record_search(result_str)
                        return result_str + (pivot_hint or "")
                    data = await resp.json()

            raw_results = data.get("results", [])[:num_results]

            if not raw_results:
                # Check if engines were rate-limited (transient, not a bad query)
                unresponsive = data.get("unresponsive_engines", [])
                rate_limited = [
                    e for e in unresponsive
                    if any(kw in str(e).lower() for kw in ("too many", "suspended", "rate", "limit"))
                ]
                if rate_limited:
                    print(f"[WEBSEARCH] Rate-limited engines: {rate_limited}")
                    result_str = (
                        f"Search returned no results — engines temporarily suspended due to "
                        f"rate limiting: {rate_limited}. This is transient — try again in "
                        f"30-60 seconds with a different query formulation, or proceed with "
                        f"available knowledge."
                    )
                    tracker = _get_search_tracker()
                    pivot_hint = tracker.record_search(result_str)
                    return result_str + (pivot_hint or "")

                print(f"[WEBSEARCH] No results for: {approved_query}")
                result_str = f"No results found for: {approved_query}"
                tracker = _get_search_tracker()
                pivot_hint = tracker.record_search(result_str)
                return result_str + (pivot_hint or "")

            # Format results as readable text for the agent
            output_lines = [f"Search results for: {approved_query}\n"]
            for i, r in enumerate(raw_results, 1):
                engines_str = ", ".join(r.get("engines", []))
                output_lines.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   URL: {r.get('url', '')}\n"
                    f"   {r.get('content', 'No description')}\n"
                    f"   Engines: {engines_str} | Score: {r.get('score', 0):.1f}\n"
                )

            print(f"[WEBSEARCH] Results: {len(raw_results)} hits for '{approved_query}'")
            for r in raw_results[:3]:
                print(f"[WEBSEARCH]   - {r.get('title', 'No title')}: {r.get('url', '')}")

            output = "\n".join(output_lines)
            tracker = _get_search_tracker()
            pivot_hint = tracker.record_search(output)
            if pivot_hint:
                output += pivot_hint
                print(f"[WEBSEARCH] Pivot hint appended after {tracker._unproductive_count} unproductive searches")
            return output

        except aiohttp.ClientError as e:
            result_str = f"Search error: Failed to connect to SearXNG: {e}"
            tracker = _get_search_tracker()
            pivot_hint = tracker.record_search(result_str)
            return result_str + (pivot_hint or "")
        except Exception as e:
            result_str = f"Search error: {e}"
            tracker = _get_search_tracker()
            pivot_hint = tracker.record_search(result_str)
            return result_str + (pivot_hint or "")


# ── Tool: Fetch URL Content (with approval gate) ───────────────────────
@tool
async def fetch_url_content(url: str, max_chars: int = 25000) -> str:
    """Fetch and extract text content from a specific URL. The user will be asked to approve before fetching.

    Use this after web_search to read the full content of a specific result.

    Args:
        url: The URL to fetch content from
        max_chars: Maximum characters to return (default 25000)
    """
    # ── Acquire approval lock — serializes the full approve→fetch cycle ──
    lock = _get_approval_lock()
    async with lock:
        # ── ENABLE + APPROVAL GATE ───────────────────────────────────
        websearch_enabled = cl.user_session.get("websearch_enabled", False)
        try:
            if not websearch_enabled:
                res = await cl.AskActionMessage(
                    content=(
                        "🌐 **The agent needs to fetch web content.**\n\n"
                        f"URL: **{url}**\n\n"
                        "Enable web access for this session?"
                    ),
                    actions=[
                        cl.Action(name="approve", payload={"value": "approve"}, label="✅ Enable & Fetch"),
                        cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
                    ],
                    timeout=120,
                ).send()
                action = _extract_action_value(res)
                if action != "approve":
                    return "URL fetch cancelled by user."
                cl.user_session.set("websearch_enabled", True)
                print(f"[WEBSEARCH] User enabled web search mid-turn (via fetch)")
            else:
                res = await cl.AskActionMessage(
                    content=f"🌐 The agent wants to fetch content from:\n\n**{url}**",
                    actions=[
                        cl.Action(name="approve", payload={"value": "approve"}, label="✅ Fetch"),
                        cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
                    ],
                    timeout=120,
                ).send()
                action = _extract_action_value(res)
                if action != "approve":
                    return "URL fetch cancelled by user."
        except Exception as e:
            return f"Approval gate error: {e}"

        # ── Only reaches here after explicit user approval ───────────
        print(f"[WEBSEARCH] Fetching URL: {url}")
        await cl.Message(content=f"🌐 Fetching: **{url}**...").send()

        # ── Fetch execution (inside lock — prevents race conditions) ──
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=True,
                ) as resp:
                    if resp.status != 200:
                        return f"Error: HTTP {resp.status} fetching {url}"
                    html = await resp.text()

            # Simple HTML to text extraction
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')

            if len(text) > max_chars:
                from core.persistence import get_work_dir
                from pathlib import Path
                import time as _time, uuid as _uuid
                _wd = get_work_dir() or "/tmp"
                output_dir = Path(_wd) / "dynamic_tasks" / ".tool_outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = f"web_fetch_{int(_time.time())}_{_uuid.uuid4().hex[:6]}.txt"
                output_path = output_dir / filename
                output_path.write_text(f"Source URL: {url}\n\n{text}", encoding="utf-8")
                preview_chars = 4000
                print(f"[WEBSEARCH] {len(text):,} chars saved to {output_path}")
                text = (
                    f"[Full web content ({len(text):,} chars) saved to {output_path}. "
                    f"First {preview_chars} chars shown below]\n\n"
                    f"{text[:preview_chars]}\n\n"
                    f"[...use read_text_file(path=\"{output_path}\") for full content]"
                )

            print(f"[WEBSEARCH] Fetched {len(text)} chars from {url}")
            return f"Content from {url}:\n\n{text}"

        except aiohttp.ClientError as e:
            return f"Error fetching URL: {e}"
        except Exception as e:
            return f"Error processing page: {e}"


# ── Auto web search (no approval gate — for stuck-detection use) ─────────
async def auto_web_search(query: str, max_results: int = 3) -> str:
    """Perform a web search without the approval gate.

    Used by stuck-detection in app.py when websearch is already enabled
    (user has already consented). Returns formatted results or error string.
    """
    import aiohttp
    import json

    if not SEARXNG_TOKEN:
        return "Error: SearXNG auth token not configured (SEARXNG_TOKEN not set)"

    params = {
        "q": query,
        "format": "json",
        "engines": "yahoo,brave,duckduckgo,pubmed,wikipedia,arxiv,startpage,core",
        "language": "en",
    }
    headers = {
        "Authorization": f"Bearer {SEARXNG_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SEARXNG_URL}/search",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return f"Search error: SearXNG returned status {resp.status}: {text[:200]}"
                data = await resp.json()

        results = data.get("results", [])[:max_results]
        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", r.get("snippet", ""))[:300]
            lines.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}\n")

        return "\n".join(lines)

    except aiohttp.ClientError as e:
        return f"Search error: Failed to connect to SearXNG: {e}"
    except Exception as e:
        return f"Search error: {e}"


# ── Tool: Fetch Web Image (with approval gate, returns vision-compatible data) ─
@tool
async def fetch_web_image(url: str) -> str:
    """Fetch an image from a URL to use as visual reference.

    Downloads the image, saves it locally, and returns it so you can SEE it.
    Use this to view reference diagrams, figures, or examples from the web
    that could help you produce better output.

    Args:
        url: The URL of the image to fetch (must be image/* content type)
    """
    import base64
    import hashlib
    import json
    from pathlib import Path

    lock = _get_approval_lock()
    async with lock:
        # ── APPROVAL GATE ────────────────────────────────────────────
        websearch_enabled = cl.user_session.get("websearch_enabled", False)
        try:
            if not websearch_enabled:
                res = await cl.AskActionMessage(
                    content=(
                        "🖼️ **The agent needs to fetch an image from the web.**\n\n"
                        f"URL: **{url}**\n\n"
                        "Enable web access for this session?"
                    ),
                    actions=[
                        cl.Action(name="approve", payload={"value": "approve"}, label="✅ Enable & Fetch"),
                        cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
                    ],
                    timeout=120,
                ).send()
                action = _extract_action_value(res)
                if action != "approve":
                    return "Image fetch cancelled by user."
                cl.user_session.set("websearch_enabled", True)
            else:
                res = await cl.AskActionMessage(
                    content=f"🖼️ The agent wants to fetch an image from:\n\n**{url}**",
                    actions=[
                        cl.Action(name="approve", payload={"value": "approve"}, label="✅ Fetch Image"),
                        cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
                    ],
                    timeout=120,
                ).send()
                action = _extract_action_value(res)
                if action != "approve":
                    return "Image fetch cancelled by user."
        except Exception as e:
            return f"Approval gate error: {e}"

        # ── DOWNLOAD ─────────────────────────────────────────────────
        print(f"[WEBSEARCH] Fetching image: {url}")
        await cl.Message(content=f"🖼️ Fetching image: **{url}**...").send()

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=True,
                ) as resp:
                    if resp.status != 200:
                        return f"Error: HTTP {resp.status} fetching image from {url}"

                    content_type = resp.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        return f"Error: URL returned {content_type}, not an image"

                    image_data = await resp.read()

            if len(image_data) > 20_000_000:
                return "Error: Image too large (>20MB)"

            # Determine media type
            media_type = content_type.split(";")[0].strip()
            if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                media_type = "image/png"

            # Save locally
            work_dir = os.environ.get("WORK_DIR", "/tmp")
            img_dir = Path(work_dir) / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            ext = media_type.split("/")[1].replace("jpeg", "jpg")
            img_path = img_dir / f"ref_{url_hash}.{ext}"
            img_path.write_bytes(image_data)

            # Base64 encode for vision
            b64_data = base64.b64encode(image_data).decode("ascii")

            print(f"[WEBSEARCH] Image saved: {img_path} ({len(image_data)} bytes)")

            # Return marker for native executor multimodal handling
            return json.dumps({
                "__image__": True,
                "base64": b64_data,
                "media_type": media_type,
                "path": str(img_path),
                "description": f"Reference image from {url} ({len(image_data)} bytes, {media_type})",
            })

        except aiohttp.ClientError as e:
            return f"Error fetching image: {e}"
        except Exception as e:
            return f"Error processing image: {e}"


# ── Websearch tools list for registration in single agent's tool pool ───
WEBSEARCH_TOOLS = [web_search, fetch_url_content, fetch_web_image]
