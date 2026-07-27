"""Spend tools — budget and usage tracking via LiteLLM proxy API.

Standalone tool module migrated from agents/spend.py (Phase 1 Step 3).
These tools query the LiteLLM proxy for:
- /user/info: user-level budget, cumulative spend, reset schedule
- /user/daily/activity: daily spend breakdown by model

No external dependencies beyond aiohttp (already in the container).
These tools are registered in the single agent's tool pool and
selected when the 'spend' skill is active.
"""
import os
from datetime import datetime

import aiohttp
from langchain.tools import tool


# ── LiteLLM proxy config ────────────────────────────────────────────────
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:8080")


def _get_api_key() -> str:
    """Get the LiteLLM virtual key from environment."""
    key = os.environ.get("LITELLM_VIRTUAL_KEY", "")
    if not key:
        raise RuntimeError("LITELLM_VIRTUAL_KEY not set")
    return key


def _get_username() -> str:
    """Get the current username from environment."""
    return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


@tool
async def get_user_budget() -> str:
    """Get current user budget and cumulative spend information.

    Returns the user's total spend, max budget, budget duration,
    budget reset time, and remaining budget.
    Use this to answer questions like "what's my budget?" or
    "how much have I spent?" or "how much budget do I have left?"
    """
    try:
        api_key = _get_api_key()
        username = _get_username()
        headers = {"Authorization": f"Bearer {api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{LITELLM_URL}/user/info",
                headers=headers,
                params={"user_id": username},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return f"Error: LiteLLM API returned status {resp.status}: {text[:500]}"
                data = await resp.json()

        # Extract user info
        user_info = data.get("user_info", data)
        spend = user_info.get("spend", 0) or 0
        max_budget = user_info.get("max_budget")
        budget_duration = user_info.get("budget_duration", "not set")
        budget_reset_at = user_info.get("budget_reset_at", "not set")
        user_id = user_info.get("user_id", username)

        # Build response
        lines = [
            "## 💰 User Budget Info",
            f"- **User:** {user_id}",
            f"- **Total spend:** ${spend:.4f}",
        ]

        if max_budget and max_budget > 0:
            remaining = max_budget - spend
            pct_used = (spend / max_budget) * 100
            lines.append(f"- **Max budget:** ${max_budget:.2f}")
            lines.append(f"- **Remaining:** ${remaining:.4f}")
            lines.append(f"- **Usage:** {pct_used:.1f}% used")
            if pct_used > 80:
                lines.append(f"- **⚠️ Warning:** Budget usage is above 80%!")
        else:
            lines.append("- **Max budget:** not set (unlimited)")

        lines.append(f"- **Budget duration:** {budget_duration}")
        lines.append(f"- **Budget resets at:** {budget_reset_at}")

        # Key count if available
        keys = data.get("keys", [])
        if keys:
            lines.append(f"- **Active keys:** {len(keys)}")

        return "\n".join(lines)

    except aiohttp.ClientError as e:
        return f"Error connecting to LiteLLM proxy: {e}"
    except Exception as e:
        return f"Error fetching user budget info: {e}"


@tool
async def get_daily_activity(start_date: str = "", end_date: str = "") -> str:
    """Get daily spend and usage activity for a date range.

    Args:
        start_date: Start date in YYYY-MM-DD format. Defaults to today.
        end_date: End date in YYYY-MM-DD format. Defaults to same as start_date.

    Returns spend, tokens, requests, and model breakdown for each day.
    Use this for questions like "how much did I spend today?",
    "show me last week's usage", "what's my daily spend breakdown?"
    """
    try:
        api_key = _get_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}

        # Default dates
        today = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = today
        if not end_date:
            end_date = start_date

        params = {"start_date": start_date, "end_date": end_date}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{LITELLM_URL}/user/daily/activity",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return f"Error: LiteLLM API returned status {resp.status}: {text[:500]}"
                data = await resp.json()

        results = data.get("results", [])
        if not results:
            return f"No activity data found for {start_date} to {end_date}."

        lines = [f"## 📊 Daily Activity: {start_date} to {end_date}\n"]

        total_spend = 0.0
        total_tokens = 0
        total_requests = 0

        for day in results:
            date = day.get("date", "unknown")
            metrics = day.get("metrics", {})
            spend = metrics.get("spend", 0)
            prompt_tokens = metrics.get("prompt_tokens", 0)
            completion_tokens = metrics.get("completion_tokens", 0)
            total_day_tokens = metrics.get("total_tokens", 0)
            requests = metrics.get("successful_requests", 0)
            failed = metrics.get("failed_requests", 0)

            total_spend += spend
            total_tokens += total_day_tokens
            total_requests += requests

            lines.append(f"### 📅 {date}")
            lines.append(f"- **Spend:** ${spend:.4f}")
            lines.append(f"- **Tokens:** {total_day_tokens:,} ({prompt_tokens:,} in / {completion_tokens:,} out)")
            lines.append(f"- **Requests:** {requests} successful, {failed} failed")

            # Model breakdown
            breakdown = day.get("breakdown", {})
            models = breakdown.get("models", {})
            if models:
                lines.append("- **By model:**")
                for model_name, model_data in models.items():
                    m = model_data.get("metrics", {})
                    m_spend = m.get("spend", 0)
                    m_tokens = m.get("total_tokens", 0)
                    m_requests = m.get("successful_requests", 0)
                    lines.append(
                        f"  - `{model_name}`: ${m_spend:.4f} · "
                        f"{m_tokens:,} tokens · {m_requests} calls"
                    )
            lines.append("")

        # Summary if multiple days
        if len(results) > 1:
            lines.append("### 📈 Period Summary")
            lines.append(f"- **Total spend:** ${total_spend:.4f}")
            lines.append(f"- **Total tokens:** {total_tokens:,}")
            lines.append(f"- **Total requests:** {total_requests:,}")

        return "\n".join(lines)

    except aiohttp.ClientError as e:
        return f"Error connecting to LiteLLM proxy: {e}"
    except Exception as e:
        return f"Error fetching daily activity: {e}"


# ── Spend tools list for registration in single agent's tool pool ───────
SPEND_TOOLS = [get_user_budget, get_daily_activity]
