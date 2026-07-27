"""Cost tracking for the native LLM provider layer.

Accumulates token usage (including cache metrics) across multiple LLM calls
within a single user turn. Uses litellm.cost_per_token() for pricing.

Replaces the LangChain BaseCallbackHandler approach for provider-based calls.
The old CostTrackingCallback in app.py continues to work for any remaining
LangChain-based calls during the transition period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.llm_provider import Usage

logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    """Accumulates token usage and cost across LLM calls in a turn."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost: float = 0.0
    num_calls: int = 0
    _call_details: list[dict] = field(default_factory=list)

    def accumulate(self, usage: Usage, model: str = "unknown") -> None:
        """Add usage from a single LLM call.

        Args:
            usage: Usage dataclass from LLMResponse.
            model: Model identifier for cost calculation.
        """
        self.num_calls += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_creation_tokens += usage.cache_creation_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens

        cost = self._calculate_cost(usage, model)
        self.total_cost += cost

        self._call_details.append({
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation": usage.cache_creation_tokens,
            "cache_read": usage.cache_read_tokens,
            "cost": cost,
        })

    def _calculate_cost(self, usage: Usage, model: str) -> float:
        """Calculate cost using litellm pricing."""
        try:
            import litellm

            # Total input includes cached tokens (they're cheaper but still count)
            prompt_tokens = usage.input_tokens + usage.cache_creation_tokens + usage.cache_read_tokens
            completion_tokens = usage.output_tokens

            if prompt_tokens == 0 and completion_tokens == 0:
                return 0.0

            # Normalize model name for litellm pricing lookup:
            # - Strip "us." cross-region prefix if present
            # - Add "bedrock/" prefix if it looks like a Bedrock model ID
            # - If lookup fails, retry with ":0" suffix (Bedrock inference profile version)
            clean_model = model
            if clean_model.startswith("us."):
                clean_model = clean_model[3:]
            if clean_model.startswith("anthropic.") and not clean_model.startswith("bedrock/"):
                clean_model = f"bedrock/{clean_model}"

            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=clean_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:
                if ":" not in clean_model.split("/")[-1]:
                    prompt_cost, completion_cost = litellm.cost_per_token(
                        model=f"{clean_model}:0",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                else:
                    raise

            # Adjust for cache pricing:
            # - cache_read tokens are 90% cheaper than regular input
            # - cache_creation tokens are 25% more expensive than regular input
            # The base cost_per_token treats all as regular input, so we adjust
            if usage.cache_read_tokens > 0 and prompt_tokens > 0:
                per_token_input_cost = prompt_cost / prompt_tokens if prompt_tokens else 0
                cache_read_savings = usage.cache_read_tokens * per_token_input_cost * 0.9
                prompt_cost -= cache_read_savings

            total = prompt_cost + completion_cost

            if total == 0.0 and (prompt_tokens > 0 or completion_tokens > 0):
                logger.warning(
                    f"Zero cost for model '{model}' (tried '{clean_model}') "
                    "— model may not be in LiteLLM price list."
                )

            return total

        except Exception as e:
            logger.warning(f"Cost calculation failed for model '{model}': {e}")
            return 0.0

    def get_summary(self) -> dict:
        """Get accumulated cost summary."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cache_creation_tokens": self.total_cache_creation_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "cost": self.total_cost,
            "num_calls": self.num_calls,
        }

    def format_cost_line(self) -> str | None:
        """Format a one-line cost summary for display.

        Returns None if no LLM calls were tracked.
        """
        if self.num_calls == 0:
            return None

        s = self.get_summary()
        parts = [
            f"~${s['cost']:.4f}",
            f"{s['total_tokens']:,} tokens ({s['input_tokens']:,} in / {s['output_tokens']:,} out)",
            f"{s['num_calls']} LLM call{'s' if s['num_calls'] != 1 else ''}",
        ]

        # Add cache info if we got any cache hits
        if s["cache_read_tokens"] > 0:
            parts.append(f"cache: {s['cache_read_tokens']:,} tokens read")

        return " · ".join(parts)

    def merge(self, other: "CostTracker") -> None:
        """Merge another tracker's data into this one."""
        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens
        self.total_cache_creation_tokens += other.total_cache_creation_tokens
        self.total_cache_read_tokens += other.total_cache_read_tokens
        self.total_cost += other.total_cost
        self.num_calls += other.num_calls
        self._call_details.extend(other._call_details)
