"""LLM Provider abstraction layer for IrisAI.

Provides a unified interface for LLM calls that supports:
- Anthropic (AsyncAnthropic) — native tool use, prompt caching, extended thinking; routes via LiteLLM proxy
- OpenAI-compatible (AsyncOpenAI) — for non-Claude models; routes via LiteLLM proxy

All providers route through LITELLM_URL. AnthropicProvider uses the Anthropic SDK request format
(preserving caching, thinking, native tools) but base_url points to the LiteLLM proxy endpoint.

Usage:
    provider = get_provider("anthropic", model_id="anthropic.claude-sonnet-4-6",
                            base_url="http://localhost:8080",
                            api_key=os.environ["LITELLM_VIRTUAL_KEY"])
    response = await provider.create_message(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class Usage:
    """Token usage from a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class ToolCall:
    """A tool call from the LLM response."""
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    stop_reason: str = ""


# ── Provider Protocol ─────────────────────────────────────────────────────────


class LLMProvider(Protocol):
    """Protocol that all providers must implement."""

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 16000,
        cache_system: bool = True,
    ) -> LLMResponse: ...

    async def create_structured(
        self,
        messages: list[dict],
        schema: Type[BaseModel],
        system: str = "",
    ) -> BaseModel: ...


# ── Anthropic Provider ────────────────────────────────────────────────────────


class AnthropicProvider:
    """Native Anthropic SDK provider via AsyncAnthropic.

    Supports prompt caching, extended thinking, and native tool use.
    Communicates with Bedrock through the LiteLLM proxy.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        temperature: float = 0,
        max_tokens: int = 16000,
        thinking_budget: int = 0,
        timeout: float = 300,
    ):
        from anthropic import AsyncAnthropic

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self._client = AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,
    ) -> LLMResponse:
        """Send a message and return the response.

        Args:
            system: System prompt text.
            messages: Conversation messages in Anthropic format.
            tools: Tool definitions in Anthropic format (from tool_converter).
            max_tokens: Override default max_tokens for this call.
            cache_system: Add cache_control to system prompt for prompt caching.
        """
        effective_max_tokens = max_tokens or self.max_tokens

        # Build system with optional cache control
        if system:
            if cache_system:
                system_param = [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                system_param = system
        else:
            system_param = None

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": effective_max_tokens,
            "messages": messages,
        }

        if system_param is not None:
            kwargs["system"] = system_param

        if tools:
            kwargs["tools"] = tools

        # Extended thinking — only when budget > 0
        # When thinking is enabled, temperature must not be set
        # and max_tokens must be > budget_tokens (Anthropic API requirement)
        if self.thinking_budget > 0:
            if effective_max_tokens <= self.thinking_budget:
                effective_max_tokens = self.thinking_budget + 4096
                kwargs["max_tokens"] = effective_max_tokens
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            kwargs["temperature"] = self.temperature

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    async def create_structured(
        self,
        messages: list[dict],
        schema: Type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        """Extract structured output by forcing a tool call.

        Uses tool_choice to force the model to call a tool whose input_schema
        matches the Pydantic model. This replaces LangChain's with_structured_output().

        The returned Pydantic model has a `_usage` attribute (Usage dataclass)
        attached for cost tracking.
        """
        tool_name = f"structured_{schema.__name__.lower()}"
        json_schema = schema.model_json_schema()
        json_schema.pop("title", None)

        tool_def = {
            "name": tool_name,
            "description": f"Return structured data matching {schema.__name__}.",
            "input_schema": json_schema,
        }

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": 4096,
            "messages": messages,
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": tool_name},
        }

        if system:
            kwargs["system"] = system

        kwargs["temperature"] = self.temperature

        response = await self._client.messages.create(**kwargs)

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )

        # Extract the tool call input and parse into Pydantic model
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                result = schema.model_validate(block.input)
                result._usage = usage
                return result

        raise ValueError(
            f"Model did not return expected tool call '{tool_name}'. "
            f"Stop reason: {response.stop_reason}"
        )

    def _parse_response(self, response) -> LLMResponse:
        """Convert Anthropic SDK response to LLMResponse."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )

        return LLMResponse(
            content="\n".join(content_parts),
            thinking="\n".join(thinking_parts),
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            stop_reason=response.stop_reason,
        )


# ── OpenAI Provider (for non-Claude models) ──────────────────────────────────


class OpenAIProvider:
    """OpenAI-compatible provider for non-Claude models via LiteLLM proxy.

    Supports standard chat completions and tool use.
    Does NOT support prompt caching or extended thinking (gracefully ignored).
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        temperature: float = 0,
        max_tokens: int = 16000,
        timeout: float = 300,
    ):
        from openai import AsyncOpenAI

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,  # ignored for OpenAI
    ) -> LLMResponse:
        """Send a message via OpenAI-compatible API."""
        effective_max_tokens = max_tokens or self.max_tokens

        # Convert Anthropic-format messages to OpenAI format
        converted_messages = self._convert_anthropic_to_openai_messages(messages)

        # Build messages with system prepended
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(converted_messages)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": effective_max_tokens,
            "messages": api_messages,
            "temperature": self.temperature,
        }

        if tools:
            # Deduplicate by name before converting to OpenAI format
            seen_names: set[str] = set()
            deduped: list[dict] = []
            for t in tools:
                if t["name"] not in seen_names:
                    seen_names.add(t["name"])
                    deduped.append(t)
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
                for t in deduped
            ]

        response = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    async def create_structured(
        self,
        messages: list[dict],
        schema: Type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        """Extract structured output via function calling."""
        import json

        tool_name = f"structured_{schema.__name__.lower()}"
        json_schema = schema.model_json_schema()
        json_schema.pop("title", None)

        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        response = await self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=4096,
            messages=api_messages,
            temperature=self.temperature,
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Return structured data matching {schema.__name__}.",
                    "parameters": json_schema,
                },
            }],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            raw_args = choice.message.tool_calls[0].function.arguments
            return schema.model_validate(json.loads(raw_args))

        raise ValueError(
            f"Model did not return expected function call '{tool_name}'. "
            f"Finish reason: {choice.finish_reason}"
        )

    def _convert_anthropic_to_openai_messages(self, messages: list[dict]) -> list[dict]:
        """Convert Anthropic-format messages to OpenAI-format messages.

        The NativeAgentExecutor builds messages in Anthropic format:
        - Assistant: {"role": "assistant", "content": [{"type": "tool_use", ...}, {"type": "text", ...}]}
        - User (tool results): {"role": "user", "content": [{"type": "tool_result", ...}]}

        OpenAI expects:
        - Assistant: {"role": "assistant", "content": "...", "tool_calls": [...]}
        - Tool results: {"role": "tool", "tool_call_id": "...", "content": "..."}
        """
        import json

        converted: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Simple string content — pass through as-is
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            # Content is a list of blocks — needs conversion
            if not isinstance(content, list):
                converted.append({"role": role, "content": str(content)})
                continue

            if role == "assistant":
                # Extract text and tool_use blocks
                text_parts: list[str] = []
                tool_calls_out: list[dict] = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls_out.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        })

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls_out:
                    assistant_msg["tool_calls"] = tool_calls_out
                converted.append(assistant_msg)

            elif role == "user":
                # User messages may contain text blocks and/or tool_result blocks
                text_parts_u: list[str] = []
                tool_results_out: list[dict] = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts_u.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tool_results_out.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", "") or "",
                        })

                # Emit tool result messages first (OpenAI expects them right after assistant)
                for tr in tool_results_out:
                    converted.append(tr)
                # Then emit user text if any
                if text_parts_u:
                    converted.append({"role": "user", "content": "\n".join(text_parts_u)})

            else:
                # system or other roles — pass through
                converted.append({"role": role, "content": str(content)})

        return converted

    _STOP_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }

    def _parse_response(self, response) -> LLMResponse:
        """Convert OpenAI response to LLMResponse."""
        import json

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        usage_obj = response.usage
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )

        raw_stop = choice.finish_reason or ""
        stop_reason = self._STOP_REASON_MAP.get(raw_stop, raw_stop)

        return LLMResponse(
            content=message.content or "",
            thinking="",
            tool_calls=tool_calls,
            usage=usage,
            model=response.model or self.model_id,
            stop_reason=stop_reason,
        )


# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # Anthropic models (use AnthropicProvider)
    "anthropic.claude-opus-4-6-v1": {"provider": "anthropic", "thinking_budget": 10000, "max_tokens": 32000},
    "anthropic.claude-sonnet-4-6": {"provider": "anthropic", "thinking_budget": 5000, "max_tokens": 16000},
    "anthropic.claude-haiku-4-5-20251001-v1": {"provider": "anthropic", "thinking_budget": 0, "max_tokens": 4096},
    # OpenAI-compatible models (use OpenAIProvider)
    "openai.gpt-oss-20b-1": {"provider": "openai", "thinking_budget": 0, "max_tokens": 8192},
    "openai.gpt-oss-120b-1": {"provider": "openai", "thinking_budget": 0, "max_tokens": 16000},
    "nvidia.nemotron-super-3-120b": {"provider": "openai", "thinking_budget": 0, "max_tokens": 16000},
}


# ── Factory ───────────────────────────────────────────────────────────────────


def get_provider(
    provider_type: str,
    model_id: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0,
    max_tokens: int = 16000,
    thinking_budget: int = 0,
    timeout: float = 300,
) -> AnthropicProvider | OpenAIProvider:
    """Factory to create an LLM provider.

    Args:
        provider_type: "anthropic" or "openai".
        model_id: Model identifier (e.g. "anthropic.claude-sonnet-4-6").
        base_url: LLM API base URL. Defaults to LITELLM_URL env var.
        api_key: API key. Defaults to LITELLM_VIRTUAL_KEY env var.
        temperature: Sampling temperature.
        max_tokens: Default max output tokens.
        thinking_budget: Extended thinking token budget (0 = disabled, Anthropic only).
        timeout: Request timeout in seconds.
    """
    effective_url = base_url or os.environ.get("LITELLM_URL", "http://localhost:8080")
    effective_key = api_key or os.environ.get("LITELLM_VIRTUAL_KEY", "")

    if not effective_key:
        raise RuntimeError("No API key provided and LITELLM_VIRTUAL_KEY not set")

    if provider_type == "anthropic":
        return AnthropicProvider(
            model_id=model_id,
            base_url=effective_url,
            api_key=effective_key,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            timeout=timeout,
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            model_id=model_id,
            base_url=effective_url,
            api_key=effective_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider_type: {provider_type!r}. Use 'anthropic' or 'openai'.")


def get_provider_for_model(
    model_id: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
    timeout: float = 300,
) -> AnthropicProvider | OpenAIProvider:
    """Auto-detect provider type from MODEL_REGISTRY and create the appropriate provider.

    Falls back to "anthropic" provider for unknown model IDs.
    Uses registry defaults for max_tokens and thinking_budget if not explicitly provided.
    """
    registry_entry = MODEL_REGISTRY.get(model_id, {})
    provider_type = registry_entry.get("provider", "anthropic")
    effective_max_tokens = max_tokens if max_tokens is not None else registry_entry.get("max_tokens", 16000)
    effective_thinking = thinking_budget if thinking_budget is not None else registry_entry.get("thinking_budget", 0)

    return get_provider(
        provider_type=provider_type,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=effective_max_tokens,
        thinking_budget=effective_thinking,
        timeout=timeout,
    )
