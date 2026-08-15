"""Cost calculation utilities for LLM usage."""

from typing import TypedDict


class UsageInfo(TypedDict, total=False):
    """Token usage information."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class CostInfo(TypedDict, total=False):
    """Cost information."""

    estimated_cost: float
    currency: str


# Per-million-token pricing for models this project has been run with. Add an
# entry here when switching to a new provider/model; unknown models simply skip
# cost estimation (usage/token counts are still logged) rather than guessing.
_MODEL_PRICING: dict[str, tuple[float, float, str]] = {
    # model prefix -> (input $/MTok, output $/MTok, currency)
    "claude-haiku-4-5": (1.0, 5.0, "USD"),
    "deepseek-chat": (2.0, 3.0, "CNY"),
}


def calculate_cost(usage: UsageInfo | None, model: str) -> CostInfo | None:
    """
    Estimate cost for a completion given token usage and the model name.

    Args:
        usage: Usage information with token counts
        model: Model name/string as configured in llm.model

    Returns:
        Cost information with estimated cost, or None if usage or pricing is unavailable
    """
    if not usage:
        return None

    pricing = next((p for prefix, p in _MODEL_PRICING.items() if model.startswith(prefix)), None)
    if pricing is None:
        return None

    input_price, output_price, currency = pricing
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0

    prompt_cost = (prompt_tokens / 1_000_000) * input_price
    completion_cost = (completion_tokens / 1_000_000) * output_price
    total_cost = prompt_cost + completion_cost

    return {
        "estimated_cost": total_cost,
        "currency": currency,
    }


def extract_usage_from_response(response: object) -> UsageInfo | None:
    """
    Extract usage information from LLM response.

    Attempts to extract usage from instructor-wrapped OpenAI response.
    Returns None if usage information is not available.

    Args:
        response: Response object from instructor/OpenAI

    Returns:
        Usage information dict or None
    """
    # Try to get raw response from instructor wrapper
    raw = getattr(response, "_raw_response", None)
    if not raw:
        return None

    usage = None
    try:
        # raw may be a dict-like or have a usage attribute
        if isinstance(raw, dict):
            usage = raw.get("usage")
        else:
            usage = getattr(raw, "usage", None)
    except Exception:
        return None

    if not usage:
        return None

    # Normalize to UsageInfo dict
    try:
        # Handle both dict and object (Pydantic model) types
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        else:
            # Pydantic object or similar
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        return {
            "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
            "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
            "total_tokens": int(total_tokens) if total_tokens is not None else None,
        }
    except (ValueError, TypeError, AttributeError):
        return None
