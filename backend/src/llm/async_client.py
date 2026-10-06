"""Async LLM client with instructor integration."""

import asyncio
from typing import Any, TypeVar

import instructor
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from .cost_calculator import (
    CostInfo,
    UsageInfo,
    calculate_cost,
    extract_usage_from_response,
)

T = TypeVar("T", bound=BaseModel)

# Without an explicit rubric, the model reads a multi-topic interest statement as
# a checklist and marks down papers that are squarely on one topic for not also
# covering the others, which pushes clearly relevant papers under the thresholds.
# SCORING_REMINDER restates the key rule at the end of the user message, where it
# isn't buried behind a long paper text in Stage 3.
SCORING_REMINDER = (
    "Score against the single closest listed interest, following the scoring guide. "
    "Do not lower the score for interests the paper does not address."
)

SCORING_GUIDE = """Scoring guide:
The user's interests are a list of alternatives, not a checklist. A paper that squarely
addresses ANY ONE listed interest is a strong match, even if it touches none of the others.
Never lower a score because the paper is narrower than the user's full set of interests or
omits other topics they mention. Lower it only for how far the paper's own subject is from
the closest listed interest, or for falling under something the user explicitly excludes.
- 0.9-1.0: The paper's main subject is one of the listed interests.
- 0.7-0.9: A listed interest is a substantial part of the paper, or the paper is a close
  neighbor (same objects, models or methods) that the user would very likely want to read.
- 0.4-0.7: Tangential: listed topics appear only as motivation, an application or an aside.
- 0.0-0.4: Unrelated, or within an explicit exclusion."""


class AsyncLLMClient:
    """
    Async LLM client with structured output using instructor.

    Supports parallel async calls for batch processing and flexible
    response schemas for different filtering stages.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        timeout: float = 60.0,
        max_concurrent: int = 10,
    ):
        """
        Initialize the async LLM client.

        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API (for custom endpoints)
            model: Model name to use
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            max_concurrent: Maximum number of concurrent requests
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # Create async OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Wrap with instructor for structured output
        self.instructor_client = instructor.from_openai(self.client)

        # Semaphore for limiting concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"AsyncLLMClient initialized: model={model}, "
            f"max_concurrent={max_concurrent}, timeout={timeout}s"
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> tuple[T, UsageInfo | None, CostInfo | None]:
        """
        Get structured completion from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model for response structure
            temperature: Optional temperature override (uses default if None)
            **kwargs: Additional arguments to pass to instructor

        Returns:
            Tuple of (response, usage_info, cost_info)
        """
        async with self.semaphore:
            try:
                # Use provided temperature or default to 0.0
                effective_temp = temperature if temperature is not None else 0.0

                # Log the conversation for debugging (without truncation)
                logger.debug(f"=== LLM Request ({response_model.__name__}) ===")
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    logger.debug(f"[{role.upper()}] {content}")

                response = await self.instructor_client.chat.completions.create(  # type: ignore
                    model=self.model,
                    messages=messages,  # type: ignore
                    response_model=response_model,
                    temperature=effective_temp,
                    **kwargs,
                )

                # Extract usage information from response
                usage = extract_usage_from_response(response)

                cost_info = calculate_cost(usage, self.model)

                # Log usage and cost
                if usage:
                    prompt = usage.get("prompt_tokens", 0)
                    completion = usage.get("completion_tokens", 0)
                    total = usage.get("total_tokens", 0)
                    if cost_info:
                        cost_str = f"{cost_info.get('currency', 'N/A')} {cost_info.get('estimated_cost', 0):.6f}"
                    else:
                        cost_str = "N/A"
                    logger.info(
                        f"LLM usage: prompt={prompt}, completion={completion}, total={total}, est_cost={cost_str}"
                    )

                # Log the response
                logger.debug(f"=== LLM Response ({response_model.__name__}) ===")
                logger.debug(f"{response}")
                logger.debug("=" * 60)

                return response, usage, cost_info  # type: ignore

            except Exception as e:
                logger.error(f"LLM completion failed: {e}")
                raise

    async def complete_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        response_model: type[T],
        **kwargs: Any,
    ) -> list[tuple[T, UsageInfo | None, CostInfo | None]]:
        """
        Get structured completions for multiple requests in parallel.

        Args:
            batch_messages: List of message lists (one per request)
            response_model: Pydantic model for response structure
            **kwargs: Additional arguments to pass to instructor

        Returns:
            List of tuples (response, usage_info, cost_info)
        """
        tasks = [self.complete(messages, response_model, **kwargs) for messages in batch_messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None to maintain list length consistency
        processed_results: list[tuple[T, UsageInfo | None, CostInfo | None] | None] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                processed_results.append(None)  # type: ignore[arg-type]
            else:
                processed_results.append(result)  # type: ignore[arg-type]

        return processed_results  # type: ignore[return-value]

    def build_stage1_messages(
        self,
        title: str,
        categories: list[str],
        user_prompt: str,
    ) -> list[dict[str, str]]:
        """
        Build messages for Stage 1 filtering (Title + Categories).

        Args:
            title: Paper title
            categories: arXiv categories
            user_prompt: User's filtering criteria

        Returns:
            List of message dicts
        """
        system_message = """You are an expert at quickly screening academic papers for relevance.
Your task is to determine if a paper is potentially relevant based ONLY on its title and categories.
This is a fast preliminary filter - be generous in passing papers that might be relevant.
Respond with a score (0-1) and a relevance statement.

""" + SCORING_GUIDE

        user_message = f"""User's interests: {user_prompt}

Paper Information:
- Title: {title}
- Categories: {", ".join(categories)}

Is this paper potentially relevant? Provide a quick assessment.
{SCORING_REMINDER}"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def build_stage2_messages(
        self,
        title: str,
        authors: list[str],
        categories: list[str],
        abstract: str,
        user_prompt: str,
    ) -> list[dict[str, str]]:
        """
        Build messages for Stage 2 filtering (Title + Authors + Categories + Abstract).

        Args:
            title: Paper title
            authors: List of author names
            categories: arXiv categories
            abstract: Paper abstract
            user_prompt: User's filtering criteria

        Returns:
            List of message dicts
        """
        system_message = """You are an expert at evaluating academic paper relevance.
Your task is to determine if a paper is relevant based on its metadata and abstract.
Provide a detailed assessment with a relevance score and reasoning.

""" + SCORING_GUIDE

        user_message = f"""User's interests: {user_prompt}

Paper Information:
- Title: {title}
- Authors: {", ".join(authors)}
- Categories: {", ".join(categories)}
- Abstract: {abstract}

Evaluate this paper's relevance to the user's interests.
{SCORING_REMINDER}"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def build_stage3_messages(
        self,
        title: str,
        authors: list[str],
        categories: list[str],
        full_text: str,
        user_prompt: str,
        custom_fields: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build messages for Stage 3 filtering (Full paper analysis).

        Args:
            title: Paper title
            authors: List of author names
            categories: arXiv categories
            full_text: Full paper text (cleaned from HTML)
            user_prompt: User's filtering criteria
            custom_fields: List of custom field dicts with 'name' and 'description'

        Returns:
            List of message dicts
        """
        system_message = """You are an expert at deeply analyzing academic papers.
Your task is to thoroughly evaluate the paper's relevance to the user's interests and
extract specific information as requested.

""" + SCORING_GUIDE

        custom_fields_prompt = ""
        if custom_fields:
            fields_list = []
            for field in custom_fields:
                field_name = field.get("name", "")
                field_desc = field.get("description", "")
                if field_name:
                    fields_list.append(f"  - {field_name}: {field_desc}")

            if fields_list:
                custom_fields_prompt = "\n\nExtract the following custom fields:\n" + "\n".join(
                    fields_list
                )
                custom_fields_prompt += (
                    "\n\nNote: The paper content includes image URLs in markdown format (![](url)). "
                    "Although you cannot view the images directly, you may include relevant figure markdown "
                    "in your custom field responses when necessary, especially if the context suggests the "
                    "figure would be valuable for users to see. However, please be selective - only include "
                    "figures that are truly essential to understanding your extracted information. "
                    "In most cases, only an image describing the methodology is needed."
                    "\n\nAdditional Instruction: Ensure the output is scannable and reader-friendly. "
                    "Avoid overusing paper-specific terms without explanation. Use bullet points or short "
                    "paragraphs for clarity."
                )

        user_message = f"""Paper Information:
- Title: {title}
- Authors: {", ".join(authors)}
- Categories: {", ".join(categories)}

Full Paper Content (partially truncated):
{full_text} (...)

User's interests: {user_prompt}

Provide a comprehensive analysis including:
1. Overall relevance score
2. Detailed reasoning for your assessment

{SCORING_REMINDER}{custom_fields_prompt}"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()
        logger.debug("AsyncLLMClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
