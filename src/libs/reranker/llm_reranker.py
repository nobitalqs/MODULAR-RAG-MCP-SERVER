"""LLM-based Reranker implementation.

Uses a Large Language Model to score and rerank candidate passages for a
given query. The LLM is prompted to assign relevance scores to each passage.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.reranker.base_reranker import BaseReranker


class LLMRerankError(RuntimeError):
    """Raised when LLM reranking fails."""


# Default prompt template for reranking
DEFAULT_RERANK_PROMPT = """You are a relevance scoring assistant. Given a query and a list of passages, score each passage for relevance to the query.

Query: {query}

Passages:
{passages}

Return a JSON array where each element has:
- "passage_id": the passage number (as string)
- "score": relevance score from 0.0 (not relevant) to 1.0 (highly relevant)

Output only the JSON array, no other text.
"""


class LLMReranker(BaseReranker):
    """LLM-based reranker using language model for relevance scoring.

    Args:
        model: Model identifier (e.g., "gpt-4o", "gpt-3.5-turbo").
        top_k: Maximum number of results to return after reranking.
        llm: Optional BaseLLM instance. If None, must be set before calling
            rerank() (typically wired by the Core layer).
        prompt_template: Optional custom prompt template. If None, tries to
            load from config/prompts/rerank.txt, falling back to built-in.
        **kwargs: Additional arguments (ignored).

    Raises:
        LLMRerankError: If reranking fails (LLM not set, invalid response, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        top_k: int = 10,
        llm: BaseLLM | None = None,
        prompt_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.llm = llm

        # Load prompt template
        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            # Try loading from config file
            prompt_path = "config/prompts/rerank.txt"
            if os.path.exists(prompt_path):
                with open(prompt_path, encoding="utf-8") as f:
                    self.prompt_template = f.read()
            else:
                # Use built-in default
                self.prompt_template = DEFAULT_RERANK_PROMPT

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using LLM-based relevance scoring.

        Args:
            query: User query string.
            candidates: List of candidate dicts to rerank.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters (ignored).

        Returns:
            Candidates sorted by LLM-assigned relevance score, limited to top_k.

        Raises:
            LLMRerankError: If LLM is not set or returns invalid response.
            ValueError: If query or candidates are invalid.
        """
        self.validate_query(query)
        self.validate_candidates(candidates)

        # Single candidate: return as-is (no need for LLM)
        if len(candidates) == 1:
            return list(candidates)

        # Check LLM is set
        if self.llm is None:
            raise LLMRerankError(
                "[LLMReranker] LLM instance not set. Must be provided during "
                "initialization or wired by Core layer."
            )

        # Build prompt with query and candidates
        prompt = self._build_rerank_prompt(query, candidates)

        # Call LLM
        try:
            response = self.llm.chat([Message(role="user", content=prompt)], trace=trace)
        except Exception as e:
            raise LLMRerankError(f"[LLMReranker] LLM chat failed: {e}") from e

        # Parse response
        scores = self._parse_llm_response(response.content)

        # Map scores back to candidates
        scored_candidates = self._map_scores_to_candidates(candidates, scores)

        # Sort by score descending and take top_k
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[: self.top_k]

    def _build_rerank_prompt(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        """Build the reranking prompt with query and passages.

        Args:
            query: User query.
            candidates: List of candidate dicts.

        Returns:
            Formatted prompt string.
        """
        # Build passages list
        passages_lines = []
        for i, candidate in enumerate(candidates):
            # Extract text from 'text' or 'content' field
            text = candidate.get("text") or candidate.get("content") or ""
            passages_lines.append(f"Passage {i}: {text}")

        passages_str = "\n".join(passages_lines)

        # Check if template has placeholders
        if "{query}" in self.prompt_template and "{passages}" in self.prompt_template:
            # Use template with placeholders
            prompt = self.prompt_template.replace("{query}", query)
            prompt = prompt.replace("{passages}", passages_str)
        else:
            # Append query and passages to template
            prompt = f"{self.prompt_template}\n\nQuery: {query}\n\nPassages:\n{passages_str}"

        return prompt

    def _parse_llm_response(self, response_text: str) -> list[dict[str, Any]]:
        """Parse LLM response into list of {passage_id, score} dicts.

        Args:
            response_text: Raw LLM response text.

        Returns:
            List of dicts with 'passage_id' and 'score' keys.

        Raises:
            LLMRerankError: If response is not valid JSON or missing fields.
        """
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            # Remove opening fence
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Parse JSON
        try:
            scores = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMRerankError(
                f"[LLMReranker] Failed to parse JSON response: {e}\nResponse: {response_text[:200]}"
            ) from e

        # Validate structure
        if not isinstance(scores, list):
            raise LLMRerankError("[LLMReranker] Expected JSON array of scores")

        for i, item in enumerate(scores):
            if not isinstance(item, dict):
                raise LLMRerankError("[LLMReranker] Each score item must be a dict")
            if "score" not in item:
                raise LLMRerankError(f"[LLMReranker] Score item at index {i} missing 'score' field")

        return scores

    def _map_scores_to_candidates(
        self,
        candidates: list[dict[str, Any]],
        scores: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map LLM scores back to candidate dicts.

        Args:
            candidates: Original candidate dicts.
            scores: Parsed scores from LLM.

        Returns:
            List of candidate dicts with 'rerank_score' field added.
        """
        # Build a mapping from passage_id to score
        score_map = {}
        for item in scores:
            passage_id = str(item.get("passage_id", ""))
            score_map[passage_id] = float(item["score"])

        # Create new candidate dicts with scores
        result = []
        for i, candidate in enumerate(candidates):
            # Make a copy to avoid mutating original
            new_candidate = dict(candidate)
            # Look up score by index
            passage_id = str(i)
            score = score_map.get(passage_id, 0.0)
            new_candidate["rerank_score"] = score
            result.append(new_candidate)

        return result
