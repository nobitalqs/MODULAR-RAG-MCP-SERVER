"""Azure Cohere Rerank API provider.

Calls the Cohere v2/rerank endpoint on Azure AI Foundry Services
via httpx. The endpoint uses the Cohere-native API format at
``{services_endpoint}/providers/cohere/v2/rerank`` with ``api-key``
header authentication.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CohereRerankError(RuntimeError):
    """Raised when Cohere reranking fails."""


class CohereReranker(BaseReranker):
    """Azure AI Foundry Cohere Rerank reranker.

    Uses httpx to POST to the Azure AI Foundry Cohere v2/rerank endpoint.
    The endpoint URL is ``{azure_endpoint}/providers/cohere/v2/rerank``.

    Credentials are resolved from constructor args, falling back to
    AZURE_OPENAI_API_KEY and AZURE_COHERE_ENDPOINT env vars.

    Args:
        model: Model name sent in the payload (e.g., "Cohere-rerank-v4.0-fast").
        top_k: Maximum number of results to return.
        api_key: Azure API key. Falls back to AZURE_OPENAI_API_KEY env var.
        azure_endpoint: Azure AI Services endpoint URL (e.g.,
            "https://myresource.services.ai.azure.com"). Falls back to
            AZURE_COHERE_ENDPOINT, then AZURE_OPENAI_ENDPOINT env vars.
        deployment_name: Unused (kept for settings compat). Defaults to model.
        api_version: Unused (kept for settings compat). Cohere v2 API has no version param.
        **kwargs: Additional arguments (ignored).

    Raises:
        ValueError: If api_key or azure_endpoint cannot be resolved.
    """

    def __init__(
        self,
        model: str = "Cohere-rerank-v4.0-fast",
        top_k: int = 5,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.top_k = top_k

        # Resolve credentials with env var fallback
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        resolved_endpoint = (
            azure_endpoint
            or os.environ.get("AZURE_COHERE_ENDPOINT", "")
            or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        )
        self.deployment_name = deployment_name or model

        if not self.api_key:
            raise ValueError("CohereReranker requires api_key or AZURE_OPENAI_API_KEY env var")
        if not resolved_endpoint:
            raise ValueError(
                "CohereReranker requires azure_endpoint or AZURE_COHERE_ENDPOINT env var"
            )

        # Build rerank URL — Cohere v2 on Azure AI Foundry Services
        endpoint = resolved_endpoint.rstrip("/")
        self._url = f"{endpoint}/providers/cohere/v2/rerank"
        self._headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            "CohereReranker initialized: model=%s, top_k=%d, url=%s",
            self.model,
            self.top_k,
            self._url,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using Azure Cohere Rerank API.

        Args:
            query: User query string.
            candidates: List of candidate dicts with 'text' or 'content'.
            trace: Optional TraceContext (unused, for interface compat).
            **kwargs: Additional arguments (ignored).

        Returns:
            Candidates sorted by relevance score, limited to top_k.

        Raises:
            CohereRerankError: If API call fails.
            ValueError: If query or candidates are invalid.
        """
        self.validate_query(query)
        self.validate_candidates(candidates)

        # Extract text from candidates
        documents = [c.get("text") or c.get("content") or "" for c in candidates]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": self.top_k,
        }

        try:
            response = httpx.post(
                self._url,
                json=payload,
                headers=self._headers,
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise CohereRerankError(
                f"[CohereReranker] API request failed: "
                f"{e.response.status_code} - {e.response.text[:200]}"
            ) from e
        except httpx.RequestError as e:
            raise CohereRerankError(f"[CohereReranker] Network error: {e}") from e

        data = response.json()
        results = data.get("results", [])

        # Map scores back to candidates (copy to avoid mutation)
        scored = []
        for item in results:
            idx = item["index"]
            if idx < len(candidates):
                new_candidate = dict(candidates[idx])
                new_candidate["rerank_score"] = float(item["relevance_score"])
                scored.append(new_candidate)

        # Sort by score descending (API may already sort, but be explicit)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[: self.top_k]
