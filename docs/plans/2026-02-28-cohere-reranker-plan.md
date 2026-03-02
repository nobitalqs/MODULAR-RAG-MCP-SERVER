# Cohere Reranker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Azure Cohere Rerank (v4.0-fast) as a pluggable reranker provider, and fix the production gap where `create_core_reranker()` bypasses `RerankerFactory`.

**Architecture:** New `CohereReranker` class extends `BaseReranker`, uses `httpx` (already in project) to call Azure's `/v1/rerank` endpoint. `RerankSettings` is extended with Azure credential fields. `create_core_reranker()` is updated to use `RerankerFactory` for provider instantiation.

**Tech Stack:** httpx (existing), Azure Cohere Rerank v1/rerank API, Registry Pattern (RerankerFactory)

---

### Task 1: CohereReranker Unit Tests (RED)

**Files:**
- Create: `tests/unit/test_cohere_reranker.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_cohere_reranker.py`:

```python
"""Tests for CohereReranker — Azure Cohere Rerank API provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCohereRerankerInit:
    """Constructor validation tests."""

    def test_raises_without_api_key(self):
        """Missing api_key and no env var raises ValueError."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="api_key"):
                CohereReranker(
                    model="Cohere-rerank-v4.0-fast",
                    api_key="",
                    azure_endpoint="https://example.com",
                )

    def test_raises_without_azure_endpoint(self):
        """Missing azure_endpoint and no env var raises ValueError."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="azure_endpoint"):
                CohereReranker(
                    model="Cohere-rerank-v4.0-fast",
                    api_key="test-key",
                    azure_endpoint="",
                )

    def test_builds_correct_url(self):
        """URL is constructed from endpoint + deployment + api_version."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(
            model="Cohere-rerank-v4.0-fast",
            api_key="test-key",
            azure_endpoint="https://myresource.cognitiveservices.azure.com",
            deployment_name="Cohere-rerank-v4.0-fast",
            api_version="2024-05-01-preview",
        )
        assert "myresource.cognitiveservices.azure.com" in reranker._url
        assert "Cohere-rerank-v4.0-fast" in reranker._url
        assert "api-version=2024-05-01-preview" in reranker._url

    def test_deployment_name_defaults_to_model(self):
        """When deployment_name is not set, it defaults to model."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(
            model="Cohere-rerank-v4.0-fast",
            api_key="test-key",
            azure_endpoint="https://example.com",
        )
        assert reranker.deployment_name == "Cohere-rerank-v4.0-fast"

    def test_reads_env_vars_as_fallback(self):
        """api_key and azure_endpoint fall back to env vars."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        env = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env-endpoint.azure.com",
        }
        with patch.dict("os.environ", env, clear=True):
            reranker = CohereReranker(model="test-model")
            assert reranker.api_key == "env-key"
            assert "env-endpoint.azure.com" in reranker._url


class TestCohereRerankerRerank:
    """Rerank method tests with mocked HTTP."""

    def _make_reranker(self):
        """Create a CohereReranker with test credentials."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        return CohereReranker(
            model="Cohere-rerank-v4.0-fast",
            top_k=2,
            api_key="test-key",
            azure_endpoint="https://example.com",
        )

    def test_rerank_sends_correct_payload(self):
        """HTTP request contains query, documents, and top_n."""
        reranker = self._make_reranker()
        candidates = [
            {"id": "c1", "text": "Python is great"},
            {"id": "c2", "text": "Java is verbose"},
            {"id": "c3", "text": "Rust is fast"},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.80},
            ]
        }

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response) as mock_post:
            reranker.rerank("best programming language", candidates)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["query"] == "best programming language"
            assert payload["documents"] == [
                "Python is great",
                "Java is verbose",
                "Rust is fast",
            ]
            assert payload["top_n"] == 2

    def test_rerank_returns_scored_candidates(self):
        """Returned candidates have rerank_score and correct ordering."""
        reranker = self._make_reranker()
        candidates = [
            {"id": "c1", "text": "Python is great", "score": 0.5},
            {"id": "c2", "text": "Java is verbose", "score": 0.7},
            {"id": "c3", "text": "Rust is fast", "score": 0.3},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.10},
            ]
        }

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response):
            result = reranker.rerank("fast language", candidates)

        # top_k=2, so only 2 results
        assert len(result) == 2
        # First result should be highest score (index 2 = Rust)
        assert result[0]["id"] == "c3"
        assert result[0]["rerank_score"] == 0.95
        # Second result (index 0 = Python)
        assert result[1]["id"] == "c1"
        assert result[1]["rerank_score"] == 0.80

    def test_rerank_does_not_mutate_original_candidates(self):
        """Original candidate dicts are not modified."""
        reranker = self._make_reranker()
        candidates = [
            {"id": "c1", "text": "Hello"},
            {"id": "c2", "text": "World"},
        ]
        original_keys = [set(c.keys()) for c in candidates]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ]
        }

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response):
            reranker.rerank("test query", candidates)

        # Original candidates should be unchanged
        for c, orig_keys in zip(candidates, original_keys):
            assert set(c.keys()) == orig_keys

    def test_rerank_raises_on_http_error(self):
        """HTTP error raises CohereRerankError."""
        import httpx as httpx_mod

        from src.libs.reranker.cohere_reranker import CohereRerankError

        reranker = self._make_reranker()
        candidates = [{"id": "c1", "text": "Hello"}]

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response,
        )

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response):
            with pytest.raises(CohereRerankError, match="Cohere"):
                reranker.rerank("test", candidates)

    def test_rerank_validates_empty_query(self):
        """Empty query raises ValueError."""
        reranker = self._make_reranker()
        with pytest.raises(ValueError, match="empty"):
            reranker.rerank("", [{"id": "c1", "text": "Hello"}])

    def test_rerank_validates_empty_candidates(self):
        """Empty candidates list raises ValueError."""
        reranker = self._make_reranker()
        with pytest.raises(ValueError, match="empty"):
            reranker.rerank("test", [])

    def test_uses_content_field_fallback(self):
        """Falls back to 'content' field when 'text' is missing."""
        reranker = self._make_reranker()
        candidates = [{"id": "c1", "content": "Using content field"}]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response) as mock_post:
            reranker.rerank("test", candidates)
            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["documents"] == ["Using content field"]


class TestCohereRerankerFactory:
    """Factory integration tests."""

    def test_factory_can_create(self):
        """CohereReranker can be registered and created via factory."""
        from src.libs.reranker.cohere_reranker import CohereReranker
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()
        factory.register_provider("cohere", CohereReranker)
        reranker = factory.create(
            "cohere",
            model="Cohere-rerank-v4.0-fast",
            api_key="test-key",
            azure_endpoint="https://example.com",
        )
        assert isinstance(reranker, CohereReranker)

    def test_is_base_reranker_subclass(self):
        """CohereReranker is a proper BaseReranker subclass."""
        from src.libs.reranker.base_reranker import BaseReranker
        from src.libs.reranker.cohere_reranker import CohereReranker

        assert issubclass(CohereReranker, BaseReranker)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cohere_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.libs.reranker.cohere_reranker'`

**Step 3: Commit**

```bash
git add tests/unit/test_cohere_reranker.py
git commit -m "test: add CohereReranker unit tests (RED)"
```

---

### Task 2: CohereReranker Implementation (GREEN)

**Files:**
- Create: `src/libs/reranker/cohere_reranker.py`
- Modify: `src/libs/reranker/__init__.py:12-23`

**Step 1: Create CohereReranker**

Create `src/libs/reranker/cohere_reranker.py`:

```python
"""Azure Cohere Rerank API provider.

Calls the Cohere v1/rerank endpoint deployed on Azure AI Services
via httpx. Follows the same pattern as other Azure providers
(AzureLLM, AzureEmbedding) — credentials from constructor or env vars.
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
    """Azure Cohere Rerank API reranker.

    Uses httpx to POST to the Azure-deployed Cohere Rerank endpoint.
    Credentials are resolved from constructor args, falling back to
    AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars.

    Args:
        model: Model/deployment name (e.g., "Cohere-rerank-v4.0-fast").
        top_k: Maximum number of results to return.
        api_key: Azure API key. Falls back to AZURE_OPENAI_API_KEY env var.
        azure_endpoint: Azure endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT.
        deployment_name: Azure deployment name. Defaults to model.
        api_version: Azure API version string.
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
        api_version: str = "2024-05-01-preview",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.api_version = api_version

        # Resolve credentials with env var fallback
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        resolved_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.deployment_name = deployment_name or model

        if not self.api_key:
            raise ValueError(
                "CohereReranker requires api_key or AZURE_OPENAI_API_KEY env var"
            )
        if not resolved_endpoint:
            raise ValueError(
                "CohereReranker requires azure_endpoint or AZURE_OPENAI_ENDPOINT env var"
            )

        # Build rerank URL
        endpoint = resolved_endpoint.rstrip("/")
        self._url = (
            f"{endpoint}/openai/deployments/{self.deployment_name}"
            f"/v1/rerank?api-version={self.api_version}"
        )
        self._headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            "CohereReranker initialized: deployment=%s, top_k=%d",
            self.deployment_name,
            self.top_k,
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
        documents = [
            c.get("text") or c.get("content") or ""
            for c in candidates
        ]

        payload = {
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
            raise CohereRerankError(
                f"[CohereReranker] Network error: {e}"
            ) from e

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
```

**Step 2: Update `__init__.py` exports**

Modify `src/libs/reranker/__init__.py`. Add `CohereReranker` import and export.

Replace lines 1-23 with:

```python
"""
Reranker - Result reranking abstraction.

Components:
- BaseReranker: Abstract base class
- NoneReranker: Pass-through (no reranking)
- LLMReranker: LLM-based reranking
- CrossEncoderReranker: Cross-encoder model reranking
- CohereReranker: Azure Cohere Rerank API
- RerankerFactory: Backend routing factory
"""

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.cohere_reranker import CohereReranker
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.reranker_factory import RerankerFactory

__all__ = [
    "BaseReranker",
    "NoneReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "RerankerFactory",
]
```

**Step 3: Run tests to verify they pass**

Run: `pytest tests/unit/test_cohere_reranker.py -v`
Expected: ALL PASS (12 tests)

**Step 4: Run existing reranker tests for regression**

Run: `pytest tests/unit/test_reranker_factory.py tests/unit/test_cross_encoder_reranker.py tests/unit/test_llm_reranker.py -v`
Expected: ALL PASS (no regression)

**Step 5: Commit**

```bash
git add src/libs/reranker/cohere_reranker.py src/libs/reranker/__init__.py
git commit -m "feat: add CohereReranker for Azure Cohere Rerank API"
```

---

### Task 3: Extend RerankSettings with Azure Credential Fields

**Files:**
- Modify: `src/core/settings.py:104-116` (provider maps)
- Modify: `src/core/settings.py:211-216` (RerankSettings dataclass)
- Modify: `src/core/settings.py:361-366` (Settings.from_dict rerank parsing)

**Step 1: Add "cohere" to env var provider maps**

In `src/core/settings.py`, modify `_PROVIDER_KEY_MAP` (line 104-108) to add cohere:

```python
_PROVIDER_KEY_MAP: dict[str, str] = {
    "azure": "AZURE_OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "AZURE_OPENAI_API_KEY",
}
```

Modify `_PROVIDER_ENDPOINT_MAP` (line 110-112) to add cohere:

```python
_PROVIDER_ENDPOINT_MAP: dict[str, str] = {
    "azure": "AZURE_OPENAI_ENDPOINT",
    "cohere": "AZURE_OPENAI_ENDPOINT",
}
```

**Step 2: Extend RerankSettings dataclass**

Modify `RerankSettings` (lines 211-216) to add optional Azure fields:

```python
@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int
    api_key: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None
```

**Step 3: Update Settings.from_dict rerank parsing**

Modify the rerank parsing block (lines 361-366) to resolve new fields:

```python
            rerank_provider = _require_str(rerank, "provider", "rerank")
```

Then change the `RerankSettings(...)` construction:

```python
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=rerank_provider,
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
                api_key=_resolve_api_key(rerank, rerank_provider),
                azure_endpoint=_resolve_azure_endpoint(rerank, rerank_provider),
                deployment_name=rerank.get("deployment_name"),
                api_version=rerank.get("api_version"),
            ),
```

**Step 4: Run settings tests for regression**

Run: `pytest tests/unit/test_settings.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/settings.py
git commit -m "feat: extend RerankSettings with Azure credential fields"
```

---

### Task 4: Wire RerankerFactory in `create_core_reranker()`

**Files:**
- Modify: `src/core/query_engine/reranker.py:16-27` (imports)
- Modify: `src/core/query_engine/reranker.py:153-167` (_get_reranker_type)
- Modify: `src/core/query_engine/reranker.py:378-391` (create_core_reranker)

**Step 1: Add imports at top of file**

In `src/core/query_engine/reranker.py`, after the existing import block (line 24), add the factory and provider imports. The new import section should look like:

```python
from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory
```

**Step 2: Update `_get_reranker_type()` to detect Cohere**

Modify `_get_reranker_type()` (lines 153-167) to add Cohere detection before the else clause:

```python
    def _get_reranker_type(self) -> str:
        """Get the type name of the current reranker backend."""
        class_name = self._reranker.__class__.__name__
        if "Cohere" in class_name:
            return "cohere"
        elif "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()
```

**Step 3: Update `create_core_reranker()` to use factory**

Replace `create_core_reranker()` (lines 378-391) with:

```python
def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """Factory function to create a CoreReranker instance.

    When no reranker is injected, uses RerankerFactory to create one
    from settings. Registers all known providers automatically.

    Args:
        settings: Application settings.
        reranker: Optional reranker backend override.

    Returns:
        Configured CoreReranker instance.
    """
    if reranker is None and settings.rerank.enabled:
        provider = settings.rerank.provider.lower()
        if provider not in ("none", ""):
            try:
                factory = RerankerFactory()
                # Lazy imports to avoid circular deps and optional deps
                from src.libs.reranker.cohere_reranker import CohereReranker
                from src.libs.reranker.cross_encoder_reranker import (
                    CrossEncoderReranker,
                )
                from src.libs.reranker.llm_reranker import LLMReranker

                factory.register_provider("cross_encoder", CrossEncoderReranker)
                factory.register_provider("llm", LLMReranker)
                factory.register_provider("cohere", CohereReranker)
                reranker = factory.create_from_settings(settings.rerank)
            except Exception:
                logger.warning(
                    "Failed to create reranker from factory, "
                    "falling back to NoneReranker",
                    exc_info=True,
                )

    return CoreReranker(settings=settings, reranker=reranker)
```

**Step 4: Run CoreReranker tests for regression**

Run: `pytest tests/unit/test_core_reranker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/query_engine/reranker.py
git commit -m "feat: wire RerankerFactory into create_core_reranker"
```

---

### Task 5: Update settings.yaml and Full Regression

**Files:**
- Modify: `config/settings.yaml:64-69` (rerank section)

**Step 1: Update settings.yaml**

Modify the rerank section (lines 64-69) to:

```yaml
# =============================================================================
# Rerank Configuration
# =============================================================================
rerank:
  enabled: true
  provider: "cohere"  # Options: none, cross_encoder, llm, cohere
  model: "Cohere-rerank-v4.0-fast"
  deployment_name: "Cohere-rerank-v4.0-fast"
  api_version: "2024-05-01-preview"
  top_k: 5
  # api_key / azure_endpoint: loaded from env vars (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All previously passing tests still pass. The 1 pre-existing failure (`test_llm_responds` — Azure deployment issue) is unrelated.

**Step 3: Run ruff**

Run: `ruff check src/libs/reranker/ src/core/settings.py src/core/query_engine/reranker.py --fix`
Run: `ruff format src/libs/reranker/ src/core/settings.py src/core/query_engine/reranker.py`

**Step 4: Post-ruff regression**

Run: `pytest tests/unit/test_cohere_reranker.py tests/unit/test_reranker_factory.py tests/unit/test_core_reranker.py tests/unit/test_settings.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add config/settings.yaml
git commit -m "feat: configure Cohere reranker in settings.yaml"
```
