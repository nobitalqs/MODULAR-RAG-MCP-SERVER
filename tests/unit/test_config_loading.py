"""Tests for settings loading and validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from dataclasses import FrozenInstanceError

from src.core.settings import (
    Settings,
    SettingsError,
    load_settings,
    validate_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 1024
embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
vector_store:
  provider: chroma
  persist_directory: ./data/db/chroma
  collection_name: knowledge_hub
retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
  rrf_k: 60
rerank:
  enabled: false
  provider: none
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 5
evaluation:
  enabled: false
  provider: custom
  metrics:
    - hit_rate
    - mrr
observability:
  log_level: INFO
  trace_enabled: true
  trace_file: ./logs/traces.jsonl
  structured_logging: true
"""


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadSettingsSuccess:
    """Tests for successful configuration loading."""

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        settings = load_settings(settings_path)

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.llm.temperature == 0.0
        assert settings.llm.max_tokens == 1024
        assert settings.embedding.provider == "openai"
        assert settings.embedding.dimensions == 1536
        assert settings.vector_store.collection_name == "knowledge_hub"
        assert settings.retrieval.rrf_k == 60
        assert settings.rerank.enabled is False
        assert settings.rerank.provider == "none"
        assert settings.evaluation.metrics == ["hit_rate", "mrr"]
        assert settings.observability.log_level == "INFO"

    def test_optional_sections_default_to_none(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        settings = load_settings(settings_path)

        assert settings.ingestion is None
        assert settings.vision_llm is None
        assert settings.dashboard is None

    def test_load_with_ingestion_section(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        ingestion:
          chunk_size: 500
          chunk_overlap: 100
          splitter: recursive
          batch_size: 50
        """)
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.ingestion is not None
        assert settings.ingestion.chunk_size == 500
        assert settings.ingestion.chunk_overlap == 100
        assert settings.ingestion.splitter == "recursive"
        assert settings.ingestion.batch_size == 50

    def test_ingestion_defaults(self, tmp_path: Path) -> None:
        """splitter and batch_size have defaults when omitted."""
        config = MINIMAL_YAML + textwrap.dedent("""\
        ingestion:
          chunk_size: 1000
          chunk_overlap: 200
        """)
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.ingestion is not None
        assert settings.ingestion.splitter == "recursive"
        assert settings.ingestion.batch_size == 100

    def test_load_with_vision_llm(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        vision_llm:
          enabled: true
          provider: azure
          model: gpt-4o
          max_image_size: 2048
        """)
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.vision_llm is not None
        assert settings.vision_llm.enabled is True
        assert settings.vision_llm.provider == "azure"
        assert settings.vision_llm.max_image_size == 2048

    def test_load_with_dashboard(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        dashboard:
          enabled: true
          port: 8501
          traces_dir: ./logs
        """)
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.dashboard is not None
        assert settings.dashboard.enabled is True
        assert settings.dashboard.port == 8501

    def test_optional_provider_fields(self, tmp_path: Path) -> None:
        """api_key, azure_endpoint, etc. are optional."""
        config = MINIMAL_YAML.replace(
            "  provider: openai\n  model: gpt-4o-mini",
            "  provider: azure\n  model: gpt-4o\n"
            "  api_key: test-key\n  azure_endpoint: https://test.openai.azure.com",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.llm.api_key == "test-key"
        assert settings.llm.azure_endpoint == "https://test.openai.azure.com"

    def test_settings_are_frozen(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        settings = load_settings(settings_path)

        with pytest.raises(AttributeError):
            settings.llm = None  # type: ignore[misc]

    def test_load_real_config(self) -> None:
        """Load the actual config/settings.yaml from repo."""
        settings = load_settings()
        assert settings.llm.provider in ("azure", "openai", "ollama", "deepseek")
        assert settings.observability.log_level == "INFO"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadSettingsErrors:
    """Tests for configuration error handling."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(SettingsError, match="Settings file not found"):
            load_settings(tmp_path / "nonexistent.yaml")

    def test_missing_llm_section(self, tmp_path: Path) -> None:
        # Remove entire llm block (key + indented children)
        lines = MINIMAL_YAML.splitlines(keepends=True)
        filtered = []
        skip = False
        for line in lines:
            if line.startswith("llm:"):
                skip = True
                continue
            if skip and (line.startswith("  ") or line.strip() == ""):
                continue
            skip = False
            filtered.append(line)
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text("".join(filtered), encoding="utf-8")

        with pytest.raises(SettingsError, match="settings.llm"):
            load_settings(settings_path)

    def test_missing_embedding_provider(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace(
            "embedding:\n  provider: openai",
            "embedding:\n  # provider missing",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with pytest.raises(SettingsError, match="embedding.provider"):
            load_settings(settings_path)

    def test_missing_vector_store_section(self, tmp_path: Path) -> None:
        lines = MINIMAL_YAML.splitlines(keepends=True)
        filtered = []
        skip = False
        for line in lines:
            if line.startswith("vector_store:"):
                skip = True
                continue
            if skip and (line.startswith("  ") or line.strip() == ""):
                continue
            skip = False
            filtered.append(line)
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text("".join(filtered), encoding="utf-8")

        with pytest.raises(SettingsError, match="settings.vector_store"):
            load_settings(settings_path)

    def test_missing_retrieval_rrf_k(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace("  rrf_k: 60", "  # rrf_k missing")
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with pytest.raises(SettingsError, match="retrieval.rrf_k"):
            load_settings(settings_path)

    def test_missing_observability_section(self, tmp_path: Path) -> None:
        lines = MINIMAL_YAML.splitlines(keepends=True)
        filtered = []
        skip = False
        for line in lines:
            if line.startswith("observability:"):
                skip = True
                continue
            if skip and (line.startswith("  ") or line.strip() == ""):
                continue
            skip = False
            filtered.append(line)
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text("".join(filtered), encoding="utf-8")

        with pytest.raises(SettingsError, match="settings.observability"):
            load_settings(settings_path)

    def test_wrong_type_temperature(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace("  temperature: 0.0", "  temperature: hot")
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with pytest.raises(SettingsError, match="llm.temperature"):
            load_settings(settings_path)

    def test_wrong_type_max_tokens(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace("  max_tokens: 1024", "  max_tokens: lots")
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with pytest.raises(SettingsError, match="llm.max_tokens"):
            load_settings(settings_path)

    def test_wrong_type_enabled(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace("  enabled: false\n  provider: none", "  enabled: yes_please\n  provider: none")
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with pytest.raises(SettingsError, match="rerank.enabled"):
            load_settings(settings_path)

    def test_empty_yaml(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text("", encoding="utf-8")

        with pytest.raises(SettingsError, match="settings.llm"):
            load_settings(settings_path)

    def test_non_mapping_root(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(SettingsError, match="Settings root must be a mapping"):
            load_settings(settings_path)

    def test_missing_metrics_list(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace(
            "  metrics:\n    - hit_rate\n    - mrr",
            "  # metrics removed",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with pytest.raises(SettingsError, match="evaluation.metrics"):
            load_settings(settings_path)


# ---------------------------------------------------------------------------
# validate_settings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateSettings:
    """Tests for the validate_settings function."""

    def test_validate_valid_settings(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        settings = load_settings(settings_path)
        validate_settings(settings)  # should not raise


# ---------------------------------------------------------------------------
# Environment variable fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnvVarFallback:
    """Tests for env var fallback when YAML values are empty."""

    def test_azure_api_key_from_env(self, tmp_path: Path) -> None:
        """api_key falls back to AZURE_OPENAI_API_KEY env var."""
        config = MINIMAL_YAML.replace(
            "  provider: openai\n  model: gpt-4o-mini",
            "  provider: azure\n  model: gpt-4o",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with mock.patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "env-key-123"}):
            settings = load_settings(settings_path)

        assert settings.llm.api_key == "env-key-123"

    def test_azure_endpoint_from_env(self, tmp_path: Path) -> None:
        """azure_endpoint falls back to AZURE_OPENAI_ENDPOINT env var."""
        config = MINIMAL_YAML.replace(
            "  provider: openai\n  model: gpt-4o-mini",
            "  provider: azure\n  model: gpt-4o",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with mock.patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"}):
            settings = load_settings(settings_path)

        assert settings.llm.azure_endpoint == "https://test.openai.azure.com"

    def test_yaml_value_takes_priority_over_env(self, tmp_path: Path) -> None:
        """Explicit YAML value should override env var."""
        config = MINIMAL_YAML.replace(
            "  provider: openai\n  model: gpt-4o-mini",
            "  provider: azure\n  model: gpt-4o\n  api_key: yaml-key",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        with mock.patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "env-key"}):
            settings = load_settings(settings_path)

        assert settings.llm.api_key == "yaml-key"

    def test_openai_api_key_from_env(self, tmp_path: Path) -> None:
        """OpenAI provider picks up OPENAI_API_KEY env var."""
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-123"}):
            settings = load_settings(settings_path)

        assert settings.llm.api_key == "sk-test-123"

    def test_no_env_var_returns_none(self, tmp_path: Path) -> None:
        """When no env var set and YAML empty, api_key is None."""
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
        }
        with mock.patch.dict(os.environ, env_clean, clear=True):
            settings = load_settings(settings_path)

        assert settings.llm.api_key is None


# ---------------------------------------------------------------------------
# VectorStoreSettings forward compatibility
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVectorStoreForwardCompat:
    """Tests for ChromaDB 1.x compatibility fields."""

    def test_default_optional_fields(self, tmp_path: Path) -> None:
        """New optional fields default to None."""
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)

        settings = load_settings(settings_path)

        assert settings.vector_store.path is None
        assert settings.vector_store.host is None
        assert settings.vector_store.port is None

    def test_chromadb_path_field(self, tmp_path: Path) -> None:
        """ChromaDB 1.x 'path' field is parsed when present."""
        config = MINIMAL_YAML.replace(
            "  collection_name: knowledge_hub",
            "  collection_name: knowledge_hub\n  path: ./data/db/chroma_v2",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.vector_store.path == "./data/db/chroma_v2"

    def test_remote_chromadb_fields(self, tmp_path: Path) -> None:
        """Host and port for remote ChromaDB HTTP client."""
        config = MINIMAL_YAML.replace(
            "  collection_name: knowledge_hub",
            "  collection_name: knowledge_hub\n  host: chromadb.example.com\n  port: 8000",
        )
        settings_path = tmp_path / "settings.yaml"
        _write_yaml(settings_path, config)

        settings = load_settings(settings_path)

        assert settings.vector_store.host == "chromadb.example.com"
        assert settings.vector_store.port == 8000


# ---------------------------------------------------------------------------
# CacheSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCacheSettings:
    """Tests for optional cache configuration section."""

    def test_cache_defaults_to_none(self, tmp_path: Path) -> None:
        """When no cache section, settings.cache is None."""
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)
        s = load_settings(settings_path)
        assert s.cache is None

    def test_load_cache_memory_provider(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        cache:
          provider: memory
          default_ttl: 3600
          max_memory_items: 10000
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        assert s.cache is not None
        assert s.cache.provider == "memory"
        assert s.cache.default_ttl == 3600
        assert s.cache.max_memory_items == 10000
        assert s.cache.redis_url is None

    def test_load_cache_redis_provider(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        cache:
          provider: redis
          redis_url: "redis://localhost:6379/0"
          default_ttl: 1800
          max_memory_items: 5000
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        assert s.cache.provider == "redis"
        assert s.cache.redis_url == "redis://localhost:6379/0"

    def test_cache_settings_are_frozen(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        cache:
          provider: memory
          default_ttl: 3600
          max_memory_items: 10000
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        with pytest.raises(FrozenInstanceError):
            s.cache.provider = "redis"


# ---------------------------------------------------------------------------
# RateLimitSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRateLimitSettings:
    """Tests for optional rate_limit configuration section."""

    def test_rate_limit_defaults_to_none(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)
        s = load_settings(settings_path)
        assert s.rate_limit is None

    def test_load_rate_limit_settings(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        rate_limit:
          enabled: true
          provider: token_bucket
          requests_per_minute: 60
          max_concurrent: 10
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        assert s.rate_limit is not None
        assert s.rate_limit.enabled is True
        assert s.rate_limit.provider == "token_bucket"
        assert s.rate_limit.requests_per_minute == 60
        assert s.rate_limit.max_concurrent == 10
        assert s.rate_limit.tokens_per_minute is None
        assert s.rate_limit.redis_url is None

    def test_rate_limit_settings_are_frozen(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        rate_limit:
          enabled: true
          provider: token_bucket
          requests_per_minute: 60
          max_concurrent: 10
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        with pytest.raises(FrozenInstanceError):
            s.rate_limit.enabled = False


# ---------------------------------------------------------------------------
# CircuitBreakerSettings / FallbackProviderSettings on LLMSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCircuitBreakerSettings:
    """Tests for circuit_breaker and fallback_providers on LLM section."""

    def test_circuit_breaker_defaults_to_none(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)
        s = load_settings(settings_path)
        assert s.llm.circuit_breaker is None
        assert s.llm.fallback_providers is None

    def test_load_circuit_breaker_settings(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace(
            "  max_tokens: 1024",
            "  max_tokens: 1024\n"
            "  circuit_breaker:\n"
            "    enabled: true\n"
            "    failure_threshold: 3\n"
            "    cooldown_seconds: 30.0\n"
            "    half_open_max_calls: 1",
        )
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        assert s.llm.circuit_breaker is not None
        assert s.llm.circuit_breaker.enabled is True
        assert s.llm.circuit_breaker.failure_threshold == 3
        assert s.llm.circuit_breaker.cooldown_seconds == 30.0

    def test_load_fallback_providers(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace(
            "  max_tokens: 1024",
            "  max_tokens: 1024\n"
            "  fallback_providers:\n"
            "    - provider: ollama\n"
            "      model: llama3\n",
        )
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        assert s.llm.fallback_providers is not None
        assert len(s.llm.fallback_providers) == 1
        assert s.llm.fallback_providers[0].provider == "ollama"
        assert s.llm.fallback_providers[0].model == "llama3"

    def test_circuit_breaker_settings_are_frozen(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML.replace(
            "  max_tokens: 1024",
            "  max_tokens: 1024\n"
            "  circuit_breaker:\n"
            "    enabled: true\n"
            "    failure_threshold: 3\n"
            "    cooldown_seconds: 30.0\n"
            "    half_open_max_calls: 1",
        )
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        with pytest.raises(FrozenInstanceError):
            s.llm.circuit_breaker.enabled = False


# ---------------------------------------------------------------------------
# QueryRewritingSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryRewritingSettings:
    """Tests for optional query_rewriting configuration section."""

    def test_query_rewriting_defaults_to_none(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, MINIMAL_YAML)
        s = load_settings(settings_path)
        assert s.query_rewriting is None

    def test_load_query_rewriting_settings(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        query_rewriting:
          enabled: true
          provider: llm
          max_rewrites: 3
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        assert s.query_rewriting is not None
        assert s.query_rewriting.enabled is True
        assert s.query_rewriting.provider == "llm"
        assert s.query_rewriting.max_rewrites == 3
        assert s.query_rewriting.model is None

    def test_query_rewriting_settings_are_frozen(self, tmp_path: Path) -> None:
        config = MINIMAL_YAML + textwrap.dedent("""\
        query_rewriting:
          enabled: true
          provider: llm
          max_rewrites: 3
        """)
        settings_path = tmp_path / "s.yaml"
        _write_yaml(settings_path, config)
        s = load_settings(settings_path)
        with pytest.raises(FrozenInstanceError):
            s.query_rewriting.enabled = False
