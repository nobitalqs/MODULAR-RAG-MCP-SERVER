"""Configuration loading and validation for the Modular RAG MCP Server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is.
    Otherwise it is resolved against REPO_ROOT.
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """Raised when settings validation fails."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_mapping(data: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: dict[str, Any], key: str, path: str) -> list[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


# ---------------------------------------------------------------------------
# Environment variable fallback helpers
# ---------------------------------------------------------------------------

# Maps (provider, field) → env var name.
# Priority: YAML explicit value > env var > None
_ENV_FALLBACKS: dict[str, str] = {
    "AZURE_OPENAI_API_KEY": "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT": "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
    "OLLAMA_BASE_URL": "OLLAMA_BASE_URL",
}

_PROVIDER_KEY_MAP: dict[str, str] = {
    "azure": "AZURE_OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "AZURE_OPENAI_API_KEY",
}

_PROVIDER_ENDPOINT_MAP: dict[str, str] = {
    "azure": "AZURE_OPENAI_ENDPOINT",
    "cohere": "AZURE_OPENAI_ENDPOINT",
}

_PROVIDER_BASE_URL_MAP: dict[str, str] = {
    "ollama": "OLLAMA_BASE_URL",
}


def _env_fallback(
    data: dict[str, Any],
    field: str,
    env_var: str | None,
) -> str | None:
    """Return YAML value if non-empty, else fall back to env var."""
    value = data.get(field)
    if value and isinstance(value, str) and value.strip():
        return value
    if env_var:
        return os.environ.get(env_var) or None
    return None


def _resolve_api_key(data: dict[str, Any], provider: str) -> str | None:
    """Resolve api_key from YAML or env var based on provider."""
    env_var = _PROVIDER_KEY_MAP.get(provider)
    return _env_fallback(data, "api_key", env_var)


def _resolve_azure_endpoint(data: dict[str, Any], provider: str) -> str | None:
    """Resolve azure_endpoint from YAML or env var."""
    env_var = _PROVIDER_ENDPOINT_MAP.get(provider)
    return _env_fallback(data, "azure_endpoint", env_var)


def _resolve_base_url(data: dict[str, Any], provider: str) -> str | None:
    """Resolve base_url from YAML or env var."""
    env_var = _PROVIDER_BASE_URL_MAP.get(provider)
    return _env_fallback(data, "base_url", env_var)


# ---------------------------------------------------------------------------
# Settings dataclasses (frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitBreakerSettings:
    enabled: bool
    failure_threshold: int
    cooldown_seconds: float
    half_open_max_calls: int = 1


@dataclass(frozen=True)
class FallbackProviderSettings:
    provider: str
    model: str
    api_key: str | None = None
    azure_endpoint: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key: str | None = None
    api_version: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    base_url: str | None = None
    circuit_breaker: CircuitBreakerSettings | None = None
    fallback_providers: list[FallbackProviderSettings] | None = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    api_key: str | None = None
    api_version: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class VisionLLMSettings:
    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: str | None = None
    api_version: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str
    path: str | None = None  # ChromaDB 1.x uses 'path' instead of persist_directory
    host: str | None = None  # Remote ChromaDB HTTP client
    port: int | None = None  # Remote ChromaDB HTTP client


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


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


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: list[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str = "recursive"
    batch_size: int = 100
    chunk_refiner: dict[str, Any] | None = None
    metadata_enricher: dict[str, Any] | None = None


@dataclass(frozen=True)
class DashboardSettings:
    enabled: bool
    port: int
    traces_dir: str


@dataclass(frozen=True)
class CacheSettings:
    provider: str             # "memory" | "redis"
    default_ttl: int          # seconds
    max_memory_items: int     # LRU cap for in-memory provider
    redis_url: str | None = None


@dataclass(frozen=True)
class RateLimitSettings:
    enabled: bool
    provider: str             # "token_bucket" | "redis"
    requests_per_minute: int
    max_concurrent: int
    tokens_per_minute: int | None = None
    redis_url: str | None = None  # reuses cache.redis_url if not set


@dataclass(frozen=True)
class QueryRewritingSettings:
    enabled: bool
    provider: str          # "none" | "llm" | "hyde"
    max_rewrites: int
    model: str | None = None


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool
    provider: str           # "memory" | "redis"
    max_turns: int
    summarize_threshold: int
    summarize_enabled: bool
    session_ttl: int


@dataclass(frozen=True)
class Settings:
    """Root settings container. Immutable after construction."""

    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings | None = None
    vision_llm: VisionLLMSettings | None = None
    dashboard: DashboardSettings | None = None
    cache: CacheSettings | None = None
    rate_limit: RateLimitSettings | None = None
    query_rewriting: QueryRewritingSettings | None = None
    memory: MemorySettings | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        """Parse a raw YAML dict into a validated Settings instance."""
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")

        # Optional sections
        ingestion_settings = None
        if "ingestion" in data:
            ing = _require_mapping(data, "ingestion", "settings")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ing, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ing, "chunk_overlap", "ingestion"),
                splitter=ing.get("splitter", "recursive"),
                batch_size=ing.get("batch_size", 100),
                chunk_refiner=ing.get("chunk_refiner"),
                metadata_enricher=ing.get("metadata_enricher"),
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vlm = _require_mapping(data, "vision_llm", "settings")
            vlm_provider = _require_str(vlm, "provider", "vision_llm")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vlm, "enabled", "vision_llm"),
                provider=vlm_provider,
                model=_require_str(vlm, "model", "vision_llm"),
                max_image_size=_require_int(vlm, "max_image_size", "vision_llm"),
                api_key=_resolve_api_key(vlm, vlm_provider),
                api_version=vlm.get("api_version"),
                azure_endpoint=_resolve_azure_endpoint(vlm, vlm_provider),
                deployment_name=vlm.get("deployment_name"),
                base_url=_resolve_base_url(vlm, vlm_provider),
            )

        dashboard_settings = None
        if "dashboard" in data:
            dash = _require_mapping(data, "dashboard", "settings")
            dashboard_settings = DashboardSettings(
                enabled=_require_bool(dash, "enabled", "dashboard"),
                port=_require_int(dash, "port", "dashboard"),
                traces_dir=_require_str(dash, "traces_dir", "dashboard"),
            )

        cache_settings = None
        if "cache" in data:
            cache_data = _require_mapping(data, "cache", "settings")
            cache_settings = CacheSettings(
                provider=_require_str(cache_data, "provider", "cache"),
                default_ttl=_require_int(cache_data, "default_ttl", "cache"),
                max_memory_items=_require_int(cache_data, "max_memory_items", "cache"),
                redis_url=cache_data.get("redis_url"),
            )

        rate_limit_settings = None
        if "rate_limit" in data:
            rl = _require_mapping(data, "rate_limit", "settings")
            rate_limit_settings = RateLimitSettings(
                enabled=_require_bool(rl, "enabled", "rate_limit"),
                provider=_require_str(rl, "provider", "rate_limit"),
                requests_per_minute=_require_int(rl, "requests_per_minute", "rate_limit"),
                max_concurrent=_require_int(rl, "max_concurrent", "rate_limit"),
                tokens_per_minute=rl.get("tokens_per_minute"),
                redis_url=rl.get("redis_url"),
            )

        query_rewriting_settings = None
        if "query_rewriting" in data:
            qr = _require_mapping(data, "query_rewriting", "settings")
            query_rewriting_settings = QueryRewritingSettings(
                enabled=_require_bool(qr, "enabled", "query_rewriting"),
                provider=_require_str(qr, "provider", "query_rewriting"),
                max_rewrites=_require_int(qr, "max_rewrites", "query_rewriting"),
                model=qr.get("model"),
            )

        memory_settings = None
        if "memory" in data:
            mem = _require_mapping(data, "memory", "settings")
            memory_settings = MemorySettings(
                enabled=_require_bool(mem, "enabled", "memory"),
                provider=_require_str(mem, "provider", "memory"),
                max_turns=_require_int(mem, "max_turns", "memory"),
                summarize_threshold=_require_int(mem, "summarize_threshold", "memory"),
                summarize_enabled=_require_bool(mem, "summarize_enabled", "memory"),
                session_ttl=_require_int(mem, "session_ttl", "memory"),
            )

        # Parse circuit_breaker sub-section of llm
        cb_settings = None
        if "circuit_breaker" in llm:
            cb_data = _require_mapping(llm, "circuit_breaker", "llm")
            cb_settings = CircuitBreakerSettings(
                enabled=_require_bool(cb_data, "enabled", "llm.circuit_breaker"),
                failure_threshold=_require_int(cb_data, "failure_threshold", "llm.circuit_breaker"),
                cooldown_seconds=_require_number(cb_data, "cooldown_seconds", "llm.circuit_breaker"),
                half_open_max_calls=cb_data.get("half_open_max_calls", 1),
            )

        # Parse fallback_providers sub-section of llm
        fallback_list = None
        if "fallback_providers" in llm:
            raw_fallbacks = llm["fallback_providers"]
            if isinstance(raw_fallbacks, list):
                fallback_list = [
                    FallbackProviderSettings(
                        provider=_require_str(fb, "provider", f"llm.fallback_providers[{i}]"),
                        model=_require_str(fb, "model", f"llm.fallback_providers[{i}]"),
                        api_key=fb.get("api_key"),
                        azure_endpoint=fb.get("azure_endpoint"),
                        base_url=fb.get("base_url"),
                    )
                    for i, fb in enumerate(raw_fallbacks)
                ]

        llm_provider = _require_str(llm, "provider", "llm")
        emb_provider = _require_str(embedding, "provider", "embedding")
        rerank_provider = _require_str(rerank, "provider", "rerank")

        return cls(
            llm=LLMSettings(
                provider=llm_provider,
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                api_key=_resolve_api_key(llm, llm_provider),
                api_version=llm.get("api_version"),
                azure_endpoint=_resolve_azure_endpoint(llm, llm_provider),
                deployment_name=llm.get("deployment_name"),
                base_url=_resolve_base_url(llm, llm_provider),
                circuit_breaker=cb_settings,
                fallback_providers=fallback_list,
            ),
            embedding=EmbeddingSettings(
                provider=emb_provider,
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                api_key=_resolve_api_key(embedding, emb_provider),
                api_version=embedding.get("api_version"),
                azure_endpoint=_resolve_azure_endpoint(embedding, emb_provider),
                deployment_name=embedding.get("deployment_name"),
                base_url=_resolve_base_url(embedding, emb_provider),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
                path=vector_store.get("path"),
                host=vector_store.get("host"),
                port=vector_store.get("port"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
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
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(
                    observability, "structured_logging", "observability"
                ),
            ),
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
            dashboard=dashboard_settings,
            cache=cache_settings,
            rate_limit=rate_limit_settings,
            query_rewriting=query_rewriting_settings,
            memory=memory_settings,
        )


def validate_settings(settings: Settings) -> None:
    """Validate semantic constraints beyond structural parsing.

    Raises SettingsError if any constraint is violated.
    """
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Environment variables from ``.env`` (if present) are loaded first,
    then YAML values take priority.  For secret fields (``api_key``,
    ``azure_endpoint``, ``base_url``), env vars serve as fallback when
    the YAML value is empty.

    Args:
        path: Path to settings YAML. Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).

    Returns:
        Validated Settings instance.

    Raises:
        SettingsError: If the file is missing or validation fails.
    """
    # Load .env file (if present) — does NOT override existing env vars
    load_dotenv(REPO_ROOT / ".env", override=False)

    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
