"""LLM-based query router — classifies query intent via structured JSON."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.core.settings import RouteConfig
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.query_router.base_router import BaseQueryRouter, RouteDecision

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """\
You are a query router. Classify the user's query into one of the available routes.

Available routes:
{routes}

Return valid JSON: {{"route": "<name>", "confidence": <0.0-1.0>, "tool_name": <string|null>, "reasoning": "<explanation>"}}

Rules:
- Choose the single best route
- confidence should reflect how certain you are
- tool_name is only used for tool_call routes; set to null otherwise
- Return ONLY the JSON object, no other text
"""

_DEFAULT_ROUTE = "knowledge_search"
_FALLBACK_CONFIDENCE = 0.5


class LLMRouter(BaseQueryRouter):
    """Routes queries by asking an LLM to classify intent.

    The LLM receives available route descriptions and returns a JSON
    decision. Falls back to ``knowledge_search`` on any error.

    Args:
        llm: A BaseLLM instance for classification.
        routes: List of RouteConfig describing available routes.
    """

    def __init__(self, llm: BaseLLM, routes: list[RouteConfig]) -> None:
        self._llm = llm
        self._routes = routes
        self._valid_names = frozenset(r.name for r in routes)

    def route(self, query: str, **kwargs: Any) -> RouteDecision:
        try:
            messages = self._build_messages(query)
            response = self._llm.chat(messages)
            return self._parse_response(response.content)
        except Exception as exc:
            logger.warning("LLMRouter failed, falling back to default: %s", exc)
            return self._fallback()

    def _build_messages(self, query: str) -> list[Message]:
        routes_text = "\n".join(
            f"- {r.name}: {r.description}" for r in self._routes
        )
        system = _SYSTEM_TEMPLATE.format(routes=routes_text)
        return [
            Message("system", system),
            Message("user", query),
        ]

    def _parse_response(self, content: str) -> RouteDecision:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLMRouter got invalid JSON, falling back")
            return self._fallback()

        route_name = data.get("route")
        if not route_name or route_name not in self._valid_names:
            logger.warning(
                "LLMRouter got unknown route '%s', falling back", route_name,
            )
            return self._fallback()

        return RouteDecision(
            route=route_name,
            confidence=float(data.get("confidence", _FALLBACK_CONFIDENCE)),
            tool_name=data.get("tool_name"),
            reasoning=data.get("reasoning"),
        )

    @staticmethod
    def _fallback() -> RouteDecision:
        return RouteDecision(
            route=_DEFAULT_ROUTE,
            confidence=_FALLBACK_CONFIDENCE,
            tool_name=None,
            reasoning="Fallback: could not classify query",
        )
