"""Recommendation synthesiser — second LLM call that turns the classifier
verdict + retrieval candidates into a structured customer-facing answer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI

from .models import ArchitectureOption, ClassifierResult, Recommendation

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "synthesizer_template.txt"


class RecommendationSynthesizer:
    def __init__(
        self,
        *,
        azure_endpoint: str,
        deployment: str,
        sync_credential,
        api_version: str = "2025-04-01-preview",
        temperature: float = 0.2,
    ) -> None:
        token_provider = get_bearer_token_provider(
            sync_credential, "https://cognitiveservices.azure.com/.default"
        )
        self._client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
        )
        self._deployment = deployment
        self._temperature = temperature
        self._template = TEMPLATE_PATH.read_text(encoding="utf-8")

    async def synthesize(
        self,
        description: str,
        classifier: ClassifierResult,
        candidates: List[dict],
    ) -> Recommendation:
        user_payload = {
            "customer_description": description,
            "classifier_verdict": classifier.to_dict(),
            "candidates": candidates,
        }

        try:
            response = await self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": self._template},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                temperature=self._temperature,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                max_completion_tokens=4000,
                timeout=60,
            )
        except Exception:
            logger.exception("Synthesizer LLM call failed")
            return self._fallback(classifier, candidates)

        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Synthesizer returned non-JSON: %r", raw[:200])
            return self._fallback(classifier, candidates)

        return self._materialise(payload, classifier, candidates)

    @staticmethod
    def _to_option(payload: dict) -> ArchitectureOption:
        return ArchitectureOption(
            title=str(payload.get("title", "")).strip(),
            url=str(payload.get("url", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            uses_ai=bool(payload.get("uses_ai", False)),
            azure_services=list(payload.get("azure_services", []) or []),
            cost_band=str(payload.get("cost_band", "unknown")).lower(),
        )

    def _materialise(
        self,
        payload: dict,
        classifier: ClassifierResult,
        candidates: List[dict],
    ) -> Recommendation:
        primary_raw = payload.get("primary")
        primary = self._to_option(primary_raw) if primary_raw else None
        alternatives = [
            self._to_option(a) for a in (payload.get("alternatives") or [])
        ]
        if not primary and candidates:
            primary = self._to_option(candidates[0])

        return Recommendation(
            primary=primary,
            alternatives=alternatives,
            why_it_fits=str(payload.get("why_it_fits", "")).strip(),
            cost_estimate=str(payload.get("cost_estimate", "Not estimated.")).strip(),
            implementation_checklist=list(
                payload.get("implementation_checklist", []) or []
            ),
            next_steps=list(payload.get("next_steps", []) or []),
            classifier=classifier,
        )

    def _fallback(
        self, classifier: ClassifierResult, candidates: List[dict]
    ) -> Recommendation:
        primary = self._to_option(candidates[0]) if candidates else None
        alternatives = [self._to_option(c) for c in candidates[1:4]]
        return Recommendation(
            primary=primary,
            alternatives=alternatives,
            why_it_fits=(
                "Automatic synthesis failed; showing top retrieval results directly."
            ),
            cost_estimate="Not estimated.",
            implementation_checklist=[],
            next_steps=[
                "Review the retrieved patterns and re-run with a more specific "
                "description."
            ],
            classifier=classifier,
        )
