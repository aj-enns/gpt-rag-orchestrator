"""AI Needs Classifier — separate Azure OpenAI call.

Uses the same auth chain as the rest of gpt-rag-orchestrator (Entra token
provider on top of the shared aio credential). The deployment used here is
intentionally a smaller / cheaper model than the synthesiser so the decision
step stays fast.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI

from .models import AINeedDecision, ClassifierResult

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "classifier_system.txt"
EXAMPLES_PATH = PROMPTS_DIR / "classifier_examples.jsonl"


class AINeedsClassifier:
    def __init__(
        self,
        *,
        azure_endpoint: str,
        deployment: str,
        sync_credential,
        api_version: str = "2025-04-01-preview",
        temperature: float = 0.0,
        max_examples: int = 6,
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
        self._max_examples = max_examples
        self._system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self._examples = list(self._load_examples())

    @staticmethod
    def _load_examples() -> Iterable[dict]:
        if not EXAMPLES_PATH.exists():
            logger.warning("classifier_examples.jsonl missing — zero-shot mode")
            return
        with EXAMPLES_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping bad example line: %s", exc)

    async def classify(self, description: str) -> ClassifierResult:
        if not description or not description.strip():
            return ClassifierResult(
                decision=AINeedDecision.MAYBE,
                confidence=0.0,
                rationale="Empty description — defer to retrieval.",
            )

        messages: List[dict] = [{"role": "system", "content": self._system_prompt}]
        for example in self._examples[: self._max_examples]:
            messages.append({"role": "user", "content": example["input"]})
            messages.append(
                {"role": "assistant", "content": json.dumps(example["output"])}
            )
        messages.append({"role": "user", "content": description.strip()})

        try:
            response = await self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                max_completion_tokens=2000,
                timeout=45,
            )
        except Exception:
            logger.exception("Classifier LLM call failed")
            return ClassifierResult(
                decision=AINeedDecision.MAYBE,
                confidence=0.0,
                rationale="Classifier call failed — defer to retrieval.",
            )

        raw = response.choices[0].message.content or "{}"
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> ClassifierResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Classifier returned non-JSON output: %r", raw[:200])
            return ClassifierResult(
                decision=AINeedDecision.MAYBE,
                confidence=0.0,
                rationale="Could not parse classifier output.",
                raw_response=raw,
            )

        decision_raw = str(payload.get("decision", "maybe")).lower()
        try:
            decision = AINeedDecision(decision_raw)
        except ValueError:
            decision = AINeedDecision.MAYBE

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        return ClassifierResult(
            decision=decision,
            confidence=confidence,
            rationale=str(payload.get("rationale", "")).strip(),
            ai_signals=list(payload.get("ai_signals", []) or []),
            non_ai_signals=list(payload.get("non_ai_signals", []) or []),
            suggested_capabilities=list(payload.get("suggested_capabilities", []) or []),
            raw_response=raw,
        )
