"""Requirements-qualification gate — an ask-first LLM step.

Given the full advisor/customer dialog, decides whether enough is known to
recommend an architecture. When not, it returns a small set of clarifying
questions. When ready, it returns a consolidated description for the
downstream classifier + synthesiser.

Uses the same Entra-token auth chain as the rest of the advisor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI

from .models import GateResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "gate_system.txt"


class RequirementsGate:
    def __init__(
        self,
        *,
        azure_endpoint: str,
        deployment: str,
        sync_credential,
        api_version: str = "2025-04-01-preview",
        temperature: float = 0.0,
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
        self._system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    async def evaluate(self, dialog: List[dict]) -> GateResult:
        """Evaluate the dialog so far.

        Args:
            dialog: ordered list of {"role": "user"|"assistant", "content": str}.
        """
        latest_user = next(
            (m["content"] for m in reversed(dialog) if m.get("role") == "user"),
            "",
        )
        if not latest_user.strip() and len(dialog) <= 1:
            return GateResult(
                ready=False,
                questions=[
                    "What are you trying to build, and what is the main goal?"
                ],
                missing=["workload"],
                consolidated_description="",
            )

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.append(
            {
                "role": "user",
                "content": json.dumps({"dialog": dialog}),
            }
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                max_completion_tokens=2000,
                timeout=45,
            )
        except Exception:
            logger.exception("Requirements gate LLM call failed")
            # Fail open: proceed to recommendation using whatever we have.
            return GateResult(
                ready=True,
                questions=[],
                missing=[],
                consolidated_description=latest_user.strip(),
            )

        raw = response.choices[0].message.content or "{}"
        return self._parse(raw, fallback_description=latest_user.strip())

    @staticmethod
    def _parse(raw: str, *, fallback_description: str) -> GateResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Gate returned non-JSON output: %r", raw[:200])
            return GateResult(
                ready=True,
                questions=[],
                missing=[],
                consolidated_description=fallback_description,
            )

        questions = [
            str(q).strip()
            for q in (payload.get("questions") or [])
            if str(q).strip()
        ][:4]
        ready = bool(payload.get("ready", False))
        # Guard: if the model says not-ready but gives no questions, treat as ready.
        if not ready and not questions:
            ready = True

        consolidated = str(
            payload.get("consolidated_description", "") or ""
        ).strip()
        if not consolidated:
            consolidated = fallback_description

        return GateResult(
            ready=ready,
            questions=questions,
            missing=list(payload.get("missing", []) or []),
            consolidated_description=consolidated,
            raw_response=raw,
        )
