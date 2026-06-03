"""Architecture Advisor strategy.

Flow
----
1. Run the AI Needs Classifier (separate LLM call).
2. Retrieve matching patterns from the Architecture Center AI Search index,
   filtered by the classifier verdict (or unfiltered if confidence is low).
3. Ask the synthesiser to produce a structured recommendation.
4. Yield a markdown response chunk for the UI.

Activation: set ``AGENT_STRATEGY=architecture_advisor`` in Azure App
Configuration (label ``gpt-rag``).

Required App Configuration keys (defaults shown):
    ARCH_ADVISOR_CLASSIFIER_DEPLOYMENT       (fallback: CHAT_DEPLOYMENT_NAME)
    ARCH_ADVISOR_SYNTHESIZER_DEPLOYMENT      (fallback: CHAT_DEPLOYMENT_NAME)
    ARCH_ADVISOR_CLASSIFIER_THRESHOLD        0.6
    ARCH_ADVISOR_TOP_K                       6
    SEARCH_ARCHITECTURE_INDEX_NAME           architecture-{token}
    SEARCH_SERVICE_QUERY_ENDPOINT
    AI_FOUNDRY_ACCOUNT_ENDPOINT
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from dependencies import get_config

from .agent_strategies import AgentStrategies
from .base_agent_strategy import BaseAgentStrategy
from .architecture_advisor.classifier import AINeedsClassifier
from .architecture_advisor.gate import RequirementsGate
from .architecture_advisor.models import (
    AINeedDecision,
    ClassifierResult,
    GateResult,
    Recommendation,
)
from .architecture_advisor.synthesizer import RecommendationSynthesizer

logger = logging.getLogger(__name__)


class ArchitectureAdvisorStrategy(BaseAgentStrategy):
    """Classifier-first architecture advisor agent strategy."""

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config()
        self.strategy_type = AgentStrategies.ARCHITECTURE_ADVISOR

        # Sync credential is required for the openai bearer token provider.
        self._sync_credential = cfg.credential

        # Endpoints + deployments.
        chat_endpoint = cfg.get("AI_FOUNDRY_ACCOUNT_ENDPOINT")
        chat_default = cfg.get("CHAT_DEPLOYMENT_NAME")
        classifier_deployment = cfg.get(
            "ARCH_ADVISOR_CLASSIFIER_DEPLOYMENT", chat_default
        )
        synthesiser_deployment = cfg.get(
            "ARCH_ADVISOR_SYNTHESIZER_DEPLOYMENT", chat_default
        )
        gate_deployment = cfg.get(
            "ARCH_ADVISOR_GATE_DEPLOYMENT", chat_default
        )

        self._gate = RequirementsGate(
            azure_endpoint=chat_endpoint,
            deployment=gate_deployment,
            sync_credential=self._sync_credential,
            api_version=self.openai_api_version,
        )
        self._classifier = AINeedsClassifier(
            azure_endpoint=chat_endpoint,
            deployment=classifier_deployment,
            sync_credential=self._sync_credential,
            api_version=self.openai_api_version,
        )
        self._synthesiser = RecommendationSynthesizer(
            azure_endpoint=chat_endpoint,
            deployment=synthesiser_deployment,
            sync_credential=self._sync_credential,
            api_version=self.openai_api_version,
        )

        # Retrieval config.
        self._search_endpoint = cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT")
        self._index_name = cfg.get("SEARCH_ARCHITECTURE_INDEX_NAME", "architecture")
        self._top_k = int(cfg.get("ARCH_ADVISOR_TOP_K", 6))
        self._confidence_threshold = float(
            cfg.get("ARCH_ADVISOR_CLASSIFIER_THRESHOLD", 0.6)
        )
        # Hardcoded: at most this many user turns may consist of clarifying
        # questions before we force a recommendation, regardless of gate state.
        # Counted off the orchestrator-maintained `conversation["questions"]`
        # list so we don't depend on a private state cache surviving Cosmos.
        self._max_question_rounds = 2

    # -------------------------------------------------------- BaseAgentStrategy

    async def initiate_agent_flow(self, user_message: str) -> AsyncIterator[str]:
        logger.info("[arch-advisor] user_message=%r", user_message[:120])

        # --- Load / initialise multi-turn state from the conversation doc.
        # The orchestrator persists `self.conversation` after every turn, so
        # anything stored under "arch_advisor" survives across turns.
        state = self._load_state()
        state["dialog"].append({"role": "user", "content": user_message})

        # Once a recommendation has been produced, treat every further reply in
        # the same thread as *added context that refines the same design* — not
        # a brand-new request. We skip the clarifying-question loop and go
        # straight to regenerating an updated recommendation.
        already_recommended = bool(state.get("recommended"))

        # Count user turns from sources we don't have to maintain ourselves.
        # `conversation["questions"]` is appended by the orchestrator BEFORE
        # this strategy runs, so on turn N it has N entries when persistence
        # works. We OR it with the in-memory dialog length (which always
        # includes the message we just appended) so the cap holds even if the
        # persisted substate is dropped between turns.
        conv = getattr(self, "conversation", {}) or {}
        questions_persisted = len(conv.get("questions") or [])
        user_turns_local = sum(
            1 for m in state["dialog"] if m.get("role") == "user"
        )
        user_turn_count = max(questions_persisted, user_turns_local)

        # The gate only sees user turns + prior clarifying questions; the large
        # recommendation markdown is excluded so it doesn't skew consolidation.
        gate_dialog = [m for m in state["dialog"] if m.get("kind") != "recommendation"]
        gate = await self._gate.evaluate(gate_dialog)
        logger.info(
            "[arch-advisor] gate: ready=%s user_turn=%d (persisted=%d local=%d) cap=%d questions=%d recommended=%s",
            gate.ready,
            user_turn_count,
            questions_persisted,
            user_turns_local,
            self._max_question_rounds,
            len(gate.questions),
            already_recommended,
        )

        # Ask clarifying questions only BEFORE the first recommendation, and
        # only while we are still inside the hardcoded round budget. The 3rd
        # user message forces a recommendation regardless of gate state.
        if (
            not already_recommended
            and not gate.ready
            and user_turn_count <= self._max_question_rounds
        ):
            questions_md = self._render_questions(gate)
            state["dialog"].append(
                {"role": "assistant", "content": questions_md, "kind": "questions"}
            )
            self._save_state(state)
            yield questions_md
            return

        # --- Enough info (or refining an existing recommendation): recommend.
        # Always fall back to the full accumulated user context so a refinement
        # turn never loses the earlier requirements.
        description = gate.consolidated_description or self._joined_user_context(
            state["dialog"]
        )
        logger.info(
            "[arch-advisor] %s on consolidated description (%d chars)",
            "refining" if already_recommended else "recommending",
            len(description),
        )

        # Emit progress immediately so the user sees activity right away — the
        # classify + retrieve + synthesise steps below take ~30s combined.
        if already_recommended:
            yield (
                "Got it \u2014 folding that into the existing design and "
                "updating the recommendation\u2026\n\n"
            )
        else:
            yield (
                "Got it \u2014 that's enough to go on. Analyzing Azure architecture "
                "options and composing a recommendation\u2026\n\n"
            )

        verdict = await self._classifier.classify(description)
        logger.info(
            "[arch-advisor] classifier: decision=%s confidence=%.2f",
            verdict.decision.value,
            verdict.confidence,
        )

        candidates = await self._retrieve(description, verdict)
        logger.info("[arch-advisor] retrieved %d candidates", len(candidates))

        recommendation = await self._synthesiser.synthesize(
            description=description,
            classifier=verdict,
            candidates=candidates,
        )
        logger.info("[arch-advisor] synthesis complete; rendering recommendation")

        rendered = self._render_markdown(recommendation)
        state["dialog"].append(
            {"role": "assistant", "content": rendered, "kind": "recommendation"}
        )
        # Mark that a recommendation now exists so subsequent replies refine it.
        state["recommended"] = True
        self._save_state(state)
        yield rendered

    # ------------------------------------------------------------ state helpers

    def _load_state(self) -> dict:
        conversation = getattr(self, "conversation", None)
        if not isinstance(conversation, dict):
            return {"dialog": [], "rounds_asked": 0, "recommended": False}
        state = conversation.get("arch_advisor")
        if not isinstance(state, dict):
            state = {"dialog": [], "rounds_asked": 0, "recommended": False}
        state.setdefault("dialog", [])
        state.setdefault("rounds_asked", 0)
        state.setdefault("recommended", False)
        return state

    def _save_state(self, state: dict) -> None:
        conversation = getattr(self, "conversation", None)
        if isinstance(conversation, dict):
            conversation["arch_advisor"] = state

    @staticmethod
    def _joined_user_context(dialog: List[dict]) -> str:
        """Concatenate every user turn so refinement keeps the full context."""
        parts = [
            str(m.get("content", "")).strip()
            for m in dialog
            if m.get("role") == "user" and str(m.get("content", "")).strip()
        ]
        return "\n\n".join(parts)

    # --------------------------------------------------------------- rendering

    @staticmethod
    def _render_questions(gate: GateResult) -> str:
        lines = [
            "Before I recommend an architecture, I need a little more detail "
            "so the recommendation actually fits your scenario:",
            "",
        ]
        for i, q in enumerate(gate.questions, start=1):
            lines.append(f"{i}. {q}")
        lines.append("")
        lines.append(
            "_Answer what you can — even rough estimates help. "
            "I'll recommend a pattern as soon as I have enough to go on._"
        )
        return "\n".join(lines)

    # ----------------------------------------------------------------- retrieve

    async def _retrieve(
        self, user_message: str, verdict: ClassifierResult
    ) -> List[dict]:
        wants_both = verdict.needs_both_paths(self._confidence_threshold)
        if wants_both:
            filter_clause = None
        elif verdict.decision == AINeedDecision.YES:
            filter_clause = "usesAi eq true"
        else:
            filter_clause = "usesAi eq false"

        boosted = user_message
        if verdict.suggested_capabilities:
            boosted = (
                f"{user_message}\n\nCapabilities: "
                + ", ".join(verdict.suggested_capabilities)
            )

        async with SearchClient(
            endpoint=self._search_endpoint,
            index_name=self._index_name,
            credential=self.credential,
        ) as client:
            results = await client.search(
                search_text=boosted,
                vector_queries=[
                    VectorizableTextQuery(
                        text=boosted,
                        k_nearest_neighbors=self._top_k * 2,
                        fields="contentVector",
                    )
                ],
                top=self._top_k,
                filter=filter_clause,
                select=[
                    "id",
                    "title",
                    "url",
                    "summary",
                    "azureServices",
                    "usesAi",
                    "costBand",
                    "categories",
                ],
                query_type="semantic",
                semantic_configuration_name="semantic-config",
            )

            candidates: List[dict] = []
            async for item in results:
                candidates.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "summary": item.get("summary", ""),
                        "uses_ai": bool(item.get("usesAi", False)),
                        "azure_services": item.get("azureServices", []) or [],
                        "cost_band": item.get("costBand", "unknown"),
                        "categories": item.get("categories", []) or [],
                        "score": item.get("@search.score", 0.0),
                    }
                )
        return candidates

    # ------------------------------------------------------------------ render

    @staticmethod
    def _render_markdown(rec: Recommendation) -> str:
        lines: List[str] = []
        v = rec.classifier
        lines.append("## Recommendation\n")
        lines.append(
            f"**AI assessment:** `{v.decision.value}` "
            f"(confidence {v.confidence:.0%})  \n"
            f"_{v.rationale}_\n"
        )
        if rec.primary:
            lines.append(f"### Primary pattern — [{rec.primary.title}]({rec.primary.url})")
            lines.append(f"{rec.primary.summary}\n")
            if rec.primary.azure_services:
                lines.append(
                    "**Key services:** " + ", ".join(rec.primary.azure_services) + "\n"
                )
            lines.append(f"**Cost band:** `{rec.primary.cost_band}`\n")
        if rec.why_it_fits:
            lines.append(f"### Why it fits\n{rec.why_it_fits}\n")
        if rec.alternatives:
            lines.append("### Alternatives")
            lines.append("| Pattern | AI? | Cost | Why consider |")
            lines.append("|---------|-----|------|--------------|")
            for alt in rec.alternatives:
                ai = "yes" if alt.uses_ai else "no"
                summary = alt.summary or ""
                summary = summary[:140] + ("…" if len(summary) > 140 else "")
                lines.append(
                    f"| [{alt.title}]({alt.url}) | {ai} | {alt.cost_band} | {summary} |"
                )
            lines.append("")
        if rec.cost_estimate:
            lines.append(f"### Cost estimate\n{rec.cost_estimate}\n")
        if rec.implementation_checklist:
            lines.append("### Implementation checklist")
            lines.extend(f"- [ ] {step}" for step in rec.implementation_checklist)
            lines.append("")
        if rec.next_steps:
            lines.append("### Next steps")
            lines.extend(f"1. {step}" for step in rec.next_steps)
        return "\n".join(lines)
