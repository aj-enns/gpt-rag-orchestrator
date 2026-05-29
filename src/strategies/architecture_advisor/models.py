"""Framework-agnostic dataclasses used by the Architecture Advisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AINeedDecision(str, Enum):
    YES = "yes"
    NO = "no"
    MAYBE = "maybe"


@dataclass
class ClassifierResult:
    decision: AINeedDecision
    confidence: float
    rationale: str
    ai_signals: List[str] = field(default_factory=list)
    non_ai_signals: List[str] = field(default_factory=list)
    suggested_capabilities: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    def needs_both_paths(self, threshold: float) -> bool:
        return self.decision == AINeedDecision.MAYBE or self.confidence < threshold

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "ai_signals": self.ai_signals,
            "non_ai_signals": self.non_ai_signals,
            "suggested_capabilities": self.suggested_capabilities,
        }


@dataclass
class GateResult:
    """Outcome of the requirements-qualification gate."""

    ready: bool
    questions: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    consolidated_description: str = ""
    raw_response: Optional[str] = None


@dataclass
class ArchitectureOption:
    title: str
    url: str
    summary: str
    uses_ai: bool
    azure_services: List[str] = field(default_factory=list)
    cost_band: str = "unknown"


@dataclass
class Recommendation:
    primary: Optional[ArchitectureOption]
    alternatives: List[ArchitectureOption]
    why_it_fits: str
    cost_estimate: str
    implementation_checklist: List[str]
    next_steps: List[str]
    classifier: ClassifierResult

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.__dict__ if self.primary else None,
            "alternatives": [a.__dict__ for a in self.alternatives],
            "why_it_fits": self.why_it_fits,
            "cost_estimate": self.cost_estimate,
            "implementation_checklist": self.implementation_checklist,
            "next_steps": self.next_steps,
            "classifier": self.classifier.to_dict(),
        }
