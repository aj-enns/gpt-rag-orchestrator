"""Architecture Advisor strategy package.

Classifier-first agent strategy that decides whether a customer-described
workload needs AI, retrieves matching patterns from the Azure Architecture
Center index, and synthesises a recommendation.
"""

from .classifier import AINeedsClassifier
from .synthesizer import RecommendationSynthesizer
from .models import AINeedDecision, ClassifierResult, ArchitectureOption, Recommendation

__all__ = [
    "AINeedsClassifier",
    "RecommendationSynthesizer",
    "AINeedDecision",
    "ClassifierResult",
    "ArchitectureOption",
    "Recommendation",
]
