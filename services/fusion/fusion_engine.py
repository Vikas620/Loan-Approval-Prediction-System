"""Transparent confidence-aware multimodal risk fusion."""

from __future__ import annotations

from settings.fusion_settings import (DISAGREEMENT_RANGE_THRESHOLD, FUSION_VERSION,
                                       LOW_CONFIDENCE_THRESHOLD, LOW_RISK_MAX,
                                       MEDIUM_RISK_MAX, fusion_weights)
from .risk_components import RiskComponent, UnifiedRiskProfile, bounded_score


class FusionEngine:
    """Combine independent risk indicators for decision support."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or fusion_weights()
        self._validate_weights(self.weights)

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> None:
        required = {"ml", "evidence", "graph"}
        if set(weights) != required or any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("Fusion weights must contain ml, evidence, graph, be non-negative, and sum to 1.")

    def fuse(self, components: dict[str, RiskComponent], ml_decision: str) -> UnifiedRiskProfile:
        missing = {"ml", "evidence", "graph"}.difference(components)
        if missing:
            raise ValueError(f"Missing risk components: {sorted(missing)}")
        scores = {name: component.score for name, component in components.items()}
        overall = bounded_score(sum(scores[name] * self.weights[name] for name in scores))
        confidence = max(0.0, min(1.0, sum(components[name].confidence * self.weights[name] for name in components)))
        score_range = max(scores.values()) - min(scores.values())
        disagreement = {"detected": score_range >= DISAGREEMENT_RANGE_THRESHOLD, "range": round(score_range, 2), "threshold": DISAGREEMENT_RANGE_THRESHOLD, "explanation": "Component risk signals differ materially and should be reviewed." if score_range >= DISAGREEMENT_RANGE_THRESHOLD else "Component risk signals are within the configured agreement range."}
        review_flags = [component.explanation for component in components.values() if component.review_required]
        if disagreement["detected"]:
            review_flags.append("Component Disagreement Detected")
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            review_flags.append("Overall system confidence is below the configured review threshold.")
        review_required = bool(review_flags)
        status = "REVIEW_REQUIRED" if review_required else ml_decision.upper()
        risk_level = "LOW" if overall <= LOW_RISK_MAX else "MEDIUM" if overall <= MEDIUM_RISK_MAX else "HIGH"
        confidence_level = "HIGH" if confidence >= .80 else "MEDIUM" if confidence >= LOW_CONFIDENCE_THRESHOLD else "LOW"
        explanations = [f"Overall risk is the weighted combination of ML ({scores['ml']:.2f}), evidence ({scores['evidence']:.2f}), and graph ({scores['graph']:.2f}) risk indicators."]
        if disagreement["detected"]:
            explanations.append(disagreement["explanation"])
        return UnifiedRiskProfile(overall, confidence, confidence_level, risk_level, status, {name: component.to_dict() for name, component in components.items()}, disagreement, review_flags, explanations, FUSION_VERSION, self.weights.copy())
