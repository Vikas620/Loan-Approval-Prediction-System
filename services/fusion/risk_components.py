"""Normalized, explainable risk component objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def bounded_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 2)


@dataclass
class RiskComponent:
    name: str
    score: float
    confidence: float
    signals: list[dict] = field(default_factory=list)
    explanation: str = ""
    source: str = ""
    review_required: bool = False

    def __post_init__(self) -> None:
        self.score = bounded_score(self.score)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnifiedRiskProfile:
    overall_risk: float
    confidence: float
    confidence_level: str
    risk_level: str
    status: str
    components: dict[str, dict]
    disagreement: dict
    review_flags: list[str]
    explanations: list[str]
    fusion_version: str
    weights: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)
