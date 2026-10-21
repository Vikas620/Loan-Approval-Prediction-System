"""Configurable Phase 8 fusion parameters.

These are engineering/demo parameters, not scientifically optimal or banking
standards. They are persisted with each fusion assessment for reproducibility.
"""

FUSION_VERSION = "phase8-baseline-v1"
ML_WEIGHT = 0.50
EVIDENCE_WEIGHT = 0.30
GRAPH_WEIGHT = 0.20
LOW_CONFIDENCE_THRESHOLD = 0.65
DISAGREEMENT_RANGE_THRESHOLD = 45.0
LOW_RISK_MAX = 33.0
MEDIUM_RISK_MAX = 66.0


def fusion_weights() -> dict[str, float]:
    return {"ml": ML_WEIGHT, "evidence": EVIDENCE_WEIGHT, "graph": GRAPH_WEIGHT}
