"""Explainable graph risk-indicator engine."""

from __future__ import annotations

from settings.graph_settings import (DOCUMENT_REUSE_WEIGHT, EVIDENCE_CONFLICT_WEIGHT,
                                      REVIEW_THRESHOLD, SHARED_DOCUMENT_ENTITY_WEIGHT,
                                      SHARED_ENTITY_WEIGHT)
from .graph_features import calculate_features


class GraphRiskEngine:
    """Analyze relationship patterns; signals are not proof of fraud."""

    def analyze(self, graph, source: dict, application_id: str) -> dict:
        features = calculate_features(graph, source, application_id)
        signals = []
        document_reuse = features["document_reuse_count"]
        if document_reuse:
            contribution = min(DOCUMENT_REUSE_WEIGHT, document_reuse * DOCUMENT_REUSE_WEIGHT)
            signals.append({"signal_name": "Document Reuse", "signal_value": document_reuse, "severity": "MEDIUM" if document_reuse == 1 else "HIGH", "evidence": [item["file_hash"] for item in source.get("documents", [])], "explanation": "This document fingerprint appears in multiple applications.", "score_contribution": contribution})
        shared_entities = features["shared_entity_count"]
        if shared_entities:
            signals.append({"signal_name": "Shared Employer Entity", "signal_value": shared_entities, "severity": "INFO", "evidence": [node for node, data in graph.nodes(data=True) if data.get("node_type") == "Employer"], "explanation": "Applicants may naturally share an employer; this relationship alone is not suspicious.", "score_contribution": SHARED_ENTITY_WEIGHT})
        if document_reuse and shared_entities:
            signals.append({"signal_name": "Shared Entity and Document Pattern", "signal_value": 1, "severity": "HIGH", "evidence": [item["file_hash"] for item in source.get("documents", [])], "explanation": "Multiple applications share an entity relationship and the same supporting document fingerprint.", "score_contribution": SHARED_DOCUMENT_ENTITY_WEIGHT})
        conflicts = [item for item in source.get("consistency_checks", []) if item.get("severity") in {"HIGH", "CRITICAL"}]
        if conflicts:
            signals.append({"signal_name": "Evidence Conflict in Relationship Network", "signal_value": len(conflicts), "severity": "HIGH", "evidence": [item.get("check_name") for item in conflicts], "explanation": "Phase 6 evidence conflicts are connected to this application graph and require review.", "score_contribution": EVIDENCE_CONFLICT_WEIGHT})
        score = max(0.0, min(100.0, sum(signal["score_contribution"] for signal in signals)))
        review_required = score >= REVIEW_THRESHOLD and bool(conflicts or document_reuse)
        return {"graph_risk_score": round(score, 2), "signals": signals, "features": features, "review_required": review_required, "status": "REVIEW_REQUIRED" if review_required else "NORMAL", "disclaimer": "Graph risk is an indicator for human review, not a probability of fraud or a fraud finding."}
