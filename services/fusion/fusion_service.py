"""Build Phase 8 components from the existing Phase 3, 6, and 7 outputs."""

from __future__ import annotations

from history_store import get_graph_records, get_phase6_data
from model.predict import predict_applicant
from phase6_analysis import evidence_summary
from services.graph.graph_builder import build_application_graph
from services.graph.graph_risk_engine import GraphRiskEngine

from .fusion_engine import FusionEngine
from .risk_components import RiskComponent


def build_ml_component(applicant: dict) -> tuple[RiskComponent, dict]:
    prediction = predict_applicant(applicant)
    approval_probability = prediction["approval_probability"]
    confidence = max(approval_probability, 1.0 - approval_probability)
    signal = {"signal": "Existing structured ML output", "value": prediction["predicted_class"], "probability": approval_probability, "source": "Phase 3 saved Gradient Boosting pipeline"}
    component = RiskComponent("ML Risk", prediction["risk_score"], confidence, [signal], "Risk is the complement of the existing model's approval probability.", "Phase 3 model", False)
    return component, prediction


def build_evidence_component(application_id: str) -> RiskComponent:
    data = get_phase6_data(application_id)
    summary = evidence_summary(data["evidence"], data["consistency_checks"], data["integrity_checks"], len(data["documents"]))
    signals = []
    for check in data["consistency_checks"]:
        signals.append({"signal": check.get("check_name"), "severity": check.get("severity"), "explanation": check.get("explanation"), "source": check.get("source_documents", [])})
    for item in data["integrity_checks"]:
        if item.get("status") != "NORMAL":
            signals.append({"signal": item.get("check_name"), "severity": item.get("status"), "explanation": item.get("detail"), "source": item.get("document_id")})
    if not data["documents"]:
        signals.append({"signal": "Missing critical evidence", "severity": "HIGH", "explanation": "Required evidence was not available."})
    risk = 100.0 - summary["evidence_consistency_score"]
    explanation = "Evidence risk is the complement of the Phase 6 evidence consistency score, with configured confidence and integrity penalties already applied."
    return RiskComponent("Evidence Risk", risk, summary["evidence_confidence"], signals, explanation, "Phase 6 evidence layer", summary["review_required"])


def build_graph_component(application_id: str) -> tuple[RiskComponent, dict]:
    source = get_graph_records(application_id)
    graph, _, _ = build_application_graph(application_id, source)
    analysis = GraphRiskEngine().analyze(graph, source, application_id)
    confidence = 0.85 if analysis["features"]["node_count"] > 2 else 0.55
    component = RiskComponent("Graph Risk", analysis["graph_risk_score"], confidence, analysis["signals"], "Graph risk is derived from explainable relationship indicators, not a fraud probability.", "Phase 7 NetworkX graph engine", analysis["review_required"])
    return component, analysis


def build_fusion_profile(application_id: str, applicant: dict, ml_decision: str | None = None) -> tuple[dict, dict]:
    ml_component, prediction = build_ml_component(applicant)
    evidence_component = build_evidence_component(application_id)
    graph_component, graph_analysis = build_graph_component(application_id)
    decision = ml_decision or ("APPROVED" if prediction["predicted_class"] == 1 else "REJECTED")
    profile = FusionEngine().fuse({"ml": ml_component, "evidence": evidence_component, "graph": graph_component}, decision).to_dict()
    profile["baseline_prediction"] = prediction
    profile["graph_analysis"] = graph_analysis
    return profile, prediction
