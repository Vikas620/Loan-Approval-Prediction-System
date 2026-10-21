"""Compose Phase 9 explanations from actual Phase 3-8 records."""

from __future__ import annotations

from history_store import get_graph_records, get_latest_assessment_for_application, get_latest_fusion_assessment, get_phase6_data
from services.fusion.fusion_service import build_fusion_profile
from services.fusion.fusion_explainer import explain_profile as fusion_explanation
from services.graph.graph_builder import build_application_graph
from services.graph.graph_risk_engine import GraphRiskEngine

from .lime_explainer import explain_local as explain_lime
from .shap_explainer import explain_local as explain_shap


def build_bundle(application_id: str) -> dict:
    assessment = get_latest_assessment_for_application(application_id)
    if assessment is None:
        raise LookupError("Assess the application before requesting explanations.")
    shap_result = explain_shap(assessment["applicant"])
    lime_result = explain_lime(assessment["applicant"])
    phase6 = get_phase6_data(application_id)
    phase7_source = get_graph_records(application_id)
    graph, _, _ = build_application_graph(application_id, phase7_source)
    graph_analysis = GraphRiskEngine().analyze(graph, phase7_source, application_id)
    fusion_record = get_latest_fusion_assessment(application_id)
    profile = fusion_record["profile"] if fusion_record else build_fusion_profile(application_id, assessment["applicant"], assessment["decision"].upper())[0]
    shap_features = {item["feature"] for item in shap_result.get("top_factors", [])} if shap_result.get("available") else set()
    lime_features = {item.get("feature") for item in lime_result.get("top_factors", [])} if lime_result.get("available") else set()
    consistency = {"status": "UNAVAILABLE", "explanation": "LIME is unavailable, so SHAP/LIME overlap was not calculated."}
    if shap_features and lime_features:
        overlap = len(shap_features & lime_features) / max(len(shap_features | lime_features), 1)
        consistency = {"status": "SIMILAR" if overlap >= .5 else "DIFFERENT", "overlap": overlap, "explanation": "Explanation methods show similar influential factors." if overlap >= .5 else "Explanation methods produced different rankings; interpret with caution."}
    return {"application_id": application_id, "assessment_id": assessment["id"], "shap": shap_result, "lime": lime_result, "explanation_consistency": consistency, "evidence": [{key: record.get(key) for key in ("evidence_id", "document_id", "field_name", "normalized_value", "page_number", "confidence", "extraction_method", "verification_status")} for record in phase6["evidence"]], "graph": {"signals": graph_analysis["signals"], "features": graph_analysis["features"], "status": graph_analysis["status"]}, "fusion": fusion_explanation(profile), "profile": profile}
