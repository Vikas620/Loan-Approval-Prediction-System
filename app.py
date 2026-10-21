"""Entry point for the Loan Approval and Risk Assessment System."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from document_intelligence import MAX_DOCUMENT_BYTES, allowed_file, extract_fields, extract_text, file_digest
from history_store import (application_exists, create_application, delete_document, get_assessment,
                           get_graph_records, get_latest_assessment_for_application,
                           get_latest_fusion_assessment, get_phase6_data, list_assessments,
                           list_explanation_audits, list_fusion_audits,
                           save_assessment, save_document, save_evidence, save_fusion_assessment,
                           save_explanation_audit, save_graph_records, save_phase6_results)
from model.predict import predict_applicant
from phase6_analysis import (consistency_checks, confidence_for_method, evidence_summary,
                             graph_edges, integrity_checks)
from report_generator import build_final_report, build_report
from services.graph.graph_builder import build_application_graph
from services.graph.graph_explainer import explain_analysis
from services.graph.graph_risk_engine import GraphRiskEngine
from services.fusion.fusion_engine import FusionEngine
from services.fusion.fusion_explainer import explain_profile
from services.fusion.fusion_service import build_fusion_profile
from services.fusion.risk_components import RiskComponent
from services.fusion.evaluation import evaluate_fusion_availability
from services.explainability.counterfactual_engine import simulate
from services.explainability.explanation_service import build_bundle
from services.explainability.fairness_engine import fairness_report
from services.explainability.model_card import model_card
from services.analytics.dashboard import application_analytics, dashboard_snapshot


FORM_FIELDS = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Dependents": {"0", "1", "2", "3+"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Property_Area": {"Urban", "Semiurban", "Rural"},
}
NUMERIC_FIELDS = {
    "ApplicantIncome": (0, None),
    "CoapplicantIncome": (0, None),
    "LoanAmount": (1, None),
    "Loan_Amount_Term": (1, None),
    "Credit_History": (0, 1),
}
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "instance" / "uploads"


def build_graph_view(application_id: str) -> tuple[dict, dict]:
    source = get_graph_records(application_id)
    graph, nodes, edges = build_application_graph(application_id, source)
    analysis = GraphRiskEngine().analyze(graph, source, application_id)
    public_nodes = [{"node_id": node["node_id"], "node_type": node["node_type"], "metadata": node["metadata"]} for node in nodes]
    public_edges = [{key: value for key, value in edge.items() if key in {"source_node", "source_type", "relationship", "target_node", "target_type", "evidence_id"}} for edge in edges]
    return {"application_id": application_id, "nodes": public_nodes, "edges": public_edges, "features": analysis["features"], "risk": analysis}, source


def validate_applicant_payload(payload: object) -> tuple[dict, dict[str, str]]:
    """Validate and normalize the public prediction input."""
    if not isinstance(payload, dict):
        return {}, {"form": "Submit the application as a JSON object."}

    errors: dict[str, str] = {}
    normalized: dict = {}
    for field, allowed in FORM_FIELDS.items():
        value = str(payload.get(field, "")).strip()
        if value not in allowed:
            errors[field] = "Choose one of the available options."
        else:
            normalized[field] = value

    for field, (minimum, maximum) in NUMERIC_FIELDS.items():
        raw_value = payload.get(field, "")
        try:
            value = float(raw_value)
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError
            normalized[field] = int(value) if value.is_integer() else value
        except (TypeError, ValueError):
            errors[field] = f"Enter a number from {minimum:g} to {maximum:g}." if maximum else f"Enter a number of at least {minimum:g}."
    return normalized, errors


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = MAX_DOCUMENT_BYTES
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):
        return jsonify({"error": "Uploaded content exceeds the configured document size limit."}), 422

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "loan-approval-system"}

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        applicant, errors = validate_applicant_payload(payload)
        if errors:
            return jsonify({"error": "Please correct the highlighted fields.", "fields": errors}), 422
        try:
            assessment = predict_applicant(applicant)
        except FileNotFoundError:
            return jsonify({"error": "The trained assessment model is not available."}), 503
        except ValueError:
            return jsonify({"error": "The application could not be assessed with the supplied values."}), 422
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify({"error": "The assessment service is temporarily unavailable."}), 500

        approved = assessment["predicted_class"] == 1
        return jsonify(
            {
                **assessment,
                "decision": "Approved" if approved else "Rejected",
                "message": "Automated model assessment only; this is not a guarantee of actual loan approval.",
            }
        )

    @app.post("/api/documents")
    @app.post("/api/documents/upload")
    def process_documents():
        """Save allowed files locally and return extracted review data."""
        uploads = request.files.getlist("documents")
        if not uploads or not any(file.filename for file in uploads):
            return jsonify({"error": "Choose at least one PDF, JPG, or PNG document."}), 422

        application_id = request.form.get("application_id") or create_application()
        document_type = request.form.get("document_type", "Other Supporting Document")
        allowed_document_types = {"Salary Slip", "Bank Statement", "Income/ITR Document", "Employment Letter", "Identity Document (Demo)", "Other Supporting Document"}
        if document_type not in allowed_document_types:
            return jsonify({"error": "Choose a supported document category."}), 422
        documents = []
        combined_fields = {}
        seen_digests = set()
        for uploaded in uploads:
            original_name = secure_filename(uploaded.filename or "")
            if not original_name or not allowed_file(original_name):
                return jsonify({"error": f"{original_name or 'A file'} is not a supported document type."}), 422
            accepted_mimes = {"application/pdf", "image/jpeg", "image/png", "application/octet-stream"}
            if uploaded.mimetype and uploaded.mimetype not in accepted_mimes:
                return jsonify({"error": f"{original_name} has an unsupported content type."}), 422
            uploaded.stream.seek(0, 2)
            size = uploaded.stream.tell()
            uploaded.stream.seek(0)
            if size > MAX_DOCUMENT_BYTES:
                return jsonify({"error": f"{original_name} exceeds the 10 MB document limit."}), 422
            digest = file_digest(uploaded)
            duplicate = digest in seen_digests
            seen_digests.add(digest)
            stored_name = f"{digest[:16]}_{original_name}"
            stored_path = UPLOAD_DIR / stored_name
            if not stored_path.exists():
                uploaded.save(stored_path)
            try:
                text, method = extract_text(stored_path)
            except RuntimeError as error:
                return jsonify({"error": str(error)}), 503
            fields = extract_fields(text)
            document_id = f"doc_{uuid4().hex}"
            method_confidence = confidence_for_method(method)
            document_record = {
                "document_id": document_id,
                "filename": original_name,
                "safe_filename": stored_name,
                "file_hash": digest,
                "document_type": document_type,
                "method": method,
                "processing_status": "PROCESSED",
                "ocr_status": "COMPLETED" if "ocr" in method else "NOT_REQUIRED",
                "extraction_status": "COMPLETED" if fields else "NO_FIELDS_FOUND",
                "confidence_score": method_confidence,
                "duplicate": duplicate,
                "page_count": 1,
                "characters": len(text),
                "fields": fields,
            }
            stored_document_id = save_document(application_id, {**document_record, "upload_timestamp": datetime.now(timezone.utc).isoformat(), "integrity_status": "NORMAL"})
            document_record["document_id"] = stored_document_id
            evidence = save_evidence(application_id, stored_document_id, fields, method, method_confidence)
            integrity = integrity_checks(document_record, stored_path, text)
            document_record["integrity_status"] = "REVIEW_REQUIRED" if any(item["status"] == "REVIEW_REQUIRED" for item in integrity) else "WARNING" if any(item["status"] == "WARNING" for item in integrity) else "NORMAL"
            documents.append({**document_record, "evidence": evidence, "integrity_checks": integrity})
            combined_fields.update({key: value for key, value in fields.items() if value})

        validation = {
            "missing": [field for field in ("Name", "ApplicantIncome", "LoanAmount") if not combined_fields.get(field)],
            "warnings": [],
        }
        if any(item["duplicate"] for item in documents):
            validation["warnings"].append("A duplicate document was detected in this upload.")
        for field in ("ApplicantIncome", "LoanAmount", "Existing liabilities"):
            value = combined_fields.get(field)
            if value and (not value.replace(".", "", 1).isdigit() or float(value) < 0):
                validation["warnings"].append(f"{field} has an invalid numeric value.")
        validation["valid"] = not validation["missing"] and not validation["warnings"]
        document_fields = [{"document_type": item["document_type"], "fields": item["fields"]} for item in documents]
        checks = consistency_checks({}, document_fields)
        integrity = [item for document in documents for item in document["integrity_checks"]]
        evidence = [item for document in documents for item in document["evidence"]]
        summary = evidence_summary(evidence, checks, integrity, len(documents))
        edges = graph_edges(application_id, documents, evidence, combined_fields)
        save_phase6_results(application_id, checks, integrity, edges)
        graph_source = get_graph_records(application_id)
        _, graph_nodes, graph_relationships = build_application_graph(application_id, graph_source)
        save_graph_records(application_id, graph_nodes, graph_relationships)
        public_documents = [{key: value for key, value in document.items() if key not in {"safe_filename", "file_hash", "fields"}} for document in documents]
        return jsonify({"application_id": application_id, "documents": public_documents, "extracted": combined_fields, "validation": validation, "consistency_checks": checks, "integrity_checks": integrity, "evidence": evidence, "evidence_summary": summary, "graph_edges": edges})

    @app.post("/api/assess")
    def assess_application():
        payload = request.get_json(silent=True) or {}
        applicant, errors = validate_applicant_payload(payload.get("applicant"))
        if errors:
            return jsonify({"error": "Review the applicant fields before assessing.", "fields": errors}), 422
        try:
            assessment_result = predict_applicant(applicant)
        except (FileNotFoundError, ValueError) as error:
            return jsonify({"error": str(error)}), 503 if isinstance(error, FileNotFoundError) else 422
        result = {**assessment_result, "decision": "Approved" if assessment_result["predicted_class"] == 1 else "Rejected", "message": "Automated model assessment only; this is not a guarantee of actual loan approval."}
        factors = [field for field in ("Credit_History", "ApplicantIncome", "LoanAmount", "Education", "Property_Area", "Self_Employed") if field in applicant]
        extraction = payload.get("extraction") or {}
        validation = payload.get("validation") or {"missing": [], "warnings": [], "valid": True}
        application_id = payload.get("application_id")
        assessment_id = save_assessment(applicant, payload.get("documents") or [], extraction, validation, result, application_id)
        phase6 = get_phase6_data(application_id) if application_id else {"documents": [], "evidence": [], "consistency_checks": [], "integrity_checks": [], "graph_edges": []}
        summary = evidence_summary(phase6["evidence"], phase6["consistency_checks"], phase6["integrity_checks"], len(phase6["documents"]))
        review_state = "REVIEW_REQUIRED" if summary["review_required"] else result["decision"].upper()
        response = {**result, "assessment_id": assessment_id, "application_id": application_id, "key_factors": factors, "created_at": datetime.now(timezone.utc).isoformat(), "review_state": review_state, "evidence_summary": summary, "consistency_checks": phase6["consistency_checks"], "integrity_checks": phase6["integrity_checks"]}
        if application_id:
            profile, _ = build_fusion_profile(application_id, applicant, result["decision"])
            fusion_id = save_fusion_assessment(assessment_id, application_id, profile)
            response["fusion_id"] = fusion_id
            response["unified_risk_profile"] = profile
        return jsonify(response)

    @app.get("/api/history")
    def assessment_history():
        return jsonify({"assessments": list_assessments()})

    @app.get("/api/documents/<application_id>")
    def application_documents(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        documents = get_phase6_data(application_id)["documents"]
        return jsonify({"documents": [{key: value for key, value in document.items() if key not in {"safe_filename", "file_hash"}} for document in documents]})

    @app.post("/api/documents/<application_id>/process")
    def process_existing_application(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        return jsonify({"application_id": application_id, "status": "Use /api/documents/upload to submit files for processing."}), 200

    @app.delete("/api/documents/<application_id>/<document_id>")
    def remove_document(application_id: str, document_id: str):
        document = delete_document(application_id, document_id)
        if document is None:
            return jsonify({"error": "Document not found."}), 404
        stored_path = (UPLOAD_DIR / document["safe_filename"]).resolve()
        if stored_path.parent == UPLOAD_DIR.resolve() and stored_path.exists():
            stored_path.unlink()
        return jsonify({"deleted": document_id}), 200

    @app.get("/api/evidence/<application_id>")
    def application_evidence(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        return jsonify({"application_id": application_id, "evidence": get_phase6_data(application_id)["evidence"]})

    @app.get("/api/consistency/<application_id>")
    def application_consistency(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        return jsonify({"application_id": application_id, "checks": get_phase6_data(application_id)["consistency_checks"]})

    @app.get("/api/document-integrity/<application_id>")
    def application_integrity(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        return jsonify({"application_id": application_id, "checks": get_phase6_data(application_id)["integrity_checks"]})

    @app.get("/api/application/<application_id>/evidence-summary")
    def application_evidence_summary(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        data = get_phase6_data(application_id)
        summary = evidence_summary(data["evidence"], data["consistency_checks"], data["integrity_checks"], len(data["documents"]))
        return jsonify({**summary, "application_id": application_id, "documents": data["documents"], "graph_edges": data["graph_edges"]})

    @app.get("/api/application/<application_id>/graph")
    def application_graph(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        return jsonify({"application_id": application_id, "edges": get_phase6_data(application_id)["graph_edges"]})

    @app.get("/api/graph/<application_id>")
    def graph_application(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        graph_view, _ = build_graph_view(application_id)
        return jsonify(graph_view)

    @app.get("/api/graph/<application_id>/features")
    def graph_features(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        graph_view, _ = build_graph_view(application_id)
        return jsonify({"application_id": application_id, "features": graph_view["features"]})

    @app.get("/api/graph/<application_id>/risk")
    def graph_risk(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        graph_view, _ = build_graph_view(application_id)
        return jsonify({"application_id": application_id, **graph_view["risk"]})

    @app.get("/api/graph/<application_id>/signals")
    def graph_signals(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        graph_view, _ = build_graph_view(application_id)
        return jsonify({"application_id": application_id, "signals": graph_view["risk"]["signals"]})

    @app.get("/api/graph/<application_id>/explanation")
    def graph_explanation(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        graph_view, _ = build_graph_view(application_id)
        return jsonify({"application_id": application_id, **explain_analysis(graph_view["risk"])})

    @app.get("/api/risk/<application_id>")
    def unified_risk(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        latest = get_latest_assessment_for_application(application_id)
        if latest is None:
            return jsonify({"error": "Assess the application before requesting unified risk."}), 404
        profile_record = get_latest_fusion_assessment(application_id)
        if profile_record is None:
            profile, _ = build_fusion_profile(application_id, latest["applicant"], latest["decision"].upper())
            save_fusion_assessment(latest["id"], application_id, profile)
        else:
            profile = profile_record["profile"]
        return jsonify({"application_id": application_id, **profile})

    @app.get("/api/risk/<application_id>/components")
    def unified_risk_components(application_id: str):
        response = unified_risk(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, "components": payload["components"]})

    @app.get("/api/risk/<application_id>/explanation")
    def unified_risk_explanation(application_id: str):
        response = unified_risk(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, **explain_profile(payload)})

    @app.get("/api/risk/<application_id>/disagreement")
    def unified_risk_disagreement(application_id: str):
        response = unified_risk(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, **payload["disagreement"]})

    @app.get("/api/risk/<application_id>/traceability")
    def unified_risk_traceability(application_id: str):
        response = unified_risk(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, "components": payload["components"], "review_flags": payload["review_flags"]})

    @app.post("/api/risk/<application_id>/scenario")
    def risk_scenario(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        latest = get_latest_assessment_for_application(application_id)
        if latest is None:
            return jsonify({"error": "Assess the original application before running a scenario."}), 404
        scenario_input = request.get_json(silent=True) or {}
        applicant = latest["applicant"].copy()
        applicant.update({key: scenario_input[key] for key in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History") if key in scenario_input})
        scenario_applicant, errors = validate_applicant_payload(applicant)
        if errors:
            return jsonify({"error": "Scenario inputs are invalid.", "fields": errors}), 422
        original, _ = build_fusion_profile(application_id, latest["applicant"], latest["decision"].upper())
        scenario, _ = build_fusion_profile(application_id, scenario_applicant, latest["decision"].upper())
        return jsonify({"application_id": application_id, "original_risk": original["overall_risk"], "scenario_risk": scenario["overall_risk"], "difference": round(scenario["overall_risk"] - original["overall_risk"], 2), "message": "Under this simulated input scenario, the model and risk indicators changed as shown. This is not a causal approval claim."})

    @app.get("/api/evaluation/fusion")
    def fusion_evaluation():
        return jsonify(evaluate_fusion_availability())

    @app.get("/api/explain/<application_id>")
    def explain_application(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        try:
            bundle = build_bundle(application_id)
            explanation_id = save_explanation_audit(application_id, bundle["assessment_id"], bundle)
            return jsonify({**bundle, "explanation_id": explanation_id})
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        except Exception:
            app.logger.exception("Explanation generation failed")
            return jsonify({"error": "Explanation generation is temporarily unavailable."}), 500

    @app.get("/api/explain/<application_id>/shap")
    def explain_shap(application_id: str):
        response = explain_application(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, **payload["shap"]})

    @app.get("/api/explain/<application_id>/lime")
    def explain_lime(application_id: str):
        response = explain_application(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, **payload["lime"]})

    @app.get("/api/explain/<application_id>/evidence")
    def explain_evidence(application_id: str):
        response = explain_application(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, "evidence": payload["evidence"]})

    @app.get("/api/explain/<application_id>/graph")
    def explain_graph(application_id: str):
        response = explain_application(application_id)
        if isinstance(response, tuple):
            return response
        payload = response.get_json()
        return jsonify({"application_id": application_id, "graph": payload["graph"]})

    @app.post("/api/explain/<application_id>/counterfactual")
    def explain_counterfactual(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        assessment = get_latest_assessment_for_application(application_id)
        if assessment is None:
            return jsonify({"error": "Assess the application before running a counterfactual."}), 404
        try:
            result = simulate(assessment["applicant"], request.get_json(silent=True) or {})
            return jsonify({"application_id": application_id, **result})
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 422

    @app.get("/api/fairness/report")
    def fairness_evaluation():
        try:
            return jsonify(fairness_report())
        except (FileNotFoundError, ValueError) as error:
            return jsonify({"error": str(error)}), 503

    @app.get("/api/model-card")
    def model_card_endpoint():
        return jsonify(model_card())

    @app.get("/api/explain/<application_id>/audit")
    def explanation_audit(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        return jsonify({"application_id": application_id, "audits": list_explanation_audits(application_id)})

    @app.get("/api/dashboard")
    def dashboard_data():
        try:
            return jsonify(dashboard_snapshot())
        except Exception:
            app.logger.exception("Dashboard aggregation failed")
            return jsonify({"error": "Dashboard data is temporarily unavailable."}), 500

    @app.get("/api/applications")
    def applications_data():
        return jsonify({"applications": application_analytics({key: request.args.get(key) for key in ("status", "risk_level", "application_id") if request.args.get(key)})})

    @app.get("/api/audit")
    def audit_data():
        return jsonify({"audits": list_fusion_audits()})

    @app.get("/api/report/<application_id>")
    def application_report(application_id: str):
        if not application_exists(application_id):
            return jsonify({"error": "Application not found."}), 404
        assessment = get_latest_assessment_for_application(application_id)
        profile_record = get_latest_fusion_assessment(application_id)
        if assessment is None or profile_record is None:
            return jsonify({"error": "A completed assessment report is not available."}), 404
        return send_file(build_final_report(assessment, profile_record["profile"]), mimetype="application/pdf", as_attachment=True, download_name=f"loanlens-{application_id}.pdf")

    @app.get("/api/system/health")
    def system_health():
        checks = {}
        try:
            from history_store import _connection
            connection = _connection()
            connection.execute("SELECT 1")
            connection.close()
            checks["database"] = "HEALTHY"
        except Exception:
            checks["database"] = "ERROR"
        checks["flask"] = "HEALTHY"
        checks["ml_model"] = "HEALTHY" if (PROJECT_ROOT / "model" / "saved_model" / "loan_approval_pipeline.joblib").exists() else "ERROR"
        checks["ocr_service"] = "HEALTHY" if __import__("shutil").which("tesseract") else "WARNING"
        try:
            import networkx
            checks["graph_engine"] = "HEALTHY"
        except ImportError:
            checks["graph_engine"] = "ERROR"
        try:
            from services.fusion.fusion_engine import FusionEngine
            FusionEngine()
            checks["fusion_engine"] = "HEALTHY"
        except Exception:
            checks["fusion_engine"] = "ERROR"
        try:
            import shap
            checks["explainability"] = "HEALTHY"
        except ImportError:
            checks["explainability"] = "WARNING"
        overall = "ERROR" if "ERROR" in checks.values() else "WARNING" if "WARNING" in checks.values() else "HEALTHY"
        return jsonify({"status": overall, "checks": checks, "debug": bool(app.config.get("DEBUG"))})

    @app.get("/reports/<int:assessment_id>.pdf")
    def assessment_report(assessment_id: int):
        assessment = get_assessment(assessment_id)
        if assessment is None:
            return jsonify({"error": "Assessment report not found."}), 404
        return send_file(build_report(assessment), mimetype="application/pdf", as_attachment=True, download_name=f"loanlens-assessment-{assessment_id}.pdf")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
