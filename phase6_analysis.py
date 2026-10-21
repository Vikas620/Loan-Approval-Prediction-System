"""Evidence-aware analysis primitives for Phase 6.

Signals are explainable heuristics, not proof of fraud or a validated
probability. The existing ML approval prediction remains independent.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from PIL import Image


INCOME_MISMATCH_REVIEW_RATIO = 0.15
INCOME_MISMATCH_HIGH_RATIO = 0.30
LOW_CONFIDENCE_THRESHOLD = 0.65


def confidence_for_method(method: str) -> float:
    return {"pdf-text": 0.94, "image-ocr": 0.74, "pdf-ocr": 0.70}.get(method, 0.50)


def integrity_checks(document: dict, path: Path, text: str) -> list[dict]:
    checks = [{"document_id": document["document_id"], "check_name": "file_hash", "status": "NORMAL", "detail": "SHA-256 hash recorded for duplicate and integrity comparison."}]
    if not text.strip():
        checks.append({"document_id": document["document_id"], "check_name": "text_quality", "status": "REVIEW_REQUIRED", "detail": "No text was extracted from the document."})
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(path) as image:
                width, height = image.size
            status = "WARNING" if width < 800 or height < 600 else "NORMAL"
            checks.append({"document_id": document["document_id"], "check_name": "image_dimensions", "status": status, "detail": f"Image dimensions: {width}x{height}."})
        except (OSError, ValueError):
            checks.append({"document_id": document["document_id"], "check_name": "image_dimensions", "status": "REVIEW_REQUIRED", "detail": "Image metadata could not be read."})
    if text.count("Name:") > 3 or text.count("Income:") > 3:
        checks.append({"document_id": document["document_id"], "check_name": "repeated_labels", "status": "WARNING", "detail": "Repeated labels may indicate a malformed or composite document."})
    return checks


def consistency_checks(declared: dict, document_fields: list[dict]) -> list[dict]:
    checks = []
    income_sources = []
    if declared.get("ApplicantIncome") is not None:
        income_sources.append(("Application Form", float(declared["ApplicantIncome"])))
    for document in document_fields:
        if document.get("fields", {}).get("ApplicantIncome"):
            try:
                income_sources.append((document["document_type"], float(document["fields"]["ApplicantIncome"])))
            except ValueError:
                pass
    if len(income_sources) >= 2:
        baseline_name, baseline = income_sources[0]
        for source_name, observed in income_sources[1:]:
            ratio = abs(observed - baseline) / max(abs(baseline), 1)
            severity = "INFO" if ratio <= 0.05 else "LOW" if ratio <= INCOME_MISMATCH_REVIEW_RATIO else "MEDIUM" if ratio <= INCOME_MISMATCH_HIGH_RATIO else "HIGH"
            checks.append({"check_name": "monthly_income_consistency", "severity": severity, "expected_value": baseline, "observed_value": observed, "source_documents": [baseline_name, source_name], "explanation": f"Income differs by {ratio * 100:.1f}% between {baseline_name} and {source_name}. Configured review thresholds are {INCOME_MISMATCH_REVIEW_RATIO * 100:.0f}% and {INCOME_MISMATCH_HIGH_RATIO * 100:.0f}%."})

    employer_sources = []
    for document in document_fields:
        employer = document.get("fields", {}).get("Employment")
        if employer:
            employer_sources.append((document["document_type"], employer))
    for index, (source_name, employer) in enumerate(employer_sources):
        if any(employer.casefold() != other.casefold() for _, other in employer_sources[index + 1:]):
            checks.append({"check_name": "employer_consistency", "severity": "HIGH", "expected_value": employer, "observed_value": [value for _, value in employer_sources], "source_documents": [source_name for source_name, _ in employer_sources], "explanation": "Employer values differ across supporting documents; human review is required."})
            break
    return checks


def evidence_summary(evidence: list[dict], checks: list[dict], integrity: list[dict], document_count: int) -> dict:
    confidence = mean([record["confidence"] for record in evidence]) if evidence else 0.0
    penalty = sum({"INFO": 0, "LOW": 3, "MEDIUM": 10, "HIGH": 25, "CRITICAL": 40}.get(check["severity"], 0) for check in checks)
    penalty += sum(15 for item in integrity if item["status"] == "WARNING")
    penalty += sum(30 for item in integrity if item["status"] == "REVIEW_REQUIRED")
    score = max(0.0, min(100.0, 100.0 - penalty - ((1.0 - confidence) * 30.0)))
    high_risk = sum(check["severity"] in {"HIGH", "CRITICAL"} for check in checks)
    review_required = high_risk > 0 or any(item["status"] == "REVIEW_REQUIRED" for item in integrity) or confidence < LOW_CONFIDENCE_THRESHOLD
    return {"evidence_consistency_score": round(score, 2), "evidence_confidence": round(confidence, 4), "document_count": document_count, "consistency_check_count": len(checks), "warning_count": sum(check["severity"] != "INFO" for check in checks), "high_risk_conflict_count": high_risk, "review_required": review_required, "review_reason": "Evidence quality or consistency requires human review." if review_required else "No configured evidence review trigger detected."}


def graph_edges(application_id: str, documents: list[dict], evidence: list[dict], extracted: dict) -> list[dict]:
    edges = [{"source_node": application_id, "source_type": "Application", "relationship": "HAS_DOCUMENT", "target_node": item["document_id"], "target_type": "Document"} for item in documents]
    for record in evidence:
        edges.append({"source_node": application_id, "source_type": "Application", "relationship": "SUPPORTED_BY", "target_node": record["document_id"], "target_type": "Document", "evidence_id": record["evidence_id"]})
    employer = extracted.get("Employment")
    if employer:
        edges.append({"source_node": application_id, "source_type": "Applicant", "relationship": "EMPLOYED_BY", "target_node": employer, "target_type": "Employer"})
    return edges