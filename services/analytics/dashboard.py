"""Actual-data dashboard aggregations for Phase 10."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from services.explainability.fairness_engine import fairness_report
from services.fusion.evaluation import evaluate_fusion_availability


DB_PATH = Path(__file__).resolve().parents[2] / "instance" / "loanlens.sqlite3"


def _records() -> list[dict]:
    if not DB_PATH.exists():
        return []
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute("""SELECT a.id, a.application_id, a.created_at, a.applicant_name,
        a.decision, a.approval_probability, a.risk_score,
        f.overall_risk, f.status AS fusion_status, f.profile_json
        FROM assessments a LEFT JOIN fusion_assessments f ON f.fusion_id =
        (SELECT MAX(f2.fusion_id) FROM fusion_assessments f2 WHERE f2.assessment_id = a.id)
        ORDER BY a.created_at DESC""").fetchall()
    connection.close()
    records = []
    for row in rows:
        record = dict(row)
        profile = json.loads(record.pop("profile_json")) if record.get("profile_json") else {}
        record["profile"] = profile
        record["status"] = record.get("fusion_status") or record["decision"].upper()
        record["overall_risk"] = record.get("overall_risk") if record.get("overall_risk") is not None else record["risk_score"]
        record["risk_level"] = profile.get("risk_level") or ("LOW" if record["overall_risk"] <= 33 else "MEDIUM" if record["overall_risk"] <= 66 else "HIGH")
        records.append(record)
    return records


def _histogram(values: list[float]) -> dict:
    bins = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for value in values:
        key = "0-20" if value <= 20 else "21-40" if value <= 40 else "41-60" if value <= 60 else "61-80" if value <= 80 else "81-100"
        bins[key] += 1
    return bins


def dashboard_snapshot() -> dict:
    records = _records()
    statuses = Counter(record["status"] for record in records)
    levels = Counter(record["risk_level"] for record in records)
    risks = [float(record["overall_risk"]) for record in records]
    component_totals = Counter()
    component_counts = Counter()
    graph_signals = Counter()
    for record in records:
        for name, component in record["profile"].get("components", {}).items():
            component_totals[name] += component.get("score", 0)
            component_counts[name] += 1
        for signal in record["profile"].get("graph_analysis", {}).get("signals", []):
            graph_signals[signal.get("signal_name", "Other configured graph signal")] += 1
    component_average = {name: round(component_totals[name] / component_counts[name], 2) for name in component_totals}
    if risks:
        component_average["overall"] = round(sum(risks) / len(risks), 2)
    evidence_distribution = Counter()
    for record in records:
        evidence_risk = record["profile"].get("components", {}).get("evidence", {}).get("score")
        if evidence_risk is None:
            continue
        consistency = 100 - float(evidence_risk)
        evidence_distribution["High Consistency" if consistency >= 67 else "Medium Consistency" if consistency >= 34 else "Low Consistency"] += 1
    return {"last_updated": datetime.now(timezone.utc).isoformat(), "kpis": {"total_applications": len(records), "approved_or_low_risk": statuses.get("APPROVED", 0) + levels.get("LOW", 0), "rejected_or_high_risk": statuses.get("REJECTED", 0) + levels.get("HIGH", 0), "review_required": statuses.get("REVIEW_REQUIRED", 0), "average_risk_score": round(sum(risks) / len(risks), 2) if risks else None}, "risk_distribution": dict(levels), "status_distribution": dict(statuses), "risk_score_histogram": _histogram(risks) if risks else {}, "component_average": component_average, "evidence_consistency": dict(evidence_distribution), "graph_signals": dict(graph_signals), "applications": [{key: record.get(key) for key in ("id", "application_id", "created_at", "applicant_name", "decision", "status", "overall_risk", "risk_level", "approval_probability")} for record in records[:100]], "model_evaluation": evaluate_fusion_availability(), "fairness": fairness_report() if records else {"available": False, "reason": "No application assessments available."}, "empty": not bool(records)}


def application_analytics(filters: dict | None = None) -> list[dict]:
    filters = filters or {}
    records = _records()
    result = []
    for record in records:
        if filters.get("status") and record["status"] != filters["status"]:
            continue
        if filters.get("risk_level") and record["risk_level"] != filters["risk_level"]:
            continue
        if filters.get("application_id") and filters["application_id"] not in (record.get("application_id") or ""):
            continue
        result.append({key: record.get(key) for key in ("id", "application_id", "created_at", "applicant_name", "decision", "status", "overall_risk", "risk_level", "approval_probability")})
    return result
