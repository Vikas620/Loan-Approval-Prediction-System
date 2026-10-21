"""Small local SQLite store for document assessments and generated reports."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / "instance" / "loanlens.sqlite3"


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        applicant_name TEXT,
        decision TEXT NOT NULL,
        approval_probability REAL NOT NULL,
        risk_score REAL NOT NULL,
        applicant_json TEXT NOT NULL,
        documents_json TEXT NOT NULL,
        extraction_json TEXT NOT NULL,
        validation_json TEXT NOT NULL
    )""")
    assessment_columns = {row["name"] for row in connection.execute("PRAGMA table_info(assessments)")}
    if "application_id" not in assessment_columns:
        connection.execute("ALTER TABLE assessments ADD COLUMN application_id TEXT")
    connection.execute("""CREATE TABLE IF NOT EXISTS applications (
        application_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'IN_PROGRESS'
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        application_id TEXT NOT NULL,
        document_type TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        safe_filename TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        upload_timestamp TEXT NOT NULL,
        processing_status TEXT NOT NULL,
        error_status TEXT,
        page_count INTEGER DEFAULT 0,
        ocr_status TEXT,
        extraction_status TEXT,
        confidence_score REAL DEFAULT 0,
        integrity_status TEXT DEFAULT 'NORMAL',
        FOREIGN KEY(application_id) REFERENCES applications(application_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS evidence_records (
        evidence_id TEXT PRIMARY KEY,
        application_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        field_name TEXT NOT NULL,
        extracted_value TEXT,
        normalized_value TEXT,
        page_number INTEGER DEFAULT 1,
        bounding_box TEXT,
        confidence REAL NOT NULL,
        extraction_method TEXT NOT NULL,
        verification_status TEXT NOT NULL DEFAULT 'UNVERIFIED',
        created_at TEXT NOT NULL,
        FOREIGN KEY(application_id) REFERENCES applications(application_id),
        FOREIGN KEY(document_id) REFERENCES documents(document_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS consistency_checks (
        check_id TEXT PRIMARY KEY,
        application_id TEXT NOT NULL,
        check_name TEXT NOT NULL,
        severity TEXT NOT NULL,
        expected_value TEXT,
        observed_value TEXT,
        source_documents TEXT NOT NULL,
        explanation TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(application_id) REFERENCES applications(application_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS integrity_checks (
        integrity_id TEXT PRIMARY KEY,
        application_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        check_name TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(application_id) REFERENCES applications(application_id),
        FOREIGN KEY(document_id) REFERENCES documents(document_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS graph_edges (
        edge_id TEXT PRIMARY KEY,
        application_id TEXT NOT NULL,
        source_node TEXT NOT NULL,
        source_type TEXT NOT NULL,
        relationship TEXT NOT NULL,
        target_node TEXT NOT NULL,
        target_type TEXT NOT NULL,
        evidence_id TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(application_id) REFERENCES applications(application_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS graph_nodes (
        node_id TEXT PRIMARY KEY,
        node_type TEXT NOT NULL,
        application_id TEXT,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS fusion_assessments (
        fusion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id INTEGER,
        application_id TEXT NOT NULL,
        ml_risk REAL NOT NULL,
        evidence_risk REAL NOT NULL,
        graph_risk REAL NOT NULL,
        overall_risk REAL NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        fusion_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS explanation_audits (
        explanation_id TEXT PRIMARY KEY,
        assessment_id INTEGER NOT NULL,
        application_id TEXT NOT NULL,
        model_version TEXT,
        fusion_version TEXT,
        explanation_method TEXT NOT NULL,
        top_features_json TEXT NOT NULL,
        evidence_references_json TEXT NOT NULL,
        graph_references_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    connection.commit()
    return connection


def create_application() -> str:
    application_id = uuid.uuid4().hex
    connection = _connection()
    connection.execute("INSERT INTO applications (application_id, created_at) VALUES (?, ?)", (application_id, datetime.now(timezone.utc).isoformat()))
    connection.commit()
    connection.close()
    return application_id


def application_exists(application_id: str) -> bool:
    connection = _connection()
    row = connection.execute("SELECT 1 FROM applications WHERE application_id = ?", (application_id,)).fetchone()
    connection.close()
    return row is not None


def save_document(application_id: str, document: dict) -> str:
    document_id = document.get("document_id") or f"doc_{uuid.uuid4().hex}"
    connection = _connection()
    try:
        connection.execute("""INSERT INTO documents
            (document_id, application_id, document_type, original_filename, safe_filename,
             file_hash, upload_timestamp, processing_status, error_status, page_count,
             ocr_status, extraction_status, confidence_score, integrity_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                document_id, application_id, document.get("document_type", "Other Supporting Document"),
                document["filename"], document["safe_filename"], document["file_hash"],
                document["upload_timestamp"], document.get("processing_status", "PROCESSED"),
                document.get("error_status"), document.get("page_count", 0), document.get("ocr_status"),
                document.get("extraction_status", "COMPLETED"), document.get("confidence_score", 0),
                document.get("integrity_status", "NORMAL"),
            ))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return document_id


def save_evidence(application_id: str, document_id: str, fields: dict, method: str, confidence: float) -> list[dict]:
    created_at = datetime.now(timezone.utc).isoformat()
    records = []
    connection = _connection()
    for field_name, value in fields.items():
        if value in (None, ""):
            continue
        evidence_id = f"ev_{uuid.uuid4().hex}"
        record = {"evidence_id": evidence_id, "application_id": application_id, "document_id": document_id, "field_name": field_name, "extracted_value": str(value), "normalized_value": str(value), "page_number": 1, "bounding_box": None, "confidence": confidence, "extraction_method": method, "verification_status": "UNVERIFIED", "created_at": created_at}
        connection.execute("""INSERT INTO evidence_records
            (evidence_id, application_id, document_id, field_name, extracted_value,
             normalized_value, page_number, bounding_box, confidence, extraction_method,
             verification_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", tuple(record.values()))
        records.append(record)
    connection.commit()
    connection.close()
    return records


def save_phase6_results(application_id: str, checks: list[dict], integrity: list[dict], edges: list[dict]) -> None:
    connection = _connection()
    created_at = datetime.now(timezone.utc).isoformat()
    for check in checks:
        connection.execute("INSERT INTO consistency_checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (f"chk_{uuid.uuid4().hex}", application_id, check["check_name"], check["severity"], json.dumps(check.get("expected_value")), json.dumps(check.get("observed_value")), json.dumps(check.get("source_documents", [])), check["explanation"], created_at))
    for item in integrity:
        connection.execute("INSERT INTO integrity_checks VALUES (?, ?, ?, ?, ?, ?, ?)", (f"int_{uuid.uuid4().hex}", application_id, item["document_id"], item["check_name"], item["status"], item["detail"], created_at))
    for edge in edges:
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (f"edge_{uuid.uuid4().hex}", application_id, edge["source_node"], edge["source_type"], edge["relationship"], edge["target_node"], edge["target_type"], edge.get("evidence_id"), created_at))
    connection.commit()
    connection.close()


def save_graph_records(application_id: str, nodes: list[dict], edges: list[dict]) -> None:
    """Persist graph nodes and evidence-linked relationship records."""
    connection = _connection()
    created_at = datetime.now(timezone.utc).isoformat()
    for node in nodes:
        connection.execute(
            "INSERT OR REPLACE INTO graph_nodes (node_id, node_type, application_id, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (node["node_id"], node["node_type"], application_id, json.dumps(node.get("metadata", {})), created_at),
        )
    for edge in edges:
        connection.execute(
            "INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"edge_{uuid.uuid4().hex}", application_id, edge["source_node"], edge["source_type"], edge["relationship"], edge["target_node"], edge["target_type"], edge.get("evidence_id"), created_at),
        )
    connection.commit()
    connection.close()


def get_graph_records(application_id: str) -> dict:
    connection = _connection()
    nodes = [dict(row) for row in connection.execute("SELECT * FROM graph_nodes WHERE application_id = ?", (application_id,))]
    edges = [dict(row) for row in connection.execute("SELECT * FROM graph_edges WHERE application_id = ?", (application_id,))]
    documents = [dict(row) for row in connection.execute("SELECT * FROM documents WHERE application_id = ?", (application_id,))]
    evidence = [dict(row) for row in connection.execute("SELECT * FROM evidence_records WHERE application_id = ?", (application_id,))]
    all_documents = [dict(row) for row in connection.execute("SELECT document_id, application_id, file_hash, document_type FROM documents")]
    all_evidence = [dict(row) for row in connection.execute("SELECT application_id, document_id, field_name, normalized_value FROM evidence_records")]
    connection.close()
    for node in nodes:
        node["metadata"] = json.loads(node.pop("metadata_json"))
    return {"nodes": nodes, "edges": edges, "documents": documents, "evidence": evidence, "all_documents": all_documents, "all_evidence": all_evidence}


def get_phase6_data(application_id: str) -> dict:
    connection = _connection()
    documents = [dict(row) for row in connection.execute("SELECT * FROM documents WHERE application_id = ? ORDER BY upload_timestamp", (application_id,))]
    evidence = [dict(row) for row in connection.execute("SELECT * FROM evidence_records WHERE application_id = ? ORDER BY created_at", (application_id,))]
    checks = [dict(row) for row in connection.execute("SELECT * FROM consistency_checks WHERE application_id = ? ORDER BY created_at", (application_id,))]
    integrity = [dict(row) for row in connection.execute("SELECT * FROM integrity_checks WHERE application_id = ? ORDER BY created_at", (application_id,))]
    edges = [dict(row) for row in connection.execute("SELECT * FROM graph_edges WHERE application_id = ? ORDER BY created_at", (application_id,))]
    connection.close()
    return {"application_id": application_id, "documents": documents, "evidence": evidence, "consistency_checks": checks, "integrity_checks": integrity, "graph_edges": edges}


def delete_document(application_id: str, document_id: str) -> dict | None:
    connection = _connection()
    row = connection.execute("SELECT * FROM documents WHERE application_id = ? AND document_id = ?", (application_id, document_id)).fetchone()
    if row is None:
        connection.close()
        return None
    document = dict(row)
    connection.execute("DELETE FROM evidence_records WHERE application_id = ? AND document_id = ?", (application_id, document_id))
    connection.execute("DELETE FROM integrity_checks WHERE application_id = ? AND document_id = ?", (application_id, document_id))
    connection.execute("DELETE FROM documents WHERE application_id = ? AND document_id = ?", (application_id, document_id))
    connection.commit()
    connection.close()
    return document


def save_assessment(applicant: dict, documents: list, extraction: dict, validation: dict, result: dict, application_id: str | None = None) -> int:
    connection = _connection()
    cursor = connection.execute(
        """INSERT INTO assessments
        (created_at, applicant_name, decision, approval_probability, risk_score,
         applicant_json, documents_json, extraction_json, validation_json, application_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            extraction.get("Name", "") or "Unnamed applicant",
            result["decision"],
            result["approval_probability"],
            result["risk_score"],
            json.dumps(applicant),
            json.dumps(documents),
            json.dumps(extraction),
            json.dumps(validation),
            application_id,
        ),
    )
    connection.commit()
    assessment_id = cursor.lastrowid
    connection.close()
    return int(assessment_id)


def get_latest_assessment_for_application(application_id: str) -> dict | None:
    connection = _connection()
    row = connection.execute("SELECT * FROM assessments WHERE application_id = ? ORDER BY id DESC LIMIT 1", (application_id,)).fetchone()
    connection.close()
    if row is None:
        return None
    result = dict(row)
    for field in ("applicant_json", "documents_json", "extraction_json", "validation_json"):
        result[field.removesuffix("_json")] = json.loads(result.pop(field))
    return result


def save_fusion_assessment(assessment_id: int | None, application_id: str, profile: dict) -> int:
    connection = _connection()
    cursor = connection.execute("""INSERT INTO fusion_assessments
        (assessment_id, application_id, ml_risk, evidence_risk, graph_risk,
         overall_risk, confidence, status, profile_json, fusion_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            assessment_id, application_id,
            profile["components"]["ml"]["score"], profile["components"]["evidence"]["score"],
            profile["components"]["graph"]["score"], profile["overall_risk"], profile["confidence"],
            profile["status"], json.dumps(profile), profile["fusion_version"], datetime.now(timezone.utc).isoformat(),
        ))
    connection.commit()
    fusion_id = cursor.lastrowid
    connection.close()
    return int(fusion_id)


def get_latest_fusion_assessment(application_id: str) -> dict | None:
    connection = _connection()
    row = connection.execute("SELECT * FROM fusion_assessments WHERE application_id = ? ORDER BY fusion_id DESC LIMIT 1", (application_id,)).fetchone()
    connection.close()
    if row is None:
        return None
    result = dict(row)
    result["profile"] = json.loads(result.pop("profile_json"))
    return result


def list_fusion_audits(limit: int = 100) -> list[dict]:
    connection = _connection()
    rows = [dict(row) for row in connection.execute("SELECT fusion_id, assessment_id, application_id, ml_risk, evidence_risk, graph_risk, overall_risk, confidence, status, fusion_version, created_at FROM fusion_assessments ORDER BY created_at DESC LIMIT ?", (limit,))]
    connection.close()
    return rows


def save_explanation_audit(application_id: str, assessment_id: int, bundle: dict) -> str:
    explanation_id = f"ex_{uuid.uuid4().hex}"
    shap_factors = bundle.get("shap", {}).get("top_factors", [])
    connection = _connection()
    connection.execute("""INSERT INTO explanation_audits
        (explanation_id, assessment_id, application_id, model_version, fusion_version,
         explanation_method, top_features_json, evidence_references_json,
         graph_references_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        explanation_id, assessment_id, application_id,
        bundle.get("profile", {}).get("baseline_prediction", {}).get("model_version"),
        bundle.get("profile", {}).get("fusion_version"),
        "SHAP + LIME adapter + evidence + graph",
        json.dumps(shap_factors), json.dumps(bundle.get("evidence", [])),
        json.dumps(bundle.get("graph", {}).get("signals", [])),
        datetime.now(timezone.utc).isoformat(),
    ))
    connection.commit()
    connection.close()
    return explanation_id


def list_explanation_audits(application_id: str) -> list[dict]:
    connection = _connection()
    rows = [dict(row) for row in connection.execute("SELECT * FROM explanation_audits WHERE application_id = ? ORDER BY created_at DESC", (application_id,))]
    connection.close()
    for row in rows:
        for field in ("top_features_json", "evidence_references_json", "graph_references_json"):
            row[field.removesuffix("_json")] = json.loads(row.pop(field))
    return rows


def get_assessment(assessment_id: int) -> dict | None:
    connection = _connection()
    row = connection.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    connection.close()
    if row is None:
        return None
    result = dict(row)
    for field in ("applicant_json", "documents_json", "extraction_json", "validation_json"):
        result[field.removesuffix("_json")] = json.loads(result.pop(field))
    return result


def list_assessments(limit: int = 20) -> list[dict]:
    connection = _connection()
    rows = connection.execute("SELECT id, created_at, applicant_name, decision, approval_probability, risk_score FROM assessments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    connection.close()
    return [dict(row) for row in rows]