"""Build a safe, graph-ready application relationship network."""

from __future__ import annotations

import networkx as nx


NODE_TYPES = {"Applicant", "Application", "Employer", "Document", "IncomeEvidence", "BankEntity"}


def _add_edge(graph: nx.MultiDiGraph, edges: list[dict], source: str, source_type: str, relationship: str, target: str, target_type: str, evidence_id: str | None = None) -> None:
    graph.add_edge(source, target, relationship=relationship, evidence_id=evidence_id)
    edges.append({"source_node": source, "source_type": source_type, "relationship": relationship, "target_node": target, "target_type": target_type, "evidence_id": evidence_id})


def build_application_graph(application_id: str, source: dict) -> tuple[nx.MultiDiGraph, list[dict], list[dict]]:
    """Build an application graph from Phase 6 records without raw document text."""
    graph = nx.MultiDiGraph()
    nodes: list[dict] = []
    edges: list[dict] = []
    applicant_id = f"applicant:{application_id}"
    application_node = f"application:{application_id}"

    def add_node(node_id: str, node_type: str, metadata: dict) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unsupported graph node type: {node_type}")
        if not graph.has_node(node_id):
            graph.add_node(node_id, node_type=node_type, **metadata)
            nodes.append({"node_id": node_id, "node_type": node_type, "metadata": metadata})

    add_node(applicant_id, "Applicant", {"label": "Applicant", "safe_id": applicant_id})
    add_node(application_node, "Application", {"label": "Application", "safe_id": application_id})
    _add_edge(graph, edges, applicant_id, "Applicant", "SUBMITTED", application_node, "Application")
    _add_edge(graph, edges, applicant_id, "Applicant", "HAS_APPLICATION", application_node, "Application")

    evidence_by_document = {}
    for record in source.get("evidence", []):
        evidence_by_document.setdefault(record["document_id"], []).append(record)

    for document in source.get("documents", []):
        document_id = f"document:{document['document_id']}"
        add_node(document_id, "Document", {"label": document.get("document_type", "Document"), "safe_id": document_id, "confidence": document.get("confidence_score", 0)})
        _add_edge(graph, edges, application_node, "Application", "SUPPORTED_BY", document_id, "Document")
        for record in evidence_by_document.get(document["document_id"], []):
            if record["field_name"] not in {"ApplicantIncome", "TotalIncome", "LoanAmount"}:
                continue
            evidence_id = f"evidence:{record['evidence_id']}"
            add_node(evidence_id, "IncomeEvidence", {"label": record["field_name"], "safe_id": evidence_id, "confidence": record["confidence"]})
            _add_edge(graph, edges, document_id, "Document", "CONTAINS", evidence_id, "IncomeEvidence", record["evidence_id"])

    employer_values = [record["normalized_value"] for record in source.get("evidence", []) if record["field_name"] == "Employment" and record.get("normalized_value")]
    for employer in sorted(set(employer_values)):
        employer_id = f"employer:{employer.casefold().replace(' ', '_')}"
        add_node(employer_id, "Employer", {"label": employer, "safe_id": employer_id})
        _add_edge(graph, edges, applicant_id, "Applicant", "EMPLOYED_BY", employer_id, "Employer")

    bank_values = [record["normalized_value"] for record in source.get("evidence", []) if record["field_name"] == "BankEntity" and str(record.get("normalized_value", "")).startswith("BANK_")]
    for bank_entity in sorted(set(bank_values)):
        bank_id = f"bank:{bank_entity}"
        add_node(bank_id, "BankEntity", {"label": bank_entity, "safe_id": bank_id, "masked": True})
        _add_edge(graph, edges, applicant_id, "Applicant", "ASSOCIATED_WITH", bank_id, "BankEntity")

    return graph, nodes, edges
