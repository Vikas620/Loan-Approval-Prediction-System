"""NetworkX graph feature calculation."""

from __future__ import annotations

import networkx as nx


def calculate_features(graph: nx.MultiDiGraph, source: dict, application_id: str) -> dict:
    undirected = graph.to_undirected()
    applicant_id = f"applicant:{application_id}"
    component = nx.node_connected_component(undirected, applicant_id) if applicant_id in undirected else set()
    document_hashes = [item["file_hash"] for item in source.get("documents", [])]
    all_documents = source.get("all_documents", [])
    reuse_count = sum(1 for file_hash in document_hashes if sum(item["file_hash"] == file_hash for item in all_documents) > 1)
    employer_nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "Employer"]
    shared_employer_count = 0
    for employer_node in employer_nodes:
        employer_label = graph.nodes[employer_node].get("label")
        shared_employer_count += sum(1 for record in source.get("all_evidence", []) if record["field_name"] == "Employment" and record.get("normalized_value", "").casefold() == str(employer_label).casefold())
    shared_bank_count = sum(1 for record in source.get("all_evidence", []) if record["field_name"] == "BankEntity" and str(record.get("normalized_value", "")).startswith("BANK_"))
    connected_applications = {item["application_id"] for item in all_documents if item["file_hash"] in document_hashes}
    return {
        "node_degree": int(graph.degree(applicant_id)) if applicant_id in graph else 0,
        "relationship_count": int(graph.number_of_edges()),
        "node_count": int(graph.number_of_nodes()),
        "connected_component_size": len(component),
        "connected_application_count": len(connected_applications),
        "document_reuse_count": reuse_count,
        "shared_entity_count": max(0, shared_employer_count - 1) + max(0, shared_bank_count - 1),
        "neighborhood_size": len(set(graph.neighbors(applicant_id))) if applicant_id in graph else 0,
        "suspicious_cluster_size": max(0, len(connected_applications) - 1),
    }
