"""Synthetic Phase 7 graph tests; no real identities or financial documents."""

from __future__ import annotations

import unittest

import networkx as nx
from tests.test_phase6 import demo_pdf

from services.graph.graph_builder import build_application_graph
from services.graph.graph_explainer import explain_analysis
from services.graph.graph_risk_engine import GraphRiskEngine


def scenario(application_id: str, document_hash: str, employer: str = "Employer_001", reused: bool = False) -> dict:
    return {
        "documents": [{"document_id": f"doc_{application_id}", "document_type": "Salary Slip", "file_hash": document_hash, "confidence_score": .94}],
        "evidence": [{"evidence_id": f"ev_{application_id}", "document_id": f"doc_{application_id}", "field_name": "ApplicantIncome", "normalized_value": "50000", "confidence": .94}, {"evidence_id": f"emp_{application_id}", "document_id": f"doc_{application_id}", "field_name": "Employment", "normalized_value": employer, "confidence": .94}],
        "all_documents": ([{"application_id": application_id, "document_id": f"doc_{application_id}", "file_hash": document_hash, "document_type": "Salary Slip"}, {"application_id": "app_other", "document_id": "doc_other", "file_hash": document_hash, "document_type": "Salary Slip"}] if reused else [{"application_id": application_id, "document_id": f"doc_{application_id}", "file_hash": document_hash, "document_type": "Salary Slip"}]),
        "all_evidence": [{"application_id": application_id, "document_id": f"doc_{application_id}", "field_name": "Employment", "normalized_value": employer}],
        "consistency_checks": [],
    }


class Phase7GraphTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_graph_nodes_and_edges(self):
        source = scenario("app_normal", "hash_normal")
        graph, nodes, edges = build_application_graph("app_normal", source)
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertGreaterEqual(len(nodes), 4)
        self.assertTrue(any(edge["relationship"] == "EMPLOYED_BY" for edge in edges))

    def test_normal_shared_employer_is_not_high_risk_by_itself(self):
        source = scenario("app_normal", "hash_normal")
        source["all_evidence"].append({"application_id": "app_other", "document_id": "doc_other", "field_name": "Employment", "normalized_value": "Employer_001"})
        graph, _, _ = build_application_graph("app_normal", source)
        result = GraphRiskEngine().analyze(graph, source, "app_normal")
        self.assertFalse(any(signal["severity"] == "HIGH" for signal in result["signals"]))

    def test_document_reuse_generates_explainable_signal(self):
        source = scenario("app_reused", "hash_shared", reused=True)
        graph, _, _ = build_application_graph("app_reused", source)
        result = GraphRiskEngine().analyze(graph, source, "app_reused")
        self.assertGreater(result["graph_risk_score"], 0)
        self.assertTrue(any(signal["signal_name"] == "Document Reuse" for signal in result["signals"]))
        self.assertLessEqual(result["graph_risk_score"], 100)
        self.assertIn("fingerprint", explain_analysis(result)["signals"][0]["why"])

    def test_shared_entity_and_reuse_requests_review(self):
        source = scenario("app_shared", "hash_shared", reused=True)
        source["all_evidence"].append({"application_id": "app_other", "document_id": "doc_other", "field_name": "Employment", "normalized_value": "Employer_001"})
        graph, _, _ = build_application_graph("app_shared", source)
        result = GraphRiskEngine().analyze(graph, source, "app_shared")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["status"], "REVIEW_REQUIRED")

    def test_empty_graph_score_is_bounded(self):
        graph, _, _ = build_application_graph("app_empty", {"documents": [], "evidence": [], "all_documents": [], "all_evidence": [], "consistency_checks": []})
        result = GraphRiskEngine().analyze(graph, {"documents": [], "evidence": [], "all_documents": [], "all_evidence": [], "consistency_checks": []}, "app_empty")
        self.assertGreaterEqual(result["graph_risk_score"], 0)
        self.assertLessEqual(result["graph_risk_score"], 100)

    def test_graph_api_endpoints(self):
        response = self.client.post("/api/documents/upload", data={"document_type": "Salary Slip", "documents": (demo_pdf(["Name: Applicant_001", "Applicant income: 50000", "Employment: Employer_001"]), "graph-synthetic.pdf")}, content_type="multipart/form-data")
        application_id = response.get_json()["application_id"]
        for suffix in ("", "/features", "/risk", "/signals", "/explanation"):
            api_response = self.client.get(f"/api/graph/{application_id}{suffix}")
            self.assertEqual(api_response.status_code, 200)
        graph = self.client.get(f"/api/graph/{application_id}").get_json()
        self.assertTrue(graph["nodes"])
        self.assertTrue(graph["edges"])


if __name__ == "__main__":
    unittest.main()
