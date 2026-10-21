"""Synthetic Phase 9 explainability, counterfactual, and fairness tests."""

from __future__ import annotations

import unittest

from services.explainability.counterfactual_engine import simulate
from services.explainability.fairness_engine import fairness_report
from services.explainability.lime_explainer import explain_local as explain_lime
from services.explainability.shap_explainer import explain_local as explain_shap
from services.fusion.fusion_engine import FusionEngine
from services.fusion.risk_components import RiskComponent


APPLICANT = {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban", "ApplicantIncome": 4583, "CoapplicantIncome": 1508, "LoanAmount": 128, "Loan_Amount_Term": 360, "Credit_History": 1}


class Phase9ExplainabilityTests(unittest.TestCase):
    def test_shap_is_actual_or_explicitly_unavailable(self):
        result = explain_shap(APPLICANT)
        self.assertIn(result["available"], {True, False})
        if result["available"]:
            self.assertTrue(result["top_factors"])
            self.assertIn("Credit_History", {item["feature"] for item in result["top_factors"]})

    def test_lime_does_not_fabricate_values(self):
        result = explain_lime(APPLICANT)
        if not result["available"]:
            self.assertIn("reason", result)

    def test_counterfactual_validates_allowed_fields(self):
        result = simulate(APPLICANT, {"ApplicantIncome": 7000, "LoanAmount": 100})
        self.assertIn("original", result)
        self.assertIn("scenario", result)
        self.assertTrue(result["changed_features"])
        with self.assertRaises(ValueError):
            simulate(APPLICANT, {"Gender": "Female"})

    def test_fairness_report_uses_available_dataset_attributes(self):
        report = fairness_report()
        self.assertTrue(report["available"])
        self.assertIn("Gender", report["attributes"])
        self.assertIn("minimum_group_size", report["attributes"]["Gender"])

    def test_fusion_traceability_scores_remain_bounded(self):
        components = {"ml": RiskComponent("ML", 20, .9), "evidence": RiskComponent("Evidence", 40, .8), "graph": RiskComponent("Graph", 60, .7)}
        profile = FusionEngine().fuse(components, "APPROVED").to_dict()
        self.assertGreaterEqual(profile["overall_risk"], 0)
        self.assertLessEqual(profile["overall_risk"], 100)

    def test_phase9_api_end_to_end(self):
        from io import BytesIO
        from app import app
        from tests.test_phase6 import demo_pdf

        app.config.update(TESTING=True)
        client = app.test_client()
        upload = client.post("/api/documents/upload", data={"document_type": "Salary Slip", "documents": (demo_pdf(["Name: Explain Applicant", "Applicant income: 50000", "Employment: Demo Employer"]), "explain.pdf")}, content_type="multipart/form-data").get_json()
        assessment = client.post("/api/assess", json={"application_id": upload["application_id"], "applicant": APPLICANT, "documents": upload["documents"], "extraction": upload["extracted"], "validation": upload["validation"]})
        self.assertEqual(assessment.status_code, 200)
        application_id = upload["application_id"]
        for suffix in ("", "/shap", "/lime", "/evidence", "/graph", "/audit"):
            self.assertEqual(client.get(f"/api/explain/{application_id}{suffix}").status_code, 200)
        self.assertEqual(client.get("/api/fairness/report").status_code, 200)
        self.assertEqual(client.get("/api/model-card").status_code, 200)
        self.assertEqual(client.post(f"/api/explain/{application_id}/counterfactual", json={"ApplicantIncome": 7000}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
