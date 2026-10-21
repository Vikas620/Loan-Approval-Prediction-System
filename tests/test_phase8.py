"""Synthetic Phase 8 fusion tests; no real personal or financial documents."""

from io import BytesIO
import unittest

from tests.test_phase6 import demo_pdf
from services.fusion.fusion_engine import FusionEngine
from services.fusion.risk_components import RiskComponent


class Phase8FusionTests(unittest.TestCase):
    def test_weight_validation_and_score_bounds(self):
        with self.assertRaises(ValueError):
            FusionEngine({"ml": .5, "evidence": .5, "graph": .5})
        components = {name: RiskComponent(name, score, .8) for name, score in (("ml", 10), ("evidence", 30), ("graph", 80))}
        profile = FusionEngine().fuse(components, "APPROVED").to_dict()
        self.assertGreaterEqual(profile["overall_risk"], 0)
        self.assertLessEqual(profile["overall_risk"], 100)
        self.assertEqual(sum(profile["weights"].values()), 1)

    def test_disagreement_triggers_review(self):
        components = {"ml": RiskComponent("ML", 5, .95), "evidence": RiskComponent("Evidence", 90, .5, review_required=True), "graph": RiskComponent("Graph", 15, .8)}
        profile = FusionEngine().fuse(components, "APPROVED").to_dict()
        self.assertTrue(profile["disagreement"]["detected"])
        self.assertEqual(profile["status"], "REVIEW_REQUIRED")
        self.assertIn("Component Disagreement Detected", profile["review_flags"])

    def test_api_persists_profile_and_supports_scenario(self):
        from app import app

        app.config.update(TESTING=True)
        client = app.test_client()
        upload = client.post("/api/documents/upload", data={"document_type": "Salary Slip", "documents": (demo_pdf(["Name: Fusion Applicant", "Applicant income: 50000", "Employment: Demo Employer"]), "fusion.pdf")}, content_type="multipart/form-data").get_json()
        applicant = {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban", "ApplicantIncome": "4583", "CoapplicantIncome": "1508", "LoanAmount": "128", "Loan_Amount_Term": "360", "Credit_History": "1"}
        assessment = client.post("/api/assess", json={"application_id": upload["application_id"], "applicant": applicant, "documents": upload["documents"], "extraction": upload["extracted"], "validation": upload["validation"]})
        self.assertEqual(assessment.status_code, 200)
        self.assertIn("unified_risk_profile", assessment.get_json())
        application_id = upload["application_id"]
        for suffix in ("", "/components", "/explanation", "/disagreement", "/traceability"):
            self.assertEqual(client.get(f"/api/risk/{application_id}{suffix}").status_code, 200)
        scenario = client.post(f"/api/risk/{application_id}/scenario", json={"ApplicantIncome": "7000", "LoanAmount": "100"})
        self.assertEqual(scenario.status_code, 200)
        self.assertIn("difference", scenario.get_json())

    def test_evaluation_reports_actual_baseline_only(self):
        from app import app

        result = app.test_client().get("/api/evaluation/fusion")
        self.assertEqual(result.status_code, 200)
        payload = result.get_json()
        self.assertTrue(payload["available"])
        self.assertIn("brier_score", payload["baseline_metrics"])
        self.assertFalse(payload["ablation"]["ml_plus_evidence"]["available"])


if __name__ == "__main__":
    unittest.main()
