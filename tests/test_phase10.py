"""Synthetic Phase 10 dashboard, reporting, health, and security tests."""

from __future__ import annotations

from io import BytesIO
import unittest

from tests.test_phase6 import demo_pdf


class Phase10PlatformTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_dashboard_analytics_are_data_driven(self):
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("kpis", payload)
        self.assertIn("risk_distribution", payload)
        self.assertIn("model_evaluation", payload)

    def test_application_filters_and_audit(self):
        self.assertEqual(self.client.get("/api/applications?status=REVIEW_REQUIRED").status_code, 200)
        self.assertEqual(self.client.get("/api/audit").status_code, 200)

    def test_system_health_and_existing_routes(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)
        health = self.client.get("/api/system/health")
        self.assertEqual(health.status_code, 200)
        self.assertIn(health.get_json()["status"], {"HEALTHY", "WARNING", "ERROR"})

    def test_application_report_and_invalid_upload(self):
        upload = self.client.post("/api/documents/upload", data={"document_type": "Salary Slip", "documents": (demo_pdf(["Name: Report Applicant", "Applicant income: 50000"]), "report.pdf")}, content_type="multipart/form-data").get_json()
        applicant = {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban", "ApplicantIncome": "4583", "CoapplicantIncome": "1508", "LoanAmount": "128", "Loan_Amount_Term": "360", "Credit_History": "1"}
        self.assertEqual(self.client.post("/api/assess", json={"application_id": upload["application_id"], "applicant": applicant, "documents": upload["documents"], "extraction": upload["extracted"], "validation": upload["validation"]}).status_code, 200)
        report = self.client.get(f"/api/report/{upload['application_id']}")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.content_type, "application/pdf")
        invalid = self.client.post("/api/documents/upload", data={"documents": (BytesIO(b"invalid"), "unsafe.txt")}, content_type="multipart/form-data")
        self.assertEqual(invalid.status_code, 422)


if __name__ == "__main__":
    unittest.main()
