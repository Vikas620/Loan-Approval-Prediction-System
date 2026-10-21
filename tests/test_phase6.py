"""Synthetic Phase 6 tests; no real identity or financial documents are used."""

from io import BytesIO
import unittest

from reportlab.pdfgen import canvas

from app import app
from phase6_analysis import evidence_summary


def demo_pdf(lines: list[str]) -> BytesIO:
    stream = BytesIO()
    document = canvas.Canvas(stream)
    for index, line in enumerate(lines):
        document.drawString(72, 780 - index * 25, line)
    document.save()
    stream.seek(0)
    return stream


class Phase6FlowTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_existing_routes_and_prediction_are_preserved(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)
        payload = {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban", "ApplicantIncome": "4583", "CoapplicantIncome": "1508", "LoanAmount": "128", "Loan_Amount_Term": "360", "Credit_History": "1"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("approval_probability", response.get_json())

    def test_upload_provenance_and_summary(self):
        response = self.client.post("/api/documents/upload", data={"document_type": "Salary Slip", "documents": (demo_pdf(["Name: Synthetic Applicant", "Applicant income: 50000", "Employment: Demo Corp"]), "salary.pdf")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result["application_id"])
        self.assertEqual(result["documents"][0]["method"], "pdf-text")
        self.assertTrue(result["evidence"])
        evidence_response = self.client.get(f"/api/evidence/{result['application_id']}")
        self.assertEqual(evidence_response.status_code, 200)
        self.assertEqual(evidence_response.get_json()["evidence"][0]["page_number"], 1)

    def test_income_mismatch_creates_high_conflict(self):
        response = self.client.post("/api/documents/upload", data={"document_type": "Salary Slip", "documents": [(demo_pdf(["Name: Synthetic Applicant", "Applicant income: 45000"]), "salary.pdf"), (demo_pdf(["Name: Synthetic Applicant", "Applicant income: 80000"]), "statement.pdf")]}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        checks = response.get_json()["consistency_checks"]
        self.assertTrue(any(check["severity"] == "HIGH" for check in checks))
        self.assertTrue(response.get_json()["evidence_summary"]["review_required"])

    def test_invalid_upload_is_rejected(self):
        response = self.client.post("/api/documents/upload", data={"documents": (BytesIO(b"not supported"), "secret.txt")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 422)

    def test_oversized_upload_is_rejected(self):
        response = self.client.post("/api/documents/upload", data={"documents": (BytesIO(b"x" * (10 * 1024 * 1024 + 1)), "large.pdf")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 422)

    def test_duplicate_documents_are_flagged(self):
        content = demo_pdf(["Name: Synthetic Applicant", "Applicant income: 50000"]).read()
        response = self.client.post("/api/documents/upload", data={"documents": [(BytesIO(content), "one.pdf"), (BytesIO(content), "two.pdf")]}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(item["duplicate"] for item in response.get_json()["documents"]))

    def test_low_confidence_and_missing_evidence_request_review(self):
        summary = evidence_summary([], [], [{"status": "REVIEW_REQUIRED"}], 0)
        self.assertTrue(summary["review_required"])
        self.assertEqual(summary["document_count"], 0)

    def test_employer_mismatch_creates_high_conflict(self):
        response = self.client.post("/api/documents/upload", data={"document_type": "Employment Letter", "documents": [(demo_pdf(["Name: Synthetic Applicant", "Employment: Demo Corp"]), "employment.pdf"), (demo_pdf(["Name: Synthetic Applicant", "Employment: Other Corp"]), "other.pdf")]}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(check["check_name"] == "employer_consistency" and check["severity"] == "HIGH" for check in response.get_json()["consistency_checks"]))

    def test_assessment_report_and_graph_endpoint(self):
        upload = self.client.post("/api/documents/upload", data={"document_type": "Employment Letter", "documents": (demo_pdf(["Name: Synthetic Applicant", "Employment: Demo Corp"]), "employment.pdf")}, content_type="multipart/form-data").get_json()
        application_id = upload["application_id"]
        applicant = {"Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban", "ApplicantIncome": "4583", "CoapplicantIncome": "0", "LoanAmount": "128", "Loan_Amount_Term": "360", "Credit_History": "1"}
        assessment = self.client.post("/api/assess", json={"application_id": application_id, "applicant": applicant, "documents": upload["documents"], "extraction": upload["extracted"], "validation": upload["validation"]}).get_json()
        self.assertIn(assessment["review_state"], {"APPROVED", "REJECTED", "REVIEW_REQUIRED"})
        self.assertEqual(self.client.get(f"/api/application/{application_id}/graph").status_code, 200)
        report = self.client.get(f"/reports/{assessment['assessment_id']}.pdf")
        self.assertEqual(report.content_type, "application/pdf")


if __name__ == "__main__":
    unittest.main()
