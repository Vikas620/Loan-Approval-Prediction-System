"""Generate a downloadable PDF assessment report."""

from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_report(assessment: dict) -> BytesIO:
    stream = BytesIO()
    document = SimpleDocTemplate(stream, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#122f3b"), spaceAfter=4 * mm))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#53656b")))
    story = [Paragraph("LoanLens Assessment Report", styles["ReportTitle"]), Paragraph("Automated model assessment for informed human review. This report is not a guarantee of loan approval.", styles["Small"]), Spacer(1, 5 * mm)]

    result_data = [["Decision", assessment["decision"]], ["Approval probability", f"{assessment['approval_probability'] * 100:.2f}%"], ["Risk score", f"{assessment['risk_score']:.2f} / 100"], ["Assessment timestamp", assessment["created_at"]]]
    applicant = assessment.get("applicant", {})
    for key, label in (("Name", "Applicant name"), ("Education", "Education"), ("Property_Area", "Property area"), ("ApplicantIncome", "Applicant income"), ("LoanAmount", "Loan amount")):
        if applicant.get(key) is not None:
            result_data.append([label, str(applicant[key])])
    table = Table(result_data, colWidths=[52 * mm, 112 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e4f3ee")), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d5e2df")), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#102331")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([table, Spacer(1, 6 * mm), Paragraph("Documents processed", styles["Heading2"])])
    documents = [["File", "Extraction mode", "Duplicate"]] + [[doc.get("filename", ""), doc.get("method", ""), "Yes" if doc.get("duplicate") else "No"] for doc in assessment.get("documents", [])]
    document_table = Table(documents, colWidths=[76 * mm, 52 * mm, 30 * mm])
    document_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#122f3b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d5e2df")), ("PADDING", (0, 0), (-1, -1), 6)]))
    story.extend([document_table, Spacer(1, 6 * mm), Paragraph("Extracted information and validation", styles["Heading2"])])
    extracted = assessment.get("extraction", {})
    validation = assessment.get("validation", {})
    extracted_rows = [[key, str(value) if value else "Not detected"] for key, value in extracted.items()]
    extracted_rows.append(["Missing fields", ", ".join(validation.get("missing", [])) or "None"])
    extracted_rows.append(["Warnings", "; ".join(validation.get("warnings", [])) or "None"])
    extracted_table = Table(extracted_rows, colWidths=[52 * mm, 112 * mm])
    extracted_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d5e2df")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f7f5")), ("PADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(extracted_table)
    document.build(story)
    stream.seek(0)
    return stream


def build_final_report(assessment: dict, profile: dict, explanation: dict | None = None) -> BytesIO:
    """Build a safe Phase 10 report from actual assessment records."""
    stream = BytesIO()
    document = SimpleDocTemplate(stream, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="FinalTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#122f3b"), spaceAfter=4 * mm))
    styles.add(ParagraphStyle(name="FinalSmall", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#53656b")))
    story = [Paragraph("LoanLens AI-Powered Risk Assessment Report", styles["FinalTitle"]), Paragraph("Academic decision-support report. This is not an autonomous lending decision or a fraud finding.", styles["FinalSmall"]), Spacer(1, 5 * mm)]
    profile_rows = [["Application ID", str(assessment.get("application_id") or "Not linked")], ["Assessment timestamp", str(assessment.get("created_at"))], ["Model version", profile.get("baseline_prediction", {}).get("model_version", "Not available")], ["Fusion version", profile.get("fusion_version", "Not available")], ["ML risk", str(profile.get("components", {}).get("ml", {}).get("score", "Insufficient data"))], ["Evidence risk", str(profile.get("components", {}).get("evidence", {}).get("score", "Insufficient data"))], ["Graph risk", str(profile.get("components", {}).get("graph", {}).get("score", "Insufficient data"))], ["Overall risk", str(profile.get("overall_risk", "Insufficient data"))], ["Confidence", str(profile.get("confidence_level", "Insufficient data"))], ["Status", str(profile.get("status", "Insufficient data"))]]
    table = Table(profile_rows, colWidths=[52 * mm, 112 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e4f3ee")), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d5e2df")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([table, Spacer(1, 5 * mm), Paragraph("Why this result", styles["Heading2"])])
    for explanation_text in profile.get("explanations", []):
        story.append(Paragraph(explanation_text, styles["BodyText"]))
    story.extend([Spacer(1, 3 * mm), Paragraph("Review flags", styles["Heading2"])])
    flags = profile.get("review_flags", []) or ["None recorded"]
    for flag in flags:
        story.append(Paragraph(str(flag), styles["BodyText"]))
    story.extend([Spacer(1, 3 * mm), Paragraph("Explainability and limitations", styles["Heading2"]), Paragraph("SHAP/LIME availability, evidence provenance, graph signals, counterfactuals, and fairness limitations are available through the application APIs. No raw document content is included in this report.", styles["FinalSmall"])])
    document.build(story)
    stream.seek(0)
    return stream