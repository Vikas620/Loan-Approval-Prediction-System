"""Local document extraction and validation for Phase 5.

This module extracts text only. It does not verify government identity
documents, perform KYC, or make a lending decision.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image
from werkzeug.utils import secure_filename

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - reported by the upload route
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - reported by the upload route
    pytesseract = None


ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_digest(file_storage) -> str:
    position = file_storage.stream.tell()
    file_storage.stream.seek(0)
    digest = hashlib.sha256(file_storage.stream.read()).hexdigest()
    file_storage.stream.seek(position)
    return digest


def extract_text(path: Path) -> tuple[str, str]:
    """Extract native PDF text or OCR an image/scanned page."""
    extension = path.suffix.lower()
    if extension == ".pdf":
        if fitz is None:
            raise RuntimeError("PDF extraction dependency is unavailable.")
        document = fitz.open(path)
        text = "\n".join(page.get_text() for page in document).strip()
        if text:
            return text, "pdf-text"
        if pytesseract is None:
            raise RuntimeError("This PDF is scanned. Install and configure Tesseract OCR to process it.")
        pages = []
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            try:
                pages.append(pytesseract.image_to_string(image))
            except pytesseract.TesseractNotFoundError as error:
                raise RuntimeError("This scanned PDF requires the native Tesseract OCR executable on PATH.") from error
        return "\n".join(pages).strip(), "pdf-ocr"

    if pytesseract is None:
        raise RuntimeError("Image OCR dependency is unavailable.")
    try:
        text = pytesseract.image_to_string(Image.open(path)).strip()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("Image OCR requires the native Tesseract OCR executable on PATH.") from error
    return text, "image-ocr"


def _first_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip(" .,:;\t")
    return ""


def _number(value: str) -> str:
    return re.sub(r"[^0-9.]", "", value)


def extract_fields(text: str) -> dict[str, str]:
    """Extract reviewable fields from common labeled document text."""
    return {
        "Name": _first_match(text, [r"(?:full\s+name|name)\s*[:\-]\s*([^\n]+)"]),
        "Date of birth": _first_match(text, [r"(?:date\s+of\s+birth|dob)\s*[:\-]\s*([^\n]+)"]),
        "ApplicantIncome": _number(_first_match(text, [r"(?:applicant\s+income|monthly\s+income|income)\s*[:\-]\s*([^\n]+)" ])),
        "Employment": _first_match(text, [r"(?:employment|employer|self[_\s-]?employed)\s*[:\-]\s*([^\n]+)"]),
        "LoanAmount": _number(_first_match(text, [r"(?:loan\s+amount|requested\s+loan)\s*[:\-]\s*([^\n]+)" ])),
        "Loan purpose": _first_match(text, [r"(?:loan\s+purpose|purpose)\s*[:\-]\s*([^\n]+)"]),
        "Existing liabilities": _number(_first_match(text, [r"(?:existing\s+liabilities|liabilities|debts)\s*[:\-]\s*([^\n]+)" ])),
        "Credit_History": _first_match(text, [r"(?:credit\s+history|credit\s+score)\s*[:\-]\s*([^\n]+)"]),
        "Education": _first_match(text, [r"education\s*[:\-]\s*([^\n]+)"]),
        "Property_Area": _first_match(text, [r"(?:property\s+area|residence)\s*[:\-]\s*([^\n]+)"]),
        "BankEntity": _first_match(text, [r"(?:bank\s+entity|masked\s+bank)\s*[:\-]\s*(BANK_[A-Z0-9_-]+)"]),
    }


def validate_extracted_fields(fields: dict[str, str]) -> dict:
    warnings = []
    missing = [name for name in ("Name", "ApplicantIncome", "LoanAmount") if not fields.get(name)]
    if not fields.get("Name"):
        warnings.append("Applicant name was not found; review the source document.")
    for field in ("ApplicantIncome", "LoanAmount", "Existing liabilities"):
        value = fields.get(field)
        if value and (not value.replace(".", "", 1).isdigit() or float(value) < 0):
            warnings.append(f"{field} has an invalid numeric value.")
    if fields.get("Credit_History") and fields["Credit_History"] not in {"0", "1"}:
        warnings.append("Credit history should be 0 or 1 for model mapping.")
    return {"missing": missing, "warnings": warnings, "valid": not warnings and not missing}