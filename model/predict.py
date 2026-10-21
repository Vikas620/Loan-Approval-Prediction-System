"""Load the Phase 3 pipeline and predict one applicant assessment."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    from .preprocess import add_domain_features
except ImportError:
    from preprocess import add_domain_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "saved_model" / "loan_approval_pipeline.joblib"


def predict_applicant(
    applicant: dict,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, int | float]:
    """Return class, approval probability, and a 0-100 risk score.

    The risk score is the complement of the model's approval probability. This
    is an automated model assessment, not a guarantee of actual approval.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found at {path}. Run model/train_model.py first.")
    if not applicant:
        raise ValueError("Applicant input must contain at least one field.")

    input_data = add_domain_features(pd.DataFrame([applicant]))
    input_data = input_data.drop(columns=["Loan_ID", "Loan_Status"], errors="ignore")
    pipeline = joblib.load(path)
    try:
        probability = float(pipeline.predict_proba(input_data)[0, 1])
        predicted_class = int(pipeline.predict(input_data)[0])
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Applicant input does not match the trained feature schema: {error}") from error
    return {
        "predicted_class": predicted_class,
        "approval_probability": probability,
        "risk_score": float((1.0 - probability) * 100.0),
        "model_name": "Gradient Boosting",
        "model_version": "phase3-gradient-boosting-v1",
    }