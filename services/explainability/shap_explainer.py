"""SHAP adapter for the saved Gradient Boosting pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

from model.preprocess import add_domain_features

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model" / "saved_model" / "loan_approval_pipeline.joblib"
ORIGINAL_FEATURES = (
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "TotalIncome", "LoanToIncomeRatio", "Gender", "Married",
    "Dependents", "Education", "Self_Employed", "Property_Area",
)


def _model_inputs(applicant: dict) -> tuple[object, pd.DataFrame, object]:
    pipeline = joblib.load(MODEL_PATH)
    data = add_domain_features(pd.DataFrame([applicant])).drop(columns=["Loan_ID", "Loan_Status"], errors="ignore")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(data)
    return model, data, transformed


def _base_feature(name: str) -> str:
    name = name.split("__", 1)[-1]
    for feature in sorted(ORIGINAL_FEATURES, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def explain_local(applicant: dict) -> dict:
    """Return actual SHAP local contributions or an explicit unavailable result."""
    if shap is None:
        return {"available": False, "method": "SHAP", "reason": "SHAP is not installed."}
    try:
        model, data, transformed = _model_inputs(applicant)
        pipeline = joblib.load(MODEL_PATH)
        names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        contributions = values[0].tolist()
        grouped = defaultdict(float)
        for name, value in zip(names, contributions):
            grouped[_base_feature(name)] += float(value)
        feature_values = data.iloc[0].to_dict()
        factors = []
        for feature, contribution in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:8]:
            factors.append({"feature": feature, "value": feature_values.get(feature), "contribution": round(contribution, 6), "direction": "increases model risk signal" if contribution > 0 else "decreases model risk signal"})
        expected = explainer.expected_value
        if hasattr(expected, "item"):
            expected = expected.item()
        return {"available": True, "method": "SHAP TreeExplainer", "model_name": "Gradient Boosting", "output_space": "raw model margin", "base_value": float(expected), "top_factors": factors, "note": "Contributions are actual SHAP values in the model's raw output space; they are not probabilities."}
    except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as error:
        return {"available": False, "method": "SHAP", "reason": f"SHAP could not be applied safely: {error}"}


def global_importance() -> dict:
    """Return actual fitted tree importances mapped to original features."""
    if not MODEL_PATH.exists():
        return {"available": False, "reason": "Saved model is unavailable."}
    try:
        pipeline = joblib.load(MODEL_PATH)
        model = pipeline.named_steps["model"]
        names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        grouped = defaultdict(float)
        for name, importance in zip(names, model.feature_importances_):
            grouped[_base_feature(name)] += float(importance)
        return {"available": True, "method": "GradientBoostingClassifier.feature_importances_", "features": [{"feature": name, "importance": round(value, 8)} for name, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)]}
    except (KeyError, AttributeError, ValueError, TypeError) as error:
        return {"available": False, "reason": f"Global importance unavailable: {error}"}
