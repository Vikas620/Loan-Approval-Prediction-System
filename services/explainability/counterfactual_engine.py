"""Validated, non-causal counterfactual simulation for demo inputs."""

from __future__ import annotations

from model.predict import predict_applicant


ALLOWED_FIELDS = {
    "ApplicantIncome": (0, 1_000_000),
    "CoapplicantIncome": (0, 1_000_000),
    "LoanAmount": (1, 10_000),
    "Loan_Amount_Term": (1, 480),
    "Credit_History": (0, 1),
}


def _validate_changes(changes: dict) -> dict:
    normalized = {}
    for field, value in changes.items():
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"Counterfactual field is not allowed: {field}")
        minimum, maximum = ALLOWED_FIELDS[field]
        numeric = float(value)
        if numeric < minimum or numeric > maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum} for simulation.")
        normalized[field] = int(numeric) if numeric.is_integer() else numeric
    return normalized


def simulate(applicant: dict, changes: dict) -> dict:
    normalized = _validate_changes(changes)
    original = predict_applicant(applicant)
    scenario_applicant = applicant.copy()
    scenario_applicant.update(normalized)
    scenario = predict_applicant(scenario_applicant)
    changed_features = [{"feature": field, "original": applicant.get(field), "simulated": value} for field, value in normalized.items() if applicant.get(field) != value]
    return {"original": {"predicted_class": original["predicted_class"], "approval_probability": original["approval_probability"], "risk_score": original["risk_score"]}, "scenario": {"predicted_class": scenario["predicted_class"], "approval_probability": scenario["approval_probability"], "risk_score": scenario["risk_score"]}, "difference": round(scenario["risk_score"] - original["risk_score"], 6), "changed_features": changed_features, "message": "Model response under simulated inputs; this is not a causal approval recommendation."}
