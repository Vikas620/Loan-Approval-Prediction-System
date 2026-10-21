"""Actual-data evaluation helpers for the Phase 8 fusion report."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from model.preprocess import DEFAULT_OUTPUT, TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports" / "ml" / "model_metrics.json"
MODEL_PATH = ROOT / "model" / "saved_model" / "loan_approval_pipeline.joblib"


def evaluate_fusion_availability() -> dict:
    """Return actual baseline metrics and honest enhanced-evaluation status."""
    if not METRICS_PATH.exists() or not MODEL_PATH.exists():
        return {"available": False, "reason": "Phase 3 evaluation artifacts are unavailable."}
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(DEFAULT_OUTPUT)
    target = data[TARGET_COLUMN].astype(int)
    features = data.drop(columns=[TARGET_COLUMN, "Loan_ID"], errors="ignore")
    _, x_test, _, y_test = train_test_split(features, target, test_size=.2, random_state=report.get("random_state", 42), stratify=target)
    pipeline = joblib.load(MODEL_PATH)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    baseline_name = report.get("selected_model", "Selected Phase 3 model")
    baseline = dict(report["metrics"][baseline_name])
    baseline["brier_score"] = float(brier_score_loss(y_test, probabilities))
    return {
        "available": True,
        "ground_truth": "Phase 3 labeled holdout",
        "baseline_model": baseline_name,
        "baseline_metrics": baseline,
        "ablation": {
            "structured_ml_only": {"available": True, "metrics": baseline},
            "ml_plus_evidence": {"available": False, "reason": "No labeled evidence-fusion ground truth is present."},
            "ml_plus_graph": {"available": False, "reason": "No labeled graph-fusion ground truth is present."},
            "ml_plus_evidence_plus_graph": {"available": False, "reason": "No labeled multimodal-fusion ground truth is present."},
        },
        "calibration": {"available": True, "metric": "Brier score", "brier_score": baseline["brier_score"], "note": "Computed on the existing labeled holdout; probabilities were not modified."},
        "leakage_notes": ["The saved Phase 3 pipeline is evaluated on its existing stratified holdout.", "No fusion component labels are fabricated.", "Document and graph signals are not used to claim supervised fusion performance."],
    }
