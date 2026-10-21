"""Dataset-specific fairness metrics with small-sample safeguards."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model.preprocess import DEFAULT_OUTPUT, TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[2]
MIN_GROUP_SIZE = 20
ATTRIBUTES = ("Gender", "Married", "Education", "Self_Employed")


def _metrics(y_true: pd.Series, predictions: object) -> dict:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"n": int(len(y_true)), "selection_rate": float(predictions.mean()), "true_positive_rate": float(tp / (tp + fn)) if tp + fn else None, "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None, "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None, "precision": float(precision_score(y_true, predictions, zero_division=0)), "recall": float(recall_score(y_true, predictions, zero_division=0)), "accuracy": float(accuracy_score(y_true, predictions))}


def fairness_report() -> dict:
    data = pd.read_csv(DEFAULT_OUTPUT)
    target = data[TARGET_COLUMN].astype(int)
    features = data.drop(columns=[TARGET_COLUMN, "Loan_ID"], errors="ignore")
    train, test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=42, stratify=target)
    pipeline = joblib.load(ROOT / "model" / "saved_model" / "loan_approval_pipeline.joblib")
    predictions = pd.Series(pipeline.predict(test), index=test.index)
    groups = {}
    warnings = []
    for attribute in ATTRIBUTES:
        if attribute not in test.columns:
            continue
        attribute_groups = {}
        for value in sorted(test[attribute].dropna().unique().tolist()):
            mask = test[attribute] == value
            if int(mask.sum()) < MIN_GROUP_SIZE:
                attribute_groups[str(value)] = {"n": int(mask.sum()), "available": False, "warning": "Insufficient sample size for reliable group comparison."}
                continue
            attribute_groups[str(value)] = {"available": True, **_metrics(y_test[mask], predictions[mask])}
        available = [item for item in attribute_groups.values() if item.get("available")]
        disparities = {}
        for metric in ("selection_rate", "true_positive_rate", "false_positive_rate"):
            values = [item[metric] for item in available if item.get(metric) is not None]
            disparities[f"{metric}_difference"] = max(values) - min(values) if len(values) >= 2 else None
        groups[attribute] = {"groups": attribute_groups, "disparities": disparities, "minimum_group_size": MIN_GROUP_SIZE}
        if any(not item.get("available", False) for item in attribute_groups.values()):
            warnings.append(f"{attribute}: one or more groups are below the minimum sample threshold.")
    return {"available": True, "attributes": groups, "warnings": warnings, "note": "Metrics are dataset-specific and do not establish fairness or regulatory compliance."}
