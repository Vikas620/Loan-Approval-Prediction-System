"""Train, compare, evaluate, and persist loan approval classifiers.

This is Phase 3 only. It trains models and saves evaluation artifacts; it does
not expose a Flask API or create SHAP explanations.

Run from the project root:
    python model/train_model.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocess import DEFAULT_OUTPUT, TARGET_COLUMN, build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
MODEL_DIR = PROJECT_ROOT / "model" / "saved_model"
MODEL_PATH = MODEL_DIR / "loan_approval_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
RANDOM_STATE = 42


def load_processed_dataset(path: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load and validate the Phase 2 processed dataset."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {source}. Run 'python model/eda.py' first."
        )
    data = pd.read_csv(source)
    required_columns = {TARGET_COLUMN}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {sorted(missing)}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target and remove the source identifier from model features."""
    target = data[TARGET_COLUMN].astype(int)
    features = data.drop(columns=[TARGET_COLUMN]).copy()
    if "Loan_ID" in features.columns:
        features = features.drop(columns=["Loan_ID"])
    return features, target


def make_models() -> dict[str, object]:
    """Return the five required classifiers with reproducible settings."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(model: object, features: pd.DataFrame) -> Pipeline:
    """Fit preprocessing only inside the model pipeline to avoid leakage."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("model", model),
        ]
    )


def evaluate_model(
    pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], object, object]:
    """Evaluate one fitted pipeline using actual holdout predictions."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return metrics, predictions, probabilities


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_confusion_matrix(name: str, y_test: pd.Series, predictions: object) -> None:
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"confusion_matrix_{slugify(name)}.png", dpi=160)
    plt.close()


def save_roc_chart(roc_data: dict[str, tuple[object, object]]) -> None:
    plt.figure(figsize=(8, 6))
    for name, (y_test, probabilities) in roc_data.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.title("Loan Approval ROC Curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curves.png", dpi=160)
    plt.close()


def save_comparison_chart(metrics: dict[str, dict[str, float]]) -> None:
    comparison = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    long_format = comparison.melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=long_format, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Model Evaluation Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test-set score")
    plt.xticks(rotation=20)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=160)
    plt.close()


def select_model(metrics: dict[str, dict[str, float]]) -> str:
    """Select by F1, then ROC-AUC, then recall; all values are test results."""
    return max(
        metrics,
        key=lambda name: (
            metrics[name]["f1_score"],
            metrics[name]["roc_auc"],
            metrics[name]["recall"],
        ),
    )


def train_all(input_path: str | Path = DEFAULT_OUTPUT) -> dict:
    """Train all models, save reports, and persist the selected full pipeline."""
    data = load_processed_dataset(input_path)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, float]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}
    roc_data: dict[str, tuple[object, object]] = {}
    for name, model in make_models().items():
        pipeline = make_pipeline(model, x_train)
        pipeline.fit(x_train, y_train)
        model_metrics, predictions, probabilities = evaluate_model(pipeline, x_test, y_test)
        metrics[name] = model_metrics
        fitted_pipelines[name] = pipeline
        roc_data[name] = (y_test, probabilities)
        save_confusion_matrix(name, y_test, predictions)

    selected_name = select_model(metrics)
    selected_pipeline = fitted_pipelines[selected_name]
    joblib.dump(selected_pipeline, MODEL_PATH)
    save_roc_chart(roc_data)
    save_comparison_chart(metrics)

    report = {
        "dataset_shape": list(data.shape),
        "target_column": TARGET_COLUMN,
        "excluded_identifier_columns": ["Loan_ID"] if "Loan_ID" in data.columns else [],
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "selection_criterion": "Highest test F1 score; ROC-AUC and recall break ties.",
        "selected_model": selected_name,
        "metrics": metrics,
        "saved_model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = train_all()
    print(f"Dataset shape: {tuple(result['dataset_shape'])}")
    print(f"Selected model: {result['selected_model']}")
    for name, values in result["metrics"].items():
        print(name, {metric: round(score, 6) for metric, score in values.items()})
    print(f"Saved pipeline: {result['saved_model']}")