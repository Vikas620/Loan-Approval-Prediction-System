"""Reusable data preparation for the loan approval dataset.

This module only prepares data. It does not train or evaluate a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "raw" / "loan_approval_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "processed" / "loan_approval_processed.csv"
DEFAULT_PREPROCESSOR = PROJECT_ROOT / "dataset" / "processed" / "preprocessor.joblib"
TARGET_COLUMN = "Loan_Status"


def load_dataset(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load the source CSV and fail clearly when it is missing or malformed."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Place the real CSV in dataset/raw/ first."
        )
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError(f"Dataset at {source} is empty.")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}', found: {list(data.columns)}"
        )
    return data


def add_domain_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add features supported by the observed source columns."""
    result = data.copy()
    required_income_columns = {"ApplicantIncome", "CoapplicantIncome"}
    if required_income_columns.issubset(result.columns):
        result["TotalIncome"] = (
            pd.to_numeric(result["ApplicantIncome"], errors="coerce")
            + pd.to_numeric(result["CoapplicantIncome"], errors="coerce")
        )

    if {"TotalIncome", "LoanAmount"}.issubset(result.columns):
        loan_amount = pd.to_numeric(result["LoanAmount"], errors="coerce")
        result["LoanToIncomeRatio"] = loan_amount / result["TotalIncome"].replace(0, pd.NA)

    return result


def clean_for_export(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning, imputation, and target encoding."""
    result = add_domain_features(data)
    result = result.drop_duplicates().copy()
    result[TARGET_COLUMN] = result[TARGET_COLUMN].map({"Y": 1, "N": 0})
    result = result.dropna(subset=[TARGET_COLUMN])

    for column in result.select_dtypes(include="number").columns:
        result[column] = result[column].fillna(result[column].median())
    for column in result.select_dtypes(exclude="number").columns:
        mode = result[column].mode(dropna=True)
        if not mode.empty:
            result[column] = result[column].fillna(mode.iloc[0])
    return result


def build_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable transformer from the cleaned dataset's actual columns."""
    feature_data = data.drop(columns=[TARGET_COLUMN], errors="ignore")
    numeric_columns = feature_data.select_dtypes(include="number").columns.tolist()
    categorical_columns = feature_data.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def prepare_dataset(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    preprocessor_path: str | Path = DEFAULT_PREPROCESSOR,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Clean, export, fit, and persist the reusable preprocessing pipeline."""
    source = load_dataset(input_path)
    cleaned = clean_for_export(source)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)

    preprocessor = build_preprocessor(cleaned)
    preprocessor.fit(cleaned.drop(columns=[TARGET_COLUMN]))
    transformer_output = Path(preprocessor_path)
    transformer_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, transformer_output)
    return cleaned, preprocessor


if __name__ == "__main__":
    prepared, _ = prepare_dataset()
    print(f"Prepared dataset shape: {prepared.shape}")
    print(f"Saved processed dataset to: {DEFAULT_OUTPUT}")
    print(f"Saved reusable preprocessor to: {DEFAULT_PREPROCESSOR}")