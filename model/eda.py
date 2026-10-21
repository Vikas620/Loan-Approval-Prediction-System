"""Run factual exploratory analysis and preprocessing for the loan dataset.

Usage from the project root:
    python model/eda.py

The script saves charts and a machine-readable summary under reports/eda/.
No model training is performed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import DEFAULT_INPUT, prepare_dataset, load_dataset, add_domain_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
TARGET_COLUMN = "Loan_Status"


def save_plot(plot_name: str) -> None:
    """Save a consistently styled chart."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{plot_name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def categorical_counts(data: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        column: data[column].fillna("<MISSING>").astype(str).value_counts().to_dict()
        for column in data.select_dtypes(exclude="number").columns
    }


def run_eda(input_path: str | Path = DEFAULT_INPUT) -> dict:
    """Create analysis outputs from the real source dataset."""
    source = load_dataset(input_path)
    data = add_domain_features(source)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    missing_values = data.isna().sum().sort_values(ascending=False)
    summary = {
        "source": str(Path(input_path).relative_to(PROJECT_ROOT)),
        "shape": list(data.shape),
        "columns": data.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": missing_values[missing_values > 0].astype(int).to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": data.select_dtypes(exclude="number").columns.tolist(),
        "categorical_value_counts": categorical_counts(data),
        "target_distribution": data[TARGET_COLUMN].fillna("<MISSING>").value_counts().to_dict(),
        "numeric_summary": json.loads(data[numeric_columns].describe().round(4).to_json()),
    }

    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x=TARGET_COLUMN, order=data[TARGET_COLUMN].dropna().value_counts().index)
    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan status")
    plt.ylabel("Applications")
    save_plot("loan_approval_distribution")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="ApplicantIncome", kde=True)
    plt.title("Applicant Income Distribution")
    save_plot("applicant_income_distribution")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="LoanAmount", kde=True)
    plt.title("Loan Amount Distribution")
    save_plot("loan_amount_distribution")

    chart_specs = [
        ("Credit_History", "Loan_Status", "Credit History vs Loan Status", "credit_history_vs_loan_status"),
        ("Education", "Loan_Status", "Education vs Loan Status", "education_vs_loan_status"),
        ("Self_Employed", "Loan_Status", "Employment Type vs Loan Status", "employment_type_vs_loan_status"),
        ("Property_Area", "Loan_Status", "Property Area vs Loan Status", "property_area_vs_loan_status"),
    ]
    for x_column, y_column, title, filename in chart_specs:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=x_column, hue=y_column)
        plt.title(title)
        plt.xticks(rotation=20)
        save_plot(filename)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="ApplicantIncome", y="LoanAmount", hue="Loan_Status", alpha=.75)
    plt.title("Applicant Income vs Loan Amount")
    save_plot("applicant_income_vs_loan_amount")

    # Inspect numerical outliers using the IQR rule; values are reported, not silently removed.
    outliers = {}
    for column in numeric_columns:
        values = data[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        outliers[column] = {"count": count, "lower_bound": float(q1 - 1.5 * iqr), "upper_bound": float(q3 + 1.5 * iqr)}
    summary["outlier_inspection_iqr"] = outliers

    with (REPORT_DIR / "eda_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, default=str)

    prepare_dataset(input_path=input_path)
    return summary


if __name__ == "__main__":
    results = run_eda()
    print(f"Source dataset shape: {tuple(results['shape'])}")
    print(f"Columns: {', '.join(results['columns'])}")
    print(f"Duplicate rows: {results['duplicate_rows']}")
    print(f"Missing values: {results['missing_values']}")
    print(f"Target distribution: {results['target_distribution']}")
    print(f"EDA outputs saved to: {REPORT_DIR}")