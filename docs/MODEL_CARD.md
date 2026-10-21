# LoanLens Model Card

## Model purpose

The saved Gradient Boosting model estimates loan approval likelihood from structured applicant data for academic decision support.

## Intended use

Demonstration, research, coursework, and human-reviewed application analysis.

## Out-of-scope use

- Autonomous lending decisions
- Fraud or forgery confirmation
- Aadhaar, PAN, KYC, or identity verification
- Regulatory compliance claims
- Production banking certification

## Training data

The model was trained in Phase 3 using the public/demo loan approval dataset prepared in Phase 2. The dataset contains 614 records and the selected holdout metrics are recorded in `reports/ml/model_metrics.json`.

## Features

Applicant and loan fields include gender, marital status, dependents, education, self-employment, applicant/co-applicant income, loan amount, loan term, credit history, property area, total income, and loan-to-income ratio. `Loan_ID` is excluded as an identifier.

## Architecture

A scikit-learn `Pipeline` contains a `ColumnTransformer` and `GradientBoostingClassifier`. The preprocessing and model are persisted together in `model/saved_model/loan_approval_pipeline.joblib`.

## Evaluation methodology

Phase 3 uses a stratified 80/20 holdout with random state 42. Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC charts are generated from actual holdout results. Phase 8 additionally calculates a Brier score without changing model probabilities.

## Explainability

Phase 9 uses SHAP `TreeExplainer` when compatible with the saved tree estimator and maps transformed columns to original feature names. SHAP contributions are reported in raw model-output space, not as probabilities. LIME is optional and currently reports a limited/unavailable status when its dependency and preprocessing contract are absent.

## Fairness evaluation

Group-wise metrics are calculated for demographic attributes actually present in the dataset: Gender, Married, Education, and Self_Employed. Groups below the configured minimum sample size are marked insufficient. These dataset-specific metrics do not establish fairness, legal compliance, or absence of bias.

## Known limitations

The source dataset is public/demo data, probabilities are not presented as calibrated production probabilities, explanations are model- and pipeline-specific, counterfactuals are simulations rather than causal advice, and document/graph signals are separate review indicators.

## Human oversight

Evidence conflicts, low confidence, graph anomalies, and component disagreement can produce `REVIEW_REQUIRED`. Human review is required for consequential use.

## Privacy and security

Use synthetic or anonymized data only. Raw documents are not returned by normal explainability APIs. No passwords, full financial identifiers, or real identity documents are used by the Phase 9 features.
