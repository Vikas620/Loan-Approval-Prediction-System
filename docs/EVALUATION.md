# LoanLens Evaluation Notes

## Baseline versus enhanced architecture

The Phase 3 structured ML baseline is evaluated on its actual stratified holdout. Phase 8 adds evidence and graph risk indicators for decision-support review. These components do not have independent ground-truth labels in the public/demo dataset, so enhanced accuracy or fairness improvements are not claimed.

The endpoint `GET /api/evaluation/fusion` reports the selected model's actual holdout metrics and Brier score. It marks ML+Evidence, ML+Graph, and ML+Evidence+Graph ablations unavailable when no corresponding labels exist.

## Explainability comparison

Phase 9 returns actual SHAP factors when `TreeExplainer` can safely operate on the saved Gradient Boosting pipeline. LIME is optional and returns an explicit unavailable status unless a compatible training-matrix and categorical mapping contract is configured. SHAP/LIME overlap is only calculated when both methods actually produce factors.

## Fairness analysis

The fairness endpoint evaluates available dataset attributes and reports group sample sizes, selection rate, TPR, FPR, FNR, precision, recall, accuracy, and available disparities. Groups below the configured threshold are warned rather than used for unreliable comparisons. Metrics are dataset-specific and do not establish regulatory compliance.

## Leakage limitations

The baseline evaluation uses the Phase 3 saved holdout and pipeline. Fusion signals are not treated as supervised labels. Document and application reuse patterns should be separated by application when future labeled experiments are designed.
