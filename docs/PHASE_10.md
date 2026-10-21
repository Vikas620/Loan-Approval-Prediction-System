# Phase 10: Final Dashboard, Analytics, Reporting, Security, and Deployment

## Objective

Phase 10 packages LoanLens as a professional academic decision-support platform. It adds data-driven executive analytics, application filters, final PDF reports, audit and health APIs, responsive dashboard panels, upload hardening, and deployment scaffolding while preserving the Phase 1–9 model, evidence, graph, fusion, and explainability layers.

## Dashboard architecture

The existing single-page workspace now includes an executive dashboard, live assessment workflow, document intelligence, evidence summary, relationship graph, unified fusion profile, explainability, fairness, Responsible AI, history, and scenario explorer. Chart.js is used as the single chart library and receives values from `/api/dashboard` only.

## Chart data sources

- Risk distribution: persisted fusion `risk_level` values.
- Risk score histogram: persisted fusion overall-risk values.
- Component comparison: persisted ML/evidence/graph component scores.
- Decision-support status: persisted assessment/fusion statuses.
- Model evaluation: actual Phase 3 holdout metrics and Phase 8 Brier score.
- Fairness: actual dataset group metrics from the saved model and labeled holdout.
- Evidence and graph detail: existing Phase 6/7 records exposed through application APIs.

When no application records exist, KPI values and chart payloads are empty or `null`; the frontend displays an insufficient-data state rather than fabricated values.

## Analytics and application reporting

- `GET /api/dashboard` returns KPI cards, chart-ready aggregates, model evaluation, fairness, and application rows.
- `GET /api/applications` supports status, risk-level, and application-ID filters.
- `GET /api/audit` returns immutable fusion audit summaries.
- `GET /api/report/<application_id>` returns a safe application-linked PDF report.

The final report includes application ID, timestamp, model/fusion versions, component scores, overall risk, confidence, status, explanations, review flags, and limitations without raw documents.

## Security hardening

Uploads remain extension-, MIME-, path-, filename-, and size-validated. `MAX_CONTENT_LENGTH` is configured to the document limit, files are stored under local ignored instance storage, raw content is not returned in analytics/explanation payloads, and error responses avoid internal details. Debug mode remains environment-controlled and is disabled by default.

The deployment files contain no secrets. Use `.env.example` for local configuration and rotate `SECRET_KEY` before deployment. This project does not claim banking-grade security, regulatory compliance, or production lending authorization.

## System health

`GET /api/system/health` checks Flask, SQLite, saved ML model, Tesseract availability, NetworkX, fusion engine, and SHAP. The expected local state is `WARNING` when native Tesseract is not installed; this does not prevent text-native PDF processing.

## Deployment

Local development:

```powershell
.\venv\Scripts\python.exe app.py
```

Production-style WSGI:

```powershell
gunicorn --bind 0.0.0.0:8000 app:app
```

The included `Dockerfile` uses Gunicorn and `FLASK_DEBUG=0`. Uploaded documents and SQLite are local-demo storage and should be replaced with managed, encrypted storage before any real deployment.

## Testing

Run all tests:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Phase 10 tests cover dashboard data, application filters, audit, health, report generation, invalid upload handling, and preserved routes. The complete suite includes all Phase 1–9 tests.

## Limitations

The dataset and all demonstration documents are public, synthetic, or anonymized. Analytics are descriptive and not production banking metrics. OCR depends on native Tesseract for image/scanned PDFs. Fairness and calibration results are dataset-specific. Fusion, graph, evidence, explainability, and counterfactual outputs support human review and are not autonomous lending or fraud decisions.
