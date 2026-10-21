# AI-Powered Loan Approval and Risk Assessment System

LoanLens is a Flask-based foundation for an AI-assisted loan assessment platform. Automated model outputs will support human review and will not represent a guaranteed lending decision.

## Phase 1: Run locally

1. Activate the existing virtual environment:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install the Phase 1 dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and replace the development secret before exposing the app beyond localhost.

4. Start the Flask application:

   ```powershell
   python app.py
   ```

5. Open <http://127.0.0.1:5000> in a browser. The health endpoint is available at <http://127.0.0.1:5000/health>.

## Current scope

Phase 1 provides the Flask application factory, environment-backed configuration, a responsive Bootstrap home page, a health endpoint, and project-level configuration files. Phase 2 adds a real loan approval dataset, reusable preprocessing, feature engineering, EDA summaries, and saved Matplotlib/Seaborn charts. Model training, database, authentication, reporting, and admin features will be added only in later confirmed phases.

## Phase 10: Final platform workflow

LoanLens is an academic, evidence-aware loan risk decision-support platform. It combines structured ML, synthetic-document extraction, evidence provenance, relationship analysis, multimodal risk fusion, SHAP explainability, counterfactual simulation, fairness reporting, data-driven analytics, PDF reports, audit records, and system health checks. It is not an autonomous lending system, identity-verification service, fraud-confirmation service, or regulatory-certified product.

### Run locally

```powershell
.\venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. Use synthetic or anonymized documents only.

### Run tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

### Production-style startup

```powershell
gunicorn --bind 0.0.0.0:8000 app:app
```

`Dockerfile`, `Procfile`, and `.env.example` contain deployment scaffolding only; never commit secrets. Native Tesseract is required for image/scanned-PDF OCR. Text-native PDF extraction works without it.

### Main API groups

- `/api/predict`, `/predict`: structured model assessment
- `/api/documents`, `/api/evidence`, `/api/consistency`: document/evidence intelligence
- `/api/graph`: relationship graph analysis
- `/api/risk`: multimodal fusion and scenarios
- `/api/explain`: SHAP, evidence, graph, counterfactual, and audit explanations
- `/api/dashboard`, `/api/applications`, `/api/audit`: analytics and audit
- `/api/report/<application_id>`: final PDF report
- `/api/system/health`: component health

See [docs/PHASE_10.md](docs/PHASE_10.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and the prior phase documents for architecture and limitations.

## Phase 2: Run EDA and preprocessing

From the project root, run:

```powershell
.\venv\Scripts\python.exe model\eda.py
```

The command reads `dataset/raw/loan_approval_train.csv`, writes the imputed and encoded-target dataset to `dataset/processed/loan_approval_processed.csv`, saves the reusable transformer to `dataset/processed/preprocessor.joblib`, and writes EDA charts plus `eda_summary.json` to `reports/eda/`. It does not train a model.

## Phase 5: Document intelligence and assessment history

Start the application:

```powershell
.\venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. The Phase 5 dashboard supports PDF, JPG, and PNG uploads, editable extracted fields, validation review, model assessment, local SQLite history, and downloadable PDF reports.

For text-native PDFs, extraction works with the Python dependencies alone. Image and scanned-PDF OCR also requires the native Tesseract executable installed and available on `PATH`; the application reports setup guidance when it is unavailable. Document processing is local demo intelligence only and does not claim Aadhaar, PAN, KYC, or identity verification.

The Phase 5 routes are:

- `POST /api/documents` for document extraction and validation
- `POST /api/assess` for reviewed applicant assessment and history persistence
- `GET /api/history` for recent local assessments
- `GET /reports/<assessment_id>.pdf` for a downloadable report
