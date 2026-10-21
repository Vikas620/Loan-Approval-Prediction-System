# LoanLens Architecture

```mermaid
flowchart TB
    Browser[Responsive dashboard]
    Browser --> Flask[Flask routes]
    Flask --> Upload[Document intelligence]
    Upload --> Processor[PyMuPDF / Tesseract abstraction]
    Processor --> Evidence[Evidence and provenance layer]
    Evidence --> SQLite[(SQLite)]
    Flask --> Predictor[Existing Phase 3 ML pipeline]
    Predictor --> Assessment[Assessment history]
    Assessment --> Report[ReportLab PDF]
    Evidence --> Consistency[Consistency and integrity heuristics]
    Consistency --> Review[APPROVED / REJECTED / REVIEW_REQUIRED]
    Evidence --> Graph[Graph-ready relationships]
    Graph --> SQLite
    SQLite --> NetworkX[Phase 7 NetworkX graph engine]
    NetworkX --> GraphRisk[Relationship risk indicators]
    GraphRisk --> GraphDashboard[Graph dashboard and review state]
    Predictor --> Fusion[Phase 8 multimodal fusion]
    Consistency --> Fusion
    GraphRisk --> Fusion
    Fusion --> Unified[Unified risk profile and abstention]
    Unified --> Audit[(Fusion audit records)]
    Unified --> Explain[Phase 9 SHAP, fairness, and counterfactual services]
    Explain --> Responsible[Responsible AI dashboard]
    User[User] --> Dashboard[Final analytics dashboard]
    Dashboard --> Flask[Flask API]
    Flask --> Analytics[Data-driven analytics]
    Analytics --> Reports[PDF reports and audit trail]
    Flask --> Health[System health and deployment checks]
```

The model decision and evidence review state are intentionally separate. The prediction is produced by the existing trained pipeline; document quality and contradictions can request human review without pretending that evidence heuristics are a lending probability.

Phase 7 adds graph nodes and relationship analysis above the evidence layer. Graph reuse and shared-entity patterns are explainable risk indicators, not fraud conclusions.

Phase 8 combines ML, evidence, and graph indicators using configurable engineering weights. The fusion layer preserves component provenance and can return `REVIEW_REQUIRED` when confidence is low or signals disagree.

Phase 9 adds model-compatible SHAP explanations, optional LIME status, validated counterfactual simulation, dataset-specific fairness metrics, model-card metadata, and explanation audit records.

Phase 10 adds the executive analytics layer, application filtering, final reporting, audit/health APIs, upload hardening, responsive charts, and Gunicorn/Docker deployment scaffolding.
