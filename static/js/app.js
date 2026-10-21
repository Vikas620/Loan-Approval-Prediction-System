"use strict";

const form = document.querySelector("#application-form");
const formError = document.querySelector("#form-error");
const resultPanel = document.querySelector("#result-panel");
const resultEmpty = resultPanel.querySelector(".result-empty");
const resultContent = resultPanel.querySelector(".result-content");
const predictButton = document.querySelector("#predict-button");
const documentInput = document.querySelector("#document-input");
const uploadZone = document.querySelector("#upload-zone");
const documentError = document.querySelector("#document-error");
const documentResults = document.querySelector("#document-results");
let extractedPayload = {};
let documentPayload = [];
let validationPayload = {};
let applicationId = null;

function showError(message, fields = {}) {
	formError.textContent = message;
	formError.classList.remove("d-none");
	Object.keys(fields).forEach((field) => {
		const input = document.querySelector(`[name="${field}"]`);
		if (input) input.classList.add("is-invalid");
	});
}

function clearErrors() {
	formError.classList.add("d-none");
	document.querySelectorAll(".is-invalid").forEach((input) => input.classList.remove("is-invalid"));
}

function setLoading(isLoading) {
	predictButton.disabled = isLoading;
	predictButton.querySelector(".button-label").textContent = isLoading ? "Assessing..." : "Run assessment";
	predictButton.querySelector(".spinner-border").classList.toggle("d-none", !isLoading);
}

function showResult(result) {
	const approved = result.predicted_class === 1;
	const approval = result.approval_probability * 100;
	const risk = result.risk_score;
	resultEmpty.classList.add("d-none");
	resultContent.classList.remove("d-none");
	resultPanel.classList.toggle("result-approved", approved);
	resultPanel.classList.toggle("result-rejected", !approved);
	document.querySelector("#decision-label").textContent = result.decision;
	document.querySelector("#approval-value").textContent = `${approval.toFixed(1)}%`;
	document.querySelector("#risk-value").textContent = `${risk.toFixed(1)} / 100`;
	document.querySelector("#approval-bar").style.width = `${approval}%`;
	document.querySelector("#risk-bar").style.width = `${risk}%`;
	document.querySelector("#result-message").textContent = result.message;
	resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();
	clearErrors();
	if (!form.checkValidity()) {
		form.classList.add("was-validated");
		showError("Please complete all required fields before running the assessment.");
		return;
	}

	const payload = Object.fromEntries(new FormData(form).entries());
	setLoading(true);
	try {
		const response = await fetch("/api/assess", {
			method: "POST",
			headers: { "Content-Type": "application/json", Accept: "application/json" },
			body: JSON.stringify({ application_id: applicationId, applicant: payload, documents: documentPayload, extraction: extractedPayload, validation: validationPayload }),
		});
		const result = await response.json();
		if (!response.ok) {
			showError(result.error || "The application could not be assessed.", result.fields || {});
			return;
		}
		showResult(result);
		if (result.review_state) {
			document.querySelector("#result-message").textContent = `${result.message} Evidence state: ${result.review_state}.`;
		}
		await loadGraph(applicationId);
		await loadFusion(applicationId);
		await loadExplainability(applicationId);
		await loadHistory();
	} catch (error) {
		showError("The assessment service could not be reached. Please try again.");
	} finally {
		setLoading(false);
	}
});

function showDocumentError(message) {
	documentError.textContent = message;
	documentError.classList.remove("d-none");
}

function renderExtraction(result) {
	applicationId = result.application_id || applicationId;
	extractedPayload = result.extracted || {};
	documentPayload = result.documents || [];
	validationPayload = result.validation || {};
	const rows = Object.entries(extractedPayload);
	document.querySelector("#extraction-table").innerHTML = rows.length
		? rows.map(([field, value]) => `<tr><td class="fw-semibold">${field}</td><td><input class="form-control form-control-sm extracted-field" data-field="${field}" value="${String(value).replaceAll('"', '&quot;')}"></td><td><span class="review-chip ${value ? "review-found" : "review-missing"}">${value ? "Found" : "Review needed"}</span></td></tr>`).join("")
		: '<tr><td colspan="3" class="text-secondary small">No labeled fields were detected. Enter values manually below.</td></tr>';
	const warningText = [...(validationPayload.missing || []).map((field) => `Missing: ${field}`), ...(validationPayload.warnings || [])];
	document.querySelector("#validation-summary").textContent = warningText.length ? warningText.join(" · ") : "Extraction passed basic validation.";
	document.querySelector("#document-status").textContent = `${documentPayload.length} document${documentPayload.length === 1 ? "" : "s"} processed`;
	document.querySelector("#document-list").innerHTML = documentPayload.map((document) => `<div class="document-row"><span><strong>${document.document_type || "Supporting document"}</strong><span class="small text-secondary ms-2">${document.method || "processed"} · ${Math.round((document.confidence_score || 0) * 100)}% confidence</span></span><button type="button" class="btn btn-sm btn-outline-danger delete-document" data-document-id="${document.document_id}">Delete</button></div>`).join("");
	const summary = result.evidence_summary || {};
	document.querySelector("#evidence-score").textContent = `${summary.evidence_consistency_score ?? "-"}/100`;
	document.querySelector("#evidence-confidence").textContent = `${((summary.evidence_confidence || 0) * 100).toFixed(0)}%`;
	document.querySelector("#evidence-documents").textContent = summary.document_count ?? documentPayload.length;
	document.querySelector("#evidence-review").textContent = summary.review_required ? "REVIEW" : "READY";
	document.querySelector("#evidence-summary").classList.remove("d-none");
	document.querySelectorAll(".delete-document").forEach((button) => button.addEventListener("click", () => deleteUploadedDocument(button.dataset.documentId)));
	documentResults.classList.remove("d-none");
}

async function deleteUploadedDocument(documentId) {
	if (!applicationId) return;
	const response = await fetch(`/api/documents/${applicationId}/${documentId}`, { method: "DELETE" });
	if (!response.ok) return;
	documentPayload = documentPayload.filter((document) => document.document_id !== documentId);
	const row = document.querySelector(`[data-document-id="${documentId}"]`);
	if (row) row.closest(".document-row").remove();
}

async function processDocuments() {
	documentError.classList.add("d-none");
	if (!documentInput.files.length) {
		showDocumentError("Choose at least one PDF, JPG, or PNG document.");
		return;
	}
	const data = new FormData();
	Array.from(documentInput.files).forEach((file) => data.append("documents", file));
	data.append("document_type", document.querySelector("#document-type").value);
	if (applicationId) data.append("application_id", applicationId);
	const button = document.querySelector("#process-documents");
	button.disabled = true;
	button.textContent = "Processing...";
	try {
		const response = await fetch("/api/documents", { method: "POST", body: data });
		const result = await response.json();
		if (!response.ok) throw new Error(result.error || "Document processing failed.");
		renderExtraction(result);
	} catch (error) {
		showDocumentError(error.message);
	} finally {
		button.disabled = false;
		button.textContent = "Process documents";
	}
}

function applyExtraction() {
	document.querySelectorAll(".extracted-field").forEach((input) => {
		extractedPayload[input.dataset.field] = input.value.trim();
	});
	const mappings = { ApplicantIncome: "ApplicantIncome", LoanAmount: "LoanAmount", Education: "Education", Property_Area: "Property_Area", Credit_History: "Credit_History" };
	Object.entries(mappings).forEach(([source, target]) => {
		const value = extractedPayload[source];
		const input = document.querySelector(`[name="${target}"]`);
		if (input && value) input.value = value;
	});
	const employment = String(extractedPayload.Employment || "").toLowerCase();
	if (employment.includes("self") || employment === "yes") document.querySelector("[name=Self_Employed]").value = "Yes";
	if (employment === "no" || employment.includes("employed by")) document.querySelector("[name=Self_Employed]").value = "No";
	document.querySelector("#application-form").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadHistory() {
	const table = document.querySelector("#history-table");
	try {
		const response = await fetch("/api/history");
		const result = await response.json();
		table.innerHTML = result.assessments.length ? result.assessments.map((item) => `<tr><td class="fw-semibold">${item.applicant_name}</td><td class="small text-secondary">${new Date(item.created_at).toLocaleString()}</td><td><span class="history-decision ${item.decision === "Approved" ? "history-approved" : "history-rejected"}">${item.decision}</span></td><td>${(item.approval_probability * 100).toFixed(1)}%</td><td>${item.risk_score.toFixed(1)}</td><td><a class="btn btn-sm btn-outline-secondary" href="/reports/${item.id}.pdf">PDF</a></td></tr>`).join("") : '<tr><td colspan="6" class="text-secondary small">No assessments saved yet.</td></tr>';
	} catch (error) {
		table.innerHTML = '<tr><td colspan="6" class="text-secondary small">History is temporarily unavailable.</td></tr>';
	}
}

function renderChart(canvasId, emptyId, type, labels, values, label, colors) {
	const canvas = document.querySelector(`#${canvasId}`);
	const empty = document.querySelector(`#${emptyId}`);
	if (!labels.length || !window.Chart) {
		canvas.classList.add("d-none");
		empty.classList.remove("d-none");
		return;
	}
	new Chart(canvas, { type, data: { labels, datasets: [{ label, data: values, backgroundColor: colors || ["#1f8a8a", "#d99a4b", "#6875b5", "#dc6b56"], borderColor: "#ffffff", borderWidth: 2 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: type === "doughnut" ? {} : { y: { beginAtZero: true, ticks: { precision: 0 } } } } });
}

async function loadDashboard() {
	try {
		const response = await fetch("/api/dashboard");
		const data = await response.json();
		if (!response.ok) throw new Error(data.error || "Dashboard unavailable");
		document.querySelector("#dashboard-updated").textContent = `Last updated ${new Date(data.last_updated).toLocaleString()}`;
		const kpis = data.kpis;
		document.querySelector("#dashboard-kpis").innerHTML = [["Total applications", kpis.total_applications], ["Approved / low risk", kpis.approved_or_low_risk], ["Rejected / high risk", kpis.rejected_or_high_risk], ["Review required", kpis.review_required], ["Average risk score", kpis.average_risk_score == null ? "Insufficient data" : `${kpis.average_risk_score} / 100`]].map(([label, value]) => `<div class="kpi-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
		renderChart("risk-distribution-chart", "risk-distribution-empty", "doughnut", Object.keys(data.risk_distribution), Object.values(data.risk_distribution), "Applications");
		renderChart("risk-score-chart", "risk-score-empty", "bar", Object.keys(data.risk_score_histogram), Object.values(data.risk_score_histogram), "Applications", ["#1f8a8a"]);
		const componentLabels = ["ML", "Evidence", "Graph", "Overall"].map((name) => name.toLowerCase()).filter((name) => data.component_average[name] != null);
		const componentValues = componentLabels.map((name) => data.component_average[name]);
		renderChart("component-chart", "component-empty", "bar", componentLabels.map((name) => name[0].toUpperCase() + name.slice(1)), componentValues, "Average risk", ["#6875b5"]);
		renderChart("status-chart", "status-empty", "doughnut", Object.keys(data.status_distribution), Object.values(data.status_distribution), "Assessments");
		renderChart("evidence-chart", "evidence-empty", "doughnut", Object.keys(data.evidence_consistency), Object.values(data.evidence_consistency), "Evidence records");
		renderChart("graph-signal-chart", "graph-signal-empty", "bar", Object.keys(data.graph_signals), Object.values(data.graph_signals), "Signals", ["#dc6b56"]);
		const metrics = data.model_evaluation?.baseline_metrics || {};
		const metricPairs = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1", "f1_score"], ["ROC-AUC", "roc_auc"]].filter((pair) => metrics[pair[1]] != null);
		renderChart("model-chart", "model-empty", "bar", metricPairs.map((pair) => pair[0]), metricPairs.map((pair) => metrics[pair[1]]), "Selected model holdout", ["#1f8a8a"]);
		const fairnessGroups = data.fairness?.attributes?.Gender?.groups || {};
		const fairnessLabels = Object.keys(fairnessGroups).filter((group) => fairnessGroups[group].available && fairnessGroups[group].true_positive_rate != null);
		renderChart("fairness-chart", "fairness-empty", "bar", fairnessLabels, fairnessLabels.map((group) => fairnessGroups[group].true_positive_rate), "Gender group TPR", ["#6875b5"]);
	} catch (error) {
		document.querySelector("#dashboard-kpis").innerHTML = '<div class="dashboard-empty">Dashboard data unavailable.</div>';
	}
}

async function loadGraph(application) {
	if (!application) return;
	try {
		const response = await fetch(`/api/graph/${application}`);
		if (!response.ok) return;
		const graph = await response.json();
		const risk = graph.risk || {};
		document.querySelector("#graph-section").classList.remove("d-none");
		document.querySelector("#graph-score").textContent = `${risk.graph_risk_score || 0}/100`;
		document.querySelector("#graph-relationships").textContent = graph.edges.length;
		document.querySelector("#graph-reuse").textContent = risk.features.document_reuse_count;
		document.querySelector("#graph-high-risk").textContent = risk.signals.filter((signal) => ["HIGH", "CRITICAL"].includes(signal.severity)).length;
		document.querySelector("#graph-status").textContent = risk.status;
		document.querySelector("#graph-status").className = `review-chip ${risk.review_required ? "review-missing" : "review-found"}`;
		document.querySelector("#graph-signals").innerHTML = risk.signals.length ? risk.signals.map((signal) => `<div class="graph-signal"><strong>${signal.signal_name}</strong><span class="review-chip ${signal.severity === "HIGH" ? "review-missing" : "review-found"}">${signal.severity}</span><p>${signal.explanation}</p></div>`).join("") : "No relationship signals detected.";
		if (window.cytoscape) {
			cytoscape({ container: document.querySelector("#graph-canvas"), elements: [...graph.nodes.map((node) => ({ data: { id: node.node_id, label: node.metadata.label || node.node_type, type: node.node_type } })), ...graph.edges.map((edge, index) => ({ data: { id: `edge-${index}`, source: edge.source_node, target: edge.target_node, label: edge.relationship } }))], layout: { name: "breadthfirst", directed: true, padding: 24 }, style: [{ selector: "node", style: { "background-color": "#1f8a8a", label: "data(label)", color: "#102331", "font-size": 11, "text-valign": "bottom", "text-margin-y": 8 } }, { selector: 'node[type="Document"]', style: { "background-color": "#d99a4b" } }, { selector: 'node[type="Employer"]', style: { "background-color": "#6875b5" } }, { selector: 'node[type="IncomeEvidence"]', style: { "background-color": "#75a86a" } }, { selector: "edge", style: { width: 1.5, "line-color": "#aabdb8", "target-arrow-color": "#aabdb8", "target-arrow-shape": "triangle", label: "data(label)", "font-size": 8, color: "#53656b", "curve-style": "bezier" } }] });
		}
	} catch (error) {
		console.warn("Graph view unavailable", error);
	}
}

async function loadFusion(application) {
	if (!application) return;
	try {
		const response = await fetch(`/api/risk/${application}`);
		if (!response.ok) return;
		const profile = await response.json();
		const components = profile.components;
		document.querySelector("#fusion-section").classList.remove("d-none");
		document.querySelector("#fusion-status").textContent = profile.status;
		document.querySelector("#fusion-status").className = `review-chip ${profile.status === "REVIEW_REQUIRED" ? "review-missing" : "review-found"}`;
		document.querySelector("#fusion-overall").textContent = `${profile.overall_risk} / 100`;
		document.querySelector("#fusion-confidence").textContent = profile.confidence_level;
		document.querySelector("#fusion-level").textContent = profile.risk_level;
		["ml", "evidence", "graph"].forEach((name) => {
			document.querySelector(`#fusion-${name}`).textContent = `${components[name].score} / 100`;
			document.querySelector(`#fusion-${name}-confidence`).textContent = `${Math.round(components[name].confidence * 100)}% confidence`;
		});
		const signals = Object.values(components).flatMap((component) => component.signals || []);
		document.querySelector("#fusion-signals").innerHTML = profile.explanations.map((item) => `<p class="mb-1">${item}</p>`).join("") + (signals.length ? signals.map((signal) => `<div class="fusion-signal"><strong>${signal.signal || signal.signal_name || "Risk signal"}</strong><span>${signal.explanation || signal.why || ""}</span></div>`).join("") : "<p>No additional component signals were generated.</p>");
		document.querySelector("#fusion-flags").innerHTML = profile.review_flags.length ? `<strong>Review flags:</strong> ${profile.review_flags.join(" · ")}` : "No configured review flags.";
		document.querySelector("#scenario-income").value = document.querySelector("#ApplicantIncome").value;
		document.querySelector("#scenario-loan").value = document.querySelector("#LoanAmount").value;
	} catch (error) {
		console.warn("Unified risk view unavailable", error);
	}
}

async function runScenario() {
	if (!applicationId) return;
	const resultBox = document.querySelector("#scenario-result");
	resultBox.textContent = "Running simulated scenario...";
	try {
		const response = await fetch(`/api/risk/${applicationId}/scenario`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ApplicantIncome: document.querySelector("#scenario-income").value, LoanAmount: document.querySelector("#scenario-loan").value }) });
		const result = await response.json();
		if (!response.ok) throw new Error(result.error || "Scenario inputs are invalid.");
		resultBox.textContent = `Original risk ${result.original_risk}; simulated risk ${result.scenario_risk}; difference ${result.difference}.`;
	} catch (error) {
		resultBox.textContent = error.message;
	}
}

async function loadExplainability(application) {
	if (!application) return;
	try {
		const [explanationResponse, fairnessResponse] = await Promise.all([fetch(`/api/explain/${application}`), fetch("/api/fairness/report")]);
		if (!explanationResponse.ok) return;
		const explanation = await explanationResponse.json();
		document.querySelector("#explain-section").classList.remove("d-none");
		document.querySelector("#explain-method-status").textContent = explanation.shap.available ? "SHAP AVAILABLE" : "EXPLAINABILITY LIMITED";
		document.querySelector("#explain-method-status").className = `review-chip ${explanation.shap.available ? "review-found" : "review-missing"}`;
		document.querySelector("#shap-factors").innerHTML = explanation.shap.available ? explanation.shap.top_factors.map((factor) => `<div class="explain-row"><strong>${factor.feature}</strong><span>${factor.contribution} · ${factor.direction}</span><small>Input: ${factor.value ?? "not provided"}</small></div>`).join("") : `<p>${explanation.shap.reason}</p>`;
		document.querySelector("#lime-status").textContent = explanation.lime.available ? "LIME factors available." : `LIME limited: ${explanation.lime.reason}`;
		document.querySelector("#evidence-trace").innerHTML = explanation.evidence.length ? explanation.evidence.map((item) => `<div class="explain-row"><strong>${item.field_name}</strong><span>${item.normalized_value}</span><small>Document ${item.document_id} · Page ${item.page_number} · ${Math.round(item.confidence * 100)}% · ${item.extraction_method}</small></div>`).join("") : "No evidence records are available.";
		document.querySelector("#graph-trace").innerHTML = explanation.graph.signals.length ? explanation.graph.signals.map((signal) => `<div class="explain-row"><strong>${signal.signal_name}</strong><span>${signal.severity}</span><small>${signal.explanation}</small></div>`).join("") : "No graph risk signals are available.";
		const fairness = await fairnessResponse.json();
		document.querySelector("#fairness-summary").innerHTML = fairness.available ? Object.entries(fairness.attributes).map(([attribute, data]) => `<div class="fairness-row"><strong>${attribute}</strong><span>${Object.entries(data.groups).map(([group, metrics]) => `${group}: N=${metrics.n}${metrics.available ? `, TPR=${metrics.true_positive_rate?.toFixed(2) ?? "-"}` : ` — ${metrics.warning}`}`).join(" · ")}</span></div>`).join("") + (fairness.warnings.length ? `<p class="mt-2">${fairness.warnings.join(" ")}</p>` : "") : "Fairness report unavailable.";
	} catch (error) {
		console.warn("Explainability view unavailable", error);
	}
}

document.querySelector("#choose-documents").addEventListener("click", () => documentInput.click());
document.querySelector("#process-documents").addEventListener("click", processDocuments);
document.querySelector("#apply-extraction").addEventListener("click", applyExtraction);
document.querySelector("#refresh-history").addEventListener("click", loadHistory);
document.querySelector("#run-scenario").addEventListener("click", runScenario);
uploadZone.addEventListener("dragover", (event) => { event.preventDefault(); uploadZone.classList.add("upload-hover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("upload-hover"));
uploadZone.addEventListener("drop", (event) => { event.preventDefault(); uploadZone.classList.remove("upload-hover"); documentInput.files = event.dataTransfer.files; });
loadHistory();
loadDashboard();
