"""Optional LIME adapter; never substitutes fabricated values."""

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # LIME is intentionally optional in this environment.
    LimeTabularExplainer = None


def explain_local(applicant: dict) -> dict:
    if LimeTabularExplainer is None:
        return {"available": False, "method": "LIME", "reason": "LIME is not installed; no local LIME values were generated."}
    return {"available": False, "method": "LIME", "reason": "LIME adapter requires a fitted training matrix and categorical mapping; it is disabled until that contract is configured."}
