"""Traceable explanations for unified risk profiles."""


def explain_profile(profile: dict) -> dict:
    signals = []
    for component_name, component in profile.get("components", {}).items():
        for signal in component.get("signals", []):
            signals.append({"component": component_name, **signal})
    return {"title": "Why this result?", "explanations": profile.get("explanations", []), "primary_signals": signals, "review_flags": profile.get("review_flags", []), "traceability": "Each component links to its Phase 3 model output, Phase 6 evidence records, or Phase 7 graph signals."}
