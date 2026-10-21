"""Safe graph explanation formatting."""


def explain_analysis(analysis: dict) -> dict:
    return {
        "summary": analysis["disclaimer"],
        "status": analysis["status"],
        "signals": [
            {
                "signal": signal["signal_name"],
                "severity": signal["severity"],
                "why": signal["explanation"],
                "score_contribution": signal["score_contribution"],
                "evidence": signal["evidence"],
            }
            for signal in analysis["signals"]
        ],
    }
