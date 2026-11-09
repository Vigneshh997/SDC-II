"""Reports serialize measured model evidence without diagnostic prose invention."""

from backend.config import DISCLAIMER


def build_report(examination, results):
    return {
        "title": "X-Fusion research examination report",
        "examination": examination,
        "predictions": results["predictions"],
        "thresholds": results["thresholds"],
        "selected_conditions": results["selected_conditions"],
        "explanations": results["explanations"],
        "model": results["model"],
        "limitations": [
            "Probabilities are model scores, not established clinical risk.",
            "An empty selected_conditions list does not establish normality.",
            "Provenance declarations require human verification.",
            "Pretrained CXR-BERT used MIMIC data; pretraining overlap must be assessed for MIMIC evaluations.",
        ],
        "disclaimer": DISCLAIMER,
    }
