"""API routes: machine catalogue and diagnosis."""
import logging

from flask import Blueprint, jsonify, request

from services.ai_service import explain
from services.diagnosis_engine import diagnose, list_machines, normalise_inputs
from services.report_builder import build_report

bp = Blueprint("diagnose", __name__)
log = logging.getLogger(__name__)


@bp.get("/api/machines")
def machines():
    """Machine list with the input fields (and options) each one needs."""
    return jsonify({"machines": list_machines()})


@bp.post("/api/diagnose")
def run_diagnosis():
    """Validate inputs, run the rule engine, add the AI explanation, return the report."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    # Bad user input -> 400 with a message the UI can show directly.
    try:
        inputs = normalise_inputs(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Anything else is our fault -> 500, details go to the server log only.
    try:
        faults = diagnose(inputs)       # rule engine: always input-dependent
        ai = explain(inputs, faults)    # Gemini -> Groq -> template, never raises
        return jsonify(build_report(inputs, faults, ai))
    except Exception:
        log.exception("Diagnosis failed")
        return jsonify({"error": "Diagnosis failed unexpectedly. Please try again."}), 500