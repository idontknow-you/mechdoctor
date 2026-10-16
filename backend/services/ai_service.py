"""AI explanation layer: Gemini -> Groq -> template fallback.

The rule engine decides WHICH faults are plausible; the LLM only explains them. If every
provider fails (no key, quota, network), a template explanation is returned so the app
never breaks. explain() never raises.
"""
import json
import logging

import config
from services.diagnosis_engine import get_machine

log = logging.getLogger(__name__)

GROQ_TIMEOUT_SECONDS = 20


def _compact_faults(faults):
    """Only what the model needs; keeps the prompt short and the answer focused."""
    return [
        {
            "name": fault["name"],
            "confidence": fault["confidence"],
            "supporting_symptoms": fault["supporting_symptoms"],
        }
        for fault in faults
    ]


def build_prompt(inputs, faults):
    machine = get_machine(inputs["machine"])
    readings = {
        key: value
        for key, value in inputs.items()
        if key not in ("machine", "machine_label") and value is not None
    }
    return f"""You are a mechanical maintenance assistant giving PRELIMINARY guidance.
Machine: {machine['label']}
Typical operating temperature range (reference only): {machine.get('typical_temp_range_c')} deg C
Readings entered: {json.dumps(readings)}
Candidate faults from a rule engine (ranked): {json.dumps(_compact_faults(faults))}

Using ONLY the readings and candidate faults above, reply with JSON in exactly this shape:
{{"summary": "3-4 sentences explaining the most plausible causes and why they fit the readings",
  "uncertainty": "1-2 sentences on what is unknown and what extra data would help",
  "safety": ["2-4 short, relevant safety precautions"]}}

Rules: never claim a fault is certain; say "possible" or "likely". The readings are illustrative,
not universal thresholds, so mention that real limits depend on the machine specification.
Do not invent readings or faults that are not listed."""


def _parse_json(text):
    """Parse the model reply, tolerating code fences or extra text around the JSON."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _clean(raw):
    """Make sure the model returned what we need; raise so the next provider is tried."""
    if not isinstance(raw, dict) or not str(raw.get("summary", "")).strip():
        raise ValueError("AI response had no summary")
    safety = raw.get("safety") or []
    if not isinstance(safety, list):
        safety = [safety]
    return {
        "summary": str(raw["summary"]).strip(),
        "uncertainty": str(raw.get("uncertainty", "")).strip(),
        "safety": [str(item).strip() for item in safety if str(item).strip()][:4],
    }


def _gemini(prompt):
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.3},
    )
    return _parse_json(response.text)


def _groq(prompt):
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return _parse_json(response.choices[0].message.content)


def _template(inputs, faults):
    names = ", ".join(fault["name"] for fault in faults[:3])
    top = faults[0] if faults else None
    detail = ""
    if top and top["supporting_symptoms"]:
        detail = f" The leading possibility ({top['name']}) fits: {'; '.join(top['supporting_symptoms'][:3])}."
    return {
        "summary": (
            f"Based on the readings entered, the most plausible causes are: {names}.{detail} "
            "This is a rule-based preliminary assessment."
        ),
        "uncertainty": (
            "Only limited readings were available, and real limits depend on the machine "
            "specification and operating conditions. Confirm by inspection before acting."
        ),
        "safety": ["Isolate and lock out the machine before any inspection."],
    }


def explain(inputs, faults):
    """Return {summary, uncertainty, safety, ai_source}. ai_source: gemini | groq | template."""
    prompt = build_prompt(inputs, faults)
    providers = (
        ("gemini", _gemini, config.GEMINI_API_KEY),
        ("groq", _groq, config.GROQ_API_KEY),
    )
    for name, call, key in providers:
        if not key:
            continue
        try:
            return {**_clean(call(prompt)), "ai_source": name}
        except Exception as exc:  # any provider failure -> try the next one
            log.warning("AI provider %s failed: %s", name, exc)
    return {**_template(inputs, faults), "ai_source": "template"}