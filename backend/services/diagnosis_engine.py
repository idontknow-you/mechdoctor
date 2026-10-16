"""Rule-based diagnosis engine.

Each machine has a JSON knowledge base (backend/data/<machine>.json). A fault lists weighted
triggers; the engine checks which triggers match the inputs, scores every fault and returns
the most plausible ones. The confidence label reflects how much evidence matched; it is not
a claim of certainty.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MAX_FAULTS = 5          # faults returned per diagnosis
MIN_MATCHED_WEIGHT = 2  # weaker matches are dropped unless nothing stronger exists
NOTES_MAX_CHARS = 500

UNKNOWN = {None, "", "unknown"}
ALLOWED_OPS = {"gte", "lte", "in", "keywords"}


def _options(*pairs):
    return [{"value": value, "label": label} for value, label in pairs]


FIELD_DEFS = {
    "temperature_c": {
        "name": "temperature_c", "label": "Temperature", "type": "number",
        "unit": "°C", "min": -50, "max": 500,
    },
    "vibration": {
        "name": "vibration", "label": "Vibration", "type": "select",
        "options": _options(("low", "Low"), ("medium", "Medium"), ("high", "High")),
    },
    "noise": {
        "name": "noise", "label": "Unusual noise", "type": "select",
        "options": _options(
            ("none", "None"), ("humming", "Humming"), ("whining", "Whining"),
            ("grinding", "Grinding"), ("squealing", "Squealing"), ("knocking", "Knocking"),
            ("rattling", "Rattling"), ("crackling", "Crackling / gravel-like"),
            ("unusual", "Unusual (type not sure)"),
        ),
    },
    "leakage": {
        "name": "leakage", "label": "Leakage", "type": "select",
        "options": _options(("none", "None"), ("minor", "Minor"), ("major", "Major")),
    },
    "speed_rpm": {
        "name": "speed_rpm", "label": "Operating speed", "type": "number",
        "unit": "RPM", "min": 0, "max": 100000,
    },
    "performance": {
        "name": "performance", "label": "Performance drop", "type": "select",
        "options": _options(("none", "No drop"), ("slight", "Slight drop"), ("major", "Major drop")),
    },
    "notes": {
        "name": "notes", "label": "Symptoms / notes", "type": "text",
        "max_length": NOTES_MAX_CHARS,
    },
}

SELECT_FIELDS = ("vibration", "noise", "leakage", "performance")
NUMBER_FIELDS = ("temperature_c", "speed_rpm")


# ---------------------------------------------------------------- knowledge base

def _validate_machine(machine, source):
    for key in ("machine", "label", "fields", "faults", "general_safety", "general_inspection"):
        if key not in machine:
            raise ValueError(f"{source}: missing key '{key}'")
    for field in machine["fields"]:
        if field not in FIELD_DEFS:
            raise ValueError(f"{source}: unknown field '{field}' in 'fields'")
    for fault in machine["faults"]:
        for key in ("id", "name", "reasoning", "triggers", "inspection_steps",
                    "maintenance_actions", "safety"):
            if key not in fault:
                raise ValueError(f"{source}: fault '{fault.get('id', '?')}' missing '{key}'")
        for trig in fault["triggers"]:
            if trig.get("field") not in FIELD_DEFS:
                raise ValueError(f"{source}: fault '{fault['id']}' has unknown trigger field {trig.get('field')!r}")
            if trig.get("op") not in ALLOWED_OPS:
                raise ValueError(f"{source}: fault '{fault['id']}' has unknown op {trig.get('op')!r}")
            if not isinstance(trig.get("weight"), (int, float)) or trig["weight"] <= 0:
                raise ValueError(f"{source}: fault '{fault['id']}' has a bad weight")
            if not trig.get("symptom"):
                raise ValueError(f"{source}: fault '{fault['id']}' has a trigger without 'symptom'")


def _load_machines():
    machines = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            machine = json.load(handle)
        _validate_machine(machine, path.name)
        machines[machine["machine"]] = machine
    if not machines:
        raise RuntimeError(f"No machine JSON files found in {DATA_DIR}")
    return machines


MACHINES = _load_machines()


def get_machine(machine_id):
    return MACHINES[machine_id]


def list_machines():
    """Catalogue for the frontend: machines and the input fields each one uses."""
    return [
        {
            "id": machine["machine"],
            "label": machine["label"],
            "description": machine.get("description", ""),
            "typical_temp_range_c": machine.get("typical_temp_range_c"),
            "fields": [FIELD_DEFS[name] for name in machine["fields"]],
        }
        for machine in MACHINES.values()
    ]


# ---------------------------------------------------------------- input handling

def _clean_text(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return None if text in UNKNOWN else text


def _clean_number(value, field):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    definition = FIELD_DEFS[field]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{definition['label']} must be a number.")
    if number != number or not (definition["min"] <= number <= definition["max"]):
        raise ValueError(
            f"{definition['label']} must be between {definition['min']} and {definition['max']}."
        )
    return number


def normalise_inputs(data):
    """Validate and clean the request. Safe to call on already-normalised data.

    Missing or 'unknown' fields become None: they are neither evidence for nor against a fault.
    Raises ValueError with a user-readable message for bad input.
    """
    machine_id = _clean_text(data.get("machine"))
    if machine_id is None:
        raise ValueError("Please select a machine.")
    if machine_id not in MACHINES:
        raise ValueError(f"Unknown machine '{machine_id}'. Choose one of: {', '.join(MACHINES)}.")
    machine = MACHINES[machine_id]

    inputs = {"machine": machine_id, "machine_label": machine["label"]}

    for field in NUMBER_FIELDS:
        inputs[field] = _clean_number(data.get(field), field)

    for field in SELECT_FIELDS:
        value = _clean_text(data.get(field))
        allowed = [option["value"] for option in FIELD_DEFS[field]["options"]]
        if value is not None and value not in allowed:
            raise ValueError(
                f"Invalid {FIELD_DEFS[field]['label'].lower()} '{value}'. Allowed: {', '.join(allowed)}."
            )
        inputs[field] = value

    notes = data.get("notes")
    notes = str(notes).strip()[:NOTES_MAX_CHARS] if notes is not None else ""
    inputs["notes"] = notes or None

    return inputs


# ---------------------------------------------------------------- scoring

def _matches(trigger, inputs):
    value = inputs[trigger["field"]]
    op = trigger["op"]
    if op == "gte":
        return value >= trigger["value"]
    if op == "lte":
        return value <= trigger["value"]
    if op == "in":
        return value in trigger["value"]
    if op == "keywords":
        text = value.lower()
        return any(keyword in text for keyword in trigger["value"])
    return False


def _score_fault(fault, inputs):
    available_weight = 0.0
    matched_weight = 0.0
    matched_symptoms = []
    matched_count = 0

    for trigger in fault["triggers"]:
        if inputs.get(trigger["field"]) is None:
            continue  # field not provided: does not count for or against
        available_weight += trigger["weight"]
        if _matches(trigger, inputs):
            matched_weight += trigger["weight"]
            matched_count += 1
            matched_symptoms.append(trigger["symptom"])

    ratio = matched_weight / available_weight if available_weight else 0.0
    return {
        "fault": fault,
        "matched_weight": matched_weight,
        "ratio": ratio,
        "matched_count": matched_count,
        "available_count": sum(
            1 for t in fault["triggers"] if inputs.get(t["field"]) is not None
        ),
        "symptoms": matched_symptoms,
    }


def _confidence(matched_count, ratio):
    """Label how much evidence matched. Needs several matching indicators to reach High."""
    if matched_count >= 4 and ratio >= 0.7:
        return "High"
    if matched_count >= 2 and ratio >= 0.4:
        return "Medium"
    return "Low"


def _to_result(scored):
    fault = scored["fault"]
    coverage = f"{scored['matched_count']} of {scored['available_count']} relevant indicators matched."
    return {
        "id": fault["id"],
        "name": fault["name"],
        "confidence": _confidence(scored["matched_count"], scored["ratio"]),
        "score": round(scored["ratio"], 2),
        "supporting_symptoms": scored["symptoms"],
        "reasoning": f"{fault['reasoning']} {coverage}",
        "inspection_steps": fault["inspection_steps"],
        "maintenance_actions": fault["maintenance_actions"],
        "safety": fault["safety"],
    }


def _no_pattern(machine):
    return {
        "id": "no_clear_pattern",
        "name": "No clear fault pattern from the data given",
        "confidence": "Low",
        "score": 0.0,
        "supporting_symptoms": [],
        "reasoning": (
            "The readings entered do not strongly match any known fault for this machine. "
            "That does not mean the machine is healthy: add more symptoms or readings."
        ),
        "inspection_steps": machine["general_inspection"],
        "maintenance_actions": ["Re-check with more complete readings before planning any repair."],
        "safety": machine["general_safety"][:1],
    }


def diagnose(data):
    """Return the most plausible faults for the given inputs, strongest first."""
    inputs = normalise_inputs(data)
    machine = MACHINES[inputs["machine"]]

    scored = [_score_fault(fault, inputs) for fault in machine["faults"]]
    scored = [s for s in scored if s["matched_weight"] > 0]
    strong = [s for s in scored if s["matched_weight"] >= MIN_MATCHED_WEIGHT]
    pool = strong or scored
    pool.sort(key=lambda s: (s["matched_weight"], s["ratio"]), reverse=True)

    if not pool:
        return [_no_pattern(machine)]
    return [_to_result(s) for s in pool[:MAX_FAULTS]]
