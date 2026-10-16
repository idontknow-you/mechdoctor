"""Merge rule-engine output and AI text into the final API response."""
from services.diagnosis_engine import get_machine

DISCLAIMER = (
    "Preliminary guidance only, not a certified engineering diagnosis. "
    "Confirm by inspection with a qualified engineer before taking action."
)

MAX_SAFETY_ITEMS = 8


def _dedupe(items, limit):
    seen, result = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result[:limit]


def build_report(inputs, faults, ai):
    machine = get_machine(inputs["machine"])

    # Machine-wide precautions first, then the AI's tailored ones, then per-fault ones.
    safety = _dedupe(
        list(machine["general_safety"])
        + list(ai.get("safety", []))
        + [item for fault in faults[:3] for item in fault.get("safety", [])],
        MAX_SAFETY_ITEMS,
    )

    public_faults = [
        {key: value for key, value in fault.items() if key not in ("score", "safety")}
        for fault in faults
    ]

    return {
        "machine": inputs["machine"],
        "machine_label": machine["label"],
        "inputs": {
            key: value
            for key, value in inputs.items()
            if key not in ("machine", "machine_label") and value is not None
        },
        "faults": public_faults,
        "safety_precautions": safety,
        "ai_summary": ai["summary"],
        "uncertainty": ai.get("uncertainty", ""),
        "ai_source": ai["ai_source"],
        "disclaimer": DISCLAIMER,
    }