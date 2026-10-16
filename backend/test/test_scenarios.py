"""Tests for the two official scenarios, input sensitivity, validation, the API and AI fallback.

Run from the backend/ folder:   pytest -v
"""
import pytest

import config
from services import ai_service
from services.diagnosis_engine import MACHINES, diagnose, normalise_inputs


def ids(faults):
    return [fault["id"] for fault in faults]


# ------------------------------------------------------------------ official scenarios

def test_scenario_1_motor_overheating(scenario_1):
    faults = diagnose(scenario_1)
    # Expected: overload, poor ventilation, bearing friction.
    assert {"overload", "poor_ventilation", "bearing_friction"} <= set(ids(faults))
    # The two classic heat-without-vibration causes lead the list.
    assert set(ids(faults)[:2]) == {"overload", "poor_ventilation"}


def test_scenario_2_pump_vibration(scenario_2):
    faults = diagnose(scenario_2)
    # Expected: misalignment, imbalance, bearing wear, cavitation where applicable.
    assert {"misalignment", "imbalance", "bearing_wear", "cavitation"} <= set(ids(faults))
    # Vibration-driven mechanical causes lead; cavitation is a possibility, not the top pick.
    assert set(ids(faults)[:2]) == {"misalignment", "imbalance"}


def test_results_are_explained_and_actionable(scenario_1, scenario_2):
    for scenario in (scenario_1, scenario_2):
        for fault in diagnose(scenario):
            assert fault["reasoning"].strip()
            assert fault["supporting_symptoms"], f"{fault['id']} has no supporting symptoms"
            assert fault["inspection_steps"]
            assert fault["maintenance_actions"]
            assert fault["safety"]
            assert fault["confidence"] in {"Low", "Medium", "High"}


# ------------------------------------------------------------------ the judge changes one input

def test_changing_vibration_changes_the_diagnosis(scenario_1):
    low = ids(diagnose(scenario_1))
    high = ids(diagnose({**scenario_1, "vibration": "high"}))
    assert low != high
    assert "shaft_misalignment" in high and "shaft_misalignment" not in low
    # Bearing friction becomes a stronger suspect once vibration is high.
    assert high.index("bearing_friction") < low.index("bearing_friction")


def test_changing_temperature_changes_the_diagnosis(scenario_1):
    cooler = diagnose({**scenario_1, "temperature_c": 60})
    assert "bearing_friction" not in ids(cooler)
    assert "winding_insulation" not in ids(cooler)
    assert all(fault["confidence"] == "Low" for fault in cooler)


def test_changing_speed_strengthens_overload(scenario_1):
    normal = {f["id"]: f for f in diagnose(scenario_1)}
    slow = {f["id"]: f for f in diagnose({**scenario_1, "speed_rpm": 1200})}
    assert slow["overload"]["score"] > normal["overload"]["score"]


def test_noise_and_performance_surface_cavitation(scenario_2):
    faults = diagnose({**scenario_2, "noise": "crackling", "performance": "slight"})
    assert ids(faults)[0] == "cavitation"


def test_leakage_surfaces_seal_problem(scenario_2):
    faults = diagnose({**scenario_2, "leakage": "major"})
    assert ids(faults)[0] == "seal_leakage"


def test_notes_text_affects_the_diagnosis(scenario_1):
    base = ids(diagnose(scenario_1))
    dusty = ids(diagnose({**scenario_1, "notes": "Hot, and the fan cover is clogged with dust"}))
    assert dusty[0] == "poor_ventilation"
    assert base != dusty


# ------------------------------------------------------------------ uncertainty handling

def test_unknown_fields_are_not_evidence_either_way():
    faults = diagnose({"machine": "gearbox"})
    assert ids(faults) == ["no_clear_pattern"]
    assert faults[0]["confidence"] == "Low"
    assert faults[0]["inspection_steps"]


def test_limited_data_never_reports_high_confidence(scenario_1, scenario_2):
    for scenario in (scenario_1, scenario_2):
        assert all(fault["confidence"] != "High" for fault in diagnose(scenario))


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_every_machine_gives_a_usable_diagnosis(machine):
    faults = diagnose({
        "machine": machine, "temperature_c": 95, "vibration": "high", "noise": "grinding",
        "leakage": "minor", "performance": "major", "notes": "",
    })
    assert faults and len(faults) <= 5
    assert all(fault["inspection_steps"] and fault["safety"] for fault in faults)


# ------------------------------------------------------------------ input validation

@pytest.mark.parametrize("payload, message", [
    ({}, "select a machine"),
    ({"machine": "toaster"}, "Unknown machine"),
    ({"machine": "gearbox", "temperature_c": "abc"}, "must be a number"),
    ({"machine": "gearbox", "temperature_c": 9999}, "between"),
    ({"machine": "gearbox", "vibration": "huge"}, "Invalid vibration"),
    ({"machine": "gearbox", "noise": "screaming"}, "Invalid unusual noise"),
])
def test_bad_input_is_rejected_with_a_clear_message(payload, message):
    with pytest.raises(ValueError, match=message):
        normalise_inputs(payload)


def test_unknown_and_empty_values_become_missing():
    inputs = normalise_inputs({
        "machine": "Electric_Motor", "temperature_c": "", "vibration": "unknown",
        "noise": " ", "notes": "   ",
    })
    assert inputs["machine"] == "electric_motor"
    assert inputs["temperature_c"] is None
    assert inputs["vibration"] is None and inputs["noise"] is None
    assert inputs["notes"] is None


def test_normalise_inputs_is_idempotent(scenario_1):
    once = normalise_inputs(scenario_1)
    assert normalise_inputs(once) == once


# ------------------------------------------------------------------ API

def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["ai"] == {"gemini_configured": False, "groq_configured": False}


def test_machines_endpoint(client):
    body = client.get("/api/machines").get_json()
    assert {m["id"] for m in body["machines"]} == set(MACHINES)
    motor = next(m for m in body["machines"] if m["id"] == "electric_motor")
    assert {f["name"] for f in motor["fields"]} >= {"temperature_c", "vibration", "noise", "notes"}


def test_diagnose_returns_the_documented_contract(client, scenario_1):
    response = client.post("/api/diagnose", json=scenario_1)
    assert response.status_code == 200
    body = response.get_json()
    for key in ("machine", "faults", "safety_precautions", "ai_summary",
                "uncertainty", "ai_source", "disclaimer"):
        assert key in body
    assert body["ai_source"] == "template"          # no keys configured in tests
    assert "not a certified" in body["disclaimer"].lower()
    assert body["safety_precautions"]
    for fault in body["faults"]:
        assert "score" not in fault and "safety" not in fault


def test_diagnose_responds_when_an_input_changes(client, scenario_1):
    low = client.post("/api/diagnose", json=scenario_1).get_json()
    high = client.post("/api/diagnose", json={**scenario_1, "vibration": "high"}).get_json()
    assert [f["id"] for f in low["faults"]] != [f["id"] for f in high["faults"]]


def test_diagnose_rejects_bad_requests(client):
    assert client.post("/api/diagnose", data="not json").status_code == 400
    response = client.post("/api/diagnose", json={"machine": "toaster"})
    assert response.status_code == 400
    assert "Unknown machine" in response.get_json()["error"]


def test_unknown_route_returns_json_404(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert "error" in response.get_json()


# ------------------------------------------------------------------ AI layer and fallback

GOOD_AI = {"summary": "AI summary.", "uncertainty": "AI uncertainty.", "safety": ["AI precaution."]}


def _explain(scenario):
    inputs = normalise_inputs(scenario)
    return ai_service.explain(inputs, diagnose(inputs))


def test_template_is_used_when_no_keys(scenario_1):
    out = _explain(scenario_1)
    assert out["ai_source"] == "template"
    assert out["summary"] and out["uncertainty"] and out["safety"]


def test_gemini_result_is_used_when_it_works(monkeypatch, scenario_1):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(ai_service, "_gemini", lambda prompt: dict(GOOD_AI))
    out = _explain(scenario_1)
    assert out["ai_source"] == "gemini" and out["summary"] == "AI summary."


def test_groq_is_the_fallback_when_gemini_fails(monkeypatch, scenario_1):
    def broken(prompt):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake")
    monkeypatch.setattr(ai_service, "_gemini", broken)
    monkeypatch.setattr(ai_service, "_groq", lambda prompt: dict(GOOD_AI))
    assert _explain(scenario_1)["ai_source"] == "groq"


def test_template_is_the_last_resort_when_every_provider_fails(monkeypatch, scenario_1):
    def broken(prompt):
        raise RuntimeError("network down")

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake")
    monkeypatch.setattr(ai_service, "_gemini", broken)
    monkeypatch.setattr(ai_service, "_groq", broken)
    assert _explain(scenario_1)["ai_source"] == "template"


def test_unusable_ai_reply_falls_through_to_template(monkeypatch, scenario_1):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(ai_service, "_gemini", lambda prompt: {"unexpected": "shape"})
    assert _explain(scenario_1)["ai_source"] == "template"


def test_ai_safety_points_reach_the_report(monkeypatch, client, scenario_1):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(ai_service, "_gemini", lambda prompt: dict(GOOD_AI))
    body = client.post("/api/diagnose", json=scenario_1).get_json()
    assert body["ai_source"] == "gemini"
    assert "AI precaution." in body["safety_precautions"]


def test_json_in_code_fences_is_parsed():
    assert ai_service._parse_json('```json\n{"summary": "x"}\n```') == {"summary": "x"}


def test_prompt_contains_readings_and_candidates(scenario_1):
    inputs = normalise_inputs(scenario_1)
    prompt = ai_service.build_prompt(inputs, diagnose(inputs))
    assert "Electric Motor" in prompt and "95" in prompt and "Overload" in prompt
    assert "never claim a fault is certain" in prompt.lower()