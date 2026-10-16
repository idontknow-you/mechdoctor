"""Shared test setup: import path, no real AI calls, API client, official scenarios."""
import sys
from pathlib import Path

import pytest

# Make "import config", "import services..." work no matter where pytest is started from.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_ai_calls(monkeypatch):
    """Tests must never hit Gemini or Groq, even if keys exist in .env."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)


@pytest.fixture
def client():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def scenario_1():
    """Official Test Scenario 1: electric motor overheating."""
    return {
        "machine": "electric_motor",
        "temperature_c": 95,
        "vibration": "low",
        "noise": "none",
        "leakage": "none",
        "speed_rpm": 1450,
        "performance": "none",
        "notes": "Unusually hot during operation",
    }


@pytest.fixture
def scenario_2():
    """Official Test Scenario 2: pump vibration."""
    return {
        "machine": "water_pump",
        "temperature_c": 55,
        "vibration": "high",
        "noise": "unusual",
        "leakage": "none",
        "speed_rpm": 1450,
        "performance": "none",
        "notes": "Excessive vibration and unusual noise",
    }