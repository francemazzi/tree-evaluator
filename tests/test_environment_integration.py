from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import create_app


def _base_payload(use_height: bool = True, use_log: bool = False):
    payload = {
        "tree": {
            "diameter_cm": 35.0,
            **({"height_m": 15.0} if use_height else {}),
            "carbon_fraction": 0.47,
        },
        "site": {"site_id": "IT-LAZ-COAST-01", "lat": 41.90, "lon": 12.50},
        "method": {"use_log_form": use_log, "rsr_override": None, "bef_mode": "none"},
        "feedback": {"observed_biomass_kg": None, "notes": None},
        "meta": {
            "request_id": "00000000-0000-0000-0000-000000000001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "api",
        },
    }
    return payload


def test_environment_estimates_with_height() -> None:
    app = create_app()
    client = TestClient(app)
    payload = _base_payload(use_height=True, use_log=False)

    r = client.post("/api/v1/environment/estimates", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["results"]["volume_dm3"] > 0
    assert data["results"]["biomass_kg"] > 0
    assert data["results"]["carbon_stock_kg"] > 0
    assert data["results"]["rsr_used"] == 0.25
    assert data["results"]["bef"] is None


def test_environment_estimates_without_height_uses_d_only_formula() -> None:
    app = create_app()
    client = TestClient(app)
    payload = _base_payload(use_height=False, use_log=False)

    r = client.post("/api/v1/environment/estimates", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "Height missing" in data["results"]["confidence"]["notes"]


def test_environment_estimates_log_form() -> None:
    app = create_app()
    client = TestClient(app)
    payload = _base_payload(use_height=True, use_log=True)

    r = client.post("/api/v1/environment/estimates", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["results"]["biomass_kg"] > 0


def test_environment_estimates_with_feedback_rd() -> None:
    app = create_app()
    client = TestClient(app)
    payload = _base_payload(use_height=True, use_log=False)
    payload["feedback"]["observed_biomass_kg"] = 1000.0

    r = client.post("/api/v1/environment/estimates", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["results"]["confidence"]["relative_error_rd"] is not None


def test_environment_estimates_validation_errors() -> None:
    app = create_app()
    client = TestClient(app)
    payload = _base_payload(use_height=True, use_log=False)
    payload["tree"]["diameter_cm"] = 0

    r = client.post("/api/v1/environment/estimates", json=payload)
    assert r.status_code == 422
    data = r.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"


