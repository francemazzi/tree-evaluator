from fastapi.testclient import TestClient

from app.main import create_app


def test_co2_calculation_endpoint_returns_expected_fields() -> None:
    app = create_app()
    client = TestClient(app)

    payload = {
        "dbh_cm": 30.0,
        "height_m": 15.0,
        "wood_density_g_cm3": 0.6,
        "carbon_fraction": 0.47,
        "root_shoot_ratio": 0.24,
        "annual_biomass_increment_t": 0.03,
    }

    response = client.post("/api/v1/co2/calc", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Ensure all expected keys are present
    assert {
        "agb_t",
        "bgb_t",
        "total_biomass_t",
        "carbon_t",
        "co2_stock_t",
        "co2_annual_t",
    }.issubset(set(data.keys()))

    # Sanity checks on numeric ranges
    assert data["agb_t"] > 0
    assert data["bgb_t"] > 0
    assert data["total_biomass_t"] > data["agb_t"]
    assert data["carbon_t"] > 0
    assert data["co2_stock_t"] > 0
    assert data["co2_annual_t"] > 0


