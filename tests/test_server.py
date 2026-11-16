"""
Unit and integration tests for CyberPulse SOC Web Server API and Security Controls.
Verifies HTTP security headers, health endpoints, threat prediction, and input validation.
"""

import io
from pathlib import Path
import pytest
from flask import Flask

from dashboard.server import app
from src.config import config


@pytest.fixture
def client():
    """Provides a configured Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_page_and_security_headers(client):
    """Verifies that the index page returns 200 and enforces strict HTTP security headers."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"CYBERPULSE SOC" in response.data

    # Security Headers Verification
    headers = response.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in headers


def test_api_health_endpoint(client):
    """Verifies health & telemetry endpoint for load balancers and SOC monitoring."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "service" in data
    assert "model_loaded" in data
    assert "active_model" in data


def test_predict_sample_endpoint(client):
    """Verifies sample inference pipeline execution and risk tier telemetry."""
    response = client.get("/api/predict_sample")
    assert response.status_code == 200
    data = response.get_json()
    assert "summary" in data
    assert "download_token" in data
    assert "top_threats" in data

    summary = data["summary"]
    assert summary["total_records_analyzed"] > 0
    assert summary["benign_records"] + summary["attack_records"] == summary["total_records_analyzed"]
    assert "risk_breakdown" in summary


def test_download_predictions_workflow(client):
    """Verifies downloading scored threat classification CSV report."""
    # Step 1: Run sample prediction to generate download token
    pred_resp = client.get("/api/predict_sample")
    assert pred_resp.status_code == 200
    token = pred_resp.get_json()["download_token"]

    # Step 2: Download scored CSV report
    dl_resp = client.get(f"/api/download_predictions/{token}")
    assert dl_resp.status_code == 200
    assert dl_resp.content_type == "text/csv; charset=utf-8"
    assert b"Predicted_Threat_Label" in dl_resp.data
    assert b"Confidence_Score" in dl_resp.data
    assert b"Risk_Tier" in dl_resp.data


def test_download_predictions_invalid_token(client):
    """Verifies 404 response when requesting an expired or non-existent token."""
    response = client.get("/api/download_predictions/non-existent-token-12345")
    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data


def test_predict_upload_no_file(client):
    """Verifies 400 error when multipart upload is missing the file field."""
    response = client.post("/api/predict_upload", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert "No file parameter" in data.get("error", "")


def test_predict_upload_invalid_extension(client):
    """Verifies rejection of non-CSV files to prevent arbitrary file upload."""
    data = {
        "file": (io.BytesIO(b"malicious executable payload"), "payload.exe")
    }
    response = client.post("/api/predict_upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_data = response.get_json()
    assert "Only standard CSV" in res_data.get("error", "")


def test_predict_upload_empty_csv(client):
    """Verifies graceful handling of empty CSV files."""
    data = {
        "file": (io.BytesIO(b""), "empty.csv")
    }
    response = client.post("/api/predict_upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_data = response.get_json()
    assert "empty" in res_data.get("error", "").lower() or "no parseable" in res_data.get("error", "").lower()


def test_predict_upload_valid_csv(client):
    """Verifies successful upload, parsing, threat classification, and report export."""
    sample_file = config.sample_data_dir / "cicids2017_sample.csv"
    assert sample_file.exists()

    with open(sample_file, "rb") as f:
        file_bytes = f.read(20000)  # Read header and first few dozen records

    data = {
        "file": (io.BytesIO(file_bytes), "upload_test.csv")
    }
    response = client.post("/api/predict_upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_data = response.get_json()
    assert "summary" in res_data
    assert "top_threats" in res_data
    assert "download_token" in res_data
    assert res_data["summary"]["total_records_analyzed"] > 0


def test_404_error_handlers(client):
    """Verifies custom 404 responses for API and web routes."""
    # API 404
    api_resp = client.get("/api/non_existent_route")
    assert api_resp.status_code == 404
    assert api_resp.is_json
    assert api_resp.get_json()["status"] == 404

    # Web 404
    web_resp = client.get("/non_existent_page")
    assert web_resp.status_code == 404
    assert b"404 - Page Not Found" in web_resp.data

