"""End-to-end tests for the FastAPI backend.

Runs the real pipeline (all four agents, DEMO_MODE, no network) through
the HTTP endpoints exactly as the Streamlit frontend calls them.
"""

import importlib
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.database import DatabaseManager

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "data" / "sample_errors"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("api") / "api_test.db"
    import os

    old_db = os.environ.get("DUCKDB_PATH")
    old_demo = os.environ.get("DEMO_MODE")
    old_key = os.environ.pop("FIREWORKS_API_KEY", None)
    os.environ["DUCKDB_PATH"] = str(db_path)
    os.environ["DEMO_MODE"] = "true"
    DatabaseManager.reset_instance()

    import main

    importlib.reload(main)
    with TestClient(main.app) as test_client:
        yield test_client

    DatabaseManager.reset_instance()
    if old_db is not None:
        os.environ["DUCKDB_PATH"] = old_db
    else:
        os.environ.pop("DUCKDB_PATH", None)
    if old_demo is not None:
        os.environ["DEMO_MODE"] = old_demo
    if old_key is not None:
        os.environ["FIREWORKS_API_KEY"] = old_key


def _read_sample(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


def test_startup_seeds_historical_incidents(client):
    response = client.get("/incidents")
    assert response.status_code == 200
    incidents = response.json()
    assert len(incidents) == 5
    recurring = [
        incident
        for incident in incidents
        if any(
            loc["file_path"] == "payment_service/processor.py"
            and loc["line_number"] == 31
            for loc in incident["stack_trace_locations"]
        )
    ]
    assert len(recurring) == 3


def test_scenario_1_payment_full_context(client):
    deployed_at = (datetime.now() - timedelta(minutes=37)).isoformat()
    body = {
        "raw_error_text": _read_sample("payment_error.txt"),
        "environment": "production",
        "affected_users": 847,
        "customer_tier": "enterprise",
        "occurrences_last_4h": 12,
        "sla_breach_minutes": 40,
        "on_call_engineer": "Marcus Webb",
        "recent_deployment": {
            "deployer": "Elena Volkov",
            "commit_hash": "9f8c2ba",
            "commit_message": "Refactor gateway retry handling",
            "deployed_at": deployed_at,
        },
    }
    response = client.post("/incidents", json=body)
    assert response.status_code == 200, response.text
    output = response.json()
    decision = output["routing_decision"]
    assert decision["owning_team"] == "payments-team"
    assert decision["priority"] == "P1"
    assert decision["manager_digest_triggered"] is True
    assert decision["routing_confidence"] > 0.85
    assert decision["probable_cause"] is not None
    assert "9f8c2ba" in decision["probable_cause"]["commit_hash"]
    assert decision["classification"]["failure_type"] == "null_pointer"
    output_types = {t["output_type"] for t in output["ticket_contents"]}
    assert "full_ticket" in output_types
    assert "manager_digest" in output_types
    assert output["jira_ticket_id"] is None  # preview mode until approved

    # Approve and send (demo mode).
    incident_id = output["incident_id"]
    approve = client.post(f"/incidents/{incident_id}/approve")
    assert approve.status_code == 200, approve.text
    approved = approve.json()
    assert approved["jira_ticket_id"].startswith("PAY-")
    assert "atlassian.net/browse/" in approved["jira_url"]
    assert len(approved["notifications"]) >= 1

    detail = client.get(f"/incidents/{incident_id}/detail")
    assert detail.status_code == 200
    assert detail.json()["routing"] is not None


def test_scenario_2_auth_minimal_context(client):
    response = client.post(
        "/incidents", json={"raw_error_text": _read_sample("auth_error.txt")}
    )
    assert response.status_code == 200, response.text
    decision = response.json()["routing_decision"]
    assert decision["owning_team"] == "security-team"
    assert decision["priority"] == "P1"
    assert decision["classification"]["failure_type"] == "auth_failure"
    assert len(decision["missing_context"]) >= 5


def test_scenario_3_ml_staging(client):
    response = client.post(
        "/incidents",
        json={
            "raw_error_text": _read_sample("ml_timeout_error.txt"),
            "environment": "staging",
            "affected_users": 85,
        },
    )
    assert response.status_code == 200, response.text
    decision = response.json()["routing_decision"]
    assert decision["owning_team"] == "ml-platform-team"
    assert decision["priority"] == "P2"
    assert decision["manager_digest_triggered"] is False
    assert decision["classification"]["failure_type"] == "timeout"


def test_scenario_4_video_error_routes(client):
    response = client.post(
        "/incidents", json={"raw_error_text": _read_sample("video_error.txt")}
    )
    assert response.status_code == 200, response.text
    decision = response.json()["routing_decision"]
    assert decision["owning_team"] == "infra-team"
    assert decision["classification"]["failure_type"] == "connection_refused"


def test_audit_detects_recurring_location(client):
    response = client.post("/audit")
    assert response.status_code == 200, response.text
    audit = response.json()
    assert audit["incidents_analyzed"] >= 5
    assert audit["patterns_found"] >= 1
    recurring = [
        flag for flag in audit["flags"] if flag["pattern_type"] == "recurring_location"
    ]
    assert len(recurring) >= 1
    payment_flags = [
        flag for flag in recurring if flag["affected_service"] == "payment-service"
    ]
    assert len(payment_flags) == 1
    flag = payment_flags[0]
    assert flag["confidence"] > 0.70
    assert len(flag["contributing_incident_ids"]) >= 3
    assert flag["plm_ticket_id"] is not None
    assert flag["plm_ticket_id"].startswith("PLM-")

    latest = client.get("/audit/latest")
    assert latest.status_code == 200
    assert latest.json()["audit_id"] == audit["audit_id"]


def test_graph_nodes_after_audit(client):
    response = client.get("/graph/nodes")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["nodes"]) > 6
    assert len(payload["edges"]) > 0
    hot = [n for n in payload["nodes"] if n["incident_count"] >= 2]
    assert any("processor" in n["node_id"] for n in hot)


def test_resolve_and_stats(client):
    incidents = client.get("/incidents").json()
    target = incidents[0]["incident_id"]
    response = client.post(f"/incidents/{target}/resolve")
    assert response.status_code == 200
    refreshed = client.get("/incidents").json()
    resolved = next(i for i in refreshed if i["incident_id"] == target)
    assert resolved["resolved"] is True

    stats = client.get("/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert "fireworks_calls" in body
    assert body["table_counts"]["gold_incident_intelligence"] >= 5
