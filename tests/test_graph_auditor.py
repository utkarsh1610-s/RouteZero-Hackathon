"""Tests for core.graph.CodeGraphBuilder and agents.auditor.ArchitecturalAuditor.

All tests run against an isolated temp DuckDB file with DEMO_MODE enabled
and FireworksClient.complete_json monkeypatched, so nothing touches the
network. Accidental network calls fail the test immediately.
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.auditor import ArchitecturalAuditor
from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.graph import CodeGraphBuilder
from core.schemas import (
    Environment,
    FailureType,
    IncidentIntelligenceRecord,
    PatternType,
    Priority,
)
from integrations.github_client import GitHubClient
from integrations.jira_client import JiraClient

HISTORICAL_FOLDER = str(REPO_ROOT / "data" / "historical_incidents")
PROCESS_PAYMENT_NODE = "payment_service/processor.py::PaymentProcessor.process_payment"
RECURRING_INCIDENT_IDS = {"INC-2026-0134", "INC-2026-0149", "INC-2026-0162"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real HTTP attempt fails the test immediately."""

    def _boom(*args, **kwargs):
        raise AssertionError("A real network call was attempted during tests.")

    monkeypatch.setattr("core.fireworks_client.requests.post", _boom)
    monkeypatch.setattr("integrations.jira_client.requests.post", _boom)
    monkeypatch.setattr("integrations.github_client.requests.get", _boom)


@pytest.fixture(autouse=True)
def fireworks_stub(monkeypatch):
    """Fireworks never hits the network: default to the empty-dict fallback."""
    monkeypatch.setattr(
        FireworksClient, "complete_json", lambda self, prompt, **kwargs: {}
    )


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A DatabaseManager on a temp DuckDB file, DEMO_MODE on, singleton reset."""
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "routezero_test.db"))
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    DatabaseManager.reset_instance()
    manager = DatabaseManager()
    yield manager
    DatabaseManager.reset_instance()


@pytest.fixture()
def org_config():
    return json.loads((REPO_ROOT / "data" / "org_config.json").read_text(encoding="utf-8"))


def make_auditor(db, org_config, graph=None):
    return ArchitecturalAuditor(
        db=db,
        org_config=org_config,
        fireworks=FireworksClient(),
        jira=JiraClient(),
        github=GitHubClient(),
        graph=graph or CodeGraphBuilder(db),
    )


def make_incident(
    incident_id,
    service="payment-service",
    failure_type=FailureType.NULL_POINTER,
    timestamp=None,
    locations=(),
):
    return IncidentIntelligenceRecord(
        incident_id=incident_id,
        jira_ticket_ref=None,
        timestamp=timestamp or datetime.now(),
        service=service,
        failure_type=failure_type,
        environment=Environment.PRODUCTION,
        is_critical_path=True,
        affected_users=100,
        priority=Priority.P2,
        stack_trace_locations=list(locations),
        probable_cause=None,
        linked_deployment=None,
        routing_confidence=0.9,
        owning_team="payments-team",
        resolved=False,
        resolution_time_minutes=None,
        agent3_summary=f"Synthetic incident {incident_id} for tests.",
    )


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


def test_graph_builds_from_demo_repo(db):
    builder = CodeGraphBuilder(db)
    builder.build_or_load()

    rows = db.get_graph_nodes()
    modules = [row for row in rows if row["node_type"] == "module"]
    defs = [row for row in rows if row["node_type"] in ("function", "class")]
    assert len(modules) >= 6
    assert len(defs) > 0

    node_ids = {row["node_id"] for row in rows}
    assert PROCESS_PAYMENT_NODE in node_ids
    assert "payment_service/processor.py" in node_ids
    assert "shared/database_pool.py::ConnectionPool" in node_ids

    edges = db.get_graph_edges()
    relationship_types = {edge["relationship_type"] for edge in edges}
    assert "contains" in relationship_types
    assert "imports" in relationship_types
    assert "calls" in relationship_types


def test_get_node_at_resolves_the_bug_line(db):
    builder = CodeGraphBuilder(db)
    builder.build_or_load()

    node = builder.get_node_at("payment_service/processor.py", 31)
    assert node is not None
    assert node.node_id == PROCESS_PAYMENT_NODE
    assert node.name == "process_payment"
    assert node.node_type.value == "function"
    assert node.start_line <= 31 <= node.end_line

    # Longer paths that merely end with the suffix resolve to the same node,
    # backslashes included.
    same = builder.get_node_at(r"D:\anywhere\demo_repo\payment_service\processor.py", 31)
    assert same is not None and same.node_id == PROCESS_PAYMENT_NODE

    # A line outside every definition falls back to the module node.
    module = builder.get_node_at("payment_service/processor.py", 1)
    assert module is not None
    assert module.node_id == "payment_service/processor.py"
    assert module.node_type.value == "module"

    # An unknown file returns None.
    assert builder.get_node_at("nonexistent/never.py", 10) is None


def test_owner_attribution_fallback_mapping(db, monkeypatch):
    # demo_repo is uncommitted, so git blame yields nothing; force that
    # explicitly so the deterministic fallback is what is under test.
    monkeypatch.setattr(CodeGraphBuilder, "_git_owner", lambda self, rel_path: None)
    builder = CodeGraphBuilder(db)
    builder.build_or_load()

    owners = {row["node_id"]: row["owner"] for row in db.get_graph_nodes()}
    assert owners["payment_service/processor.py"] == "marcus.webb"
    assert owners["auth_service/token_validator.py"] == "aisha.patel"
    assert owners["ml_service/recommender.py"] == "carlos.mendez"
    assert owners["shared/database_pool.py"] == "ben.carter"
    assert owners[PROCESS_PAYMENT_NODE] == "marcus.webb"


def test_second_build_or_load_loads_from_db_without_parsing(db, monkeypatch):
    first = CodeGraphBuilder(db)
    first.build_or_load()
    stored_count = len(db.get_graph_nodes())
    assert stored_count > 0

    # A fresh builder must load from DuckDB; any re-parse (insert) fails loudly.
    def _no_insert(*args, **kwargs):
        raise AssertionError("Graph was re-parsed: insert called on second build_or_load")

    monkeypatch.setattr(db, "insert_graph_nodes", _no_insert)
    monkeypatch.setattr(db, "insert_graph_edges", _no_insert)

    second = CodeGraphBuilder(db)
    second.build_or_load()

    assert len(db.get_graph_nodes()) == stored_count
    node = second.get_node_at("payment_service/processor.py", 31)
    assert node is not None and node.node_id == PROCESS_PAYMENT_NODE


def test_subgraph_two_hops_reaches_shared_connection_pool(db):
    builder = CodeGraphBuilder(db)
    builder.build_or_load()

    nodes, edges = builder.get_subgraph(PROCESS_PAYMENT_NODE, hops=2)
    node_ids = {node.node_id for node in nodes}
    assert PROCESS_PAYMENT_NODE in node_ids
    assert "payment_service/processor.py" in node_ids  # 1 hop: contains
    assert "shared/database_pool.py" in node_ids  # 2 hops: module imports
    # 3+ hops away must be excluded.
    assert "auth_service/token_validator.py" not in node_ids

    # Every returned edge stays inside the returned node set.
    for edge in edges:
        assert edge.source_node_id in node_ids
        assert edge.target_node_id in node_ids
    relationship_types = {edge.relationship_type for edge in edges}
    assert {"contains", "imports"} <= relationship_types

    # From the processor module, the ConnectionPool class itself is 2 hops
    # (imports edge to shared/database_pool.py, contains edge to the class).
    module_nodes, _ = builder.get_subgraph("payment_service/processor.py", hops=2)
    module_node_ids = {node.node_id for node in module_nodes}
    assert "shared/database_pool.py::ConnectionPool" in module_node_ids

    # Unknown node ids return an empty subgraph instead of raising.
    assert builder.get_subgraph("does/not/exist.py::nope") == ([], [])


def test_map_incident_locations_counts_distinct_incidents(db):
    assert db.seed_historical_incidents(HISTORICAL_FOLDER) == 5
    builder = CodeGraphBuilder(db)
    builder.build_or_load()

    counts = builder.map_incident_locations(db.get_incidents_since(7))
    assert counts[PROCESS_PAYMENT_NODE] == 3
    assert counts["auth_service/token_validator.py::TokenValidator.validate_token"] == 1
    # Re-mapping is idempotent: distinct incident counts do not inflate.
    counts_again = builder.map_incident_locations(db.get_incidents_since(7))
    assert counts_again[PROCESS_PAYMENT_NODE] == 3


# ---------------------------------------------------------------------------
# Auditor: recurring location pattern (the core demo scenario)
# ---------------------------------------------------------------------------


def test_audit_detects_recurring_location_with_fallback_assessment(db, org_config):
    assert db.seed_historical_incidents(HISTORICAL_FOLDER) == 5
    auditor = make_auditor(db, org_config)

    output = auditor.run_audit()

    assert re.fullmatch(r"AUD-\d{8}-\d{6}", output.audit_id)
    assert output.incidents_analyzed == 5

    recurring = [
        flag for flag in output.flags if flag.pattern_type == PatternType.RECURRING_LOCATION
    ]
    assert len(recurring) == 1
    flag = recurring[0]

    assert flag.affected_service == "payment-service"
    assert set(flag.contributing_incident_ids) == RECURRING_INCIDENT_IDS
    assert len(flag.flagged_locations) == 1
    assert flag.flagged_locations[0].file_path == "payment_service/processor.py"
    assert flag.flagged_locations[0].line_number == 31
    assert flag.flagged_locations[0].function_name == "process_payment"

    # Deterministic confidence: 3 incidents -> 0.55 + 0.30 = 0.85 (> 0.70).
    assert flag.confidence == pytest.approx(0.85)
    assert flag.confidence > 0.70

    # Fireworks returned {} (no API key), so the fallback assessment is built
    # only from verified facts and says so.
    assert "AI assessment unavailable" in flag.assessment
    for incident_id in RECURRING_INCIDENT_IDS:
        assert incident_id in flag.assessment
    assert "payment_service/processor.py:31" in flag.assessment

    # Graph context and attribution come along with the flag.
    connected_ids = {node.node_id for node in flag.connected_nodes}
    assert PROCESS_PAYMENT_NODE in connected_ids
    assert "shared/database_pool.py" in connected_ids
    assert flag.developer_attribution  # owners of the node + direct neighbors
    assert flag.recommended_reviewers == ["Priya Sharma", "marcus.webb"]

    # PLM ticket created in demo mode with a PLM-#### id.
    assert flag.plm_ticket_id is not None
    assert re.fullmatch(r"PLM-\d{4}", flag.plm_ticket_id)
    assert output.plm_tickets_created == [flag.plm_ticket_id]

    # Only the recurring pattern fires on the seeded data.
    assert output.patterns_found == 1
    assert not [f for f in output.flags if f.pattern_type == PatternType.SERVICE_STRESS]
    assert not [f for f in output.flags if f.pattern_type == PatternType.CASCADING_FAILURE]


def test_service_stress_does_not_fire_on_single_failure_type(db, org_config):
    # payment-service has 3 seeded incidents but ALL are null_pointer:
    # a repeated bug, not systemic stress. The stress pattern must stay silent.
    assert db.seed_historical_incidents(HISTORICAL_FOLDER) == 5
    output = make_auditor(db, org_config).run_audit()
    stress = [flag for flag in output.flags if flag.pattern_type == PatternType.SERVICE_STRESS]
    assert stress == []


def test_audit_row_is_persisted(db, org_config):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    output = make_auditor(db, org_config).run_audit()

    latest = db.get_latest_audit()
    assert latest is not None
    assert latest["audit_id"] == output.audit_id
    assert latest["incidents_analyzed"] == 5
    assert latest["patterns_found"] == 1
    assert len(latest["flags"]) == 1
    assert latest["flags"][0]["pattern_type"] == "recurring_location"
    assert latest["plm_tickets_created"] == output.plm_tickets_created


# ---------------------------------------------------------------------------
# Auditor: Fireworks response policy
# ---------------------------------------------------------------------------


def test_low_confidence_fireworks_response_skips_the_flag(db, org_config, monkeypatch):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    monkeypatch.setattr(
        FireworksClient,
        "complete_json",
        lambda self, prompt, **kwargs: {"assessment": "weak guess", "confidence": 0.3},
    )

    output = make_auditor(db, org_config).run_audit()

    assert output.flags == []
    assert output.patterns_found == 0
    assert output.plm_tickets_created == []
    # The audit run is still persisted even when everything is skipped.
    assert db.get_latest_audit()["audit_id"] == output.audit_id


def test_confident_fireworks_assessment_is_used_verbatim(db, org_config, monkeypatch):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    ai_text = (
        "The line `transaction_id = gateway_response.transaction_id` dereferences "
        "gateway_response without a None check even though charge() can return None."
    )
    monkeypatch.setattr(
        FireworksClient,
        "complete_json",
        lambda self, prompt, **kwargs: {
            "assessment": ai_text,
            "confidence": 0.9,
            "recommended_reviewers_reasoning": "Payments team owns the processor.",
        },
    )

    output = make_auditor(db, org_config).run_audit()

    recurring = [
        flag for flag in output.flags if flag.pattern_type == PatternType.RECURRING_LOCATION
    ]
    assert len(recurring) == 1
    assert recurring[0].assessment == ai_text
    # Flag confidence stays deterministic regardless of the AI's own number.
    assert recurring[0].confidence == pytest.approx(0.85)
    assert recurring[0].plm_ticket_id is not None


def test_fireworks_empty_dict_keeps_flag_and_warns(db, org_config, caplog):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="agents.auditor"):
        output = make_auditor(db, org_config).run_audit()

    assert output.patterns_found == 1
    assert any("fall" in message.lower() for message in caplog.messages)


# ---------------------------------------------------------------------------
# Auditor: other patterns fire when the evidence supports them
# ---------------------------------------------------------------------------


def test_service_stress_fires_with_multiple_failure_types(db, org_config):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    now = datetime.now()
    db.insert_intelligence_record(
        make_incident("INC-TEST-9001", failure_type=FailureType.TIMEOUT, timestamp=now - timedelta(hours=2))
    )
    db.insert_intelligence_record(
        make_incident(
            "INC-TEST-9002",
            failure_type=FailureType.CONNECTION_REFUSED,
            timestamp=now - timedelta(hours=26),
        )
    )

    output = make_auditor(db, org_config).run_audit()

    stress = [flag for flag in output.flags if flag.pattern_type == PatternType.SERVICE_STRESS]
    assert len(stress) == 1
    flag = stress[0]
    assert flag.affected_service == "payment-service"
    assert len(flag.contributing_incident_ids) == 5
    assert {"INC-TEST-9001", "INC-TEST-9002"} <= set(flag.contributing_incident_ids)
    # 5 incidents -> min(0.95, 0.55 + 0.50) = 0.95.
    assert flag.confidence == pytest.approx(0.95)
    # The fallback assessment cites the specific failure types observed.
    assert "null_pointer" in flag.assessment
    assert "timeout" in flag.assessment
    assert "connection_refused" in flag.assessment
    assert flag.plm_ticket_id is not None

    # The recurring-location flag from the seeded data is still present.
    recurring = [
        f for f in output.flags if f.pattern_type == PatternType.RECURRING_LOCATION
    ]
    assert len(recurring) == 1


def test_cascading_failure_fires_on_repeated_cross_service_pairs(db, org_config):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    now = datetime.now()
    # Two cross-service pairs, each within 30 minutes.
    db.insert_intelligence_record(
        make_incident("INC-TEST-9101", service="payment-service", timestamp=now - timedelta(hours=5))
    )
    db.insert_intelligence_record(
        make_incident(
            "INC-TEST-9102",
            service="auth-service",
            failure_type=FailureType.AUTH_FAILURE,
            timestamp=now - timedelta(hours=5) + timedelta(minutes=10),
        )
    )
    db.insert_intelligence_record(
        make_incident(
            "INC-TEST-9103",
            service="recommendation-engine",
            failure_type=FailureType.TIMEOUT,
            timestamp=now - timedelta(hours=8),
        )
    )
    db.insert_intelligence_record(
        make_incident(
            "INC-TEST-9104",
            service="auth-service",
            failure_type=FailureType.AUTH_FAILURE,
            timestamp=now - timedelta(hours=8) + timedelta(minutes=15),
        )
    )

    output = make_auditor(db, org_config).run_audit()

    cascading = [
        flag for flag in output.flags if flag.pattern_type == PatternType.CASCADING_FAILURE
    ]
    assert len(cascading) == 1
    flag = cascading[0]
    assert set(flag.contributing_incident_ids) == {
        "INC-TEST-9101",
        "INC-TEST-9102",
        "INC-TEST-9103",
        "INC-TEST-9104",
    }
    assert "payment-service" in flag.affected_service
    assert "auth-service" in flag.affected_service
    assert flag.confidence == pytest.approx(0.95)
    assert flag.plm_ticket_id is not None


# ---------------------------------------------------------------------------
# Auditor: resilience — external failures never crash the audit
# ---------------------------------------------------------------------------


def test_audit_survives_github_and_jira_failures(db, org_config, monkeypatch):
    db.seed_historical_incidents(HISTORICAL_FOLDER)

    def _github_boom(self, *args, **kwargs):
        raise RuntimeError("simulated GitHub outage")

    def _jira_boom(self, *args, **kwargs):
        raise RuntimeError("simulated Jira outage")

    monkeypatch.setattr(GitHubClient, "get_file_content_at_line", _github_boom)
    monkeypatch.setattr(JiraClient, "create_issue", _jira_boom)

    output = make_auditor(db, org_config).run_audit()

    # The flag survives (fallback assessment without a code citation),
    # only the PLM ticket is missing.
    assert output.patterns_found == 1
    flag = output.flags[0]
    assert flag.pattern_type == PatternType.RECURRING_LOCATION
    assert "AI assessment unavailable" in flag.assessment
    assert flag.plm_ticket_id is None
    assert output.plm_tickets_created == []
    assert db.get_latest_audit()["audit_id"] == output.audit_id
