"""Tests for core.database.DatabaseManager."""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.database import DatabaseManager
from core.schemas import (
    ArchitecturalFlag,
    AuditorOutput,
    BlastRadius,
    ClassificationResult,
    Environment,
    FailureType,
    GraphEdge,
    GraphNode,
    IncidentIntelligenceRecord,
    NodeType,
    NotificationRecord,
    OutputType,
    PatternType,
    Priority,
    RichContext,
    RoutingDecision,
    Stakeholder,
    StackTraceLocation,
    TicketContent,
    TicketWriterOutput,
)

EXPECTED_TABLES = {
    "bronze_raw_incidents",
    "silver_classified_incidents",
    "silver_routing_decisions",
    "gold_created_tickets",
    "gold_incident_intelligence",
    "gold_notification_log",
    "gold_audit_runs",
    "graph_nodes",
    "graph_edges",
    "graph_incident_node_mapping",
}

HISTORICAL_FOLDER = str(REPO_ROOT / "data" / "historical_incidents")


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A DatabaseManager backed by a temp DuckDB file, singleton reset around it."""
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "routezero_test.db"))
    DatabaseManager.reset_instance()
    manager = DatabaseManager()
    yield manager
    DatabaseManager.reset_instance()


def make_classification(incident_id="INC-TEST-0001", service="payment-service", timestamp=None):
    return ClassificationResult(
        incident_id=incident_id,
        raw_input=RichContext(
            raw_error_text="AttributeError: 'NoneType' object has no attribute 'charge'",
            affected_users=847,
        ),
        detected_service=service,
        failure_type=FailureType.NULL_POINTER,
        environment=Environment.PRODUCTION,
        is_critical_path=True,
        blast_radius=BlastRadius.ENTERPRISE_CUSTOMERS,
        is_first_occurrence=False,
        occurrences_last_4h=4,
        stack_trace_locations=[
            StackTraceLocation(
                file_path="payment_service/processor.py",
                line_number=31,
                function_name="process_payment",
                raw_line='  File "payment_service/processor.py", line 31, in process_payment',
            )
        ],
        classification_confidence=0.95,
        timestamp=timestamp or datetime.now(),
        missing_context=["runbook_url", "on_call_engineer"],
    )


def make_routing(classification):
    return RoutingDecision(
        classification=classification,
        owning_team="payments-team",
        assignee="marcus.webb",
        priority=Priority.P1,
        priority_reasoning="critical_path service AND production environment",
        jira_project_key="PAY",
        stakeholders=[
            Stakeholder(name="Marcus Webb", role="on-call engineer"),
            Stakeholder(name="Sofia Reyes", role="manager", output_type=OutputType.MANAGER_DIGEST),
        ],
        manager_digest_triggered=True,
        probable_cause=None,
        related_tickets=["PAY-1042"],
        runbook_url=None,
        routing_confidence=0.92,
        routing_reasoning="Service payment-service maps to payments-team in the org config.",
        missing_context=["runbook_url"],
    )


def make_intelligence(incident_id, service="payment-service", timestamp=None, resolved=False):
    return IncidentIntelligenceRecord(
        incident_id=incident_id,
        jira_ticket_ref="PAY-9999",
        timestamp=timestamp or datetime.now(),
        service=service,
        failure_type=FailureType.NULL_POINTER,
        environment=Environment.PRODUCTION,
        is_critical_path=True,
        affected_users=500,
        priority=Priority.P1,
        stack_trace_locations=[
            StackTraceLocation(
                file_path="payment_service/processor.py",
                line_number=31,
                function_name="process_payment",
            )
        ],
        probable_cause=None,
        linked_deployment=None,
        routing_confidence=0.9,
        owning_team="payments-team",
        resolved=resolved,
        resolution_time_minutes=None,
        agent3_summary="Payment processing failed for 500 production users due to a None gateway response.",
    )


# ---------------------------------------------------------------------------
# Singleton and schema
# ---------------------------------------------------------------------------


def test_singleton_identity(db):
    assert DatabaseManager() is db
    assert DatabaseManager.get_instance() is db


def test_all_tables_exist_and_start_empty(db):
    counts = db.get_table_counts()
    assert set(counts) == EXPECTED_TABLES
    assert all(count == 0 for count in counts.values())


# ---------------------------------------------------------------------------
# Insert + query round trips
# ---------------------------------------------------------------------------


def test_bronze_insert(db):
    context = RichContext(raw_error_text="Traceback (most recent call last): ...")
    db.insert_bronze_incident("INC-TEST-0001", context, source_format="python_traceback")
    assert db.get_table_counts()["bronze_raw_incidents"] == 1


def test_classification_round_trip(db):
    result = make_classification()
    db.insert_classification(result)

    row = db.get_classification("INC-TEST-0001")
    assert row is not None
    assert row["detected_service"] == "payment-service"
    assert row["failure_type"] == "null_pointer"
    assert row["environment"] == "production"
    assert row["is_critical_path"] is True
    assert row["classification_confidence"] == pytest.approx(0.95)
    assert isinstance(row["timestamp"], datetime)
    assert isinstance(row["stack_trace_locations"], list)
    assert row["stack_trace_locations"][0]["line_number"] == 31
    assert row["stack_trace_locations"][0]["file_path"] == "payment_service/processor.py"
    assert row["missing_context"] == ["runbook_url", "on_call_engineer"]
    assert row["raw_input"]["affected_users"] == 847


def test_routing_round_trip(db):
    decision = make_routing(make_classification())
    db.insert_routing_decision(decision)

    row = db.get_routing_decision("INC-TEST-0001")
    assert row is not None
    assert row["owning_team"] == "payments-team"
    assert row["assignee"] == "marcus.webb"
    assert row["priority"] == "P1"
    assert row["manager_digest_triggered"] is True
    assert isinstance(row["stakeholders"], list) and len(row["stakeholders"]) == 2
    assert row["stakeholders"][1]["output_type"] == "manager_digest"
    assert row["related_tickets"] == ["PAY-1042"]
    assert row["classification"]["incident_id"] == "INC-TEST-0001"
    assert row["probable_cause"] is None


def test_intelligence_round_trip_and_ordering(db):
    older = make_intelligence("INC-TEST-0100", timestamp=datetime.now() - timedelta(days=2))
    newer = make_intelligence("INC-TEST-0101", timestamp=datetime.now() - timedelta(hours=1))
    db.insert_intelligence_record(older)
    db.insert_intelligence_record(newer)

    incidents = db.get_all_incidents()
    assert [row["incident_id"] for row in incidents] == ["INC-TEST-0101", "INC-TEST-0100"]
    assert isinstance(incidents[0]["timestamp"], datetime)
    assert isinstance(incidents[0]["stack_trace_locations"], list)
    assert incidents[0]["stack_trace_locations"][0]["function_name"] == "process_payment"
    assert incidents[0]["resolved"] is False


def test_ticket_output_round_trip(db):
    decision = make_routing(make_classification())
    output = TicketWriterOutput(
        incident_id="INC-TEST-0001",
        routing_decision=decision,
        ticket_contents=[
            TicketContent(
                recipient_name="Marcus Webb",
                recipient_role="on-call engineer",
                output_type=OutputType.FULL_TICKET,
                title="[P1] payment-service: null pointer in process_payment",
                body="Full ticket body",
            ),
            TicketContent(
                recipient_name="Sofia Reyes",
                recipient_role="manager",
                output_type=OutputType.MANAGER_DIGEST,
                title="Digest: payment-service incident",
                body="Digest body",
            ),
        ],
        jira_ticket_id="PAY-1100",
        jira_url="https://jira.example.com/browse/PAY-1100",
        notifications_sent=[
            NotificationRecord(
                recipient="Marcus Webb",
                channel="#payments-oncall",
                output_type=OutputType.FULL_TICKET,
                success=True,
            )
        ],
        intelligence_record=make_intelligence("INC-TEST-0001"),
    )
    db.insert_created_ticket(output)

    row = db.get_ticket_output("INC-TEST-0001")
    assert row is not None
    assert row["jira_ticket_id"] == "PAY-1100"
    assert row["jira_url"] == "https://jira.example.com/browse/PAY-1100"
    assert isinstance(row["ticket_contents"], list) and len(row["ticket_contents"]) == 2
    assert row["ticket_contents"][1]["output_type"] == "manager_digest"
    assert row["notifications_sent"][0]["success"] is True
    assert isinstance(row["created_at"], datetime)

    assert db.get_ticket_output("INC-DOES-NOT-EXIST") is None


def test_notification_log(db):
    record = NotificationRecord(
        recipient="Priya Sharma",
        channel="#payments-oncall",
        output_type=OutputType.FULL_TICKET,
        success=False,
        error_message="webhook timeout",
    )
    db.insert_notification_log("INC-TEST-0001", record)
    assert db.get_table_counts()["gold_notification_log"] == 1


def test_audit_round_trip_and_latest(db):
    assert db.get_latest_audit() is None

    first = AuditorOutput(
        audit_id="AUD-0001",
        timestamp=datetime.now() - timedelta(hours=2),
        incidents_analyzed=5,
        patterns_found=1,
        flags=[
            ArchitecturalFlag(
                pattern_type=PatternType.RECURRING_LOCATION,
                affected_service="payment-service",
                contributing_incident_ids=["INC-2026-0134", "INC-2026-0149", "INC-2026-0162"],
                flagged_locations=[
                    StackTraceLocation(
                        file_path="payment_service/processor.py",
                        line_number=31,
                        function_name="process_payment",
                    )
                ],
                assessment="The gateway response is dereferenced without a None check.",
                confidence=0.88,
            )
        ],
        plm_tickets_created=["PLM-101"],
    )
    second = AuditorOutput(
        audit_id="AUD-0002",
        timestamp=datetime.now(),
        incidents_analyzed=6,
        patterns_found=0,
        flags=[],
        plm_tickets_created=[],
    )
    db.insert_audit_run(first)
    db.insert_audit_run(second)

    latest = db.get_latest_audit()
    assert latest is not None
    assert latest["audit_id"] == "AUD-0002"
    assert isinstance(latest["timestamp"], datetime)
    assert latest["flags"] == []
    assert latest["plm_tickets_created"] == []

    assert db.get_table_counts()["gold_audit_runs"] == 2


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_seed_loads_exactly_five_recent_incidents(db):
    inserted = db.seed_historical_incidents(HISTORICAL_FOLDER)
    assert inserted == 5

    incidents = db.get_all_incidents()
    assert len(incidents) == 5

    now = datetime.now()
    timestamps = [row["timestamp"] for row in incidents]
    assert all(now - timedelta(days=7) <= ts <= now for ts in timestamps)
    # Most recent incident re-based to about one day ago.
    assert now - timedelta(days=2) < max(timestamps) <= now - timedelta(hours=23)

    # The recurring location appears in exactly three incidents.
    recurring = [
        row
        for row in incidents
        if any(
            loc["file_path"] == "payment_service/processor.py" and loc["line_number"] == 31
            for loc in row["stack_trace_locations"]
        )
    ]
    assert len(recurring) == 3
    assert all(row["service"] == "payment-service" for row in recurring)
    assert all(row["failure_type"] == "null_pointer" for row in recurring)

    services = {row["service"] for row in incidents}
    assert services == {"payment-service", "auth-service", "recommendation-engine"}


def test_seed_is_idempotent(db, caplog):
    assert db.seed_historical_incidents(HISTORICAL_FOLDER) == 5
    with caplog.at_level(logging.WARNING, logger="core.database"):
        second = db.seed_historical_incidents(HISTORICAL_FOLDER)
    assert second == 0
    assert db.get_table_counts()["gold_incident_intelligence"] == 5
    assert any("skipped" in message.lower() for message in caplog.messages)


def test_seed_default_folder_resolves(db):
    # The default relative folder resolves against the repo root fallback.
    assert db.seed_historical_incidents() == 5


def test_get_incidents_since(db):
    db.seed_historical_incidents(HISTORICAL_FOLDER)
    assert len(db.get_incidents_since(7)) == 5
    # All seeded incidents are at least ~1 day old.
    assert len(db.get_incidents_since(0)) == 0


# ---------------------------------------------------------------------------
# Counting and resolution
# ---------------------------------------------------------------------------


def test_count_service_incidents_last_hour(db):
    db.insert_intelligence_record(
        make_intelligence("INC-TEST-0200", timestamp=datetime.now() - timedelta(minutes=10))
    )
    db.insert_intelligence_record(
        make_intelligence("INC-TEST-0201", timestamp=datetime.now() - timedelta(minutes=30))
    )
    db.insert_intelligence_record(
        make_intelligence("INC-TEST-0202", timestamp=datetime.now() - timedelta(hours=3))
    )
    assert db.count_service_incidents_last_hour("payment-service") == 2
    assert db.count_service_incidents_last_hour("auth-service") == 0


def test_mark_incident_resolved(db):
    started = datetime.now() - timedelta(minutes=45)
    db.insert_intelligence_record(make_intelligence("INC-TEST-0300", timestamp=started))

    db.mark_incident_resolved("INC-TEST-0300")

    row = next(r for r in db.get_all_incidents() if r["incident_id"] == "INC-TEST-0300")
    assert row["resolved"] is True
    assert 43 <= row["resolution_time_minutes"] <= 47

    # Unknown incident is a no-op, not an error.
    db.mark_incident_resolved("INC-DOES-NOT-EXIST")


# ---------------------------------------------------------------------------
# Code knowledge graph
# ---------------------------------------------------------------------------


def test_graph_nodes_edges_idempotent(db):
    nodes = [
        GraphNode(
            node_id="payment_service/processor.py::process_payment",
            name="process_payment",
            file_path="payment_service/processor.py",
            start_line=25,
            end_line=48,
            node_type=NodeType.FUNCTION,
            owner="marcus.webb",
            last_modified=datetime(2026, 6, 20, 10, 0, 0),
        ),
        GraphNode(
            node_id="shared/database_pool.py",
            name="database_pool",
            file_path="shared/database_pool.py",
            start_line=1,
            end_line=80,
            node_type=NodeType.MODULE,
        ),
    ]
    edges = [
        GraphEdge(
            edge_id="e1",
            source_node_id="payment_service/processor.py::process_payment",
            target_node_id="shared/database_pool.py",
            relationship_type="calls",
            weight=1.0,
        )
    ]

    db.insert_graph_nodes(nodes)
    db.insert_graph_edges(edges)
    # Second insert must not duplicate anything.
    db.insert_graph_nodes(nodes)
    db.insert_graph_edges(edges)

    stored_nodes = db.get_graph_nodes()
    stored_edges = db.get_graph_edges()
    assert len(stored_nodes) == 2
    assert len(stored_edges) == 1
    function_node = next(n for n in stored_nodes if n["node_type"] == "function")
    assert function_node["start_line"] == 25
    assert isinstance(function_node["last_modified"], datetime)
    assert stored_edges[0]["relationship_type"] == "calls"


def test_incident_node_mapping_counts(db):
    node_id = "payment_service/processor.py::process_payment"
    db.upsert_incident_node_mapping("INC-2026-0134", node_id, 1)
    db.upsert_incident_node_mapping("INC-2026-0149", node_id, 1)
    # Upserting the same pair again must not add a distinct incident.
    db.upsert_incident_node_mapping("INC-2026-0149", node_id, 3)
    db.upsert_incident_node_mapping("INC-2026-0141", "auth_service/token_validator.py::validate_token", 1)

    counts = db.get_node_incident_counts()
    assert counts[node_id] == 2
    assert counts["auth_service/token_validator.py::validate_token"] == 1
    assert db.get_table_counts()["graph_incident_node_mapping"] == 3
