"""Tests for agents.ticket_writer.TicketWriter.

Everything runs offline: DEMO_MODE is forced true, the DuckDB file lives in
a tmp_path, and FireworksClient.complete is always monkeypatched so no test
can make a network call. Any accidental real HTTP call fails the test
immediately.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.ticket_writer import (
    FALLBACK_PREFIX,
    NO_ACTION_SENTENCE,
    NO_PROBABLE_CAUSE_MESSAGE,
    NO_STACK_TRACE_MESSAGE,
    TicketWriter,
)
from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.schemas import (
    BlastRadius,
    ClassificationResult,
    Environment,
    FailureType,
    OutputType,
    Priority,
    ProbableCause,
    RichContext,
    RoutingDecision,
    Stakeholder,
    StackTraceLocation,
)
from integrations.jira_client import JiraClient
from integrations.slack_client import SlackClient

ORG_CONFIG = json.loads(
    (REPO_ROOT / "data" / "org_config.json").read_text(encoding="utf-8")
)

SECTION_HEADERS = [
    "## Summary",
    "## Probable Cause",
    "## Investigate First",
    "## Evidence",
    "## Blast Radius",
    "## Routing Note",
]


def _no_network(*args, **kwargs):
    raise AssertionError("A real network call was attempted during tests.")


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Isolated DB, DEMO_MODE true, and no possible network access."""
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "routezero_test.db"))
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr("integrations.jira_client.requests.post", _no_network)
    monkeypatch.setattr("integrations.slack_client.requests.post", _no_network)
    monkeypatch.setattr("core.fireworks_client.requests.post", _no_network)
    DatabaseManager.reset_instance()
    yield
    DatabaseManager.reset_instance()


@pytest.fixture()
def db():
    return DatabaseManager()


@pytest.fixture()
def fireworks_empty(monkeypatch):
    """Simulate the no-API-key behavior: complete returns ''."""
    monkeypatch.setattr(
        FireworksClient,
        "complete",
        lambda self, prompt, system=None, max_tokens=1024: "",
    )


@pytest.fixture()
def writer(db, fireworks_empty):
    return TicketWriter(
        db=db,
        org_config=ORG_CONFIG,
        fireworks=FireworksClient(),
        jira=JiraClient(),
        slack=SlackClient(),
    )


def make_writer(db):
    return TicketWriter(
        db=db,
        org_config=ORG_CONFIG,
        fireworks=FireworksClient(),
        jira=JiraClient(),
        slack=SlackClient(),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_context(**overrides):
    base = dict(
        raw_error_text=(
            "Traceback (most recent call last):\n"
            '  File "payment_service/processor.py", line 31, in process_payment\n'
            "AttributeError: 'NoneType' object has no attribute 'charge'"
        ),
        environment=Environment.PRODUCTION,
        affected_users=847,
        customer_tier="enterprise",
        sla_breach_minutes=40,
    )
    base.update(overrides)
    return RichContext(**base)


def make_decision(
    context,
    incident_id="INC-TW-0001",
    with_probable_cause=True,
    with_stack_trace=True,
):
    locations = []
    if with_stack_trace:
        locations = [
            StackTraceLocation(
                file_path="payment_service/processor.py",
                line_number=31,
                function_name="process_payment",
            ),
            StackTraceLocation(
                file_path="payment_service/gateway.py",
                line_number=88,
                function_name="submit_charge",
            ),
            StackTraceLocation(
                file_path="shared/database_pool.py",
                line_number=54,
                function_name="acquire",
            ),
            StackTraceLocation(
                file_path="shared/retry.py",
                line_number=12,
                function_name="with_retries",
            ),
        ]
    probable_cause = None
    if with_probable_cause:
        probable_cause = ProbableCause(
            description="A deployment shortly before the incident is the likely trigger.",
            commit_hash="c9d4e17",
            deployer="Elena Volkov",
            commit_message="Refactor gateway response handling",
            minutes_before_incident=37,
            confidence=0.78,
        )
    classification = ClassificationResult(
        incident_id=incident_id,
        raw_input=context,
        detected_service="payment-service",
        failure_type=FailureType.NULL_POINTER,
        environment=Environment.PRODUCTION,
        is_critical_path=True,
        blast_radius=BlastRadius.ENTERPRISE_CUSTOMERS,
        is_first_occurrence=False,
        occurrences_last_4h=4,
        stack_trace_locations=locations,
        classification_confidence=0.95,
        timestamp=datetime.now(),
        missing_context=["runbook_url", "on_call_engineer"],
    )
    return RoutingDecision(
        classification=classification,
        owning_team="payments-team",
        assignee="marcus.webb",
        priority=Priority.P1,
        priority_reasoning="critical_path service AND production environment",
        jira_project_key="PAY",
        stakeholders=[
            Stakeholder(name="Marcus Webb", role="on-call engineer"),
            Stakeholder(
                name="Sofia Reyes",
                role="engineering manager",
                output_type=OutputType.MANAGER_DIGEST,
            ),
            Stakeholder(
                name="Fatima Al-Rashid",
                role="backend team lead",
                output_type=OutputType.CROSS_FUNCTIONAL_FYI,
            ),
        ],
        manager_digest_triggered=True,
        probable_cause=probable_cause,
        related_tickets=["PAY-1042"],
        runbook_url="https://runbooks.streamco.example/payment-service",
        routing_confidence=0.92,
        routing_reasoning="payment-service maps to payments-team in the org config.",
        missing_context=["runbook_url", "on_call_engineer"],
    )


def get_content(output, output_type):
    return next(c for c in output.ticket_contents if c.output_type == output_type)


# ---------------------------------------------------------------------------
# Engineer ticket structure
# ---------------------------------------------------------------------------


def test_engineer_ticket_has_all_sections_in_order(writer):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    body = get_content(output, OutputType.FULL_TICKET).body

    positions = [body.find(header) for header in SECTION_HEADERS]
    assert all(pos >= 0 for pos in positions), f"missing section in body:\n{body}"
    assert positions == sorted(positions), "sections are out of order"

    # Deterministic list sections are built from the verified inputs only.
    assert "payment_service/processor.py:31 in process_payment" in body
    assert "Related tickets: PAY-1042" in body
    assert "Runbook: https://runbooks.streamco.example/payment-service" in body
    assert "- Affected users: 847" in body
    assert "- Critical path: yes" in body
    assert "- Routing confidence: 92%" in body
    assert "runbook_url, on_call_engineer" in body
    assert "payment-service maps to payments-team in the org config." in body
    # Probable cause section states cause and confidence percent.
    assert "c9d4e17" in body
    assert "Confidence: 78%" in body


def test_engineer_ticket_title_is_human_cased(writer):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    content = get_content(output, OutputType.FULL_TICKET)
    assert content.title == "[P1] payment-service: Null Pointer in Production"


def test_no_probable_cause_message_when_absent(writer):
    context = make_context()
    decision = make_decision(context, with_probable_cause=False)
    output = writer.write_tickets(decision, context, send=False)
    body = get_content(output, OutputType.FULL_TICKET).body
    assert NO_PROBABLE_CAUSE_MESSAGE in body
    assert "c9d4e17" not in body


def test_investigate_logs_directly_when_no_stack_trace(writer):
    context = make_context()
    decision = make_decision(context, with_stack_trace=False)
    output = writer.write_tickets(decision, context, send=False)
    body = get_content(output, OutputType.FULL_TICKET).body
    assert NO_STACK_TRACE_MESSAGE in body
    assert ".py:" not in body


def test_investigate_first_lists_at_most_three_locations(writer):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    body = get_content(output, OutputType.FULL_TICKET).body
    investigate = body.split("## Investigate First")[1].split("## Evidence")[0]
    assert "1. payment_service/processor.py:31 in process_payment" in investigate
    assert "3. shared/database_pool.py:54 in acquire" in investigate
    assert "shared/retry.py" not in investigate  # fourth location is cut


# ---------------------------------------------------------------------------
# Manager digest and FYI
# ---------------------------------------------------------------------------


def test_manager_digest_max_five_sentences_and_no_technical_tokens(writer):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    digest = get_content(output, OutputType.MANAGER_DIGEST).body
    assert _count_sentences(digest) <= 5
    assert ".py" not in digest
    assert ":31" not in digest
    # Business facts survive.
    assert "847" in digest
    assert "payments-team" in digest


def test_fyi_contains_no_action_sentence(writer):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    fyi = get_content(output, OutputType.CROSS_FUNCTIONAL_FYI).body
    assert NO_ACTION_SENTENCE in fyi
    assert "payments-team" in fyi


# ---------------------------------------------------------------------------
# Hallucination validation and fallbacks
# ---------------------------------------------------------------------------


def test_hallucinated_ai_text_is_discarded(db, monkeypatch):
    monkeypatch.setattr(
        FireworksClient,
        "complete",
        lambda self, prompt, system=None, max_tokens=1024: (
            "9999 users affected in nonexistent_file.py"
        ),
    )
    writer = make_writer(db)
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)

    full_body = get_content(output, OutputType.FULL_TICKET).body
    assert FALLBACK_PREFIX in full_body
    assert "9999" not in full_body
    assert "nonexistent_file.py" not in full_body

    digest = get_content(output, OutputType.MANAGER_DIGEST).body
    assert FALLBACK_PREFIX in digest
    assert "9999" not in digest


def test_empty_fireworks_response_uses_fallback_prefix(writer):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    full_body = get_content(output, OutputType.FULL_TICKET).body
    summary = full_body.split("## Probable Cause")[0]
    assert FALLBACK_PREFIX in summary
    # Fallback still cites the verified facts.
    assert "847" in summary
    assert "40" in summary


def test_grounded_ai_text_is_accepted(db, monkeypatch):
    monkeypatch.setattr(
        FireworksClient,
        "complete",
        lambda self, prompt, system=None, max_tokens=1024: (
            "payment-service failed in production and 847 users are affected."
        ),
    )
    writer = make_writer(db)
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    full_body = get_content(output, OutputType.FULL_TICKET).body
    summary = full_body.split("## Probable Cause")[0]
    assert "payment-service failed in production and 847 users are affected." in summary
    assert FALLBACK_PREFIX not in summary


def test_fireworks_exception_falls_back(db, monkeypatch):
    def _boom(self, prompt, system=None, max_tokens=1024):
        raise RuntimeError("simulated Fireworks outage")

    monkeypatch.setattr(FireworksClient, "complete", _boom)
    writer = make_writer(db)
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)
    assert FALLBACK_PREFIX in get_content(output, OutputType.FULL_TICKET).body


# ---------------------------------------------------------------------------
# Persistence: send=False vs send=True
# ---------------------------------------------------------------------------


def test_send_false_writes_rows_but_no_notifications(writer, db):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=False)

    counts = db.get_table_counts()
    assert counts["gold_incident_intelligence"] == 1
    assert counts["gold_created_tickets"] == 1
    assert counts["gold_notification_log"] == 0

    assert output.jira_ticket_id is None
    assert output.jira_url is None
    assert output.notifications_sent == []

    record = output.intelligence_record
    assert record.incident_id == "INC-TW-0001"
    assert record.linked_deployment == "c9d4e17"
    assert record.resolved is False
    assert record.agent3_summary == (
        "payment-service suffered a null_pointer error in production affecting "
        "847 users; routed to payments-team as P1 with probable cause linked "
        "to deployment c9d4e17."
    )


def test_send_true_demo_mode_creates_jira_and_notifications(writer, db):
    context = make_context()
    output = writer.write_tickets(make_decision(context), context, send=True)

    assert output.jira_ticket_id is not None
    assert re.fullmatch(r"PAY-\d{4}", output.jira_ticket_id)
    assert output.jira_url == (
        f"https://streamco-demo.atlassian.net/browse/{output.jira_ticket_id}"
    )

    # 1 Jira attempt + 3 Slack sends (one per stakeholder), all successful.
    assert len(output.notifications_sent) == 4
    assert all(n.success for n in output.notifications_sent)
    slack_records = [n for n in output.notifications_sent if n.channel == "#payments-oncall"]
    assert len(slack_records) == 3
    assert {n.output_type for n in slack_records} == {
        OutputType.FULL_TICKET,
        OutputType.MANAGER_DIGEST,
        OutputType.CROSS_FUNCTIONAL_FYI,
    }

    counts = db.get_table_counts()
    assert counts["gold_notification_log"] == 4
    assert counts["gold_incident_intelligence"] == 1
    assert counts["gold_created_tickets"] == 1
    assert output.intelligence_record.jira_ticket_ref == output.jira_ticket_id


# ---------------------------------------------------------------------------
# approve_and_send
# ---------------------------------------------------------------------------


def test_approve_and_send_creates_and_persists(writer, db):
    context = make_context()
    decision = make_decision(context)
    db.insert_routing_decision(decision)
    writer.write_tickets(decision, context, send=False)
    assert db.get_table_counts()["gold_notification_log"] == 0

    result = writer.approve_and_send("INC-TW-0001")

    assert result["jira_ticket_id"] is not None
    assert re.fullmatch(r"PAY-\d{4}", result["jira_ticket_id"])
    assert result["jira_url"] == (
        f"https://streamco-demo.atlassian.net/browse/{result['jira_ticket_id']}"
    )

    notifications = result["notifications"]
    assert len(notifications) == 4
    assert all(n["success"] for n in notifications)
    slack_notifications = [n for n in notifications if n["channel"] == "#payments-oncall"]
    assert len(slack_notifications) == 3
    assert {n["output_type"] for n in slack_notifications} == {
        "full_ticket",
        "manager_digest",
        "cross_functional_fyi",
    }
    assert {n["recipient"] for n in slack_notifications} == {
        "Marcus Webb",
        "Sofia Reyes",
        "Fatima Al-Rashid",
    }

    # Persisted back onto the stored rows.
    stored = db.get_ticket_output("INC-TW-0001")
    assert stored["jira_ticket_id"] == result["jira_ticket_id"]
    assert stored["jira_url"] == result["jira_url"]
    intelligence = next(
        row for row in db.get_all_incidents() if row["incident_id"] == "INC-TW-0001"
    )
    assert intelligence["jira_ticket_ref"] == result["jira_ticket_id"]
    assert db.get_table_counts()["gold_notification_log"] == 4


def test_approve_and_send_unknown_incident_raises(writer):
    with pytest.raises(ValueError):
        writer.approve_and_send("INC-DOES-NOT-EXIST")
