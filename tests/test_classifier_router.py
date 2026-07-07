"""Tests for agents.classifier.IncidentClassifier and agents.router.IncidentRouter.

Covers the four demo scenarios from the spec plus the low-confidence
Fireworks-assisted path and its fallback. The database is isolated to a
temp DuckDB file per test and FireworksClient never touches the network.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.classifier import IncidentClassifier
from agents.router import IncidentRouter
from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.schemas import (
    BlastRadius,
    DeploymentInfo,
    Environment,
    FailureType,
    OutputType,
    Priority,
    RichContext,
)

SAMPLE_DIR = REPO_ROOT / "data" / "sample_errors"
ORG_CONFIG = json.loads((REPO_ROOT / "data" / "org_config.json").read_text(encoding="utf-8"))

ALL_OPTIONAL_FIELDS = {
    "service_hint",
    "environment",
    "occurrences_last_4h",
    "affected_users",
    "customer_tier",
    "recent_deployment",
    "sla_breach_minutes",
    "on_call_engineer",
    "related_ticket_ids",
    "runbook_url",
}


def read_sample(name: str) -> str:
    return (SAMPLE_DIR / name).read_text(encoding="utf-8")


def stakeholder_by_role(decision, role):
    return [s for s in decision.stakeholders if s.role == role]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """DatabaseManager singleton backed by a temp DuckDB file."""
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.db"))
    DatabaseManager.reset_instance()
    manager = DatabaseManager()
    yield manager
    DatabaseManager.reset_instance()


@pytest.fixture()
def guarded_fireworks(monkeypatch):
    """A FireworksClient that fails the test if any completion is attempted.

    Used for the deterministic scenarios, where the router must make zero
    LLM calls.
    """

    def _forbidden(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("Fireworks must not be called on the deterministic path")

    monkeypatch.setattr(FireworksClient, "complete_json", _forbidden)
    monkeypatch.setattr(FireworksClient, "complete", _forbidden)
    return FireworksClient()


@pytest.fixture()
def classifier(db):
    return IncidentClassifier(db, ORG_CONFIG)


@pytest.fixture()
def router(db, guarded_fireworks):
    return IncidentRouter(db, ORG_CONFIG, guarded_fireworks)


# ---------------------------------------------------------------------------
# Scenario 1: payment error with full rich context
# ---------------------------------------------------------------------------


def test_scenario_1_payment_error_full_context(db, classifier, router):
    context = RichContext(
        raw_error_text=read_sample("payment_error.txt"),
        environment=Environment.PRODUCTION,
        occurrences_last_4h=3,
        affected_users=847,
        customer_tier="enterprise",
        recent_deployment=DeploymentInfo(
            deployer="Elena Volkov",
            commit_hash="9f8c2ba",
            commit_message="Refactor gateway retry handling",
            deployed_at=datetime.now() - timedelta(minutes=37),
        ),
        sla_breach_minutes=40,
        on_call_engineer="Marcus Webb",
    )

    classification = classifier.classify(context)

    assert classification.incident_id.startswith("INC-")
    assert classification.detected_service == "payment-service"
    assert classification.failure_type == FailureType.NULL_POINTER
    assert classification.environment == Environment.PRODUCTION
    assert classification.is_critical_path is True
    assert classification.blast_radius == BlastRadius.ENTERPRISE_CUSTOMERS
    assert classification.is_first_occurrence is False
    assert classification.occurrences_last_4h == 3
    assert classification.classification_confidence > 0.85
    # The processor.py:31 frame from the sample traceback must be extracted.
    assert any(
        loc.file_path == "payment_service/processor.py"
        and loc.line_number == 31
        and loc.function_name == "process_payment"
        for loc in classification.stack_trace_locations
    )
    assert set(classification.missing_context) == {"service_hint", "related_ticket_ids", "runbook_url"}

    decision = router.route(classification, context)

    assert decision.owning_team == "payments-team"
    assert decision.jira_project_key == "PAY"
    assert decision.priority == Priority.P1
    assert "critical_path" in decision.priority_reasoning
    assert "production" in decision.priority_reasoning
    # On-call engineer provided in context becomes the primary assignee.
    assert decision.assignee == "Marcus Webb"
    assert decision.manager_digest_triggered is True

    # Probable cause: deployment 37 minutes before the incident.
    assert decision.probable_cause is not None
    assert decision.probable_cause.commit_hash == "9f8c2ba"
    assert "9f8c2ba" in decision.probable_cause.description
    assert "Elena Volkov" in decision.probable_cause.description
    assert "Refactor gateway retry handling" in decision.probable_cause.description
    assert decision.probable_cause.minutes_before_incident == 37
    assert decision.probable_cause.confidence == pytest.approx(1 - 37 / 120, abs=0.02)

    assert decision.routing_confidence > 0.85

    # Stakeholders: assignee, team lead, manager digest, cross-functional FYI
    # (blast radius is enterprise_customers).
    assert [s.name for s in stakeholder_by_role(decision, "assignee")] == ["Marcus Webb"]
    assert [s.name for s in stakeholder_by_role(decision, "team_lead")] == ["Priya Sharma"]
    managers = stakeholder_by_role(decision, "manager")
    assert [s.name for s in managers] == ["Sofia Reyes"]
    assert managers[0].output_type == OutputType.MANAGER_DIGEST
    fyi = stakeholder_by_role(decision, "cross_functional_fyi")
    assert [s.name for s in fyi] == ["Omar Haddad"]
    assert fyi[0].output_type == OutputType.CROSS_FUNCTIONAL_FYI

    # Routing reasoning must let a judge verify every rule that fired.
    assert "payments-team" in decision.routing_reasoning
    assert "P1" in decision.routing_reasoning
    assert "9f8c2ba" in decision.routing_reasoning

    # Both agents persisted their outputs.
    counts = db.get_table_counts()
    assert counts["bronze_raw_incidents"] == 1
    assert counts["silver_classified_incidents"] == 1
    assert counts["silver_routing_decisions"] == 1
    assert db.get_classification(classification.incident_id) is not None
    stored = db.get_routing_decision(classification.incident_id)
    assert stored is not None
    assert stored["owning_team"] == "payments-team"


# ---------------------------------------------------------------------------
# Scenario 2: auth error with no optional context at all
# ---------------------------------------------------------------------------


def test_scenario_2_auth_error_minimal_context(db, classifier, router):
    context = RichContext(raw_error_text=read_sample("auth_error.txt"))

    classification = classifier.classify(context)

    assert classification.detected_service == "auth-service"
    assert classification.failure_type == FailureType.AUTH_FAILURE
    # The sample log mentions the pod "auth-service-prod-6d9f4b", so the
    # text-based environment detection yields production.
    assert classification.environment == Environment.PRODUCTION
    assert classification.is_critical_path is True
    assert classification.blast_radius == BlastRadius.UNKNOWN
    assert classification.is_first_occurrence is True
    assert classification.classification_confidence >= 0.65
    # Every optional field is missing.
    assert set(classification.missing_context) == ALL_OPTIONAL_FIELDS
    # Both frames of the auth traceback are extracted.
    assert any(
        loc.file_path == "auth_service/token_validator.py" and loc.line_number == 28
        for loc in classification.stack_trace_locations
    )

    decision = router.route(classification, context)

    assert decision.owning_team == "security-team"
    assert decision.jira_project_key == "AUTH"
    # critical_path service AND production environment -> P1.
    assert decision.priority == Priority.P1
    assert "critical_path" in decision.priority_reasoning
    # No on-call engineer provided: falls back to the team default assignee.
    assert decision.assignee == "aisha.patel"
    # P1 always triggers the manager digest.
    assert decision.manager_digest_triggered is True
    assert [s.name for s in stakeholder_by_role(decision, "manager")] == ["David Osei"]
    assert [s.name for s in stakeholder_by_role(decision, "team_lead")] == ["James Chen"]
    # Blast radius unknown: no cross-functional FYI.
    assert stakeholder_by_role(decision, "cross_functional_fyi") == []
    assert decision.probable_cause is None
    assert decision.related_tickets == []
    assert decision.runbook_url is None
    assert set(decision.missing_context) == ALL_OPTIONAL_FIELDS


# ---------------------------------------------------------------------------
# Scenario 3: ML timeout with medium context (staging, 85 users)
# ---------------------------------------------------------------------------


def test_scenario_3_ml_timeout_staging(db, classifier, router):
    context = RichContext(
        raw_error_text=read_sample("ml_timeout_error.txt"),
        affected_users=85,
        environment=Environment.STAGING,
    )

    classification = classifier.classify(context)

    assert classification.detected_service == "recommendation-engine"
    assert classification.failure_type == FailureType.TIMEOUT
    assert classification.environment == Environment.STAGING
    assert classification.is_critical_path is False
    assert classification.blast_radius == BlastRadius.SUBSET_OF_USERS
    assert classification.classification_confidence >= 0.65

    decision = router.route(classification, context)

    assert decision.owning_team == "ml-platform-team"
    assert decision.jira_project_key == "ML"
    # Note: the spec's ordered priority rules alone would yield P3 for a
    # non-critical-path staging service; the documented extra rule
    # (staging + affected_users >= 50 => P2) honors the demo expectation.
    assert decision.priority == Priority.P2
    assert "staging incident with material user impact" in decision.priority_reasoning
    # No digest: not P1, 85 < 500 users, no incident burst in the last hour.
    assert decision.manager_digest_triggered is False
    assert stakeholder_by_role(decision, "manager") == []
    # subset_of_users blast radius: no cross-functional FYI.
    assert stakeholder_by_role(decision, "cross_functional_fyi") == []
    assert decision.assignee == "carlos.mendez"
    assert decision.probable_cause is None


# ---------------------------------------------------------------------------
# Scenario 4: video/CDN connection refused with no context
# ---------------------------------------------------------------------------


def test_scenario_4_video_error_no_context(db, classifier, router):
    context = RichContext(raw_error_text=read_sample("video_error.txt"))

    classification = classifier.classify(context)

    assert classification.detected_service == "video-delivery"
    assert classification.failure_type == FailureType.CONNECTION_REFUSED
    # "env=production" appears in the sample log line.
    assert classification.environment == Environment.PRODUCTION
    assert classification.is_critical_path is True
    assert classification.classification_confidence >= 0.65

    decision = router.route(classification, context)

    assert decision.owning_team == "infra-team"
    assert decision.jira_project_key == "VID"
    assert decision.priority == Priority.P1
    assert decision.assignee == "lisa.park"
    assert [s.name for s in stakeholder_by_role(decision, "team_lead")] == ["Omar Haddad"]


def test_cross_functional_fyi_redirects_when_infra_owns(db, classifier, router):
    """When infra-team owns the incident, the FYI goes to backend-team's lead."""
    context = RichContext(
        raw_error_text=read_sample("video_error.txt"),
        affected_users=3120,  # > 1000 -> all_users blast radius
    )
    classification = classifier.classify(context)
    assert classification.blast_radius == BlastRadius.ALL_USERS

    decision = router.route(classification, context)
    assert decision.owning_team == "infra-team"
    fyi = stakeholder_by_role(decision, "cross_functional_fyi")
    assert [s.name for s in fyi] == ["Fatima Al-Rashid"]


# ---------------------------------------------------------------------------
# Classifier-specific behavior
# ---------------------------------------------------------------------------


def test_service_hint_wins_with_full_confidence(db, classifier):
    context = RichContext(
        raw_error_text="something exploded, nobody knows why",
        service_hint="payment-service",
    )
    classification = classifier.classify(context)
    assert classification.detected_service == "payment-service"
    assert classification.classification_confidence == pytest.approx(1.0)


def test_java_stack_trace_extraction(db, classifier):
    text = (
        'Exception in thread "main" java.lang.NullPointerException\n'
        "\tat com.streamco.payment.Processor.processPayment(Processor.java:31)\n"
        "\tat com.streamco.api.Handler.handle(Handler.java:88)\n"
    )
    classification = classifier.classify(RichContext(raw_error_text=text))
    assert classification.failure_type == FailureType.NULL_POINTER
    frames = classification.stack_trace_locations
    assert len(frames) == 2
    assert frames[0].file_path == "Processor.java"
    assert frames[0].line_number == 31
    assert frames[0].function_name == "com.streamco.payment.Processor.processPayment"
    assert frames[1].file_path == "Handler.java"
    assert frames[1].line_number == 88


def test_stack_trace_extraction_capped_at_ten_frames(db, classifier):
    frames = "\n".join(
        f'  File "svc/module_{i}.py", line {i + 1}, in func_{i}' for i in range(15)
    )
    text = f"Traceback (most recent call last):\n{frames}\nValueError: boom"
    classification = classifier.classify(RichContext(raw_error_text=text))
    assert len(classification.stack_trace_locations) == 10


def test_unmatched_error_is_unknown_low_confidence(db, classifier):
    classification = classifier.classify(
        RichContext(raw_error_text="Something completely mysterious happened overnight.")
    )
    assert classification.detected_service == "unknown"
    assert classification.failure_type == FailureType.UNKNOWN
    assert classification.classification_confidence == pytest.approx(0.3)
    assert classification.environment == Environment.UNKNOWN
    assert classification.blast_radius == BlastRadius.UNKNOWN


# ---------------------------------------------------------------------------
# Low-confidence routing: Fireworks assist and fallbacks
# ---------------------------------------------------------------------------


def test_low_confidence_routing_uses_fireworks(db, monkeypatch):
    calls = {}

    def fake_complete_json(self, prompt, **kwargs):  # noqa: ANN001, ANN003
        calls["prompt"] = prompt
        return {
            "team": "security-team",
            "reasoning": "The error mentions a rejected session credential.",
        }

    monkeypatch.setattr(FireworksClient, "complete_json", fake_complete_json)

    classifier = IncidentClassifier(db, ORG_CONFIG)
    router = IncidentRouter(db, ORG_CONFIG, FireworksClient())

    context = RichContext(raw_error_text="Something completely mysterious happened overnight.")
    classification = classifier.classify(context)
    assert classification.classification_confidence < 0.65

    decision = router.route(classification, context)

    # Fireworks was consulted with the team/service list plus error details.
    assert "prompt" in calls
    assert "security-team" in calls["prompt"]
    assert "payment-service" in calls["prompt"]
    assert "mysterious" in calls["prompt"]

    assert decision.owning_team == "security-team"
    assert decision.routing_confidence == pytest.approx(0.6)
    assert "Fireworks" in decision.routing_reasoning
    assert "rejected session credential" in decision.routing_reasoning


def test_low_confidence_fireworks_empty_falls_back_to_default_team(db, monkeypatch):
    """complete_json returning {} (e.g. no API key) must degrade cleanly."""
    monkeypatch.setattr(FireworksClient, "complete_json", lambda self, prompt, **kw: {})

    classifier = IncidentClassifier(db, ORG_CONFIG)
    router = IncidentRouter(db, ORG_CONFIG, FireworksClient())

    context = RichContext(raw_error_text="Something completely mysterious happened overnight.")
    classification = classifier.classify(context)
    decision = router.route(classification, context)

    # No keyword hits anywhere: default team fallback.
    assert decision.owning_team == "backend-team"
    assert decision.routing_confidence == pytest.approx(0.4)
    assert "fell back" in decision.routing_reasoning


def test_low_confidence_fallback_uses_best_keyword_scored_service(db, monkeypatch):
    """With {} from Fireworks, a weak keyword hit still steers the fallback."""
    monkeypatch.setattr(FireworksClient, "complete_json", lambda self, prompt, **kw: {})

    classifier = IncidentClassifier(db, ORG_CONFIG)
    router = IncidentRouter(db, ORG_CONFIG, FireworksClient())

    # "embedding" is a recommendation-engine keyword: score 1 -> confidence
    # 0.55, below the 0.65 threshold, so the router still asks Fireworks.
    context = RichContext(raw_error_text="nightly embedding batch job crashed with exit 137")
    classification = classifier.classify(context)
    assert classification.detected_service == "recommendation-engine"
    assert classification.classification_confidence == pytest.approx(0.55)

    decision = router.route(classification, context)
    assert decision.owning_team == "ml-platform-team"
    assert decision.routing_confidence == pytest.approx(0.4)
    assert "fell back" in decision.routing_reasoning


def test_low_confidence_invalid_fireworks_team_falls_back(db, monkeypatch):
    monkeypatch.setattr(
        FireworksClient,
        "complete_json",
        lambda self, prompt, **kw: {"team": "nonexistent-team", "reasoning": "made up"},
    )

    classifier = IncidentClassifier(db, ORG_CONFIG)
    router = IncidentRouter(db, ORG_CONFIG, FireworksClient())

    context = RichContext(raw_error_text="Something completely mysterious happened overnight.")
    classification = classifier.classify(context)
    decision = router.route(classification, context)

    assert decision.owning_team == "backend-team"
    assert decision.routing_confidence == pytest.approx(0.4)


def test_fireworks_exception_never_crashes_the_router(db, monkeypatch):
    def boom(self, prompt, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("network down")

    monkeypatch.setattr(FireworksClient, "complete_json", boom)

    classifier = IncidentClassifier(db, ORG_CONFIG)
    router = IncidentRouter(db, ORG_CONFIG, FireworksClient())

    context = RichContext(raw_error_text="Something completely mysterious happened overnight.")
    classification = classifier.classify(context)
    decision = router.route(classification, context)

    assert decision.owning_team == "backend-team"
    assert decision.routing_confidence == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Manager digest via incident burst in the last hour
# ---------------------------------------------------------------------------


def test_manager_digest_triggered_by_service_incident_burst(db, classifier, router):
    """Three recent incidents for the same service trip the digest threshold."""
    from core.schemas import IncidentIntelligenceRecord, StackTraceLocation

    for i in range(3):
        db.insert_intelligence_record(
            IncidentIntelligenceRecord(
                incident_id=f"INC-BURST-{i}",
                timestamp=datetime.now() - timedelta(minutes=10 * (i + 1)),
                service="recommendation-engine",
                failure_type=FailureType.TIMEOUT,
                environment=Environment.STAGING,
                is_critical_path=False,
                affected_users=10,
                priority=Priority.P3,
                stack_trace_locations=[
                    StackTraceLocation(
                        file_path="ml_service/recommender.py",
                        line_number=49,
                        function_name="_call_inference",
                    )
                ],
                routing_confidence=0.9,
                owning_team="ml-platform-team",
                agent3_summary="Recommendation inference timed out on staging.",
            )
        )

    context = RichContext(
        raw_error_text=read_sample("ml_timeout_error.txt"),
        environment=Environment.STAGING,
    )
    classification = classifier.classify(context)
    decision = router.route(classification, context)

    assert decision.manager_digest_triggered is True
    assert "incidents in the last hour" in decision.routing_reasoning
    assert [s.name for s in stakeholder_by_role(decision, "manager")] == ["Sofia Reyes"]


# ---------------------------------------------------------------------------
# Probable cause window
# ---------------------------------------------------------------------------


def test_deployment_older_than_60_minutes_is_not_probable_cause(db, classifier, router):
    context = RichContext(
        raw_error_text=read_sample("payment_error.txt"),
        environment=Environment.PRODUCTION,
        recent_deployment=DeploymentInfo(
            deployer="Dan Okafor",
            commit_hash="abc1234",
            commit_message="Bump dependencies",
            deployed_at=datetime.now() - timedelta(minutes=95),
        ),
    )
    classification = classifier.classify(context)
    decision = router.route(classification, context)
    assert decision.probable_cause is None
    assert "No deployment within 60 minutes" in decision.routing_reasoning
