"""Agent 2: The Router.

Rule-based routing with one narrow LLM escape hatch: Fireworks is consulted
only when the classification confidence from Agent 1 is below 0.65, meaning
the service could not be identified reliably. In every other case the
routing is entirely deterministic and every decision is explained in the
routing_reasoning field so a judge can verify it.

Takes a ClassificationResult plus the original RichContext and returns a
RoutingDecision, written to the silver_routing_decisions table before
returning.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from agents.classifier import LOW_CONFIDENCE_THRESHOLD, score_services
from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.schemas import (
    BlastRadius,
    ClassificationResult,
    Environment,
    NotificationChannel,
    OutputType,
    Priority,
    ProbableCause,
    RichContext,
    RoutingDecision,
    Stakeholder,
)

logger = logging.getLogger(__name__)

FALLBACK_TEAM = "backend-team"
FALLBACK_JIRA_PROJECT_KEY = "GEN"

ORG_LOOKUP_CONFIDENCE_CAP = 0.98
FIREWORKS_ASSISTED_CONFIDENCE = 0.6
FALLBACK_CONFIDENCE = 0.4

DEPLOYMENT_WINDOW_MINUTES = 60
CROSS_FUNCTIONAL_BLAST_RADII = (BlastRadius.ALL_USERS, BlastRadius.ENTERPRISE_CUSTOMERS)


def _to_naive(dt: datetime) -> datetime:
    """Normalize a datetime to naive local time so comparisons are safe."""
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


class IncidentRouter:
    """Agent 2: deterministic team/priority/stakeholder routing."""

    def __init__(self, db: DatabaseManager, org_config: dict, fireworks: FireworksClient) -> None:
        self.db = db
        self.org_config = org_config
        self.fireworks = fireworks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, classification: ClassificationResult, context: RichContext) -> RoutingDecision:
        incident_id = classification.incident_id
        logger.info("Agent 2 (Router) starting for incident %s", incident_id)

        team, jira_project_key, routing_confidence, team_reasoning = self._determine_team(
            classification, context
        )
        priority, priority_reasoning = self._determine_priority(classification, context)
        digest_triggered, digest_reasoning = self._manager_digest(
            classification, context, priority
        )
        probable_cause = self._detect_probable_cause(context, classification.timestamp)
        stakeholders, assignee = self._assemble_stakeholders(
            classification, context, team, digest_triggered
        )

        reasoning_parts = [
            team_reasoning,
            f"Priority {priority.value} assigned because: {priority_reasoning}.",
            digest_reasoning,
        ]
        if probable_cause is not None:
            reasoning_parts.append(
                f"Probable cause: deployment {probable_cause.commit_hash} by "
                f"{probable_cause.deployer} ({probable_cause.commit_message}) went out "
                f"{probable_cause.minutes_before_incident} minutes before the incident "
                f"(confidence {probable_cause.confidence:.2f})."
            )
        else:
            reasoning_parts.append(
                "No deployment within 60 minutes before the incident, so no probable "
                "cause was linked."
            )
        routing_reasoning = " ".join(part for part in reasoning_parts if part)

        decision = RoutingDecision(
            classification=classification,
            owning_team=team,
            assignee=assignee,
            priority=priority,
            priority_reasoning=priority_reasoning,
            jira_project_key=jira_project_key,
            stakeholders=stakeholders,
            manager_digest_triggered=digest_triggered,
            probable_cause=probable_cause,
            related_tickets=list(context.related_ticket_ids or []),
            runbook_url=context.runbook_url,
            routing_confidence=routing_confidence,
            routing_reasoning=routing_reasoning,
            missing_context=list(classification.missing_context),
        )

        self.db.insert_routing_decision(decision)
        logger.info(
            "Agent 2 (Router) completed for incident %s: team=%s, priority=%s, "
            "digest=%s, confidence=%.2f",
            incident_id,
            team,
            priority.value,
            digest_triggered,
            routing_confidence,
        )
        return decision

    # ------------------------------------------------------------------
    # Team determination
    # ------------------------------------------------------------------

    def _determine_team(
        self, classification: ClassificationResult, context: RichContext
    ) -> tuple[str, str, float, str]:
        """Return (team, jira_project_key, routing_confidence, reasoning)."""
        services = self.org_config.get("services", {})
        service = classification.detected_service
        confidence = classification.classification_confidence

        if confidence >= LOW_CONFIDENCE_THRESHOLD and service in services:
            team = services[service]["owning_team"]
            jira_key = services[service].get("jira_project_key", FALLBACK_JIRA_PROJECT_KEY)
            routing_confidence = min(confidence, ORG_LOOKUP_CONFIDENCE_CAP)
            reasoning = (
                f"Service '{service}' mapped to team '{team}' via org config lookup "
                f"(classification confidence {confidence:.2f})."
            )
            return team, jira_key, routing_confidence, reasoning

        logger.warning(
            "Classification confidence %.2f is below %.2f for incident %s; "
            "consulting Fireworks for team determination",
            confidence,
            LOW_CONFIDENCE_THRESHOLD,
            classification.incident_id,
        )
        return self._fireworks_team(classification, context)

    def _fireworks_team(
        self, classification: ClassificationResult, context: RichContext
    ) -> tuple[str, str, float, str]:
        services = self.org_config.get("services", {})
        teams = self.org_config.get("teams", {})

        service_summary = {
            name: {
                "owning_team": cfg.get("owning_team"),
                "keywords": cfg.get("keywords", []),
            }
            for name, cfg in services.items()
        }
        prompt = (
            "You are helping route a software incident to its owning team.\n"
            f"Available teams: {json.dumps(sorted(teams))}\n"
            f"Services and the teams that own them: {json.dumps(service_summary)}\n\n"
            "Error details:\n"
            f"Failure type: {classification.failure_type.value}\n"
            f"Environment: {classification.environment.value}\n"
            f"Raw error text:\n{context.raw_error_text[:2000]}\n\n"
            "Choose the single most likely owning team from the available teams. "
            'Respond as JSON: {"team": "<team name>", "reasoning": "<one sentence '
            'citing specific evidence from the error text>"}'
        )

        response: dict = {}
        try:
            response = self.fireworks.complete_json(prompt) or {}
        except Exception:  # noqa: BLE001 - external call must never crash the pipeline
            logger.error("Fireworks team determination failed", exc_info=True)
            response = {}

        candidate = response.get("team")
        if isinstance(candidate, str) and candidate in teams:
            reasoning_text = str(response.get("reasoning", "")).strip()
            reasoning = (
                f"Service could not be identified reliably (classification confidence "
                f"{classification.classification_confidence:.2f}), so Fireworks was "
                f"consulted and selected team '{candidate}'. Fireworks reasoning: "
                f"{reasoning_text or 'none provided'}"
            )
            jira_key = self._project_key_for_team(candidate)
            return candidate, jira_key, FIREWORKS_ASSISTED_CONFIDENCE, reasoning

        if candidate:
            logger.warning(
                "Fireworks returned team '%s' which does not exist in the org config; "
                "falling back to keyword scoring",
                candidate,
            )
        else:
            logger.warning(
                "Fireworks returned no usable team; falling back to keyword scoring"
            )

        best_service, best_score = score_services(context.raw_error_text, services)
        if best_service is not None and best_score > 0:
            team = services[best_service]["owning_team"]
            jira_key = services[best_service].get("jira_project_key", FALLBACK_JIRA_PROJECT_KEY)
            reasoning = (
                f"Service could not be identified reliably and Fireworks did not return "
                f"a valid team, so the router fell back to the best keyword-scored "
                f"service '{best_service}' (score {best_score}) owned by team '{team}'."
            )
            return team, jira_key, FALLBACK_CONFIDENCE, reasoning

        reasoning = (
            "Service could not be identified reliably, Fireworks did not return a "
            f"valid team, and no service keywords matched, so the router fell back to "
            f"the default team '{FALLBACK_TEAM}'."
        )
        return (
            FALLBACK_TEAM,
            self._project_key_for_team(FALLBACK_TEAM),
            FALLBACK_CONFIDENCE,
            reasoning,
        )

    def _project_key_for_team(self, team: str) -> str:
        for service_cfg in self.org_config.get("services", {}).values():
            if service_cfg.get("owning_team") == team:
                return service_cfg.get("jira_project_key", FALLBACK_JIRA_PROJECT_KEY)
        return FALLBACK_JIRA_PROJECT_KEY

    # ------------------------------------------------------------------
    # Priority rules (applied strictly in order)
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_priority(
        classification: ClassificationResult, context: RichContext
    ) -> tuple[Priority, str]:
        critical = classification.is_critical_path
        environment = classification.environment
        users = context.affected_users
        sla = context.sla_breach_minutes

        if critical and environment == Environment.PRODUCTION:
            return Priority.P1, "critical_path service AND production environment"
        if users is not None and users > 500:
            return Priority.P1, f"affected_users ({users}) > 500"
        if sla is not None and sla < 60 and critical:
            return Priority.P1, f"sla_breach_minutes ({sla}) < 60 AND critical_path service"
        if environment == Environment.PRODUCTION and not critical:
            return Priority.P2, "production environment AND NOT critical_path service"
        if environment == Environment.STAGING and critical:
            return Priority.P2, "staging environment AND critical_path service"
        if environment == Environment.STAGING and users is not None and users >= 50:
            return Priority.P2, (
                f"staging incident with material user impact (affected_users {users} >= 50)"
            )
        return Priority.P3, "no P1/P2 rule matched (default)"

    # ------------------------------------------------------------------
    # Manager digest threshold
    # ------------------------------------------------------------------

    def _manager_digest(
        self,
        classification: ClassificationResult,
        context: RichContext,
        priority: Priority,
    ) -> tuple[bool, str]:
        thresholds = self.org_config.get("manager_digest_thresholds", {})
        users_threshold = int(thresholds.get("affected_users_threshold", 500))
        per_hour_threshold = int(thresholds.get("same_service_incidents_per_hour", 3))

        reasons: list[str] = []
        if priority == Priority.P1:
            reasons.append("priority is P1")
        if context.affected_users is not None and context.affected_users >= users_threshold:
            reasons.append(
                f"affected_users ({context.affected_users}) >= {users_threshold}"
            )
        recent_incidents = self.db.count_service_incidents_last_hour(
            classification.detected_service
        )
        if recent_incidents >= per_hour_threshold:
            reasons.append(
                f"service '{classification.detected_service}' had {recent_incidents} "
                f"incidents in the last hour (>= {per_hour_threshold})"
            )

        if reasons:
            return True, f"Manager digest triggered because {'; '.join(reasons)}."
        return False, "Manager digest not triggered (no digest threshold met)."

    # ------------------------------------------------------------------
    # Stakeholder assembly
    # ------------------------------------------------------------------

    def _assemble_stakeholders(
        self,
        classification: ClassificationResult,
        context: RichContext,
        team: str,
        digest_triggered: bool,
    ) -> tuple[list[Stakeholder], str]:
        """Return the stakeholder list and the primary assignee name."""
        teams = self.org_config.get("teams", {})
        team_cfg = teams.get(team, {})

        assignee = context.on_call_engineer or team_cfg.get("default_jira_assignee", "unassigned")
        stakeholders: list[Stakeholder] = [
            Stakeholder(
                name=assignee,
                role="assignee",
                notify_via=NotificationChannel.JIRA,
                output_type=OutputType.FULL_TICKET,
            )
        ]

        lead = team_cfg.get("lead")
        if lead:
            stakeholders.append(
                Stakeholder(
                    name=lead,
                    role="team_lead",
                    notify_via=NotificationChannel.SLACK,
                    output_type=OutputType.FULL_TICKET,
                )
            )

        manager = team_cfg.get("manager")
        if digest_triggered and manager:
            stakeholders.append(
                Stakeholder(
                    name=manager,
                    role="manager",
                    notify_via=NotificationChannel.SLACK,
                    output_type=OutputType.MANAGER_DIGEST,
                )
            )

        if classification.blast_radius in CROSS_FUNCTIONAL_BLAST_RADII:
            fyi_team = FALLBACK_TEAM if team == "infra-team" else "infra-team"
            fyi_lead = teams.get(fyi_team, {}).get("lead")
            if fyi_lead:
                stakeholders.append(
                    Stakeholder(
                        name=fyi_lead,
                        role="cross_functional_fyi",
                        notify_via=NotificationChannel.SLACK,
                        output_type=OutputType.CROSS_FUNCTIONAL_FYI,
                    )
                )

        return stakeholders, assignee

    # ------------------------------------------------------------------
    # Probable cause (deployment linking)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_probable_cause(
        context: RichContext, incident_timestamp: datetime
    ) -> Optional[ProbableCause]:
        deployment = context.recent_deployment
        if deployment is None:
            return None

        deployed_at = _to_naive(deployment.deployed_at)
        incident_at = _to_naive(incident_timestamp)
        gap_minutes = (incident_at - deployed_at).total_seconds() / 60.0
        if gap_minutes < 0 or gap_minutes > DEPLOYMENT_WINDOW_MINUTES:
            logger.info(
                "Deployment %s was %.0f minutes before the incident (outside the "
                "%d-minute window); not linking as probable cause",
                deployment.commit_hash,
                gap_minutes,
                DEPLOYMENT_WINDOW_MINUTES,
            )
            return None

        # Confidence decays linearly with the gap: ~0.9 at 10 minutes,
        # ~0.55 at 55 minutes, floored at 0.5.
        confidence = max(0.5, 1.0 - gap_minutes / 120.0)
        minutes = int(round(gap_minutes))
        description = (
            f"Deployment {deployment.commit_hash} by {deployment.deployer} "
            f'("{deployment.commit_message}") went out {minutes} minutes before '
            f"the incident and is the probable cause."
        )
        return ProbableCause(
            description=description,
            commit_hash=deployment.commit_hash,
            deployer=deployment.deployer,
            commit_message=deployment.commit_message,
            minutes_before_incident=minutes,
            confidence=confidence,
        )
