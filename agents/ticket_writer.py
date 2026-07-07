"""Agent 3: The Ticket Writer.

Takes a RoutingDecision and the original RichContext and produces one piece
of content per stakeholder: a full engineering ticket, a manager digest, or
a cross-functional FYI. Fireworks is used ONLY for the prose parts (the
Summary narrative, the manager digest sentences, and the FYI paragraph) and
is given strictly the verified facts from the routing decision. Every AI
response is validated before acceptance: any number or file-like token in
the AI text that does not already appear in the input facts causes the text
to be discarded in favour of a deterministic template. All list-like
sections (Investigate First, Evidence, Blast Radius, Routing Note) are
always deterministic templates and never touch the AI.

Nothing in any output is invented. Every claim traces back to a verified
input data point.
"""

import logging
import re
from typing import Optional

from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.schemas import (
    IncidentIntelligenceRecord,
    NotificationChannel,
    NotificationRecord,
    OutputType,
    RichContext,
    RoutingDecision,
    Stakeholder,
    TicketContent,
    TicketWriterOutput,
)
from integrations.jira_client import JiraClient
from integrations.slack_client import SlackClient

logger = logging.getLogger(__name__)

FALLBACK_PREFIX = (
    "[AI assembly unavailable — deterministic summary; manual review recommended]"
)
NO_STACK_TRACE_MESSAGE = (
    "Investigate service logs directly — no stack trace was extracted."
)
NO_PROBABLE_CAUSE_MESSAGE = "No probable cause identified."
NO_ACTION_SENTENCE = "No action required unless escalated."

MAX_DIGEST_SENTENCES = 5

# --- hallucination validation patterns -------------------------------------
_NUMBER_RE = re.compile(r"\d+")
_CODE_FILE_RE = re.compile(
    r"[\w\-./\\]*\.(?:py|pyc|java|js|jsx|ts|tsx|go|rb|rs|c|cpp|h|hpp|json|ya?ml|sql|sh|log)\b"
)
_PATH_RE = re.compile(r"[\w\-.]+[/\\][\w\-./\\]+")

# --- manager digest sanitization patterns ----------------------------------
_PY_TOKEN_RE = re.compile(r"[\w\-./\\]*\.py\b[\w\-./\\:]*")
_LINE_REF_RE = re.compile(r":\d+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _human(value: str) -> str:
    """Human-case an enum-ish token: 'null_pointer' -> 'Null Pointer'."""
    return value.replace("_", " ").title()


class TicketWriter:
    """Agent 3: assembles and (optionally) sends all incident outputs."""

    def __init__(
        self,
        db: DatabaseManager,
        org_config: dict,
        fireworks: FireworksClient,
        jira: JiraClient,
        slack: SlackClient,
    ) -> None:
        self.db = db
        self.org_config = org_config
        self.fireworks = fireworks
        self.jira = jira
        self.slack = slack

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_tickets(
        self,
        decision: RoutingDecision,
        context: RichContext,
        send: bool = False,
    ) -> TicketWriterOutput:
        """Produce every stakeholder output and persist all records.

        With ``send=False`` (the preview flow) Jira and Slack are skipped
        entirely. With ``send=True`` the Jira ticket creation and every
        Slack notification are attempted, each wrapped in try/except, and
        every attempt is recorded in the notification log. The intelligence
        record and the gold ticket row are ALWAYS written regardless of
        send mode or external failures.
        """
        incident_id = decision.classification.incident_id
        logger.info(
            "TicketWriter starting for incident %s (send=%s, %d stakeholders)",
            incident_id,
            send,
            len(decision.stakeholders),
        )

        prose_cache: dict[str, str] = {}
        contents = [
            self._build_content(stakeholder, decision, context, prose_cache)
            for stakeholder in decision.stakeholders
        ]

        jira_ticket_id: Optional[str] = None
        jira_url: Optional[str] = None
        notifications: list[NotificationRecord] = []

        if send:
            full = next(
                (c for c in contents if c.output_type == OutputType.FULL_TICKET), None
            )
            if full is not None:
                title, body = full.title, full.body
            else:
                title = self._build_title(decision)
                body = self._full_ticket_body(decision, context, prose_cache)
            labels = [
                decision.classification.detected_service,
                decision.classification.failure_type.value,
                "routezero",
            ]
            jira_ticket_id, jira_url, jira_record = self._create_jira_issue(
                incident_id=incident_id,
                project_key=decision.jira_project_key,
                summary=title,
                description=body,
                priority=decision.priority.value,
                labels=labels,
                assignee=decision.assignee,
            )
            notifications.append(jira_record)
            self.db.insert_notification_log(incident_id, jira_record)

            channel = self._team_channel(decision.owning_team)
            for content in contents:
                record = self._send_slack_notification(
                    incident_id=incident_id,
                    recipient=content.recipient_name,
                    output_type=content.output_type,
                    title=content.title,
                    body=content.body,
                    channel=channel,
                )
                notifications.append(record)
                self.db.insert_notification_log(incident_id, record)
        else:
            logger.info(
                "Preview flow (send=False): skipping Jira and Slack for incident %s",
                incident_id,
            )

        intelligence = self._build_intelligence_record(decision, context, jira_ticket_id)
        self.db.insert_intelligence_record(intelligence)

        output = TicketWriterOutput(
            incident_id=incident_id,
            routing_decision=decision,
            ticket_contents=contents,
            jira_ticket_id=jira_ticket_id,
            jira_url=jira_url,
            notifications_sent=notifications,
            intelligence_record=intelligence,
        )
        self.db.insert_created_ticket(output)
        logger.info(
            "TicketWriter completed for incident %s: %d contents, jira_ticket=%s, %d notifications",
            incident_id,
            len(contents),
            jira_ticket_id,
            len(notifications),
        )
        return output

    def approve_and_send(self, incident_id: str) -> dict:
        """Create the real Jira ticket and Slack notifications for a stored preview.

        Loads the stored TicketWriterOutput (raises ValueError if missing),
        creates the Jira ticket from the stored full ticket content, sends
        Slack to every stored recipient on the owning team's channel, logs
        every attempt, and persists the Jira id/url back onto the stored
        rows. All external calls are wrapped; this method never crashes on
        integration failure.
        """
        logger.info("TicketWriter approve_and_send starting for incident %s", incident_id)
        stored = self.db.get_ticket_output(incident_id)
        if stored is None:
            raise ValueError(f"No stored ticket output found for incident {incident_id}")

        routing = self.db.get_routing_decision(incident_id)
        if routing is None:
            logger.warning(
                "approve_and_send: no stored routing decision for incident %s; "
                "Jira creation and channel lookup will be degraded",
                incident_id,
            )

        contents: list[dict] = stored.get("ticket_contents") or []
        full = next(
            (c for c in contents if c.get("output_type") == OutputType.FULL_TICKET.value),
            contents[0] if contents else None,
        )

        jira_ticket_id: Optional[str] = None
        jira_url: Optional[str] = None
        notifications: list[NotificationRecord] = []

        if routing is not None and full is not None:
            classification = routing.get("classification") or {}
            labels = [
                label
                for label in (
                    classification.get("detected_service"),
                    classification.get("failure_type"),
                    "routezero",
                )
                if label
            ]
            jira_ticket_id, jira_url, jira_record = self._create_jira_issue(
                incident_id=incident_id,
                project_key=routing.get("jira_project_key") or "",
                summary=full.get("title") or "",
                description=full.get("body") or "",
                priority=routing.get("priority") or "P3",
                labels=labels,
                assignee=routing.get("assignee") or "",
            )
            notifications.append(jira_record)
            self.db.insert_notification_log(incident_id, jira_record)
        else:
            logger.warning(
                "approve_and_send: skipping Jira creation for incident %s "
                "(routing decision or full ticket content missing)",
                incident_id,
            )

        channel = self._team_channel(routing.get("owning_team")) if routing else None
        for content in contents:
            record = self._send_slack_notification(
                incident_id=incident_id,
                recipient=content.get("recipient_name") or "unknown",
                output_type=OutputType(content.get("output_type") or "full_ticket"),
                title=content.get("title") or "",
                body=content.get("body") or "",
                channel=channel,
            )
            notifications.append(record)
            self.db.insert_notification_log(incident_id, record)

        if jira_ticket_id:
            self.db.update_ticket_jira_info(incident_id, jira_ticket_id, jira_url)
            self.db.update_intelligence_jira_ref(incident_id, jira_ticket_id)

        logger.info(
            "TicketWriter approve_and_send completed for incident %s: jira_ticket=%s, %d notifications",
            incident_id,
            jira_ticket_id,
            len(notifications),
        )
        return {
            "jira_ticket_id": jira_ticket_id,
            "jira_url": jira_url,
            "notifications": [
                {
                    "recipient": record.recipient,
                    "channel": record.channel,
                    "output_type": record.output_type.value,
                    "success": record.success,
                }
                for record in notifications
            ],
        }

    # ------------------------------------------------------------------
    # Content assembly
    # ------------------------------------------------------------------

    def _build_content(
        self,
        stakeholder: Stakeholder,
        decision: RoutingDecision,
        context: RichContext,
        prose_cache: dict[str, str],
    ) -> TicketContent:
        title = self._build_title(decision)
        if stakeholder.output_type == OutputType.MANAGER_DIGEST:
            body = self._manager_digest_body(decision, context, prose_cache)
        elif stakeholder.output_type == OutputType.CROSS_FUNCTIONAL_FYI:
            body = self._fyi_body(decision, prose_cache)
        else:
            if stakeholder.role == "team_lead":
                body = self._team_lead_ticket_body(decision, context, prose_cache)
            else:
                body = self._full_ticket_body(decision, context, prose_cache)
        return TicketContent(
            recipient_name=stakeholder.name,
            recipient_role=stakeholder.role,
            output_type=stakeholder.output_type,
            title=title,
            body=body,
        )

    @staticmethod
    def _build_title(decision: RoutingDecision) -> str:
        classification = decision.classification
        return (
            f"[{decision.priority.value}] {classification.detected_service}: "
            f"{_human(classification.failure_type.value)} in "
            f"{_human(classification.environment.value)}"
        )

    # --- full engineering ticket ---------------------------------------

    def _full_ticket_body(
        self,
        decision: RoutingDecision,
        context: RichContext,
        prose_cache: dict[str, str],
    ) -> str:
        summary = prose_cache.get("summary")
        if summary is None:
            summary = self._prose_or_fallback(
                kind="summary",
                fact_lines=self._summary_facts(decision, context),
                instruction=(
                    "You are drafting the Summary section of an engineering incident "
                    "ticket. Using ONLY the facts listed below, write two to four "
                    "plain sentences describing what broke and its impact. Cite the "
                    "affected user count and SLA risk only if they appear in the "
                    "facts. Do not invent or estimate any number, file name, "
                    "service, or detail that is not listed."
                ),
                fallback=self._summary_fallback(decision, context),
            )
            prose_cache["summary"] = summary

        sections = [
            f"## Summary\n{summary}",
            f"## Probable Cause\n{self._probable_cause_section(decision)}",
            f"## Investigate First\n{self._investigate_first_section(decision)}",
            f"## Evidence\n{self._evidence_section(decision)}",
            f"## Blast Radius\n{self._blast_radius_section(decision, context)}",
            f"## Routing Note\n{self._routing_note_section(decision)}",
        ]
        return "\n\n".join(sections)
    
    def _team_lead_ticket_body(
        self,
        decision: RoutingDecision,
        context: RichContext,
        prose_cache: dict[str, str],
    ) -> str:
        summary = prose_cache.get("summary") or self._summary_fallback(decision, context)
        sections = [
        f"## Summary\n{summary}",
        f"## Business Impact\n{self._blast_radius_section(decision, context)}",
        f"## Probable Cause\n{self._probable_cause_section(decision)}",
        f"## Assigned To\n{decision.assignee} ({decision.owning_team})",
        f"## Routing Note\n{self._routing_note_section(decision)}",
        ]
        return "\n\n".join(sections)

    def _summary_facts(self, decision: RoutingDecision, context: RichContext) -> list[str]:
        classification = decision.classification
        facts = [
            f"service: {classification.detected_service}",
            f"failure type: {_human(classification.failure_type.value).lower()}",
            f"environment: {classification.environment.value}",
            f"priority: {decision.priority.value}",
            f"critical path service: {'yes' if classification.is_critical_path else 'no'}",
            f"occurrences in the last 4 hours: {classification.occurrences_last_4h}",
        ]
        if context.affected_users is not None:
            facts.append(f"affected users: {context.affected_users}")
        if context.customer_tier:
            facts.append(f"customer tier: {context.customer_tier}")
        if context.sla_breach_minutes is not None:
            facts.append(f"minutes until SLA breach: {context.sla_breach_minutes}")
        if decision.probable_cause:
            pc = decision.probable_cause
            facts.append(
                f"probable cause: deployment {pc.commit_hash} by {pc.deployer} "
                f"({pc.minutes_before_incident} minutes before the incident, "
                f"confidence {int(round(pc.confidence * 100))}%)"
            )
        return facts

    @staticmethod
    def _summary_fallback(decision: RoutingDecision, context: RichContext) -> str:
        classification = decision.classification
        sentences = [
            f"{classification.detected_service} experienced a "
            f"{_human(classification.failure_type.value).lower()} failure in "
            f"{classification.environment.value}."
        ]
        if context.affected_users is not None:
            if context.customer_tier:
                sentences.append(
                    f"{context.affected_users} users ({context.customer_tier} tier) are affected."
                )
            else:
                sentences.append(f"{context.affected_users} users are affected.")
        if classification.is_critical_path:
            sentences.append("The service is on the critical path.")
        if context.sla_breach_minutes is not None:
            sentences.append(
                f"SLA breach risk in {context.sla_breach_minutes} minutes."
            )
        return " ".join(sentences)

    @staticmethod
    def _probable_cause_section(decision: RoutingDecision) -> str:
        pc = decision.probable_cause
        if pc is None:
            return NO_PROBABLE_CAUSE_MESSAGE
        return (
            f"Deployment {pc.commit_hash} by {pc.deployer} "
            f'("{pc.commit_message}") went out {pc.minutes_before_incident} minutes '
            f"before the incident. Confidence: {int(round(pc.confidence * 100))}%."
        )

    @staticmethod
    def _investigate_first_section(decision: RoutingDecision) -> str:
        locations = decision.classification.stack_trace_locations
        if not locations:
            return NO_STACK_TRACE_MESSAGE
        return "\n".join(
            f"{index}. {loc.file_path}:{loc.line_number} in {loc.function_name}"
            for index, loc in enumerate(locations[:3], start=1)
        )

    @staticmethod
    def _evidence_section(decision: RoutingDecision) -> str:
        lines: list[str] = []
        locations = decision.classification.stack_trace_locations
        if locations:
            lines.append("Stack trace locations:")
            lines.extend(
                f"- {loc.file_path}:{loc.line_number} in {loc.function_name}"
                for loc in locations
            )
        if decision.related_tickets:
            lines.append(f"Related tickets: {', '.join(decision.related_tickets)}")
        if decision.runbook_url:
            lines.append(f"Runbook: {decision.runbook_url}")
        if not lines:
            return "No additional evidence was provided."
        return "\n".join(lines)

    @staticmethod
    def _blast_radius_section(decision: RoutingDecision, context: RichContext) -> str:
        classification = decision.classification
        lines: list[str] = []
        if context.affected_users is not None:
            lines.append(f"- Affected users: {context.affected_users}")
        lines.append(f"- Environment: {classification.environment.value}")
        lines.append(
            f"- Critical path: {'yes' if classification.is_critical_path else 'no'}"
        )
        return "\n".join(lines)

    @staticmethod
    def _routing_note_section(decision: RoutingDecision) -> str:
        missing = ", ".join(decision.missing_context) if decision.missing_context else "none"
        return "\n".join(
            [
                f"- Routing confidence: {int(round(decision.routing_confidence * 100))}%",
                f"- Missing context: {missing}",
                f"- Reasoning: {decision.routing_reasoning}",
            ]
        )

    # --- manager digest --------------------------------------------------

    def _manager_digest_body(
        self,
        decision: RoutingDecision,
        context: RichContext,
        prose_cache: dict[str, str],
    ) -> str:
        cached = prose_cache.get("digest")
        if cached is not None:
            return cached

        body = self._prose_or_fallback(
            kind="manager digest",
            fact_lines=self._digest_facts(decision, context),
            instruction=(
                "You are drafting a short digest for an engineering manager. "
                "Using ONLY the facts listed below, write at most five plain "
                "sentences covering: what broke, how many users are affected, "
                "which team is handling it, the probable cause if known, and "
                "the SLA risk if any. Do not mention file names, stack traces, "
                "or any technical identifier. Do not invent or estimate any "
                "number or detail that is not listed."
            ),
            fallback=self._digest_fallback(decision, context),
        )
        body = self._strip_technical_tokens(body)
        body = self._limit_sentences(body, MAX_DIGEST_SENTENCES)
        prose_cache["digest"] = body
        return body

    @staticmethod
    def _digest_facts(decision: RoutingDecision, context: RichContext) -> list[str]:
        classification = decision.classification
        facts = [
            f"what broke: {classification.detected_service} had a "
            f"{_human(classification.failure_type.value).lower()} failure in "
            f"{classification.environment.value}",
            f"handling team: {decision.owning_team}",
            f"priority: {decision.priority.value}",
        ]
        if context.affected_users is not None:
            facts.append(f"affected users: {context.affected_users}")
        if decision.probable_cause:
            pc = decision.probable_cause
            facts.append(
                f"probable cause: a deployment {pc.minutes_before_incident} minutes "
                f"before the incident (confidence {int(round(pc.confidence * 100))}%)"
            )
        else:
            facts.append("probable cause: not identified")
        if context.sla_breach_minutes is not None:
            facts.append(f"minutes until SLA breach: {context.sla_breach_minutes}")
        return facts

    @staticmethod
    def _digest_fallback(decision: RoutingDecision, context: RichContext) -> str:
        classification = decision.classification
        sentences = [
            f"{classification.detected_service} experienced a "
            f"{_human(classification.failure_type.value).lower()} failure in "
            f"{classification.environment.value}."
        ]
        if context.affected_users is not None:
            sentences.append(f"{context.affected_users} users are affected.")
        sentences.append(
            f"{decision.owning_team} is handling the incident at priority "
            f"{decision.priority.value}."
        )
        if decision.probable_cause:
            sentences.append(
                f"The probable cause is a deployment "
                f"{decision.probable_cause.minutes_before_incident} minutes before "
                f"the incident."
            )
        if context.sla_breach_minutes is not None:
            sentences.append(
                f"SLA breach risk in {context.sla_breach_minutes} minutes."
            )
        return " ".join(sentences[:MAX_DIGEST_SENTENCES])

    # --- cross-functional FYI --------------------------------------------

    def _fyi_body(self, decision: RoutingDecision, prose_cache: dict[str, str]) -> str:
        cached = prose_cache.get("fyi")
        if cached is not None:
            return cached

        body = self._prose_or_fallback(
            kind="cross-functional FYI",
            fact_lines=self._fyi_facts(decision),
            instruction=(
                "You are drafting a brief cross-functional FYI for a neighbouring "
                "team. Using ONLY the facts listed below, write two or three plain "
                "sentences stating what the primary team is handling and what this "
                "team should monitor. End with exactly this sentence: "
                "No action required unless escalated. Do not invent or estimate "
                "any number, file name, or detail that is not listed."
            ),
            fallback=self._fyi_fallback(decision),
        )
        if NO_ACTION_SENTENCE not in body:
            body = f"{body.rstrip()} {NO_ACTION_SENTENCE}"
        prose_cache["fyi"] = body
        return body

    @staticmethod
    def _fyi_facts(decision: RoutingDecision) -> list[str]:
        classification = decision.classification
        return [
            f"primary team handling the incident: {decision.owning_team}",
            f"service: {classification.detected_service}",
            f"failure type: {_human(classification.failure_type.value).lower()}",
            f"environment: {classification.environment.value}",
            f"priority: {decision.priority.value}",
        ]

    @staticmethod
    def _fyi_fallback(decision: RoutingDecision) -> str:
        classification = decision.classification
        return (
            f"{decision.owning_team} is handling a "
            f"{_human(classification.failure_type.value).lower()} incident in "
            f"{classification.detected_service} "
            f"({classification.environment.value}, priority {decision.priority.value}). "
            f"Monitor your own dashboards and error rates for downstream impact. "
            f"{NO_ACTION_SENTENCE}"
        )

    # ------------------------------------------------------------------
    # Fireworks prose with hallucination validation
    # ------------------------------------------------------------------

    def _prose_or_fallback(
        self,
        kind: str,
        fact_lines: list[str],
        instruction: str,
        fallback: str,
    ) -> str:
        facts_text = "\n".join(f"- {line}" for line in fact_lines)
        prompt = f"{instruction}\n\nFACTS:\n{facts_text}"
        ai_text = ""
        try:
            ai_text = (self.fireworks.complete(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001 - Fireworks must never crash the pipeline
            logger.error("Fireworks call failed for %s prose: %s", kind, exc)
            ai_text = ""

        if not ai_text:
            logger.warning(
                "Fireworks returned no %s prose; using deterministic fallback.", kind
            )
            return f"{FALLBACK_PREFIX}\n{fallback}"

        if not self._ai_text_is_grounded(ai_text, facts_text):
            logger.warning(
                "Discarding %s prose from Fireworks: it contains a number or "
                "file-like token not present in the input facts.",
                kind,
            )
            return f"{FALLBACK_PREFIX}\n{fallback}"

        return ai_text

    @staticmethod
    def _ai_text_is_grounded(ai_text: str, facts_text: str) -> bool:
        """True only when every number and file-like token in the AI text
        already appears in the input facts."""
        allowed_numbers = set(_NUMBER_RE.findall(facts_text))
        for number in _NUMBER_RE.findall(ai_text):
            if number not in allowed_numbers:
                return False
        file_tokens = set(_CODE_FILE_RE.findall(ai_text)) | set(_PATH_RE.findall(ai_text))
        for token in file_tokens:
            if token not in facts_text:
                return False
        return True

    # ------------------------------------------------------------------
    # Digest sanitization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_technical_tokens(text: str) -> str:
        """Remove any .py tokens and :<line> references from manager text."""
        cleaned = _PY_TOKEN_RE.sub("", text)
        cleaned = _LINE_REF_RE.sub("", cleaned)
        return re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    @staticmethod
    def _limit_sentences(text: str, max_sentences: int) -> str:
        chunks = [c for c in _SENTENCE_SPLIT_RE.split(text.strip()) if c.strip()]
        if len(chunks) <= max_sentences:
            return text.strip()
        return " ".join(chunks[:max_sentences])

    # ------------------------------------------------------------------
    # External calls (Jira / Slack), each wrapped and recorded
    # ------------------------------------------------------------------

    def _create_jira_issue(
        self,
        incident_id: str,
        project_key: str,
        summary: str,
        description: str,
        priority: str,
        labels: list[str],
        assignee: str,
    ) -> tuple[Optional[str], Optional[str], NotificationRecord]:
        jira_ticket_id: Optional[str] = None
        jira_url: Optional[str] = None
        error: Optional[str] = None
        try:
            result = (
                self.jira.create_issue(
                    project_key=project_key,
                    summary=summary,
                    description=description,
                    priority=priority,
                    labels=labels,
                    assignee=assignee,
                )
                or {}
            )
            jira_ticket_id = result.get("ticket_id")
            jira_url = result.get("url")
            error = result.get("error")
        except Exception as exc:  # noqa: BLE001 - Jira must never crash the pipeline
            logger.error(
                "Jira issue creation raised for incident %s: %s", incident_id, exc
            )
            error = str(exc)

        success = jira_ticket_id is not None and error is None
        if not success:
            logger.warning(
                "Jira issue creation degraded for incident %s (error=%s)",
                incident_id,
                error,
            )
        record = NotificationRecord(
            recipient=assignee or "unassigned",
            channel=NotificationChannel.JIRA.value,
            output_type=OutputType.FULL_TICKET,
            success=success,
            error_message=None if success else (error or "Jira ticket was not created"),
        )
        return jira_ticket_id, jira_url, record

    def _send_slack_notification(
        self,
        incident_id: str,
        recipient: str,
        output_type: OutputType,
        title: str,
        body: str,
        channel: Optional[str],
    ) -> NotificationRecord:
        success = False
        error: Optional[str] = None
        if channel:
            try:
                success = bool(self.slack.send(channel, f"{title}\n\n{body}"))
                if not success:
                    error = "Slack client reported send failure"
            except Exception as exc:  # noqa: BLE001 - Slack must never crash the pipeline
                logger.error(
                    "Slack notification raised for incident %s (recipient %s): %s",
                    incident_id,
                    recipient,
                    exc,
                )
                error = str(exc)
        else:
            error = "No Slack channel configured for the owning team"
            logger.warning(
                "Slack notification skipped for incident %s (recipient %s): %s",
                incident_id,
                recipient,
                error,
            )
        return NotificationRecord(
            recipient=recipient,
            channel=channel or NotificationChannel.SLACK.value,
            output_type=output_type,
            success=success,
            error_message=None if success else error,
        )

    def _team_channel(self, team: Optional[str]) -> Optional[str]:
        if not team:
            return None
        team_config = (self.org_config.get("teams") or {}).get(team) or {}
        return team_config.get("slack_channel")

    # ------------------------------------------------------------------
    # Intelligence record for Agent 4
    # ------------------------------------------------------------------

    def _build_intelligence_record(
        self,
        decision: RoutingDecision,
        context: RichContext,
        jira_ticket_id: Optional[str],
    ) -> IncidentIntelligenceRecord:
        classification = decision.classification
        pc = decision.probable_cause
        return IncidentIntelligenceRecord(
            incident_id=classification.incident_id,
            jira_ticket_ref=jira_ticket_id,
            timestamp=classification.timestamp,
            service=classification.detected_service,
            failure_type=classification.failure_type,
            environment=classification.environment,
            is_critical_path=classification.is_critical_path,
            affected_users=context.affected_users,
            priority=decision.priority,
            stack_trace_locations=classification.stack_trace_locations,
            probable_cause=pc.description if pc else None,
            linked_deployment=pc.commit_hash if pc else None,
            routing_confidence=decision.routing_confidence,
            owning_team=decision.owning_team,
            resolved=False,
            resolution_time_minutes=None,
            agent3_summary=self._agent3_summary(decision, context),
        )

    @staticmethod
    def _agent3_summary(decision: RoutingDecision, context: RichContext) -> str:
        classification = decision.classification
        summary = (
            f"{classification.detected_service} suffered a "
            f"{classification.failure_type.value} error in "
            f"{classification.environment.value}"
        )
        if context.affected_users is not None:
            summary += f" affecting {context.affected_users} users"
        summary += f"; routed to {decision.owning_team} as {decision.priority.value}"
        if decision.probable_cause:
            summary += (
                f" with probable cause linked to deployment "
                f"{decision.probable_cause.commit_hash}"
            )
        else:
            summary += " with no probable cause identified"
        return summary + "."
