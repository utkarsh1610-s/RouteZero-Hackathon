"""Agent 4: the Architectural Auditor.

Reads the accumulated incident intelligence records from DuckDB, finds
patterns (recurring locations, service stress, cascading failures),
traverses the code knowledge graph, and files proactive architectural
flags with PLM tickets.

Absolute rule: the auditor only makes claims about code locations that
already appeared in real incident stack traces stored in DuckDB. It never
speculates about code it has no incident evidence for. When evidence is
insufficient it stays silent rather than guessing.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.graph import CodeGraphBuilder
from core.schemas import (
    ArchitecturalFlag,
    AuditorOutput,
    GraphNode,
    PatternType,
    StackTraceLocation,
)
from integrations.github_client import GitHubClient
from integrations.jira_client import JiraClient

logger = logging.getLogger(__name__)

PLM_TICKET_CONFIDENCE_THRESHOLD = 0.70
FIREWORKS_SKIP_CONFIDENCE = 0.5
CASCADE_WINDOW_SECONDS = 30 * 60
CASCADE_MIN_PAIRS = 2

_TRACEABILITY_NOTE = (
    "All findings in this ticket are traceable to incident records stored in "
    "DuckDB (gold_incident_intelligence)."
)


def _deterministic_confidence(num_incidents: int) -> float:
    """Evidence-scaled confidence: 2 incidents -> 0.75, 3 -> 0.85, capped at 0.95."""
    return min(0.95, 0.55 + 0.10 * num_incidents)


class ArchitecturalAuditor:
    """Agent 4. Pattern detection is deterministic; Fireworks only writes
    the plain-English assessment and can veto a flag with low confidence."""

    def __init__(
        self,
        db: DatabaseManager,
        org_config: dict,
        fireworks: FireworksClient,
        jira: JiraClient,
        github: GitHubClient,
        graph: CodeGraphBuilder,
    ) -> None:
        self.db = db
        self.org_config = org_config or {}
        self.fireworks = fireworks
        self.jira = jira
        self.github = github
        self.graph = graph

    # ------------------------------------------------------------------
    # Public API (pinned interface)
    # ------------------------------------------------------------------

    def run_audit(self) -> AuditorOutput:
        logger.info("Agent 4 (ArchitecturalAuditor) starting audit")
        audit_config = self.org_config.get("architectural_audit") or {}
        lookback_days = int(audit_config.get("lookback_days", 7))
        recurring_min = int(audit_config.get("recurring_location_min_incidents", 2))
        stress_min = int(audit_config.get("service_stress_min_incidents", 3))
        plm_project_key = audit_config.get("plm_jira_project_key", "PLM")
        plm_assignee = audit_config.get("plm_assignee", "")

        incidents = self.db.get_incidents_since(lookback_days)
        logger.info(
            "Audit window: last %d days, %d incidents loaded", lookback_days, len(incidents)
        )

        try:
            self.graph.build_or_load()
            self.graph.map_incident_locations(incidents)
        except Exception:  # noqa: BLE001 - the auditor must never crash
            logger.error("Code graph build/mapping failed; continuing audit", exc_info=True)

        flags: list[ArchitecturalFlag] = []
        flags.extend(self._recurring_location_flags(incidents, recurring_min, lookback_days))
        flags.extend(self._service_stress_flags(incidents, stress_min, lookback_days))
        flags.extend(self._cascading_failure_flags(incidents, lookback_days))

        plm_ticket_ids: list[str] = []
        for flag in flags:
            if flag.confidence > PLM_TICKET_CONFIDENCE_THRESHOLD:
                ticket_id = self._create_plm_ticket(flag, plm_project_key, plm_assignee)
                if ticket_id:
                    flag.plm_ticket_id = ticket_id
                    plm_ticket_ids.append(ticket_id)
            else:
                logger.info(
                    "Flag %s/%s at %.2f confidence is below %.2f; no PLM ticket created",
                    flag.pattern_type.value,
                    flag.affected_service,
                    flag.confidence,
                    PLM_TICKET_CONFIDENCE_THRESHOLD,
                )

        now = datetime.now()
        output = AuditorOutput(
            audit_id=f"AUD-{now:%Y%m%d-%H%M%S}",
            timestamp=now,
            incidents_analyzed=len(incidents),
            patterns_found=len(flags),
            flags=flags,
            plm_tickets_created=plm_ticket_ids,
        )
        self.db.insert_audit_run(output)
        logger.info(
            "Agent 4 (ArchitecturalAuditor) complete: %d incidents analyzed, "
            "%d patterns found, %d PLM tickets created",
            output.incidents_analyzed,
            output.patterns_found,
            len(plm_ticket_ids),
        )
        return output

    # ------------------------------------------------------------------
    # Pattern 1: recurring location
    # ------------------------------------------------------------------

    def _recurring_location_flags(
        self, incidents: list[dict], recurring_min: int, lookback_days: int
    ) -> list[ArchitecturalFlag]:
        groups: dict[tuple[str, int], dict] = {}
        for incident in incidents:
            for location in incident.get("stack_trace_locations") or []:
                file_path = (location.get("file_path") or "").replace("\\", "/")
                line_number = location.get("line_number")
                if not file_path or line_number is None:
                    continue
                group = groups.setdefault(
                    (file_path, int(line_number)),
                    {"incident_ids": set(), "services": Counter(), "function_name": ""},
                )
                group["incident_ids"].add(incident.get("incident_id"))
                group["services"][incident.get("service") or "unknown"] += 1
                group["function_name"] = group["function_name"] or (
                    location.get("function_name") or ""
                )

        flags: list[ArchitecturalFlag] = []
        for (file_path, line_number), group in sorted(groups.items()):
            incident_ids = sorted(iid for iid in group["incident_ids"] if iid)
            if len(incident_ids) < recurring_min:
                continue

            num_incidents = len(incident_ids)
            confidence = _deterministic_confidence(num_incidents)
            service = group["services"].most_common(1)[0][0]
            function_name = group["function_name"]

            code_snippet = ""
            try:
                code_snippet = self.github.get_file_content_at_line(file_path, line_number, 12)
            except Exception:  # noqa: BLE001
                logger.error(
                    "GitHub code fetch failed for %s:%d", file_path, line_number, exc_info=True
                )

            node: Optional[GraphNode] = None
            connected_nodes: list[GraphNode] = []
            try:
                node = self.graph.get_node_at(file_path, line_number)
                if node is not None:
                    connected_nodes, _ = self.graph.get_subgraph(node.node_id, hops=2)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Graph lookup failed for %s:%d", file_path, line_number, exc_info=True
                )

            connected_names = sorted({n.name for n in connected_nodes if n.node_id != (node.node_id if node else None)})
            prompt = (
                "You are an architectural auditor for the StreamCo engineering "
                "organization. One exact code location keeps failing in "
                "production incidents.\n\n"
                f"Location: {file_path} line {line_number}"
                + (f" (function {function_name})" if function_name else "")
                + f"\nThis location appeared in the stack traces of {num_incidents} "
                f"separate incidents: {', '.join(incident_ids)}.\n\n"
                "Actual code at that location (the failing line is marked with '>'):\n"
                f"{code_snippet or '(code unavailable)'}\n\n"
                f"Connected code from the knowledge graph (within 2 hops): "
                f"{', '.join(connected_names) or '(none)'}\n\n"
                "Assess what structural weakness exists at this exact location. "
                "You MUST cite the actual code shown above. Only discuss the code "
                "shown; never speculate about code you have not been shown.\n\n"
                'Respond as JSON with keys: "assessment" (2-4 sentences citing the '
                'code), "confidence" (a number between 0 and 1), and '
                '"recommended_reviewers_reasoning" (one sentence).'
            )
            response = self._fireworks_json(prompt)

            fallback = self._recurring_fallback_assessment(
                num_incidents, incident_ids, file_path, line_number, lookback_days, code_snippet
            )
            assessment = self._assessment_or_skip(response, fallback)
            if assessment is None:
                logger.info(
                    "Recurring location %s:%d skipped: Fireworks assessment "
                    "confidence below %.2f (staying silent on low confidence)",
                    file_path,
                    line_number,
                    FIREWORKS_SKIP_CONFIDENCE,
                )
                continue

            flags.append(
                ArchitecturalFlag(
                    pattern_type=PatternType.RECURRING_LOCATION,
                    affected_service=service,
                    contributing_incident_ids=incident_ids,
                    flagged_locations=[
                        StackTraceLocation(
                            file_path=file_path,
                            line_number=line_number,
                            function_name=function_name,
                        )
                    ],
                    connected_nodes=connected_nodes,
                    developer_attribution=self._attribution_for_nodes([node] if node else []),
                    assessment=assessment,
                    recommended_reviewers=self._reviewers_for_services([service]),
                    confidence=confidence,
                )
            )
            logger.info(
                "Recurring location flagged: %s:%d across %d incidents (confidence %.2f)",
                file_path,
                line_number,
                num_incidents,
                confidence,
            )
        return flags

    @staticmethod
    def _recurring_fallback_assessment(
        num_incidents: int,
        incident_ids: list[str],
        file_path: str,
        line_number: int,
        lookback_days: int,
        code_snippet: str,
    ) -> str:
        # Cite the actual flagged line of code when we have it: this is
        # verified input data, not speculation.
        flagged_line = ""
        for line in (code_snippet or "").splitlines():
            if line.startswith("> "):
                flagged_line = line.split("|", 1)[-1].strip()
                break
        code_sentence = (
            f"The flagged code at {file_path}:{line_number} is: `{flagged_line}`. "
            if flagged_line
            else f"The flagged location is {file_path}:{line_number}. "
        )
        return (
            f"This exact location failed in {num_incidents} separate incidents "
            f"({', '.join(incident_ids)}) over the last {lookback_days} days. "
            + code_sentence
            + "AI assessment unavailable — flagged on incident evidence alone."
        )

    # ------------------------------------------------------------------
    # Pattern 2: service stress
    # ------------------------------------------------------------------

    def _service_stress_flags(
        self, incidents: list[dict], stress_min: int, lookback_days: int
    ) -> list[ArchitecturalFlag]:
        by_service: dict[str, list[dict]] = defaultdict(list)
        for incident in incidents:
            service = incident.get("service")
            if service:
                by_service[service].append(incident)

        flags: list[ArchitecturalFlag] = []
        for service, service_incidents in sorted(by_service.items()):
            failure_types = sorted(
                {str(incident.get("failure_type")) for incident in service_incidents}
            )
            if len(service_incidents) < stress_min:
                continue
            if len(failure_types) < 2:
                logger.info(
                    "Service %s has %d incidents but a single failure type (%s): "
                    "a repeated bug, not systemic stress. Not flagged.",
                    service,
                    len(service_incidents),
                    failure_types[0] if failure_types else "unknown",
                )
                continue

            incident_ids = sorted(
                incident.get("incident_id") for incident in service_incidents
            )
            num_incidents = len(incident_ids)
            confidence = _deterministic_confidence(num_incidents)
            locations = self._dedup_locations(service_incidents)

            prompt = (
                "You are an architectural auditor for the StreamCo engineering "
                f"organization. The service '{service}' experienced "
                f"{num_incidents} incidents in the last {lookback_days} days "
                f"across {len(failure_types)} DISTINCT failure types: "
                f"{', '.join(failure_types)}.\n"
                f"Contributing incidents: {', '.join(incident_ids)}.\n"
                f"Stack trace locations from those incidents: "
                f"{'; '.join(f'{l.file_path}:{l.line_number}' for l in locations) or '(none)'}\n\n"
                "The same error repeating is a bug; different errors recurring in "
                "one service is a systemic problem. Assess what architectural "
                "issue this pattern of different failure types suggests. You MUST "
                "cite the specific failure types listed above. Never speculate "
                "about code locations not listed.\n\n"
                'Respond as JSON with keys: "assessment" (2-4 sentences citing the '
                'failure types), "confidence" (a number between 0 and 1), and '
                '"recommended_reviewers_reasoning" (one sentence).'
            )
            response = self._fireworks_json(prompt)

            fallback = (
                f"Service {service} experienced {num_incidents} incidents "
                f"({', '.join(incident_ids)}) across {len(failure_types)} distinct "
                f"failure types ({', '.join(failure_types)}) in the last "
                f"{lookback_days} days. Different failure types recurring in one "
                "service indicate a systemic problem rather than a single bug. "
                "AI assessment unavailable — flagged on incident evidence alone."
            )
            assessment = self._assessment_or_skip(response, fallback)
            if assessment is None:
                logger.info(
                    "Service stress for %s skipped: Fireworks assessment "
                    "confidence below %.2f (staying silent on low confidence)",
                    service,
                    FIREWORKS_SKIP_CONFIDENCE,
                )
                continue

            location_nodes = self._nodes_for_locations(locations)
            flags.append(
                ArchitecturalFlag(
                    pattern_type=PatternType.SERVICE_STRESS,
                    affected_service=service,
                    contributing_incident_ids=incident_ids,
                    flagged_locations=locations,
                    connected_nodes=location_nodes,
                    developer_attribution=self._attribution_for_nodes(location_nodes),
                    assessment=assessment,
                    recommended_reviewers=self._reviewers_for_services([service]),
                    confidence=confidence,
                )
            )
            logger.info(
                "Service stress flagged: %s with %d incidents across %d failure "
                "types (confidence %.2f)",
                service,
                num_incidents,
                len(failure_types),
                confidence,
            )
        return flags

    # ------------------------------------------------------------------
    # Pattern 3: cascading failure
    # ------------------------------------------------------------------

    def _cascading_failure_flags(
        self, incidents: list[dict], lookback_days: int
    ) -> list[ArchitecturalFlag]:
        timed = sorted(
            (incident for incident in incidents if incident.get("timestamp") is not None),
            key=lambda incident: incident["timestamp"],
        )
        pairs: list[tuple[dict, dict]] = []
        for index, first in enumerate(timed):
            for second in timed[index + 1 :]:
                delta_seconds = (second["timestamp"] - first["timestamp"]).total_seconds()
                if delta_seconds > CASCADE_WINDOW_SECONDS:
                    break
                if first.get("service") != second.get("service"):
                    pairs.append((first, second))
        if len(pairs) < CASCADE_MIN_PAIRS:
            return []

        contributing: dict[str, dict] = {}
        for first, second in pairs:
            contributing[first["incident_id"]] = first
            contributing[second["incident_id"]] = second
        incident_ids = sorted(contributing)
        services = sorted({incident.get("service") for incident in contributing.values()})
        num_incidents = len(incident_ids)
        confidence = _deterministic_confidence(num_incidents)
        locations = self._dedup_locations(list(contributing.values()))

        pair_lines = [
            f"- {first['incident_id']} ({first.get('service')}) at {first['timestamp']} "
            f"followed within 30 minutes by {second['incident_id']} "
            f"({second.get('service')}) at {second['timestamp']}"
            for first, second in pairs
        ]
        prompt = (
            "You are an architectural auditor for the StreamCo engineering "
            "organization. Incidents in DIFFERENT services repeatedly occurred "
            "within 30 minutes of each other, suggesting a shared dependency or "
            "a failure mode that propagates across service boundaries.\n\n"
            f"Observed pairs ({len(pairs)}):\n" + "\n".join(pair_lines) + "\n\n"
            f"Stack trace locations from those incidents: "
            f"{'; '.join(f'{l.file_path}:{l.line_number}' for l in locations) or '(none)'}\n\n"
            "Assess what the likely shared cause is, citing only the incidents "
            "and locations listed above. Never speculate about code you have "
            "not been shown.\n\n"
            'Respond as JSON with keys: "assessment" (2-4 sentences), '
            '"confidence" (a number between 0 and 1), and '
            '"recommended_reviewers_reasoning" (one sentence).'
        )
        response = self._fireworks_json(prompt)

        fallback = (
            f"{len(pairs)} pairs of incidents in different services occurred "
            f"within 30 minutes of each other in the last {lookback_days} days "
            f"(incidents {', '.join(incident_ids)} across services "
            f"{', '.join(services)}). This repeated cross-service timing pattern "
            "suggests a shared dependency. AI assessment unavailable — flagged "
            "on incident evidence alone."
        )
        assessment = self._assessment_or_skip(response, fallback)
        if assessment is None:
            logger.info(
                "Cascading failure pattern skipped: Fireworks assessment "
                "confidence below %.2f (staying silent on low confidence)",
                FIREWORKS_SKIP_CONFIDENCE,
            )
            return []

        location_nodes = self._nodes_for_locations(locations)
        logger.info(
            "Cascading failure flagged across %s: %d pairs, %d incidents "
            "(confidence %.2f)",
            ", ".join(services),
            len(pairs),
            num_incidents,
            confidence,
        )
        return [
            ArchitecturalFlag(
                pattern_type=PatternType.CASCADING_FAILURE,
                affected_service=", ".join(services),
                contributing_incident_ids=incident_ids,
                flagged_locations=locations,
                connected_nodes=location_nodes,
                developer_attribution=self._attribution_for_nodes(location_nodes),
                assessment=assessment,
                recommended_reviewers=self._reviewers_for_services(services),
                confidence=confidence,
            )
        ]

    # ------------------------------------------------------------------
    # Fireworks helpers
    # ------------------------------------------------------------------

    def _fireworks_json(self, prompt: str) -> dict:
        try:
            response = self.fireworks.complete_json(prompt)
            return response if isinstance(response, dict) else {}
        except Exception:  # noqa: BLE001 - the auditor must never crash
            logger.error("Fireworks call failed during audit", exc_info=True)
            return {}

    @staticmethod
    def _assessment_or_skip(response: dict, fallback: str) -> Optional[str]:
        """Apply the spec's confidence policy to a Fireworks JSON response.

        Returns None when the flag must be skipped (explicit low confidence),
        the Fireworks assessment text when usable, or the evidence-only
        fallback otherwise (empty response / missing assessment).
        """
        if response:
            raw_confidence = response.get("confidence")
            if raw_confidence is not None:
                try:
                    if float(raw_confidence) < FIREWORKS_SKIP_CONFIDENCE:
                        return None
                except (TypeError, ValueError):
                    logger.warning(
                        "Unparseable Fireworks confidence %r; ignoring it", raw_confidence
                    )
            assessment = str(response.get("assessment") or "").strip()
            if assessment:
                return assessment
        logger.warning(
            "Fireworks assessment unavailable; falling back to evidence-only "
            "assessment built from verified incident data"
        )
        return fallback

    # ------------------------------------------------------------------
    # Graph and org-config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_locations(incidents: list[dict]) -> list[StackTraceLocation]:
        """Distinct stack-trace locations across incidents, order preserved."""
        seen: set[tuple[str, int]] = set()
        locations: list[StackTraceLocation] = []
        for incident in incidents:
            for location in incident.get("stack_trace_locations") or []:
                file_path = (location.get("file_path") or "").replace("\\", "/")
                line_number = location.get("line_number")
                if not file_path or line_number is None:
                    continue
                key = (file_path, int(line_number))
                if key in seen:
                    continue
                seen.add(key)
                locations.append(
                    StackTraceLocation(
                        file_path=file_path,
                        line_number=int(line_number),
                        function_name=location.get("function_name") or "",
                    )
                )
        return locations

    def _nodes_for_locations(self, locations: list[StackTraceLocation]) -> list[GraphNode]:
        nodes: dict[str, GraphNode] = {}
        for location in locations:
            try:
                node = self.graph.get_node_at(location.file_path, location.line_number)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Graph lookup failed for %s:%d",
                    location.file_path,
                    location.line_number,
                    exc_info=True,
                )
                continue
            if node is not None:
                nodes[node.node_id] = node
        return [nodes[node_id] for node_id in sorted(nodes)]

    def _attribution_for_nodes(self, nodes: list[GraphNode]) -> list[str]:
        """Owners of the flagged nodes plus their direct graph neighbors."""
        owners: set[str] = set()
        for node in nodes:
            if node is None:
                continue
            if node.owner:
                owners.add(node.owner)
            try:
                neighbor_nodes, _ = self.graph.get_subgraph(node.node_id, hops=1)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Neighbor lookup failed for %s", node.node_id, exc_info=True
                )
                continue
            owners.update(n.owner for n in neighbor_nodes if n.owner)
        return sorted(owners)

    def _reviewers_for_services(self, services: list[str]) -> list[str]:
        """Team lead + default Jira assignee of each affected service's team."""
        reviewers: list[str] = []
        all_services = self.org_config.get("services") or {}
        all_teams = self.org_config.get("teams") or {}
        for service in services:
            team_name = (all_services.get(service) or {}).get("owning_team")
            team = all_teams.get(team_name) or {}
            for person in (team.get("lead"), team.get("default_jira_assignee")):
                if person and person not in reviewers:
                    reviewers.append(person)
        return reviewers

    # ------------------------------------------------------------------
    # PLM ticket creation
    # ------------------------------------------------------------------

    def _create_plm_ticket(
        self, flag: ArchitecturalFlag, plm_project_key: str, plm_assignee: str
    ) -> Optional[str]:
        flagged_locations = (
            "; ".join(
                f"{location.file_path}:{location.line_number}"
                for location in flag.flagged_locations
            )
            or "none extracted"
        )
        summary = (
            f"[Architectural Audit] {flag.pattern_type.value} in {flag.affected_service}"
        )
        description = "\n".join(
            [
                f"Pattern type: {flag.pattern_type.value}",
                f"Affected service: {flag.affected_service}",
                f"Contributing incidents: {', '.join(flag.contributing_incident_ids)}",
                f"Flagged files/lines: {flagged_locations}",
                f"Developer attribution: {', '.join(flag.developer_attribution) or 'unknown'}",
                f"Recommended reviewers: {', '.join(flag.recommended_reviewers) or 'unknown'}",
                f"Confidence: {flag.confidence:.2f}",
                "",
                f"Assessment: {flag.assessment}",
                "",
                _TRACEABILITY_NOTE,
            ]
        )
        try:
            result = self.jira.create_issue(
                project_key=plm_project_key,
                summary=summary,
                description=description,
                priority="P2",
                labels=["architectural-audit", flag.pattern_type.value],
                assignee=plm_assignee,
            )
            ticket_id = (result or {}).get("ticket_id")
            if ticket_id:
                logger.info(
                    "PLM ticket %s created for %s flag on %s",
                    ticket_id,
                    flag.pattern_type.value,
                    flag.affected_service,
                )
            return ticket_id
        except Exception:  # noqa: BLE001 - the auditor must never crash
            logger.error("PLM ticket creation failed", exc_info=True)
            return None
