"""Pydantic data models for RouteZero.

These are the strict contracts between agents. Nothing passes between
agents except these validated models. If data does not conform to the
schema it does not pass through.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FailureType(str, Enum):
    NULL_POINTER = "null_pointer"
    TIMEOUT = "timeout"
    AUTH_FAILURE = "auth_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONNECTION_REFUSED = "connection_refused"
    ASSERTION_ERROR = "assertion_error"
    IMPORT_ERROR = "import_error"
    UNKNOWN = "unknown"


class Environment(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class BlastRadius(str, Enum):
    ALL_USERS = "all_users"
    ENTERPRISE_CUSTOMERS = "enterprise_customers"
    SUBSET_OF_USERS = "subset_of_users"
    INTERNAL_ONLY = "internal_only"
    UNKNOWN = "unknown"


class OutputType(str, Enum):
    FULL_TICKET = "full_ticket"
    MANAGER_DIGEST = "manager_digest"
    CROSS_FUNCTIONAL_FYI = "cross_functional_fyi"


class NodeType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"


class PatternType(str, Enum):
    RECURRING_LOCATION = "recurring_location"
    SERVICE_STRESS = "service_stress"
    CASCADING_FAILURE = "cascading_failure"


class NotificationChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    JIRA = "jira"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class StackTraceLocation(BaseModel):
    """One extracted location from a stack trace."""

    file_path: str
    line_number: int
    function_name: str
    raw_line: str = ""


class DeploymentInfo(BaseModel):
    """Details about the most recent deployment, if the engineer knows them."""

    deployer: str
    commit_hash: str
    commit_message: str
    deployed_at: datetime


class Stakeholder(BaseModel):
    """One person who must be notified about an incident."""

    name: str
    role: str
    notify_via: NotificationChannel = NotificationChannel.SLACK
    output_type: OutputType = OutputType.FULL_TICKET


class ProbableCause(BaseModel):
    """A deployment-linked probable cause with decaying confidence."""

    description: str
    commit_hash: str
    deployer: str
    commit_message: str
    minutes_before_incident: int
    confidence: float = Field(ge=0.0, le=1.0)


class NotificationRecord(BaseModel):
    """The result of one notification attempt."""

    recipient: str
    channel: str
    output_type: OutputType
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline input
# ---------------------------------------------------------------------------


class RichContext(BaseModel):
    """The input to the entire pipeline.

    Only the raw error text is required. Every other field is optional
    context that improves classification and routing when provided.
    """

    raw_error_text: str
    service_hint: Optional[str] = None
    environment: Optional[Environment] = None
    occurrences_last_4h: Optional[int] = None
    affected_users: Optional[int] = None
    customer_tier: Optional[str] = None
    recent_deployment: Optional[DeploymentInfo] = None
    sla_breach_minutes: Optional[int] = None
    on_call_engineer: Optional[str] = None
    related_ticket_ids: Optional[list[str]] = None
    runbook_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent 1 output
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """The output of Agent 1, the deterministic classifier."""

    incident_id: str
    raw_input: RichContext
    detected_service: str
    failure_type: FailureType
    environment: Environment
    is_critical_path: bool
    blast_radius: BlastRadius
    is_first_occurrence: bool
    occurrences_last_4h: int = 1
    stack_trace_locations: list[StackTraceLocation] = Field(default_factory=list)
    classification_confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    missing_context: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 2 output
# ---------------------------------------------------------------------------


class RoutingDecision(BaseModel):
    """The output of Agent 2, the rule-based router."""

    classification: ClassificationResult
    owning_team: str
    assignee: str
    priority: Priority
    priority_reasoning: str
    jira_project_key: str
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    manager_digest_triggered: bool = False
    probable_cause: Optional[ProbableCause] = None
    related_tickets: list[str] = Field(default_factory=list)
    runbook_url: Optional[str] = None
    routing_confidence: float = Field(ge=0.0, le=1.0)
    routing_reasoning: str
    missing_context: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 3 output
# ---------------------------------------------------------------------------


class TicketContent(BaseModel):
    """One piece of output for one recipient."""

    recipient_name: str
    recipient_role: str
    output_type: OutputType
    title: str
    body: str


class IncidentIntelligenceRecord(BaseModel):
    """The clean structured summary Agent 3 writes for Agent 4 to read later."""

    incident_id: str
    jira_ticket_ref: Optional[str] = None
    timestamp: datetime
    service: str
    failure_type: FailureType
    environment: Environment
    is_critical_path: bool
    affected_users: Optional[int] = None
    priority: Priority
    stack_trace_locations: list[StackTraceLocation] = Field(default_factory=list)
    probable_cause: Optional[str] = None
    linked_deployment: Optional[str] = None
    routing_confidence: float = Field(ge=0.0, le=1.0)
    owning_team: str
    resolved: bool = False
    resolution_time_minutes: Optional[int] = None
    agent3_summary: str


class TicketWriterOutput(BaseModel):
    """The output of Agent 3, the ticket writer."""

    incident_id: str
    routing_decision: RoutingDecision
    ticket_contents: list[TicketContent] = Field(default_factory=list)
    jira_ticket_id: Optional[str] = None
    jira_url: Optional[str] = None
    notifications_sent: list[NotificationRecord] = Field(default_factory=list)
    intelligence_record: IncidentIntelligenceRecord


# ---------------------------------------------------------------------------
# Code knowledge graph
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """One node in the code knowledge graph."""

    node_id: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    node_type: NodeType
    owner: Optional[str] = None
    last_modified: Optional[datetime] = None


class GraphEdge(BaseModel):
    """One relationship between two graph nodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Agent 4 output
# ---------------------------------------------------------------------------


class ArchitecturalFlag(BaseModel):
    """One finding from Agent 4, the architectural auditor."""

    pattern_type: PatternType
    affected_service: str
    contributing_incident_ids: list[str] = Field(default_factory=list)
    flagged_locations: list[StackTraceLocation] = Field(default_factory=list)
    connected_nodes: list[GraphNode] = Field(default_factory=list)
    developer_attribution: list[str] = Field(default_factory=list)
    assessment: str
    recommended_reviewers: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    plm_ticket_id: Optional[str] = None


class AuditorOutput(BaseModel):
    """The output of Agent 4, the architectural auditor."""

    audit_id: str
    timestamp: datetime
    incidents_analyzed: int
    patterns_found: int
    flags: list[ArchitecturalFlag] = Field(default_factory=list)
    plm_tickets_created: list[str] = Field(default_factory=list)
