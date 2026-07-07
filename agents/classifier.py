"""Agent 1: The Classifier.

This agent has no LLM. Zero. Every classification decision is made by
deterministic Python logic: regular expressions, keyword matching, and
rule-based scoring. When a judge asks how the classification works, the
answer is a specific regex pattern or a specific rule, not a model.

Takes a RichContext as input and returns a ClassificationResult, writing
the raw input to the bronze table and the classification to the silver
table before returning.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Optional

from core.database import DatabaseManager
from core.schemas import (
    BlastRadius,
    ClassificationResult,
    Environment,
    FailureType,
    RichContext,
    StackTraceLocation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic rule tables
# ---------------------------------------------------------------------------

# Failure-type regex patterns. Evaluated in this exact order; the FIRST
# matching pattern determines the failure type.
FAILURE_TYPE_PATTERNS: tuple[tuple[FailureType, tuple[re.Pattern[str], ...]], ...] = tuple(
    (failure_type, tuple(re.compile(pattern) for pattern in patterns))
    for failure_type, patterns in (
        (
            FailureType.NULL_POINTER,
            (r"AttributeError.*NoneType", r"NoneType", r"NullPointerException"),
        ),
        (
            FailureType.TIMEOUT,
            (r"TimeoutError", r"timed out", r"ReadTimeout"),
        ),
        (
            FailureType.AUTH_FAILURE,
            (
                r"InvalidToken",
                r"\b401\b",
                r"Unauthorized",
                r"authentication failed",
                r"SignatureExpired",
            ),
        ),
        (
            FailureType.RESOURCE_EXHAUSTION,
            (
                r"MemoryError",
                r"\bOOM\b",
                r"PoolExhausted",
                r"resource exhausted",
                r"too many open files",
            ),
        ),
        (
            FailureType.CONNECTION_REFUSED,
            (r"ConnectionRefused", r"ECONNREFUSED", r"Errno 111"),
        ),
        (
            FailureType.ASSERTION_ERROR,
            (r"AssertionError",),
        ),
        (
            FailureType.IMPORT_ERROR,
            (r"ImportError", r"ModuleNotFoundError"),
        ),
    )
)

FAILURE_MATCH_CONFIDENCE = 0.95
FAILURE_NO_MATCH_CONFIDENCE = 0.40

SERVICE_HINT_CONFIDENCE = 1.0
UNKNOWN_SERVICE = "unknown"
UNKNOWN_SERVICE_CONFIDENCE = 0.30
LOW_CONFIDENCE_THRESHOLD = 0.65

MAX_STACK_FRAMES = 10

# Python traceback frame:   File "path/to/file.py", line 42, in some_function
PYTHON_FRAME_RE = re.compile(r'File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>\S+)')
# Java stack frame:   at com.example.ClassName.methodName(File.java:123)
JAVA_FRAME_RE = re.compile(r"\bat\s+(?P<qualified>[\w$.<>]+)\((?P<file>[^():]+):(?P<line>\d+)\)")

# Environment keyword search, checked in this order against the error text.
ENVIRONMENT_PATTERNS: tuple[tuple[Environment, re.Pattern[str]], ...] = (
    (Environment.PRODUCTION, re.compile(r"\b(production|prod)\b", re.IGNORECASE)),
    (Environment.STAGING, re.compile(r"\b(staging|stage)\b", re.IGNORECASE)),
    (Environment.DEVELOPMENT, re.compile(r"\b(development|dev|local)\b", re.IGNORECASE)),
)

# Every optional RichContext field, used for the missing-context report.
OPTIONAL_CONTEXT_FIELDS: tuple[str, ...] = tuple(
    name for name in RichContext.model_fields if name != "raw_error_text"
)


def score_services(error_text: str, services: dict) -> tuple[Optional[str], int]:
    """Score every org-config service against the error text.

    The score for a service is the number of its keywords plus the number of
    its file path patterns that appear (case-insensitive) in the error text.
    Returns the best-scoring service name and its score; (None, 0) when no
    service scores at all. Shared with the router's fallback path.
    """
    text = error_text.lower()
    best_service: Optional[str] = None
    best_score = 0
    for service_name, service_cfg in services.items():
        score = sum(1 for keyword in service_cfg.get("keywords", []) if keyword.lower() in text)
        score += sum(
            1 for pattern in service_cfg.get("file_path_patterns", []) if pattern.lower() in text
        )
        if score > best_score:
            best_service, best_score = service_name, score
    return best_service, best_score


class IncidentClassifier:
    """Agent 1: deterministic incident classification. No LLM calls, ever."""

    def __init__(self, db: DatabaseManager, org_config: dict) -> None:
        self.db = db
        self.org_config = org_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, context: RichContext) -> ClassificationResult:
        """Classify one incident and persist it to bronze + silver tables."""
        now = datetime.now()
        incident_id = f"INC-{now:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        logger.info("Agent 1 (Classifier) starting for incident %s", incident_id)

        failure_type, failure_confidence = self._detect_failure_type(context.raw_error_text)
        detected_service, service_confidence = self._detect_service(context)
        environment = self._detect_environment(context)
        stack_trace_locations, source_format = self._extract_stack_trace(context.raw_error_text)
        is_critical_path = self._lookup_critical_path(detected_service)
        blast_radius = self._assess_blast_radius(context.affected_users)
        missing_context = self._identify_missing_context(context)

        occurrences = context.occurrences_last_4h if context.occurrences_last_4h is not None else 1
        is_first_occurrence = context.occurrences_last_4h is None or context.occurrences_last_4h <= 1

        # The overall classification confidence is the service-detection
        # confidence: the router treats < 0.65 as "service could not be
        # identified reliably".
        classification_confidence = service_confidence
        if classification_confidence < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low classification confidence %.2f for incident %s (service '%s' "
                "could not be identified reliably)",
                classification_confidence,
                incident_id,
                detected_service,
            )
        if missing_context:
            logger.warning(
                "Missing context for incident %s: %s",
                incident_id,
                ", ".join(missing_context),
            )
        logger.info(
            "Classified incident %s: service=%s (confidence %.2f), failure=%s "
            "(pattern confidence %.2f), environment=%s, %d stack frames",
            incident_id,
            detected_service,
            service_confidence,
            failure_type.value,
            failure_confidence,
            environment.value,
            len(stack_trace_locations),
        )

        result = ClassificationResult(
            incident_id=incident_id,
            raw_input=context,
            detected_service=detected_service,
            failure_type=failure_type,
            environment=environment,
            is_critical_path=is_critical_path,
            blast_radius=blast_radius,
            is_first_occurrence=is_first_occurrence,
            occurrences_last_4h=occurrences,
            stack_trace_locations=stack_trace_locations,
            classification_confidence=classification_confidence,
            timestamp=now,
            missing_context=missing_context,
        )

        # Persist to bronze (raw, append-only) and silver before returning.
        self.db.insert_bronze_incident(incident_id, context, source_format)
        self.db.insert_classification(result)

        logger.info("Agent 1 (Classifier) completed for incident %s", incident_id)
        return result

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_failure_type(error_text: str) -> tuple[FailureType, float]:
        """First matching regex pattern wins; no match means unknown."""
        for failure_type, patterns in FAILURE_TYPE_PATTERNS:
            for pattern in patterns:
                if pattern.search(error_text):
                    return failure_type, FAILURE_MATCH_CONFIDENCE
        return FailureType.UNKNOWN, FAILURE_NO_MATCH_CONFIDENCE

    def _detect_service(self, context: RichContext) -> tuple[str, float]:
        """Two-pass service detection.

        Pass 1: if a service hint was provided and matches an org-config
        service, that service wins with confidence 1.0. Pass 2: keyword and
        file-path-pattern scoring across all org-config services; confidence
        scales with the score. A zero score yields the unknown service with
        confidence 0.3.
        """
        services = self.org_config.get("services", {})

        if context.service_hint:
            hint = context.service_hint.strip().lower()
            for service_name in services:
                normalized = service_name.lower()
                if hint == normalized or hint in normalized or normalized in hint:
                    return service_name, SERVICE_HINT_CONFIDENCE

        best_service, best_score = score_services(context.raw_error_text, services)
        if best_service is None or best_score == 0:
            return UNKNOWN_SERVICE, UNKNOWN_SERVICE_CONFIDENCE
        return best_service, min(0.95, 0.5 + 0.05 * best_score)

    @staticmethod
    def _detect_environment(context: RichContext) -> Environment:
        """Context field first; otherwise keyword search of the error text."""
        if context.environment is not None:
            return context.environment
        for environment, pattern in ENVIRONMENT_PATTERNS:
            if pattern.search(context.raw_error_text):
                return environment
        return Environment.UNKNOWN

    @staticmethod
    def _extract_stack_trace(error_text: str) -> tuple[list[StackTraceLocation], str]:
        """Extract up to MAX_STACK_FRAMES locations and detect the format.

        Supports Python tracebacks (File "path", line N, in func) and Java
        stack traces (at Class.method(File.java:123)). File paths are
        normalized to forward slashes. Returns the locations plus the
        detected source format: python_traceback, java_stacktrace, or
        plain_text.
        """
        locations: list[StackTraceLocation] = []
        python_hits = 0
        java_hits = 0
        for raw_line in error_text.splitlines():
            if len(locations) >= MAX_STACK_FRAMES:
                break
            python_match = PYTHON_FRAME_RE.search(raw_line)
            if python_match:
                python_hits += 1
                locations.append(
                    StackTraceLocation(
                        file_path=python_match.group("path").replace("\\", "/"),
                        line_number=int(python_match.group("line")),
                        function_name=python_match.group("func"),
                        raw_line=raw_line.strip(),
                    )
                )
                continue
            java_match = JAVA_FRAME_RE.search(raw_line)
            if java_match:
                java_hits += 1
                locations.append(
                    StackTraceLocation(
                        file_path=java_match.group("file").replace("\\", "/"),
                        line_number=int(java_match.group("line")),
                        function_name=java_match.group("qualified"),
                        raw_line=raw_line.strip(),
                    )
                )

        if python_hits > 0:
            source_format = "python_traceback"
        elif java_hits > 0:
            source_format = "java_stacktrace"
        else:
            source_format = "plain_text"
        return locations, source_format

    def _lookup_critical_path(self, service: str) -> bool:
        service_cfg = self.org_config.get("services", {}).get(service)
        if service_cfg is None:
            return False
        return bool(service_cfg.get("critical_path", False))

    @staticmethod
    def _assess_blast_radius(affected_users: Optional[int]) -> BlastRadius:
        if affected_users is None:
            return BlastRadius.UNKNOWN
        if affected_users > 1000:
            return BlastRadius.ALL_USERS
        if affected_users > 100:
            return BlastRadius.ENTERPRISE_CUSTOMERS
        return BlastRadius.SUBSET_OF_USERS

    @staticmethod
    def _identify_missing_context(context: RichContext) -> list[str]:
        return [name for name in OPTIONAL_CONTEXT_FIELDS if getattr(context, name) is None]
