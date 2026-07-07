"""DuckDB persistence layer for RouteZero.

All database operations for the entire application live here. The
DatabaseManager is a thread-safe singleton: Streamlit runs in multiple
threads and concurrent access to a single DuckDB connection would corrupt
the database without locking, so every database operation acquires a
threading lock.

Callers never write SQL. Every insert method takes a validated Pydantic
model from core.schemas and handles serialization internally (complex
fields are stored as JSON strings, datetimes as ISO strings). Every query
method returns a list of dictionaries or a single dictionary, with JSON
fields and timestamps parsed back into Python objects.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import duckdb
from dotenv import load_dotenv

from core.schemas import (
    AuditorOutput,
    ClassificationResult,
    GraphEdge,
    GraphNode,
    IncidentIntelligenceRecord,
    NotificationRecord,
    RichContext,
    RoutingDecision,
    TicketWriterOutput,
)

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "./database/routezero.db"


# ---------------------------------------------------------------------------
# datetime helpers: store ISO strings, parse back when returning dicts
# ---------------------------------------------------------------------------


def _now() -> datetime:
    """Naive local time; every stored timestamp is normalized to this."""
    return datetime.now()


def _to_naive(dt: datetime) -> datetime:
    """Normalize a datetime to naive local time so comparisons are safe."""
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _to_iso(dt: datetime) -> str:
    return _to_naive(dt).isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_naive(value)
    return _to_naive(datetime.fromisoformat(str(value)))


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class DatabaseManager:
    """Thread-safe singleton owning the one DuckDB connection.

    Instantiate with ``DatabaseManager()`` or ``DatabaseManager.get_instance()``;
    both always return the same object. The database file path comes from the
    ``DUCKDB_PATH`` environment variable (loaded via python-dotenv), defaulting
    to ``./database/routezero.db``. All ten tables are created on first init
    with CREATE TABLE IF NOT EXISTS; existing tables are never recreated.
    """

    TABLES: tuple[str, ...] = (
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
    )

    _instance: Optional["DatabaseManager"] = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton plumbing
    # ------------------------------------------------------------------

    def __new__(cls) -> "DatabaseManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        load_dotenv()
        db_path = Path(os.getenv("DUCKDB_PATH", _DEFAULT_DB_PATH))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        # RLock so composite operations (e.g. seeding, which calls insert
        # methods) can safely nest lock acquisitions in the same thread.
        self._lock = threading.RLock()
        self._conn = duckdb.connect(self._db_path)
        self._create_tables()
        self._initialized = True
        logger.info("DatabaseManager initialized with DuckDB file at %s", self._db_path)

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Close the connection and drop the singleton (used by tests)."""
        with cls._instance_lock:
            instance = cls._instance
            if instance is not None:
                conn = getattr(instance, "_conn", None)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001 - closing must never raise
                        logger.warning("Error closing DuckDB connection during reset", exc_info=True)
                cls._instance = None

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        ddl_statements = (
            """
            CREATE TABLE IF NOT EXISTS bronze_raw_incidents (
                incident_id VARCHAR PRIMARY KEY,
                raw_error_text VARCHAR NOT NULL,
                rich_context VARCHAR NOT NULL,
                received_at VARCHAR NOT NULL,
                source_format VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS silver_classified_incidents (
                incident_id VARCHAR PRIMARY KEY,
                raw_input VARCHAR,
                detected_service VARCHAR,
                failure_type VARCHAR,
                environment VARCHAR,
                is_critical_path BOOLEAN,
                blast_radius VARCHAR,
                is_first_occurrence BOOLEAN,
                occurrences_last_4h INTEGER,
                stack_trace_locations VARCHAR,
                classification_confidence DOUBLE,
                timestamp VARCHAR,
                missing_context VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS silver_routing_decisions (
                incident_id VARCHAR PRIMARY KEY,
                classification VARCHAR,
                owning_team VARCHAR,
                assignee VARCHAR,
                priority VARCHAR,
                priority_reasoning VARCHAR,
                jira_project_key VARCHAR,
                stakeholders VARCHAR,
                manager_digest_triggered BOOLEAN,
                probable_cause VARCHAR,
                related_tickets VARCHAR,
                runbook_url VARCHAR,
                routing_confidence DOUBLE,
                routing_reasoning VARCHAR,
                missing_context VARCHAR,
                created_at VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gold_created_tickets (
                ticket_id VARCHAR PRIMARY KEY,
                incident_id VARCHAR,
                jira_ticket_id VARCHAR,
                jira_url VARCHAR,
                ticket_contents VARCHAR,
                notifications_sent VARCHAR,
                created_at VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gold_incident_intelligence (
                incident_id VARCHAR PRIMARY KEY,
                jira_ticket_ref VARCHAR,
                timestamp VARCHAR,
                service VARCHAR,
                failure_type VARCHAR,
                environment VARCHAR,
                is_critical_path BOOLEAN,
                affected_users INTEGER,
                priority VARCHAR,
                stack_trace_locations VARCHAR,
                probable_cause VARCHAR,
                linked_deployment VARCHAR,
                routing_confidence DOUBLE,
                owning_team VARCHAR,
                resolved BOOLEAN,
                resolution_time_minutes INTEGER,
                agent3_summary VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gold_notification_log (
                log_id VARCHAR PRIMARY KEY,
                incident_id VARCHAR,
                recipient VARCHAR,
                channel VARCHAR,
                output_type VARCHAR,
                timestamp VARCHAR,
                success BOOLEAN,
                error_message VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gold_audit_runs (
                audit_id VARCHAR PRIMARY KEY,
                timestamp VARCHAR,
                incidents_analyzed INTEGER,
                patterns_found INTEGER,
                flags VARCHAR,
                plm_tickets_created VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                file_path VARCHAR,
                start_line INTEGER,
                end_line INTEGER,
                node_type VARCHAR,
                owner VARCHAR,
                last_modified VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id VARCHAR PRIMARY KEY,
                source_node_id VARCHAR,
                target_node_id VARCHAR,
                relationship_type VARCHAR,
                weight DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS graph_incident_node_mapping (
                mapping_id VARCHAR PRIMARY KEY,
                incident_id VARCHAR,
                node_id VARCHAR,
                occurrence_count INTEGER
            )
            """,
        )
        with self._lock:
            for statement in ddl_statements:
                self._conn.execute(statement)
        logger.info("Schema initialized: %d tables ensured", len(ddl_statements))

    # ------------------------------------------------------------------
    # Internal query helpers
    # ------------------------------------------------------------------

    def _fetch_dicts(self, sql: str, params: Optional[list] = None) -> list[dict]:
        with self._lock:
            cursor = self._conn.execute(sql, params or [])
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def _fetch_one_dict(self, sql: str, params: Optional[list] = None) -> Optional[dict]:
        results = self._fetch_dicts(sql, params)
        return results[0] if results else None

    @staticmethod
    def _hydrate_intelligence_row(row: dict) -> dict:
        row["stack_trace_locations"] = json.loads(row["stack_trace_locations"] or "[]")
        row["timestamp"] = _parse_dt(row["timestamp"])
        return row

    # ------------------------------------------------------------------
    # Insert methods (take Pydantic models, serialize internally)
    # ------------------------------------------------------------------

    def insert_bronze_incident(self, incident_id: str, context: RichContext, source_format: str) -> None:
        """Append the raw input exactly as received. Never modified afterwards."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO bronze_raw_incidents VALUES (?, ?, ?, ?, ?)",
                [
                    incident_id,
                    context.raw_error_text,
                    context.model_dump_json(),
                    _to_iso(_now()),
                    source_format,
                ],
            )
        logger.info("DB write: bronze_raw_incidents <- %s", incident_id)

    def insert_classification(self, result: ClassificationResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO silver_classified_incidents "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    result.incident_id,
                    result.raw_input.model_dump_json(),
                    result.detected_service,
                    result.failure_type.value,
                    result.environment.value,
                    result.is_critical_path,
                    result.blast_radius.value,
                    result.is_first_occurrence,
                    result.occurrences_last_4h,
                    _dump_json([loc.model_dump() for loc in result.stack_trace_locations]),
                    result.classification_confidence,
                    _to_iso(result.timestamp),
                    _dump_json(result.missing_context),
                ],
            )
        logger.info("DB write: silver_classified_incidents <- %s", result.incident_id)

    def insert_routing_decision(self, decision: RoutingDecision) -> None:
        incident_id = decision.classification.incident_id
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO silver_routing_decisions "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    incident_id,
                    decision.classification.model_dump_json(),
                    decision.owning_team,
                    decision.assignee,
                    decision.priority.value,
                    decision.priority_reasoning,
                    decision.jira_project_key,
                    _dump_json([s.model_dump(mode="json") for s in decision.stakeholders]),
                    decision.manager_digest_triggered,
                    decision.probable_cause.model_dump_json() if decision.probable_cause else None,
                    _dump_json(decision.related_tickets),
                    decision.runbook_url,
                    decision.routing_confidence,
                    decision.routing_reasoning,
                    _dump_json(decision.missing_context),
                    _to_iso(_now()),
                ],
            )
        logger.info("DB write: silver_routing_decisions <- %s", incident_id)

    def insert_created_ticket(self, output: TicketWriterOutput) -> None:
        ticket_id = f"TCK-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO gold_created_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ticket_id,
                    output.incident_id,
                    output.jira_ticket_id,
                    output.jira_url,
                    _dump_json([c.model_dump(mode="json") for c in output.ticket_contents]),
                    _dump_json([n.model_dump(mode="json") for n in output.notifications_sent]),
                    _to_iso(_now()),
                ],
            )
        logger.info("DB write: gold_created_tickets <- %s (incident %s)", ticket_id, output.incident_id)

    def insert_intelligence_record(self, record: IncidentIntelligenceRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO gold_incident_intelligence "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    record.incident_id,
                    record.jira_ticket_ref,
                    _to_iso(record.timestamp),
                    record.service,
                    record.failure_type.value,
                    record.environment.value,
                    record.is_critical_path,
                    record.affected_users,
                    record.priority.value,
                    _dump_json([loc.model_dump() for loc in record.stack_trace_locations]),
                    record.probable_cause,
                    record.linked_deployment,
                    record.routing_confidence,
                    record.owning_team,
                    record.resolved,
                    record.resolution_time_minutes,
                    record.agent3_summary,
                ],
            )
        logger.info("DB write: gold_incident_intelligence <- %s", record.incident_id)

    def insert_notification_log(self, incident_id: str, record: NotificationRecord) -> None:
        log_id = f"NTF-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO gold_notification_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    log_id,
                    incident_id,
                    record.recipient,
                    record.channel,
                    record.output_type.value,
                    _to_iso(_now()),
                    record.success,
                    record.error_message,
                ],
            )
        logger.info("DB write: gold_notification_log <- %s (incident %s)", log_id, incident_id)

    def insert_audit_run(self, output: AuditorOutput) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO gold_audit_runs VALUES (?, ?, ?, ?, ?, ?)",
                [
                    output.audit_id,
                    _to_iso(output.timestamp),
                    output.incidents_analyzed,
                    output.patterns_found,
                    _dump_json([flag.model_dump(mode="json") for flag in output.flags]),
                    _dump_json(output.plm_tickets_created),
                ],
            )
        logger.info("DB write: gold_audit_runs <- %s", output.audit_id)

    # ------------------------------------------------------------------
    # Query methods (return dicts; JSON and timestamps parsed back)
    # ------------------------------------------------------------------

    def get_all_incidents(self) -> list[dict]:
        rows = self._fetch_dicts(
            "SELECT * FROM gold_incident_intelligence ORDER BY timestamp DESC"
        )
        return [self._hydrate_intelligence_row(row) for row in rows]

    def get_incidents_since(self, days: int) -> list[dict]:
        cutoff = _to_iso(_now() - timedelta(days=days))
        rows = self._fetch_dicts(
            "SELECT * FROM gold_incident_intelligence WHERE timestamp >= ? "
            "ORDER BY timestamp DESC",
            [cutoff],
        )
        return [self._hydrate_intelligence_row(row) for row in rows]

    def count_service_incidents_last_hour(self, service: str) -> int:
        cutoff = _to_iso(_now() - timedelta(hours=1))
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM gold_incident_intelligence "
                "WHERE service = ? AND timestamp >= ?",
                [service, cutoff],
            ).fetchone()
        return int(row[0])

    def get_classification(self, incident_id: str) -> Optional[dict]:
        row = self._fetch_one_dict(
            "SELECT * FROM silver_classified_incidents WHERE incident_id = ?",
            [incident_id],
        )
        if row is None:
            return None
        row["raw_input"] = json.loads(row["raw_input"] or "{}")
        row["stack_trace_locations"] = json.loads(row["stack_trace_locations"] or "[]")
        row["missing_context"] = json.loads(row["missing_context"] or "[]")
        row["timestamp"] = _parse_dt(row["timestamp"])
        return row

    def get_routing_decision(self, incident_id: str) -> Optional[dict]:
        row = self._fetch_one_dict(
            "SELECT * FROM silver_routing_decisions WHERE incident_id = ?",
            [incident_id],
        )
        if row is None:
            return None
        row["classification"] = json.loads(row["classification"] or "{}")
        row["stakeholders"] = json.loads(row["stakeholders"] or "[]")
        row["probable_cause"] = json.loads(row["probable_cause"]) if row["probable_cause"] else None
        row["related_tickets"] = json.loads(row["related_tickets"] or "[]")
        row["missing_context"] = json.loads(row["missing_context"] or "[]")
        row["created_at"] = _parse_dt(row["created_at"])
        return row

    def get_latest_audit(self) -> Optional[dict]:
        row = self._fetch_one_dict(
            "SELECT * FROM gold_audit_runs ORDER BY timestamp DESC LIMIT 1"
        )
        if row is None:
            return None
        row["flags"] = json.loads(row["flags"] or "[]")
        row["plm_tickets_created"] = json.loads(row["plm_tickets_created"] or "[]")
        row["timestamp"] = _parse_dt(row["timestamp"])
        return row

    def get_ticket_output(self, incident_id: str) -> Optional[dict]:
        row = self._fetch_one_dict(
            "SELECT * FROM gold_created_tickets WHERE incident_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            [incident_id],
        )
        if row is None:
            return None
        row["ticket_contents"] = json.loads(row["ticket_contents"] or "[]")
        row["notifications_sent"] = json.loads(row["notifications_sent"] or "[]")
        row["created_at"] = _parse_dt(row["created_at"])
        return row

    def update_ticket_jira_info(
        self, incident_id: str, jira_ticket_id: Optional[str], jira_url: Optional[str]
    ) -> None:
        """Persist the Jira ticket id and URL onto the stored gold ticket row."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM gold_created_tickets WHERE incident_id = ?",
                [incident_id],
            ).fetchone()
            if int(row[0]) == 0:
                logger.warning(
                    "update_ticket_jira_info: no gold_created_tickets row for incident %s",
                    incident_id,
                )
                return
            self._conn.execute(
                "UPDATE gold_created_tickets SET jira_ticket_id = ?, jira_url = ? "
                "WHERE incident_id = ?",
                [jira_ticket_id, jira_url, incident_id],
            )
        logger.info(
            "DB write: gold_created_tickets jira info set for incident %s (%s)",
            incident_id,
            jira_ticket_id,
        )

    def update_intelligence_jira_ref(self, incident_id: str, jira_ticket_ref: Optional[str]) -> None:
        """Persist the Jira ticket reference onto the stored intelligence row."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM gold_incident_intelligence WHERE incident_id = ?",
                [incident_id],
            ).fetchone()
            if int(row[0]) == 0:
                logger.warning(
                    "update_intelligence_jira_ref: no gold_incident_intelligence row for incident %s",
                    incident_id,
                )
                return
            self._conn.execute(
                "UPDATE gold_incident_intelligence SET jira_ticket_ref = ? WHERE incident_id = ?",
                [jira_ticket_ref, incident_id],
            )
        logger.info(
            "DB write: gold_incident_intelligence jira_ticket_ref set for incident %s (%s)",
            incident_id,
            jira_ticket_ref,
        )

    def mark_incident_resolved(self, incident_id: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT timestamp FROM gold_incident_intelligence WHERE incident_id = ?",
                [incident_id],
            ).fetchone()
            if row is None:
                logger.warning("mark_incident_resolved: incident %s not found", incident_id)
                return
            started_at = _parse_dt(row[0])
            elapsed_minutes = max(0, int(round((_now() - started_at).total_seconds() / 60)))
            self._conn.execute(
                "UPDATE gold_incident_intelligence "
                "SET resolved = TRUE, resolution_time_minutes = ? WHERE incident_id = ?",
                [elapsed_minutes, incident_id],
            )
        logger.info(
            "DB write: gold_incident_intelligence %s marked resolved after %d minutes",
            incident_id,
            elapsed_minutes,
        )

    # ------------------------------------------------------------------
    # Code knowledge graph
    # ------------------------------------------------------------------

    def insert_graph_nodes(self, nodes: list[GraphNode]) -> None:
        """Idempotent: re-inserting a node_id replaces the existing row."""
        if not nodes:
            return
        rows = [
            [
                node.node_id,
                node.name,
                node.file_path,
                node.start_line,
                node.end_line,
                node.node_type.value,
                node.owner,
                _to_iso(node.last_modified) if node.last_modified else None,
            ]
            for node in nodes
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        logger.info("DB write: graph_nodes <- %d nodes", len(rows))

    def insert_graph_edges(self, edges: list[GraphEdge]) -> None:
        """Idempotent: re-inserting an edge_id replaces the existing row."""
        if not edges:
            return
        rows = [
            [
                edge.edge_id,
                edge.source_node_id,
                edge.target_node_id,
                edge.relationship_type,
                edge.weight,
            ]
            for edge in edges
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?, ?, ?)", rows
            )
        logger.info("DB write: graph_edges <- %d edges", len(rows))

    def get_graph_nodes(self) -> list[dict]:
        rows = self._fetch_dicts("SELECT * FROM graph_nodes ORDER BY node_id")
        for row in rows:
            row["last_modified"] = _parse_dt(row["last_modified"])
        return rows

    def get_graph_edges(self) -> list[dict]:
        return self._fetch_dicts("SELECT * FROM graph_edges ORDER BY edge_id")

    def upsert_incident_node_mapping(self, incident_id: str, node_id: str, occurrence_count: int) -> None:
        mapping_id = f"{incident_id}::{node_id}"
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_incident_node_mapping VALUES (?, ?, ?, ?)",
                [mapping_id, incident_id, node_id, occurrence_count],
            )
        logger.info("DB write: graph_incident_node_mapping <- %s", mapping_id)

    def get_node_incident_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, COUNT(DISTINCT incident_id) "
                "FROM graph_incident_node_mapping GROUP BY node_id"
            ).fetchall()
        return {node_id: int(count) for node_id, count in rows}

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def get_table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for table in self.TABLES:
                row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = int(row[0])
        return counts

    # ------------------------------------------------------------------
    # Historical incident seeding
    # ------------------------------------------------------------------

    def seed_historical_incidents(self, folder: str = "data/historical_incidents") -> int:
        """Load the pre-built incident JSON files into gold_incident_intelligence.

        Skips entirely (with a WARNING) if the table already has rows, so the
        seed is idempotent across restarts. Timestamps are re-based at seed
        time so the incidents always fall within the last 7 days relative to
        now: relative spacing between incidents is preserved and the most
        recent incident lands about one day ago. This guarantees the audit
        demo works no matter what date the app runs.

        Returns the number of records inserted.
        """
        with self._lock:
            existing = self._conn.execute(
                "SELECT COUNT(*) FROM gold_incident_intelligence"
            ).fetchone()[0]
        if existing > 0:
            logger.warning(
                "Historical incident seeding skipped: gold_incident_intelligence already has %d rows",
                existing,
            )
            return 0

        folder_path = self._resolve_seed_folder(folder)
        if folder_path is None:
            logger.warning("Historical incident folder not found: %s", folder)
            return 0

        json_files = sorted(folder_path.glob("*.json"))
        if not json_files:
            logger.warning("No historical incident JSON files found in %s", folder_path)
            return 0

        records: list[IncidentIntelligenceRecord] = []
        for json_file in json_files:
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                record = IncidentIntelligenceRecord.model_validate(payload)
                record = record.model_copy(update={"timestamp": _to_naive(record.timestamp)})
                records.append(record)
            except Exception:  # noqa: BLE001 - a bad seed file must not crash startup
                logger.error("Invalid historical incident file skipped: %s", json_file, exc_info=True)

        if not records:
            logger.warning("No valid historical incident records found in %s", folder_path)
            return 0

        # Re-base timestamps: preserve relative spacing, put the most recent
        # incident about one day before now.
        newest = max(record.timestamp for record in records)
        offset = (_now() - timedelta(days=1)) - newest
        for record in records:
            rebased = record.model_copy(update={"timestamp": record.timestamp + offset})
            self.insert_intelligence_record(rebased)

        logger.info(
            "Seeded %d historical incidents from %s (timestamps re-based, most recent ~1 day ago)",
            len(records),
            folder_path,
        )
        return len(records)

    @staticmethod
    def _resolve_seed_folder(folder: str) -> Optional[Path]:
        path = Path(folder)
        if path.is_dir():
            return path
        if not path.is_absolute():
            fallback = Path(__file__).resolve().parents[1] / folder
            if fallback.is_dir():
                return fallback
        return None
