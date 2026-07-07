"""RouteZero FastAPI backend.

Wires together the four agents and exposes the HTTP endpoints the
Streamlit frontend calls. The frontend never imports agents directly.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from agents.auditor import ArchitecturalAuditor
from agents.classifier import IncidentClassifier
from agents.router import IncidentRouter
from agents.ticket_writer import TicketWriter
from core.database import DatabaseManager
from core.fireworks_client import FireworksClient
from core.graph import CodeGraphBuilder
from core.schemas import AuditorOutput, RichContext, TicketWriterOutput
from integrations.github_client import GitHubClient
from integrations.jira_client import JiraClient
from integrations.slack_client import SlackClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent


def _load_org_config() -> dict:
    config_path = REPO_ROOT / "data" / "org_config.json"
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class AppState:
    """All singletons, initialized once at startup."""

    def __init__(self) -> None:
        self.org_config = _load_org_config()
        self.db = DatabaseManager.get_instance()
        self.fireworks = FireworksClient()
        self.jira = JiraClient()
        self.slack = SlackClient()
        self.github = GitHubClient()
        self.graph = CodeGraphBuilder(self.db)
        self.classifier = IncidentClassifier(self.db, self.org_config)
        self.router = IncidentRouter(self.db, self.org_config, self.fireworks)
        self.ticket_writer = TicketWriter(
            self.db, self.org_config, self.fireworks, self.jira, self.slack
        )
        self.auditor = ArchitecturalAuditor(
            self.db,
            self.org_config,
            self.fireworks,
            self.jira,
            self.github,
            self.graph,
        )


state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = AppState()
    seeded = state.db.seed_historical_incidents()
    logger.info("Startup complete; %s historical incidents seeded", seeded)
    yield


app = FastAPI(title="RouteZero", lifespan=lifespan)


@app.post("/incidents", response_model=TicketWriterOutput)
def process_incident(context: RichContext) -> TicketWriterOutput:
    """Run the full Agent 1 -> Agent 2 -> Agent 3 pipeline (preview mode)."""
    try:
        classification = state.classifier.classify(context)
        decision = state.router.route(classification, context)
        output = state.ticket_writer.write_tickets(decision, context, send=False)
        return output
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")


@app.post("/incidents/{incident_id}/approve")
def approve_incident(incident_id: str) -> dict:
    """Trigger the actual Jira creation and Slack notifications."""
    try:
        return state.ticket_writer.approve_and_send(incident_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Approve failed for %s: %s", incident_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Approve failed: {exc}")


@app.get("/incidents")
def list_incidents() -> list[dict]:
    """All incidents from the intelligence table, newest first."""
    return state.db.get_all_incidents()


@app.get("/incidents/{incident_id}/detail")
def incident_detail(incident_id: str) -> dict:
    """Full stored detail for one incident (history tab expansion)."""
    return {
        "classification": state.db.get_classification(incident_id),
        "routing": state.db.get_routing_decision(incident_id),
        "ticket_output": state.db.get_ticket_output(incident_id),
    }


@app.post("/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: str) -> dict:
    state.db.mark_incident_resolved(incident_id)
    return {"status": "resolved", "incident_id": incident_id}


@app.post("/audit", response_model=AuditorOutput)
def run_audit() -> AuditorOutput:
    """Trigger Agent 4 immediately."""
    try:
        return state.auditor.run_audit()
    except Exception as exc:
        logger.error("Audit failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit failed: {exc}")


@app.get("/audit/latest")
def latest_audit() -> dict | None:
    return state.db.get_latest_audit()


@app.get("/graph/nodes")
def graph_nodes() -> dict:
    """All graph nodes with incident counts plus edges, for visualization."""
    try:
        state.graph.build_or_load()
    except Exception as exc:
        logger.error("Graph build failed: %s", exc, exc_info=True)
    counts = state.db.get_node_incident_counts()
    nodes = []
    for node in state.db.get_graph_nodes():
        node["incident_count"] = counts.get(node["node_id"], 0)
        nodes.append(node)
    return {"nodes": nodes, "edges": state.db.get_graph_edges()}


@app.get("/stats")
def stats() -> dict:
    """Session stats for the dashboard sidebar."""
    return {
        "fireworks_calls": state.fireworks.call_count,
        "table_counts": state.db.get_table_counts(),
    }
