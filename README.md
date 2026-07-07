# RouteZero

**Zero-touch incident routing. Every claim traceable to a verified input — nothing invented.**

When an engineer finds an error, the technical work takes minutes. The administrative work that follows takes 20–40 minutes, every single time: figuring out which team owns the service, writing a ticket detailed enough that the receiving team doesn't come back with five questions, setting the right priority, and notifying the right people with the right level of detail for each of them. RouteZero eliminates that. Paste a stack trace with whatever context you have, and a four-agent pipeline classifies the failure, routes it to the owning team with an explained priority, and drafts a full Jira ticket plus role-appropriate messages for every stakeholder — engineer ticket, five-sentence manager digest, cross-functional FYI. You review a preview, click **Approve & Send**, and everything is created. A fourth agent then mines the accumulated incident history for architectural rot and files proactive flags before the next incident happens. The system's one governing rule: if a piece of information was not provided as input, it does not appear in any output. Ever.

## Architecture

Four agents, orchestrated in pure Python (no LangChain, no AutoGen — by design). LLM usage is deliberately minimal and always validated.

```
                         ┌──────────────────────────────────────────────┐
                         │        Streamlit Dashboard (:8501)           │
                         │  New Incident │ History │ Arch Intelligence  │
                         └──────────────────┬───────────────────────────┘
                                            │ HTTP
                         ┌──────────────────▼───────────────────────────┐
                         │           FastAPI Backend (:8000)            │
                         └──────────────────┬───────────────────────────┘
                                            │
  error text + context                      ▼
 ───────────────────────► [Agent 1: Classifier] ──► [Agent 2: Router] ──► [Agent 3: Ticket Writer]
                            regex + rules only        rules-first,           Fireworks prose from
                            ZERO LLM calls            LLM only if            verified facts only,
                                 │                    confidence < 0.65      hallucination-validated
                                 │                         │                      │        │
                                 ▼                         ▼                      ▼        ▼
                          ┌────────────────────────────────────────────┐   Jira + Slack (or
                          │       DuckDB (medallion layout)            │   DEMO_MODE previews)
                          │  bronze_raw_incidents                      │
                          │  silver_classified / silver_routing        │
                          │  gold_tickets / gold_incident_intelligence │
                          │  graph_nodes / graph_edges                 │
                          └───────────────────┬────────────────────────┘
                                              │ reads history
                                              ▼
                              [Agent 4: Architectural Auditor]
                          pattern detection + code graph traversal
                                              │
                                              ▼
                            PLM tickets + red/orange/green graph viz
```

### Agent 1 — Classifier (`agents/classifier.py`)

Deterministic. **Zero LLM calls.** Failure type via regex patterns, service detection via keyword/file-path scoring against the org config, stack trace extraction (Python and Java formats), environment detection, blast radius from affected-user counts, and a missing-context list. When a judge asks how a classification was made, the answer is a specific regex or rule — not a model.

### Agent 2 — Router (`agents/router.py`)

Rules-first. Team ownership, priority (with explicit reasoning for why that priority fired), stakeholder assembly, manager-digest thresholds, and deployment-based probable cause are all deterministic. Fireworks (Gemma2-9b-it) is consulted in exactly one situation: service confidence from Agent 1 below **0.65**. Every decision ships with a plain-English `routing_reasoning` field you can verify.

### Agent 3 — Ticket Writer (`agents/ticket_writer.py`)

Fireworks assembles prose (summary, manager digest, FYI) from the verified facts in the routing decision — it is never asked to invent anything. Every AI response is **hallucination-validated**: any number or file-like token not present in the input facts causes the text to be discarded in favor of a deterministic template. Evidence, blast radius, and investigate-first sections are always deterministic. Works with no API key at all via fallback templates.

### Agent 4 — Architectural Auditor (`agents/auditor.py`)

Runs on demand from the dashboard. Reads incident intelligence from DuckDB and detects three patterns: **recurring location** (same file:line in 2+ incidents), **service stress** (3+ incidents across 2+ failure types in one service), and **cascading failure** (cross-service incidents within 30 minutes). For each finding it fetches the actual code, pulls the two-hop neighborhood from the code knowledge graph, and asks Fireworks to assess the structural weakness — citing the real code. Flags above 0.70 confidence become PLM Jira tickets. It only makes claims about code that appeared in real incident stack traces; when evidence is thin, it stays silent.

### Storage and the code graph

- **DuckDB medallion layout** (`core/database.py`): `bronze_raw_incidents` (append-only raw inputs) → `silver_classified_incidents` / `silver_routing_decisions` (agent outputs) → `gold_created_tickets` / `gold_incident_intelligence` / `gold_audit_runs` (finished products), plus graph tables.
- **Code knowledge graph** (`core/graph.py`): the fictional StreamCo codebase in `demo_repo/` is parsed with the stdlib `ast` module into a `networkx` graph (modules, functions, classes; contains/imports/calls edges), persisted to DuckDB, and rendered with `pyvis` in the dashboard — **red** nodes appeared in 2+ incidents, **orange** nodes neighbor red ones, **green** nodes are clean.

## Tech stack

- **FastAPI + uvicorn** — backend API
- **Streamlit** — dashboard
- **DuckDB** — file-based medallion warehouse
- **Fireworks AI, Gemma2-9b-it** (`accounts/fireworks/models/gemma2-9b-it`) — powers **all** LLM calls in the system (qualifies for the Gemma prize)
- **networkx + ast** — code knowledge graph; **pyvis** — graph visualization
- **Pydantic v2** — strict schemas as the only contract between agents
- **requests** — direct Jira / Slack / GitHub REST integration; **python-dotenv** — configuration
- **No agent frameworks** — orchestration is pure Python, by design

## Setup

### Docker (recommended)

```bash
cp .env.example .env    # DEMO_MODE=true works with zero credentials
docker-compose up
```

- Backend: http://localhost:8000 · Dashboard: http://localhost:8501
- The database initializes automatically, the five historical incidents seed automatically, and the code graph builds on the first audit run. No manual steps.

### Local development

```bash
python3.11 -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env

uvicorn main:app --port 8000            # terminal 1
streamlit run frontend/app.py           # terminal 2
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DEMO_MODE` | `true` | **The important one.** When true, Jira and Slack render as previews in the dashboard — no external calls, no credentials needed. |
| `FIREWORKS_API_KEY` | *(empty)* | Optional. Without it, deterministic fallback templates are used everywhere. Add a real key to get AI-written prose. |
| `FIREWORKS_MODEL` | `accounts/fireworks/models/gemma2-9b-it` | Used for every LLM call. |
| `DUCKDB_PATH` | `./database/routezero.db` | Database file location. |
| `JIRA_API_TOKEN` / `JIRA_BASE_URL` / `JIRA_EMAIL` | *(empty)* | Only needed with `DEMO_MODE=false`. |
| `SLACK_WEBHOOK_URL` | *(empty)* | Only needed with `DEMO_MODE=false`. |
| `GITHUB_TOKEN` / `GITHUB_REPO_OWNER` / `GITHUB_REPO_NAME` | *(empty)* | Optional; auditor falls back to reading `demo_repo/` locally. |

**TL;DR: `DEMO_MODE=true` demos the entire system with zero credentials.**

## Demo walkthrough

Four scenarios, in order, in the dashboard.

**Scenario 1 — Payment failure, rich context.** Paste `data/sample_errors/payment_error.txt` into the New Incident tab. In *Add context*: 847 affected users, enterprise tier, production, a deployment 37 minutes ago (any commit hash/message), SLA breach in 40 minutes. Expected: **P1 → payments-team**, manager digest triggered, probable cause linked to the deployment (commit, deployer, message cited), routing confidence above 0.85.

**Scenario 2 — Auth failure, minimal context.** Paste `data/sample_errors/auth_error.txt` with no context at all. Expected: **P1 → security-team** (auth is critical-path), routing confidence around 0.75, and the missing-context list showing every field that was not provided — the system tells you what it didn't know instead of guessing.

**Scenario 3 — ML timeout, medium context.** Paste `data/sample_errors/ml_timeout_error.txt` with affected users under 100 and environment set to staging. Expected: **P2 → ml-platform-team**, no manager digest, lower blast radius.

**Scenario 4 — The Auditor.** Open the Architectural Intelligence tab and click **Run Audit Now**. Using the five pre-seeded historical incidents, Agent 4 flags the recurring location at `payment_service/processor.py:31` — the null-reference bug that appears in three separate seeded incidents — with confidence **0.85**, creates a PLM ticket listing the contributing incident IDs, and the code graph renders that node in **red**.

## Deploying to Streamlit Community Cloud

1. Host the FastAPI backend anywhere reachable — Render, Railway, or Fly.io free tiers all work (`uvicorn main:app --host 0.0.0.0 --port 8000`).
2. On [Streamlit Community Cloud](https://share.streamlit.io), deploy this repo with **`frontend/app.py`** as the entrypoint.
3. In the Streamlit app's secrets/environment, set `BACKEND_URL` to the backend's public URL (and `DEMO_MODE=true`).

**Live demo:** `<STREAMLIT_CLOUD_URL_HERE>`

## Running the tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v    # Windows
python -m pytest tests/                          # anywhere
```

80+ tests cover the schemas, database layer, all four agents, integration clients (demo mode), the code graph, and every API endpoint.

## Repository structure

```
agents/         The four agents: classifier, router, ticket_writer, auditor
core/           Schemas (Pydantic v2), DuckDB manager, Fireworks client, code graph
integrations/   Jira, Slack, and GitHub clients (all DEMO_MODE-aware)
data/           StreamCo org config, 4 sample errors, 5 seeded historical incidents
demo_repo/      Fictional StreamCo codebase the graph is built from (incl. the bug at processor.py:31)
frontend/       Streamlit dashboard (app.py)
tests/          Pytest suite
docker/         Backend and frontend Dockerfiles
main.py         FastAPI app wiring the pipeline: POST /incidents, /audit, /approve, ...
```

## License

MIT — see [LICENSE](LICENSE).

---

*RouteZero — built from real engineering pain. Every output traceable to a verified input. Nothing invented.*
