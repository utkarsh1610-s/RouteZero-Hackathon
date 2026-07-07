"""Code knowledge graph for RouteZero.

Builds a graph of the demo_repo codebase using the Python ast module and
networkx. Modules, functions and classes become nodes; "contains",
"imports" and "calls" relationships become edges. Everything is persisted
through the DatabaseManager so subsequent runs load the graph from DuckDB
instead of re-parsing the files.

Node id conventions:
- module nodes:        the relative forward-slash path, e.g.
                       "payment_service/processor.py"
- function/class nodes: "<path>::<qualname>", e.g.
                       "payment_service/processor.py::PaymentProcessor.process_payment"
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx

from core.database import DatabaseManager
from core.schemas import GraphEdge, GraphNode, NodeType

logger = logging.getLogger(__name__)

# Repo root resolved from this module's location (core/ -> project root).
# Never assume the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ORG_CONFIG_PATH = _REPO_ROOT / "data" / "org_config.json"

_GIT_TIMEOUT_SECONDS = 5

# Deterministic owner fallback: top-level demo_repo folder -> org service.
# Used when git blame cannot attribute a file (demo_repo is uncommitted).
_FOLDER_TO_SERVICE = {
    "payment_service": "payment-service",
    "auth_service": "auth-service",
    "ml_service": "recommendation-engine",
    "shared": "user-profile-service",
    "video_delivery": "video-delivery",
}


def _normalize_path(file_path: str) -> str:
    """Normalize slashes and strip leading './' from a file path string."""
    norm = (file_path or "").replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _dotted_name(node: ast.AST) -> Optional[str]:
    """Flatten an Attribute/Name chain like shared.config into a dotted string."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _ParsedModule:
    """One parsed demo_repo file plus its collected definitions."""

    def __init__(self, rel_path: str, tree: ast.Module, line_count: int) -> None:
        self.rel_path = rel_path
        self.tree = tree
        self.line_count = line_count
        # Each entry: (qualname, ast def node, enclosing class qualname or None)
        self.defs: list[tuple[str, ast.AST, Optional[str]]] = []


class CodeGraphBuilder:
    """Builds, persists and queries the code knowledge graph."""

    def __init__(self, db: DatabaseManager, demo_repo_path: Optional[str] = None) -> None:
        self.db = db
        self.demo_repo_path = (
            Path(demo_repo_path) if demo_repo_path else _REPO_ROOT / "demo_repo"
        )
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._owner_cache: dict[str, Optional[str]] = {}
        self._fallback_owners: dict[str, str] = self._load_fallback_owners()

    # ------------------------------------------------------------------
    # Public API (pinned interface)
    # ------------------------------------------------------------------

    def build_or_load(self) -> None:
        """Load the graph from DuckDB if present, otherwise parse demo_repo.

        On first run every *.py under demo_repo is parsed with ast and the
        resulting nodes and edges are stored via the DatabaseManager. On
        subsequent runs (or with a warm in-memory graph) parsing is skipped.
        """
        if self._graph.number_of_nodes() > 0:
            logger.info(
                "Code graph already in memory (%d nodes); skipping rebuild",
                self._graph.number_of_nodes(),
            )
            return

        stored_nodes = self.db.get_graph_nodes()
        if stored_nodes:
            self._load_from_db(stored_nodes, self.db.get_graph_edges())
            return

        self._build_from_source()

    def get_node_at(self, file_path: str, line_number: int) -> Optional[GraphNode]:
        """Return the innermost function/class node covering a file line.

        ``file_path`` may be the exact relative path
        ("payment_service/processor.py") or any longer path that merely ends
        with that suffix. Falls back to the module node when the line is not
        inside any definition; returns None when the file is unknown.
        """
        self.build_or_load()
        norm = _normalize_path(file_path)

        module_node: Optional[GraphNode] = None
        best: Optional[GraphNode] = None
        for node in self._nodes.values():
            if norm != node.file_path and not norm.endswith("/" + node.file_path):
                continue
            if node.node_type == NodeType.MODULE:
                module_node = node
                continue
            if node.start_line <= line_number <= node.end_line:
                # Innermost definition wins: latest start, then earliest end.
                if (
                    best is None
                    or node.start_line > best.start_line
                    or (node.start_line == best.start_line and node.end_line < best.end_line)
                ):
                    best = node
        return best or module_node

    def get_subgraph(
        self, node_id: str, hops: int = 2
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return nodes within ``hops`` of ``node_id`` in BOTH edge directions,
        plus every stored edge whose endpoints are both in that node set."""
        self.build_or_load()
        if node_id not in self._graph:
            logger.warning("get_subgraph: unknown node_id %s", node_id)
            return [], []
        ego = nx.ego_graph(self._graph, node_id, radius=hops, undirected=True)
        node_ids = set(ego.nodes)
        nodes = [self._nodes[nid] for nid in sorted(node_ids) if nid in self._nodes]
        edges = [
            edge
            for _, edge in sorted(self._edges.items())
            if edge.source_node_id in node_ids and edge.target_node_id in node_ids
        ]
        return nodes, edges

    def map_incident_locations(self, incidents: list[dict]) -> dict[str, int]:
        """Map incident stack-trace locations onto graph nodes.

        For every incident dict (as returned by db.get_incidents_since, with
        stack_trace_locations already parsed to a list of dicts) each location
        is resolved to a node and upserted into graph_incident_node_mapping.
        Returns node_id -> distinct incident count for the whole table.
        """
        self.build_or_load()
        for incident in incidents:
            incident_id = incident.get("incident_id")
            if not incident_id:
                continue
            counter: Counter[str] = Counter()
            for location in incident.get("stack_trace_locations") or []:
                file_path = location.get("file_path")
                line_number = location.get("line_number")
                if not file_path or line_number is None:
                    continue
                node = self.get_node_at(file_path, int(line_number))
                if node is None:
                    logger.warning(
                        "No graph node found for incident %s location %s:%s",
                        incident_id,
                        file_path,
                        line_number,
                    )
                    continue
                counter[node.node_id] += 1
            for node_id, occurrence_count in counter.items():
                self.db.upsert_incident_node_mapping(incident_id, node_id, occurrence_count)
        return self.db.get_node_incident_counts()

    # ------------------------------------------------------------------
    # Loading from DuckDB
    # ------------------------------------------------------------------

    def _load_from_db(self, node_rows: list[dict], edge_rows: list[dict]) -> None:
        for row in node_rows:
            node = GraphNode(
                node_id=row["node_id"],
                name=row["name"],
                file_path=row["file_path"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                node_type=NodeType(row["node_type"]),
                owner=row.get("owner"),
                last_modified=row.get("last_modified"),
            )
            self._nodes[node.node_id] = node
            self._graph.add_node(node.node_id)
        for row in edge_rows:
            edge = GraphEdge(
                edge_id=row["edge_id"],
                source_node_id=row["source_node_id"],
                target_node_id=row["target_node_id"],
                relationship_type=row["relationship_type"],
                weight=float(row.get("weight") or 1.0),
            )
            self._edges[edge.edge_id] = edge
            self._graph.add_edge(
                edge.source_node_id,
                edge.target_node_id,
                relationship_type=edge.relationship_type,
            )
        logger.info(
            "Code graph loaded from DuckDB: %d nodes, %d edges (parse skipped)",
            len(self._nodes),
            len(self._edges),
        )

    # ------------------------------------------------------------------
    # Building from source with ast
    # ------------------------------------------------------------------

    def _build_from_source(self) -> None:
        if not self.demo_repo_path.is_dir():
            logger.error("demo_repo folder not found at %s; graph is empty", self.demo_repo_path)
            return

        logger.info("Building code graph from source at %s", self.demo_repo_path)
        parsed_modules = self._parse_all_files()

        # Pass 1 created every node; pass 2 wires import and call edges,
        # which needs the full node set to resolve cross-module targets.
        module_files = {module.rel_path for module in parsed_modules}
        for module in parsed_modules:
            alias_defs, alias_modules = self._process_imports(module, module_files)
            self._process_calls(module, alias_defs, alias_modules)

        self.db.insert_graph_nodes(list(self._nodes.values()))
        self.db.insert_graph_edges(list(self._edges.values()))
        logger.info(
            "Code graph built: %d nodes, %d edges from %d modules",
            len(self._nodes),
            len(self._edges),
            len(parsed_modules),
        )

    def _parse_all_files(self) -> list[_ParsedModule]:
        parsed: list[_ParsedModule] = []
        for py_file in sorted(self.demo_repo_path.rglob("*.py")):
            rel_path = py_file.relative_to(self.demo_repo_path).as_posix()
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (OSError, UnicodeDecodeError, SyntaxError):
                logger.error("Could not parse %s; skipping file", py_file, exc_info=True)
                continue

            line_count = max(1, len(source.splitlines()))
            module = _ParsedModule(rel_path, tree, line_count)
            owner = self._owner_for(rel_path)
            last_modified = self._last_modified_for(py_file)

            self._add_node(
                GraphNode(
                    node_id=rel_path,
                    name=py_file.stem,
                    file_path=rel_path,
                    start_line=1,
                    end_line=line_count,
                    node_type=NodeType.MODULE,
                    owner=owner,
                    last_modified=last_modified,
                )
            )

            self._collect_defs(tree, [], None, module)
            for qualname, def_node, _ in module.defs:
                start_line = min(
                    [def_node.lineno]
                    + [decorator.lineno for decorator in getattr(def_node, "decorator_list", [])]
                )
                end_line = getattr(def_node, "end_lineno", None) or def_node.lineno
                node_type = (
                    NodeType.CLASS if isinstance(def_node, ast.ClassDef) else NodeType.FUNCTION
                )
                node_id = f"{rel_path}::{qualname}"
                self._add_node(
                    GraphNode(
                        node_id=node_id,
                        name=def_node.name,
                        file_path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        node_type=node_type,
                        owner=owner,
                        last_modified=last_modified,
                    )
                )
                self._add_edge(rel_path, node_id, "contains")
            parsed.append(module)
        return parsed

    def _collect_defs(
        self,
        node: ast.AST,
        stack: list[str],
        enclosing_class: Optional[str],
        module: _ParsedModule,
    ) -> None:
        """Recursively collect function/class definitions with qualnames."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = ".".join(stack + [child.name])
                module.defs.append((qualname, child, enclosing_class))
                child_class = qualname if isinstance(child, ast.ClassDef) else enclosing_class
                self._collect_defs(child, stack + [child.name], child_class, module)
            else:
                self._collect_defs(child, stack, enclosing_class, module)

    # ------------------------------------------------------------------
    # Import edges
    # ------------------------------------------------------------------

    def _process_imports(
        self, module: _ParsedModule, module_files: set[str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Add "imports" edges and return alias lookup tables for call resolution.

        Returns (alias_defs, alias_modules):
        - alias_defs maps a local name to a def node id in another module,
          e.g. "ConnectionPool" -> "shared/database_pool.py::ConnectionPool".
        - alias_modules maps a local dotted alias to a module node id,
          e.g. "shared.config" -> "shared/config.py".
        """
        alias_defs: dict[str, str] = {}
        alias_modules: dict[str, str] = {}

        for statement in ast.walk(module.tree):
            if isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
                target_file = statement.module.replace(".", "/") + ".py"
                if target_file in module_files:
                    self._add_edge(module.rel_path, target_file, "imports")
                    for alias in statement.names:
                        local = alias.asname or alias.name
                        def_id = f"{target_file}::{alias.name}"
                        if def_id in self._nodes:
                            alias_defs[local] = def_id
                        else:
                            alias_modules[local] = target_file
                else:
                    # `from shared import config` style: package + module name.
                    for alias in statement.names:
                        candidate = statement.module.replace(".", "/") + f"/{alias.name}.py"
                        if candidate in module_files:
                            self._add_edge(module.rel_path, candidate, "imports")
                            alias_modules[alias.asname or alias.name] = candidate
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    candidate = alias.name.replace(".", "/") + ".py"
                    if candidate in module_files:
                        self._add_edge(module.rel_path, candidate, "imports")
                        alias_modules[alias.asname or alias.name] = candidate
        return alias_defs, alias_modules

    # ------------------------------------------------------------------
    # Call edges (best-effort static analysis)
    # ------------------------------------------------------------------

    def _process_calls(
        self,
        module: _ParsedModule,
        alias_defs: dict[str, str],
        alias_modules: dict[str, str],
    ) -> None:
        top_defs = {
            qualname: f"{module.rel_path}::{qualname}"
            for qualname, _, _ in module.defs
            if "." not in qualname
        }
        for qualname, def_node, enclosing_class in module.defs:
            if isinstance(def_node, ast.ClassDef):
                continue
            source_id = f"{module.rel_path}::{qualname}"
            for call in self._iter_own_calls(def_node):
                target_id = self._resolve_call(
                    call, module.rel_path, enclosing_class, top_defs, alias_defs, alias_modules
                )
                if target_id and target_id != source_id:
                    self._add_edge(source_id, target_id, "calls")

    @staticmethod
    def _iter_own_calls(def_node: ast.AST):
        """Yield ast.Call nodes in a def body without descending into nested defs."""
        pending = list(ast.iter_child_nodes(def_node))
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Call):
                yield node
            pending.extend(ast.iter_child_nodes(node))

    def _resolve_call(
        self,
        call: ast.Call,
        rel_path: str,
        enclosing_class: Optional[str],
        top_defs: dict[str, str],
        alias_defs: dict[str, str],
        alias_modules: dict[str, str],
    ) -> Optional[str]:
        """Resolve a call target to a node id, or None when ambiguous."""
        func = call.func
        if isinstance(func, ast.Name):
            return top_defs.get(func.id) or alias_defs.get(func.id)
        if isinstance(func, ast.Attribute):
            # self.method() within a class body.
            if (
                isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and enclosing_class
            ):
                candidate = f"{rel_path}::{enclosing_class}.{func.attr}"
                return candidate if candidate in self._nodes else None
            # module_alias.function() for an imported demo_repo module.
            dotted = _dotted_name(func.value)
            if dotted and dotted in alias_modules:
                candidate = f"{alias_modules[dotted]}::{func.attr}"
                return candidate if candidate in self._nodes else None
        return None

    # ------------------------------------------------------------------
    # Graph mutation helpers
    # ------------------------------------------------------------------

    def _add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id)

    def _add_edge(self, source_id: str, target_id: str, relationship_type: str) -> None:
        edge_id = f"{relationship_type}:{source_id}->{target_id}"
        if edge_id in self._edges:
            return
        self._edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_type=relationship_type,
            weight=1.0,
        )
        self._graph.add_edge(source_id, target_id, relationship_type=relationship_type)

    # ------------------------------------------------------------------
    # Owner attribution
    # ------------------------------------------------------------------

    def _owner_for(self, rel_path: str) -> Optional[str]:
        """Owner attribution disabled for demo — returns None for all nodes."""
        return None

    def _git_owner(self, rel_path: str) -> Optional[str]:
        """Return the git blame author for a file, or None when unavailable.

        demo_repo is intentionally uncommitted in the hackathon repo, so this
        is expected to fail and fall back to the deterministic mapping.
        """
        if rel_path in self._owner_cache:
            return self._owner_cache[rel_path]
        owner: Optional[str] = None
        try:
            result = subprocess.run(
                ["git", "blame", "--porcelain", "-L", "1,1", "--", rel_path],
                cwd=str(self.demo_repo_path),
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("author "):
                        candidate = line[len("author ") :].strip()
                        if candidate and candidate != "Not Committed Yet":
                            owner = candidate
                        break
        except Exception:  # noqa: BLE001 - blame must never break graph building
            logger.warning("git blame failed for %s; using fallback owner", rel_path)
        self._owner_cache[rel_path] = owner
        return owner

    @staticmethod
    def _last_modified_for(py_file: Path) -> Optional[datetime]:
        try:
            return datetime.fromtimestamp(py_file.stat().st_mtime)
        except OSError:
            return None

    @staticmethod
    def _load_fallback_owners() -> dict[str, str]:
        """Map top-level demo_repo folders to team default Jira assignees."""
        try:
            config = json.loads(_ORG_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Org config not readable at %s; no fallback owners", _ORG_CONFIG_PATH)
            return {}
        services = config.get("services") or {}
        teams = config.get("teams") or {}
        owners: dict[str, str] = {}
        for folder, service_name in _FOLDER_TO_SERVICE.items():
            team_name = (services.get(service_name) or {}).get("owning_team")
            assignee = (teams.get(team_name) or {}).get("default_jira_assignee")
            if assignee:
                owners[folder] = assignee
        return owners
