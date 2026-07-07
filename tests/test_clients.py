"""Offline tests for the RouteZero integration and Fireworks clients.

All tests run without any network access. Network entry points are
monkeypatched so an accidental real call fails the test immediately.
"""

import re

import pytest
import requests

from core.fireworks_client import DEFAULT_MODEL, FireworksClient
from integrations.github_client import GitHubClient
from integrations.jira_client import JiraClient
from integrations.slack_client import SlackClient


def _no_network(*args, **kwargs):
    raise AssertionError("A real network call was attempted during tests.")


# ---------------------------------------------------------------------------
# Jira client (demo mode)
# ---------------------------------------------------------------------------


def test_demo_jira_create_issue_returns_realistic_ticket(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr("integrations.jira_client.requests.post", _no_network)

    result = JiraClient().create_issue(
        project_key="PAY",
        summary="NullPointer in payment processor",
        description="Payment charge crashed at processor.py line 31.",
        priority="P1",
        labels=["incident", "routezero"],
        assignee="alice",
    )

    assert re.fullmatch(r"PAY-\d{4}", result["ticket_id"])
    assert result["url"] == (
        f"https://streamco-demo.atlassian.net/browse/{result['ticket_id']}"
    )
    assert "error" not in result


# ---------------------------------------------------------------------------
# Slack client (demo mode)
# ---------------------------------------------------------------------------


def test_demo_slack_send_returns_true(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr("integrations.slack_client.requests.post", _no_network)

    assert SlackClient().send("#payments-alerts", "P1 incident routed") is True


# ---------------------------------------------------------------------------
# Fireworks client
# ---------------------------------------------------------------------------


def test_fireworks_without_api_key_returns_empty_without_network(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr("core.fireworks_client.requests.post", _no_network)

    client = FireworksClient()
    assert client.complete("hello") == ""
    assert client.complete_json("hello") == {}
    assert client.call_count == 0


def test_fireworks_with_blank_api_key_returns_empty_without_network(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "   ")
    monkeypatch.setattr("core.fireworks_client.requests.post", _no_network)

    client = FireworksClient()
    assert client.complete("hello") == ""
    assert client.complete_json("hello") == {}


def test_fireworks_uses_gemma_model_by_default(monkeypatch):
    monkeypatch.delenv("FIREWORKS_MODEL", raising=False)
    client = FireworksClient()
    assert client.model == "accounts/fireworks/models/gemma2-9b-it"
    assert client.model == DEFAULT_MODEL


def test_parse_json_strips_code_fences():
    assert FireworksClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_handles_plain_json_and_garbage():
    assert FireworksClient._parse_json('{"b": 2}') == {"b": 2}
    assert FireworksClient._parse_json("not json at all") == {}
    assert FireworksClient._parse_json("") == {}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = str(payload)

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fireworks_complete_success_increments_call_count(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-key")
    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {"choices": [{"message": {"content": "hello world"}}]}
        )

    monkeypatch.setattr("core.fireworks_client.requests.post", _fake_post)

    client = FireworksClient()
    assert client.complete("hi") == "hello world"
    assert client.call_count == 1
    assert captured["json"]["model"] == "accounts/fireworks/models/gemma2-9b-it"
    assert captured["json"]["temperature"] == 0.1
    assert captured["timeout"] == 30


def test_fireworks_retries_three_times_on_timeout_then_raises(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-key")
    attempts = {"count": 0}

    def _always_timeout(*args, **kwargs):
        attempts["count"] += 1
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr("core.fireworks_client.requests.post", _always_timeout)

    client = FireworksClient()
    with pytest.raises(requests.exceptions.Timeout):
        client.complete("hi")
    assert attempts["count"] == 3
    # Timed-out attempts never returned, so they do not count as calls.
    assert client.call_count == 0


# ---------------------------------------------------------------------------
# GitHub client (demo mode)
# ---------------------------------------------------------------------------


def test_github_demo_mode_reads_real_file(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr("integrations.github_client.requests.get", _no_network)

    client = GitHubClient()
    real_file = client.demo_repo_path / "payment_service" / "processor.py"
    real_lines = (
        len(real_file.read_text(encoding="utf-8").splitlines())
        if real_file.exists()
        else 0
    )

    if real_lines > 0:
        # demo_repo is being written by another agent; use it when present.
        target_line = min(31, real_lines)
    else:
        # Fall back to a tmp_path fixture with a monkeypatched repo root.
        fake_repo = tmp_path / "demo_repo"
        (fake_repo / "payment_service").mkdir(parents=True)
        content = "\n".join(f"line {i}" for i in range(1, 41))
        (fake_repo / "payment_service" / "processor.py").write_text(
            content, encoding="utf-8"
        )
        monkeypatch.setattr(client, "demo_repo_path", fake_repo)
        target_line = 31

    snippet = client.get_file_content_at_line(
        "payment_service/processor.py", target_line, context_lines=3
    )

    assert snippet
    lines = snippet.splitlines()
    assert 1 <= len(lines) <= 7  # up to 3 lines of context on each side
    # Exactly one line is marked as the target, with its line number.
    marked = [line for line in lines if line.startswith("> ")]
    assert len(marked) == 1
    assert re.match(rf"^> +{target_line} \| ", marked[0])
    # Every other line uses the two-space marker and a line number.
    for line in lines:
        if not line.startswith("> "):
            assert re.match(r"^  +\d+ \| ", line)


def test_github_demo_mode_missing_file_returns_empty(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr("integrations.github_client.requests.get", _no_network)

    client = GitHubClient()
    result = client.get_file_content_at_line("nonexistent/never_there.py", 10)
    assert result == ""
