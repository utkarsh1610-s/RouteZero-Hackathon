"""Jira integration client for RouteZero.

In DEMO_MODE the client never touches the network: it returns a realistic
fake ticket so the full pipeline and Streamlit preview work without any
credentials. In live mode it creates real issues via the Jira REST API v3
and falls back to a demo-style response on any failure.
"""

import logging
import os
import random

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEMO_BROWSE_BASE_URL = "https://streamco-demo.atlassian.net/browse"
REQUEST_TIMEOUT_SECONDS = 15


def _demo_mode() -> bool:
    """True when DEMO_MODE is 'true'/'1'/'yes' (case-insensitive, default true)."""
    return os.getenv("DEMO_MODE", "true").strip().lower() in ("true", "1", "yes")


class JiraClient:
    """Creates Jira issues, with a fully offline demo mode."""

    def create_issue(
        self,
        project_key: str,
        summary: str,
        description: str,
        priority: str,
        labels: list[str],
        assignee: str,
    ) -> dict:
        """Create a Jira issue and return {"ticket_id": ..., "url": ...}.

        In demo mode returns a fake but realistic ticket without any network
        call. In live mode, any failure falls back to a demo-style response
        with an added "error" key so the pipeline never crashes.
        """
        if _demo_mode():
            logger.warning("DEMO_MODE enabled: bypassing real Jira API call.")
            result = self._demo_response(project_key)
            logger.info(
                "demo mode bypass: simulated Jira ticket %s for project %s "
                "(summary=%r, priority=%s, assignee=%s, labels=%s).",
                result["ticket_id"],
                project_key,
                summary,
                priority,
                assignee,
                labels,
            )
            return result

        base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
        email = os.getenv("JIRA_EMAIL", "")
        api_token = os.getenv("JIRA_API_TOKEN", "")

        # Simplest single-paragraph Atlassian Document Format description.
        adf_description = {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description}],
                }
            ],
        }
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": adf_description,
                "issuetype": {"name": "Task"},
                "priority": {"name": priority},
                "labels": labels,
                "assignee": {"name": assignee},
            }
        }

        try:
            response = requests.post(
                f"{base_url}/rest/api/3/issue",
                json=payload,
                auth=(email, api_token),
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            ticket_id = data["key"]
            url = f"{base_url}/browse/{ticket_id}"
            logger.info("Jira ticket %s created successfully.", ticket_id)
            return {"ticket_id": ticket_id, "url": url}
        except Exception as exc:  # noqa: BLE001 - integrations must never crash the pipeline
            logger.error(
                "Jira issue creation failed for project %s: %s. Falling back "
                "to demo-style response.",
                project_key,
                exc,
            )
            result = self._demo_response(project_key)
            result["error"] = str(exc)
            return result

    @staticmethod
    def _demo_response(project_key: str) -> dict:
        """Build a fake but realistic ticket id and browse URL."""
        ticket_id = f"{project_key}-{random.randint(1000, 9999)}"
        return {
            "ticket_id": ticket_id,
            "url": f"{DEMO_BROWSE_BASE_URL}/{ticket_id}",
        }
