"""Slack integration client for RouteZero.

In DEMO_MODE the client logs the channel and full message that would have
been sent and reports success, so the pipeline runs fully offline. In live
mode it posts to the configured Slack incoming webhook.
"""

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10


def _demo_mode() -> bool:
    """True when DEMO_MODE is 'true'/'1'/'yes' (case-insensitive, default true)."""
    return os.getenv("DEMO_MODE", "true").strip().lower() in ("true", "1", "yes")


class SlackClient:
    """Sends Slack notifications, with a fully offline demo mode."""

    def send(self, channel: str, message: str) -> bool:
        """Send ``message`` to ``channel``. Returns True on success.

        In demo mode no network call is made; the message that would have
        been sent is logged and True is returned. In live mode any failure
        is logged at ERROR level and False is returned.
        """
        if _demo_mode():
            logger.warning("DEMO_MODE enabled: bypassing real Slack API call.")
            logger.info(
                "demo mode bypass: Slack message to channel %s would have "
                "been:\n%s",
                channel,
                message,
            )
            return True

        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        if not webhook_url:
            logger.error(
                "SLACK_WEBHOOK_URL is not set; cannot send Slack message to "
                "channel %s.",
                channel,
            )
            return False

        try:
            response = requests.post(
                webhook_url,
                json={"channel": channel, "text": message},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            logger.info("Slack notification sent to channel %s.", channel)
            return True
        except Exception as exc:  # noqa: BLE001 - integrations must never crash the pipeline
            logger.error(
                "Slack notification to channel %s failed: %s", channel, exc
            )
            return False
