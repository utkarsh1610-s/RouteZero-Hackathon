"""GitHub integration client for RouteZero.

Fetches the code around a specific file line so Agent 4 can show Fireworks
the actual code that keeps failing. In DEMO_MODE it reads directly from the
local demo_repo folder. In live mode it calls the GitHub REST API and always
falls back to the local demo_repo on any failure.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15

# Repo root computed from this file's location (integrations/ -> project root).
# Never assume the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _demo_mode() -> bool:
    """True when DEMO_MODE is 'true'/'1'/'yes' (case-insensitive, default true)."""
    return os.getenv("DEMO_MODE", "true").strip().lower() in ("true", "1", "yes")


class GitHubClient:
    """Reads file content around a line, from GitHub or the local demo_repo."""

    def __init__(self) -> None:
        self.demo_repo_path: Path = _REPO_ROOT / "demo_repo"

    def get_file_content_at_line(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 10,
    ) -> str:
        """Return the lines around ``line_number`` with line numbers prepended.

        ``file_path`` is relative with forward slashes, e.g.
        "payment_service/processor.py". The target line is marked with "> ".
        Returns an empty string when the file cannot be read anywhere.
        """
        if _demo_mode():
            logger.warning(
                "DEMO_MODE enabled: bypassing GitHub API call; reading %s "
                "from local demo_repo.",
                file_path,
            )
            return self._read_local(file_path, line_number, context_lines)

        lines = self._fetch_remote_lines(file_path)
        if lines is None:
            # Always fall back to the local demo_repo on any API failure.
            return self._read_local(file_path, line_number, context_lines)
        return self._format_window(lines, line_number, context_lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_local(
        self, file_path: str, line_number: int, context_lines: int
    ) -> str:
        local_path = self.demo_repo_path.joinpath(*file_path.split("/"))
        try:
            text = local_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "Local demo_repo file %s could not be read (%s); returning "
                "empty content.",
                local_path,
                exc,
            )
            return ""
        return self._format_window(text.splitlines(), line_number, context_lines)

    def _fetch_remote_lines(self, file_path: str) -> Optional[list[str]]:
        """Fetch and decode file content via the GitHub REST API.

        Returns the file's lines, or None on any failure so the caller can
        fall back to the local demo_repo.
        """
        owner = os.getenv("GITHUB_REPO_OWNER", "")
        repo_name = os.getenv("GITHUB_REPO_NAME", "")
        token = os.getenv("GITHUB_TOKEN", "").strip()

        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.get(
                url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            payload = response.json()
            content = base64.b64decode(payload["content"]).decode(
                "utf-8", errors="replace"
            )
            logger.info("Fetched %s from GitHub API.", file_path)
            return content.splitlines()
        except Exception as exc:  # noqa: BLE001 - integrations must never crash the pipeline
            logger.error(
                "GitHub API fetch for %s failed: %s. Falling back to local "
                "demo_repo.",
                file_path,
                exc,
            )
            return None

    @staticmethod
    def _format_window(
        lines: list[str], line_number: int, context_lines: int
    ) -> str:
        """Format the lines around ``line_number`` with line numbers prepended.

        The target line is marked with "> ", all others with two spaces, e.g.:

            29 | balance = account.balance
          > 31 | account.charge(amount)
        """
        if not lines:
            return ""
        total = len(lines)
        target = min(max(line_number, 1), total)
        start = max(target - context_lines, 1)
        end = min(target + context_lines, total)
        width = len(str(end))
        formatted = []
        for num in range(start, end + 1):
            marker = "> " if num == target else "  "
            formatted.append(f"{marker}{num:>{width}} | {lines[num - 1]}")
        return "\n".join(formatted)
