"""Fireworks AI client for RouteZero.

All Fireworks API calls go through this single client. No agent calls the
Fireworks API directly. Every call uses the Gemma2-9b-it model, which is
non-negotiable because it qualifies for the hackathon's Gemma prize.
"""

import json
import logging
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FIREWORKS_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/gemma2-9b-it"
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3

_JSON_ONLY_INSTRUCTION = (
    "\n\nRespond with valid JSON only. Do not use markdown formatting or code "
    "fences. Do not include any explanation or text outside the JSON."
)


class FireworksClient:
    """Client for the Fireworks AI chat completions API.

    Exposes two public methods: ``complete`` for plain text completions and
    ``complete_json`` for structured JSON completions. Maintains a per-instance
    ``call_count`` so the dashboard can display how many Fireworks calls have
    been made this session.
    """

    def __init__(self) -> None:
        self.model: str = os.getenv("FIREWORKS_MODEL") or DEFAULT_MODEL
        self.call_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Return a plain-text completion for ``prompt``.

        Returns an empty string without any network call when no API key is
        configured, so the pipeline stays fully functional in demo
        environments.
        """
        if not self._api_key():
            logger.warning(
                "FIREWORKS_API_KEY is not set; skipping Fireworks call and "
                "returning empty text."
            )
            return ""
        return self._call(
            prompt, system=system, max_tokens=max_tokens, temperature=0.1
        )

    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> dict:
        """Return a completion parsed as a JSON object.

        Appends an instruction to respond with valid JSON only, then parses
        the response defensively. Returns an empty dict without any network
        call when no API key is configured, and an empty dict (after logging
        the raw response at ERROR level) when the response cannot be parsed.
        """
        if not self._api_key():
            logger.warning(
                "FIREWORKS_API_KEY is not set; skipping Fireworks call and "
                "returning empty dict."
            )
            return {}
        raw = self._call(
            prompt + _JSON_ONLY_INSTRUCTION,
            system=system,
            max_tokens=max_tokens,
            temperature=0.05,
        )
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _api_key() -> str:
        return (os.getenv("FIREWORKS_API_KEY") or "").strip()

    def _call(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Make one chat-completion call with retry-on-timeout semantics.

        Retries up to ``MAX_ATTEMPTS`` times on timeout (30 second timeout per
        attempt) before re-raising. ``call_count`` is incremented on every API
        call attempt that returns a response.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    FIREWORKS_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout:
                if attempt < MAX_ATTEMPTS:
                    logger.warning(
                        "Fireworks call timed out (attempt %d of %d); retrying.",
                        attempt,
                        MAX_ATTEMPTS,
                    )
                    continue
                logger.error(
                    "Fireworks call timed out after %d attempts; giving up.",
                    MAX_ATTEMPTS,
                )
                raise
            except requests.exceptions.RequestException as exc:
                logger.error("Fireworks request failed: %s", exc)
                raise

            # The attempt returned a response, so it counts as an API call.
            self.call_count += 1

            try:
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as exc:
                logger.error("Fireworks API returned an error response: %s", exc)
                raise
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                logger.error(
                    "Fireworks response had an unexpected shape (%s). Raw "
                    "response text: %r",
                    exc,
                    getattr(response, "text", ""),
                )
                return ""

            logger.info(
                "Fireworks call succeeded (model=%s, session call count=%d).",
                self.model,
                self.call_count,
            )
            return content if isinstance(content, str) else ""

        return ""  # Unreachable: the loop always returns or raises.

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove a surrounding markdown code fence (```json ... ```)."""
        cleaned = (text or "").strip()
        match = re.match(
            r"^```[a-zA-Z0-9_-]*[ \t]*\n?(.*?)\n?[ \t]*```$",
            cleaned,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return cleaned

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Defensively parse ``raw`` into a dict.

        Strips markdown code fences before parsing. On parse failure, logs the
        raw response at ERROR level and returns an empty dict instead of
        raising.
        """
        cleaned = FireworksClient._strip_code_fences(raw)

        candidates = [cleaned]
        # Last-resort fallback: try the substring between the outermost braces
        # in case the model wrapped the JSON in stray prose.
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidate = cleaned[first_brace : last_brace + 1]
            if candidate != cleaned:
                candidates.append(candidate)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed

        logger.error(
            "Could not parse Fireworks response as JSON. Raw response: %r", raw
        )
        return {}
