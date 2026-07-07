"""JWT bearer-token validation for the StreamCo auth service."""

import base64
import hmac
import json
import logging
import time
from typing import Any, Dict

from shared.database_pool import ConnectionPool

logger = logging.getLogger(__name__)


class InvalidTokenError(Exception):
    """Raised when a bearer token fails structure, signature, or expiry checks."""


class TokenValidator:
    """Validates HS256-signed JWTs issued by the StreamCo identity provider."""

    def __init__(self, signing_key: bytes, session_pool: ConnectionPool) -> None:
        self.signing_key = signing_key
        self.session_pool = session_pool

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Verify a bearer token's signature and expiry and return its claims."""
        claims = self._verify_signature(token)
        if claims.get("exp", 0) < time.time():
            raise InvalidTokenError(
                f"Signature has expired: token for subject {claims.get('sub')} "
                f"expired at {claims.get('exp')}"
            )
        self._refresh_session(claims)
        return claims

    def _verify_signature(self, token: str) -> Dict[str, Any]:
        """Recompute the HS256 digest and decode the claims segment."""
        try:
            header_b64, claims_b64, signature_b64 = token.split(".")
        except ValueError as exc:
            raise InvalidTokenError("Malformed token: expected three segments") from exc
        signing_input = f"{header_b64}.{claims_b64}".encode("ascii")
        expected = hmac.new(self.signing_key, signing_input, "sha256").digest()
        provided = base64.urlsafe_b64decode(signature_b64 + "==")
        if not hmac.compare_digest(expected, provided):
            raise InvalidTokenError("JWT signature verification failed: digest mismatch")
        return json.loads(base64.urlsafe_b64decode(claims_b64 + "=="))

    def _refresh_session(self, claims: Dict[str, Any]) -> None:
        """Record token use against the session row for audit trails."""
        conn = self.session_pool.acquire()
        try:
            conn.execute(
                "UPDATE sessions SET last_seen_at = CURRENT_TIMESTAMP WHERE session_id = %s",
                (claims.get("sid"),),
            )
        finally:
            self.session_pool.release(conn)
