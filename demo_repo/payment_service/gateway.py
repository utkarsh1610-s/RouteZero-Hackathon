"""HTTP client for StreamCo's upstream card payment gateway."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewayResponse:
    """Successful charge acknowledgement returned by the gateway."""

    transaction_id: str
    status: str
    amount_cents: int
    currency: str


class PaymentGatewayClient:
    """Thin wrapper over the gateway REST API.

    Both ``charge`` and ``refund`` return ``None`` when the gateway declines,
    times out, or responds with a server error. Callers are responsible for
    handling the ``None`` case explicitly.
    """

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def charge(self, customer_id: str, amount_cents: int, currency: str) -> Optional[GatewayResponse]:
        """Attempt to charge the customer's stored payment method."""
        payload = {"customer": customer_id, "amount": amount_cents, "currency": currency}
        body = self._post("/v1/charges", payload)
        if body is None or body.get("status") != "succeeded":
            logger.warning("Charge declined or failed for customer %s", customer_id)
            return None
        return GatewayResponse(
            transaction_id=body["id"],
            status=body["status"],
            amount_cents=amount_cents,
            currency=currency,
        )

    def refund(self, transaction_id: str, amount_cents: int) -> Optional[Dict[str, Any]]:
        """Refund part or all of a prior charge."""
        return self._post("/v1/refunds", {"transaction": transaction_id, "amount": amount_cents})

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST to the gateway and return the JSON body, or None on any failure."""
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("Gateway request to %s failed: %s", url, exc)
            return None
        if response.status_code >= 500:
            logger.warning("Gateway returned HTTP %s for %s", response.status_code, url)
            return None
        return response.json()
