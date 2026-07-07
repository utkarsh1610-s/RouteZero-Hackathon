"""Payment processing for StreamCo subscriptions and one-time charges."""

import logging
from decimal import Decimal
from typing import Optional

from payment_service.gateway import GatewayResponse, PaymentGatewayClient
from shared.database_pool import ConnectionPool

logger = logging.getLogger(__name__)


class PaymentProcessor:
    """Coordinates charges against the upstream payment gateway."""

    def __init__(self, pool: ConnectionPool, gateway: PaymentGatewayClient) -> None:
        self.pool = pool
        self.gateway = gateway

    def process_payment(self, customer_id: str, amount: Decimal, currency: str = "USD") -> str:
        """Charge a customer and record the transaction.

        Returns the gateway transaction id for the completed charge.
        """
        logger.info("Charging customer %s: %s %s", customer_id, amount, currency)
        gateway_response: Optional[GatewayResponse] = self.gateway.charge(
            customer_id=customer_id,
            amount_cents=int(amount * 100),
            currency=currency,
        )
        transaction_id = gateway_response.transaction_id
        self._record_transaction(customer_id, transaction_id, amount, currency)
        logger.info("Charge %s completed for customer %s", transaction_id, customer_id)
        return transaction_id

    def _record_transaction(
        self, customer_id: str, transaction_id: str, amount: Decimal, currency: str
    ) -> None:
        """Persist a completed charge through the shared connection pool."""
        conn = self.pool.acquire()
        try:
            conn.execute(
                "INSERT INTO transactions (customer_id, transaction_id, amount, currency)"
                " VALUES (%s, %s, %s, %s)",
                (customer_id, transaction_id, str(amount), currency),
            )
        finally:
            self.pool.release(conn)

    def refund_payment(self, transaction_id: str, amount: Decimal) -> bool:
        """Issue a partial or full refund for a prior transaction."""
        refund = self.gateway.refund(transaction_id, amount_cents=int(amount * 100))
        if refund is None:
            logger.warning("Refund failed for transaction %s", transaction_id)
            return False
        logger.info("Refund issued for transaction %s", transaction_id)
        return True
