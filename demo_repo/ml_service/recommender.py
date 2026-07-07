"""Personalized title recommendations served by StreamCo's ML platform."""

import logging
from typing import Any, Dict, List

import requests

from shared.config import get_setting
from shared.database_pool import ConnectionPool

logger = logging.getLogger(__name__)


class Recommender:
    """Ranks catalog titles for a user via the shared model-inference endpoint."""

    def __init__(self, pool: ConnectionPool, model_version: str = "ranker-v14") -> None:
        self.pool = pool
        self.model_version = model_version
        self.inference_url = get_setting("recommendation-engine", "inference_url")
        self.timeout_ms = get_setting("recommendation-engine", "inference_timeout_ms")

    def get_recommendations(self, user_id: str, row_size: int = 20) -> List[Dict[str, Any]]:
        """Return the top ``row_size`` ranked titles for a user.

        Feature vectors come from the feature store; candidate ranking is
        delegated to the shared model-inference endpoint.
        """
        logger.info("Ranking titles for user %s with model %s", user_id, self.model_version)
        features = self._load_user_features(user_id)
        candidates = self._candidate_titles(user_id, limit=row_size * 10)
        if not candidates:
            logger.warning("No candidates for user %s; returning empty row", user_id)
            return []
        payload = {
            "model": self.model_version,
            "user_features": features,
            "candidates": candidates,
            "top_k": row_size,
        }
        logger.debug("Sending %d candidates to inference", len(candidates))
        ranked = self._call_inference(payload)
        return ranked[:row_size]

    def _call_inference(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """POST the ranking payload to the model-inference endpoint."""
        timeout_seconds = self.timeout_ms / 1000.0
        try:
            response = requests.post(self.inference_url, json=payload, timeout=timeout_seconds)
        except requests.Timeout as exc:
            raise TimeoutError(
                f"Model inference request to {self.inference_url} timed out "
                f"after {self.timeout_ms}ms (model={self.model_version})"
            ) from exc
        response.raise_for_status()
        return response.json().get("ranked", [])

    def _load_user_features(self, user_id: str) -> Dict[str, Any]:
        """Fetch the precomputed feature vector for a user from the feature store."""
        conn = self.pool.acquire()
        try:
            row = conn.execute(
                "SELECT features FROM user_features WHERE user_id = %s", (user_id,)
            ).fetchone()
        finally:
            self.pool.release(conn)
        return row[0] if row else {}

    def _candidate_titles(self, user_id: str, limit: int) -> List[str]:
        """Return recently added titles the user has not watched yet."""
        conn = self.pool.acquire()
        try:
            rows = conn.execute(
                "SELECT title_id FROM catalog_candidates WHERE user_id = %s LIMIT %s",
                (user_id, limit),
            ).fetchall()
        finally:
            self.pool.release(conn)
        return [r[0] for r in rows]
