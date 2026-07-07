"""Environment-aware configuration shared by all StreamCo services."""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "payment-service": {
        "gateway_base_url": "https://gateway.payments.internal",
        "gateway_timeout_seconds": 5.0,
    },
    "auth-service": {
        "signing_key_env": "AUTH_SIGNING_KEY",
        "token_ttl_seconds": 3600,
    },
    "recommendation-engine": {
        "inference_url": "http://ml-inference.internal/v1/rank",
        "inference_timeout_ms": 2500,
    },
}


def get_setting(service: str, key: str) -> Any:
    """Return a service setting, honouring SERVICE__KEY environment overrides."""
    env_name = f"{service}__{key}".upper().replace("-", "_")
    if env_name in os.environ:
        logger.debug("Config override %s taken from environment", env_name)
        return os.environ[env_name]
    return _DEFAULTS.get(service, {})[key]
