"""Second-generation UCM vLLM connector components.

This package is intentionally isolated from the production connector while the
v2 contract is validated.  Importing it does not alter connector registration.
"""

from .ucm_connector import UCMConnector, UCMConnectorMetadata, UCMRuntimeContext
from .ucm_kv_cache import (
    UCMGroupTag,
    UCMKVCacheLayout,
    UCMKVCacheSpec,
    parse_kv_cache_config,
)
from .ucm_proxy import UCMProxy, UCMProxyAdapter
from .ucm_scheduler import UCMLookupCoordinator, UCMDispatcher

__all__ = [
    "UCMConnector",
    "UCMConnectorMetadata",
    "UCMDispatcher",
    "UCMGroupTag",
    "UCMKVCacheLayout",
    "UCMKVCacheSpec",
    "UCMLookupCoordinator",
    "UCMProxy",
    "UCMProxyAdapter",
    "UCMRuntimeContext",
    "parse_kv_cache_config",
]
