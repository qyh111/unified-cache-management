"""Classify native specs without requiring an engine-side UCM patch."""

from enum import Enum


class KVCacheSpecKind(str, Enum):
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    MAMBA = "mamba"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    UNKNOWN = "unknown"


def get_kv_cache_spec_kind(spec):
    # Ascend specs inherit the native interfaces; check subclasses before FA.
    names = {cls.__name__ for cls in type(spec).__mro__}
    if "MambaSpec" in names:
        return KVCacheSpecKind.MAMBA
    if "SlidingWindowMLASpec" in names or "AscendSlidingWindowMLASpec" in names:
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if "SlidingWindowSpec" in names or getattr(spec, "sliding_window", None):
        return KVCacheSpecKind.SLIDING_WINDOW
    if names & {
        "ChunkedLocalAttentionSpec",
        "CrossAttentionSpec",
        "EncoderOnlyAttentionSpec",
    }:
        return KVCacheSpecKind.UNKNOWN
    if getattr(spec, "attention_chunk_size", None):
        return KVCacheSpecKind.UNKNOWN
    if "MLAAttentionSpec" in names:
        return KVCacheSpecKind.MLA_ATTENTION
    if "FullAttentionSpec" in names:
        return KVCacheSpecKind.FULL_ATTENTION
    return KVCacheSpecKind.UNKNOWN
