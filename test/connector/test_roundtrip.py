"""Read first: actual EngineCore steps save, close, load, and compare every rank."""

import pytest

pytestmark = pytest.mark.engine


def cold_roundtrip(session):
    saved = session.save_prefix()  # Real EngineCore.step until save completes.
    session.close_source()  # Join every source rank; no tensor survives.
    session.open_reader()  # New EngineCore and independent workers.
    return session.load_and_compare(saved)  # Poison -> external hit -> per-rank hashes.


def reuse_roundtrip(session):
    saved = session.save_prefix()
    session.reset_local_prefix()  # Public native operation, same instance.
    return session.load_and_compare(saved)


def test_cold_roundtrip(engine_case):
    result = engine_case.run("cold")
    assert result["source_exited_before_reader"]
    assert set(result["source_pids"]).isdisjoint(result["reader_pids"])
    assert result["external_loads"] > 0
    assert all(n > 0 for n in result["compared_bytes_by_rank"])


def test_same_instance_external_restore(engine_case):
    result = engine_case.run("reuse")
    assert result["external_loads"] > 0
    assert result["prefix_reset"]
    assert all(n > 0 for n in result["compared_bytes_by_rank"])
