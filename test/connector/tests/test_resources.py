"""Cleanup reclaims recorded SHM while leaving other processes' segments alone."""

from multiprocessing import shared_memory
from pathlib import Path
import tempfile

import pytest


def test_cleanup_only_reclaims_recorded_resources(tmp_path):
    from connector.conftest import cleanup_case_resources
    from connector.runtime.environment import observe_owned_shared_memory

    foreign = shared_memory.SharedMemory(create=True, size=8)
    owned = None
    ipc = tempfile.mkdtemp(prefix="ucm-pytest-cleanup-")
    try:
        with observe_owned_shared_memory(tmp_path):
            owned = shared_memory.SharedMemory(create=True, size=8)
        name = owned.name
        owned.close()
        cleanup = cleanup_case_resources(tmp_path, ipc)
        assert name in cleanup["removed_shm"]
        assert not Path(ipc).exists()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
        other = shared_memory.SharedMemory(name=foreign.name)
        other.close()
    finally:
        foreign.close()
        foreign.unlink()
        if owned is not None:
            try:
                owned.unlink()
            except FileNotFoundError:
                pass


def test_worker_bootstrap_failure_closes_pipe_without_queues(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from connector.runtime import worker

    class Pipe:
        closed = False
        messages = []

        def send(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    def fail(*args):
        raise RuntimeError("bootstrap failed before queues")

    monkeypatch.setattr(worker, "CacheWorker", fail)
    (tmp_path / "source").mkdir()
    pipe = Pipe()
    with pytest.raises(RuntimeError, match="bootstrap failed before queues"):
        worker.worker_main(
            pipe,
            b"unused",
            SimpleNamespace(output=str(tmp_path), phase="source"),
            0,
            None,
            None,
            None,
        )
    assert pipe.closed
    assert pipe.messages[0][0] == "error"
    assert "bootstrap failed before queues" in pipe.messages[0][1]
