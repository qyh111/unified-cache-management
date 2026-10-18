"""An explicit --confcutdir keeps the suite independent of parent conftest."""

import os
from pathlib import Path
import subprocess
import sys


def test_collection_does_not_import_engines_or_native():
    root = Path(__file__).resolve().parents[3]
    script = """
import sys
before = set(sys.modules)
import pytest
code = pytest.main(["-q", "-c", "test/connector/pytest.ini", "--confcutdir=test/connector", "test/connector/test_roundtrip.py", "test/connector/test_lifecycle.py", "--collect-only"])
assert code == 0
assert not any(k == "vllm" or k.startswith(("vllm.", "vllm_ascend", "ucm.", "torch")) for k in set(sys.modules) - before)
"""
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stdout + done.stderr
