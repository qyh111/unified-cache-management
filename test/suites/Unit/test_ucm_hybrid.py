"""Run lightweight Hybrid tests without leaking engine stubs into other tests."""

import subprocess
import sys
import unittest
from pathlib import Path

CASES = Path(__file__).with_name("hybrid_offline_cases.py")


class HybridOfflineTests(unittest.TestCase):
    def test_isolated_component_suite(self):
        result = subprocess.run(
            [sys.executable, str(CASES)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(CASES)]))
