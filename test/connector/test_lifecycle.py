"""Completion crosses steps; faults must reach the engine or byte comparator."""

import pytest

pytestmark = pytest.mark.engine


def test_delayed_rank(engine_case):
    if engine_case.options["model_config"] or engine_case.options["mode"] != "bulk":
        pytest.skip(
            "Cross-step async completion is scoped to mainline direct GQA bulk; other routes load synchronously"
        )
    result = engine_case.run("cold", "delay")
    assert result["delayed_checks"] > 0
    if engine_case.options["tp"] > 1:
        assert result["partial_completion_checks"] > 0
    assert result["external_loads"] > 0


@pytest.mark.parametrize("fault", ["load-error", "corrupt"])
def test_load_fault(engine_case, fault):
    result = engine_case.run("cold", fault)
    assert result["fault_detected"]


def test_timeout_cleanup(engine_case):
    import json

    if (
        engine_case.options["model_config"]
        or engine_case.options["mode"] != "bulk"
        or engine_case.options["store"] != "mock"
    ):
        pytest.skip("Hard-timeout cleanup uses synthetic bulk store fault")
    with pytest.raises(pytest.fail.Exception, match="Driver timed out"):
        engine_case.run("cold", "hang", timeout_override=90)
    assert list(engine_case.output.glob("hang-observed-rank-*.json"))
    cleanup = json.loads((engine_case.output / "cleanup.json").read_text())
    assert cleanup["ipc_directory_removed"]
    assert cleanup["removed_shm"] or cleanup["already_absent_shm"]
