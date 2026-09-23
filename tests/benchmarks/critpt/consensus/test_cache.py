import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import grader


@pytest.fixture
def runtime():
    result = {
        "status": "ok",
        "outputs": [{"t": "int", "v": "1"}],
        "runtime": {"python": "3.12"},
    }
    return SimpleNamespace(
        backend="docker",
        image_id="sha256:one",
        timeout=60,
        cpus=1,
        memory_mb=1024,
        run=Mock(return_value=result),
    )


def test_reference_only(tmp_path, sample_bundle, runtime):
    row = sample_bundle["problems"][0]
    for _ in range(2):
        instance = grader.Grader(sample_bundle, runtime, cache_dir=tmp_path)
        instance.execution("same code", row, reference=True)
    assert runtime.run.call_count == 1
    assert instance.cache_stats == {"hits": 1, "executions": 0}
    instance.execution("same code", row)
    instance.execution("same code", row)
    assert runtime.run.call_count == 3  # Answers never use reference caches.


@pytest.mark.parametrize(
    "field",
    [
        "code",
        "template",
        "inputs",
        "image_id",
        "timeout",
        "cpus",
        "memory_mb",
        "policy",
    ],
)
def test_invalidation(tmp_path, sample_bundle, runtime, monkeypatch, field):
    row, code = sample_bundle["problems"][0], "same code"
    grader.Grader(sample_bundle, runtime, cache_dir=tmp_path).execution(
        code, row, reference=True
    )
    if field == "code":
        code = "changed code"
    elif field == "template":
        row = {**row, "template": row["template"] + "\n# changed"}
    elif field == "inputs":
        monkeypatch.setattr(grader, "inputs", lambda *_: [{"x": 2}])
    elif field == "policy":
        monkeypatch.setattr(grader, "POLICY_VERSION", "new-policy")
    else:
        setattr(runtime, field, "sha256:two" if field == "image_id" else 2)
    grader.Grader(sample_bundle, runtime, cache_dir=tmp_path).execution(
        code, row, reference=True
    )
    assert runtime.run.call_count == 2


def test_failures_retry(tmp_path, sample_bundle, runtime):
    runtime.run.return_value = {"status": "error", "stage": "timeout"}
    instance = grader.Grader(sample_bundle, runtime, cache_dir=tmp_path)
    for _ in range(2):
        instance.execution("same code", sample_bundle["problems"][0], reference=True)
    assert runtime.run.call_count == 2
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("corrupt", ["invalid JSON", "wrong checksum"])
def test_corrupt_cache(tmp_path, sample_bundle, runtime, corrupt):
    row = sample_bundle["problems"][0]
    grader.Grader(sample_bundle, runtime, cache_dir=tmp_path).execution(
        "code", row, reference=True
    )
    path = next(tmp_path.rglob("*.json"))
    if corrupt == "wrong checksum":
        record = json.loads(path.read_text())
        record["result"]["outputs"][0]["v"] = "999"
        corrupt = json.dumps(record)
    path.write_text(corrupt)
    grader.Grader(sample_bundle, runtime, cache_dir=tmp_path).execution(
        "code", row, reference=True
    )
    assert runtime.run.call_count == 2


def test_concurrent_writes(tmp_path, sample_bundle, runtime):
    def execute(_):
        return grader.Grader(sample_bundle, runtime, cache_dir=tmp_path).execution(
            "code", sample_bundle["problems"][0], reference=True
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, range(4)))
    assert all(result == results[0] for result in results)
    count = runtime.run.call_count
    assert execute(0) == results[0]
    assert runtime.run.call_count == count
    assert len(list(tmp_path.rglob("*.json"))) == 1
