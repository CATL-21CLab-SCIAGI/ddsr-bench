from pathlib import Path

import pytest

from ddsr_bench.commands.solve import run_job


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "benchmark"),
    [
        ("benchmark: cmphysbench\n", "cmphysbench"),
        ("job_name: legacy-critpt\n", "critpt"),
    ],
)
async def test_job_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: str,
    benchmark: str,
) -> None:
    path = tmp_path / "job.yaml"
    path.write_text(config, encoding="utf-8")

    async def fake(_: Path, *, task_name: str | None = None) -> dict[str, str | None]:
        return {"benchmark": benchmark, "task_name": task_name}

    monkeypatch.setattr(
        "ddsr_bench.commands.solve.static_runner",
        lambda name: fake if name == benchmark else None,
    )

    assert await run_job(path, "7") == {"benchmark": benchmark, "task_name": "7"}


@pytest.mark.asyncio
async def test_unknown_static_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text("benchmark: scicode\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no static runner"):
        await run_job(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "generation", "grading"])
async def test_job_submission(tmp_path, monkeypatch, failure):
    path = tmp_path / "job.yaml"
    path.write_text("submission:\n  backend: internal\n  attempts: [0]\n")
    output = tmp_path / "job"
    calls = []

    async def generate(path, **kwargs):
        calls.append("generation")
        if failure == "generation":
            raise RuntimeError("generation failed")
        output.mkdir()
        (output / "response.json").write_text("saved")
        return {"output": str(output), "errors": 1}

    def submit(job, config):
        calls.append("grading")
        assert job == str(output)
        assert config == {"backend": "internal", "attempts": [0]}
        if failure == "grading":
            raise RuntimeError("grading failed")
        return "graded"

    monkeypatch.setattr("ddsr_bench.commands.solve.static_runner", lambda _: generate)
    monkeypatch.setattr("ddsr_bench.commands.solve.submitter", lambda _: submit)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            await run_job(path)
    else:
        assert (await run_job(path))["submission"] == "graded"
    assert calls == (
        ["generation"] if failure == "generation" else ["generation", "grading"]
    )
    if failure != "generation":
        assert (output / "response.json").read_text() == "saved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "submission", ["[]", "{backend: official}", "{}", "{backend: internal}"]
)
async def test_submission_preflight(tmp_path, monkeypatch, submission):
    path = tmp_path / "job.yaml"
    path.write_text(f"submission: {submission}\n")

    def forbidden(*args):
        pytest.fail("Invalid submission must not start generation")

    monkeypatch.setattr("ddsr_bench.commands.solve.static_runner", forbidden)
    with pytest.raises((TypeError, ValueError)):
        await run_job(path)
