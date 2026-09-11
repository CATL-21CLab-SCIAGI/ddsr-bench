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
