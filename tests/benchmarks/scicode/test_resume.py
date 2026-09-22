"""Exercise pinned Harbor reconciliation without a container or model service."""

import json
from pathlib import Path

import pytest
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult

from ddsr_bench.benchmarks.scicode.data.loader import load_problem
from ddsr_bench.benchmarks.scicode.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.utils import Resources


@pytest.mark.asyncio
async def test_harbor_recovery(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "scicode_19.json"
    problem = load_problem(json.loads(fixture.read_text()))
    task = compile_problem(
        problem, tmp_path / "tasks", Resources("unused", 1, 1024, 60)
    )
    config = JobConfig.model_validate(
        {
            "job_name": "recovery",
            "jobs_dir": str(tmp_path / "jobs"),
            "n_attempts": 3,
            "datasets": [{"path": str(task.parent)}],
            "agents": [
                {
                    "name": "ddsr_bench.benchmarks.scicode.evaluation.harbor:SciCodeAgent",
                    "model_name": "teacher",
                }
            ],
        }
    )
    job = await Job.create(config)
    try:
        (job.job_dir / "config.json").write_text(config.model_dump_json())
        for index, trial in enumerate(job._trial_configs):
            directory = job.job_dir / trial.trial_name
            directory.mkdir()
            (directory / "config.json").write_text(trial.model_dump_json())
            if index == 2:
                (directory / "partial.txt").write_text("unfinished generation")
                interrupted = directory
                continue
            result = TrialResult(
                task_name=task.name,
                trial_name=trial.trial_name,
                trial_uri=directory.as_uri(),
                task_id=LocalTaskId(path=task),
                task_checksum="fixture",
                config=trial,
                agent_info=AgentInfo(name="scicode", version="test"),
                exception_info=(
                    ExceptionInfo.from_exception(RuntimeError("saved failure"))
                    if index == 1
                    else None
                ),
            )
            (directory / "result.json").write_text(result.model_dump_json())
    finally:
        job._close_logger_handlers()

    resumed = await Job.create(config)
    try:
        assert len(resumed._existing_trial_results) == 2
        assert len(resumed._remaining_trial_configs) == 1
        assert not interrupted.exists()  # Harbor deletes unfinished trial folders.
        assert any(r.exception_info for r in resumed._existing_trial_results)
    finally:
        resumed._close_logger_handlers()
