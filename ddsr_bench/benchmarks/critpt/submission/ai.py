"""Internal assessment of saved AI responses using each task's grader."""

import math
from dataclasses import replace
from functools import partial
from pathlib import Path

from harbor.models.job.config import JobConfig
from harbor.models.task.config import EnvironmentConfig
from omegaconf import OmegaConf

from ddsr_bench.benchmarks.collect import load_batch
from ddsr_bench.benchmarks.utils import Resources, read_json, write_json

from ..data.loader import load_ai_task
from ..data.schemas import AIProblemSpec, CritPtSource
from ..data.utils import decode_instruction
from ..evaluation.ai import grade_response
from ..evaluation.ai.docker import execute
from ..evaluation.ai.grader import parse_problem
from ..evaluation.ai.llm.semantic import LLMSemanticJudge, SemanticJudgeConfig


def _task_paths(job: Path, datasets: list[dict] | None) -> list[Path]:
    """Use the saved job's local datasets, with the generation working directory."""
    if datasets is None:
        path = job / "job-config.yaml"
        if not path.exists():
            path = job / "config.json"  # Harbor's saved job configuration.
        if not path.is_file():
            raise ValueError("saved job config missing; supply submission.datasets")
        saved = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(saved, dict):
            raise TypeError("saved job config must be an object")
        datasets = saved.get("datasets", [])
    config = JobConfig.model_validate({"datasets": datasets})
    if not config.datasets:
        raise ValueError("submission requires local datasets")
    paths = []
    for dataset in config.datasets:
        if dataset.path is None or not dataset.path.is_dir():
            raise ValueError("dataset path unavailable; supply submission.datasets")
        paths.extend(sorted(dataset.path.glob("*/instruction.md")))
    return paths


def submit_ai(
    job: Path,
    attempts: list[int],
    *,
    datasets: list[dict] | None,
    llm: dict,
    pending: Path,
    resources: dict | None = None,
    llm_fallback: bool = False,
    grader: str | None = None,
) -> dict:
    """Preflight all selected responses before any paid assessment.

    Prepared IDs select tasks; original corpus IDs remain inside grade reports.
    Read complete saved responses, not extracted code or partial stream journals.
    Task graders are defaults; an explicit override requires compatible references.
    """
    selected = [trial for attempt in attempts for trial in load_batch(job, attempt)]
    if not selected:
        raise ValueError("AI submission requires collected trials")
    task_paths = {}
    for path in _task_paths(job, datasets):
        spec = decode_instruction(path.read_text(encoding="utf-8"))
        if spec.id in task_paths:
            raise ValueError(f"duplicate prepared problem ID: {spec.id}")
        task_paths[spec.id] = (path.parent, spec)
    inputs = []
    for trial in selected:
        if (
            trial.get("benchmark") != "critpt"
            or trial.get("benchmark_config", {}).get("source") != CritPtSource.AI
        ):
            raise ValueError("AI submission requires critpt-ai trials")
        if trial["problem_id"] not in task_paths:
            raise ValueError(f"no prepared task for {trial['problem_id']}")
        task, spec = task_paths[trial["problem_id"]]
        if not isinstance(spec, AIProblemSpec):
            raise TypeError("AI submission requires a critpt-ai task")
        if spec.grader not in {"rule", "llm"}:
            raise ValueError(f"unsupported task grader: {spec.grader!r}")
        response_path = job / trial["trial_name"] / "agent" / "response.json"
        if not response_path.resolve().is_relative_to(job.resolve()):
            raise ValueError("response must stay inside its job")
        record = read_json(response_path)
        if record.get("problem_id") != spec.id:
            raise ValueError("response problem ID does not match the collected trial")
        if record.get("problem") != {
            "statement": spec.statement,
            "code_template": spec.code_template,
        }:
            raise ValueError("prepared problem differs from the saved generation input")
        responses = record.get("responses")
        if (
            not isinstance(responses, list)
            or not responses
            or not isinstance(responses[-1], dict)
        ):
            raise ValueError("trial requires a saved final response")
        content = responses[-1].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("trial requires nonempty response content")
        problem = load_ai_task(task)
        if grader is not None:
            if grader not in ("rule", "llm"):
                raise ValueError(f"unsupported AI grader: {grader!r}")
            problem = replace(problem, spec=replace(problem.spec, grader=grader))
        parse_problem(problem)  # Validate every reference before paid assessment.
        inputs.append((trial, problem, content))
    kinds = {problem.spec.grader for _, problem, _ in inputs}
    executor = None
    if "rule" in kinds:
        if resources is None:
            raise ValueError("rule submission requires ai.resources")
        limits = Resources(**resources)
        EnvironmentConfig(
            docker_image=limits.image, cpus=limits.cpus, memory_mb=limits.memory_mb
        )
        if not math.isfinite(limits.timeout_sec) or limits.timeout_sec <= 0:
            raise ValueError("worker timeout_sec must be finite and positive")
        executor = partial(execute, resources=limits)
    if type(llm_fallback) is not bool:
        raise TypeError("llm_fallback must be boolean")
    semantic = (
        LLMSemanticJudge(config=SemanticJudgeConfig(**llm)) if "llm" in kinds else None
    )
    # Keep this claim on failure: some judge requests may already have been billed.
    write_json(
        pending,
        {"attempts": attempts, "source": str(CritPtSource.AI), "grader": grader},
        overwrite=False,
    )
    output = pending.with_name(pending.name.removesuffix(".pending.json") + ".json")
    if output.exists():
        pending.unlink()
        raise ValueError(f"submission result already exists: {output}")
    results = []
    for trial, problem, content in inputs:
        report = grade_response(
            problem,
            content,
            semantic_judge=semantic,
            llm_fallback=llm_fallback,
            execute=executor if problem.spec.grader == "rule" else None,
            attempt_id=trial["trial_name"],
        )
        results.append(
            {
                "problem_id": trial["problem_id"],
                "attempt": trial["attempt"],
                "trial_name": trial["trial_name"],
                "report": report.model_dump(mode="json"),
            }
        )
    # Private report only: do not mutate generation rewards or collection summaries.
    return {"source": str(CritPtSource.AI), "attempts": attempts, "results": results}
