from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.critpt.trajectory import normalize as normalize_critpt
from ddsr_bench.benchmarks.critpt.trajectory import sft_samples as critpt_samples
from ddsr_bench.benchmarks.scicode.trajectory import normalize as normalize_scicode
from ddsr_bench.benchmarks.scicode.trajectory import sft_samples as scicode_samples
from ddsr_bench.training.schemas import Generation, SftSample, Trajectory

_NORMALIZERS = {"critpt": normalize_critpt, "scicode": normalize_scicode}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object from {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _benchmark(trial: Path, record: dict[str, Any]) -> str:
    task_name = record.get("task_name")
    benchmark = task_name.partition("/")[0] if isinstance(task_name, str) else ""
    if benchmark not in _NORMALIZERS:
        raise ValueError(f"{trial} has an unsupported benchmark")
    return benchmark


def load_trajectory(trial_dir: str | Path) -> Trajectory:
    """Load one canonical trajectory from a static or Harbor trial directory."""
    trial = Path(trial_dir)
    response_path = trial / "agent" / "response.json"
    response_record = _json(response_path)
    trial_record = _json(trial / "result.json")
    validation_path = trial / "validation" / "result.json"
    if not validation_path.exists():
        validation_path = trial / "verifier" / "result.json"
    validation = _json(validation_path)
    benchmark = _benchmark(trial, trial_record)
    return _NORMALIZERS[benchmark](trial, response_record, trial_record, validation)


def export_trajectories(job_dir: str | Path, output: str | Path) -> Path:
    """Write deterministic canonical trajectories for one completed job."""
    job = Path(job_dir)
    trials = sorted(path.parent.parent for path in job.glob("*/agent/response.json"))
    if not trials:
        raise ValueError(f"no response records found in {job}")
    benchmarks = {_benchmark(trial, _json(trial / "result.json")) for trial in trials}
    if len(benchmarks) != 1:
        raise ValueError("cannot export multiple benchmarks together")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "trajectories.jsonl"
    with target.open("w", encoding="utf-8") as file:
        for trial in trials:
            json.dump(asdict(load_trajectory(trial)), file, ensure_ascii=False)
            file.write("\n")
    return target


def _sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    """Delegate SFT view selection to the trajectory's benchmark adapter."""
    exporters = {"critpt": critpt_samples, "scicode": scicode_samples}
    exporter = exporters.get(trajectory.benchmark)
    if exporter is None:
        raise ValueError(f"unsupported benchmark: {trajectory.benchmark!r}")
    return exporter(trajectory, view)


def export_sft(
    trajectories: str | Path,
    output: str | Path,
    view: str = "full",
) -> Path:
    """Stream canonical trajectories into conversational SFT JSONL."""
    source = Path(trajectories)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "sft.jsonl"
    with (
        source.open(encoding="utf-8") as input_file,
        target.open("w", encoding="utf-8") as output_file,
    ):
        for number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                data["generations"] = tuple(
                    Generation(**(generation | {"prompt": tuple(generation["prompt"])}))
                    for generation in data["generations"]
                )
                trajectory = Trajectory(**data)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid trajectory at line {number}") from error
            for sample in _sft_samples(trajectory, view):
                json.dump(asdict(sample), output_file, ensure_ascii=False)
                output_file.write("\n")
    return target
