from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig

from critpt_eval.benchmark.collect import collect_trials
from critpt_eval.benchmark.prepare import Resources, compile_challenges
from critpt_eval.benchmark.scicode.loader import load_split
from critpt_eval.benchmark.scicode.prepare import compile_problems
from critpt_eval.benchmark.submit import build_batch, submit_batch
from critpt_eval.loaders import load_challenges
from critpt_eval.training import export_sft, export_trajectories

Action = Callable[[DictConfig], str]
LOGGER_NAME = "critpt_eval"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure the command logger once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    return logger


def prepare(config: DictConfig) -> str:
    limits = config.harbor
    resources = Resources(
        image=str(limits.image),
        cpus=int(limits.cpus),
        memory_mb=int(limits.memory_mb),
        timeout_sec=float(limits.timeout_sec),
    )
    benchmark = str(config.benchmark.name)
    if benchmark == "scicode":
        tasks = compile_problems(
            load_split(str(config.benchmark.split)),
            config.paths.output,
            resources,
        )
    elif benchmark == "critpt":
        if config.paths.input is None:
            raise ValueError("paths.input is required for CritPt preparation")
        tasks = compile_challenges(
            load_challenges(config.paths.input), config.paths.output, resources
        )
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    return f"prepared {len(tasks)} Harbor tasks in {config.paths.output}"


def collect(config: DictConfig) -> str:
    if config.paths.input is None:
        raise ValueError("paths.input is required for collect")
    summary = collect_trials(config.paths.input)
    return f"collected {summary['complete_batches']} complete batches"


def export(config: DictConfig) -> str:
    if config.paths.input is None:
        raise ValueError("paths.input is required for export")
    trajectories = export_trajectories(config.paths.input, config.paths.output)
    sft = export_sft(trajectories, config.paths.output, str(config.training.view))
    return f"exported trajectories to {trajectories} and SFT samples to {sft}"


def submit(config: DictConfig) -> str:
    if config.paths.input is None or config.submission.attempt is None:
        raise ValueError("paths.input and submission.attempt are required for submit")
    job = Path(config.paths.input)
    attempt = int(config.submission.attempt)
    output = job / f"submission-{attempt}.json"
    if output.exists():
        raise ValueError(f"submission result already exists: {output}")
    payload = build_batch(job, attempt)
    result = submit_batch(
        payload,
        os.environ.get(str(config.submission.api_key_env), ""),
        endpoint=str(config.submission.endpoint),
        timeout=float(config.submission.timeout_sec),
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return f"submitted attempt {attempt}; result saved to {output}"


ACTIONS: dict[str, Action] = {
    "prepare": prepare,
    "collect": collect,
    "export": export,
    "submit": submit,
}


def dispatch(config: DictConfig) -> str:
    """Dispatch one validated top-level action."""
    action = str(config.action)
    try:
        run = ACTIONS[action]
    except KeyError as error:
        choices = ", ".join(ACTIONS)
        raise ValueError(
            f"unknown action {action!r}; choose from: {choices}"
        ) from error
    return run(config)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def hydra_main(config: DictConfig) -> None:
    logger = configure_logging()
    logger.info(dispatch(config))


def main() -> None:
    hydra_main()
