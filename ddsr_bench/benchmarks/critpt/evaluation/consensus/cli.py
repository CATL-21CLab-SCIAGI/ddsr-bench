"""Standalone consensus CLI; deliberately absent from legacy Hydra dispatch."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ddsr_bench.grading import validation as code_validation

from . import POLICY_VERSION
from .bundle import DEFAULT_BUNDLE, build, digest, load
from .candidates import load_candidates
from .evaluator import Evaluator, summarize
from .runtime import Runtime


def provenance(bundle_path: Path, runtime: Runtime) -> dict:
    return {
        "bundle_sha256": digest(bundle_path.read_text()),
        "policy_version": POLICY_VERSION,
        "code_validation_sha256": digest(Path(code_validation.__file__).read_text()),
        "execution": runtime.backend,
        "image": runtime.image if runtime.backend == "docker" else None,
        "image_id": runtime.image_id,
        **({"linux_sandbox": runtime.linux.info} if runtime.linux else {}),
        "host_python": platform.python_version(),
        "host_dependencies": {
            p: importlib.metadata.version(p) for p in ("sympy", "numpy", "scipy")
        },
    }


def write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent internal consensus evaluation (not official CritPt accuracy)"
    )
    commands = p.add_subparsers(dest="command", required=True)
    b = commands.add_parser(
        "build", help="build a verifier-only local bundle from the reviewed inventory"
    )
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--source-root", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    for name in ("score", "replay", "batch"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--bundle",
            type=Path,
            default=DEFAULT_BUNDLE,
            help="Reference bundle (default: the versioned bundle shipped with this package)",
        )
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--jobs", type=int, default=4)
        sub.add_argument("--timeout", type=float, default=60)
        sub.add_argument("--image", default="ddsr-critpt-consensus:local")
        execution = sub.add_mutually_exclusive_group()
        execution.add_argument(
            "--backend",
            choices=("docker", "linux"),
            help="Worker isolation backend (default: docker; linux requires bubblewrap)",
        )
        execution.add_argument(
            "--trusted-local",
            action="store_true",
            help="Run reviewed, trusted code on this host; NOT a security sandbox",
        )
        if name in ("score", "batch"):
            sub.add_argument("--candidates", type=Path, required=True)
            if name == "score":
                sub.add_argument(
                    "--attempt", type=int, help="select one native rollout attempt"
                )
            else:
                sub.add_argument("--model-label")
        else:
            sub.add_argument("--all-references", action="store_true")
            sub.add_argument(
                "--problems",
                help="optional comma-separated IDs for a partial diagnostic replay",
            )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            bundle = build(args.manifest, args.source_root, args.output)
            print(
                json.dumps(
                    {
                        "bundle": str(args.output),
                        "slots": 70,
                        "active": sum(p["mode"] != "skip" for p in bundle["problems"]),
                        "references": sum(
                            len(p["references"]) for p in bundle["problems"]
                        ),
                    }
                )
            )
            return 0
        if not 1 <= args.jobs <= 16:
            raise ValueError("jobs must be between 1 and 16")
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        bundle = load(args.bundle)
        submitted = (
            load_candidates(args.candidates, args.attempt)
            if args.command == "score"
            else None
        )
        runtime = Runtime(
            trusted_local=args.trusted_local,
            backend=args.backend,
            image=args.image,
            timeout=args.timeout,
        )
        evaluator = Evaluator(bundle, runtime)
        if args.command == "batch":
            from .batch import score_batch

            summary = score_batch(
                args.candidates,
                args.output,
                evaluator,
                jobs=args.jobs,
                metadata={**provenance(args.bundle, runtime), "timeout": args.timeout},
                model_label=args.model_label,
            )
            print(json.dumps({k: v for k, v in summary.items() if k != "problems"}))
            return 0
        if args.command == "score":

            def score(row):
                candidate = submitted.answers.get(row["id"])
                result = evaluator.grade(
                    row["id"],
                    candidate.code if candidate else None,
                    input_error=candidate.error if candidate else None,
                )
                if candidate:
                    result.update(
                        candidate_source=str(candidate.path),
                        generation=candidate.generation,
                        metadata_errors=candidate.metadata_errors,
                    )
                return result

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                results = list(pool.map(score, bundle["problems"]))
            report = {
                "provenance": provenance(args.bundle, runtime),
                "candidate_input": {
                    "directory": str(args.candidates),
                    "layout": submitted.layout,
                    "attempt": submitted.attempt,
                    "available_attempts": submitted.available_attempts,
                    "recognized_candidates": len(submitted.answers),
                },
                "summary": summarize(results),
                "results": results,
            }
        else:
            selected = (
                {int(x) for x in args.problems.split(",")}
                if args.problems
                else set(range(1, 71))
            )
            if not selected <= set(range(1, 71)):
                raise ValueError("unknown challenge in --problems")
            jobs = []
            for row in bundle["problems"]:
                if row["number"] not in selected:
                    continue
                refs = (
                    row["references"] if args.all_references else row["references"][:1]
                )
                if row["mode"] == "skip":
                    jobs.append((row["id"], None))
                else:
                    jobs.extend((row["id"], ref) for ref in refs)

            def replay(job):
                problem_id, ref = job
                return {
                    "source_reference": ref["id"] if ref else None,
                    "result": evaluator.grade(problem_id, ref["code"] if ref else None),
                }

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                runs = list(pool.map(replay, jobs))
            statuses = dict(Counter(x["result"]["status"] for x in runs))
            summary = {
                "kind": "reference_replay",
                "partial": selected != set(range(1, 71)),
                "runs": len(runs),
                "statuses": statuses,
                "healthy": all(
                    x["result"]["status"] in {"matched", "skipped"} for x in runs
                ),
                "problems": len(selected),
            }
            report = {
                "provenance": provenance(args.bundle, runtime),
                "summary": summary,
                "runs": runs,
            }
        write(args.output, report)
        print(json.dumps(report["summary"], ensure_ascii=False))
        return 0 if args.command == "score" or report["summary"]["healthy"] else 1
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"consensus: {error}", file=sys.stderr)
        return 2
