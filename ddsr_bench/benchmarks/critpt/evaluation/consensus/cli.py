"""CLI for building and replaying consensus references."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ddsr_bench.benchmarks.utils import write_json

from .execution.runtime import DiagnosticRuntime, provenance
from .grader import Grader
from .references import build_references, load_references


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent internal consensus evaluation (not official CritPt accuracy)"
    )
    commands = p.add_subparsers(dest="command", required=True)
    b = commands.add_parser(
        "build", help="build verifier-only references from the reviewed inventory"
    )
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--source-root", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("replay")
    sub.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Path to the private reference file; no machine-specific default",
    )
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--jobs", type=int, default=4)
    sub.add_argument("--timeout", type=float, default=60)
    sub.add_argument("--image", default="ddsr-bench-critpt:latest")
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
    sub.add_argument("--all-references", action="store_true")
    sub.add_argument("--problems", help="comma-separated IDs for a partial replay")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            references = build_references(args.manifest, args.source_root, args.output)
            print(
                json.dumps(
                    {
                        "path": str(args.output),
                        "slots": 70,
                        "active": sum(
                            p["mode"] != "skip" for p in references["problems"]
                        ),
                        "references": sum(
                            len(p["references"]) for p in references["problems"]
                        ),
                    }
                )
            )
            return 0
        if not 1 <= args.jobs <= 16:
            raise ValueError("jobs must be between 1 and 16")
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        references = load_references(args.references)
        runtime = DiagnosticRuntime(
            trusted_local=args.trusted_local,
            backend=args.backend,
            image=args.image,
            timeout=args.timeout,
        )
        grader = Grader(references, runtime)
        selected = (
            {int(x) for x in args.problems.split(",")}
            if args.problems
            else set(range(1, 71))
        )
        if not selected <= set(range(1, 71)):
            raise ValueError("unknown challenge in --problems")
        jobs = []
        for problem in references["problems"]:
            if problem["number"] not in selected:
                continue
            refs = (
                problem["references"]
                if args.all_references
                else problem["references"][:1]
            )
            if problem["mode"] == "skip":
                jobs.append((problem["id"], None))
            else:
                jobs.extend((problem["id"], ref) for ref in refs)

        def replay(job):
            problem_id, ref = job
            return {
                "source_reference": ref["id"] if ref else None,
                "result": grader.grade(problem_id, ref["code"] if ref else None),
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
            "provenance": provenance(args.references, runtime),
            "summary": summary,
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report, allow_nan=False, overwrite=False, newline=True)
        print(json.dumps(report["summary"], ensure_ascii=False))
        return 0 if report["summary"]["healthy"] else 1
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"consensus: {error}", file=sys.stderr)
        return 2
