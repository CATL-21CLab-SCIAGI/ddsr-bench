"""Run a CritPt job with optional local dotenv loading and durable job summary."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from ddsr_bench.commands.solve import run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-task-name")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("env file does not exist")
        load_dotenv(args.env_file, override=False)
    result = asyncio.run(
        run_job(args.config, args.include_task_name, resume=args.resume)
    )
    output = Path(result["output"])
    name = "resume-summary.json" if args.resume else "generation-summary.json"
    temporary = output / (name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / name)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
