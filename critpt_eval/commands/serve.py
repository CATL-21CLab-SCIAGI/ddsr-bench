"""Start vLLM from a native configuration file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn


def launch(config: str | Path, executable: str | None = None) -> NoReturn:
    """Replace this process with a vLLM server using its native YAML config."""
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"vLLM config not found: {path}")
    if executable == "":
        raise ValueError("vLLM executable cannot be empty")

    program = executable or str(Path(sys.executable).with_name("vllm"))
    command = [program, "serve", "--config", str(path)]
    os.execvp(program, command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a configured vLLM server")
    parser.add_argument("config", type=Path, help="native vLLM YAML configuration")
    parser.add_argument("--executable")
    args = parser.parse_args()
    launch(args.config, args.executable)
