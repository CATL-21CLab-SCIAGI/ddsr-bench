import os
import sys
from pathlib import Path

import pytest
import yaml

from ddsr_bench.commands.serve import launch


def test_launch_hands_off_to_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "server.yaml"
    config.write_text("model: test\n", encoding="utf-8")
    called: list[tuple[str, list[str]]] = []

    def execvp(executable: str, command: list[str]) -> None:
        called.append((executable, command))
        raise RuntimeError("process replaced")

    monkeypatch.setattr(os, "execvp", execvp)
    with pytest.raises(RuntimeError, match="process replaced"):
        launch(config)

    executable = str(Path(sys.executable).with_name("vllm"))
    assert called == [(executable, [executable, "serve", "--config", str(config)])]


def test_missing_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="vLLM config not found"):
        launch(tmp_path / "missing.yaml")


@pytest.mark.parametrize("name", ["linux.yaml", "macos.yaml", "macos-qwen38.yaml"])
def test_native_configs(name: str) -> None:
    path = Path(__file__).parents[2] / "configs" / "vllm" / name
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["served-model-name"] == "ddsr-local"
    assert config["host"] == "127.0.0.1"
    assert config["port"] == 8000
