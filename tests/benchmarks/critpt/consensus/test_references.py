import json
from copy import deepcopy

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.cli import (
    main,
    parser,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.matching.cases import inputs
from ddsr_bench.benchmarks.critpt.evaluation.consensus.references import (
    _parameter_names,
    checksum,
    load_references,
)


@pytest.mark.parametrize("command", ["score", "batch"])
def test_removed_commands(command):
    with pytest.raises(SystemExit) as error:
        parser().parse_args([command])
    assert error.value.code == 2


def test_parameter_names():
    template = "def answer(x, /, y=1, *, z=2): pass"
    names = _parameter_names(template)
    assert names == ["x", "y", "z"]
    assert list(inputs(1, names)[0]) == names
    assert inputs(1, []) == [{}]
    cases = inputs(62, ["k", "phi", "k_value"])
    assert [case["k_value"] for case in cases] == [1, 2, 3]


def test_bundle_checks_scope_reference_hash_and_coverage(tmp_path, sample_bundle):
    bundle = sample_bundle
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    assert len(load_references(path)["problems"]) == 70
    mutations = [
        lambda problem: problem["references"][0].update(code="def answer(): return 2"),
        lambda problem: problem.update(confidence=0.8),
        lambda problem: problem.update(reference_coverage="complete"),
        lambda problem: problem.update(parameters=["unexpected"]),
    ]
    for change in mutations:
        changed = deepcopy(bundle)
        change(changed["problems"][3])
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_references(path)


def test_external_references(tmp_path, monkeypatch, reviewed_path):
    monkeypatch.chdir(tmp_path)
    argv = ["replay", "--references", str(reviewed_path), "--output", "result.json"]
    args = parser().parse_args(argv)
    assert args.references == reviewed_path
    problems = load_references(args.references)["problems"]
    assert len(problems) == 70
    assert sum(problem["mode"] != "skip" for problem in problems) == 61
    assert sum(len(problem["references"]) for problem in problems) == 204
    assert checksum(args.references.read_text()) == (
        "6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc"
    )
    assert parser().parse_args(
        argv + ["--references", "custom.json"]
    ).references.name == ("custom.json")


def test_reference_path_required():
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["replay", "--output", "result.json"])
    assert error.value.code == 2


def test_missing_references(tmp_path, capsys):
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "replay",
                "--references",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
                "--trusted-local",
            ]
        )
        == 2
    )
    message = capsys.readouterr().err
    assert "benchmark.submission.internal.references" in message
    assert "replay --references" in message
    assert not output.exists()
