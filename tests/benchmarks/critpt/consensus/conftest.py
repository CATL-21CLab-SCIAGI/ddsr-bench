import json

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import POLICY_VERSION
from ddsr_bench.benchmarks.critpt.evaluation.consensus.bundle import (
    DEFAULT_BUNDLE,
    digest,
    load,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.policy import NOTES, mode


@pytest.fixture
def sample_bundle():
    rows = []
    for n in range(1, 71):
        active = mode(n) != "skip"
        template = "def answer():\n    pass"
        code = "def answer():\n    return 1"
        refs = (
            [{"id": "g/model", "group": "g", "code": code, "sha256": digest(code)}]
            if active
            else []
        )
        groups = [
            {
                "id": "g",
                "models": ["one", "two"],
                "references": ["g/model"] if active else [],
            }
        ]
        if n == 4:
            groups.append(
                {"id": "missing", "models": ["three", "four"], "references": []}
            )
        rows.append(
            {
                "id": f"Challenge_{n}_main",
                "number": n,
                "mode": mode(n),
                "confidence": 0.4,
                "reported_max_agreement": 2,
                "template": template,
                "template_sha256": digest(template),
                "parameters": [],
                "note": NOTES.get(n, ""),
                "references": refs,
                "groups": groups,
                "reference_coverage": "incomplete" if n == 4 else "complete",
            }
        )
    return {"schema_version": 1, "policy_version": POLICY_VERSION, "problems": rows}


@pytest.fixture
def sample_bundle_path(tmp_path, sample_bundle):
    path = tmp_path / "synthetic-bundle.json"
    path.write_text(json.dumps(sample_bundle))
    return path


@pytest.fixture
def reviewed_bundle():
    if not DEFAULT_BUNDLE.is_file():
        pytest.skip(f"External reference asset unavailable: {DEFAULT_BUNDLE}")
    return load(DEFAULT_BUNDLE)
