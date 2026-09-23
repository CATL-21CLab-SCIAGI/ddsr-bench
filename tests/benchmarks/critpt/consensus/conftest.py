import json
import os
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import POLICY_VERSION
from ddsr_bench.benchmarks.critpt.evaluation.consensus.matching.rules import NOTES, mode
from ddsr_bench.benchmarks.critpt.evaluation.consensus.references import (
    checksum,
    load_references,
)


@pytest.fixture
def sample_bundle():
    rows = []
    for n in range(1, 71):
        active = mode(n) != "skip"
        template = "def answer():\n    pass"
        code = "def answer():\n    return 1"
        refs = (
            [{"id": "g/model", "group": "g", "code": code, "sha256": checksum(code)}]
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
                "template_sha256": checksum(template),
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
def reviewed_path():
    location = os.environ.get("DDSR_CRITPT_REFERENCES")
    if not location:
        pytest.skip("set DDSR_CRITPT_REFERENCES to test the private reference file")
    path = Path(location).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"configured reference file does not exist: {path}")
    return path


@pytest.fixture
def reviewed_bundle(reviewed_path):
    return load_references(reviewed_path)
