import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import (
    references,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.execution import worker
from ddsr_bench.benchmarks.critpt.evaluation.consensus.execution.worker import evaluate
from ddsr_bench.grading import validation as code_validation


def test_generation_and_consensus_share_the_original_rules():
    assert (
        worker.validate_code
        is references.validate_code
        is code_validation.validate_code
    )
    assert references.extract_code is code_validation.extract_code


@pytest.mark.parametrize(
    "body",
    [
        "def helper(): return 1\n    return helper()",
        "return (lambda x: x)(1)",
        "try: return 1\n    except: return 2",
        "raise ValueError('bad')",
        "with thing: return 1",
        "del thing\n    return 1",
        "yield 1",
        "return sp.memmap('file')",
        "return delattr(sp, 'x')",
    ],
)
def test_worker_rejects_generation_prohibited_syntax(body):
    result = evaluate(
        {
            "action": "evaluate",
            "code": "import sympy as sp\ndef answer():\n    " + body,
            "template": "import sympy as sp\ndef answer(): pass",
            "inputs": [{}],
        }
    )
    assert result["status"] == "error" and result["stage"] == "validation"


@pytest.mark.parametrize(
    "response",
    [
        "def answer(): return 1",
        "```python\ndef answer(): return 1",  # Original extractor permits this.
        "```\nscratch\n```\n```python\ndef answer(): return 1\n```",
    ],
)
def test_extraction_uses_generation_convention(response):
    assert references.extract_code(response) == "def answer(): return 1\n"


def test_existing_reference_bundle_passes_the_original_rules(reviewed_bundle):
    for row in reviewed_bundle["problems"]:
        for reference in row["references"]:
            worker.validate_code(reference["code"], row["template"])
