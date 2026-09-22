"""Scoring contracts captured before the local-submit refactor (a52ca18)."""

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.grader import (
    Grader,
    summarize,
)


@pytest.mark.parametrize(
    "references,comparison,status,matched",
    [
        ([True, True], "matched", "matched", True),
        ([False, True], "matched", "matched", True),
        ([True, True], "different", "different", False),
        ([False, True], "different", "unknown", None),
        ([True, True], "unknown", "unknown", None),
        ([False, False], "different", "reference_error", None),
    ],
)
def test_reference_outcomes(sample_bundle, references, comparison, status, matched):
    """A match wins; a failed reference prevents a definitive mismatch."""

    class Runtime:
        def run(self, payload):
            return {
                "status": "ok",
                "comparisons": [
                    {
                        "reference": ref["id"],
                        "status": comparison,
                        "methods": ["symbolic"],
                    }
                    for ref in payload["references"]
                ],
            }

    class FixedGrader(Grader):
        def execution(self, code, row):
            if code == "candidate" or references[int(code)]:
                return {"status": "ok", "outputs": []}
            return {"status": "error", "stage": "execution"}

    row = sample_bundle["problems"][0]
    row["references"] = [
        {"id": str(i), "group": f"group-{i}", "code": str(i)}
        for i in range(len(references))
    ]
    result = FixedGrader(sample_bundle, Runtime()).grade(row["id"], "candidate")

    assert (result["status"], result["matched"]) == (status, matched)
    assert len(result["reference_errors"]) == references.count(False)
    assert result["confidence_basis"] == "historical_report_not_recomputed"
    assert "reward" not in result and "verified" not in result
    if matched:
        first = references.index(True)
        assert result["matched_reference"] == str(first)
        assert result["matched_group"] == f"group-{first}"


def test_weighted_summary():
    """Errors and uncertainty keep their weights; exclusions do not."""
    outcomes = [
        ("matched", True, 0.8),
        ("matched", True, 0.4),
        ("different", False, 0.6),
        ("unknown", None, 0.8),
        ("missing_candidate", False, 0.4),
        ("candidate_error", False, 0.6),
        ("reference_error", None, 0.8),
        ("skipped", None, 0.8),
    ]
    results = [
        {
            "problem_id": str(i),
            "status": status,
            "matched": matched,
            "confidence": weight,
            "methods": ["sampled"] if i == 0 else [],
            "reference_coverage": "incomplete" if i == 1 else "complete",
        }
        for i, (status, matched, weight) in enumerate(outcomes)
    ]

    report = summarize(results)

    assert report["policy_version"] == "consensus-61-v2"
    assert report["total_slots"] == 8
    assert report["active"] == 7 and report["skipped"] == 1
    assert report["matched"] == 2
    assert report["match_rate"] == pytest.approx(2 / 7)
    assert report["weight_total"] == 4.4
    assert report["weighted_match_rate"] == pytest.approx(1.2 / 4.4)
    assert report["sampled_matches"] == 1
    assert report["incomplete_reference_coverage"] == ["1"]
    assert report["by_confidence"]["0.8"]["match_rate"] == pytest.approx(1 / 3)
    assert report["by_confidence"]["0.6"]["match_rate"] == 0
    assert report["by_confidence"]["0.4"]["match_rate"] == 0.5
