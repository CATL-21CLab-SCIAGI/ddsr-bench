from ddsr_bench.benchmarks.hle.result import summarize


def test_summary_groups_official_dimensions() -> None:
    trial = {
        "attempt": 0,
        "status": "passed",
        "benchmark_config": {
            "subset": "Gold",
            "category": "Physics",
            "answer_type": "exactMatch",
            "correct": True,
            "confidence": 90,
            "judge_model": "judge",
            "judged": True,
        },
    }

    summary = summarize([trial])

    assert summary["overall"]["accuracy"] == 100
    assert summary["by_subset"]["Gold"]["problems"] == 1
    assert summary["by_category"]["Physics"]["accuracy"] == 100
