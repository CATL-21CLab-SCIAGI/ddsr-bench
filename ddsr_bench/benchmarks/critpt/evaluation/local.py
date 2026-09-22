"""Local submission using the existing consensus reference policy."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .consensus.bundle import DEFAULT_BUNDLE, load
from .consensus.candidates import CandidateBatch, load_collected
from .consensus.grader import Grader, grade_candidate, report
from .consensus.runtime import Runtime, provenance


def score(directory: Path, submitted: CandidateBatch, grader: Grader, jobs: int):
    """Score all policy slots, retaining missing answers in the denominator."""

    def grade(row):
        candidate = submitted.answers.get(row["id"])
        return grade_candidate(grader, row["id"], candidate)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(grade, grader.rows.values()))
    return report(directory, submitted, results)


def submit(
    job: Path,
    attempt: int,
    *,
    bundle: str | None = None,
    execution: str = "docker",
    image: str = "ddsr-critpt-consensus:local",
    timeout: float = 60,
    jobs: int = 4,
) -> dict:
    if not 1 <= jobs <= 16:
        raise ValueError("jobs must be between 1 and 16")
    path = Path(bundle) if bundle is not None else DEFAULT_BUNDLE
    references = load(path)
    submitted = load_collected(job, attempt)
    runtime = Runtime(backend=execution, image=image, timeout=timeout)
    return {
        "provenance": provenance(path, runtime),
        **score(job, submitted, Grader(references, runtime), jobs),
    }
