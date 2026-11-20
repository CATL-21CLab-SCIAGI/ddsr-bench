from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

from ddsr_bench.benchmarks.utils import benchmark_config

from .schemas import AnswerSpec, Example, Problem, ProblemSpec

_CONFIG = benchmark_config("mmlu")
_SUBCATEGORY = {
    "abstract_algebra": "math",
    "anatomy": "health",
    "astronomy": "physics",
    "business_ethics": "business",
    "clinical_knowledge": "health",
    "college_biology": "biology",
    "college_chemistry": "chemistry",
    "college_computer_science": "computer science",
    "college_mathematics": "math",
    "college_medicine": "health",
    "college_physics": "physics",
    "computer_security": "computer science",
    "conceptual_physics": "physics",
    "econometrics": "economics",
    "electrical_engineering": "engineering",
    "elementary_mathematics": "math",
    "formal_logic": "philosophy",
    "global_facts": "other",
    "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "economics",
    "high_school_physics": "physics",
    "high_school_psychology": "psychology",
    "high_school_statistics": "math",
    "high_school_us_history": "history",
    "high_school_world_history": "history",
    "human_aging": "health",
    "human_sexuality": "culture",
    "international_law": "law",
    "jurisprudence": "law",
    "logical_fallacies": "philosophy",
    "machine_learning": "computer science",
    "management": "business",
    "marketing": "business",
    "medical_genetics": "health",
    "miscellaneous": "other",
    "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy",
    "nutrition": "health",
    "philosophy": "philosophy",
    "prehistory": "history",
    "professional_accounting": "other",
    "professional_law": "law",
    "professional_medicine": "health",
    "professional_psychology": "psychology",
    "public_relations": "politics",
    "security_studies": "politics",
    "sociology": "culture",
    "us_foreign_policy": "politics",
    "virology": "health",
    "world_religions": "philosophy",
}
_CATEGORIES = {
    "STEM": {
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    },
    "humanities": {"history", "philosophy", "law"},
    "social sciences": {"politics", "culture", "economics", "geography", "psychology"},
    "other (business, health, misc.)": {"other", "business", "health"},
}


def _category(subject: str) -> str:
    subcategory = _SUBCATEGORY.get(subject)
    matches = [name for name, values in _CATEGORIES.items() if subcategory in values]
    if len(matches) != 1:
        raise ValueError(f"MMLU subject {subject!r} has no unique category")
    return matches[0]


def _rows(archive: tarfile.TarFile, name: str) -> list[list[str]]:
    member = archive.getmember(name)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read MMLU archive member {name!r}")
    with source, io.TextIOWrapper(source, encoding="utf-8", newline="") as text:
        rows = list(csv.reader(text))
    if any(len(row) != 6 or row[-1] not in "ABCD" for row in rows):
        raise ValueError(f"MMLU member {name!r} has invalid rows")
    return rows


def _example(row: list[str]) -> Example:
    return Example(row[0], tuple(row[1:5]), row[5])


def load_directory(path: str | Path) -> list[Problem]:
    """Load the official test split and its five per-subject demonstrations."""
    archive_path = Path(path) / str(_CONFIG["data_file"])
    with tarfile.open(archive_path) as archive:
        subjects = sorted(
            Path(member.name).name.removesuffix("_test.csv")
            for member in archive.getmembers()
            if member.isfile()
            and member.name.startswith("data/test/")
            and member.name.endswith("_test.csv")
        )
        if set(subjects) != set(_SUBCATEGORY):
            raise ValueError("MMLU subjects do not match the pinned taxonomy")
        problems = []
        dev_count = 0
        for subject in subjects:
            dev = _rows(archive, f"data/dev/{subject}_dev.csv")
            test = _rows(archive, f"data/test/{subject}_test.csv")
            dev_count += len(dev)
            count = int(_CONFIG["few_shot_examples"])
            if len(dev) < count:
                raise ValueError(f"MMLU subject {subject!r} has too few examples")
            examples = tuple(_example(row) for row in dev[:count])
            for index, row in enumerate(test):
                item = _example(row)
                spec = ProblemSpec(
                    id=f"{subject}-{index}",
                    subject=subject,
                    category=_category(subject),
                    question=item.question,
                    choices=item.choices,
                    examples=examples,
                )
                problems.append(Problem(spec, AnswerSpec(item.answer)))
    if len(subjects) != int(_CONFIG["expected_subjects"]):
        raise ValueError("MMLU source has an unexpected subject count")
    if len(problems) != int(_CONFIG["expected_test_records"]):
        raise ValueError("MMLU source has an unexpected test count")
    if dev_count != int(_CONFIG["expected_dev_records"]):
        raise ValueError("MMLU source has an unexpected dev count")
    return problems


def validate_source(path: Path) -> None:
    load_directory(path)
