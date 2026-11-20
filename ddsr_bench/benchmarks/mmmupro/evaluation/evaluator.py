from __future__ import annotations

import random
import string
from typing import Any

from ddsr_bench.benchmarks.mmmupro.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.utils import benchmark_config

_SEED = int(benchmark_config("mmmupro")["malformed_seed"])


def parse_answer(response: str, problem: ProblemSpec) -> tuple[str, str]:
    """Apply the official multi-choice extraction with deterministic fallback."""
    letters = string.ascii_uppercase[: len(problem.choices)]
    position = response.rfind("Answer:")
    if position != -1:
        suffix = response[position + len("Answer:") :].strip()
        matches = [letter for letter in letters if letter in suffix]
        if len(matches) == 1:
            return matches[0], "answer-marker"

    padded = response if isinstance(response, str) else ""
    for character in [",", ".", "!", "?", ";", ":", "'"]:
        padded = padded.strip(character)
    padded = f" {padded} "
    candidates = [letter for letter in letters if f"({letter})" in padded]
    pattern = "bracket"
    if not candidates:
        candidates = [letter for letter in letters if f"{letter} " in padded]
        pattern = "letter"
    if not candidates:
        candidates = [letter for letter in letters if f"{letter}." in padded]
        pattern = "period"
    if not candidates and len(padded.split()) > 5:
        candidates = [
            letter
            for letter, choice in zip(letters, problem.choices, strict=True)
            if choice.lower() in padded.lower()
        ]
        pattern = "choice-text"
    if not candidates:
        # Upstream randomly guesses. Keying the same seed by problem ID keeps
        # isolated and concurrent trials reproducible.
        selected = random.Random(f"{_SEED}:{problem.id}").choice(letters)
        return selected, "seeded-fallback"
    if len(candidates) == 1:
        return candidates[0], pattern
    if pattern == "bracket":
        positions = [padded.rfind(f"({letter})") for letter in candidates]
    elif pattern == "choice-text":
        positions = [
            padded.lower().rfind(problem.choices[letters.index(letter)].lower())
            for letter in candidates
        ]
    else:
        positions = [padded.rfind(f" {letter} ") for letter in candidates]
    return candidates[positions.index(max(positions))], f"last-{pattern}"


def evaluate(response: str, problem: ProblemSpec, answer: AnswerSpec) -> dict[str, Any]:
    selected, method = parse_answer(response, problem)
    passed = selected == answer.letter
    return {
        "reward": float(passed),
        "mode": "mmmupro",
        "status": "passed" if passed else "different",
        "setting": problem.setting,
        "subject": problem.subject,
        "domain": problem.domain,
        "selected": selected,
        "method": method,
    }
