"""Public loaders for CritPt sources; preparation selects the source."""

from .ai import load_problem as load_ai_problem
from .ai import load_problems as load_ai_problems
from .official import load_challenge, load_challenges, load_problems

__all__ = [
    "load_ai_problem",
    "load_ai_problems",
    "load_challenge",
    "load_challenges",
    "load_problems",
]
