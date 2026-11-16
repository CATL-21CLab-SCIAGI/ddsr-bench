"""CMT-Benchmark verification and execution adapters."""

from .evaluator import Evaluation, evaluate
from .verifier import verify

__all__ = ["Evaluation", "evaluate", "verify"]
