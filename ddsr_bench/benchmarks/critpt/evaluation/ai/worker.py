"""Execute code inside the restricted container; never load grading references."""

import json
import sys

from .answer import _execute_answer
from .serialization import encode
from .validation import UnsupportedAnswerOperation, _validate_answer_code


def run(code: str) -> dict:
    """Return tagged values or an explicit execution error, not a grade."""
    try:
        _validate_answer_code(code)
        value = _execute_answer(code)
        values = value if isinstance(value, dict) else {"answer": value}
        if any(not isinstance(key, str) for key in values):
            raise ValueError("answer fields must be strings")
    except UnsupportedAnswerOperation as error:
        return {"status": "unsupported", "error": str(error)}
    except ValueError as error:
        return {"status": "invalid", "error": str(error)}
    # Encoding failures are worker failures, never evidence of a wrong answer.
    return {
        "status": "ok",
        "answer": {k: encode(v) for k, v in values.items()},
    }


def main() -> None:
    code = sys.stdin.buffer.read(8_000_001)
    if len(code) > 8_000_000:
        raise ValueError("worker input too large")
    print(json.dumps(run(code.decode("utf-8")), allow_nan=False))


if __name__ == "__main__":
    main()
