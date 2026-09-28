import ast
import re
from hashlib import sha256
from pathlib import Path

from ddsr_bench.benchmarks.critpt.generation.prompts import load_prompts
from ddsr_bench.benchmarks.utils import read_json

from .schemas import AIAnswerSpec, AIProblem, AIProblemSpec, CritPtSource
from .utils import require_text

_PYTHON = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _reference(template: str, code: str, expected: object, categorical: bool) -> str:
    """Lightly check literals or expression syntax without executing code.
    This is not correctness grading; unsupported comparisons are 'unresolved'.
    """
    try:
        trees = [ast.parse(text) for text in (template, code)]
        functions = [
            [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "answer"
            ]
            for tree in trees
        ]
        if any(len(items) != 1 for items in functions):
            return "unresolved"
        original, reference = (items[0] for items in functions)
        body = reference.body
        reference.body = original.body
        # Only the function body should change. Additional setup needs later grading.
        if ast.dump(trees[0]) != ast.dump(trees[1]):
            return "unresolved"
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return "unresolved"
        expression = body[0].value
        if expression is None:
            return "unresolved"
        if isinstance(expected, str):
            # SymPy's conventional namespace differs from final_answer expressions.
            text = ast.unparse(expression)
            if "import sympy as sp" in code:
                text = re.sub(r"\bsp\.", "", text)
            same = ast.dump(ast.parse(text, mode="eval")) == ast.dump(
                ast.parse(expected, mode="eval")
            )
            # Different expressions may still be mathematically equivalent.
            return "matched" if same else "unresolved"
        actual = ast.literal_eval(expression)
        if actual is None:
            return "unresolved"
        if categorical:
            if not isinstance(actual, (list, tuple, set)) or not isinstance(
                expected, list
            ):
                return "unresolved"
            actual, expected = sorted(actual), sorted(expected)
        return "matched" if actual == expected else "different"
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return "unresolved"


def _answer(record: dict, path: Path) -> tuple[str, str, str]:
    """Extract the template, answer instructions, and reference code."""
    instructions = require_text(record, "public_solver_output_instructions", path)
    blocks = list(_PYTHON.finditer(instructions))
    if len(blocks) != 1:
        raise ValueError(f"{path}: output instructions require one Python template")
    block = blocks[0]
    template = block[1].strip() + "\n"
    solution = require_text(record, "reference_solution", path)
    references = _PYTHON.findall(solution)
    if not references:
        raise ValueError(f"{path}: reference solution requires a final Python block")
    code = references[-1].strip() + "\n"
    instructions = (instructions[: block.start()] + instructions[block.end() :]).strip()
    return template, instructions, code


def load_problem(path: str | Path) -> AIProblem:
    """Split public inputs from references for rule-based or LLM-based grading."""
    path = Path(path)
    record = read_json(path)
    reference_check = None
    if isinstance(record.get("grading_plan"), dict):
        # Answers graded by rules: executable reference and deterministic grading plan.
        final = record.get("final_answer")
        if not isinstance(final, dict) or set(final) != {"answer"}:
            raise ValueError(f"{path}: final_answer must contain only 'answer'")
        template, instructions, code = _answer(record, path)
        reference_check = _reference(
            template,
            code,
            final["answer"],
            record["answer_contract"].get("answer_type") == "categorical",
        )
        answer_key = "final_answer"
        grader = "rule"
    elif "grading_plan" not in record and isinstance(record.get("answer_items"), list):
        # Answers graded by an LLM: reference answer and items for the judge.
        default = load_prompts(CritPtSource.AI)
        instructions = require_text(record, "public_solver_output_instructions", path)
        # Use the default only when it matches; retain source-specific guidance.
        if instructions == default["ANSWER_INSTRUCTIONS"]:
            instructions = None
        template, code = "", ""
        answer_key = "reference_answer"
        grader = "llm"
        require_text(record, answer_key, path)
        require_text(record, "reference_solution", path)
    else:
        raise TypeError(
            f"{path}: expected grading_plan for rules or answer_items for an LLM judge"
        )
    spec = AIProblemSpec(
        id=f"ai-{sha256(path.read_bytes()).hexdigest()}",
        type="main",
        index=None,
        statement=require_text(record, "problem_statement", path),
        code_template=template,
        answer_instructions=instructions,
        grader=grader,
        source=CritPtSource.AI,
        source_path=path,
    )
    require_text(record, "problem_id", path)
    private = {
        key: value
        for key, value in record.items()
        if key not in ("problem_statement", "public_solver_output_instructions")
    }
    # Remove only the final_answer wrapper; preserve the answer's native JSON type.
    snippet = private.pop(answer_key)
    if answer_key == "final_answer":
        snippet = snippet["answer"]
    return AIProblem(
        spec,
        AIAnswerSpec(
            code, snippet, testcases=None, reference_check=reference_check, **private
        ),
    )


def load_problems(root: str | Path) -> list[AIProblem]:
    """Load only problem.json files, not indexes or audit observations."""
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("problem.json"))
    if not paths:
        raise ValueError(f"no AI problem files found under {root}")
    problems = [load_problem(path) for path in paths]
    if len({problem.spec.id for problem in problems}) != len(problems):
        raise ValueError("duplicate AI problem files")
    return problems
