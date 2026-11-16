from __future__ import annotations

from typing import Any

import sympy as sp

from .symbolic import parse_symbolic


def _simplify_fermions(expression: sp.Expr, operators: dict[str, Any]) -> sp.Expr:
    """Apply the official scorer's one-pair fermionic normal ordering."""
    names = [
        name
        for name in operators
        if not name.endswith("_dagger") and f"{name}_dagger" in operators
    ]
    bases = [operators[name] for name in names]
    daggers = [operators[f"{name}_dagger"] for name in names]
    base_index = {operator: index for index, operator in enumerate(bases)}
    dagger_index = {operator: index for index, operator in enumerate(daggers)}

    def head(value: sp.Expr) -> Any:
        if value in bases or value in daggers:
            return value
        return getattr(value, "func", None)

    def rewrite(product: sp.Expr) -> sp.Expr:
        factors = list(product.args)
        for index in range(len(factors) - 1):
            left, right = factors[index : index + 2]
            left_head, right_head = head(left), head(right)
            if left_head in bases and right_head in bases:
                left_index = base_index[left_head]
                right_index = base_index[right_head]
                if left_index == right_index:
                    return sp.Integer(0)
                if left_index > right_index:
                    swapped = factors[:index] + [right, left] + factors[index + 2 :]
                    return -sp.Mul(*swapped)
            elif left_head in daggers and right_head in daggers:
                left_index = dagger_index[left_head]
                right_index = dagger_index[right_head]
                if left_index == right_index:
                    return sp.Integer(0)
                if left_index > right_index:
                    swapped = factors[:index] + [right, left] + factors[index + 2 :]
                    return -sp.Mul(*swapped)
            elif left_head in bases and right_head in daggers:
                same_mode = base_index[left_head] == dagger_index[right_head]
                ordered = 1 - right * left if same_mode else -right * left
                return sp.Mul(*(factors[:index] + [ordered] + factors[index + 2 :]))
        return product

    def simplify_term(term: sp.Expr) -> sp.Expr:
        if not isinstance(term, sp.Mul):
            return term
        current = term
        while True:
            rewritten = rewrite(current)
            if rewritten == current:
                return current
            current = rewritten

    if expression.is_Add:
        return sum((simplify_term(term) for term in expression.args), sp.S.Zero)
    return simplify_term(expression)


def equivalent_functional(
    model: str,
    reference: str,
    parameter_text: str = "",
    function_text: str = "",
) -> bool:
    """Compare two boxed responses using the official functional-form rules."""
    expected = parse_symbolic(reference, parameter_text, function_text)
    actual = parse_symbolic(model, parameter_text, function_text)
    if len(actual.expressions) != len(expected.expressions):
        return False

    operators = {**expected.parameters, **expected.functions}
    fermionic = any("_dagger" in name for name in operators)
    matches = []
    for actual_expression in actual.expressions:
        row = []
        for expected_expression in expected.expressions:
            difference = (actual_expression - expected_expression).doit()
            if fermionic:
                difference = _simplify_fermions(difference, operators)
            row.append(difference.simplify() == 0)
        matches.append(row)
    return all(any(row) for row in matches)
