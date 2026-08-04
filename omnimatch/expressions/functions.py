# -*- coding: utf-8 -*-
"""Utility functions for working with expression trees."""
from typing import Iterator, Tuple, Set, Dict, Optional

from .expressions import Expression, Operation, Wildcard, NamedAtom, SymbolWrapper, Pattern

__all__ = [
    'is_constant', 'is_syntactic', 'is_anonymous', 'contains_variables_from_set',
    'create_operation_expression', 'preorder_iter_with_position', 'preorder_iter',
    'rename_variables', 'op_iter', 'op_len', 'match_head', 'get_variables',
]


def is_constant(expression: Expression) -> bool:
    """Check if the expression is constant (contains no wildcards)."""
    if isinstance(expression, Wildcard):
        return False
    if isinstance(expression, Operation):
        return all(is_constant(o) for o in expression.operands)
    return True


def is_syntactic(expression: Expression) -> bool:
    """Check if the expression is syntactic (no associative/commutative/one_identity ops or sequence wildcards)."""
    if isinstance(expression, Operation):
        if expression.head.associative or expression.head.commutative or expression.head.one_identity:
            return False
        return all(is_syntactic(o) for o in expression.operands)
    if isinstance(expression, Wildcard):
        if not expression.fixed_size:
            return False
    return True


def is_anonymous(expression: Expression) -> bool:
    """Check if the expression has no variable names."""
    if getattr(expression, 'variable_name', None) is not None:
        return False
    if isinstance(expression, Operation):
        return all(is_anonymous(o) for o in expression.operands)
    return True


def contains_variables_from_set(expression: Expression, variables: Set[str]) -> bool:
    """Check if the expression contains any variable from the given set."""
    if getattr(expression, 'variable_name', None) in variables:
        return True
    if isinstance(expression, Operation):
        return any(contains_variables_from_set(o, variables) for o in expression.operands)
    return False


def create_operation_expression(old_operation, new_operands, variable_name=True):
    """Create a new operation expression with the same head but different operands.

    Uses the raw Operation constructor (associative flattening + commutative sorting
    but NO one_identity). This preserves structure needed by the matching internals.
    """
    if variable_name is True:
        variable_name = getattr(old_operation, 'variable_name', None)
    return Operation(old_operation.head, *new_operands, variable_name=variable_name)


def preorder_iter_with_position(expression, position=()):
    """Iterate over all subexpressions with their positions (depth-first, pre-order).

    Yields:
        (expression, position) tuples.
    """
    yield expression, position
    if isinstance(expression, Operation):
        for i, operand in enumerate(expression.operands):
            yield from preorder_iter_with_position(operand, position + (i,))


def preorder_iter(expression, predicate=None):
    """Iterate over all subexpressions (depth-first, pre-order).

    Args:
        expression: The root expression to iterate over.
        predicate: Optional filter function. Only yields expressions matching the predicate.

    Yields:
        Subexpressions matching the predicate.
    """
    if predicate is None or predicate(expression):
        yield expression
    if isinstance(expression, Operation):
        for operand in expression.operands:
            yield from preorder_iter(operand, predicate)


def rename_variables(expression: Expression, renaming: Dict[str, str]) -> Expression:
    """Return a copy of the expression with variables renamed according to the renaming dict."""
    return expression.with_renamed_vars(renaming)


def op_iter(operation):
    """Iterate over the operands of an Operation, or elements of a sequence.

    For Operation objects, iterates over operation.operands.
    For other iterables (deque, list, tuple, Multiset), iterates directly.
    This dual-use pattern is needed because many_to_one.py calls op_iter(subjects)
    where subjects can be either an Operation or a deque/Multiset of expressions.
    """
    if isinstance(operation, Operation):
        return iter(operation.operands)
    return iter(operation)


def op_len(operation):
    """Get the number of operands of an Operation, or length of a sequence.

    For Operation objects, returns len(operation.operands).
    For other sequences, returns len(operation).
    This dual-use pattern matches op_iter.
    """
    if isinstance(operation, Operation):
        return len(operation.operands)
    return len(operation)


def match_head(subject, pattern) -> bool:
    """Check if the subject could potentially match the pattern based on head type.

    Used as a pre-filter before full matching. Returns True if the subject's head
    is compatible with the pattern's head. Wildcards are compatible with any head.

    Args:
        subject: The subject expression.
        pattern: A Pattern, Operation, Wildcard, or other Expression.

    Returns:
        True if the heads are compatible and full matching should be attempted.
    """
    # Extract expression from Pattern wrapper
    if isinstance(pattern, Pattern):
        pattern = pattern.expression

    # Wildcards match any subject
    if isinstance(pattern, Wildcard):
        return True

    # Operations: one_identity patterns can match any subject (collapse may occur)
    if isinstance(pattern, Operation):
        if pattern.head.one_identity:
            return True
        return isinstance(subject, Operation) and subject.head == pattern.head

    # Symbols require subject to be the same symbol
    if isinstance(pattern, NamedAtom):
        return isinstance(subject, NamedAtom)

    # SymbolWrappers match other SymbolWrappers
    if isinstance(pattern, SymbolWrapper):
        return isinstance(subject, SymbolWrapper)

    # Fallback: type match
    return type(subject) == type(pattern)


def get_variables(expression: Expression) -> Set[str]:
    """Get all variable names in the expression."""
    variables = set()
    _collect_variables(expression, variables)
    return variables


def _collect_variables(expression: Expression, variables: Set[str]) -> None:
    """Recursively collect variable names."""
    vname = getattr(expression, 'variable_name', None)
    if vname is not None:
        variables.add(vname)
    if isinstance(expression, Operation):
        for operand in expression.operands:
            _collect_variables(operand, variables)
