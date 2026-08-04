# -*- coding: utf-8 -*-
"""This module contains the CommutativePatternsParts class which is used by multiple matching algorithms."""
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Sequence, Type, cast  # pylint: disable=unused-import

from multiset import Multiset

from ..expressions.expressions import Expression, Operation, OperationHead, Wildcard
from ..expressions.substitution import Substitution
from ..expressions.functions import is_constant, is_syntactic, op_iter

__all__ = ['CommutativePatternsParts', 'Matcher', 'VarInfo', 'OPERATION_END']

OPERATION_END = ')'
"""Sentinel written after an operation's operands in the flattened pattern/subject streams."""

Matcher = Callable[[Sequence[Expression], Expression, Substitution], Iterator[Substitution]]
VarInfo = NamedTuple('VarInfo', [('min_count', int), ('default', Optional[Expression])])


class CommutativePatternsParts(object):
    """Representation of the parts of a commutative pattern expression.

    This data structure contains all the operands of a commutative operation pattern.
    They are distinguished by how they need to be matched against an expression.

    All parts are represented by a :class:`.Multiset`, because the order of operands does not matter
    in a commutative operation.

    In addition, some lengths are precalculated during the initialization of this data structure
    so that they do not have to be recalculated later.

    This data structure is meant to be immutable, so do not change any of its attributes!

    Attributes:
        operation (OperationHead):
            The OperationHead of the original pattern expression.

        constant (Multiset):
            A :class:`~.Multiset` representing the constant operands of the pattern.
        syntactic (Multiset[Operation]):
            A :class:`.Multiset` representing the syntactic operands of the pattern.
        sequence_variables (Multiset[str]):
            A :class:`.Multiset` representing the sequence variables of the pattern.
        sequence_variable_infos (Dict[str, VarInfo]):
            A dictionary mapping sequence variable names to more information about the variable.
        fixed_variables (Multiset[str]):
            A :class:`.Multiset` representing the fixed length variables of the pattern.
        fixed_variable_infos (Dict[str, VarInfo]):
            A dictionary mapping fixed variable names to more information about the variable.
        rest (Multiset):
            A :class:`.Multiset` representing the non-syntactic operation operands.

        length (int):
            The total count of operands of the commutative operation pattern.
        sequence_variable_min_length (int):
            The total combined minimum length of all sequence variables.
        fixed_variable_length (int):
            The total combined length of all fixed length variables.
        wildcard_fixed (Optional[bool]):
            Iff none of the operands is an unnamed wildcards, it is ``None``.
            Iff there are any unnamed sequence wildcards, it is ``True``.
            Otherwise, it is ``False``.
        wildcard_min_length (int):
            If :attr:`wildcard_fixed` is not ``None``, this is the total combined minimum length of all unnamed
            wildcards.
    """

    def __init__(self, operation, *expressions: Expression) -> None:
        """Create a CommutativePatternsParts instance.

        Args:
            operation:
                The OperationHead of the commutative operation (must have commutative=True).
            *expressions:
                The operands of the commutative operation.
        """
        self.operation = operation
        self.length = len(expressions)

        self.constant = Multiset()  # type: Multiset
        self.syntactic = Multiset()  # type: Multiset
        self.sequence_variables = Multiset()  # type: Multiset[str]
        self.sequence_variable_infos = dict()
        self.fixed_variables = Multiset()  # type: Multiset[str]
        self.fixed_variable_infos = dict()
        self.rest = Multiset()  # type: Multiset

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0
        self.wildcard_min_length = 0
        self.optional_count = 0
        self.wildcard_fixed = None

        for expression in expressions:
            expression = expression
            if is_constant(expression):
                self.constant[expression] += 1
            elif isinstance(expression, Wildcard):
                wc = cast(Wildcard, expression)
                if wc.variable_name:
                    name = wc.variable_name
                    if wc.fixed_size:
                        self.fixed_variables[name] += 1
                        self._update_var_info(self.fixed_variable_infos, name, wc.min_count, wc.default_value)
                        if wc.default_value is None:
                            self.fixed_variable_length += wc.min_count
                        else:
                            self.optional_count += 1
                    else:
                        self.sequence_variables[name] += 1
                        self._update_var_info(self.sequence_variable_infos, name, wc.min_count, wc.default_value)
                        if wc.default_value is None:
                            self.sequence_variable_min_length += wc.min_count
                else:
                    self.wildcard_min_length += wc.min_count
                    if self.wildcard_fixed is None:
                        self.wildcard_fixed = wc.fixed_size
                    else:
                        self.wildcard_fixed = self.wildcard_fixed and wc.fixed_size
            elif is_syntactic(expression):
                self.syntactic[expression] += 1
            else:
                self.rest[expression] += 1

    @staticmethod
    def _update_var_info(infos, name, count, default=None):
        if name not in infos:
            infos[name] = VarInfo(count, default)
        else:
            existing_info = infos[name]
            assert existing_info.min_count == count
            assert existing_info.default == default

    def __str__(self):
        parts = []
        parts.extend(map(str, self.constant))
        parts.extend(map(str, self.syntactic))
        parts.extend(map(str, self.rest))

        for name, count in self.sequence_variables.items():
            parts.extend([name] * count)

        for name, count in self.fixed_variables.items():
            parts.extend([name] * count)

        return '{}({})'.format(self.operation.name if isinstance(self.operation, OperationHead) else str(self.operation), ', '.join(parts))

def check_one_identity(operation):
    added_subst = Substitution()
    non_optional = None
    for operand in op_iter(operation):
        if isinstance(operand, Wildcard):
            try:
                if operand.default_value is not None:
                    added_subst.try_add_variable(operand.variable_name, operand.default_value)
                    continue
                elif operand.min_count == 0:
                    value = Multiset() if (isinstance(operation, Operation) and operation.head.commutative) else ()
                    added_subst.try_add_variable(operand.variable_name, value)
                    continue
            except ValueError:
                return None, None
        if non_optional is None:
            non_optional = operand
        else:
            return None, None
    return non_optional, added_subst
