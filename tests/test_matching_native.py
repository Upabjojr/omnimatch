# -*- coding: utf-8 -*-
from hypothesis import assume, given
import hypothesis.strategies as st
import pytest
from multiset import Multiset

from omnimatch.expressions.constraints import CustomConstraint
from omnimatch.expressions.expressions import (
    NamedAtom, Wildcard, Pattern, to_omnimatch_expression, from_omnimatch_expression,
    Operation, OperationHead, Arity
)
from omnimatch.matching.many_to_one import ManyToOneMatcher
from omnimatch.matching.one_to_one import match
from omnimatch.functions import substitute
from .utils import MockConstraint
from .common import *


@pytest.mark.parametrize(
    '   expression,             pattern,            expected_matches',
    [
        ({'b': 0},              {'a': x_},          []),
        (('a', 0),              {'a': x_},          []),
        ({'a': 0},              {'a': x_},          [{'x': NamedAtom('0')}]),
        ({'a': 0},              {x_: 0},            [{'x': NamedAtom('a')}]),
        ({'a': 0, 'b': 1},      {x_: 0, _: _},      [{'x': NamedAtom('a')}]),
        ({'a': 0, 'b': 0},      {x_: 0, _: _},      [{'x': NamedAtom('a')}, {'x': NamedAtom('b')}]),
        ({'a': 0, 'b': 0},      {'a': _, 'b': _},   [{}]),
        ({'a': 0, 'b': 0},      {'a': _, 'c': _},   []),
    ]
)  # yapf: disable
def test_dict_match(match, expression, pattern, expected_matches):
    expression = to_omnimatch_expression(expression)
    pattern = Pattern(to_omnimatch_expression(pattern))
    result = list(match(expression, pattern))
    for expected_match in expected_matches:
        assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
            expression, pattern, expected_match
        )
    for result_match in result:
        assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
            expression, pattern, result_match
        )


# ── MySum singledispatch tests ───────────────────────────────────────────────
# Demonstrates extending to_omnimatch_expression/from_omnimatch_expression with a user-defined class.

class MySum:
    """A plain Python class (not an Expression) representing a sum of terms."""
    def __init__(self, *terms):
        self.terms = terms

    def __repr__(self):
        return 'MySum({})'.format(', '.join(repr(t) for t in self.terms))

    def __eq__(self, other):
        return isinstance(other, MySum) and self.terms == other.terms


# Register to_omnimatch_expression for MySum
MYSUM_HEAD = OperationHead(name='MySum', arity=Arity.variadic, commutative=True)


@to_omnimatch_expression.register(MySum)
def _mysum_to_expression(obj: MySum):
    return Operation(MYSUM_HEAD, *(to_omnimatch_expression(t) for t in obj.terms))


def mysum_from_expression(expr):
    """Convert a MYSUM_HEAD Operation back to MySum (local helper, not globally registered)."""
    assert isinstance(expr, Operation) and expr.head == MYSUM_HEAD
    return MySum(*(from_omnimatch_expression(op) for op in expr.operands))


class TestMySumSingledispatch:
    """Tests that a user-defined class can be registered with to_omnimatch_expression and used for matching."""

    def test_to_expression_basic(self):
        """MySum converts to a commutative Operation."""
        expr = to_omnimatch_expression(MySum(1, 2, 3))
        assert isinstance(expr, Operation)
        assert expr.head == MYSUM_HEAD
        # Commutative, so operands are sorted
        assert len(expr.operands) == 3

    def test_from_expression_roundtrip(self):
        """to_omnimatch_expression and a custom from_omnimatch_expression are inverses (for string terms)."""
        # Note: integers become NamedAtom(str(n)), so roundtrip only preserves string terms.
        # Commutative operations sort operands, so use already-sorted terms.
        original = MySum('a', 'b', 'c')
        expr = to_omnimatch_expression(original)
        recovered = mysum_from_expression(expr)
        assert recovered == original

    def test_match_wildcard(self):
        """Wildcard matches inside MySum."""
        subject = to_omnimatch_expression(MySum(1, 2))
        pattern = Pattern(Operation(MYSUM_HEAD, Wildcard.dot('x'), Wildcard.dot('y')))
        results = list(match(subject, pattern))
        # Commutative: both orderings should match
        assert len(results) >= 1
        # Check that both values are found in some match
        values = set()
        for subst in results:
            values.add(subst['x'])
            values.add(subst['y'])
        assert NamedAtom('1') in values
        assert NamedAtom('2') in values

    def test_match_specific_value(self):
        """Match a specific symbol inside MySum."""
        subject = to_omnimatch_expression(MySum(1, 2, 3))
        # Pattern: MySum(1, x_, y_) — match with 1 as a fixed element
        pattern = Pattern(Operation(MYSUM_HEAD, NamedAtom('1'), Wildcard.dot('x'), Wildcard.dot('y')))
        results = list(match(subject, pattern))
        assert len(results) >= 1
        # x and y should be 2 and 3 (in some order)
        for subst in results:
            matched = {subst['x'], subst['y']}
            assert matched == {NamedAtom('2'), NamedAtom('3')}

    def test_match_no_match(self):
        """No match when a required element is missing."""
        subject = to_omnimatch_expression(MySum(1, 2))
        pattern = Pattern(Operation(MYSUM_HEAD, NamedAtom('5'), Wildcard.dot('x')))
        results = list(match(subject, pattern))
        assert results == []

    def test_match_sequence_variable(self):
        """Sequence wildcard collects remaining terms."""
        subject = to_omnimatch_expression(MySum(1, 2, 3))
        # Pattern: MySum(1, ___) — match 1 and collect the rest
        rest = Wildcard.star('rest')
        pattern = Pattern(Operation(MYSUM_HEAD, NamedAtom('1'), rest))
        results = list(match(subject, pattern))
        assert len(results) >= 1
        # rest should be a multiset of {2, 3} (commutative)
        for subst in results:
            rest_val = subst['rest']
            assert isinstance(rest_val, (Multiset, tuple, list, frozenset))
            rest_set = set(rest_val) if not isinstance(rest_val, Multiset) else set(rest_val)
            assert rest_set == {NamedAtom('2'), NamedAtom('3')}

    def test_many_to_one_matching(self):
        """MySum expressions work with the ManyToOneMatcher."""
        pat1 = Pattern(Operation(MYSUM_HEAD, NamedAtom('1'), Wildcard.dot('x')))
        pat2 = Pattern(Operation(MYSUM_HEAD, NamedAtom('2'), Wildcard.dot('y')))
        matcher = ManyToOneMatcher(pat1, pat2)

        subject = to_omnimatch_expression(MySum(1, 2))
        results = list(matcher.match(subject))
        # Both patterns should match
        labels = [label for label, _ in results]
        assert pat1 in labels
        assert pat2 in labels

    def test_nested_mysum(self):
        """Nested MySum instances convert and match correctly."""
        subject = to_omnimatch_expression(MySum(MySum(1, 2), 3))
        # Pattern: MySum(MySum(x_, y_), z_)
        inner_pat = Operation(MYSUM_HEAD, Wildcard.dot('x'), Wildcard.dot('y'))
        outer_pat = Operation(MYSUM_HEAD, inner_pat, Wildcard.dot('z'))
        pattern = Pattern(outer_pat)
        results = list(match(subject, pattern))
        assert len(results) >= 1
        for subst in results:
            assert subst['z'] == NamedAtom('3')
            matched_inner = {subst['x'], subst['y']}
            assert matched_inner == {NamedAtom('1'), NamedAtom('2')}
