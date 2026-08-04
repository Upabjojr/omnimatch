# -*- coding: utf-8 -*-
"""Tests for WILDCARD OPERATION HEADS.

A pattern built with :class:`WildcardOperationHead` matches an application of ANY
operation head (e.g. Rubi's ``F_[v_]``, "any function F applied to v"), binding the
matched head to a variable while its operands match in the ordinary way.

Covered here:
  * head binding (anonymous heads, repeated head variables, several heads, backtracking)
  * operand kinds: concrete / dot / optional / star / plus / nested
  * every combination of commutative x associative x one_identity, both on the
    SUBJECT's head and on an enclosing operation
  * nesting: a wildcard head applied to a wildcard-head application, and deep
    inside ordinary structure
  * arity and the negative cases (symbols, wrong arity)
  * non-regression: concrete-head patterns keep their exact previous behaviour
"""
import itertools

import pytest

from omnimatch.expressions.expressions import (
    Arity, Operation, OperationHead, Pattern, NamedAtom, Wildcard,
    WildcardOperationHead,
)
from omnimatch.matching.many_to_one import ManyToOneMatcher

# ── a spread of concrete heads ───────────────────────────────────────────────
SIN = OperationHead(name='SIN', arity=Arity.unary)
COS = OperationHead(name='COS', arity=Arity.unary)
F2 = OperationHead(name='F2', arity=Arity.binary)
VAR = OperationHead(name='VAR', arity=Arity.variadic)
LST = OperationHead(name='LST', arity=Arity.variadic)
ADD = OperationHead(name='ADD', arity=Arity.variadic, commutative=True,
                    associative=True, one_identity=True)
MUL = OperationHead(name='MUL', arity=Arity.variadic, commutative=True,
                    associative=True, one_identity=True)

a, b, x, y, z, w = (NamedAtom(n) for n in ('a', 'b', 'x', 'y', 'z', 'w'))


def anyhead(var='F'):
    """A wildcard operation head binding the matched head to `var`."""
    return WildcardOperationHead(name='__any__', variable_name=var)


def matches(pattern, subject):
    m = ManyToOneMatcher()
    m.add(pattern, label='p')
    return list(m.match(subject))


def one(pattern, subject):
    got = matches(pattern, subject)
    assert len(got) == 1, f"expected exactly 1 match, got {len(got)}"
    return got[0][1]


# =============================================================================
# Head binding
# =============================================================================

class TestHeadBinding:

    @pytest.mark.parametrize('head', [SIN, COS, VAR, ADD, MUL])
    def test_binds_any_head(self, head):
        subst = one(Pattern(Operation(anyhead(), Wildcard.dot('v'))),
                    Operation(head, x))
        assert subst['F'].value is head
        assert subst['v'] == x

    def test_anonymous_wildcard_head_matches_but_binds_nothing(self):
        pat = Pattern(Operation(WildcardOperationHead(name='__any__'),
                                Wildcard.dot('v')))
        subst = one(pat, Operation(SIN, x))
        assert 'F' not in subst
        assert subst['v'] == x

    def test_repeated_head_variable_must_be_consistent(self):
        """Two F_[..] in one pattern must bind the SAME head."""
        pat = Pattern(Operation(LST,
                                Operation(anyhead('F'), Wildcard.dot('u')),
                                Operation(anyhead('F'), Wildcard.dot('v'))))
        subst = one(pat, Operation(LST, Operation(SIN, x), Operation(SIN, y)))
        assert subst['F'].value is SIN
        assert matches(pat, Operation(LST, Operation(SIN, x), Operation(COS, y))) == []

    def test_two_independent_head_variables(self):
        pat = Pattern(Operation(LST,
                                Operation(anyhead('F'), Wildcard.dot('u')),
                                Operation(anyhead('G'), Wildcard.dot('v'))))
        subst = one(pat, Operation(LST, Operation(SIN, x), Operation(COS, y)))
        assert subst['F'].value is SIN
        assert subst['G'].value is COS
        subst = one(pat, Operation(LST, Operation(SIN, x), Operation(SIN, y)))
        assert subst['F'].value is SIN and subst['G'].value is SIN

    def test_binding_is_undone_on_backtrack(self):
        """A failed branch must not leave a stale head binding behind."""
        pat = Pattern(Operation(LST, Operation(anyhead('F'), Wildcard.dot('u')), a))
        assert matches(pat, Operation(LST, Operation(SIN, x), b)) == []
        subst = one(pat, Operation(LST, Operation(COS, y), a))
        assert subst['F'].value is COS


# =============================================================================
# Operand kinds
# =============================================================================

class TestOperandKinds:

    def test_concrete_operand(self):
        pat = Pattern(Operation(anyhead(), a))
        assert one(pat, Operation(SIN, a))['F'].value is SIN
        assert matches(pat, Operation(SIN, b)) == []

    def test_nested_operation_operand_binds_inner_wildcards(self):
        """Arguments match natively, so inner wildcards bind during the match."""
        pat = Pattern(Operation(anyhead(),
                                Operation(ADD, Wildcard.dot('p'),
                                          Operation(MUL, Wildcard.dot('q'), x))))
        subst = one(pat, Operation(SIN, Operation(ADD, a, Operation(MUL, b, x))))
        assert subst['F'].value is SIN
        assert subst['p'] == a
        assert subst['q'] == b

    def test_optional_wildcard_operand_uses_default_when_absent(self):
        pat = Pattern(Operation(anyhead(), Wildcard.optional('c', a),
                                Wildcard.dot('v')))
        subst = one(pat, Operation(VAR, b, x))
        assert subst['c'] == b and subst['v'] == x
        subst = one(pat, Operation(VAR, x))
        assert subst['c'] == a and subst['F'].value is VAR

    def test_star_sequence_operand(self):
        pat = Pattern(Operation(anyhead(), Wildcard.star('s')))
        for args in ([], [x], [x, y, z]):
            subst = one(pat, Operation(VAR, *args))
            assert subst['F'].value is VAR
            assert tuple(subst['s']) == tuple(args)

    def test_plus_sequence_operand_requires_at_least_one(self):
        pat = Pattern(Operation(anyhead(), Wildcard.plus('s')))
        assert matches(pat, Operation(VAR)) == []
        assert tuple(one(pat, Operation(VAR, x, y))['s']) == (x, y)

    def test_mixed_fixed_and_sequence_operands(self):
        pat = Pattern(Operation(anyhead(), Wildcard.dot('first'),
                                Wildcard.star('rest')))
        subst = one(pat, Operation(VAR, x, y, z))
        assert subst['first'] == x
        assert tuple(subst['rest']) == (y, z)


# =============================================================================
# commutative x associative x one_identity
# =============================================================================

_FLAGS = list(itertools.product([False, True], repeat=3))
_FLAG_IDS = [f"c{int(c)}a{int(s)}o{int(o)}" for c, s, o in _FLAGS]


def _head(name, commutative, associative, one_identity):
    return OperationHead(name=name, arity=Arity.variadic, commutative=commutative,
                         associative=associative, one_identity=one_identity)


class TestHeadFlagCombinations:

    @pytest.mark.parametrize('commutative,associative,one_identity', _FLAGS,
                             ids=_FLAG_IDS)
    def test_matches_subject_head_with_any_flags(self, commutative, associative,
                                                 one_identity):
        """The SUBJECT's head may carry any flag combination."""
        h = _head('H', commutative, associative, one_identity)
        subst = one(Pattern(Operation(anyhead(), Wildcard.dot('u'), Wildcard.dot('v'))),
                    Operation(h, x, y))
        assert subst['F'].value == h
        assert {subst['u'], subst['v']} == {x, y}

    @pytest.mark.parametrize('commutative,associative,one_identity', _FLAGS,
                             ids=_FLAG_IDS)
    def test_nested_in_enclosing_operation_with_any_flags(self, commutative,
                                                          associative, one_identity):
        """The ENCLOSING operation may carry any flag combination."""
        outer = _head('OUT', commutative, associative, one_identity)
        pat = Pattern(Operation(outer, Wildcard.dot('u'),
                                Operation(anyhead(), Wildcard.dot('v'))))
        subst = one(pat, Operation(outer, w, Operation(SIN, y)))
        assert subst['F'].value is SIN
        assert subst['v'] == y and subst['u'] == w

    def test_commutative_enclosing_matches_in_any_operand_order(self):
        pat = Pattern(Operation(MUL, Wildcard.dot('u'),
                                Operation(anyhead(), Wildcard.dot('v'))))
        for subject in (Operation(MUL, w, Operation(SIN, y)),
                        Operation(MUL, Operation(SIN, y), w)):
            subst = one(pat, subject)
            assert subst['F'].value is SIN
            assert subst['v'] == y and subst['u'] == w

    def test_associative_subject_is_flattened_before_matching(self):
        assoc = OperationHead(name='ASSOC', arity=Arity.variadic, associative=True)
        subject = Operation(assoc, x, Operation(assoc, y, z))   # -> (x, y, z)
        subst = one(Pattern(Operation(anyhead(), Wildcard.dot('p'),
                                      Wildcard.dot('q'), Wildcard.dot('r'))),
                    subject)
        assert subst['F'].value == assoc
        assert (subst['p'], subst['q'], subst['r']) == (x, y, z)

    def test_one_identity_subject_collapses_and_is_then_not_an_operation(self):
        """head(x) with one_identity collapses to x, which has no head to bind."""
        oi = _head('OI', False, False, True)
        assert oi(x) == x                                    # factory collapses
        assert matches(Pattern(Operation(anyhead(), Wildcard.dot('v'))), oi(x)) == []


# =============================================================================
# Nesting
# =============================================================================

class TestNesting:

    def test_double_nested_wildcard_heads(self):
        pat = Pattern(Operation(anyhead('F'),
                                Operation(anyhead('G'), Wildcard.dot('v'))))
        subst = one(pat, Operation(SIN, Operation(COS, x)))
        assert subst['F'].value is SIN
        assert subst['G'].value is COS
        assert subst['v'] == x

    def test_triple_nested_wildcard_heads(self):
        pat = Pattern(Operation(anyhead('F'),
                                Operation(anyhead('G'),
                                          Operation(anyhead('H'), Wildcard.dot('v')))))
        subst = one(pat, Operation(SIN, Operation(COS, Operation(VAR, x))))
        assert (subst['F'].value, subst['G'].value, subst['H'].value) == (SIN, COS, VAR)
        assert subst['v'] == x

    def test_double_nesting_requires_matching_depth(self):
        pat = Pattern(Operation(anyhead('F'),
                                Operation(anyhead('G'), Wildcard.dot('v'))))
        assert matches(pat, Operation(SIN, x)) == []

    def test_nested_double_head_must_be_consistent_when_sharing_a_variable(self):
        pat = Pattern(Operation(anyhead('F'),
                                Operation(anyhead('F'), Wildcard.dot('v'))))
        assert one(pat, Operation(SIN, Operation(SIN, x)))['F'].value is SIN
        assert matches(pat, Operation(SIN, Operation(COS, x))) == []

    def test_wildcard_head_deep_inside_ordinary_structure(self):
        pat = Pattern(Operation(ADD, Wildcard.dot('u'),
                                Operation(MUL, Wildcard.dot('c'),
                                          Operation(anyhead(), Wildcard.dot('v')))))
        subst = one(pat, Operation(ADD, a, Operation(MUL, b, Operation(SIN, x))))
        assert subst['F'].value is SIN
        assert subst['v'] == x


# =============================================================================
# Arity / negative cases
# =============================================================================

class TestArityAndNegatives:

    def test_arity_is_constrained_by_the_operand_pattern(self):
        unary = Pattern(Operation(anyhead(), Wildcard.dot('v')))
        assert len(matches(unary, Operation(SIN, x))) == 1
        assert matches(unary, Operation(F2, x, y)) == []
        binary = Pattern(Operation(anyhead(), Wildcard.dot('u'), Wildcard.dot('v')))
        assert len(matches(binary, Operation(F2, x, y))) == 1
        assert matches(binary, Operation(SIN, x)) == []

    def test_does_not_match_a_plain_symbol(self):
        assert matches(Pattern(Operation(anyhead(), Wildcard.dot('v'))), x) == []

    def test_star_operand_still_requires_an_operation(self):
        assert matches(Pattern(Operation(anyhead(), Wildcard.star('s'))), x) == []


# =============================================================================
# Non-regression: concrete-head behaviour is unchanged
# =============================================================================

class TestConcreteHeadsUnaffected:

    def test_concrete_head_pattern_does_not_broaden(self):
        pat = Pattern(Operation(SIN, Wildcard.dot('v')))
        assert len(matches(pat, Operation(SIN, x))) == 1
        assert matches(pat, Operation(COS, x)) == []

    def test_wildcard_head_not_equal_to_plain_head_of_same_name(self):
        assert WildcardOperationHead(name='SIN', variable_name='F') != SIN
        assert SIN != WildcardOperationHead(name='SIN', variable_name='F')

    def test_wildcard_heads_compare_by_variable_name(self):
        assert anyhead('F') == anyhead('F')
        assert anyhead('F') != anyhead('G')

    def test_wildcard_head_is_hashable_and_distinct_from_plain_head(self):
        assert len({anyhead('F'), anyhead('F'), anyhead('G'), SIN}) == 3

    def test_concrete_and_wildcard_patterns_coexist_in_one_matcher(self):
        m = ManyToOneMatcher()
        m.add(Pattern(Operation(SIN, Wildcard.dot('v'))), label='concrete')
        m.add(Pattern(Operation(anyhead(), Wildcard.dot('v'))), label='generic')
        assert {lbl for lbl, _ in m.match(Operation(SIN, x))} == {'concrete', 'generic'}
        assert {lbl for lbl, _ in m.match(Operation(COS, x))} == {'generic'}
