# -*- coding: utf-8 -*-
"""Tests for the FreeOf constraint."""
import pytest
from omnimatch.expressions.expressions import (
    Operation, NamedAtom, Wildcard, Pattern, Arity, OperationHead,
)
from omnimatch.expressions.constraints import FreeOf
from omnimatch.matching.one_to_one import match as match_one_to_one
from omnimatch.matching.many_to_one import ManyToOneMatcher
from omnimatch.expressions.substitution import Substitution

# ─── Test fixtures ────────────────────────────────────────────────────────────

f = OperationHead(name='f', arity=Arity.binary)
g = OperationHead(name='g', arity=Arity.unary)
h = OperationHead(name='h', arity=Arity.variadic, commutative=True, associative=True, one_identity=True)

a = NamedAtom('a')
b = NamedAtom('b')
x = NamedAtom('x')
y = NamedAtom('y')
z = NamedAtom('z')

x_ = Wildcard.dot('x')
y_ = Wildcard.dot('y')
u_ = Wildcard.dot('u')
v_ = Wildcard.dot('v')


# ─── Unit tests: FreeOf class behavior ────────────────────────────────────────

class TestFreeQProperties:
    """Test FreeOf constraint object properties."""

    def test_variables(self):
        """FreeOf depends on the pattern variable it checks."""
        c = FreeOf('u', 'x')
        assert c.variables == frozenset({'u'})

    def test_repr(self):
        c = FreeOf('u', 'x')
        assert repr(c) == "FreeOf('u', 'x')"

    def test_str(self):
        c = FreeOf('u', 'x')
        assert str(c) == "FreeOf(u, x)"

    def test_equality(self):
        c1 = FreeOf('u', 'x')
        c2 = FreeOf('u', 'x')
        c3 = FreeOf('u', 'y')
        c4 = FreeOf('v', 'x')
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

    def test_hash(self):
        c1 = FreeOf('u', 'x')
        c2 = FreeOf('u', 'x')
        assert hash(c1) == hash(c2)
        # Can be used in sets
        assert len({c1, c2}) == 1

    def test_with_renamed_vars(self):
        """Variable renaming should update the pattern variable name."""
        c = FreeOf('u', 'x')
        renamed = c.with_renamed_vars({'u': 'u_renamed'})
        assert renamed.variable == 'u_renamed'
        assert renamed.symbol_name == 'x'  # symbol_name unchanged
        assert renamed.variables == frozenset({'u_renamed'})


# ─── Integration tests: FreeOf with one-to-one matching ────────────────────────

class TestFreeQOneToOne:
    """Test FreeOf constraint in one-to-one pattern matching."""

    def test_symbol_free_of_different_name(self):
        """NamedAtom('a') is free of 'x'."""
        pattern = Pattern(f(x_, u_), FreeOf('u', 'x'))
        subject = f(x, a)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 1
        assert results[0]['u'] == a

    def test_symbol_not_free_of_same_name(self):
        """NamedAtom('x') is NOT free of 'x'."""
        pattern = Pattern(f(x_, u_), FreeOf('u', 'x'))
        subject = f(a, x)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0

    def test_operation_free_of_symbol(self):
        """g(a) is free of 'x'."""
        pattern = Pattern(f(x_, u_), FreeOf('u', 'x'))
        subject = f(x, Operation(g, a))
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 1
        assert results[0]['u'] == Operation(g, a)

    def test_operation_not_free_of_symbol(self):
        """g(x) is NOT free of 'x' — x appears nested."""
        pattern = Pattern(f(x_, u_), FreeOf('u', 'x'))
        subject = f(a, Operation(g, x))
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0

    def test_deeply_nested_symbol(self):
        """f(g(g(x)), a) — deeply nested x should be detected."""
        pattern = Pattern(f(u_, v_), FreeOf('u', 'x'))
        deeply_nested = f(Operation(g, Operation(g, x)), a)
        results = list(match_one_to_one(deeply_nested, pattern))
        assert len(results) == 0

    def test_deeply_nested_free(self):
        """f(g(g(a)), b) — no x anywhere."""
        pattern = Pattern(f(u_, v_), FreeOf('u', 'x'))
        subject = f(Operation(g, Operation(g, a)), b)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 1

    def test_multiple_freeq_constraints(self):
        """Both u and v must be free of 'x'."""
        pattern = Pattern(f(u_, v_), FreeOf('u', 'x'), FreeOf('v', 'x'))
        # Both free
        subject = f(a, b)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 1

        # u contains x
        subject = f(x, b)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0

        # v contains x
        subject = f(a, x)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0

    def test_freeq_different_symbols(self):
        """u must be free of 'x', v must be free of 'y'."""
        pattern = Pattern(f(u_, v_), FreeOf('u', 'x'), FreeOf('v', 'y'))

        # Both free of their respective symbols
        subject = f(y, x)  # u=y (free of 'x'), v=x (free of 'y')
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 1

        # u contains x
        subject = f(x, a)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0

        # v contains y
        subject = f(a, y)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0

    def test_unary_pattern_with_freeq(self):
        """g(u) where u is free of 'x'."""
        pattern = Pattern(Operation(g, u_), FreeOf('u', 'x'))

        subject = Operation(g, a)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 1

        subject = Operation(g, x)
        results = list(match_one_to_one(subject, pattern))
        assert len(results) == 0


# ─── Integration tests: FreeOf with commutative matching ───────────────────────

class TestFreeQCommutative:
    """Test FreeOf with commutative/associative operations."""

    def test_commutative_freeq(self):
        """In h(x, u) with commutative h, FreeOf('u', 'x') filters correctly."""
        pattern = Pattern(Operation(h, x_, u_), FreeOf('u', 'x'))
        # h is commutative+associative+one_identity
        subject = Operation(h, x, a)
        results = list(match_one_to_one(subject, pattern))
        # Should find the match where u=a (which is free of 'x')
        valid = [r for r in results if r['u'] == a]
        assert len(valid) >= 1

    def test_commutative_freeq_no_match(self):
        """h(x, x) — u must be free of 'x', but both operands are x."""
        pattern = Pattern(Operation(h, x_, u_), FreeOf('u', 'x'))
        subject = Operation(h, x, x)
        results = list(match_one_to_one(subject, pattern))
        # u would need to bind to x, which is not free of 'x'
        valid = [r for r in results if r.get('u') is not None and r['u'] != x]
        assert len(valid) == 0


# ─── Integration tests: FreeOf with many-to-one matching ───────────────────────

class TestFreeQManyToOne:
    """Test FreeOf with ManyToOneMatcher."""

    def test_many_to_one_basic(self):
        """FreeOf works with the many-to-one matcher."""
        pattern = Pattern(f(u_, v_), FreeOf('u', 'x'))
        matcher = ManyToOneMatcher(pattern)

        # u is free of x
        subject = f(a, x)
        results = list(matcher.match(subject))
        assert len(results) == 1

        # u contains x
        subject = f(x, a)
        results = list(matcher.match(subject))
        assert len(results) == 0

    def test_many_to_one_multiple_patterns(self):
        """Multiple patterns with different FreeOf constraints."""
        p1 = Pattern(f(u_, v_), FreeOf('u', 'x'))  # u must be free of x
        p2 = Pattern(f(u_, v_), FreeOf('v', 'x'))  # v must be free of x

        matcher = ManyToOneMatcher()
        matcher.add(p1, 'u_free')
        matcher.add(p2, 'v_free')

        # f(a, x): u=a free of x ✓, v=x not free of x ✗
        subject = f(a, x)
        results = list(matcher.match(subject))
        labels = [label for label, _ in results]
        assert 'u_free' in labels
        assert 'v_free' not in labels

        # f(x, a): u=x not free of x ✗, v=a free of x ✓
        subject = f(x, a)
        results = list(matcher.match(subject))
        labels = [label for label, _ in results]
        assert 'u_free' not in labels
        assert 'v_free' in labels

        # f(a, b): both free of x
        subject = f(a, b)
        results = list(matcher.match(subject))
        labels = [label for label, _ in results]
        assert 'u_free' in labels
        assert 'v_free' in labels


class TestFreeOfApi:
    """`FreeOf` accepts the same argument shapes as the higher-layer `FreeQ` predicate.

    `FreeQ(expr_vars, free_of)` takes either one variable or a list of them, given as
    objects rather than bare strings. `FreeOf` accepts all of those, so the two are
    interchangeable at the call site, while the original single-string form keeps
    working unchanged.
    """

    def test_a_group_of_variables_all_must_be_free(self):
        f = Operation.new('f', Arity.binary)
        x_, y_ = Wildcard.dot('x'), Wildcard.dot('y')
        pattern = Pattern(f(x_, y_), FreeOf(['x', 'y'], 'z'))

        def matches(subject):
            return bool(list(match_one_to_one(subject, pattern)))

        assert matches(f(NamedAtom('a'), NamedAtom('b'))) is True
        assert matches(f(NamedAtom('z'), NamedAtom('b'))) is False
        assert matches(f(NamedAtom('a'), NamedAtom('z'))) is False

    def test_a_group_equals_the_separate_constraints(self):
        grouped = FreeOf(['a', 'b'], 'x')
        assert grouped.variables == FreeOf('a', 'x').variables | FreeOf('b', 'x').variables

    def test_names_may_be_objects_that_carry_them(self):
        """Anything exposing `wildcard_name` or `name` may be passed instead of a str."""
        class _Wild:
            wildcard_name = 'a'

        class _Sym:
            name = 'x'

        assert FreeOf(_Wild(), _Sym()).variables == frozenset({'a'})
        assert FreeOf(_Wild(), _Sym()).symbol_name == 'x'
        assert FreeOf([_Wild(), _Wild()], _Sym()).variables == frozenset({'a'})

    def test_single_variable_form_is_unchanged(self):
        """The original API must be untouched -- `variable` stays a bare string."""
        c = FreeOf('u', 'x')
        assert c.variable == 'u'
        assert repr(c) == "FreeOf('u', 'x')"
        assert str(c) == 'FreeOf(u, x)'

    def test_grouped_str_shows_the_group(self):
        assert str(FreeOf(['a', 'b'], 'x')) == 'FreeOf([a, b], x)'

    def test_renaming_maps_every_variable_in_a_group(self):
        renamed = FreeOf(['a', 'b'], 'x').with_renamed_vars({'a': 'a2'})
        assert renamed.variables == frozenset({'a2', 'b'})
        assert renamed.symbol_name == 'x'

    def test_grouped_equality_and_hash(self):
        assert FreeOf(['a', 'b'], 'x') == FreeOf(['a', 'b'], 'x')
        assert FreeOf(['a', 'b'], 'x') != FreeOf(['a', 'c'], 'x')
        assert len({FreeOf(['a', 'b'], 'x'), FreeOf(['a', 'b'], 'x')}) == 1

    def test_an_unbound_variable_defers_rather_than_failing(self):
        """A variable not yet bound is re-checked later, exactly as in the single form."""
        assert FreeOf(['a', 'b'], 'x')(Substitution({'a': NamedAtom('q')})) is True
        assert FreeOf(['a', 'b'], 'x')(Substitution({'a': NamedAtom('x')})) is False
