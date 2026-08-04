# OmniMatch

> **⚠️ Experimental** — this package is under active development; APIs may
> change without notice. Version 0.0.2 is a pre-alpha snapshot.

OmniMatch is a library for pattern matching on symbolic expressions in Python.
It was forked by Francesco Bonazzi from
[MatchPy](https://github.com/HPAC/matchpy) by Manuel Krebber and has since
diverged (renamed API, typed expression models, SymPy integration layers).

**Work in progress**

## Installation

```
pip install omnimatch
```

Or, from a checkout, in editable mode: `pip install -e .`

## Overview

This package implements [pattern matching](https://en.wikipedia.org/wiki/Pattern_matching)
in Python. Pattern matching is a powerful tool for symbolic computations,
operating on symbolic expressions. Given a pattern and an expression (which is
usually called *subject*), the goal of pattern matching is to find a
substitution for all the variables in the pattern such that the pattern becomes
the subject. As an example, consider the pattern `f(x)`, where `f` is a
function and `x` is a variable, and the subject `f(a)`, where `a` is a constant
symbol. Then the substitution that replaces `x` with `a` is a match. OmniMatch
supports associative and/or commutative function symbols, as well as sequence
variables, similar to pattern matching in
[Mathematica](https://reference.wolfram.com/language/guide/Patterns.html).

A detailed example of how to use OmniMatch can be found in `docs/example.md`.

OmniMatch supports both one-to-one and many-to-one pattern matching. The latter
makes use of similarities between patterns to efficiently find matches for
multiple patterns at the same time.

### Expressions

Expressions are tree-like data structures, consisting of operations (functions,
internal nodes) and symbols (constants, leaves):

```python
>>> from omnimatch import Operation, NamedAtom, Arity
>>> f = Operation.new('f', Arity.binary)
>>> a = NamedAtom('a')
>>> print(f(a, a))
f(a, a)

```

Patterns are expressions which may contain wildcards (variables):

```python
>>> from omnimatch import Pattern, Wildcard
>>> x = Wildcard.dot('x')
>>> print(Pattern(f(a, x)))
f(a, x_)

```

In the previous example, `x` is the name of the variable. However, it is also
possible to use wildcards without names:

```python
>>> w = Wildcard.dot()
>>> print(Pattern(f(w, w)))
f(_, _)

```

It is also possible to assign variable names to entire subexpressions:

```python
>>> print(Pattern(f(w, a, variable_name='y')))
y: f(_, a)

```

### Pattern Matching

Given a pattern and an expression (which is usually called subject), the idea
of pattern matching is to find a substitution that maps wildcards to
expressions such that the pattern becomes the subject. In OmniMatch, a
substitution is a dict that maps variable names to expressions.

```python
>>> from omnimatch import match
>>> y = Wildcard.dot('y')
>>> b = NamedAtom('b')
>>> subject = f(a, b)
>>> pattern = Pattern(f(x, y))
>>> substitution = next(match(subject, pattern))
>>> print(substitution)
{x ↦ a, y ↦ b}

```

Applying the substitution to the pattern results in the original expression.

```python
>>> from omnimatch import substitute
>>> print(substitute(pattern, substitution))
f(a, b)

```

### Sequence Wildcards

Sequence wildcards are wildcards that can match a sequence of expressions
instead of just a single expression:

```python
>>> z = Wildcard.plus('z')
>>> pattern = Pattern(f(z))
>>> subject = f(a, b)
>>> substitution = next(match(subject, pattern))
>>> print(substitution)
{z ↦ (a, b)}

```

### Associativity and Commutativity

OmniMatch natively supports associative and/or commutative operations. Nested
associative operators are automatically flattened, the operands in commutative
operations are sorted:

```python
>>> g = Operation.new('g', Arity.variadic, associative=True, commutative=True)
>>> print(g(a, g(b, a)))
g(a, a, b)

```

Associativity and commutativity is also considered for pattern matching:

```python
>>> pattern = Pattern(g(b, x))
>>> subject = g(a, a, b)
>>> print(next(match(subject, pattern)))
{x ↦ g(a, a)}
>>> h = Operation.new('h', Arity.variadic)
>>> pattern = Pattern(h(b, x))
>>> subject = h(a, a, b)
>>> list(match(subject, pattern))
[]

```

### Many-to-One Matching

When a fixed set of patterns is matched repeatedly against different subjects,
matching can be sped up significantly by using many-to-one matching. The idea
of many-to-one matching is to construct an automaton-like data structure
(similar to a decision tree) that exploits similarities between patterns. In
OmniMatch this is the `ManyToOneMatcher`, which supports associative and/or
commutative matching with sequence variables. (An older syntactic-only
`DiscriminationNet` has been removed from this fork.)

```python
>>> pattern1 = Pattern(f(a, x))
>>> pattern2 = Pattern(f(y, b))
>>> matcher = ManyToOneMatcher(pattern1, pattern2)
>>> subject = f(a, b)
>>> matches = matcher.match(subject)
>>> for matched_pattern, substitution in sorted(map(lambda m: (str(m[0]), str(m[1])), matches)):
...     print('{} matched with {}'.format(matched_pattern, substitution))
f(a, x_) matched with {x ↦ b}
f(y_, b) matched with {y ↦ a}

```

## Roadmap

Besides the existing features, we plan on adding the following to OmniMatch:

- Support for Mathematica's `Alternatives`: For example `f(a | b)` would match
  either `f(a)` or `f(b)`.
- Support for Mathematica's `Repeated`: For example `f(a..)` would match
  `f(a)`, `f(a, a)`, `f(a, a, a)`, etc.
- Support pattern sequences (`PatternSequence` in Mathematica). These are
  mainly useful in combination with `Alternatives` or `Repeated`, e.g.
  `f(a | (b, c))` would match either `f(a)` or `f(b, c)`. `f((a a)..)` would
  match any `f` with an even number of `a` arguments.
- All these additional pattern features need to be supported in the
  `ManyToOneMatcher` as well.
- Better integration with existing types such as `dict`.
- Code generation for both one-to-one and many-to-one matching. There is
  already an experimental implementation, but it still has some dependencies on
  OmniMatch which can probably be removed.
- Improving the documentation with more examples.
- Better test coverage with more randomized tests.
- Implementation of the matching algorithms in a lower-level language, for
  example C, both for performance and to make OmniMatch's functionality
  available in other languages.

## Contributing

If you have some issue or want to contribute, please feel free to open an issue
or create a pull request. Help is always appreciated!

The Makefile has several tasks to help development:

- To install all needed packages, you can use `make init`.
- To run the tests you can use `make test`. The tests use
  [pytest](https://docs.pytest.org/).
- To generate the documentation you can use `make docs`.
- To run the style checker ([pylint](https://www.pylint.org/)) you can use
  `make check`.

If you have any questions or need help with setting things up, please open an
issue and we will try the best to assist you.

## License

OmniMatch is licensed under the MIT License, the same license as MatchPy, of
which it is a fork. Two copyright notices apply: Copyright (c) 2016 Manuel
Krebber (the original [MatchPy](https://github.com/HPAC/matchpy)) and
Copyright (c) 2026 Francesco Bonazzi (the OmniMatch modifications and
additions). See the `LICENSE` file, which also carries MatchPy's third-party
notices.
