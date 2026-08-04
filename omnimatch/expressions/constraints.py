# -*- coding: utf-8 -*-
"""Contains several pattern constraint classes.

A pattern constraint is used to further filter which subjects a pattern matches.

The most common use would be the :class:`CustomConstraint`, which wraps a lambda or function to act as a constraint:

>>> a_symbol_constraint = CustomConstraint(lambda x: x.name.startswith('a'))
>>> pattern = Pattern(x_, a_symbol_constraint)
>>> is_match(NamedAtom('a1'), pattern)
True
>>> is_match(NamedAtom('b1'), pattern)
False

There is also the :class:`EqualVariablesConstraint` which will try to unify the substitutions of the variables and only
match if it succeeds:

>>> equal_constraint = EqualVariablesConstraint('x', 'y')
>>> pattern = Pattern(f(x_, y_), equal_constraint)
>>> is_match(f(a, a), pattern)
True
>>> is_match(f(a, b), pattern)
False

You can also create a subclass of the :class:`Constraint` class to create your own custom constraint type.
"""
import inspect
from collections import OrderedDict
from typing import Callable, FrozenSet, Dict, Iterable, Tuple, Union
from functools import cached_property

from .._typed import TypedModel

from . import substitution
from ..utils import get_short_lambda_source


__all__ = ['Constraint', 'EqualVariablesConstraint', 'CustomConstraint', 'FreeOf']

_CO_VARARGS = 0x04       # CO_VARARGS  (*args)
_CO_VARKEYWORDS = 0x08   # CO_VARKEYWORDS (**kwargs)


def _constraint_parameter_names(constraint):
    """Ordered parameter names of a constraint callback.

    Fast path: read them straight off ``constraint.__code__`` (plain
    functions/lambdas), which avoids the very expensive ``inspect.signature``
    machinery — this is called hundreds of thousands of times while building a
    large matcher. Falls back to ``inspect.signature`` for exotic callables
    (functools.partial, callable objects, builtins). Raises the same ``ValueError``
    as before for positional-only / ``*args`` / ``**kwargs`` parameters.
    """
    code = getattr(constraint, '__code__', None)
    if (code is not None
            and not (code.co_flags & (_CO_VARARGS | _CO_VARKEYWORDS))
            and not getattr(code, 'co_posonlyargcount', 0)
            and not hasattr(constraint, '__wrapped__')):
        # Plain function/lambda with only positional-or-keyword / keyword-only
        # params and no functools.wraps redirection: its code object's names ARE
        # the signature, so read them directly (avoids inspect.signature).
        n = code.co_argcount + code.co_kwonlyargcount
        return code.co_varnames[:n]
    # Wrappers (functools.wraps), partials, callable objects, or functions with
    # *args/**kwargs: defer to inspect.signature (it follows __wrapped__ and
    # raises for genuinely disallowed parameter kinds).
    names = []
    for param in inspect.signature(constraint).parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            names.append(param.name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError("Constraint cannot have variable keyword arguments ({})".format(param.name))
        else:
            raise ValueError(
                "Constraint cannot have positional-only or variable positional arguments ({})".format(param.name)
            )
    return names


class Constraint(TypedModel):  # pylint: disable=too-few-public-methods
    """Base for pattern constraints.

    A constraint is essentially a callback, that receives the match :class:`Substitution` and returns a :class:`bool`
    indicating whether the match is valid.

    You have to override all the abstract methods if you wish to create your own subclass.
    """
    def __call__(self, match: substitution.Substitution) -> bool:  # pylint: disable=missing-raises-doc
        """Return True, iff the constraint is fulfilled by the substitution.

        Override this in your subclass to define the actual constraint behavior.

        Args:
            match:
                The (current) match substitution. Note that the matching is done from left to right, so not all
                variables may have a value yet. You need to override `variables` so that the constraint gets
                called once all the variables it depends on have a value assigned to them.

        Returns:
            True, iff the constraint is fulfilled by the substitution.
        """
        raise NotImplementedError

    def __eq__(self, other):
        """Constraints need to be equatable."""
        raise NotImplementedError

    def __hash__(self):
        """Constraints need to be hashable."""
        raise NotImplementedError

    @property
    def variables(self) -> FrozenSet[str]:
        """The names of the variables the constraint depends upon.

        Used by matchers to decide when a constraint can be evaluated (which is when all
        the dependency variables have been assigned a value). If the set is empty, the constraint will
        only be evaluated once the whole match is complete.
        """
        return frozenset()

    def with_renamed_vars(self, renaming: Dict[str, str]) -> 'Constraint':  # pylint: disable=missing-raises-doc
        """Return a *copy* of the constraint with renamed variables.
        This is called when the variables in the expression are renamed and hence the ones in the constraint have to be
        renamed as well. A later invocation of :meth:`__call__` will have the new variable names.
        You will have to implement this if your constraint needs to use the variables of the match substitution.
        Note that this can be called multiple times and you might have to account for that.
        Also, this should not modify the original constraint but rather return a copy.
        Args:
            renaming:
                A dictionary mapping old names to new names.
        Returns:
            A copy of the constraint with renamed variables.
        """
        raise NotImplementedError


class EqualVariablesConstraint(Constraint):  # pylint: disable=too-few-public-methods
    """A constraint that ensure multiple variables are equal.

    The constraint tries to unify the substitutions for the variables and is fulfilled iff that succeeds.
    """

    def __init__(self, *variables: str, **kwargs) -> None:
        """
        Args:
            *variables: The names of the variables to check for equality.
        """
        super().__init__(**kwargs)
        self._variables = frozenset(variables)

    @property
    def variables(self):
        return self._variables

    def __call__(self, match: substitution.Substitution) -> bool:
        subst = substitution.Substitution()
        for name in self._variables:
            try:
                subst.try_add_variable('_', match[name])
            except ValueError:
                return False
        return True

    def __str__(self):
        return '({!s})'.format(' == '.join(sorted(self._variables)))

    def __repr__(self):
        return 'EqualVariablesConstraint({!s})'.format(' == '.join(sorted(self._variables)))

    def __eq__(self, other):
        return isinstance(other, EqualVariablesConstraint) and self._variables == other._variables

    def __hash__(self):
        return hash(self._variables)

    def with_renamed_vars(self, renaming):
        return EqualVariablesConstraint(*(renaming.get(v, v) for v in self.variables))


class CustomConstraint(Constraint):  # pylint: disable=too-few-public-methods
    """Wrapper for lambdas of functions as constraints.

    The parameter names have to be the same as the the variable names in the expression:

    >>> constraint = CustomConstraint(lambda x, y: x.name < y.name)
    >>> pattern = Pattern(f(x_, y_), constraint)
    >>> is_match(f(a, b), pattern)
    True
    >>> is_match(f(b, a), pattern)
    False

    The ordering of the parameters is not important. You only need to have the parameters needed for the constraint,
    not all variables occurring in the pattern.

    Note, that the matching happens from left left to right, so not all variables may have been assigned a value when
    constraint is called. For constraints over multiple variables you should attach the constraint to the last
    variable occurring in the pattern or a surrounding operation.
    """
    constraint: Callable[..., bool]
    # Always assigned in __init__; a static default avoids Pydantic re-introspecting
    # the default_factory (inspect.signature(OrderedDict)) on every construction.

    def __init__(self, constraint: Callable[..., bool], **kwargs) -> None:
        """
        Args:
            constraint:
                The constraint callback.

        Raises:
            ValueError:
                If the callback has positional-only or variable parameters (\\*args and \\*\\*kwargs).
        """
        super().__init__(constraint=constraint, **kwargs)
        self._variables = OrderedDict((name, name) for name in _constraint_parameter_names(constraint))

    @cached_property
    def variables(self):
        return frozenset(self._variables.values())

    def __call__(self, match: substitution.Substitution) -> bool:
        try:
            args = dict((name, match[var_name]) for name, var_name in self._variables.items())
        except KeyError:
            return True  # Not all variables bound yet; constraint will be re-checked later
        return self.constraint(**args)

    def _get_name(self):
        try:
            return get_short_lambda_source(self.constraint) or self.constraint.__name__
        except Exception:
            return 'UNKNOWN'

    def __str__(self):
        return '({!s})'.format(self._get_name())

    def __repr__(self):
        return 'CustomConstraint({!s})'.format(self._get_name())

    def __eq__(self, other):
        return (
            isinstance(other, CustomConstraint) and self.constraint == other.constraint and
            self._variables == other._variables
        )

    def __hash__(self):
        return hash(self.constraint)

    def with_renamed_vars(self, renaming):
        cc = CustomConstraint(self.constraint)
        for param_name in cc._variables.keys():
            old_name = self._variables[param_name]
            cc._variables[param_name] = renaming.get(old_name, old_name)
        return cc


def _constraint_name(obj) -> str:
    """Name of `obj` for constraint purposes: a plain string, or a named object.

    Accepts a bare name, or anything carrying one -- a pattern wildcard exposes
    ``wildcard_name``, a symbol exposes ``name``. This is what lets `FreeOf` be called
    with the same argument shapes as a higher-layer predicate that works in objects
    rather than strings, without this module knowing anything about those layers.
    """
    for attr in ('wildcard_name', 'name'):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(obj, str):
        return obj
    return str(obj)


def _constraint_names(obj) -> Tuple[str, ...]:
    """Normalise one name, or an iterable of them, to a tuple of names."""
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return (_constraint_name(obj),)
    return tuple(_constraint_name(o) for o in obj)


class FreeOf(Constraint):
    """Constraint that checks expressions are free of a given symbol.

    ``FreeOf(variables, symbol)`` succeeds when NONE of the expressions bound to
    `variables` contains an atom named `symbol` anywhere in its tree.

    The argument shapes mirror the higher-layer ``FreeQ`` predicate, so the two are
    interchangeable at the call site:

    * one variable or MANY -- ``FreeOf('a', 'x')`` and ``FreeOf(['a', 'b'], 'x')``;
      a group succeeds only if EVERY member is free of the symbol, so the second is
      exactly ``FreeOf('a', 'x')`` and ``FreeOf('b', 'x')`` together;
    * names as bare strings, or as the objects carrying them -- a pattern wildcard
      (``wildcard_name``) or a symbol (``name``) may be passed directly.

    Optimized over a CustomConstraint because:
    - Uses a dedicated iterative traversal with early exit (no generator overhead)
    - No lambda/closure indirection
    - Proper __repr__ for debugging and code generation

    Example::

        >>> from omnimatch import *
        >>> x_ = Wildcard.dot('x')
        >>> y_ = Wildcard.dot('y')
        >>> f = Operation.new('f', Arity.binary)
        >>> # y must not contain symbol 'x'
        >>> pattern = Pattern(f(x_, y_), FreeOf('y', 'x'))
        >>> is_match(f(NamedAtom('x'), NamedAtom('a')), pattern)
        True
        >>> is_match(f(NamedAtom('x'), NamedAtom('x')), pattern)
        False

        A group of variables, all of which must be free of the symbol::

        >>> both = Pattern(f(x_, y_), FreeOf(['x', 'y'], 'z'))
        >>> is_match(f(NamedAtom('a'), NamedAtom('b')), both)
        True
        >>> is_match(f(NamedAtom('a'), NamedAtom('z')), both)
        False
    """

    variable: Union[str, Tuple[str, ...]]
    symbol_name: str

    def __init__(self, variables, symbol_name, **kwargs) -> None:
        """
        Args:
            variables:
                The pattern variable whose bound expression will be checked, or an
                iterable of them (all must be free of the symbol). Each may be a bare
                name or an object carrying one (``wildcard_name`` / ``name``).
            symbol_name:
                The symbol that must NOT appear anywhere in those expressions, as a
                name or as an object carrying one.
        """
        names = _constraint_names(variables)
        # A single variable is stored unwrapped so that `.variable`, `repr` and equality
        # are unchanged for existing callers.
        stored = names[0] if len(names) == 1 else names
        super().__init__(variable=stored,
                         symbol_name=_constraint_name(symbol_name), **kwargs)

    @property
    def variable_names(self) -> Tuple[str, ...]:
        """The checked variables, always as a tuple -- one entry or many."""
        return self.variable if isinstance(self.variable, tuple) else (self.variable,)

    @cached_property
    def variables(self) -> FrozenSet[str]:
        return frozenset(self.variable_names)

    def __call__(self, match: substitution.Substitution) -> bool:
        for name in self.variable_names:
            try:
                expr = match[name]
            except KeyError:
                continue  # Variable not yet bound; will be re-checked later
            if not self._is_free(expr):
                return False
        return True

    def _is_free(self, expr) -> bool:
        """Check that expr does not contain an atom with the given name.

        Uses an explicit stack for tree traversal (no recursion limit issues,
        no generator overhead) and exits immediately upon finding the symbol.
        Checks both NamedAtom (by name) and SymbolWrapper (by str(value)).
        """
        from .expressions import NamedAtom, SymbolWrapper, Operation

        # Handle sequence variable values (tuples, lists, Multisets)
        if isinstance(expr, (tuple, list)):
            stack = list(expr)
        else:
            stack = [expr]

        while stack:
            node = stack.pop()
            if isinstance(node, NamedAtom):
                if node.name == self.symbol_name:
                    return False
            elif isinstance(node, SymbolWrapper):
                if node.name == self.symbol_name:
                    return False
            elif isinstance(node, Operation):
                stack.extend(node.operands)
            elif isinstance(node, (tuple, list)):
                stack.extend(node)
        return True

    def __str__(self):
        if isinstance(self.variable, tuple):
            return 'FreeOf([{}], {})'.format(', '.join(self.variable), self.symbol_name)
        return 'FreeOf({}, {})'.format(self.variable, self.symbol_name)

    def __repr__(self):
        return 'FreeOf({!r}, {!r})'.format(self.variable, self.symbol_name)

    def __eq__(self, other):
        return (
            isinstance(other, FreeOf) and
            self.variable == other.variable and
            self.symbol_name == other.symbol_name
        )

    def __hash__(self):
        return hash(('FreeOf', self.variable, self.symbol_name))

    def with_renamed_vars(self, renaming: Dict[str, str]) -> 'FreeOf':
        renamed = tuple(renaming.get(n, n) for n in self.variable_names)
        return FreeOf(renamed, self.symbol_name)
