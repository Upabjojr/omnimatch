# -*- coding: utf-8 -*-
"""Expression classes for the OmniMatch pattern matching library.

This refactored version uses:
- TypedModel (omnimatch._typed) for all expression types
- OperationHead objects (instead of class-based Operation subclasses)
- Singledispatch `to_omnimatch_expression` / `from_omnimatch_expression` for conversions
- `Operation(head, *operands)` as raw constructor (normalize only, no one_identity)
- `head(*operands)` as factory (applies one_identity, may return non-Operation)
"""
import keyword
from enum import Enum
from functools import singledispatch, cached_property
from typing import Callable, Iterator, List, Optional, Set, Tuple, Type, Union, Any

from multiset import Multiset
from .._typed import TypedModel, field

__all__ = [
    'Expression', 'Arity', 'AtomExpr', 'NamedAtom', 'SymbolWrapper', 'Wildcard', 'Operation', 'Pattern',
    'OperationHead', 'to_omnimatch_expression', 'from_omnimatch_expression',
    'make_dot_variable', 'make_plus_variable', 'make_star_variable',
    'LIST_HEAD', 'TUPLE_HEAD', 'DICT_HEAD', 'DICT_PAIR_HEAD',
]

ExprPredicate = Callable[['Expression'], bool]
ExpressionsWithPos = Iterator[Tuple['Expression', Tuple[int, ...]]]


# ─── Base Expression ──────────────────────────────────────────────────────────

class Expression(TypedModel):
    """Base class for all OmniMatch expressions.

    All expressions have:
    - head: identifies the type (self for atoms, OperationHead for operations)
    - variable_name: optional name when used as a pattern variable
    """
    head: Any = None
    variable_name: Optional[str] = None

    def collect_symbols(self, symbols: Set[str]):
        """Collect all symbol names in this expression into the given set."""
        pass

    def with_renamed_vars(self, renaming: dict) -> 'Expression':
        """Return a copy with variable names renamed according to the mapping."""
        return self

    @property
    def is_constant(self) -> bool:
        """True if this expression contains no wildcards (is fully ground)."""
        for expr, _ in self.preorder_iter():
            if isinstance(expr, Wildcard):
                return False
            if expr.variable_name is not None:
                return False
        return True

    @property
    def is_syntactic(self) -> bool:
        """True if this expression can be matched purely by structural comparison.

        An expression is syntactic when it contains no sequence wildcards
        and no commutative/associative operations.
        """
        for expr, _ in self.preorder_iter():
            if isinstance(expr, Wildcard) and not expr.fixed_size:
                return False
            if isinstance(expr, Operation):
                if expr.head.commutative or expr.head.associative:
                    return False
        return True

    @property
    def symbols(self) -> 'Multiset':
        """Multiset of all symbol names in this expression (includes head names for operations)."""
        result = []
        for expr, _ in self.preorder_iter():
            if isinstance(expr, Operation):
                result.append(expr.head.name)
            elif isinstance(expr, (NamedAtom, SymbolWrapper)) and not isinstance(expr, Wildcard):
                result.append(expr.name)
        return Multiset(result)

    @property
    def variables(self) -> 'Multiset':
        """Multiset of all variable names in this expression."""
        result = []
        for expr, _ in self.preorder_iter():
            if expr.variable_name is not None:
                result.append(expr.variable_name)
        return Multiset(result)

    def preorder_iter(self, predicate=None) -> 'ExpressionsWithPos':
        """Iterate over all subexpressions in preorder, yielding (expression, position).

        Args:
            predicate: If given, only yield expressions where predicate(expr) is True.
        """
        yield from self._preorder_iter((), predicate)

    def _preorder_iter(self, position, predicate):
        if predicate is None or predicate(self):
            yield (self, position)
        if isinstance(self, Operation):
            for i, operand in enumerate(self.operands):
                yield from operand._preorder_iter(position + (i,), predicate)

    def __getitem__(self, position):
        """Access subexpression by position tuple or slice.

        position can be:
        - A tuple of ints: path to subexpression (e.g., (1, 0))
        - A slice with start/end position tuples: range of children
        """
        if isinstance(position, slice):
            return self._getitem_slice(position.start, position.stop)
        # Tuple position access
        if len(position) == 0:
            return self
        if not isinstance(self, Operation):
            raise IndexError("Invalid position")
        idx = position[0]
        if idx < 0 or idx >= len(self.operands):
            raise IndexError("Invalid position")
        return self.operands[idx][position[1:]]

    def _getitem_slice(self, start, stop):
        """Get a range of subexpressions."""
        if len(start) != len(stop):
            raise IndexError('Invalid slice: Start and stop must have the same length')
        if len(start) == 0:
            return [self]
        if not isinstance(self, Operation):
            raise IndexError('Invalid slice: Parent expression is not an operation')
        if len(start) == 1:
            s, e = start[0], stop[0]
            if s > e:
                raise IndexError('Invalid slice')
            e = min(e + 1, len(self.operands))
            return list(self.operands[s:e])
        # Recurse into the child at start[0] (must equal stop[0])
        if start[0] != stop[0]:
            raise IndexError('Invalid slice: Start and stop must have the same length')
        child_idx = start[0]
        if child_idx < 0 or child_idx >= len(self.operands):
            raise IndexError('Invalid position')
        return self.operands[child_idx]._getitem_slice(start[1:], stop[1:])

    def __contains__(self, expression) -> bool:
        """Check if expression is contained anywhere in this expression tree."""
        for expr, _ in self.preorder_iter():
            if expr == expression:
                return True
        return False


# ─── Arity enum ───────────────────────────────────────────────────────────────

class Arity(Enum):
    """Defines how many operands an operation accepts."""
    nullary = (0, True)
    unary = (1, True)
    binary = (2, True)
    ternary = (3, True)
    variadic = (0, False)   # 0 or more operands

    def __init__(self, min_count, fixed_size):
        self.min_count = min_count
        self.fixed_size = fixed_size

    def __getitem__(self, index):
        """Support arity[0] → min_count, arity[1] → fixed_size for backward compat."""
        if index == 0:
            return self.min_count
        if index == 1:
            return self.fixed_size
        raise IndexError(index)


# ─── OperationHead ────────────────────────────────────────────────────────────

class OperationHead(TypedModel):
    """Metadata describing an operation type.

    An OperationHead defines the structural properties of an operation:
    name, arity, whether it's commutative, associative, or one_identity.

    OperationHead objects are callable — calling them is the factory that
    applies one_identity logic (may return a non-Operation for single operands).
    Use `Operation(head, *operands)` directly for the raw constructor that
    only normalizes (flatten associative, sort commutative) without one_identity.
    """
    name: str
    arity: Arity = Arity.variadic
    commutative: bool = False
    associative: bool = False
    one_identity: bool = False
    infix: bool = False

    def __call__(self, *operands, variable_name=None):
        """Factory: create an expression, applying one_identity if appropriate.

        If one_identity is True and there's exactly one operand after normalization,
        returns that operand directly (possibly with variable_name attached).
        Otherwise creates an Operation.
        """
        result = _check_one_identity(self, list(operands))
        if result is not None:
            # one_identity collapsed to single operand
            if variable_name and hasattr(result, 'variable_name'):
                result = result.with_renamed_vars({result.variable_name: variable_name} if result.variable_name else {})
                if hasattr(result, 'variable_name'):
                    object.__setattr__(result, 'variable_name', variable_name)
            return result
        return Operation(self, *operands, variable_name=variable_name)

    def __hash__(self):
        # Cached: rebuilt-tuple hash, computed ~1.9M times during Rubi matcher build.
        # OperationHead is immutable after construction.
        h = self.__dict__.get('_cached_hash')
        if h is None:
            h = hash((self.name, self.arity, self.commutative, self.associative, self.one_identity))
            self.__dict__['_cached_hash'] = h
        return h

    def __eq__(self, other):
        if not isinstance(other, OperationHead):
            return NotImplemented
        return (self.name == other.name and self.arity == other.arity and
                self.commutative == other.commutative and self.associative == other.associative and
                self.one_identity == other.one_identity)

    def __repr__(self):
        parts = [f"name={self.name!r}"]
        if self.arity != Arity.variadic:
            parts.append(f"arity={self.arity.name}")
        if self.commutative:
            parts.append("commutative=True")
        if self.associative:
            parts.append("associative=True")
        if self.one_identity:
            parts.append("one_identity=True")
        return f"OperationHead({', '.join(parts)})"


class WildcardOperationHead(OperationHead):
    """An operation head that matches ANY operation head, binding the matched one.

    OmniMatch normally indexes an operation by its concrete head, so a pattern whose
    *head itself* is a wildcard could not be expressed. Some rule sets need exactly
    that -- e.g. Rubi's ``F_[v_]``, "any function ``F`` applied to ``v``".

    An ``Operation`` whose head is a ``WildcardOperationHead`` matches an operation
    with ANY head (its operands are matched normally against the pattern's
    operands, so ordinary argument wildcards bind as usual). The subject's head is
    bound to ``variable_name`` in the substitution, wrapped so it is a regular
    OmniMatch expression.

    Matching support lives in :mod:`omnimatch.matching.many_to_one`: such a pattern
    is keyed under the ``_HEAD_ANY_OP`` transition key (rather than under its own
    head), and every operation subject also offers that key.
    """

    variable_name: Optional[str] = None

    def __hash__(self):
        return hash(('WildcardOperationHead', self.name, self.arity,
                     self.commutative, self.associative, self.one_identity,
                     self.variable_name))

    def __eq__(self, other):
        # Deliberately NOT equal to a plain OperationHead with the same name:
        # a wildcard head is a different kind of thing.
        if type(other) is not WildcardOperationHead:
            return NotImplemented if isinstance(other, OperationHead) else False
        return (self.name == other.name and self.arity == other.arity and
                self.commutative == other.commutative and
                self.associative == other.associative and
                self.one_identity == other.one_identity and
                self.variable_name == other.variable_name)

    def __repr__(self):
        return f"WildcardOperationHead(variable_name={self.variable_name!r})"


# ─── Helper functions for Operation construction ──────────────────────────────

def _check_one_identity(head, operands):
    """Check if one_identity should collapse the operation to a single operand.

    Returns the single operand if collapsing should happen, None otherwise.
    Called by OperationHead.__call__ (the factory), NOT by Operation.__init__ (the raw constructor).
    """
    if not head.one_identity:
        return None
    # First normalize (flatten associative)
    if head.associative:
        flat = []
        for op in operands:
            if isinstance(op, Operation) and op.head == head and not op.variable_name:
                flat.extend(op.operands)
            else:
                flat.append(op)
        operands = flat
    if len(operands) == 1:
        return operands[0]
    return None


def _normalize_operands(head, operands):
    """Normalize operands: flatten associative, sort commutative.

    Modifies operands list in place. Called by Operation.__init__.
    """
    # Flatten associative
    if head.associative:
        flat = []
        for op in operands:
            if isinstance(op, Operation) and op.head == head and not op.variable_name:
                flat.extend(op.operands)
            else:
                flat.append(op)
        operands.clear()
        operands.extend(flat)
    # Sort commutative
    if head.commutative:
        operands.sort()


def _check_arity(head, operands):
    """Validate operand count against arity constraints.

    Raises ValueError if the arity constraint is violated.
    For non-fixed arity (variadic), no check is needed.
    For fixed arity, wildcards with fixed_size=False (sequence wildcards) provide
    flexibility: they count as their min_count contribution.
    """
    if not head.arity.fixed_size:
        return  # variadic — anything goes
    required = head.arity.min_count
    # Count how many positions the operands occupy
    min_ops = 0
    has_unbounded = False
    for op in operands:
        if isinstance(op, Wildcard) and not op.fixed_size:
            min_ops += op.min_count
            has_unbounded = True
        else:
            min_ops += 1
    if has_unbounded:
        # Sequence wildcards present: minimum must not exceed required
        if min_ops > required:
            raise ValueError(
                f"Operation {head.name!r} with arity {head.arity.name} (requires {required} operands) "
                f"got too many operands: minimum is {min_ops}")
    else:
        # All fixed: must be exact
        if len(operands) != required:
            raise ValueError(
                f"Operation {head.name!r} with arity {head.arity.name} requires exactly {required} operands, "
                f"got {len(operands)}")


# ─── Operation ────────────────────────────────────────────────────────────────

class Operation(Expression):
    """A compound expression consisting of a head (OperationHead) and operands.

    The raw constructor `Operation(head, *operands)` applies normalization
    (flatten associative, sort commutative) but does NOT apply one_identity.
    Use `head(*operands)` (OperationHead.__call__) for the full factory.
    """
    operands: List[Expression] = field(default_factory=list)

    def __init__(self, head, *operands, variable_name=None, **kwargs) -> None:
        # Auto-convert non-Expression operands to expressions
        op_list = []
        for op in operands:
            if isinstance(op, Expression):
                op_list.append(op)
            elif isinstance(op, str):
                op_list.append(NamedAtom(op))
            else:
                op_list.append(SymbolWrapper(op))
        _normalize_operands(head, op_list)
        _check_arity(head, op_list)
        super().__init__(head=head, operands=op_list, variable_name=variable_name, **kwargs)

    @classmethod
    def new(cls, name: str, arity: Arity = Arity.variadic, **kwargs) -> 'OperationHead':
        """Create a new OperationHead (backward-compatible factory).

        Returns an OperationHead which is callable to create Operations.

        Raises:
            ValueError: If name is a Python keyword or not a valid identifier.
            TypeError: If one_identity is used with non-variadic arity.
            TypeError: If infix is used with unary arity.
        """
        # Name validation: reject non-identifiers and statement keywords unless infix=True
        _OPERATOR_KEYWORDS = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'}
        if not kwargs.get('infix') and (not name.isidentifier() or (keyword.iskeyword(name) and name not in _OPERATOR_KEYWORDS)):
            raise ValueError(f"Invalid operation name: {name!r}. Must be a valid non-keyword identifier, or use infix=True.")
        if kwargs.get('one_identity') and arity != Arity.variadic:
            raise TypeError("one_identity=True is only allowed for variadic operations.")
        if kwargs.get('infix') and arity == Arity.unary:
            raise TypeError("infix=True is not allowed for unary operations.")
        return OperationHead(name=name, arity=arity, **kwargs)

    def collect_symbols(self, symbols):
        for op in self.operands:
            op.collect_symbols(symbols)

    def with_renamed_vars(self, renaming) -> 'Operation':
        new_operands = [op.with_renamed_vars(renaming) for op in self.operands]
        new_vname = renaming.get(self.variable_name, self.variable_name)
        return Operation(self.head, *new_operands, variable_name=new_vname)

    def __copy__(self) -> 'Operation':
        return Operation(self.head, *self.operands, variable_name=self.variable_name)

    def __str__(self):
        value = '{!s}({!s})'.format(self.head.name, ', '.join(str(o) for o in self.operands))
        if self.variable_name:
            value = '{}: {}'.format(self.variable_name, value)
        return value

    def __repr__(self):
        operand_str = ', '.join(map(repr, self.operands))
        if self.variable_name:
            return 'Operation({!s}, {!s}, variable_name={})'.format(self.head.name, operand_str, self.variable_name)
        return 'Operation({!s}, {!s})'.format(self.head.name, operand_str)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if not isinstance(other, Operation):
            return False  # Operations sort after all atoms (NamedAtom, Wildcard)
        if self.head != other.head:
            return self.head.name < other.head.name
        if len(self.operands) != len(other.operands):
            return len(self.operands) < len(other.operands)
        for left, right in zip(self.operands, other.operands):
            if left < right:
                return True
            elif right < left:
                return False
        return (self.variable_name or '') < (other.variable_name or '')

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return (
            self.head == other.head and
            len(self.operands) == len(other.operands) and
            all(x == y for x, y in zip(self.operands, other.operands)) and
            self.variable_name == other.variable_name
        )

    def __hash__(self):
        # CACHED: this hash is recursive over the whole subtree (operands tuple), and
        # matcher construction / transition-table lookups hash the same node millions of
        # times. Safe to cache because expressions are immutable after construction --
        # the only post-construction mutation in the codebase is on a NamedAtom copy
        # (many_to_one._get_label_and_head), never on an Operation.
        h = self.__dict__.get('_cached_hash')
        if h is None:
            h = hash((Operation, self.head, tuple(self.operands), self.variable_name))
            self.__dict__['_cached_hash'] = h
        return h


# ─── AtomExpr base ────────────────────────────────────────────────────────────────

class AtomExpr(Expression):
    """Base for all atomic (leaf) expressions."""
    __iter__ = None


# ─── NamedAtom ───────────────────────────────────────────────────────────────────

class NamedAtom(AtomExpr):
    """An atomic constant expression term, uniquely identified by its name."""
    name: str = ""

    def __init__(self, name: str, variable_name=None, **kwargs) -> None:
        super().__init__(variable_name=variable_name, name=name, **kwargs)
        self.head = self

    def __str__(self):
        if self.variable_name:
            return '{}: {}'.format(self.name, self.variable_name)
        return self.name

    def __repr__(self):
        if self.variable_name:
            return '{!s}({!r}, variable_name={})'.format(type(self).__name__, self.name, self.variable_name)
        return '{!s}({!r})'.format(type(self).__name__, self.name)

    def collect_symbols(self, symbols):
        symbols.add(self.name)

    def with_renamed_vars(self, renaming) -> 'NamedAtom':
        return type(self)(self.name, variable_name=renaming.get(self.variable_name, self.variable_name))

    def __copy__(self) -> 'NamedAtom':
        return type(self)(self.name, variable_name=self.variable_name)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, NamedAtom):
            if self.name == other.name:
                return (self.variable_name or '') < (other.variable_name or '')
            return self.name < other.name
        if isinstance(other, SymbolWrapper):
            return self.name < other.name
        if isinstance(other, Wildcard):
            return True  # Symbols sort before Wildcards
        if isinstance(other, Operation):
            return True  # Atoms sort before Operations
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, NamedAtom):
            return self.name == other.name and self.variable_name == other.variable_name
        if isinstance(other, SymbolWrapper):
            return self.name == other.name and self.variable_name == other.variable_name
        return NotImplemented

    def __hash__(self):
        return hash(('_named_atom_', self.name, self.variable_name))


# ─── SymbolWrapper ────────────────────────────────────────────────────────────

class SymbolWrapper(AtomExpr):
    """An atomic expression wrapping an arbitrary Python object.

    Unlike NamedAtom (which stores a string name), SymbolWrapper stores the
    original object directly. This enables lossless roundtripping when
    converting expressions from external libraries (e.g. SymPy integers,
    constants like I, pi) without string-based encoding/decoding.

    SymbolWrapper is cross-compatible with NamedAtom for matching: a NamedAtom('x')
    pattern will match a SymbolWrapper whose name property returns 'x'.
    This allows patterns to be written with plain NamedAtom('2') and still match
    against SymbolWrapper(Integer(2)).
    """
    value: object = None

    def __init__(self, value, variable_name=None, **kwargs) -> None:
        super().__init__(value=value, variable_name=variable_name, **kwargs)
        self.head = self

    @cached_property
    def name(self) -> str:
        """String representation for display and cross-type matching with NamedAtom.

        Fast path: for objects with a string ``.name`` (sympy Symbol/WildSymbol —
        the overwhelming majority of wrapped values), that attribute IS the str()
        form, and ``str()`` on a sympy object invokes the full printing machinery
        (~23x slower). Profiling showed ~14% of ManyToOneMatcher build time was
        sympy's printer called from ``SymbolWrapper.__hash__`` via this property.

        NOT taken for sympy ``Dummy`` (``.name`` is ``'d'`` but ``str()`` is ``'_d'`` —
        the underscore matters for name-based matching); detected duck-typed via
        ``is_Dummy`` so this layer stays sympy-free.
        """
        n = getattr(self.value, 'name', None)
        if type(n) is str and getattr(self.value, 'is_Dummy', False) is not True:
            return n
        return str(self.value)

    def __str__(self):
        if self.variable_name:
            return '{}: {}'.format(self.variable_name, self.value)
        return self.name

    def __repr__(self):
        if self.variable_name:
            return 'SymbolWrapper({!r}, variable_name={!r})'.format(self.value, self.variable_name)
        return 'SymbolWrapper({!r})'.format(self.value)

    def collect_symbols(self, symbols):
        symbols.add(self.name)

    def with_renamed_vars(self, renaming) -> 'SymbolWrapper':
        return SymbolWrapper(self.value, variable_name=renaming.get(self.variable_name, self.variable_name))

    def __copy__(self) -> 'SymbolWrapper':
        return SymbolWrapper(self.value, variable_name=self.variable_name)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, SymbolWrapper):
            if self.name == other.name:
                return (self.variable_name or '') < (other.variable_name or '')
            return self.name < other.name
        if isinstance(other, NamedAtom):
            return self.name < other.name
        if isinstance(other, Operation):
            return True  # Atoms sort before Operations
        return True  # Before Wildcards

    def __eq__(self, other):
        if isinstance(other, SymbolWrapper):
            return self.value == other.value and self.variable_name == other.variable_name
        if isinstance(other, NamedAtom):
            return self.name == other.name and self.variable_name == other.variable_name
        # Allow direct comparison with the wrapped value (e.g. SymbolWrapper(Integer(2)) == Integer(2))
        if not isinstance(other, Expression) and self.variable_name is None:
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(('_named_atom_', self.name, self.variable_name))

    def __gt__(self, other):
        if not isinstance(other, Expression):
            return self.value > other
        return NotImplemented

    def __ge__(self, other):
        if not isinstance(other, Expression):
            return self.value >= other
        return NotImplemented

    def __le__(self, other):
        if not isinstance(other, Expression):
            return self.value <= other
        return NotImplemented


# ─── Wildcard ─────────────────────────────────────────────────────────────────

class Wildcard(AtomExpr):
    """A wildcard that matches any expression.

    Attributes:
        min_count: Minimum number of expressions this wildcard matches.
        fixed_size: If True, matches exactly min_count expressions.
        default_value: Default value for optional wildcards.
    """
    min_count: int = 1
    fixed_size: bool = True
    default_value: object = None

    def __init__(self, min_count=1, fixed_size=True, variable_name=None, default_value=None, **kwargs) -> None:
        if min_count < 0:
            raise ValueError("min_count must be non-negative")
        if fixed_size and min_count == 0 and default_value is None:
            raise ValueError("Wildcard with min_count=0 and fixed_size=True requires a default_value (use Wildcard.optional)")
        super().__init__(min_count=min_count, fixed_size=fixed_size, variable_name=variable_name,
                         default_value=default_value, **kwargs)
        self.head = None

    @staticmethod
    def dot(name=None) -> 'Wildcard':
        """Create a fixed-size wildcard (matches exactly one expression)."""
        return Wildcard(min_count=1, fixed_size=True, variable_name=name)

    @staticmethod
    def star(name=None) -> 'Wildcard':
        """Create a star wildcard (matches zero or more expressions)."""
        return Wildcard(min_count=0, fixed_size=False, variable_name=name)

    @staticmethod
    def plus(name=None) -> 'Wildcard':
        """Create a plus wildcard (matches one or more expressions)."""
        return Wildcard(min_count=1, fixed_size=False, variable_name=name)

    @staticmethod
    def optional(name, default) -> 'Wildcard':
        """Create an optional wildcard with a default value.

        Optional wildcards match exactly 1 expression (like dot), but fall back
        to the default_value when no subject is available in variadic operations.
        """
        return Wildcard(min_count=1, fixed_size=True, variable_name=name, default_value=default)


    def __str__(self):
        if self.variable_name:
            if self.fixed_size:
                return self.variable_name + '_'
            elif self.min_count == 0:
                return self.variable_name + '___'
            else:
                return self.variable_name + '__'
        if not self.fixed_size:
            if self.min_count == 0:
                return '___'
            return '__'
        return '_'

    def __repr__(self):
        if self.variable_name:
            suffix = 'dot' if self.fixed_size else ('star' if self.min_count == 0 else 'plus')
            return f'Wildcard.{suffix}({self.variable_name!r})'
        return f'Wildcard({self.min_count}, {self.fixed_size})'

    def collect_symbols(self, symbols):
        pass

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        new_name = renaming.get(self.variable_name, self.variable_name)
        return Wildcard(self.min_count, self.fixed_size, variable_name=new_name, default_value=self.default_value)

    def __copy__(self) -> 'Wildcard':
        return Wildcard(self.min_count, self.fixed_size, variable_name=self.variable_name,
                        default_value=self.default_value)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, (NamedAtom, SymbolWrapper)) and not isinstance(other, Wildcard):
            return False  # Wildcards sort after Symbols
        if isinstance(other, Operation):
            return True  # Atoms sort before Operations
        if isinstance(other, Wildcard):
            # Ordering: fixed_size → min_count → variable_name
            if self.fixed_size != other.fixed_size:
                return self.fixed_size  # True (dot) < False (sequence)
            if self.min_count != other.min_count:
                return self.min_count < other.min_count
            if (self.variable_name or '') != (other.variable_name or ''):
                return (self.variable_name or '') < (other.variable_name or '')
            return False  # equal
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Wildcard):
            return NotImplemented
        return (self.min_count == other.min_count and self.fixed_size == other.fixed_size and
                self.variable_name == other.variable_name and self.default_value == other.default_value)

    def __hash__(self):
        return hash((Wildcard, self.min_count, self.fixed_size, self.variable_name))



# ─── Pattern ──────────────────────────────────────────────────────────────────

class Pattern(TypedModel):
    """A pattern wrapping an expression with constraints.

    Attributes:
        expression: The expression to match against.
        constraints: Tuple of constraints that must be satisfied.
    """
    expression: Expression
    constraints: Tuple['Constraint', ...] = ()

    def __init__(self, expression, *constraints, **kwargs):
        if 'constraints' in kwargs and not constraints:
            # Reconstructing from serialized data — constraints already in kwargs
            super().__init__(expression=expression, **kwargs)
        elif constraints:
            super().__init__(expression=expression, constraints=tuple(constraints), **kwargs)
        else:
            super().__init__(expression=expression, **kwargs)

    @property
    def head(self):
        return self.expression.head

    @property
    def variable_name(self):
        return self.expression.variable_name

    @cached_property
    def local_constraints(self):
        """Constraints that depend on pattern variables (checked during matching).

        cached_property: rebuilt tuples on every access were measurable during matcher
        construction (accessed once per _internal_add, ~100k times on the Rubi set).
        """
        return tuple(c for c in self.constraints if c.variables)

    @cached_property
    def global_constraints(self):
        """Constraints with no variables (checked after matching completes).

        cached_property: accessed once per yielded match in _internal_iter.
        """
        return tuple(c for c in self.constraints if not c.variables)

    def __eq__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        # frozenset comparison via a cached view -- the old `set(a) == set(b)` built
        # two fresh sets on every comparison (Pattern is a dict key in the matcher).
        return self.expression == other.expression and self._constraint_set == other._constraint_set

    @cached_property
    def _constraint_set(self):
        return frozenset(self.constraints)

    def __hash__(self):
        h = self.__dict__.get('_cached_hash')
        if h is None:
            h = hash((Pattern, self.expression, self.constraints))
            self.__dict__['_cached_hash'] = h
        return h

    def __repr__(self):
        if self.constraints:
            return f'Pattern({self.expression!r}, {", ".join(repr(c) for c in self.constraints)})'
        return f'Pattern({self.expression!r})'

    def __str__(self):
        return str(self.expression)




# ─── Built-in operation heads ─────────────────────────────────────────────────

LIST_HEAD = OperationHead(name='list', arity=Arity.variadic)
TUPLE_HEAD = OperationHead(name='tuple', arity=Arity.variadic)
DICT_HEAD = OperationHead(name='dict', arity=Arity.variadic, commutative=True)
DICT_PAIR_HEAD = OperationHead(name='dictpair', arity=Arity.binary)


# ─── Singledispatch converters ────────────────────────────────────────────────

@singledispatch
def to_omnimatch_expression(obj) -> Expression:
    """Convert ANY object into a OmniMatch :class:`Expression` (the ingestion point).

    True singledispatch: every supported input type is a registration -- Python
    containers below, and e.g. the whole SymPy tree via ``sympy_matching``.
    The default turns an unknown object into a ``NamedAtom`` of its ``str()``.
    """
    return NamedAtom(str(obj))


@to_omnimatch_expression.register(Expression)
def _expression_to_expression(obj: Expression) -> Expression:
    """A OmniMatch expression is already converted -- identity."""
    return obj


@to_omnimatch_expression.register(dict)
def _dict_to_expression(obj: dict) -> Expression:
    pairs = [Operation(DICT_PAIR_HEAD, to_omnimatch_expression(k), to_omnimatch_expression(v))
             for k, v in obj.items()]
    return Operation(DICT_HEAD, *pairs)


@to_omnimatch_expression.register(list)
def _list_to_expression(obj: list) -> Expression:
    return Operation(LIST_HEAD, *[to_omnimatch_expression(item) for item in obj])


@to_omnimatch_expression.register(tuple)
def _tuple_to_expression(obj: tuple) -> Expression:
    return Operation(TUPLE_HEAD, *[to_omnimatch_expression(item) for item in obj])


@singledispatch
def from_omnimatch_expression(expr):
    """Convert a OmniMatch expression back to a GENERIC Python object.

    This is the domain-agnostic reverse of :func:`to_omnimatch_expression`: atoms unwrap to
    their names/values, everything else passes through unchanged. A conversion
    targeting a specific library belongs in its OWN dispatch function -- e.g.
    ``sympy_matching.omnimatch_to_sympy`` maps ``NamedAtom -> sympy.Symbol`` and
    operation heads to SymPy classes, which would be wrong to impose here (the
    registries are global, so registering SymPy semantics on this function would
    change behaviour for every non-SymPy user of omnimatch).
    """
    return expr


@from_omnimatch_expression.register(NamedAtom)
def _named_atom_from_expression(expr: NamedAtom):
    return expr.name


@from_omnimatch_expression.register(Wildcard)
def _wildcard_from_expression(expr: Wildcard):
    # A Wildcard is not a NamedAtom-with-a-name from the caller's perspective;
    # pass it through unchanged (the old default's `and not isinstance(Wildcard)`
    # exclusion, expressed as a proper registration).
    return expr


@from_omnimatch_expression.register(SymbolWrapper)
def _symbol_wrapper_from_expression(expr: 'SymbolWrapper'):
    """Unwrap the original wrapped object -- omnimatch-generic, so it lives here.

    (``sympy_matching`` overrides this registration with a HeadRef-aware version
    for wrapped operation heads.)
    """
    return expr.value


# ─── Factory helpers ──────────────────────────────────────────────────────────

def make_dot_variable(name: str) -> Wildcard:
    """Create a named dot wildcard (matches exactly one expression)."""
    return Wildcard.dot(name)


def make_plus_variable(name: str) -> Wildcard:
    """Create a named plus wildcard (matches one or more expressions)."""
    return Wildcard.plus(name)


def make_star_variable(name: str) -> Wildcard:
    """Create a named star wildcard (matches zero or more expressions)."""
    return Wildcard.star(name)


