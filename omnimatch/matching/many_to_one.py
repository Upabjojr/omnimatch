# -*- coding: utf-8 -*-
"""Contains the :class:`ManyToOneMatcher` which can be used for fast many-to-one matching.

You can initialize the matcher with a list of the patterns that you wish to match:

>>> pattern1 = Pattern(f(a, x_))
>>> pattern2 = Pattern(f(y_, b))
>>> matcher = ManyToOneMatcher(pattern1, pattern2)

You can also add patterns later:

>>> pattern3 = Pattern(f(a, b))
>>> matcher.add(pattern3)

A pattern can be added with a label which is yielded instead of the pattern during matching:

>>> pattern4 = Pattern(f(x_, y_))
>>> matcher.add(pattern4, "some label")

Then you can match a subject against all the patterns at once:

>>> subject = f(a, b)
>>> matches = matcher.match(subject)
>>> for matched_pattern, substitution in sorted(map(lambda m: (str(m[0]), str(m[1])), matches)):
...     print('{} matched with {}'.format(matched_pattern, substitution))
f(a, b) matched with {}
f(a, x_) matched with {x ↦ b}
f(y_, b) matched with {y ↦ a}
some label matched with {x ↦ a, y ↦ b}

Also contains the :class:`ManyToOneReplacer` which can replace a set :class:`ReplacementRule` at one using a
:class:`ManyToOneMatcher` for finding the matches.
"""
import math
import html
import itertools
import copy
from collections import deque
from operator import itemgetter
from typing import ClassVar, Container, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Type

from .._typed import TypedModel, field

try:
    from graphviz import Digraph, Graph
except ImportError:
    Digraph = None
    Graph = None
from multiset import Multiset

from ..expressions.expressions import (
    Expression, Operation, OperationHead, NamedAtom, SymbolWrapper,
    Wildcard, WildcardOperationHead, Pattern
)
from ..expressions.constraints import Constraint
from ..expressions.substitution import Substitution
from ..expressions.functions import (
    is_anonymous, contains_variables_from_set, create_operation_expression, preorder_iter_with_position,
    rename_variables, op_iter, preorder_iter, op_len
)
from ..utils import (VariableWithCount, commutative_sequence_variable_partition_iter)
from .. import functions
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter, LEFT, RIGHT
from ._common import check_one_identity, OPERATION_END

__all__ = ['ManyToOneMatcher', 'ManyToOneReplacer']

MultisetOfInt = Multiset
MultisetOfExpression = Multiset

# Used only by visualization code (debug)
_VISITED = set()


# ── LabelType hierarchy ──────────────────────────────────────────────────────
# Represents the value stored in _Transition.label.

class LabelType:
    """Base class for transition labels (plain __slots__ class: internal to the
    matcher, always constructed by omnimatch itself, so the TypedModel type-check
    layer bought nothing and cost ~3.1M generic-init calls per Rubi build)."""
    __slots__ = ()

    def unwrap(self):
        """Return the raw wrapped value (for downstream isinstance checks)."""
        raise NotImplementedError

class LabelTypeExpression(LabelType):
    """Label wrapping an expression value (Expression or native Python object)."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def unwrap(self):
        return self.value

    def __hash__(self):
        return hash(('LabelTypeExpression', self.value))

    def __eq__(self, other):
        if isinstance(other, LabelTypeExpression):
            return self.value == other.value
        return NotImplemented

    def __repr__(self):
        return f'LabelTypeExpression({self.value!r})'

class LabelTypeOperation(LabelType):
    """Label wrapping an OperationHead."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def unwrap(self):
        return self.value

    def __hash__(self):
        return hash(('LabelTypeOperation', self.value))

    def __eq__(self, other):
        if isinstance(other, LabelTypeOperation):
            return self.value == other.value
        return NotImplemented

    def __repr__(self):
        return f'LabelTypeOperation({self.value!r})'

class LabelTypeEpsilon(LabelType):
    """Singleton epsilon (empty) transition label."""
    __slots__ = ()

    def unwrap(self):
        return None

    def __hash__(self):
        return hash('LabelTypeEpsilon')

    def __eq__(self, other):
        return isinstance(other, LabelTypeEpsilon) and not isinstance(other, LabelTypeEnd)

    def __repr__(self):
        return '_EPS'

class LabelTypeEnd(LabelType):
    """Label for OPERATION_END transitions (end of operands)."""
    __slots__ = ()

    def unwrap(self):
        return OPERATION_END

    def __hash__(self):
        return hash('LabelTypeEnd')

    def __eq__(self, other):
        return isinstance(other, LabelTypeEnd)

    def __repr__(self):
        return 'OPERATION_END'

_EPS = LabelTypeEpsilon()
_LABEL_END = LabelTypeEnd()


# ── TransitionKey / HeadType hierarchy ───────────────────────────────────────
# TransitionKey is the base for all dict keys in _State.transitions.
# HeadType is a subclass of TransitionKey representing expression heads.

class TransitionKey:
    """Base class for transition dictionary keys (plain __slots__ class)."""
    __slots__ = ()

class HeadType(TransitionKey):
    """Base for expression heads used as transition keys."""
    __slots__ = ()

class HeadTypeExpression(HeadType):
    """Head is a specific expression value (e.g. a NamedAtom instance or native object)."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(('HeadTypeExpression', self.value))

    def __eq__(self, other):
        if isinstance(other, HeadTypeExpression):
            return self.value == other.value
        return NotImplemented

class HeadTypeOperation(HeadType):
    """Head is an OperationHead instance."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, HeadTypeOperation):
            return self.value == other.value
        return NotImplemented

class HeadTypeNone(HeadType):
    """Wildcard head (matches anything). Represents the None catch-all."""
    __slots__ = ()

    def __hash__(self):
        return hash('HeadTypeNone')

    def __eq__(self, other):
        return isinstance(other, HeadTypeNone)

# Singleton for the None/wildcard head
_HEAD_NONE = HeadTypeNone()


class HeadTypeAnyOperation(HeadType):
    """Key for patterns whose operation HEAD is itself a wildcard.

    A pattern built with a :class:`WildcardOperationHead` (e.g. Rubi's ``F_[v_]``)
    must be reachable for an operation subject with ANY head, so it is filed under
    this single key instead of under its own head; every operation subject offers
    this key in addition to its concrete head (see ``_get_heads``).
    """
    __slots__ = ()

    def __hash__(self):
        return hash('HeadTypeAnyOperation')

    def __eq__(self, other):
        return isinstance(other, HeadTypeAnyOperation)

# Singleton for the "any operation head" key
_HEAD_ANY_OP = HeadTypeAnyOperation()

class TransitionKeyEnd(TransitionKey):
    """Key for OPERATION_END transitions."""
    __slots__ = ()

    def __hash__(self):
        return hash('TransitionKeyEnd')

    def __eq__(self, other):
        return isinstance(other, TransitionKeyEnd)

# Singleton for OPERATION_END key
_TRANSITION_END = TransitionKeyEnd()

class TransitionKeyPatternId(TransitionKey):
    """Key for commutative subpattern id transitions."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(('TransitionKeyPatternId', self.value))

    def __eq__(self, other):
        if isinstance(other, TransitionKeyPatternId):
            return self.value == other.value
        return NotImplemented


# ── _PatternKey / _PatternValue ──────────────────────────────────────────────

class _PatternKey:
    """Hashable key for CommutativeMatcher.patterns dict (plain __slots__ class)."""
    __slots__ = ('subpatterns', 'variables')

    def __init__(self, subpatterns, variables):
        # subpatterns: Tuple[int, ...]; variables: Tuple[Tuple[VariableWithCount, bool], ...]
        self.subpatterns = subpatterns
        self.variables = variables

    def __hash__(self):
        return hash((self.subpatterns, self.variables))

    def __eq__(self, other):
        if isinstance(other, _PatternKey):
            return self.subpatterns == other.subpatterns and self.variables == other.variables
        return NotImplemented

class _PatternValue:
    """Value stored in CommutativeMatcher.patterns dict (plain __slots__ class)."""
    __slots__ = ('index', 'pattern_set', 'variables')

    def __init__(self, index, pattern_set, variables):
        self.index = index
        self.pattern_set = pattern_set
        self.variables = variables

class _State:
    __slots__ = ('number', 'transitions', 'matcher')

    def __init__(self, number, transitions=None, matcher=None):
        self.number = number                                  # int
        self.transitions = {} if transitions is None else transitions  # Dict[TransitionKey, List[_Transition]]
        self.matcher = matcher                                # Optional[CommutativeMatcher]

    # NOTE: no custom model_dump / transitions field_validator here. JSON
    # (de)serialization is hand-rolled in omnimatch.matching.json_serialization
    # (it constructs _State with a dict and populates it), so a `mode='before'`
    # validator only added overhead on every one of the ~260K state constructions
    # during a matcher build (and referenced an un-imported name for the list
    # branch that was never exercised).

    def __hash__(self):
        return hash(self.number)

    def __eq__(self, other):
        if isinstance(other, _State):
            return self.number == other.number
        return NotImplemented


class _Transition:
    __slots__ = ('label', 'target', 'variable_name', 'patterns', 'check_constraints', 'subst')

    def __init__(self, label, target, variable_name=None, patterns=None, check_constraints=None, subst=None):
        self.label = label                                    # LabelType
        self.target = target                                  # _State
        self.variable_name = variable_name                    # Optional[str]
        self.patterns = set() if patterns is None else patterns  # Set[int]
        self.check_constraints = check_constraints            # Optional[Set[int]]
        self.subst = subst                                    # Optional[Substitution]

    def __hash__(self):
        return hash((self.label, self.target.number, self.variable_name))

    def __eq__(self, other):
        if isinstance(other, _Transition):
            return self.label == other.label and self.target.number == other.target.number and self.variable_name == other.variable_name
        return NotImplemented



class _MatchIter:
    def __init__(self, matcher, subject, intial_associative=None):
        self.matcher = matcher
        self.subjects = deque([subject]) if subject is not None else deque()
        self.patterns = set(range(len(matcher.patterns)))
        self.substitution = Substitution()
        self.constraints = set(range(len(matcher.constraints)))
        self.associative = [intial_associative]

    def __iter__(self):
        for _ in self._match(self.matcher.root):
            yield from self._internal_iter()

    def grouped(self):
        """
        Yield the matches grouped by their final state in the automaton, i.e. structurally identical patterns
        only differing in constraints will be yielded together. Each group is yielded as a list of tuples consisting of
        a pattern and a match substitution.

        Yields:
            The grouped matches.
        """
        for _ in self._match(self.matcher.root):
            yield list(self._internal_iter())

    def any(self):
        """
        Returns:
            True, if any match is found.
        """
        try:
            next(self)
        except StopIteration:
            return False
        return True

    def _internal_iter(self):
        for pattern_index in self.patterns:
            renaming = self.matcher.pattern_vars[pattern_index]
            new_substitution = self.substitution.rename({renamed: original for original, renamed in renaming.items()})
            pattern, label, _ = self.matcher.patterns[pattern_index]
            valid = True
            for constraint in pattern.global_constraints:
                if not constraint(new_substitution):
                    valid = False
                    break
            if valid:
                yield label, new_substitution

    def _match(self, state: _State) -> Iterator[_State]:
        # (a write-only `self.visited` set was removed here -- it was populated on
        # every state visit and never read; the debug visualization uses the module
        # level _VISITED instead.)
        if len(self.subjects) == 0:
            if state.number in self.matcher.finals or _TRANSITION_END in state.transitions:
                yield state
            heads = [_HEAD_NONE]
        else:
            heads = list(self._get_heads(self.subjects[0]))
        for head in heads:
            for transition in state.transitions.get(head, []):
                yield from self._match_transition(transition)

    def _match_transition(self, transition: _Transition) -> Iterator[_State]:
        if self.patterns.isdisjoint(transition.patterns):
            return
        label = transition.label
        if isinstance(label, LabelTypeEpsilon):
            subject = self.subjects[0] if self.subjects else None
            yield from self._check_transition(transition, subject, False)
            return
        if isinstance(label, LabelTypeOperation):
            if transition.target.matcher:
                yield from self._match_commutative_operation(transition.target)
            else:
                yield from self._match_regular_operation(transition)
            return
        if isinstance(label, LabelTypeExpression):
            expr = label.value
            if isinstance(expr, Wildcard):
                min_count = expr.min_count
                if expr.default_value is not None and min_count > 0:
                    yield from self._check_transition(transition, expr.default_value, False)
                if expr.fixed_size and not self.associative[-1]:
                    assert min_count == 1, "Fixed wildcards with length != 1 are not supported."
                    if not self.subjects:
                        return
                else:
                    yield from self._match_sequence_variable(expr, transition)
                    return
        subject = self.subjects.popleft() if self.subjects else None
        yield from self._check_transition(transition, subject)

    def _check_transition(self, transition, subject, restore_subject=True):
        if self.patterns.isdisjoint(transition.patterns):
            return
        restore_constraints = set()
        restore_patterns = self.patterns - transition.patterns
        self.patterns &= transition.patterns
        old_values = {}
        try:
            if transition.subst is not None:
                try:
                    for name, value in transition.subst.items():
                        old_values[name] = self.substitution.get(name, None)
                        self.substitution.try_add_variable(name, value)
                except ValueError:
                    for k, v in old_values.items():
                        if v is None:
                            del self.substitution[k]
                        else:
                            self.substitution[k] = v
                    return

            if transition.variable_name is not None:
                try:
                    old_values[transition.variable_name] = self.substitution.get(transition.variable_name, None)
                    self.substitution.try_add_variable(transition.variable_name, subject)
                except ValueError:
                    return
                self._check_constraints(transition.check_constraints, restore_constraints, restore_patterns)
                if not self.patterns:
                    return

            yield from self._match(transition.target)

        finally:
            if restore_subject and subject is not None:
                self.subjects.appendleft(subject)
            self.constraints |= restore_constraints
            self.patterns |= restore_patterns
            for k, v in old_values.items():
                if v is None:
                    del self.substitution[k]
                else:
                    self.substitution[k] = v

    def _check_constraints(self, variable: str, restore_constraints, restore_patterns) -> bool:
        if isinstance(variable, str):
            by_name = self.matcher.constraint_vars.get(variable)
            if not by_name:
                return
            # Only constraints whose patterns still overlap the viable patterns are
            # processed anyway (the disjoint check below). Common variable names
            # (x, a, b, m, ...) appear in thousands of rules, so `by_name` can be
            # huge while there are few viable patterns; in that case gather the
            # candidates from the per-(variable, pattern) index rather than
            # scanning every constraint on the name. This yields exactly the same
            # set of processed constraints (a subset the disjoint check would keep).
            cpm = self.matcher.constraint_pattern_map
            # `cpm` is a build-time index; it is empty on a matcher rebuilt from a
            # serialized form (constraints/constraint_vars set directly). Fall back
            # to the full scan in that case — it is always correct.
            if cpm and len(self.patterns) < len(by_name):
                check_constraints = set()
                for p in self.patterns:
                    cs = cpm.get((variable, p))
                    if cs:
                        check_constraints |= cs
            else:
                check_constraints = by_name
        else:
            check_constraints = variable
        variables = set(self.substitution.keys())
        for constraint_index in check_constraints:
            if constraint_index not in self.constraints:
                continue
            constraint, patterns = self.matcher.constraints[constraint_index]
            if constraint.variables <= variables and not self.patterns.isdisjoint(patterns):
                self.constraints.remove(constraint_index)
                restore_constraints.add(constraint_index)
                if not constraint(self.substitution):
                    restore_patterns |= self.patterns & patterns
                    self.patterns -= patterns
                    if not self.patterns:
                        break

    @staticmethod
    def _get_heads(expression: Expression) -> Iterator[TransitionKey]:
        if isinstance(expression, Operation):
            yield HeadTypeOperation(value=expression.head)
            # Also offer the "any operation head" key so that patterns whose head
            # is a WildcardOperationHead are reachable for this subject.
            yield _HEAD_ANY_OP
        else:
            yield HeadTypeExpression(value=expression)
        yield _HEAD_NONE

    def _match_sequence_variable(self, wildcard: Wildcard, transition: _Transition) -> Iterator[_State]:
        min_count = wildcard.min_count
        if len(self.subjects) < min_count:
            return
        matched_subject = []
        for _ in range(min_count):
            matched_subject.append(self.subjects.popleft())
        while True:
            if self.associative[-1] and wildcard.fixed_size:
                assert min_count == 1, "Fixed wildcards with length != 1 are not supported."
                if len(matched_subject) > 1:
                    wrapped = self.associative[-1](*matched_subject)
                else:
                    wrapped = matched_subject[0]
            else:
                if len(matched_subject) == 0 and wildcard.default_value is not None:
                    wrapped = wildcard.default_value
                else:
                    wrapped = tuple(matched_subject)
            yield from self._check_transition(transition, wrapped, False)
            if not self.subjects:
                break
            matched_subject.append(self.subjects.popleft())
        self.subjects.extendleft(reversed(matched_subject))

    def _match_commutative_operation(self, state: _State) -> Iterator[_State]:
        subject = self.subjects.popleft()
        matcher = state.matcher
        substitution = self.substitution
        matcher.add_subject(None)
        for operand in op_iter(subject):
            matcher.add_subject(operand)
        # `self.patterns` is the set of still-viable pattern ids -- one entry per
        # rule, so ~8500 for the Rubi rule set. Every set operation on it is O(n),
        # and the body below runs once PER MATCH, which made this the hottest
        # function once _build_bipartite was fixed. Three things are done about it:
        #
        #  * the entry value is captured ONCE, so restoring at the end of each
        #    iteration is a plain rebind instead of an O(n) union;
        #  * the narrowing is a single intersection instead of an intersection plus
        #    a difference (the difference existed only to rebuild the entry value);
        #  * the transition lookup, which builds a type-checked TransitionKey, is
        #    done once per match instead of twice.
        #
        # `restore_patterns` is still passed to _check_constraints because that
        # accumulates into it, but its contents are no longer needed here: rebinding
        # `saved_patterns` restores exactly the same set it would have rebuilt.
        saved_patterns = self.patterns
        transitions = state.transitions
        for matched_pattern, new_substitution in matcher.match(subject, substitution):
            restore_constraints = set()
            # dict key views support set algebra directly, with no intermediate sets
            diff = new_substitution.keys() - substitution.keys()
            self.substitution = new_substitution
            transition_set = transitions[TransitionKeyPatternId(value=matched_pattern)]
            if len(transition_set) == 1:
                potential_patterns = transition_set[0].patterns
            else:
                t_iter = iter(t.patterns for t in transition_set)
                potential_patterns = next(t_iter).union(*t_iter)
            restore_patterns = set()
            self.patterns = saved_patterns & potential_patterns
            for variable in diff:
                self._check_constraints(variable, restore_constraints, restore_patterns)
                if not self.patterns:
                    break
            if self.patterns:
                for next_transition in transition_set:
                    yield from self._check_transition(next_transition, subject, False)
            self.constraints |= restore_constraints
            self.patterns = saved_patterns
        self.substitution = substitution
        self.subjects.appendleft(subject)

    def _match_regular_operation(self, transition: _Transition) -> Iterator[_State]:
        subject = self.subjects.popleft()
        after_subjects = self.subjects
        operand_subjects = self.subjects = deque(op_iter(subject))
        new_associative = transition.label.value if isinstance(transition.label, LabelTypeOperation) and isinstance(transition.label.value, OperationHead) and transition.label.value.associative else None
        # A WildcardOperationHead pattern matches an operation with ANY head; bind
        # the subject's head to the head variable (wrapped so it is a regular
        # expression). Restored on backtrack, like any other binding.
        head_var = None
        head_old = None
        label_head = transition.label.value if isinstance(transition.label, LabelTypeOperation) else None
        if isinstance(label_head, WildcardOperationHead) and label_head.variable_name:
            head_var = label_head.variable_name
            try:
                head_old = self.substitution.get(head_var, None)
                self.substitution.try_add_variable(head_var, SymbolWrapper(subject.head))
            except ValueError:
                # Conflicts with an existing binding for the same head variable.
                self.subjects = after_subjects
                self.subjects.appendleft(subject)
                return
        self.associative.append(new_associative)
        for new_state in self._check_transition(transition, subject, False):
            self.subjects = after_subjects
            self.associative.pop()
            for end_transition in new_state.transitions[_TRANSITION_END]:
                yield from self._check_transition(end_transition, None, False)
            self.subjects = operand_subjects
            self.associative.append(new_associative)
        self.subjects = after_subjects
        self.subjects.appendleft(subject)
        self.associative.pop()
        if head_var is not None:  # undo the wildcard-head binding on backtrack
            if head_old is None:
                self.substitution.pop(head_var, None)
            else:
                self.substitution[head_var] = head_old


class ManyToOneMatcher(TypedModel):

    patterns: List[Tuple[Pattern, object, List[int]]] = field(default_factory=list)
    states: List[_State] = field(default_factory=list)
    root: Optional[_State] = None
    pattern_vars: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[Tuple[Constraint, Set[int]]] = field(default_factory=list)
    constraint_vars: Dict[str, Set[int]] = field(default_factory=dict)
    finals: Set[int] = field(default_factory=set)
    rename: bool = True
    commutative_matchers: List['CommutativeMatcher'] = field(default_factory=list)
    # O(1) lookup index: maps Pattern -> index in self.patterns list (optimization)
    pattern_to_index: Dict[Pattern, int] = field(default_factory=dict)
    # O(1) lookup index: maps Constraint -> index in self.constraints list (optimization)
    constraint_to_index: Dict[Constraint, int] = field(default_factory=dict)
    # O(1) lookup index: maps (variable_name, pattern_index) -> the constraint
    # indices on that variable that apply to that pattern. Avoids scanning every
    # constraint sharing a variable name (a, b, m, x, ... are reused across
    # thousands of rules) when building each transition (optimization).
    constraint_pattern_map: Dict[Tuple[str, int], Set[int]] = field(default_factory=dict)

    _state_id: ClassVar[int] = 0

    def __init__(self, *patterns: Expression, rename=True, **kwargs) -> None:
        """
        Args:
            *patterns: The patterns which the matcher should match.
        """
        super().__init__(rename=rename, **kwargs)
        # When reconstructing from serialized data, self.root is already set
        # by TypedModel field initialization. Only create a fresh root for new
        # matcher instances.
        if self.root is None:
            self.root = self._create_state()

        for pattern in patterns:
            self.add(pattern)

    def clear(self):
        """Removes all cached data."""
        for commutative_matcher in self.commutative_matchers:
            commutative_matcher.clear()

    def add(self, pattern: Pattern, label=None) -> None:
        """Add a new pattern to the matcher.

        The optional label defaults to the pattern itself and is yielded during matching. The same pattern can be
        added with different labels which means that every match for the pattern will result in every associated label
        being yielded with that match individually.

        Equivalent patterns with the same label are not added again. However, patterns that are structurally equivalent,
        but have different constraints or different variable names are distinguished by the matcher.

        Args:
            pattern:
                The pattern to add.
            label:
                An optional label for the pattern. Defaults to the pattern itself.
        """
        if label is None:
            label = pattern
        # O(1) duplicate check using pattern_to_index (was O(n) linear scan)
        if pattern in self.pattern_to_index:
            idx = self.pattern_to_index[pattern]
            # Still need to check label matches
            _, l, _ = self.patterns[idx]
            if label == l:
                return idx
        # TODO: Avoid renaming in the pattern, use variable indices instead
        renaming = self._collect_variable_renaming(pattern.expression) if self.rename else {}
        self._internal_add(pattern, label, renaming)

    def _internal_add(self, pattern: Pattern, label, renaming) -> int:
        """Add a new pattern to the matcher.

        Equivalent patterns are not added again. However, patterns that are structurally equivalent,
        but have different constraints or different variable names are distinguished by the matcher.

        Args:
            pattern: The pattern to add.

        Returns:
            The internal id for the pattern. This is mainly used by the :class:`CommutativeMatcher`.
        """
        pattern_index = len(self.patterns)
        # Skip the rename entirely for an identity renaming: the CommutativeMatcher
        # sub-pattern path passes {n: n for n in vnames}, so with_renamed_vars would
        # rebuild every constraint (re-running CustomConstraint's signature
        # introspection) and re-sort every commutative operand for a provable no-op --
        # measured ~185k redundant CustomConstraint reconstructions on the Rubi set.
        is_identity = all(k == v for k, v in renaming.items())
        if is_identity:
            renamed_constraints = list(pattern.local_constraints)
        else:
            renamed_constraints = [c.with_renamed_vars(renaming) for c in pattern.local_constraints]
        constraint_indices = [self._add_constraint(c, pattern_index) for c in renamed_constraints]
        self.patterns.append((pattern, label, constraint_indices))
        self.pattern_to_index[pattern] = pattern_index  # O(1) lookup index
        self.pattern_vars.append(renaming)
        pattern = pattern.expression if is_identity else rename_variables(pattern.expression, renaming)
        state = self.root
        patterns_stack = [deque([pattern])]

        self._process_pattern_stack(state, patterns_stack, renamed_constraints, pattern_index)

        return pattern_index

    def _process_pattern_stack(self, state, patterns_stack, renamed_constraints, pattern_index):
        while patterns_stack:
            if patterns_stack[-1]:
                subpattern = patterns_stack[-1].popleft()
                variable_name = getattr(subpattern, 'variable_name', None)
                if isinstance(subpattern, Operation):
                    if isinstance(subpattern, Operation) and subpattern.head.one_identity:
                        non_optional, added_subst = check_one_identity(subpattern)
                        if non_optional is not None:
                            stack = [q.copy() for q in patterns_stack]
                            stack[-1].appendleft(non_optional)
                            new_state = self._create_expression_transition(state, _EPS, variable_name, pattern_index, added_subst)
                            self._process_pattern_stack(new_state, stack, renamed_constraints, pattern_index)
                    if not (isinstance(subpattern, Operation) and subpattern.head.commutative):
                        patterns_stack.append(deque(op_iter(subpattern)))
                state = self._create_expression_transition(state, subpattern, variable_name, pattern_index)
                if isinstance(subpattern, Operation) and subpattern.head.commutative:
                    subpattern_id = state.matcher.add_pattern(subpattern, renamed_constraints)
                    state = self._create_simple_transition(state, subpattern_id, pattern_index)
            else:
                patterns_stack.pop()
                if len(patterns_stack) > 0:
                    state = self._create_simple_transition(state, OPERATION_END, pattern_index)
        self.finals.add(state.number)

    def _add_constraint(self, constraint, pattern):
        # O(1) lookup using constraint_to_index (was O(n) linear scan)
        index = self.constraint_to_index.get(constraint)
        if index is not None:
            _, patterns_set = self.constraints[index]
            patterns_set.add(pattern)
        else:
            index = len(self.constraints)
            self.constraints.append((constraint, set([pattern])))
            self.constraint_to_index[constraint] = index
        # try/except instead of setdefault: setdefault eagerly allocates a throwaway
        # set() (and a key tuple) on every call -- ~2.3M wasted allocations building
        # the Rubi matcher (measured ~0.5-1s).
        cvars = self.constraint_vars
        cpm = self.constraint_pattern_map
        for var in constraint.variables:
            try:
                cvars[var].add(index)
            except KeyError:
                cvars[var] = {index}
            key = (var, pattern)
            try:
                cpm[key].add(index)
            except KeyError:
                cpm[key] = {index}
        return index

    def match(self, subject: Expression) -> Iterator[Tuple[Expression, Substitution]]:
        """Match the subject against all the matcher's patterns.

        Args:
            subject: The subject to match.

        Yields:
            For every match, a tuple of the matching pattern and the match substitution.
        """
        return _MatchIter(self, subject)

    def is_match(self, subject: Expression) -> bool:
        """Check if the subject matches any of the matcher's patterns.

        Args:
            subject: The subject to match.

        Return:
            True, if the subject is matched by any of the matcher's patterns.
            False, otherwise.
        """
        return _MatchIter(self, subject).any()

    def _create_expression_transition(
            self, state: _State, expression, variable_name: Optional[str], index: int, subst=None
    ) -> _State:
        label, head = self._get_label_and_head(expression)
        transitions = state.transitions.setdefault(head, [])
        commutative = isinstance(expression, Operation) and expression.head.commutative
        matcher = None
        for transition in transitions:
            if transition.variable_name == variable_name and transition.label == label and transition.subst == subst:
                transition.patterns.add(index)
                if variable_name is not None:
                    # The constraints newly applicable to this transition are exactly
                    # those on `variable_name` that apply to the pattern `index` we
                    # just merged in; every constraint applying to a previously-added
                    # pattern is already in check_constraints (see below).
                    new_constraints = self.constraint_pattern_map.get((variable_name, index))
                    if new_constraints:
                        transition.check_constraints.update(new_constraints)
                state = transition.target
                break
        else:
            if commutative:
                matcher = CommutativeMatcher(expression.head if (isinstance(expression, Operation) and expression.head.associative) else None)
                self.commutative_matchers.append(matcher)
            state = self._create_state(matcher)
            if variable_name is not None:
                # Constraints on `variable_name` that apply to pattern `index`.
                constraints = set(self.constraint_pattern_map.get((variable_name, index), ()))
            else:
                constraints = None
            transition = _Transition(label=label, target=state, variable_name=variable_name, patterns={index}, check_constraints=constraints, subst=subst)
            transitions.append(transition)
        return state

    def _create_simple_transition(self, state: _State, label, index: int, variable_name=None) -> _State:
        # Wrap raw label into proper TransitionKey and LabelType
        if isinstance(label, str):
            key = _TRANSITION_END
            wrapped_label = _LABEL_END  # OPERATION_END transitions
        elif isinstance(label, int):
            key = TransitionKeyPatternId(value=label)
            wrapped_label = _EPS  # Pattern ID transitions use epsilon as label placeholder
        else:
            key = label
            wrapped_label = label
        if key in state.transitions:
            transition = state.transitions[key][0]
            transition.patterns.add(index)
            return transition.target
        new_state = self._create_state()
        transition = _Transition(label=wrapped_label, target=new_state, variable_name=variable_name, patterns={index}, check_constraints=None, subst=None)
        state.transitions[key] = [transition]
        return new_state

    @staticmethod
    def _get_label_and_head(expression: Expression) -> Tuple[LabelType, TransitionKey]:
        if isinstance(expression, LabelTypeEpsilon) or expression is _EPS:
            return _EPS, _HEAD_NONE
        if isinstance(expression, Operation):
            if isinstance(expression.head, WildcardOperationHead):
                # A wildcard head matches any operation head, so this pattern must
                # not be filed under its own head -- file it under the shared
                # "any operation" key, which every operation subject offers.
                return LabelTypeOperation(value=expression.head), _HEAD_ANY_OP
            return LabelTypeOperation(value=expression.head), HeadTypeOperation(value=expression.head)
        else:
            if isinstance(expression, Wildcard):
                head = _HEAD_NONE
                label_expr = Wildcard(expression.min_count, expression.fixed_size, default_value=expression.default_value)
                return LabelTypeExpression(value=label_expr), head
            elif isinstance(expression, NamedAtom):
                label_copy = copy.copy(expression)
                label_copy.variable_name = None
                return LabelTypeExpression(value=label_copy), HeadTypeExpression(value=label_copy)
            else:
                return LabelTypeExpression(value=expression), HeadTypeExpression(value=expression)

    def _create_state(self, matcher: 'CommutativeMatcher'=None) -> _State:
        state = _State(number=ManyToOneMatcher._state_id, transitions=dict(), matcher=matcher)
        self.states.append(state)
        ManyToOneMatcher._state_id += 1
        return state

    @classmethod
    def _collect_variable_renaming(
            cls, expression: Expression, position: List[int]=None, variables: Dict[str, str]=None
    ) -> Dict[str, str]:
        """Return renaming for the variables in the expression.

        The variable names are generated according to the position of the variable in the expression. The goal is to
        rename variables in structurally identical patterns so that the automaton contains less redundant states.
        """
        if position is None:
            position = [0]
        if variables is None:
            variables = {}
        if getattr(expression, 'variable_name', False):
            if expression.variable_name not in variables:
                variables[expression.variable_name] = cls._get_name_for_position(position, variables.values())
        position[-1] += 1
        if isinstance(expression, Operation):
            if isinstance(expression, Operation) and expression.head.commutative:
                for operand in op_iter(expression):
                    position.append(0)
                    cls._collect_variable_renaming(operand, position, variables)
                    position.pop()
            else:
                for operand in op_iter(expression):
                    cls._collect_variable_renaming(operand, position, variables)

        return variables

    @staticmethod
    def _get_name_for_position(position: List[int], variables: Container[str]) -> str:
        new_name = 'i{}'.format('.'.join(map(str, position)))
        if new_name in variables:
            counter = 1
            while '{}_{}'.format(new_name, counter) in variables:
                counter += 1
            new_name = '{}_{}'.format(new_name, counter)
        return new_name

    def as_graph(self) -> Digraph:  # pragma: no cover
        return self._as_graph(None)

    _PATTERN_COLORS = [
        '#2E4272',
        '#7887AB',
        '#4F628E',
        '#162955',
        '#061539',
        '#403075',
        '#887CAF',
        '#615192',
        '#261758',
        '#13073A',
        '#226666',
        '#669999',
        '#407F7F',
        '#0D4D4D',
        '#003333',
    ]

    _CONSTRAINT_COLORS = [
        '#AA3939',
        '#D46A6A',
        '#801515',
        '#550000',
        '#AA6C39',
        '#D49A6A',
        '#804515',
        '#552600',
        '#882D61',
        '#AA5585',
        '#661141',
        '#440027',
    ]

    _VARIABLE_COLORS = [
        '#8EA336',
        '#B9CC66',
        '#677B14',
        '#425200',
        '#5C9632',
        '#B5E196',
        '#85BC5E',
        '#3A7113',
        '#1F4B00',
        '#AAA139',
        '#807715',
        '#554E00',
    ]

    @classmethod
    def _colored_pattern(cls, pid):  # pragma: no cover
        color = cls._PATTERN_COLORS[pid % len(cls._PATTERN_COLORS)]
        return '<font color="{}"><b>p{}</b></font>'.format(color, pid)

    @classmethod
    def _colored_constraint(cls, cid):  # pragma: no cover
        color = cls._CONSTRAINT_COLORS[cid % len(cls._CONSTRAINT_COLORS)]
        return '<font color="{}"><b>c{}</b></font>'.format(color, cid)

    @classmethod
    def _colored_variable(cls, var):  # pragma: no cover
        color = cls._VARIABLE_COLORS[hash(var) % len(cls._VARIABLE_COLORS)]
        return '<font color="{}"><b>{}</b></font>'.format(color, var)

    @classmethod
    def _format_pattern_set(cls, patterns):  # pragma: no cover
        return '{{{}}}'.format(', '.join(map(cls._colored_pattern, patterns)))

    @classmethod
    def _format_constraint_set(cls, constraints):  # pragma: no cover
        return '{{{}}}'.format(', '.join(map(cls._colored_constraint, constraints)))

    def _as_graph(self, finals: Optional[List[str]]) -> Digraph:  # pragma: no cover
        if Digraph is None:
            raise ImportError('The graphviz package is required to draw the graph.')
        graph = Digraph()
        if finals is None:
            patterns = [
                '{}: {} with {}'.format(
                    self._colored_pattern(i), html.escape(str(p.expression)), self._format_constraint_set(c)
                ) for i, (p, l, c) in enumerate(self.patterns)
            ]
            graph.node('patterns', '<<b>Patterns:</b><br/>\n{}>'.format('<br/>\n'.join(patterns)), {'shape': 'box'})

        self._make_graph_nodes(graph, finals)
        if finals is None:
            constraints = [
                '{}: {} for {}'.format(self._colored_constraint(i), html.escape(str(c)), self._format_pattern_set(p))
                for i, (c, p) in enumerate(self.constraints)
            ]
            graph.node(
                'constraints', '<<b>Constraints:</b><br/>\n{}>'.format('<br/>\n'.join(constraints)), {'shape': 'box'}
            )
        self._make_graph_edges(graph)
        return graph

    def _make_graph_nodes(self, graph: Digraph, finals: Optional[List[str]]) -> None:  # pragma: no cover
        state_patterns = {}
        for state in self.states:
            state_patterns.setdefault(state.number, set())
            for transition in itertools.chain.from_iterable(state.transitions.values()):
                state_patterns.setdefault(transition.target.number, set()).update(transition.patterns)
        for state in self.states:
            name = 'n{!s}'.format(state.number)
            if state.matcher:
                has_states = len(state.matcher.automaton.states) > 1
                if has_states:
                    graph.node(name, 'Sub Matcher', {'shape': 'box'})
                subfinals = []
                if has_states:
                    graph.subgraph(state.matcher.automaton._as_graph(subfinals))
                submatch_label = '<<b>Sub Matcher End</b>' if has_states else '<<b>Sub Matcher</b>'
                for pattern_index, subpatterns, variables in state.matcher.patterns.values():
                    var_formatted = ', '.join(
                        '{}[{}]x{}{}{}'.format(self._colored_variable(n), m, c, 'W' if w else '', ': {}'.format(d) if d is not None else '')
                        for (n, c, m, d), w in variables
                    )
                    submatch_label += '<br/>\n{}: {} {}'.format(
                        self._colored_pattern(pattern_index), subpatterns, var_formatted
                    )
                submatch_label += '>'
                end_name = (name + '-end') if has_states else name
                graph.node(end_name, submatch_label, {'shape': 'box'})
                for f in subfinals:
                    graph.edge(f, end_name)
                if has_states:
                    graph.edge(name, 'n{}'.format(state.matcher.automaton.root.number))
            else:
                attrs = {'shape': ('doublecircle' if state.number in self.finals else 'circle')}
                if state.number in _VISITED:
                    attrs['color'] = 'red'
                graph.node(name, str(state.number), attrs)
                if state.number in self.finals:
                    sp = state_patterns[state.number]
                    if finals is not None:
                        finals.append(name + '-out')
                    variables = [
                        '{}: {}'.format(
                            self._colored_pattern(i),
                            ', '.join('{} -&gt; {}'.format(self._colored_variable(o), n) for n, o in r.items())
                        ) for i, r in enumerate(self.pattern_vars) if i in sp
                    ]
                    graph.node(
                        name + '-out', '<<b>Pattern Variables:</b><br/>\n{}>'.format('<br/>\n'.join(variables)),
                        {'shape': 'box'}
                    )
                    graph.edge(name, name + '-out')

    def _make_graph_edges(self, graph: Digraph) -> None:  # pragma: no cover
        for state in self.states:
            for _, transitions in state.transitions.items():
                for transition in transitions:
                    t_label = '<'
                    if transition.variable_name:
                        t_label += '{}: '.format(self._colored_variable(transition.variable_name))
                    t_label += '&epsilon;' if isinstance(transition.label, LabelTypeEpsilon) else html.escape(str(transition.label))
                    if isinstance(transition.label, LabelTypeOperation):
                        t_label += '('
                    t_label += '<br/>{}'.format(self._format_pattern_set(transition.patterns))
                    if transition.check_constraints is not None:
                        t_label += '<br/>{}'.format(self._format_constraint_set(transition.check_constraints))
                    if transition.subst is not None:
                        t_label += '<br/>{}'.format(html.escape(str(transition.subst)))
                    t_label += '>'

                    start = 'n{!s}'.format(state.number)
                    if state.matcher and len(state.matcher.automaton.states) > 1:
                        start += '-end'
                    end = 'n{!s}'.format(transition.target.number)
                    graph.edge(start, end, t_label)


class ManyToOneReplacer:
    """Class that contains a set of replacement rules and can apply them efficiently to an expression."""

    def __init__(self, *rules):
        """
        A replacement rule consists of a *pattern*, that is matched against any subexpression
        of the expression. If a match is found, the *replacement* callback of the rule is called with
        the variables from the match substitution. Whatever the callback returns is used as a replacement for the
        matched subexpression. This can either be a single expression or a sequence of expressions, which is then
        integrated into the surrounding operation in place of the subexpression.

        Note that the pattern can therefore not be a single sequence variable/wildcard, because only single expressions
        will be matched.

        Args:
            *rules:
                The replacement rules.
        """
        self.matcher = ManyToOneMatcher()
        for rule in rules:
            self.add(rule)

    def add(self, rule: 'functions.ReplacementRule') -> None:
        """Add a new rule to the replacer.

        Args:
            rule:
                The rule to add.
        """
        self.matcher.add(rule.pattern, rule.replacement)

    def replace(self, expression: Expression, max_count: int=math.inf) -> Expression:
        """Replace all occurrences of the patterns according to the replacement rules.

        Args:
            expression:
                The expression to which the replacement rules are applied.
            max_count:
                If given, at most *max_count* applications of the rules are performed. Otherwise, the rules
                are applied until there is no more match. If the set of replacement rules is not confluent,
                the replacement might not terminate without a *max_count* set.

        Returns:
            The resulting expression after the application of the replacement rules. This can also be a sequence of
            expressions, if the root expression is replaced with a sequence of expressions by a rule.
        """
        replaced = True
        replace_count = 0
        while replaced and replace_count < max_count:
            replaced = False
            for subexpr, pos in preorder_iter_with_position(expression):
                try:
                    replacement, subst = next(iter(self.matcher.match(subexpr)))
                    result = replacement(**subst)
                    expression = functions.replace(expression, pos, result)
                    replaced = True
                    break
                except StopIteration:
                    pass
            replace_count += 1
        return expression

    def replace_post_order(self, expression: Expression) -> Expression:
        """Replace all occurrences of the patterns according to the replacement rules.

        Replaces innermost expressions first.

        Args:
            expression:
                The expression to which the replacement rules are applied.
            max_count:
                If given, at most *max_count* applications of the rules are performed. Otherwise, the rules
                are applied until there is no more match. If the set of replacement rules is not confluent,
                the replacement might not terminate without a *max_count* set.

        Returns:
            The resulting expression after the application of the replacement rules. This can also be a sequence of
            expressions, if the root expression is replaced with a sequence of expressions by a rule.
        """
        return self._replace_post_order(expression)[0]

    def _replace_post_order(self, expression):
        any_replaced = False
        while True:
            if isinstance(expression, Operation):
                new_operands = [self._replace_post_order(o) for o in op_iter(expression)]
                if any(r for _, r in new_operands):
                    new_operands = [o for o, _ in new_operands]
                    expression = create_operation_expression(expression, new_operands)
                    any_replaced = True
            try:
                replacement, subst = next(iter(self.matcher.match(expression)))
                expression = replacement(**subst)
                any_replaced = True
            except StopIteration:
                break
        return expression, any_replaced


Subgraph = BipartiteGraph[Tuple[int, int], Tuple[int, int], Substitution]
Matching = Dict[Tuple[int, int], Tuple[int, int]]


class CommutativeMatcher(TypedModel):

    patterns: Dict[_PatternKey, '_PatternValue'] = field(default_factory=dict)
    subjects: Dict[Expression, Tuple[int, Set[int]]] = field(default_factory=dict)
    subjects_by_id: Dict[int, Expression] = field(default_factory=dict)
    automaton: Optional[ManyToOneMatcher] = None
    bipartite: BipartiteGraph[Tuple[int, int], Tuple[int, int], List[Substitution]] = field(default_factory=BipartiteGraph)
    associative: Optional[OperationHead] = None
    max_optional_count: int = 0
    anonymous_patterns: Set[int] = field(default_factory=set)
    # Dedicated tracking for optional wildcards - avoids mixing None into subjects dict
    optional_subject_id: int = -1  # -1 means not initialized
    optional_pattern_ids: Set[int] = field(default_factory=set)

    def __init__(self, associative: Optional[type] = None, **kwargs) -> None:
        super().__init__(associative=associative, **kwargs)
        # Only create fresh automaton/bipartite for new instances.
        # When reconstructing from serialized data, these fields are already
        # populated by TypedModel field initialization.
        if self.automaton is None:
            self.automaton = ManyToOneMatcher()
        if self.bipartite is None:
            self.bipartite = BipartiteGraph()

    def clear(self):
        """Removes all cached data."""
        if isinstance(self.subjects, dict):
            self.subjects.clear()
        else:
            self.subjects = {}
        if isinstance(self.subjects_by_id, dict):
            self.subjects_by_id.clear()
        else:
            self.subjects_by_id = {}
        self.optional_subject_id = -1
        self.optional_pattern_ids.clear()
        self.automaton.clear()
        self.bipartite.clear()

    def add_pattern(self, operands: Iterable[Expression], constraints) -> int:
        pattern_set, pattern_vars = self._extract_sequence_wildcards(operands, constraints)
        sorted_vars = tuple(sorted(pattern_vars.values(), key=lambda v: (v[0][0] or '', v[0][1], v[0][2], v[1])))
        sorted_subpatterns = tuple(sorted(pattern_set))
        pattern_key = _PatternKey(subpatterns=sorted_subpatterns, variables=sorted_vars)
        if pattern_key not in self.patterns:
            inserted_id = len(self.patterns)
            self.patterns[pattern_key] = _PatternValue(index=inserted_id, pattern_set=pattern_set, variables=sorted_vars)
        else:
            pv = self.patterns[pattern_key]
            inserted_id = pv.index if isinstance(pv, _PatternValue) else pv[0]
        return inserted_id

    def get_match_iter(self, subject):
        match_iter = _MatchIter(self.automaton, subject, self.associative)
        for _ in match_iter._match(self.automaton.root):
            for pattern_index in match_iter.patterns:
                substitution = Substitution(match_iter.substitution)
                yield pattern_index, substitution

    def add_subject(self, subject: Expression) -> int:
        # Handle the optional sentinel case separately
        if subject is None:
            if self.optional_subject_id == -1:
                # Allocate a unique ID for the optional slot (large negative to never clash)
                self.optional_subject_id = -1000000
                for pattern_index, substitution in self.get_match_iter(subject):
                    self.bipartite.setdefault((self.optional_subject_id, pattern_index), []).append(Substitution(substitution))
                    self.optional_pattern_ids.add(pattern_index)
            return self.optional_subject_id
        # Normal subject handling
        if subject not in self.subjects:
            subject_id, pattern_set = self.subjects[subject] = (len(self.subjects), set())
            self.subjects_by_id[subject_id] = subject
            for pattern_index, substitution in self.get_match_iter(subject):
                self.bipartite.setdefault((subject_id, pattern_index), []).append(Substitution(substitution))
                pattern_set.add(pattern_index)
        else:
            subject_id, _ = self.subjects[subject]
        return subject_id

    def match(self, subjects: Sequence[Expression], substitution: Substitution) -> Iterator[Tuple[int, Substitution]]:
        subject_ids = Multiset()
        pattern_ids = Multiset()
        if self.max_optional_count > 0 and self.optional_subject_id != -1:
            subject_ids.add(self.optional_subject_id)
            for _ in range(self.max_optional_count):
                pattern_ids.update(self.optional_pattern_ids)
        for subject in op_iter(subjects):
            subject_id, subject_pattern_ids = self.subjects[subject]
            subject_ids.add(subject_id)
            pattern_ids.update(subject_pattern_ids)
        for pv in self.patterns.values():
            if isinstance(pv, _PatternValue):
                pattern_index, pattern_set, pattern_vars = pv.index, pv.pattern_set, pv.variables
            else:
                pattern_index, pattern_set, pattern_vars = pv
            if pattern_set:
                if not pattern_set <= pattern_ids:
                    continue
                bipartite_match_iter = self._match_with_bipartite(subject_ids, pattern_set, substitution)
                for bipartite_substitution, matched_subjects in bipartite_match_iter:
                    ids = subject_ids - matched_subjects
                    remaining = Multiset(self.subjects_by_id[id] for id in ids if id != self.optional_subject_id)
                    if pattern_vars:
                        sequence_var_iter = self._match_sequence_variables(
                            remaining, pattern_vars, bipartite_substitution
                        )
                        for result_substitution in sequence_var_iter:
                            yield pattern_index, result_substitution
                    elif len(remaining) == 0:
                        yield pattern_index, bipartite_substitution
            elif pattern_vars:
                sequence_var_iter = self._match_sequence_variables(Multiset(op_iter(subjects)), pattern_vars, substitution)
                for variable_substitution in sequence_var_iter:
                    yield pattern_index, variable_substitution
            elif op_len(subjects) == 0:
                yield pattern_index, substitution

    def _extract_sequence_wildcards(self, operands: Iterable[Expression],
                                    constraints) -> Tuple[MultisetOfInt, Dict[str, Tuple[VariableWithCount, bool]]]:
        pattern_set = Multiset()
        pattern_vars = dict()
        opt_count = 0
        for operand in op_iter(operands):
            if isinstance(operand, Wildcard) and operand.default_value is not None:
                opt_count += 1
            if not self._is_sequence_wildcard(operand):
                actual_constraints = [c for c in constraints if contains_variables_from_set(operand, c.variables)]
                pattern = Pattern(operand, *actual_constraints)
                # O(1) lookup using pattern_to_index dict (was O(n) linear scan)
                index = self.automaton.pattern_to_index.get(pattern)
                if index is None:
                    vnames = set(e.variable_name for e in preorder_iter(pattern.expression) if hasattr(e, 'variable_name') and e.variable_name is not None)
                    renaming = {n: n for n in vnames}
                    index = self.automaton._internal_add(pattern, None, renaming)
                    if is_anonymous(pattern.expression):
                        self.anonymous_patterns.add(index)
                pattern_set.add(index)
            else:
                varname = operand.variable_name
                if varname is None:
                    if varname in pattern_vars:
                        (_, _, min_count, _), _ = pattern_vars[varname]
                    else:
                        min_count = 0
                    pattern_vars[varname] = (VariableWithCount(varname, 1, operand.min_count + min_count, None), False)
                else:
                    if varname in pattern_vars:
                        (_, count, _, _), wrap = pattern_vars[varname]
                    else:
                        count = 0
                        wrap = operand.fixed_size and self.associative
                    pattern_vars[varname] = (
                        VariableWithCount(varname, count + 1, operand.min_count, operand.default_value), wrap
                    )
        if opt_count > self.max_optional_count:
            self.max_optional_count = opt_count
        return pattern_set, pattern_vars

    def _is_sequence_wildcard(self, expression: Expression) -> bool:
        if isinstance(expression, Wildcard):
            return not expression.fixed_size or self.associative
        return False

    def _match_with_bipartite(
            self,
            subject_ids: MultisetOfInt,
            pattern_set: MultisetOfInt,
            substitution: Substitution,
    ) -> Iterator[Tuple[Substitution, MultisetOfInt]]:
        bipartite = self._build_bipartite(subject_ids, pattern_set)
        for matching in enum_maximum_matchings_iter(bipartite):
            if len(matching) < len(pattern_set):
                break
            if not self._is_canonical_matching(matching):
                continue
            # matched_subjects depends only on `matching`, not on `substs` -- hoisted
            # out of the Cartesian-product loop (it was rebuilt per product element).
            matched_subjects = Multiset(subexpression for subexpression, _ in matching)
            for substs in itertools.product(*(bipartite[edge] for edge in matching.items())):
                try:
                    bipartite_substitution = substitution.union(*substs)
                except ValueError:
                    continue
                yield bipartite_substitution, matched_subjects

    def _match_sequence_variables(
            self,
            subjects: MultisetOfExpression,
            pattern_vars: Sequence[VariableWithCount],
            substitution: Substitution,
    ) -> Iterator[Substitution]:
        only_counts = [info for info, _ in pattern_vars]
        wrapped_vars = [name for (name, _, _, _), wrap in pattern_vars if wrap and name]
        for variable_substitution in commutative_sequence_variable_partition_iter(subjects, only_counts):
            for var in wrapped_vars:
                operands = variable_substitution[var]
                if isinstance(operands, (tuple, list, Multiset)):
                    if len(operands) > 1:
                        variable_substitution[var] = self.associative(*operands)
                    else:
                        variable_substitution[var] = next(iter(operands))
            try:
                result_substitution = substitution.union(variable_substitution)
            except ValueError:
                continue
            yield result_substitution

    def _build_bipartite(self, subjects: MultisetOfInt, patterns: MultisetOfInt) -> Subgraph:
        bipartite = BipartiteGraph()
        n = 0
        m = 0
        p_states = {}
        # This is the hottest loop in commutative matching. It used to scan EVERY
        # edge of the subject and test `pattern in patterns` -- a Python-level
        # Multiset.__contains__ -- which on one Rubi integration meant ~51 MILLION
        # tests for ~27000 edges actually created: about 4100 candidates examined per
        # call to keep 2.
        #
        # The loop is therefore inverted. `patterns` (the patterns admissible at this
        # transition) is small, while a subject's edge set is large, so we iterate the
        # SMALL side and probe the large one: O(len(patterns)) C-level set lookups
        # instead of O(len(edges)) interpreted iterations. `_graph` stores the
        # neighbours as (RIGHT, pattern) tuples, hence the tagged probe.
        edges_of = self.bipartite._graph.get
        edge_values = self.bipartite._edges
        for subject, s_count in subjects.items():
            edges = edges_of((LEFT, subject))
            if edges is None:
                continue
            any_patterns = False
            for pattern, p_count in patterns.items():
                if (RIGHT, pattern) not in edges:
                    continue
                any_patterns = True
                subst = edge_values[subject, pattern]
                if pattern in p_states:
                    p_start = p_states[pattern]
                else:
                    p_start = p_states[pattern] = m
                    m += p_count
                for i in range(n, n + s_count):
                    for j in range(p_start, p_start + p_count):
                        bipartite[(subject, i), (pattern, j)] = subst
            if any_patterns:
                n += s_count

        return bipartite

    def _is_canonical_matching(self, matching: Matching) -> bool:
        anonymous_patterns = self.anonymous_patterns
        for (s1, n1), (p1, m1) in matching.items():
            for (s2, n2), (p2, m2) in matching.items():
                if p1 in anonymous_patterns and p2 in anonymous_patterns:
                    if n1 < n2 and m1 > m2:
                        return False
                elif s1 == s2 and n1 < n2 and m1 > m2:
                    return False
        return True

    def bipartite_as_graph(self) -> Graph:  # pragma: no cover
        """Returns a :class:`graphviz.Graph` representation of this bipartite graph."""
        if Graph is None:
            raise ImportError('The graphviz package is required to draw the graph.')
        graph = Graph()
        nodes_left = {}  # type: Dict[TLeft, str]
        nodes_right = {}  # type: Dict[TRight, str]
        node_id = 0
        for (left, right), value in self.bipartite._edges.items():
            if left not in nodes_left:
                name = 'node{:d}'.format(node_id)
                nodes_left[left] = name
                label = str(self.subjects_by_id[left])
                graph.node(name, label=label)
                node_id += 1
            if right not in nodes_right:
                name = 'node{:d}'.format(node_id)
                nodes_right[right] = name
                label = str(self.automaton.patterns[right][0])
                graph.node(name, label=label)
                node_id += 1
            edge_label = value is not True and str(value) or ''
            graph.edge(nodes_left[left], nodes_right[right], edge_label)
        return graph

    def concrete_bipartite_as_graph(self, subjects, patterns) -> Graph:  # pragma: no cover
        """Returns a :class:`graphviz.Graph` representation of this bipartite graph."""
        if Graph is None:
            raise ImportError('The graphviz package is required to draw the graph.')
        bipartite = self._build_bipartite(subjects, patterns)
        graph = Graph()
        nodes_left = {}  # type: Dict[TLeft, str]
        nodes_right = {}  # type: Dict[TRight, str]
        node_id = 0
        for (left, right), value in bipartite._edges.items():
            if left not in nodes_left:
                subject_id, i = left
                name = 'node{:d}'.format(node_id)
                nodes_left[left] = name
                label = '{}, {}'.format(i, self.subjects_by_id[subject_id])
                graph.node(name, label=label)
                node_id += 1
            if right not in nodes_right:
                pattern, i = right
                name = 'node{:d}'.format(node_id)
                nodes_right[right] = name
                label = '{}, {}'.format(i, self.automaton.patterns[pattern][0])
                graph.node(name, label=label)
                node_id += 1
            edge_label = value is not True and str(value) or ''
            graph.edge(nodes_left[left], nodes_right[right], edge_label)
        return graph


class SecondaryAutomaton():  # pragma: no cover
    # TODO: Decide whether to integrate this
    def __init__(self, k):
        self.k = k
        self.states = self._build(k)

    def match(self, edges):
        raise NotImplementedError

    @staticmethod
    def _build(k):
        states = dict()
        queue = [frozenset([0])]

        while queue:
            state_id = queue.pop(0)
            state = states[state_id] = dict()
            for i in range(1, 2**k):
                new_state = set()
                for t in [2**j for j in range(k) if i & 2**j]:
                    for v in state_id:
                        new_state.add(t | v)
                new_state = frozenset(new_state - state_id)
                if new_state:
                    if new_state != state_id:
                        state[i] = new_state
                    if new_state not in states and new_state not in queue:
                        queue.append(new_state)

        keys = sorted(states.keys())
        new_states = []

        for state in keys:
            new_states.append(states[state])

        for i, state in enumerate(new_states):
            new_state = dict()
            for key, value in state.items():
                new_state[key] = keys.index(value)
            new_states[i] = new_state

        return new_states

    def as_graph(self):
        if Digraph is None:
            raise ImportError('The graphviz package is required to draw the graph.')
        graph = Digraph()
        for i in range(len(self.states)):
            graph.node(str(i), str(i))

        for state, edges in enumerate(self.states):
            for target, labels in itertools.groupby(sorted(edges.items()), key=itemgetter(1)):
                label = '\n'.join(bin(l)[2:].zfill(self.k) for l, _ in labels)
                graph.edge(str(state), str(target), label)

        return graph
