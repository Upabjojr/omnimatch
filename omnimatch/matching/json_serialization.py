# -*- coding: utf-8 -*-
"""JSON serialization/deserialization for ManyToOneMatcher.

Provides to_json() and from_json() for complete roundtrip serialization
of the ManyToOneMatcher automaton to JSON format.

Serialization uses functools.singledispatch to dispatch on object type.
Deserialization uses registry dicts keyed on type-tag strings.
"""
import json
from functools import singledispatch
from typing import Any, Dict, List, Optional, Set, Tuple

from multiset import Multiset

from ..expressions.expressions import (
    Expression, Operation, OperationHead, NamedAtom, SymbolWrapper, Wildcard, Pattern, Arity
)
from ..expressions.constraints import Constraint, EqualVariablesConstraint, CustomConstraint
from ..expressions.substitution import Substitution
from ..utils import VariableWithCount



@singledispatch
def serialize_wrapped_value(val):
    """Serialize a value wrapped in SymbolWrapper. Extensible via singledispatch.

    Extensions (e.g. sympy_matching) register handlers for their types.
    The default fallback stores a repr string.
    """
    if val is None:
        return None
    return {'_val_type': 'repr', 'repr': repr(val)}

@serialize_wrapped_value.register(int)
def _ser_wrapped_int(val):
    return {'_val_type': 'int', 'value': val}

@serialize_wrapped_value.register(float)
def _ser_wrapped_float(val):
    return {'_val_type': 'float', 'value': val}

@serialize_wrapped_value.register(str)
def _ser_wrapped_str(val):
    return {'_val_type': 'str', 'value': val}


# Registry for wrapped value deserialization (keyed on '_val_type' tag)
_WRAPPED_VALUE_DESERIALIZERS = {
    'int': lambda data: data['value'],
    'float': lambda data: data['value'],
    'str': lambda data: data['value'],
}


def register_wrapped_value_deserializer(tag: str, fn=None):
    """PUBLIC extension point: register ``fn(data) -> value`` for ``_val_type == tag``.

    The deserialization counterpart of the ``serialize_wrapped_value`` singledispatch
    (dispatch is on the JSON ``_val_type`` tag rather than a Python type, so a plain
    tag registry replaces singledispatch here). Usable as a decorator::

        @register_wrapped_value_deserializer('sympy')
        def _deser(data): ...

    External libraries (e.g. ``sympy_matching.json_ext``) use this instead of
    reaching into the private registry dict.
    """
    if fn is None:
        def _decorator(f):
            _WRAPPED_VALUE_DESERIALIZERS[tag] = f
            return f
        return _decorator
    _WRAPPED_VALUE_DESERIALIZERS[tag] = fn
    return fn


def deserialize_wrapped_value(data):
    """Deserialize a wrapped value from JSON dict.

    Extensible via :func:`register_wrapped_value_deserializer`."""
    if data is None:
        return None
    val_type = data.get('_val_type')
    if val_type is None:
        return None
    handler = _WRAPPED_VALUE_DESERIALIZERS.get(val_type)
    if handler:
        return handler(data)
    # 'repr' fallback: return as string (lossy, but won't crash)
    return data.get('repr', str(data))


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION — singledispatch on object type
# ══════════════════════════════════════════════════════════════════════════════


# ── Expression serialization ─────────────────────────────────────────────────

@singledispatch
def _serialize_expression(expr) -> Optional[dict]:
    """Convert an Expression to a JSON-safe dict. Dispatches on type."""
    if expr is None:
        return None
    # Fallback for unknown expression types
    return {
        '_expr_type': 'NamedAtom',
        'name': str(expr),
        'variable_name': getattr(expr, 'variable_name', None),
    }

@_serialize_expression.register(type(None))
def _serialize_none_expr(expr):
    return None

@_serialize_expression.register(Operation)
def _serialize_operation_expr(expr):
    return {
        '_expr_type': 'Operation',
        'head': _serialize_operation_head(expr.head),
        'operands': [_serialize_expression(op) for op in expr.operands],
        'variable_name': expr.variable_name,
    }


@_serialize_expression.register(Wildcard)
def _serialize_wildcard_expr(expr):
    return {
        '_expr_type': 'Wildcard',
        'min_count': expr.min_count,
        'fixed_size': expr.fixed_size,
        'variable_name': expr.variable_name,
        'default_value': _serialize_expression(expr.default_value) if isinstance(expr.default_value, Expression) else None,
    }

@_serialize_expression.register(SymbolWrapper)
def _serialize_symbol_wrapper_expr(expr):
    return {
        '_expr_type': 'SymbolWrapper',
        'value': serialize_wrapped_value(expr.value),
        'variable_name': expr.variable_name,
    }

@_serialize_expression.register(NamedAtom)
def _serialize_symbol_expr(expr):
    return {
        '_expr_type': 'NamedAtom',
        'name': expr.name,
        'variable_name': expr.variable_name,
    }


# ── Substitution value serialization ─────────────────────────────────────────

@singledispatch
def _serialize_subst_value(val):
    """Serialize a substitution value. Dispatches on type."""
    if val is None:
        return None
    # Fallback: treat as expression
    return {'_subst_type': 'expression', 'value': _serialize_expression(val)}

@_serialize_subst_value.register(type(None))
def _serialize_subst_none(val):
    return None

@_serialize_subst_value.register(tuple)
def _serialize_subst_tuple(val):
    return {'_subst_type': 'tuple', 'items': [_serialize_subst_value(v) for v in val]}

@_serialize_subst_value.register(Multiset)
def _serialize_subst_multiset(val):
    return {'_subst_type': 'multiset', 'items': [_serialize_subst_value(v) for v in val]}

@_serialize_subst_value.register(Expression)
def _serialize_subst_expression(val):
    return {'_subst_type': 'expression', 'value': _serialize_expression(val)}


# ── OperationHead serialization ──────────────────────────────────────────────

def _serialize_operation_head(head: OperationHead) -> Optional[dict]:
    """Serialize OperationHead to dict."""
    if head is None:
        return None
    return {
        'name': head.name,
        'arity': head.arity.name,
        'commutative': head.commutative,
        'associative': head.associative,
        'one_identity': head.one_identity,
        'infix': head.infix,
    }


# ── Constraint serialization ─────────────────────────────────────────────────

@singledispatch
def _serialize_constraint(c) -> dict:
    """Serialize a Constraint to dict. Dispatches on type."""
    return {'_constraint_type': 'Unknown', 'variables': sorted(c.variables) if hasattr(c, 'variables') else []}

@_serialize_constraint.register(EqualVariablesConstraint)
def _serialize_equal_vars_constraint(c):
    return {'_constraint_type': 'EqualVariablesConstraint', 'variables': sorted(c.variables)}

@_serialize_constraint.register(CustomConstraint)
def _serialize_custom_constraint(c):
    return {
        '_constraint_type': 'CustomConstraint',
        'variables': sorted(c.variables),
        '_callable_name': getattr(c.constraint, '__name__', '<lambda>'),
    }


# ── LabelType serialization ─────────────────────────────────────────────────

@singledispatch
def _serialize_label(label) -> dict:
    """Serialize a LabelType to dict. Dispatches on type."""
    from .many_to_one import LabelTypeEpsilon
    return {'_label_type': 'LabelTypeEpsilon'}

def _register_label_serializers():
    """Register label serializers (deferred to avoid circular imports)."""
    from .many_to_one import LabelTypeExpression, LabelTypeOperation, LabelTypeEpsilon, LabelTypeEnd

    @_serialize_label.register(LabelTypeEnd)
    def _ser_label_end(label):
        return {'_label_type': 'LabelTypeEnd'}

    @_serialize_label.register(LabelTypeEpsilon)
    def _ser_label_eps(label):
        return {'_label_type': 'LabelTypeEpsilon'}

    @_serialize_label.register(LabelTypeOperation)
    def _ser_label_op(label):
        return {'_label_type': 'LabelTypeOperation', 'value': _serialize_operation_head(label.value)}

    @_serialize_label.register(LabelTypeExpression)
    def _ser_label_expr(label):
        return {'_label_type': 'LabelTypeExpression', 'value': _serialize_expression(label.value)}


# ── TransitionKey serialization ──────────────────────────────────────────────

@singledispatch
def _serialize_transition_key(key) -> dict:
    """Serialize a TransitionKey to dict. Dispatches on type."""
    return {'_key_type': 'HeadTypeNone'}

def _register_transition_key_serializers():
    """Register transition key serializers (deferred to avoid circular imports)."""
    from .many_to_one import (
        TransitionKeyEnd, TransitionKeyPatternId,
        HeadTypeExpression, HeadTypeOperation, HeadTypeNone,
    )

    @_serialize_transition_key.register(TransitionKeyEnd)
    def _ser_key_end(key):
        return {'_key_type': 'TransitionKeyEnd'}

    @_serialize_transition_key.register(TransitionKeyPatternId)
    def _ser_key_pattern_id(key):
        return {'_key_type': 'TransitionKeyPatternId', 'value': key.value}

    @_serialize_transition_key.register(HeadTypeOperation)
    def _ser_key_head_op(key):
        return {'_key_type': 'HeadTypeOperation', 'value': _serialize_operation_head(key.value)}

    @_serialize_transition_key.register(HeadTypeExpression)
    def _ser_key_head_expr(key):
        return {'_key_type': 'HeadTypeExpression', 'value': _serialize_expression(key.value)}


    @_serialize_transition_key.register(HeadTypeNone)
    def _ser_key_head_none(key):
        return {'_key_type': 'HeadTypeNone'}


# ── VariableWithCount / pattern variables ────────────────────────────────────

def _serialize_variable_with_count(vwc) -> dict:
    """Serialize a VariableWithCount to dict."""
    default = None
    if vwc.default is not None and isinstance(vwc.default, Expression):
        default = _serialize_expression(vwc.default)
    return {'name': vwc.name, 'count': vwc.count, 'minimum': vwc.minimum, 'default': default}


def _serialize_pattern_variables(variables) -> list:
    """Serialize pattern variables: Tuple[Tuple[VariableWithCount, bool|OperationHead], ...]"""
    result = []
    for var_tuple in variables:
        vwc = var_tuple[0]
        flag = var_tuple[1]
        if isinstance(flag, bool):
            flag_data = {'_flag_type': 'bool', 'value': flag}
        else:
            flag_data = {'_flag_type': 'OperationHead', 'value': _serialize_operation_head(flag)}
        result.append([_serialize_variable_with_count(vwc), flag_data])
    return result


# ── State/Transition serialization ───────────────────────────────────────────

def _serialize_state(state) -> dict:
    """Serialize _State to dict (handles recursive structure)."""
    transitions = []
    for key, trans_list in state.transitions.items():
        transitions.append([
            _serialize_transition_key(key),
            [_serialize_transition(t) for t in trans_list]
        ])
    return {
        'number': state.number,
        'transitions': transitions,
        'matcher': _serialize_commutative_matcher(state.matcher) if state.matcher is not None else None,
    }


def _serialize_transition(t) -> dict:
    """Serialize _Transition to dict."""
    subst = None
    if t.subst is not None:
        subst = {k: _serialize_subst_value(v) for k, v in t.subst.items()}
    return {
        'label': _serialize_label(t.label),
        'target': _serialize_state(t.target),
        'variable_name': t.variable_name,
        'patterns': list(t.patterns),
        'check_constraints': list(t.check_constraints) if t.check_constraints is not None else None,
        'subst': subst,
    }


# ── CommutativeMatcher serialization ─────────────────────────────────────────

def _serialize_commutative_matcher(cm) -> Optional[dict]:
    """Serialize CommutativeMatcher to dict."""
    if cm is None:
        return None
    patterns = []
    for pk, pv in cm.patterns.items():
        patterns.append({
            'subpatterns': list(pk.subpatterns),
            'pk_variables': _serialize_pattern_variables(pk.variables),
            'index': pv.index,
            'pattern_set': {k: v for k, v in pv.pattern_set.items()},
            'pv_variables': _serialize_pattern_variables(pv.variables),
        })
    return {
        'automaton': serialize_matcher(cm.automaton),
        'patterns': patterns,
        'associative': _serialize_operation_head(cm.associative) if cm.associative else None,
        'max_optional_count': cm.max_optional_count,
        'anonymous_patterns': list(cm.anonymous_patterns),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DESERIALIZATION — registry dicts keyed on type-tag strings
# ══════════════════════════════════════════════════════════════════════════════


# ── Expression deserialization ────────────────────────────────────────────────

def _deser_operation(data):
    head = _deserialize_operation_head(data['head'])
    operands = [_deserialize_expression(op) for op in data['operands']]
    return Operation(head, *operands, variable_name=data.get('variable_name'))


def _deser_wildcard(data):
    default_value = _deserialize_expression(data.get('default_value'))
    return Wildcard(
        min_count=data['min_count'],
        fixed_size=data['fixed_size'],
        variable_name=data.get('variable_name'),
        default_value=default_value,
    )

def _deser_symbol_wrapper(data):
    val = deserialize_wrapped_value(data.get('value'))
    variable_name = data.get('variable_name')
    if val is not None:
        return SymbolWrapper(val, variable_name=variable_name)
    # Legacy fallback (old format with just 'name')
    return NamedAtom(data.get('name', ''), variable_name=variable_name)

def _deser_symbol(data):
    return NamedAtom(data['name'], variable_name=data.get('variable_name'))

_EXPRESSION_DESERIALIZERS = {
    'Operation': _deser_operation,
    'Wildcard': _deser_wildcard,
    'SymbolWrapper': _deser_symbol_wrapper,
    'NamedAtom': _deser_symbol,
}


def register_expression_deserializer(tag: str, fn=None):
    """PUBLIC extension point: register ``fn(data) -> Expression`` for ``_expr_type == tag``.

    Counterpart of the ``_serialize_expression`` singledispatch for the JSON side
    (dispatch is on the ``_expr_type`` tag). Usable as a decorator.
    """
    if fn is None:
        def _decorator(f):
            _EXPRESSION_DESERIALIZERS[tag] = f
            return f
        return _decorator
    _EXPRESSION_DESERIALIZERS[tag] = fn
    return fn

def _deserialize_expression(data) -> Optional[Expression]:
    """Reconstruct an Expression from a JSON-safe dict."""
    if data is None:
        return None
    expr_type = data.get('_expr_type', 'NamedAtom')
    handler = _EXPRESSION_DESERIALIZERS.get(expr_type, _deser_symbol)
    return handler(data)


# ── Substitution value deserialization ────────────────────────────────────────

def _deser_subst_tuple(data):
    return tuple(_deserialize_subst_value(v) for v in data.get('items', []))

def _deser_subst_multiset(data):
    return Multiset(_deserialize_subst_value(v) for v in data.get('items', []))

def _deser_subst_expression(data):
    return _deserialize_expression(data.get('value'))

_SUBST_VALUE_DESERIALIZERS = {
    'tuple': _deser_subst_tuple,
    'multiset': _deser_subst_multiset,
    'expression': _deser_subst_expression,
}

def _deserialize_subst_value(data):
    """Deserialize a substitution value."""
    if data is None:
        return None
    st = data.get('_subst_type', 'expression')
    handler = _SUBST_VALUE_DESERIALIZERS.get(st, _deser_subst_expression)
    return handler(data)


# ── OperationHead deserialization ────────────────────────────────────────────

def _deserialize_operation_head(data) -> Optional[OperationHead]:
    """Deserialize OperationHead from dict."""
    if data is None:
        return None
    return OperationHead(
        name=data['name'],
        arity=Arity[data['arity']],
        commutative=data.get('commutative', False),
        associative=data.get('associative', False),
        one_identity=data.get('one_identity', False),
        infix=data.get('infix', False),
    )


# ── Constraint deserialization ────────────────────────────────────────────────

def _deser_equal_vars_constraint(data, constraint_lookup, index):
    return EqualVariablesConstraint(*data['variables'])

def _deser_custom_constraint(data, constraint_lookup, index):
    if constraint_lookup and index < len(constraint_lookup):
        return constraint_lookup[index]
    return CustomConstraint(lambda: True)

def _deser_unknown_constraint(data, constraint_lookup, index):
    return EqualVariablesConstraint()

_CONSTRAINT_DESERIALIZERS = {
    'EqualVariablesConstraint': _deser_equal_vars_constraint,
    'CustomConstraint': _deser_custom_constraint,
    'Unknown': _deser_unknown_constraint,
}

def _deserialize_constraint(data: dict, constraint_lookup: Optional[List] = None, index: int = 0) -> Constraint:
    """Deserialize a Constraint from dict."""
    ctype = data.get('_constraint_type', 'Unknown')
    handler = _CONSTRAINT_DESERIALIZERS.get(ctype, _deser_unknown_constraint)
    return handler(data, constraint_lookup, index)


# ── LabelType deserialization ────────────────────────────────────────────────

def _deser_label_end(data):
    from .many_to_one import _LABEL_END
    return _LABEL_END

def _deser_label_eps(data):
    from .many_to_one import _EPS
    return _EPS

def _deser_label_operation(data):
    from .many_to_one import LabelTypeOperation
    return LabelTypeOperation(value=_deserialize_operation_head(data['value']))

def _deser_label_expression(data):
    from .many_to_one import LabelTypeExpression
    return LabelTypeExpression(value=_deserialize_expression(data['value']))

_LABEL_DESERIALIZERS = {
    'LabelTypeEnd': _deser_label_end,
    'LabelTypeEpsilon': _deser_label_eps,
    'LabelTypeOperation': _deser_label_operation,
    'LabelTypeExpression': _deser_label_expression,
}

def _deserialize_label(data: dict):
    """Deserialize a LabelType from dict."""
    ltype = data.get('_label_type', 'LabelTypeEpsilon')
    handler = _LABEL_DESERIALIZERS.get(ltype, _deser_label_eps)
    return handler(data)


# ── TransitionKey deserialization ────────────────────────────────────────────

def _deser_key_end(data):
    from .many_to_one import _TRANSITION_END
    return _TRANSITION_END

def _deser_key_pattern_id(data):
    from .many_to_one import TransitionKeyPatternId
    return TransitionKeyPatternId(value=data['value'])

def _deser_key_head_operation(data):
    from .many_to_one import HeadTypeOperation
    return HeadTypeOperation(value=_deserialize_operation_head(data['value']))

def _deser_key_head_expression(data):
    from .many_to_one import HeadTypeExpression
    return HeadTypeExpression(value=_deserialize_expression(data['value']))


def _deser_key_head_none(data):
    from .many_to_one import _HEAD_NONE
    return _HEAD_NONE

_TRANSITION_KEY_DESERIALIZERS = {
    'TransitionKeyEnd': _deser_key_end,
    'TransitionKeyPatternId': _deser_key_pattern_id,
    'HeadTypeOperation': _deser_key_head_operation,
    'HeadTypeExpression': _deser_key_head_expression,
    'HeadTypeNone': _deser_key_head_none,
}

def _deserialize_transition_key(data: dict):
    """Deserialize a TransitionKey from dict."""
    ktype = data.get('_key_type', 'HeadTypeNone')
    handler = _TRANSITION_KEY_DESERIALIZERS.get(ktype, _deser_key_head_none)
    return handler(data)


# ── VariableWithCount / pattern variables deserialization ─────────────────────

def _deserialize_variable_with_count(data: dict):
    """Deserialize a VariableWithCount from dict."""
    default = _deserialize_expression(data.get('default')) if data.get('default') else None
    return VariableWithCount(name=data.get('name'), count=data.get('count', 0),
                             minimum=data.get('minimum', 0), default=default)


def _deserialize_pattern_variables(data: list) -> tuple:
    """Deserialize pattern variables from list."""
    result = []
    for vwc_data, flag_data in data:
        vwc = _deserialize_variable_with_count(vwc_data)
        if flag_data.get('_flag_type') == 'OperationHead':
            flag = _deserialize_operation_head(flag_data['value'])
        else:
            flag = flag_data.get('value', False)
        result.append((vwc, flag))
    return tuple(result)


# ── State/Transition deserialization ─────────────────────────────────────────

def _deserialize_state(data: dict, states_cache: dict, commutative_matchers: list):
    """Deserialize _State from dict with caching for shared references."""
    from .many_to_one import _State
    number = data['number']
    if number in states_cache:
        return states_cache[number]
    state = _State(number=number, transitions={}, matcher=None)
    states_cache[number] = state
    for key_data, trans_list_data in data['transitions']:
        key = _deserialize_transition_key(key_data)
        transitions = [_deserialize_transition(td, states_cache, commutative_matchers) for td in trans_list_data]
        state.transitions[key] = transitions
    if data.get('matcher') is not None:
        cm = _deserialize_commutative_matcher(data['matcher'])
        state.matcher = cm
        commutative_matchers.append(cm)
    return state


def _deserialize_transition(data: dict, states_cache: dict, commutative_matchers: list):
    """Deserialize _Transition from dict."""
    from .many_to_one import _Transition
    label = _deserialize_label(data['label'])
    target = _deserialize_state(data['target'], states_cache, commutative_matchers)
    subst = None
    if data.get('subst') is not None:
        subst = Substitution({k: _deserialize_subst_value(v) for k, v in data['subst'].items()})
    return _Transition(
        label=label,
        target=target,
        variable_name=data.get('variable_name'),
        patterns=set(data.get('patterns', [])),
        check_constraints=set(data['check_constraints']) if data.get('check_constraints') is not None else None,
        subst=subst,
    )


# ── CommutativeMatcher deserialization ────────────────────────────────────────

def _deserialize_commutative_matcher(data: dict):
    """Deserialize CommutativeMatcher from dict."""
    from .many_to_one import CommutativeMatcher, _PatternKey, _PatternValue
    assoc = _deserialize_operation_head(data['associative']) if data.get('associative') else None
    cm = CommutativeMatcher(associative=assoc)
    cm.automaton = deserialize_matcher(data['automaton'])
    cm.patterns = {}
    for p_data in data.get('patterns', []):
        subpatterns = tuple(p_data['subpatterns'])
        pk_variables = _deserialize_pattern_variables(p_data.get('pk_variables', []))
        pk = _PatternKey(subpatterns=subpatterns, variables=pk_variables)
        # JSON converts int dict keys to strings — convert back
        pattern_set = Multiset({int(k): v for k, v in p_data.get('pattern_set', {}).items()})
        pv_variables = _deserialize_pattern_variables(p_data.get('pv_variables', []))
        pv = _PatternValue(index=p_data['index'], pattern_set=pattern_set, variables=pv_variables)
        cm.patterns[pk] = pv
    cm.max_optional_count = data.get('max_optional_count', 0)
    cm.anonymous_patterns = set(data.get('anonymous_patterns', []))
    return cm


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL ManyToOneMatcher serialization / deserialization
# ══════════════════════════════════════════════════════════════════════════════

def serialize_matcher(matcher) -> dict:
    """Serialize ManyToOneMatcher to a JSON-safe dict."""
    patterns = []
    for pattern, label, constraint_indices in matcher.patterns:
        pat_data = {
            'expression': _serialize_expression(pattern.expression),
            'constraints': [_serialize_constraint(c) for c in pattern.constraints],
        }
        if label is None:
            label_data = {'_kind': 'none'}
        elif isinstance(label, Pattern):
            label_data = {'_kind': 'pattern', 'expression': _serialize_expression(label.expression),
                          'constraints': [_serialize_constraint(c) for c in label.constraints]}
        elif isinstance(label, str):
            label_data = {'_kind': 'string', 'value': label}
        elif callable(label) and (getattr(label, '_rubi_replacement_expr', None) is not None
                                  or (hasattr(label, '__closure__') and label.__closure__)):
            # Replacement function from ManyToOneReplacer. Prefer the explicit
            # attribute set by the (tracing) replacement factory; fall back to the
            # closure for plain replacement functions.
            replacement_expr = getattr(label, '_rubi_replacement_expr', None)
            if replacement_expr is None:
                replacement_expr = label.__closure__[0].cell_contents
            label_data = {'_kind': 'replacement_fn',
                          'replacement_expr': serialize_wrapped_value(replacement_expr)}
        else:
            label_data = {'_kind': 'pattern', 'expression': _serialize_expression(label.expression),
                          'constraints': [_serialize_constraint(c) for c in label.constraints]}
        patterns.append([pat_data, label_data, constraint_indices])

    constraints = []
    for c, pat_set in matcher.constraints:
        constraints.append([_serialize_constraint(c), list(pat_set)])

    return {
        'patterns': patterns,
        'root': _serialize_state(matcher.root) if matcher.root else None,
        'pattern_vars': matcher.pattern_vars,
        'constraints': constraints,
        'constraint_vars': {k: list(v) for k, v in matcher.constraint_vars.items()},
        'finals': list(matcher.finals),
        'rename': matcher.rename,
    }


def _rebuild_replacement_fn(replacement_expr):
    """Rebuild a replacement function from a deserialized SymPy expression."""
    def _replacement(**match_dict):
        from sympy_matching.conversion import omnimatch_to_sympy, to_omnimatch_expression
        from sympy_matching.wild import WildSymbol
        sympy_subs = {}
        for name, omnimatch_val in match_dict.items():
            sympy_subs[name] = omnimatch_to_sympy(omnimatch_val)
        result = replacement_expr
        for atom in replacement_expr.atoms():
            if isinstance(atom, WildSymbol) and atom.wildcard_name in sympy_subs:
                result = result.subs(atom, sympy_subs[atom.wildcard_name])
        return to_omnimatch_expression(result)
    return _replacement


def deserialize_matcher(data: dict, original_constraints: Optional[List] = None):
    """Deserialize ManyToOneMatcher from a JSON-safe dict."""
    from .many_to_one import ManyToOneMatcher

    matcher = ManyToOneMatcher(rename=data.get('rename', True))

    # Collect commutative matchers during state deserialization so that
    # state.matcher IS the same object as in matcher.commutative_matchers
    commutative_matchers = []
    states_cache = {}
    if data.get('root'):
        matcher.root = _deserialize_state(data['root'], states_cache, commutative_matchers)
    matcher.states = list(states_cache.values())
    matcher.commutative_matchers = commutative_matchers

    matcher.patterns = []
    for pat_data, label_data, constraint_indices in data['patterns']:
        expr = _deserialize_expression(pat_data['expression'])
        pat_constraints = tuple(
            _deserialize_constraint(cd, original_constraints, i)
            for i, cd in enumerate(pat_data.get('constraints', []))
        )
        pattern = Pattern(expr, *pat_constraints)
        if label_data.get('_kind') == 'none':
            label = None
        elif label_data.get('_kind') == 'string':
            label = label_data['value']
        elif label_data.get('_kind') == 'replacement_fn':
            # Rebuild the replacement function from the stored SymPy expression
            replacement_sympy = deserialize_wrapped_value(label_data['replacement_expr'])
            label = _rebuild_replacement_fn(replacement_sympy)
        else:
            label_expr = _deserialize_expression(label_data['expression'])
            label_constraints = tuple(
                _deserialize_constraint(cd, original_constraints, i)
                for i, cd in enumerate(label_data.get('constraints', []))
            )
            label = Pattern(label_expr, *label_constraints)
        matcher.patterns.append((pattern, label, constraint_indices))

    matcher.pattern_vars = data.get('pattern_vars', [])

    matcher.constraints = []
    for c_data, pat_set in data.get('constraints', []):
        idx = len(matcher.constraints)
        c = _deserialize_constraint(c_data, original_constraints, idx)
        matcher.constraints.append((c, set(pat_set)))

    matcher.constraint_vars = {k: set(v) for k, v in data.get('constraint_vars', {}).items()}
    matcher.finals = set(data.get('finals', []))

    return matcher


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def to_json(matcher) -> str:
    """Serialize a ManyToOneMatcher to a JSON string."""
    # Ensure singledispatch registrations are done (deferred for circular imports)
    _ensure_registered()
    return json.dumps(serialize_matcher(matcher), separators=(',', ':'))


def from_json(json_str: str, original_constraints=None):
    """Deserialize a ManyToOneMatcher from a JSON string."""
    return deserialize_matcher(json.loads(json_str), original_constraints)


# ── Deferred registration (avoids circular imports) ──────────────────────────

_registered = False

def _ensure_registered():
    """Register singledispatch handlers that depend on many_to_one types."""
    global _registered
    if _registered:
        return
    _registered = True
    _register_label_serializers()
    _register_transition_key_serializers()
