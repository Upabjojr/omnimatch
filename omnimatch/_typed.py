# -*- coding: utf-8 -*-
"""Lightweight typed base replacing Pydantic ``BaseModel``.

`TypedModel` declares fields via class annotations (like a dataclass), assigns
them in ``__init__(**kwargs)``, applies defaults, and enforces a **shallow**
``isinstance`` type-check per field at construction time — Python does not
otherwise check that an assigned value matches the field's declared type.

The check is deliberately shallow (outer type only; e.g. ``List[Expression]`` is
checked as ``isinstance(value, list)``, not element-by-element) and is compiled
once per field, so it stays cheap on the matcher's hot construction path. This is
a fraction of Pydantic's per-instance schema validation cost while still catching
"wrong type assigned to field".

Use ``field(default_factory=...)`` for mutable defaults (list/dict/set), exactly
like ``dataclasses.field`` / Pydantic ``Field``.
"""
import typing

_MISSING = object()
_NoneType = type(None)


class _FieldSpec:
    __slots__ = ('default', 'default_factory')

    def __init__(self, default=_MISSING, default_factory=None):
        self.default = default
        self.default_factory = default_factory


def field(*, default=_MISSING, default_factory=None):
    """Declare a field default (mirrors ``dataclasses.field`` / Pydantic ``Field``)."""
    return _FieldSpec(default, default_factory)


def _make_checker(typ):
    """Compile ``(type_name, predicate)`` for a field annotation, or ``None`` to skip.

    Returns ``None`` for ``Any``/``object``, forward refs, ``TypeVar``, and unions
    that include a non-class member (checked leniently). Generic aliases are
    reduced to their origin (``List[X]`` -> ``list``).
    """
    if typ is None or typ is typing.Any or typ is object:
        return None
    if type(typ) is type:
        return (typ.__name__, lambda v, t=typ: isinstance(v, t))
    origin = typing.get_origin(typ)
    if origin is None:
        return None  # ForwardRef / TypeVar / special form -> no check
    if origin is typing.Union:
        args = typing.get_args(typ)
        if any(a is not _NoneType and type(a) is not type for a in args):
            return None  # a member is a forward ref etc. -> be lenient
        concrete = tuple(a for a in args if a is not _NoneType)
        has_none = _NoneType in args
        name = ' | '.join(a.__name__ for a in concrete) + (' | None' if has_none else '')

        def _check_union(v, concrete=concrete, has_none=has_none):
            if v is None:
                return has_none
            return isinstance(v, concrete)

        return (name, _check_union)
    if type(origin) is type:
        return (getattr(origin, '__name__', str(origin)), lambda v, o=origin: isinstance(v, o))
    return None


def _resolve_fields(cls):
    """(name -> (_FieldSpec, checker)) for ``cls`` across its MRO; cached on the class."""
    cached = cls.__dict__.get('__typed_fields__')
    if cached is not None:
        return cached
    fields = {}
    for klass in reversed(cls.__mro__):  # base-to-derived so subclasses override
        for name, typ in klass.__dict__.get('__annotations__', {}).items():
            if name.startswith('_'):  # private attrs are not fields
                continue
            if typ is typing.ClassVar or typing.get_origin(typ) is typing.ClassVar:
                # ClassVar (bare `ClassVar` has no typing origin, `ClassVar[T]`
                # does). A subclass may redefine an inherited field as a ClassVar
                # (a fixed class-level value); it then stops being an instance field.
                fields.pop(name, None)
                continue
            raw_default = klass.__dict__.get(name, _MISSING)
            if isinstance(raw_default, _FieldSpec):
                spec = raw_default
            elif raw_default is _MISSING:
                spec = _FieldSpec()
            else:
                spec = _FieldSpec(default=raw_default)
            fields[name] = (spec, _make_checker(typ))
    cls.__typed_fields__ = fields
    return fields


def _compile_setter(cls):
    """Code-generate a specialized ``(self, kwargs) -> None`` initializer for ``cls``.

    Semantically identical to the old generic ``TypedModel.__init__`` loop, but with
    the per-field work (name lookup, default/factory, type-check, setattr) unrolled
    into straight-line code -- the same trick ``dataclasses`` uses. This matters
    because every omnimatch ``Expression`` node is a TypedModel: building the Rubi
    ManyToOneMatcher creates ~3.1 million nodes, and the generic loop's field-dict
    iteration + spec unpacking was one of the top construction hotspots.

    Cached on the class as ``__typed_setter__`` (checked via ``cls.__dict__`` so a
    subclass never inherits a base's setter, which would miss its extra fields).
    """
    fields = _resolve_fields(cls)
    env = {'_MISSING': _MISSING, '_setattr': object.__setattr__}
    lines = ['def __typed_setter__(self, kwargs):']
    if not fields:
        lines.append('    pass')
    for i, (name, (spec, checker)) in enumerate(fields.items()):
        get = f"    value = kwargs.pop({name!r}, _MISSING)"
        lines.append(get)
        if spec.default_factory is not None:
            env[f'_fac{i}'] = spec.default_factory
            lines.append(f"    if value is _MISSING: value = _fac{i}()")
        elif spec.default is not _MISSING:
            env[f'_def{i}'] = spec.default
            lines.append(f"    if value is _MISSING: value = _def{i}")
        else:
            lines.append(f"    if value is _MISSING:")
            lines.append(f"        raise TypeError({cls.__name__ + ': missing required argument ' + repr(name)!r})")
        if checker is not None:
            env[f'_chk{i}'] = checker[1]
            msg = "{}.{} expected {}, got {{}}".format(cls.__name__, name, checker[0])
            lines.append(f"    if not _chk{i}(value):")
            lines.append(f"        raise TypeError({msg!r}.format(type(value).__name__))")
        lines.append(f"    _setattr(self, {name!r}, value)")
    # Leftover kwargs: same lenient semantics as before -- ignore names that exist as
    # class-level attributes (a field a subclass turned into a ClassVar, still passed
    # by a base __init__); error only on genuinely unknown names.
    lines.append('    if kwargs:')
    lines.append('        unexpected = [k for k in kwargs if not hasattr(type(self), k)]')
    lines.append('        if unexpected:')
    lines.append(f"            raise TypeError('{cls.__name__}: unexpected keyword arguments ' + repr(unexpected))")
    exec('\n'.join(lines), env)
    setter = env['__typed_setter__']
    cls.__typed_setter__ = setter
    return setter


class TypedModel:
    """Base for annotation-declared, type-checked value objects (replaces BaseModel)."""

    def __init__(self, **kwargs):
        cls = type(self)
        setter = cls.__dict__.get('__typed_setter__')
        if setter is None:
            setter = _compile_setter(cls)
        setter(self, kwargs)
