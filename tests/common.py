# -*- coding: utf-8 -*-
from omnimatch.expressions.expressions import (
    Arity, Operation, NamedAtom, Wildcard, make_dot_variable, make_plus_variable, make_star_variable
)

from .utils import MockConstraint


class SpecialSymbol(NamedAtom):
    pass


f = Operation.new('f', Arity.variadic)
f2 = Operation.new('f2', Arity.variadic)
f_u = Operation.new('f_u', Arity.unary)
f_i = Operation.new('f_i', Arity.variadic, one_identity=True)
f_c = Operation.new('f_c', Arity.variadic, commutative=True)
f_ci = Operation.new('f_ci', Arity.variadic, commutative=True, one_identity=True)
f2_c = Operation.new('f2_c', Arity.variadic, commutative=True)
f_a = Operation.new('f_a', Arity.variadic, associative=True)
f_ac = Operation.new('f_ac', Arity.variadic, associative=True, commutative=True)
a = NamedAtom('a')
b = NamedAtom('b')
c = NamedAtom('c')
d = NamedAtom('d')
a_x = NamedAtom('a', variable_name='x')
s = SpecialSymbol('s')
_ = Wildcard.dot()
x_ = make_dot_variable('x')
y_ = make_dot_variable('y')
z_ = make_dot_variable('z')
__ = Wildcard.plus()
x__ = make_plus_variable('x')
y__ = make_plus_variable('y')
z__ = make_plus_variable('z')
___ = Wildcard.star()
x___ = make_star_variable('x')
y___ = make_star_variable('y')
z___ = make_star_variable('z')
oa_ = Wildcard.optional('o', a)
oa__ = Wildcard(1, False, 'o', a)
oa___ = Wildcard(0, False, 'o', a)
o2b_ = Wildcard.optional('o2', b)

mock_constraint_false = MockConstraint(False)
mock_constraint_true = MockConstraint(True)

del Arity
del Operation
del NamedAtom
del Wildcard
del MockConstraint

__all__ = [name for name in dir() if not name.startswith('__') or all(c == '_' for c in name)]
