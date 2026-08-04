# -*- coding: utf-8 -*-
import pytest

from omnimatch.expressions.expressions import Operation, NamedAtom, Arity, Wildcard, make_dot_variable, make_star_variable, make_plus_variable
import omnimatch

@pytest.fixture(autouse=True)
def add_default_expressions(doctest_namespace):
    doctest_namespace['f'] = Operation.new('f', Arity.variadic)
    doctest_namespace['a'] = NamedAtom('a')
    doctest_namespace['b'] = NamedAtom('b')
    doctest_namespace['c'] = NamedAtom('c')
    doctest_namespace['x_'] = make_dot_variable('x')
    doctest_namespace['y_'] = make_dot_variable('y')
    doctest_namespace['_'] = Wildcard.dot()
    doctest_namespace['__'] = Wildcard.plus()
    doctest_namespace['___'] = Wildcard.star()
    doctest_namespace['__name__'] = '__main__'

    for name in omnimatch.__all__:
        doctest_namespace[name] = getattr(omnimatch, name)
