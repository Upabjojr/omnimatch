# -*- coding: utf-8 -*-
import pytest
from types import ModuleType

from omnimatch.expressions.expressions import Wildcard, Operation
from omnimatch.matching.one_to_one import match as match_one_to_one
from omnimatch.matching.many_to_one import ManyToOneMatcher
from omnimatch.expressions.functions import preorder_iter
from omnimatch.matching.code_generation import CodeGenerator
from omnimatch.matching.json_serialization import to_json, from_json

def pytest_configure():
    pytest.matcher = None

def pytest_generate_tests(metafunc):
    if 'match' in metafunc.fixturenames:
        metafunc.parametrize('match', ['one-to-one', 'many-to-one', 'generated', 'json-roundtrip'], indirect=True)
    if 'match_many' in metafunc.fixturenames:
        metafunc.parametrize('match_many', ['many-to-one', 'generated', 'json-roundtrip'], indirect=True)


def match_many_to_one(expression, *patterns):
    try:
        pattern = patterns[0]
        commutative = next(
            p for p in preorder_iter(pattern.expression) if isinstance(p, Operation) and p.head.commutative
        )
        next(wc for wc in preorder_iter(commutative) if isinstance(wc, Wildcard) and wc.min_count > 1)
    except StopIteration:
        pass
    else:
        pytest.xfail('Matcher does not support fixed wildcards with length != 1 in commutative operations')
    matcher = ManyToOneMatcher(*patterns)
    for _, substitution in matcher.match(expression):
        yield substitution


GENERATED_TEMPLATE = '''
# -*- coding: utf-8 -*-
from omnimatch import *
from tests.common import *
from tests.utils import *

{}

{}
'''.strip()


def match_generated(expression, *patterns):
    matcher = ManyToOneMatcher(*patterns)
    generator = CodeGenerator(matcher)
    gc, code = generator.generate_code()
    code = GENERATED_TEMPLATE.format(gc, code)
    compiled = compile(code, '', 'exec')
    module = ModuleType("generated_code")
    print(code)
    # Inject constraint objects into module namespace (avoids repr issues with lambdas)
    module.__dict__.update(generator.constraint_objects)
    exec(compiled, module.__dict__)
    for _, substitution in module.match_root(expression):
        yield substitution



def match_json_roundtrip(expression, *patterns):
    try:
        pattern = patterns[0]
        commutative = next(
            p for p in preorder_iter(pattern.expression) if isinstance(p, Operation) and p.head.commutative
        )
        next(wc for wc in preorder_iter(commutative) if isinstance(wc, Wildcard) and wc.min_count > 1)
    except StopIteration:
        pass
    else:
        pytest.xfail('Matcher does not support fixed wildcards with length != 1 in commutative operations')
    matcher = ManyToOneMatcher(*patterns)
    # Save state that can't be fully serialized (CustomConstraint callables)
    original_patterns = matcher.patterns[:]
    original_constraints = matcher.constraints[:]
    original_constraint_vars = dict(matcher.constraint_vars)
    json_str = to_json(matcher)
    matcher2 = from_json(json_str)
    # Restore constraint objects (CustomConstraint has non-serializable callables)
    # Patterns contain global constraints, matcher.constraints has local ones
    matcher2.patterns = original_patterns
    matcher2.constraints = original_constraints
    matcher2.constraint_vars = original_constraint_vars
    for _, substitution in matcher2.match(expression):
        yield substitution


@pytest.fixture
def match(request):
    pytest.matcher = request.param
    if request.param == 'one-to-one':
        return match_one_to_one
    elif request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'generated':
        return match_generated
    elif request.param == 'json-roundtrip':
        return match_json_roundtrip
    else:
        raise ValueError("Invalid internal test config")


@pytest.fixture
def match_many(request):
    pytest.matcher = request.param
    if request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'generated':
        return match_generated
    elif request.param == 'json-roundtrip':
        return match_json_roundtrip
    else:
        raise ValueError("Invalid internal test config")
