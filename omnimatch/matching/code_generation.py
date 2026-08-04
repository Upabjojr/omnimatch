import re

from ..expressions.expressions import Wildcard, Operation, OperationHead
from ..expressions.constraints import CustomConstraint
from ..expressions.functions import op_iter, get_variables
from ._common import OPERATION_END
from .many_to_one import LabelTypeEpsilon, LabelTypeEnd, LabelTypeOperation, LabelTypeExpression
from ..utils import get_short_lambda_source

# COLLAPSE_IF_RE disabled - had indentation bugs with comments between nested ifs
# COLLAPSE_IF_RE = re.compile(
#     r'\n(?P<indent1>\s*)if (?P<cond1>[^\n]+):\n+\1(?P<indent2>\s+)'
#     r'(?P<comment>(?:\#[^\n]*\n+\1\3)*)'
#     r'if (?P<cond2>[^\n]+):\n+'
#     r'(?P<block>\1\3(?P<indent3>\s+)[^\n]*\n+(?:\1\3\7[^\n]*\n+)*)'
#     r'(?!\1(?:\3|elif|else))'
# )


class CodeGenerator:
    def __init__(self, matcher):
        self._matcher = matcher
        self._var_number = 0
        self._indentation = '    '
        self._level = 0
        self._code = ''
        self._subjects = ['subjects']
        self._substs = 0
        self._patterns = set(range(len(matcher.patterns)))
        self._associative = 0
        self._associative_stack = [None]
        self._global_code = []
        self._imports = set()
        self._constraint_objects = {}

    def indent(self):
        self._level += 1

    def dedent(self):
        if self._level > 0:
            self._level -= 1

    def add_line(self, line):
        self._code += (self._indentation * self._level) + str(line) + '\n'

    def get_var_name(self, prefix):
        self._var_number += 1
        return prefix + str(self._var_number)

    def generate_code(self, func_name='match_root', add_imports=True):
        self._imports.add('from collections import deque')
        self._imports.add('from omnimatch.expressions.expressions import Operation')
        self.add_line('def {}(subject):'.format(func_name))
        self.indent()
        self.add_line('{} = deque([subject]) if subject is not None else deque()'.format(self._subjects[-1]))
        self.add_line('subst{} = Substitution()'.format(self._substs))
        self.generate_state_code(self._matcher.root)
        self.add_line('return')
        self.add_line('yield')
        self.dedent()

        if add_imports:
            self._global_code.insert(0, '\n'.join(self._imports))

        return self.clean_code('\n\n'.join(p for p in self._global_code if p)), self.clean_code(self._code)

    @property
    def constraint_objects(self):
        """Dict of constraint name -> constraint object for injection into generated code namespace."""
        return self._constraint_objects

    def final_label(self, index, subst_name):
        return str(index)

    def generate_state_code(self, state, valid_patterns=None):
        if state.matcher is not None:
            self._imports.add('from omnimatch.matching.many_to_one import CommutativeMatcher')
            self._imports.add('from typing import ClassVar')
            self._imports.add('from multiset import Multiset')
            self._imports.add('from omnimatch.utils import VariableWithCount')
            generator = type(self)(state.matcher.automaton)
            generator.indent()
            global_code, code = generator.generate_code(func_name='get_match_iter', add_imports=False)
            self._global_code.append(global_code)
            patterns = self.commutative_patterns(state.matcher.patterns)
            subjects = repr(state.matcher.subjects)
            subjects_by_id = repr(state.matcher.subjects_by_id)
            associative = self.operation_symbol(state.matcher.associative)
            max_optional_count = repr(state.matcher.max_optional_count)
            anonymous_patterns = repr(state.matcher.anonymous_patterns)
            self._global_code.append(
                '''
class CommutativeMatcher{0}(CommutativeMatcher):
{8}_instance: ClassVar = None
{8}patterns: ClassVar = {1}
{8}subjects: ClassVar = {2}
{8}subjects_by_id: ClassVar = {7}
{8}bipartite: ClassVar = BipartiteGraph()
{8}associative: ClassVar = {3}
{8}max_optional_count: ClassVar = {4}
{8}anonymous_patterns: ClassVar = {5}

{8}def __init__(self):
{8}{8}super().__init__()
{8}{8}self.add_subject(None)

{8}@staticmethod
{8}def get():
{8}{8}if CommutativeMatcher{0}._instance is None:
{8}{8}{8}CommutativeMatcher{0}._instance = CommutativeMatcher{0}()
{8}{8}return CommutativeMatcher{0}._instance

{8}@staticmethod
{6}'''.strip().format(
                    state.number, patterns, subjects, associative, max_optional_count, anonymous_patterns, code,
                    subjects_by_id, self._indentation
                )
            )
            self.add_line('matcher = CommutativeMatcher{}.get()'.format(state.number))
            tmp = self.get_var_name('tmp')
            self.add_line('{} = {}'.format(tmp, self._subjects[-1]))
            self.add_line('{} = []'.format(self._subjects[-1]))
            self.add_line('for s in {}:'.format(tmp))
            self.indent()
            self.add_line('matcher.add_subject(s)')
            subjects = self._subjects.pop()
            self.dedent()
            self.add_line(
                'for pattern_index, subst{} in matcher.match({}, subst{}):'.format(self._substs + 1, tmp, self._substs)
            )
            self._substs += 1
            self.indent()
            self.add_line('pass')
            for tk, transitions in state.transitions.items():
                pattern_index = tk.value if hasattr(tk, 'value') else tk
                self.add_line('if pattern_index == {}:'.format(pattern_index))
                self.indent()
                self.add_line('pass')
                patterns, variables = next((pv.pattern_set, pv.variables) for pv in state.matcher.patterns.values() if pv.index == pattern_index)
                variables = set(v[0][0] for v in variables)
                pvars = iter(get_variables(state.matcher.automaton.patterns[i][0].expression) for i in patterns)
                variables.update(*pvars)
                constraints = []
                if variables:
                    constraints = sorted(
                        set.union(*iter(self._matcher.constraint_vars.get(v, set()) for v in variables))
                    )
                self.generate_constraints(constraints, transitions)
                self.dedent()
            self.dedent()
            self._substs -= 1
            self._subjects.append(subjects)
        else:
            self.add_line('# State {}'.format(state.number))
            if state.number in self._matcher.finals:
                self.add_line('if len({}) == 0:'.format(self._subjects[-1]))
                self.indent()
                self.add_line('pass')
                # Only yield patterns valid for this final state
                patterns_to_yield = valid_patterns if valid_patterns is not None else self._patterns
                for pattern_index in patterns_to_yield:
                    constraints = self._matcher.patterns[pattern_index][0].global_constraints
                    subst_name = self._emit_final_subst_build(pattern_index)
                    for constraint in constraints:
                        self.enter_global_constraint(constraint, subst_name)
                    self._emit_final_yield(pattern_index, subst_name)
                    for constraint in constraints:
                        self.exit_global_constraint(constraint)
                self.dedent()
            else:
                for transitions in state.transitions.values():
                    for transition in transitions:
                        self.generate_transition_code(transition)

    def commutative_var_entry(self, entry):
        return '(VariableWithCount({!r}, {}, {}, {}), {})'.format(
            entry[0][0], entry[0][1], entry[0][2],
            self.expr(entry[0][3]), self.operation_symbol(entry[1]) if isinstance(entry[1], OperationHead) else repr(entry[1])
        )

    def commutative_patterns(self, patterns):
        sorted_patterns = sorted(patterns.values(), key=lambda pv: pv.index)
        return '{{\n    {}\n}}'.format(
            ',\n    '.join(
                '{0}: ({0}, {1!r}, [\n      {2}\n])'.format(pv.index, pv.pattern_set, ',\n      '.join(map(self.commutative_var_entry, pv.variables)))
                for pv in sorted_patterns
            )
        )

    def generate_transition_code(self, transition):
        enter_func = None
        exit_func = None
        label = transition.label
        # Unwrap the label to get the raw value for code generation
        raw_label = label.unwrap() if hasattr(label, 'unwrap') else label
        if isinstance(label, LabelTypeOperation):
            enter_func = self.enter_operation
            exit_func = self.exit_operation
            raw_label = label.value
        elif isinstance(label, LabelTypeEnd):
            enter_func = self.enter_operation_end
            exit_func = self.exit_operation_end
            raw_label = OPERATION_END
        elif isinstance(label, LabelTypeEpsilon):
            enter_func = self.enter_eps
            exit_func = self.exit_eps
        elif isinstance(label, LabelTypeExpression):
            expr = label.value
            if isinstance(expr, Wildcard):
                wc = expr
                if wc.default_value is not None:
                    self.enter_variable_assignment(transition.variable_name, self.optional_expr(wc.default_value))
                    constraints = sorted(transition.check_constraints) if transition.check_constraints is not None else []
                    self.generate_constraints(constraints, [transition])
                    self.exit_variable_assignment()
                if wc.fixed_size and self._associative_stack[-1] is None:
                    enter_func = self.enter_fixed_wildcard
                    exit_func = self.exit_fixed_wildcard
                else:
                    enter_func = self.enter_sequence_wildcard
                    exit_func = self.exit_sequence_wildcard
                raw_label = wc
            else:
                enter_func = self.enter_symbol
                exit_func = self.exit_symbol
                raw_label = expr
        else:
            # Fallback for any other label type
            enter_func = self.enter_symbol
            exit_func = self.exit_symbol
            raw_label = label.unwrap() if hasattr(label, 'unwrap') else label

        value = enter_func(raw_label)
        value, var_value = value if isinstance(value, tuple) else (value, value)
        if transition.variable_name is not None:
            self.enter_variable_assignment(transition.variable_name, var_value)
        if transition.subst is not None:
            self.enter_subst(transition.subst)
        constraints = sorted(transition.check_constraints) if transition.check_constraints is not None else []
        self.generate_constraints(constraints, [transition])

        if transition.subst is not None:
            self.exit_subst(transition.subst)
        if transition.variable_name is not None:
            self.exit_variable_assignment()
        exit_func(value)

    def get_args(self, operation, operation_type):
        return 'op_iter({})'.format(operation)

    def push_subjects(self, value, operation):
        self._subjects.append(self.get_var_name('subjects'))
        self.add_line('{} = deque({})'.format(self._subjects[-1], self.get_args(value, operation)))

    def push_subst(self):
        new_subst = self.get_var_name('subst')
        self.add_line('subst{} = Substitution(subst{})'.format(self._substs + 1, self._substs))
        self._substs += 1

    def enter_eps(self, _):
        return '{0}'.format(self._subjects[-1])

    def exit_eps(self, _):
        pass

    def enter_operation(self, operation):
        # In the new design, operations are identified by OperationHead, not by type.
        # Generate: isinstance(X, Operation) and X.head == <head>
        op_name = self.operation_symbol(operation)
        self.add_line(
            'if len({0}) >= 1 and isinstance({0}[0], Operation) and {0}[0].head == {1}:'.
            format(self._subjects[-1], op_name)
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        atype = operation if (isinstance(operation, OperationHead) and operation.associative) else None
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1
            self.add_line('associative{} = {}'.format(self._associative, tmp))
            self.add_line('associative_type{} = type({})'.format(self._associative, tmp))
        self.push_subjects(tmp, operation)
        return tmp

    def operation_symbol(self, operation):
        if operation is None:
            return 'None'
        if isinstance(operation, OperationHead):
            # Store OperationHead as a module-level variable in the generated code
            var_name = '_op_head_' + re.sub(r'\W', '_', operation.name)
            head_repr = 'OperationHead(name={!r}, arity=Arity({}, {}), commutative={!r}, associative={!r}, one_identity={!r}, infix={!r})'.format(
                operation.name, operation.arity.min_count, operation.arity.fixed_size,
                operation.commutative, operation.associative, operation.one_identity, operation.infix
            )
            assign_line = '{} = {}'.format(var_name, head_repr)
            if assign_line not in '\n'.join(self._global_code):
                self._imports.add('from omnimatch.expressions.expressions import OperationHead, Arity')
                self._global_code.append(assign_line)
            return var_name
        return getattr(operation, '__name__', str(operation))

    def exit_operation(self, value):
        self._subjects.pop()
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1

    def enter_fixed_wildcard(self, wildcard):
        self.add_line('if len({}) >= 1:'.format(self._subjects[-1]))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        return tmp

    def exit_fixed_wildcard(self, value):
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()

    def enter_variable_assignment(self, variable_name, value):
        self.push_subst()
        self.add_line('try:')
        self.indent()
        self.add_line('subst{}.try_add_variable({!r}, {})'.format(self._substs, variable_name, value))
        self.dedent()
        self.add_line('except ValueError:')
        self.indent()
        self.add_line('pass')
        self.dedent()
        self.add_line('else:')
        self.indent()
        self.add_line('pass')

    def enter_subst(self, subst):
        self.push_subst()
        self.add_line('try:')
        self.indent()
        for name, value in subst.items():
            self.add_line('subst{}.try_add_variable({!r}, {})'.format(self._substs, name, self.expr(value)))
        self.dedent()
        self.add_line('except ValueError:')
        self.indent()
        self.add_line('pass')
        self.dedent()
        self.add_line('else:')
        self.indent()
        self.add_line('pass')

    def expr(self, expr):
        return repr(expr)

    def exit_subst(self, subst):
        self._substs -= 1
        self.dedent()

    def exit_variable_assignment(self):
        self._substs -= 1
        self.dedent()

    def enter_optional_wildcard(self, wildcard, variable_name):
        self.enter_variable_assignment(variable_name, self.optional_expr(wildcard.default_value))

    def optional_expr(self, expr):
        return repr(expr)

    def exit_optional_wildcard(self, value):
        self.exit_variable_assignment()

    def enter_symbol(self, symbol):
        self.add_line('if len({0}) >= 1 and {0}[0] == {1}:'.format(self._subjects[-1], self.symbol_repr(symbol)))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        return tmp

    def symbol_repr(self, symbol):
        return repr(symbol)

    def exit_symbol(self, value):
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()

    def enter_operation_end(self, _):
        self.add_line('if len({0}) == 0:'.format(self._subjects[-1]))
        self.indent()
        self.add_line('pass')
        subjects = self._subjects.pop()
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1
        return [subjects, atype]

    def exit_operation_end(self, value):
        subjects, atype = value
        self._subjects.append(subjects)
        self.dedent()
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1

    def enter_sequence_wildcard(self, wildcard):
        tmp = self.get_var_name('tmp')
        tmp2 = self.get_var_name('tmp')
        mc = wildcard.min_count if wildcard.default_value is None or wildcard.min_count > 0 else 1
        self.add_line('if len({}) >= {}:'.format(self._subjects[-1], mc))
        self.indent()
        self.add_line('{} = []'.format(tmp))
        for _ in range(mc):
            self.add_line('{}.append({}.popleft())'.format(tmp, self._subjects[-1]))
        self.add_line('while True:')
        self.indent()
        if self._associative_stack[-1] is not None and wildcard.fixed_size:
            self.add_line('if len({}) > {}:'.format(tmp, wildcard.min_count))
            self.indent()
            self.add_line(
                '{} = {}'.format(
                    tmp2,
                    self.create_operation(
                        'associative{}'.format(self._associative), 'associative{}'.format(self._associative), tmp
                    )
                )
            )
            self.dedent()
            self.add_line('elif len({}) == 1:'.format(tmp))
            self.indent()
            self.add_line('{} = {}[0]'.format(tmp2, tmp))
            self.dedent()
            self.add_line('else:')
            self.indent()
            self.add_line('assert False, "Unreachable"')
            self.dedent()
        else:
            self.add_line('{} = tuple({})'.format(tmp2, tmp))
        return tmp, tmp2

    def create_operation(self, operation, operation_type, args):
        return 'create_operation_expression({}, {})'.format(operation, args)

    def exit_sequence_wildcard(self, value):
        self.add_line('if len({}) == 0:'.format(self._subjects[-1]))
        self.indent()
        self.add_line('break')
        self.dedent()
        self.add_line('{}.append({}.popleft())'.format(value, self._subjects[-1]))
        self.dedent()
        self.add_line('{}.extendleft(reversed({}))'.format(self._subjects[-1], value))
        self.dedent()

    def _emit_final_subst_build(self, pattern_index):
        """Emit code to build the final substitution with original var names. Returns subst_name."""
        renaming = self._matcher.pattern_vars[pattern_index]
        subst_name = 'subst{}'.format(self._substs)
        if any(k != v for k, v in renaming.items()):
            self.add_line('tmp_subst = Substitution()')
            for original, renamed in renaming.items():
                self.add_line('tmp_subst[{!r}] = subst{}[{!r}]'.format(original, self._substs, renamed))
            subst_name = 'tmp_subst'
        return subst_name

    def _emit_final_yield(self, pattern_index, subst_name):
        """Emit the yield statement for the final substitution."""
        self.add_line('# {}: {}'.format(pattern_index, self._matcher.patterns[pattern_index][0]))
        self.add_line('yield {}, {}'.format(self.final_label(pattern_index, subst_name), subst_name))

    def yield_final_substitution(self, pattern_index):
        """Emit both the subst build and the yield (legacy interface)."""
        subst_name = self._emit_final_subst_build(pattern_index)
        self._emit_final_yield(pattern_index, subst_name)

    def generate_constraints(self, constraints, transitions):
        # Count the blocks actually OPENED: a constraint disjoint from the transitions'
        # patterns is `continue`d without emitting an `if`/indent, but the old cleanup
        # loop dedented once per constraint INCLUDING the skipped ones. Whenever a
        # pattern-index branch carried skipped constraints (routine in Rubi's
        # commutative rule sets), the indentation stack underflowed and the emitted
        # code dedented out of the enclosing function -- `return` at column 0,
        # SyntaxError, "code generation is unviable for Rubi".
        opened = 0
        for constraint_index in constraints:
            constraint, patterns = self._matcher.constraints[constraint_index]
            t_iter = iter(t.patterns for t in transitions)
            potential = next(t_iter).union(*t_iter)
            if patterns.isdisjoint(potential):
                continue
            if isinstance(constraint, CustomConstraint):
                source = get_short_lambda_source(constraint.constraint)
                if source:
                    self.add_line('# Constraint: {}'.format(source.strip()))
            subst_name = 'subst{}'.format(self._substs)
            constraint_name = 'constraint{}'.format(constraint_index)
            self._constraint_objects[constraint_name] = constraint
            self.add_line('if {}({}):'.format(constraint_name, subst_name))
            self.indent()
            self.add_line('pass')
            opened += 1

        for transition in transitions:
            self.generate_state_code(transition.target, valid_patterns=transition.patterns)

        for _ in range(opened):
            self.dedent()

    def enter_global_constraint(self, constraint, subst_name=None):
        if subst_name is None:
            subst_name = 'subst{}'.format(self._substs)
        constraint_name = 'constraint{}'.format(id(constraint))
        self._constraint_objects[constraint_name] = constraint
        self.add_line('if {}({}):'.format(constraint_name, subst_name))
        self.indent()
        self.add_line('pass')

    def exit_global_constraint(self, constraint):
        self.dedent()

    @staticmethod
    def clean_code(code):
        # Remove unnecessary 'pass' statements, but preserve pass when it's the only
        # content in a block that precedes else/elif/except/finally (which would leave the block empty)
        # First regex: remove pass followed by new statements at same indent (but not else/elif/except/finally)
        code = re.sub(r'\n(\s*)pass\n((?:\1(?:if|for|while|try|with|def|class)[^\n]*\n)+)', r'\n\2', code)
        # Second regex: remove pass followed by non-whitespace, but use negative lookahead to skip else/elif/except/finally
        code = re.sub(r'\n(\s*)pass\n(\s*\n)*(\1(?!else|elif|except|finally)\S)', r'\n\3', code)
        return code
