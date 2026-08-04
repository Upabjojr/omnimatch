# -*- coding: utf-8 -*-
from pydantic import ConfigDict, Field, PrivateAttr
from omnimatch.expressions.constraints import Constraint
from omnimatch.expressions.substitution import Substitution
from omnimatch.expressions.expressions import Pattern


class MockConstraint(Constraint):
    model_config = ConfigDict(extra='allow')

    return_value: object = None
    called_with: list = Field(default_factory=list)
    renaming: dict = Field(default_factory=dict)
    _variables: set = PrivateAttr(default_factory=set)

    def __init__(self, return_value, *variables, renaming=None, **kwargs):
        super().__init__(
            return_value=return_value,
            called_with=[],
            renaming=renaming or {},
            **kwargs
        )
        self._variables = set(variables)

    def __call__(self, match):
        self.called_with.append(Substitution(match))
        return self.return_value

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        if self.variables:
            return 'MockConstraint({!r}, {}, renaming={!r})'.format(
                self.return_value, ', '.join(map(repr, self.variables)), self.renaming)
        return 'MockConstraint({!r}, renaming={!r})'.format(self.return_value, self.renaming)

    def with_renamed_vars(self, renaming):
        self.renaming.update(renaming)
        return self

    @property
    def variables(self):
        return set(self.renaming.get(v, v) for v in self._variables)

    @property
    def call_count(self):
        return len(self.called_with)

    def assert_called_with(self, args):
        args = dict((self.renaming.get(n, n), v) for n, v in args.items())
        assert args in self.called_with, "Constraint was not called with {}. List of calls: {}".format(
            args, self.called_with
        )


def assert_match_as_expected(match, subject, pattern, expected_matches):
    pattern = Pattern(pattern)
    matches = list(match(subject, pattern))
    assert len(matches) == len(expected_matches), 'Unexpected number of matches'
    for expected_match in expected_matches:
        assert expected_match in matches, "Subject {!s} and pattern {!s} did not yield the match {!s} but were supposed to".format(
            subject, pattern, expected_match
        )
