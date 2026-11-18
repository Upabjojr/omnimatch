# -*- coding: utf-8 -*-
import itertools
import math

import hypothesis.strategies as st
from hypothesis import given
import pytest

from omnimatch.matching.bipartite import BipartiteGraph, _DirectedMatchGraph, enum_maximum_matchings_iter


@st.composite
def bipartite_graph(draw):
    m = draw(st.integers(min_value=1, max_value=4))
    n = draw(st.integers(min_value=m, max_value=5))

    graph = BipartiteGraph()
    for i in range(n):
        for j in range(m):
            b = draw(st.booleans())
            if b:
                graph[i, j] = b

    return graph


@given(bipartite_graph())
def test_enum_maximum_matchings_iter_correctness(graph):
    size = None
    matchings = set()
    for matching in enum_maximum_matchings_iter(graph):
        if size is None:
            size = len(matching)
        assert len(matching) == size, "Matching has a different size than the first one"
        for edge in matching.items():
            assert edge in graph, "Matching contains an edge that was not in the graph"
        frozen_matching = frozenset(matching.items())
        assert frozen_matching not in matchings, "Matching was duplicate"
        matchings.add(frozen_matching)


@pytest.mark.parametrize('n, m', filter(lambda x: x[0] >= x[1], itertools.product(range(1, 6), range(0, 4))))
def test_completeness(n, m):
    graph = BipartiteGraph(map(lambda x: (x, True), itertools.product(range(n), range(m))))
    count = sum(1 for _ in enum_maximum_matchings_iter(graph))
    expected_count = m > 0 and math.factorial(n) / math.factorial(n - m) or 0
    assert count == expected_count


@pytest.mark.parametrize(
    '   graph,                      expected_cycle',
    [
        ({},                        []),
        ({0: {1}},                  []),
        ({0: {1}, 1: {2}},          []),
        ({0: {1}, 1: {0}},          [0, 1]),
        ({0: {1}, 1: {0}},          [1, 0]),
        ({0: {1}, 1: {0, 2}},       [0, 1]),
        ({0: {1, 2}, 1: {0, 2}},    [0, 1]),
        ({0: {1, 2}, 1: {0}},       [0, 1]),
        ({0: {1}, 1: {2}, 2: {0}},  [0, 1, 2]),
        ({0: {2}, 1: {2}},          []),
        ({0: {2}, 1: {2}, 2: {0}},  [0, 2]),
        ({0: {2}, 1: {2}, 2: {1}},  [1, 2]),
    ]
)  # yapf: disable
def test_directed_graph_find_cycle(graph, expected_cycle):
    dmg = _DirectedMatchGraph({}, {})
    dmg.update(graph)
    cycle = dmg.find_cycle()
    if len(expected_cycle) > 0:
        assert expected_cycle[0] in cycle
        start = cycle.index(expected_cycle[0])
        cycle = cycle[start:] + cycle[:start]
    assert cycle == expected_cycle


class TestBipartiteGraphTest:
    def test_setitem(self):
        graph = BipartiteGraph()

        graph[0, 1] = True

        with pytest.raises(TypeError):
            graph[0] = True

        with pytest.raises(TypeError):
            graph[0, ] = True

        with pytest.raises(TypeError):
            graph[0, 1, 2] = True

    def test_getitem(self):
        graph = BipartiteGraph({(0, 0): True})

        assert graph[0, 0] == True

        with pytest.raises(TypeError):
            _ = graph[0]

        with pytest.raises(TypeError):
            _ = graph[0, ]

        with pytest.raises(TypeError):
            _ = graph[0, 1, 2]

        with pytest.raises(KeyError):
            _ = graph[0, 1]

    def test_delitem(self):
        graph = BipartiteGraph({(0, 0): True})

        assert (0, 0) in graph

        del graph[0, 0]

        assert (0, 0) not in graph

        with pytest.raises(TypeError):
            del graph[0]

        with pytest.raises(TypeError):
            del graph[0, ]

        with pytest.raises(TypeError):
            del graph[0, 1, 2]

        with pytest.raises(KeyError):
            del graph[0, 1]

    def test_limited_to(self):
        graph = BipartiteGraph({(0, 0): True, (1, 0): True, (1, 1): True, (0, 1): True})

        assert graph.limited_to({0}, {0}) == {(0, 0): True}
        assert graph.limited_to({0, 1}, {1}) == {(0, 1): True, (1, 1): True}
        assert graph.limited_to({1}, {1}) == {(1, 1): True}
        assert graph.limited_to({1}, {0, 1}) == {(1, 0): True, (1, 1): True}
        assert graph.limited_to({0, 1}, {0, 1}) == graph

    def test_eq(self):
        assert BipartiteGraph() == {}
        assert {} == BipartiteGraph()
        assert BipartiteGraph({(1, 1): True}) == {(1, 1): True}
        assert {(1, 1): True} == BipartiteGraph({(1, 1): True})
        assert not BipartiteGraph({(1, 2): True}) == {(1, 1): True}
        assert not {(1, 2): True} == BipartiteGraph({(1, 1): True})
        assert not BipartiteGraph() == ''
        assert not '' == BipartiteGraph()


def _brute_force_maximum_matchings(edges):
    """All maximum matchings of the bipartite graph with the given edge list, by brute force."""
    best = set()
    for size in range(len(edges), 0, -1):
        for subset in itertools.combinations(edges, size):
            lefts = {l for l, _ in subset}
            rights = {r for _, r in subset}
            if len(lefts) == len(rights) == size:
                best.add(frozenset(subset))
        if best:
            break
    return best


@pytest.mark.parametrize(
    '   edges,                          expected',
    [
        # A right vertex that can be left free: previously only {0: 0} was produced.
        ([(0, 0), (0, 1)],              [{0: 0}, {0: 1}]),
        ([(0, 0), (0, 1), (0, 2)],      [{0: 0}, {0: 1}, {0: 2}]),
        ([(0, 0), (1, 0)],              [{0: 0}, {1: 0}]),
        ([(0, 0), (0, 1), (1, 1)],      [{0: 0, 1: 1}]),
        ([(0, 0), (1, 1), (1, 2)],      [{0: 0, 1: 1}, {0: 0, 1: 2}]),
    ]
)
def test_enum_maximum_matchings_iter_leaves_right_vertex_free(edges, expected):
    graph = BipartiteGraph((edge, True) for edge in edges)
    result = list(enum_maximum_matchings_iter(graph))
    assert sorted(map(sorted, (m.items() for m in result))) == sorted(map(sorted, (m.items() for m in expected)))


@pytest.mark.parametrize('n, m', [(n, m) for n in range(1, 4) for m in range(1, 4)])
def test_enum_maximum_matchings_iter_exhaustive(n, m):
    """Compare against brute force on every bipartite graph with n left and m right vertices."""
    all_edges = list(itertools.product(range(n), range(m)))
    for mask in range(1 << len(all_edges)):
        edges = [e for i, e in enumerate(all_edges) if mask >> i & 1]
        graph = BipartiteGraph((edge, True) for edge in edges)
        result = [frozenset(matching.items()) for matching in enum_maximum_matchings_iter(graph)]
        expected = _brute_force_maximum_matchings(edges)
        assert len(result) == len(set(result)), "duplicate matching for {}".format(edges)
        assert set(result) == expected, "wrong matchings for {}".format(edges)
