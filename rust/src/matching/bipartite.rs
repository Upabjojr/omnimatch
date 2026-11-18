//! Bipartite graphs and enumeration of their maximum matchings.
//!
//! Port of `omnimatch/matching/bipartite.py`. The enumeration follows Takeaki
//! Uno, "Algorithms for Enumerating All Perfect, Maximum and Maximal Matchings
//! in Bipartite Graphs" (ISAAC '97), as the Python version does.
//!
//! The graphviz helpers (`as_graph`) are not ported.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;
use std::ops::ControlFlow;

use crate::matching::hopcroft_karp::HopcroftKarp;

/// A matching: each left vertex mapped to its right partner.
pub type Matching<L, R> = BTreeMap<L, R>;

/// A vertex of a bipartite graph, tagged with its side (Python: `(LEFT, x)` / `(RIGHT, x)`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Node<L, R> {
    Left(L),
    Right(R),
}

/// A bipartite graph whose edges `(left, right)` carry a value of type `V`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BipartiteGraph<L: Ord, R: Ord, V> {
    edges: BTreeMap<(L, R), V>,
    left_adjacency: BTreeMap<L, BTreeSet<R>>,
    right_adjacency: BTreeMap<R, BTreeSet<L>>,
}

impl<L: Ord, R: Ord, V> Default for BipartiteGraph<L, R, V> {
    fn default() -> Self {
        BipartiteGraph { edges: BTreeMap::new(), left_adjacency: BTreeMap::new(), right_adjacency: BTreeMap::new() }
    }
}

impl<L: Ord + Clone, R: Ord + Clone, V> BipartiteGraph<L, R, V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Add (or relabel) an edge, returning the previous value.
    pub fn insert(&mut self, left: L, right: R, value: V) -> Option<V> {
        self.left_adjacency.entry(left.clone()).or_default().insert(right.clone());
        self.right_adjacency.entry(right.clone()).or_default().insert(left.clone());
        self.edges.insert((left, right), value)
    }

    /// The value of an edge, inserting `default()` first if it is missing
    /// (Python: `graph.setdefault(edge, default)`).
    pub fn get_or_insert_with(&mut self, left: L, right: R, default: impl FnOnce() -> V) -> &mut V {
        if !self.edges.contains_key(&(left.clone(), right.clone())) {
            self.insert(left.clone(), right.clone(), default());
        }
        self.edges.get_mut(&(left, right)).expect("just inserted")
    }

    pub fn get(&self, left: &L, right: &R) -> Option<&V>
    where
        L: Clone,
        R: Clone,
    {
        self.edges.get(&(left.clone(), right.clone()))
    }

    pub fn contains(&self, left: &L, right: &R) -> bool {
        self.get(left, right).is_some()
    }

    /// Remove an edge, returning its value.
    pub fn remove(&mut self, left: &L, right: &R) -> Option<V> {
        let value = self.edges.remove(&(left.clone(), right.clone()))?;
        if let Some(rs) = self.left_adjacency.get_mut(left) {
            rs.remove(right);
            if rs.is_empty() {
                self.left_adjacency.remove(left);
            }
        }
        if let Some(ls) = self.right_adjacency.get_mut(right) {
            ls.remove(left);
            if ls.is_empty() {
                self.right_adjacency.remove(right);
            }
        }
        Some(value)
    }

    pub fn clear(&mut self) {
        self.edges.clear();
        self.left_adjacency.clear();
        self.right_adjacency.clear();
    }

    /// Edges with their values, in sorted order.
    pub fn edges_with_labels(&self) -> impl Iterator<Item = (&(L, R), &V)> {
        self.edges.iter()
    }

    /// Edges, in sorted order.
    pub fn edges(&self) -> impl Iterator<Item = &(L, R)> {
        self.edges.keys()
    }

    pub fn left_nodes(&self) -> impl Iterator<Item = &L> {
        self.left_adjacency.keys()
    }

    pub fn right_nodes(&self) -> impl Iterator<Item = &R> {
        self.right_adjacency.keys()
    }

    /// Right vertices adjacent to `left`.
    pub fn neighbors_of_left(&self, left: &L) -> Option<&BTreeSet<R>> {
        self.left_adjacency.get(left)
    }

    /// Left vertices adjacent to `right`.
    pub fn neighbors_of_right(&self, right: &R) -> Option<&BTreeSet<L>> {
        self.right_adjacency.get(right)
    }

    /// A maximum matching, found with Hopcroft-Karp.
    pub fn find_matching(&self) -> Matching<L, R>
    where
        R: Hash,
    {
        let graph = self.left_adjacency.iter().map(|(l, rs)| (l.clone(), rs.iter().cloned().collect())).collect();
        HopcroftKarp::new(graph).get_maximum_matching()
    }

    fn filtered(&self, mut keep: impl FnMut(&L, &R) -> bool) -> Self
    where
        V: Clone,
    {
        let mut g = Self::new();
        for ((l, r), v) in &self.edges {
            if keep(l, r) {
                g.insert(l.clone(), r.clone(), v.clone());
            }
        }
        g
    }

    /// A copy without the edge's two endpoints (and all their edges).
    pub fn without_nodes(&self, left: &L, right: &R) -> Self
    where
        V: Clone,
    {
        self.filtered(|l, r| l != left && r != right)
    }

    /// A copy without the given edge.
    pub fn without_edge(&self, left: &L, right: &R) -> Self
    where
        V: Clone,
    {
        self.filtered(|l, r| !(l == left && r == right))
    }

    /// The subgraph induced by the given vertex sets.
    pub fn limited_to(&self, left: &BTreeSet<L>, right: &BTreeSet<R>) -> Self
    where
        V: Clone,
    {
        self.filtered(|l, r| left.contains(l) && right.contains(r))
    }
}

impl<L: Ord + Clone, R: Ord + Clone, V> FromIterator<((L, R), V)> for BipartiteGraph<L, R, V> {
    fn from_iter<I: IntoIterator<Item = ((L, R), V)>>(iter: I) -> Self {
        let mut g = Self::new();
        for ((l, r), v) in iter {
            g.insert(l, r, v);
        }
        g
    }
}

// ─── directed graphs ─────────────────────────────────────────────────────────

/// A directed graph as an adjacency map (Python: `_DirectedMatchGraph`, a dict of sets).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedGraph<N: Ord> {
    pub adjacency: BTreeMap<N, BTreeSet<N>>,
}

impl<N: Ord + Clone> DirectedGraph<N> {
    pub fn new() -> Self {
        DirectedGraph { adjacency: BTreeMap::new() }
    }

    /// Some cycle of the graph (as its list of nodes), or an empty list.
    pub fn find_cycle(&self) -> Vec<N> {
        let mut visited = BTreeSet::new();
        for node in self.adjacency.keys() {
            let cycle = self.find_cycle_from(node, &mut Vec::new(), &mut visited);
            if !cycle.is_empty() {
                return cycle;
            }
        }
        Vec::new()
    }

    fn find_cycle_from(&self, node: &N, path: &mut Vec<N>, visited: &mut BTreeSet<N>) -> Vec<N> {
        if visited.contains(node) {
            return match path.iter().position(|n| n == node) {
                Some(index) => path[index..].to_vec(),
                None => Vec::new(),
            };
        }
        visited.insert(node.clone());
        let Some(successors) = self.adjacency.get(node) else { return Vec::new() };
        for other in successors {
            path.push(node.clone());
            let cycle = self.find_cycle_from(other, path, visited);
            path.pop();
            if !cycle.is_empty() {
                return cycle;
            }
        }
        Vec::new()
    }
}

impl<N: Ord + Clone> Default for DirectedGraph<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// The directed graph D(G, M): matched edges point left → right, all other
/// edges right → left.
fn directed_match_graph<L: Ord + Clone, R: Ord + Clone, V>(
    graph: &BipartiteGraph<L, R, V>,
    matching: &Matching<L, R>,
) -> DirectedGraph<Node<L, R>> {
    let mut d = DirectedGraph::new();
    for (tail, head) in graph.edges() {
        if matching.get(tail) == Some(head) {
            d.adjacency.insert(Node::Left(tail.clone()), BTreeSet::from([Node::Right(head.clone())]));
        } else {
            d.adjacency.entry(Node::Right(head.clone())).or_default().insert(Node::Left(tail.clone()));
        }
    }
    d
}

// ─── enumeration ─────────────────────────────────────────────────────────────

/// Call `on_matching` for every maximum matching of `graph` (each exactly
/// once), stopping early if it returns [`ControlFlow::Break`].
pub fn for_each_maximum_matching<L, R, V>(
    graph: &BipartiteGraph<L, R, V>,
    on_matching: &mut dyn FnMut(&Matching<L, R>) -> ControlFlow<()>,
) -> ControlFlow<()>
where
    L: Ord + Clone,
    R: Ord + Clone + Hash,
    V: Clone,
{
    let matching = graph.find_matching();
    if matching.is_empty() {
        return ControlFlow::Continue(());
    }
    on_matching(&matching)?;
    let mut graph = graph.clone();
    let dmg = directed_match_graph(&graph, &matching);
    enum_rec(&mut graph, &matching, &dmg, on_matching)
}

/// All maximum matchings of `graph` (Python: `enum_maximum_matchings_iter`).
pub fn enum_maximum_matchings<L, R, V>(graph: &BipartiteGraph<L, R, V>) -> Vec<Matching<L, R>>
where
    L: Ord + Clone,
    R: Ord + Clone + Hash,
    V: Clone,
{
    let mut out = Vec::new();
    let _ = for_each_maximum_matching(graph, &mut |m| {
        out.push(m.clone());
        ControlFlow::Continue(())
    });
    out
}

fn left_of<L: Clone, R>(n: &Node<L, R>) -> L {
    match n {
        Node::Left(l) => l.clone(),
        Node::Right(_) => unreachable!("expected a left node"),
    }
}

fn right_of<L, R: Clone>(n: &Node<L, R>) -> R {
    match n {
        Node::Right(r) => r.clone(),
        Node::Left(_) => unreachable!("expected a right node"),
    }
}

fn enum_rec<L, R, V>(
    graph: &mut BipartiteGraph<L, R, V>,
    matching: &Matching<L, R>,
    dmg: &DirectedGraph<Node<L, R>>,
    k: &mut dyn FnMut(&Matching<L, R>) -> ControlFlow<()>,
) -> ControlFlow<()>
where
    L: Ord + Clone,
    R: Ord + Clone + Hash,
    V: Clone,
{
    // Step 1
    if graph.is_empty() {
        return ControlFlow::Continue(());
    }

    // Step 2: a cycle in D(G, M); it alternates between left and right nodes.
    let raw_cycle = dmg.find_cycle();
    if !raw_cycle.is_empty() {
        // Make the cycle start on the left.
        let cycle: Vec<Node<L, R>> = if matches!(raw_cycle[0], Node::Right(_)) {
            let mut c = vec![raw_cycle[raw_cycle.len() - 1].clone()];
            c.extend_from_slice(&raw_cycle[..raw_cycle.len() - 1]);
            c
        } else {
            raw_cycle
        };
        let n = cycle.len();

        // Step 3 (as in Python, simply the first edge of the cycle)
        let (edge_l, edge_r) = (left_of(&cycle[0]), right_of(&cycle[1]));

        // Step 5: flip the edges along the cycle.
        let mut new_match = matching.clone();
        for i in (0..n).step_by(2) {
            new_match.insert(left_of(&cycle[i]), right_of(&cycle[(i + n - 1) % n]));
        }
        k(&new_match)?;

        // Step 7: recurse with M' on G - e.
        let old_value = graph.remove(&edge_l, &edge_r).expect("cycle edge is in the graph");
        let dmg_minus = directed_match_graph(graph, &new_match);
        let flow = enum_rec(graph, &new_match, &dmg_minus, k);
        graph.insert(edge_l.clone(), edge_r.clone(), old_value);
        flow?;

        // Step 6: recurse with M on G + e (the endpoints' other edges removed).
        let removed: Vec<((L, R), V)> = graph
            .edges_with_labels()
            .filter(|((l, r), _)| *l == edge_l || *r == edge_r)
            .map(|(e, v)| (e.clone(), v.clone()))
            .collect();
        for ((l, r), _) in &removed {
            graph.remove(l, r);
        }
        let dmg_plus = directed_match_graph(graph, matching);
        let flow = enum_rec(graph, matching, &dmg_plus, k);
        for ((l, r), v) in removed {
            graph.insert(l, r, v);
        }
        return flow;
    }

    // Step 8: a feasible path left1 -> right -> left2 in D(G, M), with left1
    // matched and left2 unmatched.
    let mut found = None;
    'search: for node in dmg.adjacency.keys() {
        let Node::Left(left1) = node else { continue };
        let Some(right) = matching.get(left1) else { continue };
        if let Some(predecessors) = dmg.adjacency.get(&Node::Right(right.clone())) {
            for n2 in predecessors {
                if let Node::Left(left2) = n2 {
                    if !matching.contains_key(left2) {
                        found = Some((left1.clone(), right.clone(), left2.clone()));
                        break 'search;
                    }
                }
            }
        }
    }
    // M' is M with the path's direction swapped; e is the edge M' gains.
    let (new_match, edge_l, edge_r) = if let Some((left1, right, left2)) = found {
        let mut new_match = matching.clone();
        new_match.remove(&left1);
        new_match.insert(left2.clone(), right.clone());
        (new_match, left2, right)
    } else {
        // Mirror image, missing from the Python version (which therefore never
        // enumerates matchings that differ only in which right vertex stays
        // free): a path right2 -> left1 -> right1 with right2 unmatched.
        let matched_rights: BTreeSet<&R> = matching.values().collect();
        let mut found = None;
        'search_right: for (node, successors) in &dmg.adjacency {
            let Node::Right(right2) = node else { continue };
            if matched_rights.contains(right2) {
                continue;
            }
            for n1 in successors {
                if let Node::Left(left1) = n1 {
                    if matching.contains_key(left1) {
                        found = Some((left1.clone(), right2.clone()));
                        break 'search_right;
                    }
                }
            }
        }
        let Some((left1, right2)) = found else { return ControlFlow::Continue(()) };
        let mut new_match = matching.clone();
        new_match.insert(left1.clone(), right2.clone());
        (new_match, left1, right2)
    };
    k(&new_match)?;

    let mut graph_plus = graph.without_nodes(&edge_l, &edge_r);
    let mut graph_minus = graph.without_edge(&edge_l, &edge_r);
    let dgm_plus = directed_match_graph(&graph_plus, &new_match);
    let dgm_minus = directed_match_graph(&graph_minus, matching);

    // Steps 9 and 10
    enum_rec(&mut graph_plus, &new_match, &dgm_plus, k)?;
    enum_rec(&mut graph_minus, matching, &dgm_minus, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(n: usize, m: usize) -> BipartiteGraph<usize, usize, bool> {
        (0..n).flat_map(|i| (0..m).map(move |j| ((i, j), true))).collect()
    }

    #[test]
    fn completeness() {
        // tests/test_bipartite.py::test_completeness
        for n in 1..6usize {
            for m in 0..4usize {
                if n < m {
                    continue;
                }
                let expected = if m > 0 { (n - m + 1..=n).product::<usize>() } else { 0 };
                assert_eq!(enum_maximum_matchings(&complete(n, m)).len(), expected, "K({n}, {m})");
            }
        }
    }

    /// All matchings of maximum size, by brute force.
    fn brute_force(graph: &BipartiteGraph<usize, usize, bool>) -> BTreeSet<Vec<(usize, usize)>> {
        let edges: Vec<(usize, usize)> = graph.edges().cloned().collect();
        let mut best: BTreeSet<Vec<(usize, usize)>> = BTreeSet::new();
        let mut best_size = 0;
        for mask in 0u32..(1 << edges.len()) {
            let chosen: Vec<(usize, usize)> =
                edges.iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, e)| *e).collect();
            let ls: BTreeSet<_> = chosen.iter().map(|e| e.0).collect();
            let rs: BTreeSet<_> = chosen.iter().map(|e| e.1).collect();
            if ls.len() != chosen.len() || rs.len() != chosen.len() {
                continue;
            }
            if chosen.len() > best_size {
                best_size = chosen.len();
                best.clear();
            }
            if chosen.len() == best_size && best_size > 0 {
                best.insert(chosen);
            }
        }
        best
    }

    /// Every graph with up to 4 + 4 vertices (and at most 12 possible edges): the enumeration
    /// yields exactly the maximum matchings, each once (a stronger version of
    /// the hypothesis test `test_enum_maximum_matchings_iter_correctness`).
    #[test]
    fn exhaustive_small_graphs() {
        for (n, m) in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (1, 3), (3, 3), (4, 2), (4, 3), (2, 4), (3, 4)] {
            let slots: Vec<(usize, usize)> = (0..n).flat_map(|i| (0..m).map(move |j| (i, j))).collect();
            for mask in 0u32..(1 << slots.len()) {
                let graph: BipartiteGraph<usize, usize, bool> =
                    slots.iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, &e)| (e, true)).collect();
                let got = enum_maximum_matchings(&graph);
                let as_sets: BTreeSet<Vec<(usize, usize)>> =
                    got.iter().map(|m| m.iter().map(|(l, r)| (*l, *r)).collect()).collect();
                assert_eq!(as_sets.len(), got.len(), "duplicate matching for {mask:b}");
                assert_eq!(as_sets, brute_force(&graph), "graph {n}x{m} mask {mask:b}");
            }
        }
    }

    #[test]
    fn find_cycle() {
        // tests/test_bipartite.py::test_directed_graph_find_cycle
        type Adjacency = Vec<(u8, Vec<u8>)>;
        let cases: Vec<(Adjacency, Vec<u8>)> = vec![
            (vec![], vec![]),
            (vec![(0, vec![1])], vec![]),
            (vec![(0, vec![1]), (1, vec![2])], vec![]),
            (vec![(0, vec![1]), (1, vec![0])], vec![0, 1]),
            (vec![(0, vec![1]), (1, vec![0, 2])], vec![0, 1]),
            (vec![(0, vec![1, 2]), (1, vec![0, 2])], vec![0, 1]),
            (vec![(0, vec![1, 2]), (1, vec![0])], vec![0, 1]),
            (vec![(0, vec![1]), (1, vec![2]), (2, vec![0])], vec![0, 1, 2]),
            (vec![(0, vec![2]), (1, vec![2])], vec![]),
            (vec![(0, vec![2]), (1, vec![2]), (2, vec![0])], vec![0, 2]),
            (vec![(0, vec![2]), (1, vec![2]), (2, vec![1])], vec![1, 2]),
        ];
        for (adjacency, expected) in cases {
            let g =
                DirectedGraph { adjacency: adjacency.iter().map(|(n, s)| (*n, s.iter().copied().collect())).collect() };
            let mut cycle = g.find_cycle();
            if let Some(first) = expected.first() {
                let start = cycle.iter().position(|n| n == first).expect("cycle contains the expected start");
                cycle.rotate_left(start);
            }
            assert_eq!(cycle, expected, "{adjacency:?}");
        }
    }

    #[test]
    fn graph_operations() {
        let mut g: BipartiteGraph<u8, u8, bool> =
            [((0, 0), true), ((1, 0), true), ((1, 1), true), ((0, 1), true)].into_iter().collect();
        let set = |xs: &[u8]| xs.iter().copied().collect::<BTreeSet<u8>>();
        let edges = |g: &BipartiteGraph<u8, u8, bool>| g.edges().cloned().collect::<Vec<_>>();
        assert_eq!(edges(&g.limited_to(&set(&[0]), &set(&[0]))), vec![(0, 0)]);
        assert_eq!(edges(&g.limited_to(&set(&[0, 1]), &set(&[1]))), vec![(0, 1), (1, 1)]);
        assert_eq!(g.limited_to(&set(&[0, 1]), &set(&[0, 1])), g);
        assert_eq!(edges(&g.without_nodes(&0, &0)), vec![(1, 1)]);
        assert_eq!(edges(&g.without_edge(&0, &0)), vec![(0, 1), (1, 0), (1, 1)]);
        assert_eq!(g.remove(&0, &0), Some(true));
        assert_eq!(g.remove(&0, &0), None);
        assert_eq!(g.neighbors_of_left(&0), Some(&set(&[1])));
        assert_eq!(g.neighbors_of_right(&0), Some(&set(&[1])));
        *g.get_or_insert_with(2, 2, || false) = true;
        assert_eq!(g.get(&2, &2), Some(&true));
        assert_eq!(g.len(), 4);
    }
}
