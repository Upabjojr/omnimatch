//! Maximum bipartite matching with the Hopcroft-Karp algorithm.
//!
//! Port of `omnimatch/matching/hopcroft_karp.py`. Like the Python version, it
//! is deterministic: vertices are visited in the order given.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;

const INF: i64 = -1; // Python: FAKE_INFINITY

/// Hopcroft-Karp on a graph given as left vertices with their adjacent right
/// vertices (an ordered list, so results are reproducible).
#[derive(Debug, Clone)]
pub struct HopcroftKarp<L, R> {
    left: Vec<L>,
    right: Vec<R>,
    adjacency: Vec<Vec<usize>>,
}

impl<L: Clone + Ord, R: Clone + Eq + Hash + Ord> HopcroftKarp<L, R> {
    /// `graph` maps each left vertex to its neighbours (without duplicates).
    pub fn new(graph: Vec<(L, Vec<R>)>) -> Self {
        let mut right = Vec::new();
        let mut right_index: HashMap<R, usize> = HashMap::new();
        let mut left = Vec::with_capacity(graph.len());
        let mut adjacency = Vec::with_capacity(graph.len());
        for (l, rs) in graph {
            left.push(l);
            adjacency.push(
                rs.into_iter()
                    .map(|r| {
                        *right_index.entry(r.clone()).or_insert_with(|| {
                            right.push(r);
                            right.len() - 1
                        })
                    })
                    .collect(),
            );
        }
        HopcroftKarp { left, right, adjacency }
    }

    /// The size of a maximum matching and one such matching.
    pub fn get_maximum_matching_num(&self) -> (usize, BTreeMap<L, R>) {
        let (count, pair_left) = run(&self.adjacency, self.right.len());
        let matching = pair_left
            .iter()
            .enumerate()
            .filter_map(|(l, r)| r.map(|r| (self.left[l].clone(), self.right[r].clone())))
            .collect();
        (count, matching)
    }

    /// One maximum matching.
    pub fn get_maximum_matching(&self) -> BTreeMap<L, R> {
        self.get_maximum_matching_num().1
    }
}

struct State<'a> {
    adjacency: &'a [Vec<usize>],
    pair_left: Vec<Option<usize>>,
    pair_right: Vec<Option<usize>>,
    dist: Vec<i64>,
    reference_distance: i64,
}

/// Hopcroft-Karp on vertex indices; returns the matching size and each left
/// vertex's partner.
pub(crate) fn run(adjacency: &[Vec<usize>], n_right: usize) -> (usize, Vec<Option<usize>>) {
    let mut st = State {
        adjacency,
        pair_left: vec![None; adjacency.len()],
        pair_right: vec![None; n_right],
        dist: vec![INF; adjacency.len()],
        reference_distance: INF,
    };
    let mut matchings = 0;
    while st.bfs() {
        for left in 0..adjacency.len() {
            if st.pair_left[left].is_none() && st.dfs(left) {
                matchings += 1;
            }
        }
    }
    (matchings, st.pair_left)
}

impl State<'_> {
    fn bfs(&mut self) -> bool {
        let mut queue = VecDeque::new();
        for left in 0..self.adjacency.len() {
            if self.pair_left[left].is_none() {
                queue.push_back(left);
                self.dist[left] = 0;
            } else {
                self.dist[left] = INF;
            }
        }
        self.reference_distance = INF;
        while let Some(left) = queue.pop_front() {
            if self.dist[left] == INF && self.reference_distance == INF {
                continue;
            }
            if self.reference_distance != INF && self.dist[left] >= self.reference_distance {
                continue;
            }
            for &right in &self.adjacency[left] {
                match self.pair_right[right] {
                    None => {
                        if self.reference_distance == INF {
                            self.reference_distance = self.dist[left] + 1;
                        }
                    }
                    Some(other) => {
                        if self.dist[other] == INF {
                            self.dist[other] = self.dist[left] + 1;
                            queue.push_back(other);
                        }
                    }
                }
            }
        }
        self.reference_distance != INF
    }

    fn dfs(&mut self, left: usize) -> bool {
        for i in 0..self.adjacency[left].len() {
            let right = self.adjacency[left][i];
            let found = match self.pair_right[right] {
                None => self.reference_distance == self.dist[left] + 1,
                Some(other) => self.dist[other] == self.dist[left] + 1 && self.dfs(other),
            };
            if found {
                self.pair_left[left] = Some(right);
                self.pair_right[right] = Some(left);
                return true;
            }
        }
        self.dist[left] = INF;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<L: Clone + Ord + std::fmt::Debug, R: Clone + Eq + Hash + Ord + std::fmt::Debug>(
        graph: Vec<(L, Vec<R>)>,
        expected: Vec<(L, R)>,
    ) {
        let (n, m) = HopcroftKarp::new(graph).get_maximum_matching_num();
        assert_eq!(m, expected.iter().cloned().collect::<BTreeMap<_, _>>());
        assert_eq!(n, expected.len());
    }

    /// The cases from `tests/test_hopcroft_karp.py`, with the same expected matchings.
    #[test]
    fn python_cases() {
        check(
            vec![
                (0, vec!["v0", "v1"]),
                (1, vec!["v0", "v4"]),
                (2, vec!["v2", "v3"]),
                (3, vec!["v0", "v4"]),
                (4, vec!["v0", "v3"]),
            ],
            vec![(0, "v1"), (1, "v4"), (2, "v2"), (3, "v0"), (4, "v3")],
        );
        check(
            vec![
                ("A", vec![1, 2]),
                ("B", vec![2, 3]),
                ("C", vec![2]),
                ("D", vec![3, 4, 5, 6]),
                ("E", vec![4, 7]),
                ("F", vec![7]),
                ("G", vec![7]),
            ],
            vec![("A", 1), ("B", 3), ("C", 2), ("D", 5), ("E", 4), ("F", 7)],
        );
        check(
            vec![(1, vec!['a', 'c']), (2, vec!['a', 'c']), (3, vec!['c', 'b']), (4, vec!['e'])],
            vec![(1, 'a'), (2, 'c'), (3, 'b'), (4, 'e')],
        );
        check(
            vec![
                ("A", vec![3, 4]),
                ("B", vec![3, 4]),
                ("C", vec![3]),
                ("D", vec![1, 5, 7]),
                ("E", vec![1, 2, 7]),
                ("F", vec![2, 8]),
                ("G", vec![6]),
                ("H", vec![2, 4, 8]),
            ],
            vec![("A", 3), ("B", 4), ("D", 1), ("E", 7), ("F", 8), ("G", 6), ("H", 2)],
        );
    }
}
