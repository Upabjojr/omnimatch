//! Port of the numeric parts of `tests/test_utils.py`.
//!
//! The hypothesis-based tests become exhaustive loops over small ranges.
//! Not ported (Python-only helpers): `test_get_short_lambda_source`,
//! `test_cached_property`, `test_slot_cached_property`, and the negative-argument
//! errors of `weak_composition_iter` (its arguments are `usize`).

#![allow(clippy::type_complexity)]

use std::collections::BTreeSet;
use std::panic;

use omnimatch::utils::*;
use omnimatch::*;

fn is_unique<T: Ord + Clone>(v: &[T]) -> bool {
    v.iter().cloned().collect::<BTreeSet<_>>().len() == v.len()
}

/// All vectors with `len` components, each in `0..=max`.
fn all_vectors(len: usize, max: usize) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for _ in 0..len {
        out = out.into_iter().flat_map(|v| (0..=max).map(move |x| [v.clone(), vec![x]].concat())).collect();
    }
    out
}

#[test]
fn fixed_integer_vectors() {
    for len in 0..=3 {
        for max_vector in all_vectors(len, 3) {
            for sum in 0..=10 {
                let got = fixed_integer_vector_iter(&max_vector, sum);
                // Correctness and completeness against brute force, in lexicographic order.
                let expected: Vec<Vec<usize>> = all_vectors(len, 3)
                    .into_iter()
                    .filter(|v| v.iter().sum::<usize>() == sum && v.iter().zip(&max_vector).all(|(x, m)| x <= m))
                    .collect();
                assert_eq!(got, expected, "{max_vector:?} sum {sum}");
            }
        }
    }
    // Python's hypothesis test used sums >= 50, which have no solutions here.
    assert!(fixed_integer_vector_iter(&[10, 10, 10, 10, 10], 51).is_empty());
}

#[test]
fn extended_euclid_identity() {
    for a in 1..=60i64 {
        for b in 1..=60i64 {
            let (x, y, d) = extended_euclid(a, b);
            assert_eq!(a % d, 0);
            assert_eq!(b % d, 0);
            assert_eq!(a * x + b * y, d, "{a} {b}");
        }
    }
}

#[test]
fn base_solution_linear_correct_complete_unique() {
    for a in 1..=12i64 {
        for b in 1..=12i64 {
            for c in 0..=40i64 {
                let solutions = base_solution_linear(a, b, c);
                for &(x, y) in &solutions {
                    assert!(x >= 0 && y >= 0);
                    assert_eq!(a * x + b * y, c, "invalid solution {x},{y} for {a} {b} {c}");
                }
                assert!(is_unique(&solutions), "duplicate solution for {a} {b} {c}");
                let set: BTreeSet<_> = solutions.into_iter().collect();
                for x in 0..=c {
                    for y in 0..=c {
                        if a * x + b * y == c {
                            assert!(set.contains(&(x, y)), "missing {x},{y} for {a} {b} {c}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn base_solution_linear_errors() {
    for (a, b, c) in [(0, 1, 1), (-1, 1, 1), (1, 0, 1), (1, -1, 1), (1, 1, -1)] {
        let result = panic::catch_unwind(|| base_solution_linear(a, b, c));
        assert!(result.is_err(), "({a}, {b}, {c}) should be rejected");
    }
}

#[test]
fn weak_compositions() {
    for n in 0..=10usize {
        for m in 0..4usize {
            let parts = weak_composition_iter(n, m);
            for part in &parts {
                assert_eq!(part.len(), m);
                assert_eq!(part.iter().sum::<usize>(), n);
            }
            // (n+m-1)! / ((m-1)! n!)
            let expected = if m == 0 {
                usize::from(n == 0)
            } else {
                let mut count = 1;
                for i in 1..m {
                    count = count * (n + m - i) / i;
                }
                count
            };
            assert_eq!(parts.len(), expected, "n={n} m={m}");
            assert!(is_unique(&parts));
        }
    }
}

/// Brute-force number of non-negative solutions of `coeffs · x = total`.
fn count_solutions(total: usize, coeffs: &[usize]) -> usize {
    match coeffs.split_first() {
        None => usize::from(total == 0),
        Some((&k, rest)) => (0..=total / k).map(|x| count_solutions(total - x * k, rest)).sum(),
    }
}

#[test]
fn solve_linear_diop_correct_complete_unique() {
    let coefficient_choices = [1usize, 2, 3, 5];
    let mut coefficient_lists: Vec<Vec<usize>> = vec![vec![], vec![1, 2, 2]];
    for len in 1..=4 {
        for indices in all_vectors(len, coefficient_choices.len() - 1) {
            coefficient_lists.push(indices.iter().map(|&i| coefficient_choices[i]).collect());
        }
    }
    for coeffs in &coefficient_lists {
        for c in 0..=12usize {
            let solutions = solve_linear_diop(c, coeffs);
            for s in &solutions {
                assert_eq!(s.len(), coeffs.len());
                assert_eq!(s.iter().zip(coeffs).map(|(x, k)| x * k).sum::<usize>(), c, "{coeffs:?} {c}");
            }
            assert!(is_unique(&solutions), "duplicates for {coeffs:?} {c}");
            let expected = count_solutions(c, coeffs);
            assert_eq!(solutions.len(), expected, "{coeffs:?} {c}");
        }
    }
}

fn values(s: &str) -> Multiset<Expr> {
    s.chars().map(|ch| Expr::atom(ch.to_string())).collect()
}

/// Every result must give each variable at least its minimum and, counting
/// multiplicities, partition `values` exactly. Returns the number of results.
fn check_partitions(variables: &[VariableWithCount], values: &Multiset<Expr>) -> usize {
    let results = commutative_sequence_variable_partition_iter(values, variables);
    for subst in &results {
        assert_eq!(subst.len(), variables.len(), "wrong number of variables in {subst}");
        let mut union = Multiset::new();
        for var in variables {
            let Some(Replacement::Multiset(m)) = subst.get(var.name.as_deref().unwrap()) else {
                panic!("{subst} has no multiset for {:?}", var.name)
            };
            assert!(m.len() >= var.minimum, "{subst}: {:?} below its minimum", var.name);
            union = union.combine(&m.times(var.count));
        }
        assert_eq!(&union, values, "{subst} is not a partition");
    }
    results.len()
}

fn variables(spec: &[(usize, usize)]) -> Vec<VariableWithCount> {
    spec.iter()
        .enumerate()
        .map(|(i, &(count, minimum))| VariableWithCount::new(Some(&format!("var{i}")), count, minimum, None))
        .collect()
}

#[test]
fn partition_expected_counts() {
    // (variables as (count, minimum), values, expected number of results)
    let cases: &[(&[(usize, usize)], &str, usize)] = &[
        (&[], "a", 0),
        (&[], "", 1),
        (&[(1, 0)], "", 1),
        (&[(1, 0)], "a", 1),
        (&[(1, 1)], "", 0),
        (&[(1, 1)], "a", 1),
        (&[(1, 2)], "a", 0),
        (&[(2, 0)], "", 1),
        (&[(2, 0)], "a", 0),
        (&[(2, 1)], "", 0),
        (&[(2, 1)], "a", 0),
        (&[(2, 2)], "a", 0),
        (&[(2, 0)], "ab", 0),
        (&[(2, 1)], "ab", 0),
        (&[(2, 2)], "ab", 0),
        (&[(2, 0)], "aa", 1),
        (&[(2, 1)], "aa", 1),
        (&[(2, 2)], "aa", 0),
        (&[(2, 0)], "aaa", 0),
        (&[(2, 1)], "aaa", 0),
        (&[(2, 2)], "aaa", 0),
        (&[(2, 0)], "aabb", 1),
        (&[(2, 1)], "aabb", 1),
        (&[(2, 2)], "aabb", 1),
        (&[(1, 0), (1, 0)], "", 1),
        (&[(1, 0), (1, 0)], "a", 2),
        (&[(1, 1), (1, 0)], "", 0),
        (&[(1, 1), (1, 0)], "a", 1),
        (&[(1, 0), (1, 0)], "aa", 3),
        (&[(1, 1), (1, 0)], "aa", 2),
        (&[(1, 0), (1, 0)], "ab", 4),
        (&[(1, 1), (1, 0)], "ab", 3),
        (&[(1, 0), (1, 0)], "aaa", 4),
        (&[(1, 1), (1, 0)], "aaa", 3),
        (&[(1, 0), (1, 0)], "aab", 6),
        (&[(1, 1), (1, 0)], "aab", 5),
        (&[(2, 0), (1, 0)], "", 1),
        (&[(2, 0), (1, 0)], "aa", 2),
        (&[(2, 1), (1, 0)], "", 0),
        (&[(2, 1), (1, 0)], "aa", 1),
        (&[(2, 0), (1, 0)], "ab", 1),
        (&[(2, 1), (1, 0)], "ab", 0),
        (&[(2, 0), (1, 0)], "aaa", 2),
        (&[(2, 1), (1, 0)], "aaa", 1),
        (&[(2, 0), (1, 0)], "aab", 2),
        (&[(2, 1), (1, 0)], "aab", 1),
    ];
    for &(spec, vals, expected) in cases {
        assert_eq!(check_partitions(&variables(spec), &values(vals)), expected, "{spec:?} {vals:?}");
    }
}

/// Replaces `test_correctness_randomized`: every combination of up to three
/// variables (count 1..=3, minimum 0..=2) over small value multisets.
#[test]
fn partition_exhaustive_small() {
    let value_sets = ["a", "ab", "aab", "aabb", "abc", "aaab", "aabbc"];
    let var_specs: Vec<(usize, usize)> = (1..=3).flat_map(|c| (0..=2).map(move |m| (c, m))).collect();
    let mut checked = 0;
    for n_vars in 1..=3usize {
        for choice in all_vectors(n_vars, var_specs.len() - 1) {
            let spec: Vec<(usize, usize)> = choice.iter().map(|&i| var_specs[i]).collect();
            for vals in value_sets {
                check_partitions(&variables(&spec), &values(vals));
                checked += 1;
            }
        }
    }
    assert!(checked > 5000);
}
