//! Combinatorial helpers used by the matchers.
//!
//! Port of the numeric/combinatorial parts of `omnimatch/utils.py`. The Python
//! generators become functions returning `Vec`s, in the same order. The
//! Python-specific helpers (`get_short_lambda_source`, cached properties,
//! `generator_chain`) have no Rust counterpart; `generator_chain` is replaced by
//! continuation-passing recursion in the matchers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::expressions::Expr;
use crate::multiset::Multiset;
use crate::substitution::{Replacement, Substitution};

/// All non-negative integer vectors that are componentwise `<= max_vector` and
/// whose components sum to `vector_sum`, in lexicographic order.
pub fn fixed_integer_vector_iter(max_vector: &[usize], vector_sum: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    fn rec(max_vector: &[usize], sum: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        let Some((&first, rest)) = max_vector.split_first() else {
            if sum == 0 {
                out.push(prefix.clone());
            }
            return;
        };
        let total: usize = max_vector.iter().sum();
        if sum > total {
            return;
        }
        let start = (first + sum).saturating_sub(total);
        let end = first.min(sum);
        for j in start..=end {
            prefix.push(j);
            rec(rest, sum - j, prefix, out);
            prefix.pop();
        }
    }
    rec(max_vector, vector_sum, &mut Vec::new(), &mut out);
    out
}

/// All weak compositions of `n` into `num_parts` parts, in lexicographic order.
pub fn weak_composition_iter(n: usize, num_parts: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    fn rec(n: usize, parts: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if parts == 0 {
            if n == 0 {
                out.push(prefix.clone());
            }
            return;
        }
        if parts == 1 {
            prefix.push(n);
            out.push(prefix.clone());
            prefix.pop();
            return;
        }
        for first in 0..=n {
            prefix.push(first);
            rec(n - first, parts - 1, prefix, out);
            prefix.pop();
        }
    }
    rec(n, num_parts, &mut Vec::new(), &mut out);
    out
}

/// All 0/1 vectors of length `count`, in the order of
/// `itertools.product([0, 1], repeat=count)`.
pub fn binary_vectors(count: usize) -> Vec<Vec<usize>> {
    (0..1usize << count).map(|bits| (0..count).map(|i| (bits >> (count - 1 - i)) & 1).collect()).collect()
}

/// Extended Euclidean algorithm: `(x, y, d)` with `a*x + b*y = d = gcd(a, b)`.
pub fn extended_euclid(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (1, 0, a);
    }
    let (x0, y0, d) = extended_euclid(b, a.rem_euclid(b));
    (y0, x0 - a.div_euclid(b) * y0, d)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Non-negative solutions of `a*x + b*y = c` (`a, b > 0`, `c >= 0`).
///
/// # Panics
/// If `a` or `b` is not positive or `c` is negative.
pub fn base_solution_linear(a: i64, b: i64, c: i64) -> Vec<(i64, i64)> {
    assert!(a > 0 && b > 0, "Coefficients a and b must be positive integers.");
    assert!(c >= 0, "Constant c must not be negative.");
    let d = gcd(a, gcd(b, c));
    let (a, b, c) = (a / d, b / d, c / d);
    let mut out = Vec::new();
    if c == 0 {
        out.push((0, 0));
        return out;
    }
    let (x0, y0, d) = extended_euclid(a, b);
    if c % d != 0 {
        return out;
    }
    let (mut x, mut y) = (c * x0, c * y0);
    if x <= 0 {
        while y >= 0 {
            if x >= 0 {
                out.push((x, y));
            }
            x += b;
            y -= a;
        }
    } else {
        while x >= 0 {
            if y >= 0 {
                out.push((x, y));
            }
            x -= b;
            y += a;
        }
    }
    out
}

/// Non-negative integer solutions of `c_1 x_1 + ... + c_n x_n = total`.
pub fn solve_linear_diop(total: usize, coeffs: &[usize]) -> Vec<Vec<usize>> {
    let total = total as i64;
    let coeffs: Vec<i64> = coeffs.iter().map(|&c| c as i64).collect();
    solve_rec(total, &coeffs).into_iter().map(|v| v.into_iter().map(|x| x as usize).collect()).collect()
}

fn solve_rec(total: i64, coeffs: &[i64]) -> Vec<Vec<i64>> {
    match coeffs.len() {
        0 => {
            if total == 0 {
                vec![vec![]]
            } else {
                vec![]
            }
        }
        1 => {
            if total % coeffs[0] == 0 {
                vec![vec![total / coeffs[0]]]
            } else {
                vec![]
            }
        }
        2 => base_solution_linear(coeffs[0], coeffs[1], total).into_iter().map(|(x, y)| vec![x, y]).collect(),
        _ => {
            let remainder_gcd = coeffs[1..].iter().fold(0, |g, &c| gcd(g, c));
            let new_coeffs: Vec<i64> = coeffs[1..].iter().map(|c| c / remainder_gcd).collect();
            let mut out = Vec::new();
            for (x0, rest_total) in base_solution_linear(coeffs[0], remainder_gcd, total) {
                for mut rest in solve_rec(rest_total, &new_coeffs) {
                    rest.insert(0, x0);
                    out.push(rest);
                }
            }
            out
        }
    }
}

type Solutions = Rc<Vec<Vec<usize>>>;

thread_local! {
    /// Memoized `solve_linear_diop` results (Python: `_linear_diop_solution_cache`).
    static DIOP_CACHE: RefCell<HashMap<(usize, Vec<usize>), Solutions>> = RefCell::new(HashMap::new());
}

fn solve_linear_diop_cached(total: usize, coeffs: &[usize]) -> Solutions {
    let key = (total, coeffs.to_vec());
    if let Some(hit) = DIOP_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit;
    }
    let solutions = Rc::new(solve_linear_diop(total, coeffs));
    DIOP_CACHE.with(|c| c.borrow_mut().insert(key, solutions.clone()));
    solutions
}

/// A sequence variable occurring `count` times in a commutative pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariableWithCount {
    /// `None` for an unnamed wildcard.
    pub name: Option<String>,
    pub count: usize,
    pub minimum: usize,
    pub default: Option<Expr>,
}

impl VariableWithCount {
    pub fn new(name: Option<&str>, count: usize, minimum: usize, default: Option<Expr>) -> Self {
        VariableWithCount { name: name.map(str::to_owned), count, minimum, default }
    }
}

/// All ways to distribute the multiset `values` among the sequence `variables`
/// (each occurring `count` times, needing at least `minimum` values).
///
/// Unnamed variables take part in the distribution but are left out of the
/// returned substitutions. Results are in no particular order.
pub fn commutative_sequence_variable_partition_iter(
    values: &Multiset<Expr>,
    variables: &[VariableWithCount],
) -> Vec<Substitution> {
    if variables.len() == 1 {
        return single_variable_partition(values, &variables[0]).into_iter().collect();
    }
    let items: Vec<(Expr, usize)> = values.items().map(|(e, &n)| (e.clone(), n)).collect();
    let counts: Vec<usize> = variables.iter().map(|v| v.count).collect();
    let mut assignment: Vec<Multiset<Expr>> = vec![Multiset::new(); variables.len()];
    let mut out = Vec::new();

    fn rec(
        idx: usize,
        items: &[(Expr, usize)],
        counts: &[usize],
        variables: &[VariableWithCount],
        assignment: &mut Vec<Multiset<Expr>>,
        out: &mut Vec<Substitution>,
    ) {
        if idx == items.len() {
            let mut subst = Substitution::new();
            for (var, assigned) in variables.iter().zip(assignment.iter()) {
                let value = match (&var.default, assigned.is_empty()) {
                    (Some(default), true) => Replacement::Single(default.clone()),
                    _ if assigned.len() < var.minimum => return,
                    _ => Replacement::Multiset(assigned.clone()),
                };
                if let Some(name) = &var.name {
                    subst.insert(name.clone(), value);
                }
            }
            out.push(subst);
            return;
        }
        let (value, total) = &items[idx];
        for solution in solve_linear_diop_cached(*total, counts).iter() {
            for (assigned, &n) in assignment.iter_mut().zip(solution) {
                assigned.set_count(value.clone(), n);
            }
            rec(idx + 1, items, counts, variables, assignment, out);
        }
        for assigned in assignment.iter_mut() {
            assigned.remove_all(value);
        }
    }

    rec(0, &items, &counts, variables, &mut assignment, &mut out);
    out
}

fn single_variable_partition(values: &Multiset<Expr>, var: &VariableWithCount) -> Option<Substitution> {
    let named = |value: Replacement| {
        let mut s = Substitution::new();
        if let Some(name) = &var.name {
            s.insert(name.clone(), value);
        }
        s
    };
    if values.is_empty() {
        if let Some(default) = &var.default {
            return Some(named(Replacement::Single(default.clone())));
        }
    }
    if var.count == 1 {
        return (values.len() >= var.minimum).then(|| named(Replacement::Multiset(values.clone())));
    }
    let mut new_values = Multiset::new();
    for (element, &n) in values.items() {
        if n % var.count != 0 {
            return None;
        }
        new_values.add_n(element.clone(), n / var.count);
    }
    (new_values.len() >= var.minimum).then(|| named(Replacement::Multiset(new_values)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_integer_vectors() {
        assert_eq!(fixed_integer_vector_iter(&[2, 2], 2), vec![vec![0, 2], vec![1, 1], vec![2, 0]]);
        assert_eq!(fixed_integer_vector_iter(&[], 0), vec![Vec::<usize>::new()]);
        assert!(fixed_integer_vector_iter(&[1, 1], 3).is_empty());
    }

    #[test]
    fn weak_compositions() {
        assert_eq!(
            weak_composition_iter(5, 2),
            vec![vec![0, 5], vec![1, 4], vec![2, 3], vec![3, 2], vec![4, 1], vec![5, 0]]
        );
        assert_eq!(weak_composition_iter(0, 0), vec![Vec::<usize>::new()]);
        assert!(weak_composition_iter(1, 0).is_empty());
        assert_eq!(weak_composition_iter(3, 3).len(), 10);
    }

    #[test]
    fn binary() {
        assert_eq!(binary_vectors(2), vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
        assert_eq!(binary_vectors(0), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn euclid() {
        assert_eq!(extended_euclid(42, 12), (1, -3, 6));
    }

    #[test]
    fn diophantine_matches_brute_force() {
        for coeffs in [vec![1], vec![2, 3], vec![1, 2, 3], vec![2, 4, 6], vec![3, 5, 7, 2]] {
            for total in 0..20usize {
                let mut got = solve_linear_diop(total, &coeffs);
                got.sort();
                fn brute(rest: usize, coeffs: &[usize], prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
                    match coeffs.split_first() {
                        None if rest == 0 => out.push(prefix.clone()),
                        None => {}
                        Some((&c, tail)) => {
                            for x in 0..=rest / c {
                                prefix.push(x);
                                brute(rest - x * c, tail, prefix, out);
                                prefix.pop();
                            }
                        }
                    }
                }
                let mut expected = Vec::new();
                brute(total, &coeffs, &mut Vec::new(), &mut expected);
                assert_eq!(got, expected, "coeffs {coeffs:?} total {total}");
            }
        }
    }

    #[test]
    fn partition_doc_example() {
        let x = VariableWithCount::new(Some("x"), 1, 1, None);
        let y = VariableWithCount::new(Some("y"), 2, 0, None);
        let values: Multiset<Expr> = "aaabbc".chars().map(|c| Expr::atom(c.to_string())).collect();
        let mut got: Vec<String> =
            commutative_sequence_variable_partition_iter(&values, &[x, y]).iter().map(|s| s.to_string()).collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                "{x ↦ {a, a, a, b, b, c}, y ↦ {}}",
                "{x ↦ {a, a, a, c}, y ↦ {b}}",
                "{x ↦ {a, b, b, c}, y ↦ {a}}",
                "{x ↦ {a, c}, y ↦ {a, b}}",
            ]
        );
    }
}
