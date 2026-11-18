//! Port of the one-to-one cases from `tests/test_matching.py`.

// The tables mirror the Python ones literally, `&[a.clone()]` included.
#![allow(clippy::cloned_ref_to_slice_refs)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use omnimatch::*;

// ─── fixtures (tests/common.py) ──────────────────────────────────────────────

fn head(name: &str) -> Head {
    let b = Head::builder(name);
    match name {
        "f" | "f2" => b,
        "f_u" => b.arity(Arity::Unary),
        "f_i" => b.one_identity(true),
        "f_c" | "f2_c" => b.commutative(true),
        "f_ci" => b.commutative(true).one_identity(true),
        "f_a" => b.associative(true),
        "f_ac" => b.associative(true).commutative(true),
        _ => panic!("unknown head {name}"),
    }
    .build()
    .unwrap()
}

macro_rules! op {
    ($name:ident) => {
        #[allow(dead_code)]
        fn $name(ops: &[Expr]) -> Expr {
            head(stringify!($name)).call(ops.to_vec())
        }
    };
}
op!(f);
op!(f2);
op!(f_i);
op!(f_c);
op!(f_ci);
op!(f2_c);
op!(f_a);
op!(f_ac);

fn a() -> Expr {
    Expr::atom("a")
}
fn b() -> Expr {
    Expr::atom("b")
}
fn c() -> Expr {
    Expr::atom("c")
}
fn a_x() -> Expr {
    a().with_variable_name(Some("x"))
}
fn _w() -> Expr {
    Expr::dot(None)
}
fn x_() -> Expr {
    make_dot_variable("x")
}
fn y_() -> Expr {
    make_dot_variable("y")
}
fn z_() -> Expr {
    make_dot_variable("z")
}
fn __() -> Expr {
    Expr::plus(None)
}
fn x__() -> Expr {
    make_plus_variable("x")
}
fn y__() -> Expr {
    make_plus_variable("y")
}
fn z__() -> Expr {
    make_plus_variable("z")
}
fn ___() -> Expr {
    Expr::star(None)
}
fn x___() -> Expr {
    make_star_variable("x")
}
fn y___() -> Expr {
    make_star_variable("y")
}
fn z___() -> Expr {
    make_star_variable("z")
}
fn oa_() -> Expr {
    Expr::optional("o", a())
}
fn oa__() -> Expr {
    Expr::wildcard(1, false, Some("o"), Some(a())).unwrap()
}
fn oa___() -> Expr {
    Expr::wildcard(0, false, Some("o"), Some(a())).unwrap()
}
fn o2b_() -> Expr {
    Expr::optional("o2", b())
}

// ─── substitution builders ───────────────────────────────────────────────────

fn one(e: Expr) -> Replacement {
    Replacement::Single(e)
}
fn seq(es: &[Expr]) -> Replacement {
    Replacement::Sequence(es.to_vec())
}
fn ms(es: &[Expr]) -> Replacement {
    Replacement::Multiset(es.iter().cloned().collect())
}
fn sub(pairs: &[(&str, Replacement)]) -> Substitution {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

fn matches(subject: &Expr, pattern: &Expr) -> Vec<Substitution> {
    match_pattern(subject, &Pattern::new(pattern.clone())).unwrap()
}

/// Python's check: every expected match is found and every found match is expected.
fn assert_matches(subject: Expr, pattern: Expr, expected: Vec<Substitution>) {
    let result = matches(&subject, &pattern);
    for e in &expected {
        assert!(result.contains(e), "{subject} / {pattern}: missing {e}; got {}", show(&result));
    }
    for r in &result {
        assert!(expected.contains(r), "{subject} / {pattern}: unexpected {r}; got {}", show(&result));
    }
}

/// Same, and the number of matches must agree too.
fn assert_matches_exactly(subject: Expr, pattern: Expr, expected: Vec<Substitution>) {
    let n = matches(&subject, &pattern).len();
    assert_eq!(n, expected.len(), "{subject} / {pattern}: wrong number of matches");
    assert_matches(subject, pattern, expected);
}

fn show(result: &[Substitution]) -> String {
    format!("[{}]", result.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[test]
fn constant_match() {
    let cases = [
        (a(), a(), true),
        (b(), a(), false),
        (f(&[]), f(&[]), true),
        (f(&[a()]), f(&[]), false),
        (f(&[a()]), f(&[a()]), true),
        (f(&[b()]), f(&[a()]), false),
        (f(&[]), f(&[a()]), false),
        (f2(&[a()]), f(&[a()]), false),
        (f(&[a(), b()]), f(&[a()]), false),
        (f(&[a(), b()]), f(&[a(), b()]), true),
        (f(&[a()]), f(&[a(), b()]), false),
        (f(&[b(), a()]), f(&[a(), b()]), false),
        (f(&[a(), b(), c()]), f(&[a(), b()]), false),
        (f(&[a(), f2(&[b()])]), f(&[a(), b()]), false),
        (f(&[f2(&[a()]), f2(&[b()])]), f(&[a(), b()]), false),
        (f(&[f2(&[a()]), b()]), f(&[a(), b()]), false),
        (f(&[f(&[a(), b()])]), f(&[a(), b()]), false),
        (f2(&[a(), b()]), f(&[a(), b()]), false),
        (f(&[a(), f2(&[b()])]), f(&[a(), f2(&[b()])]), true),
        (f(&[f2(&[a()]), b()]), f(&[f2(&[a()]), b()]), true),
        (f(&[f(&[a(), b()])]), f(&[f(&[a(), b()])]), true),
    ];
    for (subject, pattern, is_match) in cases {
        let expected = if is_match { vec![Substitution::new()] } else { vec![] };
        assert_eq!(matches(&subject, &pattern), expected, "{subject} / {pattern}");
    }
}

#[test]
fn commutative_match() {
    let cases = [
        (f_c(&[]), f_c(&[]), true),
        (f_c(&[a()]), f_c(&[]), false),
        (f_c(&[]), f_c(&[a()]), false),
        (f_c(&[a(), b()]), f_c(&[a(), b()]), true),
        (f_c(&[b(), a()]), f_c(&[a(), b()]), true),
        (f_c(&[b(), a(), c()]), f_c(&[a(), b(), c()]), true),
        (f_c(&[c(), a(), b()]), f_c(&[a(), b(), c()]), true),
        (f_c(&[b(), a(), c()]), f_c(&[c(), b(), a()]), true),
        (f_c(&[b(), a(), a()]), f_c(&[a(), a(), b()]), true),
        (f_c(&[a(), b(), a()]), f_c(&[a(), a(), b()]), true),
        (f_c(&[b(), b(), a()]), f_c(&[a(), a(), b()]), false),
        (f_c(&[c(), a(), f2(&[b()])]), f_c(&[a(), f2(&[b()]), c()]), true),
        (f_c(&[c(), a(), f2(&[b()])]), f_c(&[f2(&[a()]), b(), c()]), false),
        (f2(&[c(), f_c(&[a(), b()])]), f2(&[c(), f_c(&[b(), a()])]), true),
        (f2(&[c(), f_c(&[a(), b()])]), f2(&[f_c(&[a(), b()]), c()]), false),
        (f_c(&[c(), f_c(&[a(), b()])]), f_c(&[f_c(&[a(), b()]), c()]), true),
        (f_c(&[c(), f_c(&[b(), a()])]), f_c(&[f_c(&[a(), b()]), c()]), true),
        (f_c(&[c(), f_c(&[b(), b()])]), f_c(&[f_c(&[a(), b()]), c()]), false),
        (f_c(&[a(), f_c(&[b(), a()])]), f_c(&[f_c(&[a(), b()]), c()]), false),
    ];
    for (subject, pattern, is_match) in cases {
        let expected = if is_match { vec![Substitution::new()] } else { vec![] };
        assert_eq!(matches(&subject, &pattern), expected, "{subject} / {pattern}");
    }
}

#[test]
fn commutative_syntactic_match() {
    let cases = [
        (f_c(&[f(&[a()])]), f_c(&[f(&[x_()])]), 1),
        (f_c(&[f(&[a()]), f(&[b()])]), f_c(&[f(&[x_()]), f(&[y_()])]), 2),
        (f_c(&[f(&[a()]), f(&[b()])]), f_c(&[f(&[x_()]), f(&[x_()])]), 0),
        (f_c(&[f(&[a()]), f(&[a()])]), f_c(&[f(&[x_()]), f(&[x_()])]), 1),
        (f_c(&[f(&[a()]), f(&[b()])]), f_c(&[f(&[x_()]), y_()]), 2),
        (f_c(&[f(&[a()]), f(&[a()])]), f_c(&[f(&[x_()]), x_()]), 0),
        (f_c(&[f(&[a()]), a()]), f_c(&[f(&[x_()]), x_()]), 1),
        (f_c(&[f(&[a()]), a()]), f_c(&[f(&[x_()])]), 0),
        (f_c(&[f(&[a()]), a()]), f_c(&[f(&[x_()]), f(&[y_()])]), 0),
        (f_c(&[f(&[a()]), f(&[a()])]), f_c(&[f(&[x_()]), f(&[y_()]), f(&[z_()])]), 0),
        (f_c(&[f2(&[a()]), f2(&[a()])]), f_c(&[f(&[x_()]), f(&[y_()])]), 0),
        (f2_c(&[f(&[a()]), f(&[a()])]), f_c(&[f(&[x_()]), f(&[y_()])]), 0),
        (f_c(&[f_c(&[a(), b()]), c()]), f_c(&[f_c(&[x__()]), y__()]), 1),
    ];
    for (subject, pattern, count) in cases {
        let result = matches(&subject, &pattern);
        assert_eq!(result.len(), count, "{subject} / {pattern}: {}", show(&result));
        for s in &result {
            assert_eq!(substitute(&pattern, s).unwrap(), one(subject.clone()), "invalid match {s}");
        }
    }
}

#[test]
fn wildcard_dot_match() {
    let cases: Vec<(Expr, Expr, Option<Substitution>)> = vec![
        (a(), x_(), Some(sub(&[("x", one(a()))]))),
        (b(), x_(), Some(sub(&[("x", one(b()))]))),
        (f(&[a()]), f(&[x_()]), Some(sub(&[("x", one(a()))]))),
        (f(&[b()]), f(&[x_()]), Some(sub(&[("x", one(b()))]))),
        (f(&[a()]), x_(), Some(sub(&[("x", one(f(&[a()])))]))),
        (f2(&[a()]), f(&[x_()]), None),
        (f(&[a(), b()]), f(&[x_()]), None),
        (f(&[a(), b()]), f(&[x_(), b()]), Some(sub(&[("x", one(a()))]))),
        (f(&[a(), b()]), f(&[x_(), a()]), None),
        (f(&[a(), b()]), f(&[a(), x_()]), Some(sub(&[("x", one(b()))]))),
        (f(&[a(), b()]), f(&[x_(), x_()]), None),
        (f(&[a(), a()]), f(&[x_(), x_()]), Some(sub(&[("x", one(a()))]))),
        (f(&[a(), b()]), f(&[x_(), y_()]), Some(sub(&[("x", one(a())), ("y", one(b()))]))),
        (f(&[a()]), f(&[x_(), y_()]), None),
        (f(&[a(), b(), c()]), f(&[x_(), y_()]), None),
        (f(&[a(), f2(&[b()])]), f(&[x_(), y_()]), Some(sub(&[("x", one(a())), ("y", one(f2(&[b()])))]))),
        (f(&[a(), f2(&[b()])]), f(&[x_(), f2(&[y_()])]), Some(sub(&[("x", one(a())), ("y", one(b()))]))),
        (f(&[a(), f2(&[b()])]), f(&[x_(), f2(&[x_()])]), None),
        (f(&[a(), f2(&[a()])]), f(&[x_(), f2(&[x_()])]), Some(sub(&[("x", one(a()))]))),
        (f(&[f2(&[a()]), f2(&[b()])]), f(&[x_(), x_()]), None),
        (f(&[f2(&[a()]), f2(&[b()])]), f(&[x_(), y_()]), Some(sub(&[("x", one(f2(&[a()]))), ("y", one(f2(&[b()])))]))),
        (f(&[f2(&[a()]), a()]), f(&[x_(), x_()]), None),
        (f(&[f2(&[a()]), a()]), f(&[f2(&[x_()]), x_()]), Some(sub(&[("x", one(a()))]))),
        (f(&[f(&[a(), b()])]), f(&[x_(), y_()]), None),
        (f(&[f(&[a(), b()])]), f(&[x_()]), Some(sub(&[("x", one(f(&[a(), b()])))]))),
        (f2(&[a(), b()]), f(&[x_(), y_()]), None),
        (f(&[f(&[a(), b()])]), f(&[f(&[x_(), y_()])]), Some(sub(&[("x", one(a())), ("y", one(b()))]))),
    ];
    for (subject, pattern, expected) in cases {
        assert_eq!(matches(&subject, &pattern), expected.into_iter().collect::<Vec<_>>(), "{subject} / {pattern}");
    }
}

#[test]
fn optional_wildcard_match() {
    let o = |r: Replacement| vec![sub(&[("o", r)])];
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (f(&[a()]), f(&[a(), oa_()]), o(one(a()))),
        (f(&[a()]), f(&[a(), oa__()]), o(one(a()))),
        (f(&[a()]), f(&[a(), oa___()]), o(one(a()))),
        (f_a(&[a()]), f_a(&[a(), oa_()]), o(one(a()))),
        (f_a(&[a()]), f_a(&[a(), oa__()]), o(one(a()))),
        (f_a(&[a()]), f_a(&[a(), oa___()]), o(one(a()))),
        (f(&[a(), b()]), f(&[a(), oa_()]), o(one(b()))),
        (f(&[a(), b()]), f(&[a(), oa__()]), o(seq(&[b()]))),
        (f(&[a(), b()]), f(&[a(), oa___()]), o(seq(&[b()]))),
        (f_a(&[a(), b()]), f_a(&[a(), oa_()]), o(one(b()))),
        (f_a(&[a(), b()]), f_a(&[a(), oa__()]), o(seq(&[b()]))),
        (f_a(&[a(), b()]), f_a(&[a(), oa___()]), o(seq(&[b()]))),
        (f(&[a(), b(), c()]), f(&[a(), oa_()]), vec![]),
        (f(&[a(), b(), c()]), f(&[a(), oa__()]), o(seq(&[b(), c()]))),
        (f(&[a(), b(), c()]), f(&[a(), oa___()]), o(seq(&[b(), c()]))),
        (f_a(&[a(), b(), c()]), f_a(&[a(), oa_()]), o(one(f_a(&[b(), c()])))),
        (f_a(&[a(), b(), c()]), f_a(&[a(), oa__()]), o(seq(&[b(), c()]))),
        (f_a(&[a(), b(), c()]), f_a(&[a(), oa___()]), o(seq(&[b(), c()]))),
        (f_c(&[a()]), f_c(&[a(), oa_()]), o(one(a()))),
        (f_c(&[a()]), f_c(&[a(), oa__()]), o(one(a()))),
        (f_c(&[a()]), f_c(&[a(), oa___()]), o(one(a()))),
        (f_ac(&[a()]), f_ac(&[a(), oa_()]), o(one(a()))),
        (f_ac(&[a()]), f_ac(&[a(), oa__()]), o(one(a()))),
        (f_ac(&[a()]), f_ac(&[a(), oa___()]), o(one(a()))),
        (f_c(&[a(), b()]), f_c(&[a(), oa_()]), o(one(b()))),
        (f_c(&[a(), b()]), f_c(&[a(), oa__()]), o(ms(&[b()]))),
        (f_c(&[a(), b()]), f_c(&[a(), oa___()]), o(ms(&[b()]))),
        (f_ac(&[a(), b()]), f_ac(&[a(), oa_()]), o(one(b()))),
        (f_ac(&[a(), b()]), f_ac(&[a(), oa__()]), o(ms(&[b()]))),
        (f_ac(&[a(), b()]), f_ac(&[a(), oa___()]), o(ms(&[b()]))),
        (f_c(&[a(), b(), c()]), f_c(&[a(), oa_()]), vec![]),
        (f_c(&[a(), b(), c()]), f_c(&[a(), oa__()]), o(ms(&[b(), c()]))),
        (f_c(&[a(), b(), c()]), f_c(&[a(), oa___()]), o(ms(&[b(), c()]))),
        (f_ac(&[a(), b(), c()]), f_ac(&[a(), oa_()]), o(one(f_ac(&[b(), c()])))),
        (f_ac(&[a(), b(), c()]), f_ac(&[a(), oa__()]), o(ms(&[b(), c()]))),
        (f_ac(&[a(), b(), c()]), f_ac(&[a(), oa___()]), o(ms(&[b(), c()]))),
        (f(&[a()]), f(&[a(), oa_(), oa_()]), o(one(a()))),
        (f(&[a()]), f(&[a(), oa_(), o2b_()]), vec![sub(&[("o", one(a())), ("o2", one(b()))])]),
        (f(&[a(), a()]), f(&[a(), oa_(), oa_()]), o(one(a()))),
        (f(&[a(), b()]), f(&[a(), oa_(), oa_()]), vec![]),
        (f(&[f(&[a()])]), f(&[f(&[oa_()]), oa_()]), o(one(a()))),
        (f(&[f(&[b()])]), f(&[f(&[oa_()]), oa_()]), vec![]),
        (f(&[f(&[b()]), b()]), f(&[f(&[oa_()]), oa_()]), o(one(b()))),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

#[test]
fn associative_wildcard_dot_match() {
    let x = |r: Replacement| sub(&[("x", r)]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (f_a(&[a()]), f_a(&[x_()]), vec![x(one(a()))]),
        (f_a(&[a(), b()]), f_a(&[x_()]), vec![x(one(f_a(&[a(), b()])))]),
        (f_a(&[a(), b()]), f_a(&[a(), x_()]), vec![x(one(b()))]),
        (f_a(&[a(), b(), c()]), f_a(&[a(), x_()]), vec![x(one(f_a(&[b(), c()])))]),
        (f_a(&[a(), b(), c()]), f_a(&[x_(), c()]), vec![x(one(f_a(&[a(), b()])))]),
        (f_a(&[a(), b(), c()]), f_a(&[x_()]), vec![x(one(f_a(&[a(), b(), c()])))]),
        (f_a(&[a(), b(), a(), b()]), f_a(&[x_(), x_()]), vec![x(one(f_a(&[a(), b()])))]),
        (f_a(&[a(), b(), a()]), f_a(&[x_(), b(), x_()]), vec![x(one(a()))]),
        (f_a(&[a(), a(), b(), a(), a()]), f_a(&[x_(), b(), x_()]), vec![x(one(f_a(&[a(), a()])))]),
        (
            f_a(&[a(), b(), c()]),
            f_a(&[x_(), y_()]),
            vec![
                sub(&[("x", one(a())), ("y", one(f_a(&[b(), c()])))]),
                sub(&[("x", one(f_a(&[a(), b()]))), ("y", one(c()))]),
            ],
        ),
        (f_a(&[a()]), f_a(&[_w()]), vec![Substitution::new()]),
        (f_a(&[a(), b()]), f_a(&[_w()]), vec![Substitution::new()]),
        // associative + commutative
        (f_ac(&[a()]), f_ac(&[_w()]), vec![Substitution::new()]),
        (f_ac(&[a()]), f_ac(&[x_()]), vec![x(one(a()))]),
        (f_ac(&[a(), b()]), f_ac(&[_w()]), vec![Substitution::new()]),
        (f_ac(&[a(), b()]), f_ac(&[x_()]), vec![x(one(f_ac(&[a(), b()])))]),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

fn xy(x: &[Expr], y: &[Expr]) -> Substitution {
    sub(&[("x", seq(x)), ("y", seq(y))])
}
fn xs(x: &[Expr]) -> Substitution {
    sub(&[("x", seq(x))])
}

#[test]
fn wildcard_star_match() {
    let (a, b, c) = (a(), b(), c());
    let f2b = f2(&[b.clone()]);
    let f2a = f2(&[a.clone()]);
    let fab = f(&[a.clone(), b.clone()]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (a.clone(), x___(), vec![xs(&[a.clone()])]),
        (f(&[a.clone()]), f(&[x___()]), vec![xs(&[a.clone()])]),
        (f(&[]), f(&[x___()]), vec![xs(&[])]),
        (f(&[a.clone()]), x___(), vec![xs(&[f(&[a.clone()])])]),
        (f2(&[a.clone()]), f(&[x___()]), vec![]),
        (fab.clone(), f(&[x___()]), vec![xs(&[a.clone(), b.clone()])]),
        (fab.clone(), f(&[x___(), b.clone()]), vec![xs(&[a.clone()])]),
        (fab.clone(), f(&[x___(), a.clone()]), vec![]),
        (fab.clone(), f(&[a.clone(), x___()]), vec![xs(&[b.clone()])]),
        (fab.clone(), f(&[x___(), x___()]), vec![]),
        (f(&[a.clone(), a.clone()]), f(&[x___(), x___()]), vec![xs(&[a.clone()])]),
        (
            fab.clone(),
            f(&[x___(), y___()]),
            vec![xy(&[], &[a.clone(), b.clone()]), xy(&[a.clone()], &[b.clone()]), xy(&[a.clone(), b.clone()], &[])],
        ),
        (f(&[a.clone()]), f(&[x___(), y___()]), vec![xy(&[], &[a.clone()]), xy(&[a.clone()], &[])]),
        (
            f(&[a.clone(), b.clone(), c.clone()]),
            f(&[x___(), y___()]),
            vec![
                xy(&[], &[a.clone(), b.clone(), c.clone()]),
                xy(&[a.clone()], &[b.clone(), c.clone()]),
                xy(&[a.clone(), b.clone()], &[c.clone()]),
                xy(&[a.clone(), b.clone(), c.clone()], &[]),
            ],
        ),
        (
            f(&[a.clone(), f2b.clone()]),
            f(&[x___(), y___()]),
            vec![
                xy(&[], &[a.clone(), f2b.clone()]),
                xy(&[a.clone()], &[f2b.clone()]),
                xy(&[a.clone(), f2b.clone()], &[]),
            ],
        ),
        (f(&[a.clone(), f2b.clone()]), f(&[x___(), f2(&[y___()])]), vec![xy(&[a.clone()], &[b.clone()])]),
        (f(&[a.clone(), f2b.clone()]), f(&[x___(), f2(&[x___()])]), vec![]),
        (f(&[a.clone(), f2a.clone()]), f(&[x___(), f2(&[x___()])]), vec![xs(&[a.clone()])]),
        (f(&[f2a.clone(), f2b.clone()]), f(&[x___(), x___()]), vec![]),
        (
            f(&[f2a.clone(), f2b.clone()]),
            f(&[x___(), y___()]),
            vec![
                xy(&[f2a.clone(), f2b.clone()], &[]),
                xy(&[f2a.clone()], &[f2b.clone()]),
                xy(&[], &[f2a.clone(), f2b.clone()]),
            ],
        ),
        (f(&[f2a.clone(), a.clone()]), f(&[x___(), x___()]), vec![]),
        (f(&[f2a.clone(), a.clone()]), f(&[f2(&[x___()]), x___()]), vec![xs(&[a.clone()])]),
        (f(&[fab.clone()]), f(&[x___(), y___()]), vec![xy(&[fab.clone()], &[]), xy(&[], &[fab.clone()])]),
        (f(&[fab.clone()]), f(&[x___()]), vec![xs(&[fab.clone()])]),
        (f2(&[a.clone(), b.clone()]), f(&[x___(), y___()]), vec![]),
        (f(&[a.clone(), a.clone(), a.clone()]), f(&[x___(), b.clone(), y___()]), vec![]),
        (
            f(&[a.clone(), a.clone(), a.clone()]),
            f(&[x___(), a.clone(), y___()]),
            vec![xy(&[], &[a.clone(), a.clone()]), xy(&[a.clone()], &[a.clone()]), xy(&[a.clone(), a.clone()], &[])],
        ),
        (f(&[a.clone()]), f(&[x___(), a.clone(), y___()]), vec![xy(&[], &[])]),
        (
            f(&[a.clone(), a.clone()]),
            f(&[x___(), a.clone(), y___()]),
            vec![xy(&[a.clone()], &[]), xy(&[], &[a.clone()])],
        ),
        (
            f(&[a.clone(), b.clone(), a.clone()]),
            f(&[x___(), a.clone(), y___()]),
            vec![xy(&[], &[b.clone(), a.clone()]), xy(&[a.clone(), b.clone()], &[])],
        ),
        (f(&[a.clone(), b.clone(), a.clone(), b.clone()]), f(&[x___(), x___()]), vec![xs(&[a.clone(), b.clone()])]),
        (f(&[a.clone(), b.clone(), a.clone(), a.clone()]), f(&[x___(), x___()]), vec![]),
        (f(&[a.clone(), b.clone(), a.clone()]), f(&[x___(), b.clone(), x___()]), vec![xs(&[a.clone()])]),
        (f(&[a.clone(), b.clone(), a.clone(), a.clone()]), f(&[x___(), b.clone(), x___()]), vec![]),
        (f(&[a.clone(), a.clone(), b.clone(), a.clone()]), f(&[x___(), b.clone(), x___()]), vec![]),
        (
            f(&[a.clone(), b.clone(), a.clone(), b.clone(), a.clone(), b.clone(), a.clone()]),
            f(&[x___(), b.clone(), x___()]),
            vec![xs(&[a.clone(), b.clone(), a.clone()])],
        ),
        (
            f(&[a.clone(), b.clone(), a.clone(), b.clone()]),
            f(&[x___(), b.clone(), y___()]),
            vec![xy(&[a.clone(), b.clone(), a.clone()], &[]), xy(&[a.clone()], &[a.clone(), b.clone()])],
        ),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

#[test]
fn wildcard_plus_match() {
    let (a, b, c) = (a(), b(), c());
    let f2b = f2(&[b.clone()]);
    let f2a = f2(&[a.clone()]);
    let fab = f(&[a.clone(), b.clone()]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (a.clone(), x__(), vec![xs(&[a.clone()])]),
        (f(&[a.clone()]), f(&[x__()]), vec![xs(&[a.clone()])]),
        (f(&[]), f(&[x__()]), vec![]),
        (f(&[a.clone()]), x__(), vec![xs(&[f(&[a.clone()])])]),
        (f2(&[a.clone()]), f(&[x__()]), vec![]),
        (fab.clone(), f(&[x__()]), vec![xs(&[a.clone(), b.clone()])]),
        (fab.clone(), f(&[x__(), b.clone()]), vec![xs(&[a.clone()])]),
        (fab.clone(), f(&[x__(), a.clone()]), vec![]),
        (fab.clone(), f(&[a.clone(), x__()]), vec![xs(&[b.clone()])]),
        (fab.clone(), f(&[x__(), x__()]), vec![]),
        (f(&[a.clone(), a.clone()]), f(&[x__(), x__()]), vec![xs(&[a.clone()])]),
        (fab.clone(), f(&[x__(), y__()]), vec![xy(&[a.clone()], &[b.clone()])]),
        (f(&[a.clone()]), f(&[x__(), y__()]), vec![]),
        (
            f(&[a.clone(), b.clone(), c.clone()]),
            f(&[x__(), y__()]),
            vec![xy(&[a.clone()], &[b.clone(), c.clone()]), xy(&[a.clone(), b.clone()], &[c.clone()])],
        ),
        (f(&[a.clone(), f2b.clone()]), f(&[x__(), y__()]), vec![xy(&[a.clone()], &[f2b.clone()])]),
        (f(&[a.clone(), f2b.clone()]), f(&[x__(), f2(&[y__()])]), vec![xy(&[a.clone()], &[b.clone()])]),
        (f(&[a.clone(), f2b.clone()]), f(&[x__(), f2(&[x__()])]), vec![]),
        (f(&[a.clone(), f2a.clone()]), f(&[x__(), f2(&[x__()])]), vec![xs(&[a.clone()])]),
        (f(&[f2a.clone(), f2b.clone()]), f(&[x__(), x__()]), vec![]),
        (f(&[f2a.clone(), f2b.clone()]), f(&[x__(), y__()]), vec![xy(&[f2a.clone()], &[f2b.clone()])]),
        (f(&[f2a.clone(), a.clone()]), f(&[x__(), x__()]), vec![]),
        (f(&[f2a.clone(), a.clone()]), f(&[f2(&[x__()]), x__()]), vec![xs(&[a.clone()])]),
        (f(&[fab.clone()]), f(&[x__(), y__()]), vec![]),
        (f(&[fab.clone()]), f(&[x__()]), vec![xs(&[fab.clone()])]),
        (f2(&[a.clone(), b.clone()]), f(&[x__(), y__()]), vec![]),
        (f(&[a.clone(), a.clone(), a.clone()]), f(&[x__(), b.clone(), y__()]), vec![]),
        (f(&[a.clone(), a.clone(), a.clone()]), f(&[x__(), a.clone(), y__()]), vec![xy(&[a.clone()], &[a.clone()])]),
        (f(&[a.clone()]), f(&[x__(), a.clone(), y__()]), vec![]),
        (f(&[a.clone(), a.clone()]), f(&[x__(), a.clone(), y__()]), vec![]),
        (f(&[a.clone(), b.clone(), a.clone()]), f(&[x__(), a.clone(), y__()]), vec![]),
        (f(&[a.clone(), b.clone(), a.clone(), b.clone()]), f(&[x__(), x__()]), vec![xs(&[a.clone(), b.clone()])]),
        (f(&[a.clone(), b.clone(), a.clone(), a.clone()]), f(&[x__(), x__()]), vec![]),
        (f(&[a.clone(), b.clone(), a.clone()]), f(&[x__(), b.clone(), x__()]), vec![xs(&[a.clone()])]),
        (f(&[a.clone(), b.clone(), a.clone(), a.clone()]), f(&[x__(), b.clone(), x__()]), vec![]),
        (f(&[a.clone(), a.clone(), b.clone(), a.clone()]), f(&[x__(), b.clone(), x__()]), vec![]),
        (
            f(&[a.clone(), b.clone(), a.clone(), b.clone(), a.clone(), b.clone(), a.clone()]),
            f(&[x__(), b.clone(), x__()]),
            vec![xs(&[a.clone(), b.clone(), a.clone()])],
        ),
        (
            f(&[a.clone(), b.clone(), a.clone(), b.clone()]),
            f(&[x__(), b.clone(), y__()]),
            vec![xy(&[a.clone()], &[a.clone(), b.clone()])],
        ),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

#[test]
fn wildcard_mixed_match() {
    let (a, b, c) = (a(), b(), c());
    let abc = f(&[a.clone(), b.clone(), c.clone()]);
    let xyz = |x: Replacement, y: Replacement, z: Replacement| sub(&[("x", x), ("y", y), ("z", z)]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (
            f(&[a.clone(), b.clone()]),
            f(&[x__(), y___()]),
            vec![xy(&[a.clone(), b.clone()], &[]), xy(&[a.clone()], &[b.clone()])],
        ),
        (
            f(&[a.clone(), b.clone()]),
            f(&[x___(), y__()]),
            vec![xy(&[a.clone()], &[b.clone()]), xy(&[], &[a.clone(), b.clone()])],
        ),
        (abc.clone(), f(&[x_(), y__()]), vec![sub(&[("x", one(a.clone())), ("y", seq(&[b.clone(), c.clone()]))])]),
        (abc.clone(), f(&[x__(), y_()]), vec![sub(&[("x", seq(&[a.clone(), b.clone()])), ("y", one(c.clone()))])]),
        (abc.clone(), f(&[x_(), y___()]), vec![sub(&[("x", one(a.clone())), ("y", seq(&[b.clone(), c.clone()]))])]),
        (abc.clone(), f(&[x___(), y_()]), vec![sub(&[("x", seq(&[a.clone(), b.clone()])), ("y", one(c.clone()))])]),
        (
            abc.clone(),
            f(&[x___(), y_(), z___()]),
            vec![
                xyz(seq(&[a.clone(), b.clone()]), one(c.clone()), seq(&[])),
                xyz(seq(&[a.clone()]), one(b.clone()), seq(&[c.clone()])),
                xyz(seq(&[]), one(a.clone()), seq(&[b.clone(), c.clone()])),
            ],
        ),
        (
            abc.clone(),
            f(&[x__(), y_(), z___()]),
            vec![
                xyz(seq(&[a.clone(), b.clone()]), one(c.clone()), seq(&[])),
                xyz(seq(&[a.clone()]), one(b.clone()), seq(&[c.clone()])),
            ],
        ),
        (
            abc.clone(),
            f(&[x___(), y_(), z__()]),
            vec![
                xyz(seq(&[a.clone()]), one(b.clone()), seq(&[c.clone()])),
                xyz(seq(&[]), one(a.clone()), seq(&[b.clone(), c.clone()])),
            ],
        ),
        (f(&[a.clone(), a.clone()]), f(&[x_(), x__()]), vec![]),
        (f(&[a.clone(), a.clone()]), f(&[x__(), x_()]), vec![]),
        (f_c(&[a.clone(), a.clone()]), f_c(&[x__(), x_()]), vec![]),
        (f(&[a.clone(), f_c(&[a.clone()])]), f(&[x__(), f_c(&[x_()])]), vec![]),
        (f(&[a.clone(), f_c(&[a.clone()])]), f(&[x_(), f_c(&[x__()])]), vec![]),
        (f(&[f_c(&[a.clone()]), a.clone()]), f(&[f_c(&[x_()]), x__()]), vec![]),
        (f(&[f_c(&[a.clone()]), a.clone()]), f(&[f_c(&[x__()]), x_()]), vec![]),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

#[test]
fn commutative_multiple_fixed_vars() {
    let (a, b) = (a(), b());
    let xy1 = |x: &Expr, y: &Expr| sub(&[("x", one(x.clone())), ("y", one(y.clone()))]);
    let y1 = |y: &Expr| sub(&[("y", one(y.clone()))]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (
            f_c(&[a.clone(), b.clone(), a.clone(), a.clone(), b.clone()]),
            f_c(&[x_(), x_(), x_(), y_(), y_()]),
            vec![xy1(&a, &b)],
        ),
        (
            f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]),
            f_c(&[x_(), x_(), y_(), y_()]),
            vec![xy1(&a, &b), xy1(&b, &a)],
        ),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[x_(), x_(), x_(), y_(), y_()]), vec![]),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[x_(), y_(), y_()]), vec![]),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[x_(), a.clone(), y_(), y_()]), vec![xy1(&a, &b)]),
        (f_c(&[a.clone(), b.clone(), b.clone(), b.clone()]), f_c(&[x_(), b.clone(), y_(), y_()]), vec![xy1(&a, &b)]),
        (
            f_c(&[a.clone(), b.clone(), a.clone(), a.clone(), b.clone()]),
            f_c(&[_w(), _w(), _w(), y_(), y_()]),
            vec![y1(&a), y1(&b)],
        ),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[_w(), _w(), y_(), y_()]), vec![y1(&b), y1(&a)]),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[_w(), _w(), _w(), y_(), y_()]), vec![]),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[_w(), y_(), y_()]), vec![]),
        (f_c(&[a.clone(), b.clone(), b.clone(), b.clone()]), f_c(&[_w(), _w(), y_(), y_()]), vec![y1(&b)]),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[_w(), a.clone(), y_(), y_()]), vec![y1(&b)]),
        (f_c(&[a.clone(), b.clone(), a.clone(), b.clone()]), f_c(&[_w(), b.clone(), y_(), y_()]), vec![y1(&a)]),
        (f_c(&[a.clone(), b.clone(), b.clone(), b.clone()]), f_c(&[_w(), b.clone(), y_(), y_()]), vec![y1(&b)]),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches_exactly(subject, pattern, expected);
    }
}

#[test]
fn commutative_multiple_sequence_vars() {
    let (a, b) = (a(), b());
    let xym = |x: &[Expr], y: &[Expr]| sub(&[("x", ms(x)), ("y", ms(y))]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (
            f_c(&[a.clone(), b.clone()]),
            f_c(&[x___(), y___()]),
            vec![
                xym(&[a.clone()], &[b.clone()]),
                xym(&[b.clone()], &[a.clone()]),
                xym(&[a.clone(), b.clone()], &[]),
                xym(&[], &[a.clone(), b.clone()]),
            ],
        ),
        (f_c(&[a.clone(), b.clone()]), f_c(&[x___(), x___()]), vec![]),
        (f_c(&[a.clone(), a.clone()]), f_c(&[x___(), x___()]), vec![sub(&[("x", ms(&[a.clone()]))])]),
        (
            f_c(&[a.clone(), a.clone()]),
            f_c(&[x___(), x___(), y___()]),
            vec![xym(&[a.clone()], &[]), xym(&[], &[a.clone(), a.clone()])],
        ),
        (
            f(&[a.clone(), b.clone(), f_c(&[a.clone(), b.clone()])]),
            f(&[x___(), f_c(&[x___()])]),
            vec![xs(&[a.clone(), b.clone()])],
        ),
        (
            f(&[b.clone(), a.clone(), f_c(&[a.clone(), b.clone()])]),
            f(&[x___(), f_c(&[x___()])]),
            vec![xs(&[b.clone(), a.clone()])],
        ),
        (f(&[a.clone(), a.clone(), f_c(&[a.clone(), b.clone()])]), f(&[x___(), f_c(&[x___()])]), vec![]),
        (f(&[a.clone(), b.clone(), f_c(&[a.clone(), a.clone()])]), f(&[x___(), f_c(&[x___()])]), vec![]),
        (f_c(&[a.clone()]), f_c(&[__(), __()]), vec![]),
        (f_c(&[a.clone(), a.clone()]), f_c(&[__(), __()]), vec![Substitution::new()]),
        (f_c(&[a.clone(), b.clone()]), f_c(&[__(), __()]), vec![Substitution::new()]),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches_exactly(subject, pattern, expected);
    }
}

#[test]
fn mixed_commutative_vars() {
    let (a, b) = (a(), b());
    let cases = [
        (f(&[f(&[a.clone()]), f_c(&[a.clone()])]), f(&[f(&[x_()]), f_c(&[x_()])])),
        (f(&[f(&[a.clone(), b.clone()]), f_c(&[a.clone(), b.clone()])]), f(&[f(&[x__()]), f_c(&[x__()])])),
        (f(&[f(&[a.clone(), b.clone()]), f_c(&[b.clone(), a.clone()])]), f(&[f(&[x__()]), f_c(&[x__()])])),
        (f(&[f(&[b.clone(), a.clone()]), f_c(&[b.clone(), a.clone()])]), f(&[f(&[x__()]), f_c(&[x__()])])),
        (f(&[f(&[b.clone(), a.clone()]), f_c(&[a.clone(), b.clone()])]), f(&[f(&[x__()]), f_c(&[x__()])])),
        (f(&[f_c(&[a.clone(), b.clone()]), f(&[a.clone(), b.clone()])]), f(&[f_c(&[x__()]), f(&[x__()])])),
        (f(&[f_c(&[b.clone(), a.clone()]), f(&[a.clone(), b.clone()])]), f(&[f_c(&[x__()]), f(&[x__()])])),
        (f(&[f_c(&[b.clone(), a.clone()]), f(&[b.clone(), a.clone()])]), f(&[f_c(&[x__()]), f(&[x__()])])),
        (f(&[f_c(&[a.clone(), b.clone()]), f(&[b.clone(), a.clone()])]), f(&[f_c(&[x__()]), f(&[x__()])])),
        (f(&[f_c(&[a.clone(), b.clone()]), f_c(&[a.clone(), b.clone()])]), f(&[f_c(&[x__()]), f_c(&[x__()])])),
    ];
    for (subject, pattern) in cases {
        assert!(!matches(&subject, &pattern).is_empty(), "{subject} / {pattern}");
    }
}

#[test]
fn mixed_associative_commutative_vars() {
    let (a, b) = (a(), b());
    let ab = || vec![a.clone(), b.clone()];
    let cases = [
        (f(&[f(&[a.clone()]), f_ac(&[a.clone()])]), f(&[f(&[x_()]), f_ac(&[x_()])]), true),
        (f(&[f_ac(&[a.clone()]), f(&[a.clone()])]), f(&[f_ac(&[x_()]), f(&[x_()])]), true),
        (f(&[f(&ab()), f_ac(&ab())]), f(&[f(&[x_()]), f_ac(&[x_()])]), false),
        (f(&[f_ac(&ab()), f(&ab())]), f(&[f_ac(&[x_()]), f(&[x_()])]), false),
        (f(&[f_c(&[a.clone()]), f_ac(&[a.clone()])]), f(&[f_c(&[x_()]), f_ac(&[x_()])]), true),
        (f(&[f_ac(&[a.clone()]), f_c(&[a.clone()])]), f(&[f_ac(&[x_()]), f_c(&[x_()])]), true),
        (f(&[f_c(&ab()), f_ac(&ab())]), f(&[f_c(&[x_()]), f_ac(&[x_()])]), false),
        (f(&[f_ac(&ab()), f_c(&ab())]), f(&[f_ac(&[x_()]), f_c(&[x_()])]), false),
        (f(&[f_a(&[a.clone()]), f_ac(&[a.clone()])]), f(&[f_a(&[x_()]), f_ac(&[x_()])]), true),
        (f(&[f_ac(&[a.clone()]), f_a(&[a.clone()])]), f(&[f_ac(&[x_()]), f_a(&[x_()])]), true),
        (f(&[f_a(&ab()), f_ac(&ab())]), f(&[f_a(&[x_()]), f_ac(&[x_()])]), false),
        (f(&[f_ac(&ab()), f_a(&ab())]), f(&[f_ac(&[x_()]), f_a(&[x_()])]), false),
        (f(&[f(&[f_ac(&ab())]), f_ac(&ab())]), f(&[f(&[x_()]), f_ac(&[x_()])]), true),
        (f(&[f_ac(&ab()), f(&[f_ac(&ab())])]), f(&[f_ac(&[x_()]), f(&[x_()])]), true),
        (f(&[f_a(&ab()), f_ac(&[f_a(&ab())])]), f(&[f_a(&[x_()]), f_ac(&[x_()])]), true),
        (f(&[f_ac(&[f_a(&ab())]), f_a(&ab())]), f(&[f_ac(&[x_()]), f_a(&[x_()])]), true),
        (f(&[f_a(&ab()), f_c(&ab())]), f(&[f_a(&[x_()]), f_c(&[x_(), ___()])]), false),
        (f(&[f_a(&[a.clone()]), f_c(&[a.clone(), a.clone()])]), f(&[f_a(&[x_()]), f_c(&[x_(), x_()])]), true),
        (
            f(&[f_ac(&ab()), f_ac(&[a.clone(), a.clone(), b.clone(), b.clone()])]),
            f(&[f_ac(&[x_()]), f_ac(&[x_(), x_()])]),
            true,
        ),
    ];
    for (subject, pattern, is_match) in cases {
        assert_eq!(!matches(&subject, &pattern).is_empty(), is_match, "{subject} / {pattern}");
    }
}

#[test]
fn variable_name() {
    let (a, b) = (a(), b());
    let fh = head("f");
    let x = |e: Expr| vec![sub(&[("x", one(e))])];
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (a.clone(), a_x(), x(a.clone())),
        (f(&[a.clone()]), f(&[a_x()]), x(a.clone())),
        (f(&[b.clone()]), f(&[a_x()]), vec![]),
        (f(&[a.clone(), b.clone()]), f(&[a_x(), b.clone()]), x(a.clone())),
        (f(&[a.clone()]), fh.call_named(vec![a.clone()], "x"), x(f(&[a.clone()]))),
        (f2(&[a.clone()]), fh.call_named(vec![a.clone()], "x"), vec![]),
        (f(&[a.clone(), b.clone()]), fh.call_named(vec![a.clone(), b.clone()], "x"), x(f(&[a.clone(), b.clone()]))),
        (f2(&[f(&[a.clone()])]), f2(&[fh.call_named(vec![a.clone()], "x")]), x(f(&[a.clone()]))),
        (
            f2(&[f(&[a.clone(), b.clone()])]),
            f2(&[fh.call_named(vec![a.clone(), b.clone()], "x")]),
            x(f(&[a.clone(), b.clone()])),
        ),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

#[test]
fn one_identity_match() {
    let (a, b, c) = (a(), b(), c());
    let oo2 = |o: &Expr, o2: &Expr| sub(&[("o", one(o.clone())), ("o2", one(o2.clone()))]);
    let cases: Vec<(Expr, Expr, Vec<Substitution>)> = vec![
        (f(&[a.clone()]), f_i(&[oa_(), f(&[a.clone()])]), vec![sub(&[("o", one(a.clone()))])]),
        (f(&[a.clone()]), f_i(&[x___(), f(&[a.clone()])]), vec![xs(&[])]),
        (a.clone(), f_i(&[oa_(), f_i(&[o2b_(), a.clone()])]), vec![oo2(&a, &b)]),
        (f_i(&[c.clone(), a.clone()]), f_i(&[oa_(), f_i(&[o2b_(), a.clone()])]), vec![oo2(&c, &b), oo2(&a, &c)]),
        (f_i(&[c.clone(), f_i(&[c.clone(), a.clone()])]), f_i(&[oa_(), f_i(&[o2b_(), a.clone()])]), vec![oo2(&c, &c)]),
        (a.clone(), f_i(&[oa_(), o2b_(), a.clone()]), vec![oo2(&a, &b)]),
        (a.clone(), f_i(&[oa_(), x___(), a.clone()]), vec![sub(&[("o", one(a.clone())), ("x", seq(&[]))])]),
        (
            f_i(&[b.clone(), a.clone()]),
            f_i(&[oa_(), x___(), a.clone()]),
            vec![
                sub(&[("o", one(b.clone())), ("x", seq(&[]))]),
                sub(&[("o", one(a.clone())), ("x", seq(&[b.clone()]))]),
            ],
        ),
        (f(&[a.clone()]), f_ci(&[oa_(), f(&[a.clone()])]), vec![sub(&[("o", one(a.clone()))])]),
        (f(&[a.clone()]), f_ci(&[x___(), f(&[a.clone()])]), vec![sub(&[("x", ms(&[]))])]),
        (a.clone(), f_ci(&[oa_(), f_i(&[o2b_(), a.clone()])]), vec![oo2(&a, &b)]),
        (f_ci(&[c.clone(), a.clone()]), f_ci(&[oa_(), f_i(&[o2b_(), a.clone()])]), vec![oo2(&c, &b)]),
        (f_ci(&[c.clone(), a.clone()]), f_ci(&[oa_(), f_ci(&[o2b_(), a.clone()])]), vec![oo2(&c, &b), oo2(&a, &c)]),
        (
            f_ci(&[c.clone(), f_i(&[c.clone(), a.clone()])]),
            f_ci(&[oa_(), f_i(&[o2b_(), a.clone()])]),
            vec![oo2(&c, &c)],
        ),
        (a.clone(), f_ci(&[oa_(), o2b_(), a.clone()]), vec![oo2(&a, &b)]),
        (a.clone(), f_ci(&[oa_(), x___(), a.clone()]), vec![sub(&[("o", one(a.clone())), ("x", ms(&[]))])]),
        (
            f_ci(&[b.clone(), a.clone()]),
            f_ci(&[oa_(), x___(), a.clone()]),
            vec![sub(&[("o", one(b.clone())), ("x", ms(&[]))]), sub(&[("o", one(a.clone())), ("x", ms(&[b.clone()]))])],
        ),
    ];
    for (subject, pattern, expected) in cases {
        assert_matches(subject, pattern, expected);
    }
}

// ─── constraints ─────────────────────────────────────────────────────────────

/// Port of `tests/utils.py::MockConstraint`: returns a fixed value and counts calls.
#[derive(Debug)]
struct MockConstraint {
    value: bool,
    vars: Vec<String>,
    calls: AtomicUsize,
}

impl MockConstraint {
    fn new(value: bool, vars: &[&str]) -> Arc<MockConstraint> {
        Arc::new(MockConstraint {
            value,
            vars: vars.iter().map(|s| s.to_string()).collect(),
            calls: AtomicUsize::new(0),
        })
    }
}

impl Constraint for MockConstraint {
    fn variables(&self) -> std::collections::BTreeSet<String> {
        self.vars.iter().cloned().collect()
    }
    fn check(&self, _subst: &Substitution) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.value
    }
    fn with_renamed_vars(&self, _renaming: &std::collections::HashMap<String, String>) -> Arc<dyn Constraint> {
        unimplemented!()
    }
}

fn count_with_mocks(subject: &Expr, pattern: &Expr, values: &[bool], vars: &[&str]) -> usize {
    let constraints: Vec<Arc<dyn Constraint>> =
        values.iter().map(|&v| MockConstraint::new(v, vars) as Arc<dyn Constraint>).collect();
    match_pattern(subject, &Pattern::with_constraints(pattern.clone(), constraints)).unwrap().len()
}

#[test]
fn global_and_local_constraints() {
    let (a, b) = (a(), b());
    let syntactic = [
        (a.clone(), _w(), vec![false], 0),
        (a.clone(), _w(), vec![true], 1),
        (f(&[a.clone(), b.clone()]), f(&[_w(), _w()]), vec![false, true], 0),
        (f(&[a.clone(), b.clone()]), f(&[_w(), _w()]), vec![true, true], 1),
        (a.clone(), x_(), vec![true], 1),
        (a.clone(), x_(), vec![false], 0),
        (f(&[a.clone(), a.clone()]), f(&[x_(), x_()]), vec![true, false], 0),
        (f(&[a.clone(), a.clone()]), f(&[x_(), x_()]), vec![true, true], 1),
        (f(&[a.clone()]), f(&[a.clone()]), vec![false], 0),
        (f(&[a.clone()]), f(&[a.clone()]), vec![true], 1),
    ];
    for (subject, pattern, values, count) in syntactic {
        assert_eq!(count_with_mocks(&subject, &pattern, &values, &[]), count, "global {subject} / {pattern}");
    }
    let fa_c = f_c(&[a.clone()]);
    let non_syntactic = [
        (f(&[a.clone(), b.clone()]), f(&[___(), ___()]), vec![true, true], 3, false),
        (f(&[a.clone(), b.clone()]), f(&[___(), ___()]), vec![true, false], 0, false),
        (f_c(&[a.clone(), a.clone()]), f_c(&[x_(), x_()]), vec![true, false], 0, true),
        (f_c(&[a.clone(), a.clone()]), f_c(&[x_(), x_()]), vec![true, true], 1, true),
        (f(&[a.clone(), fa_c.clone()]), f(&[x_(), f_c(&[x_()])]), vec![false, true], 0, true),
        (f(&[a.clone(), fa_c.clone()]), f(&[x_(), f_c(&[x_()])]), vec![true, true], 1, true),
        (f_c(&[a.clone(), f(&[a.clone()])]), f_c(&[x_(), f(&[x_()])]), vec![true, false], 0, true),
        (f_c(&[a.clone(), f(&[a.clone()])]), f_c(&[x_(), f(&[x_()])]), vec![true, true], 1, true),
        (f_c(&[a.clone(), a.clone()]), f_c(&[x___()]), vec![false], 0, true),
        (f_c(&[a.clone(), a.clone()]), f_c(&[x___()]), vec![true], 1, true),
    ];
    for (subject, pattern, values, count, also_local) in non_syntactic {
        assert_eq!(count_with_mocks(&subject, &pattern, &values, &[]), count, "global {subject} / {pattern}");
        if also_local {
            assert_eq!(count_with_mocks(&subject, &pattern, &values, &["x"]), count, "local {subject} / {pattern}");
        }
    }
}

#[test]
fn selective_constraint() {
    let c = CustomConstraint::new(["x"], |args: &[&Replacement]| args[0].to_string().len() > 1);
    let pattern = Pattern::with_constraints(f(&[___(), x_(), ___()]), vec![Arc::new(c)]);
    let subject = f(&[a(), Expr::atom("aa"), b(), Expr::atom("bb")]);
    let result = match_pattern(&subject, &pattern).unwrap();
    assert_eq!(result.len(), 2);
    assert!(result.contains(&sub(&[("x", one(Expr::atom("aa")))])));
    assert!(result.contains(&sub(&[("x", one(Expr::atom("bb")))])));
}

#[test]
fn free_of_constraint() {
    let fb = Head::new("f", Arity::Binary).unwrap();
    let pattern = Pattern::with_constraints(fb.call(vec![x_(), y_()]), vec![Arc::new(FreeOf::single("y", "x"))]);
    assert!(is_match(&fb.call(vec![Expr::atom("x"), a()]), &pattern).unwrap());
    assert!(!is_match(&fb.call(vec![Expr::atom("x"), Expr::atom("x")]), &pattern).unwrap());
}

#[test]
fn non_constant_subject_is_rejected() {
    assert_eq!(match_pattern(&x_(), &Pattern::new(x_())), Err(MatchError::NonConstantSubject));
}

#[test]
fn match_anywhere_positions() {
    let subject = f(&[a(), f(&[b()])]);
    let result = match_anywhere(&subject, &Pattern::new(f(&[x_()]))).unwrap();
    assert_eq!(result, vec![(sub(&[("x", one(b()))]), vec![1])]);
}

#[test]
fn rewriting() {
    // f(x_) -> x, applied repeatedly
    let rule = ReplacementRule::new(Pattern::new(f(&[x_()])), |s: &Substitution| s.get("x").unwrap().clone());
    let expr = f2(&[f(&[f(&[a()])]), b()]);
    assert_eq!(replace_all(&expr, &[rule.clone()], None).unwrap(), one(f2(&[a(), b()])));
    assert_eq!(replace_all(&expr, &[rule.clone()], Some(1)).unwrap(), one(f2(&[f(&[a()]), b()])));
    assert_eq!(replace_all_post_order(&expr, &[rule]).unwrap(), f2(&[a(), b()]));

    // substitute splices sequences and sorted multisets
    let s = sub(&[("x", seq(&[a(), b()]))]);
    assert_eq!(substitute(&f(&[x_(), c()]), &s).unwrap(), one(f(&[a(), b(), c()])));
    let s = sub(&[("x", ms(&[b(), a(), b()]))]);
    assert_eq!(substitute(&f(&[x_(), c()]), &s).unwrap().to_string(), "f(a, b, b, c)");

    // replace splices too
    let r = replace(&f(&[a(), b()]), &[0], seq(&[c(), c()])).unwrap();
    assert_eq!(r, one(f(&[c(), c(), b()])));
}
