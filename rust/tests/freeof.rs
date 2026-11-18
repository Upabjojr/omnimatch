//! Port of `tests/test_freeof.py`.
//!
//! Every matching case runs through both `match_pattern` and `ManyToOneMatcher`.
//!
//! Not ported: `test_repr` / the `repr` half of `test_single_variable_form_is_unchanged`
//! (Python `repr`; `Display` is tested instead) and `test_names_may_be_objects_that_carry_them`
//! (duck-typed `wildcard_name` / `name` arguments have no Rust analogue).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use omnimatch::*;

fn f(l: Expr, r: Expr) -> Expr {
    Head::new("f", Arity::Binary).unwrap().call(vec![l, r])
}
fn g(e: Expr) -> Expr {
    Head::new("g", Arity::Unary).unwrap().call(vec![e])
}
fn h(ops: &[Expr]) -> Expr {
    Head::builder("h").commutative(true).associative(true).one_identity(true).build().unwrap().call(ops.to_vec())
}
fn at(n: &str) -> Expr {
    Expr::atom(n)
}
fn var(n: &str) -> Expr {
    make_dot_variable(n)
}

fn free_of(variable: &str, symbol: &str) -> Arc<dyn Constraint> {
    Arc::new(FreeOf::single(variable, symbol))
}

fn pattern(expression: Expr, constraints: Vec<Arc<dyn Constraint>>) -> Pattern {
    Pattern::with_constraints(expression, constraints)
}

/// One-to-one matches, after checking that many-to-one finds the same set.
fn matches(subject: &Expr, pattern: &Pattern) -> Vec<Substitution> {
    let one_to_one = match_pattern(subject, pattern).unwrap();
    let matcher = ManyToOneMatcher::from_patterns([pattern.clone()]);
    let many: Vec<Substitution> = matcher.match_all(subject).into_iter().map(|(_, s)| s).collect();
    let set = |v: &[Substitution]| v.iter().map(ToString::to_string).collect::<BTreeSet<_>>();
    assert_eq!(set(&many), set(&one_to_one), "matchers disagree on {subject} / {pattern}");
    one_to_one
}

fn bound(s: &Substitution, n: &str) -> Expr {
    s.get(n).and_then(Replacement::as_single).cloned().unwrap()
}

fn set(xs: &[&str]) -> BTreeSet<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

// ─── properties ──────────────────────────────────────────────────────────────

#[test]
fn properties() {
    let c = FreeOf::single("u", "x");
    assert_eq!(c.variables(), set(&["u"]));
    assert_eq!(c.to_string(), "FreeOf(u, x)");
    assert_eq!(c.variable_names(), ["u"]);

    assert_eq!(FreeOf::single("u", "x"), FreeOf::single("u", "x"));
    assert_ne!(FreeOf::single("u", "x"), FreeOf::single("u", "y"));
    assert_ne!(FreeOf::single("u", "x"), FreeOf::single("v", "x"));
    let distinct: HashSet<FreeOf> = [FreeOf::single("u", "x"), FreeOf::single("u", "x")].into();
    assert_eq!(distinct.len(), 1);
    // The same, through the type-erased handle the matchers use.
    let refs: HashSet<ConstraintRef> =
        [ConstraintRef(free_of("u", "x")), ConstraintRef(free_of("u", "x")), ConstraintRef(free_of("u", "y"))].into();
    assert_eq!(refs.len(), 2);

    let renamed = c.with_renamed_vars(&HashMap::from([("u".into(), "u_renamed".into())]));
    assert_eq!(renamed.variables(), set(&["u_renamed"]));
    assert_eq!(renamed.eq_key(), FreeOf::single("u_renamed", "x").eq_key());
}

// ─── one-to-one (and many-to-one) ────────────────────────────────────────────

#[test]
fn symbols_and_operations() {
    let p = pattern(f(var("x"), var("u")), vec![free_of("u", "x")]);
    let r = matches(&f(at("x"), at("a")), &p);
    assert_eq!(r.len(), 1);
    assert_eq!(bound(&r[0], "u"), at("a"));
    assert!(matches(&f(at("a"), at("x")), &p).is_empty());

    let r = matches(&f(at("x"), g(at("a"))), &p);
    assert_eq!(r.len(), 1);
    assert_eq!(bound(&r[0], "u"), g(at("a")));
    assert!(matches(&f(at("a"), g(at("x"))), &p).is_empty());
}

#[test]
fn deep_nesting() {
    let p = pattern(f(var("u"), var("v")), vec![free_of("u", "x")]);
    assert!(matches(&f(g(g(at("x"))), at("a")), &p).is_empty());
    assert_eq!(matches(&f(g(g(at("a"))), at("b")), &p).len(), 1);
}

#[test]
fn multiple_constraints() {
    let p = pattern(f(var("u"), var("v")), vec![free_of("u", "x"), free_of("v", "x")]);
    assert_eq!(matches(&f(at("a"), at("b")), &p).len(), 1);
    assert!(matches(&f(at("x"), at("b")), &p).is_empty());
    assert!(matches(&f(at("a"), at("x")), &p).is_empty());

    let p = pattern(f(var("u"), var("v")), vec![free_of("u", "x"), free_of("v", "y")]);
    assert_eq!(matches(&f(at("y"), at("x")), &p).len(), 1);
    assert!(matches(&f(at("x"), at("a")), &p).is_empty());
    assert!(matches(&f(at("a"), at("y")), &p).is_empty());
}

#[test]
fn unary_pattern() {
    let p = pattern(g(var("u")), vec![free_of("u", "x")]);
    assert_eq!(matches(&g(at("a")), &p).len(), 1);
    assert!(matches(&g(at("x")), &p).is_empty());
}

// ─── commutative ─────────────────────────────────────────────────────────────

#[test]
fn commutative() {
    let p = pattern(h(&[var("x"), var("u")]), vec![free_of("u", "x")]);
    let r = matches(&h(&[at("x"), at("a")]), &p);
    assert!(r.iter().any(|s| bound(s, "u") == at("a")), "{r:?}");
    assert!(r.iter().all(|s| bound(s, "u") != at("x")), "{r:?}");

    let r = matches(&h(&[at("x"), at("x")]), &p);
    assert!(r.is_empty(), "u can only be x, which is not free of x: {r:?}");
}

// ─── many-to-one with several patterns ───────────────────────────────────────

#[test]
fn many_to_one_multiple_patterns() {
    let mut matcher = ManyToOneMatcher::<&str>::new();
    matcher.add_with_label(pattern(f(var("u"), var("v")), vec![free_of("u", "x")]), "u_free");
    matcher.add_with_label(pattern(f(var("u"), var("v")), vec![free_of("v", "x")]), "v_free");
    let labels = |s: &Expr| matcher.match_all(s).into_iter().map(|(l, _)| *l).collect::<BTreeSet<_>>();
    assert_eq!(labels(&f(at("a"), at("x"))), ["u_free"].into());
    assert_eq!(labels(&f(at("x"), at("a"))), ["v_free"].into());
    assert_eq!(labels(&f(at("a"), at("b"))), ["u_free", "v_free"].into());
}

// ─── group API ───────────────────────────────────────────────────────────────

#[test]
fn group_of_variables() {
    let p = pattern(f(var("x"), var("y")), vec![Arc::new(FreeOf::new(["x", "y"], "z"))]);
    assert!(!matches(&f(at("a"), at("b")), &p).is_empty());
    assert!(matches(&f(at("z"), at("b")), &p).is_empty());
    assert!(matches(&f(at("a"), at("z")), &p).is_empty());

    let grouped = FreeOf::new(["a", "b"], "x");
    let separate: BTreeSet<String> =
        FreeOf::single("a", "x").variables().union(&FreeOf::single("b", "x").variables()).cloned().collect();
    assert_eq!(grouped.variables(), separate);
    assert_eq!(grouped.to_string(), "FreeOf([a, b], x)");

    let renamed = grouped.with_renamed_vars(&HashMap::from([("a".into(), "a2".into())]));
    assert_eq!(renamed.variables(), set(&["a2", "b"]));
    assert_eq!(renamed.eq_key(), FreeOf::new(["a2", "b"], "x").eq_key());

    assert_eq!(FreeOf::new(["a", "b"], "x"), FreeOf::new(["a", "b"], "x"));
    assert_ne!(FreeOf::new(["a", "b"], "x"), FreeOf::new(["a", "c"], "x"));
    let distinct: HashSet<FreeOf> = [FreeOf::new(["a", "b"], "x"), FreeOf::new(["a", "b"], "x")].into();
    assert_eq!(distinct.len(), 1);
}

#[test]
fn unbound_variable_defers() {
    let c = FreeOf::new(["a", "b"], "x");
    assert!(c.check(&[("a", at("q"))].into_iter().collect()));
    assert!(!c.check(&[("a", at("x"))].into_iter().collect()));
}
