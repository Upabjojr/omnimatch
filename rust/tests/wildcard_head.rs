//! Port of `tests/test_wildcard_operation_head.py`.
//!
//! Python only supports wildcard heads in the many-to-one matcher; here every
//! case also runs through the one-to-one matcher, and the two must agree.

use std::collections::BTreeSet;

use omnimatch::*;

fn h(name: &str, arity: Arity) -> Head {
    Head::builder(name).arity(arity).build_unchecked()
}
fn sin() -> Head {
    h("SIN", Arity::Unary)
}
fn cos() -> Head {
    h("COS", Arity::Unary)
}
fn f2() -> Head {
    h("F2", Arity::Binary)
}
fn var() -> Head {
    h("VAR", Arity::Variadic)
}
fn lst() -> Head {
    h("LST", Arity::Variadic)
}
fn acm(name: &str) -> Head {
    Head::builder(name).commutative(true).associative(true).one_identity(true).build_unchecked()
}
fn add() -> Head {
    acm("ADD")
}
fn mul() -> Head {
    acm("MUL")
}
fn at(n: &str) -> Expr {
    Expr::atom(n)
}
fn dot(n: &str) -> Expr {
    make_dot_variable(n)
}
fn anyhead(v: &str) -> Head {
    Head::any(Some(v))
}
fn op(head: &Head, operands: &[Expr]) -> Expr {
    Expr::operation(head.clone(), operands.to_vec())
}

/// Many-to-one matches, after checking that one-to-one finds the same set.
fn matches(pattern: &Expr, subject: &Expr) -> Vec<Substitution> {
    let pattern = Pattern::new(pattern.clone());
    let mut m = ManyToOneMatcher::<&str>::new();
    m.add_with_label(pattern.clone(), "p");
    let many: Vec<Substitution> = m.match_all(subject).into_iter().map(|(_, s)| s).collect();
    let one_to_one = match_pattern(subject, &pattern).unwrap();
    let set = |v: &[Substitution]| v.iter().map(ToString::to_string).collect::<BTreeSet<_>>();
    assert_eq!(set(&many), set(&one_to_one), "matchers disagree on {subject} / {pattern}");
    many
}

fn one(pattern: &Expr, subject: &Expr) -> Substitution {
    let got = matches(pattern, subject);
    assert_eq!(got.len(), 1, "expected exactly 1 match of {pattern} on {subject}, got {got:?}");
    got.into_iter().next().unwrap()
}

fn bound_head(s: &Substitution, var: &str) -> Head {
    s.get(var).and_then(Replacement::as_head).cloned().unwrap_or_else(|| panic!("{var} not bound to a head in {s}"))
}

fn single(s: &Substitution, var: &str) -> Expr {
    s.get(var).and_then(Replacement::as_single).cloned().unwrap_or_else(|| panic!("{var} not bound in {s}"))
}

fn sequence(s: &Substitution, var: &str) -> Vec<Expr> {
    s.get(var).unwrap().to_vec()
}

// ─── head binding ────────────────────────────────────────────────────────────

#[test]
fn binds_any_head() {
    for head in [sin(), cos(), var(), add(), mul()] {
        let s = one(&op(&anyhead("F"), &[dot("v")]), &op(&head, &[at("x")]));
        assert_eq!(bound_head(&s, "F"), head);
        assert_eq!(single(&s, "v"), at("x"));
    }
}

#[test]
fn anonymous_wildcard_head_binds_nothing() {
    let s = one(&op(&Head::any(None), &[dot("v")]), &op(&sin(), &[at("x")]));
    assert!(!s.contains("F"));
    assert_eq!(single(&s, "v"), at("x"));
}

#[test]
fn repeated_head_variable_must_be_consistent() {
    let pat = op(&lst(), &[op(&anyhead("F"), &[dot("u")]), op(&anyhead("F"), &[dot("v")])]);
    let s = one(&pat, &op(&lst(), &[op(&sin(), &[at("x")]), op(&sin(), &[at("y")])]));
    assert_eq!(bound_head(&s, "F"), sin());
    assert!(matches(&pat, &op(&lst(), &[op(&sin(), &[at("x")]), op(&cos(), &[at("y")])])).is_empty());
}

#[test]
fn two_independent_head_variables() {
    let pat = op(&lst(), &[op(&anyhead("F"), &[dot("u")]), op(&anyhead("G"), &[dot("v")])]);
    let s = one(&pat, &op(&lst(), &[op(&sin(), &[at("x")]), op(&cos(), &[at("y")])]));
    assert_eq!((bound_head(&s, "F"), bound_head(&s, "G")), (sin(), cos()));
    let s = one(&pat, &op(&lst(), &[op(&sin(), &[at("x")]), op(&sin(), &[at("y")])]));
    assert_eq!((bound_head(&s, "F"), bound_head(&s, "G")), (sin(), sin()));
}

#[test]
fn binding_is_undone_on_backtrack() {
    let pat = op(&lst(), &[op(&anyhead("F"), &[dot("u")]), at("a")]);
    assert!(matches(&pat, &op(&lst(), &[op(&sin(), &[at("x")]), at("b")])).is_empty());
    let s = one(&pat, &op(&lst(), &[op(&cos(), &[at("y")]), at("a")]));
    assert_eq!(bound_head(&s, "F"), cos());
}

// ─── operand kinds ───────────────────────────────────────────────────────────

#[test]
fn concrete_operand() {
    let pat = op(&anyhead("F"), &[at("a")]);
    assert_eq!(bound_head(&one(&pat, &op(&sin(), &[at("a")])), "F"), sin());
    assert!(matches(&pat, &op(&sin(), &[at("b")])).is_empty());
}

#[test]
fn nested_operation_operand_binds_inner_wildcards() {
    let pat = op(&anyhead("F"), &[op(&add(), &[dot("p"), op(&mul(), &[dot("q"), at("x")])])]);
    let s = one(&pat, &op(&sin(), &[op(&add(), &[at("a"), op(&mul(), &[at("b"), at("x")])])]));
    assert_eq!(bound_head(&s, "F"), sin());
    assert_eq!((single(&s, "p"), single(&s, "q")), (at("a"), at("b")));
}

#[test]
fn optional_wildcard_operand_uses_default_when_absent() {
    let pat = op(&anyhead("F"), &[Expr::optional("c", at("a")), dot("v")]);
    let s = one(&pat, &op(&var(), &[at("b"), at("x")]));
    assert_eq!((single(&s, "c"), single(&s, "v")), (at("b"), at("x")));
    let s = one(&pat, &op(&var(), &[at("x")]));
    assert_eq!(single(&s, "c"), at("a"));
    assert_eq!(bound_head(&s, "F"), var());
}

#[test]
fn sequence_operands() {
    let star = op(&anyhead("F"), &[make_star_variable("s")]);
    for args in [vec![], vec![at("x")], vec![at("x"), at("y"), at("z")]] {
        let s = one(&star, &op(&var(), &args));
        assert_eq!(bound_head(&s, "F"), var());
        assert_eq!(sequence(&s, "s"), args);
    }
    let plus = op(&anyhead("F"), &[make_plus_variable("s")]);
    assert!(matches(&plus, &op(&var(), &[])).is_empty());
    assert_eq!(sequence(&one(&plus, &op(&var(), &[at("x"), at("y")])), "s"), vec![at("x"), at("y")]);

    let mixed = op(&anyhead("F"), &[dot("first"), make_star_variable("rest")]);
    let s = one(&mixed, &op(&var(), &[at("x"), at("y"), at("z")]));
    assert_eq!(single(&s, "first"), at("x"));
    assert_eq!(sequence(&s, "rest"), vec![at("y"), at("z")]);
}

// ─── commutative x associative x one_identity ────────────────────────────────

fn flag_heads(name: &str) -> Vec<Head> {
    let mut heads = Vec::new();
    for bits in 0..8 {
        heads.push(
            Head::builder(name)
                .commutative(bits & 1 != 0)
                .associative(bits & 2 != 0)
                .one_identity(bits & 4 != 0)
                .build_unchecked(),
        );
    }
    heads
}

#[test]
fn matches_subject_head_with_any_flags() {
    for head in flag_heads("H") {
        let s = one(&op(&anyhead("F"), &[dot("u"), dot("v")]), &op(&head, &[at("x"), at("y")]));
        assert_eq!(bound_head(&s, "F"), head);
        let uv: BTreeSet<Expr> = [single(&s, "u"), single(&s, "v")].into();
        assert_eq!(uv, [at("x"), at("y")].into());
    }
}

#[test]
fn nested_in_enclosing_operation_with_any_flags() {
    for outer in flag_heads("OUT") {
        let pat = op(&outer, &[dot("u"), op(&anyhead("F"), &[dot("v")])]);
        let s = one(&pat, &op(&outer, &[at("w"), op(&sin(), &[at("y")])]));
        assert_eq!(bound_head(&s, "F"), sin(), "{outer:?}");
        assert_eq!((single(&s, "v"), single(&s, "u")), (at("y"), at("w")));
    }
}

#[test]
fn commutative_enclosing_matches_in_any_operand_order() {
    let pat = op(&mul(), &[dot("u"), op(&anyhead("F"), &[dot("v")])]);
    for subject in [op(&mul(), &[at("w"), op(&sin(), &[at("y")])]), op(&mul(), &[op(&sin(), &[at("y")]), at("w")])] {
        let s = one(&pat, &subject);
        assert_eq!(bound_head(&s, "F"), sin());
        assert_eq!((single(&s, "v"), single(&s, "u")), (at("y"), at("w")));
    }
}

#[test]
fn associative_subject_is_flattened_before_matching() {
    let assoc = Head::builder("ASSOC").associative(true).build_unchecked();
    let subject = op(&assoc, &[at("x"), op(&assoc, &[at("y"), at("z")])]);
    let s = one(&op(&anyhead("F"), &[dot("p"), dot("q"), dot("r")]), &subject);
    assert_eq!(bound_head(&s, "F"), assoc);
    assert_eq!((single(&s, "p"), single(&s, "q"), single(&s, "r")), (at("x"), at("y"), at("z")));
}

#[test]
fn one_identity_subject_collapses_and_is_then_not_an_operation() {
    let oi = Head::builder("OI").one_identity(true).build_unchecked();
    assert_eq!(oi.call(vec![at("x")]), at("x"));
    assert!(matches(&op(&anyhead("F"), &[dot("v")]), &oi.call(vec![at("x")])).is_empty());
}

// ─── nesting ─────────────────────────────────────────────────────────────────

#[test]
fn nested_wildcard_heads() {
    let double = op(&anyhead("F"), &[op(&anyhead("G"), &[dot("v")])]);
    let s = one(&double, &op(&sin(), &[op(&cos(), &[at("x")])]));
    assert_eq!((bound_head(&s, "F"), bound_head(&s, "G"), single(&s, "v")), (sin(), cos(), at("x")));
    assert!(matches(&double, &op(&sin(), &[at("x")])).is_empty());

    let triple = op(&anyhead("F"), &[op(&anyhead("G"), &[op(&anyhead("H"), &[dot("v")])])]);
    let s = one(&triple, &op(&sin(), &[op(&cos(), &[op(&var(), &[at("x")])])]));
    assert_eq!((bound_head(&s, "F"), bound_head(&s, "G"), bound_head(&s, "H")), (sin(), cos(), var()));

    let shared = op(&anyhead("F"), &[op(&anyhead("F"), &[dot("v")])]);
    assert_eq!(bound_head(&one(&shared, &op(&sin(), &[op(&sin(), &[at("x")])])), "F"), sin());
    assert!(matches(&shared, &op(&sin(), &[op(&cos(), &[at("x")])])).is_empty());
}

#[test]
fn wildcard_head_deep_inside_ordinary_structure() {
    let pat = op(&add(), &[dot("u"), op(&mul(), &[dot("c"), op(&anyhead("F"), &[dot("v")])])]);
    let s = one(&pat, &op(&add(), &[at("a"), op(&mul(), &[at("b"), op(&sin(), &[at("x")])])]));
    assert_eq!(bound_head(&s, "F"), sin());
    assert_eq!(single(&s, "v"), at("x"));
}

// ─── arity / negative cases ──────────────────────────────────────────────────

#[test]
fn arity_and_negatives() {
    let unary = op(&anyhead("F"), &[dot("v")]);
    assert_eq!(matches(&unary, &op(&sin(), &[at("x")])).len(), 1);
    assert!(matches(&unary, &op(&f2(), &[at("x"), at("y")])).is_empty());
    let binary = op(&anyhead("F"), &[dot("u"), dot("v")]);
    assert_eq!(matches(&binary, &op(&f2(), &[at("x"), at("y")])).len(), 1);
    assert!(matches(&binary, &op(&sin(), &[at("x")])).is_empty());
    assert!(matches(&unary, &at("x")).is_empty());
    assert!(matches(&op(&anyhead("F"), &[make_star_variable("s")]), &at("x")).is_empty());
}

// ─── concrete heads are unaffected ───────────────────────────────────────────

#[test]
fn concrete_heads_unaffected() {
    let pat = op(&sin(), &[dot("v")]);
    assert_eq!(matches(&pat, &op(&sin(), &[at("x")])).len(), 1);
    assert!(matches(&pat, &op(&cos(), &[at("x")])).is_empty());

    let named_sin = Head::builder("SIN").arity(Arity::Unary).wildcard(Some("F")).build_unchecked();
    assert_ne!(named_sin, sin());
    assert_eq!(anyhead("F"), anyhead("F"));
    assert_ne!(anyhead("F"), anyhead("G"));
    let set: std::collections::HashSet<Head> = [anyhead("F"), anyhead("F"), anyhead("G"), sin()].into();
    assert_eq!(set.len(), 3);

    let mut m = ManyToOneMatcher::<&str>::new();
    m.add_with_label(Pattern::new(op(&sin(), &[dot("v")])), "concrete");
    m.add_with_label(Pattern::new(op(&anyhead("F"), &[dot("v")])), "generic");
    let labels = |s: &Expr| m.match_all(s).into_iter().map(|(l, _)| *l).collect::<BTreeSet<_>>();
    assert_eq!(labels(&op(&sin(), &[at("x")])), ["concrete", "generic"].into());
    assert_eq!(labels(&op(&cos(), &[at("x")])), ["generic"].into());
}

// ─── Rust-only extras ────────────────────────────────────────────────────────

#[test]
fn substitute_puts_the_bound_head_back() {
    let pat = op(&anyhead("F"), &[dot("v")]);
    let subject = op(&sin(), &[at("x")]);
    let s = one(&pat, &subject);
    assert_eq!(substitute(&pat, &s).unwrap(), Replacement::Single(subject));
}

#[test]
fn constraint_on_head_variable() {
    let not_sin: std::sync::Arc<dyn Constraint> =
        std::sync::Arc::new(CustomConstraint::new(["F"], |args| args[0].as_head().is_some_and(|h| h.name != "SIN")));
    let pattern = Pattern::with_constraints(op(&anyhead("F"), &[dot("v")]), vec![not_sin]);
    let m = ManyToOneMatcher::from_patterns([pattern.clone()]);
    for (subject, expected) in [(op(&sin(), &[at("x")]), false), (op(&cos(), &[at("x")]), true)] {
        assert_eq!(m.is_match(&subject), expected, "many-to-one {subject}");
        assert_eq!(is_match(&subject, &pattern).unwrap(), expected, "one-to-one {subject}");
    }
}
