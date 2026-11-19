//! Port of `tests/test_expressions.py`.
//!
//! Not ported (no Rust counterpart):
//! * `test_getitem_slice`, `test_getitem_slice_symbol` — slice access by start/end positions;
//! * `test_lt_error` — comparing an expression with a non-expression is a type error at compile time;
//! * `test_copy` — `Expr` is an immutable `Arc`; cloning shares the node;
//! * `Wildcard(-1, False)` in `test_wildcard_error` — `min_count` is a `usize`.

#![allow(clippy::type_complexity)]

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use omnimatch::*;

fn head(name: &str) -> Head {
    let b = Head::builder(name);
    match name {
        "f" | "f2" | "special" => b,
        "f_i" => b.one_identity(true),
        "f_c" => b.commutative(true),
        "f_a" => b.associative(true),
        "f_ac" => b.associative(true).commutative(true),
        _ => panic!("unknown head {name}"),
    }
    .build()
    .unwrap()
}

macro_rules! op {
    ($name:ident) => {
        fn $name(ops: &[Expr]) -> Expr {
            head(stringify!($name)).call(ops.to_vec())
        }
    };
}
op!(f);
op!(f2);
op!(f_i);
op!(f_c);
op!(f_a);
op!(f_ac);
op!(special);

fn a() -> Expr {
    Expr::atom("a")
}
fn b() -> Expr {
    Expr::atom("b")
}
fn c() -> Expr {
    Expr::atom("c")
}
fn w() -> Expr {
    Expr::dot(None)
}
fn x_() -> Expr {
    make_dot_variable("x")
}
fn y_() -> Expr {
    make_dot_variable("y")
}
fn __() -> Expr {
    Expr::plus(None)
}
fn x__() -> Expr {
    make_plus_variable("x")
}
fn ___() -> Expr {
    Expr::star(None)
}
fn x___() -> Expr {
    make_star_variable("x")
}

#[test]
fn operation_simplify() {
    let cases = [
        (f_i(&[a()]), a()),
        (f_i(&[a(), b()]), f_i(&[a(), b()])),
        (f_i(&[w()]), w()),
        (f_i(&[___()]), f_i(&[___()])),
        (f_i(&[__()]), f_i(&[__()])),
        (f_i(&[x_()]), x_()),
        (f_i(&[x___()]), f_i(&[x___()])),
        (f_i(&[x__()]), f_i(&[x__()])),
        (f_a(&[f_a(&[a()])]), f_a(&[a()])),
        (f_a(&[f_a(&[a(), b()])]), f_a(&[a(), b()])),
        (f_a(&[a(), f_a(&[b()])]), f_a(&[a(), b()])),
        (f_a(&[f_a(&[a()]), b()]), f_a(&[a(), b()])),
        (f_a(&[f(&[a()])]), f_a(&[f(&[a()])])),
        (f_c(&[a(), b()]), f_c(&[a(), b()])),
        (f_c(&[b(), a()]), f_c(&[a(), b()])),
    ];
    for (expression, simplified) in cases {
        assert_eq!(expression, simplified);
    }
    // The raw constructor flattens and sorts too, but never applies one_identity.
    assert!(Expr::operation(head("f_i"), vec![a()]).is_operation());
}

#[test]
fn operation_errors() {
    let unary = Head::new("f", Arity::Unary).unwrap();
    let variadic = Head::new("f", Arity::Variadic).unwrap();
    let binary = Head::new("f", Arity::Binary).unwrap();
    let binary_assoc = Head::builder("f").arity(Arity::Binary).associative(true).build().unwrap();
    let cases: Vec<(&Head, Vec<Expr>, bool)> = vec![
        (&unary, vec![], true),
        (&unary, vec![a(), b()], true),
        (&variadic, vec![], false),
        (&variadic, vec![a()], false),
        (&variadic, vec![a(), b()], false),
        (&binary_assoc, vec![a(), a(), b()], true),
        (&binary, vec![x_(), x___()], false),
        (&binary, vec![x_(), x__()], false),
        (&binary, vec![x_(), x_(), x__()], true),
        (&binary, vec![x_(), x_(), x___()], false),
        (&binary, vec![x_(), x_()], false),
        (&binary, vec![x_(), x_(), x_()], true),
    ];
    for (h, operands, is_error) in cases {
        let result = h.try_call(operands.clone());
        assert_eq!(result.is_err(), is_error, "{} {operands:?}: {result:?}", h.name);
        if is_error {
            assert!(matches!(result, Err(ExprError::Arity(_))));
        }
    }
}

#[test]
fn is_constant() {
    let cases =
        [(a(), true), (x_(), false), (w(), false), (f(&[a()]), true), (f(&[a(), b()]), true), (f(&[x_()]), false)];
    for (expression, expected) in cases {
        assert_eq!(expression.is_constant(), expected, "{expression}");
    }
}

#[test]
fn is_syntactic() {
    let cases = [
        (a(), true),
        (x_(), true),
        (w(), true),
        (x___(), false),
        (___(), false),
        (x__(), false),
        (__(), false),
        (f(&[a()]), true),
        (f(&[a(), b()]), true),
        (f(&[x_()]), true),
        (f(&[x__()]), false),
        (f_a(&[a()]), false),
        (f_a(&[a(), b()]), false),
        (f_a(&[x_()]), false),
        (f_a(&[x__()]), false),
        (f_c(&[a()]), false),
        (f_c(&[a(), b()]), false),
        (f_c(&[x_()]), false),
        (f_c(&[x__()]), false),
        (f_ac(&[a()]), false),
        (f_ac(&[a(), b()]), false),
        (f_ac(&[x_()]), false),
        (f_ac(&[x__()]), false),
    ];
    for (expression, expected) in cases {
        assert_eq!(expression.is_syntactic(), expected, "{expression}");
    }
}

fn names(xs: &[&str]) -> Multiset<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

#[test]
fn symbols() {
    let cases: Vec<(Expr, Vec<&str>)> = vec![
        (a(), vec!["a"]),
        (x_(), vec![]),
        (w(), vec![]),
        (f(&[a()]), vec!["a", "f"]),
        (f(&[a(), b()]), vec!["a", "b", "f"]),
        (f(&[x_()]), vec!["f"]),
        (f(&[a(), a()]), vec!["a", "a", "f"]),
        (f(&[f(&[a()]), f(&[b(), c()])]), vec!["a", "b", "c", "f", "f", "f"]),
    ];
    for (expression, expected) in cases {
        assert_eq!(expression.symbols(), names(&expected), "{expression}");
    }
}

#[test]
fn variables() {
    let fh = head("f");
    let cases: Vec<(Expr, Vec<&str>)> = vec![
        (a(), vec![]),
        (x_(), vec!["x"]),
        (w(), vec![]),
        (f(&[a()]), vec![]),
        (f(&[x_()]), vec!["x"]),
        (f(&[x_(), x_()]), vec!["x", "x"]),
        (f(&[x_(), a()]), vec!["x"]),
        (f(&[x_(), a(), y_()]), vec!["x", "y"]),
        (f(&[f(&[x_()]), f(&[b(), x_()])]), vec!["x", "x"]),
        (fh.call_named(vec![a()], "x"), vec!["x"]),
        (fh.call_named(vec![f(&[y_()])], "x"), vec!["x", "y"]),
    ];
    for (expression, expected) in cases {
        assert_eq!(expression.variables(), names(&expected), "{expression}");
    }
}

#[test]
fn preorder_iter() {
    let pos = |e: Expr, p: &[usize]| (e, p.to_vec());

    let e = f(&[a(), x_()]);
    assert_eq!(e.preorder_iter(), vec![pos(e.clone(), &[]), pos(a(), &[0]), pos(x_(), &[1])]);

    // Python: predicate `e.head is None` (only wildcards have no head).
    let e = f(&[a(), f(&[x_()])]);
    let wildcards: Vec<_> = e.preorder_iter().into_iter().filter(|(x, _)| x.is_wildcard()).collect();
    assert_eq!(wildcards, vec![pos(x_(), &[1, 0])]);

    // Python: predicate `e.head == f`.
    let with_f: Vec<_> = e.preorder_iter().into_iter().filter(|(x, _)| x.head() == Some(&head("f"))).collect();
    assert_eq!(with_f, vec![pos(e.clone(), &[]), pos(f(&[x_()]), &[1])]);
}

#[test]
fn getitem() {
    let e = f(&[a(), f(&[x_(), b()]), w()]);
    let cases: Vec<(Vec<usize>, Option<Expr>)> = vec![
        (vec![], Some(e.clone())),
        (vec![0], Some(a())),
        (vec![0, 0], None),
        (vec![1], Some(f(&[x_(), b()]))),
        (vec![1, 0], Some(x_())),
        (vec![1, 0, 0], None),
        (vec![1, 1], Some(b())),
        (vec![1, 1, 0], None),
        (vec![1, 2], None),
        (vec![2], Some(w())),
        (vec![3], None),
    ];
    for (position, expected) in cases {
        match expected {
            Some(expected) => assert_eq!(e.get(&position).unwrap(), &expected, "{position:?}"),
            None => assert!(matches!(e.get(&position), Err(ExprError::InvalidPosition(_))), "{position:?}"),
        }
    }
}

#[test]
fn ordering() {
    let a_x = a().with_variable_name(Some("x"));
    let a_y = a().with_variable_name(Some("y"));
    let cases = [
        (a(), b()),
        (a(), a_x.clone()),
        (a_x, a_y),
        (a(), w()),
        (a(), x_()),
        (w(), x_()),
        (x_(), y_()),
        (x_(), x__()),
        (f(&[a()]), f(&[b()])),
        (f(&[a()]), f2(&[a()])),
        (f(&[a()]), f(&[a(), a()])),
        (f(&[b()]), f(&[a(), a()])),
        (f(&[a(), a()]), f(&[a(), b()])),
        (f(&[a(), a()]), f(&[a(), a(), a()])),
        (a(), f(&[a()])),
        (x_(), f(&[a()])),
        (w(), f(&[a()])),
        (f(&[a()]), special(&[a()])),
    ];
    for (e1, e2) in cases {
        assert!(e1 < e2, "{e1} < {e2} did not hold");
        assert!(!(e2 < e1), "inconsistent order: both {e2} < {e1} and {e1} < {e2}");
    }
}

#[test]
fn head_and_wildcard_errors() {
    assert_eq!(Head::new("if", Arity::Variadic), Err(ExprError::InvalidName("if".into())));
    assert_eq!(Head::new("+", Arity::Variadic), Err(ExprError::InvalidName("+".into())));
    assert!(matches!(Expr::wildcard(0, true, None, None), Err(ExprError::InvalidWildcard(_))));
    for arity in [Arity::Unary, Arity::Binary] {
        assert_eq!(
            Head::builder("Invalid").arity(arity).one_identity(true).build(),
            Err(ExprError::OneIdentityNotVariadic)
        );
    }
    assert_eq!(Head::builder("Invalid").arity(Arity::Unary).infix(true).build(), Err(ExprError::InfixUnary));
}

#[test]
fn with_renamed_vars() {
    let none: HashMap<String, String> = HashMap::new();
    let x_to_y: HashMap<String, String> = HashMap::from([("x".into(), "y".into())]);
    let cases = [
        (a(), &none, a()),
        (a(), &x_to_y, a()),
        (x_(), &none, x_()),
        (x_(), &x_to_y, y_()),
        (f(&[x_()]), &none, f(&[x_()])),
        (f(&[x_()]), &x_to_y, f(&[y_()])),
    ];
    for (expression, renaming, expected) in cases {
        assert_eq!(expression.with_renamed_vars(renaming), expected, "{expression}");
    }
}

fn hash_of(e: &Expr) -> u64 {
    let mut h = DefaultHasher::new();
    e.hash(&mut h);
    h.finish()
}

#[test]
fn hash_and_eq() {
    let simple = [a(), b(), f(&[a(), b()]), x_(), ___(), head("f").call_named(vec![w()], "x")];
    for (i, e1) in simple.iter().enumerate() {
        for (j, e2) in simple.iter().enumerate() {
            assert_eq!(e1 == e2, i == j, "{e1} vs {e2}");
            assert_eq!(hash_of(e1) == hash_of(e2), i == j, "hash({e1}) vs hash({e2})");
        }
    }
}

#[test]
fn contains() {
    let cases = [
        (a(), a(), true),
        (a(), b(), false),
        (f(&[a()]), a(), true),
        (f(&[a()]), b(), false),
        (f(&[a()]), f(&[a()]), true),
        (f(&[a(), b()]), f(&[a()]), false),
        (f(&[a()]), f(&[a(), b()]), false),
        (f(&[x_(), y_()]), x_(), true),
        (f(&[x_(), y_()]), y_(), true),
        (f(&[x_(), y_()]), a(), false),
    ];
    for (expression, sub, expected) in cases {
        assert_eq!(expression.contains(&sub), expected, "{sub} in {expression}");
    }
}
