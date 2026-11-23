//! The patterns and subjects compiled by `build.rs` and checked by the tests.
//! This file is compiled twice: into the build script and into the library.

// The pattern tables read best as literal operand lists.
#![allow(clippy::cloned_ref_to_slice_refs)]

use omnimatch::{
    make_dot_variable, make_plus_variable, make_star_variable, Arity, Constraint, CustomConstraint,
    EqualVariablesConstraint, Expr, FreeOf, Head, Pattern, Replacement,
};
use std::sync::Arc;

pub fn head(name: &str) -> Head {
    let b = Head::builder(name);
    match name {
        "f" | "f2" | "g" => b,
        "f_u" => b.arity(Arity::Unary),
        "f_i" => b.one_identity(true),
        "f_c" | "f2_c" => b.commutative(true),
        "f_ci" => b.commutative(true).one_identity(true),
        "f_a" => b.associative(true),
        "f_ac" => b.associative(true).commutative(true),
        "f_aci" => b.associative(true).commutative(true).one_identity(true),
        _ => panic!("unknown head {name}"),
    }
    .build()
    .unwrap()
}

pub fn op(name: &str, operands: &[Expr]) -> Expr {
    head(name).call(operands.to_vec())
}

fn named(name: &str, operands: &[Expr], variable: &str) -> Expr {
    head(name).call_named(operands.to_vec(), variable)
}

fn any_head(variable: &str, operands: &[Expr]) -> Expr {
    Head::any(Some(variable)).call(operands.to_vec())
}

pub fn a() -> Expr {
    Expr::atom("a")
}
pub fn b() -> Expr {
    Expr::atom("b")
}
pub fn c() -> Expr {
    Expr::atom("c")
}

// ── functions for custom constraints (named in `with_rust_path`) ──

/// `x` is the atom `a`.
pub fn is_a(args: &[&Replacement]) -> bool {
    args[0].as_single() == Some(&Expr::atom("a"))
}

/// The two values differ.
pub fn distinct(args: &[&Replacement]) -> bool {
    args[0] != args[1]
}

/// A global constraint (no variables) that never holds.
pub fn never(_: &[&Replacement]) -> bool {
    false
}

fn constrained(expression: Expr, constraints: Vec<Arc<dyn Constraint>>) -> Pattern {
    Pattern::with_constraints(expression, constraints)
}

/// The compiled patterns. Their positions are the pattern indices.
pub fn patterns() -> Vec<Pattern> {
    let (x_, y_, z_) = (make_dot_variable("x"), make_dot_variable("y"), make_dot_variable("z"));
    let (x__, y__) = (make_plus_variable("x"), make_plus_variable("y"));
    let (x___, y___, z___) = (make_star_variable("x"), make_star_variable("y"), make_star_variable("z"));
    let oa_ = Expr::optional("o", a());
    let oa___ = Expr::wildcard(0, false, Some("o"), Some(a())).unwrap();
    let w_ = Expr::dot(None);
    let w__ = Expr::plus(None);
    let w___ = Expr::star(None);

    let plain = vec![
        // constants
        a(),
        op("f", &[a()]),
        op("f", &[a(), b()]),
        op("f_c", &[a(), b()]),
        op("f", &[op("f_c", &[a(), b()])]),
        op("f_a", &[a(), b()]),
        op("f_ac", &[a(), b(), a()]),
        // dot wildcards
        x_.clone(),
        op("f", &[x_.clone()]),
        op("f", &[x_.clone(), y_.clone()]),
        op("f", &[x_.clone(), y_.clone(), z_.clone()]),
        op("f", &[x_.clone(), x_.clone()]),
        op("f", &[a(), x_.clone()]),
        op("f", &[w_.clone(), w_.clone()]),
        op("f_u", &[x_.clone()]),
        op("f_c", &[x_.clone(), a()]),
        op("f_c", &[x_.clone(), y_.clone()]),
        op("f_c", &[x_.clone(), x_.clone()]),
        op("f_c", &[w_.clone(), a()]),
        op("f_a", &[x_.clone(), y_.clone()]),
        op("f_a", &[a(), x_.clone()]),
        op("f_ac", &[x_.clone(), y_.clone()]),
        op("f_ac", &[x_.clone(), a()]),
        op("f_ac", &[x_.clone(), x_.clone(), y_.clone()]),
        // optional wildcards
        op("f", &[x_.clone(), oa_.clone()]),
        op("f", &[oa_.clone()]),
        op("f_c", &[x_.clone(), oa_.clone()]),
        op("f_ac", &[oa_.clone(), x_.clone()]),
        op("f", &[x_.clone(), oa___.clone()]),
        // sequence wildcards
        op("f", &[x___.clone()]),
        op("f", &[x__.clone(), y__.clone()]),
        op("f", &[x___.clone(), a(), y___.clone()]),
        op("f", &[x__.clone(), x__.clone()]),
        op("f", &[w___.clone(), b(), w__.clone()]),
        op("f", &[x___.clone(), y___.clone(), z___.clone()]),
        op("f_c", &[x___.clone()]),
        op("f_c", &[x__.clone(), a()]),
        op("f_c", &[x___.clone(), y___.clone()]),
        op("f_c", &[x__.clone(), x__.clone()]),
        op("f_c", &[w__.clone(), x_.clone()]),
        op("f_a", &[x__.clone(), y_.clone()]),
        op("f_a", &[x___.clone(), b()]),
        op("f_ac", &[x__.clone(), y__.clone()]),
        op("f_ac", &[x___.clone(), a()]),
        // one identity
        op("f_i", &[x_.clone(), oa_.clone()]),
        op("f_i", &[x___.clone(), a()]),
        op("f_ci", &[x_.clone(), a()]),
        op("f_ci", &[x_.clone(), y___.clone()]),
        op("f_aci", &[x_.clone(), z___.clone()]),
        // nesting
        op("f", &[op("f", &[x_.clone()]), y___.clone()]),
        op("f_c", &[op("f2_c", &[x_.clone(), a()]), y___.clone()]),
        op("f_c", &[op("f", &[x_.clone()]), op("f", &[y_.clone()])]),
        op("f_c", &[op("f_c", &[x_.clone(), b()]), x_.clone()]),
        op("f_ac", &[op("f_c", &[x___.clone()]), y_.clone()]),
        op("f", &[op("f_a", &[x__.clone(), y_.clone()]), x__.clone()]),
        // named subexpressions
        op("f", &[named("f_c", &[a(), x_.clone()], "y")]),
        named("f_c", &[a(), x_.clone()], "z"),
        op("f", &[named("f", &[a(), x_.clone()], "y"), y_.clone()]),
        op("f", &[a().with_variable_name(Some("x")), x_.clone()]),
        // wildcard heads
        any_head("F", &[x_.clone()]),
        any_head("F", &[a(), x___.clone()]),
        op("f", &[any_head("F", &[x_.clone()]), any_head("F", &[y_.clone()])]),
        op("f_c", &[any_head("G", &[a()]), y___.clone()]),
    ];
    let mut patterns: Vec<Pattern> = plain.into_iter().map(Pattern::new).collect();

    let path = |c: CustomConstraint, p: &str| -> Arc<dyn Constraint> { Arc::new(c.with_rust_path(p)) };
    patterns.extend([
        constrained(op("f", &[x_.clone(), y_.clone()]), vec![Arc::new(EqualVariablesConstraint::new(["x", "y"]))]),
        constrained(op("f_c", &[x_.clone(), y___.clone()]), vec![Arc::new(FreeOf::single("y", "a"))]),
        constrained(op("f", &[x___.clone()]), vec![Arc::new(FreeOf::single("x", "b"))]),
        constrained(
            op("f", &[x_.clone(), y_.clone()]),
            vec![path(CustomConstraint::new(["x"], is_a), "crate::cases::is_a")],
        ),
        constrained(
            op("f_c", &[x_.clone(), y_.clone(), z___.clone()]),
            vec![path(CustomConstraint::new(["x", "y"], distinct), "crate::cases::distinct")],
        ),
        constrained(
            op("f", &[x_.clone()]),
            vec![path(CustomConstraint::new(Vec::<String>::new(), never), "crate::cases::never")],
        ),
        constrained(
            any_head("F", &[x_.clone(), y_.clone()]),
            vec![path(CustomConstraint::new(["x", "y"], distinct), "crate::cases::distinct")],
        ),
    ]);
    patterns
}

/// Subjects: a hand-picked list plus every small expression over a few heads.
pub fn subjects() -> Vec<Expr> {
    let heads = ["f", "f_c", "f_a", "f_ac", "f_i", "f_ci", "f_u", "g"];
    let atoms = [a(), b()];
    let mut level1: Vec<Expr> = atoms.to_vec();
    for &h in &heads {
        let arities: &[usize] = if h == "f_u" { &[1] } else { &[0, 1, 2, 3] };
        for &n in arities {
            for operands in sequences(&atoms, n) {
                level1.push(op(h, &operands));
            }
        }
    }
    let pool: Vec<Expr> = vec![
        a(),
        b(),
        op("f", &[a()]),
        op("f", &[b()]),
        op("f", &[a(), b()]),
        op("f_c", &[a(), b()]),
        op("f_c", &[a()]),
        op("f_a", &[a(), b()]),
        op("f_ac", &[a(), b()]),
        op("g", &[a()]),
        op("f2_c", &[a(), b()]),
        op("f_c", &[b(), b()]),
    ];
    let mut out = level1;
    for &h in &["f", "f_c", "f_ac", "f_a"] {
        for n in 1..=3 {
            for operands in sequences(&pool, n) {
                if n == 3 && operands.iter().filter(|e| e.is_operation()).count() > 1 {
                    continue;
                }
                out.push(op(h, &operands));
            }
        }
    }
    out.extend([
        op("f", &[a(), b(), c(), a(), b()]),
        op("f_c", &[a(), a(), b(), b(), c()]),
        op("f_ac", &[a(), b(), c(), op("f_c", &[a()])]),
        op("f", &[op("f_a", &[a(), b(), c()]), a(), b(), c()]),
        op("f_c", &[op("f_c", &[a(), b()]), a()]),
        op("f_c", &[op("f2_c", &[a(), c()]), op("f2_c", &[a(), b()]), b()]),
    ]);
    out
}

fn sequences(items: &[Expr], n: usize) -> Vec<Vec<Expr>> {
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                items.iter().map(move |e| {
                    let mut v = prefix.clone();
                    v.push(e.clone());
                    v
                })
            })
            .collect();
    }
    out
}
