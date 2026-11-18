//! Port of `tests/test_constraints.py`.
//!
//! Not ported: `test_custom_constraint_errors` (`*args` / `**kwargs` callbacks have
//! no Rust analogue: variable names are given explicitly) and `test_constraint_vars`
//! (there is no instantiable base `Constraint`).
//!
//! Python's `==` / `hash` on constraints correspond to `Constraint::eq_key`, and to
//! `ConstraintRef` for type-erased constraints.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use omnimatch::*;

fn n(i: u32) -> Expr {
    Expr::atom(i.to_string())
}

fn subst(pairs: &[(&str, u32)]) -> Substitution {
    pairs.iter().map(|(k, v)| (k.to_string(), n(*v))).collect()
}

fn set(xs: &[&str]) -> BTreeSet<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

fn hash_of(c: &Arc<dyn Constraint>) -> u64 {
    let mut h = DefaultHasher::new();
    ConstraintRef(c.clone()).hash(&mut h);
    h.finish()
}

/// Constraints that are pairwise different must be unequal and hash differently;
/// each must equal (and hash like) itself.
fn assert_pairwise_distinct(constraints: &[Arc<dyn Constraint>]) {
    for (i, c1) in constraints.iter().enumerate() {
        for (j, c2) in constraints.iter().enumerate() {
            let equal = ConstraintRef(c1.clone()) == ConstraintRef(c2.clone());
            assert_eq!(equal, i == j, "{c1:?} vs {c2:?}");
            assert_eq!(hash_of(c1) == hash_of(c2), i == j, "hash {c1:?} vs {c2:?}");
        }
    }
}

fn is_zero(r: &Replacement) -> bool {
    *r == Replacement::Single(n(0))
}

#[test]
fn equal_variables_call() {
    let c = EqualVariablesConstraint::new(["x", "y"]);
    assert!(c.check(&subst(&[("x", 0), ("y", 0)])));
    assert!(!c.check(&subst(&[("x", 0), ("y", 1)])));
}

#[test]
fn equal_variables_eq_and_hash() {
    let constraints: Vec<Arc<dyn Constraint>> = vec![
        Arc::new(EqualVariablesConstraint::new(Vec::<String>::new())),
        Arc::new(EqualVariablesConstraint::new(["x"])),
        Arc::new(EqualVariablesConstraint::new(["y"])),
        Arc::new(EqualVariablesConstraint::new(["x", "y"])),
    ];
    assert_pairwise_distinct(&constraints);
    // Equal by value, not identity.
    assert_eq!(
        ConstraintRef(Arc::new(EqualVariablesConstraint::new(["x", "y"]))),
        ConstraintRef(Arc::new(EqualVariablesConstraint::new(["y", "x"])))
    );
}

fn custom_constraints() -> Vec<Arc<dyn Constraint>> {
    vec![
        Arc::new(CustomConstraint::new(["x"], |args| is_zero(args[0]))),
        Arc::new(CustomConstraint::new(["y"], |args| is_zero(args[0]))),
        Arc::new(CustomConstraint::new(["x", "y"], |args| args[0] == args[1])),
    ]
}

#[test]
fn custom_call() {
    let cs = custom_constraints();
    let cases = [
        (0, [("x", 0), ("y", 0)], true),
        (0, [("x", 1), ("y", 0)], false),
        (1, [("x", 0), ("y", 0)], true),
        (1, [("x", 0), ("y", 1)], false),
        (2, [("x", 0), ("y", 0)], true),
        (2, [("x", 0), ("y", 1)], false),
        (2, [("x", 1), ("y", 0)], false),
        (2, [("x", 1), ("y", 1)], true),
    ];
    for (i, s, expected) in cases {
        assert_eq!(cs[i].check(&subst(&s)), expected, "constraint {i} on {s:?}");
    }
}

#[test]
fn custom_eq_and_hash() {
    assert_pairwise_distinct(&custom_constraints());
    // Clones share the closure, so they compare equal (Python: same callable).
    let c = CustomConstraint::new(["x"], |args| is_zero(args[0]));
    assert_eq!(ConstraintRef(Arc::new(c.clone())), ConstraintRef(Arc::new(c)));
}

#[test]
fn variables() {
    assert_eq!(EqualVariablesConstraint::new(["x", "y"]).variables(), set(&["x", "y"]));
    assert_eq!(CustomConstraint::new(["x", "y"], |_| true).variables(), set(&["x", "y"]));
}

#[test]
fn equal_variables_with_renamed_vars() {
    let c2 = EqualVariablesConstraint::new(["x", "y"]).with_renamed_vars(&HashMap::from([("x".into(), "z".into())]));
    assert!(!c2.check(&subst(&[("x", 1), ("z", 2), ("y", 1)])));
    assert!(c2.check(&subst(&[("x", 1), ("z", 2), ("y", 2)])));
}

#[test]
fn custom_with_renamed_vars() {
    let seen: Arc<Mutex<Option<(Replacement, Replacement)>>> = Arc::new(Mutex::new(None));
    let record = seen.clone();
    let c1 = CustomConstraint::new(["x", "y"], move |args| {
        *record.lock().unwrap() = Some((args[0].clone(), args[1].clone()));
        args[0] == args[1]
    });
    let c2 = c1.with_renamed_vars(&HashMap::from([("x".into(), "z".into())]));

    assert!(!c2.check(&subst(&[("x", 1), ("z", 2), ("y", 1)])));
    assert_eq!(*seen.lock().unwrap(), Some((Replacement::Single(n(2)), Replacement::Single(n(1)))));
    assert!(c2.check(&subst(&[("x", 1), ("z", 3), ("y", 3)])));
    assert_eq!(*seen.lock().unwrap(), Some((Replacement::Single(n(3)), Replacement::Single(n(3)))));
}

#[test]
fn custom_repeated_renaming() {
    let rename =
        |c: &dyn Constraint, from: &str, to: &str| c.with_renamed_vars(&HashMap::from([(from.into(), to.into())]));
    let c1 = CustomConstraint::new(["x1", "y"], |_| true);
    assert_eq!(c1.variables(), set(&["x1", "y"]));
    let c2 = rename(&c1, "x1", "x2");
    assert_eq!(c2.variables(), set(&["x2", "y"]));
    let c3 = rename(c2.as_ref(), "x2", "x3");
    assert_eq!(c3.variables(), set(&["x3", "y"]));
    let c4 = rename(c3.as_ref(), "z1", "z2");
    assert_eq!(c4.variables(), set(&["x3", "y"]));
}
