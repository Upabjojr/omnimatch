//! Pattern constraints.
//!
//! Port of `omnimatch/expressions/constraints.py`.
//!
//! A constraint is a predicate on the match substitution. Constraints that
//! depend on variables are checked during matching as soon as all those
//! variables are bound; constraints without variables are checked once a match
//! is complete.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use crate::expressions::{Expr, ExprKind};
use crate::substitution::{Replacement, Substitution};

/// A predicate on match substitutions.
pub trait Constraint: fmt::Debug + Send + Sync {
    /// The variables the constraint depends on. Empty means "global": the
    /// constraint is only evaluated once the whole match is complete.
    fn variables(&self) -> BTreeSet<String>;

    /// True iff the constraint is satisfied by `subst`.
    fn check(&self, subst: &Substitution) -> bool;

    /// A copy of the constraint with variables renamed.
    fn with_renamed_vars(&self, renaming: &HashMap<String, String>) -> Arc<dyn Constraint>;

    /// A value identifying the constraint for equality and hashing (Python:
    /// `__eq__` / `__hash__`). `None` means "only equal to itself" (compared by
    /// pointer), which is the right default for opaque constraints.
    fn eq_key(&self) -> Option<String> {
        None
    }

    /// A Rust expression (of a type implementing `Constraint`) that rebuilds
    /// this constraint, used by matchers compiled with `omnimatch-compile`.
    /// `None` (the default) means the constraint cannot be compiled.
    fn rust_source(&self) -> Option<String> {
        None
    }
}

/// `["a", "b"]` as Rust source.
fn rust_str_array<'a>(items: impl IntoIterator<Item = &'a String>) -> String {
    let items: Vec<String> = items.into_iter().map(|s| format!("{s:?}")).collect();
    format!("[{}] as [&str; {}]", items.join(", "), items.len())
}

/// An `Arc<dyn Constraint>` compared and hashed by [`Constraint::eq_key`]
/// (falling back to pointer identity), so it can be used in sets and maps.
#[derive(Clone, Debug)]
pub struct ConstraintRef(pub Arc<dyn Constraint>);

impl ConstraintRef {
    fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
}

impl PartialEq for ConstraintRef {
    fn eq(&self, other: &Self) -> bool {
        match (self.0.eq_key(), other.0.eq_key()) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.address() == other.address(),
            _ => false,
        }
    }
}

impl Eq for ConstraintRef {}

impl std::hash::Hash for ConstraintRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.0.eq_key() {
            Some(key) => key.hash(state),
            None => self.address().hash(state),
        }
    }
}

fn rename(name: &str, renaming: &HashMap<String, String>) -> String {
    renaming.get(name).cloned().unwrap_or_else(|| name.to_owned())
}

// ─── EqualVariablesConstraint ────────────────────────────────────────────────

/// Satisfied iff the replacements of all the given variables can be unified.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EqualVariablesConstraint {
    variables: BTreeSet<String>,
}

impl EqualVariablesConstraint {
    pub fn new<I, S>(variables: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        EqualVariablesConstraint { variables: variables.into_iter().map(Into::into).collect() }
    }
}

impl Constraint for EqualVariablesConstraint {
    fn variables(&self) -> BTreeSet<String> {
        self.variables.clone()
    }

    fn check(&self, subst: &Substitution) -> bool {
        let mut unified = Substitution::new();
        for name in &self.variables {
            let Some(value) = subst.get(name) else {
                return true; // not bound yet; checked again later
            };
            if unified.try_add_variable("_", value.clone()).is_err() {
                return false;
            }
        }
        true
    }

    fn with_renamed_vars(&self, renaming: &HashMap<String, String>) -> Arc<dyn Constraint> {
        Arc::new(EqualVariablesConstraint::new(self.variables.iter().map(|v| rename(v, renaming))))
    }

    fn eq_key(&self) -> Option<String> {
        Some(format!("EqualVariables{:?}", self.variables))
    }

    fn rust_source(&self) -> Option<String> {
        Some(format!("::omnimatch::EqualVariablesConstraint::new({})", rust_str_array(&self.variables)))
    }
}

impl fmt::Display for EqualVariablesConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.variables.iter().map(String::as_str).collect();
        write!(f, "({})", names.join(" == "))
    }
}

// ─── CustomConstraint ────────────────────────────────────────────────────────

type Predicate = dyn Fn(&[&Replacement]) -> bool + Send + Sync;

/// A constraint wrapping a closure.
///
/// Python's `CustomConstraint` reads the variable names from the callback's
/// parameter names; Rust closures have none, so the names are given explicitly
/// and the closure receives the replacements positionally, in that order:
///
/// ```
/// use omnimatch::{CustomConstraint, Replacement};
/// let c = CustomConstraint::new(["x", "y"], |args: &[&Replacement]| args[0] != args[1]);
/// ```
#[derive(Clone)]
pub struct CustomConstraint {
    /// Variable names, in the order the closure receives them.
    variables: Vec<String>,
    predicate: Arc<Predicate>,
    label: Option<String>,
    /// Path of a `fn(&[&Replacement]) -> bool` equivalent to the predicate, for compiled matchers.
    rust_path: Option<String>,
}

impl CustomConstraint {
    pub fn new<I, S, F>(variables: I, predicate: F) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        F: Fn(&[&Replacement]) -> bool + Send + Sync + 'static,
    {
        CustomConstraint {
            variables: variables.into_iter().map(Into::into).collect(),
            predicate: Arc::new(predicate),
            label: None,
            rust_path: None,
        }
    }

    /// Name a function equivalent to the predicate, so that matchers using
    /// this constraint can be compiled with `omnimatch-compile`: the compiled
    /// matcher calls the function at `path` (e.g. `"crate::rules::is_positive"`,
    /// resolved where the generated code is included), which must have the
    /// signature `fn(&[&Replacement]) -> bool`.
    ///
    /// ```
    /// use omnimatch::{CustomConstraint, Replacement};
    /// fn distinct(args: &[&Replacement]) -> bool {
    ///     args[0] != args[1]
    /// }
    /// let c = CustomConstraint::new(["x", "y"], distinct).with_rust_path("crate::distinct");
    /// ```
    pub fn with_rust_path(mut self, path: impl Into<String>) -> Self {
        self.rust_path = Some(path.into());
        self
    }

    /// Attach a human-readable label, used by `Debug` (Python shows the lambda source).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Constraint for CustomConstraint {
    fn variables(&self) -> BTreeSet<String> {
        self.variables.iter().cloned().collect()
    }

    fn check(&self, subst: &Substitution) -> bool {
        let mut args = Vec::with_capacity(self.variables.len());
        for name in &self.variables {
            match subst.get(name) {
                Some(value) => args.push(value),
                None => return true, // not bound yet; checked again later
            }
        }
        (self.predicate)(&args)
    }

    fn with_renamed_vars(&self, renaming: &HashMap<String, String>) -> Arc<dyn Constraint> {
        Arc::new(CustomConstraint {
            variables: self.variables.iter().map(|v| rename(v, renaming)).collect(),
            predicate: self.predicate.clone(),
            label: self.label.clone(),
            rust_path: self.rust_path.clone(),
        })
    }

    /// Equal iff the same closure (Python: the same callable) on the same variables.
    fn eq_key(&self) -> Option<String> {
        Some(format!("Custom{:p}{:?}", Arc::as_ptr(&self.predicate) as *const (), self.variables))
    }

    fn rust_source(&self) -> Option<String> {
        let path = self.rust_path.as_ref()?;
        let mut source = format!(
            "::omnimatch::CustomConstraint::new({}, {path}).with_rust_path({path:?})",
            rust_str_array(&self.variables)
        );
        if let Some(label) = &self.label {
            source.push_str(&format!(".with_label({label:?})"));
        }
        Some(source)
    }
}

impl fmt::Debug for CustomConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CustomConstraint({})", self.label.as_deref().unwrap_or("UNKNOWN"))
    }
}

// ─── FreeOf ──────────────────────────────────────────────────────────────────

/// Satisfied when none of the expressions bound to `variables` contains an atom
/// named `symbol_name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FreeOf {
    variables: Vec<String>,
    symbol_name: String,
}

impl FreeOf {
    pub fn new<I, S>(variables: I, symbol_name: impl Into<String>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        FreeOf { variables: variables.into_iter().map(Into::into).collect(), symbol_name: symbol_name.into() }
    }

    /// Single-variable shorthand: `FreeOf::single("y", "x")`.
    pub fn single(variable: impl Into<String>, symbol_name: impl Into<String>) -> Self {
        FreeOf { variables: vec![variable.into()], symbol_name: symbol_name.into() }
    }

    pub fn variable_names(&self) -> &[String] {
        &self.variables
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    fn is_free(&self, exprs: &[Expr]) -> bool {
        let mut stack: Vec<&Expr> = exprs.iter().collect();
        while let Some(node) = stack.pop() {
            match node.kind() {
                ExprKind::Atom { name } if *name == self.symbol_name => return false,
                ExprKind::Operation { operands, .. } => stack.extend(operands),
                _ => {}
            }
        }
        true
    }
}

impl Constraint for FreeOf {
    fn variables(&self) -> BTreeSet<String> {
        self.variables.iter().cloned().collect()
    }

    /// Note: unlike the Python version, multiset-valued bindings (sequence
    /// variables in commutative operations) are checked too; Python's `_is_free`
    /// only descends into tuples/lists and silently accepts multisets.
    fn check(&self, subst: &Substitution) -> bool {
        self.variables.iter().all(|name| match subst.get(name) {
            Some(value) => self.is_free(&value.to_vec()),
            None => true, // not bound yet; checked again later
        })
    }

    fn with_renamed_vars(&self, renaming: &HashMap<String, String>) -> Arc<dyn Constraint> {
        Arc::new(FreeOf {
            variables: self.variables.iter().map(|v| rename(v, renaming)).collect(),
            symbol_name: self.symbol_name.clone(),
        })
    }

    fn eq_key(&self) -> Option<String> {
        Some(format!("FreeOf{:?}{:?}", self.variables, self.symbol_name))
    }

    fn rust_source(&self) -> Option<String> {
        Some(format!("::omnimatch::FreeOf::new({}, {:?})", rust_str_array(&self.variables), self.symbol_name))
    }
}

impl fmt::Display for FreeOf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.variables.len() == 1 {
            write!(f, "FreeOf({}, {})", self.variables[0], self.symbol_name)
        } else {
            write!(f, "FreeOf([{}], {})", self.variables.join(", "), self.symbol_name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::{Arity, Head};

    #[test]
    fn free_of() {
        let f = Head::new("f", Arity::Variadic).unwrap();
        let c = FreeOf::single("y", "x");
        let mut s = Substitution::new();
        assert!(c.check(&s));
        s.insert("y", f.call(vec![Expr::atom("a")]));
        assert!(c.check(&s));
        s.insert("y", f.call(vec![Expr::atom("x")]));
        assert!(!c.check(&s));
        s.insert("y", Replacement::Multiset([Expr::atom("x")].into_iter().collect()));
        assert!(!c.check(&s));
        let renamed = c.with_renamed_vars(&HashMap::from([("y".into(), "z".into())]));
        assert_eq!(renamed.variables(), BTreeSet::from(["z".to_string()]));
    }

    #[test]
    fn equal_variables() {
        let c = EqualVariablesConstraint::new(["x", "y"]);
        let mut s: Substitution = [("x", Expr::atom("a")), ("y", Expr::atom("a"))].into_iter().collect();
        assert!(c.check(&s));
        s.insert("y", Expr::atom("b"));
        assert!(!c.check(&s));
        assert_eq!(c.to_string(), "(x == y)");
    }

    #[test]
    fn custom() {
        let c = CustomConstraint::new(["x"], |args: &[&Replacement]| {
            args[0].as_single().and_then(Expr::atom_name).is_some_and(|n| n.starts_with('a'))
        });
        let s: Substitution = [("x", Expr::atom("a1"))].into_iter().collect();
        assert!(c.check(&s));
        let s: Substitution = [("x", Expr::atom("b1"))].into_iter().collect();
        assert!(!c.check(&s));
    }
}
