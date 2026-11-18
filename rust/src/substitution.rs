//! Substitutions: maps from variable names to their replacements.
//!
//! Port of `omnimatch/expressions/substitution.py`.

use std::collections::btree_map;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::expressions::{Expr, ExprKind};
use crate::multiset::Multiset;

/// The value bound to a variable.
///
/// Python uses a bare expression, a tuple, or a `Multiset`; these are the three
/// variants here.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Replacement {
    /// A single expression (dot wildcards, named subexpressions).
    Single(Expr),
    /// An ordered sequence (sequence wildcards in non-commutative operations).
    Sequence(Vec<Expr>),
    /// An unordered collection (sequence wildcards in commutative operations).
    Multiset(Multiset<Expr>),
}

impl Replacement {
    /// The bound expressions as a list (a single expression becomes a one-element list).
    pub fn to_vec(&self) -> Vec<Expr> {
        match self {
            Replacement::Single(e) => vec![e.clone()],
            Replacement::Sequence(v) => v.clone(),
            Replacement::Multiset(m) => m.iter().cloned().collect(),
        }
    }

    pub fn as_single(&self) -> Option<&Expr> {
        match self {
            Replacement::Single(e) => Some(e),
            _ => None,
        }
    }

    fn is_collection(&self) -> bool {
        !matches!(self, Replacement::Single(_))
    }
}

impl From<Expr> for Replacement {
    fn from(e: Expr) -> Self {
        Replacement::Single(e)
    }
}

impl From<Vec<Expr>> for Replacement {
    fn from(v: Vec<Expr>) -> Self {
        Replacement::Sequence(v)
    }
}

impl From<Multiset<Expr>> for Replacement {
    fn from(m: Multiset<Expr>) -> Self {
        Replacement::Multiset(m)
    }
}

impl fmt::Display for Replacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Replacement::Single(e) => write!(f, "{e}"),
            Replacement::Sequence(v) => {
                write!(f, "(")?;
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            Replacement::Multiset(m) => write!(f, "{m}"),
        }
    }
}

/// Two replacements for the same variable could not be unified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conflict;

impl fmt::Display for Conflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conflicting replacements for a variable")
    }
}

impl std::error::Error for Conflict {}

/// A substitution, mapping variable names to replacements.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Substitution {
    map: BTreeMap<String, Replacement>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Replacement> {
        self.map.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Insert unconditionally, overwriting any existing replacement.
    pub fn insert(&mut self, name: impl Into<String>, replacement: impl Into<Replacement>) {
        self.map.insert(name.into(), replacement.into());
    }

    pub fn remove(&mut self, name: &str) -> Option<Replacement> {
        self.map.remove(name)
    }

    pub fn iter(&self) -> btree_map::Iter<'_, String, Replacement> {
        self.map.iter()
    }

    /// Add a variable, merging with an existing replacement if possible.
    ///
    /// Two replacements merge if they are equal, or if the existing one is a
    /// multiset and the new one is an ordering of the same elements (the ordered
    /// version then wins).
    pub fn try_add_variable(&mut self, name: &str, replacement: Replacement) -> Result<(), Conflict> {
        let Some(existing) = self.map.get(name) else {
            self.map.insert(name.to_owned(), replacement);
            return Ok(());
        };
        match existing {
            Replacement::Sequence(seq) => {
                let same = match &replacement {
                    Replacement::Multiset(m) => &seq.iter().cloned().collect::<Multiset<_>>() == m,
                    other => other == existing,
                };
                if same {
                    Ok(())
                } else {
                    Err(Conflict)
                }
            }
            Replacement::Multiset(m) => {
                if !replacement.is_collection() {
                    return Err(Conflict);
                }
                let compare: Multiset<Expr> = replacement.to_vec().into_iter().collect();
                if &compare != m {
                    return Err(Conflict);
                }
                if let Replacement::Sequence(_) = replacement {
                    self.map.insert(name.to_owned(), replacement);
                }
                Ok(())
            }
            Replacement::Single(_) => {
                if &replacement == existing {
                    Ok(())
                } else {
                    Err(Conflict)
                }
            }
        }
    }

    /// A copy with the variable added (see [`Substitution::try_add_variable`]).
    pub fn union_with_variable(&self, name: &str, replacement: Replacement) -> Result<Substitution, Conflict> {
        let mut new = self.clone();
        new.try_add_variable(name, replacement)?;
        Ok(new)
    }

    /// Merge several substitutions into a new one.
    pub fn union(&self, others: &[&Substitution]) -> Result<Substitution, Conflict> {
        let mut new = self.clone();
        for other in others {
            for (name, replacement) in other.iter() {
                new.try_add_variable(name, replacement.clone())?;
            }
        }
        Ok(new)
    }

    /// Extract the variable bindings from a syntactic `pattern` that is already
    /// known to match `subject` when treated as linear. Returns false if the
    /// bindings cannot be unified. Mutates `self` even on failure.
    pub fn extract_substitution(&mut self, subject: &Expr, pattern: &Expr) -> bool {
        if let Some(name) = pattern.variable_name() {
            return self.try_add_variable(name, Replacement::Single(subject.clone())).is_ok();
        }
        if let ExprKind::Operation { operands, .. } = pattern.kind() {
            debug_assert_eq!(subject.operands().len(), operands.len());
            for (s, p) in subject.operands().iter().zip(operands) {
                if !self.extract_substitution(s, p) {
                    return false;
                }
            }
        }
        true
    }

    /// A copy with variables renamed; names not in `renaming` stay the same.
    pub fn rename(&self, renaming: &HashMap<String, String>) -> Substitution {
        Substitution { map: self.map.iter().map(|(k, v)| (renaming.get(k).unwrap_or(k).clone(), v.clone())).collect() }
    }
}

impl<K: Into<String>, V: Into<Replacement>> FromIterator<(K, V)> for Substitution {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Substitution { map: iter.into_iter().map(|(k, v)| (k.into(), v.into())).collect() }
    }
}

impl<'a> IntoIterator for &'a Substitution {
    type Item = (&'a String, &'a Replacement);
    type IntoIter = btree_map::Iter<'a, String, Replacement>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl fmt::Display for Substitution {
    /// `{x ↦ a, y ↦ (a, b), z ↦ {a, b}}`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (k, v)) in self.map.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k} ↦ {v}")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::{make_dot_variable, Arity, Head};

    fn a() -> Expr {
        Expr::atom("a")
    }
    fn b() -> Expr {
        Expr::atom("b")
    }

    #[test]
    fn display() {
        let s: Substitution = [
            ("x", Replacement::Single(a())),
            ("y", Replacement::Sequence(vec![a(), b()])),
            ("z", Replacement::Multiset([b(), a()].into_iter().collect())),
        ]
        .into_iter()
        .collect();
        assert_eq!(s.to_string(), "{x ↦ a, y ↦ (a, b), z ↦ {a, b}}");
    }

    #[test]
    fn multiset_then_sequence_merges_to_sequence() {
        let mut s = Substitution::new();
        s.insert("x", Replacement::Multiset([a(), b()].into_iter().collect()));
        s.try_add_variable("x", Replacement::Sequence(vec![a(), b()])).unwrap();
        assert_eq!(s.to_string(), "{x ↦ (a, b)}");
        assert!(s.try_add_variable("x", Replacement::Sequence(vec![b(), a()])).is_err());
        assert!(s.try_add_variable("x", Replacement::Multiset([b(), a()].into_iter().collect())).is_ok());
    }

    #[test]
    fn conflicts() {
        let mut s = Substitution::new();
        s.insert("x", a());
        assert!(s.try_add_variable("x", Replacement::Single(b())).is_err());
        assert!(s.try_add_variable("x", Replacement::Sequence(vec![a()])).is_err());
        assert!(s.try_add_variable("x", Replacement::Single(a())).is_ok());
    }

    #[test]
    fn union() {
        let s1: Substitution =
            [("x", Replacement::Multiset([a(), b()].into_iter().collect())), ("z", Replacement::Single(a()))]
                .into_iter()
                .collect();
        let s2: Substitution =
            [("x", Replacement::Sequence(vec![a(), b()])), ("y", Replacement::Sequence(vec![Expr::atom("c")]))]
                .into_iter()
                .collect();
        assert_eq!(s1.union(&[&s2]).unwrap().to_string(), "{x ↦ (a, b), y ↦ (c), z ↦ a}");
    }

    #[test]
    fn extract() {
        let f = Head::new("f", Arity::Variadic).unwrap();
        let (x, y) = (make_dot_variable("x"), make_dot_variable("y"));
        let mut s = Substitution::new();
        assert!(s.extract_substitution(&f.call(vec![a(), b()]), &f.call(vec![x.clone(), y])));
        assert_eq!(s.to_string(), "{x ↦ a, y ↦ b}");
        assert!(!s.extract_substitution(&b(), &x));
        let mut s = Substitution::new();
        assert!(!s.extract_substitution(&f.call(vec![a(), b()]), &f.call(vec![x.clone(), x])));
        assert_eq!(s.to_string(), "{x ↦ a}");
    }

    #[test]
    fn rename() {
        let s: Substitution = [("x", a())].into_iter().collect();
        let r = s.rename(&HashMap::from([("x".into(), "y".into())]));
        assert_eq!(r.get("y"), Some(&Replacement::Single(a())));
    }
}
