//! Data structures shared by the matching algorithms.
//!
//! Port of `omnimatch/matching/_common.py` (plus `match_head` from
//! `expressions/functions.py`).

use std::collections::BTreeMap;
use std::fmt;

use crate::expressions::{Expr, ExprKind, Head};
use crate::multiset::Multiset;
use crate::substitution::{Replacement, Substitution};

/// Minimum length and default value of a pattern variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarInfo {
    pub min_count: usize,
    pub default: Option<Expr>,
}

/// The operands of a commutative pattern, split by how they are matched.
#[derive(Debug, Clone)]
pub struct CommutativePatternsParts {
    pub head: Head,
    /// Total number of operands.
    pub length: usize,
    /// Operands without wildcards.
    pub constant: Multiset<Expr>,
    /// Syntactic (purely structural) non-constant operands.
    pub syntactic: Multiset<Expr>,
    pub sequence_variables: Multiset<String>,
    pub sequence_variable_infos: BTreeMap<String, VarInfo>,
    pub fixed_variables: Multiset<String>,
    pub fixed_variable_infos: BTreeMap<String, VarInfo>,
    /// Non-syntactic operation operands.
    pub rest: Multiset<Expr>,
    /// Combined minimum length of the named sequence variables.
    pub sequence_variable_min_length: usize,
    /// Combined length of the named fixed-length variables.
    pub fixed_variable_length: usize,
    /// Combined minimum length of the unnamed wildcards.
    pub wildcard_min_length: usize,
    pub optional_count: usize,
    /// `None` if there are no unnamed wildcards, `Some(false)` if any of them
    /// is a sequence wildcard, `Some(true)` otherwise.
    pub wildcard_fixed: Option<bool>,
}

impl CommutativePatternsParts {
    pub fn new(head: &Head, operands: &[Expr]) -> Self {
        let mut parts = CommutativePatternsParts {
            head: head.clone(),
            length: operands.len(),
            constant: Multiset::new(),
            syntactic: Multiset::new(),
            sequence_variables: Multiset::new(),
            sequence_variable_infos: BTreeMap::new(),
            fixed_variables: Multiset::new(),
            fixed_variable_infos: BTreeMap::new(),
            rest: Multiset::new(),
            sequence_variable_min_length: 0,
            fixed_variable_length: 0,
            wildcard_min_length: 0,
            optional_count: 0,
            wildcard_fixed: None,
        };
        for expression in operands {
            if expression.is_constant() {
                parts.constant.add(expression.clone());
            } else if let ExprKind::Wildcard(w) = expression.kind() {
                match expression.variable_name() {
                    Some(name) => {
                        let info = VarInfo { min_count: w.min_count, default: w.default.clone() };
                        if w.fixed_size {
                            parts.fixed_variables.add(name.to_owned());
                            update_var_info(&mut parts.fixed_variable_infos, name, info);
                            if w.default.is_none() {
                                parts.fixed_variable_length += w.min_count;
                            } else {
                                parts.optional_count += 1;
                            }
                        } else {
                            parts.sequence_variables.add(name.to_owned());
                            update_var_info(&mut parts.sequence_variable_infos, name, info);
                            if w.default.is_none() {
                                parts.sequence_variable_min_length += w.min_count;
                            }
                        }
                    }
                    None => {
                        parts.wildcard_min_length += w.min_count;
                        parts.wildcard_fixed = Some(parts.wildcard_fixed.unwrap_or(true) && w.fixed_size);
                    }
                }
            } else if expression.is_syntactic() {
                parts.syntactic.add(expression.clone());
            } else {
                parts.rest.add(expression.clone());
            }
        }
        parts
    }
}

fn update_var_info(infos: &mut BTreeMap<String, VarInfo>, name: &str, info: VarInfo) {
    let existing = infos.entry(name.to_owned()).or_insert_with(|| info.clone());
    debug_assert_eq!(*existing, info, "variable {name} used with inconsistent wildcard kinds");
}

impl fmt::Display for CommutativePatternsParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        parts.extend(self.constant.iter().map(ToString::to_string));
        parts.extend(self.syntactic.iter().map(ToString::to_string));
        parts.extend(self.rest.iter().map(ToString::to_string));
        parts.extend(self.sequence_variables.iter().cloned());
        parts.extend(self.fixed_variables.iter().cloned());
        write!(f, "{}({})", self.head.name, parts.join(", "))
    }
}

/// For a `one_identity` operation pattern: if every operand but one can take
/// its default (optional wildcards) or the empty sequence (star wildcards),
/// return that remaining operand together with the default bindings.
pub fn check_one_identity(pattern: &Expr) -> Option<(Expr, Substitution)> {
    let (head, operands) = pattern.as_operation()?;
    let mut added = Substitution::new();
    let mut non_optional: Option<Expr> = None;
    for operand in operands {
        if let Some(w) = operand.as_wildcard() {
            let value = if let Some(default) = &w.default {
                Some(Replacement::Single(default.clone()))
            } else if w.min_count == 0 {
                Some(if head.commutative {
                    Replacement::Multiset(Multiset::new())
                } else {
                    Replacement::Sequence(Vec::new())
                })
            } else {
                None
            };
            if let Some(value) = value {
                if let Some(name) = operand.variable_name() {
                    added.try_add_variable(name, value).ok()?;
                }
                continue;
            }
        }
        if non_optional.is_some() {
            return None;
        }
        non_optional = Some(operand.clone());
    }
    non_optional.map(|e| (e, added))
}

/// Cheap pre-filter: could `subject` match `pattern`, judging by heads only?
pub fn match_head(subject: &Expr, pattern: &Expr) -> bool {
    match pattern.kind() {
        ExprKind::Wildcard(_) => true,
        ExprKind::Operation { head, .. } => {
            head.one_identity || (head.wildcard && subject.is_operation()) || subject.head() == Some(head)
        }
        ExprKind::Atom { .. } => subject.is_atom(),
    }
}
