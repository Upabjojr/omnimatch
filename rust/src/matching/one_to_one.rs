//! One-to-one matching: one pattern against one subject.
//!
//! Port of `omnimatch/matching/one_to_one.py`.
//!
//! The Python implementation is a tower of lazy generators chained with
//! `generator_chain`. Here every step is written in continuation-passing style:
//! each function receives a callback `k` that it calls once per partial match,
//! and the callback returns [`ControlFlow::Break`] to stop the whole search
//! early (e.g. for [`is_match`]). The search order is the same as in Python.
//!
//! One deliberate simplification: Python removes a local constraint from the
//! active set after it has been checked (and restores it on backtracking).
//! Here every constraint whose variables are all bound is checked at each
//! binding point. Because a constraint's result only depends on the values of
//! its variables, which never change once bound, the set of matches is the
//! same; constraints may just be evaluated more than once.

use std::fmt;
use std::ops::ControlFlow;
use std::slice;
use std::sync::Arc;

use crate::constraints::Constraint;
use crate::expressions::{Expr, ExprKind, Head, Pattern};
use crate::matching::common::{check_one_identity, match_head, CommutativePatternsParts, VarInfo};
use crate::multiset::Multiset;
use crate::substitution::{Replacement, Substitution};
use crate::utils::{
    binary_vectors, commutative_sequence_variable_partition_iter, fixed_integer_vector_iter, weak_composition_iter,
    VariableWithCount,
};

type Flow = ControlFlow<()>;
const CONTINUE: Flow = ControlFlow::Continue(());

/// Errors reported by the matching entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchError {
    /// The subject contains wildcards.
    NonConstantSubject,
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchError::NonConstantSubject => write!(f, "The subject for matching must be constant."),
        }
    }
}

impl std::error::Error for MatchError {}

// ─── public API ──────────────────────────────────────────────────────────────

/// Call `on_match` for every match of `pattern` against `subject`, stopping as
/// soon as it returns [`ControlFlow::Break`].
pub fn for_each_match(
    subject: &Expr,
    pattern: &Pattern,
    on_match: &mut dyn FnMut(Substitution) -> ControlFlow<()>,
) -> Result<ControlFlow<()>, MatchError> {
    if !subject.is_constant() {
        return Err(MatchError::NonConstantSubject);
    }
    let cx = Ctx::new(&pattern.local_constraints());
    let global = pattern.global_constraints();
    Ok(match_(slice::from_ref(subject), &pattern.expression, Substitution::new(), &cx, &mut |subst| {
        if global.iter().all(|c| c.check(&subst)) {
            on_match(subst)
        } else {
            CONTINUE
        }
    }))
}

/// All matches of `pattern` against `subject`.
pub fn match_pattern(subject: &Expr, pattern: &Pattern) -> Result<Vec<Substitution>, MatchError> {
    let mut out = Vec::new();
    let _ = for_each_match(subject, pattern, &mut |s| {
        out.push(s);
        CONTINUE
    })?;
    Ok(out)
}

/// The first match of `pattern` against `subject`, if any.
pub fn first_match(subject: &Expr, pattern: &Pattern) -> Result<Option<Substitution>, MatchError> {
    let mut found = None;
    let _ = for_each_match(subject, pattern, &mut |s| {
        found = Some(s);
        ControlFlow::Break(())
    })?;
    Ok(found)
}

/// True iff `pattern` matches `subject`.
pub fn is_match(subject: &Expr, pattern: &Pattern) -> Result<bool, MatchError> {
    Ok(first_match(subject, pattern)?.is_some())
}

/// All matches of `pattern` against any subexpression of `subject`, together
/// with the position of that subexpression.
pub fn match_anywhere(subject: &Expr, pattern: &Pattern) -> Result<Vec<(Substitution, Vec<usize>)>, MatchError> {
    if !subject.is_constant() {
        return Err(MatchError::NonConstantSubject);
    }
    let mut out = Vec::new();
    for (child, pos) in subject.preorder_iter() {
        if match_head(&child, &pattern.expression) {
            for subst in match_pattern(&child, pattern)? {
                out.push((subst, pos.clone()));
            }
        }
    }
    Ok(out)
}

// ─── internals ───────────────────────────────────────────────────────────────

/// The local constraints, with their variable lists precomputed.
struct Ctx {
    constraints: Vec<(Arc<dyn Constraint>, Vec<String>)>,
}

impl Ctx {
    fn new(constraints: &[Arc<dyn Constraint>]) -> Self {
        Ctx { constraints: constraints.iter().map(|c| (c.clone(), c.variables().into_iter().collect())).collect() }
    }

    /// False iff a constraint whose variables are all bound is violated.
    fn check(&self, subst: &Substitution) -> bool {
        self.constraints.iter().all(|(c, vars)| !vars.iter().all(|v| subst.contains(v)) || c.check(subst))
    }
}

type K<'a> = &'a mut dyn FnMut(Substitution) -> Flow;

fn bind(var: &str, value: Replacement, subst: &Substitution, cx: &Ctx, k: K) -> Flow {
    match subst.union_with_variable(var, value) {
        Ok(s) if cx.check(&s) => k(s),
        _ => CONTINUE,
    }
}

/// Match `pattern` against the sequence `subjects` (a single subject unless
/// `pattern` is a sequence wildcard or an optional wildcard).
fn match_(subjects: &[Expr], pattern: &Expr, subst: Substitution, cx: &Ctx, k: K) -> Flow {
    let var = pattern.variable_name();
    match pattern.kind() {
        ExprKind::Wildcard(w) => {
            // Size checks happened in the caller.
            let value = match (&w.default, subjects.first()) {
                (Some(default), None) => Some(Replacement::Single(default.clone())),
                _ if !w.fixed_size => Some(Replacement::Sequence(subjects.to_vec())),
                (_, first) => first.cloned().map(Replacement::Single),
            };
            match (var, value) {
                (Some(var), Some(value)) => bind(var, value, &subst, cx, k),
                _ => k(subst),
            }
        }
        ExprKind::Atom { name } => {
            if subjects.len() != 1 || subjects[0].atom_name() != Some(name) {
                return CONTINUE;
            }
            match var {
                Some(var) => bind(var, Replacement::Single(subjects[0].clone()), &subst, cx, k),
                None => k(subst),
            }
        }
        ExprKind::Operation { head, .. } => {
            if head.one_identity {
                match_one_identity(subjects, pattern, &subst, cx, k)?;
            }
            let [subject] = subjects else { return CONTINUE };
            let Some(subject_head) = subject.head() else { return CONTINUE };
            if !(head.wildcard || subject_head == head) {
                return CONTINUE;
            }
            // A wildcard head matches any head and binds it (Python only
            // supports wildcard heads in the many-to-one matcher).
            let subst = match head.head_variable.as_deref() {
                Some(head_var) if head.wildcard => {
                    match subst.union_with_variable(head_var, Replacement::Head(subject_head.clone())) {
                        Ok(s) if cx.check(&s) => s,
                        _ => return CONTINUE,
                    }
                }
                _ => subst,
            };
            match var {
                Some(var) => {
                    let value = Replacement::Single(subject.clone());
                    match_operation(subject, pattern, subst, cx, &mut |s| bind(var, value.clone(), &s, cx, k))
                }
                None => match_operation(subject, pattern, subst, cx, k),
            }
        }
    }
}

fn match_one_identity(subjects: &[Expr], pattern: &Expr, subst: &Substitution, cx: &Ctx, k: K) -> Flow {
    let Some((non_optional, added)) = check_one_identity(pattern) else { return CONTINUE };
    match subst.union(&[&added]) {
        Ok(new_subst) => match_(subjects, &non_optional, new_subst, cx, k),
        Err(_) => CONTINUE,
    }
}

fn match_operation(subject: &Expr, pattern: &Expr, subst: Substitution, cx: &Ctx, k: K) -> Flow {
    let (head, pattern_ops) = pattern.as_operation().expect("pattern is an operation");
    let subject_ops = subject.operands();
    if pattern_ops.is_empty() {
        return if subject_ops.is_empty() { k(subst) } else { CONTINUE };
    }
    if head.commutative {
        let parts = CommutativePatternsParts::new(head, pattern_ops);
        match_commutative_operation(subject_ops, &parts, subst, cx, k)
    } else {
        non_commutative_match(subject_ops, head, pattern_ops, subst, cx, k)
    }
}

// ─── non-commutative ─────────────────────────────────────────────────────────

/// Returns `(free subjects to distribute, #sequence variables, #optional wildcards)`,
/// or `None` if the pattern needs more operands than the subject has.
fn count_seq_vars(subject_len: usize, head: &Head, pattern_ops: &[Expr]) -> Option<(usize, usize, usize)> {
    let mut remaining = subject_len as isize;
    let mut sequence_var_count = 0;
    let mut optional_count = 0;
    for operand in pattern_ops {
        match operand.as_wildcard() {
            Some(w) if !w.fixed_size || head.associative => {
                sequence_var_count += 1;
                if w.default.is_none() {
                    remaining -= w.min_count as isize;
                }
            }
            Some(w) if w.default.is_some() => optional_count += 1,
            Some(w) => remaining -= w.min_count as isize,
            None => remaining -= 1,
        }
        if remaining < 0 {
            return None;
        }
    }
    Some((remaining as usize, sequence_var_count, optional_count))
}

/// Distribute the subject operands among the pattern operands.
///
/// `optional_parts[i]` is 1 iff the i-th optional wildcard takes a subject;
/// `sequence_var_partition[i]` is how many extra operands the i-th sequence
/// variable gets. In associative operations, fixed-size wildcards that receive
/// several operands get them wrapped in a new operation.
fn build_full_partition(
    optional_parts: &[usize],
    sequence_var_partition: &[usize],
    subjects: &[Expr],
    head: &Head,
    pattern_ops: &[Expr],
) -> Option<Vec<Vec<Expr>>> {
    let mut i = 0;
    let mut var_index = 0;
    let mut opt_index = 0;
    let mut result = Vec::with_capacity(pattern_ops.len());
    for operand in pattern_ops {
        let mut wrap_associative = 0;
        let count = match operand.as_wildcard() {
            Some(w) => {
                let mut count = if w.default.is_none() { w.min_count } else { 0 };
                if !w.fixed_size || head.associative {
                    count += sequence_var_partition[var_index];
                    var_index += 1;
                    if w.fixed_size {
                        wrap_associative = w.min_count;
                    }
                } else if w.default.is_some() {
                    count = optional_parts[opt_index];
                    opt_index += 1;
                }
                count
            }
            None => 1,
        };
        let start = i.min(subjects.len());
        let end = (i + count).min(subjects.len());
        let mut exprs = subjects[start..end].to_vec();
        i += count;
        if wrap_associative > 0 && exprs.len() > wrap_associative {
            let tail = exprs.split_off(wrap_associative - 1);
            exprs.push(head.try_call(tail).ok()?);
        }
        result.push(exprs);
    }
    Some(result)
}

fn non_commutative_match(
    subject_ops: &[Expr],
    head: &Head,
    pattern_ops: &[Expr],
    subst: Substitution,
    cx: &Ctx,
    k: K,
) -> Flow {
    let Some((remaining, sequence_var_count, optional_count)) = count_seq_vars(subject_ops.len(), head, pattern_ops)
    else {
        return CONTINUE;
    };
    for optional in binary_vectors(optional_count) {
        let used: usize = optional.iter().sum();
        if used > remaining {
            continue;
        }
        for part in weak_composition_iter(remaining - used, sequence_var_count) {
            let Some(partition) = build_full_partition(&optional, &part, subject_ops, head, pattern_ops) else {
                continue;
            };
            match_operands(&partition, pattern_ops, subst.clone(), cx, k)?;
        }
    }
    CONTINUE
}

/// Match `partition[i]` against `pattern_ops[i]` for every `i`, left to right.
fn match_operands(partition: &[Vec<Expr>], pattern_ops: &[Expr], subst: Substitution, cx: &Ctx, k: K) -> Flow {
    let Some((first, rest)) = partition.split_first() else { return k(subst) };
    match_(first, &pattern_ops[0], subst, cx, &mut |s| match_operands(rest, &pattern_ops[1..], s, cx, k))
}

// ─── commutative ─────────────────────────────────────────────────────────────

/// One step of the commutative search: each consumes part of the remaining
/// subject multiset.
enum Step<'a> {
    /// Match a non-variable operand against one of the subjects.
    Expr(&'a Expr),
    /// A fixed-length variable occurring `count` times (`None` = unnamed wildcards).
    FixedVar { name: Option<&'a str>, count: usize, length: usize, default: Option<&'a Expr> },
}

type KRest<'a> = &'a mut dyn FnMut(Multiset<Expr>, Substitution) -> Flow;

fn match_commutative_operation(
    subject_ops: &[Expr],
    parts: &CommutativePatternsParts,
    subst: Substitution,
    cx: &Ctx,
    k: K,
) -> Flow {
    let mut subjects: Multiset<Expr> = subject_ops.iter().cloned().collect();
    if !parts.constant.is_subset(&subjects) {
        return CONTINUE;
    }
    subjects = subjects.difference(&parts.constant);
    let rest_expr = parts.rest.combine(&parts.syntactic);
    let needed_length =
        parts.sequence_variable_min_length + parts.fixed_variable_length + rest_expr.len() + parts.wildcard_min_length;
    if subjects.len() < needed_length {
        return CONTINUE;
    }

    let associative = parts.head.associative;

    // Fixed variables that are already bound consume their values up front.
    let mut fixed_vars = parts.fixed_variables.clone();
    for (name, &count) in parts.fixed_variables.items() {
        let Some(replacement) = subst.get(name) else { continue };
        let needed: Multiset<Expr> = match replacement {
            Replacement::Single(e) if associative && e.head() == Some(&parts.head) => {
                e.operands().iter().cloned().collect()
            }
            Replacement::Single(e) => std::iter::once(e.clone()).collect(),
            _ => return CONTINUE,
        };
        let needed = needed.times(count);
        if !needed.is_subset(&subjects) {
            return CONTINUE;
        }
        subjects = subjects.difference(&needed);
        fixed_vars.remove_all(name);
    }

    let mut steps: Vec<Step> = rest_expr.iter().map(Step::Expr).collect();
    if !associative {
        for (name, &count) in fixed_vars.items() {
            let info = &parts.fixed_variable_infos[name];
            steps.push(Step::FixedVar {
                name: Some(name),
                count,
                length: info.min_count,
                default: info.default.as_ref(),
            });
        }
        if parts.wildcard_fixed == Some(true) {
            steps.push(Step::FixedVar { name: None, count: 1, length: parts.wildcard_min_length, default: None });
        }
    }

    // Sequence variables (plus, for associative operations, fixed variables)
    // take whatever is left.
    let mut sequence_vars = variables_with_counts(&parts.sequence_variables, &parts.sequence_variable_infos);
    let unnamed = VariableWithCount::new(None, 1, parts.wildcard_min_length, None);
    if associative {
        sequence_vars.extend(variables_with_counts(&fixed_vars, &parts.fixed_variable_infos));
        if parts.wildcard_fixed == Some(true) {
            sequence_vars.push(unnamed.clone());
        }
    }
    if parts.wildcard_fixed == Some(false) {
        sequence_vars.push(unnamed);
    }

    run_steps(&steps, subjects, subst, cx, &mut |remaining, substitution| {
        for mut sequence_subst in commutative_sequence_variable_partition_iter(&remaining, &sequence_vars) {
            if associative {
                for v in fixed_vars.distinct_elements() {
                    let Some(Replacement::Multiset(value)) = sequence_subst.get(v).cloned() else { continue };
                    let l = parts.fixed_variable_infos[v].min_count;
                    if value.len() > l {
                        let mut normal: Multiset<Expr> = value.iter().take(l.saturating_sub(1)).cloned().collect();
                        let Ok(wrapped) = parts.head.try_call(value.difference(&normal).iter().cloned().collect())
                        else {
                            continue;
                        };
                        if l > 1 {
                            normal.add(wrapped);
                            sequence_subst.insert(v.clone(), Replacement::Multiset(normal));
                        } else {
                            sequence_subst.insert(v.clone(), Replacement::Single(wrapped));
                        }
                    } else if let Some(only) = value.iter().next() {
                        debug_assert!(value.len() == 1 && l == 1, "fixed variables with length != 1 are not supported");
                        sequence_subst.insert(v.clone(), Replacement::Single(only.clone()));
                    }
                }
            }
            if let Ok(result) = substitution.union(&[&sequence_subst]) {
                if cx.check(&result) {
                    k(result)?;
                }
            }
        }
        CONTINUE
    })
}

fn variables_with_counts(
    variables: &Multiset<String>,
    infos: &std::collections::BTreeMap<String, VarInfo>,
) -> Vec<VariableWithCount> {
    variables
        .items()
        .map(|(name, &count)| {
            let info = &infos[name];
            VariableWithCount::new(Some(name), count, info.min_count, info.default.clone())
        })
        .collect()
}

fn run_steps(steps: &[Step], subjects: Multiset<Expr>, subst: Substitution, cx: &Ctx, k: KRest) -> Flow {
    let Some((step, rest)) = steps.split_first() else { return k(subjects, subst) };
    let mut next = |remaining: Multiset<Expr>, s: Substitution| run_steps(rest, remaining, s, cx, k);
    match *step {
        Step::Expr(expression) => {
            let candidates: Vec<Expr> = subjects.distinct_elements().cloned().collect();
            for expr in candidates {
                if match_head(&expr, expression) {
                    let remaining = subjects.without_one(&expr);
                    match_(slice::from_ref(&expr), expression, subst.clone(), cx, &mut |s| next(remaining.clone(), s))?;
                }
            }
            CONTINUE
        }
        Step::FixedVar { name, count, length, default } => {
            if let Some(existing) = name.and_then(|n| subst.get(n)) {
                let value = existing.to_vec();
                if let Some(default) = default {
                    if value.len() == 1 && &value[0] == default {
                        next(subjects.clone(), subst.clone())?;
                    }
                }
                let existing: Multiset<Expr> = value.into_iter().collect::<Multiset<_>>().times(count);
                if !existing.is_subset(&subjects) {
                    return CONTINUE;
                }
                return next(subjects.difference(&existing), subst);
            }
            if let (Some(default), Some(name)) = (default, name) {
                let mut with_default = subst.clone();
                with_default.insert(name, default.clone());
                next(subjects.clone(), with_default)?;
            }
            if length == 1 {
                let candidates: Vec<(Expr, usize)> = subjects.items().map(|(e, &n)| (e.clone(), n)).collect();
                for (expr, expr_count) in candidates {
                    if expr_count < count {
                        continue;
                    }
                    let mut taken = Multiset::new();
                    taken.add_n(expr.clone(), count);
                    let remaining = subjects.difference(&taken);
                    match name {
                        Some(name) => {
                            let mut new_subst = subst.clone();
                            new_subst.insert(name, expr);
                            if cx.check(&new_subst) {
                                next(remaining, new_subst)?;
                            }
                        }
                        None => next(remaining, subst.clone())?,
                    }
                }
            } else {
                debug_assert!(name.is_none(), "fixed variables with length != 1 are not supported");
                let items: Vec<(Expr, usize)> = subjects.items().map(|(e, &n)| (e.clone(), n)).collect();
                let max: Vec<usize> = items.iter().map(|(_, n)| n / count).collect();
                for subset in fixed_integer_vector_iter(&max, length) {
                    let mut taken = Multiset::new();
                    for ((expr, _), c) in items.iter().zip(subset) {
                        taken.add_n(expr.clone(), c * count);
                    }
                    next(subjects.difference(&taken), subst.clone())?;
                }
            }
            CONTINUE
        }
    }
}
