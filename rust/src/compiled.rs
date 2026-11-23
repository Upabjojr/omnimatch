//! Support for many-to-one matchers compiled ahead of time by the
//! `omnimatch-compile` crate.
//!
//! The compiler reads a [`ManyToOneMatcher`](crate::ManyToOneMatcher)'s
//! automaton through [`ManyToOneMatcher::export`](crate::ManyToOneMatcher::export)
//! ([`AutomatonInfo`] and friends) and turns every state into a Rust function.
//! The dispatch on the subject (which transitions apply) becomes a `match` on
//! head and atom names, and the sets of still-viable patterns become bitsets.
//! What the transitions *do* (bind a variable and undo it on backtracking, check
//! constraints, split a sequence, descend into operands, combine the operands
//! of a commutative operation) is the same for every matcher. It lives here, as
//! functions the generated code calls with the next state's function as an
//! argument, so that rustc can inline them.
//!
//! This module is public so that generated code can use it. Its items follow
//! the generator, not semver.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::constraints::Constraint;
use crate::expressions::{Expr, ExprKind, Head, Pattern};
use crate::substitution::{Replacement, Substitution};

pub use crate::matching::many_to_one::{CommutativeCore, CommutativePatternSpec, OperandMatches};

// ─── automaton snapshot (input of the compiler) ──────────────────────────────

/// The automaton of a [`ManyToOneMatcher`](crate::ManyToOneMatcher) (or of
/// one commutative operation's operands).
#[derive(Debug, Clone)]
pub struct AutomatonInfo {
    /// State 0 is the root.
    pub states: Vec<StateInfo>,
    pub finals: BTreeSet<usize>,
    pub patterns: Vec<PatternInfo>,
    /// Local constraints (with automaton variable names) and the patterns they belong to.
    pub constraints: Vec<(Arc<dyn Constraint>, BTreeSet<usize>)>,
    /// Automaton variable name -> constraints on it.
    pub constraint_vars: BTreeMap<String, BTreeSet<usize>>,
}

#[derive(Debug, Clone)]
pub struct PatternInfo {
    pub pattern: Pattern,
    /// Original variable name -> automaton variable name.
    pub renaming: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct StateInfo {
    /// Transitions grouped by the key they are looked up under, sorted by key.
    /// Within a key, the order is the order in which they are tried.
    pub transitions: Vec<(KeyInfo, Vec<TransitionInfo>)>,
    /// Set if the state is reached by a commutative operation.
    pub commutative: Option<CommutativeInfo>,
}

/// What a subject must be for a transition to be tried.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyInfo {
    /// This atom (without variable name).
    Atom(Expr),
    /// An operation with this head.
    Operation(Head),
    /// Any operation (wildcard heads).
    AnyOperation,
    /// Anything, including no subject (wildcards, epsilon transitions).
    Any,
    /// The end of an operation's operands.
    End,
    /// A pattern id of the state's commutative matcher.
    PatternId(usize),
}

#[derive(Debug, Clone)]
pub enum LabelInfo {
    /// An atom or a wildcard (without variable name).
    Expr(Expr),
    Operation(Head),
    Epsilon,
    End,
}

#[derive(Debug, Clone)]
pub struct TransitionInfo {
    pub label: LabelInfo,
    pub target: usize,
    pub variable_name: Option<String>,
    pub patterns: BTreeSet<usize>,
    pub check_constraints: BTreeSet<usize>,
    pub subst: Option<Substitution>,
}

#[derive(Debug, Clone)]
pub struct CommutativeInfo {
    pub associative: Option<Head>,
    pub max_optional_count: usize,
    pub anonymous_patterns: BTreeSet<usize>,
    /// Per commutative pattern id: operand pattern ids as `(id, count)`, and
    /// the sequence variables with whether they are wrapped in the associative head.
    pub patterns: Vec<CommutativePatternSpec>,
    /// The automaton matching a single operand.
    pub automaton: Box<AutomatonInfo>,
}

// ─── runtime data ────────────────────────────────────────────────────────────

pub type Flow = ControlFlow<()>;
pub const CONTINUE: Flow = ControlFlow::Continue(());

/// Called with the run and the state reached once all subjects are consumed.
pub type K<'k> = &'k mut dyn FnMut(&mut Run, usize) -> Flow;
/// A generated state function.
pub type StateFn<M> = fn(&M, &mut Run, K<'_>) -> Flow;
/// A generated function taking the `End` transitions of the given state.
pub type EndFn<M> = fn(&M, &mut Run, usize, K<'_>) -> Flow;
/// A generated function taking the transitions of a commutative pattern id.
pub type PatternIdFn<M> = fn(&M, &mut Run, usize, &Expr, K<'_>) -> Flow;
/// A generated function matching one operand of a commutative operation.
pub type OperandFn<M> = fn(&M, Option<&Expr>) -> Vec<(usize, Substitution)>;

/// A transition's static data. Pattern sets are bitsets with one bit per
/// pattern of the automaton.
#[derive(Debug)]
pub struct Transition {
    pub patterns: &'static [u64],
    pub variable: Option<&'static str>,
    /// Constraints to check after binding `variable`.
    pub constraints: &'static [usize],
    /// Index into [`Tables::substs`] of the bindings added by the transition.
    pub subst: Option<usize>,
}

/// Per-pattern data used when a final state is reached.
#[derive(Debug)]
pub struct PatternTail {
    /// Automaton variable name -> original variable name.
    pub inverse_renaming: HashMap<String, String>,
    pub global_constraints: Vec<Arc<dyn Constraint>>,
}

/// The run-time data of one automaton that cannot be `static`.
#[derive(Debug)]
pub struct Tables {
    pub pattern_count: usize,
    pub substs: Vec<Substitution>,
    pub constraints: Vec<Arc<dyn Constraint>>,
    pub constraint_patterns: Vec<&'static [u64]>,
    pub constraint_vars: HashMap<&'static str, &'static [usize]>,
    pub patterns: Vec<PatternTail>,
}

fn words(count: usize) -> usize {
    (count + 63) / 64
}

fn is_disjoint(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x & y == 0)
}

fn is_empty(a: &[u64]) -> bool {
    a.iter().all(|&x| x == 0)
}

/// The mutable state of one search (the interpreter's `MatchIter`).
#[derive(Debug)]
pub struct Run {
    pub subjects: VecDeque<Expr>,
    /// Bitset of the patterns still viable on the current path.
    pub patterns: Vec<u64>,
    pub substitution: Substitution,
    active_constraints: Vec<bool>,
    associative: Vec<Option<Head>>,
}

impl Run {
    pub fn new(tables: &Tables, subject: Option<Expr>, associative: Option<Head>) -> Self {
        let mut patterns = vec![0u64; words(tables.pattern_count)];
        for i in 0..tables.pattern_count {
            patterns[i / 64] |= 1 << (i % 64);
        }
        Run {
            subjects: subject.into_iter().collect(),
            patterns,
            substitution: Substitution::new(),
            active_constraints: vec![true; tables.constraints.len()],
            associative: vec![associative],
        }
    }

    /// The indices of the viable patterns, in increasing order.
    pub fn viable(&self) -> impl Iterator<Item = usize> + '_ {
        self.patterns
            .iter()
            .enumerate()
            .flat_map(|(w, &bits)| (0..64).filter(move |b| bits >> b & 1 == 1).map(move |b| w * 64 + b))
    }

    fn in_associative(&self) -> bool {
        self.associative.last().is_some_and(Option::is_some)
    }
}

/// Which kind of subject is next: the generated dispatch `match`es on this.
pub enum Front<'a> {
    None,
    Operation(&'a Head),
    Atom(&'a str),
    Other,
}

pub fn front(r: &Run) -> Front<'_> {
    match r.subjects.front() {
        None => Front::None,
        Some(subject) => match subject.kind() {
            ExprKind::Operation { head, .. } => Front::Operation(head),
            ExprKind::Atom { name } if subject.variable_name().is_none() => Front::Atom(name),
            _ => Front::Other,
        },
    }
}

// ─── transitions ─────────────────────────────────────────────────────────────

enum ConstraintSource<'s> {
    Indices(&'s [usize]),
    Variable(&'s str),
}

fn check_constraints(tables: &Tables, r: &mut Run, source: ConstraintSource<'_>, restore: &mut Vec<usize>) {
    let candidates: &[usize] = match source {
        ConstraintSource::Indices(indices) => indices,
        ConstraintSource::Variable(var) => match tables.constraint_vars.get(var) {
            Some(indices) => indices,
            None => return,
        },
    };
    for &index in candidates {
        if !r.active_constraints[index] {
            continue;
        }
        let constraint = &tables.constraints[index];
        let patterns = tables.constraint_patterns[index];
        let all_bound = constraint.variables().iter().all(|v| r.substitution.contains(v));
        if all_bound && !is_disjoint(&r.patterns, patterns) {
            r.active_constraints[index] = false;
            restore.push(index);
            if !constraint.check(&r.substitution) {
                for (w, p) in r.patterns.iter_mut().zip(patterns) {
                    *w &= !p;
                }
                if is_empty(&r.patterns) {
                    break;
                }
            }
        }
    }
}

fn restore_bindings(r: &mut Run, old_values: Vec<(String, Option<Replacement>)>) {
    for (name, old) in old_values.into_iter().rev() {
        match old {
            Some(value) => r.substitution.insert(name, value),
            None => {
                r.substitution.remove(&name);
            }
        }
    }
}

/// Take transition `t`, binding its variable to `subject`, and continue with
/// `next`. Every change is undone before returning.
#[inline]
pub fn check_transition<M>(
    m: &M,
    tables: &Tables,
    r: &mut Run,
    t: &Transition,
    subject: Option<Replacement>,
    next: StateFn<M>,
    k: K<'_>,
) -> Flow {
    if is_disjoint(&r.patterns, t.patterns) {
        return CONTINUE;
    }
    // The viable patterns change if `t` excludes some of them or if a failing
    // constraint removes some; save them only then.
    let narrows = r.patterns.iter().zip(t.patterns).any(|(w, p)| w & !p != 0);
    let saved_patterns =
        if narrows || (t.variable.is_some() && !t.constraints.is_empty()) { Some(r.patterns.clone()) } else { None };
    if narrows {
        for (w, p) in r.patterns.iter_mut().zip(t.patterns) {
            *w &= p;
        }
    }
    let mut old_values: Vec<(String, Option<Replacement>)> = Vec::new();
    // The previous value of `t.variable`, if it was bound by this transition.
    let mut old_variable: Option<(&str, Option<Replacement>)> = None;
    let mut restore = Vec::new();

    let flow = 'body: {
        if let Some(index) = t.subst {
            for (name, value) in &tables.substs[index] {
                old_values.push((name.clone(), r.substitution.get(name).cloned()));
                if r.substitution.try_add_variable(name, value.clone()).is_err() {
                    break 'body CONTINUE;
                }
            }
        }
        if let Some(var) = t.variable {
            old_variable = Some((var, r.substitution.get(var).cloned()));
            if let Some(subject) = subject {
                if r.substitution.try_add_variable(var, subject).is_err() {
                    break 'body CONTINUE;
                }
            }
            check_constraints(tables, r, ConstraintSource::Indices(t.constraints), &mut restore);
            if is_empty(&r.patterns) {
                break 'body CONTINUE;
            }
        }
        next(m, r, k)
    };

    for index in restore {
        r.active_constraints[index] = true;
    }
    if let Some(saved) = saved_patterns {
        r.patterns = saved;
    }
    if let Some((var, old)) = old_variable {
        match old {
            Some(value) => r.substitution.insert(var, value),
            None => {
                r.substitution.remove(var);
            }
        }
    }
    restore_bindings(r, old_values);
    flow
}

/// An epsilon transition: binds the next subject without consuming it.
#[inline]
pub fn epsilon<M>(m: &M, tables: &Tables, r: &mut Run, t: &Transition, next: StateFn<M>, k: K<'_>) -> Flow {
    if is_disjoint(&r.patterns, t.patterns) {
        return CONTINUE;
    }
    let subject = r.subjects.front().cloned().map(Replacement::Single);
    check_transition(m, tables, r, t, subject, next, k)
}

/// An atom or `End` transition: consumes the next subject (if any).
#[inline]
pub fn consume<M>(m: &M, tables: &Tables, r: &mut Run, t: &Transition, next: StateFn<M>, k: K<'_>) -> Flow {
    if is_disjoint(&r.patterns, t.patterns) {
        return CONTINUE;
    }
    let subject = r.subjects.pop_front();
    let flow = check_transition(m, tables, r, t, subject.clone().map(Replacement::Single), next, k);
    if let Some(subject) = subject {
        r.subjects.push_front(subject);
    }
    flow
}

/// A wildcard transition (`wildcard` is the label, without variable name).
pub fn wildcard<M>(
    m: &M,
    tables: &Tables,
    r: &mut Run,
    t: &Transition,
    wildcard: &Expr,
    next: StateFn<M>,
    k: K<'_>,
) -> Flow {
    if is_disjoint(&r.patterns, t.patterns) {
        return CONTINUE;
    }
    let w = wildcard.as_wildcard().expect("a wildcard label");
    if let Some(default) = &w.default {
        if w.min_count > 0 {
            check_transition(m, tables, r, t, Some(Replacement::Single(default.clone())), next, k)?;
        }
    }
    if w.fixed_size && !r.in_associative() {
        if r.subjects.is_empty() {
            return CONTINUE;
        }
        let subject = r.subjects.pop_front();
        let flow = check_transition(m, tables, r, t, subject.clone().map(Replacement::Single), next, k);
        if let Some(subject) = subject {
            r.subjects.push_front(subject);
        }
        return flow;
    }

    // A sequence wildcard (or a fixed one inside an associative operation).
    let min_count = w.min_count;
    if r.subjects.len() < min_count {
        return CONTINUE;
    }
    let mut matched: Vec<Expr> = r.subjects.drain(..min_count).collect();
    let associative = r.associative.last().cloned().flatten();
    let mut flow = CONTINUE;
    loop {
        let wrapped = match &associative {
            Some(head) if w.fixed_size => {
                if matched.len() > 1 {
                    head.try_call(matched.clone()).ok().map(Replacement::Single)
                } else {
                    matched.first().cloned().map(Replacement::Single)
                }
            }
            _ => match (&w.default, matched.is_empty()) {
                (Some(default), true) => Some(Replacement::Single(default.clone())),
                _ => Some(Replacement::Sequence(matched.clone())),
            },
        };
        if let Some(wrapped) = wrapped {
            flow = check_transition(m, tables, r, t, Some(wrapped), next, k);
            if flow.is_break() {
                break;
            }
        }
        match r.subjects.pop_front() {
            Some(e) => matched.push(e),
            None => break,
        }
    }
    for e in matched.into_iter().rev() {
        r.subjects.push_front(e);
    }
    flow
}

/// A non-commutative operation: match the operands from `next`, then take the
/// `End` transitions (`end`) of the state reached.
#[allow(clippy::too_many_arguments)]
pub fn regular_operation<M>(
    m: &M,
    tables: &Tables,
    r: &mut Run,
    t: &Transition,
    head: &Head,
    next: StateFn<M>,
    end: EndFn<M>,
    k: K<'_>,
) -> Flow {
    if is_disjoint(&r.patterns, t.patterns) {
        return CONTINUE;
    }
    let subject = r.subjects.pop_front().expect("an operation subject");

    // A wildcard head binds the subject's head (undone on backtracking).
    let mut head_binding: Option<(&str, Option<Replacement>)> = None;
    if let (true, Some(var)) = (head.wildcard, head.head_variable.as_deref()) {
        let old = r.substitution.get(var).cloned();
        let bound = Replacement::Head(subject.head().expect("operation subject").clone());
        if r.substitution.try_add_variable(var, bound).is_err() {
            r.subjects.push_front(subject);
            return CONTINUE;
        }
        head_binding = Some((var, old));
    }
    let saved_patterns = r.patterns.clone();
    let mut restore = Vec::new();
    if let Some((var, _)) = head_binding {
        check_constraints(tables, r, ConstraintSource::Variable(var), &mut restore);
    }
    let flow = if is_empty(&r.patterns) {
        CONTINUE
    } else {
        let operands: VecDeque<Expr> = subject.operands().iter().cloned().collect();
        let mut after: Option<VecDeque<Expr>> = Some(std::mem::replace(&mut r.subjects, operands));
        r.associative.push(head.associative.then(|| head.clone()));
        let flow =
            check_transition(m, tables, r, t, Some(Replacement::Single(subject.clone())), next, &mut |r, state| {
                let inner = std::mem::replace(&mut r.subjects, after.take().expect("outer subjects"));
                let saved_associative = r.associative.pop();
                let flow = end(m, r, state, k);
                after = Some(std::mem::replace(&mut r.subjects, inner));
                r.associative.push(saved_associative.flatten());
                flow
            });
        r.subjects = after.take().expect("outer subjects");
        r.associative.pop();
        flow
    };
    for index in restore {
        r.active_constraints[index] = true;
    }
    r.patterns = saved_patterns;
    if let Some((var, old)) = head_binding {
        match old {
            Some(value) => r.substitution.insert(var, value),
            None => {
                r.substitution.remove(var);
            }
        }
    }
    r.subjects.push_front(subject);
    flow
}

/// A commutative operation: match its operands with `core` and take the
/// transitions of each commutative pattern id that matched (`dispatch`).
/// `pattern_id_patterns[id]` is the union of those transitions' pattern sets.
#[allow(clippy::too_many_arguments)]
pub fn commutative_operation<M>(
    m: &M,
    tables: &Tables,
    r: &mut Run,
    t: &Transition,
    core: &CommutativeCore,
    operand_matches: OperandFn<M>,
    pattern_id_patterns: &[&[u64]],
    dispatch: PatternIdFn<M>,
    k: K<'_>,
) -> Flow {
    if is_disjoint(&r.patterns, t.patterns) {
        return CONTINUE;
    }
    let subject = r.subjects.pop_front().expect("an operation subject");
    let operand = |s: Option<&Expr>| operand_matches(m, s);
    core.add_subject(None, &operand);
    for o in subject.operands() {
        core.add_subject(Some(o), &operand);
    }
    let original = r.substitution.clone();
    let saved_patterns = r.patterns.clone();
    let mut flow = CONTINUE;

    for (matched_pattern, mut new_substitution) in core.match_operands(subject.operands(), &original) {
        if let Some(var) = t.variable {
            if new_substitution.try_add_variable(var, Replacement::Single(subject.clone())).is_err() {
                continue;
            }
        }
        let Some(potential) = pattern_id_patterns.get(matched_pattern) else { continue };
        let diff: Vec<String> =
            new_substitution.iter().map(|(k, _)| k).filter(|k| !original.contains(k)).cloned().collect();
        r.substitution = new_substitution;
        for ((w, s), p) in r.patterns.iter_mut().zip(&saved_patterns).zip(potential.iter()) {
            *w = s & p;
        }
        let mut restore = Vec::new();
        for var in &diff {
            check_constraints(tables, r, ConstraintSource::Variable(var), &mut restore);
            if is_empty(&r.patterns) {
                break;
            }
        }
        if !is_empty(&r.patterns) {
            flow = dispatch(m, r, matched_pattern, &subject, k);
        }
        for index in restore {
            r.active_constraints[index] = true;
        }
        r.patterns.clone_from(&saved_patterns);
        if flow.is_break() {
            break;
        }
    }
    r.substitution = original;
    r.subjects.push_front(subject);
    flow
}

// ─── entry points ────────────────────────────────────────────────────────────

/// Run a (top-level) automaton from `root` and call `on_group` with the
/// matches reaching each final state, with the original variable names and
/// after the global constraints.
pub fn for_each_group<M>(
    m: &M,
    tables: &Tables,
    root: StateFn<M>,
    subject: &Expr,
    on_group: &mut dyn FnMut(Vec<(usize, Substitution)>) -> Flow,
) -> Flow {
    let mut r = Run::new(tables, Some(subject.clone()), None);
    root(m, &mut r, &mut |r, _state| {
        let mut group = Vec::new();
        for index in r.viable() {
            let tail = &tables.patterns[index];
            let subst = r.substitution.rename(&tail.inverse_renaming);
            if tail.global_constraints.iter().all(|c| c.check(&subst)) {
                group.push((index, subst));
            }
        }
        on_group(group)
    })
}

/// Run an operand automaton on one operand: all `(operand pattern, substitution)` matches.
pub fn operand_matches<M>(
    m: &M,
    tables: &Tables,
    root: StateFn<M>,
    subject: Option<&Expr>,
    associative: Option<&Head>,
) -> Vec<(usize, Substitution)> {
    let mut r = Run::new(tables, subject.cloned(), associative.cloned());
    let mut out = Vec::new();
    let _ = root(m, &mut r, &mut |r, _| {
        for index in r.viable() {
            out.push((index, r.substitution.clone()));
        }
        CONTINUE
    });
    out
}
