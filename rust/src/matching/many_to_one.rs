//! Many-to-one matching: match one subject against many patterns at once.
//!
//! Port of `omnimatch/matching/many_to_one.py`.
//!
//! All patterns are compiled into one automaton over the pre-order stream of
//! a pattern's nodes (operation heads, atoms, wildcards, and an "end of
//! operands" marker). Patterns sharing a prefix share states. The operands of
//! commutative operations cannot be matched in order; each state reached by a
//! commutative operation owns a `CommutativeMatcher`, with its own automaton
//! for the operands, which matches the operands individually and then
//! combines them through maximum bipartite matchings.
//!
//! Matching is a backtracking search that mutates one `MatchIter` (remaining
//! subjects, still-viable patterns, the substitution, ...) and undoes each
//! change when it backtracks, like the Python version. Python's generators
//! become continuation-passing: each step calls `k` for every state reached and
//! stops early if `k` returns [`ControlFlow::Break`].
//!
//! Differences from Python:
//! * The graphviz helpers and `SecondaryAutomaton` (unused) are not ported.
//! * Results of a `CommutativeMatcher` are collected before being consumed
//!   rather than produced lazily, so an early stop still pays for all the
//!   matches of the innermost commutative operation.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::constraints::{Constraint, ConstraintRef};
use crate::expressions::{Expr, ExprKind, Head, Pattern, WildcardSpec};
use crate::functions::{replace, ReplacementFn, ReplacementRule, RewriteError};
use crate::matching::bipartite::{enum_maximum_matchings, BipartiteGraph, Matching};
use crate::matching::bitset::BitSet;
use crate::matching::common::check_one_identity;
use crate::multiset::Multiset;
use crate::substitution::{Replacement, Substitution};
use crate::utils::{commutative_sequence_variable_partition_iter, VariableWithCount};

type Flow = ControlFlow<()>;
const CONTINUE: Flow = ControlFlow::Continue(());

// ─── automaton data ──────────────────────────────────────────────────────────

/// What a transition consumes (Python: the `LabelType` hierarchy).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Label {
    /// An atom or a wildcard; wildcards are stored without their variable name.
    Expr(Expr),
    /// An operation with this head.
    Operation(Head),
    /// Nothing (one-identity shortcuts and commutative pattern ids).
    Epsilon,
    /// The end of an operation's operands.
    End,
}

/// The key under which a state stores its transitions (Python: `TransitionKey`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    /// A specific atom (without variable name).
    Expr(Expr),
    /// A specific operation head.
    Operation(Head),
    /// Any operation (patterns with a wildcard head; Python: `HeadTypeAnyOperation`).
    AnyOperation,
    /// Matches any subject: wildcards and epsilon transitions.
    Any,
    End,
    /// A pattern id of the state's commutative matcher.
    PatternId(usize),
}

#[derive(Debug)]
struct Transition {
    label: Label,
    target: usize,
    variable_name: Option<String>,
    /// The patterns that use this transition.
    patterns: BTreeSet<usize>,
    /// `patterns` as a bit set (what the search intersects with).
    bits: BitSet,
    /// Constraints to check after binding `variable_name`.
    check_constraints: BTreeSet<usize>,
    /// Bindings added when taking the transition (defaults for one-identity shortcuts).
    subst: Option<Substitution>,
}

#[derive(Debug, Default)]
struct State {
    transitions: HashMap<Key, Vec<Transition>>,
    matcher: Option<Box<CommutativeMatcher>>,
}

/// The pattern automaton, without labels (shared by [`ManyToOneMatcher`] and
/// the operand automata of commutative matchers).
#[derive(Debug)]
struct Automaton {
    /// Each pattern and the indices of its (renamed) local constraints.
    patterns: Vec<(Pattern, Vec<usize>)>,
    states: Vec<State>,
    /// Per pattern: original variable name -> automaton variable name.
    pattern_vars: Vec<HashMap<String, String>>,
    /// Each local constraint and the patterns it belongs to.
    constraints: Vec<(Arc<dyn Constraint>, BTreeSet<usize>)>,
    /// The pattern sets of `constraints` as bit sets.
    constraint_bits: Vec<BitSet>,
    constraint_vars: HashMap<String, BTreeSet<usize>>,
    finals: HashSet<usize>,
    pattern_to_index: HashMap<Pattern, usize>,
    constraint_to_index: HashMap<ConstraintRef, usize>,
    /// (variable, pattern) -> constraints on that variable belonging to that pattern.
    constraint_pattern_map: HashMap<(String, usize), BTreeSet<usize>>,
}

const ROOT: usize = 0;

impl Automaton {
    fn new() -> Self {
        Automaton {
            patterns: Vec::new(),
            states: vec![State::default()],
            pattern_vars: Vec::new(),
            constraints: Vec::new(),
            constraint_bits: Vec::new(),
            constraint_vars: HashMap::new(),
            finals: HashSet::new(),
            pattern_to_index: HashMap::new(),
            constraint_to_index: HashMap::new(),
            constraint_pattern_map: HashMap::new(),
        }
    }

    fn clear_caches(&self) {
        for state in &self.states {
            if let Some(m) = &state.matcher {
                m.clear();
            }
        }
    }

    fn internal_add(&mut self, pattern: Pattern, renaming: HashMap<String, String>) -> usize {
        let pattern_index = self.patterns.len();
        let is_identity = renaming.iter().all(|(k, v)| k == v);
        let renamed_constraints: Vec<Arc<dyn Constraint>> = if is_identity {
            pattern.local_constraints()
        } else {
            pattern.local_constraints().iter().map(|c| c.with_renamed_vars(&renaming)).collect()
        };
        let constraint_indices = renamed_constraints.iter().map(|c| self.add_constraint(c, pattern_index)).collect();
        let expression =
            if is_identity { pattern.expression.clone() } else { pattern.expression.with_renamed_vars(&renaming) };
        self.pattern_to_index.insert(pattern.clone(), pattern_index);
        self.patterns.push((pattern, constraint_indices));
        self.pattern_vars.push(renaming);
        self.process_pattern_stack(ROOT, vec![VecDeque::from([expression])], &renamed_constraints, pattern_index);
        pattern_index
    }

    fn process_pattern_stack(
        &mut self,
        mut state: usize,
        mut stack: Vec<VecDeque<Expr>>,
        constraints: &[Arc<dyn Constraint>],
        pattern_index: usize,
    ) {
        while let Some(top) = stack.last_mut() {
            let Some(subpattern) = top.pop_front() else {
                stack.pop();
                if !stack.is_empty() {
                    state = self.create_simple_transition(state, Key::End, pattern_index);
                }
                continue;
            };
            let variable_name = subpattern.variable_name().map(str::to_owned);
            let mut commutative = false;
            if let Some((head, operands)) = subpattern.as_operation() {
                commutative = head.commutative;
                if head.one_identity {
                    if let Some((non_optional, added)) = check_one_identity(&subpattern) {
                        let mut branch = stack.clone();
                        branch.last_mut().expect("non-empty").push_front(non_optional);
                        let new_state = self.create_expression_transition(
                            state,
                            None,
                            variable_name.clone(),
                            pattern_index,
                            Some(added),
                        );
                        self.process_pattern_stack(new_state, branch, constraints, pattern_index);
                    }
                }
                if !commutative {
                    stack.push(operands.iter().cloned().collect());
                }
            }
            state = self.create_expression_transition(state, Some(&subpattern), variable_name, pattern_index, None);
            if commutative {
                let matcher = self.states[state].matcher.as_mut().expect("commutative state has a matcher");
                let subpattern_id = matcher.add_pattern(subpattern.operands(), constraints);
                state = self.create_simple_transition(state, Key::PatternId(subpattern_id), pattern_index);
            }
        }
        self.finals.insert(state);
    }

    fn add_constraint(&mut self, constraint: &Arc<dyn Constraint>, pattern: usize) -> usize {
        let key = ConstraintRef(constraint.clone());
        let index = match self.constraint_to_index.get(&key) {
            Some(&index) => {
                self.constraints[index].1.insert(pattern);
                self.constraint_bits[index].insert(pattern);
                index
            }
            None => {
                let index = self.constraints.len();
                self.constraints.push((constraint.clone(), BTreeSet::from([pattern])));
                self.constraint_bits.push(BitSet::singleton(pattern));
                self.constraint_to_index.insert(key, index);
                index
            }
        };
        for var in constraint.variables() {
            self.constraint_vars.entry(var.clone()).or_default().insert(index);
            self.constraint_pattern_map.entry((var, pattern)).or_default().insert(index);
        }
        index
    }

    /// `expression == None` means an epsilon transition.
    fn create_expression_transition(
        &mut self,
        state: usize,
        expression: Option<&Expr>,
        variable_name: Option<String>,
        index: usize,
        subst: Option<Substitution>,
    ) -> usize {
        let (label, key) = label_and_key(expression);
        let applicable = |s: &Self| -> BTreeSet<usize> {
            variable_name
                .as_ref()
                .and_then(|v| s.constraint_pattern_map.get(&(v.clone(), index)))
                .cloned()
                .unwrap_or_default()
        };
        let new_constraints = applicable(self);
        let transitions = self.states[state].transitions.entry(key.clone()).or_default();
        for t in transitions.iter_mut() {
            if t.variable_name == variable_name && t.label == label && t.subst == subst {
                t.patterns.insert(index);
                t.bits.insert(index);
                t.check_constraints.extend(new_constraints);
                return t.target;
            }
        }
        let matcher = match expression.and_then(Expr::as_operation) {
            Some((head, _)) if head.commutative => {
                Some(Box::new(CommutativeMatcher::new(head.associative.then(|| head.clone()))))
            }
            _ => None,
        };
        let target = self.states.len();
        self.states.push(State { transitions: HashMap::new(), matcher });
        self.states[state].transitions.get_mut(&key).expect("just created").push(Transition {
            label,
            target,
            variable_name,
            patterns: BTreeSet::from([index]),
            bits: BitSet::singleton(index),
            check_constraints: new_constraints,
            subst,
        });
        target
    }

    fn create_simple_transition(&mut self, state: usize, key: Key, index: usize) -> usize {
        if let Some(ts) = self.states[state].transitions.get_mut(&key) {
            ts[0].patterns.insert(index);
            ts[0].bits.insert(index);
            return ts[0].target;
        }
        let target = self.states.len();
        self.states.push(State::default());
        let label = if key == Key::End { Label::End } else { Label::Epsilon };
        self.states[state].transitions.insert(
            key,
            vec![Transition {
                label,
                target,
                variable_name: None,
                patterns: BTreeSet::from([index]),
                bits: BitSet::singleton(index),
                check_constraints: BTreeSet::new(),
                subst: None,
            }],
        );
        target
    }

    /// Rename variables by position so that structurally identical patterns
    /// share automaton states.
    fn collect_variable_renaming(expression: &Expr) -> HashMap<String, String> {
        fn rec(expression: &Expr, position: &mut Vec<usize>, variables: &mut Vec<(String, String)>) {
            if let Some(v) = expression.variable_name() {
                if !variables.iter().any(|(k, _)| k == v) {
                    let name = name_for_position(position, variables);
                    variables.push((v.to_owned(), name));
                }
            }
            *position.last_mut().expect("non-empty") += 1;
            if let Some((head, operands)) = expression.as_operation() {
                for operand in operands {
                    if head.commutative {
                        position.push(0);
                        rec(operand, position, variables);
                        position.pop();
                    } else {
                        rec(operand, position, variables);
                    }
                }
            }
        }
        let mut variables = Vec::new();
        rec(expression, &mut vec![0], &mut variables);
        variables.into_iter().collect()
    }
}

fn name_for_position(position: &[usize], variables: &[(String, String)]) -> String {
    let taken = |name: &str| variables.iter().any(|(_, v)| v == name);
    let base = format!("i{}", position.iter().map(ToString::to_string).collect::<Vec<_>>().join("."));
    if !taken(&base) {
        return base;
    }
    let mut counter = 1;
    while taken(&format!("{base}_{counter}")) {
        counter += 1;
    }
    format!("{base}_{counter}")
}

fn label_and_key(expression: Option<&Expr>) -> (Label, Key) {
    let Some(expression) = expression else { return (Label::Epsilon, Key::Any) };
    match expression.kind() {
        ExprKind::Operation { head, .. } if head.wildcard => (Label::Operation(head.clone()), Key::AnyOperation),
        ExprKind::Operation { head, .. } => (Label::Operation(head.clone()), Key::Operation(head.clone())),
        ExprKind::Wildcard(w) => {
            let anonymous = Expr::wildcard(w.min_count, w.fixed_size, None, w.default.clone()).expect("valid wildcard");
            (Label::Expr(anonymous), Key::Any)
        }
        ExprKind::Atom { .. } => {
            let anonymous = expression.with_variable_name(None);
            (Label::Expr(anonymous.clone()), Key::Expr(anonymous))
        }
    }
}

// ─── the public matcher ──────────────────────────────────────────────────────

/// Matches a subject against many patterns at once.
///
/// Each pattern is added with a label of type `T` that is reported with its
/// matches; [`ManyToOneMatcher::add`] uses the pattern itself as label.
///
/// ```
/// use omnimatch::*;
/// let f = Head::new("f", Arity::Variadic).unwrap();
/// let (a, b) = (Expr::atom("a"), Expr::atom("b"));
/// let x_ = make_dot_variable("x");
/// let mut matcher = ManyToOneMatcher::<String>::new();
/// matcher.add_with_label(Pattern::new(f.call(vec![a.clone(), x_.clone()])), "f(a, x_)".into());
/// matcher.add_with_label(Pattern::new(f.call(vec![x_, b.clone()])), "f(x_, b)".into());
/// let mut found: Vec<String> = matcher
///     .match_all(&f.call(vec![a, b]))
///     .iter()
///     .map(|(label, subst)| format!("{label} matched with {subst}"))
///     .collect();
/// found.sort();
/// assert_eq!(found, ["f(a, x_) matched with {x ↦ b}", "f(x_, b) matched with {x ↦ a}"]);
/// ```
pub struct ManyToOneMatcher<T = Pattern> {
    automaton: Automaton,
    labels: Vec<T>,
    rename: bool,
}

impl<T> std::fmt::Debug for ManyToOneMatcher<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManyToOneMatcher")
            .field("patterns", &self.automaton.patterns.len())
            .field("states", &self.automaton.states.len())
            .finish()
    }
}

impl<T> Default for ManyToOneMatcher<T> {
    fn default() -> Self {
        ManyToOneMatcher { automaton: Automaton::new(), labels: Vec::new(), rename: true }
    }
}

impl ManyToOneMatcher<Pattern> {
    /// Add a pattern, labelled with itself.
    pub fn add(&mut self, pattern: Pattern) -> usize {
        self.add_with_label(pattern.clone(), pattern)
    }

    /// A matcher for the given patterns.
    pub fn from_patterns(patterns: impl IntoIterator<Item = Pattern>) -> Self {
        let mut matcher = Self::new();
        for p in patterns {
            matcher.add(p);
        }
        matcher
    }
}

impl<T> ManyToOneMatcher<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Without renaming, structurally identical patterns with different
    /// variable names do not share states (Python: `rename=False`).
    pub fn without_renaming() -> Self {
        ManyToOneMatcher { rename: false, ..Self::default() }
    }

    /// Number of patterns (counting a pattern once per distinct label).
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Number of automaton states (of the top-level automaton).
    pub fn state_count(&self) -> usize {
        self.automaton.states.len()
    }

    /// Add a pattern with a label. Adding the same pattern with an equal label
    /// again is a no-op. Returns the pattern's index.
    pub fn add_with_label(&mut self, pattern: Pattern, label: T) -> usize
    where
        T: PartialEq,
    {
        if let Some(&index) = self.automaton.pattern_to_index.get(&pattern) {
            if self.labels[index] == label {
                return index;
            }
        }
        let renaming =
            if self.rename { Automaton::collect_variable_renaming(&pattern.expression) } else { HashMap::new() };
        let index = self.automaton.internal_add(pattern, renaming);
        self.labels.push(label);
        index
    }

    /// Drop the subjects cached by the commutative matchers.
    pub fn clear(&self) {
        self.automaton.clear_caches();
    }

    /// Call `on_match` with the label and substitution of every match.
    pub fn for_each_match(&self, subject: &Expr, on_match: &mut dyn FnMut(&T, Substitution) -> Flow) -> Flow {
        self.for_each_group(subject, &mut |group| {
            for (index, subst) in group {
                on_match(&self.labels[index], subst)?;
            }
            CONTINUE
        })
    }

    /// Like [`ManyToOneMatcher::for_each_match`], but the matches reaching the
    /// same final state (structurally identical patterns differing only in
    /// constraints or labels) are reported together (Python: `match(...).grouped()`).
    pub fn for_each_group(&self, subject: &Expr, on_group: &mut dyn FnMut(Vec<(usize, Substitution)>) -> Flow) -> Flow {
        let automaton = &self.automaton;
        let mut it = MatchIter::new(automaton, Some(subject.clone()), None);
        it.match_state(ROOT, &mut |it, _state| {
            let mut group = Vec::new();
            for index in it.patterns.iter() {
                let inverse: HashMap<String, String> = automaton.pattern_vars[index]
                    .iter()
                    .map(|(orig, renamed)| (renamed.clone(), orig.clone()))
                    .collect();
                let subst = it.substitution.rename(&inverse);
                let (pattern, _) = &automaton.patterns[index];
                if pattern.global_constraints().iter().all(|c| c.check(&subst)) {
                    group.push((index, subst));
                }
            }
            on_group(group)
        })
    }

    /// All matches, as `(label, substitution)` pairs.
    pub fn match_all(&self, subject: &Expr) -> Vec<(&T, Substitution)> {
        let mut out: Vec<(usize, Substitution)> = Vec::new();
        let _ = self.for_each_group(subject, &mut |group| {
            out.extend(group);
            CONTINUE
        });
        out.into_iter().map(|(i, s)| (&self.labels[i], s)).collect()
    }

    /// The first match, if any.
    pub fn first_match(&self, subject: &Expr) -> Option<(&T, Substitution)> {
        let mut found = None;
        let _ = self.for_each_group(subject, &mut |group| match group.into_iter().next() {
            Some(m) => {
                found = Some(m);
                ControlFlow::Break(())
            }
            None => CONTINUE,
        });
        found.map(|(i, s)| (&self.labels[i], s))
    }

    /// True iff any pattern matches.
    pub fn is_match(&self, subject: &Expr) -> bool {
        self.first_match(subject).is_some()
    }
}

// ─── matching ────────────────────────────────────────────────────────────────

/// The mutable state of one search (Python: `_MatchIter`).
struct MatchIter<'a> {
    automaton: &'a Automaton,
    subjects: VecDeque<Expr>,
    /// Patterns still viable on the current path.
    patterns: BitSet,
    substitution: Substitution,
    /// Constraints not yet checked on the current path.
    active_constraints: Vec<bool>,
    /// Innermost-last stack of enclosing associative heads.
    associative: Vec<Option<Head>>,
}

type K<'k, 'a> = &'k mut dyn FnMut(&mut MatchIter<'a>, usize) -> Flow;

enum ConstraintSource<'s> {
    Indices(&'s BTreeSet<usize>),
    Variable(&'s str),
}

impl<'a> MatchIter<'a> {
    fn new(automaton: &'a Automaton, subject: Option<Expr>, associative: Option<Head>) -> Self {
        MatchIter {
            automaton,
            subjects: subject.into_iter().collect(),
            patterns: BitSet::full(automaton.patterns.len()),
            substitution: Substitution::new(),
            active_constraints: vec![true; automaton.constraints.len()],
            associative: vec![associative],
        }
    }

    /// Run the automaton from `state`; call `k` with every state reached once
    /// all subjects are consumed.
    fn match_state(&mut self, state: usize, k: K<'_, 'a>) -> Flow {
        let automaton = self.automaton;
        let st = &automaton.states[state];
        let keys: Vec<Key> = match self.subjects.front() {
            None => {
                if automaton.finals.contains(&state) || st.transitions.contains_key(&Key::End) {
                    k(self, state)?;
                }
                vec![Key::Any]
            }
            Some(subject) => match subject.head() {
                Some(head) => vec![Key::Operation(head.clone()), Key::AnyOperation, Key::Any],
                None => vec![Key::Expr(subject.clone()), Key::Any],
            },
        };
        for key in keys {
            if let Some(transitions) = st.transitions.get(&key) {
                for t in transitions {
                    self.match_transition(t, k)?;
                }
            }
        }
        CONTINUE
    }

    fn match_transition(&mut self, t: &'a Transition, k: K<'_, 'a>) -> Flow {
        if self.patterns.is_disjoint(&t.bits) {
            return CONTINUE;
        }
        match &t.label {
            Label::Epsilon => {
                let subject = self.subjects.front().cloned().map(Replacement::Single);
                return self.check_transition(t, subject, k);
            }
            Label::Operation(_) => {
                return if self.automaton.states[t.target].matcher.is_some() {
                    self.match_commutative_operation(t, k)
                } else {
                    self.match_regular_operation(t, k)
                };
            }
            Label::Expr(e) => {
                if let ExprKind::Wildcard(w) = e.kind() {
                    if let Some(default) = &w.default {
                        if w.min_count > 0 {
                            self.check_transition(t, Some(Replacement::Single(default.clone())), k)?;
                        }
                    }
                    let in_associative = self.associative.last().is_some_and(Option::is_some);
                    if w.fixed_size && !in_associative {
                        debug_assert_eq!(w.min_count, 1, "fixed wildcards with length != 1 are not supported");
                        if self.subjects.is_empty() {
                            return CONTINUE;
                        }
                    } else {
                        return self.match_sequence_variable(w, t, k);
                    }
                }
            }
            Label::End => {}
        }
        let subject = self.subjects.pop_front();
        let flow = self.check_transition(t, subject.clone().map(Replacement::Single), k);
        if let Some(subject) = subject {
            self.subjects.push_front(subject);
        }
        flow
    }

    /// Take transition `t`, binding its variable to `subject`, and continue
    /// from its target. Every change is undone before returning.
    fn check_transition(&mut self, t: &'a Transition, subject: Option<Replacement>, k: K<'_, 'a>) -> Flow {
        let narrowed = self.patterns.intersection(&t.bits);
        if narrowed.is_empty() {
            return CONTINUE;
        }
        let saved_patterns = std::mem::replace(&mut self.patterns, narrowed);
        let mut old_values: Vec<(String, Option<Replacement>)> = Vec::new();
        let mut restore_constraints = Vec::new();

        let flow = 'body: {
            if let Some(subst) = &t.subst {
                for (name, value) in subst {
                    old_values.push((name.clone(), self.substitution.get(name).cloned()));
                    if self.substitution.try_add_variable(name, value.clone()).is_err() {
                        break 'body CONTINUE;
                    }
                }
            }
            if let Some(var) = &t.variable_name {
                old_values.push((var.clone(), self.substitution.get(var).cloned()));
                if let Some(subject) = subject {
                    if self.substitution.try_add_variable(var, subject).is_err() {
                        break 'body CONTINUE;
                    }
                }
                self.check_constraints(ConstraintSource::Indices(&t.check_constraints), &mut restore_constraints);
                if self.patterns.is_empty() {
                    break 'body CONTINUE;
                }
            }
            self.match_state(t.target, k)
        };

        for index in restore_constraints {
            self.active_constraints[index] = true;
        }
        self.patterns = saved_patterns;
        for (name, old) in old_values.into_iter().rev() {
            match old {
                Some(value) => self.substitution.insert(name, value),
                None => {
                    self.substitution.remove(&name);
                }
            }
        }
        flow
    }

    /// Evaluate the applicable constraints whose variables are all bound,
    /// removing the patterns of violated ones from the viable set.
    fn check_constraints(&mut self, source: ConstraintSource<'_>, restore_constraints: &mut Vec<usize>) {
        let automaton = self.automaton;
        let gathered: BTreeSet<usize>;
        let candidates: &BTreeSet<usize> = match source {
            ConstraintSource::Indices(indices) => indices,
            ConstraintSource::Variable(var) => {
                let Some(by_name) = automaton.constraint_vars.get(var) else { return };
                if self.patterns.len() < by_name.len() {
                    gathered = self
                        .patterns
                        .iter()
                        .filter_map(|p| automaton.constraint_pattern_map.get(&(var.to_owned(), p)))
                        .flatten()
                        .copied()
                        .collect();
                    &gathered
                } else {
                    by_name
                }
            }
        };
        for &index in candidates {
            if !self.active_constraints[index] {
                continue;
            }
            let constraint = &automaton.constraints[index].0;
            let patterns = &automaton.constraint_bits[index];
            if self.patterns.is_disjoint(patterns) {
                continue;
            }
            let all_bound = constraint.variables().iter().all(|v| self.substitution.contains(v));
            if all_bound {
                self.active_constraints[index] = false;
                restore_constraints.push(index);
                if !constraint.check(&self.substitution) {
                    self.patterns.difference_with(patterns);
                    if self.patterns.is_empty() {
                        break;
                    }
                }
            }
        }
    }

    fn match_sequence_variable(&mut self, wildcard: &WildcardSpec, t: &'a Transition, k: K<'_, 'a>) -> Flow {
        let min_count = wildcard.min_count;
        if self.subjects.len() < min_count {
            return CONTINUE;
        }
        let mut matched: Vec<Expr> = self.subjects.drain(..min_count).collect();
        let associative = self.associative.last().cloned().flatten();
        let mut flow = CONTINUE;
        loop {
            let wrapped = match &associative {
                Some(head) if wildcard.fixed_size => {
                    debug_assert_eq!(min_count, 1, "fixed wildcards with length != 1 are not supported");
                    if matched.len() > 1 {
                        match head.try_call(matched.clone()) {
                            Ok(e) => Some(Replacement::Single(e)),
                            Err(_) => None,
                        }
                    } else {
                        matched.first().cloned().map(Replacement::Single)
                    }
                }
                _ => match (&wildcard.default, matched.is_empty()) {
                    (Some(default), true) => Some(Replacement::Single(default.clone())),
                    _ => Some(Replacement::Sequence(matched.clone())),
                },
            };
            if let Some(wrapped) = wrapped {
                flow = self.check_transition(t, Some(wrapped), k);
                if flow.is_break() {
                    break;
                }
            }
            match self.subjects.pop_front() {
                Some(next) => matched.push(next),
                None => break,
            }
        }
        for e in matched.into_iter().rev() {
            self.subjects.push_front(e);
        }
        flow
    }

    fn match_regular_operation(&mut self, t: &'a Transition, k: K<'_, 'a>) -> Flow {
        let Label::Operation(head) = &t.label else { unreachable!("operation transition") };
        let subject = self.subjects.pop_front().expect("an operation subject");

        // A wildcard head binds the subject's head (undone on backtracking).
        let mut head_binding: Option<(&str, Option<Replacement>)> = None;
        if let (true, Some(var)) = (head.wildcard, head.head_variable.as_deref()) {
            let old = self.substitution.get(var).cloned();
            let bound = Replacement::Head(subject.head().expect("operation subject").clone());
            if self.substitution.try_add_variable(var, bound).is_err() {
                self.subjects.push_front(subject);
                return CONTINUE;
            }
            head_binding = Some((var, old));
        }
        let saved_patterns = self.patterns.clone();
        let mut restore_constraints = Vec::new();
        if let Some((var, _)) = head_binding {
            self.check_constraints(ConstraintSource::Variable(var), &mut restore_constraints);
        }
        let flow = if self.patterns.is_empty() { CONTINUE } else { self.match_operation_operands(t, &subject, k) };
        for index in restore_constraints {
            self.active_constraints[index] = true;
        }
        self.patterns = saved_patterns;
        if let Some((var, old)) = head_binding {
            match old {
                Some(value) => self.substitution.insert(var, value),
                None => {
                    self.substitution.remove(var);
                }
            }
        }
        self.subjects.push_front(subject);
        flow
    }

    fn match_operation_operands(&mut self, t: &'a Transition, subject: &Expr, k: K<'_, 'a>) -> Flow {
        let Label::Operation(head) = &t.label else { unreachable!("operation transition") };
        let operands: VecDeque<Expr> = subject.operands().iter().cloned().collect();
        let mut after: Option<VecDeque<Expr>> = Some(std::mem::replace(&mut self.subjects, operands));
        let new_associative = head.associative.then(|| head.clone());
        self.associative.push(new_associative);
        let automaton = self.automaton;

        let flow = self.check_transition(t, Some(Replacement::Single(subject.clone())), &mut |it, new_state| {
            let inner = std::mem::replace(&mut it.subjects, after.take().expect("outer subjects"));
            let saved_associative = it.associative.pop();
            let mut flow = CONTINUE;
            if let Some(ends) = automaton.states[new_state].transitions.get(&Key::End) {
                for end in ends {
                    flow = it.check_transition(end, None, k);
                    if flow.is_break() {
                        break;
                    }
                }
            }
            after = Some(std::mem::replace(&mut it.subjects, inner));
            it.associative.push(saved_associative.flatten());
            flow
        });

        self.subjects = after.take().expect("outer subjects");
        self.associative.pop();
        flow
    }

    fn match_commutative_operation(&mut self, t: &'a Transition, k: K<'_, 'a>) -> Flow {
        let state = t.target;
        let automaton = self.automaton;
        let subject = self.subjects.pop_front().expect("an operation subject");
        let matcher = automaton.states[state].matcher.as_deref().expect("commutative matcher");
        matcher.add_subject(None);
        for operand in subject.operands() {
            matcher.add_subject(Some(operand));
        }
        let original = self.substitution.clone();
        let saved_patterns = self.patterns.clone();
        let transitions = &automaton.states[state].transitions;
        let mut flow = CONTINUE;

        for (matched_pattern, mut new_substitution) in matcher.match_operands(subject.operands(), &original) {
            // Bind a named commutative subexpression (`f_c(..., variable_name='y')`).
            // The Python version never takes this transition's binding step, so
            // it silently drops `y`; its one-to-one matcher does bind it.
            if let Some(var) = &t.variable_name {
                if new_substitution.try_add_variable(var, Replacement::Single(subject.clone())).is_err() {
                    continue;
                }
            }
            let diff: Vec<String> =
                new_substitution.iter().map(|(k, _)| k).filter(|k| !original.contains(k)).cloned().collect();
            self.substitution = new_substitution;
            let transition_set = &transitions[&Key::PatternId(matched_pattern)];
            let mut potential = BitSet::new();
            for t in transition_set {
                potential.union_with(&t.bits);
            }
            self.patterns = saved_patterns.intersection(&potential);
            let mut restore_constraints = Vec::new();
            for var in &diff {
                self.check_constraints(ConstraintSource::Variable(var), &mut restore_constraints);
                if self.patterns.is_empty() {
                    break;
                }
            }
            if !self.patterns.is_empty() {
                for next in transition_set {
                    flow = self.check_transition(next, Some(Replacement::Single(subject.clone())), k);
                    if flow.is_break() {
                        break;
                    }
                }
            }
            for index in restore_constraints {
                self.active_constraints[index] = true;
            }
            self.patterns = saved_patterns.clone();
            if flow.is_break() {
                break;
            }
        }
        self.substitution = original;
        self.subjects.push_front(subject);
        flow
    }
}

// ─── commutative matcher ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatternKey {
    subpatterns: Vec<usize>,
    variables: Vec<(VariableWithCount, bool)>,
}

#[derive(Debug)]
struct PatternValue {
    index: usize,
    /// Operand patterns (indices into the operand automaton).
    pattern_set: Multiset<usize>,
    /// Sequence variables, and whether to wrap their values in the associative head.
    variables: Vec<(VariableWithCount, bool)>,
}

/// Subject id used for the "optional" slot (defaults of optional wildcards).
const OPTIONAL_SUBJECT: i64 = -1_000_000;

/// Subjects seen so far and which operand patterns they match.
#[derive(Debug, Default)]
struct Cache {
    subjects: HashMap<Expr, (i64, BTreeSet<usize>)>,
    subjects_by_id: Vec<Expr>,
    /// Subject id -> operand pattern -> the substitutions of the matches.
    bipartite: BipartiteGraph<i64, usize, Vec<Substitution>>,
    optional_initialized: bool,
    optional_pattern_ids: BTreeSet<usize>,
}

/// Matches the operands of one commutative operation (Python: `CommutativeMatcher`).
#[derive(Debug)]
struct CommutativeMatcher {
    core: CommutativeCore,
    pattern_index: HashMap<PatternKey, usize>,
    /// Matches a single operand against the operand patterns.
    automaton: Automaton,
}

/// The part of a commutative matcher that combines operand matches through
/// bipartite matchings. It does not know how single operands are matched:
/// the interpreted [`ManyToOneMatcher`] runs an operand automaton, a compiled
/// matcher (see [`crate::compiled`]) calls generated code.
#[derive(Debug)]
pub struct CommutativeCore {
    patterns: Vec<PatternValue>,
    associative: Option<Head>,
    max_optional_count: usize,
    anonymous_patterns: BTreeSet<usize>,
    cache: RefCell<Cache>,
}

/// A commutative pattern: its operand pattern ids as `(id, count)` pairs,
/// and its sequence variables with whether their values are wrapped in the
/// associative head.
pub type CommutativePatternSpec = (Vec<(usize, usize)>, Vec<(VariableWithCount, bool)>);

/// Matches of one operand (`None`: the optional slot): `(operand pattern, substitution)` pairs.
pub type OperandMatches<'f> = &'f dyn Fn(Option<&Expr>) -> Vec<(usize, Substitution)>;

type Edge = ((i64, usize), (usize, usize));

impl CommutativeMatcher {
    fn new(associative: Option<Head>) -> Self {
        CommutativeMatcher {
            core: CommutativeCore::new(associative, 0, [], Vec::new()),
            pattern_index: HashMap::new(),
            automaton: Automaton::new(),
        }
    }

    fn clear(&self) {
        self.core.clear();
        self.automaton.clear_caches();
    }

    fn is_sequence_wildcard(&self, expression: &Expr) -> bool {
        expression.as_wildcard().is_some_and(|w| !w.fixed_size || self.core.associative.is_some())
    }

    fn add_pattern(&mut self, operands: &[Expr], constraints: &[Arc<dyn Constraint>]) -> usize {
        let mut pattern_set = Multiset::new();
        // Insertion-ordered map from variable name to (info, wrap).
        let mut pattern_vars: Vec<(Option<String>, (VariableWithCount, bool))> = Vec::new();
        let mut opt_count = 0;
        for operand in operands {
            if operand.as_wildcard().is_some_and(|w| w.default.is_some()) {
                opt_count += 1;
            }
            if !self.is_sequence_wildcard(operand) {
                let actual: Vec<Arc<dyn Constraint>> = constraints
                    .iter()
                    .filter(|c| operand.contains_variables_from_set(&c.variables()))
                    .cloned()
                    .collect();
                let pattern = Pattern::with_constraints(operand.clone(), actual);
                let index = match self.automaton.pattern_to_index.get(&pattern) {
                    Some(&index) => index,
                    None => {
                        let renaming = operand.variable_set().into_iter().map(|n| (n.clone(), n)).collect();
                        let anonymous = operand.is_anonymous();
                        let index = self.automaton.internal_add(pattern, renaming);
                        if anonymous {
                            self.core.anonymous_patterns.insert(index);
                        }
                        index
                    }
                };
                pattern_set.add(index);
                continue;
            }
            let w = operand.as_wildcard().expect("sequence wildcard");
            let name = operand.variable_name().map(str::to_owned);
            let existing = pattern_vars.iter().position(|(n, _)| *n == name);
            let entry = match &name {
                None => {
                    let prev_min = existing.map_or(0, |i| pattern_vars[i].1 .0.minimum);
                    (VariableWithCount::new(None, 1, w.min_count + prev_min, None), false)
                }
                Some(n) => {
                    let (count, wrap) = match existing {
                        Some(i) => (pattern_vars[i].1 .0.count, pattern_vars[i].1 .1),
                        None => (0, w.fixed_size && self.core.associative.is_some()),
                    };
                    (VariableWithCount::new(Some(n), count + 1, w.min_count, w.default.clone()), wrap)
                }
            };
            match existing {
                Some(i) => pattern_vars[i].1 = entry,
                None => pattern_vars.push((name, entry)),
            }
        }
        self.core.max_optional_count = self.core.max_optional_count.max(opt_count);

        let mut sorted_vars: Vec<(VariableWithCount, bool)> = pattern_vars.into_iter().map(|(_, v)| v).collect();
        sorted_vars.sort_by(|(a, wa), (b, wb)| {
            (a.name.as_deref().unwrap_or(""), a.count, a.minimum, *wa).cmp(&(
                b.name.as_deref().unwrap_or(""),
                b.count,
                b.minimum,
                *wb,
            ))
        });
        let key = PatternKey { subpatterns: pattern_set.iter().copied().collect(), variables: sorted_vars.clone() };
        if let Some(&index) = self.pattern_index.get(&key) {
            return index;
        }
        let index = self.core.patterns.len();
        self.pattern_index.insert(key, index);
        self.core.patterns.push(PatternValue { index, pattern_set, variables: sorted_vars });
        index
    }

    /// All `(operand pattern, substitution)` matches of the operand automaton.
    fn operand_matches(&self, subject: Option<&Expr>) -> Vec<(usize, Substitution)> {
        let mut it = MatchIter::new(&self.automaton, subject.cloned(), self.core.associative.clone());
        let mut out = Vec::new();
        let _ = it.match_state(ROOT, &mut |it, _| {
            for index in it.patterns.iter() {
                out.push((index, it.substitution.clone()));
            }
            CONTINUE
        });
        out
    }

    fn add_subject(&self, subject: Option<&Expr>) -> i64 {
        self.core.add_subject(subject, &|s| self.operand_matches(s))
    }

    fn match_operands(&self, subjects: &[Expr], substitution: &Substitution) -> Vec<(usize, Substitution)> {
        self.core.match_operands(subjects, substitution)
    }
}

impl CommutativeCore {
    /// A core for the given commutative patterns: each is a multiset of
    /// operand pattern ids (as `(id, count)` pairs) plus its sequence
    /// variables, and whether their values are wrapped in the associative head.
    pub fn new(
        associative: Option<Head>,
        max_optional_count: usize,
        anonymous_patterns: impl IntoIterator<Item = usize>,
        patterns: Vec<CommutativePatternSpec>,
    ) -> Self {
        let patterns = patterns
            .into_iter()
            .enumerate()
            .map(|(index, (set, variables))| {
                let mut pattern_set = Multiset::new();
                for (id, count) in set {
                    pattern_set.add_n(id, count);
                }
                PatternValue { index, pattern_set, variables }
            })
            .collect();
        CommutativeCore {
            patterns,
            associative,
            max_optional_count,
            anonymous_patterns: anonymous_patterns.into_iter().collect(),
            cache: RefCell::new(Cache::default()),
        }
    }

    /// Forget the operands seen so far.
    pub fn clear(&self) {
        *self.cache.borrow_mut() = Cache::default();
    }

    /// The associative head the operands' sequence variables are wrapped in.
    pub fn associative(&self) -> Option<&Head> {
        self.associative.as_ref()
    }

    /// Register a subject operand (`None`: the optional slot) and match it
    /// against the operand patterns, unless that was done before.
    pub fn add_subject(&self, subject: Option<&Expr>, operand_matches: OperandMatches<'_>) -> i64 {
        match subject {
            None => {
                if !self.cache.borrow().optional_initialized {
                    let matches = operand_matches(None);
                    let mut cache = self.cache.borrow_mut();
                    cache.optional_initialized = true;
                    for (pattern_index, subst) in matches {
                        cache.bipartite.get_or_insert_with(OPTIONAL_SUBJECT, pattern_index, Vec::new).push(subst);
                        cache.optional_pattern_ids.insert(pattern_index);
                    }
                }
                OPTIONAL_SUBJECT
            }
            Some(subject) => {
                if let Some((id, _)) = self.cache.borrow().subjects.get(subject) {
                    return *id;
                }
                let matches = operand_matches(Some(subject));
                let mut cache = self.cache.borrow_mut();
                let id = cache.subjects_by_id.len() as i64;
                cache.subjects_by_id.push(subject.clone());
                let mut pattern_set = BTreeSet::new();
                for (pattern_index, subst) in matches {
                    cache.bipartite.get_or_insert_with(id, pattern_index, Vec::new).push(subst);
                    pattern_set.insert(pattern_index);
                }
                cache.subjects.insert(subject.clone(), (id, pattern_set));
                id
            }
        }
    }

    /// All `(commutative pattern id, substitution)` matches for the operands
    /// `subjects` (which must all have been registered with `add_subject`).
    pub fn match_operands(&self, subjects: &[Expr], substitution: &Substitution) -> Vec<(usize, Substitution)> {
        let cache = self.cache.borrow();
        let mut subject_ids: Multiset<i64> = Multiset::new();
        let mut pattern_ids: Multiset<usize> = Multiset::new();
        if self.max_optional_count > 0 && cache.optional_initialized {
            subject_ids.add(OPTIONAL_SUBJECT);
            for _ in 0..self.max_optional_count {
                for &p in &cache.optional_pattern_ids {
                    pattern_ids.add(p);
                }
            }
        }
        for subject in subjects {
            let (id, ids) = &cache.subjects[subject];
            subject_ids.add(*id);
            for &p in ids {
                pattern_ids.add(p);
            }
        }

        let mut out = Vec::new();
        for pv in &self.patterns {
            if !pv.pattern_set.is_empty() {
                if !pv.pattern_set.is_subset(&pattern_ids) {
                    continue;
                }
                for (bipartite_subst, matched_subjects) in
                    self.match_with_bipartite(&cache, &subject_ids, &pv.pattern_set, substitution)
                {
                    let remaining: Multiset<Expr> = subject_ids
                        .difference(&matched_subjects)
                        .iter()
                        .filter(|&&id| id != OPTIONAL_SUBJECT)
                        .map(|&id| cache.subjects_by_id[id as usize].clone())
                        .collect();
                    if !pv.variables.is_empty() {
                        for result in self.match_sequence_variables(&remaining, &pv.variables, &bipartite_subst) {
                            out.push((pv.index, result));
                        }
                    } else if remaining.is_empty() {
                        out.push((pv.index, bipartite_subst));
                    }
                }
            } else if !pv.variables.is_empty() {
                let all: Multiset<Expr> = subjects.iter().cloned().collect();
                for result in self.match_sequence_variables(&all, &pv.variables, substitution) {
                    out.push((pv.index, result));
                }
            } else if subjects.is_empty() {
                out.push((pv.index, substitution.clone()));
            }
        }
        out
    }

    fn match_with_bipartite(
        &self,
        cache: &Cache,
        subject_ids: &Multiset<i64>,
        pattern_set: &Multiset<usize>,
        substitution: &Substitution,
    ) -> Vec<(Substitution, Multiset<i64>)> {
        let bipartite = Self::build_bipartite(cache, subject_ids, pattern_set);
        let mut out = Vec::new();
        for matching in enum_maximum_matchings(&bipartite) {
            if matching.len() < pattern_set.len() {
                break;
            }
            if !self.is_canonical_matching(&matching) {
                continue;
            }
            let matched_subjects: Multiset<i64> = matching.keys().map(|(s, _)| *s).collect();
            let choices: Vec<&Vec<Substitution>> =
                matching.iter().map(|(l, r)| bipartite.get(l, r).expect("matching edge")).collect();
            // Cartesian product of the substitution choices of each edge.
            fn product(
                choices: &[&Vec<Substitution>],
                acc: &Substitution,
                matched: &Multiset<i64>,
                out: &mut Vec<(Substitution, Multiset<i64>)>,
            ) {
                let Some((first, rest)) = choices.split_first() else {
                    out.push((acc.clone(), matched.clone()));
                    return;
                };
                for s in first.iter() {
                    if let Ok(merged) = acc.union(&[s]) {
                        product(rest, &merged, matched, out);
                    }
                }
            }
            product(&choices, substitution, &matched_subjects, &mut out);
        }
        out
    }

    fn match_sequence_variables(
        &self,
        subjects: &Multiset<Expr>,
        pattern_vars: &[(VariableWithCount, bool)],
        substitution: &Substitution,
    ) -> Vec<Substitution> {
        let only_counts: Vec<VariableWithCount> = pattern_vars.iter().map(|(v, _)| v.clone()).collect();
        let wrapped: Vec<&str> =
            pattern_vars.iter().filter(|(_, wrap)| *wrap).filter_map(|(v, _)| v.name.as_deref()).collect();
        let mut out = Vec::new();
        for mut var_subst in commutative_sequence_variable_partition_iter(subjects, &only_counts) {
            for &var in &wrapped {
                if let Some(Replacement::Multiset(operands)) = var_subst.get(var).cloned() {
                    let value = if operands.len() > 1 {
                        match self
                            .associative
                            .as_ref()
                            .and_then(|h| h.try_call(operands.iter().cloned().collect()).ok())
                        {
                            Some(e) => e,
                            None => continue,
                        }
                    } else {
                        match operands.iter().next() {
                            Some(e) => e.clone(),
                            None => continue,
                        }
                    };
                    var_subst.insert(var, value);
                }
            }
            if let Ok(result) = substitution.union(&[&var_subst]) {
                out.push(result);
            }
        }
        out
    }

    /// The bipartite graph between the (copies of the) subjects and the
    /// (copies of the) operand patterns, with the match substitutions as labels.
    fn build_bipartite(
        cache: &Cache,
        subjects: &Multiset<i64>,
        patterns: &Multiset<usize>,
    ) -> BipartiteGraph<(i64, usize), (usize, usize), Vec<Substitution>> {
        let mut bipartite = BipartiteGraph::new();
        let mut n = 0;
        let mut m = 0;
        let mut p_states: BTreeMap<usize, usize> = BTreeMap::new();
        for (&subject, &s_count) in subjects.items() {
            let Some(edges) = cache.bipartite.neighbors_of_left(&subject) else { continue };
            let mut any_patterns = false;
            for (&pattern, &p_count) in patterns.items() {
                if !edges.contains(&pattern) {
                    continue;
                }
                any_patterns = true;
                let subst = cache.bipartite.get(&subject, &pattern).expect("edge exists");
                let p_start = *p_states.entry(pattern).or_insert_with(|| {
                    let start = m;
                    m += p_count;
                    start
                });
                for i in n..n + s_count {
                    for j in p_start..p_start + p_count {
                        bipartite.insert((subject, i), (pattern, j), subst.clone());
                    }
                }
            }
            if any_patterns {
                n += s_count;
            }
        }
        bipartite
    }

    /// Matchings that only permute copies of the same subject (or of
    /// anonymous patterns) are equivalent; keep one of them.
    fn is_canonical_matching(&self, matching: &Matching<(i64, usize), (usize, usize)>) -> bool {
        let edges: Vec<Edge> = matching.iter().map(|(l, r)| (*l, *r)).collect();
        for &((s1, n1), (p1, m1)) in &edges {
            for &((s2, n2), (p2, m2)) in &edges {
                if self.anonymous_patterns.contains(&p1) && self.anonymous_patterns.contains(&p2) {
                    if n1 < n2 && m1 > m2 {
                        return false;
                    }
                } else if s1 == s2 && n1 < n2 && m1 > m2 {
                    return false;
                }
            }
        }
        true
    }
}

// ─── replacer ────────────────────────────────────────────────────────────────

/// Applies a set of replacement rules using a [`ManyToOneMatcher`].
#[derive(Default)]
pub struct ManyToOneReplacer {
    matcher: ManyToOneMatcher<ReplacementFn>,
}

impl ManyToOneReplacer {
    pub fn new(rules: impl IntoIterator<Item = ReplacementRule>) -> Self {
        let mut replacer = Self::default();
        for rule in rules {
            replacer.add(rule);
        }
        replacer
    }

    /// Add a rule. (Callbacks cannot be compared, so rules are never deduplicated.)
    pub fn add(&mut self, rule: ReplacementRule) {
        let renaming = if self.matcher.rename {
            Automaton::collect_variable_renaming(&rule.pattern.expression)
        } else {
            HashMap::new()
        };
        self.matcher.automaton.internal_add(rule.pattern, renaming);
        self.matcher.labels.push(rule.replacement);
    }

    fn first(&self, expression: &Expr) -> Option<Replacement> {
        self.matcher.first_match(expression).map(|(replacement, subst)| replacement(&subst))
    }

    /// Rewrite at the first (pre-order) position where any rule matches, until
    /// nothing matches or `max_count` rewrites were done.
    pub fn replace(&self, expression: &Expr, max_count: Option<usize>) -> Result<Replacement, RewriteError> {
        let mut expression = expression.clone();
        let mut count = 0;
        'outer: while max_count.map_or(true, |m| count < m) {
            for (subexpr, pos) in expression.preorder_iter() {
                if let Some(result) = self.first(&subexpr) {
                    let replaced = replace(&expression, &pos, result)?;
                    count += 1;
                    match replaced {
                        Replacement::Single(e) => expression = e,
                        other => return Ok(other),
                    }
                    continue 'outer;
                }
            }
            break;
        }
        Ok(Replacement::Single(expression))
    }

    /// Rewrite innermost expressions first, until nothing matches.
    pub fn replace_post_order(&self, expression: &Expr) -> Result<Expr, RewriteError> {
        Ok(self.post_order_rec(expression.clone())?.0)
    }

    fn post_order_rec(&self, mut expression: Expr) -> Result<(Expr, bool), RewriteError> {
        let mut any_replaced = false;
        loop {
            if let Some((head, operands)) = expression.as_operation() {
                let mut changed = false;
                let mut new_operands = Vec::with_capacity(operands.len());
                for operand in operands {
                    let (e, r) = self.post_order_rec(operand.clone())?;
                    changed |= r;
                    new_operands.push(e);
                }
                if changed {
                    expression = Expr::try_operation(head.clone(), new_operands, expression.variable_name())?;
                    any_replaced = true;
                }
            }
            match self.first(&expression) {
                Some(Replacement::Single(e)) => {
                    expression = e;
                    any_replaced = true;
                }
                Some(_) => return Err(RewriteError::NonSingleRoot),
                None => return Ok((expression, any_replaced)),
            }
        }
    }
}

// ─── export (for compiled matchers) ──────────────────────────────────────────

impl<T> ManyToOneMatcher<T> {
    /// A snapshot of the matcher's automaton, for `omnimatch-compile`.
    pub fn export(&self) -> crate::compiled::AutomatonInfo {
        export_automaton(&self.automaton)
    }
}

fn export_automaton(a: &Automaton) -> crate::compiled::AutomatonInfo {
    use crate::compiled::{AutomatonInfo, CommutativeInfo, KeyInfo, LabelInfo, PatternInfo, StateInfo, TransitionInfo};
    let key_info = |key: &Key| match key {
        Key::Expr(e) => KeyInfo::Atom(e.clone()),
        Key::Operation(h) => KeyInfo::Operation(h.clone()),
        Key::AnyOperation => KeyInfo::AnyOperation,
        Key::Any => KeyInfo::Any,
        Key::End => KeyInfo::End,
        Key::PatternId(i) => KeyInfo::PatternId(*i),
    };
    let states = a
        .states
        .iter()
        .map(|state| {
            let mut transitions: Vec<(KeyInfo, Vec<TransitionInfo>)> = state
                .transitions
                .iter()
                .map(|(key, ts)| {
                    let ts = ts
                        .iter()
                        .map(|t| TransitionInfo {
                            label: match &t.label {
                                Label::Expr(e) => LabelInfo::Expr(e.clone()),
                                Label::Operation(h) => LabelInfo::Operation(h.clone()),
                                Label::Epsilon => LabelInfo::Epsilon,
                                Label::End => LabelInfo::End,
                            },
                            target: t.target,
                            variable_name: t.variable_name.clone(),
                            patterns: t.patterns.clone(),
                            check_constraints: t.check_constraints.clone(),
                            subst: t.subst.clone(),
                        })
                        .collect();
                    (key_info(key), ts)
                })
                .collect();
            transitions.sort_by(|x, y| x.0.cmp(&y.0));
            let commutative = state.matcher.as_deref().map(|m| CommutativeInfo {
                associative: m.core.associative.clone(),
                max_optional_count: m.core.max_optional_count,
                anonymous_patterns: m.core.anonymous_patterns.clone(),
                patterns: m
                    .core
                    .patterns
                    .iter()
                    .map(|pv| (pv.pattern_set.items().map(|(&i, &n)| (i, n)).collect(), pv.variables.clone()))
                    .collect(),
                automaton: Box::new(export_automaton(&m.automaton)),
            });
            StateInfo { transitions, commutative }
        })
        .collect();
    AutomatonInfo {
        states,
        finals: a.finals.iter().copied().collect(),
        patterns: a
            .patterns
            .iter()
            .zip(&a.pattern_vars)
            .map(|((pattern, _), renaming)| PatternInfo {
                pattern: pattern.clone(),
                renaming: renaming.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            })
            .collect(),
        constraints: a.constraints.clone(),
        constraint_vars: a.constraint_vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    }
}
