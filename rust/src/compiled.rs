//! Support for many-to-one matchers compiled ahead of time by the
//! `omnimatch-compile` crate.
//!
//! The compiler reads a [`ManyToOneMatcher`](crate::ManyToOneMatcher)'s
//! automaton through [`ManyToOneMatcher::export`](crate::ManyToOneMatcher::export)
//! ([`AutomatonInfo`] and friends) and turns every state into a Rust function.
//! The dispatch on the subject (which transitions apply) becomes a `match` on
//! head and atom names. Everything else (pattern sets, bindings, expressions,
//! constraints) becomes static tables ([`Data`]), from which the values that
//! must be objects are built when first used. What the transitions *do* (bind a variable and undo it on backtracking, check
//! constraints, split a sequence, descend into operands, combine the operands
//! of a commutative operation) is the same for every matcher. It lives here, as
//! functions the generated code calls with the next state's function as an
//! argument, so that rustc can inline them.
//!
//! This module is public so that generated code can use it. Its items follow
//! the generator, not semver.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::ops::ControlFlow;
use std::sync::{Arc, OnceLock};

use crate::constraints::Constraint;
use crate::expressions::{Arity, Expr, ExprKind, Head, Pattern};
use crate::substitution::{Replacement, Substitution};
use crate::utils::VariableWithCount;

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

// ─── static data (output of the compiler) ────────────────────────────────────
//
// Everything a compiled matcher knows about its patterns is in `static` tables
// of plain integers that refer to each other by index. They live in the
// binary's read-only data: the OS maps those pages from the file when they are
// first touched and can drop them again under memory pressure, and no
// relocations are applied to them when the program loads. The few values that
// must be real objects (expressions, heads, bindings, constraints, commutative
// matchers) are built from the tables the first time they are used.

/// "No index".
pub const NONE: u32 = u32::MAX;

/// A range `off..off + len` of one of the [`Data`] tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub off: u32,
    pub len: u32,
}

impl Span {
    pub const EMPTY: Span = Span { off: 0, len: 0 };

    pub const fn new(off: u32, len: u32) -> Span {
        Span { off, len }
    }

    fn range(self) -> std::ops::Range<usize> {
        self.off as usize..(self.off + self.len) as usize
    }
}

/// The code of an arity in [`HeadDesc::arity`].
pub fn arity_code(arity: Arity) -> u8 {
    match arity {
        Arity::Nullary => 0,
        Arity::Unary => 1,
        Arity::Binary => 2,
        Arity::Ternary => 3,
        Arity::Variadic => 4,
    }
}

pub const HEAD_COMMUTATIVE: u8 = 1;
pub const HEAD_ASSOCIATIVE: u8 = 2;
pub const HEAD_ONE_IDENTITY: u8 = 4;
pub const HEAD_INFIX: u8 = 8;
pub const HEAD_WILDCARD: u8 = 16;

/// An operation head.
#[derive(Debug)]
pub struct HeadDesc {
    /// String index.
    pub name: u32,
    /// 0 to 4: nullary, unary, binary, ternary, variadic.
    pub arity: u8,
    /// `HEAD_*` bits.
    pub flags: u8,
    /// String index of the head variable, or [`NONE`].
    pub variable: u32,
}

pub const EXPR_ATOM: u8 = 0;
pub const EXPR_WILDCARD: u8 = 1;
pub const EXPR_OPERATION: u8 = 2;

/// An expression node.
#[derive(Debug)]
pub struct ExprDesc {
    /// `EXPR_*`.
    pub kind: u8,
    /// Wildcards: `fixed_size`.
    pub fixed: bool,
    /// String index of the variable name, or [`NONE`].
    pub variable: u32,
    /// Atoms: string index of the name. Wildcards: `min_count`. Operations: head index.
    pub a: u32,
    /// Wildcards: expression index of the default, or [`NONE`].
    pub b: u32,
    /// Operations: the operands (expression indices in [`Data::u32s`]).
    pub operands: Span,
}

pub const REPLACEMENT_SINGLE: u8 = 0;
pub const REPLACEMENT_SEQUENCE: u8 = 1;
pub const REPLACEMENT_MULTISET: u8 = 2;
pub const REPLACEMENT_HEAD: u8 = 3;

/// A variable's value.
#[derive(Debug)]
pub struct ReplacementDesc {
    /// `REPLACEMENT_*`.
    pub kind: u8,
    /// Single: expression index. Head: head index.
    pub a: u32,
    /// Sequence and multiset: expression indices in [`Data::u32s`].
    pub items: Span,
}

/// A transition's data. Pattern sets are sorted lists of pattern indices.
#[derive(Debug)]
pub struct Transition {
    /// Patterns (in [`Data::u32s`]).
    pub patterns: Span,
    /// String index, or [`NONE`].
    pub variable: u32,
    /// Local constraint indices to check after binding `variable` (in [`Data::u32s`]).
    pub constraints: Span,
    /// Index of the bindings added by the transition, or [`NONE`].
    pub subst: u32,
}

/// One automaton: the top-level one, or the one matching the operands of a
/// commutative operation.
#[derive(Debug)]
pub struct AutomatonDesc {
    pub pattern_count: u32,
    /// Constraint index of each local constraint (in [`Data::u32s`]).
    pub constraints: Span,
    /// Patterns of each local constraint (in [`Data::spans`]).
    pub constraint_patterns: Span,
    /// `(variable, offset, length)` triples in [`Data::u32s`], sorted by
    /// variable name: the local constraints on each variable.
    pub constraint_vars: Span,
    /// One per pattern (in [`Data::tails`]).
    pub tails: Span,
}

/// Per-pattern data used when a final state is reached.
#[derive(Debug)]
pub struct TailDesc {
    /// `(automaton name, original name)` pairs of strings in [`Data::u32s`],
    /// sorted by automaton name.
    pub inverse_renaming: Span,
    /// Constraint indices (in [`Data::u32s`]).
    pub global_constraints: Span,
}

/// A commutative matcher.
#[derive(Debug)]
pub struct CoreDesc {
    /// Head index, or [`NONE`].
    pub associative: u32,
    pub max_optional_count: u32,
    /// In [`Data::u32s`].
    pub anonymous_patterns: Span,
    /// In [`Data::core_patterns`].
    pub patterns: Span,
}

/// A commutative pattern.
#[derive(Debug)]
pub struct CorePatternDesc {
    /// `(operand pattern id, count)` pairs in [`Data::u32s`].
    pub operands: Span,
    /// In [`Data::variables`].
    pub variables: Span,
}

/// A sequence variable of a commutative pattern.
#[derive(Debug)]
pub struct VariableDesc {
    /// String index, or [`NONE`].
    pub name: u32,
    pub count: u32,
    pub minimum: u32,
    /// Expression index, or [`NONE`].
    pub default: u32,
    /// Whether the value is wrapped in the associative head.
    pub wrap: bool,
}

/// All static data of a compiled matcher, and the cells holding the values
/// built from it.
pub struct Data {
    pub text: &'static str,
    /// Byte ranges of `text`.
    pub strings: &'static [Span],
    pub u32s: &'static [u32],
    /// Ranges of `u32s`.
    pub spans: &'static [Span],
    pub heads: &'static [HeadDesc],
    pub head_cells: &'static [OnceLock<Head>],
    pub exprs: &'static [ExprDesc],
    pub expr_cells: &'static [OnceLock<Expr>],
    pub replacements: &'static [ReplacementDesc],
    /// `(name, replacement)` pairs in `u32s`.
    pub substs: &'static [Span],
    pub subst_cells: &'static [OnceLock<Substitution>],
    pub make_constraint: fn(u32) -> Arc<dyn Constraint>,
    pub constraint_cells: &'static [OnceLock<Arc<dyn Constraint>>],
    pub automata: &'static [AutomatonDesc],
    pub tails: &'static [TailDesc],
    pub cores: &'static [CoreDesc],
    pub core_patterns: &'static [CorePatternDesc],
    pub variables: &'static [VariableDesc],
}

impl fmt::Debug for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Data")
            .field("automata", &self.automata.len())
            .field("exprs", &self.exprs.len())
            .field("constraints", &self.constraint_cells.len())
            .finish()
    }
}

fn opt(i: u32) -> Option<u32> {
    (i != NONE).then_some(i)
}

impl Data {
    pub fn str(&'static self, i: u32) -> &'static str {
        &self.text[self.strings[i as usize].range()]
    }

    fn opt_str(&'static self, i: u32) -> Option<&'static str> {
        opt(i).map(|i| self.str(i))
    }

    pub fn list(&'static self, s: Span) -> &'static [u32] {
        &self.u32s[s.range()]
    }

    pub fn head(&'static self, i: u32) -> &'static Head {
        self.head_cells[i as usize].get_or_init(|| {
            let h = &self.heads[i as usize];
            let arity = match h.arity {
                0 => Arity::Nullary,
                1 => Arity::Unary,
                2 => Arity::Binary,
                3 => Arity::Ternary,
                _ => Arity::Variadic,
            };
            let flag = |bit: u8| h.flags & bit != 0;
            let mut b = Head::builder(self.str(h.name))
                .arity(arity)
                .commutative(flag(HEAD_COMMUTATIVE))
                .associative(flag(HEAD_ASSOCIATIVE))
                .one_identity(flag(HEAD_ONE_IDENTITY))
                .infix(flag(HEAD_INFIX));
            if flag(HEAD_WILDCARD) {
                b = b.wildcard(self.opt_str(h.variable));
            }
            b.build_unchecked()
        })
    }

    /// True iff `head` equals head `i` (compared without building it).
    pub fn head_is(&'static self, head: &Head, i: u32) -> bool {
        let h = &self.heads[i as usize];
        let flag = |bit: u8| h.flags & bit != 0;
        arity_code(head.arity) == h.arity
            && head.commutative == flag(HEAD_COMMUTATIVE)
            && head.associative == flag(HEAD_ASSOCIATIVE)
            && head.one_identity == flag(HEAD_ONE_IDENTITY)
            && head.wildcard == flag(HEAD_WILDCARD)
            && head.head_variable.as_deref() == self.opt_str(h.variable)
            && head.name == self.str(h.name)
    }

    pub fn expr(&'static self, i: u32) -> &'static Expr {
        self.expr_cells[i as usize].get_or_init(|| {
            let e = &self.exprs[i as usize];
            let variable = self.opt_str(e.variable);
            match e.kind {
                EXPR_ATOM => {
                    let atom = Expr::atom(self.str(e.a));
                    if variable.is_some() {
                        atom.with_variable_name(variable)
                    } else {
                        atom
                    }
                }
                EXPR_WILDCARD => {
                    Expr::wildcard(e.a as usize, e.fixed, variable, opt(e.b).map(|d| self.expr(d).clone()))
                        .expect("a valid wildcard")
                }
                _ => {
                    let operands = self.list(e.operands).iter().map(|&o| self.expr(o).clone()).collect();
                    Expr::try_operation(self.head(e.a).clone(), operands, variable).expect("a valid operation")
                }
            }
        })
    }

    fn replacement(&'static self, i: u32) -> Replacement {
        let r = &self.replacements[i as usize];
        let exprs = || self.list(r.items).iter().map(|&e| self.expr(e).clone());
        match r.kind {
            REPLACEMENT_SINGLE => Replacement::Single(self.expr(r.a).clone()),
            REPLACEMENT_SEQUENCE => Replacement::Sequence(exprs().collect()),
            REPLACEMENT_MULTISET => Replacement::Multiset(exprs().collect()),
            _ => Replacement::Head(self.head(r.a).clone()),
        }
    }

    pub fn subst(&'static self, i: u32) -> &'static Substitution {
        self.subst_cells[i as usize].get_or_init(|| {
            let pairs = self.list(self.substs[i as usize]);
            let mut s = Substitution::new();
            for pair in pairs.chunks(2) {
                s.insert(self.str(pair[0]), self.replacement(pair[1]));
            }
            s
        })
    }

    pub fn constraint(&'static self, i: u32) -> &'static Arc<dyn Constraint> {
        self.constraint_cells[i as usize].get_or_init(|| (self.make_constraint)(i))
    }

    /// Build commutative matcher `i`.
    pub fn core(&'static self, i: usize) -> CommutativeCore {
        let c = &self.cores[i];
        let patterns = self.core_patterns[c.patterns.range()]
            .iter()
            .map(|p| {
                let operands =
                    self.list(p.operands).chunks(2).map(|pair| (pair[0] as usize, pair[1] as usize)).collect();
                let variables = self.variables[p.variables.range()]
                    .iter()
                    .map(|v| {
                        let default = opt(v.default).map(|d| self.expr(d).clone());
                        let var =
                            VariableWithCount::new(self.opt_str(v.name), v.count as usize, v.minimum as usize, default);
                        (var, v.wrap)
                    })
                    .collect();
                (operands, variables)
            })
            .collect();
        CommutativeCore::new(
            opt(c.associative).map(|h| self.head(h).clone()),
            c.max_optional_count as usize,
            self.list(c.anonymous_patterns).iter().map(|&p| p as usize),
            patterns,
        )
    }
}

/// One automaton of a compiled matcher.
#[derive(Debug, Clone, Copy)]
pub struct Tables {
    pub data: &'static Data,
    pub automaton: &'static AutomatonDesc,
}

impl Tables {
    pub fn new(data: &'static Data, automaton: usize) -> Tables {
        Tables { data, automaton: &data.automata[automaton] }
    }

    /// The global index of local constraint `local`.
    fn constraint(self, local: usize) -> &'static Arc<dyn Constraint> {
        self.data.constraint(self.data.list(self.automaton.constraints)[local])
    }

    fn constraint_patterns(self, local: usize) -> &'static [u32] {
        self.data.list(self.data.spans[self.automaton.constraint_patterns.off as usize + local])
    }

    /// The local constraints on automaton variable `var`.
    fn constraints_on(self, var: &str) -> &'static [u32] {
        let triples = self.data.list(self.automaton.constraint_vars);
        let n = triples.len() / 3;
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.data.str(triples[3 * mid]).cmp(var) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return self.data.list(Span::new(triples[3 * mid + 1], triples[3 * mid + 2])),
            }
        }
        &[]
    }

    fn tail(self, pattern: usize) -> &'static TailDesc {
        &self.data.tails[self.automaton.tails.off as usize + pattern]
    }
}

// ─── runtime state ───────────────────────────────────────────────────────────

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

fn contains(sorted: &[u32], x: u32) -> bool {
    sorted.binary_search(&x).is_ok()
}

/// The set of patterns still viable on the current path: all of them, or a
/// sorted list.
#[derive(Debug, Clone)]
pub struct Viable {
    all: bool,
    list: Vec<u32>,
    count: u32,
}

impl Viable {
    fn all(count: u32) -> Viable {
        Viable { all: count > 0, list: Vec::new(), count }
    }

    fn is_empty(&self) -> bool {
        !self.all && self.list.is_empty()
    }

    /// True iff some viable pattern is in `set`.
    fn meets(&self, set: &[u32]) -> bool {
        if self.all {
            return !set.is_empty();
        }
        let (small, large) = if self.list.len() <= set.len() { (&self.list[..], set) } else { (set, &self.list[..]) };
        small.iter().any(|&p| contains(large, p))
    }

    /// True iff some viable pattern is not in `set`.
    fn exceeds(&self, set: &[u32]) -> bool {
        if self.all {
            return set.len() < self.count as usize;
        }
        self.list.iter().any(|&p| !contains(set, p))
    }

    fn intersect(&mut self, set: &[u32]) {
        if self.all {
            self.all = false;
            self.list = set.to_vec();
        } else {
            self.list.retain(|&p| contains(set, p));
        }
    }

    fn remove(&mut self, set: &[u32]) {
        if self.all {
            self.all = false;
            self.list = (0..self.count).filter(|&p| !contains(set, p)).collect();
        } else {
            self.list.retain(|&p| !contains(set, p));
        }
    }

    /// The viable pattern indices, in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let all = if self.all { 0..self.count } else { 0..0 };
        all.map(|p| p as usize).chain(self.list.iter().map(|&p| p as usize))
    }
}

/// The mutable state of one search (the interpreter's `MatchIter`).
#[derive(Debug)]
pub struct Run {
    pub subjects: VecDeque<Expr>,
    pub patterns: Viable,
    pub substitution: Substitution,
    active_constraints: Vec<bool>,
    associative: Vec<Option<Head>>,
}

impl Run {
    pub fn new(tables: Tables, subject: Option<Expr>, associative: Option<Head>) -> Self {
        Run {
            subjects: subject.into_iter().collect(),
            patterns: Viable::all(tables.automaton.pattern_count),
            substitution: Substitution::new(),
            active_constraints: vec![true; tables.automaton.constraints.len as usize],
            associative: vec![associative],
        }
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

fn check_constraints(tables: Tables, r: &mut Run, candidates: &[u32], restore: &mut Vec<usize>) {
    for &local in candidates {
        let local = local as usize;
        if !r.active_constraints[local] {
            continue;
        }
        let constraint = tables.constraint(local);
        let patterns = tables.constraint_patterns(local);
        let all_bound = constraint.variables().iter().all(|v| r.substitution.contains(v));
        if all_bound && r.patterns.meets(patterns) {
            r.active_constraints[local] = false;
            restore.push(local);
            if !constraint.check(&r.substitution) {
                r.patterns.remove(patterns);
                if r.patterns.is_empty() {
                    break;
                }
            }
        }
    }
}

fn restore_binding(r: &mut Run, name: &str, old: Option<Replacement>) {
    match old {
        Some(value) => r.substitution.insert(name, value),
        None => {
            r.substitution.remove(name);
        }
    }
}

/// Take transition `t`, binding its variable to `subject`, and continue with
/// `next`. Every change is undone before returning.
#[inline]
pub fn check_transition<M>(
    m: &M,
    tables: Tables,
    r: &mut Run,
    t: &Transition,
    subject: Option<Replacement>,
    next: StateFn<M>,
    k: K<'_>,
) -> Flow {
    let data = tables.data;
    let patterns = data.list(t.patterns);
    if !r.patterns.meets(patterns) {
        return CONTINUE;
    }
    let variable = data.opt_str(t.variable);
    // The viable patterns change if `t` excludes some of them or if a failing
    // constraint removes some; save them only then.
    let narrows = r.patterns.exceeds(patterns);
    let saved_patterns =
        if narrows || (variable.is_some() && t.constraints.len > 0) { Some(r.patterns.clone()) } else { None };
    if narrows {
        r.patterns.intersect(patterns);
    }
    let mut old_values: Vec<(&str, Option<Replacement>)> = Vec::new();
    // The previous value of `variable`, if it was bound by this transition.
    let mut old_variable: Option<(&str, Option<Replacement>)> = None;
    let mut restore = Vec::new();

    let flow = 'body: {
        if let Some(index) = opt(t.subst) {
            for (name, value) in data.subst(index) {
                old_values.push((name, r.substitution.get(name).cloned()));
                if r.substitution.try_add_variable(name, value.clone()).is_err() {
                    break 'body CONTINUE;
                }
            }
        }
        if let Some(var) = variable {
            old_variable = Some((var, r.substitution.get(var).cloned()));
            if let Some(subject) = subject {
                if r.substitution.try_add_variable(var, subject).is_err() {
                    break 'body CONTINUE;
                }
            }
            check_constraints(tables, r, data.list(t.constraints), &mut restore);
            if r.patterns.is_empty() {
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
        restore_binding(r, var, old);
    }
    for (name, old) in old_values.into_iter().rev() {
        restore_binding(r, name, old);
    }
    flow
}

/// An epsilon transition: binds the next subject without consuming it.
#[inline]
pub fn epsilon<M>(m: &M, tables: Tables, r: &mut Run, t: &Transition, next: StateFn<M>, k: K<'_>) -> Flow {
    if !r.patterns.meets(tables.data.list(t.patterns)) {
        return CONTINUE;
    }
    let subject = r.subjects.front().cloned().map(Replacement::Single);
    check_transition(m, tables, r, t, subject, next, k)
}

/// An atom or `End` transition: consumes the next subject (if any).
#[inline]
pub fn consume<M>(m: &M, tables: Tables, r: &mut Run, t: &Transition, next: StateFn<M>, k: K<'_>) -> Flow {
    if !r.patterns.meets(tables.data.list(t.patterns)) {
        return CONTINUE;
    }
    let subject = r.subjects.pop_front();
    let flow = check_transition(m, tables, r, t, subject.clone().map(Replacement::Single), next, k);
    if let Some(subject) = subject {
        r.subjects.push_front(subject);
    }
    flow
}

/// A wildcard transition (`wildcard` is the expression index of the label,
/// a wildcard without variable name).
pub fn wildcard<M>(
    m: &M,
    tables: Tables,
    r: &mut Run,
    t: &Transition,
    wildcard: u32,
    next: StateFn<M>,
    k: K<'_>,
) -> Flow {
    let data = tables.data;
    if !r.patterns.meets(data.list(t.patterns)) {
        return CONTINUE;
    }
    let w = &data.exprs[wildcard as usize];
    let (min_count, fixed_size) = (w.a as usize, w.fixed);
    let default = opt(w.b).map(|d| data.expr(d));
    if let Some(default) = default {
        if min_count > 0 {
            check_transition(m, tables, r, t, Some(Replacement::Single(default.clone())), next, k)?;
        }
    }
    if fixed_size && !r.in_associative() {
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
    if r.subjects.len() < min_count {
        return CONTINUE;
    }
    let mut matched: Vec<Expr> = r.subjects.drain(..min_count).collect();
    let associative = r.associative.last().cloned().flatten();
    let mut flow = CONTINUE;
    loop {
        let wrapped = match &associative {
            Some(head) if fixed_size => {
                if matched.len() > 1 {
                    head.try_call(matched.clone()).ok().map(Replacement::Single)
                } else {
                    matched.first().cloned().map(Replacement::Single)
                }
            }
            _ => match (default, matched.is_empty()) {
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

/// A non-commutative operation with head index `head`: match the operands
/// from `next`, then take the `End` transitions (`end`) of the state reached.
#[allow(clippy::too_many_arguments)]
pub fn regular_operation<M>(
    m: &M,
    tables: Tables,
    r: &mut Run,
    t: &Transition,
    head: u32,
    next: StateFn<M>,
    end: EndFn<M>,
    k: K<'_>,
) -> Flow {
    let data = tables.data;
    if !r.patterns.meets(data.list(t.patterns)) {
        return CONTINUE;
    }
    let desc = &data.heads[head as usize];
    let subject = r.subjects.pop_front().expect("an operation subject");

    // A wildcard head binds the subject's head (undone on backtracking).
    let mut head_binding: Option<(&str, Option<Replacement>)> = None;
    if let (true, Some(var)) = (desc.flags & HEAD_WILDCARD != 0, data.opt_str(desc.variable)) {
        let old = r.substitution.get(var).cloned();
        let bound = Replacement::Head(subject.head().expect("operation subject").clone());
        if r.substitution.try_add_variable(var, bound).is_err() {
            r.subjects.push_front(subject);
            return CONTINUE;
        }
        head_binding = Some((var, old));
    }
    let saved_patterns = head_binding.is_some().then(|| r.patterns.clone());
    let mut restore = Vec::new();
    if let Some((var, _)) = head_binding {
        check_constraints(tables, r, tables.constraints_on(var), &mut restore);
    }
    let flow = if r.patterns.is_empty() {
        CONTINUE
    } else {
        let operands: VecDeque<Expr> = subject.operands().iter().cloned().collect();
        let mut after: Option<VecDeque<Expr>> = Some(std::mem::replace(&mut r.subjects, operands));
        let associative = (desc.flags & HEAD_ASSOCIATIVE != 0).then(|| data.head(head).clone());
        r.associative.push(associative);
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
    if let Some(saved) = saved_patterns {
        r.patterns = saved;
    }
    if let Some((var, old)) = head_binding {
        restore_binding(r, var, old);
    }
    r.subjects.push_front(subject);
    flow
}

/// A commutative operation: match its operands with `core` and take the
/// transitions of each commutative pattern id that matched (`dispatch`).
/// `pattern_id_patterns` (in [`Data::spans`]) holds, per pattern id, the
/// union of those transitions' pattern sets.
#[allow(clippy::too_many_arguments)]
pub fn commutative_operation<M>(
    m: &M,
    tables: Tables,
    r: &mut Run,
    t: &Transition,
    core: &CommutativeCore,
    operand_matches: OperandFn<M>,
    pattern_id_patterns: Span,
    dispatch: PatternIdFn<M>,
    k: K<'_>,
) -> Flow {
    let data = tables.data;
    if !r.patterns.meets(data.list(t.patterns)) {
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
    let variable = data.opt_str(t.variable);
    let potentials = &data.spans[pattern_id_patterns.range()];
    let mut flow = CONTINUE;

    for (matched_pattern, mut new_substitution) in core.match_operands(subject.operands(), &original) {
        if let Some(var) = variable {
            if new_substitution.try_add_variable(var, Replacement::Single(subject.clone())).is_err() {
                continue;
            }
        }
        let Some(&potential) = potentials.get(matched_pattern) else { continue };
        let diff: Vec<String> =
            new_substitution.iter().map(|(k, _)| k).filter(|k| !original.contains(k)).cloned().collect();
        r.substitution = new_substitution;
        r.patterns.intersect(data.list(potential));
        let mut restore = Vec::new();
        for var in &diff {
            check_constraints(tables, r, tables.constraints_on(var), &mut restore);
            if r.patterns.is_empty() {
                break;
            }
        }
        if !r.patterns.is_empty() {
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

/// `subst` with automaton variable names replaced by the original ones.
fn rename(data: &'static Data, subst: &Substitution, inverse: &[u32]) -> Substitution {
    let n = inverse.len() / 2;
    let original = |name: &str| -> Option<&'static str> {
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match data.str(inverse[2 * mid]).cmp(name) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(data.str(inverse[2 * mid + 1])),
            }
        }
        None
    };
    subst.iter().map(|(name, value)| (original(name).unwrap_or(name).to_owned(), value.clone())).collect()
}

/// Run a (top-level) automaton from `root` and call `on_group` with the
/// matches reaching each final state, with the original variable names and
/// after the global constraints.
pub fn for_each_group<M>(
    m: &M,
    tables: Tables,
    root: StateFn<M>,
    subject: &Expr,
    on_group: &mut dyn FnMut(Vec<(usize, Substitution)>) -> Flow,
) -> Flow {
    let data = tables.data;
    let mut r = Run::new(tables, Some(subject.clone()), None);
    root(m, &mut r, &mut |r, _state| {
        let mut group = Vec::new();
        for index in r.patterns.iter() {
            let tail = tables.tail(index);
            let subst = rename(data, &r.substitution, data.list(tail.inverse_renaming));
            if data.list(tail.global_constraints).iter().all(|&c| data.constraint(c).check(&subst)) {
                group.push((index, subst));
            }
        }
        on_group(group)
    })
}

/// Run an operand automaton on one operand: all `(operand pattern, substitution)` matches.
pub fn operand_matches<M>(
    m: &M,
    tables: Tables,
    root: StateFn<M>,
    subject: Option<&Expr>,
    associative: Option<&Head>,
) -> Vec<(usize, Substitution)> {
    let mut r = Run::new(tables, subject.cloned(), associative.cloned());
    let mut out = Vec::new();
    let _ = root(m, &mut r, &mut |r, _| {
        for index in r.patterns.iter() {
            out.push((index, r.substitution.clone()));
        }
        CONTINUE
    });
    out
}
