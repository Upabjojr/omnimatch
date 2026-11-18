//! Expression trees: atoms, wildcards and operations.
//!
//! Port of `omnimatch/expressions/expressions.py`.
//!
//! Python's class hierarchy (`NamedAtom`, `Wildcard`, `Operation`, ...) becomes a
//! single immutable, cheaply clonable [`Expr`] (an `Arc` around a node) whose
//! [`ExprKind`] says which variant it is. Operation metadata lives in an
//! [`OperationHead`], shared through the [`Head`] handle.
//!
//! Not ported yet: `SymbolWrapper` (wrapping arbitrary host-language objects) and
//! `WildcardOperationHead` (only used by the many-to-one matcher).

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use crate::multiset::Multiset;

/// An error constructing an expression or operation head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprError {
    /// The operation name is not a valid identifier (and `infix` was not set).
    InvalidName(String),
    /// `one_identity` was requested for a non-variadic operation.
    OneIdentityNotVariadic,
    /// `infix` was requested for a unary operation.
    InfixUnary,
    /// The operands do not satisfy the head's arity.
    Arity(String),
    /// Invalid wildcard parameters.
    InvalidWildcard(String),
    /// Invalid position or slice.
    InvalidPosition(String),
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprError::InvalidName(name) => write!(
                f,
                "Invalid operation name: {name:?}. Must be a valid non-keyword identifier, or use infix=true."
            ),
            ExprError::OneIdentityNotVariadic => {
                write!(f, "one_identity=true is only allowed for variadic operations.")
            }
            ExprError::InfixUnary => write!(f, "infix=true is not allowed for unary operations."),
            ExprError::Arity(msg) | ExprError::InvalidWildcard(msg) | ExprError::InvalidPosition(msg) => {
                write!(f, "{msg}")
            }
        }
    }
}

impl std::error::Error for ExprError {}

// ─── Arity ───────────────────────────────────────────────────────────────────

/// How many operands an operation accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arity {
    Nullary,
    Unary,
    Binary,
    Ternary,
    /// Zero or more operands.
    Variadic,
}

impl Arity {
    pub fn min_count(self) -> usize {
        match self {
            Arity::Nullary | Arity::Variadic => 0,
            Arity::Unary => 1,
            Arity::Binary => 2,
            Arity::Ternary => 3,
        }
    }

    pub fn fixed_size(self) -> bool {
        self != Arity::Variadic
    }

    pub fn name(self) -> &'static str {
        match self {
            Arity::Nullary => "nullary",
            Arity::Unary => "unary",
            Arity::Binary => "binary",
            Arity::Ternary => "ternary",
            Arity::Variadic => "variadic",
        }
    }
}

// ─── OperationHead ───────────────────────────────────────────────────────────

/// Metadata describing an operation type.
///
/// Equality and hashing ignore `infix` (it only affects printing), matching the
/// Python implementation.
#[derive(Debug, Clone)]
pub struct OperationHead {
    pub name: String,
    pub arity: Arity,
    pub commutative: bool,
    pub associative: bool,
    pub one_identity: bool,
    pub infix: bool,
}

impl OperationHead {
    fn key(&self) -> (&str, Arity, bool, bool, bool) {
        (&self.name, self.arity, self.commutative, self.associative, self.one_identity)
    }
}

impl PartialEq for OperationHead {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for OperationHead {}

impl Hash for OperationHead {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl PartialOrd for OperationHead {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OperationHead {
    /// By name first (as in Python); the remaining fields only break ties so the
    /// order is total and consistent with `Eq`.
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

/// A shared handle to an [`OperationHead`]. Calling [`Head::call`] is the
/// factory that applies `one_identity`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Head(Arc<OperationHead>);

impl Deref for Head {
    type Target = OperationHead;
    fn deref(&self) -> &OperationHead {
        &self.0
    }
}

/// Builder returned by [`Head::builder`].
#[derive(Debug, Clone)]
pub struct HeadBuilder {
    head: OperationHead,
}

impl HeadBuilder {
    pub fn arity(mut self, arity: Arity) -> Self {
        self.head.arity = arity;
        self
    }
    pub fn commutative(mut self, value: bool) -> Self {
        self.head.commutative = value;
        self
    }
    pub fn associative(mut self, value: bool) -> Self {
        self.head.associative = value;
        self
    }
    pub fn one_identity(mut self, value: bool) -> Self {
        self.head.one_identity = value;
        self
    }
    pub fn infix(mut self, value: bool) -> Self {
        self.head.infix = value;
        self
    }

    /// Validate and build the head (Python: `Operation.new`).
    pub fn build(self) -> Result<Head, ExprError> {
        let h = &self.head;
        if !h.infix && !is_valid_operation_name(&h.name) {
            return Err(ExprError::InvalidName(h.name.clone()));
        }
        if h.one_identity && h.arity != Arity::Variadic {
            return Err(ExprError::OneIdentityNotVariadic);
        }
        if h.infix && h.arity == Arity::Unary {
            return Err(ExprError::InfixUnary);
        }
        Ok(Head(Arc::new(self.head)))
    }

    /// Build without validation (Python: constructing `OperationHead(...)` directly).
    pub fn build_unchecked(self) -> Head {
        Head(Arc::new(self.head))
    }
}

/// Python statement keywords that may not be used as operation names. Operator
/// keywords (`and`, `or`, `not`, `in`, `is`, `True`, `False`, `None`) are allowed.
const FORBIDDEN_KEYWORDS: &[&str] = &[
    "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "lambda", "nonlocal", "pass", "raise", "return", "try", "while", "with",
    "yield",
];

fn is_valid_operation_name(name: &str) -> bool {
    let mut chars = name.chars();
    let starts_ok = matches!(chars.next(), Some(c) if c == '_' || c.is_alphabetic());
    starts_ok && chars.all(|c| c == '_' || c.is_alphanumeric()) && !FORBIDDEN_KEYWORDS.contains(&name)
}

impl Head {
    /// Start building a head with the given name (variadic, no flags).
    pub fn builder(name: impl Into<String>) -> HeadBuilder {
        HeadBuilder {
            head: OperationHead {
                name: name.into(),
                arity: Arity::Variadic,
                commutative: false,
                associative: false,
                one_identity: false,
                infix: false,
            },
        }
    }

    /// Shorthand for `Head::builder(name).arity(arity).build()`.
    pub fn new(name: impl Into<String>, arity: Arity) -> Result<Head, ExprError> {
        Head::builder(name).arity(arity).build()
    }

    /// Factory: build an expression, applying `one_identity` (which may return
    /// the single operand instead of an operation).
    pub fn try_call(&self, operands: Vec<Expr>) -> Result<Expr, ExprError> {
        self.try_call_named(operands, None)
    }

    /// Like [`Head::try_call`], attaching `variable_name` to the result.
    pub fn try_call_named(&self, operands: Vec<Expr>, variable_name: Option<&str>) -> Result<Expr, ExprError> {
        if let Some(single) = check_one_identity(self, &operands) {
            return Ok(match variable_name {
                Some(name) => single.with_variable_name(Some(name)),
                None => single,
            });
        }
        Expr::try_operation(self.clone(), operands, variable_name)
    }

    /// Panicking version of [`Head::try_call`], convenient for building
    /// expressions whose arity is known to be right.
    pub fn call(&self, operands: Vec<Expr>) -> Expr {
        self.try_call(operands).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Panicking version of [`Head::try_call_named`].
    pub fn call_named(&self, operands: Vec<Expr>, variable_name: &str) -> Expr {
        self.try_call_named(operands, Some(variable_name)).unwrap_or_else(|e| panic!("{e}"))
    }
}

/// Helper for `one_identity`: the single operand the call collapses to, if any.
fn check_one_identity(head: &Head, operands: &[Expr]) -> Option<Expr> {
    if !head.one_identity {
        return None;
    }
    if head.associative {
        let flat = flatten_associative(head, operands.to_vec());
        return if flat.len() == 1 { flat.into_iter().next() } else { None };
    }
    if operands.len() == 1 {
        Some(operands[0].clone())
    } else {
        None
    }
}

fn flatten_associative(head: &Head, operands: Vec<Expr>) -> Vec<Expr> {
    let mut flat = Vec::with_capacity(operands.len());
    for op in operands {
        match op.kind() {
            ExprKind::Operation { head: h, operands: inner } if h == head && op.variable_name().is_none() => {
                flat.extend(inner.iter().cloned())
            }
            _ => flat.push(op),
        }
    }
    flat
}

fn check_arity(head: &Head, operands: &[Expr]) -> Result<(), ExprError> {
    if !head.arity.fixed_size() {
        return Ok(());
    }
    let required = head.arity.min_count();
    let mut min_ops = 0;
    let mut has_unbounded = false;
    for op in operands {
        match op.as_wildcard() {
            Some(w) if !w.fixed_size => {
                min_ops += w.min_count;
                has_unbounded = true;
            }
            _ => min_ops += 1,
        }
    }
    if has_unbounded {
        if min_ops > required {
            return Err(ExprError::Arity(format!(
                "Operation {:?} with arity {} (requires {} operands) got too many operands: minimum is {}",
                head.name,
                head.arity.name(),
                required,
                min_ops
            )));
        }
    } else if operands.len() != required {
        return Err(ExprError::Arity(format!(
            "Operation {:?} with arity {} requires exactly {} operands, got {}",
            head.name,
            head.arity.name(),
            required,
            operands.len()
        )));
    }
    Ok(())
}

// ─── Expr ────────────────────────────────────────────────────────────────────

/// Parameters of a wildcard.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WildcardSpec {
    /// Minimum number of expressions matched.
    pub min_count: usize,
    /// If true, matches exactly `min_count` expressions.
    pub fixed_size: bool,
    /// Default value for optional wildcards.
    pub default: Option<Expr>,
}

/// The variant of an expression node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprKind {
    /// A constant symbol identified by its name (Python: `NamedAtom`).
    Atom { name: String },
    /// A wildcard (pattern variable when it has a variable name).
    Wildcard(WildcardSpec),
    /// An operation applied to operands.
    Operation { head: Head, operands: Vec<Expr> },
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct Node {
    kind: ExprKind,
    variable_name: Option<String>,
}

/// An immutable expression. Cloning is cheap (reference counted).
#[derive(Clone)]
pub struct Expr(Arc<Node>);

impl Expr {
    fn from_parts(kind: ExprKind, variable_name: Option<String>) -> Expr {
        Expr(Arc::new(Node { kind, variable_name }))
    }

    // ── constructors ──

    /// A constant atom (Python: `NamedAtom(name)`).
    pub fn atom(name: impl Into<String>) -> Expr {
        Expr::from_parts(ExprKind::Atom { name: name.into() }, None)
    }

    /// A general wildcard. `min_count == 0 && fixed_size` requires a default.
    pub fn wildcard(
        min_count: usize,
        fixed_size: bool,
        variable_name: Option<&str>,
        default: Option<Expr>,
    ) -> Result<Expr, ExprError> {
        if fixed_size && min_count == 0 && default.is_none() {
            return Err(ExprError::InvalidWildcard(
                "Wildcard with min_count=0 and fixed_size=true requires a default value (use Expr::optional)".into(),
            ));
        }
        Ok(Expr::from_parts(
            ExprKind::Wildcard(WildcardSpec { min_count, fixed_size, default }),
            variable_name.map(str::to_owned),
        ))
    }

    /// A wildcard matching exactly one expression (`x_`).
    pub fn dot(variable_name: Option<&str>) -> Expr {
        Expr::wildcard(1, true, variable_name, None).expect("valid wildcard")
    }

    /// A wildcard matching zero or more expressions (`x___`).
    pub fn star(variable_name: Option<&str>) -> Expr {
        Expr::wildcard(0, false, variable_name, None).expect("valid wildcard")
    }

    /// A wildcard matching one or more expressions (`x__`).
    pub fn plus(variable_name: Option<&str>) -> Expr {
        Expr::wildcard(1, false, variable_name, None).expect("valid wildcard")
    }

    /// A dot wildcard that falls back to `default` when no subject is available.
    pub fn optional(variable_name: &str, default: Expr) -> Expr {
        Expr::wildcard(1, true, Some(variable_name), Some(default)).expect("valid wildcard")
    }

    /// Raw operation constructor: flattens associative operands and sorts
    /// commutative ones, but does NOT apply `one_identity` (see [`Head::call`]).
    pub fn try_operation(head: Head, operands: Vec<Expr>, variable_name: Option<&str>) -> Result<Expr, ExprError> {
        let mut operands = if head.associative { flatten_associative(&head, operands) } else { operands };
        if head.commutative {
            operands.sort();
        }
        check_arity(&head, &operands)?;
        Ok(Expr::from_parts(ExprKind::Operation { head, operands }, variable_name.map(str::to_owned)))
    }

    /// Panicking version of [`Expr::try_operation`].
    pub fn operation(head: Head, operands: Vec<Expr>) -> Expr {
        Expr::try_operation(head, operands, None).unwrap_or_else(|e| panic!("{e}"))
    }

    /// A copy of this expression with a different variable name.
    pub fn with_variable_name(&self, variable_name: Option<&str>) -> Expr {
        Expr::from_parts(self.0.kind.clone(), variable_name.map(str::to_owned))
    }

    // ── accessors ──

    pub fn kind(&self) -> &ExprKind {
        &self.0.kind
    }

    pub fn variable_name(&self) -> Option<&str> {
        self.0.variable_name.as_deref()
    }

    pub fn is_atom(&self) -> bool {
        matches!(self.kind(), ExprKind::Atom { .. })
    }

    pub fn is_wildcard(&self) -> bool {
        matches!(self.kind(), ExprKind::Wildcard(_))
    }

    pub fn is_operation(&self) -> bool {
        matches!(self.kind(), ExprKind::Operation { .. })
    }

    /// The atom's name, if this is an atom.
    pub fn atom_name(&self) -> Option<&str> {
        match self.kind() {
            ExprKind::Atom { name } => Some(name),
            _ => None,
        }
    }

    pub fn as_wildcard(&self) -> Option<&WildcardSpec> {
        match self.kind() {
            ExprKind::Wildcard(w) => Some(w),
            _ => None,
        }
    }

    /// The head and operands, if this is an operation.
    pub fn as_operation(&self) -> Option<(&Head, &[Expr])> {
        match self.kind() {
            ExprKind::Operation { head, operands } => Some((head, operands)),
            _ => None,
        }
    }

    pub fn head(&self) -> Option<&Head> {
        self.as_operation().map(|(h, _)| h)
    }

    /// The operands of an operation, or an empty slice for atoms and wildcards.
    pub fn operands(&self) -> &[Expr] {
        self.as_operation().map(|(_, ops)| ops).unwrap_or(&[])
    }

    // ── structural queries ──

    /// Pre-order traversal yielding `(subexpression, position)`.
    pub fn preorder_iter(&self) -> Vec<(Expr, Vec<usize>)> {
        fn walk(e: &Expr, pos: &mut Vec<usize>, out: &mut Vec<(Expr, Vec<usize>)>) {
            out.push((e.clone(), pos.clone()));
            for (i, op) in e.operands().iter().enumerate() {
                pos.push(i);
                walk(op, pos, out);
                pos.pop();
            }
        }
        let mut out = Vec::new();
        walk(self, &mut Vec::new(), &mut out);
        out
    }

    /// Visit every subexpression in pre-order.
    pub fn for_each_preorder(&self, f: &mut dyn FnMut(&Expr)) {
        f(self);
        for op in self.operands() {
            op.for_each_preorder(f);
        }
    }

    /// True if no wildcard occurs anywhere (Python: `expressions.functions.is_constant`).
    ///
    /// Note: this ignores variable names on non-wildcards, like the Python
    /// function of the same name (the `Expression.is_constant` *property* does not).
    pub fn is_constant(&self) -> bool {
        match self.kind() {
            ExprKind::Wildcard(_) => false,
            ExprKind::Operation { operands, .. } => operands.iter().all(Expr::is_constant),
            ExprKind::Atom { .. } => true,
        }
    }

    /// True if the expression can be matched purely structurally: no sequence
    /// wildcards and no associative/commutative/one_identity operations.
    pub fn is_syntactic(&self) -> bool {
        match self.kind() {
            ExprKind::Operation { head, operands } => {
                !(head.associative || head.commutative || head.one_identity) && operands.iter().all(Expr::is_syntactic)
            }
            ExprKind::Wildcard(w) => w.fixed_size,
            ExprKind::Atom { .. } => true,
        }
    }

    /// True if no subexpression has a variable name.
    pub fn is_anonymous(&self) -> bool {
        self.variable_name().is_none() && self.operands().iter().all(Expr::is_anonymous)
    }

    /// Multiset of symbol names (atom names and operation head names).
    pub fn symbols(&self) -> Multiset<String> {
        let mut result = Multiset::new();
        self.for_each_preorder(&mut |e| match e.kind() {
            ExprKind::Operation { head, .. } => result.add(head.name.clone()),
            ExprKind::Atom { name } => result.add(name.clone()),
            ExprKind::Wildcard(_) => {}
        });
        result
    }

    /// Multiset of variable names.
    pub fn variables(&self) -> Multiset<String> {
        let mut result = Multiset::new();
        self.for_each_preorder(&mut |e| {
            if let Some(v) = e.variable_name() {
                result.add(v.to_owned());
            }
        });
        result
    }

    /// Set of variable names (Python: `get_variables`).
    pub fn variable_set(&self) -> BTreeSet<String> {
        self.variables().distinct_elements().cloned().collect()
    }

    /// True if any variable name from `variables` occurs in the expression.
    pub fn contains_variables_from_set(&self, variables: &BTreeSet<String>) -> bool {
        self.variable_name().is_some_and(|v| variables.contains(v))
            || self.operands().iter().any(|o| o.contains_variables_from_set(variables))
    }

    /// True if `needle` occurs anywhere in this expression tree.
    pub fn contains(&self, needle: &Expr) -> bool {
        self == needle || self.operands().iter().any(|o| o.contains(needle))
    }

    /// The subexpression at `position`.
    pub fn get(&self, position: &[usize]) -> Result<&Expr, ExprError> {
        let mut current = self;
        for &idx in position {
            current =
                current.operands().get(idx).ok_or_else(|| ExprError::InvalidPosition("Invalid position".into()))?;
        }
        Ok(current)
    }

    /// A copy with variable names renamed according to `renaming`.
    pub fn with_renamed_vars(&self, renaming: &HashMap<String, String>) -> Expr {
        let new_name = self.variable_name().map(|v| renaming.get(v).map_or(v, String::as_str));
        match self.kind() {
            ExprKind::Operation { head, operands } => {
                let new_ops = operands.iter().map(|o| o.with_renamed_vars(renaming)).collect();
                Expr::try_operation(head.clone(), new_ops, new_name).expect("renaming preserves arity")
            }
            _ => self.with_variable_name(new_name),
        }
    }

    fn rank(&self) -> u8 {
        match self.kind() {
            ExprKind::Atom { .. } => 0,
            ExprKind::Wildcard(_) => 1,
            ExprKind::Operation { .. } => 2,
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Variable names compare as in Python (`None` sorts like `""`), with `None`
/// before `Some("")` to keep the order total.
fn cmp_var(a: Option<&str>, b: Option<&str>) -> Ordering {
    a.unwrap_or("").cmp(b.unwrap_or("")).then(a.is_some().cmp(&b.is_some()))
}

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Expr {
    /// The canonical order used to sort commutative operands:
    /// atoms < wildcards < operations.
    ///
    /// * atoms: by name, then variable name;
    /// * wildcards: dot before sequence, then `min_count`, then variable name;
    /// * operations: by head, then operand count, then operands, then variable name.
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.0, &other.0) {
            return Ordering::Equal;
        }
        let by_kind = match (self.kind(), other.kind()) {
            (ExprKind::Atom { name: a }, ExprKind::Atom { name: b }) => a.cmp(b),
            (ExprKind::Wildcard(a), ExprKind::Wildcard(b)) => b
                .fixed_size
                .cmp(&a.fixed_size)
                .then(a.min_count.cmp(&b.min_count))
                .then_with(|| cmp_var(self.variable_name(), other.variable_name()))
                .then_with(|| a.default.cmp(&b.default)),
            (ExprKind::Operation { head: ha, operands: oa }, ExprKind::Operation { head: hb, operands: ob }) => {
                ha.cmp(hb).then(oa.len().cmp(&ob.len())).then_with(|| oa.cmp(ob))
            }
            _ => return self.rank().cmp(&other.rank()),
        };
        by_kind.then_with(|| cmp_var(self.variable_name(), other.variable_name()))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            ExprKind::Atom { name } => match self.variable_name() {
                // Same (odd) order as Python's NamedAtom.__str__.
                Some(v) => write!(f, "{name}: {v}"),
                None => write!(f, "{name}"),
            },
            ExprKind::Wildcard(w) => {
                let suffix = if w.fixed_size {
                    "_"
                } else if w.min_count == 0 {
                    "___"
                } else {
                    "__"
                };
                write!(f, "{}{}", self.variable_name().unwrap_or(""), suffix)
            }
            ExprKind::Operation { head, operands } => {
                if let Some(v) = self.variable_name() {
                    write!(f, "{v}: ")?;
                }
                write!(f, "{}(", head.name)?;
                for (i, op) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{op}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            ExprKind::Atom { name } => match self.variable_name() {
                Some(v) => write!(f, "NamedAtom({name:?}, variable_name={v})"),
                None => write!(f, "NamedAtom({name:?})"),
            },
            ExprKind::Wildcard(w) => match self.variable_name() {
                Some(v) => {
                    let kind = if w.fixed_size {
                        "dot"
                    } else if w.min_count == 0 {
                        "star"
                    } else {
                        "plus"
                    };
                    write!(f, "Wildcard.{kind}({v:?})")
                }
                None => write!(f, "Wildcard({}, {})", w.min_count, w.fixed_size),
            },
            ExprKind::Operation { head, operands } => {
                write!(f, "Operation({}", head.name)?;
                for op in operands {
                    write!(f, ", {op:?}")?;
                }
                if let Some(v) = self.variable_name() {
                    write!(f, ", variable_name={v}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Built-in heads used to represent host-language containers.
pub mod builtin_heads {
    use super::*;
    use std::sync::OnceLock;

    pub fn list() -> Head {
        static H: OnceLock<Head> = OnceLock::new();
        H.get_or_init(|| Head::builder("list").build_unchecked()).clone()
    }
    pub fn tuple() -> Head {
        static H: OnceLock<Head> = OnceLock::new();
        H.get_or_init(|| Head::builder("tuple").build_unchecked()).clone()
    }
    pub fn dict() -> Head {
        static H: OnceLock<Head> = OnceLock::new();
        H.get_or_init(|| Head::builder("dict").commutative(true).build_unchecked()).clone()
    }
    pub fn dict_pair() -> Head {
        static H: OnceLock<Head> = OnceLock::new();
        H.get_or_init(|| Head::builder("dictpair").arity(Arity::Binary).build_unchecked()).clone()
    }
}

// ─── Pattern ─────────────────────────────────────────────────────────────────

use crate::constraints::Constraint;

/// An expression together with constraints that a match must satisfy.
#[derive(Clone)]
pub struct Pattern {
    pub expression: Expr,
    pub constraints: Vec<Arc<dyn Constraint>>,
}

impl Pattern {
    pub fn new(expression: Expr) -> Pattern {
        Pattern { expression, constraints: Vec::new() }
    }

    pub fn with_constraints(expression: Expr, constraints: Vec<Arc<dyn Constraint>>) -> Pattern {
        Pattern { expression, constraints }
    }

    /// Constraints that depend on pattern variables (checked during matching).
    pub fn local_constraints(&self) -> Vec<Arc<dyn Constraint>> {
        self.constraints.iter().filter(|c| !c.variables().is_empty()).cloned().collect()
    }

    /// Constraints with no variables (checked once a match is complete).
    pub fn global_constraints(&self) -> Vec<Arc<dyn Constraint>> {
        self.constraints.iter().filter(|c| c.variables().is_empty()).cloned().collect()
    }
}

impl From<Expr> for Pattern {
    fn from(expression: Expr) -> Pattern {
        Pattern::new(expression)
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expression)
    }
}

impl fmt::Debug for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pattern({:?}", self.expression)?;
        for c in &self.constraints {
            write!(f, ", {c:?}")?;
        }
        write!(f, ")")
    }
}

// ─── factory helpers ─────────────────────────────────────────────────────────

/// A named dot wildcard (`x_`).
pub fn make_dot_variable(name: &str) -> Expr {
    Expr::dot(Some(name))
}

/// A named plus wildcard (`x__`).
pub fn make_plus_variable(name: &str) -> Expr {
    Expr::plus(Some(name))
}

/// A named star wildcard (`x___`).
pub fn make_star_variable(name: &str) -> Expr {
    Expr::star(Some(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a() -> Expr {
        Expr::atom("a")
    }
    fn b() -> Expr {
        Expr::atom("b")
    }

    #[test]
    fn head_validation() {
        assert!(Head::new("f", Arity::Variadic).is_ok());
        assert!(Head::new("and", Arity::Binary).is_ok());
        assert_eq!(Head::new("1f", Arity::Variadic), Err(ExprError::InvalidName("1f".into())));
        assert_eq!(Head::new("class", Arity::Variadic), Err(ExprError::InvalidName("class".into())));
        assert!(Head::builder("+").infix(true).build().is_ok());
        assert_eq!(
            Head::builder("g").arity(Arity::Unary).one_identity(true).build(),
            Err(ExprError::OneIdentityNotVariadic)
        );
    }

    #[test]
    fn display() {
        let f = Head::new("f", Arity::Variadic).unwrap();
        assert_eq!(f.call(vec![a(), a()]).to_string(), "f(a, a)");
        assert_eq!(f.call(vec![a(), Expr::dot(Some("x"))]).to_string(), "f(a, x_)");
        let w = Expr::dot(None);
        assert_eq!(f.call(vec![w.clone(), w]).to_string(), "f(_, _)");
        assert_eq!(f.call_named(vec![Expr::dot(None), a()], "y").to_string(), "y: f(_, a)");
        assert_eq!(make_star_variable("x").to_string(), "x___");
        assert_eq!(Expr::plus(None).to_string(), "__");
    }

    #[test]
    fn normalization() {
        let fc = Head::builder("f_c").commutative(true).build().unwrap();
        assert_eq!(fc.call(vec![b(), a()]), fc.call(vec![a(), b()]));
        let fa = Head::builder("f_a").associative(true).build().unwrap();
        let nested = fa.call(vec![a(), fa.call(vec![b(), a()])]);
        assert_eq!(nested.operands().len(), 3);
        let fi = Head::builder("f_i").one_identity(true).build().unwrap();
        assert_eq!(fi.call(vec![a()]), a());
        assert!(Expr::operation(fi, vec![a()]).is_operation());
    }

    #[test]
    fn arity_check() {
        let g = Head::new("g", Arity::Binary).unwrap();
        assert!(g.try_call(vec![a()]).is_err());
        assert!(g.try_call(vec![a(), b()]).is_ok());
        assert!(g.try_call(vec![Expr::star(None)]).is_ok());
        assert!(g.try_call(vec![Expr::plus(None), Expr::plus(None), Expr::plus(None)]).is_err());
    }

    #[test]
    fn ordering() {
        let f = Head::new("f", Arity::Variadic).unwrap();
        let x = Expr::dot(Some("x"));
        let xs = Expr::star(Some("x"));
        let op = f.call(vec![a()]);
        assert!(a() < b());
        assert!(b() < x);
        assert!(x < xs);
        assert!(xs < op);
        assert!(f.call(vec![a()]) < f.call(vec![a(), a()]));
    }

    #[test]
    fn positions_and_queries() {
        let f = Head::new("f", Arity::Variadic).unwrap();
        let e = f.call(vec![a(), f.call(vec![b(), make_dot_variable("x")])]);
        assert_eq!(e.get(&[1, 0]).unwrap(), &b());
        assert!(e.get(&[2]).is_err());
        assert!(!e.is_constant());
        assert!(e.is_syntactic());
        assert_eq!(e.symbols().count(&"f".to_string()), 2);
        assert!(e.variable_set().contains("x"));
        assert!(e.contains(&b()));
        let renamed = e.with_renamed_vars(&HashMap::from([("x".to_string(), "y".to_string())]));
        assert!(renamed.variable_set().contains("y"));
        assert_eq!(e.preorder_iter().len(), 5);
    }
}
