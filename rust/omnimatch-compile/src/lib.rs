//! Compile [`omnimatch`] many-to-one matchers to Rust code at build time.
//!
//! This replaces Python's `omnimatch.matching.code_generation`. Instead of
//! printing source text, the compiler turns the automaton of a
//! [`ManyToOneMatcher`] into Rust tokens: one
//! function per state, whose dispatch on the next subject is a `match` on head
//! and atom names. All other data (pattern sets, bindings, expressions,
//! constraints) becomes `static` tables of integers, so a compiled matcher
//! holds its rules in the binary's read-only data instead of on the heap.
//! The few values that must be objects are built from those tables the first
//! time they are used. rustc (and LLVM) compile the code together with the
//! [`omnimatch::compiled`] runtime, which does the variable binding,
//! backtracking and commutative (bipartite) matching shared by all matchers.
//!
//! Use it from a build script, then include the generated file:
//!
//! ```no_run
//! // build.rs
//! use omnimatch::*;
//! use omnimatch_compile::Compiler;
//!
//! let f = Head::new("f", Arity::Binary).unwrap();
//! let mut compiler = Compiler::new("RuleMatcher");
//! compiler.add(Pattern::new(f.call(vec![Expr::atom("a"), make_dot_variable("x")])));
//! compiler.add(Pattern::new(f.call(vec![make_dot_variable("x"), Expr::atom("b")])));
//! let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("rules.rs");
//! compiler.write(out).unwrap();
//! ```
//!
//! ```ignore
//! // src/lib.rs
//! include!(concat!(env!("OUT_DIR"), "/rules.rs"));
//!
//! let matcher = RuleMatcher::new();
//! for (pattern_index, substitution) in matcher.match_all(&subject) { /* ... */ }
//! ```
//!
//! The generated type has the same matching methods as `ManyToOneMatcher`,
//! with the index returned by [`Compiler::add`] in place of the label, and
//! finds the same matches in the same order.
//!
//! Constraints are rebuilt in the generated code from
//! [`Constraint::rust_source`].
//! `EqualVariablesConstraint` and `FreeOf` support it. A `CustomConstraint`
//! needs a named function, given with
//! [`CustomConstraint::with_rust_path`](omnimatch::CustomConstraint::with_rust_path).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;

use omnimatch::compiled::{
    arity_code, AutomatonInfo, CommutativeInfo, KeyInfo, LabelInfo, Span, TransitionInfo, HEAD_ASSOCIATIVE,
    HEAD_COMMUTATIVE, HEAD_INFIX, HEAD_ONE_IDENTITY, HEAD_WILDCARD, NONE,
};
use omnimatch::{Constraint, Expr, ExprKind, Head, ManyToOneMatcher, Pattern, Replacement, Substitution};
use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote, ToTokens};

/// Why a matcher could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The struct name is not a valid Rust identifier.
    InvalidName(String),
    /// A constraint has no [`Constraint::rust_source`] (e.g. a
    /// `CustomConstraint` without [`with_rust_path`](omnimatch::CustomConstraint::with_rust_path)).
    UnsupportedConstraint(String),
    /// A constraint's `rust_source` is not valid Rust.
    InvalidConstraintSource(String),
    /// Writing the output file failed.
    Io(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::InvalidName(name) => write!(f, "{name:?} is not a valid Rust type name"),
            CompileError::UnsupportedConstraint(c) => {
                write!(f, "constraint {c} cannot be compiled (a CustomConstraint needs `with_rust_path`)")
            }
            CompileError::InvalidConstraintSource(s) => write!(f, "constraint source is not valid Rust: {s}"),
            CompileError::Io(e) => write!(f, "cannot write the generated code: {e}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Collects patterns and generates a compiled matcher for them.
pub struct Compiler {
    name: String,
    matcher: ManyToOneMatcher<()>,
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compiler").field("name", &self.name).field("patterns", &self.matcher.len()).finish()
    }
}

impl Compiler {
    /// A compiler for a matcher type called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Compiler { name: name.into(), matcher: ManyToOneMatcher::new() }
    }

    /// Add a pattern. Returns its index, which the compiled matcher reports
    /// with its matches. Adding the same pattern again returns the same index.
    pub fn add(&mut self, pattern: Pattern) -> usize {
        self.matcher.add_with_label(pattern, ())
    }

    /// Number of distinct patterns added.
    pub fn len(&self) -> usize {
        self.matcher.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matcher.is_empty()
    }

    /// The interpreted matcher for the same patterns (same pattern indices).
    pub fn interpreted(&self) -> &ManyToOneMatcher<()> {
        &self.matcher
    }

    /// The generated code as tokens.
    pub fn tokens(&self) -> Result<TokenStream, CompileError> {
        let name = syn_ident(&self.name)?;
        Generator::new(self.matcher.export()).generate(&name)
    }

    /// The generated code as a string (unformatted).
    pub fn generate(&self) -> Result<String, CompileError> {
        Ok(self.tokens()?.to_string())
    }

    /// Write the generated code to `path` (typically in `OUT_DIR`).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), CompileError> {
        let code = self.generate()?;
        std::fs::write(path, code).map_err(|e| CompileError::Io(e.to_string()))
    }
}

fn syn_ident(name: &str) -> Result<Ident, CompileError> {
    let mut chars = name.chars();
    let ok = matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric());
    if !ok || name == "_" {
        return Err(CompileError::InvalidName(name.to_owned()));
    }
    Ok(Ident::new(name, proc_macro2::Span::call_site()))
}

fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

// ─── generation ──────────────────────────────────────────────────────────────

/// One automaton to generate: the top-level one (index 0) or the operand
/// automaton of a commutative state.
struct Unit {
    info: AutomatonInfo,
    /// For operand automata: the index of the commutative core it belongs to.
    core: Option<usize>,
}

struct Core {
    operands: usize,
    info: CommutativeInfo,
    /// Per commutative pattern id, the union of its transitions' pattern sets (in `spans`).
    pid_sets: Span,
}

fn lit(x: impl TryInto<u32> + Copy + fmt::Debug) -> Literal {
    Literal::u32_unsuffixed(x.try_into().unwrap_or_else(|_| panic!("{x:?} does not fit in u32")))
}

fn span_tokens(s: Span) -> TokenStream {
    let (off, len) = (lit(s.off), lit(s.len));
    quote!(rt::Span { off: #off, len: #len })
}

/// The static tables, as they are filled.
#[derive(Default)]
struct Tables {
    text: String,
    strings: Vec<Span>,
    string_index: HashMap<String, u32>,
    u32s: Vec<u32>,
    list_index: HashMap<Vec<u32>, Span>,
    spans: Vec<Span>,
    heads: Vec<TokenStream>,
    head_index: HashMap<Head, u32>,
    exprs: Vec<TokenStream>,
    expr_index: HashMap<Expr, u32>,
    replacements: Vec<TokenStream>,
    substs: Vec<Span>,
    constraints: Vec<TokenStream>,
    constraint_index: HashMap<String, u32>,
    automata: Vec<TokenStream>,
    tails: Vec<TokenStream>,
    cores: Vec<TokenStream>,
    core_patterns: Vec<TokenStream>,
    variables: Vec<TokenStream>,
    transitions: Vec<TokenStream>,
}

impl Tables {
    fn string(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.string_index.get(s) {
            return i;
        }
        let i = self.strings.len() as u32;
        self.strings.push(Span::new(self.text.len() as u32, s.len() as u32));
        self.text.push_str(s);
        self.string_index.insert(s.to_owned(), i);
        i
    }

    fn opt_string(&mut self, s: Option<&str>) -> u32 {
        s.map_or(NONE, |s| self.string(s))
    }

    /// A list in `u32s` (identical lists are shared).
    fn list(&mut self, items: Vec<u32>) -> Span {
        if items.is_empty() {
            return Span::EMPTY;
        }
        if let Some(&s) = self.list_index.get(&items) {
            return s;
        }
        let s = Span::new(self.u32s.len() as u32, items.len() as u32);
        self.u32s.extend(&items);
        self.list_index.insert(items, s);
        s
    }

    fn set(&mut self, set: &BTreeSet<usize>) -> Span {
        self.list(set.iter().map(|&i| i as u32).collect())
    }

    /// Consecutive entries of `spans`.
    fn spans(&mut self, items: Vec<Span>) -> Span {
        let s = Span::new(self.spans.len() as u32, items.len() as u32);
        self.spans.extend(items);
        s
    }

    fn head(&mut self, head: &Head) -> u32 {
        // `Head` equality ignores `infix`; keep the first one seen.
        if let Some(&i) = self.head_index.get(head) {
            return i;
        }
        let name = lit(self.string(&head.name));
        let arity = lit(arity_code(head.arity));
        let mut flags = 0u8;
        for (on, bit) in [
            (head.commutative, HEAD_COMMUTATIVE),
            (head.associative, HEAD_ASSOCIATIVE),
            (head.one_identity, HEAD_ONE_IDENTITY),
            (head.infix, HEAD_INFIX),
            (head.wildcard, HEAD_WILDCARD),
        ] {
            if on {
                flags |= bit;
            }
        }
        let flags = lit(flags);
        let variable = lit(self.opt_string(head.head_variable.as_deref()));
        let i = self.heads.len() as u32;
        self.heads.push(quote!(rt::HeadDesc { name: #name, arity: #arity, flags: #flags, variable: #variable }));
        self.head_index.insert(head.clone(), i);
        i
    }

    fn expr(&mut self, expr: &Expr) -> u32 {
        if let Some(&i) = self.expr_index.get(expr) {
            return i;
        }
        let variable = self.opt_string(expr.variable_name());
        let (kind, fixed, a, b, operands) = match expr.kind() {
            ExprKind::Atom { name } => (quote!(rt::EXPR_ATOM), false, self.string(name), NONE, Span::EMPTY),
            ExprKind::Wildcard(w) => {
                let default = w.default.as_ref().map_or(NONE, |d| self.expr(d));
                (quote!(rt::EXPR_WILDCARD), w.fixed_size, w.min_count as u32, default, Span::EMPTY)
            }
            ExprKind::Operation { head, operands } => {
                let head = self.head(head);
                let operands = operands.iter().map(|o| self.expr(o)).collect();
                (quote!(rt::EXPR_OPERATION), false, head, NONE, self.list(operands))
            }
        };
        let (variable, a, b, operands) = (lit(variable), lit(a), lit(b), span_tokens(operands));
        let i = self.exprs.len() as u32;
        self.exprs.push(quote! {
            rt::ExprDesc { kind: #kind, fixed: #fixed, variable: #variable, a: #a, b: #b, operands: #operands }
        });
        self.expr_index.insert(expr.clone(), i);
        i
    }

    fn replacement(&mut self, r: &Replacement) -> u32 {
        let (kind, a, items) = match r {
            Replacement::Single(e) => (quote!(rt::REPLACEMENT_SINGLE), self.expr(e), Span::EMPTY),
            Replacement::Sequence(es) => {
                let items = es.iter().map(|e| self.expr(e)).collect();
                (quote!(rt::REPLACEMENT_SEQUENCE), NONE, self.list(items))
            }
            Replacement::Multiset(ms) => {
                let items = ms.iter().map(|e| self.expr(e)).collect();
                (quote!(rt::REPLACEMENT_MULTISET), NONE, self.list(items))
            }
            Replacement::Head(h) => (quote!(rt::REPLACEMENT_HEAD), self.head(h), Span::EMPTY),
        };
        let (a, items) = (lit(a), span_tokens(items));
        let i = self.replacements.len() as u32;
        self.replacements.push(quote!(rt::ReplacementDesc { kind: #kind, a: #a, items: #items }));
        i
    }

    fn subst(&mut self, s: &Substitution) -> u32 {
        let mut pairs = Vec::new();
        for (name, value) in s {
            pairs.push(self.string(name));
            pairs.push(self.replacement(value));
        }
        let span = self.list(pairs);
        let i = self.substs.len() as u32;
        self.substs.push(span);
        i
    }

    fn constraint(&mut self, c: &dyn Constraint) -> Result<u32, CompileError> {
        let source = c.rust_source().ok_or_else(|| CompileError::UnsupportedConstraint(format!("{c:?}")))?;
        if let Some(&i) = self.constraint_index.get(&source) {
            return Ok(i);
        }
        let tokens: TokenStream =
            source.parse().map_err(|e| CompileError::InvalidConstraintSource(format!("{source}: {e}")))?;
        let i = self.constraints.len() as u32;
        self.constraints.push(tokens);
        self.constraint_index.insert(source, i);
        Ok(i)
    }

    fn transition(&mut self, t: &TransitionInfo) -> Literal {
        let patterns = span_tokens(self.set(&t.patterns));
        let variable = lit(self.opt_string(t.variable_name.as_deref()));
        let constraints = span_tokens(self.set(&t.check_constraints));
        let subst = lit(t.subst.as_ref().map_or(NONE, |s| self.subst(s)));
        let i = lit(self.transitions.len());
        self.transitions.push(quote! {
            rt::Transition { patterns: #patterns, variable: #variable, constraints: #constraints, subst: #subst }
        });
        i
    }
}

/// A `static` array of `items` called `name`.
fn table(name: &str, ty: TokenStream, items: &[TokenStream]) -> TokenStream {
    let name = format_ident!("{}", name);
    let n = items.len();
    quote!(static #name: [#ty; #n] = [#(#items),*];)
}

/// A `static` array of empty cells called `name`.
fn cells(name: &str, ty: TokenStream, n: usize) -> TokenStream {
    let name = format_ident!("{}", name);
    let empty = format_ident!("{}_EMPTY", name);
    quote! {
        const #empty: ::std::sync::OnceLock<#ty> = ::std::sync::OnceLock::new();
        static #name: [::std::sync::OnceLock<#ty>; #n] = [#empty; #n];
    }
}

/// Arms per constraint constructor function (no function grows with the number of constraints).
const CONSTRAINTS_PER_FN: usize = 64;

struct Generator {
    units: Vec<Unit>,
    cores: Vec<Core>,
    /// (unit, state) -> core index.
    core_of: HashMap<(usize, usize), usize>,
    tables: Tables,
}

impl Generator {
    fn new(top: AutomatonInfo) -> Self {
        let mut g =
            Generator { units: Vec::new(), cores: Vec::new(), core_of: HashMap::new(), tables: Tables::default() };
        g.add_unit(top, None);
        g
    }

    fn add_unit(&mut self, info: AutomatonInfo, core: Option<usize>) -> usize {
        let unit = self.units.len();
        let commutative: Vec<(usize, CommutativeInfo)> =
            info.states.iter().enumerate().filter_map(|(i, s)| s.commutative.clone().map(|c| (i, c))).collect();
        self.units.push(Unit { info, core });
        for (state, mut c) in commutative {
            let core_index = self.cores.len();
            let operand_info = std::mem::replace(
                &mut *c.automaton,
                AutomatonInfo {
                    states: Vec::new(),
                    finals: Default::default(),
                    patterns: Vec::new(),
                    constraints: Vec::new(),
                    constraint_vars: Default::default(),
                },
            );
            self.cores.push(Core { operands: usize::MAX, info: c, pid_sets: Span::EMPTY });
            self.core_of.insert((unit, state), core_index);
            let operands = self.add_unit(operand_info, Some(core_index));
            self.cores[core_index].operands = operands;
        }
        unit
    }

    fn generate(mut self, name: &Ident) -> Result<TokenStream, CompileError> {
        let mut items = Vec::new();
        for u in 0..self.units.len() {
            items.push(self.unit(u)?);
        }
        for c in 0..self.cores.len() {
            self.core(c);
        }

        let t = &self.tables;
        let constraint_fns = constraint_fns(&t.constraints);
        let text = Literal::string(&t.text);
        let strings: Vec<TokenStream> = t.strings.iter().map(|&s| span_tokens(s)).collect();
        let u32s: Vec<TokenStream> = t.u32s.iter().map(|&x| lit(x).into_token_stream()).collect();
        let spans: Vec<TokenStream> = t.spans.iter().map(|&s| span_tokens(s)).collect();
        let substs: Vec<TokenStream> = t.substs.iter().map(|&s| span_tokens(s)).collect();
        let statics = [
            table("STRINGS", quote!(rt::Span), &strings),
            table("U32S", quote!(u32), &u32s),
            table("SPANS", quote!(rt::Span), &spans),
            table("HEADS", quote!(rt::HeadDesc), &t.heads),
            cells("HEAD_CELLS", quote!(::omnimatch::Head), t.heads.len()),
            table("EXPRS", quote!(rt::ExprDesc), &t.exprs),
            cells("EXPR_CELLS", quote!(Expr), t.exprs.len()),
            table("REPLACEMENTS", quote!(rt::ReplacementDesc), &t.replacements),
            table("SUBSTS", quote!(rt::Span), &substs),
            cells("SUBST_CELLS", quote!(Substitution), t.substs.len()),
            cells("CONSTRAINT_CELLS", quote!(::std::sync::Arc<dyn ::omnimatch::Constraint>), t.constraints.len()),
            table("AUTOMATA", quote!(rt::AutomatonDesc), &t.automata),
            table("TAILS", quote!(rt::TailDesc), &t.tails),
            table("CORES", quote!(rt::CoreDesc), &t.cores),
            table("CORE_PATTERNS", quote!(rt::CorePatternDesc), &t.core_patterns),
            table("VARIABLES", quote!(rt::VariableDesc), &t.variables),
            table("T", quote!(rt::Transition), &t.transitions),
        ];
        let core_count = self.cores.len();
        let pid_sets = self.cores.iter().enumerate().map(|(c, core)| {
            let name = format_ident!("PID_SETS_{}", c);
            let span = span_tokens(core.pid_sets);
            quote!(const #name: rt::Span = #span;)
        });
        let module = format_ident!("__omnimatch_{}", snake_case(&name.to_string()));
        let pattern_count = self.units[0].info.patterns.len();

        Ok(quote! {
            #[allow(unused, non_snake_case, non_upper_case_globals, clippy::all, clippy::pedantic)]
            mod #module {
                use ::omnimatch::compiled as rt;
                use ::omnimatch::{Expr, Substitution};

                static TEXT: &str = #text;
                #(#statics)*
                #constraint_fns
                #(#pid_sets)*

                static DATA: rt::Data = rt::Data {
                    text: TEXT,
                    strings: &STRINGS,
                    u32s: &U32S,
                    spans: &SPANS,
                    heads: &HEADS,
                    head_cells: &HEAD_CELLS,
                    exprs: &EXPRS,
                    expr_cells: &EXPR_CELLS,
                    replacements: &REPLACEMENTS,
                    substs: &SUBSTS,
                    subst_cells: &SUBST_CELLS,
                    make_constraint: make_constraint,
                    constraint_cells: &CONSTRAINT_CELLS,
                    automata: &AUTOMATA,
                    tails: &TAILS,
                    cores: &CORES,
                    core_patterns: &CORE_PATTERNS,
                    variables: &VARIABLES,
                };

                #[inline(always)]
                fn tb(automaton: usize) -> rt::Tables {
                    rt::Tables::new(&DATA, automaton)
                }

                /// A many-to-one matcher compiled by `omnimatch-compile`.
                ///
                /// The patterns are static data. A matcher only allocates the
                /// commutative matchers it uses, with their caches of operand
                /// matches; expressions and constraints are built once per
                /// program when first needed.
                pub struct Matcher {
                    cores: ::std::boxed::Box<[::std::cell::OnceCell<rt::CommutativeCore>]>,
                }

                impl ::core::default::Default for Matcher {
                    fn default() -> Self {
                        Self::new()
                    }
                }

                impl ::core::fmt::Debug for Matcher {
                    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                        f.debug_struct(stringify!(#name)).field("patterns", &#pattern_count).finish()
                    }
                }

                impl Matcher {
                    /// Number of patterns.
                    pub const PATTERN_COUNT: usize = #pattern_count;

                    pub fn new() -> Self {
                        Matcher { cores: (0..#core_count).map(|_| ::std::cell::OnceCell::new()).collect() }
                    }

                    fn core(&self, c: usize) -> &rt::CommutativeCore {
                        self.cores[c].get_or_init(|| DATA.core(c))
                    }

                    /// Drop the operands cached by the commutative matchers.
                    pub fn clear(&self) {
                        for core in self.cores.iter() {
                            if let ::core::option::Option::Some(core) = core.get() {
                                core.clear();
                            }
                        }
                    }

                    /// Call `on_group` with the matches reaching each final state
                    /// (patterns differing only in constraints are grouped).
                    pub fn for_each_group(
                        &self,
                        subject: &Expr,
                        on_group: &mut dyn FnMut(::std::vec::Vec<(usize, Substitution)>) -> rt::Flow,
                    ) -> rt::Flow {
                        rt::for_each_group(self, tb(0), a0_s0, subject, on_group)
                    }

                    /// Call `on_match` with the pattern index and substitution of every match.
                    pub fn for_each_match(
                        &self,
                        subject: &Expr,
                        on_match: &mut dyn FnMut(usize, Substitution) -> rt::Flow,
                    ) -> rt::Flow {
                        self.for_each_group(subject, &mut |group| {
                            for (index, subst) in group {
                                on_match(index, subst)?;
                            }
                            rt::CONTINUE
                        })
                    }

                    /// All matches, as `(pattern index, substitution)` pairs.
                    pub fn match_all(&self, subject: &Expr) -> ::std::vec::Vec<(usize, Substitution)> {
                        let mut out = ::std::vec::Vec::new();
                        let _ = self.for_each_group(subject, &mut |group| {
                            out.extend(group);
                            rt::CONTINUE
                        });
                        out
                    }

                    /// The first match, if any.
                    pub fn first_match(&self, subject: &Expr) -> ::core::option::Option<(usize, Substitution)> {
                        let mut found = ::core::option::Option::None;
                        let _ = self.for_each_group(subject, &mut |group| match group.into_iter().next() {
                            ::core::option::Option::Some(m) => {
                                found = ::core::option::Option::Some(m);
                                ::core::ops::ControlFlow::Break(())
                            }
                            ::core::option::Option::None => rt::CONTINUE,
                        });
                        found
                    }

                    /// True iff any pattern matches.
                    pub fn is_match(&self, subject: &Expr) -> bool {
                        self.first_match(subject).is_some()
                    }
                }

                #(#items)*
            }

            pub use #module::Matcher as #name;
        })
    }

    /// The static data of commutative core `c` (after its automata).
    fn core(&mut self, c: usize) {
        let info = &self.cores[c].info;
        let t = &mut self.tables;
        let associative = lit(info.associative.as_ref().map_or(NONE, |h| t.head(h)));
        let max_optional = lit(info.max_optional_count);
        let anonymous = span_tokens(t.set(&info.anonymous_patterns));
        let first = t.core_patterns.len();
        for (operands, variables) in &info.patterns {
            let operands = operands.iter().flat_map(|&(id, n)| [id as u32, n as u32]).collect();
            let operands = span_tokens(t.list(operands));
            let first_var = t.variables.len();
            for (v, wrap) in variables {
                let name = lit(t.opt_string(v.name.as_deref()));
                let (count, minimum) = (lit(v.count), lit(v.minimum));
                let default = lit(v.default.as_ref().map_or(NONE, |d| t.expr(d)));
                t.variables.push(quote! {
                    rt::VariableDesc { name: #name, count: #count, minimum: #minimum, default: #default, wrap: #wrap }
                });
            }
            let vars = span_tokens(Span::new(first_var as u32, (t.variables.len() - first_var) as u32));
            t.core_patterns.push(quote!(rt::CorePatternDesc { operands: #operands, variables: #vars }));
        }
        let patterns = span_tokens(Span::new(first as u32, (t.core_patterns.len() - first) as u32));
        t.cores.push(quote! {
            rt::CoreDesc {
                associative: #associative,
                max_optional_count: #max_optional,
                anonymous_patterns: #anonymous,
                patterns: #patterns,
            }
        });
    }

    /// The functions of automaton `u`; adds its static data to the tables.
    fn unit(&mut self, u: usize) -> Result<TokenStream, CompileError> {
        let info = std::mem::replace(
            &mut self.units[u].info,
            AutomatonInfo {
                states: Vec::new(),
                finals: Default::default(),
                patterns: Vec::new(),
                constraints: Vec::new(),
                constraint_vars: Default::default(),
            },
        );
        let result = self.unit_with(u, &info);
        self.units[u].info = info;
        result
    }

    fn unit_with(&mut self, u: usize, info: &AutomatonInfo) -> Result<TokenStream, CompileError> {
        let state_fn = |s: usize| format_ident!("a{}_s{}", u, s);
        let end_fn = format_ident!("a{}_end", u);
        let ul = lit(u);

        let mut fns = Vec::new();
        let mut end_arms = Vec::new();
        for (s, state) in info.states.iter().enumerate() {
            let this = state_fn(s);
            let mut op_arms = Vec::new();
            let mut atom_arms = Vec::new();
            let mut keyed_bodies = Vec::new();
            let mut any_op = Vec::new();
            let mut any = Vec::new();
            let mut next_sel = 3usize; // 0: no subject, 1: other operation, 2: other atom/expression
            for (key, ts) in &state.transitions {
                match key {
                    KeyInfo::End => {
                        let calls: Vec<TokenStream> = ts
                            .iter()
                            .map(|t| {
                                let id = self.tables.transition(t);
                                let next = state_fn(t.target);
                                quote!(rt::check_transition(m, tb(#ul), r, &T[#id], ::core::option::Option::None, #next, k)?;)
                            })
                            .collect();
                        let s_lit = lit(s);
                        end_arms.push(quote!(#s_lit => { #(#calls)* }));
                        continue;
                    }
                    KeyInfo::PatternId(_) => continue, // handled by the commutative dispatch below
                    _ => {}
                }
                let calls: Vec<TokenStream> = ts
                    .iter()
                    .map(|t| {
                        let id = self.tables.transition(t);
                        self.transition_call(u, t, &id)
                    })
                    .collect();
                match key {
                    KeyInfo::Operation(h) => {
                        let sel = lit(next_sel);
                        next_sel += 1;
                        let name = Literal::string(&h.name);
                        let hi = lit(self.tables.head(h));
                        op_arms.push(quote!(#name if DATA.head_is(head, #hi) => #sel,));
                        keyed_bodies.push(quote!(#sel => { #(#calls)* }));
                    }
                    KeyInfo::Atom(e) => {
                        let sel = lit(next_sel);
                        next_sel += 1;
                        let name = Literal::string(e.atom_name().expect("atom key"));
                        atom_arms.push(quote!(#name => #sel,));
                        keyed_bodies.push(quote!(#sel => { #(#calls)* }));
                    }
                    KeyInfo::AnyOperation => any_op.extend(calls),
                    KeyInfo::Any => any.extend(calls),
                    KeyInfo::End | KeyInfo::PatternId(_) => unreachable!(),
                }
            }
            let has_end = state.transitions.iter().any(|(k, _)| *k == KeyInfo::End);
            let report = if info.finals.contains(&s) || has_end {
                let s_lit = lit(s);
                quote!(if sel == 0 { k(r, #s_lit)?; })
            } else {
                quote!()
            };
            let keyed =
                if keyed_bodies.is_empty() { quote!() } else { quote!(match sel { #(#keyed_bodies)* _ => {} }) };
            let any_op = if any_op.is_empty() { quote!() } else { quote!(if is_operation { #(#any_op)* }) };
            fns.push(quote! {
                fn #this(m: &Matcher, r: &mut rt::Run, k: rt::K<'_>) -> rt::Flow {
                    let (sel, is_operation): (u32, bool) = match rt::front(r) {
                        rt::Front::None => (0, false),
                        rt::Front::Operation(head) => (match head.name.as_str() { #(#op_arms)* _ => 1 }, true),
                        rt::Front::Atom(name) => (match name { #(#atom_arms)* _ => 2 }, false),
                        rt::Front::Other => (2, false),
                    };
                    #report
                    #keyed
                    #any_op
                    #(#any)*
                    rt::CONTINUE
                }
            });

            // A commutative state: dispatch on the matched commutative pattern id.
            if let Some(&c) = self.core_of.get(&(u, s)) {
                let pid_fn = format_ident!("a{}_pid{}", u, s);
                let mut unions = Vec::new();
                let mut arms = Vec::new();
                for pid in 0..self.cores[c].info.patterns.len() {
                    let ts: Vec<&TransitionInfo> = state
                        .transitions
                        .iter()
                        .filter(|(k, _)| *k == KeyInfo::PatternId(pid))
                        .flat_map(|(_, ts)| ts)
                        .collect();
                    let union: BTreeSet<usize> = ts.iter().flat_map(|t| t.patterns.iter().copied()).collect();
                    unions.push(self.tables.set(&union));
                    let calls: Vec<TokenStream> = ts
                        .iter()
                        .map(|t| {
                            let id = self.tables.transition(t);
                            let next = state_fn(t.target);
                            quote! {
                                rt::check_transition(
                                    m, tb(#ul), r, &T[#id],
                                    ::core::option::Option::Some(::omnimatch::Replacement::Single(subject.clone())),
                                    #next, k,
                                )?;
                            }
                        })
                        .collect();
                    if !calls.is_empty() {
                        let pid = lit(pid);
                        arms.push(quote!(#pid => { #(#calls)* }));
                    }
                }
                self.cores[c].pid_sets = self.tables.spans(unions);
                fns.push(quote! {
                    fn #pid_fn(m: &Matcher, r: &mut rt::Run, pid: usize, subject: &Expr, k: rt::K<'_>) -> rt::Flow {
                        match pid {
                            #(#arms)*
                            _ => {}
                        }
                        rt::CONTINUE
                    }
                });
            }
        }

        fns.push(quote! {
            fn #end_fn(m: &Matcher, r: &mut rt::Run, state: usize, k: rt::K<'_>) -> rt::Flow {
                match state {
                    #(#end_arms)*
                    _ => {}
                }
                rt::CONTINUE
            }
        });

        // An operand automaton: its entry point for the commutative core.
        if let Some(c) = self.units[u].core {
            let operands_fn = format_ident!("a{}_operands", u);
            let root = state_fn(0);
            let c = lit(c);
            fns.push(quote! {
                fn #operands_fn(m: &Matcher, subject: ::core::option::Option<&Expr>) -> ::std::vec::Vec<(usize, Substitution)> {
                    rt::operand_matches(m, tb(#ul), #root, subject, m.core(#c).associative())
                }
            });
        }

        // The automaton's tables.
        let t = &mut self.tables;
        let mut constraint_ids = Vec::new();
        let mut constraint_patterns = Vec::new();
        for (c, patterns) in &info.constraints {
            constraint_ids.push(t.constraint(c.as_ref())?);
            constraint_patterns.push(t.set(patterns));
        }
        let constraints = span_tokens(t.list(constraint_ids));
        let constraint_patterns = span_tokens(t.spans(constraint_patterns));
        let mut triples = Vec::new();
        for (var, cs) in &info.constraint_vars {
            // Sorted by name: `BTreeMap<String, _>` orders like `str`.
            let name = t.string(var);
            let list = t.set(cs);
            triples.extend([name, list.off, list.len]);
        }
        let constraint_vars = span_tokens(t.list(triples));
        let first_tail = t.tails.len();
        for p in &info.patterns {
            let inverse: BTreeMap<&String, &String> = p.renaming.iter().map(|(o, r)| (r, o)).collect();
            let pairs = inverse.into_iter().flat_map(|(r, o)| [t.string(r), t.string(o)]).collect::<Vec<_>>();
            let inverse = span_tokens(t.list(pairs));
            let globals = if self.units[u].core.is_none() {
                p.pattern
                    .global_constraints()
                    .iter()
                    .map(|c| t.constraint(c.as_ref()))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            };
            let globals = span_tokens(t.list(globals));
            t.tails.push(quote!(rt::TailDesc { inverse_renaming: #inverse, global_constraints: #globals }));
        }
        let tails = span_tokens(Span::new(first_tail as u32, info.patterns.len() as u32));
        let pattern_count = lit(info.patterns.len());
        t.automata.push(quote! {
            rt::AutomatonDesc {
                pattern_count: #pattern_count,
                constraints: #constraints,
                constraint_patterns: #constraint_patterns,
                constraint_vars: #constraint_vars,
                tails: #tails,
            }
        });
        Ok(quote!(#(#fns)*))
    }

    /// The call taking transition `t` (index `id` of `T`) of automaton `u`.
    fn transition_call(&mut self, u: usize, t: &TransitionInfo, id: &Literal) -> TokenStream {
        let ul = lit(u);
        let next = format_ident!("a{}_s{}", u, t.target);
        match &t.label {
            LabelInfo::Epsilon => quote!(rt::epsilon(m, tb(#ul), r, &T[#id], #next, k)?;),
            LabelInfo::End => quote!(rt::consume(m, tb(#ul), r, &T[#id], #next, k)?;),
            LabelInfo::Expr(e) if e.is_wildcard() => {
                let e = lit(self.tables.expr(e));
                quote!(rt::wildcard(m, tb(#ul), r, &T[#id], #e, #next, k)?;)
            }
            LabelInfo::Expr(_) => quote!(rt::consume(m, tb(#ul), r, &T[#id], #next, k)?;),
            LabelInfo::Operation(head) => match self.core_of.get(&(u, t.target)) {
                Some(&c) => {
                    let operands_fn = format_ident!("a{}_operands", self.cores[c].operands);
                    let pid_fn = format_ident!("a{}_pid{}", u, t.target);
                    let pid_sets = format_ident!("PID_SETS_{}", c);
                    let c = lit(c);
                    quote!(rt::commutative_operation(m, tb(#ul), r, &T[#id], m.core(#c), #operands_fn, #pid_sets, #pid_fn, k)?;)
                }
                None => {
                    let h = lit(self.tables.head(head));
                    let end_fn = format_ident!("a{}_end", u);
                    quote!(rt::regular_operation(m, tb(#ul), r, &T[#id], #h, #next, #end_fn, k)?;)
                }
            },
        }
    }
}

/// `make_constraint(i)`, split into functions of at most [`CONSTRAINTS_PER_FN`] arms.
fn constraint_fns(constraints: &[TokenStream]) -> TokenStream {
    let ty = quote!(::std::sync::Arc<dyn ::omnimatch::Constraint>);
    let mut chunks = Vec::new();
    let mut dispatch = Vec::new();
    for (n, chunk) in constraints.chunks(CONSTRAINTS_PER_FN).enumerate() {
        let f = format_ident!("make_constraint_{}", n);
        let arms = chunk.iter().enumerate().map(|(j, c)| {
            let i = lit(n * CONSTRAINTS_PER_FN + j);
            quote!(#i => ::std::sync::Arc::new(#c),)
        });
        chunks.push(quote! {
            fn #f(i: u32) -> #ty {
                match i {
                    #(#arms)*
                    _ => unreachable!(),
                }
            }
        });
        let n_lit = lit(n);
        dispatch.push(quote!(#n_lit => #f(i),));
    }
    let per = lit(CONSTRAINTS_PER_FN);
    quote! {
        fn make_constraint(i: u32) -> #ty {
            match i / #per {
                #(#dispatch)*
                _ => unreachable!(),
            }
        }
        #(#chunks)*
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omnimatch::{make_dot_variable, Arity, CustomConstraint, FreeOf};
    use std::sync::Arc;

    fn f() -> Head {
        Head::new("f", Arity::Binary).unwrap()
    }

    fn compiler_with(constraint: Arc<dyn Constraint>) -> Compiler {
        let mut c = Compiler::new("M");
        let pattern = f().call(vec![make_dot_variable("x"), make_dot_variable("y")]);
        c.add(Pattern::with_constraints(pattern, vec![constraint]));
        c
    }

    #[test]
    fn rejects_invalid_names() {
        for name in ["", "_", "1abc", "a-b", "a b"] {
            assert_eq!(Compiler::new(name).generate(), Err(CompileError::InvalidName(name.to_owned())));
        }
        assert!(Compiler::new("Rules_2").generate().is_ok());
    }

    #[test]
    fn rejects_custom_constraints_without_a_path() {
        let c = compiler_with(Arc::new(CustomConstraint::new(["x"], |_: &[&Replacement]| true)));
        assert!(matches!(c.generate(), Err(CompileError::UnsupportedConstraint(_))));
    }

    #[test]
    fn rejects_an_invalid_rust_path() {
        let constraint = CustomConstraint::new(["x"], |_: &[&Replacement]| true).with_rust_path("not a ) path");
        let c = compiler_with(Arc::new(constraint));
        assert!(matches!(c.generate(), Err(CompileError::InvalidConstraintSource(_))));
    }

    #[test]
    fn accepts_built_in_constraints_and_named_functions() {
        let c = compiler_with(Arc::new(FreeOf::single("x", "a")));
        assert!(c.generate().unwrap().contains("FreeOf"));
        let constraint = CustomConstraint::new(["x"], |_: &[&Replacement]| true).with_rust_path("crate::check");
        let code = compiler_with(Arc::new(constraint)).generate().unwrap();
        assert!(code.contains("crate :: check"), "{code}");
    }

    #[test]
    fn output_is_deterministic() {
        let build = || {
            let mut c = Compiler::new("M");
            let fc = Head::builder("f_c").commutative(true).build().unwrap();
            for p in [
                f().call(vec![Expr::atom("a"), make_dot_variable("x")]),
                fc.call(vec![Expr::atom("a"), make_dot_variable("x"), Expr::star(Some("y"))]),
                Head::any(Some("F")).call(vec![make_dot_variable("x")]),
            ] {
                c.add(Pattern::new(p));
            }
            c.generate().unwrap()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn duplicate_patterns_share_an_index() {
        let mut c = Compiler::new("M");
        let p = Pattern::new(f().call(vec![Expr::atom("a"), make_dot_variable("x")]));
        assert_eq!(c.add(p.clone()), 0);
        assert_eq!(c.add(p), 0);
        assert_eq!(c.len(), 1);
    }
}
