//! Compile [`omnimatch`] many-to-one matchers to Rust code at build time.
//!
//! This replaces Python's `omnimatch.matching.code_generation`. Instead of
//! printing source text, the compiler turns the automaton of a
//! [`ManyToOneMatcher`] into Rust tokens: one
//! function per state, whose dispatch on the next subject is a `match` on head
//! and atom names, with pattern sets as constant bitsets. rustc (and LLVM)
//! compile and optimize it together with the [`omnimatch::compiled`] runtime,
//! which does the variable binding, backtracking and commutative (bipartite)
//! matching shared by all matchers.
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

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use omnimatch::compiled::{AutomatonInfo, KeyInfo, LabelInfo, TransitionInfo};
use omnimatch::utils::VariableWithCount;
use omnimatch::{Arity, Constraint, Expr, ExprKind, Head, ManyToOneMatcher, Pattern, Replacement, Substitution};
use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};

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
    Ok(Ident::new(name, Span::call_site()))
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
    /// The automaton and state it belongs to.
    owner: (usize, usize),
    operands: usize,
    info: omnimatch::compiled::CommutativeInfo,
}

struct Generator {
    units: Vec<Unit>,
    cores: Vec<Core>,
    /// (unit, state) -> core index.
    core_of: HashMap<(usize, usize), usize>,
    heads: Vec<Head>,
    head_index: HashMap<Head, usize>,
    exprs: Vec<Expr>,
    expr_index: HashMap<Expr, usize>,
}

fn u64_lit(x: u64) -> Literal {
    Literal::u64_suffixed(x)
}

fn usize_lit(x: usize) -> Literal {
    Literal::usize_suffixed(x)
}

fn str_lit(s: &str) -> Literal {
    Literal::string(s)
}

fn option_str(s: Option<&str>) -> TokenStream {
    match s {
        Some(s) => {
            let s = str_lit(s);
            quote!(::core::option::Option::Some(#s))
        }
        None => quote!(::core::option::Option::None),
    }
}

impl Generator {
    fn new(top: AutomatonInfo) -> Self {
        let mut g = Generator {
            units: Vec::new(),
            cores: Vec::new(),
            core_of: HashMap::new(),
            heads: Vec::new(),
            head_index: HashMap::new(),
            exprs: Vec::new(),
            expr_index: HashMap::new(),
        };
        g.add_unit(top, None);
        g
    }

    fn add_unit(&mut self, info: AutomatonInfo, core: Option<usize>) -> usize {
        let unit = self.units.len();
        let commutative: Vec<(usize, omnimatch::compiled::CommutativeInfo)> =
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
            self.cores.push(Core { owner: (unit, state), operands: usize::MAX, info: c });
            self.core_of.insert((unit, state), core_index);
            let operands = self.add_unit(operand_info, Some(core_index));
            self.cores[core_index].operands = operands;
        }
        unit
    }

    fn head(&mut self, head: &Head) -> usize {
        if let Some(&i) = self.head_index.get(head) {
            // `Head` equality ignores `infix`; keep the first one seen.
            return i;
        }
        let i = self.heads.len();
        self.heads.push(head.clone());
        self.head_index.insert(head.clone(), i);
        i
    }

    fn expr(&mut self, expr: &Expr) -> usize {
        if let Some(&i) = self.expr_index.get(expr) {
            return i;
        }
        self.intern_heads(expr);
        let i = self.exprs.len();
        self.exprs.push(expr.clone());
        self.expr_index.insert(expr.clone(), i);
        i
    }

    fn intern_heads(&mut self, expr: &Expr) {
        match expr.kind() {
            ExprKind::Operation { head, operands } => {
                self.head(head);
                for o in operands {
                    self.intern_heads(o);
                }
            }
            ExprKind::Wildcard(w) => {
                if let Some(d) = &w.default {
                    self.intern_heads(d);
                }
            }
            ExprKind::Atom { .. } => {}
        }
    }

    // ── value constructors (inside `new()`, where `heads` is in scope) ──

    fn head_tokens(head: &Head) -> TokenStream {
        let name = str_lit(&head.name);
        let arity = format_ident!(
            "{}",
            match head.arity {
                Arity::Nullary => "Nullary",
                Arity::Unary => "Unary",
                Arity::Binary => "Binary",
                Arity::Ternary => "Ternary",
                Arity::Variadic => "Variadic",
            }
        );
        let (c, a, o, i) = (head.commutative, head.associative, head.one_identity, head.infix);
        let wildcard = if head.wildcard {
            let var = option_str(head.head_variable.as_deref());
            quote!(.wildcard(#var))
        } else {
            quote!()
        };
        quote! {
            ::omnimatch::Head::builder(#name)
                .arity(::omnimatch::Arity::#arity)
                .commutative(#c)
                .associative(#a)
                .one_identity(#o)
                .infix(#i)
                #wildcard
                .build_unchecked()
        }
    }

    fn expr_tokens(&self, expr: &Expr) -> TokenStream {
        let var = option_str(expr.variable_name());
        match expr.kind() {
            ExprKind::Atom { name } => {
                let name = str_lit(name);
                match expr.variable_name() {
                    None => quote!(::omnimatch::Expr::atom(#name)),
                    Some(_) => quote!(::omnimatch::Expr::atom(#name).with_variable_name(#var)),
                }
            }
            ExprKind::Wildcard(w) => {
                let min = usize_lit(w.min_count);
                let fixed = w.fixed_size;
                let default = match &w.default {
                    Some(d) => {
                        let d = self.expr_tokens(d);
                        quote!(::core::option::Option::Some(#d))
                    }
                    None => quote!(::core::option::Option::None),
                };
                quote!(::omnimatch::Expr::wildcard(#min, #fixed, #var, #default).unwrap())
            }
            ExprKind::Operation { head, operands } => {
                let h = usize_lit(self.head_index[head]);
                let operands = operands.iter().map(|o| self.expr_tokens(o));
                quote!(::omnimatch::Expr::try_operation(heads[#h].clone(), vec![#(#operands),*], #var).unwrap())
            }
        }
    }

    fn replacement_tokens(&self, r: &Replacement) -> TokenStream {
        match r {
            Replacement::Single(e) => {
                let e = self.expr_tokens(e);
                quote!(::omnimatch::Replacement::Single(#e))
            }
            Replacement::Sequence(es) => {
                let es = es.iter().map(|e| self.expr_tokens(e));
                quote!(::omnimatch::Replacement::Sequence(vec![#(#es),*]))
            }
            Replacement::Multiset(ms) => {
                let es = ms.iter().map(|e| self.expr_tokens(e));
                quote!(::omnimatch::Replacement::Multiset(vec![#(#es),*].into_iter().collect()))
            }
            Replacement::Head(h) => {
                let h = usize_lit(self.head_index[h]);
                quote!(::omnimatch::Replacement::Head(heads[#h].clone()))
            }
        }
    }

    fn substitution_tokens(&self, s: &Substitution) -> TokenStream {
        let inserts = s.iter().map(|(name, value)| {
            let name = str_lit(name);
            let value = self.replacement_tokens(value);
            quote!(s.insert(#name, #value);)
        });
        quote!({
            let mut s = ::omnimatch::Substitution::new();
            #(#inserts)*
            s
        })
    }

    fn constraint_tokens(c: &dyn Constraint) -> Result<TokenStream, CompileError> {
        let source = c.rust_source().ok_or_else(|| CompileError::UnsupportedConstraint(format!("{c:?}")))?;
        let tokens: TokenStream =
            source.parse().map_err(|e| CompileError::InvalidConstraintSource(format!("{source}: {e}")))?;
        Ok(quote!(::std::sync::Arc::new(#tokens) as ::std::sync::Arc<dyn ::omnimatch::Constraint>))
    }

    fn intern_values(&mut self, sub: &Substitution) {
        for (_, value) in sub {
            match value {
                Replacement::Single(e) => self.intern_heads(e),
                Replacement::Sequence(es) => es.iter().for_each(|e| self.intern_heads(e)),
                Replacement::Multiset(ms) => ms.iter().for_each(|e| self.intern_heads(e)),
                Replacement::Head(h) => {
                    self.head(h);
                }
            }
        }
    }

    fn generate(mut self, name: &Ident) -> Result<TokenStream, CompileError> {
        // Intern every head and expression referenced by labels, keys and bindings first.
        for u in 0..self.units.len() {
            let states = self.units[u].info.states.clone();
            for state in &states {
                for (key, ts) in &state.transitions {
                    match key {
                        KeyInfo::Operation(h) => {
                            self.head(h);
                        }
                        KeyInfo::Atom(e) => self.intern_heads(e),
                        _ => {}
                    }
                    for t in ts {
                        match &t.label {
                            LabelInfo::Expr(e) => {
                                self.expr(e);
                            }
                            LabelInfo::Operation(h) => {
                                self.head(h);
                            }
                            _ => {}
                        }
                        if let Some(s) = &t.subst {
                            self.intern_values(s);
                        }
                    }
                }
            }
        }
        for c in 0..self.cores.len() {
            if let Some(h) = self.cores[c].info.associative.clone() {
                self.head(&h);
            }
            for (_, vars) in self.cores[c].info.patterns.clone() {
                for (v, _) in vars {
                    if let Some(d) = &v.default {
                        self.intern_heads(d);
                    }
                }
            }
        }

        let mut items = Vec::new();
        let mut table_fields = Vec::new();
        let mut table_inits = Vec::new();
        for u in 0..self.units.len() {
            let (unit_items, init) = self.unit(u)?;
            items.push(unit_items);
            let field = format_ident!("a{}", u);
            table_fields.push(quote!(#field: rt::Tables));
            table_inits.push(quote!(#field: #init));
        }
        let mut core_fields = Vec::new();
        let mut core_inits = Vec::new();
        let mut core_clears = Vec::new();
        for (c, core) in self.cores.iter().enumerate() {
            let field = format_ident!("c{}", c);
            core_fields.push(quote!(#field: rt::CommutativeCore));
            core_clears.push(quote!(self.#field.clear();));
            let associative = match &core.info.associative {
                Some(h) => {
                    let h = usize_lit(self.head_index[h]);
                    quote!(::core::option::Option::Some(heads[#h].clone()))
                }
                None => quote!(::core::option::Option::None),
            };
            let max_optional = usize_lit(core.info.max_optional_count);
            let anonymous = core.info.anonymous_patterns.iter().map(|&i| usize_lit(i));
            let patterns = core.info.patterns.iter().map(|(set, vars)| {
                let set = set.iter().map(|&(i, n)| {
                    let (i, n) = (usize_lit(i), usize_lit(n));
                    quote!((#i, #n))
                });
                let vars = vars.iter().map(|(v, wrap)| self.variable_with_count(v, *wrap));
                quote!((vec![#(#set),*], vec![#(#vars),*]))
            });
            core_inits.push(quote! {
                #field: rt::CommutativeCore::new(#associative, #max_optional, { let v: ::std::vec::Vec<usize> = vec![#(#anonymous),*]; v }, vec![#(#patterns),*])
            });
        }

        let head_inits = self.heads.iter().map(Self::head_tokens);
        let expr_inits: Vec<TokenStream> = self.exprs.iter().map(|e| self.expr_tokens(e)).collect();
        let module = format_ident!("__omnimatch_{}", snake_case(&name.to_string()));
        let pattern_count = usize_lit(self.units[0].info.patterns.len());

        Ok(quote! {
            #[allow(unused, non_snake_case, clippy::all, clippy::pedantic)]
            mod #module {
                use ::omnimatch::compiled as rt;
                use ::omnimatch::{Expr, Substitution};

                /// A many-to-one matcher compiled by `omnimatch-compile`.
                pub struct Matcher {
                    heads: ::std::vec::Vec<::omnimatch::Head>,
                    exprs: ::std::vec::Vec<Expr>,
                    #(#table_fields,)*
                    #(#core_fields,)*
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
                        let heads: ::std::vec::Vec<::omnimatch::Head> = vec![#(#head_inits),*];
                        let exprs: ::std::vec::Vec<Expr> = vec![#(#expr_inits),*];
                        Matcher {
                            #(#table_inits,)*
                            #(#core_inits,)*
                            heads,
                            exprs,
                        }
                    }

                    /// Drop the operands cached by the commutative matchers.
                    pub fn clear(&self) {
                        #(#core_clears)*
                    }

                    /// Call `on_group` with the matches reaching each final state
                    /// (patterns differing only in constraints are grouped).
                    pub fn for_each_group(
                        &self,
                        subject: &Expr,
                        on_group: &mut dyn FnMut(::std::vec::Vec<(usize, Substitution)>) -> rt::Flow,
                    ) -> rt::Flow {
                        rt::for_each_group(self, &self.a0, a0_s0, subject, on_group)
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

    fn variable_with_count(&self, v: &VariableWithCount, wrap: bool) -> TokenStream {
        let name = option_str(v.name.as_deref());
        let (count, minimum) = (usize_lit(v.count), usize_lit(v.minimum));
        let default = match &v.default {
            Some(d) => {
                let d = self.expr_tokens(d);
                quote!(::core::option::Option::Some(#d))
            }
            None => quote!(::core::option::Option::None),
        };
        quote!((::omnimatch::utils::VariableWithCount::new(#name, #count, #minimum, #default), #wrap))
    }

    /// The items of one automaton, and the expression building its `Tables`.
    fn unit(&self, u: usize) -> Result<(TokenStream, TokenStream), CompileError> {
        let info = &self.units[u].info;
        let n_words = (info.patterns.len() + 63) / 64;
        let bits = |set: &std::collections::BTreeSet<usize>| -> TokenStream {
            let mut words = vec![0u64; n_words];
            for &p in set {
                words[p / 64] |= 1 << (p % 64);
            }
            let words = words.into_iter().map(u64_lit);
            quote!(&[#(#words),*])
        };
        let tables = format_ident!("a{}", u);
        let state_fn = |s: usize| format_ident!("a{}_s{}", u, s);
        let end_fn = format_ident!("a{}_end", u);

        let mut statics = Vec::new();
        let mut substs: Vec<TokenStream> = Vec::new();
        let mut transition_count = 0usize;
        // Emit the static data of a transition and return its name.
        let mut transition = |t: &TransitionInfo, statics: &mut Vec<TokenStream>, substs: &mut Vec<TokenStream>| {
            let ident = format_ident!("A{}_T{}", u, transition_count);
            transition_count += 1;
            let patterns = bits(&t.patterns);
            let variable = option_str(t.variable_name.as_deref());
            let constraints = t.check_constraints.iter().map(|&c| usize_lit(c));
            let subst = match &t.subst {
                Some(s) => {
                    let i = usize_lit(substs.len());
                    substs.push(self.substitution_tokens(s));
                    quote!(::core::option::Option::Some(#i))
                }
                None => quote!(::core::option::Option::None),
            };
            statics.push(quote! {
                static #ident: rt::Transition = rt::Transition {
                    patterns: #patterns,
                    variable: #variable,
                    constraints: &[#(#constraints),*],
                    subst: #subst,
                };
            });
            ident
        };

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
                        let calls = ts.iter().map(|t| {
                            let ident = transition(t, &mut statics, &mut substs);
                            let next = state_fn(t.target);
                            quote!(rt::check_transition(m, &m.#tables, r, &#ident, ::core::option::Option::None, #next, k)?;)
                        });
                        let s_lit = usize_lit(s);
                        let calls: Vec<TokenStream> = calls.collect();
                        end_arms.push(quote!(#s_lit => { #(#calls)* }));
                        continue;
                    }
                    KeyInfo::PatternId(_) => continue, // handled by the commutative dispatch below
                    _ => {}
                }
                let calls: Vec<TokenStream> = ts
                    .iter()
                    .map(|t| {
                        let ident = transition(t, &mut statics, &mut substs);
                        self.transition_call(u, t, &ident)
                    })
                    .collect();
                match key {
                    KeyInfo::Operation(h) => {
                        let sel = usize_lit(next_sel);
                        next_sel += 1;
                        let name = str_lit(&h.name);
                        let hi = usize_lit(self.head_index[h]);
                        op_arms.push(quote!(#name if head == &m.heads[#hi] => #sel,));
                        keyed_bodies.push(quote!(#sel => { #(#calls)* }));
                    }
                    KeyInfo::Atom(e) => {
                        let sel = usize_lit(next_sel);
                        next_sel += 1;
                        let name = str_lit(e.atom_name().expect("atom key"));
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
                let s_lit = usize_lit(s);
                quote!(if sel == 0 { k(r, #s_lit)?; })
            } else {
                quote!()
            };
            let keyed =
                if keyed_bodies.is_empty() { quote!() } else { quote!(match sel { #(#keyed_bodies)* _ => {} }) };
            let any_op = if any_op.is_empty() { quote!() } else { quote!(if is_operation { #(#any_op)* }) };
            fns.push(quote! {
                fn #this(m: &Matcher, r: &mut rt::Run, k: rt::K<'_>) -> rt::Flow {
                    let (sel, is_operation): (usize, bool) = match rt::front(r) {
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
                let core = &self.cores[c];
                let pid_fn = format_ident!("a{}_pid{}", u, s);
                let pid_static = format_ident!("A{}_PID{}", u, s);
                let mut unions = Vec::new();
                let mut arms = Vec::new();
                for pid in 0..core.info.patterns.len() {
                    let ts: Vec<&TransitionInfo> = state
                        .transitions
                        .iter()
                        .filter(|(k, _)| *k == KeyInfo::PatternId(pid))
                        .flat_map(|(_, ts)| ts)
                        .collect();
                    let union: std::collections::BTreeSet<usize> =
                        ts.iter().flat_map(|t| t.patterns.iter().copied()).collect();
                    unions.push(bits(&union));
                    let calls: Vec<TokenStream> = ts
                        .iter()
                        .map(|t| {
                            let ident = transition(t, &mut statics, &mut substs);
                            let next = state_fn(t.target);
                            quote! {
                                rt::check_transition(
                                    m, &m.#tables, r, &#ident,
                                    ::core::option::Option::Some(::omnimatch::Replacement::Single(subject.clone())),
                                    #next, k,
                                )?;
                            }
                        })
                        .collect();
                    let pid = usize_lit(pid);
                    arms.push(quote!(#pid => { #(#calls)* }));
                }
                fns.push(quote! {
                    static #pid_static: &[&[u64]] = &[#(#unions),*];
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
            let core_field = format_ident!("c{}", c);
            let root = state_fn(0);
            fns.push(quote! {
                fn #operands_fn(m: &Matcher, subject: ::core::option::Option<&Expr>) -> ::std::vec::Vec<(usize, Substitution)> {
                    rt::operand_matches(m, &m.#tables, #root, subject, m.#core_field.associative())
                }
            });
        }

        // The tables.
        let constraints =
            info.constraints.iter().map(|(c, _)| Self::constraint_tokens(c.as_ref())).collect::<Result<Vec<_>, _>>()?;
        let constraint_patterns = info.constraints.iter().map(|(_, ps)| bits(ps));
        let constraint_vars = info.constraint_vars.iter().map(|(v, cs)| {
            let v = str_lit(v);
            let cs = cs.iter().map(|&c| usize_lit(c));
            quote!((#v, &[#(#cs),*] as &'static [usize]))
        });
        let tails = info
            .patterns
            .iter()
            .map(|p| {
                let inverse: BTreeMap<&String, &String> = p.renaming.iter().map(|(o, r)| (r, o)).collect();
                let inverse = inverse.into_iter().map(|(r, o)| {
                    let (r, o) = (str_lit(r), str_lit(o));
                    quote!((#r.to_owned(), #o.to_owned()))
                });
                let globals = if self.units[u].core.is_none() {
                    p.pattern
                        .global_constraints()
                        .iter()
                        .map(|c| Self::constraint_tokens(c.as_ref()))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    Vec::new()
                };
                Ok(quote! {
                    rt::PatternTail {
                        inverse_renaming: vec![#(#inverse),*].into_iter().collect(),
                        global_constraints: vec![#(#globals),*],
                    }
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;
        let pattern_count = usize_lit(info.patterns.len());
        let init = quote! {
            rt::Tables {
                pattern_count: #pattern_count,
                substs: vec![#(#substs),*],
                constraints: vec![#(#constraints),*],
                constraint_patterns: vec![#(#constraint_patterns),*],
                constraint_vars: vec![#(#constraint_vars),*].into_iter().collect(),
                patterns: vec![#(#tails),*],
            }
        };
        Ok((quote!(#(#statics)* #(#fns)*), init))
    }

    /// The call taking transition `t` (static data `ident`) of automaton `u`.
    fn transition_call(&self, u: usize, t: &TransitionInfo, ident: &Ident) -> TokenStream {
        let tables = format_ident!("a{}", u);
        let next = format_ident!("a{}_s{}", u, t.target);
        match &t.label {
            LabelInfo::Epsilon => quote!(rt::epsilon(m, &m.#tables, r, &#ident, #next, k)?;),
            LabelInfo::End => quote!(rt::consume(m, &m.#tables, r, &#ident, #next, k)?;),
            LabelInfo::Expr(e) if e.is_wildcard() => {
                let e = usize_lit(self.expr_index[e]);
                quote!(rt::wildcard(m, &m.#tables, r, &#ident, &m.exprs[#e], #next, k)?;)
            }
            LabelInfo::Expr(_) => quote!(rt::consume(m, &m.#tables, r, &#ident, #next, k)?;),
            LabelInfo::Operation(head) => match self.core_of.get(&(u, t.target)) {
                Some(&c) => {
                    let core = &self.cores[c];
                    debug_assert_eq!(core.owner, (u, t.target));
                    let core_field = format_ident!("c{}", c);
                    let operands_fn = format_ident!("a{}_operands", core.operands);
                    let pid_fn = format_ident!("a{}_pid{}", u, t.target);
                    let pid_static = format_ident!("A{}_PID{}", u, t.target);
                    quote!(rt::commutative_operation(m, &m.#tables, r, &#ident, &m.#core_field, #operands_fn, #pid_static, #pid_fn, k)?;)
                }
                None => {
                    let h = usize_lit(self.head_index[head]);
                    let end_fn = format_ident!("a{}_end", u);
                    quote!(rt::regular_operation(m, &m.#tables, r, &#ident, &m.heads[#h], #next, #end_fn, k)?;)
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omnimatch::{make_dot_variable, CustomConstraint, FreeOf};
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
