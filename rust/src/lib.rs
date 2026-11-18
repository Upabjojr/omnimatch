//! Pattern matching on symbolic expressions — Rust port of the
//! [omnimatch](https://pypi.org/project/omnimatch/) Python package.
//!
//! ```
//! use omnimatch::*;
//!
//! let f = Head::new("f", Arity::Binary).unwrap();
//! let a = Expr::atom("a");
//! let pattern = Pattern::new(f.call(vec![a.clone(), make_dot_variable("x")]));
//! let subject = f.call(vec![a.clone(), Expr::atom("b")]);
//! let matches = match_pattern(&subject, &pattern).unwrap();
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].to_string(), "{x ↦ b}");
//! ```
//!
//! See `README.md` in the crate directory for which parts of the Python
//! package have been ported.

pub mod constraints;
pub mod expressions;
pub mod functions;
pub mod matching;
pub mod multiset;
pub mod substitution;
pub mod utils;

pub use constraints::{Constraint, CustomConstraint, EqualVariablesConstraint, FreeOf};
pub use expressions::{
    make_dot_variable, make_plus_variable, make_star_variable, Arity, Expr, ExprError, ExprKind, Head, HeadBuilder,
    OperationHead, Pattern, WildcardSpec,
};
pub use functions::{replace, replace_all, replace_all_post_order, substitute, ReplacementRule, RewriteError};
pub use matching::one_to_one::{first_match, for_each_match, is_match, match_anywhere, match_pattern, MatchError};
pub use multiset::Multiset;
pub use substitution::{Conflict, Replacement, Substitution};
