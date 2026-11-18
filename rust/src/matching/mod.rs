//! Pattern matching algorithms.
//!
//! Ported so far: one-to-one matching ([`one_to_one`]), bipartite graphs and
//! maximum-matching enumeration ([`bipartite`], [`hopcroft_karp`]).
//! Not yet ported: many-to-one matching, code generation and JSON serialization.

pub mod bipartite;
pub mod common;
pub mod hopcroft_karp;
pub mod one_to_one;
