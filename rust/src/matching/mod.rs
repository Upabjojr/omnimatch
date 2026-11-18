//! Pattern matching algorithms.
//!
//! Ported so far: one-to-one matching ([`one_to_one`]), bipartite graphs and
//! maximum-matching enumeration ([`bipartite`], [`hopcroft_karp`]), and
//! many-to-one matching ([`many_to_one`]).
//! Not yet ported: code generation and JSON serialization.

pub mod bipartite;
pub mod common;
pub mod hopcroft_karp;
pub mod many_to_one;
pub mod one_to_one;
