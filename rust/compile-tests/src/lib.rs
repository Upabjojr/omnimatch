//! Matchers compiled by `build.rs` with `omnimatch-compile`; `tests/`
//! compares them with the interpreted `ManyToOneMatcher`.

pub mod cases;
pub mod rules;

include!(concat!(env!("OUT_DIR"), "/corpus.rs"));
include!(concat!(env!("OUT_DIR"), "/empty.rs"));
include!(concat!(env!("OUT_DIR"), "/rule_set.rs"));
