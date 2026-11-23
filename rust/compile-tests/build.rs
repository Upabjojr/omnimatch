//! Compiles the matchers tested by this crate.

#[path = "src/cases.rs"]
#[allow(dead_code)]
mod cases;
#[path = "src/rules.rs"]
#[allow(dead_code)]
mod rules;

use std::path::PathBuf;

use omnimatch_compile::Compiler;

fn main() {
    println!("cargo:rerun-if-changed=src/cases.rs");
    println!("cargo:rerun-if-changed=src/rules.rs");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut corpus = Compiler::new("CorpusMatcher");
    for pattern in cases::patterns() {
        corpus.add(pattern);
    }
    corpus.write(out.join("corpus.rs")).unwrap();

    let mut rule_set = Compiler::new("RuleSetMatcher");
    for pattern in rules::rules(rules::RULE_COUNT) {
        rule_set.add(pattern);
    }
    rule_set.write(out.join("rule_set.rs")).unwrap();

    Compiler::new("EmptyMatcher").write(out.join("empty.rs")).unwrap();
}
