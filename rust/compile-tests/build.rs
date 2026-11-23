//! Compiles the matchers tested by this crate.

#[path = "src/cases.rs"]
#[allow(dead_code)]
mod cases;

use std::path::PathBuf;

use omnimatch_compile::Compiler;

fn main() {
    println!("cargo:rerun-if-changed=src/cases.rs");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut corpus = Compiler::new("CorpusMatcher");
    for pattern in cases::patterns() {
        corpus.add(pattern);
    }
    corpus.write(out.join("corpus.rs")).unwrap();

    Compiler::new("EmptyMatcher").write(out.join("empty.rs")).unwrap();
}
