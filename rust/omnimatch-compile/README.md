# omnimatch-compile

Compiles an `omnimatch` many-to-one matcher into Rust code at build time. It
replaces the Python package's `matching/code_generation.py`, which printed
Python source for a matcher.

This crate does not print source text for people to read. It reads the automaton that
`ManyToOneMatcher` builds and emits Rust tokens: one function per state.
Each function's dispatch on the next subject is a `match` on head and atom
names, and the sets of still-viable patterns are constant bitsets. rustc and
LLVM then compile that code together with `omnimatch::compiled`, the small
runtime that does the work every matcher shares: binding variables and
undoing them on backtracking, checking constraints, splitting sequences, and
matching the operands of commutative operations through bipartite matchings.
Nothing is interpreted when the matcher runs, and no LLVM is needed at that point.

## Usage

```toml
# Cargo.toml
[dependencies]
omnimatch = "0.0.3"

[build-dependencies]
omnimatch = "0.0.3"
omnimatch-compile = "0.0.3"
```

```rust
// build.rs
use omnimatch::*;
use omnimatch_compile::Compiler;

fn main() {
    let f = Head::new("f", Arity::Binary).unwrap();
    let mut compiler = Compiler::new("RuleMatcher");
    compiler.add(Pattern::new(f.call(vec![Expr::atom("a"), make_dot_variable("x")]))); // index 0
    compiler.add(Pattern::new(f.call(vec![make_dot_variable("x"), Expr::atom("b")]))); // index 1
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("rules.rs");
    compiler.write(out).unwrap();
}
```

```rust
// src/lib.rs
include!(concat!(env!("OUT_DIR"), "/rules.rs"));

fn demo(subject: &omnimatch::Expr) {
    let matcher = RuleMatcher::new();
    for (pattern_index, substitution) in matcher.match_all(subject) {
        println!("pattern {pattern_index} matched with {substitution}");
    }
}
```

The generated type has the same methods as `ManyToOneMatcher`:
`match_all`, `first_match`, `is_match`, `for_each_match`, `for_each_group`
and `clear`. Matches are reported with the index that `Compiler::add`
returned, and come in the same order as from the interpreted matcher.

## Constraints

The generated code rebuilds each constraint from `Constraint::rust_source`:

* `EqualVariablesConstraint` and `FreeOf` work as they are.
* A `CustomConstraint` holds a closure, which cannot be moved into generated
  code. Name an equivalent function with `with_rust_path`. The path is
  resolved where the generated file is included:

  ```rust
  pub fn distinct(args: &[&Replacement]) -> bool {
      args[0] != args[1]
  }
  let c = CustomConstraint::new(["x", "y"], distinct).with_rust_path("crate::distinct");
  ```

* Your own `Constraint` types can implement `rust_source`.

`Compiler::generate` and `write` return `CompileError::UnsupportedConstraint`
for a constraint they cannot rebuild.

## Performance

`cargo run --release -p omnimatch-compile-tests --example bench` times both
matchers on the test corpus (70 patterns, 1251 subjects). On the development
machine, the compiled matcher was about 1.5–1.8× faster overall. On
subjects without commutative operations it was about 2.5×, because there the
automaton's control flow is all the work. Commutative operations spend most of
their time in the bipartite matching, which both matchers share.

## Tests

`compile-tests/` (crate `omnimatch-compile-tests`, not published) compiles a
matcher from `src/cases.rs` in its build script. Its tests check that the
compiled matcher finds exactly the same matches, in the same order, as the
interpreted `ManyToOneMatcher` on more than 1200 subjects.
