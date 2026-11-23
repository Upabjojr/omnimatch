# omnimatch-compile

Compiles an `omnimatch` many-to-one matcher into Rust code at build time. It
replaces the Python package's `matching/code_generation.py`, which printed
Python source for a matcher.

This crate does not print source text for people to read. It reads the automaton that
`ManyToOneMatcher` builds and emits Rust tokens: one function per state.
Each function's dispatch on the next subject is a `match` on head and atom
names. Everything else the matcher knows goes into `static` tables of
integers (see [Memory](#memory)). rustc and LLVM then compile that code
together with `omnimatch::compiled`, the small runtime that does the work
every matcher shares: binding variables and undoing them on backtracking,
checking constraints, splitting sequences, and matching the operands of
commutative operations through bipartite matchings. No LLVM is needed when
the matcher runs.

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

## Memory

A compiled matcher keeps its rules out of the heap. The sets of viable
patterns (sorted index lists), the variable bindings, the expressions, the
constraints' parameters and the commutative patterns are `static` arrays of
plain integers that refer to each other by index. They contain no pointers,
so the loader applies no relocations to them: they stay in the binary's
read-only pages. The OS reads those pages from the file when they are first
touched, can drop them again under memory pressure, and shares them between
processes running the same binary.

Some values must be real objects: the expressions used as labels or
bindings, the constraints, and the commutative matchers. Each one is built
from the tables the first time a match needs it. Expressions and
constraints are built once per program; commutative matchers once per
`Matcher`, since they cache operand matches (`clear()` empties the caches).
`new()` allocates only one empty cell per commutative matcher.

Measured on a synthetic set of integration rules in the style of Rubi
(products and sums that are commutative and associative), matching an
instance of every tenth rule:

| Rules | Build (rustc, release) | Interpreted heap | Compiled heap after matching | Compiled read-only data |
|---|---|---|---|---|
| 2,000 | 33 s, 0.6 GB | 6.4 MB | 0.5 MB | 0.3 MB |
| 5,000 | 46 s, 1.2 GB | 14.7 MB | 0.9 MB | 0.6 MB |
| 8,500 | 72 s, 1.8 GB | 24.1 MB | 1.2 MB | 0.9 MB |

Most of the compiled heap is the commutative matchers' caches. The
compiled matcher was also 2.1–2.5× faster on these rules. Before this
layout, the generated `new()` built all of it on the heap, and rustc ran
out of memory (over 16 GB) compiling 2,000 rules.

`compile-tests` compiles 1,000 such rules and checks that the matches are
the same as the interpreted matcher's, and that what remains after `clear()`
is less than a tenth of the interpreted matcher's heap.

## Performance

`cargo run --release -p omnimatch-compile-tests --example bench` times both
matchers on the test corpus (70 patterns, 1251 subjects). On the development
machine, the compiled matcher was about 1.6–1.8× faster overall. On
subjects without commutative operations it was about 2.2×, because there the
automaton's control flow is all the work. Commutative operations spend most of
their time in the bipartite matching, which both matchers share.

## Tests

`compile-tests/` (crate `omnimatch-compile-tests`, not published) compiles
matchers in its build script: one from `src/cases.rs`, whose 70 patterns
cover each kind of wildcard, head and constraint, and the 1,000 rules of
`src/rules.rs`. Its tests check that the compiled matchers find exactly the
same matches as the interpreted `ManyToOneMatcher` (for `cases.rs`, in the
same order, on more than 1,200 subjects), and the heap use described above.
