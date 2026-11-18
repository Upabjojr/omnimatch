# omnimatch (Rust port)

A work-in-progress Rust port of the `omnimatch` Python package in the parent
directory. It has no dependencies (the Python `multiset` package is replaced by
`src/multiset.rs`).

```
cargo test      # unit tests, ported one-to-one matching tests, doctests
cargo clippy --all-targets
```

## Status

| Python module                           | Rust                                | Status |
|-----------------------------------------|-------------------------------------|--------|
| `expressions/expressions.py`            | `src/expressions.rs`                | ported (`WildcardOperationHead` is `Head::any`), except `SymbolWrapper` and `to_/from_omnimatch_expression` |
| `expressions/substitution.py`           | `src/substitution.rs`               | ported |
| `expressions/constraints.py`            | `src/constraints.rs`                | ported (`EqualVariablesConstraint`, `CustomConstraint`, `FreeOf`) |
| `expressions/functions.py`              | methods on `Expr`, `matching/common.rs` | ported |
| `utils.py` (combinatorics)              | `src/utils.rs`                      | ported; Python-only helpers (lambda source, cached properties, `generator_chain`) dropped |
| `matching/_common.py`                   | `src/matching/common.rs`            | ported |
| `matching/one_to_one.py`                | `src/matching/one_to_one.rs`        | ported |
| `functions.py`                          | `src/functions.rs`                  | ported (`sort_key` is `substitute_sorted_by`) |
| `matching/hopcroft_karp.py`            | `src/matching/hopcroft_karp.rs`     | ported (same matchings as Python on the Python test cases) |
| `matching/bipartite.py`                 | `src/matching/bipartite.rs`         | ported, except the graphviz helpers |
| `matching/many_to_one.py`               | `src/matching/many_to_one.rs`       | ported (`ManyToOneMatcher`, `ManyToOneReplacer`, `CommutativeMatcher`, wildcard heads), except graphviz helpers and the unused `SecondaryAutomaton` |
| `matching/code_generation.py`           | –                                   | not yet; may not make sense in Rust |
| `matching/json_serialization.py`        | –                                   | not yet (would add a `serde` dependency) |
| `_typed.py`                             | –                                   | not needed |

`tests/matching.rs` ports the tables from `tests/test_matching.py` (constant,
commutative, dot, optional, associative, star/plus/mixed sequence wildcards,
one-identity, variable names, constraints) and `tests/test_matching_many_to_one.py`.
As in Python's `conftest.py`, every table runs through both matchers, and
`many_to_one_all_patterns_at_once` builds one matcher per subject holding all
of that subject's patterns (Python: `test_many_to_one`). `tests/wildcard_head.rs`
and `tests/functions.rs` port `test_wildcard_operation_head.py` and
`test_functions.py` (including the logic-simplification rewrite system, run
through both `replace_all` and `ManyToOneReplacer`).

## Design

* **Expressions.** `Expr` is an immutable, reference-counted node whose
  `ExprKind` is `Atom`, `Wildcard` or `Operation`. Operation metadata is an
  `OperationHead` shared through `Head`. As in Python there are two
  constructors: `Expr::try_operation` (raw: flattens associative operands and
  sorts commutative ones) and `Head::call` (the factory, which also applies
  `one_identity`). `Head::builder(..).build()` does the validation that
  `Operation.new` does.
* **Ordering.** `Ord for Expr` is Python's `__lt__` (atoms < wildcards <
  operations, …), extended with tie-breaks so that it is a total order
  consistent with `Eq`.
* **Replacements.** A variable is bound to a `Replacement`: `Single(Expr)`,
  `Sequence(Vec<Expr>)` or `Multiset(Multiset<Expr>)`, matching Python's
  expression / tuple / `Multiset`.
* **Matching.** Python's chain of lazy generators is written in
  continuation-passing style: every step takes a callback that it calls once per
  partial match, and the callback can return `ControlFlow::Break` to stop early.
  The search order is the same as in Python. Entry points: `match_pattern`
  (all matches), `for_each_match` (streaming), `first_match`, `is_match`,
  `match_anywhere`.
* **Errors.** Python's `ValueError`/`IndexError` become `Result`s
  (`ExprError`, `MatchError`, `RewriteError`). `Head::call` and
  `Expr::operation` are panicking shorthands for when the arity is known to be right.

## Differences from the Python package

* `CustomConstraint` cannot read variable names from a closure's parameters, so
  they are given explicitly and the closure receives the replacements in that
  order: `CustomConstraint::new(["x", "y"], |args| ...)`.
* Local constraints are not removed from the active set once checked (Python
  removes them and restores them on backtracking). Every constraint whose
  variables are all bound is checked at each binding point. The matches are the
  same; a constraint may just be evaluated more often.
* `FreeOf` also checks multiset-valued bindings. The Python `_is_free` only looks
  inside tuples and lists, so it accepts any multiset (looks like a bug there).
* Unnamed star wildcards in `one_identity` patterns are not bound under a
  `None` key, which the Python version does.
* `enum_maximum_matchings` also handles maximum matchings that leave a *right*
  vertex free. The Python `_enum_maximum_matchings_iter` (Step 8) only looks for
  alternating paths ending at a free left vertex, so for the graph
  `{(0, 0), (0, 1)}` it yields `{0: 0}` but never `{0: 1}`. `many_to_one` is
  not affected: it puts patterns on the right and requires them all to be
  matched. The Rust tests check the enumeration exhaustively against brute
  force on every graph with up to 4 + 4 vertices.
* `BipartiteGraph` keeps its edges in sorted order (Python: insertion order), so
  enumeration order can differ.
* `ManyToOneMatcher<T>` is generic over the label type; `add` labels a pattern
  with itself, `add_with_label` takes any label. Matches are delivered through
  a callback (`for_each_match`, `for_each_group`) or collected (`match_all`,
  `first_match`, `is_match`).
* The many-to-one matcher binds the variable of a named commutative
  subexpression. Python's `ManyToOneMatcher` drops it: for the pattern
  `f(f_c(a, b, variable_name='y'))` and subject `f(f_c(a, b))` its one-to-one
  `match` gives `{y ↦ f_c(a, b)}` but the many-to-one matcher gives `{}`
  (`_match_transition` never runs `_check_transition` for a commutative
  operation's own transition).
* A `CommutativeMatcher` computes all matches of an operation's operands
  before handing them on (Python produces them lazily). The matches are the
  same; stopping early just saves less work.
* Constraint equality (used to share constraints and to detect duplicate
  patterns) goes through `Constraint::eq_key`: built-in constraints compare by
  value, `CustomConstraint`s by closure identity, other constraints by pointer.
* Wildcard heads (`Head::any(Some("F"))`, Python: `WildcardOperationHead`) bind
  the matched head as `Replacement::Head` (Python wraps it in a `SymbolWrapper`).
  They work in both matchers (Python: many-to-one only), constraints on the head
  variable are checked, and `substitute` puts a bound head back into the
  expression. An operation with a wildcard head is not constant (Python's
  `is_constant` overlooks wildcard heads, so e.g. `F_(a)` inside a commutative
  pattern would be treated as a constant operand).
* `SymbolWrapper` (wrapping host-language objects) is not ported. Atoms are
  identified by name only.
