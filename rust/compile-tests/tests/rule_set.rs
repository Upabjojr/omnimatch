//! The large rule set: the compiled matcher finds the same matches as the
//! interpreted one, and keeps its rules out of the heap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use omnimatch::{Expr, ManyToOneMatcher, Substitution};
use omnimatch_compile_tests::rules::{self, RULE_COUNT};
use omnimatch_compile_tests::RuleSetMatcher;

/// Counts the bytes allocated and not yet freed by the current thread (tests
/// run in parallel threads).
struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = LIVE.try_with(|c| c.set(c.get() + layout.size() as isize));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = LIVE.try_with(|c| c.set(c.get() - layout.size() as isize));
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> isize {
    LIVE.with(Cell::get)
}

fn interpreted() -> ManyToOneMatcher<()> {
    let mut matcher = ManyToOneMatcher::new();
    for (i, pattern) in rules::rules(RULE_COUNT).into_iter().enumerate() {
        assert_eq!(matcher.add_with_label(pattern, ()), i);
    }
    matcher
}

fn sorted(mut matches: Vec<(usize, Substitution)>) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = matches.drain(..).map(|(i, s)| (i, s.to_string())).collect();
    out.sort();
    out
}

#[test]
fn same_matches_as_interpreted() {
    let interpreted = interpreted();
    let compiled = RuleSetMatcher::new();
    assert_eq!(RuleSetMatcher::PATTERN_COUNT, RULE_COUNT);
    let mut total = 0;
    for subject in rules::subjects() {
        let expected = sorted(interpreted_indexed(&interpreted, &subject));
        let actual = sorted(compiled.match_all(&subject));
        assert_eq!(actual, expected, "subject {subject}");
        total += actual.len();
    }
    assert!(total > rules::subjects().len(), "the subjects should match: {total} matches");
}

fn interpreted_indexed(matcher: &ManyToOneMatcher<()>, subject: &Expr) -> Vec<(usize, Substitution)> {
    let mut out = Vec::new();
    let _ = matcher.for_each_group(subject, &mut |group| {
        out.extend(group);
        std::ops::ControlFlow::Continue(())
    });
    out
}

#[test]
fn rules_are_not_on_the_heap() {
    let subjects = rules::subjects();

    let before = live();
    let interpreted = interpreted();
    let interpreted_heap = live() - before;

    let before = live();
    let compiled = RuleSetMatcher::new();
    let new_heap = live() - before;
    for subject in &subjects {
        assert!(!compiled.match_all(subject).is_empty());
    }
    let compiled_heap = live() - before;
    compiled.clear();
    let cleared_heap = live() - before;
    drop(interpreted);

    // `new()` allocates one empty cell per commutative matcher.
    assert!(new_heap < 16 * 1024, "RuleSetMatcher::new() allocated {new_heap} bytes");
    // Matching fills the commutative matchers' caches of operand matches
    // (as in the interpreted matcher); `clear()` empties them. What remains
    // is what was built from the static tables: the commutative matchers that
    // were used, and the expressions and constraints they needed.
    assert!(compiled_heap > cleared_heap);
    assert!(
        cleared_heap * 10 < interpreted_heap,
        "compiled matcher: {cleared_heap} bytes ({compiled_heap} before clear()), interpreted: {interpreted_heap} bytes"
    );
}
