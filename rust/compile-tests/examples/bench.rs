//! Time the compiled matcher against the interpreted one on the test corpus:
//! `cargo run --release -p omnimatch-compile-tests --example bench`

use std::hint::black_box;
use std::time::{Duration, Instant};

use omnimatch::ManyToOneMatcher;
use omnimatch_compile_tests::{cases, CorpusMatcher};

fn time(label: &str, rounds: usize, mut f: impl FnMut() -> usize) -> Duration {
    let mut matches = 0;
    f(); // warm up (fills the commutative caches)
    let start = Instant::now();
    for _ in 0..rounds {
        matches += f();
    }
    let elapsed = start.elapsed();
    println!("{label:<12} {:>9.2?} per round  ({} matches per round)", elapsed / rounds as u32, matches / rounds);
    elapsed
}

fn main() {
    let subjects = cases::subjects();
    let mut interpreted = ManyToOneMatcher::<()>::new();
    for pattern in cases::patterns() {
        interpreted.add_with_label(pattern, ());
    }
    let compiled = CorpusMatcher::new();
    println!("{} patterns, {} subjects", CorpusMatcher::PATTERN_COUNT, subjects.len());

    let rounds = 20;
    let i = time("interpreted", rounds, || subjects.iter().map(|s| black_box(interpreted.match_all(s)).len()).sum());
    let c = time("compiled", rounds, || subjects.iter().map(|s| black_box(compiled.match_all(s)).len()).sum());
    println!("speed-up: {:.2}x", i.as_secs_f64() / c.as_secs_f64());

    // Without the commutative caches: every round starts cold.
    let i = time("interp. cold", rounds, || {
        interpreted.clear();
        subjects.iter().map(|s| black_box(interpreted.match_all(s)).len()).sum()
    });
    let c = time("compiled cold", rounds, || {
        compiled.clear();
        subjects.iter().map(|s| black_box(compiled.match_all(s)).len()).sum()
    });
    println!("speed-up: {:.2}x", i.as_secs_f64() / c.as_secs_f64());

    // Only subjects without commutative operations: no bipartite matching,
    // the automaton's control flow is all the work.
    let mut has_commutative = false;
    let plain: Vec<_> = subjects
        .iter()
        .filter(|s| {
            has_commutative = false;
            s.for_each_preorder(&mut |e| has_commutative |= e.head().is_some_and(|h| h.commutative));
            !has_commutative
        })
        .cloned()
        .collect();
    println!("{} subjects without commutative operations", plain.len());
    let i = time("interpreted", rounds, || plain.iter().map(|s| black_box(interpreted.match_all(s)).len()).sum());
    let c = time("compiled", rounds, || plain.iter().map(|s| black_box(compiled.match_all(s)).len()).sum());
    println!("speed-up: {:.2}x", i.as_secs_f64() / c.as_secs_f64());
}
