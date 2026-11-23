//! A compiled matcher must report the same matches, in the same order, as the
//! interpreted `ManyToOneMatcher` built from the same patterns.

use std::ops::ControlFlow;

use omnimatch::{Expr, ManyToOneMatcher, Substitution};
use omnimatch_compile_tests::cases::{self, a, b, op};
use omnimatch_compile_tests::{CorpusMatcher, EmptyMatcher};

fn interpreted() -> ManyToOneMatcher<()> {
    let mut matcher = ManyToOneMatcher::new();
    for (i, pattern) in cases::patterns().into_iter().enumerate() {
        assert_eq!(matcher.add_with_label(pattern, ()), i, "patterns must be distinct");
    }
    matcher
}

fn interpreted_matches(matcher: &ManyToOneMatcher<()>, subject: &Expr) -> Vec<(usize, Substitution)> {
    let mut out = Vec::new();
    let _ = matcher.for_each_group(subject, &mut |group| {
        out.extend(group);
        ControlFlow::Continue(())
    });
    out
}

#[test]
fn same_matches_as_interpreted() {
    let interpreted = interpreted();
    let compiled = CorpusMatcher::new();
    assert_eq!(CorpusMatcher::PATTERN_COUNT, interpreted.len());
    let subjects = cases::subjects();
    let mut total = 0;
    for subject in &subjects {
        let expected = interpreted_matches(&interpreted, subject);
        let actual = compiled.match_all(subject);
        assert_eq!(actual, expected, "subject {subject}");
        total += expected.len();
    }
    // Make sure the corpus exercises the matcher.
    assert!(subjects.len() > 1000, "{} subjects", subjects.len());
    assert!(total > 5000, "{total} matches");
}

#[test]
fn every_pattern_matches_something() {
    let compiled = CorpusMatcher::new();
    let mut matched = [false; CorpusMatcher::PATTERN_COUNT];
    for subject in cases::subjects() {
        for (index, _) in compiled.match_all(&subject) {
            matched[index] = true;
        }
    }
    let unmatched: Vec<usize> = (0..matched.len()).filter(|&i| !matched[i]).collect();
    // Only the pattern whose global constraint never holds.
    let never: Vec<usize> = cases::patterns()
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.global_constraints().is_empty())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(never.len(), 1);
    assert_eq!(unmatched, never);
}

#[test]
fn first_match_and_is_match() {
    let interpreted = interpreted();
    let compiled = CorpusMatcher::new();
    for subject in cases::subjects() {
        let expected = interpreted.first_match(&subject).map(|(_, s)| s);
        let actual = compiled.first_match(&subject).map(|(_, s)| s);
        assert_eq!(actual, expected, "subject {subject}");
        assert_eq!(compiled.is_match(&subject), expected.is_some());
    }
}

#[test]
fn stops_early() {
    let compiled = CorpusMatcher::new();
    let subject = op("f", &[a(), b()]);
    let all = compiled.match_all(&subject).len();
    assert!(all > 3);
    let mut seen = 0;
    let flow = compiled.for_each_match(&subject, &mut |_, _| {
        seen += 1;
        if seen == 2 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert!(flow.is_break());
    assert_eq!(seen, 2);
}

#[test]
fn caches_can_be_cleared() {
    let compiled = CorpusMatcher::new();
    let subject = op("f_c", &[a(), b(), op("f", &[a()])]);
    let before = compiled.match_all(&subject);
    compiled.clear();
    assert_eq!(compiled.match_all(&subject), before);
}

#[test]
fn empty_matcher() {
    let compiled = EmptyMatcher::new();
    assert_eq!(EmptyMatcher::PATTERN_COUNT, 0);
    assert!(compiled.match_all(&a()).is_empty());
    assert!(!compiled.is_match(&op("f", &[a()])));
}
