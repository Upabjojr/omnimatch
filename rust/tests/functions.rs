//! Port of `tests/test_functions.py`.

// The case tables mirror the Python parametrize lists.
#![allow(clippy::type_complexity)]

use omnimatch::*;

fn f(ops: &[Expr]) -> Expr {
    Head::new("f", Arity::Variadic).unwrap().call(ops.to_vec())
}
fn f2(ops: &[Expr]) -> Expr {
    Head::new("f2", Arity::Variadic).unwrap().call(ops.to_vec())
}
fn f_c(ops: &[Expr]) -> Expr {
    Head::builder("f_c").commutative(true).build().unwrap().call(ops.to_vec())
}
fn a() -> Expr {
    Expr::atom("a")
}
fn b() -> Expr {
    Expr::atom("b")
}
fn c() -> Expr {
    Expr::atom("c")
}
fn x_() -> Expr {
    make_dot_variable("x")
}
fn y_() -> Expr {
    make_dot_variable("y")
}
fn one(e: Expr) -> Replacement {
    Replacement::Single(e)
}
fn seq(es: &[Expr]) -> Replacement {
    Replacement::Sequence(es.to_vec())
}
fn sub(pairs: &[(&str, Replacement)]) -> Substitution {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[test]
fn is_match_cases() {
    assert!(is_match(&a(), &Pattern::new(a())).unwrap());
    assert!(!is_match(&a(), &Pattern::new(b())).unwrap());
    assert!(is_match(&f(&[a()]), &Pattern::new(f(&[x_()]))).unwrap());
}

#[test]
fn substitute_cases() {
    let cases: Vec<(Expr, Substitution, Replacement)> = vec![
        (a(), sub(&[]), one(a())),
        (a(), sub(&[("x", one(b()))]), one(a())),
        (x_(), sub(&[("x", one(b()))]), one(b())),
        (x_(), sub(&[("x", seq(&[a(), b()]))]), seq(&[a(), b()])),
        (y_(), sub(&[("x", one(b()))]), one(y_())),
        (f(&[x_()]), sub(&[("x", one(b()))]), one(f(&[b()]))),
        (f(&[x_()]), sub(&[("y", one(b()))]), one(f(&[x_()]))),
        (f(&[x_()]), sub(&[]), one(f(&[x_()]))),
        (f(&[a(), x_()]), sub(&[("x", one(b()))]), one(f(&[a(), b()]))),
        (f(&[x_()]), sub(&[("x", seq(&[a(), b()]))]), one(f(&[a(), b()]))),
        (f(&[x_()]), sub(&[("x", seq(&[]))]), one(f(&[]))),
        (f(&[x_(), c()]), sub(&[("x", seq(&[a(), b()]))]), one(f(&[a(), b(), c()]))),
        (f(&[x_(), y_()]), sub(&[("x", one(a())), ("y", one(b()))]), one(f(&[a(), b()]))),
        (f(&[x_(), y_()]), sub(&[("x", seq(&[a(), c()])), ("y", one(b()))]), one(f(&[a(), c(), b()]))),
        (f(&[x_(), y_()]), sub(&[("x", one(a())), ("y", seq(&[b(), c()]))]), one(f(&[a(), b(), c()]))),
    ];
    for (expression, substitution, expected) in cases {
        assert_eq!(substitute(&expression, &substitution).unwrap(), expected, "{expression} / {substitution}");
    }
}

#[test]
fn substitute_custom_sorting_key() {
    let s = sub(&[("x", one(a())), ("y", Replacement::Multiset([b(), c()].into_iter().collect()))]);
    let expression = f(&[x_(), y_()]);
    let reverse = |l: &Expr, r: &Expr| r.to_string().cmp(&l.to_string());
    assert_eq!(substitute_sorted_by(&expression, &s, &reverse).unwrap(), one(f(&[a(), c(), b()])));
    assert_eq!(substitute(&expression, &s).unwrap(), one(f(&[a(), b(), c()])));
}

fn replace_one_via_many(
    expression: &Expr,
    position: &[usize],
    replacement: Replacement,
) -> Result<Replacement, ExprError> {
    replace_many(expression, vec![(position.to_vec(), replacement)])
}

#[test]
fn replace_cases() {
    let fab = || f(&[a(), b()]);
    let cases: Vec<(Expr, Vec<usize>, Replacement, Expr)> = vec![
        (a(), vec![], one(b()), b()),
        (f(&[a()]), vec![], one(b()), b()),
        (a(), vec![], one(f(&[b()])), f(&[b()])),
        (f(&[a()]), vec![], one(f(&[b()])), f(&[b()])),
        (f(&[a()]), vec![0], one(b()), f(&[b()])),
        (fab(), vec![0], one(c()), f(&[c(), b()])),
        (fab(), vec![1], one(c()), f(&[a(), c()])),
        (f(&[a()]), vec![0], seq(&[b(), c()]), f(&[b(), c()])),
        (fab(), vec![0], seq(&[b(), c()]), f(&[b(), c(), b()])),
        (fab(), vec![1], seq(&[b(), c()]), f(&[a(), b(), c()])),
        (f(&[f(&[a()])]), vec![0], one(b()), f(&[b()])),
        (f(&[f(&[a()])]), vec![0, 0], one(b()), f(&[f(&[b()])])),
        (f(&[fab()]), vec![0, 0], one(c()), f(&[f(&[c(), b()])])),
        (f(&[fab()]), vec![0, 1], one(c()), f(&[f(&[a(), c()])])),
        (f(&[fab(), fab()]), vec![0, 0], one(c()), f(&[f(&[c(), b()]), fab()])),
        (f(&[fab(), fab()]), vec![0, 1], one(c()), f(&[f(&[a(), c()]), fab()])),
        (f(&[fab(), fab()]), vec![1, 0], one(c()), f(&[fab(), f(&[c(), b()])])),
        (f(&[fab(), fab()]), vec![1, 1], one(c()), f(&[fab(), f(&[a(), c()])])),
        (f(&[fab(), fab()]), vec![0], one(c()), f(&[c(), fab()])),
        (f(&[fab(), fab()]), vec![1], one(c()), f(&[fab(), c()])),
    ];
    type ReplaceFn = fn(&Expr, &[usize], Replacement) -> Result<Replacement, ExprError>;
    for replace_fn in [replace as ReplaceFn, replace_one_via_many] {
        for (expression, position, replacement, expected) in &cases {
            assert_eq!(
                replace_fn(expression, position, replacement.clone()).unwrap(),
                one(expected.clone()),
                "{expression} {position:?}"
            );
        }
        assert!(replace_fn(&a(), &[0], one(b())).is_err());
        assert!(replace_fn(&f(&[a()]), &[0, 0], one(b())).is_err());
        assert!(replace_fn(&f(&[a()]), &[1], one(b())).is_err());
        assert!(replace_fn(&fab(), &[2], one(b())).is_err());
    }
}

#[test]
fn replace_many_cases() {
    let r = |p: &[usize], e: Replacement| (p.to_vec(), e);
    let f2ab = || f2(&[a(), b()]);
    let cases: Vec<(Expr, Vec<(Vec<usize>, Replacement)>, Expr)> = vec![
        (f(&[a(), b()]), vec![r(&[0], one(b())), r(&[1], one(a()))], f(&[b(), a()])),
        (f(&[a(), b()]), vec![r(&[0], seq(&[c(), c()])), r(&[1], one(a()))], f(&[c(), c(), a()])),
        (f(&[a(), b()]), vec![r(&[0], one(b())), r(&[1], seq(&[c(), c()]))], f(&[b(), c(), c()])),
        (f(&[f2ab(), c()]), vec![r(&[0, 0], one(b())), r(&[0, 1], one(a()))], f(&[f2(&[b(), a()]), c()])),
        (f_c(&[c(), f2ab()]), vec![r(&[1, 0], one(b())), r(&[1, 1], one(a()))], f_c(&[c(), f2(&[b(), a()])])),
        (f(&[f2ab(), f2(&[c()])]), vec![r(&[1, 0], one(b())), r(&[0, 1], one(a()))], f(&[f2(&[a(), a()]), f2(&[b()])])),
        (f(&[f2ab(), f2(&[c()])]), vec![r(&[0, 1], one(a())), r(&[1, 0], one(b()))], f(&[f2(&[a(), a()]), f2(&[b()])])),
        (
            f_c(&[f2(&[c()]), f2ab()]),
            vec![r(&[0, 0], one(b())), r(&[1, 1], one(a()))],
            f_c(&[f2(&[b()]), f2(&[a(), a()])]),
        ),
        (
            f_c(&[f2(&[c()]), f2ab()]),
            vec![r(&[1, 1], one(a())), r(&[0, 0], one(b()))],
            f_c(&[f2(&[b()]), f2(&[a(), a()])]),
        ),
    ];
    for (expression, replacements, expected) in cases {
        assert_eq!(replace_many(&expression, replacements).unwrap(), one(expected), "{expression}");
    }
    assert!(replace_many(&f(&[a()]), vec![r(&[], one(b())), r(&[0], one(b()))]).is_err());
    assert!(replace_many(&a(), vec![r(&[], one(b())), r(&[0], one(b()))]).is_err());
    assert!(replace_many(&a(), vec![r(&[0], one(b())), r(&[1], one(b()))]).is_err());
    assert_eq!(replace_many(&f(&[a(), b()]), vec![]).unwrap(), one(f(&[a(), b()])));
}

#[test]
fn match_anywhere_cases() {
    let pos = |s: Substitution, p: &[usize]| (s, p.to_vec());
    let x = |e: Expr| sub(&[("x", one(e))]);
    let cases: Vec<(Expr, Expr, Vec<(Substitution, Vec<usize>)>)> = vec![
        (f(&[a()]), f(&[x_()]), vec![pos(x(a()), &[])]),
        (f(&[a()]), x_(), vec![pos(x(f(&[a()])), &[]), pos(x(a()), &[0])]),
        (
            f(&[a(), f2(&[b()]), f2(&[f2(&[c()]), f2(&[a()]), f2(&[f2(&[b()])])]), f2(&[c()]), c()]),
            f2(&[x_()]),
            vec![
                pos(x(b()), &[1]),
                pos(x(c()), &[2, 0]),
                pos(x(a()), &[2, 1]),
                pos(x(f2(&[b()])), &[2, 2]),
                pos(x(b()), &[2, 2, 0]),
                pos(x(c()), &[3]),
            ],
        ),
    ];
    for (expression, pattern, expected) in cases {
        let results = match_anywhere(&expression, &Pattern::new(pattern.clone())).unwrap();
        assert_eq!(results.len(), expected.len(), "{expression} / {pattern}");
        for e in &expected {
            assert!(results.contains(e), "{expression} / {pattern}: missing {e:?}");
        }
    }
    assert_eq!(match_anywhere(&f(&[x_()]), &Pattern::new(f(&[x_()]))), Err(MatchError::NonConstantSubject));
}

fn get(s: &Substitution, n: &str) -> Replacement {
    s.get(n).unwrap().clone()
}

fn single(s: &Substitution, n: &str) -> Expr {
    get(s, n).as_single().unwrap().clone()
}

/// `test_logic_simplify`: a boolean formula reduced to ⊥ by a rewrite system,
/// with both `replace_all` and `ManyToOneReplacer`.
#[test]
fn logic_simplify() {
    let acm = |name: &str| Head::builder(name).associative(true).one_identity(true).commutative(true).build().unwrap();
    let (land, lor, lxor) = (acm("and"), acm("or"), acm("xor"));
    let lnot = Head::new("not", Arity::Unary).unwrap();
    let limplies = Head::new("implies", Arity::Binary).unwrap();
    let iff = Head::new("iff", Arity::Binary).unwrap();
    let bot = Expr::atom("⊥");
    let top = Expr::atom("⊤");
    let v: Vec<Expr> = (0..=11).map(|i| Expr::atom(format!("a{i}"))).collect();

    let and = |o: &[Expr]| land.call(o.to_vec());
    let or = |o: &[Expr]| lor.call(o.to_vec());
    let xor = |o: &[Expr]| lxor.call(o.to_vec());
    let not = |e: Expr| lnot.call(vec![e]);
    let implies = |l: Expr, r: Expr| limplies.call(vec![l, r]);
    let eqv = |l: Expr, r: Expr| iff.call(vec![l, r]);

    let part1 = eqv(
        eqv(
            or(&[v[1].clone(), v[2].clone()]),
            or(&[not(v[3].clone()), eqv(xor(&[v[4].clone(), v[5].clone()]), not(not(not(v[6].clone()))))]),
        ),
        not(and(&[
            and(&[v[7].clone(), v[8].clone()]),
            not(xor(&[
                xor(&[or(&[v[9].clone(), and(&[v[10].clone(), v[11].clone()])]), v[2].clone()]),
                and(&[
                    and(&[v[11].clone(), xor(&[v[2].clone(), eqv(v[5].clone(), v[5].clone())])]),
                    xor(&[xor(&[v[7].clone(), v[7].clone()]), eqv(v[9].clone(), v[4].clone())]),
                ]),
            ])),
        ])),
    );
    let part2 = not(and(&[
        implies(
            and(&[v[1].clone(), v[2].clone()]),
            not(xor(&[
                or(&[
                    or(&[
                        xor(&[
                            implies(and(&[v[3].clone(), v[4].clone()]), implies(v[5].clone(), v[6].clone())),
                            or(&[v[7].clone(), v[8].clone()]),
                        ]),
                        xor(&[eqv(v[9].clone(), v[10].clone()), v[11].clone()]),
                    ]),
                    xor(&[xor(&[v[2].clone(), v[2].clone()]), v[7].clone()]),
                ]),
                eqv(or(&[v[4].clone(), v[9].clone()]), xor(&[not(v[6].clone()), v[6].clone()])),
            ])),
        ),
        not(eqv(not(v[11].clone()), not(v[9].clone()))),
    ]));
    let expression = implies(and(&[part1.clone(), implies(part1, part2.clone())]), part2);

    let (x__, y___, z_) = (make_plus_variable("x"), make_star_variable("y"), make_dot_variable("z"));
    let rule = |p: Expr, r: Box<dyn Fn(&Substitution) -> Expr + Send + Sync>| {
        ReplacementRule::new(Pattern::new(p), move |s| Replacement::Single(r(s)))
    };
    let (h_and, h_xor, h_not, bot2) = (land.clone(), lxor.clone(), lnot.clone(), bot.clone());
    let (h_and2, h_xor2, h_and3, h_xor3, h_and4, h_xor4) =
        (land.clone(), lxor.clone(), land.clone(), lxor.clone(), land.clone(), lxor.clone());
    let (h_xor5, h_and5, h_xor6, top2, h_xor7) = (lxor.clone(), land.clone(), lxor.clone(), top.clone(), lxor.clone());
    let rules = vec![
        // xor(x, ⊥) → x
        rule(xor(&[x__.clone(), bot.clone()]), Box::new(move |s| h_xor.call(get(s, "x").to_vec()))),
        // xor(x, x) → ⊥
        rule(xor(&[x_(), x_(), Expr::star(None)]), Box::new(move |_| bot2.clone())),
        // and(x, ⊤) → x
        rule(and(&[x__.clone(), top.clone()]), Box::new(move |s| h_and.call(get(s, "x").to_vec()))),
        // and(x, ⊥) → ⊥
        rule(and(&[Expr::plus(None), bot.clone()]), {
            let bot = bot.clone();
            Box::new(move |_| bot.clone())
        }),
        // and(x, x) → x
        rule(
            and(&[x_(), x_(), y___.clone()]),
            Box::new(move |s| {
                let mut ops = vec![single(s, "x")];
                ops.extend(get(s, "y").to_vec());
                h_and2.call(ops)
            }),
        ),
        // and(x, xor(y, z)) → xor(and(x, y), and(x, z))
        rule(
            and(&[x_(), xor(&[y_(), z_.clone()])]),
            Box::new(move |s| {
                let (x, y, z) = (single(s, "x"), single(s, "y"), single(s, "z"));
                h_xor2.call(vec![h_and3.call(vec![x.clone(), y]), h_and3.call(vec![x, z])])
            }),
        ),
        // implies(x, y) → not(xor(x, and(x, y)))
        rule(
            implies(x_(), y_()),
            Box::new(move |s| {
                let (x, y) = (single(s, "x"), single(s, "y"));
                h_not.call(vec![h_xor3.call(vec![x.clone(), h_and4.call(vec![x, y])])])
            }),
        ),
        // not(x) → xor(x, ⊤)
        rule(not(x_()), Box::new(move |s| h_xor4.call(vec![single(s, "x"), top2.clone()]))),
        // or(x, y) → xor(and(x, y), xor(x, y))
        rule(
            or(&[x_(), y_()]),
            Box::new(move |s| {
                let (x, y) = (single(s, "x"), single(s, "y"));
                h_xor5.call(vec![h_and5.call(vec![x.clone(), y.clone()]), h_xor6.call(vec![x, y])])
            }),
        ),
        // iff(x, y) → not(xor(x, y))
        rule(eqv(x_(), y_()), Box::new(move |s| lnot.call(vec![h_xor7.call(vec![single(s, "x"), single(s, "y")])]))),
    ];
    assert_eq!(replace_all(&expression, &rules, None).unwrap(), one(bot.clone()), "replace_all");
    let replacer = ManyToOneReplacer::new(rules);
    assert_eq!(replacer.replace(&expression, None).unwrap(), one(bot), "ManyToOneReplacer");
}
