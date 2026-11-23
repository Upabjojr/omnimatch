//! A large synthetic rule set shaped like an integration rule table (in the
//! style of Rubi): `Int(Times(factors...), x)` with commutative, associative
//! products and sums. Compiled by `build.rs` to check that compiled matchers
//! scale. This file is compiled twice: into the build script and into the
//! library.

use omnimatch::{make_dot_variable, Arity, Expr, ExprKind, Head, Pattern};

/// Number of rules compiled into `RuleSetMatcher`.
pub const RULE_COUNT: usize = 1000;

/// A small deterministic random number generator (xorshift).
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// `n` distinct rules; the same `n` always gives the same rules.
pub fn rules(n: usize) -> Vec<Pattern> {
    let int = Head::new("Int", Arity::Binary).unwrap();
    let times = Head::builder("Times").commutative(true).associative(true).one_identity(true).build().unwrap();
    let plus = Head::builder("Plus").commutative(true).associative(true).one_identity(true).build().unwrap();
    let power = Head::new("Power", Arity::Binary).unwrap();
    let functions: Vec<Head> = ["Sin", "Cos", "Tan", "Log", "Exp", "Sqrt", "ArcSin", "ArcTan"]
        .iter()
        .map(|name| Head::new(*name, Arity::Unary).unwrap())
        .collect();
    let vars = ["a", "b", "c", "d", "e", "m", "n", "p"];
    let x = || Expr::atom("x");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    while out.len() < n {
        let factor = |rng: &mut Rng| -> Expr {
            let v = make_dot_variable(vars[rng.below(vars.len())]);
            match rng.below(4) {
                0 => power.call(vec![
                    plus.call(vec![make_dot_variable("a"), times.call(vec![make_dot_variable("b"), x()])]),
                    v,
                ]),
                1 => functions[rng.below(functions.len())].call(vec![times.call(vec![v, x()])]),
                2 => power.call(vec![x(), v]),
                _ => plus.call(vec![make_dot_variable(vars[rng.below(vars.len())]), power.call(vec![x(), v])]),
            }
        };
        let k = 1 + rng.below(3);
        let mut factors: Vec<Expr> = (0..k).map(|_| factor(&mut rng)).collect();
        if rng.below(2) == 0 {
            factors.push(make_dot_variable("u"));
        }
        let pattern = int.call(vec![times.call(factors), x()]);
        if seen.insert(pattern.clone()) {
            out.push(Pattern::new(pattern));
        }
    }
    out
}

/// `pattern` with every wildcard replaced by `atom`: a subject it matches.
pub fn instance(pattern: &Expr, atom: &str) -> Expr {
    match pattern.kind() {
        ExprKind::Wildcard(_) => Expr::atom(atom),
        ExprKind::Atom { .. } => pattern.clone(),
        ExprKind::Operation { head, operands } => head.call(operands.iter().map(|o| instance(o, atom)).collect()),
    }
}

/// Subjects for the rule set: instances of every tenth rule.
pub fn subjects() -> Vec<Expr> {
    rules(RULE_COUNT)
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 10 == 0)
        .flat_map(|(i, p)| [instance(&p.expression, "k"), instance(&p.expression, if i % 20 == 0 { "x" } else { "a" })])
        .collect()
}
